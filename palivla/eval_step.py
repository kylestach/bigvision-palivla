from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import multihost_utils


from functools import partial

from palivla.tokenizer import Tokenizer
from palivla.train_step import compute_action_metrics, compute_stats
from palivla.palivla_types import TrainingBatch, RolloutBatch

from jax.sharding import PartitionSpec
from scalax.sharding import MeshShardingHelper

import tensorflow as tf
import wandb
import numpy as np

# get the next seven tokens after the begin of action token; otherwise, use zeros
def extract_action_tokens(step_tokens, begin_of_action_token):
    action_starts = (step_tokens == begin_of_action_token).astype(jnp.int32)
    first_action_start_idx = jnp.argmax(action_starts)+1 # first index of action token. will be 0+1 if none found
    
    action_tokens = lax.cond(
        (first_action_start_idx == 1) | (first_action_start_idx + 7 > step_tokens.shape[0]),
        lambda _: jnp.zeros(7, dtype=step_tokens.dtype),
        lambda _: lax.dynamic_slice(step_tokens, (first_action_start_idx,), (7,)),
        operand=None
    
    )
    
    return action_tokens

# get the cot tokens (tokens btwn begin CoT token and begin action token)
def extract_cot_strs(step_tokens, masked_prompt_tokens, beg_cot_token, beg_action_token, detokenize_lang_fn):

    # get prompt
    masked_prompt_tokens = np.array(masked_prompt_tokens)
    first_zero_idx = np.where(masked_prompt_tokens == 0)[0][0] if 0 in masked_prompt_tokens else len(masked_prompt_tokens)
    prompt_tokens = masked_prompt_tokens[:first_zero_idx]

    detokenize_prompt = detokenize_lang_fn(tf.convert_to_tensor(prompt_tokens, dtype=tf.int32))
    prompt_str = tf.strings.reduce_join(detokenize_prompt, separator="").numpy().decode("utf-8")

    # get cot
    step_tokens_np = np.array(step_tokens)

    try:
        action_start_idx = np.where(step_tokens_np == beg_action_token)[0][0]
    except IndexError:
        return prompt_str, "" 
    
    if action_start_idx<=0:
        return prompt_str, ""

    #the output sequence (step_tokens) directly starts from CoT, bc the prompt now includes the beg_cot_token
    cot_tokens = step_tokens_np[:action_start_idx]

    detokenized_cot = detokenize_lang_fn(tf.convert_to_tensor(cot_tokens, dtype=tf.int32))
    cot_str = tf.strings.reduce_join(detokenized_cot, separator="").numpy().decode("utf-8")

    return prompt_str, cot_str 

def get_cot_table_metrics(lang_and_cot_strs):
    table = wandb.Table(columns=["GT Language Label", "Predicted CoT"])
    for lang_str, cot_str in lang_and_cot_strs:
        table.add_data(lang_str, cot_str)
    dct = {"CoT Outputs": table}
    return dct


def compute_gen_stats(
    decode_fn,
    tokenize_fn,
    detokenize_fn,
    detokenize_lang_fn,
    mesh: MeshShardingHelper,
    batch: TrainingBatch,
    prefix: str,
    tokenizer_config: Tokenizer.TokenizerConfig,
    target_key_order: Sequence[str] | None = None,
    use_cot: bool = False,
):
    """
    Compute generative (rollout) statistics on a batch of data.
    """

    def _split_tokens(tokens, mask, mask_ar, gen_start):
        seq_len = tokens.shape[0]
        prompt = jnp.where(
            jnp.arange(seq_len) < gen_start,
            tokens,
            jnp.zeros_like(tokens),
        )
        prompt_mask = jnp.where(
            jnp.arange(seq_len) < gen_start,
            mask,
            jnp.zeros_like(mask),
        )
        prompt_ar = jnp.where(
            jnp.arange(seq_len) < gen_start,
            mask_ar,
            jnp.zeros_like(mask_ar),
        )

        gen_start = jnp.argmax(tokens == tokenizer_config.begin_of_action_token)+1

        gen, gen_mask = jax.lax.cond(
            gen_start == 1,
            lambda: (jnp.zeros_like(jnp.arange(seq_len)), jnp.zeros_like(jnp.arange(seq_len)).astype(mask.dtype)),
            lambda: (
                jnp.where(
                        jnp.roll(mask, -gen_start, axis=0),
                        jnp.roll(tokens, -gen_start, axis=0),
                        0,
                      ),
                jnp.roll(mask, -gen_start, axis=0).astype(mask.dtype)
            ),
        )
         
        gen_ar = jnp.ones_like(gen_mask)

        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "prompt_ar": prompt_ar,
            "gen": gen,
            "gen_mask": gen_mask,
            "gen_ar": gen_ar,
        }

    split_tokens = jax.vmap(_split_tokens)(batch.tokens, batch.tokens_mask, batch.tokens_ar, batch.gen_start)

    rollout_batch = RolloutBatch(
        sensor_data=batch.sensors,
        sensor_masks=batch.sensors_mask,
        prompt=split_tokens["prompt"],
        prompt_mask=split_tokens["prompt_mask"],
        prompt_ar=split_tokens["prompt_ar"],
    )

    out_tokens = decode_fn(
        rollout_batch,
        target_key_order=target_key_order,
    )

    def _compute_action_metrics_shim(
        pred_action_tokens,
        pred_action_logits,
        gt_action_tokens,
        gt_actions,
    ):
        return compute_action_metrics(
            detokenize_fn=detokenize_fn,
            pred_action_tokens=pred_action_tokens,
            pred_action_logits=pred_action_logits,
            gt_action_tokens=gt_action_tokens,
            gt_actions=gt_actions,
            action_dim=batch.actions.shape[-1],
            tokenizer_config=tokenizer_config,
            log_segment_prefix=prefix,
        )
    
    extract_action_tokens_vmap = jax.vmap(extract_action_tokens, in_axes=(0, None)) # only map over batch argument
    out_action_tokens = extract_action_tokens_vmap(out_tokens, tokenizer_config.begin_of_action_token)

    if use_cot:
        # batch.tokens refers to the masked prompt tokens, so we'd have to add batch.cot_tokens to get the ground truth
        gathered_batch_tokens = multihost_utils.process_allgather(batch.tokens)
        gathered_out_tokens = multihost_utils.process_allgather(out_tokens)

        batch_tokens_np = np.asarray(gathered_batch_tokens)
        out_tokens_np = np.asarray(gathered_out_tokens)

        lang_and_cot_strs = [
            extract_cot_strs(tokens, prompt_tokens, tokenizer_config.begin_of_cot_token, tokenizer_config.begin_of_action_token, detokenize_lang_fn)
            for prompt_tokens, tokens in zip(batch_tokens_np, out_tokens_np)
        ]
        cot_metrics = get_cot_table_metrics(lang_and_cot_strs)

    action_metrics = mesh.sjit(
        _compute_action_metrics_shim,
        out_shardings=PartitionSpec(),
    )(
        out_action_tokens,
        None,
        split_tokens["gen"][:, :tokenizer_config.num_action_tokens],
        batch.actions,
    )

    action_metrics = jax.device_get(action_metrics)

    if use_cot:
        return {**action_metrics, **cot_metrics}
    
    return action_metrics


def compute_eval_stats(
    predict_fn,
    detokenize_fn,
    batch: TrainingBatch,
    prefix: str,
    tokenizer_config: Tokenizer.TokenizerConfig,
    target_key_order: Sequence[str] | None = None,
):
    all_inputs = batch.sensors | {"text": batch.tokens[..., :-1]}
    all_masks = batch.sensors_mask | {
        "text": jnp.ones_like(batch.tokens[..., :-1], dtype=jnp.bool_)
    }

    logits, _ = predict_fn(
        all_inputs,
        all_masks,
        text_ar_mask=batch.tokens_ar[..., :-1],
        proprio=batch.get("proprio_single_arm", None),
        train=False,
        target_key_order=target_key_order,
    )

    results = compute_stats(
        detokenize_fn=detokenize_fn,
        pred_logits=logits,
        tokens=batch.tokens,
        actions=batch.actions,
        mask_loss=batch.tokens_loss,
        tokenizer_config=tokenizer_config,
        log_segment_prefix=prefix,
    )[1]

    return jax.device.get(results)
