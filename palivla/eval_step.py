from typing import Sequence

import jax
import jax.numpy as jnp

from functools import partial

from palivla.tokenizer import Tokenizer
from palivla.train_step import compute_action_metrics, compute_stats
from palivla.palivla_types import TrainingBatch, RolloutBatch

from jax.sharding import PartitionSpec
from scalax.sharding import MeshShardingHelper


def _compute_action_metrics_shim(
    detokenize_fn,
    log_segment_prefix: str | None,
    action_dim: int,
    tokenizer_config: Tokenizer.TokenizerConfig,
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
        action_dim=action_dim,
        tokenizer_config=tokenizer_config,
        log_segment_prefix=log_segment_prefix,
    )


# ria todo: for cot, just change this method to incorporate the cot token structure
def compute_gen_stats(
    decode_fn,
    tokenize_fn,
    detokenize_fn,
    mesh: MeshShardingHelper,
    batch: TrainingBatch,
    prefix: str,
    tokenizer_config: Tokenizer.TokenizerConfig,
    target_key_order: Sequence[str] | None = None,
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

        gen_mask = jnp.arange(seq_len) >= gen_start
        gen_mask = jnp.roll(mask, -gen_start, axis=0)
        gen = jnp.where(
            gen_mask,
            jnp.roll(tokens, -gen_start, axis=0),
            0,
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

    import pdb; pdb.set_trace()
    out_tokens = decode_fn(
        rollout_batch,
        target_key_order=target_key_order,
    ) 

    # if we're using CoT, all the out tokens don't necessarily correspond to the action 
    # to deconflict, take all tokens after generated begin of action token as the generated action
    
    gt_action_tokens = tokenize_fn(batch.actions)

    token_exists = jnp.any(out_tokens == tokenizer_config.begin_of_action_token, axis=1) # check if begin_of_action_token appears in generated sequences
    action_start_indices = jnp.argmax(out_tokens == tokenizer_config.begin_of_action_token, axis=1) # use argmax to find the first occurrence for every timestep
    action_start_indices = jnp.where(token_exists, action_start_indices, -1) # for steps where it's not found, replace with "-1"

    # get the next seven tokens after the begin of action token; otherwise, use zeros
    extract_tokens_vmap = jax.vmap(
        lambda seq, idx: jnp.where(
            (idx == -1) | (idx + 7 > seq.shape[0]),
            jnp.zeros(7, dtype=seq.dtype),
            seq[idx:idx + 7]
        ),
        in_axes=(0, 0)  # maps across every step of trajectory, and every token generated for that step
    )

    out_action_tokens = extract_tokens_vmap(out_tokens, action_start_indices)

    __compute_action_metrics_shim = partial(
        _compute_action_metrics_shim,
        detokenize_fn,
        prefix,
        batch.actions.shape[-1],
        tokenizer_config,
    )

    return mesh.sjit(
        __compute_action_metrics_shim,
        out_shardings=PartitionSpec(),
    )(
        out_action_tokens,
        None,
        gt_action_tokens,
        batch.actions,
    )


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

    return compute_stats(
        detokenize_fn=detokenize_fn,
        pred_logits=logits,
        tokens=batch.tokens,
        actions=batch.actions,
        mask_loss=batch.tokens_loss,
        tokenizer_config=tokenizer_config,
        log_segment_prefix=prefix,
    )[1]
