from typing import Sequence

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import multihost_utils


from functools import partial

from palivla.tokenizer import Tokenizer
from palivla.train_step import compute_action_metrics, compute_stats
from palivla.palivla_types import TrainingBatch, RolloutBatch
from palivla.cot_utils import extract_action_tokens, extract_cot_strs, get_cot_table_metrics, viz_cot

from jax.sharding import PartitionSpec
from scalax.sharding import MeshShardingHelper

import tensorflow as tf
import wandb
import numpy as np

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

    def _split_tokens(tokens, mask, mask_ar, gen_start_idx, action_start_idx):
        seq_len = tokens.shape[0]
        prompt = jnp.where(
            jnp.arange(seq_len) < gen_start_idx,
            tokens,
            jnp.zeros_like(tokens),
        )
        prompt_mask = jnp.where(
            jnp.arange(seq_len) < gen_start_idx,
            mask,
            jnp.zeros_like(mask),
        )
        prompt_ar = jnp.where(
            jnp.arange(seq_len) < gen_start_idx,
            mask_ar,
            jnp.zeros_like(mask_ar),
        )

        # we will always be able to find an action token, bc this is ground truth
        gen = jnp.roll(tokens, -action_start_idx, axis=0)
        gen_mask = jnp.roll(mask, -action_start_idx, axis=0).astype(mask.dtype)
        gen_ar = jnp.ones_like(gen_mask)

        return {
            "prompt": prompt,
            "prompt_mask": prompt_mask,
            "prompt_ar": prompt_ar,
            "gen": gen,
            "gen_mask": gen_mask,
            "gen_ar": gen_ar,
        }

    split_tokens = jax.vmap(_split_tokens)(batch.tokens, batch.tokens_mask, batch.tokens_ar, batch.gen_start_idx, batch.action_start_idx)

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

    if use_cot:
        extract_action_tokens_vmap = jax.vmap(extract_action_tokens, in_axes=(0, None)) # only map over batch argument
        out_action_tokens = extract_action_tokens_vmap(out_tokens, tokenizer_config.begin_of_action_token) # check that this actually contains actions
    else:
        out_action_tokens = out_tokens # all generated tokens are action tokens if we don't use chain of thought

    action_metrics = mesh.sjit(
        _compute_action_metrics_shim,
        out_shardings=PartitionSpec(),
    )(
        out_action_tokens,
        None,
        split_tokens["gen"][:, :tokenizer_config.num_action_tokens],
        batch.actions
    )

    metrics = jax.device_get(action_metrics)

    if use_cot:
        # batch.tokens refers to the masked prompt tokens, so we'd have to add batch.cot_tokens to get the ground truth
        gathered_batch_tokens = multihost_utils.process_allgather(batch.tokens)
        gathered_batch_images = multihost_utils.process_allgather(batch.sensors)['image_primary']
        gathered_out_tokens = multihost_utils.process_allgather(out_tokens)

        batch_tokens_np = np.asarray(gathered_batch_tokens)
        out_tokens_np = np.asarray(gathered_out_tokens)

        lang_and_cot_strs = [
            extract_cot_strs(step_out_tokens, step_prompt_tokens, tokenizer_config.begin_of_action_token, detokenize_lang_fn)
            for step_out_tokens, step_prompt_tokens in zip(out_tokens_np, batch_tokens_np)
        ]

        # now visualize just one of the chain of thoughts 
        viz_metrics = viz_cot(image=gathered_batch_images[0], lang_str=lang_and_cot_strs[0][0], reasoning_str=lang_and_cot_strs[0][1])
        cot_metrics = get_cot_table_metrics(lang_and_cot_strs)
        metrics = metrics | cot_metrics
        metrics = metrics | viz_metrics
    
    return metrics


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
