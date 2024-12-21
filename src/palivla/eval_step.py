from typing import Sequence

import jax
import jax.numpy as jnp

from palivla.train_step import compute_stats
from palivla.typing import TrainingBatch, RolloutBatch


def compute_gen_stats(
    decode_fn,
    detokenize_fn,
    batch: TrainingBatch,
    prefix: str,
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

    split_tokens = jax.vmap(_split_tokens)(
        batch.tokens, batch.tokens_mask, batch.tokens_ar, batch.gen_start
    )

    rollout_batch = RolloutBatch(
        sensor_data=batch.sensors,
        sensor_masks=batch.sensors_mask,
        prompt=split_tokens["prompt"],
        prompt_mask=split_tokens["prompt_mask"],
        prompt_ar=split_tokens["prompt_ar"],
    )
    out_tokens = decode_fn(rollout_batch)
    out_actions = detokenize_fn(out_tokens)
    num_tokens = out_tokens.shape[1]
    target_tokens = batch.tokens[..., :num_tokens]
    gen_mask = split_tokens["gen_mask"][..., :num_tokens]

    # Compute L2 metrics
    metrics = {
        "mse": jnp.mean(jnp.square(out_actions - batch.actions)),
        "mae": jnp.mean(jnp.abs(out_actions - batch.actions)),
        "accuracy": jnp.mean(
            (out_tokens == target_tokens) * gen_mask
        )
        / jnp.mean(gen_mask),
    }

    metrics = {prefix + k: v for k, v in metrics.items()}

    return metrics


def compute_eval_stats(
    predict_fn,
    batch: TrainingBatch,
    prefix: str,
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
        proprio=batch.get("proprio", None),
        train=False,
        target_key_order=target_key_order,
    )

    metrics = compute_stats(
        pred_logits=logits,
        tokens=batch.tokens,
        mask_loss=batch.tokens_loss,
    )[1]

    metrics = {prefix + k: v for k, v in metrics.items()}
    return metrics
