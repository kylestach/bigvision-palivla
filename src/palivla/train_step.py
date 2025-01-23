from typing import Any

import chex
import jax
import jax.numpy as jnp
import optax

from palivla.components.train_state import TrainState


def compute_stats(
    *,
    pred_logits,
    target_tokens,
    target_mask_loss,
):
    loss = jnp.mean(
        target_mask_loss
        * optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_tokens),
        axis=-1,
    ) / jnp.mean(target_mask_loss, axis=-1)
    loss = jnp.mean(loss)

    accuracy = jnp.mean(
        target_mask_loss * (jnp.argmax(pred_logits, axis=-1) == target_tokens)
    ) / jnp.mean(target_mask_loss)
    metrics = {"loss": loss, "accuracy": accuracy}

    return loss, metrics


def step_fn(
    train_state: TrainState,
    batch: Any,
    key: chex.PRNGKey,
    train: bool,
):
    def loss_fn(params, batch, key: chex.PRNGKey):
        logits, _ = train_state.apply_fn(
            {"params": params},
            batch["sensors"],
            batch["sensors_mask"],
            batch["prompt"],
            batch["gen"],
            train=train,
            rngs={"dropout": key},
        )

        return compute_stats(
            pred_logits=logits[..., :-1, :],
            target_tokens=batch["gen"]["tokens"][..., 1:],
            target_mask_loss=batch["gen"]["mask_loss"][..., 1:],
        )

    grad_fn = jax.grad(loss_fn, has_aux=True)

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)
    train_state, info["optimizer"] = train_state.apply_gradients_with_info(grads=grads)

    return train_state, info, key
