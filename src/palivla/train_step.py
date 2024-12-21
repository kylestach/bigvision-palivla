from typing import Any
import jax
import jax.numpy as jnp
import chex
import optax
from flax.training.train_state import TrainState
from palivla.optimizer import components_by_label


def smooth_nll_loss(logits, labels, sigma, base_action_token, action_vocab_size):
    # Normal PDF with mean `label` and std `sigma`
    bin_cutoffs = jnp.arange(action_vocab_size - 1) + 0.5
    cdf = jax.scipy.stats.norm.cdf(
        bin_cutoffs, labels[..., None] - base_action_token, sigma
    )
    cdf = jnp.concatenate(
        [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., :1])], axis=-1
    )
    label_probs = jnp.diff(cdf, axis=-1)

    logits = jax.nn.log_softmax(logits, axis=-1)
    entropy = -jnp.sum(jax.scipy.special.xlogy(label_probs, label_probs), axis=-1)
    return (
        -jnp.sum(
            label_probs
            * logits[..., base_action_token : base_action_token + action_vocab_size],
            axis=-1,
        )
        - entropy
    )


def get_action_tokens(
    pred_logits, tokens, num_action_tokens: int, begin_of_action_token: int
):
    _get_action_tokens = jax.vmap(
        lambda x, i: jax.lax.dynamic_slice(x, (i,), (num_action_tokens,))
    )

    action_token_starts = jnp.argmax(tokens == begin_of_action_token, axis=-1) + 1
    pred_tokens = jnp.argmax(pred_logits, axis=-1)
    pred_action_tokens = _get_action_tokens(pred_tokens, action_token_starts)
    pred_action_logits = jax.vmap(_get_action_tokens, in_axes=(-1, None), out_axes=-1)(
        pred_logits, action_token_starts
    )
    gt_action_tokens = _get_action_tokens(tokens, action_token_starts)

    return {
        "pred_action_tokens": pred_action_tokens,
        "pred_action_logits": pred_action_logits,
        "gt_action_tokens": gt_action_tokens,
    }


def compute_stats(
    *,
    pred_logits,
    target_tokens,
    target_mask_loss,
):
    loss = jnp.mean(
        target_mask_loss
        * optax.softmax_cross_entropy_with_integer_labels(pred_logits, target_tokens)
    ) / jnp.mean(target_mask_loss)

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
    updates, opt_state = train_state.tx.update(
        grads, train_state.opt_state, params=train_state.params
    )
    params = optax.apply_updates(train_state.params, updates)

    train_state = train_state.replace(
        params=params, opt_state=opt_state, step=train_state.step + 1
    )

    info = info | train_state.opt_state.hyperparams

    def _norm_info(values, prefix):
        components = components_by_label(values)
        result = {f"{prefix}_{k}": optax.global_norm(v) for k, v in components.items()}
        result[prefix] = jnp.sqrt(sum(x**2 for x in result.values()))
        return result

    info = (
        info
        | _norm_info(grads, "norm/grad")
        | _norm_info(updates, "norm/update")
        | _norm_info(train_state.params, "norm/param")
    )

    return train_state, info, key
