from functools import partial
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import optax
from palivla.components.train_state import TrainState
from palivla.critic.vla_critic import PaliVLACritic
from palivla.typing import Data, Params


def hl_gauss_target(
    q_min: float, q_max: float, num_bins: int, critic_value: float, sigma: float
) -> jnp.ndarray:
    xs = jnp.linspace(q_min, q_max, num_bins - 1)
    cdf = jax.scipy.stats.norm.cdf(xs, critic_value[..., None], sigma)
    cdf = jnp.concatenate(
        [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., -1:])], axis=-1
    )
    return jnp.diff(cdf, axis=-1)


def loss_fn(
    params: Params,
    batch: Data,
    rng: jax.random.PRNGKey,
    *,
    train: bool,
    regress_to_mc_returns: bool = False,
    train_with_sarsa: bool = False,
    zero_out_actions: bool = False,
    model: PaliVLACritic,
    ema_params: Params,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    rng, target_key, critic_key = jax.random.split(rng, 3)

    action = batch["action"]

    if regress_to_mc_returns:
        target_value = batch["mc_return"]
    else:
        if train_with_sarsa:
            next_action = batch["next_action"]
        elif zero_out_actions:
            next_action = jnp.zeros_like(batch["next_action"])
            action = jnp.zeros_like(action)
        else:
            next_action = batch["counterfactual_next_actions"]

        next_target_value_dist, _ = model.apply(
            {"params": ema_params},
            batch["next_sensors"],
            batch["next_sensors_mask"],
            batch["next_prompt"],
            next_action,
            train=train,
            rngs={"dropout": target_key},
        )
        next_target_value = next_target_value_dist.mean()

        # Maximize over next action options
        if next_target_value.ndim == 2:
            chex.assert_shape(
                next_target_value,
                (
                    batch["rewards"].shape[0],
                    batch["counterfactual_next_actions"].shape[1],
                ),
            )
            next_target_value = jnp.max(next_target_value, axis=1)

        chex.assert_shape(next_target_value, (batch["rewards"].shape[0],))
        target_value = batch["rewards"] + model.discount * next_target_value * (
            batch["td_mask"]
        )
    
    chex.assert_shape(target_value, (batch["rewards"].shape[0],))

    critic_target_probs = hl_gauss_target(
        q_min=model.q_min,
        q_max=model.q_max,
        num_bins=model.num_critic_bins,
        critic_value=target_value,
        sigma=model.critic_sigma,
    )

    critic_distribution, critic_info = model.apply(
        {"params": params},
        batch["sensors"],
        batch["sensors_mask"],
        batch["prompt"],
        action,
        train=train,
        rngs={"dropout": critic_key},
    )

    critic_value = critic_distribution.mean()
    critic_std = critic_distribution.std()

    loss = optax.softmax_cross_entropy(critic_distribution.logits, critic_target_probs).mean()
    info = {
        "q_target": jnp.mean(target_value),
        "q_value": jnp.mean(critic_value),
        "q_mse": jnp.mean(jnp.square(critic_value - target_value)),
        "q_std": jnp.mean(critic_std),
        "q - mc": jnp.mean(critic_value - batch["mc_return"]),
    }

    if regress_to_mc_returns:
        info["loss.mc"] = loss
    else:
        info["loss.td"] = loss

    if model.aux_mc_prediction:
        chex.assert_shape(target_value, (batch["rewards"].shape[0],))

        aux_mc_head_distribution = critic_info["mc_head"]
        mc_target_probs = hl_gauss_target(
            q_min=model.q_min,
            q_max=model.q_max,
            num_bins=model.num_critic_bins,
            critic_value=batch["mc_return"],
            sigma=model.critic_sigma,
        )
        chex.assert_equal_shape([aux_mc_head_distribution.logits, mc_target_probs])
        aux_loss = optax.softmax_cross_entropy(aux_mc_head_distribution.logits, mc_target_probs).mean()
        info["loss.mc"] = aux_loss
        info["q_value_aux_mc"] = jnp.mean(aux_mc_head_distribution.mean())

        loss = loss + model.aux_loss_weight * aux_loss

    info["loss.total"] = loss

    return loss, info


def train_step(
    train_state: TrainState,
    batch: Data,
    key: jax.random.PRNGKey,
    train: bool,
    regress_to_mc_returns: bool = False,
    train_with_sarsa: bool = False,
    zero_out_actions: bool = False,
) -> Tuple[TrainState, Dict[str, Any]]:
    grad_fn = jax.grad(
        partial(
            loss_fn,
            model=train_state.model,
            train=train,
            ema_params=train_state.opt_state["ema"],
            regress_to_mc_returns=regress_to_mc_returns,
            train_with_sarsa=train_with_sarsa,
            zero_out_actions=zero_out_actions,
        ),
        has_aux=True,
    )

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)

    train_state, info["optimizer"] = train_state.apply_gradients_with_info(grads=grads)

    return train_state, info, key
