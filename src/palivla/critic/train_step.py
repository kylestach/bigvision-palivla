from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import optax

from palivla.critic.train_state import EMATrainState
from palivla.optimizer import components_by_label
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


def train_step(
    train_state: EMATrainState, batch: Data, key: jax.random.PRNGKey, train: bool
) -> Tuple[EMATrainState, Dict[str, Any]]:
    def loss_fn(
        params: Params, batch: Data, rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        rng, target_key, critic_key = jax.random.split(rng, 3)
        _, next_target_value, _ = train_state.apply_fn(
            {"params": train_state.ema_params},
            batch["next_sensors"],
            batch["next_sensors_mask"],
            batch["next_prompt"],
            batch["next_actions"],
            train=train,
            rngs={"dropout": target_key},
        )

        # Maximize over next action options
        next_target_value = jnp.max(next_target_value, axis=-1)

        target_value = batch[
            "rewards"
        ] + train_state.model.discount * next_target_value * (1 - batch["terminals"])

        critic_target_probs = hl_gauss_target(
            q_min=train_state.model.q_min,
            q_max=train_state.model.q_max,
            num_bins=train_state.model.num_critic_bins,
            critic_value=target_value,
            sigma=train_state.model.critic_sigma,
        )

        critic_logits, critic_value, critic_info = train_state.apply_fn(
            {"params": params},
            batch["sensors"],
            batch["sensors_mask"],
            batch["prompt"],
            batch["action"],
            train=train,
            rngs={"dropout": critic_key},
        )

        critic_probs = jax.nn.softmax(critic_logits)
        critic_atoms = jnp.linspace(
            train_state.model.q_min,
            train_state.model.q_max,
            train_state.model.num_critic_bins,
        )
        critic_std = jnp.sqrt(
            jnp.sum(
                critic_probs * (critic_atoms - critic_value[..., None]) ** 2, axis=-1
            )
        )

        loss = optax.softmax_cross_entropy(critic_logits, critic_target_probs).mean()
        return loss, {
            "loss": loss,
            "q_target": jnp.mean(target_value),
            "q_value": jnp.mean(critic_value),
            "q_mse": jnp.mean(jnp.square(critic_value - target_value)),
            "q_std": jnp.mean(critic_std),
        }

    grad_fn = jax.grad(loss_fn, has_aux=True)

    key, dropout_key = jax.random.split(key)
    grads, info = grad_fn(train_state.params, batch, dropout_key)

    train_state = train_state.apply_gradients(
        grads=grads,
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
        | _norm_info(train_state.params, "norm/param")
    )

    return train_state, info, key
