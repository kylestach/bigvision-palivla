import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat
from flax.struct import dataclass

from big_vision.models.proj.paligemma.paligemma import make_attn_mask
from palivla.components.model import PaliVLAModel
from palivla.typing import Data


def make_attn_mask_for_critic(input_mask, input_mask_ar, n_actions):
    """Returns attention mask bool[B, N, N] to use in transformer for a critic where many actions are fed in. Each action will be able to attend to all input tokens, but the actions will not be able to attend to each other.

    Example (batch dimension omitted):
    input_mask    = [1, 1, 1, 1, 1, 0]
    input_mask_ar = [0, 0, 0, 1, 1, 0]
    n_actions     = 2

    Resulting mask:
    [[1 1 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [1 1 1 0 0 0 0 0]
     [1 1 1 1 0 0 0 0]
     [1 1 1 1 1 0 0 0]
     [0 0 0 0 0 0 0 0]
     [1 1 1 1 1 0 1 0]
     [1 1 1 1 1 0 0 1]]
    """
    batch_size, n_inputs = input_mask.shape
    input_to_input = make_attn_mask(input_mask, input_mask_ar)
    action_to_input = repeat(input_mask, "b n -> b n_actions n", n_actions=n_actions)
    action_to_action = jnp.eye(n_actions, dtype=jnp.bool_)[None].repeat(
        batch_size, axis=0
    )
    input_to_action = jnp.zeros((batch_size, n_inputs, n_actions), dtype=jnp.bool_)

    is_new_position = jnp.concatenate(
        [
            input_mask,
            jnp.ones([batch_size, 1], dtype=jnp.bool_),
            jnp.zeros([batch_size, n_actions - 1], dtype=jnp.bool_),
        ],
        axis=-1,
    )
    positions = (
        jnp.cumsum(
            is_new_position,
            axis=1,
        )
        - 1
    )

    return (
        jnp.concatenate(
            [
                jnp.concatenate([input_to_input, input_to_action], axis=2),
                jnp.concatenate([action_to_input, action_to_action], axis=2),
            ],
            axis=1,
        ),
        positions,
    )


@dataclass
class DistributionalValue:
    logits: jnp.ndarray
    atoms: jnp.ndarray

    @classmethod
    def from_hl(
        cls, mean: jnp.ndarray, qmin: float, qmax: float, num_bins: int, sigma: float
    ):
        xs = jnp.linspace(qmin, qmax, num_bins - 1)
        cdf = jax.scipy.stats.norm.cdf(xs, mean[..., None], sigma)
        cdf = jnp.concatenate(
            [jnp.zeros_like(cdf[..., :1]), cdf, jnp.ones_like(cdf[..., -1:])], axis=-1
        )
        probs = jnp.diff(cdf, axis=-1)
        atoms = jnp.linspace(qmin, qmax, num_bins)
        logits = jnp.log(jnp.clip(probs, a_min=1e-8, a_max=None))
        return cls(logits=logits, atoms=atoms)

    def mean(self):
        probs = jax.nn.softmax(self.logits, axis=-1)
        return jnp.sum(probs * self.atoms, axis=-1)

    def std(self):
        probs = jax.nn.softmax(self.logits, axis=-1)
        mean = jnp.sum(probs * self.atoms, axis=-1, keepdims=True)
        var = jnp.sum(probs * (self.atoms - mean) ** 2, axis=-1)
        return jnp.sqrt(var)


class PaliVLACritic(PaliVLAModel):
    num_critic_bins: int = 256
    q_min: float = 0.0
    q_max: float = 1.0
    critic_sigma: float = 0.03
    discount: float = 0.99
    target_ema_rate: float = 0.005
    aux_mc_prediction: bool = False
    aux_loss_weight: float | None = None

    def setup(self):
        super().setup()
        self.action_proj = nn.Dense(self.llm.embdim)
        self.critic_head = nn.Dense(self.num_critic_bins)

        if self.aux_mc_prediction:
            self.critic_head_mc = nn.Dense(self.num_critic_bins)

    def __call__(
        self,
        sensors: Data,
        sensors_mask: Data,
        prompt_seq: Data,
        actions: Data,
        *,
        train: bool = False,
    ):
        # Concatenate the prompt/gen sequences
        embeds, masks, masks_ar, info, prompt_end = self.embed_sensors_and_text(
            sensors, sensors_mask, prompt_seq, None, train=train
        )

        # Action embeddings
        # Actions should have shape [batch, n_counterfactuals, action_dim] or [batch, action_dim]
        actions_squeeze = False
        if actions.ndim == 2:
            actions = actions[:, None, :]
            actions_squeeze = True
        chex.assert_rank(actions, 3)
        actions_embeds = self.action_proj(actions)

        attn_mask, positions = make_attn_mask_for_critic(
            masks, masks_ar, actions.shape[1]
        )

        embeds = jnp.concatenate([embeds, actions_embeds], axis=1)
        _, llm_info = self.llm(embeds, mask=attn_mask, train=train, positions=positions)

        info = llm_info | {
            f"{modality}_info": m_info for modality, m_info in info.items()
        }

        pre_logits = llm_info["pre_logits"][..., prompt_end:, :]
        critic_logits = self.critic_head(pre_logits)

        info["text_pre_logits"] = pre_logits
        info["text_logits"] = critic_logits
        info["text_tokens"] = jnp.argmax(critic_logits, axis=-1)

        if actions_squeeze:
            critic_logits = jnp.squeeze(critic_logits, axis=1)

        atoms = jnp.linspace(self.q_min, self.q_max, self.num_critic_bins)

        info = {}

        # Auxiliary head for MC regression
        if self.aux_mc_prediction:
            mc_logits = self.critic_head_mc(pre_logits)
            if actions_squeeze:
                mc_logits = jnp.squeeze(mc_logits, axis=1)
            info["mc_head"] = DistributionalValue(
                logits=mc_logits,
                atoms=atoms,
            )

        return DistributionalValue(logits=critic_logits, atoms=atoms), info
