from functools import partial
from typing import Dict, Sequence, Tuple

import chex
import einops
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import repeat
from flax.core import FrozenDict

from big_vision.models.proj.paligemma.gemma_bv import Model as GemmaModel
from big_vision.models.proj.paligemma.paligemma import make_attn_mask
from big_vision.models.vit import Model as ViTModel
from palivla.spec import ModuleSpec
from palivla.typing import Data, Info


def get_default_config():
    return {
        "llm_spec": {
            "__ctor": "big_vision.models.proj.paligemma.gemma_bv.Model",
            "config": {"vocab_size": 257_152},
        },
        "img_spec": {
            "__ctor": "big_vision.models.vit.Model",
            "config": {"variant": "So400m/14", "pool_type": "none", "scan": True},
        },
        "encoder_specs": {},
        "modality_mappings": {"image_primary": "img"},
        "prompt_autoregressive": False,
        "target_key_order": ("image_primary",),
    }


def collect_embeddings(
    embeds: Dict[str, jax.Array],
    embed_masks: Dict[str, jax.Array],
    *,
    target_key_order: Sequence[str] | None = None,
):
    """
    Collects embeddings for a single sample.
    """
    # Make sure we don't have any batch axes.
    _, embed_dim = jax.tree.leaves(embeds)[0].shape
    chex.assert_shape([*embeds.values()], (None, embed_dim))
    chex.assert_shape([*embed_masks.values()], (None,))

    values = jnp.concatenate([embeds[k] for k in target_key_order], axis=0)
    masks = jnp.concatenate([embed_masks[k] for k in target_key_order], axis=0)

    return values, masks


class PaliVLAModel(nn.Module):
    # Specifications for the basic modules
    llm_spec: ModuleSpec
    img_spec: ModuleSpec

    # Encoders by name
    # Should exclude the "image" modality.
    encoder_specs: FrozenDict[str, ModuleSpec]

    # Mapping from modality to embedding
    modality_mappings: FrozenDict[str, str]

    prompt_autoregressive: bool
    target_key_order: Sequence[str]

    def setup(self):
        self.llm: GemmaModel = self.llm_spec.instantiate(name="llm")
        self.image: ViTModel = self.img_spec.instantiate(
            name="img", num_classes=self.llm.embdim
        )

        def _encode_image(data, mask, *, train: bool = False):
            # Normalize the image to be in the range [-1, 1]
            data = data / 127.5 - 1

            # Shim to process the stack dimension as a batch dimension
            orig_shape = data.shape
            num_batch_dims = len(orig_shape) - 3
            mask = jnp.any(mask, axis=tuple(range(num_batch_dims, mask.ndim)))

            if len(orig_shape) == 5:
                data = einops.rearrange(data, "... h w c -> (...) h w c")
                mask = einops.rearrange(mask, "... -> (...)")

            result, info = self.image(data, train=train)
            mask = einops.repeat(mask, "... -> ... k", k=result.shape[1])
            if len(orig_shape) == 5:
                result = einops.rearrange(
                    result, "(b k) t d -> b (k t) d", b=orig_shape[0]
                )
                mask = einops.rearrange(mask, "(b k) t -> b (k t)", b=orig_shape[0])

            return result, mask, info

        self.encoders = {
            "img": _encode_image,
        } | {
            name: spec.instantiate(name=name)
            for name, spec in self.encoder_specs.items()
        }

        self.modality_start_tokens = {
            modality: self.param(
                name=f"start_{modality}",
                init_fn=nn.linear.default_embed_init,
                shape=(
                    1,
                    self.llm.embdim,
                ),
            )
            for modality in self.modality_mappings.keys()
        }

    def embed_modalities(
        self, data: Data, masks: Dict[str, jax.Array], *, train: bool = False
    ) -> Tuple[Data, Data, Info]:
        """
        Get embeddings for all modalities.
        """

        embeds = {}
        embed_masks = {}
        info = {}

        for modality, encoder_name in self.modality_mappings.items():
            # Embed the data
            encoder = self.encoders[encoder_name]

            embed, mask, m_info = encoder(data[modality], masks[modality], train=train)
            m_info["embed"] = embed

            # If only one embedding was produced, expand it to match the sequence length
            if embed.ndim == 2:
                embed = jnp.expand_dims(embed, axis=1)

            # Add the modality start tokens
            start_token = einops.repeat(
                self.modality_start_tokens[modality],
                "1 e -> b 1 e",
                b=data[modality].shape[0],
            )
            embed = jnp.concatenate([start_token, embed], axis=1)

            # Create a mask to match the shape of the embeddings
            # Add one to the length to account for the modality start tokens
            if mask.ndim < embed.ndim - 1:
                mask = repeat(
                    mask,
                    "... -> ... t",
                    t=embed.shape[1],
                )
            else:
                mask = jnp.concatenate([mask[:, :1], mask], axis=1)
            chex.assert_shape(mask, embed.shape[:-1])

            info.update({f"{modality}/{k}": v for k, v in m_info.items()})

            embeds[modality] = embed
            embed_masks[modality] = mask

        batch_size = jax.tree.leaves(data)[0].shape[0]
        embed_dim = self.llm.embdim
        chex.assert_shape(list(embeds.values()), (batch_size, None, embed_dim))

        return embeds, embed_masks, info

    def embed_sensors(
        self,
        sensors: Data,
        sensors_mask: Data,
        *,
        train: bool = False,
    ):
        sensors_embeds, sensors_masks, info = self.embed_modalities(
            sensors, sensors_mask, train=train
        )

        packed_embeds, packed_masks = jax.vmap(
            partial(collect_embeddings, target_key_order=self.target_key_order)
        )(
            sensors_embeds,
            sensors_masks,
        )

        return packed_embeds, packed_masks, info

    def embed_sensors_and_text(
        self,
        sensors: Data,
        sensors_mask: Data,
        prompt_seq: Data,
        gen_seq: Data | None,
        *,
        train: bool = False,
    ):
        sequence = prompt_seq
        if gen_seq is None:
            sequence = prompt_seq
        else:
            sequence = {
                "tokens": jnp.concatenate(
                    [prompt_seq["tokens"], gen_seq["tokens"]], axis=1
                ),
                "mask_ar": jnp.concatenate(
                    [prompt_seq["mask_ar"], gen_seq["mask_ar"]], axis=1
                ),
                "mask": jnp.concatenate([prompt_seq["mask"], gen_seq["mask"]], axis=1),
            }

        sensors_embeds, sensors_masks, sensors_info = self.embed_sensors(
            sensors, sensors_mask, train=train
        )
        text_embeds, text_info = self.embed_text(sequence["tokens"], train=train)

        all_embeds = jnp.concatenate([sensors_embeds, text_embeds], axis=1)
        all_masks = jnp.concatenate([sensors_masks, sequence["mask"]], axis=1)
        all_masks_ar = jnp.concatenate(
            [jnp.zeros_like(sensors_masks), sequence["mask_ar"]], axis=1
        )

        prompt_end = sensors_embeds.shape[1] + prompt_seq["tokens"].shape[1]

        return all_embeds, all_masks, all_masks_ar, sensors_info | text_info, prompt_end

    def __call__(
        self,
        sensors: Data,
        sensors_mask: Data,
        prompt_seq: Data,
        gen_seq: Data,
        *,
        train: bool = False,
    ):
        # Concatenate the prompt/gen sequences
        embeds, masks, masks_ar, info, prompt_end = self.embed_sensors_and_text(
            sensors, sensors_mask, prompt_seq, gen_seq, train=train
        )

        positions = jnp.cumsum(masks, axis=1) - 1
        attn_mask = make_attn_mask(masks, masks_ar)
        _, llm_info = self.llm(embeds, mask=attn_mask, train=train, positions=positions)

        info = llm_info | {
            f"{modality}_info": m_info for modality, m_info in info.items()
        }

        # Get only the logits for the text tokens, which should be the last `n` tokens
        pre_logits = llm_info["pre_logits"][..., prompt_end:, :]
        logits = self.llm.compute_logits(pre_logits, train=train)

        info["text_pre_logits"] = pre_logits
        info["text_logits"] = logits
        info["text_tokens"] = jnp.argmax(logits, axis=-1)

        return logits, info

    def embed_text(self, tokens, train=False):
        out = {}
        ztxt = out["llm/ztxt"] = self.llm.embed_tokens(tokens, train=train)
        return ztxt, out

    def prefill_cache(
        self,
        x: jax.Array,
        input_mask: jax.Array,
        mask_ar: jax.Array,
        *,
        cache_size,
    ):
        """Initializes decoding cache with `x` [B, N, E] as prompt."""
        if hasattr(self.llm, "prefill_cache"):
            attn_mask = make_attn_mask(input_mask, mask_ar)
            return self.llm.prefill_cache(
                x, input_mask, attn_mask, cache_size=cache_size
            )
        else:
            return self._fallback_prefill_cache(x, input_mask, mask_ar, cache_size)

    def extend_cache(self, x):
        """Advances decoding cache with `x` [B, 1, E]."""
        if hasattr(self.llm, "prefill_cache"):
            return self.llm.extend_cache(x)
        else:
            return self._fallback_extend_cache(x)
