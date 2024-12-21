from functools import partial
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict, FrozenConfigDict
import optax

from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
from flax import linen as nn

from palivla.tokenizer import Tokenizer
from palivla.spec import ModuleSpec

model_config = {
    "llm": {"vocab_size": 257_152},
    "img": {
        "variant": "So400m/14",
        "pool_type": "none",
        "scan": True,
    },
}


def get_model_spec(**kwargs):
    return ModuleSpec.create(
        paligemma.Model,
        model_config | kwargs,
    )


def load_pretrained_params(
    checkpoint_path: str,
    dtype: jnp.dtype = jnp.float32,
):
    params = paligemma.load(None, checkpoint_path, FrozenConfigDict(model_config))
    params = jax.tree.map(lambda x: x.astype(dtype), params)

    return params


def get_decode_fn(model: nn.Module, tokenizer: Tokenizer):
    decode_fn = predict_fns.get_all(model)["decode"]
    decode = partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.config.eos_token,
    )
    return decode


def load_model_params_decode(config: ConfigDict, tokenizer: Tokenizer):
    from big_vision.models.proj.paligemma import paligemma
    from big_vision.trainers.proj.paligemma import predict_fns

    # Define model
    model = paligemma.Model(**FrozenConfigDict(model_config))

    # Load params - this can take up to 1 minute in T4 colabs.
    params = paligemma.load(None, config.model_path, FrozenConfigDict(model_config))

    # Change params to fp32
    params = jax.tree.map(lambda x: x.astype(jnp.float32), params)

    decode_fn = predict_fns.get_all(model)["decode"]
    decode = partial(
        decode_fn,
        devices=jax.devices(),
        eos_token=tokenizer.config.eos_token,
    )

    return model, params, decode


def adamw_cosine_warmup(
    *,
    learning_rate,
    warmup_steps,
    total_steps,
    global_norm,
    weight_decay,
    b1=0.9,
    b2=0.999,
):
    @optax.inject_hyperparams
    def _make_optimizer(lr):
        return optax.chain(
            optax.clip_by_global_norm(global_norm),
            optax.adamw(lr, weight_decay=weight_decay, b1=b1, b2=b2),
        )

    return _make_optimizer(
        optax.warmup_cosine_decay_schedule(
            0, learning_rate, warmup_steps, total_steps - warmup_steps
        )
    )
