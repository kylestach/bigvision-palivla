from typing import Literal
import chex
import jax
import jax.numpy as jnp

from big_vision.utils import Registry
from palivla.model_components import ModelComponents
from palivla.typing import Params


@Registry.register("load.paligemma_weights")
def load_paligemma_weights(
    model: ModelComponents, *, path: str | None, param_dtype: jnp.dtype = jnp.float32
):
    from big_vision.models.proj.paligemma.paligemma import load as load_paligemma
    from ml_collections import FrozenConfigDict

    # TODO(Kyle): Allow loading other variants of PaliGemma
    base_model_cfg = FrozenConfigDict(
        {
            "llm": {"vocab_size": 257_152},
            "img": {
                "variant": "So400m/14",
                "pool_type": "none",
                "scan": True,
            },
        }
    )
    if path is None:
        base_params = {}
    else:
        base_params = load_paligemma(
            None,
            path,
            base_model_cfg,
        )

    def _replace_single_param(
        param,
        load_param,
        mismatch_strategy: Literal["overwrite", "skip", "subarray", "error"],
    ):
        if load_param is None:
            return param

        if load_param.shape == param.shape:
            return load_param

        if mismatch_strategy == "overwrite":
            return load_param
        elif mismatch_strategy == "skip":
            return param
        elif mismatch_strategy == "subarray":
            return jax.lax.dynamic_update_slice(
                param.astype(param_dtype),
                load_param.astype(param_dtype),
                (0,) * param.ndim,
            )
        elif mismatch_strategy == "error":
            raise ValueError(
                f"Mismatch in shape between param {param.shape} and load_param {load_param.shape}"
            )
        else:
            raise ValueError(f"Invalid mismatch strategy: {mismatch_strategy}")

    def _replace_params_fn(params: Params, param_replacements: Params, path_str=""):
        if isinstance(param_replacements, dict):
            return {
                k: _replace_params_fn(
                    params.get(k, None),
                    param_replacements.get(k, None),
                    f"{path_str}/{k}",
                )
                for k in set(params.keys()) | set(param_replacements.keys())
            }

        if path_str in ["/llm/embedder/input_embedding"]:
            strategy = "subarray"
            jax.debug.print(f"Replacing param {path_str} with subarray strategy")
        else:
            strategy = "error"

        try:
            return _replace_single_param(
                params,
                param_replacements,
                strategy,
            ).astype(param_dtype)
        except ValueError as e:
            raise ValueError(f"Error replacing param {path_str}: {e}")

    replace_params_fn = model.sharding.mesh.sjit(
        _replace_params_fn,
        in_shardings=(model.sharding.model_sharding_rule, None),
        out_shardings=model.sharding.model_sharding_rule,
        donate_argnums=(0,),
    )

    model.train_state = model.train_state.replace(
        params=replace_params_fn(model.train_state.params, base_params)
    )
