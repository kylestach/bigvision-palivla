from importlib.machinery import ModuleSpec
from typing import Any

import jax
import optax
from flax.struct import dataclass, field

from palivla.components.train_state import ShardingMetadata, TrainState
from palivla.spec import OptimizerSpec
from palivla.typing import Params


def initialize_train_state(
    example_batch: Any,
    model_spec: ModuleSpec,
    optimizer_spec: OptimizerSpec,
    sharding: ShardingMetadata,
    rng: jax.Array,
):
    model = model_spec.instantiate()
    tx = optimizer_spec.instantiate()

    def init_train_state(example_batch: Any):
        params = model.lazy_init(
            rng,
            *example_batch,
            train=False,
        )["params"]

        return EMATrainState.create(
            apply_fn=model.apply,
            model_spec=model_spec,
            optimizer_spec=optimizer_spec,
            model=model,
            tx=tx,
            params=params,
            ema_params=params,
            ema_rate=model.target_ema_rate,
        )

    _init_train_state = sharding.mesh.sjit(
        init_train_state,
        out_shardings=sharding.model_sharding_rule,
    )
    # Add the example batch in at the end so it doesn't get treated as a
    return _init_train_state(example_batch)


@dataclass
class EMATrainState(TrainState):
    ema_params: Params
    ema_rate: float = field(pytree_node=False)

    def apply_gradients(self, *, grads, **kwargs):
        ema_params = optax.incremental_update(
            old_tensors=self.ema_params,
            new_tensors=self.params,
            step_size=self.ema_rate,
        )
        return super().apply_gradients(grads=grads, ema_params=ema_params, **kwargs)

    @classmethod
    def initialize(
        cls,
        *,
        rng: jax.Array,
        sharding: ShardingMetadata,
        model_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        example_batch: Any,
    ):
        return initialize_train_state(
            example_batch,
            model_spec,
            optimizer_spec,
            sharding,
            rng,
        )
