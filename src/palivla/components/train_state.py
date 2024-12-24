from os import PathLike
from typing import Any, Optional

import cloudpickle
import jax.experimental.multihost_utils
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
from flax import linen as nn
from flax.struct import field
from flax.training.train_state import TrainState as FlaxTrainState
from scalax.sharding import MeshShardingHelper, PartitionSpec, ShardingRule

from palivla.optimizer import components_by_label
from palivla.spec import ModuleSpec, OptimizerSpec


class ShardingMetadata:
    mesh: MeshShardingHelper
    model_sharding_rule: ShardingRule | PartitionSpec

    def __init__(
        self,
        mesh: MeshShardingHelper,
        model_sharding_rule: ShardingRule | PartitionSpec,
    ):
        self.mesh = mesh
        self.model_sharding_rule = model_sharding_rule


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
            example_batch["sensors"],
            example_batch["sensors_mask"],
            example_batch["prompt"],
            example_batch["gen"],
            train=False,
        )["params"]

        return TrainState.create(
            apply_fn=model.apply,
            model_spec=model_spec,
            optimizer_spec=optimizer_spec,
            model=model,
            tx=tx,
            params=params,
        )

    _init_train_state = sharding.mesh.sjit(
        init_train_state,
        out_shardings=sharding.model_sharding_rule,
    )
    # Add the example batch in at the end so it doesn't get treated as a
    return _init_train_state(example_batch)


class TrainState(FlaxTrainState):
    model_spec: ModuleSpec = field(pytree_node=False)
    model: nn.Module = field(pytree_node=False)
    optimizer_spec: Optional[OptimizerSpec] = field(pytree_node=False)

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

    def save_static(self, path: PathLike):
        with tf.io.gfile.GFile(path / "model_spec.json", "w") as f:
            f.write(self.model_spec.to_json())
        with tf.io.gfile.GFile(path / "optimizer_spec.json", "w") as f:
            f.write(self.optimizer_spec.to_json())

        # Write example batch as pickle
        with tf.io.gfile.GFile(path / "example_batch.pkl", "wb") as f:
            cloudpickle.dump(self.example_batch, f)

    @classmethod
    def load_static(
        cls,
        path: PathLike,
        *,
        sharding: ShardingMetadata,
    ):
        with tf.io.gfile.GFile(path / "model_spec.json", "r") as f:
            model_spec = ModuleSpec.from_json(f.read())
        with tf.io.gfile.GFile(path / "optimizer_spec.json", "r") as f:
            optimizer_spec = OptimizerSpec.from_json(f.read())
        with tf.io.gfile.GFile(path / "example_batch.pkl", "rb") as f:
            example_batch = cloudpickle.load(f)

        # Initialize the model
        return initialize_train_state(
            example_batch,
            model_spec,
            optimizer_spec,
            sharding,
            rng=jax.random.PRNGKey(0),
        )

    def save_state(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        checkpoint_manager.save(step, ocp.args.StandardSave(self))

    def load_state(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        return checkpoint_manager.restore(step, ocp.args.StandardRestore(self))

    def get_params(self, *, use_ema_params: bool = False):
        if use_ema_params:
            return self.opt_state["ema"]
        return self.params

    def apply_gradients_with_info(self, *, grads: jax.Array, **kwargs):
        updates, opt_state = self.tx.update(grads, self.opt_state, params=self.params)
        params = optax.apply_updates(self.params, updates)
        self.apply_gradients

        def _norm_info(values, prefix):
            components = components_by_label(values)
            result = {
                f"{prefix}_{k}": optax.global_norm(v) for k, v in components.items()
            }
            result[prefix] = jnp.sqrt(sum(x**2 for x in result.values()))
            return result

        info = (
            self.opt_state["optimizer"].hyperparams
            | _norm_info(grads, "grad_norm")
            | _norm_info(updates, "update_norm")
            | _norm_info(self.params, "param_norm")
        )

        return (
            self.replace(
                params=params, opt_state=opt_state, step=self.step + 1, **kwargs
            ),
            info,
        )
