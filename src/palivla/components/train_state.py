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


def initialize_train_state_fn(
    example_batch: Any,
    model_spec: ModuleSpec,
    optimizer_spec: OptimizerSpec,
    sharding: ShardingMetadata,
    rng: jax.Array,
):
    model = model_spec.instantiate()
    tx = optimizer_spec.instantiate()

    def init_train_state():
        params = model.lazy_init(
            rng,
            *example_batch,
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

    return sharding.mesh.sjit(
        init_train_state,
        out_shardings=sharding.model_sharding_rule,
    )


def initialize_train_state(
    example_batch: Any,
    model_spec: ModuleSpec,
    optimizer_spec: OptimizerSpec,
    sharding: ShardingMetadata,
    rng: jax.Array,
):
    return initialize_train_state_fn(
        example_batch,
        model_spec,
        optimizer_spec,
        sharding,
        rng,
    )()


def initialize_abstract_train_state(
    example_batch: Any,
    model_spec: ModuleSpec,
    optimizer_spec: OptimizerSpec,
    sharding: ShardingMetadata,
    rng: jax.Array,
):
    return jax.eval_shape(
        initialize_train_state_fn(
            example_batch,
            model_spec,
            optimizer_spec,
            sharding,
            rng,
        )
    )


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
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "model_spec.json"), "w") as f:
            f.write(self.model_spec.to_json())
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "optimizer_spec.json"), "w") as f:
            f.write(self.optimizer_spec.to_json())

    @classmethod
    def load_static(
        cls,
        path: PathLike,
        *,
        sharding: ShardingMetadata,
        example_batch: Any,
        weights_only: bool = False,
    ):
        with tf.io.gfile.GFile(tf.io.gfile.join(path, "model_spec.json"), "r") as f:
            model_spec = ModuleSpec.from_json(f.read())

        if weights_only:
            optimizer_spec = OptimizerSpec(optax.set_to_zero, {})
        else:
            with tf.io.gfile.GFile(
                tf.io.gfile.join(path, "optimizer_spec.json"), "r"
            ) as f:
                optimizer_spec = OptimizerSpec.from_json(f.read())

        # Initialize the model
        abstract_train_state = initialize_abstract_train_state(
            example_batch,
            model_spec,
            optimizer_spec,
            sharding,
            rng=jax.random.PRNGKey(0),
        )

        # Shard the abstract train state
        if isinstance(sharding.model_sharding_rule, ShardingRule):
            shardings = sharding.model_sharding_rule.apply(abstract_train_state)
        elif isinstance(sharding.model_sharding_rule, PartitionSpec):
            shardings = jax.tree.map(
                lambda x: jax.sharding.NamedSharding(
                    sharding.mesh.mesh, sharding.model_sharding_rule
                ),
                abstract_train_state,
            )
        else:
            raise ValueError(
                "Sharding rule must be either ShardingRule or PartitionSpec"
            )

        return jax.tree.map(
            lambda x, s: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=s),
            abstract_train_state,
            shardings,
        )

    def save_state(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        checkpoint_manager.save(step, args=ocp.args.StandardSave(self))

    def load_state(
        self,
        step: int,
        checkpoint_manager: ocp.CheckpointManager,
        *,
        weights_only: bool = False,
    ):
        if weights_only:
            restore_args = jax.tree.map(
                lambda x: ocp.ArrayRestoreArgs(sharding=x.sharding), self
            )
            return checkpoint_manager.restore(
                step,
                args=ocp.args.PyTreeRestore(
                    self,
                    restore_args=restore_args,
                    transforms={
                        r"params/([a-z]+)": ocp.Transform(original_key=r"params/\1")
                    },
                ),
            )
        else:
            return checkpoint_manager.restore(step, args=ocp.args.StandardRestore(self))

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
