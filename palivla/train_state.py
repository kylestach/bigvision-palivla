from functools import cached_property, partial
from typing import Dict, Optional, Sequence, Type
from flax.training.train_state import TrainState as FlaxTrainState
from flax.struct import dataclass, field
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze
from ml_collections.config_dict.config_dict import _is_type_safety_violation
import jax
import jax.experimental
import jax.experimental.multihost_utils
import jax.numpy as jnp
import orbax.checkpoint as ocp
from palivla import model
from scalax.sharding import (
    MeshShardingHelper,
    ShardingRule,
    PartitionSpec,
)
import optax
import numpy as np
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer

from palivla.eval_step import compute_eval_stats, compute_gen_stats
from palivla.model import load_from_pretrained
from palivla.spec import ModuleSpec, OptimizerSpec, restore_gluon_module
from palivla.tokenizer import Tokenizer
# from palivla.preprocess.sentencepiece_model_pb2 import ModelProto as SentencepieceModelProto
from palivla.sentencepiece_model_pb2 import ModelProto as SentencepieceModelProto
from palivla.train_step import TrainingBatch, step_fn
from palivla.types import Params, RolloutBatch
from palivla.predict_fns import _decode
from palivla.utils import merge_params

def fmt_handler_name(name, suffix, is_legacy=False):
    return f"{name}/{suffix}" if is_legacy else f"{name}_{suffix}"

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


class TrainState(FlaxTrainState):
    name: str = field(pytree_node=False)
    model_spec: ModuleSpec = field(pytree_node=False)
    model: nn.Module = field(pytree_node=False)
    optimizer_spec: Optional[OptimizerSpec] = field(pytree_node=False)

    sharding_metadata: ShardingMetadata | None = field(pytree_node=False)

    @classmethod
    def get_checkpoint_handlers(cls, name: str, is_legacy: bool = False):
        return {
            fmt_handler_name(name, "model_spec", is_legacy=is_legacy): ocp.JsonCheckpointHandler(),
            fmt_handler_name(name, "model_params", is_legacy=is_legacy): ocp.StandardCheckpointHandler(),
            fmt_handler_name(name, "optimizer_spec", is_legacy=is_legacy): ocp.JsonCheckpointHandler(),
            fmt_handler_name(name, "opt_state", is_legacy=is_legacy): ocp.StandardCheckpointHandler(),
        }

    @classmethod
    def create(
        cls,
        name: str,
        model_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        rng: jax.Array,
        batch_spec: FrozenDict[str, jax.ShapeDtypeStruct],
        sharding_metadata: ShardingMetadata | None = None,
    ):
        model = model_spec.instantiate()
        optimizer = optimizer_spec.instantiate()

        def init_fn(rng):
            params = model.lazy_init(rng, batch_spec)
            opt_state = optimizer.init(params)

            return cls(
                name=name,
                params=params,
                opt_state=opt_state,
                model_spec=model_spec,
                optimizer_spec=optimizer_spec,
                sharding_metadata=sharding_metadata,
                model=model,
                step=0,
                apply_fn=model.apply,
                tx=optimizer,
            )

        if sharding_metadata is not None:
            init_fn = sharding_metadata.mesh.sjit(
                init_fn,
                out_shardings=sharding_metadata.model_sharding_rule,
            )
        else:
            init_fn = jax.jit(init_fn)

        return init_fn(rng)

    @classmethod
    def with_params(
        cls,
        name: str,
        params: FrozenDict,
        model_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        sharding_metadata: ShardingMetadata | None = None,
    ):
        model = model_spec.instantiate()
        optimizer = optimizer_spec.instantiate()

        def init_fn(params):
            opt_state = optimizer.init(params)

            return cls(
                name=name,
                params=params,
                opt_state=opt_state,
                model_spec=model_spec,
                optimizer_spec=optimizer_spec,
                sharding_metadata=sharding_metadata,
                model=model,
                step=0,
                apply_fn=model.apply,
                tx=optimizer,
            )

        if sharding_metadata is not None:
            init_fn = sharding_metadata.mesh.sjit(
                init_fn,
                out_shardings=sharding_metadata.model_sharding_rule,
            )
        else:
            init_fn = jax.jit(init_fn)

        return init_fn(params)

    def save_args(self):
        args = {
            f"{self.name}_model_spec": ocp.args.JsonSave(self.model_spec.to_dict()),
            f"{self.name}_model_params": ocp.args.StandardSave({"params": self.params}),
        }

        if self.optimizer_spec is not None:
            args[f"{self.name}_optimizer_spec"] = ocp.args.JsonSave(
                self.optimizer_spec.to_dict()
            )
            args[f"{self.name}_opt_state"] = ocp.args.StandardSave(
                {"opt_state": self.opt_state}
            )

        return ocp.args.Composite(**args)

    @classmethod
    def restore(
        cls,
        name: str,
        checkpoint_manager: ocp.CheckpointManager,
        *,
        load_optimizer: bool,
        sharding_metadata: ShardingMetadata | None = None,
        step: int | None = None,
        is_legacy: bool = False
    ):
        if step is None:
            step = checkpoint_manager.latest_step()

        model_params_name = fmt_handler_name(name, "model_params", is_legacy=is_legacy)
        model_spec_name = fmt_handler_name(name, "model_spec", is_legacy=is_legacy)
        optimizer_spec_name = fmt_handler_name(name, "optimizer_spec", is_legacy=is_legacy)
        opt_state_name = fmt_handler_name(name, "opt_state", is_legacy=is_legacy)

        # breakpoint()
        try:
            abstract_params = checkpoint_manager.item_metadata(step).get(model_params_name)[
                "params"
            ]
        except TypeError as e:
            if jax.process_index() == 0:
                breakpoint()
            raise e
        if sharding_metadata is not None:
            shardings = sharding_metadata.mesh.match_sharding_rule(
                sharding_metadata.model_sharding_rule, {"params": abstract_params}
            )["params"]
            abstract_params = jax.tree_map(
                lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
                abstract_params,
                shardings,
            )

        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                **{
                    model_spec_name: ocp.args.JsonRestore(),
                    model_params_name: ocp.args.StandardRestore(
                        {"params": abstract_params}
                    ),
                }
            ),
        )

        params = restored.get(model_params_name)["params"]
        model_spec = ModuleSpec.from_dict(restored.get(model_spec_name))
        model = model_spec.instantiate()

        if load_optimizer:
            # First, load the optimizer spec
            optimizer_spec_json = checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    **{
                        optimizer_spec_name: ocp.args.JsonRestore(),
                    }
                ),
            )[optimizer_spec_name]
            optimizer_spec = OptimizerSpec.from_dict(optimizer_spec_json)
            optimizer = optimizer_spec.instantiate()

            # Initialize a dummy optimizer state with the correct sharding
            abstract_opt_state = jax.eval_shape(optimizer.init, params)

            # Add the shardings
            if sharding_metadata is not None:
                shardings = sharding_metadata.mesh.match_sharding_rule(
                    sharding_metadata.model_sharding_rule, {"opt_state": abstract_opt_state}
                )["opt_state"]
                abstract_opt_state = jax.tree_map(
                    lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s),
                    abstract_opt_state,
                    shardings,
                )

            opt_state = checkpoint_manager.restore(
                step,
                args=ocp.args.Composite(
                    **{
                        opt_state_name: ocp.args.StandardRestore(
                            {"opt_state": abstract_opt_state}
                        ),
                    }
                ),
            )[opt_state_name]["opt_state"]
        else:
            # If we don't load the optimizer, just use a dummy optimizer
            optimizer_spec = None
            optimizer = optax.set_to_zero()
            opt_state = optimizer.init(params)

        return cls(
            name=name,
            params=params,
            opt_state=opt_state,
            model_spec=model_spec,
            optimizer_spec=optimizer_spec,
            sharding_metadata=sharding_metadata,
            step=jnp.asarray(step, jnp.int32),
            model=model,
            apply_fn=model.apply,
            tx=optimizer,
        )


class PaliVLATrainState:
    def __init__(
        self,
        model_state: TrainState,
        action_tokenizer_state: TrainState,
        rng: jax.Array,
        language_tokenizer: SentencepieceTokenizer,
        config: dict,
        dataset_statistics: dict,
        mesh: MeshShardingHelper,
        data_sharding: ShardingRule,
        tokenizer_config: Tokenizer.TokenizerConfig,
    ):
        self.model_state = model_state
        self.action_tokenizer_state = action_tokenizer_state
        self.rng = rng
        self.language_tokenizer = language_tokenizer
        self.config = config
        self.dataset_statistics = dataset_statistics
        self.mesh = mesh
        self.data_sharding = data_sharding
        self.tokenizer_config = tokenizer_config
        self.tokenizer = Tokenizer.from_components(
            language_tokenizer=self.language_tokenizer,
            action_tokenizer=self.action_tokenizer_state.model,
            action_tokenizer_params=self.action_tokenizer_state.params,
            config=self.tokenizer_config,
        )

    @classmethod
    def from_components(
        cls,
        paligemma_weights_path: str,
        action_tokenizer_weights_path: str | None,
        language_tokenizer_path: str,
        config: dict,
        dataset_statistics: dict,
        language_tokenizer: SentencepieceTokenizer,
        optimizer_spec: OptimizerSpec,
        *,
        action_dim: int | None,
        action_horizon: int | None,
        model_sharding: ShardingMetadata,
        data_sharding: ShardingMetadata,
        mesh: MeshShardingHelper,
        seed: int,
        param_dtype: jnp.dtype,
        batch_shape: Dict[str, jax.ShapeDtypeStruct],
        pretrained_params: Params | None = None,
        pretrained_action_tokenizer_state: TrainState | None = None,
        loaded_language_tokenizer: SentencepieceTokenizer | None = None,
        skip_optimizer: bool = False,
    ):
        model_spec, params = load_from_pretrained(
            paligemma_weights_path,
            config,
            batch_shape,
            mesh=mesh,
            sharding_rule=model_sharding,
            seed=seed,
            param_dtype=param_dtype,
        )
        if pretrained_params is not None:
            params = merge_params(params, pretrained_params)

        model_sharding_metadata = ShardingMetadata(
            mesh=mesh,
            model_sharding_rule=model_sharding,
        )

        model_state = TrainState.with_params(
            name="model",
            params=params,
            model_spec=model_spec,
            optimizer_spec=OptimizerSpec.create(optax.set_to_zero, {}) if skip_optimizer else optimizer_spec,
            sharding_metadata=model_sharding_metadata,
        )

        if pretrained_action_tokenizer_state is not None:
            action_tokenizer_spec = pretrained_action_tokenizer_state.model_spec
            action_tokenizer_params = pretrained_action_tokenizer_state.params
        else:
            if action_tokenizer_weights_path is None:
                action_tokenizer_spec = ModuleSpec.from_name(
                    "palivla.tokenizer.BinActionTokenizer",
                    {
                        "min_action_value": -2,
                        "max_action_value": 2,
                        "action_dim": action_dim,
                        "action_horizon": action_horizon,
                        "action_vocab_size": 256,
                    },
                )
                action_tokenizer_params = {}
            else:
                action_tokenizer_spec, action_tokenizer_params = restore_gluon_module(
                    action_tokenizer_weights_path,
                    mesh=mesh,
                )

        action_tokenizer_state = TrainState.with_params(
            name="action_tokenizer",
            params=action_tokenizer_params,
            model_spec=action_tokenizer_spec,
            optimizer_spec=OptimizerSpec.create(optax.set_to_zero, {}),
            sharding_metadata=ShardingMetadata(mesh, PartitionSpec()),
        )

        if loaded_language_tokenizer is None:
            with tf.io.gfile.GFile(language_tokenizer_path, "rb") as f:
                language_tokenizer = SentencepieceTokenizer(f.read())
        else:
            language_tokenizer = loaded_language_tokenizer

        return cls(
            model_state=model_state,
            action_tokenizer_state=action_tokenizer_state,
            language_tokenizer=language_tokenizer,
            config=config,
            dataset_statistics=dataset_statistics,
            mesh=mesh,
            data_sharding=data_sharding,
            rng=jax.random.PRNGKey(seed),
            tokenizer_config=Tokenizer.TokenizerConfig.create(
                action_tokenizer=action_tokenizer_state.model,
                language_tokenizer=language_tokenizer,
                prompt_autoregressive=config["prompt_autoregressive"],
            ),
        )

    @cached_property
    def tokenizer(self, device=None):
        if device is None:
            action_tokenizer_params = self.action_tokenizer_state.params
        elif device == "cpu":
            action_tokenizer_params = jax.device_get(self.action_tokenizer_state.params)
        else:
            action_tokenizer_params = jax.device_put(
                self.action_tokenizer_state.params, device
            )

        return Tokenizer.from_components(
            language_tokenizer=self.language_tokenizer,
            action_tokenizer=self.action_tokenizer_state.model,
            action_tokenizer_params=action_tokenizer_params,
            config=self.tokenizer_config,
        )

    def save_args(self):
        dataset_statistics_save = jax.tree.map(
            lambda x: x.tolist(), self.dataset_statistics
        )
        language_tokenizer_proto = SentencepieceModelProto()
        language_tokenizer_proto.ParseFromString(
            self.language_tokenizer._model_resource._model
        )
        return ocp.args.Composite(
            **self.model_state.save_args(),
            **self.action_tokenizer_state.save_args(),
            config=ocp.args.JsonSave(unfreeze(self.config)),
            dataset_statistics=ocp.args.JsonSave(dataset_statistics_save),
            language_tokenizer=ocp.args.ProtoSave(language_tokenizer_proto),
        )

    def item_names(self):
        return self.save_args().keys()

    @classmethod
    def get_checkpoint_handlers(cls, is_legacy: bool = False):
        return {
            **TrainState.get_checkpoint_handlers("model", is_legacy),
            **TrainState.get_checkpoint_handlers("action_tokenizer", is_legacy),
            "config": ocp.JsonCheckpointHandler(),
            "dataset_statistics": ocp.JsonCheckpointHandler(),
            "language_tokenizer": ocp.ProtoCheckpointHandler(
                "language_tokenizer.proto"
            ),
        }
                
    
    @classmethod
    def get_old_checkpoint_handlers(cls):
        return dict(
            params=ocp.StandardCheckpointHandler(),
            opt_state=ocp.StandardCheckpointHandler(),
            step=ocp.StandardCheckpointHandler(),
            rng=ocp.StandardCheckpointHandler(),
            model_spec=ocp.JsonCheckpointHandler(),
            optimizer_spec=ocp.JsonCheckpointHandler(),
            dataset_statistics=ocp.JsonCheckpointHandler(),
            tokenizer_config=ocp.JsonCheckpointHandler(),
            action_tokenizer_spec=ocp.JsonCheckpointHandler(),
            action_tokenizer_params=ocp.StandardCheckpointHandler(),
        )

    @classmethod
    def load_components(
        cls,
        checkpoint_manager: ocp.CheckpointManager,
        *,
        step: int | None = None,
        load_optimizer: bool = False,
        mesh: MeshShardingHelper,
        model_sharding: ShardingRule,
        data_sharding: ShardingRule,
        is_legacy: bool = False
    ):
        step = step or checkpoint_manager.latest_step()

        # Replicate the action tokenizer across all devices
        action_tokenizer_sharding_metadata = ShardingMetadata(mesh, PartitionSpec())
        model_sharding_metadata = ShardingMetadata(mesh, model_sharding)

        restored_model_state = TrainState.restore(
            name="model",
            checkpoint_manager=checkpoint_manager,
            load_optimizer=load_optimizer,
            step=step,
            sharding_metadata=model_sharding_metadata,
            is_legacy=is_legacy,
        )
        restored_action_tokenizer_state = TrainState.restore(
            name="action_tokenizer",
            checkpoint_manager=checkpoint_manager,
            load_optimizer=False,
            step=step,
            sharding_metadata=action_tokenizer_sharding_metadata,
            is_legacy=is_legacy,
        )
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                config=ocp.args.JsonRestore(),
                dataset_statistics=ocp.args.JsonRestore(),
                language_tokenizer=ocp.args.ProtoRestore(SentencepieceModelProto),
            ),
        )

        language_tokenizer = SentencepieceTokenizer(
            model=restored["language_tokenizer"].SerializeToString()
        )

        dataset_statistics = restored["dataset_statistics"]
        dataset_statistics = jax.tree.map(
            lambda x: np.array(x),
            dataset_statistics,
            is_leaf=lambda x: isinstance(x, list),
        )

        return restored_model_state, restored_action_tokenizer_state, language_tokenizer, dataset_statistics

    @classmethod
    def restore(
        cls,
        checkpoint_manager: ocp.CheckpointManager,
        *,
        step: int | None = None,
        load_optimizer: bool = False,
        mesh: MeshShardingHelper,
        model_sharding: ShardingRule,
        data_sharding: ShardingRule,
        is_legacy: bool = False
    ):
        step = step or checkpoint_manager.latest_step()

        # Replicate the action tokenizer across all devices
        action_tokenizer_sharding_metadata = ShardingMetadata(mesh, PartitionSpec())
        model_sharding_metadata = ShardingMetadata(mesh, model_sharding)

        restored_model_state = TrainState.restore(
            name="model",
            checkpoint_manager=checkpoint_manager,
            load_optimizer=load_optimizer,
            step=step,
            sharding_metadata=model_sharding_metadata,
            is_legacy=is_legacy,
        )
        restored_action_tokenizer_state = TrainState.restore(
            name="action_tokenizer",
            checkpoint_manager=checkpoint_manager,
            load_optimizer=False,
            step=step,
            sharding_metadata=action_tokenizer_sharding_metadata,
            is_legacy=is_legacy,
        )

        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                config=ocp.args.JsonRestore(),
                dataset_statistics=ocp.args.JsonRestore(),
                language_tokenizer=ocp.args.ProtoRestore(SentencepieceModelProto),
            ),
        )

        config = freeze(restored["config"])
        dataset_statistics = restored["dataset_statistics"]
        dataset_statistics = jax.tree.map(
            lambda x: np.array(x),
            dataset_statistics,
            is_leaf=lambda x: isinstance(x, list),
        )

        language_tokenizer = SentencepieceTokenizer(
            model=restored["language_tokenizer"].SerializeToString()
        )

        seed = 0
        rng = jax.random.PRNGKey(seed)

        return cls(
            model_state=restored_model_state,
            action_tokenizer_state=restored_action_tokenizer_state,
            config=config,
            dataset_statistics=dataset_statistics,
            language_tokenizer=language_tokenizer,
            rng=rng,
            tokenizer_config=Tokenizer.TokenizerConfig.create(
                restored_action_tokenizer_state.model,
                language_tokenizer,
                config["prompt_autoregressive"],
            ),
            mesh=mesh,
            data_sharding=data_sharding,
        )

    @cached_property
    def step_fn(self):
        # for some reason, static argnums not working, can't be bothered to figure it out
        __step_fn = partial(step_fn, self.detokenize_action, True, self.tokenizer.config, False)
        if self.mesh is None:
            _step_fn = partial(jax.jit, __step_fn)
        else:
            _step_fn = partial(
                self.mesh.sjit,
                __step_fn,
                args_sharding_constraint=(
                    self.model_state.sharding_metadata.model_sharding_rule,
                    self.data_sharding,
                    None,
                ),
            )

        return _step_fn(
            in_shardings=(
                self.model_state.sharding_metadata.model_sharding_rule,
                self.data_sharding,
                None,
            ),
            out_shardings=(
                self.model_state.sharding_metadata.model_sharding_rule,
                None,
                None,
            ),
        )
    
    @cached_property
    def fuse_step_fn(self):
        __step_fn = partial(step_fn, self.detokenize_action, True, self.tokenizer.config, True)
        if self.mesh is None:
            _step_fn = partial(jax.jit, __step_fn)
        else:
            _step_fn = partial(
                self.mesh.sjit,
                __step_fn,
                args_sharding_constraint=(
                    self.model_state.sharding_metadata.model_sharding_rule,
                    self.data_sharding,
                    None,
                ),
            )

        return _step_fn(
            in_shardings=(
                self.model_state.sharding_metadata.model_sharding_rule,
                self.data_sharding,
                None,
            ),
            out_shardings=(
                self.model_state.sharding_metadata.model_sharding_rule,
                None,
                None,
            ),
        )

    def prepare_sensors(self, sensors: Dict[str, jax.Array]):
        return {k: (v / 127.5 - 1.0) if "image" in k and "digit" not in k else v for k, v in sensors.items()}

    def prepare_batch(self, batch: TrainingBatch):
        return batch.replace(sensors=self.prepare_sensors(batch.sensors))

    def train_step(self, batch: TrainingBatch,):
        with self.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            self.model_state, base_info, self.rng = self.step_fn(
                self.model_state,
                self.prepare_batch(batch),
                self.rng,
            )
        


            # self.model_state, fuse_info, self.rng = self.fuse_step_fn(
            #     self.model_state,
            #     self.prepare_batch(batch),
            #     self.rng,
            # )

            # info = base_info | fuse_info
            info = base_info

        return info

    def decode(
        self, batch: RolloutBatch, target_key_order: Sequence[str] | None = None
    ):
        return _decode(
            self.model_state.params,
            batch.sensor_data | {"text": batch.prompt},
            batch.sensor_masks | {"text": batch.prompt_mask},
            batch.prompt_ar,
            target_key_order=target_key_order or self.config.get("target_key_order"),
            model=self.model_state.model,
            devices=self.mesh.mesh.devices,
            max_decode_len=self.action_tokenizer_state.model.num_tokens,
            eos_token=self.tokenizer_config.eos_token,
            best_of_n=1,
            sampler="greedy",
            replicate_out=False,
            eos_look_behind=0,
        )

    def tokenize_action(self, actions, obs):
        return self.tokenizer.tokenize_action(
            actions,
            obs=obs,
        )

    def detokenize_action(self, tokens, obs):
        return self.tokenizer.detokenize_action(
            tokens,
            obs=obs,
        )

    def eval_step(
        self,
        batch: TrainingBatch,
        prefix: str,
        target_key_order: Sequence[str] | None = None,
        include_regular_stats: bool = True,
    ):
        batch = self.prepare_batch(batch)

        results = compute_gen_stats(
            decode_fn=self.decode,
            tokenize_fn=partial(self.tokenize_action, obs=batch.sensors),
            detokenize_fn=partial(self.detokenize_action, obs=batch.sensors),
            mesh=self.mesh,
            batch=batch,
            prefix=prefix,
            tokenizer_config=self.tokenizer.config,
            target_key_order=target_key_order,
        )

        if include_regular_stats:
            results = results | compute_eval_stats(
                partial(
                    self.mesh.sjit(self.model_state.apply_fn, out_shardings=(None,)),
                    {"params": self.model_state.params},
                ),
                partial(self.detokenize_action, obs=batch.sensors),
                batch,
                prefix,
                self.tokenizer.config,
                target_key_order=target_key_order,
            )

        return jax.device_get(results)


def load_old_checkpoint(
    load_directory: str,
    language_tokenizer_path: str,
    step: int,
    save_directory: Optional[str] = None,
    mesh: Optional[MeshShardingHelper] = None,
    model_sharding: Optional[ShardingRule] = None,
    data_sharding: Optional[ShardingRule] = None,
    load_optimizer: bool = False,
    model_dtype: Optional[jnp.dtype] = None,
):
    
    load_checkpoint_manager = ocp.CheckpointManager(
            directory=load_directory,
            item_names=[
                "params",
                "opt_state",
                "step",
                "model_spec",
                "optimizer_spec",
                "dataset_statistics",
                "tokenizer_config",
                "rng",
                "action_tokenizer_spec",
                "action_tokenizer_params",
            ],
            options=ocp.CheckpointManagerOptions(),
        )
    # breakpoint() 
    rng = jax.random.PRNGKey(0)
    # train_step_sharding = jax.sharding.NamedSharding(
    #     mesh.mesh,
    #     model_sharding.apply({"step": jax.ShapeDtypeStruct((), jnp.int64)}),
    # )["step"]

    restored_metadata = load_checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                step=ocp.args.StandardRestore(
                    jax.ShapeDtypeStruct((), jnp.int64)
                ),
                model_spec=ocp.args.JsonRestore(),
                optimizer_spec=ocp.args.JsonRestore(),
                dataset_statistics=ocp.args.JsonRestore(),
                rng=ocp.args.StandardRestore(rng),
                tokenizer_config=ocp.args.JsonRestore(),
                action_tokenizer_spec=ocp.args.JsonRestore(),
                action_tokenizer_params=ocp.args.StandardRestore(
                    jax.tree.map(
                        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                        load_checkpoint_manager.item_metadata(step)[
                            "action_tokenizer_params"
                        ],
                    )
                ),
            ),
        )
    pass


# @classmethod
#     def from_checkpoint(
#         cls,
#         load_directory: str,
#         language_tokenizer_path: str,
#         step: int,
#         save_directory: Optional[str] = None,
#         mesh: Optional[MeshShardingHelper] = None,
#         model_sharding: Optional[ShardingRule] = None,
#         data_sharding: Optional[ShardingRule] = None,
#         load_optimizer: bool = True,
#         model_dtype: Optional[jnp.dtype] = None,
#     ):
#         load_checkpoint_manager = ocp.CheckpointManager(
#             directory=load_directory,
#             item_names=[
#                 "params",
#                 "opt_state",
#                 "step",
#                 "model_spec",
#                 "optimizer_spec",
#                 "dataset_statistics",
#                 "tokenizer_config",
#                 "rng",
#                 "action_tokenizer_spec",
#                 "action_tokenizer_params",
#             ],
#             options=ocp.CheckpointManagerOptions(),
#         )

#         # Restore JSON saved objects
#         rng = jax.random.PRNGKey(0)
#         train_step_sharding = jax.sharding.NamedSharding(
#             mesh.mesh,
#             model_sharding.apply({"step": jax.ShapeDtypeStruct((), jnp.int64)}),
#         )["step"]

#         restored_metadata = load_checkpoint_manager.restore(
#             step,
#             args=ocp.args.CompositeRestore(
#                 step=ocp.args.StandardRestore(
#                     jax.ShapeDtypeStruct((), jnp.int64, sharding=train_step_sharding)
#                 ),
#                 model_spec=ocp.args.JsonRestore(),
#                 optimizer_spec=ocp.args.JsonRestore(),
#                 dataset_statistics=ocp.args.JsonRestore(),
#                 rng=ocp.args.StandardRestore(rng),
#                 tokenizer_config=ocp.args.JsonRestore(),
#                 action_tokenizer_spec=ocp.args.JsonRestore(),
#                 action_tokenizer_params=ocp.args.StandardRestore(
#                     jax.tree.map(
#                         lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
#                         load_checkpoint_manager.item_metadata(step)[
#                             "action_tokenizer_params"
#                         ],
#                     )
#                 ),
#             ),
#         )
#         rng = restored_metadata["rng"]
#         train_step = restored_metadata["step"]
#         model_spec = ModuleSpec.from_dict(restored_metadata["model_spec"])
#         optimizer_spec = OptimizerSpec.from_dict(restored_metadata["optimizer_spec"])
#         action_tokenizer_spec = ModuleSpec.from_dict(
#             restored_metadata["action_tokenizer_spec"]
#         )

#         dataset_statistics = restored_metadata["dataset_statistics"]
#         with tf.io.gfile.GFile(language_tokenizer_path, "rb") as f:
#             language_tokenizer = SentencepieceTokenizer(f.read())

#         tokenizer = Tokenizer.from_components(
#             language_tokenizer=language_tokenizer,
#             action_tokenizer=action_tokenizer_spec.instantiate(),
#             action_tokenizer_params=restored_metadata["action_tokenizer_params"],
#             prompt_autoregressive=restored_metadata["tokenizer_config"][
#                 "prompt_autoregressive"
#             ],
#         )

#         # Load sharded params
#         params_metadata = load_checkpoint_manager.item_metadata(step)["params"]

#         def _make_abstract_array(metadata):
#             return jax.ShapeDtypeStruct(
#                 shape=metadata.shape,
#                 dtype=metadata.dtype,
#             )

#         def _shard_abstract_array(abstract_array, sharding_rule):
#             return jax.ShapeDtypeStruct(
#                 shape=abstract_array.shape,
#                 dtype=model_dtype or abstract_array.dtype,
#                 sharding_rule=sharding_rule,
#             )

#         abstract_params = jax.tree_map(_make_abstract_array, params_metadata)
#         params_sharding_rules = model_sharding.apply(abstract_params)
#         abstract_params = jax.tree.map(
#             _shard_abstract_array,
#             abstract_params,
#             params_sharding_rules,
#         )
#         params = load_checkpoint_manager.restore(
#             step,
#             args=ocp.args.Composite(params=ocp.args.StandardRestore(abstract_params)),
#         )["params"]

#         if load_optimizer:
#             tx = optimizer_spec.instantiate()
#             abstract_optimizer_state = jax.eval_shape(tx.init, abstract_params)
#             optimizer_sharding_rules = model_sharding.apply(abstract_optimizer_state)
#             abstract_optimizer_state = jax.tree.map(
#                 _shard_abstract_array,
#                 abstract_optimizer_state,
#                 optimizer_sharding_rules,
#             )
#             opt_state = load_checkpoint_manager.restore(
#                 step,
#                 args=ocp.args.Composite(
#                     opt_state=ocp.args.StandardRestore(abstract_optimizer_state)
#                 ),
#             )["opt_state"]
#         else:
#             optimizer_spec = OptimizerSpec(optax.set_to_zero, {})
#             opt_state = {}

#         model = model_spec.instantiate()
#         tx = optimizer_spec.instantiate()
#         decode_fn = get_decode_fn(model, tokenizer)

#         if save_directory is not None:
#             checkpoint_manager = ocp.CheckpointManager(
#                 directory=save_directory,
#                 item_names=[
#                     "params",
#                     "opt_state",
#                     "step",
#                     "model_spec",
#                     "optimizer_spec",
#                     "dataset_statistics",
#                     "tokenizer_config",
#                     "rng",
#                 ],
#                 options=ocp.CheckpointManagerOptions(
#                     max_to_keep=1,
#                 ),
#             )
#         else:
#             checkpoint_manager = None

#         return cls(
#             train_state=TrainState(
#                 step=train_step,
#                 apply_fn=model.apply,
#                 params=params,
#                 tx=tx,
#                 opt_state=opt_state,
#             ),
#             module_spec=model_spec,
#             optimizer_spec=optimizer_spec,
#             dataset_statistics=dataset_statistics,
#             tokenizer=tokenizer,
#             module=model_spec.instantiate(),
#             model_sharding=model_sharding,
#             data_sharding=data_sharding,
#             mesh=mesh,
#             decode_fn=decode_fn,
#             checkpoint_manager=checkpoint_manager,
#             rng=rng,
#         )
