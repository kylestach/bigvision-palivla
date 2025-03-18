from functools import partial
from os import PathLike
from typing import Any

import cloudpickle
import flax.linen as nn
import jax
import orbax.checkpoint as ocp
from jax.sharding import PartitionSpec
from transformers import AutoTokenizer
import numpy as np

from palivla.components.action_tokenizer import ActionTokenizer
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata, TrainState
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.train_step import step_fn
from palivla.utils import read_staging_directory, write_staging_directory


def make_step_fn(sharding: ShardingMetadata):
    return sharding.mesh.sjit(
        partial(step_fn, train=True),
        in_shardings=(sharding.model_sharding_rule, PartitionSpec("fsdp"), None),
        out_shardings=(sharding.model_sharding_rule, None, None),
        args_sharding_constraint=(
            sharding.model_sharding_rule,
            PartitionSpec("fsdp"),
            None,
        ),
        donate_argnums=(0,),
    )


def make_gather_fn(mesh):
    jax_gather_fn = jax.jit(
        lambda x: x,
        out_shardings=jax.NamedSharding(mesh, PartitionSpec()),
    )
    return lambda tensor: jax.device_get(jax_gather_fn(tensor))


class ModelComponents:
    __slots__ = [
        "language_tokenizer",
        "action_tokenizer",
        "sequence_builder",
        "train_state",
        "sharding",
        "rng",
        "step_fn",
        "data_gather_fn",
        "example_batch",
    ]

    def __init__(
        self,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        sequence_builder: SequenceBuilder,
        train_state: TrainState,
        sharding: ShardingMetadata,
        rng: jax.Array,
        example_batch: Any,
    ):
        self.language_tokenizer = language_tokenizer
        self.action_tokenizer = action_tokenizer
        self.sequence_builder = sequence_builder
        self.train_state = train_state
        self.sharding = sharding
        self.rng = rng
        self.step_fn = make_step_fn(sharding)
        self.data_gather_fn = make_gather_fn(sharding.mesh.mesh)
        self.example_batch = example_batch
    @classmethod
    def initialize(
        cls,
        *,
        model_spec: ModuleSpec,
        optimizer_spec: OptimizerSpec,
        seed: int,
        language_tokenizer: AutoTokenizer,
        action_tokenizer: ActionTokenizer,
        sequence_builder: SequenceBuilder,
        sharding_metadata: ShardingMetadata,
        example_batch: Any,
    ):
        rng, key = jax.random.split(jax.random.PRNGKey(seed))
        return cls(
            language_tokenizer=language_tokenizer,
            action_tokenizer=action_tokenizer,
            sequence_builder=sequence_builder,
            sharding=sharding_metadata,
            rng=rng,
            train_state=TrainState.initialize(
                model_spec=model_spec,
                optimizer_spec=optimizer_spec,
                example_batch=example_batch,
                sharding=sharding_metadata,
                rng=key,
            ),
            example_batch=example_batch,
        )

    def save_static(self, path: Any):
        from tensorflow import io

        io.gfile.makedirs(path)

        # Huggingface can't load from GCS, so we need to stage the tokenizer to a local directory
        with write_staging_directory(io.gfile.join(path, "language_tokenizer")) as temp_dir:
            self.language_tokenizer.save_pretrained(temp_dir)

        self.action_tokenizer.save(path)
        self.sequence_builder.save(path)
        self.train_state.save_static(path)
        with io.gfile.GFile(io.gfile.join(path, "rng.pkl"), "wb") as f:
            cloudpickle.dump(jax.device_get(self.rng), f)
        with io.gfile.GFile(io.gfile.join(path, "example_batch.pkl"), "wb") as f:
            cloudpickle.dump(self.example_batch, f)

    def save_state(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        self.train_state.save_state(step, checkpoint_manager)

    @classmethod
    def load_static(cls, path: PathLike, sharding: ShardingMetadata):
        from tensorflow import io

        # Huggingface can't load from GCS, so we need to stage the tokenizer to a local directory
        with read_staging_directory(io.gfile.join(path, "language_tokenizer")) as temp_dir:
            language_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

        action_tokenizer = ActionTokenizer.load(path)
        sequence_builder = SequenceBuilder.load(path)

        with io.gfile.GFile(io.gfile.join(path, "example_batch.pkl"), "rb") as f:
            example_batch = cloudpickle.load(f)
        with io.gfile.GFile(io.gfile.join(path, "rng.pkl"), "rb") as f:
            rng = cloudpickle.load(f)

        train_state = TrainState.load_static(
            path,
            sharding=sharding,
            example_batch=example_batch,
        )
        return cls(
            language_tokenizer=language_tokenizer,
            action_tokenizer=action_tokenizer,
            sequence_builder=sequence_builder,
            train_state=train_state,
            sharding=sharding,
            rng=rng,
            example_batch=example_batch,
        )

    def load_state(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        self.train_state = self.train_state.load_state(step, checkpoint_manager)

    def train_step(self, batch: Any):
        # Tokenize the batch and build sequences
        sequences = self.build_sequence(batch, begin_is_prompt=False)

        # Shard the batch to devices
        batch = {
            "sensors": batch["observation"],
            "sensors_mask": batch["observation"]["pad_mask_dict"],
            "prompt": sequences["prompt"],
            "gen": sequences["gen"],
        }
        batch = self.sharding.mesh.local_data_to_global_array(batch)

        # Run the train step
        with self.sharding.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            self.train_state, info, self.rng = self.step_fn(
                self.train_state, batch, self.rng
            )

        return info

    def eval_step(self, batch):
        gt_actions = batch["action"][:, -1, :, :]

        predicted_actions, actions_mask, tokens = self.predict(
            batch, action_dim=gt_actions.shape[-1], return_tokens=True,
        )

        gt_actions = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(gt_actions))
        predicted_actions = np.nan_to_num(predicted_actions)

        return {
            "gen_valid_pct": actions_mask.mean(),
            "gen_l2": np.mean(np.square(predicted_actions - gt_actions) * actions_mask)
            / actions_mask.mean(),
            "gen_l1": np.mean(np.abs(predicted_actions - gt_actions) * actions_mask)
            / actions_mask.mean(),
            "gen_acc": np.mean(
                (tokens["predicted"] == tokens["target"]) * tokens["mask"]
            )
            / tokens["mask"].mean(),
        }

    def build_sequence(self, batch: Any, begin_is_prompt: bool = True):
        return self.sequence_builder.build_sequence(
            batch,
            self.language_tokenizer,
            self.action_tokenizer,
            begin_is_prompt=begin_is_prompt,
        )

    def predict_tokens(self, batch, sequences: Any | None, *, use_ema_params: bool = False, replicate: bool = False):
        if sequences is None:
            sequences = self.build_sequence(batch, begin_is_prompt=True)

        # Shard the batch to devices
        inputs = {
            "sensors": batch["observation"],
            "sensors_mask": batch["observation"]["pad_mask_dict"],
            "prompt": sequences["prompt"],
            "gen": sequences["gen"],
        }

        if not replicate:
            inputs = self.sharding.mesh.local_data_to_global_array(inputs)

        # Run the train step
        with self.sharding.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            from palivla.predict_fns import _decode

            params = self.train_state.get_params(use_ema_params=use_ema_params)

            tokens = _decode(
                params,
                inputs,
                model=self.train_state.model,
                mesh=self.sharding.mesh.mesh,
                out_sharding=PartitionSpec() if replicate else PartitionSpec("fsdp"),
                max_decode_len=self.sequence_builder.max_decode_length,
                eos_token=self.language_tokenizer.eos_token_id,
            )

        return self.data_gather_fn(tokens)

    def predict(
        self,
        batch,
        action_dim: int,
        *,
        use_ema_params: bool = False,
        return_tokens: bool = False,
        replicate: bool = False,
    ):
        sequences = self.build_sequence(batch, begin_is_prompt=True)
        tokens = self.predict_tokens(batch, sequences, use_ema_params=use_ema_params, replicate=replicate)

        actions, actions_mask = self.sequence_builder.batch_get_actions(
            tokens,
            self.language_tokenizer,
            self.action_tokenizer,
            begin_is_prompt=True,
            action_dim=action_dim,
        )

        if return_tokens:
            sequences = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(sequences))
            return (
                actions,
                actions_mask,
                {
                    "predicted": tokens,
                    "target": sequences["gen"]["tokens"],
                    "mask": sequences["gen"]["mask"],
                },
            )
        else:
            return actions, actions_mask
