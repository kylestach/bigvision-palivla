from functools import partial
from os import PathLike
from typing import Any

import cloudpickle
import flax.linen as nn
import jax
import orbax.checkpoint as ocp
import jax
from jax.sharding import PartitionSpec
import jax.experimental.multihost_utils as mhu
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
        gt_actions = batch["action"][:, -1, :, :]  # (batch, max_chunk_size, dimension)
        predicted_actions, actions_mask, tokens = self.predict(
            batch, action_dim=gt_actions.shape[-1], return_tokens=True,
        )

        gt_actions = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(gt_actions))
        gt_use_actions = batch['use_actions']
        gt_use_actions = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(gt_use_actions))
        predicted_actions = np.nan_to_num(predicted_actions)

        # Get metrics for datasets that use actions
        gt_actions_real = np.array([ac for use, ac in zip(gt_use_actions, gt_actions) if use])
        predicted_actions_real = np.array([ac for use, ac in zip(gt_use_actions, predicted_actions) if use])
        actions_mask_real = np.array([mask for use, mask in zip(gt_use_actions, actions_mask) if use])

        # Get action token IDs from the tokenizer
        action_token_start = self.language_tokenizer.convert_tokens_to_ids("<begin_of_action>")
        
        # Convert token IDs back to strings for comparison
        token_strings = np.array([self.language_tokenizer.convert_ids_to_tokens(int(token_id)) for token_id in tokens["target"].flatten()]).reshape(tokens["target"].shape)
        
        # Create masks for action and representation tokens
        # Check for both <begin_of_action> and <act tokens
        is_action_token = np.array([token == "<begin_of_action>" or token.startswith("<act") for token in token_strings.flatten()]).reshape(token_strings.shape)
        is_special_token = np.array([token == "<pad>" or token == "<eos>" for token in token_strings.flatten()]).reshape(token_strings.shape)
        is_rep_token = ~(is_action_token | is_special_token)
        
        # Convert boolean masks to float32 for JAX compatibility
        action_token_mask = (tokens["mask"].astype(np.float32) * is_action_token.astype(np.float32))
        rep_token_mask = (tokens["mask"].astype(np.float32) * is_rep_token.astype(np.float32))
        total_token_mask = tokens["mask"].astype(np.float32)
        
        # Compute accuracy separately for action and representation tokens
        action_acc = np.mean((tokens["predicted"] == tokens["target"]) * action_token_mask) / np.mean(action_token_mask) if np.mean(action_token_mask) > 0 else 0.0
        rep_acc = np.mean((tokens["predicted"] == tokens["target"]) * rep_token_mask) / np.mean(rep_token_mask) if np.mean(rep_token_mask) > 0 else 0.0
        total_acc = np.mean((tokens["predicted"] == tokens["target"]) * total_token_mask) / np.mean(total_token_mask) if np.mean(total_token_mask) > 0 else 0.0

        # Initialize metrics dictionary with global metrics
        metrics = {
            "gen_l2": np.mean(np.square(predicted_actions - gt_actions) * actions_mask) / np.mean(actions_mask),
            "gen_l1": np.mean(np.abs(predicted_actions - gt_actions) * actions_mask) / np.mean(actions_mask),
            "gen_acc": total_acc,
            "gen_acc_actions": action_acc,
            "gen_acc_representations": rep_acc,
            "gen_l2_for_datasets_using_actions": np.mean(np.square(predicted_actions_real - gt_actions_real) * actions_mask_real) / np.mean(actions_mask_real),
            "gen_l1_for_datasets_using_actions": np.mean(np.abs(predicted_actions_real - gt_actions_real) * actions_mask_real) / np.mean(actions_mask_real),
        }

        # Get dataset names for each sample
        dataset_names = batch.get('dataset_name', None)
        if dataset_names is not None:
            # Since dataset names are non-numeric, we need to handle them differently
            # First, convert to a format that can be gathered
            if isinstance(dataset_names[0], bytes):
                dataset_names = np.array([name.decode() if isinstance(name, bytes) else name for name in dataset_names])
            
            # Create a deterministic numeric encoding for dataset names
            unique_names = np.unique(dataset_names)
            # Use simple enumeration for stable numeric IDs
            name_to_id = {name: i for i, name in enumerate(unique_names)}
            numeric_ids = np.array([name_to_id[name] for name in dataset_names])
            
            # Gather the numeric IDs
            gathered_ids = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(numeric_ids))
            
            # Create reverse mapping for gathered IDs
            id_to_name = {i: name for i, name in enumerate(unique_names)}
            
            # Compute metrics for each unique dataset in the gathered batch
            for dataset_id in np.unique(gathered_ids):
                if dataset_id in id_to_name:  # Only process if we have the mapping
                    dataset = id_to_name[dataset_id]
                    dataset_mask = gathered_ids == dataset_id
                    if np.any(dataset_mask):
                        # Action metrics
                        dataset_actions = gt_actions[dataset_mask]
                        dataset_preds = predicted_actions[dataset_mask]
                        dataset_action_mask = actions_mask[dataset_mask].astype(np.float32)
                        
                        # Token metrics
                        dataset_tokens = {
                            "predicted": tokens["predicted"][dataset_mask],
                            "target": tokens["target"][dataset_mask],
                            "mask": tokens["mask"][dataset_mask].astype(np.float32)
                        }
                        dataset_is_action = is_action_token[dataset_mask].astype(np.float32)
                        dataset_is_rep = is_rep_token[dataset_mask].astype(np.float32)
                        
                        dataset_action_token_mask = dataset_tokens["mask"] * dataset_is_action
                        dataset_rep_token_mask = dataset_tokens["mask"] * dataset_is_rep
                        
                        dataset_action_acc = np.mean((dataset_tokens["predicted"] == dataset_tokens["target"]) * dataset_action_token_mask) / np.mean(dataset_action_token_mask) if np.mean(dataset_action_token_mask) > 0 else 0.0
                        dataset_rep_acc = np.mean((dataset_tokens["predicted"] == dataset_tokens["target"]) * dataset_rep_token_mask) / np.mean(dataset_rep_token_mask) if np.mean(dataset_rep_token_mask) > 0 else 0.0
                        dataset_total_acc = np.mean((dataset_tokens["predicted"] == dataset_tokens["target"]) * dataset_tokens["mask"]) / np.mean(dataset_tokens["mask"]) if np.mean(dataset_tokens["mask"]) > 0 else 0.0
                        
                        metrics.update({
                            f"{dataset}_l2": np.mean(np.square(dataset_preds - dataset_actions) * dataset_action_mask) / np.mean(dataset_action_mask),
                            f"{dataset}_l1": np.mean(np.abs(dataset_preds - dataset_actions) * dataset_action_mask) / np.mean(dataset_action_mask),
                            f"{dataset}_acc": dataset_total_acc,
                            f"{dataset}_acc_actions": dataset_action_acc,
                            f"{dataset}_acc_representations": dataset_rep_acc,
                        })

        return metrics

    def build_sequence(self, batch: Any, begin_is_prompt: bool = True, include_action_tokens: bool = True):
        return self.sequence_builder.build_sequence(
            batch,
            self.language_tokenizer,
            self.action_tokenizer,
            begin_is_prompt=begin_is_prompt,
            include_action_tokens=include_action_tokens,
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
        include_action_tokens: bool = True,
        inference_mode: bool = False,
    ):
        sequences = self.build_sequence(batch, begin_is_prompt=True, include_action_tokens=include_action_tokens)

        if inference_mode:
            batch, sequences = mhu.broadcast_one_to_all(
                (batch, sequences)
            )
            action_chunk_sizes = batch['action_chunk_size']
        else:
            action_chunk_sizes = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(batch['action_chunk_size']))

        tokens = self.predict_tokens(batch, sequences, use_ema_params=use_ema_params, replicate=replicate)
        
        # since our gt sequence doesn't include actions for certain datasets, our actions for those datasets will be masked out
        actions, actions_mask = self.sequence_builder.batch_get_actions(
            tokens,
            self.language_tokenizer,
            self.action_tokenizer,
            begin_is_prompt=True,
            action_dim=action_dim,
            action_chunk_sizes=action_chunk_sizes,
        )

        if return_tokens:
            sequences = self.data_gather_fn(self.sharding.mesh.local_data_to_global_array(sequences)) if not inference_mode else sequences
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

    def load_params(self, step: int, checkpoint_manager: ocp.CheckpointManager):
        """
        Loads the model state from a checkpoint but keeps the current optimizer state.
        This preserves all model variables while using fresh optimizer states.
        """
        # Create a temporary state for loading
        tmp_state = checkpoint_manager.restore(step, args=ocp.args.StandardRestore(self.train_state))
        
        # Keep all state variables except optimizer state
        new_state = self.train_state.replace(
            step=tmp_state.step,
            params=tmp_state.params,
            # Keep current optimizer state
            opt_state=self.train_state.opt_state
        )
        self.train_state = new_state
