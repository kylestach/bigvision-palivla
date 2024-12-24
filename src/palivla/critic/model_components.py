from functools import partial
from typing import Any

import flax.linen as nn
import jax
import numpy as np
from jax.sharding import PartitionSpec
from transformers import AutoTokenizer

from palivla.components.action_tokenizer import ActionTokenizer
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata
from palivla.critic.train_state import EMATrainState
from palivla.critic.train_step import train_step
from palivla.model_components import ModelComponents
from palivla.spec import ModuleSpec, OptimizerSpec


def make_step_fn(sharding: ShardingMetadata, donate_train_state: bool = True, **kwargs):
    donate_argnums = (0,) if donate_train_state else None
    return sharding.mesh.sjit(
        partial(train_step, **kwargs),
        in_shardings=(
            sharding.model_sharding_rule,
            PartitionSpec("fsdp"),
            None,
        ),
        out_shardings=(sharding.model_sharding_rule, None, None),
        args_sharding_constraint=(
            sharding.model_sharding_rule,
            PartitionSpec("fsdp"),
            None,
        ),
        donate_argnums=donate_argnums,
    )


class CriticModelComponents(ModelComponents):
    def __init__(self, *args, critic_train_step_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_step_fn = make_step_fn(
            self.sharding,
            **critic_train_step_kwargs,
            donate_train_state=True,
            train=True,
        )
        self.eval_step_fn = make_step_fn(
            self.sharding,
            **critic_train_step_kwargs,
            donate_train_state=False,
            train=False,
        )

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
        critic_train_step_kwargs: dict,
    ):
        rng, key = jax.random.split(jax.random.PRNGKey(seed))
        return cls(
            language_tokenizer=language_tokenizer,
            action_tokenizer=action_tokenizer,
            sequence_builder=sequence_builder,
            sharding=sharding_metadata,
            rng=rng,
            train_state=EMATrainState.initialize(
                model_spec=model_spec,
                optimizer_spec=optimizer_spec,
                example_batch=example_batch,
                sharding=sharding_metadata,
                rng=key,
            ),
            critic_train_step_kwargs=critic_train_step_kwargs,
        )

    def prepare_batch_for_train_step(self, batch: dict):
        # Tokenize the batch and build sequences
        sequences = self.sequence_builder.build_sequence(
            batch, self.language_tokenizer, self.action_tokenizer
        )

        # TODO: load counterfactual_next_actions
        batch["counterfactual_next_actions"] = np.zeros_like(batch["action"])

        # Shard the batch to devices
        batch = {
            "sensors": batch["observation"],
            "sensors_mask": batch["observation"]["pad_mask_dict"],
            "next_sensors": batch["next_observation"],
            "next_sensors_mask": batch["next_observation"]["pad_mask_dict"],
            "prompt": sequences["prompt"],
            "next_prompt": sequences["prompt"],
            "action": batch["action"][:, -1, -1, :],
            "next_action": batch["next_action"][:, -1, -1, :],
            "counterfactual_next_actions": batch["counterfactual_next_actions"][
                :, -1, :, :
            ],
            "rewards": batch["reward"],
            "td_mask": batch["td_mask"],
            "mc_return": batch["mc_return"],
        }
        batch = self.sharding.mesh.local_data_to_global_array(batch)
        return batch

    def train_step(
        self,
        batch: dict,
    ):
        batch = self.prepare_batch_for_train_step(batch)

        # Run the train step
        with self.sharding.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            self.train_state, info, self.rng = self.train_step_fn(
                self.train_state,
                batch,
                self.rng,
            )

        return info

    def eval_step(self, batch: dict):
        batch = self.prepare_batch_for_train_step(batch)
        with self.sharding.mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
            _, info, self.rng = self.eval_step_fn(
                self.train_state,
                batch,
                self.rng,
            )
        return info

    def save_args(self):
        return self.train_state.save_args()
