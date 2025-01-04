import os
from collections import OrderedDict
from pathlib import Path

from big_vision.utils import Registry
from palivla.components.action_tokenizer import ActionTokenizer
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata
from palivla.critic.model_components import CriticModelComponents
from palivla.critic.visualize_critic import visualize_critic
from palivla.critic.vla_critic import PaliVLACritic


import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import tqdm
import wandb
from absl import app, flags
from absl import logging as absl_logging
from flax.core.frozen_dict import freeze
from ml_collections import ConfigDict, config_flags
import palivla.load_fns
from palivla.dataset import make_base_dataset, make_trajectory_dataset
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.utils import flatten_wandb_dict, host_broadcast_str
from scalax.sharding import FSDPShardingRule, MeshShardingHelper
from transformers import AutoTokenizer

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def make_sharding(config: ConfigDict):
    mesh = MeshShardingHelper([-1], ["fsdp"])
    sharding_metadata = ShardingMetadata(
        mesh=mesh,
        model_sharding_rule=FSDPShardingRule(
            "fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"]
        ),
    )
    return sharding_metadata


def get_batch_info(batch: dict):
    return {
        "rewards_mean": np.mean(batch["reward"]),
        "rewards_std": np.std(batch["reward"]),
        "rewards_min": np.min(batch["reward"]),
        "rewards_max": np.max(batch["reward"]),
        "actions_mean": np.mean(batch["action"]),
        "actions_std": np.std(batch["action"]),
        "actions_min": np.min(batch["action"]),
        "actions_max": np.max(batch["action"]),
        "next_actions_mean": np.mean(batch["next_action"]),
        "next_actions_std": np.std(batch["next_action"]),
        "next_actions_min": np.min(batch["next_action"]),
        "next_actions_max": np.max(batch["next_action"]),
        "td_mask_mean": np.mean(batch["td_mask"]),
        "td_mask_std": np.std(batch["td_mask"]),
        "td_mask_min": np.min(batch["td_mask"]),
        "td_mask_max": np.max(batch["td_mask"]),
        "mc_returns_mean": np.mean(batch["mc_return"]),
        "mc_returns_std": np.std(batch["mc_return"]),
        "mc_returns_min": np.min(batch["mc_return"]),
        "mc_returns_max": np.max(batch["mc_return"]),
    }


def get_example_batch():
    return OrderedDict(
        {
            "sensors": {
                "image_primary": jax.ShapeDtypeStruct(
                    shape=(1, 224, 224, 3), dtype=jnp.uint8
                ),
                "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.float32),
            },
            "sensors_mask": {
                "image_primary": jax.ShapeDtypeStruct(
                    shape=(1, 224, 224, 3), dtype=jnp.bool_
                ),
                "proprio": jax.ShapeDtypeStruct(shape=(1, 7), dtype=jnp.bool_),
            },
            "prompt": {
                "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
                "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
                "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
                "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
            },
            "actions": jax.ShapeDtypeStruct(shape=(1, 1, 7), dtype=jnp.float32),
        }
    )


def create_model(config: ConfigDict, sharding_metadata: ShardingMetadata):
    example_batch = get_example_batch()

    language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer)
    action_tokenizer: ActionTokenizer = Registry.lookup(config.action_tokenizer)()
    sequence_builder: SequenceBuilder = Registry.lookup(config.sequence_builder)()

    extra_tokens = [
        "<begin_of_action>",
    ] + [f"<act{i}>" for i in range(action_tokenizer.vocab_size)]
    language_tokenizer.add_tokens(extra_tokens)
    language_tokenizer.add_bos_token = False

    model_config = config.model_config.to_dict()
    model_config["llm_spec"]["config"]["vocab_size"] = len(language_tokenizer)

    model_spec = ModuleSpec(
        PaliVLACritic,
        freeze(model_config),
    )
    optimizer_spec = OptimizerSpec.create(
        make_optimizer,
        config.optimizer.kwargs.to_dict(),
    )

    return CriticModelComponents.initialize(
        model_spec=model_spec,
        optimizer_spec=optimizer_spec,
        seed=config.get("seed", 0),
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        sequence_builder=sequence_builder,
        sharding_metadata=sharding_metadata,
        example_batch=(
            example_batch["sensors"],
            example_batch["sensors_mask"],
            example_batch["prompt"],
            example_batch["actions"],
        ),
        critic_train_step_kwargs=config.critic_train_step_kwargs.to_dict(),
    )


def main(_):
    if flags.FLAGS.platform == "tpu":
        jax.distributed.initialize()

    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    sharding_metadata = make_sharding(config)

    if config.resume_checkpoint_dir is not None:
        # Load the model from a checkpoint
        model = CriticModelComponents.load_static(
            config.resume_checkpoint_dir, sharding_metadata
        )
        restore_manager = ocp.CheckpointManager(
            config.resume_checkpoint_dir, options=ocp.CheckpointManagerOptions()
        )
        model.load_state(config.resume_checkpoint_step, restore_manager)
    else:
        # Otherwise, create the model from scratch and apply any load_fns
        model = create_model(config, sharding_metadata)
        for load_fn, load_fn_kwargs in config.load_fns:
            load_fn = Registry.lookup(load_fn)
            load_fn(model, **load_fn_kwargs)

    # Make the basic dataset
    # We have to do this first, since we need to know how the dataset is set up before we can construct the model
    train_ds = make_base_dataset(
        **config.dataset_kwargs.to_dict(),
        train=True,
    )

    viz_datasets = {
        k: make_trajectory_dataset(
            **viz_dataset_kwargs.to_dict(),
            train=False,
        )
        for k, viz_dataset_kwargs in config.viz_traj_datasets.items()
    }
    viz_dataset_iters = {k: v.iterator() for k, v in viz_datasets.items()}

    viz_trajectories = {
        k: [next(v_iter) for _ in range(config.viz_num_trajectories)]
        for k, v_iter in viz_dataset_iters.items()
    }
    validation_ds = make_base_dataset(**config.dataset_kwargs.to_dict(), train=False)

    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    def make_training_batch(batch):
        return batch

    def make_validation_batch(batch):
        return batch

    train_it = map(
        make_training_batch,
        train_ds.batch(per_host_train_batch_size).iterator(),
    )

    if config.overfit_dataset:

        def _make_overfit_it(it):
            overfit_batch = next(it)
            while True:
                yield overfit_batch

        train_it = _make_overfit_it(train_it)

    eval_it = map(
        make_validation_batch,
        validation_ds.batch(per_host_eval_batch_size).iterator(),
    )

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {
            "project": config.wandb_project,
            "tags": [],
            "mode": config.wandb_mode,
            "name": config.wandb_experiment_name,
        }

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    if config.save_path is not None:
        checkpoint_save_path = tf.io.gfile.join(config.save_path, run_name)

        checkpoint_save_manager = ocp.CheckpointManager(
            checkpoint_save_path,
            options=ocp.CheckpointManagerOptions(max_to_keep=config.max_to_keep),
        )

        model.save_static(Path(tf.io.gfile.join(checkpoint_save_path)))

    train_infos = []

    # Main training loop
    start_step = model.train_state.step.item()

    with tqdm.trange(
        start_step, config.num_steps, desc="Training", dynamic_ncols=True
    ) as pbar:
        for i in pbar:
            batch = next(train_it)

            # Train step
            train_info = model.train_step(batch)
            train_info = jax.device_get(train_info)
            train_infos.append(train_info)
            pbar.set_postfix(
                loss=f"{train_info['loss']:.2f}",
                q_value=f"{train_info['q_value']:.2f}",
            )

            # Main logging
            if (i + 1) % config.log_interval == 0:
                avg_train_info = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *train_infos
                )
                if jax.process_index() == 0:
                    wandb.log(
                        flatten_wandb_dict(
                            {
                                "training": avg_train_info,
                                "batch_info": get_batch_info(batch),
                            }
                        ),
                        step=i,
                    )
                train_infos = []

            # Visualizations
            if (i + 1) % config.viz_interval == 0:
                for viz_ds_name, trajectories in viz_trajectories.items():
                    visualizations = {}
                    for j, trajectory in enumerate(trajectories):
                        image = visualize_critic(model, trajectory)
                        visualizations[f"{viz_ds_name}_{j}"] = wandb.Image(image)
                    if jax.process_index() == 0:
                        wandb.log(
                            flatten_wandb_dict({"visualizations": visualizations}),
                            commit=False,
                            step=i + 1,
                        )

            # Validation
            if (i + 1) % config.eval_interval == 0:
                eval_batch = next(eval_it)
                eval_info = model.eval_step(eval_batch)
                if jax.process_index() == 0:
                    wandb.log(
                        flatten_wandb_dict({"validation": eval_info}),
                        commit=False,
                        step=i,
                    )

            # Checkpointing
            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    checkpoint_save_manager.save(i + 1, args=model.save_args())

    if config.save_path is not None:
        checkpoint_save_manager.wait_until_finished()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test_critic.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)
