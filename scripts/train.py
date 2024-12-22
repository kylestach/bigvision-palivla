import os

from big_vision.utils import Registry
from palivla.components.model import PaliVLAModel
from palivla.components.sequence_builder import SequenceBuilder
from palivla.components.train_state import ShardingMetadata
from palivla.components.action_tokenizer import ActionTokenizer

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from transformers import AutoTokenizer
import tensorflow as tf
import tqdm
from absl import app, flags, logging as absl_logging
from ml_collections import ConfigDict, config_flags
from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
)

import wandb
import numpy as np
from flax.core.frozen_dict import freeze

import palivla.load_fns
from palivla.dataset import make_base_dataset
from palivla.optimizer import make_optimizer
from palivla.spec import ModuleSpec, OptimizerSpec
from palivla.model_components import ModelComponents
from palivla.utils import host_broadcast_str

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


def create_model(config: ConfigDict, sharding_metadata: ShardingMetadata):
    example_batch = {
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
        "gen": {
            "tokens": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.int32),
            "mask": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_ar": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.bool_),
            "mask_loss": jax.ShapeDtypeStruct(shape=(1, 10), dtype=jnp.float32),
        },
    }

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
        PaliVLAModel,
        freeze(model_config),
    )
    optimizer_spec = OptimizerSpec.create(
        make_optimizer,
        config.optimizer.kwargs.to_dict(),
    )

    return ModelComponents.initialize(
        model_spec=model_spec,
        optimizer_spec=optimizer_spec,
        seed=config.get("seed", 0),
        language_tokenizer=language_tokenizer,
        action_tokenizer=action_tokenizer,
        sequence_builder=sequence_builder,
        sharding_metadata=sharding_metadata,
        example_batch=example_batch,
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
        model = ModelComponents.load_static(
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
    train_ds = make_base_dataset(**config.dataset_kwargs.to_dict(), train=True)

    # Construct the final dataset
    # We need to do this after the model is constructed, since we need to have a tokenizer
    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    def make_training_batch(batch):
        return batch

    train_it = map(
        make_training_batch,
        train_ds.batch(per_host_train_batch_size).iterator(),
    )

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {
            "project": config.wandb_project,
            "tags": [],
            "mode": config.wandb_mode,
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

    wandb_logs = []

    # Main training loop
    start_step = model.train_state.step.item()

    if config.overfit_dataset:
        batch = next(train_it)

    with tqdm.trange(
        start_step, config.num_steps, desc="Training", dynamic_ncols=True
    ) as pbar:
        for i in pbar:
            if not config.overfit_dataset:
                batch = next(train_it)
            info = model.train_step(batch)

            info = jax.device_get(info)
            wandb_logs.append(info)
            pbar.set_postfix(
                loss=f"{info['loss']:.4f}",
            )

            if (i + 1) % config.log_interval == 0:
                avg_info = jax.tree.map(
                    lambda *xs: np.mean(np.stack(xs), axis=0), *wandb_logs
                )
                if jax.process_index() == 0:
                    wandb.log(avg_info, step=i)
                wandb_logs = []

            if (i + 1) % config.eval_interval == 0:
                print(model.predict(batch))
                '''
                eval_info = {}
                eval_batch = next(gen_eval_it)
                eval_info = model.eval_step(
                    eval_batch, "eval/gen_", include_regular_stats=False
                )

                train_batch_for_eval = next(gen_train_it)
                train_info = model.eval_step(train_batch_for_eval, "train/gen_")

                if jax.process_index() == 0:
                    wandb.log(
                        eval_info | train_info,
                        commit=False,
                        step=i,
                    )
                '''

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    checkpoint_save_manager.save(i + 1, args=model.save_args())
    if config.save_path is not None:
        checkpoint_save_manager.wait_until_finished()


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    app.run(main)
