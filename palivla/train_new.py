import jax
import jax.numpy as jnp

import tensorflow as tf
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorflow_text import SentencepieceTokenizer
from scalax.sharding import (
    MeshShardingHelper,
    FSDPShardingRule,
    PartitionSpec,
)

import wandb
import numpy as np
from flax import linen as nn
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp

from palivla.model import make_optimizer
from palivla.spec import OptimizerSpec
from palivla.train_state import PaliVLA
from palivla.train_step import step_fn
from palivla.dataset import make_dataset
from palivla.types import Data

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def host_broadcast_str(x: str | None) -> str:
    """
    Broadcast_one_to_all, but with a string.

    Works by padding the string to the length of the longest string and then
    broadcasting the result, then stripping the padding.

    Note: this will remove the padding from the end of the string.
    """
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()


def compute_gen_stats(model: PaliVLA, batch: Data, prefix: str):
    """
    Compute generative (rollout) statistics on a batch of data.
    """
    out_tokens = model.decode(
        model.process_image(batch["observation"]),
        batch["tokens"],
        batch["mask_ar"],
        batch["mask_input"],
    )
    out_tokens = jax.device_get(multihost_utils.process_allgather(out_tokens, tiled=True))

    decoded_actions = model.tokenizer.bin_detokenize(out_tokens)

    gt_action = jax.device_get(multihost_utils.process_allgather(batch["action"], tiled=True))
    gt_action = np.squeeze(gt_action, axis=(1, 2))
    gt_action_tokens = model.tokenizer.bin_tokenize(gt_action)

    info = {
        f"{prefix}/rollout_mae": np.mean(np.abs(decoded_actions - gt_action)),
        f"{prefix}/rollout_mse": np.mean(np.square(decoded_actions - gt_action)),
        f"{prefix}/rollout_acc": np.mean(out_tokens == gt_action_tokens),
    }
    for j in range(gt_action.shape[-1]):
        error = decoded_actions[..., j] - gt_action[..., j]
        info[f"details/{prefix}/rollout_mse_{j}"] = np.mean(np.square(error))
        info[f"details/{prefix}/rollout_mae_{j}"] = np.mean(np.abs(error))
    for j in range(gt_action_tokens.shape[-1]):
        info[f"details/{prefix}/rollout_acc_{j}"] = np.mean(
            out_tokens[:, j] == gt_action_tokens[:, j]
        )

    return info


def main(_):
    jax.distributed.initialize()

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    predict_prompt = config.get("predict_prompt", False)

    # Setup mesh and sharding for model and data
    mesh = MeshShardingHelper(
        [-1], ["fsdp"]
    )

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec("fsdp")

    print("Loading model...")
    # Load model
    if config.resume_from_checkpoint_dir is not None:
        model = PaliVLA.from_checkpoint(
            config.resume_from_checkpoint_dir,
            tokenizer_path=config.tokenizer_path,
            step=config.resume_from_checkpoint_step,
            save_directory=config.save_path,
            mesh=mesh,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            load_optimizer=config.load_optimizer,
            model_dtype=jnp.float32,
            image_keys=config.image_keys,
        )
    else:
        model = PaliVLA.from_pretrained(
            config.model_path,
            config.tokenizer_path,
            prompt_autoregressive=predict_prompt,
            optimizer_spec=OptimizerSpec(
                make_optimizer, config.optimizer_kwargs.to_dict()
            ),
            mesh=mesh,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
            model_dtype=jnp.float32,
            dataset_statistics={},
            rng_seed=config.get("seed", 0),
            checkpoint_save_path=config.save_path,
            image_keys=config.image_keys,
        )

    print("Constructing dataset...")
    # Load dataset
    train_ds = make_dataset(config, model.tokenizer, train=True, generation=False)
    model.dataset_statistics = train_ds.dataset_statistics

    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    train_it = train_ds.batch(per_host_train_batch_size).iterator()
    gen_eval_it = (
        make_dataset(config, model.tokenizer, train=False, generation=True)
        .batch(per_host_eval_batch_size)
        .iterator()
    )
    gen_train_it = (
        make_dataset(config, model.tokenizer, train=True, generation=True)
        .batch(per_host_eval_batch_size)
        .iterator()
    )

    # W&B setup
    if jax.process_index() == 0:
        wandb_kwargs = {"project": config.wandb_project}

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    wandb_logs = []

    # Main training loop
    with tqdm.trange(config.num_steps, desc="Training") as pbar:
        # Skip ahead if resuming from checkpoint
        pbar.update(model.train_state.step.item())

        for i in pbar:
            batch = next(train_it)
            batch_train = {
                "image": model.process_image(batch["observation"]),
                "tokens": batch["tokens"],
                "input_mask": batch["mask_input"],
                "mask_ar": batch["mask_ar"],
                "mask_loss": batch["mask_loss"],
            }

            info = model.train_step(batch_train)

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
                info = compute_gen_stats(
                    model, next(gen_eval_it), "gen/eval"
                ) | compute_gen_stats(model, next(gen_train_it), "gen/train")

                if jax.process_index() == 0:
                    wandb.log(
                        info,
                        commit=False,
                        step=i,
                    )

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    print(f"Saving model to {config.save_path}/{i}")
                    model.save(i + 1)


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "bridge_config.py", "Path to the config file."
    )
    app.run(main)
