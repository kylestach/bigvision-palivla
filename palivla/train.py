from functools import partial
import json

import jax
import tensorflow as tf
import tqdm
from absl import app, flags
from flax.training.train_state import TrainState
from ml_collections import ConfigDict, config_flags
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

from palivla.model import load_model, make_optimizer
from palivla.tokenizer import Tokenizer
from palivla.train_step import step_fn
from palivla.dataset import make_dataset

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def host_broadcast_str(x: str | None) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()


def main(_):
    jax.distributed.initialize()

    tf.random.set_seed(jax.process_index())

    config = flags.FLAGS.config

    predict_prompt = config.get("predict_prompt", False)

    with open(config.tokenizer_path, "rb") as f:
        language_tokenizer = SentencepieceTokenizer(f.read())
    tokenizer = Tokenizer.from_tokenizer(language_tokenizer, prompt_autoregressive=predict_prompt)

    print("Constructing dataset...")
    per_host_train_batch_size = config.batch_size // jax.process_count()
    per_host_eval_batch_size = config.eval_batch_size // jax.process_count()

    train_ds = make_dataset(config, tokenizer, train=True, generation=False)
    train_it = train_ds.batch(per_host_train_batch_size).iterator()
    gen_eval_it = (
        make_dataset(config, tokenizer, train=False, generation=True)
        .batch(per_host_eval_batch_size)
        .iterator()
    )
    gen_train_it = (
        make_dataset(config, tokenizer, train=True, generation=True)
        .batch(per_host_eval_batch_size)
        .iterator()
    )

    print("Loading model params...")
    model, params, decode = load_model(config, tokenizer)

    print("Initializing model...")

    optimizer = make_optimizer(config)

    mesh = MeshShardingHelper(
        [config.data_axis_size, config.fsdp_axis_size], ["data", "fsdp"]
    )

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec(("data", "fsdp"))

    @partial(mesh.sjit, in_shardings=None, out_shardings=model_sharding)
    def init_fn(params):
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )

    train_state = init_fn(params)
    del params

    key = jax.random.PRNGKey(0)

    if jax.process_index() == 0:
        wandb_kwargs = {"project": config.wandb_project}

        wandb.init(**wandb_kwargs)
        wandb.config.update(config.to_dict())

        run_name = wandb.run.name
    else:
        run_name = None

    run_name = host_broadcast_str(run_name)

    if config.save_path is not None:
        checkpoint_manager = ocp.CheckpointManager(
            f"{config.save_path}/{run_name}/checkpoints",
            item_names=["state", "config", "dataset_statistics"],
            options=ocp.CheckpointManagerOptions(
                max_to_keep=1
            ),
        )

    dataset_statistics = jax.tree.map(lambda x: x.tolist(), train_ds.dataset_statistics)

    # Save config and dataset statistics
    if config.save_path is not None:
        with tf.io.gfile.GFile(f"{config.save_path}/{run_name}/config.json", "w") as f:
            json.dump(config.to_dict(), f)
        with tf.io.gfile.GFile(
            f"{config.save_path}/{run_name}/dataset_statistics.json", "w"
        ) as f:
            json.dump(dataset_statistics, f)

    if config.resume_from_checkpoint_dir is not None:
        assert (
            config.save_path is not None
        ), "Must provide save_path to resume from checkpoint"

        train_state = ocp.CheckpointManager(
            config.resume_from_checkpoint_dir,
            item_names=["state"],
            options=ocp.CheckpointManagerOptions(),
        ).restore(
            config.resume_from_checkpoint_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(train_state)
            )
        )['state']

    jit_step_fn = mesh.sjit(
        step_fn,
        in_shardings=(model_sharding, data_sharding, None),
        out_shardings=(model_sharding, None, None),
        static_argnums=(3,),
        args_sharding_constraint=(
            model_sharding,
            data_sharding,
            None,
        ),
        donate_argnums=(0,),
    )

    wandb_logs = []

    # Main training loop
    with tqdm.trange(train_state.step, config.num_steps, desc="Training") as pbar:
        for i in pbar:
            batch = next(train_it)
            batch_train = mesh.local_data_to_global_array(
                {
                    "image": batch["observation"]["images"],
                    "tokens": batch["tokens"],
                    "input_mask": batch["mask_input"],
                    "mask_ar": batch["mask_ar"],
                    "mask_loss": batch["mask_loss"],
                }
            )

            with mesh.mesh, nn.logical_axis_rules([("act_batch", "fsdp")]):
                train_state, info, key = jit_step_fn(
                    train_state,
                    batch_train,
                    key,
                    tokenizer.config,
                )

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
                def compute_gen_stats(batch, prefix):
                    batch = mesh.local_data_to_global_array(
                        {
                            "image": batch["observation"]["images"],
                            "text": batch["tokens"],
                            "mask_ar": batch["mask_ar"],
                            "mask_input": batch["mask_input"],
                            "action": np.squeeze(batch["action"], axis=(1, 2)),
                            "_mask": np.ones(batch["tokens"].shape[0], dtype=np.bool_),
                        }
                    )
                    out_tokens = decode(
                        {"params": train_state.params},
                        batch,
                        model=model,
                        devices=jax.devices(),
                        max_decode_len=tokenizer.config.num_action_tokens,
                        replicate_out=True,
                        mesh=mesh.mesh,
                    )
                    out_tokens = jax.device_get(
                        multihost_utils.process_allgather(out_tokens)
                    )

                    decoded_actions = tokenizer.bin_detokenize(out_tokens)

                    gt_action = jax.device_get(
                        multihost_utils.process_allgather(batch["action"])
                    )
                    gt_action_tokens = tokenizer.bin_tokenize(gt_action)

                    info = {
                        f"{prefix}/rollout_mae": np.mean(
                            np.abs(decoded_actions - gt_action)
                        ),
                        f"{prefix}/rollout_mse": np.mean(
                            np.square(decoded_actions - gt_action)
                        ),
                        f"{prefix}/rollout_acc": np.mean(
                            out_tokens == gt_action_tokens
                        ),
                    }
                    for j in range(gt_action.shape[-1]):
                        error = decoded_actions[:, j] - gt_action[:, j]
                        info[f"details/{prefix}/rollout_mse_{j}"] = np.mean(
                            np.square(error)
                        )
                        info[f"details/{prefix}/rollout_mae_{j}"] = np.mean(
                            np.abs(error)
                        )
                    for j in range(gt_action_tokens.shape[-1]):
                        info[f"details/{prefix}/rollout_acc_{j}"] = np.mean(
                            out_tokens[:, j] == gt_action_tokens[:, j]
                        )

                    return info

                info = compute_gen_stats(
                    next(gen_eval_it), "gen/eval"
                ) | compute_gen_stats(next(gen_train_it), "gen/train")

                if jax.process_index() == 0:
                    wandb.log(
                        info,
                        commit=False,
                        step=i,
                    )

            if (i + 1) % config.save_interval == 0:
                if config.save_path is not None:
                    print(f"Saving model to {config.save_path}/{i}")
                    checkpoint_manager.save(
                        i + 1,
                        args=ocp.args.Composite(
                            state=ocp.args.StandardSave(train_state),
                            config=ocp.args.JsonSave(config.to_dict()),
                            dataset_statistics=ocp.args.JsonSave(
                                dataset_statistics
                            ),
                        ),
                    )


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "bridge_config.py", "Path to the config file."
    )
    app.run(main)
