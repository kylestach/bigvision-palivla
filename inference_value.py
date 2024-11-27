import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import jax
import jax.numpy as jnp
from absl import app, flags
import optax
import json
import numpy as np
from functools import partial
import orbax.checkpoint as ocp
from tensorflow_text import SentencepieceTokenizer
# from palivla.dataset import prepare_image
from palivla.tokenizer import Tokenizer
from palivla.load_model import load_model_params_decode
from scalax.sharding import MeshShardingHelper, FSDPShardingRule, PartitionSpec
from flax.training.train_state import TrainState
from jax.experimental import multihost_utils
from ml_collections import config_flags
from palivla.train_state import PaliVLATrainState
from palivla.types import TrainingBatch, RolloutBatch

def unnormalize_action_minmax(action, unnormalization_statistics):
    mask = unnormalization_statistics.get(
        "mask", jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = jnp.where(
        mask,
        (action + 1) / 2 * (unnormalization_statistics["p99"] - unnormalization_statistics["p01"]) + unnormalization_statistics["p01"],
        action,
    )

    return action

def unnormalize_action(action, unnormalization_statistics):
    mask = unnormalization_statistics.get(
        "mask", jnp.ones_like(unnormalization_statistics["mean"], dtype=bool)
    )
    action = action[..., : len(mask)]
    action = jnp.where(
        mask,
        (action * unnormalization_statistics["std"])
        + unnormalization_statistics["mean"],
        action,
    )
    return action


def main(_):
    # jax.distributed.initialize()

    config = flags.FLAGS.config
    # Sharding
    mesh = MeshShardingHelper([-1], ["fsdp"])

    model_sharding = FSDPShardingRule("fsdp", fsdp_axis_size=mesh.mesh.shape["fsdp"])
    data_sharding = PartitionSpec("fsdp")
    # data_sharding = jax.sharding.SingleDeviceSharding(jax.local_devices()[0])

    restore_checkpoint_manager = ocp.CheckpointManager(
            flags.FLAGS.resume_from_checkpoint_dir,
            item_handlers=PaliVLATrainState.get_checkpoint_handlers(),
    )

    model = PaliVLATrainState.restore(
            checkpoint_manager=restore_checkpoint_manager,
            step=flags.FLAGS.resume_from_checkpoint_step,
            load_optimizer=False,
            mesh=mesh,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
        )

    tokenizer = model.tokenizer
    decode = model.decode
    dataset_statistics = model.dataset_statistics

    optimizer = optax.identity()
    def get_value(pred_values, tokens, tokenizer_config: Tokenizer.TokenizerConfig):
        value_token_starts = jnp.argmax(tokens == tokenizer_config.end_of_action_token, axis=-1)
        _get_values = jax.vmap(
            lambda x, i: jax.lax.dynamic_slice(x, (i,), (1,))
        )
        qs = jax.vmap(_get_values, in_axes=(-1, None), out_axes=-1)(
            pred_values, value_token_starts
        ).squeeze()
        return qs
    # action_mean = np.array(dataset_statistics[flags.FLAGS.dataset_name]["action"]["mean"])
    # action_std = np.array(dataset_statistics[flags.FLAGS.dataset_name]["action"]["std"])
    # action_mask = np.array(dataset_statistics[flags.FLAGS.dataset_name]["action"]["mask"])
    def make_inference_batch(batch):
        sensors = {
            k: batch["observation"][k][None].numpy()
            for k in batch["observation"]
            if k in model.model_state.model.modality_mappings and k != "text"
        }
        sensors_mask = {
            k: batch["observation"]["pad_mask_dict"][k].numpy()
            for k in model.model_state.model.modality_mappings
            if k != "text"
        }
        return TrainingBatch(
                sensors=sensors,
                sensors_mask=sensors_mask,
                sensors_next=None,
                sensors_next_mask=None,
                actions=batch["action"].numpy(),
                tokens=batch["tokens"].numpy(),
                tokens_ar=batch["mask_ar"].numpy(),
                tokens_loss=batch.get("mask_loss", None),
                tokens_mask=batch["mask_input"].numpy(),
                rewards=None,
                td_mask=None,
                mc_returns=None,
                next_actions=None,
                next_tokens=None,
                gen_start=None,
            )

    # Do inference
    def do_inference(images, instructions, action):
        data = {
            "observation": {"image_primary": images, "pad_mask_dict": {"image_primary": tf.ones(len(images), dtype=tf.bool)}},
            "task": {"language_instruction": instructions},
            "action": action,
        }
        language_token_instructions = tokenizer.tokenize_language_instruction(data)
        # batch = prepare_image(batch)
        batch = tokenizer.prepare_tokens_for_training(data, language_token_instructions)
        batch = batch | data
        batch = make_inference_batch(batch)
        key = jax.random.PRNGKey(0)
        key, key_value = jax.random.split(key)
        all_inputs = batch.sensors | {"text": batch.tokens[..., :-1][None]}
        all_masks = batch.sensors_mask | {
            "text": jnp.ones_like(batch.tokens[..., :-1][None], dtype=jnp.bool_)
        }
        logits, info = model.model_state.apply_fn(
            {"params": model.model_state.params},
            all_inputs,
            data_masks=all_masks,
            text_ar_mask=batch.tokens_ar[..., :-1][None],
            train=False,
            rngs={"dropout": key},
        )
        values = info["values"]
        # qs = get_value(values, batch.tokens[..., 1:][None], tokenizer.config)
        qs = get_value(values, batch.tokens[..., :-1][None], tokenizer.config)

        return qs

    for _ in range(10):
        images = tf.random.normal((1, 224, 224, 3))
        # images = tf.cast(images, tf.float32)  / 127.5 - 1.0
        instructions = tf.constant("place the mushroom in the pot")
        action = tf.random.normal((1, 1, 7))
        qs = do_inference(images, instructions, action)
        print(qs)


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "/nfs/nfs2/users/mitsuhiko/codes/bigvision-palivla/palivla/configs/bridge_config.py", "Path to the config file."
    )
    flags.DEFINE_string(
        "dataset_name", "bridge_dataset", "Name of the dataset to use for inference."
    )
    flags.DEFINE_string(
        "resume_from_checkpoint_dir", "gs://rail-tpus-mitsuhiko-central2/logs/test/clean-field-7/", "Path to the checkpoint directory."
        # "resume_from_checkpoint_dir", "/nfs/nfs2/users/mitsuhiko/codes/bigvision-palivla/checkpoints", "Path to the checkpoint directory."
    )
    flags.DEFINE_integer(
        "resume_from_checkpoint_step", 100000, "Step to resume from."
    )
    app.run(main)
