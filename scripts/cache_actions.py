"""
Create a pickle with a dictionary of (trajectory_index, timestep) to actions.

Only supports Bridge and OpenVLA for now.
"""

import os
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import orbax.checkpoint as ocp
import tensorflow as tf
from tqdm import tqdm
import wandb

# OpenVLA imports
import torch
from absl import app, flags
from absl import logging as absl_logging
from ml_collections import config_flags
from palivla.dataset import make_base_single_dataset
from PIL import Image
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform
from transformers import AutoProcessor

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def main(_):
    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_INIT_TIMEOUT"] = "120"
    wandb.require("core")

    config = flags.FLAGS.config

    wandb.init(
        project=config.wandb_project,
        name=f"bridge_openvla_action_caching_worker_{flags.FLAGS.worker_id}_of_{flags.FLAGS.num_workers}",
    )
    wandb.config.update(config.to_dict())

    os.makedirs(os.path.dirname(flags.FLAGS.output_path), exist_ok=True)

    # train_ds = make_base_dataset(**config.dataset_kwargs.to_dict(), train=True)
    train_ds = make_base_single_dataset(
        name="bridge_dataset",
        **config.dataset_kwargs.to_dict(),
        **config.dataset_kwargs.oxe_kwargs.to_dict(),
        train=flags.FLAGS.train_partition,
    )
    num_trajectories = 53192 if flags.FLAGS.train_partition else 6872
    it = train_ds.iterator()
    # Create OpenVLA agent
    openvla_processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )
    openvla_model = OpenVLAForActionPrediction.from_pretrained(
        "openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    assert (
        torch.cuda.is_available()
    ), "OpenVLA requires a GPU to run, but no GPU was found"
    device_id = "cuda:0"
    torch.cuda.empty_cache()
    openvla_model = openvla_model.to(device_id)
    openvla_action_tokenizer = ActionTokenizer(openvla_processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        openvla_action_tokenizer,
        openvla_processor.tokenizer,
        image_transform=openvla_processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )
    collator = PaddedCollatorForActionPrediction(
        openvla_processor.tokenizer.model_max_length,
        openvla_processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    frame_key_to_actions = {}

    def get_actions(observation: dict, language_instruction: str):
        image = observation["image_primary"][0]
        assert len(image.shape) == 3
        images: List[Image.Image] = [
            Image.fromarray(image)
        ] * flags.FLAGS.num_action_samples
        openvla_inputs = openvla_processor(
            [language_instruction] * flags.FLAGS.num_action_samples,
            images,
        ).to(device_id, dtype=torch.bfloat16)

        actions = openvla_model.predict_action(
            **openvla_inputs,
            unnorm_key="bridge_orig",
            do_sample=True,
        ).reshape(flags.FLAGS.num_action_samples, 7)
        return actions

    trajectory_index = 0
    processed_trajectories = 0
    num_total_partitions = (
        num_trajectories // flags.FLAGS.num_workers // flags.FLAGS.save_interval
    )
    with tqdm(total=(num_trajectories)) as pbar:
        while True:
            pbar.update(1)
            try:
                trajectory = next(it)

            except StopIteration:
                break

            trajectory_index += 1

            if (
                trajectory_index - 1
            ) % flags.FLAGS.num_workers != flags.FLAGS.worker_id:
                continue

            save_path = flags.FLAGS.output_path.format(
                id=flags.FLAGS.worker_id,
                workers=flags.FLAGS.num_workers,
                partition=processed_trajectories // flags.FLAGS.save_interval,
                num_partitions=num_total_partitions,
            )
            processed_trajectories += 1

            # If the file already exists, skip
            if os.path.exists(save_path):
                continue

            trajectory_length = len(trajectory["action"])
            language_instruction = trajectory["task"]["language_instruction"][0].decode(
                "utf-8"
            )

            if flags.FLAGS.dry_run:
                continue
            for i in range(trajectory_length):
                frame_key = trajectory["frame_key"][i].decode("utf-8")
                if frame_key not in frame_key_to_actions:
                    frame_key_to_actions[frame_key] = get_actions(
                        jax.tree.map(lambda x: x[i], trajectory["next_observation"]),
                        language_instruction,
                    )

            if processed_trajectories % flags.FLAGS.save_interval == 0:
                print(
                    f"Saving partition {(processed_trajectories-1) // flags.FLAGS.save_interval}"
                )
                with open(save_path, "wb") as f:
                    pickle.dump(frame_key_to_actions, f)
                frame_key_to_actions = {}
                wandb.log(
                    {
                        "partitions_left": num_total_partitions
                        - processed_trajectories // flags.FLAGS.save_interval
                    }
                )

    print(f"Processed {trajectory_index} trajectories")

    # Save the remaining actions
    if not flags.FLAGS.dry_run:
        with open(
            flags.FLAGS.output_path.format(
                id=flags.FLAGS.worker_id,
                workers=flags.FLAGS.num_workers,
                partition="final",
                num_partitions=num_total_partitions,
            ),
            "wb",
        ) as f:
            pickle.dump(frame_key_to_actions, f)


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/bridge_critic_config.py", "Path to the config file."
    )
    flags.DEFINE_integer("num_workers", 1, "Number of workers.")
    flags.DEFINE_integer("worker_id", 0, "Worker ID.")
    flags.DEFINE_integer(
        "num_action_samples", 8, "Number of times to repeat the action sampling."
    )
    flags.DEFINE_boolean("dry_run", False, "Dry run.")
    flags.DEFINE_string(
        "output_path",
        "bridge_openvla_cached_actions/worker_{id}_of_{workers}_partition_{partition}_of_{num_partitions}.pkl",
        "Output path.",
    )
    flags.DEFINE_boolean(
        "train_partition", True, "Use train partition of Bridge (otherwise validation)."
    )
    flags.DEFINE_integer(
        "save_interval", 100, "Will save every `save_interval` trajectories."
    )
    app.run(main)
