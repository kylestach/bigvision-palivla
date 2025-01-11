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
import imageio
from typing import List
from PIL import Image, ImageDraw, ImageFont

try:
    from simpler_env import simpler_env
    from simpler_env.simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )
except ImportError:
    import simpler_env
    from simpler_env.utils.env.observation_utils import (
        get_image_from_maniskill2_obs_dict,
    )

from transforms3d.euler import euler2axangle
from scripts.train_critic import make_sharding, create_model
from scripts.cache_actions import initialize_openvla, get_openvla_actions


jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
tf.config.set_visible_devices([], "GPU")


def main(_):
    # Turn off debug logs
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity(absl_logging.WARNING)

    tf.random.set_seed(0)
    FLAGS = flags.FLAGS

    config = FLAGS.config

    sharding_metadata = make_sharding(config)

    if config.resume_checkpoint_dir != "":
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

    # Create the environment
    env = simpler_env.make(FLAGS.task_name)

    # Initialize OpenVLA
    openvla_variables = initialize_openvla()

    # Get dataset for normalization statistics
    train_ds = make_base_dataset(**config.dataset_kwargs.to_dict(), train=True)
    bridge_metadata = train_ds.dataset_statistics["bridge_dataset"]
    example_batch = next(
        train_ds.batch(FLAGS.num_action_samples_from_base_policy).iterator()
    )

    trajectory_images = []
    trajectory_rewards = []
    for i in tqdm.trange(FLAGS.num_eval_episodes, desc="Evaluating episodes"):
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()

        image = get_image_from_maniskill2_obs_dict(
            env, obs
        )  # np.ndarray of shape (H, W, 3), uint8
        images = [image]
        rewards = [0.0]
        done = False

        # Sticky gripper variables
        num_consecutive_gripper_change_actions = 0
        is_gripper_closed = False

        while not done:

            # Resize image to 224x224
            image = tf.image.resize(
                image, (224, 224), method="lanczos3", antialias=True
            )
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            raw_actions = get_openvla_actions(
                openvla_variables,
                observation={"image_primary": image[None]},
                language_instruction=instruction,
                num_action_samples=FLAGS.num_action_samples_from_base_policy,
            )
            assert raw_actions.shape == (FLAGS.num_action_samples_from_base_policy, 7)

            # Normalize actions before running them through the critic.
            mask = bridge_metadata["action"]["mask"]
            p01 = bridge_metadata["action"]["p01"]
            p99 = bridge_metadata["action"]["p99"]
            normalized_actions = np.where(
                mask,
                np.clip(2 * (raw_actions - p01) / (p99 - p01 + 1e-8) - 1, -1, 1),
                raw_actions,
            )
            critic_observation = {
                "image_primary": image[None].repeat(
                    FLAGS.num_action_samples_from_base_policy, axis=0
                ),
                "pad_mask_dict": example_batch["observation"]["pad_mask_dict"],
            }
            values = model.predict(
                {
                    "observation": critic_observation,
                    "action": normalized_actions,
                    "task": {
                        "language_instruction": [instruction]
                        * FLAGS.num_action_samples_from_base_policy
                    },
                }
            )

            # Get the action with the highest value
            best_action_idx = np.argmax(values)
            best_action = raw_actions[best_action_idx]

            roll, pitch, yaw = best_action[3:6]
            gripper_open = best_action[6:7]

            action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
            action_rotation_axangle = action_rotation_ax * action_rotation_angle

            # Sticky gripper logic

            if (gripper_open.item() < 0.5) != is_gripper_closed:
                num_consecutive_gripper_change_actions += 1
            else:
                num_consecutive_gripper_change_actions = 0

            if num_consecutive_gripper_change_actions >= FLAGS.sticky_gripper_num_steps:
                is_gripper_closed = not is_gripper_closed
                num_consecutive_gripper_change_actions = 0

            gripper_action = -1 if is_gripper_closed else 1

            # action = {
            #     "world_vector": best_action[:3],
            #     "rot_axangle": action_rotation_axangle,
            #     "gripper": np.array([gripper_action]),
            # }
            action = np.concatenate(
                [best_action[:3], action_rotation_axangle, np.array([gripper_action])]
            )
            obs, reward, success, truncated, info = env.step(action)
            image = get_image_from_maniskill2_obs_dict(env, obs)
            images.append(image)
            rewards.append(reward)

            done = success or truncated

        trajectory_images.append(images)
        trajectory_rewards.append(rewards)

    # Define video path
    video_path = f"simpler_eval_{FLAGS.task_name}.mp4"

    # Save the video
    all_frames: List[np.ndarray] = []
    for trajectory_idx, (trajectory, rewards) in enumerate(
        zip(trajectory_images, trajectory_rewards)
    ):
        for frame_idx, (frame, reward) in enumerate(zip(trajectory, rewards)):
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)

            # Create semi-transparent black box for text
            box_coords = (10, 10, 400, 80)
            draw.rectangle(box_coords, fill=(0, 0, 0, 128))

            # Add instruction and reward text
            draw.text(
                (20, 20),
                f"Instruction: {env.get_language_instruction()[:50]}...",
                fill=(255, 255, 255),
            )
            draw.text((20, 45), f"Reward: {reward:.3f}", fill=(255, 255, 255))

            frame_with_text = np.array(pil_image)
            all_frames.append(frame_with_text)

        # Add blank frames between trajectories
        blank_frame = np.zeros_like(trajectory[0])
        all_frames.extend([blank_frame] * 30)

    print(f"Saving video with {len(all_frames)} frames to {video_path}")
    imageio.mimsave(video_path, all_frames, fps=30, quality=8, macro_block_size=1)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    config_flags.DEFINE_config_file(
        "config", "configs/smoke_test_critic.py", "Path to the config file."
    )
    flags.DEFINE_string("task_name", "widowx_spoon_on_towel", "Task name.")
    flags.DEFINE_integer("num_eval_episodes", 100, "Number of evaluation episodes.")
    flags.DEFINE_integer(
        "sticky_gripper_num_steps", 1, "Number of sticky gripper steps."
    )
    flags.DEFINE_integer(
        "num_action_samples_from_base_policy",
        8,
        "Number of action samples from the base policy (E.g. OpenVLA).",
    )

    app.run(main)
