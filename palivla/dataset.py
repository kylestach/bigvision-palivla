from functools import partial
from typing import Optional, Sequence
import tensorflow as tf
from ml_collections import ConfigDict
import numpy as np
import dlimp
import jax
import jax.numpy as jnp

from octo.data.utils.data_utils import NormalizationType
from octo.utils.fuse_constants import create_batch
from palivla.tokenizer import Tokenizer
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights, make_oxe_dataset_kwargs
from palivla.types import TrainingBatch, Data

# Filter unmasked modalities to match modality_idx
# the modalities, in order, are:
# [('simple'), (), ('visual',), ('tactile',), ('audio',), ('visual', 'tactile'), ('visual', 'audio'), ('tactile', 'audio'), ('visual', 'tactile', 'audio')]
def create_fuse_modal_mask(data):
    sensor_masks = data["observation"]["pad_mask_dict"]
    modality_idx = data["observation"]["modality_idx"]

    modal_masks = jax.tree.map(lambda leaf: leaf, sensor_masks)

    modal_masks["image_primary"] = sensor_masks["image_primary"] & (
        (modality_idx == 0) | (modality_idx == 2) | (modality_idx == 5) | (modality_idx == 6) | (modality_idx == 8)
    )
    modal_masks["image_wrist"] = sensor_masks["image_wrist"] & (
        (modality_idx == 0) | (modality_idx == 2) | (modality_idx == 5) | (modality_idx == 6) | (modality_idx == 8)
    )
    modal_masks["image_digit_left"] = sensor_masks["image_digit_left"] & (
        (modality_idx == 0) | (modality_idx == 3) | (modality_idx == 5) | (modality_idx == 7) | (modality_idx == 8)
    )
    modal_masks["image_digit_right"] = sensor_masks["image_digit_right"] & (
        (modality_idx == 0) | (modality_idx == 3) | (modality_idx == 5) | (modality_idx == 7) | (modality_idx == 8)
    )
    modal_masks["mel_spectro"] = sensor_masks["mel_spectro"] & (
        (modality_idx == 0) | (modality_idx == 4) | (modality_idx == 6) | (modality_idx == 7) | (modality_idx == 8)
    )
    data["observation"]["modal_pad_mask_dict"] = modal_masks
    return data

# some invalid instructions are included for batching reasons
# e.g. a tactile-based command for a button-pressing task
# we mask out the loss for these invalid instructions
def enforce_valid_language_instruction(
    batch: TrainingBatch,
    mask_loss_fuse: jax.Array,
):
    mask_loss_fuse = mask_loss_fuse & jnp.expand_dims(batch.language_validity[np.arange(batch.modality_idx.shape[0]), batch.modality_idx[:, 0]], axis=-1)
    return mask_loss_fuse


def process_rephrase_tf(
    data,
):
    data['task'].pop('modal_commands')
    return data


def mel_spectro_to_image(data):
    mel_spectro_mean = -6.163405
    mel_spectro_std = 14.010228

    # Normalize and clip the mel spectrogram
    normalized_mel_spectro = tf.clip_by_value(
        (data["observation"]["mel_spectro"][-1] - mel_spectro_mean) / mel_spectro_std,
        -1,
        1,
    )

    # Repeat the channel dimension to create a 3-channel image
    grayscale_mel_spectro = tf.repeat(
        normalized_mel_spectro[..., tf.newaxis], repeats=3, axis=-1
    )

    # Resize the mel spectrogram to 224x224
    resized_mel_spectro = tf.image.resize(grayscale_mel_spectro, [224, 224])

    data["observation"]["mel_spectro"] = resized_mel_spectro
    return data


def tactile_to_image(data):
    def normalize_and_clip(image, mean, std):
        resized_image = tf.image.resize(tf.cast(image[-1], tf.float32), [224, 224])
        normalized_image = (resized_image - mean) / std / 3
        return tf.clip_by_value(normalized_image, -1, 1)

    tactile_mean = tf.constant([0.209282, -0.23046867, 0.07245745])
    tactile_std = tf.constant([6.41063034, 3.83920391, 4.75675555])

    data["observation"]["image_digit_left"] = normalize_and_clip(
        data["observation"]["image_digit_left"], tactile_mean, tactile_std
    )
    data["observation"]["image_digit_right"] = normalize_and_clip(
        data["observation"]["image_digit_right"], tactile_mean, tactile_std
    )
    return data


def make_base_dataset(
    *,
    oxe_kwargs: dict,
    train: bool,
    shuffle_buffer_size: int,
    frame_transform_kwargs: dict,
    traj_transform_kwargs: dict,
    batch_size: Optional[int] = None,
    balance_weights: bool,
    traj_transform_threads: int,
    traj_read_threads: int,
) -> dlimp.DLataset:
    dataset_kwargs_list, sample_weights = (
        make_oxe_dataset_kwargs_and_weights(**oxe_kwargs)
    )
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=shuffle_buffer_size,
        frame_transform_kwargs=frame_transform_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        batch_size=batch_size,
        balance_weights=balance_weights,
        traj_transform_threads=traj_transform_threads,
        traj_read_threads=traj_read_threads,
    )

    return dataset

# quick hack to make digit dataset in the same way
def make_base_dataset_digit(
    *,
    dataset_kwargs_list: list[dict],
    train: bool,
    shuffle_buffer_size: int,
    frame_transform_kwargs: dict,
    traj_transform_kwargs: dict,
    batch_size: Optional[int] = None,
    balance_weights: bool,
    traj_transform_threads: int,
    traj_read_threads: int,
) -> dlimp.DLataset:
    sample_weights = [ 1.0 for _ in dataset_kwargs_list ]
    sample_weight_norm = sum(sample_weights)
    sample_weights = [weight / sample_weight_norm for weight in sample_weights]
    sample_weights[-1] += 1 - sum(sample_weights)
    
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        shuffle_buffer_size=shuffle_buffer_size,
        frame_transform_kwargs=frame_transform_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
        batch_size=batch_size,
        balance_weights=balance_weights,
        traj_transform_threads=traj_transform_threads,
        traj_read_threads=traj_read_threads,
    )

    return dataset


def make_base_single_dataset(
    *,
    name: str,
    data_dir: str,
    load_camera_views: Sequence[str],
    load_depth: bool,
    load_proprio: bool,
    load_language: bool,
    force_recompute_dataset_statistics: bool,
    action_proprio_normalization_type: NormalizationType,
    train: bool,
    frame_transform_kwargs: dict,
    traj_transform_kwargs: dict,
    batch_size: Optional[int] = None,
) -> dlimp.DLataset:
    dataset = make_single_dataset(
        make_oxe_dataset_kwargs(
            name,
            data_dir,
            load_camera_views,
            load_depth,
            load_proprio,
            load_language,
            force_recompute_dataset_statistics,
            action_proprio_normalization_type,
        ),
        train=train,
        frame_transform_kwargs=frame_transform_kwargs,
        traj_transform_kwargs=traj_transform_kwargs,
    )
    dataset_statistics = dataset.dataset_statistics
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    dataset.dataset_statistics = dataset_statistics
    return dataset

def make_frame_transform(
    multimodal_rephrasings: bool,
    chunk_relative_actions: bool,
    gripper_relative_actions: bool,
    proprio_dropout_prob: float,
    tokenizer: Tokenizer | None,
    generation: bool,
):
    def frame_transform(data):
        if multimodal_rephrasings:
            data = process_rephrase_tf(data)
            data = mel_spectro_to_image(data)
            data = tactile_to_image(data)
            data = create_fuse_modal_mask(data)
            
        if chunk_relative_actions:
            # Gripper is absolute, rest is relative
            if gripper_relative_actions:
                relative_mask = 1
            else:
                relative_mask = tf.constant([True] * 6 + [False] + [True] * 6 + [False])
            initial_offset = data["observation"]["proprio"][-1] * tf.cast(
                relative_mask, tf.float32
            )
            data["action"] = data["action"] - initial_offset
            data["initial_offset"] = initial_offset

        if proprio_dropout_prob > 0:
            mask = tf.random.uniform((1,)) > proprio_dropout_prob
            data["observation"]["proprio"] = tf.where(
                mask,
                data["observation"]["proprio"],
                tf.zeros_like(data["observation"]["proprio"]),
            )
            data["observation"]["pad_mask_dict"]["proprio"] &= mask

        if tokenizer is not None:
            language_token_instructions = tokenizer.tokenize_language_instruction(data)

            if generation:
                data = data | tokenizer.prepare_tokens_for_generation(
                    data, language_token_instructions
                )
            else:
                data = data | tokenizer.prepare_tokens_for_training(
                    data, language_token_instructions
                )

        data["proprio"] = tf.squeeze(data["observation"]["proprio"], axis=0)
        data["action"] = tf.squeeze(data["action"], axis=0)

        return data

    return frame_transform


def transform_dataset(
    dataset: dlimp.DLataset,
    tokenizer: Tokenizer | None,
    generation: bool,
    *,
    multimodal_rephrasings: bool = False,
    chunk_relative_actions: bool = False,
    gripper_relative_actions: bool = True,
    require_language: bool = True,
    proprio_dropout_prob: float = 0.0,
):
    dataset_statistics = dataset.dataset_statistics

    if require_language:
        dataset = dataset.filter(has_language)
    dataset = dataset.map(
        make_frame_transform(
            multimodal_rephrasings,
            chunk_relative_actions,
            gripper_relative_actions,
            proprio_dropout_prob,
            tokenizer,
            generation,
        ),
        num_parallel_calls=None,
    )
    options = tf.data.Options()
    options.autotune.enabled = False
    dataset = dataset.with_options(options)

    dataset.dataset_statistics = dataset_statistics
    return dataset


def has_language(data):
    return tf.strings.length(data["task"]["language_instruction"]) > 0
