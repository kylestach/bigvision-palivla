from typing import Optional, Sequence
import tensorflow as tf
import numpy as np

from octo.data.utils.data_utils import NormalizationType
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights, make_oxe_dataset_kwargs
import dlimp


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
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        **oxe_kwargs
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
