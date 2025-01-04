from typing import Optional, Sequence

import dlimp
from octo.data.dataset import make_interleaved_dataset, make_single_dataset
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.utils.data_utils import NormalizationType


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
    **kwargs,
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
        **kwargs,
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
    **kwargs,
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
            **kwargs,
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


def make_trajectory_dataset(
    *,
    name: str,
    data_dir: str,
    load_camera_views: Sequence[str],
    load_depth: bool,
    load_proprio: bool,
    load_language: bool,
    action_proprio_normalization_type: NormalizationType,
    force_recompute_dataset_statistics: bool,
    train: bool,
    **kwargs,
):
    import tensorflow as tf

    def squeeze_fn(traj):
        traj["action"] = tf.squeeze(traj["action"], axis=(1, 2))
        traj["action_pad_mask"] = tf.squeeze(traj["action_pad_mask"], axis=(1, 2))
        traj["observation"] = tf.nest.map_structure(
            lambda x: tf.squeeze(x, axis=1), traj["observation"]
        )
        return traj

    dataset_kwargs = make_oxe_dataset_kwargs(
        name,
        data_dir,
        load_camera_views,
        load_depth,
        load_proprio,
        load_language,
        force_recompute_dataset_statistics,
        action_proprio_normalization_type,
        **kwargs,
    )
    dataset_kwargs["num_parallel_reads"] = None
    dataset_kwargs["num_parallel_calls"] = None

    return make_single_dataset(
        dataset_kwargs,
        train=train,
        frame_transform_kwargs={},
        traj_transform_kwargs={},
    ).map(squeeze_fn)
