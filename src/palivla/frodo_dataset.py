from enum import Enum
import numpy as np
from typing import Tuple, Callable
from pathlib import Path
import einops
import tqdm
import zarr
import jax
import gcsfs
import tensorstore as ts

import torch
import torch.utils.data
import torchvision.transforms.functional as TF

import torchvision

import logging

# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot.common.datasets.video_utils import load_from_videos

from PIL import Image

def load_from_videos(
    item: dict[str, torch.Tensor],
    video_frame_keys: list[str],
    videos_dir: Path,
    tolerance_s: float,
    backend: str = "pyav",
):
    """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
    in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a Segmentation Fault.
    This probably happens because a memory reference to the video loader is created in the main process and a
    subprocess fails to access it.
    """
    # since video path already contains "videos" (e.g. videos_dir="data/videos", path="videos/episode_0.mp4")
    data_dir = videos_dir.parent

    for key in video_frame_keys:
        if isinstance(item[key], list):
            # load multiple frames at once (expected when delta_timestamps is not None)
            timestamps = [frame["timestamp"] for frame in item[key]]
            paths = [frame["path"] for frame in item[key]]
            if len(set(paths)) > 1:
                raise NotImplementedError("All video paths are expected to be the same for now.")
            video_path = data_dir / paths[0]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames
        else:
            # load one frame
            timestamps = [item[key]["timestamp"]]
            video_path = data_dir / item[key]["path"]

            frames = decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
            item[key] = frames[0]

    return item



def decode_video_frames_torchvision(
    video_path: str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesnt support accuracte seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = timestamps[0]
    last_ts = timestamps[-1]

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    # assert is_within_tol.all(), (
    #     f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
    #     "It means that the closest frame that can be loaded from the video is too far away in time."
    #     "This might be due to synchronization issues with timestamps during data collection."
    #     "To be safe, we advise to ignore this item during training."
    #     f"\nqueried timestamps: {query_ts}"
    #     f"\nloaded timestamps: {loaded_ts}"
    #     f"\nvideo: {video_path}"
    #     f"\nbackend: {backend}"
    # )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class ActionFormat(Enum):
    WAYPOINT = 1
    WAYPOINT_ANGLE = 2
    LINEAR_ANGULAR = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_str(s: str) -> "ActionFormat":
        return ActionFormat[s.upper()]


def yaw_rotmat(yaw: float | np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    if isinstance(yaw, torch.Tensor):
        return torch.tensor(
            [
                [torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw)],
                [torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw)],
                [torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)],
            ],
        )
    else:
        return np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
        )



# class InfiniteWeightedRandomSampler(torch.utils.data.Sampler[int]):
#     def __init__(self, weights):
#         self.weights = weights
#         print(weights length)

#     def __iter__(self):
#         while True:
#             # Block in chunks of 1024
#             yield from iter(torch.multinomial(self.weights, 1024, True).tolist())

#     def __len__(self):
#         return len(self.weights)

class InfiniteWeightedRandomSampler(torch.utils.data.Sampler[int]):
    def __init__(self, weights):
        """
        Sampler that samples infinitely with weights, handling cases where 
        the number of categories exceeds 2^24 by sampling in smaller chunks.
        
        Args:
            weights (torch.Tensor): The tensor of weights, should be of shape (N,).
        """
        self.weights = weights
        self.chunk_size = 2**24  # Max categories that can be handled in one call by CUDA
        print(f"Weight length: {len(weights)}")

    def __iter__(self):
        while True:
            # First, break the weights into chunks so that we don't exceed the CUDA limit of 2^24 categories
            num_categories = len(self.weights)
            start = 0
            
            while start < num_categories:
                end = min(start + self.chunk_size, num_categories)  # Determine the chunk
                chunk_weights = self.weights[start:end]
                
                # Perform multinomial sampling on the chunk of weights
                sampled_indices = torch.multinomial(chunk_weights, 1024, replacement=True)
                
                # Yield the sampled indices as a list
                yield from sampled_indices.tolist()
                
                # Move to the next chunk
                start = end

    def __len__(self):
        # Return the number of categories (length of the weights tensor)
        return len(self.weights)
    
    


def to_local_coords(
    positions: np.ndarray | torch.Tensor, curr_pos: np.ndarray | torch.Tensor, curr_yaw: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos) @ rotmat


def load_frames_zarr(
    dataset: zarr.Array,
    index: int,
    episode_data_index: dict[str, np.ndarray],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[np.ndarray]:
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = dataset["episode_index"][index].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    ep_data_ids = np.arange(ep_data_id_from, ep_data_id_to, 1)

    # load timestamps
    ep_timestamps = dataset["timestamp"][ep_data_id_from:ep_data_id_to]

    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = dataset["timestamp"][index]

    item = {}

    for key, delta_ts in delta_timestamps.items():
        # if it is a video frame
        timestamp_key = f"{key}.timestamp"
        path_key = f"{key}.path"
        is_video = timestamp_key in dataset.keys() and path_key in dataset.keys()

        # get timestamps used as query to retrieve data of previous/future frames
        if delta_ts is None:
            if key in dataset.keys():
                item[key] = np.asarray(dataset[key][index])
            elif is_video:
                item[key] = [
                    {"path": dataset[path_key][i.item()], "timestamp": dataset[timestamp_key][i.item()]}
                    for i in ep_data_ids
                ]
            else:
                raise ValueError(f"Timestamp key {timestamp_key} not found in dataset")
        else:
            query_ts = current_ts + np.array(delta_ts)

            # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
            right_idcs = np.searchsorted(ep_timestamps, query_ts, side="right")
            left_idcs = right_idcs - 1
            left_idcs = np.clip(left_idcs, 0, len(ep_timestamps) - 1)
            right_idcs = np.clip(right_idcs, 0, len(ep_timestamps) - 1)
            right_vals = ep_timestamps[right_idcs]
            left_vals = ep_timestamps[left_idcs]
            best_idcs = np.where(
                np.abs(right_vals - query_ts) < np.abs(left_vals - query_ts),
                right_idcs,
                left_idcs,
            )
            time_error = np.abs(ep_timestamps[best_idcs] - query_ts)

            is_pad = time_error > tolerance_s

            # check violated query timestamps are all outside the episode range
            assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
                f"One or several timestamps unexpectedly violate the tolerance ({time_error.min()} > {tolerance_s=}) inside episode range."
                "This might be due to synchronization issues with timestamps during data collection."
            )

            # get dataset indices corresponding to frames to be loaded
            data_ids = ep_data_ids[best_idcs]

            if is_video:
                # video mode where frame are expressed as dict of path and timestamp
                item[key] = [
                    {"path": dataset[path_key][i], "timestamp": float(dataset[timestamp_key][i])}
                    for i in data_ids
                ]
            else:
                item[key] = dataset[key][data_ids]

            item[f"{key}_is_pad"] = is_pad

    return item


# class FrodoDataset(LeRobotDataset):
class FrodoDataset:
    def __init__(
        self,
        repo_id: str,
        root: str | None,
        split: str = "train",

        action_format: ActionFormat | str = ActionFormat.WAYPOINT,

        action_horizon: int = 8,
        action_spacing: int = 1,
        goal_horizon: int = 20,
        context_size: int = 5,
        context_spacing: int = 1,

        dataset_framerate: int = 10,

        image_size: Tuple[int, int] = (112, 112),

        image_transforms: Callable | None = None,
        load_goal_image: bool = False,

        item_transform: Callable | None = None,

        action_key: str = "action",
    ):
        """
        Main ViNT dataset class
        """
        if isinstance(action_format, str):
            action_format = ActionFormat.from_str(action_format)
        self.action_format = action_format

        if action_format == ActionFormat.WAYPOINT:
            self.num_action_params = 2
        elif action_format == ActionFormat.WAYPOINT_ANGLE:
            self.num_action_params = 3
        elif action_format == ActionFormat.LINEAR_ANGULAR:
            self.num_action_params = 2

        self.dt = 1 / dataset_framerate
        self.action_spacing = action_spacing
        self.action_horizon = action_horizon
        self.goal_horizon = goal_horizon
        self.context_size = context_size
        self.context_spacing = context_spacing
        self.image_size = image_size
        self.load_goal_image = load_goal_image
        self.fps = dataset_framerate
        self.tolerance_s = 1 / self.fps - 1e-4
        self.video_backend = "pyav"
        self.videos_dir = f"{root}/frodobots_dataset/videos"

        self.action_key = action_key

        # super().__init__(
        #     repo_id=repo_id,
        #     root=root,
        #     split=split,
        #     image_transforms=image_transforms,
        #     delta_timestamps={
        #         "observation.filtered_position": [0.0],
        #         "observation.relative_position": [0.0],
        #         "observation.filtered_heading": [0.0],
        #         "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
        #         "action": [i * self.dt for i in range(action_spacing * action_horizon)],
        #     },
        # )
        fs = gcsfs.GCSFileSystem(project="rail-tpus")
        gcs_dir = "frodo-bucket-c2/frodobots_v2_export/frodobots_dataset/dataset_cache.zarr"

        folders = fs.ls(gcs_dir)
        self.dataset_cache = {}
        for folder in folders:
            folder_name = folder.split("/")[-1]
            breakpoint()
            if folder_name == ".zgroup":
                continue
            ts_spec = {
                "driver": "zarr",
                "kvstore": {
                    "driver": "gcs",
                    "bucket": "frodo-bucket-c2",
                    "path": f"frodobots_v2_export/frodobots_dataset/dataset_cache.zarr/{folder_name}",
                }
            }
            subcache = ts.open(ts_spec).result()
            self.dataset_cache[folder] = subcache
        
        ep_from = []
        ep_to = []
        for ep_id in tqdm.trange(self.dataset_cache["episode_index"].max() + 1, desc="Building episode data index..."):
            ep_from.append(np.searchsorted(self.dataset_cache["episode_index"], ep_id, side="left"))
            ep_to.append(np.searchsorted(self.dataset_cache["episode_index"], ep_id + 1, side="left"))
        
        self.episode_data_index = {
            "from": np.asarray(ep_from),
            "to": np.asarray(ep_to),
        }
        self.image_transforms = image_transforms
        self.delta_timestamps = {
            "observation.filtered_position": [0.0],
            "observation.relative_position": [0.0],
            "observation.filtered_heading": [0.0],
            "observation.images.front": [i * context_spacing * self.dt for i in range(-context_size, 1)],
            self.action_key: [i * self.dt for i in range(action_spacing * action_horizon)],
        }

        self.item_transform = item_transform
    
    def __len__(self):
        return len(self.dataset_cache["episode_index"])

    def _image_transforms(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            img (np.ndarray): image tensor
        Returns:
            np.ndarray: transformed image
        """
        if self.image_transforms is not None:
            img = self.image_transforms(img)

        # img = np.asarray(Image.fromarray(img).resize(self.image_size))
        img = TF.resize(img, self.image_size)
        img = img.numpy()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

    def viz_rollout(self, actions: np.ndarray) -> np.ndarray:
        if self.action_format == ActionFormat.WAYPOINT:
            positions = actions
        elif self.action_format == ActionFormat.WAYPOINT_ANGLE:
            positions = actions[..., :2]
        elif self.action_format == ActionFormat.LINEAR_ANGULAR:
            # Roll out actions
            positions = np.zeros_like(actions)
            heading = np.zeros_like(actions[..., 0, 0])

            for i in range(1, actions.shape[-2]):
                vel = actions[..., i - 1, 0]
                angvel = actions[..., i - 1, 1]

                direction = np.stack([np.cos(heading), np.sin(heading)], axis=-1)
                positions[..., i, :] = positions[..., i - 1, :] + vel[..., None] * direction * self.dt
                heading = heading + angvel * self.dt
        else:
            raise ValueError(f"Unknown action format {self.action_format}")

        return positions

    def __getitem__(self, idx):
        # Sample a goal timestamp
        ep_id = self.dataset_cache["episode_index"][idx].item()
        episode_length_remaining = self.episode_data_index["to"][ep_id] - idx
        goal_dist = np.random.randint(0, min(self.goal_horizon, episode_length_remaining))
        goal_dist = int(np.random.exponential(scale=self.goal_horizon))
        goal_dist = min(goal_dist, episode_length_remaining.item() - 1)

        # Add the goal to the list of delta timestamps
        delta_timestamps = self.delta_timestamps or {k: [0.0] for k in item.keys()}
        delta_timestamps = delta_timestamps | {k: None for k in ["episode_index", "frame_index", "timestamp"]}

        delta_timestamps["observation.filtered_position"].append(goal_dist * self.dt * self.action_spacing)
        delta_timestamps["observation.filtered_heading"].append(goal_dist * self.dt * self.action_spacing)
        delta_timestamps["observation.relative_position"].append(goal_dist * self.dt * self.action_spacing)
        if self.load_goal_image:
            delta_timestamps["observation.images.front"].append(goal_dist * self.dt * self.action_spacing)

        item = load_frames_zarr(
            self.dataset_cache,
            idx,
            self.episode_data_index,
            delta_timestamps,
            self.tolerance_s,
        )

        def get_obs_image_keys(keys):
            if self.load_goal_image:
                return keys[:-1]
            return keys

        image_obs = self._image_transforms(load_from_videos(
            {"observation.images.front": get_obs_image_keys(item["observation.images.front"])},
            ["observation.images.front"],
            self.videos_dir,
            self.tolerance_s,
            self.video_backend,
        )["observation.images.front"])
        image_obs = einops.rearrange(image_obs, "t c h w -> t h w c")
        image_obs_mask = ~get_obs_image_keys(item["observation.images.front_is_pad"])

        if self.load_goal_image:
            image_goal = self._image_transforms(load_from_videos(
                {"observation.images.front": item["observation.images.front"][-1]},
                ["observation.images.front"],
                self.videos_dir,
                self.tolerance_s,
                self.video_backend,
            )["observation.images.front"])
            image_goal = einops.rearrange(image_goal, "c h w -> h w c")
            image_goal_pad = item["observation.images.front_is_pad"][-1]
        else:
            image_goal = None
            image_goal_pad = None

        unnorm_position = item["observation.filtered_position"][:-1]
        current_heading = item["observation.filtered_heading"][0]

        goal_pos_relative = to_local_coords(item["observation.filtered_position"][-1, None], unnorm_position[0], current_heading)[0]

        action = einops.reduce(item[self.action_key], "(a s) d -> a d", reduction="mean", s=self.action_spacing)
        action_mask = np.array([1]) # ~item["action_is_pad"]

        # Mask out which goals we use: gps, image, none  
        a, b = np.random.rand(2)
        goal_pos_mask = a < 0.5 
        image_goal_mask = b < 0.5

        item = {
            "action": action,
            "pad_mask_dict": {
                "action": action_mask,
            },
            "observation": {
                "image_front": image_obs[:self.context_size + 1],
                "image_goal": image_goal, 
                "filtered_position": item["observation.relative_position"],
                "filtered_heading": item["observation.filtered_heading"],
                "gps_goal": goal_pos_relative,
                "pad_mask_dict": {
                    "image_front": image_obs_mask[:self.context_size + 1],
                    "filtered_position": ~item["observation.relative_position_is_pad"],
                    "filtered_heading": ~item["observation.filtered_heading_is_pad"],
                    # "gps_goal": np.ones((1,), dtype=bool),
                    "gps_goal": goal_pos_mask,
                    "image_goal": image_goal_mask,
                },
            },
            "task": {
                "language_instruction": f"Go to {goal_pos_relative}" if goal_pos_mask else "",
                "goal_position": goal_pos_relative,
                "goal_image": image_goal,
            }
        }

        if self.item_transform is not None:
            item = self.item_transform(item)

        return item

    def get_sampler(self, base_rate: float = 0.1):
        """
        Create a sampler that samples dataset elements proportionally to the sum of squared future turning actions (+ base_rate).

        A sample that drives straight will be weighted by base_rate, while a sample that is constantly turning at max speed will be weighted by 1.
        """
        import torch
        indices = torch.arange(len(self))
        to_indices = self.episode_data_index["to"] - 1
        to_indices = torch.from_numpy(to_indices[self.dataset_cache["episode_index"]])

        target_indices = indices[:, None] + torch.arange(self.action_horizon) * self.action_spacing
        target_next_indices = target_indices + 1
        target_indices.clip_(indices[:, None], to_indices[:, None])
        target_next_indices.clip_(indices[:, None], to_indices[:, None])

        headings = torch.tensor(self.dataset_cache["observation.filtered_heading"])
        heading_diff = (headings[target_indices] - headings[target_next_indices]).clip_(-0.2, 0.2).abs_().sum(dim=-1)

        future_steer = torch.clip(heading_diff, -1, 1)
        weights = base_rate + (1 - base_rate) * future_steer ** 2

        print("NUM WEIGHTS", weights.shape)

        return InfiniteWeightedRandomSampler(weights)


    
