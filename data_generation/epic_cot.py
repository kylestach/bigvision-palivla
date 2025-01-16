import tensorflow as tf
import numpy as np
import argparse
import json
import os

from octo.data.oxe import make_oxe_dataset_kwargs
from octo.data.dataset import make_single_dataset
TRAJS_TO_PROCESS = 5200
WRITE_INTERVAL=25
parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=0)
parser.add_argument("--dataset", type=str, default="ego4d_hamer")
args = parser.parse_args()

FINAL_JSON = f"/home/oiermees/generated_ecot/{args.dataset}/jsons/batch{args.id}.json"


tf.config.set_visible_devices(
    [], device_type="gpu"
)
tf.config.set_visible_devices(
    [], device_type="GPU"
)
dataset_paths = {
    "epic_kitchens": "/raid/datasets/openx/",
    "h2_o_dataset": "/raid/users/homanga/h2o/",
    "fpha_dataset": "/raid/users/homanga/fpha/",
    "ss_v2_dataset": "/raid/users/homanga/ssv2_full/",
    "ego4d_hamer": "/raid/datasets/openx/"
}

# load chunk of dataset in based on ID
print("Loading dataset ", args.dataset)
start_idx = TRAJS_TO_PROCESS*args.id
dataset_kwargs = make_oxe_dataset_kwargs(args.dataset,dataset_paths[args.dataset])
dataset = dataset = make_single_dataset(dataset_kwargs, frame_transform_kwargs=dict(resize_size={"primary": (256, 256)},),train=True)
dataset = dataset.skip(start_idx).take(TRAJS_TO_PROCESS)
iterator = dataset.iterator()

# if final json path already exists, load it in
results_dict = {}
if os.path.exists(FINAL_JSON):
    print("loading existing JSON!", flush=True)
    with open(FINAL_JSON, "r") as file:
        results_dict = json.load(file)

assert results_dict is not None

for i, episode in enumerate(iterator):
    # write dictionary if it's time
    if (i + 1) % WRITE_INTERVAL == 0:
        with open(FINAL_JSON, "w") as f:
            json.dump(results_dict, f)

    traj_id = int(episode['traj_idx'][0])
    print(f"Processing traj {traj_id} **********", flush=True)
    traj_id = int(episode['traj_idx'][0])
    print(f"Processing traj {traj_id} **********", flush=True)

    # skip traj if we've already processed it
    if str(traj_id) in results_dict:
        print("skipping traj ", traj_id, flush=True)
        continue

    images = episode['observation']['image_primary'].squeeze()
    language_label = episode['task']['language_instruction'][0].decode()

    # create results dict for this trajectory
    results_dict[traj_id] = {}
    results_dict[traj_id]['language_label'] = language_label

with open(FINAL_JSON, "w") as f:
    json.dump(results_dict, f)

print(f"FINISHED BATCH {args.id} FOR DATASET {args.dataset}!", flush=True)