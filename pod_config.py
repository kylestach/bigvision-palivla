import os
import re

def get_tpu_config(zone, tpu_type, num_tpus):
    GCP_PROJECT_NAME = "rail-tpus"
    USERNAME = "kstachowicz"
    NFS_DIRS = {
        "europe-west4-b": f"/nfs/nfs3/users/{USERNAME}",
        "us-central2-b": f"/nfs/nfs2/users/{USERNAME}",
    }
    CHECKPOINT_DIRS = {
        "europe-west4-b": "gs://kyle-checkpoints-eu4",
        "us-central2-b": "gs://kyle-checkpoints-c2",
    }
    DATASET_DIRS = {
        "europe-west4-b": "gs://rail-datasets-europe-west4/oxe/resize_256_256",
        "us-central2-b": "gs://rail-orca-central2/resize_256_256",
    }

    SOURCE_DIR_NAME = "big_vision_rl"

    nfs = NFS_DIRS[zone]
    checkpoints_dir = CHECKPOINT_DIRS[zone]
    dataset_dir = DATASET_DIRS[zone]

    runtime_versions = {"v4": "tpu-ubuntu2204-base", "v5": "v2-alpha-tpuv5-lite"}
    accelerator_types = {"v4": f"v4-{num_tpus}", "v5": f"v5litepod-{num_tpus}"}

    return {
        "tpc_args": {
            "project": "rail-tpus",
            "zone": zone,
            "accelerator_type": accelerator_types[tpu_type],
            "runtime_version": runtime_versions[tpu_type],
            "reserved": True,
        },
        "setup_script": "source $HOME/.bashrc",
        "src_dir": f"{nfs}/{SOURCE_DIR_NAME}",
        "train_args": {
            "batch_size": num_tpus * 8,
            "save_path": f"{checkpoints_dir}/paligemma-checkpoints",
            "dataset_kwargs.oxe_kwargs.data_dir": dataset_dir,
        },
    }

DEFAULT_TRAIN_ARGS = {
    "eval_interval": 100,
    "save_interval": 1000,
    "data_axis_size": 1,
    "fsdp_axis_size": -1,
}


TPU_POD_CONFIGS = {
    "eu-v5-64": get_tpu_config("europe-west4-b", "v5", 64),
    "eu-v5-128": get_tpu_config("europe-west4-b", "v5", 128),
    "eu-v5-256": get_tpu_config("europe-west4-b", "v5", 256),
    "us-v4-8": get_tpu_config("us-central2-b", "v4", 8),
    "us-v4-16": get_tpu_config("us-central2-b", "v4", 16),
    "us-v4-32": get_tpu_config("us-central2-b", "v4", 32),
    "us-v4-64": get_tpu_config("us-central2-b", "v4", 64),
    "us-v4-128": get_tpu_config("us-central2-b", "v4", 128),
}

TPU_POD_TYPES = {
    "kyle-pod-64": "eu-v5-64",
    "kyle-pod-128": "eu-v5-128",
    "kyle-pod-256": "eu-v5-256",
    "homer-pod-64": "eu-v5-64",
    "homer-pod-128": "eu-v5-128",
    "dibya-pod-64-1": "eu-v5-64",
    "dibya-pod-64-2": "eu-v5-64",
    "v4-vm-*": "us-v4-8",
    "v4-pod-16": "us-v4-16",
}

def parse_args(args_str):
    args = {}
    if args_str:
        for arg in args_str.split(','):
            key, value = arg.split('=')
            args[key] = value
    return args

pod_name = os.environ.get("POD_NAME")
for config_re, maybe_pod_type in TPU_POD_TYPES.items():
    if re.match(config_re, pod_name):
        pod_type = maybe_pod_type
        break

config = TPU_POD_CONFIGS[pod_type]

train_args = os.environ.get("TRAIN_ARGS")
config_file = os.environ.get("CONFIG_FILE", "configs/bridge_critic_config.py")
train_args = DEFAULT_TRAIN_ARGS | config["train_args"] | parse_args(train_args)
train_args_str = " \\\n\t".join([f"--config.{k} {v}" for k, v in train_args.items()])
train_script = os.environ.get("TRAIN_SCRIPT", "scripts/train_critic.py")

launch_script = f"""
#!/bin/bash

{config["setup_script"]}
cd {config["src_dir"]}

source .venv/bin/activate
python {train_script} --config {config_file} \
    {train_args_str} \
    --platform tpu

bash
read -p "Press any key to continue..."
"""

if os.environ.get("VERBOSE", "0") == "1":
    import pprint
    print("*" * 100)
    print("RUNNING WITH CONFIG")
    pprint.pprint(config)
    print("LAUNCH SCRIPT:")
    print(launch_script)
    print("*" * 100)

configure_tpc(
    **config["tpc_args"],
    name=pod_name,
    tmux_session_name=f"tpc_{pod_name}",
    launch_script=launch_script,
)
