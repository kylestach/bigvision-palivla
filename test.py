import os
import re

DEFAULT_TRAIN_ARGS = {
    "eval_interval": 1000,
    "save_interval": 10000,
    "data_axis_size": 1,
    "fsdp_axis_size": -1,
    "paligemma_weights_path": "models/paligemma-3b-pt-224.f16.npz",
    "language_tokenizer_path": "models/paligemma_tokenizer.model",
    # "resume_from_checkpoint_dir": "gs://rail-tpus-mitsuhiko-central2/logs/test/zesty-bird-3/",
    # "resume_from_checkpoint_step": 10000,
}

if not os.path.exists(DEFAULT_TRAIN_ARGS["paligemma_weights_path"]):
    import kagglehub

    MODEL_PATH = "./pt_224_128.params.f16.npz"
    if not os.path.exists(MODEL_PATH):
        print("Downloading the checkpoint from Kaggle, this could take a few minutes....")
        # Note: kaggle archive contains the same checkpoint in multiple formats.
        # Download only the float16 model.
        MODEL_PATH = kagglehub.model_download('google/paligemma/jax/paligemma-3b-pt-224', 'paligemma-3b-pt-224.f16.npz')
        print(f"Model path: {MODEL_PATH}")

    TOKENIZER_PATH = "./paligemma_tokenizer.model"
    if not os.path.exists(TOKENIZER_PATH):
        print("Downloading the model tokenizer...")
        #!gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}
        os.system(f"gsutil cp gs://big_vision/paligemma_tokenizer.model {TOKENIZER_PATH}")
        print(f"Tokenizer path: {TOKENIZER_PATH}")



TPU_PODS = {
    "v4-vm-.*": {
        "tpc_args": {
            "project": "rail-tpus",
            "zone": "us-central2-b",
            "accelerator_type": "v4-8",
            "runtime_version": "tpu-vm-v4-base",
            "reserved": False,
        },
        "setup_script": "source /nfs/nfs2/users/mitsuhiko/miniconda/etc/profile.d/conda.sh && conda activate big_vision",
        "src_dir": "/nfs/nfs2/users/mitsuhiko/codes/bigvision-palivla",
        "train_args": {
            "batch_size": 32,
            "save_path": "gs://rail-tpus-mitsuhiko-central2/logs/test",
            "dataset_kwargs.oxe_kwargs.data_dir": "gs://rail-orca-central2/resize_256_256",
        },
    },
}

def parse_args(args_str):
    args = {}
    if args_str:
        for arg in args_str.split(','):
            key, value = arg.split('=')
            args[key] = value
    return args

# pod_name = os.environ.get("POD_NAME")
pod_name = "v4-vm-30"
config = None
for key, maybe_config in TPU_PODS.items():
    if re.fullmatch(key, pod_name) is not None:
        config = maybe_config
        break

if config is None:
    raise ValueError(f"No matching configuration found for pod name: {pod_name}")

train_args = os.environ.get("TRAIN_ARGS")
# config_file = os.environ.get("CONFIG_FILE", "bridge_config.py")
config_file = "/nfs/nfs2/users/mitsuhiko/codes/bigvision-palivla/palivla/configs/bridge_config.py"
train_args = DEFAULT_TRAIN_ARGS | config["train_args"] | parse_args(train_args)
train_args_str = " \\\n\t".join([f"--config.{k} {v}" for k, v in train_args.items()])
train_script = os.environ.get("TRAIN_SCRIPT", "palivla/train.py")

launch_script = f"""
#!/bin/bash

{config["setup_script"]}
cd {config["src_dir"]}

PYTHONPATH=. python {train_script} --config {config_file} \
    {train_args_str}

"""

if os.environ.get("VERBOSE", "0") == "1":
    import pprint
    print("*" * 100)
    print("RUNNING WITH CONFIG")
    pprint.pprint(config)
    print("LAUNCH SCRIPT:")
    print(launch_script)
    print("*" * 100)

# configure_tpc(
#     **config["tpc_args"],
#     name=pod_name,
#     tmux_session_name=f"tpc_{pod_name}",
#     launch_script=launch_script,
# )

# execute the launch script
print(launch_script)
os.system(launch_script)