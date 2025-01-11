from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.components.model import get_default_config


def get_config(variant_config: str):
    num_train_steps = FieldReference(100000, int)
    data_dir = FieldReference("/data/rlds/", str)

    model_config = get_default_config()

    config = {
        "data_dir": data_dir,

        # W&B settings
        "wandb_project": placeholder(str),
        "wandb_mode": "online",
        # Tokenizers
        "language_tokenizer": "google/paligemma2-3b-pt-224",
        "action_tokenizer": "action_tokenizer.bin(min_action_value=-3, max_action_value=3)",
        "sequence_builder": "sequence_builder.default(prompt_pad_length=50, gen_pad_length=10)",
        # Initialization
        "load_fns": [
            (
                "load.paligemma_weights",
                {
                    "hf_repo": "google/paligemma-3b-mix-224-jax",
                    "path": "paligemma-3b-mix-224.npz",
                },
            )
        ],
        "resume_checkpoint_dir": placeholder(str),
        "resume_checkpoint_step": placeholder(int),
        # Overfit the dataset (for smoke tests/debugging)
        "overfit_dataset": False,
        # Training settings
        "batch_size": placeholder(int),
        "eval_batch_size": placeholder(int),
        "num_steps": num_train_steps,
        # Checkpoint settings
        "save_path": placeholder(str),
        "save_interval": 10000,
        "max_to_keep": 1,
        # Logging and visualization
        "eval_interval": 100,
        "log_interval": 10,
        "viz_interval": 1000,
        # Multi-device settings
        "data_axis_size": 1,
        "fsdp_axis_size": -1,
        # Model
        "model_config": model_config,
        # Optimizer settings
        "optimizer": {
            "name": "optimizer.default_optimizer",
            "kwargs": {
                "optimizer": "sgd",
                "num_train_steps": num_train_steps,
                "base_learning_rate": 1e-3,
                "ema_rate": 0.999,
            },
        },
        # Dataset settings
        "dataset_kwargs": {
            "oxe_kwargs": {
                "data_mix": "bridge",
                "data_dir": data_dir,
                "load_camera_views": ["primary"],
                "load_depth": False,
                "load_proprio": True,
                "load_language": True,
                "force_recompute_dataset_statistics": False,
                "action_proprio_normalization_type": NormalizationType.NORMAL,
            },
            "traj_transform_kwargs": {
                "window_size": 1,
                "action_horizon": 1,
            },
            "frame_transform_kwargs": {
                "image_augment_kwargs": {},
                "resize_size": {"primary": [224, 224]},
            },
            "balance_weights": True,
            "shuffle_buffer_size": 50000,
            "traj_transform_threads": 16,
            "traj_read_threads": 16,
        },
        "viz_trajectories_per_dataset": 4,
        "visualization_datasets": {
            "bridge": {
                "name": "bridge_dataset",
                "data_dir": data_dir,
                "load_camera_views": ["primary"],
                "load_depth": False,
                "load_proprio": True,
                "load_language": True,
                "force_recompute_dataset_statistics": False,
                "action_proprio_normalization_type": NormalizationType.NORMAL,
            },
        },
        "visualizations": {
            "bridge_sanity_print": {
                "dataset": "bridge",
                "visualization": "viz.sanity_print",
            }
        },
    }

    if variant_config == "smoke_test":
        config["overfit_dataset"] = True
        config["batch_size"] = 16
        config["eval_batch_size"] = 16
        config["wandb_mode"] = "online"
        config["wandb_project"] = "palivla-debug"
        config["load_fns"] = []
        model_config["llm_spec"]["config"]["variant"] = "smoke_test"
        model_config["img_spec"]["config"]["variant"] = "S/14"

    return config

