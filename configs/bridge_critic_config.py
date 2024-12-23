from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.components.model import get_default_config

placeholder(int)._value


def get_config():
    num_train_steps = FieldReference(100000, int)

    model_config = get_default_config()

    model_config["num_critic_bins"] = 256
    model_config["q_min"] = 0.0
    model_config["q_max"] = 1.0
    model_config["critic_sigma"] = 0.01
    model_config["discount"] = 0.5
    model_config["target_ema_rate"] = 0.5

    return ConfigDict(
        {
            # W&B settings
            "wandb_project": "palivla-bridge",
            "wandb_mode": "online",
            # Tokenizers
            "language_tokenizer": "google/paligemma-3b-pt-224",
            "action_tokenizer": "action_tokenizer.bin(min_action_value=-3, max_action_value=3)",
            "sequence_builder": "sequence_builder.default(prompt_pad_length=50, gen_pad_length=10)",
            # Initialization
            "load_fns": [
                (
                    "load.paligemma_weights",
                    {
                        "hf_repo": "google/paligemma-3b-pt-224-jax",
                        "path": "paligemma-3b-pt-224.npz",
                    },
                )
            ],
            "resume_checkpoint_dir": None,
            "resume_checkpoint_step": None,
            # Overfit the dataset (for smoke tests/debugging)
            "overfit_dataset": False,
            # Training settings
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": placeholder(str),
            "save_interval": 10000,
            "max_to_keep": 1,
            # Logging and visualization
            "eval_interval": 100,
            "log_interval": 1,
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
                    "base_learning_rate": 1e-4,
                },
            },
            # Dataset settings
            "dataset_kwargs": {
                "oxe_kwargs": {
                    "data_mix": "bridge",
                    "data_dir": "/data/rlds/",
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
        }
    )