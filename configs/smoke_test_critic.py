from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.components.model import get_default_config

placeholder(int)._value


def get_config(variant_config: str | None = None):
    data_dir = FieldReference("/data/rlds/", str)
    num_train_steps = FieldReference(100000, int)

    model_config = get_default_config()
    if variant_config is not None:
        variant_config_kv_pairs = variant_config.split(",")
        variant_config_dict = {
            k: v for k, v in [pair.split("=") for pair in variant_config_kv_pairs]
        }
    else:
        variant_config_dict = {}

    model_config["llm_spec"]["config"]["variant"] = variant_config_dict.get(
        "llm", "smoke_test"
    )
    model_config["img_spec"]["config"]["variant"] = variant_config_dict.get(
        "img", "S/14"
    )

    model_config["num_critic_bins"] = 256
    model_config["q_min"] = 0.0
    model_config["q_max"] = 1.0
    model_config["critic_sigma"] = 0.01
    model_config["discount"] = 0.5
    model_config["target_ema_rate"] = 0.5

    return ConfigDict(
        {
            # Shared field references
            "data_dir": data_dir,

            # W&B settings
            "wandb_project": "palivla-critic-debug",
            "wandb_mode": "disabled",
            "wandb_experiment_name": "critic-smoke-test",
            # Tokenizers
            "language_tokenizer": "google/paligemma-3b-pt-224",
            "action_tokenizer": "action_tokenizer.bin(min_action_value=-3, max_action_value=3)",
            "sequence_builder": "sequence_builder.default(prompt_pad_length=50, gen_pad_length=10)",
            # Initialization
            "load_fns": [],
            "resume_checkpoint_dir": None,
            "resume_checkpoint_step": None,
            # Overfit the dataset (for smoke tests/debugging)
            "overfit_dataset": True,
            # Training settings
            "batch_size": 16,
            "eval_batch_size": 16,
            "num_steps": num_train_steps,
            # Checkpoint settings
            "save_path": placeholder(str),
            "save_interval": 10000,
            "max_to_keep": 1,
            # Logging and visualization
            "eval_interval": 100,
            "viz_interval": 1000,
            "log_interval": 10,
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
                    "ema_rate": 0.005,
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
            "viz_traj_datasets": {
                "bridge": {
                    "name": "bridge_dataset",
                    "data_dir": data_dir,
                    "load_camera_views": ["primary"],
                    "load_depth": False,
                    "load_proprio": True,
                    "load_language": True,
                    "force_recompute_dataset_statistics": False,
                    "action_proprio_normalization_type": NormalizationType.NORMAL,
                }
            },
            "viz_num_trajectories": 4,
            "critic_train_step_kwargs": {
                "regress_to_mc_returns": True,
                "train_with_sarsa": False,
            }
        }
    )
