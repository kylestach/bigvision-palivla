from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

from palivla.components.model import get_default_config

placeholder(int)._value


def get_config():
    num_train_steps = FieldReference(100000, int)
    data_dir = FieldReference("/data/rlds/", str)
    ema_rate = FieldReference(0.005, float)
    discount = FieldReference(0.98, float)

    model_config = get_default_config()

    model_config["num_critic_bins"] = 128
    model_config["q_min"] = 0.0
    model_config["q_max"] = 1.0
    # set to default value suggested by the paper
    # https://arxiv.org/pdf/2403.03950.pdf
    # model_config["critic_sigma"] = 0.01
    model_config["critic_sigma"] = (
        0.75
        * (model_config["q_max"] - model_config["q_min"])
        / model_config["num_critic_bins"]
    )
    model_config["discount"] = discount
    model_config["target_ema_rate"] = 0.005
    model_config["target_ema_rate"] = ema_rate

    return ConfigDict(
        {
            # Shared field references
            "data_dir": data_dir,

            # W&B settings
            "wandb_project": "palivla-bridge",
            "wandb_mode": "online",
            "wandb_experiment_name": "bridge-critic",
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
            "resume_checkpoint_dir": placeholder(str),
            "resume_checkpoint_step": placeholder(int),
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
            "viz_interval": 1000,
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
                    "optimizer": "adamw",
                    "num_train_steps": num_train_steps,
                    "base_learning_rate": 5e-5,
                    "ema_rate": ema_rate,
                    "llm_optimizer_kwargs": {
                        "init_learning_rate": 0,
                        "learning_rate": 5e-5,
                        "warmup_steps": 500,
                        "weight_decay": 5e-6,
                        "grad_norm_clip": 10.0,
                    },
                    "embed_optimizer_kwargs": {
                        "init_learning_rate": 0,
                        "learning_rate": 5e-5,
                        "warmup_steps": 100,
                        "weight_decay": 0.0,
                        "grad_norm_clip": 10.0,
                    },
                    "img_optimizer_kwargs": {
                        "init_learning_rate": 0,
                        "learning_rate": 5e-5,
                        "warmup_steps": 500,
                        "weight_decay": 5e-6,
                        "grad_norm_clip": 10.0,
                    },
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
                "mc_discount": discount,
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
                    "mc_discount": discount,
                },
            },
            # Critic training kwargs
            "viz_num_trajectories": 4,
            "critic_train_step_kwargs": {
                "regress_to_mc_returns": False,
                "train_with_sarsa": False,
                "zero_out_actions": False,
            },
        }
    )
