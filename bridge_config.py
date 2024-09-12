from octo.data.utils.data_utils import NormalizationType
from ml_collections.config_dict import placeholder, ConfigDict

placeholder(int)._value

def get_config():
    return ConfigDict(
        {
            "model_load_fn": "big_vision.models.proj.paligemma.paligemma.load",
            "tokenizer_path": "models/paligemma_tokenizer.model",
            "model_path": "models/paligemma",
            "save_path": placeholder(str),
            "batch_size": 192,
            "eval_batch_size": 128,
            "num_steps": 100000,
            "eval_interval": 100,
            "save_interval": 5000,
            "log_interval": 1,
            "profile": False,
            "data_axis_size": 1,
            "fsdp_axis_size": -1,
            "resume_from_checkpoint_dir": placeholder(str),
            "resume_from_checkpoint_step": placeholder(int),
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
                    # "task_augment_strategy": "rephrase_instruction",
                    # "task_augment_kwargs": {
                    #     "pickle_file_path": "/data/rlds/paraphrases_oxe.pkl",
                    #     "rephrase_prob": 0.5,
                    # },
                },
                "frame_transform_kwargs": {
                    "image_augment_kwargs": {},
                    "resize_size": {"primary": [224, 224]},
                },
                "shuffle_buffer_size": 50000,
            },
            # "optimizer": "sgd",
            # "llm_optimizer_kwargs": {
            #     "init_learning_rate": 0,
            #     "learning_rate": 1e-3,
            #     "warmup_steps": 50,
            #     "weight_decay": 0,
            #     "grad_norm_clip": 10.0,
            # },
            # "embed_optimizer_kwargs": {
            #     "init_learning_rate": 0,
            #     "learning_rate": 1e-3,
            #     "warmup_steps": 10,
            #     "weight_decay": 0.0,
            #     "grad_norm_clip": 10.0,
            # },
            # "img_optimizer_kwargs": {
            #     "init_learning_rate": 0,
            #     "learning_rate": 1e-3,
            #     "warmup_steps": 50,
            #     "weight_decay": 0,
            #     "grad_norm_clip": 10.0,
            # },
            "optimizer": "adamw",
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
        }
    )
