from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config
from octo.data.utils.data_utils import NormalizationType

BIMANUAL_ACTION_DIM = 14
MAX_CHUNK_SIZE = 50

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    
    config["model_config"]["num_proprio_tokens"] = 14

    config["model_config"]["modality_mappings"] =  {
        "image_primary": "img",
        "image_left_wrist": "img",
        "image_right_wrist": "img",
        "proprio_bimanual": "proprio",
    }
    config["model_config"]["target_key_order"] = (
        "proprio_bimanual",
        "image_left_wrist", 
        "image_right_wrist", 
        "image_primary",
    )
    
    config["sequence_builder"] = f"sequence_builder.default(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"

    config["cot_path"] = FieldReference(None, str)
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"

    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = False

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "aloha_pp_mix" 
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = ["primary", "left_wrist", "right_wrist"]

    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": BIMANUAL_ACTION_DIM,
    }

    config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"] = {"primary": [224, 224], "left_wrist": [224, 224], "right_wrist": [224, 224]}

    config["dataset_kwargs"]["traj_read_threads"] = 1

    config['visualization_datasets']['aloha_pick_place_full_dataset'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['aloha_pick_place_full_dataset']['name'] = "aloha_pick_place_full_dataset"

    config["visualization_datasets"] = {
        "aloha_pick_place_full_dataset": {
            "name": "aloha_pick_place_full_dataset",
            "data_dir": config['data_dir'],
            "load_camera_views": ["primary", "left_wrist", "right_wrist"],
            "load_depth": False,
            "load_proprio": True,
            "load_language": True,
            "force_recompute_dataset_statistics": False,
            "action_proprio_normalization_type": NormalizationType.NORMAL,
            "frame_transform_kwargs": {
                "image_augment_kwargs": {},
                "resize_size": {"primary": [224, 224], "left_wrist": [224, 224], "right_wrist": [224, 224]},
            },
        },
    }

    
    
    if variant_config == "smoke_test":
        config["visualizations"] = {
            "overfit_sanity_print": {
                "dataset": "overfit",
                "visualization": "viz.sanity_print",
            },
            "overfit_chain_of_thought": {
                "dataset": "overfit",
                "visualization": "viz.chain_of_thought",
            }
        }
    else:
        # Use mix checkpoints instead of pt
        config["load_fns"] = [
            (
                "load.paligemma_weights",
                {
                    "hf_repo": "google/paligemma-3b-mix-224-jax",
                    "path": "paligemma-3b-mix-224.npz",
                },
            )
        ]

        config["visualizations"] = {
            "aloha_pick_place_full_dataset_sanity_print": {
                "dataset": "aloha_pick_place_full_dataset",
                "visualization": "viz.sanity_print"
            }
        }

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
