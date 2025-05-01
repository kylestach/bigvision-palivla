from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

SINGLE_ARM_ACTION_DIM = 7
MAX_CHUNK_SIZE = 4

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"

    config["cot_path"] = FieldReference(None, str)
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"

    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "human_bridge" 

    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": SINGLE_ARM_ACTION_DIM,
    }

    config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"] = {"primary": [224, 224]}

    config["dataset_kwargs"]["traj_read_threads"] = 3

    config['visualization_datasets']['ego4d_hamer'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['ego4d_hamer']['name'] = "ego4d_hamer"

    config['visualization_datasets']['epic_kitchens'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['epic_kitchens']['name'] = "epic_kitchens"

    for v in config["visualization_datasets"].values():
        v["use_cot"] = True
        v["cot_data_path"] = config["cot_path"]
    
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
        
        config["visualizations"]["bridge_chain_of_thought"] = {
            "dataset": "bridge",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["ego4d_hamer_chain_of_thought"] = {
            "dataset": "ego4d_hamer",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["epic_kitchens_chain_of_thought"] = {
            "dataset": "epic_kitchens",
            "visualization": "viz.chain_of_thought"
        }
        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
