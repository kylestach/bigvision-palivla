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

    # REASONINGS ONLY (NO ACTIONS)
    config["dataset_kwargs"]["oxe_kwargs"]["override_and_use_reasonings_only"] = True

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "bridge"
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = ["primary"]

    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": SINGLE_ARM_ACTION_DIM,
    }

    config["dataset_kwargs"]["traj_read_threads"] = 1
    
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

        for v in config["visualization_datasets"].values():
            v["use_cot"] = True
            v["cot_data_path"] = config["cot_path"]
            v["override_and_use_reasonings_only"] = True # REASONINGS ONLY

        config['visualization_datasets']['hard_bridge_eval'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['hard_bridge_eval']['name'] = "hard_bridge_eval" # we don't wanna use reasonings for this one 
        config['visualization_datasets']['hard_bridge_eval']['override_and_use_reasonings_only'] = True # gt should literally be nothing for this

        config["visualizations"]["hard_bridge_eval_chain_of_thought"] = {
            "dataset": "hard_bridge_eval",
            "visualization": "viz.chain_of_thought"
        }
        
        config["visualizations"]["bridge_chain_of_thought"] = {
            "dataset": "bridge",
            "visualization": "viz.chain_of_thought"
        }

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
