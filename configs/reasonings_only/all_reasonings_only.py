from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

# this can be arbitrary (but at least the length of the longest action for dataloading purposes), since we're not using actions
SINGLE_ARM_ACTION_DIM = 14
MAX_CHUNK_SIZE = 4 

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"

    config["cot_path"] = FieldReference(None, str)
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"
    
    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]

    # REASONINGS ONLY (NO ACTIONS)
    config["dataset_kwargs"]["oxe_kwargs"]["use_actions_dct"] = {
        "bridge_dataset": False,
        'aloha_spoons_in_bowls_dataset': False,
        'aloha_long_horizon_dataset': False,
        'aloha_pick_place_full_dataset': False,
        'ego4d_hamer': False,
        'libero_90': False,
    }

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "all_mix"
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = ["primary"] # intentionally just using high view, even for aloha (no proprio)

    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": SINGLE_ARM_ACTION_DIM,
    }

    config["dataset_kwargs"]["traj_read_threads"] = 6

    # learning rate
    config['optimizer']['kwargs']['llm_optimizer_kwargs'] = {
        "learning_rate": 5e-5,
        "schedule_type": "warmup_constant",

    }
    config['optimizer']['kwargs']['embed_optimizer_kwargs'] = config['optimizer']['kwargs']['llm_optimizer_kwargs']
    config['optimizer']['kwargs']['img_optimizer_kwargs'] = config['optimizer']['kwargs']['llm_optimizer_kwargs']
    
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

        config['visualization_datasets']['libero_90'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['libero_90']['name'] = "libero_90"

        config['visualization_datasets']['ego4d_hamer'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['ego4d_hamer']['name'] = "ego4d_hamer"

        config['visualization_datasets']['aloha_spoons_in_bowls_dataset'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['aloha_spoons_in_bowls_dataset']['name'] = "aloha_spoons_in_bowls_dataset"

        config['visualization_datasets']['aloha_long_horizon_dataset'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['aloha_long_horizon_dataset']['name'] = "aloha_long_horizon_dataset"

        config['visualization_datasets']['aloha_pick_place_full_dataset'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['aloha_pick_place_full_dataset']['name'] = "aloha_pick_place_full_dataset"
        
        config['visualization_datasets']['hard_bridge_eval'] = config['visualization_datasets']['bridge'].copy()
        config['visualization_datasets']['hard_bridge_eval']['name'] = "hard_bridge_eval" # we don't wanna use reasonings for this one 
        
        for k, v in config["visualization_datasets"].items():
            if k != "hard_bridge_eval":
                v["use_cot"] = True
                v["cot_data_path"] = config["cot_path"]
            v["use_actions"] = False


        config["visualizations"]["bridge_chain_of_thought"] = {
            "dataset": "bridge",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["libero_90_chain_of_thought"] = {
            "dataset": "libero_90",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["ego4d_hamer_chain_of_thought"] = {
            "dataset": "ego4d_hamer",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["aloha_spoons_in_bowls_dataset_chain_of_thought"] = {
            "dataset": "aloha_spoons_in_bowls_dataset",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["aloha_long_horizon_dataset_chain_of_thought"] = {
            "dataset": "aloha_long_horizon_dataset",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["aloha_pick_place_full_dataset_chain_of_thought"] = {
            "dataset": "aloha_pick_place_full_dataset",
            "visualization": "viz.chain_of_thought"
        }
        config["visualizations"]["hard_bridge_eval_chain_of_thought"] = {
            "dataset": "hard_bridge_eval",
            "visualization": "viz.chain_of_thought"
        }
        # config["visualizations"]["libero_90_sanity_print"] = {
        #     "dataset": "libero_90",
        #     "visualization": "viz.sanity_print"
        # }
        # config["visualizations"]["ego4d_hamer_sanity_print"] = {
        #     "dataset": "ego4d_hamer",
        #     "visualization": "viz.sanity_print"
        # }

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
