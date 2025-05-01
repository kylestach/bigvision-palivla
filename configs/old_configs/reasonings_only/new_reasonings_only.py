from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config
from octo.data.utils.data_utils import NormalizationType

BIMANUAL_ACTION_DIM = 14
MAX_CHUNK_SIZE = 50
ALOHA_PROPRIO_DIM = 14

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    
    config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"

    config["cot_path"] = FieldReference(None, str)
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"

    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "cotraining_mix" 
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = ["primary"]
    config["dataset_kwargs"]["oxe_kwargs"]["use_actions_dct"] = {
        "bridge_dataset": False,
        'aloha_spoons_in_bowls_dataset': False,
        'aloha_long_horizon_dataset': False,
        'aloha_pick_place_full_dataset': False,
        'aloha_bread_dataset': False,
        'ego4d_hamer': False,
        'libero_90': False,
        'fractal20220817_data': False, 
        'droid_dataset': False,
        'aria_dataset': False,
    }

    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": BIMANUAL_ACTION_DIM,
    }

    config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"] = {"primary": [224, 224]}

    config["dataset_kwargs"]["traj_read_threads"] = 10

    # learning rate
    config['optimizer']['kwargs']['llm_optimizer_kwargs'] = {
        "learning_rate": 5e-5,
        "schedule_type": "warmup_constant",

    }
    config['optimizer']['kwargs']['embed_optimizer_kwargs'] = config['optimizer']['kwargs']['llm_optimizer_kwargs']
    config['optimizer']['kwargs']['img_optimizer_kwargs'] = config['optimizer']['kwargs']['llm_optimizer_kwargs']

    config['visualization_datasets']['aloha_spoons_in_bowls_dataset'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['aloha_spoons_in_bowls_dataset']['name'] = "aloha_spoons_in_bowls_dataset"

    config['visualization_datasets']['hard_bridge_eval'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['hard_bridge_eval']['name'] = "hard_bridge_eval"

    config['visualization_datasets']['fractal20220817_data'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['fractal20220817_data']['name'] = "fractal20220817_data"

    config['visualization_datasets']['droid_dataset'] = config['visualization_datasets']['bridge'].copy()
    config['visualization_datasets']['droid_dataset']['name'] = "droid_dataset"

    # fix settings
    for ds, ds_config in config['visualization_datasets'].items():
        ds_config['load_proprio'] = True
        ds_config['load_camera_views'] = ["primary"]
        ds_config['frame_transform_kwargs']['resize_size'] = {
            "primary": [224, 224]
        }
        if ds != "hard_bridge_eval":
            ds_config["use_cot"] = True
            ds_config["cot_data_path"] = config["cot_path"]
        ds_config["use_actions"] = False

    
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

        config["visualizations"]["aloha_spoons_in_bowls_dataset_chain_of_thought"] = {
            "dataset": "aloha_spoons_in_bowls_dataset",
            "visualization": "viz.chain_of_thought"
        }

        config["visualizations"]["hard_bridge_eval_chain_of_thought"] = {
            "dataset": "hard_bridge_eval",
            "visualization": "viz.chain_of_thought"
        }

        config["visualizations"]["fractal20220817_data_chain_of_thought"] = {
            "dataset": "fractal20220817_data",
            "visualization": "viz.chain_of_thought"
        }

        config["visualizations"]["droid_dataset_chain_of_thought"] = {
            "dataset": "droid_dataset",
            "visualization": "viz.chain_of_thought"
        }

        config["visualizations"]["fractal20220817_data_sanity_print"] = {
            "dataset": "fractal20220817_data",
            "visualization": "viz.sanity_print"
        }

        config["visualizations"]["droid_dataset_sanity_print"] = {
            "dataset": "droid_dataset",
            "visualization": "viz.sanity_print"
        }

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
