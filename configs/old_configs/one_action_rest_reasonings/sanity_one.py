from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

PAD_TO_ACTION_DIM = 14
MAX_CHUNK_SIZE = 4 # since this is aloha chunk size ..

# Specify target dataset
TARGET_DATASET = "bridge_dataset"

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    # Resume from reasoning checkpoint
    config["resume_checkpoint_dir"] = "gs://multi-robot-bucket3/runs/vla/all_reasonings_only_21042025_085309"
    config["resume_checkpoint_step"] = 40000

    # CoT sequence builder & action tokenizer
    config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"

    # Turn on CoT 
    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["cot_path"] = FieldReference(None, str)
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]
    
    # Turn on/off actions per dataset
    datasets = [
        "bridge_dataset", 
    ]
    config["dataset_kwargs"]["oxe_kwargs"]["use_actions_dct"] = {
        ds: (True if ds==TARGET_DATASET else False)
        for ds in datasets
    }

    # Specify data mix
    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "bridge"

    # Specify chunk size
    config["dataset_kwargs"]["oxe_kwargs"]["override_chunk_size"] = MAX_CHUNK_SIZE

    # Load camera views
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = ["primary"] # intentionally just using high view, even for aloha (no proprio)

    # Traj transform kwargs
    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": PAD_TO_ACTION_DIM,
    }

    # Specify traj read threads
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

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
