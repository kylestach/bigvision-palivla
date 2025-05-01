from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

PAD_TO_ACTION_DIM = 8
MAX_CHUNK_SIZE = 4 # since this is aloha chunk size ..


def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["model_config"]["num_proprio_tokens"] = 14

    config["model_config"]["modality_mappings"] =  {
        "image_primary": "img",
        # "proprio_franka": "proprio",
    }
    config["model_config"]["target_key_order"] = (
        # "proprio_franka",
        "image_primary",
    )

    # Resume from reasoning checkpoint
    config["resume_checkpoint_dir"] = "gs://multi-robot-bucket3/runs/vla/more_data_reasonings_only_29042025_200001"
    config["resume_checkpoint_step"] = 55000

    # CoT sequence builder & action tokenizer
    config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"

    # Turn on CoT 
    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["cot_path"] = FieldReference(None, str)
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]
    
    # Turn on/off actions per dataset
    config["dataset_kwargs"]["oxe_kwargs"]["use_actions_dct"] = {
        'droid_dataset': True,
    }

    # Specify data mix
    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "franka"

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
        "max_proprio_dim": 14,
    }

    # Specify traj read threads
    config["dataset_kwargs"]["traj_read_threads"] = 1
    config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"] = {"primary": [224, 224]}
    
    # learning rate
    config['optimizer']['kwargs']['llm_optimizer_kwargs'] = {
        "learning_rate": 1e-5,
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

        # Setup visualization datasets
        viz_datasets = ['droid_dataset']  
        for ds in viz_datasets:
            config['visualization_datasets'][ds] = config['visualization_datasets']['bridge'].copy()
            config['visualization_datasets'][ds]['name'] = ds
            config['visualization_datasets'][ds]['use_actions'] = True
            config['visualization_datasets'][ds]['use_cot'] = True
            config['visualization_datasets'][ds]['cot_data_path'] = config['cot_path']
            config['visualization_datasets'][ds]['traj_transform_kwargs'] = {
                "max_action_dim": PAD_TO_ACTION_DIM,
                "max_proprio_dim": 14,
            }

        # CoT visualizations
        config["visualizations"]["droid_dataset_chain_of_thought"] = {
            "dataset": "droid_dataset",
            "visualization": "viz.chain_of_thought"
        }

        # Sanity prints
        config["visualizations"]["droid_dataset_sanity_print"] = {
            "dataset": "droid_dataset",
            "visualization": "viz.sanity_print"
        }

        config['visualization_datasets'].pop('bridge')
        config['visualizations'].pop('bridge_sanity_print')

        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
