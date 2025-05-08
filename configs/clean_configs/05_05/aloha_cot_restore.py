from ml_collections import ConfigDict, FieldReference
from palivla.base_config import get_config as get_base_config

MAX_ACTION_DIM = 14
MAX_CHUNK_SIZE = 50
MAX_PROPRIO_DIM = 14 # testing proprio padding 

USE_COT = True

IMAGE_KEYS = ["left_wrist", "right_wrist", "primary"] # in the order you want them to appear in the sequence
PROPRIO_KEYS = ["bimanual"] # in the order you want them to appear in the sequence

DATA_MIX = "new_aloha_mix"
USE_ACTIONS_DCT = {
    "aloha_bread_dataset": True,
    "aloha_pick_place_full_dataset": True,
    "aloha_spoons_in_bowls_dataset": True,
    "aloha_long_horizon_dataset": True,
}

VISUALIZATIONS = {
    "aloha_bread_dataset": ['sanity_print', 'chain_of_thought'],
    "aloha_pick_place_full_dataset": ['sanity_print', 'chain_of_thought'],
    "aloha_spoons_in_bowls_dataset": ['sanity_print', 'chain_of_thought'],
    "aloha_long_horizon_dataset": ['sanity_print', 'chain_of_thought'],
}

RESTORE = {
    'path': 'alohaonly_reasonings_06052025_085733',
    'step': 25000
}

####################################################################################
## DON'T CHANGE ANYTHING BELOW 
####################################################################################
def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["cot_path"] = FieldReference(None, str)
    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = USE_COT

    # restore from checkpoint
    if RESTORE['path'] is not None:
        config["resume_checkpoint_dir"] = f"gs://multi-robot-bucket3/runs/vla/{RESTORE['path']}"
        config["resume_checkpoint_step"] = RESTORE['step']

    # dataset configs
    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = DATA_MIX 
    config["dataset_kwargs"]["traj_read_threads"] = len(USE_ACTIONS_DCT)
    config["dataset_kwargs"]["oxe_kwargs"]["use_actions_dct"] = USE_ACTIONS_DCT

    # modality mappings 
    config["model_config"]["num_proprio_tokens"] = MAX_PROPRIO_DIM
    config["model_config"]["target_key_order"] = (
        tuple(f"proprio_{k}" for k in PROPRIO_KEYS) +
        tuple(f"image_{k}" for k in IMAGE_KEYS)
    )

    # sequence builders & action tokenizers 
    if USE_COT:
        config["sequence_builder"] = f"sequence_builder.cot(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"
        config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]
    else:
        config["sequence_builder"] = f"sequence_builder.default(prompt_pad_length=50, gen_pad_length=150, action_chunk_pad_length={MAX_CHUNK_SIZE})"
    config['action_tokenizer'] = "action_tokenizer.fast(min_action_value=-3, max_action_value=3)"
    
    # loading appropriate image keys
    config["dataset_kwargs"]["oxe_kwargs"]["load_camera_views"] = IMAGE_KEYS
    config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"] = {k: [224, 224] for k in IMAGE_KEYS}

    # traj transform kwargs
    config["dataset_kwargs"]["traj_transform_kwargs"] = {
        "window_size": 1,
        "action_horizon": MAX_CHUNK_SIZE,
        "task_augment_strategy": "delete_task_conditioning",
        "task_augment_kwargs": {
            "keep_image_prob": 0,
        },
        "max_action_dim": MAX_ACTION_DIM,
        "max_proprio_dim": MAX_PROPRIO_DIM,
    }

    ####################################################################################
    # VISUALIZATIONS
    ####################################################################################

    # use actions for datasets that you've specified
    base_viz_config = config['visualization_datasets']['bridge_dataset'].copy()
    config['visualization_datasets'] = {}
    
    for ds_name, viz_types in VISUALIZATIONS.items():
        ds_viz_config = base_viz_config.copy()
        ds_viz_config['name'] = ds_name
        
        # visualize actions depending on whether you're training on them for each dataset
        ds_viz_config['use_actions'] =  config['dataset_kwargs']['oxe_kwargs']['use_actions_dct'].get(ds_name, False)

        # if a max proprio dim is specified in the traj transform kwargs, add it to the viz config
        if 'max_proprio_dim' in config['dataset_kwargs']['traj_transform_kwargs']:
            ds_viz_config['traj_transform_kwargs'] = {
                'max_proprio_dim': config['dataset_kwargs']['traj_transform_kwargs']['max_proprio_dim']
            }

        # load all the relevant cam views, and set their resizes
        ds_viz_config['load_camera_views'] = config['dataset_kwargs']['oxe_kwargs']['load_camera_views']
        ds_viz_config['frame_transform_kwargs']['resize_size'] = config['dataset_kwargs']["frame_transform_kwargs"]["resize_size"]

        # load in cot if you're using it (for non hard bridge eval)
        if ds_name != 'hard_bridge_eval':
            ds_viz_config['use_cot'] = config['dataset_kwargs']['oxe_kwargs']['use_cot']
            if ds_viz_config['use_cot']:
                ds_viz_config['cot_data_path'] = config['cot_path']

        config['visualization_datasets'][ds_name] = ds_viz_config
        # add in the visualizations
        for viz_type in viz_types:
            config['visualizations'][f'{ds_name}_{viz_type}'] = {
                'dataset': ds_name,
                'visualization': f"viz.{viz_type}"
            }

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
