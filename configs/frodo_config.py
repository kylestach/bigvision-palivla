from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["batch_size"] = 20 # 256
    config["eval_batch_size"] = 20


    config["wandb_mode"] = "online"
    config["wandb_project"] = "palivla-debug-frodo"

    config["dataset"] = "frodobots"
    config["dataset_root"] = "/mnt/ephemeral2/frodobots_v2_export"

    config["save_path"] = "/mnt/ephemeral2/models/palivla"
    config["run_name"] = "palivla_relabeled_frodo"
    config["save_interval"] = 1000
    config["eval_interval"] = 10

    config["model_config"]["modality_mappings"] = {"image_front": "img",
                                                    "image_goal": "img"}
    config["model_config"]["target_key_order"] = *("image_front", "image_goal",),

    config["optimizer"]["kwargs"]["optimizer"] = "adamw"
    config["optimizer"]["kwargs"]["base_learning_rate"] = 1e-4
        

    return ConfigDict(config)
