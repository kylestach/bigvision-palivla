from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    config["sequence_builder"] = "sequence_builder.cot(prompt_pad_length=50, gen_pad_length=500)"

    config["cot_path"] = FieldReference(None, str)

    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]

    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = "bridge"
    config["dataset_kwargs"]["traj_read_threads"] = 1

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
    
    config["optimizer"]["kwargs"]["embed_optimizer_kwargs"] = {
        "warmup_steps": 100,
    }
    config["optimizer"]["kwargs"]["llm_optimizer_kwargs"] = {
        "warmup_steps": 1000,
    }
    config["optimizer"]["kwargs"]["img_optimizer_kwargs"] = {
        "warmup_steps": 1000,
    }

    if variant_config == "smoke_test":
        config["visualizations"] = {
            # "overfit_sanity_print": {
            #     "dataset": "overfit",
            #     "visualization": "viz.sanity_print",
            # }
            "overfit_chain_of_thought": {
                "dataset": "overfit",
                "visualization": "viz.chain_of_thought",
            }
        }
    else:
        config["visualizations"]["bridge_chain_of_thought"] = {
            "dataset": "bridge",
            "visualization": "viz.chain_of_thought"
        }
        config["wandb_project"] = "palivla-cot"

    return ConfigDict(config)
