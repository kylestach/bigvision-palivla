from ml_collections import ConfigDict, FieldReference
from ml_collections.config_dict import placeholder
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config).to_dict()
    config["sequence_builder"] = "sequence_builder.cot(prompt_pad_length=50, gen_pad_length=110)"

    config["cot_path"] = FieldReference(None, str)

    config["dataset_kwargs"]["oxe_kwargs"]["use_cot"] = True
    config["dataset_kwargs"]["oxe_kwargs"]["cot_data_path"] = config["cot_path"]
    for v in config["visualization_datasets"].values():
        v["use_cot"] = True
        v["cot_data_path"] = config["cot_path"]

    if variant_config == "smoke_test":
        config["visualizations"] = {
            "overfit_sanity_print": {
                "dataset": "overfit",
                "visualization": "viz.sanity_print",
            }
        }

    return ConfigDict(config)
