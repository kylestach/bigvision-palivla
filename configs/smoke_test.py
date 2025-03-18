from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "smoke_test"):
    config = get_base_config(variant_config)
    config["visualizations"] = {
        "overfit_sanity_print": {
            "dataset": "overfit",
            "visualization": "viz.sanity_print",
        }
    }
    return ConfigDict(config)
