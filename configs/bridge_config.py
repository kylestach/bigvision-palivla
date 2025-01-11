from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    return ConfigDict(config)
