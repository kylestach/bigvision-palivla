import tempfile
from contextlib import contextmanager

import jax
import numpy as np
from jax.experimental import multihost_utils


def freeze_structure(structure):
    return jax.tree_util.tree_map(
        lambda x: tuple(freeze_structure(y) for y in x) if isinstance(x, list) else x,
        structure,
        is_leaf=lambda x: isinstance(x, list),
    )


def key_string(path, separator="/") -> str:
    def _component_to_string(component) -> str:
        if isinstance(component, jax.tree_util.SequenceKey):
            return str(component.idx)
        elif isinstance(component, jax.tree_util.DictKey):
            return str(component.key)
        elif isinstance(component, jax.tree_util.GetAttrKey):
            return str(component.name)
        elif isinstance(component, jax.tree_util.FlattenedIndexKey):
            return str(component.key)
        else:
            return str(component)

    return separator.join(_component_to_string(component) for component in path)


def host_broadcast_str(x: str | None) -> str:
    """
    Broadcast_one_to_all, but with a string.

    Works by padding the string to the length of the longest string and then
    broadcasting the result, then stripping the padding.

    Note: this will remove the padding from the end of the string.
    """
    if x is None:
        x = ""

    max_len = multihost_utils.broadcast_one_to_all(len(x))
    padded = x.ljust(max_len)

    encoded = np.array([ord(c) for c in padded], dtype=np.uint8)[:max_len]
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    decoded = "".join([chr(u) for u in encoded])

    return decoded.rstrip()


def gcs_recursive_copy(src: str, dst: str):
    from tensorflow import io as io

    io.gfile.makedirs(dst)
    for item in io.gfile.listdir(src):
        src_path = io.gfile.join(src, item)
        dst_path = io.gfile.join(dst, item)
        if io.gfile.isdir(src_path):
            gcs_recursive_copy(src_path, dst_path)
        else:
            io.gfile.copy(src_path, dst_path, overwrite=True)


def flatten_wandb_dict(nested_dict: dict, prefix: str = "") -> dict:
    """
    Flatten a nested dictionary for logging to Weights & Biases.
    """
    flat_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_wandb_dict(value, f"{prefix}{key}/"))
        else:
            flat_dict[f"{prefix}{key}"] = value
    return flat_dict


@contextmanager
def write_staging_directory(target_dir: str):
    """Creates a temporary staging directory and copies its contents to target_dir on exit.
    
    Args:
        target_dir: Directory to copy staged files to (can be local or GCS path)
    
    Yields:
        Path to temporary staging directory
    """
    from tensorflow import io as io

    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

        # Create target dir if it doesn't exist
        io.gfile.makedirs(target_dir)

        gcs_recursive_copy(temp_dir, target_dir)

@contextmanager
def read_staging_directory(target_dir: str):
    """Stages a directory from GCS to a temporary directory. The temporary directory is deleted on exit.

    Args:
        target_dir: Directory to stage from (can be local or GCS path)
    
    Yields:
        Path to temporary staging directory
    """
    from tensorflow import io as io

    with tempfile.TemporaryDirectory() as temp_dir:
        gcs_recursive_copy(target_dir, temp_dir)

        yield temp_dir