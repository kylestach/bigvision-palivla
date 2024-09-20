from flax.core.frozen_dict import FrozenDict
import jax


def freeze_structure(structure):
    return jax.tree_util.tree_map(
        lambda x: tuple(freeze_structure(y) for y in x) if isinstance(x, list) else x,
        structure,
        is_leaf=lambda x: isinstance(x, list),
    )
