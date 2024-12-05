from typing import Any, Dict, Sequence, Mapping, Union

import jax
from flax.typing import Collection, VariableDict
from flax.struct import dataclass
import chex

Array = chex.Array
ArrayTree = Union[chex.Array, Mapping[str, "ArrayTree"], Sequence["ArrayTree"]]
Params = Collection
Variables = VariableDict
Updates = ArrayTree
Data = ArrayTree
Info = Dict[str, Any]


@dataclass
class TrainingBatch:
    sensors: Dict[str, jax.Array]
    sensors_mask: jax.Array
    sensors_next: Dict[str, jax.Array]
    sensors_next_mask: jax.Array
    # actions_mask: jax.Array
    actions: jax.Array
    tokens: jax.Array
    tokens_ar: jax.Array
    tokens_loss: jax.Array
    tokens_mask: jax.Array
    rewards: jax.Array
    td_mask: jax.Array
    mc_returns: jax.Array
    # next_actions: jax.Array
    next_tokens: jax.Array
    # next_mask_ar: jax.Array
    next_mask_input: jax.Array
    gen_start: jax.Array | None = None
    gen_tokens: jax.Array | None = None
    gen_mask_input: jax.Array | None = None



@dataclass
class RolloutBatch:
    sensor_data: Dict[str, jax.Array]
    sensor_masks: Dict[str, jax.Array]
    prompt: jax.Array
    prompt_mask: jax.Array
    prompt_ar: jax.Array
