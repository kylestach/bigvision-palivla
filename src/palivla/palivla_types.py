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
    actions: jax.Array
    tokens: jax.Array
    tokens_ar: jax.Array
    tokens_loss: jax.Array
    tokens_mask: jax.Array
    gen_start_idx: jax.Array | None = None # refers to where the prompt ends in the ground truth sequence. either CoT token or action token idx
    action_start_idx: jax.Array | None = None # refers to where the actions are in the GROUND TRUTH sequence


@dataclass
class RolloutBatch:
    sensor_data: Dict[str, jax.Array]
    sensor_masks: Dict[str, jax.Array]
    prompt: jax.Array
    prompt_mask: jax.Array
    prompt_ar: jax.Array
