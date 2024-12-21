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
