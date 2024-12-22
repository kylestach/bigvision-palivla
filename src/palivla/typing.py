from typing import Any, Dict, Mapping, Sequence, Union

import chex
from flax.typing import Collection, VariableDict

Array = chex.Array
ArrayTree = Union[chex.Array, Mapping[str, "ArrayTree"], Sequence["ArrayTree"]]
Params = Collection
Variables = VariableDict
Updates = ArrayTree
Data = ArrayTree
Info = Dict[str, Any]
