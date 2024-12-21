from typing import Any, Dict, Sequence, Mapping, Union

from flax.typing import Collection, VariableDict
import chex

Array = chex.Array
ArrayTree = Union[chex.Array, Mapping[str, "ArrayTree"], Sequence["ArrayTree"]]
Params = Collection
Variables = VariableDict
Updates = ArrayTree
Data = ArrayTree
Info = Dict[str, Any]
