import importlib
import json
from typing import Any, Callable, Dict, Generic, TypeVar

import optax
from flax import linen as nn
from flax import struct
from flax.core import frozen_dict

from palivla.utils import freeze_structure

T = TypeVar("T")


@struct.dataclass
class CtorSpec(Generic[T]):
    ctor: Callable[..., T]
    config: Dict[str, Any]

    @classmethod
    def from_name(cls, ctor_full_name: str, config: Dict[str, Any]):
        ctor_module = importlib.import_module(".".join(ctor_full_name.split(".")[:-1]))
        ctor_name = ctor_full_name.split(".")[-1]
        ctor = getattr(ctor_module, ctor_name)
        return cls(ctor=ctor, config=config)

    def instantiate(self, *args, **kwargs) -> T:
        return self.ctor(**frozen_dict.freeze(freeze_structure(self.config)), **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {"ctor": self.ctor.__module__ + "." + self.ctor.__name__, "config": self.config}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CtorSpec":
        return cls.from_name(ctor_full_name=data["ctor"], config=data["config"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "CtorSpec":
        data = json.loads(json_str)
        return cls.from_dict(data)


OptimizerSpec = CtorSpec[optax.GradientTransformation]
ModuleSpec = CtorSpec[nn.Module]
