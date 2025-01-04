import importlib
import json
from typing import Any, Callable, Dict, Generic, Mapping, TypeVar

import jax
import optax
from flax import linen as nn
from flax import struct
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze

from palivla.utils import freeze_structure

T = TypeVar("T")


@struct.dataclass
class CtorSpec(Generic[T]):
    ctor: Callable[..., T]
    config: FrozenDict[str, Any]

    @classmethod
    def is_ctor_spec_dict(cls, data: Any) -> bool:
        return isinstance(data, Mapping) and "__ctor" in data and "config" in data

    @classmethod
    def create(
        cls, ctor: Callable[..., T] | str, config: Dict[str, Any]
    ) -> "CtorSpec[T]":
        config = jax.tree.map(
            lambda x: CtorSpec.from_dict(x) if CtorSpec.is_ctor_spec_dict(x) else x,
            config,
            is_leaf=CtorSpec.is_ctor_spec_dict,
        )
        config = freeze_structure(config)
        return cls(ctor=ctor, config=freeze(config))

    @classmethod
    def from_name(cls, ctor_full_name: str, config: Dict[str, Any]):
        ctor_module = importlib.import_module(".".join(ctor_full_name.split(".")[:-1]))
        ctor_name = ctor_full_name.split(".")[-1]
        ctor = getattr(ctor_module, ctor_name)
        return cls.create(ctor, config)

    def instantiate(self, **kwargs) -> T:
        return self.ctor(**self.config, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        config = jax.tree.map(
            lambda x: x.to_dict() if isinstance(x, CtorSpec) else x,
            unfreeze(self.config),
            is_leaf=lambda x: isinstance(x, CtorSpec),
        )
        return {
            "__ctor": self.ctor.__module__ + "." + self.ctor.__name__,
            "config": config,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], overrides: Dict[str, Any] | None = None
    ) -> "CtorSpec":
        if isinstance(data, CtorSpec):
            return data

        if overrides:
            data["config"].update(overrides)

        return cls.from_name(ctor_full_name=data["__ctor"], config=data["config"])

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(
        cls, json_str: str, overrides: Dict[str, Any] | None = None
    ) -> "CtorSpec":
        data = json.loads(json_str)
        return cls.from_dict(data, overrides=overrides)


OptimizerSpec = CtorSpec[optax.GradientTransformation]
ModuleSpec = CtorSpec[nn.Module]
