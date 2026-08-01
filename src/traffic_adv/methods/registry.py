from __future__ import annotations

from importlib import import_module

from traffic_adv.config import ExperimentConfig
from traffic_adv.methods.base import AttackMethod

_METHODS = {
    "mmad": "traffic_adv.methods.mmad.method:MMADMethod",
}


def register_method(name: str, import_path: str) -> None:
    if name in _METHODS:
        raise ValueError(f"Method already registered: {name}")
    _METHODS[name] = import_path


def create_method(config: ExperimentConfig) -> AttackMethod:
    try:
        module_name, class_name = _METHODS[config.method].split(":", 1)
    except KeyError as exc:
        raise ValueError(
            f"Unknown method {config.method!r}; available: {sorted(_METHODS)}"
        ) from exc
    method_type = getattr(import_module(module_name), class_name)
    return method_type(config)

