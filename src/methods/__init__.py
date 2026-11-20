"""Method registry mapping CLI names to adapter classes."""

from .base import BaseMethod
from .kcs2o import KCS2OMethod
from .kite import KITEMethod
from .transkite import TransKITEMethod

METHOD_REGISTRY: dict[str, type[BaseMethod]] = {
    "kite": KITEMethod,
    "transkite": TransKITEMethod,
    "kcs2o": KCS2OMethod,
}


def get_method(name: str, config: dict) -> BaseMethod:
    key = name.lower()
    if key not in METHOD_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[key](config)


__all__ = ["BaseMethod", "KITEMethod", "TransKITEMethod", "KCS2OMethod", "METHOD_REGISTRY", "get_method"]
