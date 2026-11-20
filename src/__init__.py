"""Unified UAV & UUV trajectory forecasting package (KITE, TransKITE, KC-S2O)."""

from .methods import METHOD_REGISTRY, get_method

__all__ = ["METHOD_REGISTRY", "get_method"]
