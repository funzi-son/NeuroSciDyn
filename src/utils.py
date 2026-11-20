"""Shared utilities: seeding, device selection, config loading, logging."""

import ast
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal fallback parser for flat key: value YAML configs when PyYAML is absent."""
    cfg: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val_str = [s.strip() for s in line.split(":", 1)]
        low = val_str.lower()
        if low == "true":
            val: Any = True
        elif low == "false":
            val = False
        elif low in ("null", "none", "~"):
            val = None
        elif val_str.startswith("[") and val_str.endswith("]"):
            inner = val_str[1:-1].strip()
            val = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
        else:
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                val = val_str.strip("'\"")
        cfg[key] = val
    return cfg


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str | None = None) -> torch.device:
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    with open(config_path, "r") as f:
        text = f.read()
    cfg = (yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)) or {}
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


def setup_logger(name: str = "uav_methods") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)
