"""Abstract base interface for all trajectory forecasting methods."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class BaseMethod(ABC):
    """Unified interface so train.py, evaluate.py, and export scripts stay method-agnostic."""

    name: str = ""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Instantiate the underlying PyTorch model from self.config."""

    @abstractmethod
    def train(self, dataset: str, device: torch.device) -> dict[str, Any]:
        """Train on 'uav' or 'uuv' and save best checkpoint(s). Return summary dict."""

    @abstractmethod
    def evaluate(self, dataset: str, device: torch.device, checkpoint: str | None = None) -> dict[str, float]:
        """Evaluate on 'uav' or 'uuv' test split and return aggregated metrics."""

    @abstractmethod
    def load_for_inference(
        self,
        device: torch.device,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, dict[str, Any]]:
        """Load trained weights and any normalization metadata for UAV inference."""

    @abstractmethod
    def predict_uav_trajectory(
        self,
        model: nn.Module,
        raw_coords: np.ndarray,
        meta: dict[str, Any],
        device: torch.device,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Autoregressively forecast a full UAV trajectory: returns (history, truth, pred)."""

    @abstractmethod
    def build_onnx_export_spec(
        self,
        checkpoint: str | None = None,
    ) -> tuple[nn.Module, tuple[torch.Tensor, ...], list[str], dict[str, Any]]:
        """Return (eval_module, example_inputs, input_names, meta_dict) for ONNX export."""
