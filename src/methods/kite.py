"""KITE method adapter."""

import torch.nn as nn

from ..models.kite import KITEModel
from .kinematic_seq2seq import KinematicSeq2SeqBase


class KITEMethod(KinematicSeq2SeqBase):
    name = "kite"
    checkpoint_filename = "best_kite.pth"

    def build_model(self) -> nn.Module:
        cfg = self.config
        in_dim = cfg["input_dim"] * 2 if cfg.get("use_kinematics", True) else cfg["input_dim"]
        return KITEModel(
            input_dim=in_dim,
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            pred_len=cfg["pred_len"],
            dropout=cfg.get("dropout", 0.5),
        )
