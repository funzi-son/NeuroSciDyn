"""TransKITE method adapter."""

import torch.nn as nn

from ..models.transkite import TransKITEModel
from .kinematic_seq2seq import KinematicSeq2SeqBase


class TransKITEMethod(KinematicSeq2SeqBase):
    name = "transkite"
    checkpoint_filename = "best_transkite.pth"

    def build_model(self) -> nn.Module:
        cfg = self.config
        in_dim = cfg["input_dim"] * 2 if cfg.get("use_kinematics", True) else cfg["input_dim"]
        return TransKITEModel(
            input_dim=in_dim,
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            nhead=cfg.get("nhead", 4),
            seq_len=cfg["seq_len"],
            pred_len=cfg["pred_len"],
            dropout=cfg.get("dropout", 0.1),
        )
