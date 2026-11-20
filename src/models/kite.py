"""KITE: Kinematics-Informed Trajectory Estimation (Seq2Seq GRU)."""

import torch
import torch.nn as nn


class KITEModel(nn.Module):
    """Seq2Seq GRU with zero-input decoder and dual position/velocity heads."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        pred_len: int = 20,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        self.gru1 = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        out_dim = input_dim // 2 if input_dim == 6 else 3
        self.fc_pos = nn.Linear(hidden_dim, out_dim)
        self.fc_vel = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), self.pred_len, self.hidden_dim, device=x.device, dtype=x.dtype)
        out, _ = self.gru2(dec_input, h_n)
        out_pos = self.fc_pos(out)
        out_vel = self.fc_vel(out)
        return torch.cat([out_pos, out_vel], dim=2)
