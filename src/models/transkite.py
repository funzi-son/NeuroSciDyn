"""TransKITE: Hybrid Transformer-GRU Architecture for Trajectory Forecasting."""

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence order."""

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransKITEModel(nn.Module):
    """Transformer Encoder + Latent Bridge + Zero-Input GRU Decoder with dual heads."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
        seq_len: int = 50,
        pred_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.bridge = nn.Linear(hidden_dim, hidden_dim * num_layers)
        self.decoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_vel = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        enc_out = self.transformer_encoder(self.pos_encoder(self.input_proj(x)))
        hidden_mapped = self.bridge(enc_out.mean(dim=1))
        h_0 = hidden_mapped.view(b, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()

        decoder_input = torch.zeros(b, self.pred_len, x.size(2), device=x.device, dtype=x.dtype)
        dec_out, _ = self.decoder(decoder_input, h_0)
        return torch.cat([self.fc_pos(dec_out), self.fc_vel(dec_out)], dim=2)
