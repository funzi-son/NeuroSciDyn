import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Injects temporal positional information into the sequence using sine/cosine functions.
    This allows the Transformer to recognize the sequential order of historical time steps.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransKITEModel(nn.Module):
    """
    TransKITE: Hybrid Transformer-GRU Architecture for Trajectory Prediction.
    - Encoder: Transformer Encoder to extract long-term spatio-temporal global contexts.
    - Bridge: Compresses historical context into an initial momentum state (h_0).
    - Decoder: Zero-Input GRU that autonomously unrolls future states to mitigate exposure bias.
    - Dual-Head Output: Independent linear projections for position and velocity.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, nhead=4, seq_len=50, pred_len=20, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        
        # --- Transformer Encoder (Global Historical Context) ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- Latent State Bridge (Information Bottleneck) ---
        self.bridge = nn.Linear(hidden_dim, hidden_dim * num_layers)
        
        # --- Zero-Input GRU Decoder (Autonomous Unrolling) ---
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # --- Dual-Head Neural Projections ---
        self.fc_pos = nn.Linear(hidden_dim, 3)
        self.fc_vel = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        B = x.size(0)
        
        # 1. Encoding Phase
        out = self.input_proj(x)
        out = self.pos_encoder(out)
        enc_out = self.transformer_encoder(out)
        
        # 2. Context Compression & Bridging
        mean_enc = enc_out.mean(dim=1)
        hidden_mapped = self.bridge(mean_enc)
        
        # Reshape to match GRU's expected h_0 format: [num_layers, B, hidden_dim]
        h_0 = hidden_mapped.view(B, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        # 3. Decoding Phase (Zero-Input Rollout)
        # Force the model to rely entirely on h_0 to prevent noise accumulation (drift)
        decoder_input = torch.zeros(B, self.pred_len, x.size(2), device=x.device)
        dec_out, _ = self.decoder(decoder_input, h_0)
        
        # 4. Dual-Head Decoding
        out_pos = self.fc_pos(dec_out)
        out_vel = self.fc_vel(dec_out)
        
        # Return concatenated tensor: [Position (3) + Velocity (3)] -> [B, pred_len, 6]
        return torch.cat([out_pos, out_vel], dim=2)