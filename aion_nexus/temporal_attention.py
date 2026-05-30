"""TemporalSelfAttention — noise-robust aggregation for AION-NEXUS v2.0.

Replaces the v1.0 BiGRU + dual-pool with a multi-head self-attention block
plus a learned-query pooling. Lighter, faster, and noise-robust by design:
the attention mechanism learns to down-weight noisy timesteps.

Frozen architecture (must match training-time `aion_nexus_v6.TemporalSelfAttention`):
- channels: 192 (output of multi-scale CNN: 64 × 3 branches)
- num_heads: 4
- dropout: 0.1
- use_learned_pooling: True (query vector attends to all timesteps for pooling)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention + learned-query pooling.

    Input:  [B, 192, 640]   (channels-first temporal features)
    Output: [B, 192]         (aggregated per-sample features)

    Three internal stages:
      1. Multi-head self-attention over timesteps (residual + LayerNorm)
      2. Feed-forward (residual + LayerNorm)
      3. Learned-query pooling: a single learnable query vector attends to
         all timesteps to produce one aggregated representation.
    """

    def __init__(
        self,
        channels: int = 192,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_learned_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_learned_pooling = use_learned_pooling

        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )
        if use_learned_pooling:
            self.pool_query = nn.Parameter(torch.randn(1, 1, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: [B, C, T]; Returns: [B, C]."""
        x = x.transpose(1, 2)  # [B, T, C]
        b = x.size(0)

        attn_out, attn_weights = self.mha(x, x, x)
        x = self.norm1(attn_out + x)

        ffn_out = self.ffn(x)
        x = self.norm2(ffn_out + x)

        if self.use_learned_pooling:
            pq = self.pool_query.expand(b, -1, -1)
            pooled, _ = self.mha(pq, x, x)
            return pooled.squeeze(1)
        # Attention-weighted pooling fallback
        weights = attn_weights.mean(dim=1).mean(dim=1)
        weights = torch.softmax(weights, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)
