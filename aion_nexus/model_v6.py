"""AION-NEXUS v6 — production architecture for v2.0.

Combines proven v1.0 components (MultiScaleTemporalCNN, AttentionFusion)
with new v6 noise-robust aggregation and recursive refinement:

  Input [B, 2, 2560]
       │ MultiScaleTemporalCNN  (3 parallel branches; ~50K params)
       ↓ [B, 192, 640]
       │ AttentionFusion        (channel-wise SE-style; ~5K params)
       ↓ [B, 192, 640]
       │ TemporalSelfAttention  (multi-head + learned-query pool; NEW v6)
       ↓ [B, 192]
       │ TinyRecursiveReasoner  (TRM-inspired refinement; NEW v6)
       ↓ [B, num_classes]

Verified parameter count (must match training-time aion_nexus_v6):
  - 4-class FEMTO bearing fault: 716,577 params (frozen contract).
  - Analytical derivation matches `nexus_ultra_v6_results/training.log` exactly (delta = 0).
  - 2.73 MB on disk (FP32).

Same preprocessing as v1.0 (z-score per channel + HP-Butterworth 1 Hz).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from aion_nexus.config import NUM_CHANNELS, NUM_CLASSES
from aion_nexus.model import AttentionFusion, MultiScaleTemporalCNN
from aion_nexus.recursive_reasoner import TinyRecursiveReasoner
from aion_nexus.temporal_attention import TemporalSelfAttention

_logger = logging.getLogger(__name__)

V6_PARAM_COUNT_4CLASS = 716_577  # Verified against nexus_ultra_v6_results/training.log line 20 (delta = 0)


class AIONNexusV6(nn.Module):
    """v6 architecture: CNN → ChannelAttn → TemporalSelfAttn → TinyRecursiveReasoner."""

    def __init__(
        self,
        in_channels: int = NUM_CHANNELS,
        num_classes: int = NUM_CLASSES,
        cnn_out_channels: int = 64,
        latent_dim: int = 128,
        num_recursive_layers: int = 2,
        halt_threshold: float = 0.9,
        temporal_attention_heads: int = 4,
        temporal_attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.halt_threshold = halt_threshold

        self.multi_scale_cnn = MultiScaleTemporalCNN(in_channels, cnn_out_channels)
        # Note: original v6 source uses attribute name `channel_attention`;
        # we keep that to match state_dict keys.
        self.channel_attention = AttentionFusion(in_channels=cnn_out_channels * 3, reduction=8)
        self.temporal_attention = TemporalSelfAttention(
            channels=cnn_out_channels * 3,
            num_heads=temporal_attention_heads,
            dropout=temporal_attention_dropout,
            use_learned_pooling=True,
        )
        self.recursive_reasoner = TinyRecursiveReasoner(
            feature_dim=cnn_out_channels * 3,
            latent_dim=latent_dim,
            num_classes=num_classes,
            num_layers=num_recursive_layers,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d | nn.LayerNorm | nn.GroupNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.multi_scale_cnn(x)
        f = self.channel_attention(f)
        return self.temporal_attention(f)

    def forward(
        self,
        x: torch.Tensor,
        N_supervision: int = 1,    # noqa: N803 — public API kwarg, mirrors TRM paper notation
        n_reasoning: int = 6,
        T_recursions: int = 3,     # noqa: N803 — public API kwarg, mirrors TRM paper notation
        rpm=None,
        geometry=None,
    ) -> dict:
        features = self.extract_features(x)

        if N_supervision == 1:
            out = self.recursive_reasoner(
                features, y_init=None, z_init=None,
                n=n_reasoning, T=T_recursions, train_mode=self.training,
            )
            return {"logits": out["logits"], "features": out["latent"]}

        # Deep supervision (training only)
        b = x.size(0)
        y = torch.full((b, self.num_classes), 1.0 / self.num_classes, device=x.device)
        z = torch.zeros(b, self.latent_dim, device=x.device)
        history = []
        out = None
        for _ in range(N_supervision):
            out = self.recursive_reasoner(
                features, y_init=y.clone(), z_init=z.clone(),
                n=n_reasoning, T=T_recursions, train_mode=True,
            )
            history.append({"logits": out["logits"].clone(), "halt_prob": out["halt_prob"].clone()})
            y = out["logits"].detach()
            z = out["latent"].detach()
            if not self.training and out["halt_prob"].mean() > self.halt_threshold:
                break
        return {
            "logits": out["logits"],
            "features": out["latent"],
            "halt_prob": out["halt_prob"],
            "supervision_history": history,
        }

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_aion_nexus_v6(num_classes: int = NUM_CLASSES,
                         halt_threshold: float = 0.9,
                         temporal_attention_heads: int = 4,
                         strict_param_check: bool = True) -> AIONNexusV6:
    """Factory: instantiate AION-NEXUS v6 with verified parameter count."""
    model = AIONNexusV6(
        in_channels=NUM_CHANNELS,
        num_classes=num_classes,
        cnn_out_channels=64,
        latent_dim=128,
        num_recursive_layers=2,
        halt_threshold=halt_threshold,
        temporal_attention_heads=temporal_attention_heads,
        temporal_attention_dropout=0.1,
    )
    n = model.get_num_params()
    if strict_param_check and num_classes == NUM_CLASSES and n != V6_PARAM_COUNT_4CLASS:
        raise ValueError(
            f"v6 architecture drift: got {n:,} params, expected {V6_PARAM_COUNT_4CLASS:,}. "
            "This indicates the v6 model code has diverged from the trained checkpoint. "
            "Pass strict_param_check=False to bypass during architecture experimentation."
        )
    _logger.info("AION-NEXUS v6 instantiated: %d params (%.2f MB)", n, n * 4 / 1024 / 1024)
    return model
