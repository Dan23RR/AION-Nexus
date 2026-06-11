"""AION-NEXUS architecture — exact replica of the trained model.

Architecture: Multi-Scale CNN (3 parallel branches) + Channel Attention +
Bidirectional GRU + 3-layer MLP classifier.

This file is FROZEN — any modification breaks checkpoint compatibility.
Total parameters: 1,061,724.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from aion_nexus.config import NUM_CHANNELS, NUM_CLASSES, PENULTIMATE_FEATURE_DIM

_logger = logging.getLogger(__name__)


class MultiScaleTemporalCNN(nn.Module):
    """3 parallel CNN branches at different receptive-field scales."""

    def __init__(self, in_channels: int = NUM_CHANNELS, out_channels: int = 64) -> None:
        super().__init__()
        # Short-term branch (high-freq, kernels 3+7)
        self.short_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )
        # Medium-term (mid-freq, kernels 15+31)
        self.medium_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, out_channels, kernel_size=31, padding=15),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )
        # Long-term (low-freq, kernels 63+127)
        self.long_conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=63, padding=31),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, out_channels, kernel_size=127, padding=63),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_conv(x)
        medium = self.medium_conv(x)
        long = self.long_conv(x)
        return torch.cat([short, medium, long], dim=1)  # [B, 192, 640]


class AttentionFusion(nn.Module):
    """SE-style channel attention with concurrent avg+max pooling."""

    def __init__(self, in_channels: int = 192, reduction: int = 8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        avg_feat = self.avg_pool(x).view(b, c)
        max_feat = self.max_pool(x).view(b, c)
        attention = self.fc(avg_feat) + self.fc(max_feat)
        return x * attention.unsqueeze(-1)


class TemporalEncoder(nn.Module):
    """Bidirectional GRU with avg+max pooling. Output dim = 4 × hidden_dim."""

    def __init__(self, input_dim: int = 192, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # [B, T, C]
        out, _ = self.gru(x)                # [B, T, 2*H]
        out = out.transpose(1, 2)           # [B, 2*H, T]
        avg_feat = self.avg_pool(out).squeeze(-1)
        max_feat = self.max_pool(out).squeeze(-1)
        return torch.cat([avg_feat, max_feat], dim=1)  # [B, 4*H = 512]


class ClassificationHead(nn.Module):
    """3-layer MLP with BatchNorm and graded dropout."""

    def __init__(self, input_dim: int = PENULTIMATE_FEATURE_DIM,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.67),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class AIONNexus(nn.Module):
    """Full AION-NEXUS production model.

    Forward returns ``{'logits': [B, num_classes], 'features': [B, 512]}``.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.multi_scale_cnn = MultiScaleTemporalCNN(NUM_CHANNELS, 64)
        self.attention = AttentionFusion(192, 8)
        self.temporal_encoder = TemporalEncoder(192, 128, num_layers=2, dropout=0.2)
        self.classifier = ClassificationHead(PENULTIMATE_FEATURE_DIM, num_classes, 0.3)
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
            elif isinstance(m, nn.BatchNorm1d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, rpm=None, geometry=None) -> dict:
        multi_scale = self.multi_scale_cnn(x)
        attended = self.attention(multi_scale)
        feat = self.temporal_encoder(attended)
        logits = self.classifier(feat)
        return {"logits": logits, "features": feat}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_aion_nexus(num_classes: int = NUM_CLASSES) -> AIONNexus:
    """Factory: instantiate AION-NEXUS with verified parameter count.

    Raises ValueError if param count does not match the frozen 1,061,724.
    """
    model = AIONNexus(num_classes=num_classes)
    n = model.get_num_params()
    expected = 1_061_724
    if num_classes == NUM_CLASSES and n != expected:
        raise ValueError(
            f"Architecture drift: got {n:,} params, expected {expected:,}. "
            "This indicates the model code has diverged from the trained checkpoint."
        )
    _logger.info("AION-NEXUS instantiated: %d params (%.1f MB)", n, n * 4 / 1024 / 1024)
    return model
