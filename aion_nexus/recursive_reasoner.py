"""TinyRecursiveReasoner — TRM-inspired progressive refinement head.

Inspired by Jolicoeur-Martineau (2025), "Less is More: Recursive Reasoning
with Tiny Networks". Two tiny networks plus a halting head:
- reasoning_net: refines latent z given (features, y, z)
- answer_net: produces y' from (z, y)
- halt_head: scalar halting probability from y

Forward unrolls T recursions × n reasoning steps, with optional gradient
detachment for the first T-1 recursions during training (deep-supervision
emulation of a deeper network without the parameter cost).

Frozen for v2.0:
- feature_dim = 192 (multi-scale CNN × 3 branches)
- latent_dim = 128
- num_layers = 2 (TRM "tiny")
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TinyRecursiveReasoner(nn.Module):
    """Tiny recursive head with optional deep-supervision unrolling."""

    def __init__(
        self,
        feature_dim: int = 192,
        latent_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        layers: list[nn.Module] = []
        in_dim = feature_dim + num_classes + latent_dim
        for i in range(num_layers):
            in_size = in_dim if i == 0 else latent_dim
            layers.extend([
                nn.Linear(in_size, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.reasoning_net = nn.Sequential(*layers)

        self.answer_net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, num_classes),
        )

        self.halt_head = nn.Sequential(nn.Linear(num_classes, 1), nn.Sigmoid())

    def latent_recursion(
        self,
        features: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        n: int = 6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(n):
            combined = torch.cat([features, y, z], dim=1)
            z = self.reasoning_net(combined)
        y_new = self.answer_net(torch.cat([z, y], dim=1))
        return y_new, z

    def forward(
        self,
        features: torch.Tensor,
        y_init: torch.Tensor | None = None,
        z_init: torch.Tensor | None = None,
        n: int = 6,
        T: int = 3,  # noqa: N803 — public API kwarg, mirrors TRM paper notation
        train_mode: bool = True,
    ) -> dict:
        b = features.size(0)
        device = features.device
        y = y_init if y_init is not None else torch.full(
            (b, self.num_classes), 1.0 / self.num_classes, device=device
        )
        z = z_init if z_init is not None else torch.zeros(b, self.latent_dim, device=device)

        if train_mode and T > 1:
            with torch.no_grad():
                for _ in range(T - 1):
                    y, z = self.latent_recursion(features, y, z, n)

        y, z = self.latent_recursion(features, y, z, n)
        halt_prob = self.halt_head(y)
        return {"logits": y, "latent": z, "halt_prob": halt_prob}
