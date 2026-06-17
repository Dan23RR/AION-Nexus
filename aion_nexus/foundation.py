"""Ride a foundation encoder, don't out-pretrain one — a model-agnostic adapter.

Why this module exists (the binding constraint)
-----------------------------------------------
The architecture-leap research found that the wall to cross-machine bearing
diagnosis is DATA DIVERSITY (the number of distinct physical bearings seen in
pretraining), NOT model capacity. A solo founder cannot out-pretrain a foundation
encoder such as UniFault (pretrained on >9 billion points across 10 datasets, MIT-
licensed) on a ~9300-window corpus — and AION's own contrastive substrate v3 is
transductively contaminated (its "0.783 LOBO" is leaked, see RETRACTIONS.md).

So the move is to RIDE the foundation encoder and OWN the verification + adaptation
layer: wrap ANY frozen encoder that maps a vibration window ``[B, C, N]`` to an
embedding ``[B, D]``, put a small few-shot head on top, and run the result through
AION's existing certified inference + conformal verification + physics second
opinion. The same adapter wraps UniFault, MOMENT, Mantis, a customer's encoder, or
AION's own — embodying the thesis: *verify / adapt ANY model, don't compete with it.*

What this is and is NOT (workspace 6.31)
----------------------------------------
- It is a thin, model-agnostic ADAPTER, not a foundation model. It adds no
  pretraining diversity of its own.
- Riding a foundation encoder only fixes the diversity deficit IF that encoder did
  NOT see your held-out bearings in pretraining — otherwise a "LOBO" number is
  transductive, exactly the flaw it is meant to fix. A clean INDUCTIVE LOBO
  benchmark requires the encoder's pretraining corpus to exclude the test bearings;
  this module cannot verify that for a black-box encoder, so the honest number is
  the caller's to establish (see :func:`aion_nexus.evaluation.evaluate_leave_one_group_out`).
- The few-shot head is a linear probe on a FROZEN encoder: the ceiling is the
  encoder's representation quality on bearings, not magic. We say so.

The adapter conforms to the same forward contract as v1/v3/v6
(``{"logits": [B, K], "features": [B, D]}``), so it drops into
:class:`~aion_nexus.inference.InferenceEngine`, :class:`~aion_nexus.few_shot.FewShotAdapter`
(architecture ``"ext"``), the certified serving path, and the physics second opinion.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — standard PyTorch convention

from aion_nexus.config import NUM_CHANNELS, NUM_CLASSES

_logger = logging.getLogger(__name__)

# Encoder contract: a torch Module OR a plain callable mapping [B, C, N] -> [B, D].
EncoderLike = Callable[[torch.Tensor], torch.Tensor]


class ExternalEncoderAdapter(nn.Module):
    """Wrap a FROZEN external encoder + a few-shot linear head.

    Parameters
    ----------
    encoder:
        A torch ``nn.Module`` or any callable mapping a harmonised input
        ``[B, C, N]`` to an embedding ``[B, embed_dim]``. It is frozen by default
        (no gradients, kept in eval mode even while the head trains, so a frozen
        encoder's BatchNorm running stats never drift during few-shot adaptation).
    embed_dim:
        The encoder's output embedding dimension D (the few-shot head's input).
    num_classes:
        Output classes for the linear head.
    target_length:
        If set and the input's time length differs, the window is resampled
        (linear interpolation) to this length before the encoder — the input
        harmonisation a foundation encoder usually needs (it was trained at a
        specific length/rate; AION serves ``[2, 2560]`` at 25.6 kHz). ``None`` =
        pass the length through unchanged.
    target_channels:
        If set and differs from the input channel count, the channels are reduced
        to this many. Only reduction to 1 (via ``channel_reduce``) or pass-through
        is supported; expanding channels is rejected (it would fabricate data).
    channel_reduce:
        ``"mean"`` (average the channels) or ``"first"`` (take channel 0) when
        reducing to ``target_channels=1``.
    normalize:
        Optional callable applied to the (resampled, channel-reduced) window before
        the encoder. Normalisation is ENCODER-SPECIFIC — e.g. UniFault expects
        per-channel min-max, not AION's z-score + 1 Hz high-pass — so the right
        normalisation belongs with the encoder, here, not in AION's preprocessing.
        ``None`` = pass through unchanged.
    freeze:
        Freeze the encoder (default True — the whole point: ride it, adapt the head).
    """

    def __init__(self, encoder: EncoderLike, embed_dim: int,
                 num_classes: int = NUM_CLASSES, *, target_length: int | None = None,
                 target_channels: int | None = None, channel_reduce: str = "mean",
                 normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
                 freeze: bool = True) -> None:
        super().__init__()
        if embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if channel_reduce not in ("mean", "first"):
            raise ValueError("channel_reduce must be 'mean' or 'first'")
        self.encoder = encoder
        self.embed_dim = int(embed_dim)
        self.target_length = target_length
        self.target_channels = target_channels
        self.channel_reduce = channel_reduce
        self.normalize = normalize
        self.frozen = bool(freeze)
        self.head = nn.Linear(self.embed_dim, num_classes)   # the few-shot head
        if self.frozen and isinstance(self.encoder, nn.Module):
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    # -- keep a frozen encoder in eval() even when the adapter is put in train() --
    def train(self, mode: bool = True) -> ExternalEncoderAdapter:
        super().train(mode)
        if self.frozen and isinstance(self.encoder, nn.Module):
            self.encoder.eval()                              # never train the frozen encoder's BN
        return self

    def _harmonize(self, x: torch.Tensor) -> torch.Tensor:
        """Resample length + reduce channels to what the encoder expects."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, C, N], got shape {tuple(x.shape)}")
        b, c, _ = x.shape
        if self.target_channels is not None and self.target_channels != c:
            if self.target_channels != 1:
                raise ValueError(
                    f"target_channels={self.target_channels} but input has {c}: only "
                    "reduction to 1 channel or pass-through is supported (expanding "
                    "channels would fabricate data)")
            x = x[:, :1, :] if self.channel_reduce == "first" else x.mean(dim=1, keepdim=True)
        if self.target_length is not None and self.target_length != x.shape[-1]:
            x = F.interpolate(x, size=self.target_length, mode="linear", align_corners=False)
        if self.normalize is not None:
            x = self.normalize(x)
        return x

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self._harmonize(x)
        if self.frozen:
            with torch.no_grad():
                emb = self.encoder(x)
        else:
            emb = self.encoder(x)
        if emb.dim() != 2 or emb.shape[1] != self.embed_dim:
            raise ValueError(
                f"encoder must return [B, {self.embed_dim}]; got {tuple(emb.shape)}. "
                "Check embed_dim and that the encoder returns a pooled embedding.")
        return emb

    def forward(self, x: torch.Tensor, rpm=None, geometry=None) -> dict:
        emb = self._embed(x)
        # Detach a frozen encoder's embedding so the graph for head training is tiny
        # (and so no_grad/grad mode can't leak into the encoder).
        feat = emb.detach() if self.frozen else emb
        return {"logits": self.head(feat), "features": feat}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def wrap_foundation_encoder(encoder: EncoderLike, embed_dim: int, *,
                            num_classes: int = NUM_CLASSES,
                            input_length: int | None = None,
                            input_channels: int | None = None,
                            channel_reduce: str = "mean") -> ExternalEncoderAdapter:
    """Convenience factory: wrap a frozen foundation ``encoder`` for AION serving.

    ``input_length`` / ``input_channels`` are what the ENCODER expects (the adapter
    harmonises AION's ``[2, 2560]`` window to them). Returns an adapter ready for
    :class:`~aion_nexus.inference.InferenceEngine` (``architecture_version="ext"``)
    and few-shot adaptation of the head only.
    """
    adapter = ExternalEncoderAdapter(
        encoder, embed_dim, num_classes=num_classes, target_length=input_length,
        target_channels=input_channels, channel_reduce=channel_reduce, freeze=True)
    _logger.info("Wrapped frozen foundation encoder: embed_dim=%d, head %d->%d, "
                 "harmonise to %sx%s, trainable params=%d", embed_dim,
                 embed_dim, num_classes, input_channels or NUM_CHANNELS,
                 input_length or "(native)", adapter.get_trainable_params())
    return adapter
