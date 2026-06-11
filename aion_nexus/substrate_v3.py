"""AION-NEXUS v3 — self-supervised SUBSTRATE backbone (foundation model).

A PatchTST-style patch transformer pretrained self-supervised (contrastive
NT-Xent) on pooled unlabeled vibration (FEMTO + MFPT + CWRU). Unlike v1 (BiGRU)
and v6 (TempAttn+TRM) — which are supervised classifiers — v3 is a FROZEN
foundation ENCODER: the classification head is trained per-deployment with a few
labels (few-shot adaptation), then served through the verified trust layer
(conformal + physics verifier -> certificate).

POSITIONING (honest, §6.31):
  v3 is NOT a higher-accuracy in-distribution classifier. v1 in-distribution
  FEMTO F1 = 0.884 is unbeaten by v3 for that task. v3's value is CROSS-DOMAIN
  FEW-SHOT: validated under leave-one-bearing-out and cross-DATASET (FEMTO↔MFPT↔
  CWRU), 10-shot binary-health macro-F1 0.91–1.00 vs random-init 0.5–0.8.
  Use v3 for: adapting to a NEW machine/rig with ~10 labels/class, and as the
  trustworthy backbone behind the AION-2 certified pipeline. Zero-shot cross-rig
  is NOT reliable — collect the few labels.

Frozen architecture (must match the pretrained checkpoint
`checkpoints/aion_nexus_substrate_v3.pth`, objective contrastive-ntxent-patchTST):
  patch_len=64, d_model=192, depth=4, nhead=4, dim_ff=384  ->  encoder 1,220,928 params.

Same input contract as v1/v6: [B, 2, 2560], z-score per channel + 1 Hz HP.
"""
import logging

import torch
import torch.nn as nn

from aion_nexus.config import NUM_CHANNELS, NUM_CLASSES

_logger = logging.getLogger(__name__)

V3_ENCODER_PARAM_COUNT = 1_220_928    # v3 (d_model 192, depth 4) — verified against checkpoint
V3_1_ENCODER_PARAM_COUNT = 3_206_400  # v3.1 (d_model 256, depth 6) PRODUCTION encoder; the
# pretraining encoder is +256 (the mask_token, dropped on load). Load via AIONNexusV3.from_checkpoint.
V3_EMBED_DIM = 192


class SubstrateEncoderV3(nn.Module):
    """[B,2,2560] -> patches -> transformer -> mean-pooled embedding [B,192]."""

    def __init__(self, in_ch: int = NUM_CHANNELS, seq: int = 2560, patch_len: int = 64,
                 d_model: int = 192, depth: int = 4, nhead: int = 4, dim_ff: int = 384,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if seq % patch_len != 0:
            raise ValueError("seq must be divisible by patch_len")
        self.patch_len = patch_len
        self.n_patches = seq // patch_len
        self.out_dim = d_model
        self.proj = nn.Linear(in_ch * patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        p = x.unfold(2, self.patch_len, self.patch_len)
        p = p.permute(0, 2, 1, 3).reshape(b, self.n_patches, -1)
        h = self.tr(self.proj(p) + self.pos)
        return self.norm(h.mean(1))


class AIONNexusV3(nn.Module):
    """Substrate encoder + linear severity/health head.

    Forward returns ``{'logits': [B,num_classes], 'features': [B,192]}`` — same
    contract as v1/v6 so it drops into the certified inference + few-shot paths.
    The head is trained per-deployment (few-shot); the encoder stays frozen.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, freeze_encoder: bool = True,
                 encoder=None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else SubstrateEncoderV3()
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Linear(self.encoder.out_dim, num_classes)   # few-shot head (dim follows encoder)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, num_classes: int = NUM_CLASSES,
                        map_location: str = "cpu") -> "AIONNexusV3":
        """Build v3 OR v3.1 from a checkpoint, reading the architecture from its `cfg`
        (so a bigger v3.1 substrate drops in with no code change). The pretraining-only
        `mask_token` (v3.1) is dropped — the embed path matches exactly."""
        sd = torch.load(checkpoint_path, map_location=map_location)
        cfg = sd.get("cfg", {})
        enc = SubstrateEncoderV3(
            patch_len=cfg.get("patch_len", 64), d_model=cfg.get("d_model", 192),
            depth=cfg.get("depth", 4), nhead=cfg.get("nhead", 4), dim_ff=cfg.get("dim_ff", 384),
        )
        model = cls(num_classes=num_classes, encoder=enc)
        enc_sd = {k: v for k, v in sd.get("encoder", sd).items() if k != "mask_token"}
        model.encoder.load_state_dict(enc_sd, strict=True)
        _logger.info("Loaded substrate v3/v3.1 (objective=%s, d_model=%d, depth=%d, enc_params=%d)",
                     sd.get("objective", "?"), enc.out_dim, cfg.get("depth", 4),
                     sum(p.numel() for p in enc.parameters()))
        return model

    def forward(self, x: torch.Tensor, rpm=None, geometry=None) -> dict:
        emb = self.encoder.embed(x)
        return {"logits": self.head(emb), "features": emb}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def load_substrate(self, checkpoint_path: str, map_location: str = "cpu") -> None:
        """Load the pretrained substrate encoder weights (leaves head fresh)."""
        sd = torch.load(checkpoint_path, map_location=map_location)
        enc_sd = sd.get("encoder", sd)
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        if unexpected:
            raise ValueError(f"unexpected substrate keys: {unexpected[:4]}...")
        _logger.info("Loaded v3 substrate encoder (objective=%s)", sd.get("objective", "?"))


def create_substrate_v3(num_classes: int = NUM_CLASSES, strict_param_check: bool = True) -> AIONNexusV3:
    """Factory with encoder param-count verification against the frozen checkpoint."""
    model = AIONNexusV3(num_classes=num_classes)
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    if strict_param_check and n_enc != V3_ENCODER_PARAM_COUNT:
        raise ValueError(
            f"v3 substrate drift: encoder has {n_enc:,} params, expected "
            f"{V3_ENCODER_PARAM_COUNT:,}. Architecture diverged from the pretrained checkpoint."
        )
    _logger.info("AION-NEXUS v3 substrate: encoder %d params, embed_dim %d", n_enc, V3_EMBED_DIM)
    return model
