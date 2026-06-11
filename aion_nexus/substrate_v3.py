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
import hashlib
import logging

import torch
import torch.nn as nn

from aion_nexus.config import NUM_CHANNELS, NUM_CLASSES

_logger = logging.getLogger(__name__)


def _sha256_file(path: str, _chunk: int = 1 << 20) -> str:
    """Stream the SHA-256 of a file without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def _safe_torch_load(checkpoint_path: str, map_location: str = "cpu",
                     allow_unsafe_load: bool = False,
                     expected_sha256: str | None = None):
    """Hardened ``torch.load``: SHA-256 integrity check + ``weights_only=True`` default.

    Mirrors the pattern in :meth:`InferenceEngine.from_checkpoint` (inference.py):
    ``weights_only=True`` blocks pickle-based arbitrary-code execution; the unsafe
    fallback is opt-in only. Registered hashes are in checkpoints/README.md.
    """
    if expected_sha256 is not None:
        actual = _sha256_file(checkpoint_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {checkpoint_path}: expected "
                f"{expected_sha256}, got {actual}. Refusing to load a checkpoint "
                "whose integrity cannot be verified."
            )
        _logger.info("Checkpoint SHA-256 verified: %s", actual)
    try:
        return torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not allow_unsafe_load:
            raise ValueError(
                f"Failed to load {checkpoint_path} with weights_only=True "
                f"({type(exc).__name__}: {exc}). This is the safe path that "
                "blocks arbitrary code execution from pickled checkpoints. If "
                "you trust this file, pass allow_unsafe_load=True (and prefer "
                "also passing expected_sha256)."
            ) from exc
        _logger.warning(
            "weights_only=True failed (%s); falling back to UNSAFE "
            "weights_only=False because allow_unsafe_load=True. Only do this for "
            "checkpoints you produced or whose SHA-256 you verified.",
            type(exc).__name__,
        )
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False)

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
                        map_location: str = "cpu", allow_unsafe_load: bool = False,
                        expected_sha256: str | None = None) -> "AIONNexusV3":
        """Build v3 OR v3.1 from a checkpoint, reading the architecture from its `cfg`
        (so a bigger v3.1 substrate drops in with no code change). The pretraining-only
        `mask_token` (v3.1) is dropped — the embed path matches exactly.

        Loads with ``weights_only=True`` by default (blocks pickle-based RCE);
        ``allow_unsafe_load=True`` is the opt-in fallback for trusted files only.
        Pass ``expected_sha256`` (see checkpoints/README.md) to also verify
        file integrity before loading.
        """
        sd = _safe_torch_load(checkpoint_path, map_location=map_location,
                              allow_unsafe_load=allow_unsafe_load,
                              expected_sha256=expected_sha256)
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

    def load_substrate(self, checkpoint_path: str, map_location: str = "cpu",
                       allow_unsafe_load: bool = False,
                       expected_sha256: str | None = None) -> None:
        """Load the pretrained substrate encoder weights (leaves head fresh).

        Safe by default (``weights_only=True``); see :meth:`from_checkpoint` for
        the ``allow_unsafe_load`` / ``expected_sha256`` semantics.
        """
        sd = _safe_torch_load(checkpoint_path, map_location=map_location,
                              allow_unsafe_load=allow_unsafe_load,
                              expected_sha256=expected_sha256)
        enc_sd = sd.get("encoder", sd)
        missing, unexpected = self.encoder.load_state_dict(enc_sd, strict=False)
        if unexpected:
            raise ValueError(f"unexpected substrate keys: {unexpected[:4]}...")
        _logger.info("Loaded v3 substrate encoder (objective=%s)", sd.get("objective", "?"))


def v3_from_model_state_dict(state_dict: dict) -> AIONNexusV3:
    """Rebuild a full AIONNexusV3 (encoder + trained head) from a model state dict.

    Used by :meth:`InferenceEngine.from_checkpoint` to serve adapted v3 checkpoints
    (e.g. saved by :meth:`FewShotAdapter.save`). The architecture is inferred from
    tensor shapes: ``encoder.proj.weight`` gives d_model/patch_len, the layer count
    gives depth, ``linear1`` gives dim_ff. ``nhead`` is NOT recoverable from a state
    dict, so the frozen contracts are assumed: nhead=4 for d_model 192 (v3),
    nhead=8 for d_model 256 (v3.1); anything else falls back to 4 with a warning.

    Raises:
        ValueError: if the state dict has no trained head (``head.weight``) — a
            pretrained substrate ENCODER alone must not be served: its head would
            be random. Adapt it first (see checkpoints/README.md).
    """
    if "head.weight" not in state_dict:
        raise ValueError(
            "This v3 checkpoint contains a pretrained substrate ENCODER but no "
            "trained classification head; serving it would emit predictions from "
            "an untrained (random) head. Adapt it first:\n"
            "  model = AIONNexusV3.from_checkpoint(path)\n"
            "  engine = InferenceEngine(model, architecture_version='v3')\n"
            "  adapter = FewShotAdapter(engine)\n"
            "  adapter.adapt(target_signals, target_labels)\n"
            "  adapter.save('checkpoints/aion_nexus_v3_adapted.pth')\n"
            "then serve the adapted checkpoint (e.g. via AION_CHECKPOINT)."
        )
    proj_w = state_dict["encoder.proj.weight"]
    d_model = proj_w.shape[0]
    patch_len = proj_w.shape[1] // NUM_CHANNELS
    depth = 1 + max(
        int(k.split(".")[3]) for k in state_dict if k.startswith("encoder.tr.layers.")
    )
    dim_ff = state_dict["encoder.tr.layers.0.linear1.weight"].shape[0]
    nhead = {192: 4, 256: 8}.get(d_model)
    if nhead is None:
        _logger.warning(
            "v3 state dict has non-standard d_model=%d; assuming nhead=4 "
            "(nhead is not recoverable from a state dict)", d_model,
        )
        nhead = 4
    num_classes = state_dict["head.weight"].shape[0]
    enc = SubstrateEncoderV3(patch_len=patch_len, d_model=d_model, depth=depth,
                             nhead=nhead, dim_ff=dim_ff)
    model = AIONNexusV3(num_classes=num_classes, encoder=enc)
    model.load_state_dict(state_dict, strict=True)
    return model


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
