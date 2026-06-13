"""Inference engine — load checkpoint, predict on validated signals.

Public API:
    engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")
    result = engine.predict(signal_2x2560)
    results = engine.predict_batch([signal1, signal2, ...])
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from aion_nexus.config import (
    CLASS_ACTIONS,
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
)
from aion_nexus.model import create_aion_nexus
from aion_nexus.model_v6 import create_aion_nexus_v6
from aion_nexus.ood import OODConfig, OODResult, check_signal_plausibility
from aion_nexus.preprocessing import preprocess_batch, preprocess_signal
from aion_nexus.version import __version__

_logger = logging.getLogger(__name__)


def _sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Stream the SHA-256 of a file without loading it fully into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class PredictionResult:
    """Single-sample prediction result with full provenance.

    The ``ood_*`` fields are an ADDITIVE plausibility-gate annotation (see
    ``aion_nexus.ood``). They never change the ``predicted_class_*`` /
    ``probabilities`` / ``confidence`` fields (the raw classifier output is
    always preserved for transparency and downstream auditing). When
    ``ood_flag`` is True the input is implausible as bearing vibration, the
    engine ABSTAINS, and ``recommended_action`` is overridden to a no-escalation
    abstain action (alert_level 0, stop_machine False) regardless of the
    classifier's verdict.
    """
    predicted_class_index: int
    predicted_class_name: str
    description: str
    probabilities: dict[str, float]    # name -> probability (raw classifier output)
    confidence: float                  # max probability (raw classifier output)
    confidence_band: str               # "high" / "medium" / "low"
    recommended_action: dict
    latency_ms: float
    model_version: str
    # Additive OOD / plausibility-gate fields (backward-compatible).
    ood_flag: bool = False
    ood_score: float = 0.0
    ood_reason: str | None = None
    abstain: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# Action returned when the plausibility gate abstains: no escalation, ever.
_ABSTAIN_ACTION: dict = {
    "alert_level": 0,
    "stop_machine": False,
    "schedule_inspection": False,
    "abstain": True,
}


class InferenceEngine:
    """Production inference engine. Thread-safe for read-only inference."""

    def __init__(self, model, device: torch.device | str = "cpu",
                 architecture_version: str = "v1",
                 ood_config: OODConfig | None = None) -> None:
        """Wrap a torch model in an inference engine.

        ``model`` may be an instance of :class:`AIONNexus` (v1) or
        :class:`AIONNexusV6` (v6).

        ``ood_config`` tunes the heuristic plausibility gate (see
        ``aion_nexus.ood``); defaults to thresholds from ``AION_OOD_*`` env vars.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.architecture_version = architecture_version
        self.ood_config = ood_config if ood_config is not None else OODConfig.from_env()
        self._latency_ms_running_avg = 0.0
        self._inference_count = 0

    def _apply_ood_gate(
        self, signal: np.ndarray, idx: int, conf: float, band: str
    ) -> tuple[int, float, str, dict, OODResult]:
        """Run the plausibility gate on a RAW signal window.

        Returns ``(idx, conf, band, recommended_action, ood_result)``. The class
        index / confidence / band are passed through UNCHANGED (raw classifier
        output is always preserved); only ``recommended_action`` is replaced with
        the abstain action when the gate fires, so an implausible input can never
        escalate to an automated stop-machine.
        """
        ood = check_signal_plausibility(signal, self.ood_config)
        if ood.ood_flag:
            return idx, conf, band, dict(_ABSTAIN_ACTION), ood
        return idx, conf, band, CLASS_ACTIONS[CLASS_NAMES[idx]], ood

    @staticmethod
    def detect_architecture(state_dict: dict) -> str:
        """Inspect state_dict keys to determine v1 / v6 / v3 architecture.

        v1 has `temporal_encoder.gru.*` (BiGRU) and `classifier.classifier.*` (MLP).
        v6 has `temporal_attention.mha.*` and `recursive_reasoner.reasoning_net.*`.
        v3 (PatchTST substrate) comes in two forms:
          - full model state dict: `encoder.proj.*` / `encoder.tr.*` (+ `head.*`
            when a few-shot head has been trained, e.g. `FewShotAdapter.save`);
          - raw pretrained substrate checkpoint: a nested dict under the
            `encoder` key (with `proj.*` / `tr.*` inside) plus `cfg`/`objective`.

        Returns "v1", "v6" or "v3". Raises ValueError if no pattern matches.
        """
        keys = list(state_dict.keys())
        has_gru = any(k.startswith("temporal_encoder.gru") for k in keys)
        has_mha = any(k.startswith("temporal_attention.mha") for k in keys)
        has_trm = any(k.startswith("recursive_reasoner") for k in keys)
        if has_gru and not has_mha:
            return "v1"
        if has_mha and has_trm:
            return "v6"
        # v3 full-model state dict (encoder.* flattened tensors)
        if any(k.startswith(("encoder.proj.", "encoder.tr.")) for k in keys):
            return "v3"
        # v3 raw substrate checkpoint ({'encoder': {...}, 'cfg': ..., 'objective': ...})
        enc = state_dict.get("encoder")
        if isinstance(enc, dict) and any(k.startswith(("proj.", "tr.")) for k in enc):
            return "v3"
        raise ValueError(
            "Unrecognized checkpoint architecture. Keys do not match v1 "
            "(BiGRU + classifier), v6 (TemporalSelfAttention + TRM) or v3 "
            "(PatchTST substrate encoder). "
            f"First 10 keys: {keys[:10]}"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cpu",
        num_classes: int = NUM_CLASSES,
        strict: bool = True,
        version: str | None = None,
        expected_sha256: str | None = None,
        allow_unsafe: bool = False,
    ) -> InferenceEngine:
        """Load a saved checkpoint and instantiate an inference engine.

        Args:
            checkpoint_path: path to a .pth file produced by training.
            device: 'cpu' / 'cuda' / 'cuda:0'.
            num_classes: must match the trained head dimensionality.
            strict: if True, fail on unexpected/missing keys in state_dict.
            version: 'v1' or 'v6' to force a specific architecture.
                If None (default), architecture is auto-detected from state_dict
                keys. Use this only when auto-detection fails (e.g., partial
                checkpoints).
            expected_sha256: if provided, the file's SHA-256 is verified before
                loading and a mismatch raises ValueError (integrity / supply-chain
                guard). Registered hashes are in checkpoints/README.md.
            allow_unsafe: by default the checkpoint is loaded with
                ``weights_only=True``, which blocks arbitrary-code execution from a
                maliciously-pickled file. Set True only for fully-trusted files to
                fall back to the unsafe ``weights_only=False`` path.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "See checkpoints/README.md for how to obtain a model file."
            )

        if expected_sha256 is not None:
            actual = _sha256_file(checkpoint_path)
            if actual.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch for {checkpoint_path}: expected "
                    f"{expected_sha256}, got {actual}. Refusing to load a checkpoint "
                    "whose integrity cannot be verified."
                )
            _logger.info("Checkpoint SHA-256 verified: %s", actual)

        _logger.info("Loading checkpoint: %s", checkpoint_path)
        # Safe by default: weights_only=True prevents pickle-based RCE. All shipped
        # checkpoints load under this path; the unsafe fallback is opt-in only.
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            if not allow_unsafe:
                raise ValueError(
                    f"Failed to load {checkpoint_path} with weights_only=True "
                    f"({type(exc).__name__}: {exc}). This is the safe path that "
                    "blocks arbitrary code execution from pickled checkpoints. If "
                    "you trust this file, pass allow_unsafe=True (and prefer also "
                    "passing expected_sha256)."
                ) from exc
            _logger.warning(
                "weights_only=True failed (%s); falling back to UNSAFE "
                "weights_only=False because allow_unsafe=True. Only do this for "
                "checkpoints you produced or whose SHA-256 you verified.",
                type(exc).__name__,
            )
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        if version is None:
            version = cls.detect_architecture(state_dict)
            _logger.info("Auto-detected architecture: %s", version)
        elif version not in ("v1", "v6", "v3"):
            raise ValueError(f"Unknown version: {version!r}; must be 'v1', 'v6' or 'v3'")
        else:
            _logger.info("Forced architecture: %s", version)

        if version == "v3":
            # v3 substrate: a full model state dict (trained few-shot head) is
            # rebuilt and served like any other architecture. A raw pretrained
            # ENCODER-only checkpoint is refused with an actionable error —
            # serving it would emit predictions from an untrained head (the
            # server then starts in degraded mode and reports this via /health).
            from aion_nexus.substrate_v3 import v3_from_model_state_dict
            model = v3_from_model_state_dict(state_dict)
            _logger.info("Checkpoint loaded successfully (v3 architecture)")
            return cls(model=model, device=device, architecture_version="v3")

        if version == "v1":
            model = create_aion_nexus(num_classes=num_classes)
        else:
            model = create_aion_nexus_v6(num_classes=num_classes)

        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            _logger.warning(
                "Loaded checkpoint with missing=%s unexpected=%s", missing, unexpected
            )
        _logger.info("Checkpoint loaded successfully (%s architecture)", version)
        engine = cls(model=model, device=device)
        engine.architecture_version = version
        return engine

    def _confidence_band(self, conf: float) -> str:
        if conf >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        if conf >= LOW_CONFIDENCE_THRESHOLD:
            return "medium"
        return "low"

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> PredictionResult:
        """Predict on a single 2-channel signal.

        Args:
            signal: numpy array, shape [2, N] or [N, 2] with N >= 2560.

        Returns:
            PredictionResult with probabilities, confidence, action.
        """
        t0 = time.perf_counter()
        x = preprocess_signal(signal).to(self.device)
        out = self.model(x)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        band = self._confidence_band(conf)

        # Plausibility gate on the RAW signal (before z-score/highpass erase
        # amplitude). On an implausible input we ABSTAIN: the raw class/conf are
        # preserved, but recommended_action is forced to no-escalation.
        idx, conf, band, action, ood = self._apply_ood_gate(signal, idx, conf, band)
        latency = (time.perf_counter() - t0) * 1000.0

        # Update running latency average for monitoring
        self._inference_count += 1
        alpha = 1.0 / min(self._inference_count, 100)
        self._latency_ms_running_avg = (
            (1 - alpha) * self._latency_ms_running_avg + alpha * latency
        )

        name = CLASS_NAMES[idx]
        return PredictionResult(
            predicted_class_index=idx,
            predicted_class_name=name,
            description=CLASS_DESCRIPTIONS[name],
            probabilities={n: float(probs[i]) for i, n in enumerate(CLASS_NAMES)},
            confidence=conf,
            confidence_band=band,
            recommended_action=action,
            latency_ms=latency,
            model_version=__version__,
            ood_flag=ood.ood_flag,
            ood_score=ood.ood_score,
            ood_reason=ood.ood_reason,
            abstain=ood.ood_flag,
        )

    @torch.no_grad()
    def predict_batch(self, signals: Sequence[np.ndarray]) -> list[PredictionResult]:
        """Predict on a batch of signals (more efficient than looped predict)."""
        if not signals:
            return []
        signals = list(signals)
        t0 = time.perf_counter()
        x = preprocess_batch(signals).to(self.device)
        out = self.model(x)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
        latency_per = ((time.perf_counter() - t0) * 1000.0) / len(signals)

        results = []
        for i in range(probs.shape[0]):
            idx = int(np.argmax(probs[i]))
            conf = float(probs[i, idx])
            band = self._confidence_band(conf)
            # Per-window plausibility gate on the corresponding RAW signal.
            idx, conf, band, action, ood = self._apply_ood_gate(
                signals[i], idx, conf, band
            )
            name = CLASS_NAMES[idx]
            results.append(PredictionResult(
                predicted_class_index=idx,
                predicted_class_name=name,
                description=CLASS_DESCRIPTIONS[name],
                probabilities={n: float(probs[i, j]) for j, n in enumerate(CLASS_NAMES)},
                confidence=conf,
                confidence_band=band,
                recommended_action=action,
                latency_ms=latency_per,
                model_version=__version__,
                ood_flag=ood.ood_flag,
                ood_score=ood.ood_score,
                ood_reason=ood.ood_reason,
                abstain=ood.ood_flag,
            ))
        return results

    @torch.no_grad()
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Return the penultimate feature vector (512-dim on v1, 128-dim on v6, 192-dim on v3)."""
        x = preprocess_signal(signal).to(self.device)
        out = self.model(x)
        return out["features"].cpu().numpy()[0]

    def get_health(self) -> dict:
        """Return a snapshot of engine health for /health endpoint."""
        return {
            "version": __version__,
            "architecture_version": self.architecture_version,
            "device": str(self.device),
            "model_param_count": self.model.get_num_params(),
            "inference_count": self._inference_count,
            "running_avg_latency_ms": round(self._latency_ms_running_avg, 2),
        }
