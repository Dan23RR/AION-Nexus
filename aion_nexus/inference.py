"""Inference engine — load checkpoint, predict on validated signals.

Public API:
    engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")
    result = engine.predict(signal_2x2560)
    results = engine.predict_batch([signal1, signal2, ...])
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from aion_nexus.config import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    CLASS_ACTIONS,
    LOW_CONFIDENCE_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
)
from aion_nexus.model import AIONNexus, create_aion_nexus
from aion_nexus.model_v6 import AIONNexusV6, create_aion_nexus_v6
from aion_nexus.preprocessing import preprocess_signal, preprocess_batch
from aion_nexus.version import __version__


_logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Single-sample prediction result with full provenance."""
    predicted_class_index: int
    predicted_class_name: str
    description: str
    probabilities: dict[str, float]    # name -> probability
    confidence: float                  # max probability
    confidence_band: str               # "high" / "medium" / "low"
    recommended_action: dict
    latency_ms: float
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class InferenceEngine:
    """Production inference engine. Thread-safe for read-only inference."""

    def __init__(self, model, device: torch.device | str = "cpu",
                 architecture_version: str = "v1") -> None:
        """Wrap a torch model in an inference engine.

        ``model`` may be an instance of :class:`AIONNexus` (v1) or
        :class:`AIONNexusV6` (v6).
        """
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.architecture_version = architecture_version
        self._latency_ms_running_avg = 0.0
        self._inference_count = 0

    @staticmethod
    def detect_architecture(state_dict: dict) -> str:
        """Inspect state_dict keys to determine v1 vs v6 architecture.

        v1 has `temporal_encoder.gru.*` (BiGRU) and `classifier.classifier.*` (MLP).
        v6 has `temporal_attention.mha.*` and `recursive_reasoner.reasoning_net.*`.

        Returns "v1" or "v6". Raises ValueError if neither pattern matches.
        """
        keys = list(state_dict.keys())
        has_gru = any(k.startswith("temporal_encoder.gru") for k in keys)
        has_mha = any(k.startswith("temporal_attention.mha") for k in keys)
        has_trm = any(k.startswith("recursive_reasoner") for k in keys)
        if has_gru and not has_mha:
            return "v1"
        if has_mha and has_trm:
            return "v6"
        raise ValueError(
            "Unrecognized checkpoint architecture. Keys do not match v1 "
            "(BiGRU + classifier) or v6 (TemporalSelfAttention + TRM). "
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
    ) -> "InferenceEngine":
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
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "See checkpoints/README.md for how to obtain a model file."
            )

        _logger.info("Loading checkpoint: %s", checkpoint_path)
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
        elif version not in ("v1", "v6"):
            raise ValueError(f"Unknown version: {version!r}; must be 'v1' or 'v6'")
        else:
            _logger.info("Forced architecture: %s", version)

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
            recommended_action=CLASS_ACTIONS[name],
            latency_ms=latency,
            model_version=__version__,
        )

    @torch.no_grad()
    def predict_batch(self, signals: Sequence[np.ndarray]) -> list[PredictionResult]:
        """Predict on a batch of signals (more efficient than looped predict)."""
        if not signals:
            return []
        t0 = time.perf_counter()
        x = preprocess_batch(list(signals)).to(self.device)
        out = self.model(x)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()
        latency_per = ((time.perf_counter() - t0) * 1000.0) / len(signals)

        results = []
        for i in range(probs.shape[0]):
            idx = int(np.argmax(probs[i]))
            conf = float(probs[i, idx])
            name = CLASS_NAMES[idx]
            results.append(PredictionResult(
                predicted_class_index=idx,
                predicted_class_name=name,
                description=CLASS_DESCRIPTIONS[name],
                probabilities={n: float(probs[i, j]) for j, n in enumerate(CLASS_NAMES)},
                confidence=conf,
                confidence_band=self._confidence_band(conf),
                recommended_action=CLASS_ACTIONS[name],
                latency_ms=latency_per,
                model_version=__version__,
            ))
        return results

    @torch.no_grad()
    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Return the 512-dim penultimate feature vector (for embedding/clustering)."""
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
