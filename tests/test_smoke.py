"""Smoke tests — no checkpoint required, no real data required.

These tests verify that the package can be imported, the model can be
instantiated with the expected parameter count, and a forward pass produces
output of the expected shape on synthetic input.

Run: pytest tests/test_smoke.py -v
"""
import numpy as np
import pytest
import torch

from aion_nexus import (
    MODEL_PARAM_COUNT,
    NUM_CHANNELS,
    NUM_CLASSES,
    SIGNAL_LENGTH,
    InferenceEngine,
    create_aion_nexus,
    preprocess_signal,
    validate_signal,
)
from aion_nexus.preprocessing import SignalValidationError


def test_model_param_count():
    """Architecture parameter count is locked at 1,061,724."""
    model = create_aion_nexus(num_classes=NUM_CLASSES)
    assert model.get_num_params() == MODEL_PARAM_COUNT


def test_forward_pass_shape():
    """Forward pass produces correct output shapes."""
    model = create_aion_nexus()
    model.eval()
    x = torch.randn(4, NUM_CHANNELS, SIGNAL_LENGTH)
    with torch.no_grad():
        out = model(x)
    assert out["logits"].shape == (4, NUM_CLASSES)
    assert out["features"].shape == (4, 512)


def test_preprocess_accepts_2xn():
    sig = np.random.randn(NUM_CHANNELS, 3000).astype(np.float32)
    t = preprocess_signal(sig)
    assert t.shape == (1, NUM_CHANNELS, SIGNAL_LENGTH)
    # After z-score + HP-Butterworth filter, mean is near zero (HP removes DC)
    # but std differs from 1 because HP attenuates low frequencies.
    arr = t.numpy()[0, 0]
    assert abs(arr.mean()) < 1e-2  # HP removes DC bias
    assert 0.5 < arr.std() < 1.5    # HP modifies std but stays in reasonable range


def test_preprocess_accepts_nx2_transpose():
    sig = np.random.randn(3000, NUM_CHANNELS).astype(np.float32)
    t = preprocess_signal(sig)
    assert t.shape == (1, NUM_CHANNELS, SIGNAL_LENGTH)


def test_preprocess_rejects_wrong_dim():
    with pytest.raises(SignalValidationError):
        validate_signal(np.random.randn(2560))


def test_preprocess_rejects_too_short():
    with pytest.raises(SignalValidationError):
        validate_signal(np.random.randn(2, 1000))


def test_preprocess_rejects_nan():
    sig = np.random.randn(2, 3000)
    sig[0, 100] = np.nan
    with pytest.raises(SignalValidationError):
        validate_signal(sig)


def test_preprocess_rejects_stuck_sensor():
    sig = np.zeros((2, 3000), dtype=np.float32)  # both channels stuck
    with pytest.raises(SignalValidationError):
        validate_signal(sig)


def test_inference_engine_synthetic_no_checkpoint():
    """Wrap a freshly initialized model — predicts with random weights, but
    pipeline must produce valid output structure."""
    model = create_aion_nexus()
    engine = InferenceEngine(model, device="cpu")
    sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
    result = engine.predict(sig)
    assert result.predicted_class_index in range(NUM_CLASSES)
    assert 0.0 <= result.confidence <= 1.0
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert result.confidence_band in ["low", "medium", "high"]
    assert result.latency_ms > 0
    assert isinstance(result.recommended_action, dict)


def test_predict_batch_consistent_with_predict():
    """Batched inference should produce same predictions as looped single calls."""
    model = create_aion_nexus()
    engine = InferenceEngine(model, device="cpu")
    sigs = [np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32) for _ in range(3)]

    batched = engine.predict_batch(sigs)
    individual = [engine.predict(s) for s in sigs]

    assert len(batched) == 3
    for b, i in zip(batched, individual, strict=False):
        assert b.predicted_class_index == i.predicted_class_index
        for cls, p in b.probabilities.items():
            assert p == pytest.approx(i.probabilities[cls], abs=1e-5)


def test_extract_features_shape():
    model = create_aion_nexus()
    engine = InferenceEngine(model, device="cpu")
    sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
    feat = engine.extract_features(sig)
    assert feat.shape == (512,)


def test_health_snapshot():
    model = create_aion_nexus()
    engine = InferenceEngine(model, device="cpu")
    sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
    engine.predict(sig)
    h = engine.get_health()
    assert h["model_param_count"] == MODEL_PARAM_COUNT
    assert h["inference_count"] == 1
    assert h["running_avg_latency_ms"] > 0
