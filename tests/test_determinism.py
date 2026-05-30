"""Determinism + reproducibility tests.

Verifies that:
- Same input + same seed → same output (bit-exact)
- Same input across batched vs single → same output (within float tolerance)
- Architecture instantiation is reproducible given seed
"""
import numpy as np
import pytest
import torch

from aion_nexus import (
    create_aion_nexus, InferenceEngine,
    NUM_CHANNELS, SIGNAL_LENGTH, NUM_CLASSES,
)


def _set_seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)


class TestDeterminism:
    def test_same_seed_same_logits_random_weights(self):
        """Two model instances with same seed produce identical forward."""
        x = torch.randn(2, NUM_CHANNELS, SIGNAL_LENGTH)

        _set_seed(42)
        model1 = create_aion_nexus()
        model1.eval()
        with torch.no_grad():
            out1 = model1(x)

        _set_seed(42)
        model2 = create_aion_nexus()
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6)
        assert torch.allclose(out1["features"], out2["features"], atol=1e-6)

    def test_same_input_same_output_eval_mode(self):
        """Same input passed twice in eval mode produces identical output."""
        _set_seed(0)
        model = create_aion_nexus()
        model.eval()
        x = torch.randn(1, NUM_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out_a = model(x)
            out_b = model(x)
        assert torch.allclose(out_a["logits"], out_b["logits"], atol=1e-7)

    def test_dropout_disabled_in_eval(self):
        """Eval mode disables dropout: predictions are deterministic."""
        _set_seed(0)
        model = create_aion_nexus()
        model.eval()
        engine = InferenceEngine(model)
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        r1 = engine.predict(sig)
        r2 = engine.predict(sig)
        for cls in r1.probabilities:
            assert r1.probabilities[cls] == pytest.approx(
                r2.probabilities[cls], abs=1e-6
            )

    def test_batch_vs_single_consistency(self):
        """Batched inference produces same per-sample results as single calls."""
        _set_seed(0)
        engine = InferenceEngine(create_aion_nexus())
        sigs = [np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
                for _ in range(4)]

        single = [engine.predict(s) for s in sigs]
        batched = engine.predict_batch(sigs)

        for s, b in zip(single, batched):
            assert s.predicted_class_index == b.predicted_class_index
            for cls in s.probabilities:
                assert s.probabilities[cls] == pytest.approx(
                    b.probabilities[cls], abs=1e-5
                )

    def test_extract_features_consistency(self):
        """extract_features returns same vector on identical input."""
        _set_seed(0)
        engine = InferenceEngine(create_aion_nexus())
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        f1 = engine.extract_features(sig)
        f2 = engine.extract_features(sig)
        np.testing.assert_array_almost_equal(f1, f2, decimal=6)


class TestArchitectureFrozen:
    def test_param_count_locked(self):
        """Architecture parameter count is exactly 1,061,724."""
        model = create_aion_nexus()
        assert model.get_num_params() == 1_061_724

    def test_param_count_with_alternate_classes(self):
        """Different num_classes changes only the final linear layer."""
        m4 = create_aion_nexus(num_classes=4)
        # Different class count is allowed (param check is bypassed for non-default)
        from aion_nexus.model import AIONNexus
        m6 = AIONNexus(num_classes=6)  # bypass create_aion_nexus guard
        # Difference: final Linear(128 -> num_classes); diff = (6-4)*128 + (6-4) = 258 params
        assert m6.get_num_params() - m4.get_num_params() == 2 * 128 + 2

    def test_feature_dim_locked(self):
        """Penultimate feature dim is 512."""
        model = create_aion_nexus()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, NUM_CHANNELS, SIGNAL_LENGTH))
        assert out["features"].shape == (1, 512)

    def test_logits_dim_locked(self):
        model = create_aion_nexus()
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, NUM_CHANNELS, SIGNAL_LENGTH))
        assert out["logits"].shape == (2, NUM_CLASSES)
