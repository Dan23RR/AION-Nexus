"""v6 architecture tests — instantiation, forward, parameter count, auto-detect.

Mirrors test_smoke.py but for the v6 architecture (TemporalSelfAttention + TRM).
Verifies:
- 716,577 parameter count (matches training log)
- Forward pass shapes
- v1 vs v6 architecture detection from state_dict keys
- Backward-compat: v1 still works after v2.0 changes
"""
import numpy as np
import pytest
import torch

from aion_nexus import (
    AIONNexus, AIONNexusV6,
    create_aion_nexus, create_aion_nexus_v6,
    V6_PARAM_COUNT_4CLASS,
    InferenceEngine,
    NUM_CHANNELS, SIGNAL_LENGTH, NUM_CLASSES, MODEL_PARAM_COUNT,
    TemporalSelfAttention, TinyRecursiveReasoner,
)


class TestV6Architecture:
    def test_v6_param_count_locked(self):
        """v6 architecture parameter count is exactly 716,577."""
        model = create_aion_nexus_v6(num_classes=NUM_CLASSES)
        assert model.get_num_params() == V6_PARAM_COUNT_4CLASS == 716_577

    def test_v6_smaller_than_v1(self):
        """v6 has fewer parameters than v1 (~32.5% reduction)."""
        v1 = create_aion_nexus()
        v6 = create_aion_nexus_v6()
        assert v6.get_num_params() < v1.get_num_params()
        ratio = v6.get_num_params() / v1.get_num_params()
        assert 0.65 < ratio < 0.70  # ~67.5% of v1

    def test_v6_forward_shape(self):
        """Forward pass produces correct output shapes."""
        model = create_aion_nexus_v6()
        model.eval()
        x = torch.randn(4, NUM_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out["logits"].shape == (4, NUM_CLASSES)
        # v6 features are latent_dim=128 (from TRM), not 512 like v1
        assert out["features"].shape == (4, 128)

    def test_v6_forward_with_supervision(self):
        """Deep supervision mode returns supervision_history."""
        model = create_aion_nexus_v6()
        model.train()
        x = torch.randn(2, NUM_CHANNELS, SIGNAL_LENGTH)
        out = model(x, N_supervision=3)
        assert "supervision_history" in out
        assert len(out["supervision_history"]) >= 1
        assert "halt_prob" in out

    def test_v6_extract_features_shape(self):
        """extract_features returns the temporal-attention output [B, 192]."""
        model = create_aion_nexus_v6()
        model.eval()
        x = torch.randn(3, NUM_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            f = model.extract_features(x)
        # Output of TemporalSelfAttention: 192 channels (64 × 3 branches)
        assert f.shape == (3, 192)


class TestArchitectureAutoDetect:
    def test_detect_v1_from_state_dict(self):
        """v1 state_dict contains BiGRU keys → detected as v1."""
        v1_model = create_aion_nexus()
        sd = v1_model.state_dict()
        detected = InferenceEngine.detect_architecture(sd)
        assert detected == "v1"

    def test_detect_v6_from_state_dict(self):
        """v6 state_dict contains MHA + TRM keys → detected as v6."""
        v6_model = create_aion_nexus_v6()
        sd = v6_model.state_dict()
        detected = InferenceEngine.detect_architecture(sd)
        assert detected == "v6"

    def test_detect_unknown_architecture_raises(self):
        """Unrelated state_dict raises ValueError with diagnostic message."""
        bogus = {"foo.bar.weight": torch.zeros(1)}
        with pytest.raises(ValueError, match="Unrecognized checkpoint architecture"):
            InferenceEngine.detect_architecture(bogus)


class TestEngineDualArchitecture:
    def test_engine_v1_predict(self):
        """InferenceEngine wraps v1 model and predicts."""
        engine = InferenceEngine(create_aion_nexus(), architecture_version="v1")
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        r = engine.predict(sig)
        assert r.predicted_class_index in range(NUM_CLASSES)
        assert engine.architecture_version == "v1"

    def test_engine_v6_predict(self):
        """InferenceEngine wraps v6 model and predicts."""
        engine = InferenceEngine(create_aion_nexus_v6(), architecture_version="v6")
        sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
        r = engine.predict(sig)
        assert r.predicted_class_index in range(NUM_CLASSES)
        assert engine.architecture_version == "v6"

    def test_engine_health_includes_arch_version(self):
        """get_health() reports architecture_version."""
        engine = InferenceEngine(create_aion_nexus_v6(), architecture_version="v6")
        h = engine.get_health()
        assert h["architecture_version"] == "v6"
        assert h["model_param_count"] == 716_577


class TestTemporalSelfAttentionUnit:
    def test_temporal_attention_shape(self):
        """Standalone module: [B, 192, 640] → [B, 192]."""
        ta = TemporalSelfAttention(channels=192, num_heads=4)
        ta.eval()
        x = torch.randn(2, 192, 640)
        with torch.no_grad():
            out = ta(x)
        assert out.shape == (2, 192)

    def test_temporal_attention_deterministic_eval(self):
        """Same input + eval mode → identical output."""
        torch.manual_seed(0)
        ta = TemporalSelfAttention()
        ta.eval()
        x = torch.randn(1, 192, 640)
        with torch.no_grad():
            o1 = ta(x)
            o2 = ta(x)
        assert torch.allclose(o1, o2, atol=1e-7)


class TestTinyRecursiveReasonerUnit:
    def test_trm_shape(self):
        """[B, 192] features → [B, num_classes] logits."""
        trm = TinyRecursiveReasoner(feature_dim=192, latent_dim=128, num_classes=4)
        trm.eval()
        f = torch.randn(3, 192)
        with torch.no_grad():
            out = trm(f)
        assert out["logits"].shape == (3, 4)
        assert out["latent"].shape == (3, 128)
        assert out["halt_prob"].shape == (3, 1)

    def test_trm_halt_prob_in_unit_interval(self):
        """halt_prob is sigmoid-bounded."""
        trm = TinyRecursiveReasoner()
        trm.eval()
        with torch.no_grad():
            out = trm(torch.randn(5, 192))
        assert (out["halt_prob"] >= 0).all()
        assert (out["halt_prob"] <= 1).all()


class TestBackwardCompatV1:
    """Ensure v1 (1,061,724 params, F1=0.884) still works after v2.0 changes."""

    def test_v1_param_count_unchanged(self):
        assert create_aion_nexus().get_num_params() == MODEL_PARAM_COUNT == 1_061_724

    def test_v1_forward_unchanged(self):
        m = create_aion_nexus()
        m.eval()
        x = torch.randn(2, NUM_CHANNELS, SIGNAL_LENGTH)
        with torch.no_grad():
            out = m(x)
        assert out["logits"].shape == (2, NUM_CLASSES)
        assert out["features"].shape == (2, 512)  # v1 features = 512-dim
