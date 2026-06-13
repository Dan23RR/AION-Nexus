"""Tests for the heuristic plausibility gate (aion_nexus.ood) and its
integration into the inference engine (red-team kill-shot #4).

The gate must:
  * FLAG white/uniform noise, saturated/clipped signals, and quasi-constant
    traces as implausible (out-of-distribution for bearing vibration);
  * NOT flag real FEMTO windows at any life stage, nor structured synthetic
    signals (sine + harmonics);
  * when integrated, force the engine to ABSTAIN (no escalation) on flagged
    inputs while PRESERVING the raw classifier output (class/conf/probs).

Threshold basis is documented in aion_nexus/ood.py; the FEMTO numbers below
were measured on this machine (2026-06-13). When the FEMTO dataset is present
we assert real windows pass; otherwise that portion is skipped (the synthetic
discrimination tests always run).
"""
import glob
import os

import numpy as np
import pytest
import torch

from aion_nexus import InferenceEngine, create_aion_nexus
from aion_nexus.ood import (
    OODConfig,
    check_signal_plausibility,
)

# Locate the FEMTO dataset if present (tests degrade gracefully without it).
_FEMTO_BASE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "FEMTO+Bearing", "10. FEMTO Bearing", "FEMTOBearingDataSet",
)


def _femto_windows(max_files: int = 6):
    """Yield real [2, 2560] FEMTO windows spanning early/mid/late life, if any."""
    out = []
    for sub in (
        os.path.join("Test_set", "Test_set", "Bearing1_3"),
        os.path.join("Test_set", "Training_set", "Learning_set", "Bearing1_1"),
    ):
        d = os.path.join(_FEMTO_BASE, sub)
        files = sorted(glob.glob(os.path.join(d, "acc_*.csv")))
        if not files:
            continue
        # Sample first / middle / last (early, mid, near-failure).
        for f in (files[0], files[len(files) // 2], files[-1]):
            arr = np.loadtxt(f, delimiter=",")
            out.append(arr[:, [4, 5]].T[:, :2560])
            if len(out) >= max_files:
                return out
    return out


# ---- Discrimination on synthetic signals (always run) -----------------------

class TestPlausibilityGateSynthetic:
    def test_white_gaussian_noise_flagged(self):
        rng = np.random.default_rng(0)
        r = check_signal_plausibility(rng.standard_normal((2, 2560)))
        assert r.ood_flag is True
        assert "spectrum" in r.ood_reason
        # flatness for white noise is ~0.57, well above the 0.45 threshold
        assert r.spectral_flatness > 0.45

    def test_uniform_noise_flagged(self):
        rng = np.random.default_rng(1)
        r = check_signal_plausibility(rng.uniform(-1, 1, (2, 2560)))
        assert r.ood_flag is True

    def test_saturated_clipped_flagged(self):
        rng = np.random.default_rng(2)
        sat = np.clip(rng.standard_normal((2, 2560)) * 100, -10, 10)
        r = check_signal_plausibility(sat)
        assert r.ood_flag is True
        # A railed signal has crest factor near 1 (peak ~ RMS).
        assert "saturated" in r.ood_reason or "spectrum" in r.ood_reason

    def test_quasi_constant_flagged(self):
        rng = np.random.default_rng(3)
        const = np.full((2, 2560), 3.0) + rng.standard_normal((2, 2560)) * 1e-8
        r = check_signal_plausibility(const)
        assert r.ood_flag is True
        assert "quasi-constant" in r.ood_reason

    def test_structured_sine_not_flagged(self):
        """A spectrally-structured signal (tones + light noise) is in-dist-like."""
        rng = np.random.default_rng(4)
        t = np.arange(2560)
        sig = np.vstack([
            np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.13 * t)
            + 0.05 * rng.standard_normal(2560),
            np.cos(2 * np.pi * 0.05 * t) + 0.05 * rng.standard_normal(2560),
        ])
        r = check_signal_plausibility(sig)
        assert r.ood_flag is False
        assert r.ood_reason is None

    def test_nx2_orientation_handled(self):
        """Gate auto-transposes [N, 2] like the rest of the pipeline."""
        rng = np.random.default_rng(5)
        noise_n2 = rng.standard_normal((2560, 2))
        r = check_signal_plausibility(noise_n2)
        assert r.ood_flag is True  # still detected as noise after transpose

    def test_score_is_flatness_in_unit_interval(self):
        rng = np.random.default_rng(6)
        r = check_signal_plausibility(rng.standard_normal((2, 2560)))
        assert 0.0 <= r.ood_score <= 1.0
        assert r.ood_score == r.spectral_flatness

    def test_custom_config_overrides_threshold(self):
        """A permissive flatness threshold lets noise through (tunability)."""
        rng = np.random.default_rng(7)
        noise = rng.standard_normal((2, 2560))
        permissive = OODConfig(flatness_max=0.99, crest_min=0.0, std_min=0.0)
        r = check_signal_plausibility(noise, permissive)
        assert r.ood_flag is False

    def test_config_from_env(self, monkeypatch):
        monkeypatch.setenv("AION_OOD_FLATNESS_MAX", "0.99")
        cfg = OODConfig.from_env()
        assert cfg.flatness_max == 0.99


# ---- Discrimination on REAL FEMTO windows (skipped if dataset absent) --------

class TestPlausibilityGateRealFemto:
    def test_real_femto_windows_not_flagged(self):
        windows = _femto_windows()
        if not windows:
            pytest.skip("FEMTO dataset not present in data/")
        for w in windows:
            r = check_signal_plausibility(w)
            assert r.ood_flag is False, (
                f"real FEMTO window wrongly flagged: flatness={r.spectral_flatness:.3f} "
                f"reason={r.ood_reason}"
            )
            # Real bearing vibration sits well under the 0.45 flatness threshold.
            assert r.spectral_flatness < 0.45


# ---- Engine integration: abstain on flagged, preserve raw output -------------

class TestEngineOodIntegration:
    @pytest.fixture
    def engine(self):
        torch.manual_seed(0)
        return InferenceEngine(create_aion_nexus())

    def test_noise_triggers_abstain_no_escalation(self, engine):
        rng = np.random.default_rng(0)
        res = engine.predict(rng.standard_normal((2, 2560)))
        assert res.ood_flag is True
        assert res.abstain is True
        # Action is forced to no-escalation regardless of the predicted class.
        assert res.recommended_action["alert_level"] == 0
        assert res.recommended_action["stop_machine"] is False
        assert res.recommended_action.get("abstain") is True

    def test_noise_preserves_raw_classifier_output(self, engine):
        """Abstaining must NOT erase the raw class/confidence/probabilities."""
        rng = np.random.default_rng(1)
        res = engine.predict(rng.standard_normal((2, 2560)))
        assert res.predicted_class_name in ("normal", "early", "medium", "advanced")
        assert 0.0 <= res.confidence <= 1.0
        assert abs(sum(res.probabilities.values()) - 1.0) < 1e-5

    def test_structured_signal_not_abstained(self, engine):
        rng = np.random.default_rng(2)
        t = np.arange(2560)
        sig = np.vstack([
            np.sin(2 * np.pi * 0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.13 * t)
            + 0.05 * rng.standard_normal(2560),
            np.cos(2 * np.pi * 0.05 * t) + 0.05 * rng.standard_normal(2560),
        ])
        res = engine.predict(sig)
        assert res.ood_flag is False
        assert res.abstain is False

    def test_real_femto_not_abstained(self, engine):
        windows = _femto_windows(max_files=3)
        if not windows:
            pytest.skip("FEMTO dataset not present in data/")
        for w in windows:
            res = engine.predict(w)
            assert res.ood_flag is False, f"real FEMTO abstained: {res.ood_reason}"

    def test_batch_per_window_ood(self, engine):
        """predict_batch gates each window independently against its raw signal."""
        rng = np.random.default_rng(3)
        noise = rng.standard_normal((2, 2560))
        t = np.arange(2560)
        structured = np.vstack([
            np.sin(2 * np.pi * 0.05 * t) + 0.05 * rng.standard_normal(2560),
            np.cos(2 * np.pi * 0.05 * t) + 0.05 * rng.standard_normal(2560),
        ])
        results = engine.predict_batch([noise, structured])
        assert results[0].ood_flag is True and results[0].abstain is True
        assert results[1].ood_flag is False

    def test_to_dict_includes_ood_fields(self, engine):
        rng = np.random.default_rng(4)
        d = engine.predict(rng.standard_normal((2, 2560))).to_dict()
        for key in ("ood_flag", "ood_score", "ood_reason", "abstain"):
            assert key in d
