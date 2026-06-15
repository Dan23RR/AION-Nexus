"""Tests for the first-class degradation-STAGE output (aion_nexus.degradation).

Covers the honesty-critical invariants:
- ``degradation_index`` is monotone in the stage the probability mass concentrates on;
- the conformal stage set (via aion_nexus.verify) covers the true stage at the
  target coverage on synthetic exchangeable data;
- an OOD-flagged window ABSTAINS;
- the uncalibrated path is honestly labelled (calibrated=False, no stage set);
- the /predict_degradation endpoint responds and the EXISTING /predict contract
  is unchanged (additive, non-breaking).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from aion_nexus.config import CLASS_NAMES, NUM_CLASSES
from aion_nexus.degradation import (
    DEGRADATION_DISCLAIMER,
    STAGE_LABELS,
    DegradationEstimate,
    degradation_index_from_probs,
    estimate_degradation,
)
from aion_nexus.verify import ConformalCalibrator

# Keep the server out-of-band env vars from leaking in (mirrors test_api_integration).
os.environ.pop("AION_CHECKPOINT", None)
os.environ.pop("AION_API_KEY", None)


def _onehot(stage: int, n: int = NUM_CLASSES) -> np.ndarray:
    v = np.full(n, 0.01 / (n - 1), dtype=np.float64)
    v[stage] = 0.99
    return v


# --------------------------------------------------------------------------- #
# degradation_index monotonicity
# --------------------------------------------------------------------------- #

class TestDegradationIndexMonotone:
    def test_index_increases_with_stage_onehot(self):
        indices = [degradation_index_from_probs(_onehot(k)) for k in range(NUM_CLASSES)]
        assert all(b > a for a, b in zip(indices, indices[1:], strict=False)), indices

    def test_pristine_index_near_zero_endoflife_near_one(self):
        # Perfectly-confident earliest stage -> ~0; latest stage -> ~1.
        v0 = np.array([1.0, 0.0, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 0.0, 1.0])
        assert degradation_index_from_probs(v0) == pytest.approx(0.0, abs=1e-9)
        assert degradation_index_from_probs(v3) == pytest.approx(1.0, abs=1e-9)

    def test_shifting_mass_upward_strictly_increases_index(self):
        base = np.array([0.4, 0.3, 0.2, 0.1])
        shifted = np.array([0.1, 0.2, 0.3, 0.4])  # mass moved to higher stages
        assert degradation_index_from_probs(shifted) > degradation_index_from_probs(base)

    def test_index_matches_estimate_field(self):
        v = np.array([0.1, 0.5, 0.3, 0.1])
        est = estimate_degradation(v)
        assert est.degradation_index == pytest.approx(degradation_index_from_probs(v))

    def test_stage_ordinal_matches_argmax(self):
        for k in range(NUM_CLASSES):
            est = estimate_degradation(_onehot(k))
            assert est.stage_ordinal == k
            assert est.stage_label == STAGE_LABELS[k]


# --------------------------------------------------------------------------- #
# input handling
# --------------------------------------------------------------------------- #

class TestProbabilityInputForms:
    def test_accepts_dict_keyed_by_class_names(self):
        d = {name: 1.0 / NUM_CLASSES for name in CLASS_NAMES}
        est = estimate_degradation(d)
        assert est.degradation_index == pytest.approx(0.5)  # uniform -> middle

    def test_unnormalised_vector_is_normalised_for_index(self):
        # Scaling the vector must not change the index (it is mass-weighted).
        v = np.array([0.2, 0.6, 0.4, 0.2])
        assert degradation_index_from_probs(v) == pytest.approx(
            degradation_index_from_probs(v * 10.0)
        )

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            estimate_degradation(np.array([0.5, 0.5]))

    def test_negative_prob_raises(self):
        with pytest.raises(ValueError):
            estimate_degradation(np.array([1.2, -0.2, 0.0, 0.0]))

    def test_missing_dict_key_raises(self):
        with pytest.raises(ValueError):
            estimate_degradation({"normal": 1.0})


# --------------------------------------------------------------------------- #
# conformal stage set: coverage + abstain
# --------------------------------------------------------------------------- #

def _synthetic_calib(n_per_class: int = 300, sharpness: float = 4.0, seed: int = 0):
    """Synthetic EXCHANGEABLE calibration probs + labels for 4 ordinal stages.

    For each true stage we draw a logit vector peaked at the true stage with
    Gaussian noise, softmax it -> a realistic, imperfect classifier. Calibration
    and a fresh test draw come from the SAME generator, so exchangeability holds
    and the marginal coverage guarantee is expected to bind.
    """
    rng = np.random.default_rng(seed)

    def draw(n):
        labels = rng.integers(0, NUM_CLASSES, size=n)
        logits = rng.normal(0.0, 1.0, size=(n, NUM_CLASSES))
        logits[np.arange(n), labels] += sharpness
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        return probs, labels

    return draw(n_per_class * NUM_CLASSES), draw(n_per_class * NUM_CLASSES)


class TestConformalStageSet:
    def test_set_covers_true_stage_at_target_coverage(self):
        (pc, lc), (pt, lt) = _synthetic_calib()
        alpha = 0.1
        cal = ConformalCalibrator(alpha=alpha, score="aps", rng_seed=0)
        cal.fit(pc, lc)
        hits = 0
        for i in range(len(lt)):
            est = estimate_degradation(pt[i], calibrator=cal)
            assert est.calibrated is True
            assert est.conformal_stage_set is not None
            if lt[i] in est.conformal_stage_set:
                hits += 1
        coverage = hits / len(lt)
        # Marginal coverage should be near 1 - alpha under exchangeability. Allow
        # finite-sample slack; the honest claim is "approximately 1 - alpha".
        assert coverage >= (1 - alpha) - 0.05, coverage

    def test_conformal_labels_align_with_set(self):
        (pc, lc), _ = _synthetic_calib()
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit(pc, lc)
        est = estimate_degradation(_onehot(2), calibrator=cal)
        assert est.conformal_stage_labels == [STAGE_LABELS[i] for i in est.conformal_stage_set]

    def test_ambiguous_set_triggers_abstain(self):
        # alpha very small -> large sets -> non-singleton -> abstain.
        (pc, lc), _ = _synthetic_calib(sharpness=0.3)  # weak classifier
        cal = ConformalCalibrator(alpha=0.001)
        cal.fit(pc, lc)
        flat = np.array([0.25, 0.25, 0.25, 0.25])
        est = estimate_degradation(flat, calibrator=cal)
        assert len(est.conformal_stage_set) > 1
        assert est.abstain is True
        assert "ambiguous" in (est.abstain_reason or "")

    def test_singleton_set_does_not_abstain(self):
        (pc, lc), _ = _synthetic_calib(sharpness=6.0)
        cal = ConformalCalibrator(alpha=0.2)
        cal.fit(pc, lc)
        est = estimate_degradation(_onehot(0), calibrator=cal)
        assert est.conformal_stage_set == [0]
        assert est.abstain is False

    def test_coverage_caveat_surfaced_when_calibrated(self):
        (pc, lc), _ = _synthetic_calib()
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit(pc, lc)
        est = estimate_degradation(_onehot(1), calibrator=cal)
        assert est.coverage_caveat is not None
        assert "exchangeab" in est.coverage_caveat.lower()

    def test_unfitted_calibrator_raises(self):
        cal = ConformalCalibrator(alpha=0.1)  # never fit
        with pytest.raises(ValueError):
            estimate_degradation(_onehot(0), calibrator=cal)


# --------------------------------------------------------------------------- #
# honest uncalibrated path + OOD abstain
# --------------------------------------------------------------------------- #

class TestHonestyAndAbstain:
    def test_uncalibrated_path_is_labelled(self):
        est = estimate_degradation(_onehot(1))
        assert est.calibrated is False
        assert est.conformal_stage_set is None
        assert est.conformal_stage_labels is None
        assert est.coverage_caveat is None
        assert est.abstain is False  # no fabricated abstain without a calibrator

    def test_disclaimer_always_present(self):
        est = estimate_degradation(_onehot(2))
        assert est.disclaimer == DEGRADATION_DISCLAIMER
        assert "not" in est.disclaimer.lower() and "rul" in est.disclaimer.lower()

    def test_disclaimer_never_claims_calibrated_rul(self):
        est = estimate_degradation(_onehot(3))
        low = est.disclaimer.lower()
        assert "time-to-failure" in low
        # Must not assert it IS a calibrated RUL / hours.
        assert "calibrated time-to-failure" not in low.replace("not a calibrated time-to-failure", "")

    def test_ood_flag_forces_abstain(self):
        est = estimate_degradation(_onehot(3), ood_flag=True, ood_reason="white noise")
        assert est.abstain is True
        assert est.abstain_reason == "white noise"

    def test_ood_abstain_wins_even_with_singleton_set(self):
        (pc, lc), _ = _synthetic_calib(sharpness=6.0)
        cal = ConformalCalibrator(alpha=0.2)
        cal.fit(pc, lc)
        est = estimate_degradation(_onehot(0), calibrator=cal, ood_flag=True)
        assert est.conformal_stage_set == [0]  # would otherwise NOT abstain
        assert est.abstain is True

    def test_to_dict_roundtrip(self):
        est = estimate_degradation(_onehot(1))
        d = est.to_dict()
        assert d["stage_ordinal"] == 1
        assert d["disclaimer"] == DEGRADATION_DISCLAIMER
        assert isinstance(d, dict)


# --------------------------------------------------------------------------- #
# InferenceEngine integration (additive, non-breaking)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def engine():
    from aion_nexus import InferenceEngine
    from aion_nexus.model import create_aion_nexus

    torch.manual_seed(0)
    return InferenceEngine(create_aion_nexus())


def _real_like_signal(seed: int = 0) -> np.ndarray:
    """A spectrally-structured 2-channel window that passes the plausibility gate."""
    rng = np.random.default_rng(seed)
    t = np.arange(2560) / 25_600.0
    base = (np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 300 * t))
    sig = np.stack([base, 0.8 * base + 0.1 * rng.standard_normal(2560)])
    return sig.astype(np.float32)


class TestEngineIntegration:
    def test_predict_degradation_populates_field(self, engine):
        result = engine.predict_degradation(_real_like_signal())
        assert result.degradation is not None
        assert isinstance(result.degradation, DegradationEstimate)
        assert 0.0 <= result.degradation.degradation_index <= 1.0
        # Same prediction fields as plain predict (additive output).
        assert result.predicted_class_name in CLASS_NAMES

    def test_predict_degradation_with_calibrator_adds_set(self, engine):
        (pc, lc), _ = _synthetic_calib()
        cal = ConformalCalibrator(alpha=0.1)
        cal.fit(pc, lc)
        result = engine.predict_degradation(_real_like_signal(), calibrator=cal)
        assert result.degradation.calibrated is True
        assert result.degradation.conformal_stage_set is not None

    def test_plain_predict_has_none_degradation(self, engine):
        # Backward-compat: predict()/predict_batch() must NOT populate degradation.
        result = engine.predict(_real_like_signal())
        assert result.degradation is None
        batch = engine.predict_batch([_real_like_signal()])
        assert batch[0].degradation is None

    def test_predict_degradation_ood_abstains(self, engine):
        noise = np.random.default_rng(1).standard_normal((2, 2560)).astype(np.float32)
        result = engine.predict_degradation(noise)
        # White noise should trip the plausibility gate -> degradation abstains.
        if result.ood_flag:
            assert result.degradation.abstain is True


# --------------------------------------------------------------------------- #
# Server endpoint + existing-API-intact
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def app():
    from server.main import app as _app
    return _app


@pytest.fixture(autouse=True)
def fresh_engine(app):
    from aion_nexus import InferenceEngine
    from aion_nexus.model import create_aion_nexus

    torch.manual_seed(0)
    saved = getattr(app.state, "engine", None)
    saved_err = getattr(app.state, "startup_error", None)
    app.state.engine = InferenceEngine(create_aion_nexus())
    app.state.startup_error = None
    yield
    app.state.engine = saved
    app.state.startup_error = saved_err


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestDegradationEndpoint:
    def test_endpoint_responds_200_with_degradation(self, client):
        sig = _real_like_signal().tolist()
        r = client.post("/predict_degradation", json={"signal": sig})
        assert r.status_code == 200, r.text
        body = r.json()
        # Inherits the full predict contract...
        assert body["predicted_class_name"] in CLASS_NAMES
        assert 0.0 <= body["confidence"] <= 1.0
        # ...plus the additive degradation object.
        deg = body["degradation"]
        assert 0 <= deg["stage_ordinal"] <= 3
        assert deg["stage_label"] in STAGE_LABELS
        assert 0.0 <= deg["degradation_index"] <= 1.0
        assert deg["disclaimer"] == DEGRADATION_DISCLAIMER
        # Served without a calibrator -> honest "not calibrated" point estimate.
        assert deg["calibrated"] is False
        assert deg["conformal_stage_set"] is None

    def test_endpoint_rejects_ragged_signal_400(self, client):
        r = client.post("/predict_degradation", json={"signal": [[1, 2, 3], [1]]})
        assert r.status_code == 400

    def test_endpoint_rejects_short_signal_400(self, client):
        r = client.post("/predict_degradation", json={"signal": [[0.0] * 10, [0.0] * 10]})
        assert r.status_code == 400

    def test_existing_predict_endpoint_unchanged(self, client):
        # The pre-existing /predict response must NOT gain a degradation field.
        sig = _real_like_signal().tolist()
        r = client.post("/predict", json={"signal": sig})
        assert r.status_code == 200, r.text
        assert "degradation" not in r.json()

    def test_health_still_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
