"""Continuous monitoring — rolling SLO + PSI drift over the decision stream (v2.21.0)."""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.monitoring import (
    PSI_MODERATE,
    Monitor,
    population_stability_index,
)


def test_psi_is_small_without_drift_and_large_with_drift():
    rng = np.random.default_rng(0)
    ref = rng.beta(5, 2, 1000)
    assert population_stability_index(ref, rng.beta(5, 2, 400)) < 0.1
    assert population_stability_index(ref, rng.beta(2, 5, 400)) > PSI_MODERATE


def test_psi_zero_on_tiny_samples():
    assert population_stability_index(np.array([0.5, 0.6]), np.array([0.5])) == 0.0


def test_monitor_status_rolling_rates():
    m = Monitor(window=100)
    for v, c in [("CERTIFIED", 0.95), ("REVIEW", 0.6), ("ABSTAIN", 0.3), ("CERTIFIED", 0.9)]:
        m.record(c, v)
    s = m.status()
    assert s["n"] == 4
    assert s["certified_rate"] == 0.5 and s["review_rate"] == 0.25 and s["abstain_rate"] == 0.25
    assert abs(s["mean_confidence"] - 0.6875) < 1e-6


def test_monitor_flags_significant_drift():
    rng = np.random.default_rng(1)
    m = Monitor(window=500, reference_confidence=rng.beta(5, 2, 1000))
    for c in rng.beta(2, 5, 400):  # drifted low-confidence stream
        m.record(float(c), "ABSTAIN" if c < 0.5 else "REVIEW")
    s = m.status()
    assert s["drift_level"] == "significant"
    assert any("drift" in a.lower() for a in s["alerts"])


def test_monitor_window_is_bounded():
    m = Monitor(window=50)
    for _ in range(200):
        m.record(0.9, "CERTIFIED")
    assert len(m) == 50


def test_realized_metrics_with_delayed_labels():
    m = Monitor()
    r = m.realized_metrics(np.array([2, 3, 0, 2]), np.array([0, 3, 0, 1]))
    assert r["accuracy"] == 0.5
    assert r["false_healthy_rate"] == 0.5  # 2 degraded bearings called healthy of 4


def test_empty_monitor_and_bad_window():
    assert Monitor().status()["n"] == 0
    with pytest.raises(ValueError):
        Monitor(window=0)
