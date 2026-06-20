"""Calibrated RUL with conformal intervals (CQR) — proven, not asserted.

The headline test shows the conformal interval HOLDS its target coverage on a
held-out set (workspace 6.31: a guarantee demonstrated empirically), plus the API
contracts (non-negative RUL, point inside the interval, true run-to-failure labels).
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.rul import (
    ConformalRUL,
    RULEstimate,
    health_features,
    rul_labels_for_run,
)


def _synthetic(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (n, 5))
    base = 1000 * (1 - x[:, 0]) + 300 * x[:, 1] - 200 * x[:, 2]
    y = np.clip(base + rng.normal(0, 50 + 150 * x[:, 0], n), 0, None)
    return x, y


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_cqr_covers_at_target_in_distribution(alpha):
    xf, yf = _synthetic(2000, 0)
    xc, yc = _synthetic(1500, 1)
    xt, yt = _synthetic(5000, 2)
    m = ConformalRUL(alpha=alpha).fit(xf, yf).calibrate(xc, yc)
    cov = m.coverage(xt, yt)
    assert cov["coverage"] >= (1 - alpha) - 0.03, cov
    assert cov["mean_width"] > 0


def test_rul_is_non_negative_and_point_in_interval():
    xf, yf = _synthetic(1500, 0)
    xc, yc = _synthetic(800, 1)
    m = ConformalRUL(alpha=0.1).fit(xf, yf).calibrate(xc, yc)
    for e in m.predict(_synthetic(200, 3)[0]):
        assert e.lower >= 0.0
        assert e.lower <= e.point <= e.upper
        assert isinstance(e, RULEstimate)


def test_rul_labels_are_true_remaining_life():
    # FEMTO: one acquisition = 10 s; the last file is failure (RUL 0).
    np.testing.assert_array_equal(rul_labels_for_run(5), [40.0, 30.0, 20.0, 10.0, 0.0])
    assert rul_labels_for_run(1).tolist() == [0.0]


def test_health_features_deterministic_shape():
    sig = np.random.default_rng(0).standard_normal((2, 2560))
    f1 = health_features(sig)
    f2 = health_features(sig)
    assert f1.shape == (14,)  # 7 features x 2 channels
    np.testing.assert_allclose(f1, f2)


def test_predict_before_calibrate_uses_no_correction():
    xf, yf = _synthetic(800, 0)
    m = ConformalRUL(alpha=0.1).fit(xf, yf)  # fit but not calibrate
    est = m.predict_one(xf[0])
    assert est.upper >= est.lower >= 0.0  # raw quantile interval, still valid shape


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        ConformalRUL().predict_one(np.zeros(5))


def test_alpha_out_of_range_rejected():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            ConformalRUL(alpha=bad)


def test_estimate_as_dict_carries_caveat():
    xf, yf = _synthetic(800, 0)
    m = ConformalRUL(alpha=0.1).fit(xf, yf).calibrate(*_synthetic(400, 1))
    d = m.predict_one(xf[0]).as_dict()
    assert {"point", "lower", "upper", "width", "alpha", "unit", "coverage_caveat"} <= d.keys()
    assert "exchangeability" in d["coverage_caveat"].lower()
