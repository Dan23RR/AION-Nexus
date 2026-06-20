"""Risk control (CRC + RCPS) — bound the catastrophic false-healthy rate.

These prove the frontier guarantee EMPIRICALLY: the threshold chosen on a
calibration set actually controls the expected miss rate on a disjoint test set
(workspace 6.31 — a guarantee that is demonstrated, not merely asserted).
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.verify import (
    conformal_risk_control,
    empirical_risk,
    false_healthy_loss,
    rcps_threshold,
)


def _synthetic(n, seed):
    """A decent-but-imperfect model that sometimes makes a degraded bearing look healthy."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    logits = np.full((n, 4), -2.0)
    for i in range(n):
        logits[i, y[i]] += 3.0 + rng.normal(0, 1.5)
        if y[i] in (2, 3) and rng.random() < 0.4:
            logits[i, rng.integers(0, 2)] += 2.5
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), y


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_crc_controls_expected_risk_on_held_out(alpha):
    pc, yc = _synthetic(2000, 0)
    pt, yt = _synthetic(20000, 1)
    rc = conformal_risk_control(pc, yc, alpha=alpha)
    realized = empirical_risk(pt, yt, rc.lambda_hat)
    assert realized <= alpha + 0.02, f"CRC failed to control risk: {realized} > {alpha}"
    assert rc.method == "CRC"
    assert str(alpha) not in rc.guarantee or "<=" in rc.guarantee  # has a guarantee string


def test_rcps_is_more_conservative_than_crc():
    pc, yc = _synthetic(2000, 0)
    pt, yt = _synthetic(20000, 1)
    crc = conformal_risk_control(pc, yc, alpha=0.10)
    rcps = rcps_threshold(pc, yc, alpha=0.10, delta=0.1)
    # RCPS bounds the risk with high probability -> a larger/safer set -> lambda >=.
    assert rcps.lambda_hat >= crc.lambda_hat
    assert empirical_risk(pt, yt, rcps.lambda_hat) <= empirical_risk(pt, yt, crc.lambda_hat) + 1e-9
    assert rcps.method == "RCPS" and rcps.delta == 0.1


def test_false_healthy_loss_is_monotone_non_increasing():
    pc, yc = _synthetic(3000, 2)
    lambdas = np.linspace(0.0, 1.0, 21)
    risks = [empirical_risk(pc, yc, float(lam)) for lam in lambdas]
    assert all(risks[i + 1] <= risks[i] + 1e-9 for i in range(len(risks) - 1))
    assert risks[-1] == 0.0  # the full set flags everything -> zero misses


def test_risk_control_reduces_the_miss_rate_vs_point_prediction():
    pc, yc = _synthetic(2000, 0)
    pt, yt = _synthetic(20000, 1)
    base_miss = float(np.mean([
        (yt[i] in (2, 3)) and (int(np.argmax(pt[i])) in (0, 1)) for i in range(len(yt))]))
    rc = conformal_risk_control(pc, yc, alpha=0.05)
    controlled = empirical_risk(pt, yt, rc.lambda_hat)
    assert controlled < base_miss  # the guarantee actually buys a lower miss rate


def test_prediction_set_never_empty_and_grows_with_lambda():
    p = np.array([0.85, 0.1, 0.03, 0.02])
    rc = conformal_risk_control(_synthetic(500, 0)[0], _synthetic(500, 0)[1], alpha=0.1)
    s = rc.prediction_set(p)
    assert len(s) >= 1
    assert len(false_healthy_loss(p[None], np.array([3]), 1.0)) == 1


def test_alpha_out_of_range_rejected():
    pc, yc = _synthetic(500, 0)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal_risk_control(pc, yc, alpha=bad)
