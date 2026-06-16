"""Empirical-coverage proofs for the conditional conformal calibrators.

These tests do not merely exercise the API — they SIMULATE coverage on synthetic
data and assert each calibrator delivers the guarantee it claims, AND that the
plain marginal calibrator FAILS the same scenario (the value proposition). All
generators are seeded, so the assertions are deterministic.

Scenarios:
- class-conditional: a rare, hard class is under-covered by marginal CP; class-
  conditional CP restores per-class coverage.
- Mondrian: a hard covariate group is under-covered by marginal CP; per-group CP
  restores per-group coverage.
- weighted: under covariate shift, unweighted CP under-covers; correctly-weighted
  CP recovers coverage.
- ACI: on a drifting (non-exchangeable) stream, a fixed-level gate drifts off
  target while ACI keeps the long-run miscoverage frequency near alpha.
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.verify import (
    AdaptiveConformalGate,
    ClassConditionalConformalCalibrator,
    ConformalCalibrator,
    MondrianConformalCalibrator,
    WeightedConformalCalibrator,
)

ALPHA = 0.10
TARGET = 1 - ALPHA
TOL = 0.035   # Monte-Carlo slack for empirical coverage


def _prob_vector(rng, true_class, conf, k):
    """A valid prob vector with p[true_class]=conf and the rest spread randomly."""
    conf = float(np.clip(conf, 1e-4, 1 - 1e-4))
    others = rng.dirichlet(np.ones(k - 1)) * (1 - conf)
    vec = np.empty(k)
    vec[true_class] = conf
    vec[[c for c in range(k) if c != true_class]] = others
    return vec


def _fit_base(probs, labels, score="lac"):
    """Construct + fit a base ConformalCalibrator (its fit() returns qhat, not self)."""
    cal = ConformalCalibrator(alpha=ALPHA, score=score)
    cal.fit(probs, labels)
    return cal


def _coverage(sets, labels):
    return float(np.mean([labels[i] in sets[i] for i in range(len(labels))]))


def _per_key_coverage(sets, labels, keys):
    out = {}
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        out[key] = float(np.mean([labels[i] in sets[i] for i in idx]))
    return out


# --------------------------------------------------------------------------- #
# 1. Class-conditional: rescue a rare, hard class from under-coverage
# --------------------------------------------------------------------------- #

def _gen_classwise(n, rng, k=4):
    freqs = np.array([0.31, 0.31, 0.30, 0.08])           # class 3 is rare
    beta = {0: (12, 2), 1: (12, 2), 2: (12, 2), 3: (1.5, 6)}  # class 3 is hard
    labels = rng.choice(k, size=n, p=freqs)
    probs = np.array([_prob_vector(rng, y, rng.beta(*beta[y]), k) for y in labels])
    return probs, labels


def test_class_conditional_covers_every_class_while_marginal_fails():
    rng = np.random.default_rng(7)
    p_cal, y_cal = _gen_classwise(4000, rng)
    p_te, y_te = _gen_classwise(6000, rng)

    cc = ClassConditionalConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    cc_sets = cc.predict_set(p_te)
    cc_cov = _per_key_coverage(cc_sets, y_te, y_te)

    base = _fit_base(p_cal, y_cal)
    base_sets = base.predict_set(p_te)
    base_cov = _per_key_coverage(base_sets, y_te, y_te)

    # Class-conditional: EVERY class meets the per-class guarantee.
    for c, cov in cc_cov.items():
        assert cov >= TARGET - TOL, f"class {c} under-covered by class-conditional: {cov:.3f}"
    # Marginal: the rare hard class (3) is badly under-covered (the failure it fixes).
    assert base_cov[3] < 0.5, f"expected marginal CP to fail class 3, got {base_cov[3]:.3f}"
    assert cc_cov[3] >= TARGET - TOL


# --------------------------------------------------------------------------- #
# 2. Mondrian: rescue a hard covariate group from under-coverage
# --------------------------------------------------------------------------- #

def _gen_grouped(n, rng, k=4):
    groups = rng.integers(0, 2, size=n)                  # 0 = easy regime, 1 = hard
    labels = rng.integers(0, k, size=n)
    conf = np.where(groups == 0, rng.beta(12, 2, size=n), rng.beta(1.5, 6, size=n))
    probs = np.array([_prob_vector(rng, labels[i], conf[i], k) for i in range(n)])
    return probs, labels, groups


def test_mondrian_covers_every_group_while_marginal_fails():
    rng = np.random.default_rng(11)
    p_cal, y_cal, g_cal = _gen_grouped(5000, rng)
    p_te, y_te, g_te = _gen_grouped(6000, rng)

    mon = MondrianConformalCalibrator(alpha=ALPHA, score="lac").fit(p_cal, y_cal, g_cal)
    mon_cov = _per_key_coverage(mon.predict_set(p_te, g_te), y_te, g_te)

    base = _fit_base(p_cal, y_cal)
    base_cov = _per_key_coverage(base.predict_set(p_te), y_te, g_te)

    for g, cov in mon_cov.items():
        assert cov >= TARGET - TOL, f"group {g} under-covered by Mondrian: {cov:.3f}"
    # Marginal under-covers the hard group 1.
    assert base_cov[1] < TARGET - 0.05, f"expected marginal to fail group 1: {base_cov[1]:.3f}"


def test_mondrian_flags_unseen_group_fallback():
    rng = np.random.default_rng(3)
    p_cal, y_cal, _ = _gen_grouped(800, rng)
    g_cal = np.zeros(len(y_cal), dtype=int)              # only group 0 in calibration
    mon = MondrianConformalCalibrator(alpha=ALPHA, score="lac").fit(p_cal, y_cal, g_cal)
    # Predict on a group never calibrated -> must fall back AND record it.
    mon.predict_set(p_cal[:5], np.full(5, 999))
    assert 999 in mon.fell_back_groups


# --------------------------------------------------------------------------- #
# 3. Weighted: recover coverage under covariate shift
# --------------------------------------------------------------------------- #

def _gen_shift(n, rng, p_hard, k=4):
    """Regime z in {0=easy,1=hard}; P(z=1)=p_hard. Returns probs, labels, z."""
    z = (rng.random(n) < p_hard).astype(int)
    labels = rng.integers(0, k, size=n)
    conf = np.where(z == 0, rng.beta(12, 2, size=n), rng.beta(1.5, 6, size=n))
    probs = np.array([_prob_vector(rng, labels[i], conf[i], k) for i in range(n)])
    return probs, labels, z


def test_weighted_recovers_coverage_under_covariate_shift():
    rng = np.random.default_rng(5)
    p_cal_hard, p_te_hard = 0.2, 0.8                     # calib easy-heavy, test hard-heavy
    p_cal, y_cal, z_cal = _gen_shift(3000, rng, p_cal_hard)
    p_te, y_te, z_te = _gen_shift(4000, rng, p_te_hard)

    # Likelihood ratio w(z) = P_test(z) / P_cal(z).
    w_of = {0: (1 - p_te_hard) / (1 - p_cal_hard), 1: p_te_hard / p_cal_hard}
    w_cal = np.array([w_of[z] for z in z_cal])
    w_te = np.array([w_of[z] for z in z_te])

    wcp = WeightedConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal, weight_calib=w_cal)
    cov_weighted = _coverage(wcp.predict_set(p_te, weight_test=w_te), y_te)
    # Honest unweighted baseline: a calibrator that ignores the shift entirely.
    unweighted = WeightedConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    cov_unweighted = _coverage(unweighted.predict_set(p_te, weight_test=1.0), y_te)

    # Unweighted under-covers under the shift; correctly-weighted recovers coverage.
    assert cov_unweighted < TARGET - 0.03, f"expected unweighted to under-cover: {cov_unweighted:.3f}"
    assert cov_weighted >= TARGET - TOL, f"weighted failed to recover coverage: {cov_weighted:.3f}"


def test_weighted_reduces_to_marginal_coverage_without_shift():
    # With w=1 and NO shift, weighted CP must still hit the marginal target.
    rng = np.random.default_rng(9)
    p_cal, y_cal, _ = _gen_shift(3000, rng, 0.5)
    p_te, y_te, _ = _gen_shift(4000, rng, 0.5)
    wcp = WeightedConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    cov = _coverage(wcp.predict_set(p_te, weight_test=1.0), y_te)
    assert cov >= TARGET - TOL


# --------------------------------------------------------------------------- #
# 4. ACI: long-run coverage on a non-stationary stream
# --------------------------------------------------------------------------- #

def _gen_confusable(n, rng, boost_mean, boost_sd=1.2, k=4):
    """A genuinely UNCERTAIN model: logits ~ N(0,1), true class boosted by
    N(boost_mean, sd). Lower boost -> more confusable -> coverage is binding (sets
    of size > 1 are needed), so the conformal level actually does work. A high-
    confidence model would make even top-1 cover ~99%, leaving nothing for the
    gate to adapt — not a meaningful ACI test."""
    from aion_nexus.verify import softmax
    y = rng.integers(0, k, size=n)
    logits = rng.standard_normal((n, k))
    logits[np.arange(n), y] += rng.normal(boost_mean, boost_sd, size=n)
    return softmax(logits), y


def test_aci_holds_long_run_coverage_on_drifting_stream_while_fixed_drifts():
    rng = np.random.default_rng(13)
    # Calibrate on an uncertain regime (top-1 covers ~65%, so coverage is binding).
    p_cal, y_cal = _gen_confusable(2500, rng, boost_mean=1.5)

    aci = AdaptiveConformalGate(alpha=ALPHA, gamma=0.05).fit(p_cal, y_cal)
    fixed = _fit_base(p_cal, y_cal)

    horizon = 6000
    fixed_err = 0
    for t in range(horizon):
        # Non-stationary drift: the true-class boost falls 2.5 -> -0.5 over the
        # stream (the model gets steadily worse — concept drift).
        boost = 2.5 - 3.0 * (t / horizon)
        probs, yy = _gen_confusable(1, rng, boost_mean=boost)
        row, y = probs[0], int(yy[0])
        aci.step(row, true_label=y)
        if y not in fixed.predict(row[None]).sets[0]:
            fixed_err += 1

    aci_miscov = aci.empirical_miscoverage
    fixed_miscov = fixed_err / horizon
    # ACI keeps the realised miscoverage frequency near alpha despite the drift...
    assert abs(aci_miscov - ALPHA) < 0.03, f"ACI miscoverage {aci_miscov:.3f} not near {ALPHA}"
    # ...while the fixed-level gate drifts off target (under-covers as it worsens).
    assert fixed_miscov > ALPHA + 0.03, f"expected fixed gate to drift: {fixed_miscov:.3f}"


def test_aci_no_feedback_does_not_adapt():
    rng = np.random.default_rng(1)
    k = 4
    y = rng.integers(0, k, size=500)
    conf = rng.beta(8, 2, size=500)
    p = np.array([_prob_vector(rng, y[i], conf[i], k) for i in range(500)])
    aci = AdaptiveConformalGate(alpha=ALPHA).fit(p, y)
    before = aci.alpha_t
    for row in p[:50]:
        aci.step(row)                                    # no true_label -> read-only
    assert aci.alpha_t == before
    assert aci.n_steps == 0


# --------------------------------------------------------------------------- #
# API / honesty surface
# --------------------------------------------------------------------------- #

def test_every_calibrator_exposes_its_guarantee_and_assumption():
    for cal in (ClassConditionalConformalCalibrator(), MondrianConformalCalibrator(),
                WeightedConformalCalibrator(), AdaptiveConformalGate()):
        assert cal.guarantee and isinstance(cal.guarantee, str)
        assert cal.coverage_valid_under and isinstance(cal.coverage_valid_under, str)


def test_sets_are_never_empty():
    rng = np.random.default_rng(2)
    p_cal, y_cal = _gen_classwise(1000, rng)
    p_te, _ = _gen_classwise(200, rng)
    cc = ClassConditionalConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    assert all(len(s) >= 1 for s in cc.predict_set(p_te))


def test_alpha_out_of_range_rejected():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ClassConditionalConformalCalibrator(alpha=bad).__post_init__()


# --------------------------------------------------------------------------- #
# 5. Deploy-time covariate-shift weights — ESTIMATED (not oracle) from features
# --------------------------------------------------------------------------- #

def _gen_feature_shift(n, rng, z_mean, k=4):
    """A 1-D covariate z (the shift axis) that also DRIVES difficulty: higher z =>
    lower true-class confidence. Calibration is low-z (easy), target is high-z
    (hard) — so vanilla CP under-covers the target and the weights must be inferred
    from z alone (no labels)."""
    z = rng.normal(z_mean, 1.0, size=n)
    labels = rng.integers(0, k, size=n)
    conf = 1.0 / (1.0 + np.exp(-(2.2 - 1.6 * z)))     # high z -> low confidence
    probs = np.array([_prob_vector(rng, labels[i], conf[i], k) for i in range(n)])
    return probs, labels, z.reshape(-1, 1)


def test_estimated_weights_recover_coverage_under_shift():
    from aion_nexus.verify import deploy_weighted_calibrator

    rng = np.random.default_rng(17)
    p_cal, y_cal, f_cal = _gen_feature_shift(3000, rng, z_mean=0.0)
    p_te, y_te, f_te = _gen_feature_shift(4000, rng, z_mean=1.4)   # covariate shift in z

    # Vanilla split conformal (ignores the shift) under-covers the target.
    base = WeightedConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    cov_vanilla = _coverage(base.predict_set(p_te, weight_test=1.0), y_te)

    # Weights ESTIMATED from unlabeled features (the deploy reality) recover coverage.
    cal, weight_fn = deploy_weighted_calibrator(p_cal, y_cal, f_cal, f_te, alpha=ALPHA)
    cov_est = _coverage(cal.predict_set(p_te, weight_test=weight_fn(f_te)), y_te)

    assert cov_vanilla < TARGET - 0.03, f"expected vanilla to under-cover: {cov_vanilla:.3f}"
    assert cov_est >= TARGET - TOL, f"estimated-weight CP failed to recover: {cov_est:.3f}"


def test_estimated_weights_are_near_uniform_without_shift():
    from aion_nexus.verify import estimate_covariate_shift_weights

    # Same distribution for cal and "target" -> no shift -> weights ~ uniform, so
    # the estimator must not invent a shift and break the no-shift case.
    rng = np.random.default_rng(23)
    _, _, f_cal = _gen_feature_shift(2500, rng, z_mean=0.0)
    _, _, f_tgt = _gen_feature_shift(2500, rng, z_mean=0.0)
    w_cal, _ = estimate_covariate_shift_weights(f_cal, f_tgt)
    # Coefficient of variation small => essentially uniform weights.
    assert w_cal.std() / w_cal.mean() < 0.35
