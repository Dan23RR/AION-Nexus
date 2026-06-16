"""Example 8: CONDITIONAL conformal — past the marginal guarantee.

The base conformal layer gives MARGINAL coverage, valid only under exchangeability
— the red-team's kill-shot #5. This shows the four conditional calibrators turning
that weakness into the claimable frontier, each demonstrated against the failure it
fixes (the plain marginal calibrator under-covers; the conditional one does not).

    1. CLASS-CONDITIONAL — a rare, hard class is under-covered marginally; per-class
       CP restores its coverage.
    2. MONDRIAN — a hard operating regime is under-covered marginally; per-group CP
       restores per-group coverage.
    3. ACI (online) — on a drifting stream, a fixed gate drifts off target; ACI
       keeps the long-run miscoverage frequency near alpha WITHOUT exchangeability.

Runs with no checkpoint and no optional dependency (the verifier is pure numpy).
HONESTY: synthetic data, for demonstration. Each method still needs its own
assumption (within-class / within-group exchangeability; online label feedback).

Run:
    python examples/08_conditional_conformal.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus.verify import (
    AdaptiveConformalGate,
    ClassConditionalConformalCalibrator,
    ConformalCalibrator,
    MondrianConformalCalibrator,
    softmax,
)

ALPHA = 0.10
K = 4


def _prob_vector(rng, y, conf):
    conf = float(np.clip(conf, 1e-4, 1 - 1e-4))
    others = rng.dirichlet(np.ones(K - 1)) * (1 - conf)
    vec = np.empty(K)
    vec[y] = conf
    vec[[c for c in range(K) if c != y]] = others
    return vec


def _per_key_coverage(sets, labels, keys):
    out = {}
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        out[key] = float(np.mean([labels[i] in sets[i] for i in idx]))
    return out


def _fit_base(probs, labels):
    cal = ConformalCalibrator(alpha=ALPHA, score="lac")
    cal.fit(probs, labels)
    return cal


def main() -> int:
    rng = np.random.default_rng(0)

    # ---- 1. Class-conditional: rescue a rare, hard class ---------------------
    def gen_classwise(n):
        freqs = np.array([0.31, 0.31, 0.30, 0.08])           # class 3 rare
        beta = {0: (12, 2), 1: (12, 2), 2: (12, 2), 3: (1.5, 6)}  # class 3 hard
        y = rng.choice(K, size=n, p=freqs)
        p = np.array([_prob_vector(rng, c, rng.beta(*beta[c])) for c in y])
        return p, y

    p_cal, y_cal = gen_classwise(4000)
    p_te, y_te = gen_classwise(6000)
    cc = ClassConditionalConformalCalibrator(alpha=ALPHA).fit(p_cal, y_cal)
    base = _fit_base(p_cal, y_cal)
    cc_cov = _per_key_coverage(cc.predict_set(p_te), y_te, y_te)
    base_cov = _per_key_coverage(base.predict_set(p_te), y_te, y_te)
    print("--- 1. Class-conditional vs marginal (target coverage 0.90) ---")
    print(f"  class 3 (rare, hard): marginal={base_cov[3]:.2f}  "
          f"class-conditional={cc_cov[3]:.2f}")
    print(f"  per-class (class-conditional): "
          f"{ {int(c): round(v, 2) for c, v in cc_cov.items()} }")
    print("  -> marginal silently under-covers the rare hard class; per-class fixes it.")

    # ---- 2. Mondrian: rescue a hard operating regime -------------------------
    def gen_grouped(n):
        g = rng.integers(0, 2, size=n)                       # 0 easy, 1 hard regime
        y = rng.integers(0, K, size=n)
        conf = np.where(g == 0, rng.beta(12, 2, size=n), rng.beta(1.5, 6, size=n))
        p = np.array([_prob_vector(rng, y[i], conf[i]) for i in range(n)])
        return p, y, g

    p_cal, y_cal, g_cal = gen_grouped(5000)
    p_te, y_te, g_te = gen_grouped(6000)
    mon = MondrianConformalCalibrator(alpha=ALPHA, score="lac").fit(p_cal, y_cal, g_cal)
    mon_cov = _per_key_coverage(mon.predict_set(p_te, g_te), y_te, g_te)
    base2 = _fit_base(p_cal, y_cal)
    base2_cov = _per_key_coverage(base2.predict_set(p_te), y_te, g_te)
    print("\n--- 2. Mondrian (per-group) vs marginal ---")
    print(f"  hard regime (group 1): marginal={base2_cov[1]:.2f}  "
          f"per-group={mon_cov[1]:.2f}")
    print("  -> calibrate per regime/bearing; each group keeps its own guarantee.")

    # ---- 3. ACI: long-run coverage on a drifting stream ----------------------
    def gen_confusable(n, boost):
        y = rng.integers(0, K, size=n)
        logits = rng.standard_normal((n, K))
        logits[np.arange(n), y] += rng.normal(boost, 1.2, size=n)
        return softmax(logits), y

    p_cal, y_cal = gen_confusable(2500, 1.5)
    aci = AdaptiveConformalGate(alpha=ALPHA, gamma=0.05).fit(p_cal, y_cal)
    fixed = _fit_base(p_cal, y_cal)
    horizon, fixed_err = 6000, 0
    for t in range(horizon):
        boost = 2.5 - 3.0 * (t / horizon)                    # concept drift
        probs, yy = gen_confusable(1, boost)
        row, y = probs[0], int(yy[0])
        aci.step(row, true_label=y)
        if y not in fixed.predict(row[None]).sets[0]:
            fixed_err += 1
    print("\n--- 3. ACI (online) vs fixed gate on a DRIFTING stream (target miscov 0.10) ---")
    print(f"  fixed-gate miscoverage = {fixed_err / horizon:.3f}  (drifts off target)")
    print(f"  ACI miscoverage        = {aci.empirical_miscoverage:.3f}  "
          f"(tracks alpha; final alpha_t={aci.alpha_t:.3f})")
    print("  -> ACI holds long-run coverage with NO exchangeability assumption.")

    print("\nAll conditional methods deliver their guarantee where the marginal one "
          "fails. Each still carries its own honest assumption (see docs/CONFORMAL_ADVANCED.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
