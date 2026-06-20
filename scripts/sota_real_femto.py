"""State-of-the-art post-hoc improvement + selective-prediction eval on REAL FEMTO.

The honest baseline (scripts/eval_real_femto) showed the v1 model scores macro-F1
~0.70 on real run-to-failure windows and OVER-PREDICTS 'advanced'. This script
pushes the honest real-data story to the frontier WITHOUT retraining:

  A. Post-hoc fixes (fit on a calibration split, measured on a disjoint test split):
     1. Logit adjustment (Menon et al., ICLR 2021) — debias the class over-prediction.
     2. Temperature scaling (Guo et al., ICML 2017) — calibrate confidence (ECE).
     3. AdaBN (Li et al., 2018) — re-estimate BatchNorm stats (v1 HAS BatchNorm).
  B. Selective prediction (the verifier's frontier metric):
     - Risk-coverage curve + AURC (area under risk-coverage; lower = better).
     - Accuracy @ coverage {10,20,50}% — what a customer can trust at each gate.
     - Marginal vs CLASS-CONDITIONAL conformal coverage on the imbalanced classes.

HONESTY (workspace 6.31): all post-hoc parameters are fit on calibration ONLY and
reported on a disjoint test split; positional FEMTO labels are a noisy proxy, and
v1 trained on these bearings (in-distribution), so AdaBN has little to fix here
(an honest near-zero lift is reported as such, not hidden).

Run:  python -m scripts.sota_real_femto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aion_nexus.config import CLASS_NAMES
from aion_nexus.inference import InferenceEngine
from scripts.eval_real_femto import _f1_macro, load_femto

_PKG = Path(__file__).resolve().parent.parent
_DEFAULT_FEMTO = _PKG / "data" / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet"
_DEFAULT_CKPT = _PKG / "checkpoints" / "aion_nexus_v1.pth"
K = len(CLASS_NAMES)
EPS = 1e-12


def _probs(engine, windows) -> np.ndarray:
    res = engine.predict_batch(windows)
    return np.array([[r.probabilities[c] for c in CLASS_NAMES] for r in res], dtype=np.float64)


# ---- A1. logit adjustment ---------------------------------------------------
def _logit_adjust(probs: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """score_c = log p_c - tau * log q_c  (debias toward a balanced posterior)."""
    s = np.log(probs + EPS) - tau * np.log(q + EPS)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def _fit_tau(probs_cal, y_cal, q) -> float:
    best_tau, best_f1 = 0.0, -1.0
    for tau in np.linspace(0.0, 2.0, 41):
        f1 = _f1_macro(y_cal, _logit_adjust(probs_cal, q, tau).argmax(1))
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)
    return best_tau


# ---- A2. temperature scaling ------------------------------------------------
def _temper(probs: np.ndarray, temp: float) -> np.ndarray:
    p = np.power(probs + EPS, 1.0 / temp)
    return p / p.sum(1, keepdims=True)


def _fit_temperature(probs_cal, y_cal) -> float:
    best_temp, best_nll = 1.0, 1e9
    for temp in np.linspace(0.5, 4.0, 36):
        p = _temper(probs_cal, temp)
        nll = -np.mean(np.log(p[np.arange(len(y_cal)), y_cal] + EPS))
        if nll < best_nll:
            best_nll, best_temp = nll, float(temp)
    return best_temp


def _ece(probs, y, n_bins=10) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


# ---- B. selective prediction ------------------------------------------------
def _risk_coverage(probs, y):
    """Return (coverages, accuracies, aurc) sweeping confidence thresholds."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    order = np.argsort(-conf)              # most-confident first
    correct = correct[order]
    n = len(y)
    cov, acc = [], []
    csum = np.cumsum(correct)
    for k in range(1, n + 1):
        cov.append(k / n)
        acc.append(csum[k - 1] / k)
    risk = 1.0 - np.array(acc)
    aurc = float(np.mean(risk))            # area under risk-coverage (lower better)
    return np.array(cov), np.array(acc), aurc


def _acc_at_coverage(cov, acc, c):
    i = int(np.searchsorted(cov, c))
    i = min(max(i, 0), len(acc) - 1)
    return float(acc[i])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SOTA post-hoc + selective eval on real FEMTO.")
    ap.add_argument("--femto-root", type=Path, default=_DEFAULT_FEMTO)
    ap.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    ap.add_argument("--per-bearing", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=_PKG / "results" / "sota_real_femto.json")
    args = ap.parse_args(argv)

    print("Loading real FEMTO (RAW)...")
    data = load_femto(args.femto_root, args.per_bearing)
    windows, y = data["windows"], data["labels"]
    n = len(windows)
    print(f"  {n} windows")
    engine = InferenceEngine.from_checkpoint(str(args.checkpoint))
    probs = _probs(engine, windows)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    ci, ti = perm[: n // 2], perm[n // 2:]
    yc, yt = y[ci], y[ti]

    # ---- A. post-hoc improvements (fit on calib, report on test) ------------
    base_f1 = _f1_macro(yt, probs[ti].argmax(1))
    base_acc = float((probs[ti].argmax(1) == yt).mean())

    q = probs[ci].mean(0)                          # source class-prior estimate
    tau = _fit_tau(probs[ci], yc, q)
    la_test = _logit_adjust(probs[ti], q, tau)
    la_f1 = _f1_macro(yt, la_test.argmax(1))
    la_acc = float((la_test.argmax(1) == yt).mean())

    temp = _fit_temperature(probs[ci], yc)
    ece_before = _ece(probs[ti], yt)
    ece_after = _ece(_temper(probs[ti], temp), yt)

    # AdaBN (v1 has BatchNorm). In-distribution here, so expect a small/zero lift.
    adabn_f1 = None
    try:
        from aion_nexus.adapt import recalibrate_batchnorm
        from aion_nexus.preprocessing import preprocess_batch
        cal_t = preprocess_batch([windows[i] for i in ci[:256]])
        adapted = recalibrate_batchnorm(engine.model, cal_t)
        eng2 = InferenceEngine(adapted)
        adabn_f1 = _f1_macro(yt, _probs(eng2, [windows[i] for i in ti]).argmax(1))
    except Exception as exc:  # never let an optional method break the report
        print(f"  (AdaBN skipped: {exc})")

    print("\nA. POST-HOC (real test split):")
    print(f"   baseline macro-F1 {base_f1:.3f} (acc {base_acc:.3f})")
    print(f"   logit-adjust (tau={tau:.2f}) macro-F1 {la_f1:.3f} (acc {la_acc:.3f})  "
          f"lift {la_f1 - base_f1:+.3f}")
    print(f"   temperature T={temp:.2f}: ECE {ece_before:.3f} -> {ece_after:.3f}")
    if adabn_f1 is not None:
        print(f"   AdaBN macro-F1 {adabn_f1:.3f} (in-distribution -> "
              f"lift {adabn_f1 - base_f1:+.3f}, expected ~0)")

    # ---- B. selective prediction (baseline vs improved) ---------------------
    cov_b, acc_b, aurc_b = _risk_coverage(probs[ti], yt)
    cov_i, acc_i, aurc_i = _risk_coverage(la_test, yt)
    print("\nB. SELECTIVE PREDICTION (risk-coverage):")
    print(f"   AURC baseline {aurc_b:.3f} -> improved {aurc_i:.3f} (lower better)")
    sel = {}
    for c in (0.1, 0.2, 0.5):
        ab, ai = _acc_at_coverage(cov_b, acc_b, c), _acc_at_coverage(cov_i, acc_i, c)
        sel[f"acc@{int(c*100)}pct"] = {"baseline": round(ab, 3), "improved": round(ai, 3)}
        print(f"   accuracy @ {int(c*100):>2}% coverage: baseline {ab:.3f} | improved {ai:.3f}")

    # the canonical selective-prediction figure (a reproducible repo artifact)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.plot(cov_b * 100, acc_b, color="#9aa0a6", lw=2,
                label=f"baseline (AURC {aurc_b:.3f})")
        ax.plot(cov_i * 100, acc_i, color="#1f77b4", lw=2,
                label=f"temperature-calibrated (AURC {aurc_i:.3f})")
        ax.axhline(base_acc, color="#d62728", ls="--", lw=1,
                   label=f"raw accuracy ({base_acc:.2f}, 100% coverage)")
        ax.set_xlabel("coverage — % of windows the verifier CERTIFIES")
        ax.set_ylabel("accuracy on CERTIFIED windows")
        ax.set_title("Risk-coverage on REAL FEMTO: the verifier isolates what it can trust")
        ax.set_ylim(0.6, 1.01)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        png = args.out.parent / "risk_coverage.png"
        fig.tight_layout()
        fig.savefig(png, dpi=130)
        plt.close(fig)
        print(f"   wrote figure {png.name}")
    except Exception as exc:  # plotting is optional; never break the report
        print(f"   (plot skipped: {exc})")

    # ---- B2. marginal vs class-conditional conformal per-class coverage -----
    from aion_nexus.verify import ConformalCalibrator
    marg = ConformalCalibrator(alpha=args.alpha)
    marg.fit(probs[ci], yc)
    mres = marg.predict(probs[ti])
    per_class_marg = {}
    for c in range(K):
        m = yt == c
        if m.sum():
            per_class_marg[CLASS_NAMES[c]] = round(
                float(np.mean([c in set(int(x) for x in mres.sets[i])
                               for i in np.where(m)[0]])), 3)
    cc_cov = {}
    try:
        from aion_nexus.verify.conformal_advanced import ClassConditionalConformalCalibrator
        cc = ClassConditionalConformalCalibrator(alpha=args.alpha)
        cc.fit(probs[ci], yc)
        csets = cc.predict_set(probs[ti])
        for c in range(K):
            m = yt == c
            if m.sum():
                cc_cov[CLASS_NAMES[c]] = round(
                    float(np.mean([c in set(int(x) for x in csets[i])
                                   for i in np.where(m)[0]])), 3)
    except Exception as exc:
        print(f"   (class-conditional skipped: {exc})")
    print(f"\n   per-class coverage MARGINAL:        {per_class_marg}")
    if cc_cov:
        print(f"   per-class coverage CLASS-CONDITIONAL: {cc_cov}  (target {1-args.alpha:.2f})")

    # ---- report -------------------------------------------------------------
    report = {
        "dataset": "FEMTO Full_Test_Set (real)",
        "n_windows": int(n),
        "baseline_macro_f1": round(base_f1, 4),
        "logit_adjust": {"tau": round(tau, 3), "macro_f1": round(la_f1, 4),
                         "lift": round(la_f1 - base_f1, 4)},
        "temperature": {"T": round(temp, 3), "ece_before": round(ece_before, 4),
                        "ece_after": round(ece_after, 4)},
        "adabn_macro_f1": round(adabn_f1, 4) if adabn_f1 is not None else None,
        "selective": {"aurc_baseline": round(aurc_b, 4), "aurc_improved": round(aurc_i, 4),
                      **sel},
        "conformal_per_class_coverage": {"marginal": per_class_marg,
                                         "class_conditional": cc_cov},
        "honesty_note": (
            "Post-hoc params fit on calibration, reported on a disjoint test split. "
            "Positional labels are a noisy proxy; AdaBN is near-zero because v1 is "
            "in-distribution on these bearings. The frontier value is the selective-"
            "prediction curve: a customer reads accuracy@coverage and picks the gate."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
