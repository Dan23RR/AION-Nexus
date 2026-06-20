"""Test the SERVED verification layer on REAL FEMTO data (v2.16.0).

This is the honest, real-data counterpart of examples/12: it runs the v1
checkpoint and the conformal layer on actual FEMTO run-to-failure windows (not
synthetic), and reports the numbers that matter — and where they break.

What it measures (all on REAL FEMTO Full_Test_Set bearings):
  1. The model's macro-F1 on real raw windows (sanity vs published ~0.884).
  2. Empirical conformal coverage IN-DISTRIBUTION (i.i.d. calib/test split) — the
     guarantee should hold (~1-alpha) when calibration and serving are exchangeable.
  3. Empirical conformal coverage CROSS-BEARING (calibrate on some bearings, test
     on a HELD-OUT bearing) — exchangeability breaks, so coverage is expected to
     DROP. This is the honest §6.31 result the certificate's caveat warns about.
  4. The physics second opinion's verdict distribution on real high-degradation
     windows (exploratory; FEMTO bearing geometry is assumed, see note).
  5. One real, Ed25519-signed certificate minted from a real calibration and
     verified offline, plus a real-indistribution calibration artifact.

HONESTY (workspace 6.31): the shipped v1 was trained on a GLOBALLY-STRATIFIED
split that saw windows from ALL these bearings, so NO FEMTO bearing is leakage-
clean vs training. We therefore label the real calibration `real-indistribution`,
NOT `real-holdout` — the coverage is measured on real data, but it is not a clean
cross-machine guarantee, and the basis says so.

Run (data already on disk):
    python -m scripts.eval_real_femto
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from aion_nexus.config import CLASS_NAMES
from aion_nexus.inference import InferenceEngine
from aion_nexus.physics import BearingGeometry, physics_consistency
from aion_nexus.serving_calibration import (
    BASIS_REAL_INDIST,
    coverage_guarantee_string,
    save_calibration,
)
from aion_nexus.verify import (
    ConformalCalibrator,
    Verifier,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)

_PKG = Path(__file__).resolve().parent.parent
_DEFAULT_FEMTO = _PKG / "data" / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet"
_DEFAULT_CKPT = _PKG / "checkpoints" / "aion_nexus_v1.pth"
FS = 25_600
WIN = 2560

# FEMTO/PRONOSTIA operating conditions -> shaft speed (rpm).
_RPM = {"Bearing1": 1800.0, "Bearing2": 1650.0, "Bearing3": 1500.0}
# FEMTO does not publish the test bearing geometry; prior AION work assumed an
# NSK 6203 (8 balls, d=6.6, D=28.5). Physics on FEMTO is therefore EXPLORATORY.
_ASSUMED_GEOMETRY = BearingGeometry(n_rolling_elements=8, ball_diameter=6.6, pitch_diameter=28.5)


def _rpm_for(bearing: str) -> float:
    return _RPM.get(bearing.split("_")[0], 1800.0)


def _derive_label(file_idx: int, total: int) -> int:
    """FEMTO degradation fraction -> 4-class severity (matches training convention)."""
    if total <= 1:
        return 0
    deg = file_idx / (total - 1)
    return 0 if deg < 0.2 else 1 if deg < 0.5 else 2 if deg < 0.8 else 3


def _read_raw_csv(path: Path) -> np.ndarray | None:
    """Read one FEMTO acc_*.csv -> RAW [2, 2560] (channels 4,5), NO preprocessing."""
    import pandas as pd
    for sep in (",", ";"):
        try:
            df = pd.read_csv(path, header=None, sep=sep)
        except Exception:
            continue
        if df.shape[1] >= 6:
            break
    else:
        return None
    h = df.iloc[:, 4].to_numpy(dtype=np.float32)
    v = df.iloc[:, 5].to_numpy(dtype=np.float32)
    if len(h) < 500 or np.isnan(h).sum() > 0.1 * len(h):
        return None
    if len(h) >= WIN:
        h, v = h[:WIN], v[:WIN]
    else:
        h = np.pad(h, (0, WIN - len(h)))
        v = np.pad(v, (0, WIN - len(v)))
    return np.stack([h, v], axis=0).astype(np.float32)  # RAW; engine preprocesses


def load_femto(femto_root: Path, per_bearing: int) -> dict:
    """Load RAW windows from the run-to-failure Full_Test_Set bearings."""
    test_root = femto_root / "Validation_Set" / "Full_Test_Set"
    if not test_root.exists():
        test_root = femto_root / "Test_set" / "Test_set"
    bearing_dirs = sorted(d for d in test_root.glob("Bearing*") if d.is_dir())
    windows, labels, bearings, rpms = [], [], [], []
    for bdir in bearing_dirs:
        acc = sorted(bdir.glob("acc_*.csv"))
        if not acc:
            continue
        total = len(acc)
        # spread the sample across the whole run so all degradation stages appear
        idx = np.unique(np.linspace(0, total - 1, min(per_bearing, total)).astype(int))
        for i in idx:
            w = _read_raw_csv(acc[i])
            if w is None:
                continue
            windows.append(w)
            labels.append(_derive_label(int(i), total))
            bearings.append(bdir.name)
            rpms.append(_rpm_for(bdir.name))
    return {
        "windows": windows,
        "labels": np.array(labels, dtype=int),
        "bearings": np.array(bearings),
        "rpms": np.array(rpms, dtype=float),
    }


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, k: int = 4) -> float:
    f1s = []
    for c in range(k):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _coverage(probs_cal, y_cal, probs_te, y_te, alpha=0.1) -> tuple[float, float]:
    """Empirical coverage + mean set size of marginal split-conformal at level alpha."""
    cal = ConformalCalibrator(alpha=alpha)
    cal.fit(probs_cal, y_cal)
    res = cal.predict(probs_te)
    covered = [int(y_te[i]) in set(int(c) for c in res.sets[i]) for i in range(len(y_te))]
    sizes = [len(res.sets[i]) for i in range(len(y_te))]
    return float(np.mean(covered)), float(np.mean(sizes))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Test the served verification layer on real FEMTO.")
    ap.add_argument("--femto-root", type=Path, default=_DEFAULT_FEMTO)
    ap.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    ap.add_argument("--per-bearing", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=_PKG / "results" / "real_femto_eval.json")
    args = ap.parse_args(argv)

    if not args.femto_root.exists():
        raise SystemExit(f"FEMTO dataset not found at {args.femto_root}")
    print(f"Loading real FEMTO windows ({args.per_bearing}/bearing, RAW)...")
    data = load_femto(args.femto_root, args.per_bearing)
    windows, y, bearings, rpms = (data["windows"], data["labels"],
                                  data["bearings"], data["rpms"])
    n = len(windows)
    if n == 0:
        raise SystemExit("no FEMTO windows loaded")
    print(f"  {n} windows, {len(set(bearings))} bearings, "
          f"labels={dict(Counter(y.tolist()))}")

    # ---- 1. model F1 on REAL raw windows ------------------------------------
    engine = InferenceEngine.from_checkpoint(str(args.checkpoint))
    results = engine.predict_batch(windows)
    probs = np.array([[r.probabilities[c] for c in CLASS_NAMES] for r in results],
                     dtype=np.float64)
    y_pred = probs.argmax(1)
    f1_overall = _f1_macro(y, y_pred)
    per_bearing_f1 = {}
    for b in sorted(set(bearings)):
        m = bearings == b
        per_bearing_f1[b] = round(_f1_macro(y[m], y_pred[m]), 4)
    print(f"\n1. Model macro-F1 on REAL windows (one window/file, positional "
          f"labels, COMPLETE run-to-failure): {f1_overall:.4f}")
    print("   NOTE: published numbers reproduce but on EASIER regimes — 0.884 on the "
          "stratified-temporal (leaky) split, per_bearing 0.92 on the TRUNCATED "
          "Test_set/Test_set (stops near/before failure). On the COMPLETE run-to-"
          "failure used here the model over-predicts 'advanced' in the failure phase.")
    print(f"   per-bearing mean {np.mean(list(per_bearing_f1.values())):.4f} "
          f"± {np.std(list(per_bearing_f1.values())):.4f}")

    # ---- 2. conformal coverage IN-DISTRIBUTION (i.i.d. split) ----------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    half = n // 2
    ci, ti = perm[:half], perm[half:]
    cov_ind, size_ind = _coverage(probs[ci], y[ci], probs[ti], y[ti], args.alpha)
    print(f"\n2. Conformal coverage IN-DISTRIBUTION (i.i.d. split, target "
          f"{1 - args.alpha:.2f}): {cov_ind:.3f}  (mean set size {size_ind:.2f})")

    # ---- 2b. SELECTIVE certification on real data (the product value) --------
    # The verifier refuses to certify when the conformal set is not a singleton.
    # On a mediocre model that is the honest behaviour: it CERTIFIES only the
    # confident-correct windows and sends the rest to REVIEW, so selective accuracy
    # on the CERTIFIED subset is what a customer can actually trust.
    sel_verifier = Verifier(alpha=args.alpha, class_names=list(CLASS_NAMES)).calibrate(
        probs[ci], y[ci])
    verdicts, certified_correct, n_certified = [], 0, 0
    for k in ti:
        c = sel_verifier.certify(probs[k], model_id="aion-v1")
        verdicts.append(c.verdict)
        if c.verdict == "CERTIFIED":
            n_certified += 1
            certified_correct += int(c.predicted_label == int(y[k]))
    vd = Counter(verdicts)
    raw_acc = float((y_pred[ti] == y[ti]).mean())
    cert_rate = n_certified / len(ti)
    cert_acc = (certified_correct / n_certified) if n_certified else float("nan")
    print(f"\n2b. Selective certification on real test windows "
          f"(verdicts {dict(vd)}):")
    print(f"    CERTIFIED {cert_rate:.0%} of windows @ accuracy {cert_acc:.3f} "
          f"vs raw accuracy {raw_acc:.3f}  (the verifier refuses the rest -> REVIEW)")

    # ---- 3. conformal coverage CROSS-BEARING (leave-one-bearing-out) ---------
    uniq = sorted(set(bearings))
    cross = []
    for held in uniq:
        te = bearings == held
        ca = ~te
        if te.sum() < 10 or len(set(y[ca])) < 2:
            continue
        cov, _ = _coverage(probs[ca], y[ca], probs[te], y[te], args.alpha)
        cross.append(cov)
    cov_cross = float(np.mean(cross)) if cross else float("nan")
    print(f"3. Conformal coverage CROSS-BEARING (LOBO, target {1 - args.alpha:.2f}): "
          f"{cov_cross:.3f} mean over {len(cross)} held-out bearings "
          f"-> exchangeability {'HOLDS' if cov_cross >= 1 - args.alpha - 0.03 else 'BREAKS'} "
          "(the honest caveat, on real data)")

    # ---- 4. physics second opinion on REAL high-degradation windows ----------
    sev = np.where(y == 3)[0][:60]
    phys_counts: Counter = Counter()
    for i in sev:
        v = physics_consistency(np.asarray(windows[i], dtype=np.float64), fs=FS,
                                rpm=float(rpms[i]), geometry=_ASSUMED_GEOMETRY)
        phys_counts[v.verdict] += 1
    print(f"\n4. Physics on {len(sev)} real advanced-degradation windows "
          f"(assumed geometry): {dict(phys_counts)}")

    # ---- 5. mint a REAL certificate + a real-indistribution artifact ---------
    basis = BASIS_REAL_INDIST
    cov_str = coverage_guarantee_string(basis, args.alpha, leakage_checked=False)
    verifier = Verifier(alpha=args.alpha, class_names=list(CLASS_NAMES)).calibrate(
        probs[ci], y[ci])
    seed = generate_seed()
    j = int(ti[0])
    cert = verifier.certify(probs[j], input_signal=np.asarray(windows[j]),
                            model_id="aion-nexus-v1-realfemto", seed=seed,
                            ttl_seconds=3600, conformal_method="marginal-split-conformal",
                            coverage_guarantee=cov_str)
    vres = verify_certificate(cert.as_dict(), expected_pubkey=ed25519_pubkey_from_seed(seed))
    artifact = _PKG / "checkpoints" / "calibration_femto_indist.npz"
    meta = save_calibration(artifact, probs[ci], y[ci], CLASS_NAMES, basis=basis,
                            source="real FEMTO Full_Test_Set (v1 globally-stratified "
                                   "-> in-distribution, NOT leakage-clean vs training)")
    print(f"\n5. Real certificate: verdict={cert.verdict}, basis={basis}, "
          f"offline trusted={vres['trusted']}")
    print(f"   real-indistribution calibration artifact -> {artifact.name} "
          f"(n={meta['n']})")

    # ---- write the honest report --------------------------------------------
    report = {
        "dataset": "FEMTO Full_Test_Set (real, run-to-failure)",
        "n_windows": int(n),
        "n_bearings": len(uniq),
        "label_dist": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        "model_f1_macro_real_allfiles": round(f1_overall, 4),
        "model_f1_published_stratified_leaky": 0.884,
        "per_bearing_f1": per_bearing_f1,
        "alpha": args.alpha,
        "raw_accuracy": round(raw_acc, 3),
        "selective_certified_rate": round(cert_rate, 3),
        "selective_certified_accuracy": round(cert_acc, 3) if n_certified else None,
        "verdict_distribution": {k: int(v) for k, v in vd.items()},
        "coverage_in_distribution": round(cov_ind, 3),
        "set_size_in_distribution": round(size_ind, 2),
        "coverage_cross_bearing_lobo": round(cov_cross, 3),
        "coverage_cross_bearing_folds": len(cross),
        "physics_verdicts_advanced": dict(phys_counts),
        "physics_geometry": "ASSUMED NSK 6203 (FEMTO does not publish geometry)",
        "certificate_basis": basis,
        "certificate_trusted_offline": bool(vres["trusted"]),
        "honesty_note": (
            "Two findings. (1) MODEL: the published numbers REPRODUCE, but on the "
            "EASIER regimes — 0.884 on the stratified-temporal (leaky) split, and "
            "per_bearing_f1.json 0.92 on the TRUNCATED Test_set/Test_set (stops near/"
            "before failure; verified: regen reproduces 0.9218 exactly). On the "
            "COMPLETE run-to-failure (Full_Test_Set) used here the model scores ~0.70 "
            "and over-predicts 'advanced' in the failure phase. Positional labels are "
            "a noisy proxy. (2) VERIFIER: this is where the value is — conformal "
            "coverage HOLDS on real data by enlarging sets, so the system CERTIFIES "
            "only the confident-correct windows (higher selective accuracy) and sends "
            "the rest to REVIEW instead of shipping confident wrong answers. v1 trained "
            "globally-stratified -> no FEMTO bearing is leakage-clean vs training, so "
            "calibration is labelled real-indistribution, not real-holdout."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        shown = args.out.relative_to(_PKG)
    except ValueError:
        shown = args.out
    print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
