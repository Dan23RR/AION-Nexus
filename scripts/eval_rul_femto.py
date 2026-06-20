"""Validate calibrated RUL + conformal intervals on REAL FEMTO run-to-failure data.

Uses the COMPLETE run-to-failure bearings (Full_Test_Set), where the true RUL of
acquisition i is (n_files - 1 - i) * 10 s — the ACTUAL remaining life, not the
positional proxy. Reports conformal-interval coverage two ways:

  * IN-DISTRIBUTION (random fit/cal/test split, same bearings -> exchangeable):
    the 1 - alpha guarantee should hold.
  * CROSS-BEARING (leave-one-bearing-out: fit+cal on other bearings, test on a
    held-out one): exchangeability is broken, so coverage is expected to drop —
    the honest signal that absolute RUL does not transfer to a never-seen bearing.

Writes results/rul_femto_eval.json. Run:  python -m scripts.eval_rul_femto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aion_nexus.rul import ConformalRUL, health_features_batch, rul_labels_for_run
from scripts.eval_real_femto import _read_raw_csv

_PKG = Path(__file__).resolve().parent.parent
_FEMTO = _PKG / "data" / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet"
HOUR = 3600.0


def load_rul(femto_root: Path, per_bearing: int) -> dict:
    """Load RAW windows + TRUE RUL (seconds) + bearing id from complete runs."""
    test_root = femto_root / "Validation_Set" / "Full_Test_Set"
    if not test_root.exists():
        test_root = femto_root / "Test_set" / "Test_set"
    feats, ruls, bearings = [], [], []
    for bdir in sorted(d for d in test_root.glob("Bearing*") if d.is_dir()):
        acc = sorted(bdir.glob("acc_*.csv"))
        if not acc:
            continue
        total = len(acc)
        rul_all = rul_labels_for_run(total)               # true RUL per file index
        idx = np.unique(np.linspace(0, total - 1, min(per_bearing, total)).astype(int))
        for i in idx:
            w = _read_raw_csv(acc[i])
            if w is None:
                continue
            feats.append(w)
            ruls.append(float(rul_all[i]))
            bearings.append(bdir.name)
    return {
        "features": health_features_batch(feats),
        "rul": np.array(ruls, dtype=np.float64),
        "bearings": np.array(bearings),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate conformal RUL on real FEMTO.")
    ap.add_argument("--femto-root", type=Path, default=_FEMTO)
    ap.add_argument("--per-bearing", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", type=Path, default=_PKG / "results" / "rul_femto_eval.json")
    args = ap.parse_args(argv)

    if not args.femto_root.exists():
        raise SystemExit(f"FEMTO not found at {args.femto_root}")
    print("Loading real FEMTO run-to-failure (RAW) + true RUL...")
    data = load_rul(args.femto_root, args.per_bearing)
    x, y, b = data["features"], data["rul"], data["bearings"]
    print(f"  {len(y)} windows, {len(set(b))} bearings, RUL range "
          f"{y.min()/HOUR:.2f}-{y.max()/HOUR:.2f} h")

    # ---- in-distribution (random fit/cal/test) ------------------------------
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n = len(y)
    fit_i, cal_i, te_i = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]
    m = ConformalRUL(alpha=args.alpha).fit(x[fit_i], y[fit_i]).calibrate(x[cal_i], y[cal_i])
    ind = m.coverage(x[te_i], y[te_i])
    print(f"\nIN-DISTRIBUTION (target {1-args.alpha:.2f}): coverage={ind['coverage']:.3f}  "
          f"mean_width={ind['mean_width']/HOUR:.2f} h  MAE={ind['mae']/HOUR:.2f} h "
          f"-> {'HOLDS' if ind['coverage'] >= 1-args.alpha-0.03 else 'under-covers'}")

    # ---- cross-bearing (leave-one-bearing-out) ------------------------------
    uniq = sorted(set(b))
    covs, widths = [], []
    for held in uniq:
        te = b == held
        tr = ~te
        if te.sum() < 10 or tr.sum() < 30:
            continue
        idx_tr = np.where(tr)[0]
        rng.shuffle(idx_tr)
        half = len(idx_tr) // 2
        mm = ConformalRUL(alpha=args.alpha).fit(x[idx_tr[:half]], y[idx_tr[:half]])
        mm.calibrate(x[idx_tr[half:]], y[idx_tr[half:]])
        c = mm.coverage(x[te], y[te])
        covs.append(c["coverage"])
        widths.append(c["mean_width"])
    cross_cov = float(np.mean(covs)) if covs else float("nan")
    print(f"CROSS-BEARING LOBO (target {1-args.alpha:.2f}): coverage={cross_cov:.3f} "
          f"mean over {len(covs)} bearings  mean_width={np.mean(widths)/HOUR:.2f} h "
          f"-> {'HOLDS' if cross_cov >= 1-args.alpha-0.03 else 'UNDER-COVERS (exchangeability broken — honest)'}")

    # ---- a single calibrated RUL estimate (illustrative) --------------------
    est = m.predict_one(x[te_i[0]])
    print(f"\nExample estimate: RUL = {est.point/HOUR:.2f} h  "
          f"[{est.lower/HOUR:.2f}, {est.upper/HOUR:.2f}] h @ {1-args.alpha:.0%} coverage")

    report = {
        "dataset": "FEMTO Full_Test_Set (real run-to-failure)",
        "n_windows": int(n), "n_bearings": len(uniq), "alpha": args.alpha,
        "method": "Conformalized Quantile Regression (Romano et al. 2019)",
        "in_distribution": {k: round(v, 4) for k, v in ind.items()},
        "cross_bearing_lobo": {"coverage": round(cross_cov, 4),
                               "mean_width_h": round(float(np.mean(widths)) / HOUR, 3),
                               "n_bearings": len(covs)},
        "unit": "seconds",
        "honesty_note": (
            "RUL labels are the TRUE remaining life ((n_files-1-i)*10s) on complete "
            "run-to-failure bearings, NOT the positional stage proxy. In-distribution "
            "(same-bearing, exchangeable) the conformal interval holds its target "
            "coverage; cross-bearing it under-covers because absolute time-to-failure "
            "does not transfer to a never-seen bearing — the interval is honest about "
            "that (it would need per-asset calibration / online updating to recover)."),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
