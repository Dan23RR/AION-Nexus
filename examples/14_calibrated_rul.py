"""Example 14: calibrated RUL with conformal intervals on real FEMTO (v2.19.0).

Unlike /predict_degradation (a coarse 4-class STAGE), this emits a TIME-TO-FAILURE
with a distribution-free conformal interval (Conformalized Quantile Regression).
It fits on real run-to-failure data, validates that the interval holds its target
coverage, persists the model the way the server loads it, and prints one estimate.

    python examples/14_calibrated_rul.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from aion_nexus.rul import ConformalRUL, load_rul
from scripts.eval_rul_femto import load_rul as load_femto_rul

_PKG = Path(__file__).resolve().parent.parent
_FEMTO = _PKG / "data" / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet"
HOUR = 3600.0


def main() -> int:
    if not _FEMTO.exists():
        print(f"FEMTO not found at {_FEMTO}; skipping.")
        return 0
    print("Loading real FEMTO run-to-failure + true RUL...")
    data = load_femto_rul(_FEMTO, per_bearing=150)
    x, y = data["features"], data["rul"]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n = len(y)
    fit_i, cal_i, te_i = perm[: n // 2], perm[n // 2: 3 * n // 4], perm[3 * n // 4:]

    model = ConformalRUL(alpha=0.1).fit(x[fit_i], y[fit_i]).calibrate(x[cal_i], y[cal_i])
    cov = model.coverage(x[te_i], y[te_i])
    print(f"\nIn-distribution conformal coverage: {cov['coverage']:.3f} "
          f"(target {cov['target']:.2f}); MAE {cov['mae']/HOUR:.2f} h, "
          f"mean interval {cov['mean_width']/HOUR:.2f} h")

    # persist + reload exactly as the server does (AION_RUL_ARTIFACT)
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "rul.joblib"
        model.save(art)
        reloaded = load_rul(art)
        e1 = model.predict_one(x[te_i[0]])
        e2 = reloaded.predict_one(x[te_i[0]])
        assert abs(e1.point - e2.point) < 1e-6 and abs(e1.upper - e2.upper) < 1e-6

    est = model.predict_one(x[te_i[0]])
    print(f"\nExample certified RUL: {est.point/HOUR:.2f} h  "
          f"[{est.lower/HOUR:.2f}, {est.upper/HOUR:.2f}] h @ 90% coverage")
    print(f"  caveat: {est.coverage_caveat[:80]}...")

    assert cov["coverage"] >= cov["target"] - 0.03, "conformal RUL must hold coverage"
    assert est.lower >= 0.0 and est.lower <= est.point <= est.upper
    print("\nOK — a calibrated time-to-failure with a coverage-valid interval, fit on "
          "real run-to-failure data and reloadable by the server (/predict_rul). The "
          "honest cross-bearing limit is reported by scripts/eval_rul_femto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
