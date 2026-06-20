"""Build a conformal-calibration artifact for certified serving (v2.16.0).

This is the bridge that turns the served certificate's coverage number from a
synthetic PLACEHOLDER into a REAL, leakage-checked guarantee — the concrete fix
for "the verification layer is a library, not a served product".

Two modes:

  Demo (runnable now, no real data):
      python -m scripts.build_calibration --demo
  builds a CLEARLY-LABELLED synthetic-demo artifact (basis='synthetic-demo') from
  the loaded checkpoint, so the end-to-end pipeline and server run. It is NOT a
  real coverage guarantee and is labelled as such.

  Real (a deployer with labelled field data):
      python -m scripts.build_calibration --from-npz field.npz --train-groups-key train_groups
  where field.npz holds:
      signals      float array [M, 2, >=2560]  raw held-out CALIBRATION windows
      labels       int   array [M]              ground-truth class per window
      groups       array [M]                    bearing/recording/machine id per window
      train_groups array [T]                    the group ids used in TRAINING
  The engine is run on the windows to produce probabilities, the calibration
  groups are proven DISJOINT from the training groups (the leakage gate — a leaked
  split is REFUSED), and a basis='real-holdout' artifact is written.

Output defaults to checkpoints/calibration_v1.npz (what the server loads by
default; override with --out or, at serve time, AION_CALIBRATION_NPZ).
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import numpy as np

from aion_nexus import InferenceEngine
from aion_nexus.config import CLASS_NAMES
from aion_nexus.serving_calibration import (
    BASIS_DEMO,
    BASIS_REAL,
    save_calibration,
    synthetic_demo_probs,
)

_PKG_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CKPT = _PKG_ROOT / "checkpoints" / "aion_nexus_v1.pth"
_DEFAULT_OUT = _PKG_ROOT / "checkpoints" / "calibration_v1.npz"


def _engine(checkpoint: Path) -> InferenceEngine:
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    return InferenceEngine.from_checkpoint(str(checkpoint))


def _probs_for(engine: InferenceEngine, signal: np.ndarray) -> np.ndarray:
    res = engine.predict(signal)
    return np.array([res.probabilities[n] for n in CLASS_NAMES], dtype=np.float64)


def build_demo(engine: InferenceEngine, out: Path, *, per_class: int = 8) -> dict:
    probs, labels = synthetic_demo_probs(
        lambda sig: _probs_for(engine, sig), len(CLASS_NAMES), per_class=per_class)
    return save_calibration(
        out, probs, labels, CLASS_NAMES,
        basis=BASIS_DEMO,
        source="scripts.build_calibration --demo (synthetic, NOT real coverage)",
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )


def build_real(engine: InferenceEngine, field_npz: Path, out: Path) -> dict:
    with np.load(field_npz, allow_pickle=False) as data:
        for key in ("signals", "labels", "groups", "train_groups"):
            if key not in data:
                raise SystemExit(f"{field_npz} is missing required array '{key}'")
        signals = np.asarray(data["signals"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int64)
        groups = [str(g) for g in data["groups"]]
        train_groups = [str(g) for g in data["train_groups"]]
    if signals.ndim != 3:
        raise SystemExit(f"signals must be [M, 2, N]; got shape {signals.shape}")
    probs = np.vstack([_probs_for(engine, sig) for sig in signals])
    # save_calibration runs the leakage gate and REFUSES a leaked real artifact.
    return save_calibration(
        out, probs, labels, CLASS_NAMES,
        basis=BASIS_REAL,
        train_groups=train_groups,
        calib_groups=groups,
        source=f"scripts.build_calibration --from-npz {field_npz.name}",
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a conformal-calibration artifact.")
    ap.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="synthetic-demo artifact (runnable, NOT real coverage)")
    mode.add_argument("--from-npz", type=Path, metavar="FIELD_NPZ",
                      help="real held-out field data (signals/labels/groups/train_groups)")
    ap.add_argument("--per-class", type=int, default=8, help="demo windows per class")
    args = ap.parse_args(argv)

    engine = _engine(args.checkpoint)
    if args.demo:
        meta = build_demo(engine, args.out, per_class=args.per_class)
    else:
        meta = build_real(engine, args.from_npz, args.out)

    print(f"Wrote calibration artifact -> {args.out}")
    print(f"  basis            : {meta['basis']}")
    print(f"  n / classes      : {meta['n']} / {meta['n_classes']}")
    print(f"  leakage_checked  : {meta['leakage_checked']} (disjoint={meta['disjoint']})")
    if meta["basis"] != BASIS_REAL:
        print("  NOTE: this is NOT a real coverage guarantee — the served certificate "
              "will report coverage_basis accordingly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
