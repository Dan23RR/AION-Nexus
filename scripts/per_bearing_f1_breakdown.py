"""Per-bearing F1 macro breakdown of the production v1 checkpoint.

This is NOT a strict leave-one-bearing-out (LOBO) cross-validation — that
would require retraining the model 11 times, holding out one bearing each
time. This is a per-bearing breakdown of the EXISTING released checkpoint.

What it shows: the variance of F1 macro across the 11 run-to-failure
bearings in FEMTO Test_set. High variance = checkpoint generalizes
unevenly = stratified-random split likely inflated the headline F1=0.884.
Low variance = checkpoint generalizes evenly = headline F1 is honest.

This is the cheapest possible signal on the "is F1=0.884 inflated?"
question, runnable in minutes instead of hours.

Output:
  results/per_bearing_f1.json     — raw per-bearing F1 + summary stats
  results/per_bearing_f1.md       — human-readable table

Usage:
  python -m scripts.per_bearing_f1_breakdown \\
      --checkpoint checkpoints/aion_nexus_v1.pth \\
      --femto-root "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet" \\
      --out-dir results
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import mean, stdev

import numpy as np


_logger = logging.getLogger("aion_nexus.per_bearing")


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def load_signal_csv(path: Path) -> np.ndarray | None:
    """Load single FEMTO acc_*.csv with HP-filter + z-score (matches training)."""
    import pandas as pd
    try:
        data = pd.read_csv(path, header=None)
        n_cols = len(data.columns)
        if n_cols >= 6:
            h = data.iloc[:, 4].values.astype(np.float32)
            v = data.iloc[:, 5].values.astype(np.float32)
        elif n_cols >= 2:
            h = data.iloc[:, 0].values.astype(np.float32)
            v = data.iloc[:, 1].values.astype(np.float32)
        else:
            return None
        if len(h) < 500 or len(h) > 10000:
            return None
        if np.isnan(h).sum() > len(h) * 0.1:
            return None
        L = 2560
        if len(h) != L:
            if len(h) > L:
                h, v = h[:L], v[:L]
            else:
                h = np.pad(h, (0, L - len(h)), mode='constant')
                v = np.pad(v, (0, L - len(v)), mode='constant')
        h = (h - np.mean(h)) / (np.std(h) + 1e-8)
        v = (v - np.mean(v)) / (np.std(v) + 1e-8)
        try:
            from scipy import signal as scipy_signal
            sos = scipy_signal.butter(2, 1.0, 'highpass', fs=25600, output='sos')
            h = scipy_signal.sosfilt(sos, h).astype(np.float32)
            v = scipy_signal.sosfilt(sos, v).astype(np.float32)
        except Exception:
            h = (h - np.mean(h)).astype(np.float32)
            v = (v - np.mean(v)).astype(np.float32)
        return np.stack([h, v], axis=0).astype(np.float32)
    except Exception:
        return None


def derive_label(file_idx: int, total_files: int) -> int:
    """FEMTO degradation_pct → 4-class severity (matches training convention)."""
    if total_files <= 1:
        return 0
    deg = file_idx / (total_files - 1)
    if deg < 0.2:
        return 0  # normal
    if deg < 0.5:
        return 1  # early
    if deg < 0.8:
        return 2  # medium
    return 3  # advanced


def evaluate_one_bearing(engine, bearing_dir: Path, num_classes: int = 4) -> dict:
    """Load all acc_*.csv from one bearing folder, run inference, compute F1 macro."""
    acc_files = sorted(bearing_dir.glob("acc_*.csv"))
    if not acc_files:
        return {"bearing": bearing_dir.name, "status": "no_files", "n": 0}

    total = len(acc_files)
    sigs, labs = [], []
    for i, p in enumerate(acc_files):
        sig = load_signal_csv(p)
        if sig is None:
            continue
        sigs.append(sig)
        labs.append(derive_label(i, total))

    if not sigs:
        return {"bearing": bearing_dir.name, "status": "no_valid_samples", "n": 0}

    results = engine.predict_batch(sigs)
    y_pred = np.asarray([r.predicted_class_index for r in results])
    y_true = np.asarray(labs)
    f1 = f1_macro(y_true, y_pred, num_classes)
    acc = float((y_pred == y_true).mean())

    # Class distribution actually observed
    class_dist = [int((y_true == c).sum()) for c in range(num_classes)]
    pred_dist = [int((y_pred == c).sum()) for c in range(num_classes)]

    return {
        "bearing": bearing_dir.name,
        "status": "ok",
        "n": len(sigs),
        "n_files_total": total,
        "f1_macro": f1,
        "accuracy": acc,
        "true_class_distribution": class_dist,
        "pred_class_distribution": pred_dist,
    }


def write_report(results: list[dict], out_md: Path, out_json: Path, checkpoint: str):
    """Write human-readable Markdown + machine-readable JSON."""
    valid = [r for r in results if r.get("status") == "ok"]
    f1s = [r["f1_macro"] for r in valid]
    if f1s:
        f1_mean = mean(f1s)
        f1_std = stdev(f1s) if len(f1s) > 1 else 0.0
        f1_min = min(f1s)
        f1_max = max(f1s)
        f1_range = f1_max - f1_min
    else:
        f1_mean = f1_std = f1_min = f1_max = f1_range = 0.0

    summary = {
        "checkpoint": checkpoint,
        "n_bearings_evaluated": len(valid),
        "n_bearings_total": len(results),
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "f1_macro_min": f1_min,
        "f1_macro_max": f1_max,
        "f1_macro_range": f1_range,
        "per_bearing": results,
        "interpretation": (
            "High variance across bearings (std > 0.10 or range > 0.30) suggests the "
            "model generalizes UNEVENLY across the 11 run-to-failure bearings, which "
            "is consistent with the hypothesis that the stratified-random 80/20 split "
            "INFLATES the headline F1=0.884 by leaking bearing identity into both "
            "train and test. Low variance (std < 0.05) means the headline number is "
            "honest. This is NOT a true LOBO measurement (would require retraining); "
            "it is a per-bearing breakdown of the existing checkpoint."
        ),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    lines = []
    lines.append("# Per-bearing F1 macro breakdown — production v1 checkpoint\n")
    lines.append(f"**Checkpoint**: `{checkpoint}`\n")
    lines.append(f"**Bearings evaluated**: {len(valid)} / {len(results)}\n")
    lines.append(f"**Generated**: {Path().absolute()}\n")
    lines.append("\n## Headline summary\n")
    lines.append(f"- **F1 macro mean** across {len(valid)} bearings: **{f1_mean:.4f}**")
    lines.append(f"- **F1 macro std** across bearings: **{f1_std:.4f}**")
    lines.append(f"- **F1 macro range** (max - min): **{f1_range:.4f}** "
                 f"(min {f1_min:.4f} on `{[r['bearing'] for r in valid if r['f1_macro']==f1_min][0]}`, "
                 f"max {f1_max:.4f} on `{[r['bearing'] for r in valid if r['f1_macro']==f1_max][0]}`)")
    lines.append("\n## Interpretation\n")
    lines.append(summary["interpretation"])
    lines.append("\n## Per-bearing detail\n")
    lines.append("| Bearing | n samples | F1 macro | Accuracy | True class dist | Pred class dist |")
    lines.append("|---|---:|---:|---:|---|---|")
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r['bearing']} | — | — | — | — (status={r['status']}) | — |")
        else:
            lines.append(f"| {r['bearing']} | {r['n']} | **{r['f1_macro']:.4f}** | "
                         f"{r['accuracy']:.3f} | {r['true_class_distribution']} | "
                         f"{r['pred_class_distribution']} |")
    lines.append("\n## Honest framing for external use\n")
    lines.append("This number is the per-bearing F1 macro variance of the released v1 "
                 "checkpoint on the 11 FEMTO run-to-failure bearings, evaluated independently. "
                 "It is the cheapest available proxy for the question 'is F1=0.884 inflated "
                 "by stratified-random split data leakage?'. A true leave-one-bearing-out "
                 "(LOBO) F1 number would require retraining the model 11 times and is "
                 "scheduled for the next iteration.\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/aion_nexus_v1.pth")
    parser.add_argument("--femto-root", required=True,
                        help="Path to FEMTOBearingDataSet (containing Test_set/, Validation_Set/, etc.)")
    parser.add_argument("--bearing-subset", default="Test_set/Test_set",
                        help="Subdirectory containing the 11 run-to-failure bearing folders")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from aion_nexus import InferenceEngine

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        _logger.error("Checkpoint not found: %s", ckpt)
        return 2

    femto_root = Path(args.femto_root)
    bearing_parent = femto_root / args.bearing_subset
    if not bearing_parent.exists():
        _logger.error("Bearing subset path not found: %s", bearing_parent)
        return 2

    bearing_dirs = sorted(d for d in bearing_parent.iterdir() if d.is_dir())
    if not bearing_dirs:
        _logger.error("No bearing folders found under %s", bearing_parent)
        return 2

    _logger.info("Found %d bearings: %s", len(bearing_dirs),
                 [d.name for d in bearing_dirs])

    engine = InferenceEngine.from_checkpoint(ckpt, device=args.device)
    _logger.info("Loaded checkpoint: %s (architecture_version=%s)",
                 ckpt, getattr(engine, "architecture_version", "?"))

    results = []
    for i, bdir in enumerate(bearing_dirs):
        _logger.info("[%d/%d] Evaluating %s ...", i+1, len(bearing_dirs), bdir.name)
        r = evaluate_one_bearing(engine, bdir)
        results.append(r)
        if r.get("status") == "ok":
            _logger.info("  → F1=%.4f (n=%d)", r["f1_macro"], r["n"])
        else:
            _logger.info("  → SKIPPED (%s)", r.get("status"))

    out_md = Path(args.out_dir) / "per_bearing_f1.md"
    out_json = Path(args.out_dir) / "per_bearing_f1.json"
    write_report(results, out_md, out_json, str(ckpt))
    _logger.info("Wrote %s and %s", out_md, out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
