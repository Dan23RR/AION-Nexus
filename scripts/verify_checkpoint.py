"""Independent verification of AION-NEXUS published F1 numbers.

Reproduces the exact (train, val, test) split used during training so the
"test set" matches the 2792-sample subset on which F1=0.884 was reported.

Two paths:

A. PREFERRED — import the original training-time loader from the source
   repo (`aion_data.AIONDataset` + `create_stratified_temporal_splits`).
   Guarantees byte-equivalent preprocessing AND identical split
   (StratifiedShuffleSplit with random_state=42).

B. FALLBACK — directory scan with label-in-filename pattern. Useful for
   sanity checks but NOT bit-equivalent to training test set.

Plus a static `diagnose_checkpoint` step that runs without any data.

Usage:
    python -m scripts.verify_checkpoint \\
        --checkpoint checkpoints/aion_nexus_v1.pth \\
        --aion-data-repo ../DefinitiveAION/clean \\
        --femto-root data/FEMTO+Bearing/FEMTOBearingDataSet/Learning_set \\
        --metadata data/FEMTO+Bearing/metadata.json \\
        --mfpt data/mfpt
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

_logger = logging.getLogger("aion_nexus.verify")


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


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> list[list[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred, strict=False):
        cm[int(t)][int(p)] += 1
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> list[float]:
    out = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        out.append(f1)
    return out


# ---- Path A: use the original training loader ---------------------------

def evaluate_via_original_loader(engine, aion_data_repo: Path,
                                  femto_root: Path, metadata_path: Path,
                                  num_classes: int = 4) -> dict:
    """Reproduce the exact training-time test set and evaluate.

    The original `AIONDataset._process_csv_file` produces signals that are
    ALREADY z-scored AND HP-Butterworth-filtered. We bypass our own
    preprocessing and feed the dataset's signals directly to the model.
    """
    sys.path.insert(0, str(aion_data_repo))
    try:
        from aion_data import AIONDataset, TemporalConfig, create_stratified_temporal_splits
    except ImportError as exc:
        return {"path": "A", "status": "import_failed", "error": str(exc)}

    config = TemporalConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    _logger.info("Loading FEMTO via original AIONDataset (HP-filter + z-score applied inside)...")
    dataset = AIONDataset(data_root=str(femto_root), metadata_path=str(metadata_path))
    train_idx, val_idx, test_idx = create_stratified_temporal_splits(dataset, config)
    _logger.info("Splits: train=%d val=%d test=%d", len(train_idx), len(val_idx), len(test_idx))

    if not test_idx:
        return {"path": "A", "status": "empty_test", "n_samples": 0}

    import torch
    model = engine.model.eval()

    test_signals_pre, test_labels = [], []
    for idx in test_idx:
        signal, label, _ = dataset[idx]
        test_signals_pre.append(signal)
        test_labels.append(int(label.item() if hasattr(label, "item") else label))

    _logger.info("Inference on %d test samples (using already-preprocessed dataset signals)...",
                 len(test_signals_pre))
    preds = []
    BATCH = 64  # noqa: N806 — constant-style local, intentional
    with torch.no_grad():
        for i in range(0, len(test_signals_pre), BATCH):
            x = torch.stack(test_signals_pre[i:i + BATCH], dim=0)
            out = model(x)
            preds.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())

    y_true = np.asarray(test_labels)
    y_pred = np.asarray(preds)
    return {
        "path": "A_original_loader",
        "status": "ok",
        "n_samples": len(test_idx),
        "f1_macro": f1_macro(y_true, y_pred, num_classes),
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_f1": per_class_f1(y_true, y_pred, num_classes),
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes),
    }


# ---- Path B: sanity directory scan -------------------------------------

def evaluate_via_directory_scan(engine, root: Path, max_samples: int | None = None,
                                 num_classes: int = 4) -> dict:
    """Best-effort scan of `<name>_class<N>.csv` files; uses production preprocessing.

    NOT bit-equivalent to training test set; for canonical verification use Path A.
    """
    label_re = re.compile(r"_class(\d+)")
    sigs, labs = [], []
    for p in sorted(root.rglob("*.csv")):
        try:
            m = label_re.search(p.name)
            if not m:
                continue
            raw = np.loadtxt(p, delimiter=",")
            if raw.ndim != 2:
                continue
            n_cols = raw.shape[1]
            ci, cj = (4, 5) if n_cols >= 6 else (0, 1)
            sig = raw[:, [ci, cj]].T.astype(np.float32, copy=False)
            if sig.shape[1] < 2560:
                continue
            sigs.append(sig)
            labs.append(int(m.group(1)))
            if max_samples and len(sigs) >= max_samples:
                break
        except Exception:
            continue
    if not sigs:
        return {"path": "B", "status": "no_samples", "n_samples": 0}

    results = engine.predict_batch(sigs)
    y_pred = np.asarray([r.predicted_class_index for r in results])
    y_true = np.asarray(labs)
    return {
        "path": "B_directory_scan",
        "status": "ok",
        "n_samples": len(sigs),
        "f1_macro": f1_macro(y_true, y_pred, num_classes),
        "accuracy": float((y_pred == y_true).mean()),
    }


# ---- MFPT loader: read the real *.mat structure ------------------------

def evaluate_mfpt_mat(engine, root: Path, max_samples: int | None = None,
                      num_classes: int = 4) -> dict:
    """Load MFPT *.mat files and run the model on them as a *load/smoke* check.

    The MFPT distribution ships `.mat` files (not `.csv`). Each file holds a
    `bearing` struct with a single-channel signal `gs` (we duplicate it to the
    2 channels the model expects, per the FAQ Q25 workaround).

    HONESTY CAVEAT — why this is NOT a labelled F1 evaluation:
      MFPT filenames encode FAULT TYPE (baseline / InnerRaceFault /
      OuterRaceFault), but AION-NEXUS predicts a 4-class LIFE-STAGE / SEVERITY
      index (0=normal ... 3=advanced). Fault-type labels do NOT map to the
      severity taxonomy, so computing an "F1 vs the 4 severity classes" here
      would be meaningless. We therefore report the *prediction distribution*
      and prove the data flows end-to-end, but emit NO F1 and NO PASS. The
      historical MFPT zero-shot F1=0.615 used a separate windowing/selection
      recipe that is documented as lost (see docs/reproduce.md); it is NOT
      reproduced by this loader.
    """
    try:
        import scipy.io as sio
    except ImportError as exc:
        return {"path": "MFPT_mat", "status": "scipy_unavailable",
                "n_samples": 0, "error": str(exc)}

    mat_files = sorted(root.rglob("*.mat"))
    if not mat_files:
        return {"path": "MFPT_mat", "status": "no_mat_files", "n_samples": 0,
                "message": f"no *.mat under {root}"}

    sigs = []
    fault_types: list[str] = []
    WINDOW = 2560  # noqa: N806 — constant-style local, intentional
    for mf in mat_files:
        try:
            m = sio.loadmat(str(mf))
            if "bearing" not in m:
                continue
            bearing = m["bearing"]
            gs = np.asarray(bearing["gs"][0, 0]).astype(np.float32).reshape(-1)
            if gs.size < WINDOW:
                continue
            # One non-overlapping window per file keeps the smoke check fast.
            seg = gs[:WINDOW]
            # Single-channel -> duplicate to 2 channels (FAQ Q25 workaround).
            sig = np.stack([seg, seg], axis=0).astype(np.float32)
            sigs.append(sig)
            name = mf.name.lower()
            if name.startswith("baseline"):
                fault_types.append("baseline")
            elif "innerrace" in name:
                fault_types.append("inner_race")
            elif "outerrace" in name:
                fault_types.append("outer_race")
            else:
                fault_types.append("unknown")
            if max_samples and len(sigs) >= max_samples:
                break
        except Exception:
            continue

    if not sigs:
        return {"path": "MFPT_mat", "status": "no_samples", "n_samples": 0,
                "message": f"found {len(mat_files)} .mat files but none yielded a usable window"}

    results = engine.predict_batch(sigs)
    y_pred = [int(r.predicted_class_index) for r in results]
    pred_dist = [y_pred.count(c) for c in range(num_classes)]

    # status = "loaded_no_labels": data flows, but no severity ground truth.
    return {
        "path": "MFPT_mat",
        "status": "loaded_no_labels",
        "n_samples": len(sigs),
        "n_mat_files": len(mat_files),
        "severity_prediction_distribution": pred_dist,
        "fault_type_counts": {ft: fault_types.count(ft) for ft in sorted(set(fault_types))},
        "note": ("MFPT labels are fault-TYPE; model predicts severity-STAGE. "
                 "No F1 computed (label taxonomies differ). Data load verified."),
    }


# ---- Loader with HP-filter (matches training pipeline) ------------

def load_signal_csv_original(path: str | Path):
    """Load single CSV with HP-filter + z-score (matches training)."""
    import numpy as np
    import pandas as pd
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = pd.read_csv(path, header=None)
        n_cols = len(data.columns)

        # Handle any column count
        if n_cols >= 6:
            h = data.iloc[:, 4].values.astype(np.float32)
            v = data.iloc[:, 5].values.astype(np.float32)
        elif n_cols >= 2:
            h = data.iloc[:, 0].values.astype(np.float32)
            v = data.iloc[:, 1].values.astype(np.float32)
        elif n_cols == 1:
            val = data.iloc[:, 0].values.astype(np.float32)
            h = val.copy()
            v = val.copy()
        else:
            return None

        # Filter bad samples
        if len(h) < 500 or len(h) > 10000:
            return None
        if np.isnan(h).sum() > len(h) * 0.1:
            return None

        # Resize to 2560
        L = 2560  # noqa: N806 — constant-style local, intentional
        if len(h) != L:
            if len(h) > L:
                h, v = h[:L], v[:L]
            else:
                h = np.pad(h, (0, L - len(h)), mode='constant')
                v = np.pad(v, (0, L - len(v)), mode='constant')

        # Z-score normalize
        h = (h - np.mean(h)) / (np.std(h) + 1e-8)
        v = (v - np.mean(v)) / (np.std(v) + 1e-8)

        # HP filter (1 Hz)
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


# ---- FEMTO native: direct scan with HP-filter loader -----------------------------

def evaluate_femto_native(engine, femto_root: Path, max_samples: int | None = None,
                        num_classes: int = 4) -> dict:
    """Load FEMTO native structure with HP-filter preprocessing.

    Scans Validation_Set subdirectories (not training) and derives labels from
    degradation_pct = file_idx / (total_files - 1).
    """

    sigs, labs = [], []

    # Find bearing directories in Validation_Set
    val_base = femto_root / "10. FEMTO Bearing" / "FEMTOBearingDataSet" / "Validation_Set" / "Full_Test_Set"
    if not val_base.exists():
        return {"path": "FEMTO_native", "status": "path_not_found", "message": str(val_base)}

    for bearing_dir in sorted(val_base.iterdir()):
        if not bearing_dir.is_dir():
            continue

        acc_files = sorted(bearing_dir.glob("acc_*.csv"))
        if not acc_files:
            continue

        total_files = len(acc_files)

        for file_idx, csv_path in enumerate(acc_files):
            if max_samples and len(sigs) >= max_samples:
                break

            try:
                signal = load_signal_csv_original(csv_path)
                if signal is None:
                    continue

                # Derive label from degradation_pct
                degradation_pct = file_idx / (total_files - 1) if total_files > 1 else 0.0
                if degradation_pct < 0.2:
                    label = 0
                elif degradation_pct < 0.5:
                    label = 1
                elif degradation_pct < 0.8:
                    label = 2
                else:
                    label = 3

                sigs.append(signal)
                labs.append(label)

            except Exception:
                continue

        if max_samples and len(sigs) >= max_samples:
            break

    if not sigs:
        return {"path": "FEMTO_native", "status": "no_samples", "n_samples": 0}

    _logger.info("FEMTO_native: loaded %d samples", len(sigs))

    # Inferenza
    results = engine.predict_batch(sigs)
    y_pred = np.asarray([r.predicted_class_index for r in results])
    y_true = np.asarray(labs)

    return {
        "path": "FEMTO_native",
        "status": "ok",
        "n_samples": len(sigs),
        "f1_macro": f1_macro(y_true, y_pred, num_classes),
        "accuracy": float((y_pred == y_true).mean()),
        "per_class_f1": per_class_f1(y_true, y_pred, num_classes),
    }


# ---- Sanity diagnostics (no data needed) -------------------------------

def diagnose_checkpoint(engine) -> dict:
    """Check param count + class distribution on synthetic input.

    Picks expected param count based on architecture_version detected on the
    engine. v1 → 1,061,724; v6 → 716,577.
    """
    n_params = engine.model.get_num_params()
    arch_version = getattr(engine, "architecture_version", "v1")
    expected = 1_061_724 if arch_version == "v1" else 716_577
    rng = np.random.default_rng(0)
    preds = []
    for _ in range(50):
        sig = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
        preds.append(engine.predict(sig).predicted_class_index)
    counts = [preds.count(c) for c in range(4)]
    return {
        "architecture_version": arch_version,
        "param_count": n_params,
        "expected_param_count": expected,
        "param_match": n_params == expected,
        "synthetic_class_distribution": counts,
        "collapsed_to_one_class": max(counts) >= 45,
    }


# ---- Main --------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/aion_nexus_v1.pth")
    parser.add_argument("--aion-data-repo", default="../DefinitiveAION/clean",
                        help="Path to source training repo (must contain aion_data.py)")
    parser.add_argument("--femto-root", default=None,
                        help="FEMTO bearing data root for Path A")
    parser.add_argument("--metadata", default=None,
                        help="metadata.json path for Path A")
    parser.add_argument("--mfpt", default=None, help="MFPT data root (Path B)")
    parser.add_argument("--out", default="results/verification.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-original", action="store_true",
                        help="Skip Path A even if available")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick test (e.g., 500)")
    parser.add_argument("--femto", default=None,
                        help="FEMTO data directory (native structure with acc_*.csv)")
    args = parser.parse_args()

    from aion_nexus import InferenceEngine

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        _logger.error("Checkpoint not found: %s", ckpt)
        return 2

    engine = InferenceEngine.from_checkpoint(ckpt, device=args.device)

    summary = {
        "checkpoint": str(ckpt),
        "preprocessing_version": "1.0.2",
        "diagnostics": diagnose_checkpoint(engine),
        "evaluations": [],
    }
    _logger.info("Diagnostics: param_match=%s collapsed=%s dist=%s",
                 summary["diagnostics"]["param_match"],
                 summary["diagnostics"]["collapsed_to_one_class"],
                 summary["diagnostics"]["synthetic_class_distribution"])
    if summary["diagnostics"]["collapsed_to_one_class"]:
        _logger.error("Model predicts only one class on random input; checkpoint may be broken.")

    # Path A
    if not args.skip_original:
        repo = Path(args.aion_data_repo)
        femto_root = Path(args.femto_root) if args.femto_root else None
        metadata = Path(args.metadata) if args.metadata else None
        if (repo.exists() and (repo / "aion_data.py").exists()
                and femto_root and metadata
                and femto_root.exists() and metadata.exists()):
            _logger.info("=" * 70)
            _logger.info("Path A: original training-time loader")
            _logger.info("=" * 70)
            ev = evaluate_via_original_loader(engine, repo, femto_root, metadata)
            summary["evaluations"].append(ev)
            if ev.get("status") == "ok":
                _logger.info("Path A: F1=%.4f Accuracy=%.4f n=%d",
                             ev["f1_macro"], ev["accuracy"], ev["n_samples"])
                _logger.info("Per-class F1: %s", ev["per_class_f1"])
        else:
            _logger.warning("Path A skipped: aion-data-repo / femto-root / metadata unavailable")

    # MFPT: the real distribution ships *.mat. Try the .mat loader first; if a
    # custom labelled *_classN.csv tree is present, also do the directory scan.
    if args.mfpt:
        mfpt = Path(args.mfpt)
        if not mfpt.exists():
            _logger.warning("MFPT path does not exist: %s", mfpt)
        else:
            mat_files = list(mfpt.rglob("*.mat"))
            csv_labelled = any(re.search(r"_class\d+", p.name) for p in mfpt.rglob("*.csv"))
            ran_any = False
            if mat_files:
                ev = evaluate_mfpt_mat(engine, mfpt, max_samples=args.max_samples)
                ev["name"] = "MFPT_mat_load"
                summary["evaluations"].append(ev)
                ran_any = True
                if ev.get("status") == "loaded_no_labels":
                    _logger.info("MFPT .mat load OK: n=%d (no F1 — fault-type labels != severity classes). "
                                 "Pred dist=%s", ev["n_samples"],
                                 ev.get("severity_prediction_distribution"))
                else:
                    _logger.warning("MFPT .mat load status=%s: %s",
                                    ev.get("status"), ev.get("message", ""))
            if csv_labelled:
                ev = evaluate_via_directory_scan(engine, mfpt, max_samples=args.max_samples)
                ev["name"] = "MFPT_zero_shot"
                summary["evaluations"].append(ev)
                ran_any = True
                if ev.get("status") == "ok":
                    _logger.info("MFPT zero-shot: F1=%.4f n=%d", ev["f1_macro"], ev["n_samples"])
            if not ran_any:
                _logger.warning("MFPT: neither *.mat nor labelled *_classN.csv found under %s", mfpt)

    # FEMTO native (direct scan with HP-filter loader)
    if args.femto:
        femto = Path(args.femto)
        if femto.exists():
            ev = evaluate_femto_native(engine, femto, max_samples=args.max_samples)
            ev["name"] = "FEMTO_test"
            summary["evaluations"].append(ev)
            if ev.get("status") == "ok":
                _logger.info("FEMTO_test: F1=%.4f Accuracy=%.4f n=%d",
                             ev["f1_macro"], ev["accuracy"], ev["n_samples"])

    # Published numbers depend on architecture version
    arch = summary["diagnostics"].get("architecture_version", "v1")
    if arch == "v6":
        PUBLISHED = {  # noqa: N806 — constant-style local, intentional
            "A_original_loader": {"f1": 0.9343, "tol": 0.01,
                                   "label": "FEMTO test (training split, v6 ULTRA)"},
            "MFPT_zero_shot":    {"f1": 0.615, "tol": 0.02,
                                   "label": "MFPT zero-shot"},
        }
    else:
        PUBLISHED = {  # noqa: N806 — constant-style local, intentional
            "A_original_loader": {"f1": 0.884, "tol": 0.01,
                                   "label": "FEMTO test (training split, v1)"},
            "MFPT_zero_shot":    {"f1": 0.615, "tol": 0.02,
                                   "label": "MFPT zero-shot"},
        }
    # A PASS must mean: at least one published number was actually checked
    # against a real evaluation AND matched. Loads with no labels, empty test
    # sets, import failures, etc. NEVER count as a verification and NEVER PASS.
    n_checked = 0          # evaluations compared against a published number
    n_verified = 0         # of those, the ones that matched within tolerance
    n_deviation = 0
    n_data_produced = 0    # evaluations that produced *any* usable signal load
    for ev in summary["evaluations"]:
        status = ev.get("status")
        # Track that at least *some* real data ran through the model.
        if status in ("ok", "VERIFIED", "DEVIATION", "loaded_no_labels") and ev.get("n_samples", 0) > 0:
            n_data_produced += 1

        key = ev.get("path") or ev.get("name")
        pub = PUBLISHED.get(key)
        if not pub or status not in ("ok",):
            continue
        delta = abs(ev["f1_macro"] - pub["f1"])
        ok = delta <= pub["tol"]
        ev["published_f1"] = pub["f1"]
        ev["delta"] = delta
        ev["status"] = "VERIFIED" if ok else "DEVIATION"
        n_checked += 1
        if ok:
            n_verified += 1
        else:
            n_deviation += 1
        _logger.info("  %s: %.4f vs published %.4f -> %s (delta %.4f)",
                     pub["label"], ev["f1_macro"], pub["f1"], ev["status"], delta)

    summary["verification_summary"] = {
        "n_evaluations": len(summary["evaluations"]),
        "n_checked_against_published": n_checked,
        "n_verified": n_verified,
        "n_deviation": n_deviation,
        "n_data_produced": n_data_produced,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    _logger.info("Saved: %s", out)

    # ---- Decide exit status — PASSED is reserved for a real verification ----
    if n_deviation > 0:
        _logger.error("VERIFICATION FAILED: %d published number(s) deviated beyond tolerance.",
                      n_deviation)
        return 1

    if n_checked == 0:
        # Nothing was compared against a published number. This is NOT a PASS.
        if n_data_produced > 0:
            _logger.warning(
                "NO VERIFICATION PERFORMED: %d evaluation(s) loaded data but none "
                "could be checked against a published number "
                "(e.g. MFPT .mat has fault-type labels, not severity classes). "
                "Nothing verified. To reproduce F1=0.884 provide "
                "--aion-data-repo + --femto-root + --metadata (Path A).",
                n_data_produced)
            return 3
        _logger.error(
            "NO SAMPLES EVALUATED — nothing verified. "
            "Provide a real dataset: Path A (--aion-data-repo + --femto-root + "
            "--metadata) for F1=0.884, or --femto for a native FEMTO scan.")
        return 2

    _logger.info("VERIFICATION PASSED: %d/%d published number(s) reproduced within tolerance.",
                 n_verified, n_checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
