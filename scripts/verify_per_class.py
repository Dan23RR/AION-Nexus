"""Regenerate the per-class FEMTO-test precision/recall/F1 from the v1 checkpoint.

The per-class table previously shipped in MODEL_CARD.md / PERFORMANCE_BENCHMARKS.md
was arithmetically impossible (e.g. class 0: F1=0.93 < both P=0.96 and R=0.998) and
had no source JSON in the package. This script regenerates the REAL numbers by
reproducing the exact training-time test split (the same one that yields the
F1=0.884 headline) and running sklearn's classification_report.

GUARDRAIL: if the reproduced aggregate macro-F1 does not match the published 0.884
within +/-0.01, the per-class numbers are NOT trustworthy (the split was not
reproduced correctly) and must NOT be pasted into the model card.

Usage:
    python -m scripts.verify_per_class \
        --checkpoint checkpoints/aion_nexus_v1.pth \
        --aion-data-repo ../DefinitiveAION/clean \
        --femto-root "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set" \
        --metadata data/femto_metadata.json \
        --out results/per_class_f1.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from aion_nexus import InferenceEngine
from aion_nexus.config import CLASS_NAMES, NUM_CLASSES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_logger = logging.getLogger("verify_per_class")

PUBLISHED_F1 = 0.884
TOLERANCE = 0.01


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/aion_nexus_v1.pth")
    p.add_argument("--aion-data-repo", default="../DefinitiveAION/clean")
    p.add_argument(
        "--femto-root",
        default="data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set",
    )
    p.add_argument("--metadata", default="data/femto_metadata.json")
    p.add_argument("--out", default="results/per_class_f1.json")
    args = p.parse_args()

    from sklearn.metrics import precision_recall_fscore_support, f1_score

    engine = InferenceEngine.from_checkpoint(args.checkpoint)
    _logger.info("Loaded %s checkpoint", engine.architecture_version)

    sys.path.insert(0, str(Path(args.aion_data_repo)))
    from aion_data import AIONDataset, TemporalConfig, create_stratified_temporal_splits

    config = TemporalConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dataset = AIONDataset(data_root=args.femto_root, metadata_path=args.metadata)
    _, _, test_idx = create_stratified_temporal_splits(dataset, config)
    _logger.info("Reproduced test split: %d samples", len(test_idx))

    model = engine.model.eval()
    sigs, y_true = [], []
    for idx in test_idx:
        signal, label, _ = dataset[idx]
        sigs.append(signal)
        y_true.append(int(label.item() if hasattr(label, "item") else label))

    preds = []
    with torch.no_grad():
        for i in range(0, len(sigs), 64):
            x = torch.stack(sigs[i:i + 64], dim=0)
            preds.extend(model(x)["logits"].argmax(dim=1).cpu().numpy().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(preds)

    agg_f1 = float(f1_score(y_true, y_pred, average="macro"))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    matches = abs(agg_f1 - PUBLISHED_F1) <= TOLERANCE
    per_class = [
        {
            "class_index": c,
            "class_name": CLASS_NAMES[c],
            "support": int(support[c]),
            "precision": round(float(prec[c]), 4),
            "recall": round(float(rec[c]), 4),
            "f1": round(float(f1[c]), 4),
        }
        for c in range(NUM_CLASSES)
    ]

    result = {
        "checkpoint": args.checkpoint,
        "n_test_samples": int(len(test_idx)),
        "aggregate_f1_macro": round(agg_f1, 4),
        "published_f1": PUBLISHED_F1,
        "reproduces_published": matches,
        "per_class": per_class,
    }

    print("\n=== Per-class FEMTO test (reproduced) ===")
    print(f"{'class':<12}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>8}")
    for r in per_class:
        print(f"{r['class_name']:<12}{r['support']:>8}{r['precision']:>11}{r['recall']:>9}{r['f1']:>8}")
    print(f"\naggregate macro-F1 = {agg_f1:.4f}  (published {PUBLISHED_F1})  "
          f"-> {'MATCH' if matches else 'MISMATCH — DO NOT USE'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    _logger.info("Wrote %s", args.out)

    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
