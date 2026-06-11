"""Discriminating test: load checkpoint into the SOURCE v6 model class
(from `DefinitiveAION/clean/aion_nexus_v6.py`) instead of my reconstruction.

If F1 ≈ 0.934 with the source model → my port has a subtle bug; fix the port.
If F1 ≈ 0.30 with the source model → bug is elsewhere (data, preprocessing).

Usage:
    python -m scripts.verify_with_source_v6 \
        --checkpoint checkpoints/aion_nexus_v6.pth \
        --aion-data-repo ../DefinitiveAION/clean \
        --femto-root data/FEMTO+Bearing/.../Test_set/Test_set \
        --metadata data/femto_metadata.json
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_logger = logging.getLogger(__name__)


def f1_macro(y_true, y_pred, num_classes=4):
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


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aion-data-repo", required=True,
                        help="Path containing aion_nexus_v6.py + aion_data.py")
    parser.add_argument("--femto-root", required=True)
    parser.add_argument("--metadata", required=True)
    args = parser.parse_args()

    repo = Path(args.aion_data_repo)
    if not (repo / "aion_nexus_v6.py").exists():
        _logger.error("aion_nexus_v6.py not found in %s", repo)
        return 2

    sys.path.insert(0, str(repo))
    from aion_data import AIONDataset, TemporalConfig, create_stratified_temporal_splits
    from aion_nexus_v6 import create_aion_nexus_v6 as create_source_v6

    _logger.info("Creating SOURCE v6 model (not the production-package port)...")
    model = create_source_v6(num_classes=4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info("  Source v6 params: %d", n_params)
    assert n_params == 716_577, f"Source v6 has {n_params} params, expected 716,577"

    _logger.info("Loading checkpoint %s into source v6 model...", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=True)
    _logger.info("  load_state_dict: missing=%s unexpected=%s", missing, unexpected)
    model.eval()

    _logger.info("Loading FEMTO test set (same path A as standard verify)...")
    config = TemporalConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dataset = AIONDataset(data_root=str(args.femto_root), metadata_path=str(args.metadata))
    _, _, test_idx = create_stratified_temporal_splits(dataset, config)
    _logger.info("  Test set: %d samples", len(test_idx))

    signals, labels = [], []
    for idx in test_idx:
        sig, lab, _ = dataset[idx]
        signals.append(sig)
        labels.append(int(lab.item() if hasattr(lab, "item") else lab))

    _logger.info("Running inference with SOURCE model...")
    preds = []
    BATCH = 64  # noqa: N806 — constant-style local, intentional
    with torch.no_grad():
        for i in range(0, len(signals), BATCH):
            x = torch.stack(signals[i:i + BATCH], dim=0)
            out = model(x, N_supervision=1)
            preds.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())

    y_true = np.asarray(labels)
    y_pred = np.asarray(preds)
    f1 = f1_macro(y_true, y_pred)
    acc = float((y_pred == y_true).mean())

    print()
    print("=" * 70)
    print("SOURCE V6 model verification:")
    print(f"  F1 macro:  {f1:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print("  Published: 0.9343")
    print(f"  Delta:     {abs(f1 - 0.9343):.4f}")
    print("=" * 70)
    if f1 > 0.85:
        print("PASS: source v6 reproduces published F1.")
        print("→ Conclusion: bug is in the production-package port (aion_nexus/model_v6.py)")
        print("  Most likely: subtle architectural difference (layer order, init, forward).")
    else:
        print("FAIL: source v6 also gives low F1.")
        print("→ Conclusion: bug is upstream (data loading, preprocessing, label assignment).")
    print("=" * 70)
    return 0 if f1 > 0.85 else 1


if __name__ == "__main__":
    sys.exit(main())
