"""Diagnose why v6 verification gives F1=0.30 instead of 0.934.

Three checks:
1. state_dict key match: every expected key present, no surprises
2. Per-layer weight statistics: detect uninitialized layers
3. Forward hyperparameter sweep: try (T_recursions, N_supervision, n_reasoning)
   combinations to find the one matching training-time inference

Usage:
    python -m scripts.diagnose_v6 \
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


def check_state_dict_keys(ckpt_path: Path, num_classes: int = 4) -> dict:
    """Compare state_dict in checkpoint vs model's expected keys."""
    from aion_nexus.model_v6 import create_aion_nexus_v6

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    model = create_aion_nexus_v6(num_classes=num_classes)
    expected = set(model.state_dict().keys())
    got = set(sd.keys())

    missing = expected - got
    unexpected = got - expected
    matched = expected & got

    # Shape mismatch
    shape_mismatch = []
    for k in matched:
        if sd[k].shape != model.state_dict()[k].shape:
            shape_mismatch.append((k, tuple(sd[k].shape), tuple(model.state_dict()[k].shape)))

    return {
        "n_expected": len(expected),
        "n_got": len(got),
        "n_matched": len(matched),
        "missing_in_ckpt": sorted(missing),
        "unexpected_in_ckpt": sorted(unexpected),
        "shape_mismatch": shape_mismatch,
        "all_match": not (missing or unexpected or shape_mismatch),
        "ckpt_meta": {k: type(v).__name__ for k, v in ckpt.items() if isinstance(ckpt, dict)} if isinstance(ckpt, dict) else None,
    }


def per_layer_stats(model) -> dict:
    """Mean/std of weights per layer — detect uninitialized layers."""
    out = {}
    for name, p in model.named_parameters():
        if p.numel() == 0:
            continue
        out[name] = {
            "shape": list(p.shape),
            "mean": float(p.mean()),
            "std": float(p.std()),
            "min": float(p.min()),
            "max": float(p.max()),
        }
    return out


def hyperparameter_sweep(engine, dataset, test_idx, configs: list[dict]) -> list[dict]:
    """Try multiple forward hyperparameter configurations on the test set."""
    import torch
    model = engine.model.eval()

    # Cache features once (they don't depend on N_supervision / T_recursions)
    test_features = []
    test_labels = []
    BATCH = 64  # noqa: N806 — constant-style local, intentional
    with torch.no_grad():
        signals = []
        for idx in test_idx:
            sig, lab, _ = dataset[idx]
            signals.append(sig)
            test_labels.append(int(lab.item() if hasattr(lab, "item") else lab))
        for i in range(0, len(signals), BATCH):
            x = torch.stack(signals[i:i + BATCH], dim=0)
            f = model.extract_features(x)
            test_features.append(f)
    test_features = torch.cat(test_features, dim=0)
    y_true = np.asarray(test_labels)

    results = []
    for cfg in configs:
        T = cfg.get("T_recursions", 3)        # noqa: N806 — mirrors TRM paper notation
        n = cfg.get("n_reasoning", 6)
        N_sup = cfg.get("N_supervision", 1)   # noqa: N806 — mirrors TRM paper notation
        _logger.info("  Sweep: T=%d n=%d N_sup=%d", T, n, N_sup)
        preds = []
        with torch.no_grad():
            for i in range(0, test_features.shape[0], BATCH):
                f_batch = test_features[i:i + BATCH]
                if N_sup == 1:
                    out = model.recursive_reasoner(
                        f_batch, y_init=None, z_init=None,
                        n=n, T=T, train_mode=False,
                    )
                    preds.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())
                else:
                    # Mimic deep supervision at eval time
                    b = f_batch.size(0)
                    y = torch.full((b, model.num_classes), 1.0/model.num_classes)
                    z = torch.zeros(b, model.latent_dim)
                    out = None
                    for _ in range(N_sup):
                        out = model.recursive_reasoner(
                            f_batch, y_init=y, z_init=z, n=n, T=T, train_mode=False,
                        )
                        y = out["logits"]
                        z = out["latent"]
                    preds.extend(out["logits"].argmax(dim=1).cpu().numpy().tolist())
        y_pred = np.asarray(preds)
        f1 = f1_macro(y_true, y_pred)
        cfg_result = dict(cfg)
        cfg_result["f1_macro"] = f1
        cfg_result["accuracy"] = float((y_pred == y_true).mean())
        results.append(cfg_result)
        _logger.info("    -> F1=%.4f Acc=%.4f", f1, cfg_result["accuracy"])

    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--aion-data-repo", default="../DefinitiveAION/clean")
    parser.add_argument("--femto-root", default=None)
    parser.add_argument("--metadata", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        _logger.error("Checkpoint not found: %s", ckpt)
        return 2

    # === Check 1: state_dict key match ===
    _logger.info("=" * 70)
    _logger.info("CHECK 1: state_dict key match")
    _logger.info("=" * 70)
    sd_check = check_state_dict_keys(ckpt)
    _logger.info("Expected: %d keys, got: %d keys, matched: %d",
                 sd_check["n_expected"], sd_check["n_got"], sd_check["n_matched"])
    if sd_check["missing_in_ckpt"]:
        _logger.error("MISSING in checkpoint (model expects but ckpt lacks):")
        for k in sd_check["missing_in_ckpt"][:20]:
            _logger.error("  %s", k)
    if sd_check["unexpected_in_ckpt"]:
        _logger.warning("UNEXPECTED in checkpoint (ckpt has but model doesn't expect):")
        for k in sd_check["unexpected_in_ckpt"][:20]:
            _logger.warning("  %s", k)
    if sd_check["shape_mismatch"]:
        _logger.error("SHAPE MISMATCH:")
        for k, ckpt_sh, mod_sh in sd_check["shape_mismatch"][:20]:
            _logger.error("  %s: ckpt=%s vs model=%s", k, ckpt_sh, mod_sh)
    _logger.info("Checkpoint top-level keys: %s", sd_check.get("ckpt_meta"))

    # === Check 2: per-layer weight statistics ===
    _logger.info("=" * 70)
    _logger.info("CHECK 2: per-layer weight statistics")
    _logger.info("=" * 70)
    from aion_nexus import InferenceEngine
    engine = InferenceEngine.from_checkpoint(ckpt)
    stats = per_layer_stats(engine.model)
    for name, s in list(stats.items())[:10]:
        _logger.info("  %s: shape=%s mean=%.4f std=%.4f range=[%.3f, %.3f]",
                     name, s["shape"], s["mean"], s["std"], s["min"], s["max"])
    # Check for uninitialized layers (std too large = random init still in place)
    uninitialized = [n for n, s in stats.items()
                     if s["std"] > 0.5 and "norm" not in n.lower() and "pool_query" not in n.lower()]
    if uninitialized:
        _logger.warning("Layers with suspiciously large std (possibly uninitialized):")
        for n in uninitialized[:10]:
            _logger.warning("  %s: std=%.3f", n, stats[n]["std"])

    # === Check 3: hyperparameter sweep on FEMTO test set ===
    if not args.femto_root or not args.metadata:
        _logger.warning("Skipping CHECK 3 (need --femto-root and --metadata)")
        return 0

    _logger.info("=" * 70)
    _logger.info("CHECK 3: hyperparameter sweep on FEMTO test set")
    _logger.info("=" * 70)
    sys.path.insert(0, str(Path(args.aion_data_repo)))
    from aion_data import AIONDataset, TemporalConfig, create_stratified_temporal_splits

    config = TemporalConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    dataset = AIONDataset(data_root=str(args.femto_root), metadata_path=str(args.metadata))
    _, _, test_idx = create_stratified_temporal_splits(dataset, config)

    sweeps = [
        {"T_recursions": 1, "n_reasoning": 6, "N_supervision": 1},
        {"T_recursions": 3, "n_reasoning": 6, "N_supervision": 1},  # default
        {"T_recursions": 6, "n_reasoning": 6, "N_supervision": 1},
        {"T_recursions": 3, "n_reasoning": 1, "N_supervision": 1},
        {"T_recursions": 3, "n_reasoning": 12, "N_supervision": 1},
        {"T_recursions": 3, "n_reasoning": 6, "N_supervision": 4},
        {"T_recursions": 1, "n_reasoning": 1, "N_supervision": 1},  # minimal
    ]
    sweep_results = hyperparameter_sweep(engine, dataset, test_idx, sweeps)
    _logger.info("=" * 70)
    _logger.info("Best configuration:")
    best = max(sweep_results, key=lambda r: r["f1_macro"])
    _logger.info("  T=%d n=%d N_sup=%d -> F1=%.4f Acc=%.4f",
                 best["T_recursions"], best["n_reasoning"], best["N_supervision"],
                 best["f1_macro"], best["accuracy"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
