"""Quantize AION-NEXUS to INT8 for edge / mobile deployment.

Uses PyTorch dynamic quantization (linear + GRU layers → INT8). Produces a
~3.5× smaller model with ~1.5–2× faster CPU inference; expected accuracy
loss < 1 percentage point F1 (verify with --validate).

Usage:
    python -m scripts.quantize \\
        --in checkpoints/aion_nexus_v1.pth \\
        --out checkpoints/aion_nexus_v1_int8.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from aion_nexus import InferenceEngine, NUM_CHANNELS, SIGNAL_LENGTH
from aion_nexus.preprocessing import preprocess_signal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validate", action="store_true",
                        help="Compare quantized vs FP32 on synthetic samples")
    parser.add_argument("--n-validate", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading FP32 model from {args.in_path} ...")
    engine_fp32 = InferenceEngine.from_checkpoint(args.in_path)
    fp32_size = Path(args.in_path).stat().st_size / 1024 / 1024
    print(f"  FP32 size: {fp32_size:.2f} MB")

    print("Applying dynamic quantization (Linear + GRU → INT8) ...")
    quantized = torch.quantization.quantize_dynamic(
        engine_fp32.model,
        qconfig_spec={torch.nn.Linear, torch.nn.GRU},
        dtype=torch.qint8,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": quantized.state_dict()}, out_path)
    int8_size = out_path.stat().st_size / 1024 / 1024
    print(f"  INT8 size: {int8_size:.2f} MB ({fp32_size/int8_size:.1f}× smaller)")

    if args.validate:
        print(f"\nValidating on {args.n_validate} synthetic samples ...")
        engine_int8 = InferenceEngine(quantized)
        rng = np.random.default_rng(0)
        agree = 0
        max_prob_drift = 0.0
        for _ in range(args.n_validate):
            sig = rng.standard_normal((NUM_CHANNELS, SIGNAL_LENGTH)).astype(np.float32)
            r_fp32 = engine_fp32.predict(sig)
            r_int8 = engine_int8.predict(sig)
            if r_fp32.predicted_class_index == r_int8.predicted_class_index:
                agree += 1
            for cls in r_fp32.probabilities:
                d = abs(r_fp32.probabilities[cls] - r_int8.probabilities[cls])
                max_prob_drift = max(max_prob_drift, d)
        agree_pct = 100 * agree / args.n_validate
        print(f"  Class agreement:    {agree}/{args.n_validate} ({agree_pct:.1f}%)")
        print(f"  Max prob drift:     {max_prob_drift:.4f}")
        if agree_pct < 95:
            print(f"WARN: agreement < 95%; quantization may be hurting accuracy too much.")
            print(f"  Consider per-channel quantization or quantization-aware training.")
            return 1

    print(f"\nSaved INT8 checkpoint to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
