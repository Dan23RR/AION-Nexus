"""Verify ONNX export produces numerically identical predictions to PyTorch.

After exporting via `scripts/export_onnx.py`, run this to confirm the ONNX
runtime gives the same predictions as the PyTorch reference. Critical for
edge deployment confidence.

Usage:
    python -m scripts.verify_onnx_parity \\
        --pth checkpoints/aion_nexus_v1.pth \\
        --onnx checkpoints/aion_nexus.onnx \\
        --n-samples 100
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from aion_nexus import NUM_CHANNELS, SIGNAL_LENGTH, InferenceEngine
from aion_nexus.preprocessing import preprocess_signal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, help="PyTorch checkpoint")
    parser.add_argument("--onnx", required=True, help="ONNX export path")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--logits-tol", type=float, default=1e-4,
                        help="Max allowed |logit_pt - logit_onnx|")
    parser.add_argument("--prob-tol", type=float, default=1e-4)
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed. `pip install onnxruntime`.")
        return 2

    print(f"Loading PyTorch model from {args.pth} ...")
    engine = InferenceEngine.from_checkpoint(args.pth)

    print(f"Loading ONNX model from {args.onnx} ...")
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    max_logit_diff = 0.0
    max_prob_diff = 0.0
    class_disagreements = 0

    for _ in range(args.n_samples):
        sig = rng.standard_normal((NUM_CHANNELS, SIGNAL_LENGTH)).astype(np.float32)
        x_torch = preprocess_signal(sig)  # [1, 2, 2560]
        x_onnx = x_torch.numpy()

        with torch.no_grad():
            out_pt = engine.model(x_torch)
        logits_pt = out_pt["logits"].numpy()[0]

        outputs = session.run(None, {"input": x_onnx})
        logits_onnx = outputs[0][0]  # first output is logits

        # Logit diff
        d_logit = float(np.max(np.abs(logits_pt - logits_onnx)))
        max_logit_diff = max(max_logit_diff, d_logit)

        # Probability diff
        probs_pt = np.exp(logits_pt) / np.sum(np.exp(logits_pt))
        probs_onnx = np.exp(logits_onnx) / np.sum(np.exp(logits_onnx))
        d_prob = float(np.max(np.abs(probs_pt - probs_onnx)))
        max_prob_diff = max(max_prob_diff, d_prob)

        # Class agreement
        if int(np.argmax(logits_pt)) != int(np.argmax(logits_onnx)):
            class_disagreements += 1

    print()
    print(f"Tested {args.n_samples} synthetic samples")
    print(f"  max |logit diff|:  {max_logit_diff:.2e} (tolerance {args.logits_tol:.0e})")
    print(f"  max |prob diff|:   {max_prob_diff:.2e} (tolerance {args.prob_tol:.0e})")
    print(f"  class disagreements: {class_disagreements}")

    ok = (max_logit_diff <= args.logits_tol
          and max_prob_diff <= args.prob_tol
          and class_disagreements == 0)

    if ok:
        print("\nPASS: ONNX export is numerically equivalent to PyTorch.")
        return 0
    else:
        print("\nFAIL: ONNX export differs from PyTorch beyond tolerance.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
