"""Example 1: load checkpoint, predict on a single signal.

Run: python examples/01_basic_inference.py [path/to/signal.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

from aion_nexus import InferenceEngine
from aion_nexus.utils import load_signal_csv


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        signal = load_signal_csv(argv[1])
        print(f"Loaded signal: {argv[1]}  shape={signal.shape}")
    else:
        # Synthetic fallback
        rng = np.random.default_rng(0)
        signal = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
        print("No CSV provided — using synthetic random signal as demo.")

    ckpt = Path("checkpoints/aion_nexus_v1.pth")
    if not ckpt.exists():
        print(f"Checkpoint not found at {ckpt}. See checkpoints/README.md.")
        print("Running with random weights for demo only — predictions are NOT meaningful.")
        from aion_nexus.model import create_aion_nexus
        engine = InferenceEngine(create_aion_nexus())
    else:
        engine = InferenceEngine.from_checkpoint(ckpt)

    result = engine.predict(signal)
    print()
    print(f"Predicted class:  {result.predicted_class_name} (index {result.predicted_class_index})")
    print(f"  Description:    {result.description}")
    print(f"  Confidence:     {result.confidence:.3f} ({result.confidence_band})")
    print("  Probabilities:")
    for cls, p in result.probabilities.items():
        bar = "#" * int(p * 30)
        print(f"    {cls:10s} {p:.4f}  {bar}")
    print(f"  Latency:        {result.latency_ms:.2f} ms")
    print(f"  Action:         {result.recommended_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
