"""Example 2: batched inference over multiple CSVs in a directory.

Run: python examples/02_batch_inference.py [path/to/csv_dir]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from aion_nexus import InferenceEngine
from aion_nexus.utils import load_signal_csv


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python 02_batch_inference.py <csv_dir>")
        return 2

    csv_dir = Path(argv[1])
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {csv_dir}")
        return 1

    ckpt = Path("checkpoints/aion_nexus_v1.pth")
    if not ckpt.exists():
        from aion_nexus.model import create_aion_nexus
        print("WARNING: no checkpoint, using random weights for demo.")
        engine = InferenceEngine(create_aion_nexus())
    else:
        engine = InferenceEngine.from_checkpoint(ckpt)

    print(f"Predicting on {len(csvs)} files...")
    signals = [load_signal_csv(p) for p in csvs]

    t0 = time.perf_counter()
    results = engine.predict_batch(signals)
    elapsed = time.perf_counter() - t0

    counts = {"normal": 0, "early": 0, "medium": 0, "advanced": 0}
    for path, result in zip(csvs, results):
        counts[result.predicted_class_name] += 1
        flag = "[!]" if result.recommended_action.get("stop_machine") else "   "
        print(
            f"{flag} {path.name:40s}  {result.predicted_class_name:9s}  "
            f"conf={result.confidence:.3f} ({result.confidence_band})"
        )

    print()
    print(f"Summary: {counts}")
    print(f"Latency: {elapsed*1000:.1f} ms total, {elapsed*1000/len(csvs):.2f} ms/sample")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
