"""Benchmark inference latency / throughput on synthetic input.

Run: python -m scripts.benchmark_inference [--device cpu|cuda] [--batch-size 1|8|32]
"""
from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

from aion_nexus import InferenceEngine, NUM_CHANNELS, SIGNAL_LENGTH
from aion_nexus.model import create_aion_nexus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-iter", type=int, default=200)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint; uses random weights if absent")
    args = parser.parse_args()

    if args.checkpoint:
        engine = InferenceEngine.from_checkpoint(args.checkpoint, device=args.device)
    else:
        engine = InferenceEngine(create_aion_nexus(), device=args.device)

    rng = np.random.default_rng(0)
    if args.batch_size == 1:
        sig = rng.standard_normal((NUM_CHANNELS, SIGNAL_LENGTH)).astype(np.float32)
        # Warmup
        for _ in range(args.n_warmup):
            engine.predict(sig)
        latencies = []
        for _ in range(args.n_iter):
            t0 = time.perf_counter()
            engine.predict(sig)
            latencies.append((time.perf_counter() - t0) * 1000)
    else:
        sigs = [rng.standard_normal((NUM_CHANNELS, SIGNAL_LENGTH)).astype(np.float32)
                for _ in range(args.batch_size)]
        for _ in range(args.n_warmup):
            engine.predict_batch(sigs)
        latencies = []
        for _ in range(args.n_iter):
            t0 = time.perf_counter()
            engine.predict_batch(sigs)
            latencies.append((time.perf_counter() - t0) * 1000)

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    p99 = sorted(latencies)[int(0.99 * len(latencies))]
    mean = statistics.mean(latencies)
    throughput = (args.batch_size * 1000) / mean

    print(f"AION-NEXUS inference benchmark")
    print(f"  device:        {args.device}")
    print(f"  batch_size:    {args.batch_size}")
    print(f"  iterations:    {args.n_iter}  (warmup: {args.n_warmup})")
    print(f"  latency (ms):  mean={mean:.2f}  p50={p50:.2f}  p95={p95:.2f}  p99={p99:.2f}")
    print(f"  throughput:    {throughput:.1f} samples/sec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
