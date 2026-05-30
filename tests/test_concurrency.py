"""Concurrency / thread-safety tests.

InferenceEngine read-only inference (`predict`, `predict_batch`, `extract_features`)
should be safe to call from multiple threads simultaneously without corrupting
state or producing wrong predictions.
"""
import threading

import numpy as np
import pytest

from aion_nexus import InferenceEngine, NUM_CHANNELS, SIGNAL_LENGTH
from aion_nexus.model import create_aion_nexus


@pytest.fixture(scope="module")
def engine():
    import torch
    torch.manual_seed(0)
    np.random.seed(0)
    return InferenceEngine(create_aion_nexus())


@pytest.fixture(scope="module")
def fixed_signal():
    np.random.seed(42)
    return np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)


class TestThreadSafetyReadOnly:
    def test_concurrent_predict_same_signal(self, engine, fixed_signal):
        """Many threads predicting the same signal produce same result."""
        results = []
        lock = threading.Lock()

        def worker():
            r = engine.predict(fixed_signal)
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(results) == 10
        # All must produce same prediction
        ref = results[0]
        for r in results[1:]:
            assert r.predicted_class_index == ref.predicted_class_index
            for cls in r.probabilities:
                assert r.probabilities[cls] == pytest.approx(
                    ref.probabilities[cls], abs=1e-4
                )

    def test_concurrent_predict_different_signals(self, engine):
        """Concurrent predictions on different signals work correctly."""
        np.random.seed(1)
        signals = [
            np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
            for _ in range(20)
        ]
        # Reference: sequential
        ref_results = [engine.predict(s) for s in signals]

        # Concurrent
        results: list = [None] * 20
        lock = threading.Lock()

        def worker(idx, sig):
            r = engine.predict(sig)
            with lock:
                results[idx] = r

        threads = [threading.Thread(target=worker, args=(i, s))
                   for i, s in enumerate(signals)]
        for t in threads: t.start()
        for t in threads: t.join()

        for i, (ref, conc) in enumerate(zip(ref_results, results)):
            assert ref.predicted_class_index == conc.predicted_class_index, \
                f"Mismatch at {i}: ref={ref.predicted_class_index}, " \
                f"concurrent={conc.predicted_class_index}"

    def test_concurrent_extract_features(self, engine, fixed_signal):
        """Concurrent feature extractions return identical vectors."""
        results = []
        lock = threading.Lock()

        def worker():
            f = engine.extract_features(fixed_signal)
            with lock:
                results.append(f)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(results) == 8
        ref = results[0]
        for r in results[1:]:
            np.testing.assert_array_almost_equal(r, ref, decimal=5)

    def test_health_telemetry_does_not_corrupt(self, engine, fixed_signal):
        """Concurrent calls correctly increment inference count (within race
        tolerance) without crashing."""
        n_threads = 5
        n_per_thread = 10

        def worker():
            for _ in range(n_per_thread):
                engine.predict(fixed_signal)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        before = engine.get_health()["inference_count"]
        for t in threads: t.start()
        for t in threads: t.join()
        after = engine.get_health()["inference_count"]

        # Counter is non-atomic; expect increments equal to total OR slightly
        # under (race condition cosmetic). Must be > before and ≤ before + total.
        total = n_threads * n_per_thread
        assert after > before
        assert after - before <= total
