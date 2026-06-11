"""Utility functions: CSV loading, signal segmentation, file IO helpers."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from aion_nexus.config import NUM_CHANNELS, SIGNAL_LENGTH

_logger = logging.getLogger(__name__)


def load_signal_csv(path: str | Path, channels: tuple[int, int] = (4, 5)) -> np.ndarray:
    """Load a 2-channel vibration signal from a CSV file.

    Default column indices (4, 5) match the FEMTO PRONOSTIA `acc_*.csv` format
    where columns 0-3 are timestamps and 4-5 are horizontal/vertical
    accelerometer channels. Override ``channels`` for other formats.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D CSV; got shape {raw.shape}")
    ci, cj = channels
    if max(ci, cj) >= raw.shape[1]:
        raise ValueError(
            f"Requested channels ({ci}, {cj}) but CSV has only {raw.shape[1]} columns"
        )
    return raw[:, [ci, cj]].T.astype(np.float32, copy=False)


def segment_signal(signal: np.ndarray, window: int = SIGNAL_LENGTH,
                   stride: int | None = None) -> list[np.ndarray]:
    """Sliding-window segmentation of a long signal into [2, window] chunks.

    Args:
        signal: [2, N] array (or [N, 2]; auto-transposed).
        window: window length (default 2560).
        stride: hop size (default = window, i.e. non-overlapping).

    Returns:
        List of [2, window] arrays.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal; got {signal.ndim}D")
    if signal.shape[0] != NUM_CHANNELS and signal.shape[1] == NUM_CHANNELS:
        signal = signal.T
    if signal.shape[0] != NUM_CHANNELS:
        raise ValueError(f"Need {NUM_CHANNELS} channels; got shape {signal.shape}")

    if stride is None:
        stride = window
    n = signal.shape[1]
    if n < window:
        raise ValueError(f"Signal length {n} < window {window}")

    return [signal[:, i:i + window] for i in range(0, n - window + 1, stride)]


def aggregate_window_predictions(
    probabilities: list[np.ndarray],
    method: str = "mean",
) -> tuple[int, np.ndarray]:
    """Aggregate per-window probabilities into a single bearing-level decision.

    Args:
        probabilities: list of [num_classes] probability vectors.
        method: 'mean' (probability averaging — recommended), 'majority' (hard vote),
                or 'max_class' (highest single-window confidence).

    Returns:
        (predicted_class_index, aggregated_probability_vector)
    """
    arr = np.stack(probabilities, axis=0)
    if method == "mean":
        agg = arr.mean(axis=0)
        return int(np.argmax(agg)), agg
    if method == "max_class":
        # pick the window with highest top-class probability, return its full distribution
        top_per_window = arr.max(axis=1)
        best = int(np.argmax(top_per_window))
        return int(np.argmax(arr[best])), arr[best]
    if method == "majority":
        votes = np.argmax(arr, axis=1)
        counts = np.bincount(votes, minlength=arr.shape[1])
        winner = int(np.argmax(counts))
        agg = (votes == winner).mean() * np.eye(arr.shape[1])[winner]
        return winner, agg
    raise ValueError(f"Unknown aggregation method: {method}")
