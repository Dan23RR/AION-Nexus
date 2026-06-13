"""Signal preprocessing pipeline.

Production contract (MUST match training-time preprocessing in `aion_data.py`):

1. Validate input shape, NaN/Inf, stuck-sensor.
2. Center-crop or pad to SIGNAL_LENGTH (2560 samples).
3. Per-channel z-score: ``(x - mean) / (std + 1e-8)`` per channel per sample.
4. High-pass Butterworth filter at 1 Hz cutoff (2nd order, sosfilt) to remove
   DC bias and low-frequency drift. Same filter as training.
5. Return as torch tensor [1, 2, 2560] (or [B, 2, 2560] for batch).

The order is critical: z-score BEFORE high-pass, exactly as in training.
Skipping the HP filter degrades F1 from 0.884 → 0.28 (verified empirically
2026-04-27 via independent verify_checkpoint run before the fix).
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from aion_nexus.config import NUM_CHANNELS, SAMPLING_RATE_HZ, SIGNAL_LENGTH

_logger = logging.getLogger(__name__)

# Warn at most once per process if scipy is missing and we degrade to the
# mean-removal fallback (which is NOT the training-time preprocessing).
_SCIPY_FALLBACK_WARNED = False


class SignalValidationError(ValueError):
    """Raised when an input signal violates the production contract."""


def validate_signal(signal: np.ndarray) -> np.ndarray:
    """Validate, normalize shape to [2, 2560], return as float32.

    Accepts:
      - [2, N]   — preferred shape
      - [N, 2]   — auto-transposed
      - [2*N]    — interleaved 2-channel raised as error (ambiguous)

    Raises SignalValidationError on:
      - wrong dimensionality
      - NaN / Inf
      - constant signal (sensor stuck)
      - signal length too short or too long after centering

    Truncates / center-crops if longer than SIGNAL_LENGTH.
    """
    if not isinstance(signal, np.ndarray):
        try:
            signal = np.asarray(signal, dtype=np.float32)
        except Exception as exc:
            raise SignalValidationError(f"Cannot convert input to ndarray: {exc}") from exc

    if signal.ndim != 2:
        raise SignalValidationError(
            f"Expected 2D array of shape [2, N] or [N, 2]; got {signal.ndim}D shape {signal.shape}"
        )

    # Auto-transpose if needed
    if signal.shape[0] != NUM_CHANNELS and signal.shape[1] == NUM_CHANNELS:
        signal = signal.T
    if signal.shape[0] != NUM_CHANNELS:
        raise SignalValidationError(
            f"Expected {NUM_CHANNELS} channels; got shape {signal.shape}"
        )

    # Cast to float32 FIRST, then check finiteness (kill-shot #3: cast-then-check).
    # The serving dtype is float32. A finite-in-float64 value with |x| >= ~3.4e38
    # overflows to +/-Inf on the float32 cast; checking finiteness in float64
    # BEFORE the cast lets such a value pass, after which it becomes Inf, the
    # z-score yields NaN, and predict() silently returns class=normal conf=nan
    # stop_machine=False on corrupted input. Casting first makes the overflow
    # observable here and rejected with an actionable error.
    signal = signal.astype(np.float32, copy=False)
    # Check NaN/Inf BEFORE center-crop so anomalies in any portion of the
    # input signal are caught (not silently dropped by cropping).
    if not np.all(np.isfinite(signal)):
        raise SignalValidationError(
            "Signal contains NaN or Inf values (after float32 cast; note that "
            "magnitudes >= ~3.4e38 overflow to Inf in float32 and are rejected here)"
        )

    n = signal.shape[1]
    if n < SIGNAL_LENGTH:
        raise SignalValidationError(
            f"Signal too short: {n} samples (need at least {SIGNAL_LENGTH} = "
            f"{SIGNAL_LENGTH / 25_600:.3f}s at 25.6 kHz)"
        )
    if n > SIGNAL_LENGTH:
        # Center-crop
        start = (n - SIGNAL_LENGTH) // 2
        signal = signal[:, start:start + SIGNAL_LENGTH]

    # Detect stuck sensors: per-channel std must exceed the stuck threshold.
    # Threshold = 1e-7 chosen for two reasons:
    #   1. Real industrial accelerometers below this std are effectively dead
    #      (typical noise floor of PCB 352C03 is ~5 μg = 5e-6 g, well above 1e-7).
    #   2. Tighter thresholds (e.g. 1e-9) leak past float32 precision noise:
    #      np.asarray([0.1]*N, dtype=float32).std() ≈ 2e-8 due to rounding.
    # Compute std in float64 to avoid float32 precision drift altering the
    # stuck-detection at the boundary.
    stds = signal.astype(np.float64).std(axis=1)
    stuck_threshold = 1e-7
    if np.any(stds < stuck_threshold):
        bad_ch = int(np.argmin(stds))
        raise SignalValidationError(
            f"Channel {bad_ch} appears stuck (std={stds[bad_ch]:.2e} < {stuck_threshold:.0e}); "
            "check sensor calibration"
        )

    return signal.astype(np.float32, copy=False)


def _highpass_filter(arr: np.ndarray, cutoff_hz: float = 1.0,
                     fs_hz: int = SAMPLING_RATE_HZ, order: int = 2) -> np.ndarray:
    """High-pass Butterworth filter (sosfilt) per channel.

    Matches the training-time filter in `aion_data.py`. Removes DC bias and
    low-frequency drift below `cutoff_hz`. Falls back to mean-removal if
    scipy unavailable (graceful degradation, but training used scipy).
    """
    try:
        from scipy import signal as scipy_signal
    except ImportError:
        # Mean-removal fallback. This is a degraded path, NOT the training-time
        # preprocessing: it drops FEMTO F1 from 0.884 to ~0.28. scipy is a hard
        # runtime dependency (pinned in requirements.txt and pyproject.toml); if
        # we are here, the install is broken. Make the degradation observable.
        global _SCIPY_FALLBACK_WARNED
        if not _SCIPY_FALLBACK_WARNED:
            _logger.warning(
                "scipy unavailable: high-pass Butterworth replaced by mean-removal "
                "fallback. This is NOT training-time preprocessing and degrades "
                "FEMTO F1 0.884 -> ~0.28. Install scipy>=1.10. (warned once)"
            )
            _SCIPY_FALLBACK_WARNED = True
        return arr - arr.mean(axis=1, keepdims=True)

    sos = scipy_signal.butter(order, cutoff_hz, btype="highpass",
                               fs=fs_hz, output="sos")
    out = np.empty_like(arr)
    for ch in range(arr.shape[0]):
        out[ch] = scipy_signal.sosfilt(sos, arr[ch])
    return out


def preprocess_signal(signal: np.ndarray) -> torch.Tensor:
    """Validate + z-score per channel + HP-Butterworth 1Hz + tensor [1, 2, 2560].

    Pipeline matches training-time `aion_data.AIONDataset._process_csv_file`:
      1. Validate (shape, NaN, stuck)
      2. Crop/pad to 2560
      3. Z-score per channel: ``(x - mean) / (std + 1e-8)``
      4. High-pass Butterworth (2nd order, 1 Hz cutoff @ 25.6 kHz)
    """
    arr = validate_signal(signal)
    # Step 3: z-score per channel
    means = arr.mean(axis=1, keepdims=True)
    stds = arr.std(axis=1, keepdims=True) + 1e-8
    arr = (arr - means) / stds
    # Step 4: HP-Butterworth (training uses 1 Hz cutoff @ 25.6 kHz)
    arr = _highpass_filter(arr, cutoff_hz=1.0, fs_hz=SAMPLING_RATE_HZ, order=2)
    arr = arr.astype(np.float32, copy=False)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, 2, 2560]
    return tensor


def preprocess_batch(signals: list[np.ndarray]) -> torch.Tensor:
    """Stack a list of validated/normalized signals into a [B, 2, 2560] batch."""
    if not signals:
        raise SignalValidationError("Empty batch")
    tensors = [preprocess_signal(s) for s in signals]
    return torch.cat(tensors, dim=0)
