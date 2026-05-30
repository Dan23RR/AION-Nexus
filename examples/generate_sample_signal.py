"""Generate a synthetic 2-channel vibration signal for offline demo.

Run: python examples/generate_sample_signal.py
Produces: examples/sample_signal.csv (FEMTO acc_*.csv format with 6 columns).

The synthetic signal mixes:
- Shaft rotation tone (typical 30-60 Hz on lab bearings)
- Bearing fault tone (BPFO ~ 100 Hz at 30 RPS shaft speed)
- White noise floor

This is for plumbing tests only, not a realistic fault signal.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def main() -> int:
    rng = np.random.default_rng(0)
    fs = 25_600                        # sampling rate in Hz
    n = 2_700                          # > 2560 (model needs 2560)
    t = np.arange(n) / fs

    shaft_hz = 30.0                    # ~1800 RPM
    bpfo_hz  = 100.0                   # bearing outer-race tone
    sig_h = (
        0.4 * np.sin(2 * np.pi * shaft_hz * t)
        + 0.2 * np.sin(2 * np.pi * bpfo_hz * t)
        + 0.05 * rng.standard_normal(n)
    )
    sig_v = (
        0.3 * np.sin(2 * np.pi * shaft_hz * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * bpfo_hz * t + 1.0)
        + 0.05 * rng.standard_normal(n)
    )

    # FEMTO acc_*.csv format: 6 columns
    # cols 0-3: hour, minute, second, microsecond (we use dummy values)
    # cols 4, 5: horizontal, vertical accelerometer (g)
    hour = np.zeros(n)
    minute = np.zeros(n)
    second = (t * 1000).astype(int) % 60
    microsec = ((t * 1_000_000).astype(int) % 1_000_000)
    arr = np.column_stack([hour, minute, second, microsec, sig_h, sig_v])

    out = Path(__file__).parent / "sample_signal.csv"
    np.savetxt(out, arr, delimiter=",",
               fmt=["%.0f", "%.0f", "%.0f", "%.0f", "%.6f", "%.6f"])
    print(f"Wrote: {out}  ({n} samples, FEMTO format)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
