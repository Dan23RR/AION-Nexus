"""Generate a VALID 2-channel vibration signal for the offline quickstart demo.

Run: python examples/generate_sample_signal.py
Produces: examples/sample_signal.csv

Output format matches the FEMTO PRONOSTIA ``acc_*.csv`` layout that
``aion_nexus.utils.load_signal_csv`` expects:

  - 6 numeric columns, comma-separated, NO header, NO trailing comment lines
  - cols 0-3: timestamp fields (hour, minute, second, microsecond)
  - cols 4, 5: horizontal / vertical accelerometer channels (g)
  - exactly 2560 rows (>= SIGNAL_LENGTH, so ``validate_signal`` accepts it)

Source selection (honest provenance):

  1. PREFERRED — copy a REAL FEMTO window if the bundled FEMTO data is present
     under ``data/FEMTO+Bearing/.../Test_set``. A real 2560-row window is the
     most faithful demo input. The chosen file path is printed so provenance is
     traceable.
  2. FALLBACK — a SYNTHETIC plausible signal (shaft tone + bearing-fault tone +
     white noise). This is plumbing-only and is clearly labelled as synthetic.

Either way the resulting CSV is a structurally valid input that
``examples/01_basic_inference.py examples/sample_signal.csv`` can consume.
NOTE on labels: AION-NEXUS predicts a 4-class life-stage / severity index
(0=normal ... 3=advanced), NOT a fault *type*. A single demo window has no
ground-truth label; the demo only exercises the inference path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIGNAL_LENGTH = 2560
N_ROWS = 2560  # exactly one FEMTO acquisition window
OUT_PATH = Path(__file__).parent / "sample_signal.csv"

# Candidate locations of a real FEMTO acc_*.csv (relative to package root).
_PKG_ROOT = Path(__file__).resolve().parent.parent
_FEMTO_GLOB = (
    "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/"
    "Test_set/Test_set"
)


def _find_real_femto_window() -> Path | None:
    """Return the path of a real FEMTO acc_*.csv with >= SIGNAL_LENGTH rows, or None."""
    base = _PKG_ROOT / _FEMTO_GLOB
    if not base.exists():
        return None
    for bearing_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        accs = sorted(bearing_dir.glob("acc_*.csv"))
        for acc in accs:
            try:
                raw = np.loadtxt(acc, delimiter=",")
            except Exception:
                continue
            if raw.ndim == 2 and raw.shape[0] >= SIGNAL_LENGTH and raw.shape[1] >= 6:
                return acc
    return None


def _write_csv(arr: np.ndarray) -> None:
    """Write a [N, 6] array as a FEMTO-format CSV (no header, no comments)."""
    np.savetxt(
        OUT_PATH,
        arr,
        delimiter=",",
        fmt=["%.0f", "%.0f", "%.0f", "%.0f", "%.6f", "%.6f"],
    )


def _from_real_femto(src: Path) -> None:
    raw = np.loadtxt(src, delimiter=",")
    arr = raw[:N_ROWS, :6]
    _write_csv(arr)
    # src is under the package, so make the printed path relative when possible
    try:
        rel = src.relative_to(_PKG_ROOT)
    except ValueError:
        rel = src
    print(f"Wrote: {OUT_PATH}  ({arr.shape[0]} rows, REAL FEMTO window)")
    print(f"Source: {rel}")
    print("Provenance: real PRONOSTIA acc window — structurally valid demo input.")


def _from_synthetic() -> None:
    rng = np.random.default_rng(0)
    fs = 25_600                        # sampling rate in Hz
    t = np.arange(N_ROWS) / fs

    shaft_hz = 30.0                    # ~1800 RPM
    bpfo_hz = 100.0                    # bearing outer-race tone
    sig_h = (
        0.4 * np.sin(2 * np.pi * shaft_hz * t)
        + 0.2 * np.sin(2 * np.pi * bpfo_hz * t)
        + 0.05 * rng.standard_normal(N_ROWS)
    )
    sig_v = (
        0.3 * np.sin(2 * np.pi * shaft_hz * t + 0.5)
        + 0.15 * np.sin(2 * np.pi * bpfo_hz * t + 1.0)
        + 0.05 * rng.standard_normal(N_ROWS)
    )

    hour = np.zeros(N_ROWS)
    minute = np.zeros(N_ROWS)
    second = (t * 1000).astype(int) % 60
    microsec = (t * 1_000_000).astype(int) % 1_000_000
    arr = np.column_stack([hour, minute, second, microsec, sig_h, sig_v])
    _write_csv(arr)
    print(f"Wrote: {OUT_PATH}  ({N_ROWS} rows, SYNTHETIC signal)")
    print("Provenance: synthetic shaft+BPFO tone — plumbing demo only, NOT a real fault.")


def main() -> int:
    real = _find_real_femto_window()
    if real is not None:
        _from_real_femto(real)
    else:
        _from_synthetic()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
