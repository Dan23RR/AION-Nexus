"""Signal-plausibility gate (heuristic out-of-distribution guard).

WHAT THIS IS — AND IS NOT
-------------------------
This is a **lightweight, hand-tuned heuristic** that asks a single honest
question: *does this input look like a real bearing-vibration window at all?*
It is NOT a learned OOD detector, NOT a density model, NOT a calibrated novelty
score. It cannot tell an early defect from an advanced one, and it makes no
claim about the classifier's reliability inside the in-distribution region.
It exists for one purpose: to stop the classifier from emitting a confident,
actionable verdict on inputs that are physically implausible as bearing
vibration (white noise, saturated/clipped sensors, quasi-constant traces).

WHY IT IS NEEDED (red-team kill-shot #4)
----------------------------------------
The 4-class softmax head never outputs "I don't know". Fed white Gaussian
noise, the deployed v1 model produced mean confidence ~0.788 with ~44% of
windows above the HIGH threshold (automated-action band) and *never* predicted
``normal`` — i.e. it manufactured high-confidence fault alarms from pure noise.
A random-perturbed input reached 99.99% on the stop-machine class. Softmax
confidence is not evidence of in-distribution-ness. This gate adds an explicit
abstain path so noise/garbage de-escalates instead of escalating.

HOW IT DISCRIMINATES
--------------------
The single most discriminating cheap feature is **spectral flatness** (Wiener
entropy): the ratio of the geometric to the arithmetic mean of the power
spectrum. It is ~1 for a flat (white) spectrum and ->0 for a peaky one. Real
bearing vibration has strong spectral structure (shaft/cage/ball tones,
resonances), so its flatness is low; broadband noise sits near the white limit.

Empirically measured on this machine (2026-06-13, FEMTO PRONOSTIA acc_*.csv,
[2,2560] windows, flatness computed per channel on the DC-removed raw signal
then averaged):

    white Gaussian noise (10 draws)   flatness 0.566 +/- 0.010   crest 3.72
    uniform broadband noise           flatness 0.551
    saturated/clipped (+/-rail, 92%)  flatness 0.569             crest 1.03
    near-constant + 1e-4 noise        flatness 0.546
    FEMTO Bearing1_1 early (first 5)  flatness 0.231 +/- 0.008   crest 3.55
    FEMTO Bearing1_1 near-failure     flatness 0.195 +/- 0.020   crest 7.94
    FEMTO Bearing1_3 early (first 5)  flatness 0.275 +/- 0.003   crest 3.92
    FEMTO Bearing1_3 near-failure     flatness 0.055 +/- 0.004   crest 6.59
    pure tone (sine)                  flatness 0.000             (PASS, in-dist-like)

Real FEMTO windows top out at flatness ~0.275 across the whole life cycle;
broadband noise floors at ~0.551. The gap is wide. The flatness threshold is
set at 0.45 — comfortably between the two populations, biased toward false
negatives (letting a borderline-real signal through) over false positives
(abstaining on a real bearing), because a missed abstain on real data only
costs us the normal classifier path, whereas a spurious abstain on real data
would suppress a genuine alarm.

NOTE on crest factor: white noise (3.72) overlaps FEMTO early-life (3.55), so
crest is USELESS as a noise discriminator and is NOT used for that. It is used
only to catch **saturation/clipping**, where a railed signal has crest -> 1.

These thresholds are operational defaults, overridable via the ``OODConfig``
constructor or the ``AION_OOD_*`` environment variables, and they should be
re-tuned per deployment if the accelerometer model or mounting changes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Default thresholds. See module docstring for the empirical basis.
# Flatness >= this => spectrum too white to be bearing vibration.
DEFAULT_FLATNESS_MAX = 0.45
# Crest factor <= this => signal is clipped/railed (saturated sensor).
DEFAULT_CREST_MIN = 1.5
# Per-channel std (raw units) <= this => quasi-constant / near-dead trace.
# Set well above the preprocessing stuck-threshold (1e-7) so this gate is a
# softer, earlier net; truly stuck channels are still rejected by validate_signal.
DEFAULT_STD_MIN = 1e-5


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class OODConfig:
    """Thresholds for the plausibility gate (operational defaults)."""

    flatness_max: float = DEFAULT_FLATNESS_MAX
    crest_min: float = DEFAULT_CREST_MIN
    std_min: float = DEFAULT_STD_MIN

    @classmethod
    def from_env(cls) -> OODConfig:
        """Build a config, overriding any threshold from AION_OOD_* env vars."""
        return cls(
            flatness_max=_env_float("AION_OOD_FLATNESS_MAX", DEFAULT_FLATNESS_MAX),
            crest_min=_env_float("AION_OOD_CREST_MIN", DEFAULT_CREST_MIN),
            std_min=_env_float("AION_OOD_STD_MIN", DEFAULT_STD_MIN),
        )


@dataclass(frozen=True)
class OODResult:
    """Outcome of the plausibility gate for a single window."""

    ood_flag: bool          # True => input is implausible as bearing vibration
    ood_score: float        # spectral flatness in [0, 1]; higher = more noise-like
    ood_reason: str | None  # human-readable reason, or None when plausible
    spectral_flatness: float
    crest_factor: float
    min_channel_std: float


def _spectral_flatness(signal: np.ndarray) -> float:
    """Mean per-channel spectral flatness (Wiener entropy) in [0, 1].

    Flatness = geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum).
    ~1 for a flat (white) spectrum, ->0 for a peaky (tonal/structured) one.
    Computed on the DC-removed signal; the DC bin is dropped to avoid a single
    huge component dominating the arithmetic mean and masking flatness.
    """
    if signal.ndim != 2:
        signal = np.atleast_2d(signal)
    flats: list[float] = []
    for ch in range(signal.shape[0]):
        s = signal[ch].astype(np.float64)
        s = s - s.mean()
        spec = np.abs(np.fft.rfft(s)) ** 2
        if spec.size > 1:
            spec = spec[1:]  # drop DC bin
        spec = spec + 1e-12  # guard log(0) and division by an all-zero spectrum
        geo_mean = np.exp(np.mean(np.log(spec)))
        arith_mean = float(np.mean(spec))
        flats.append(float(geo_mean / arith_mean))
    return float(np.mean(flats)) if flats else 1.0


def _crest_factor(signal: np.ndarray) -> float:
    """Minimum per-channel crest factor (peak / RMS).

    A clean impulsive vibration has a high crest factor; a clipped/saturated
    sensor rails out and its crest collapses toward 1. We take the MIN across
    channels so a single saturated channel is caught.
    """
    if signal.ndim != 2:
        signal = np.atleast_2d(signal)
    crests: list[float] = []
    for ch in range(signal.shape[0]):
        s = signal[ch].astype(np.float64)
        rms = float(np.sqrt(np.mean(s ** 2))) + 1e-12
        peak = float(np.max(np.abs(s)))
        crests.append(peak / rms)
    return float(np.min(crests)) if crests else 0.0


def _min_channel_std(signal: np.ndarray) -> float:
    if signal.ndim != 2:
        signal = np.atleast_2d(signal)
    stds = signal.astype(np.float64).std(axis=1)
    return float(np.min(stds)) if stds.size else 0.0


def check_signal_plausibility(
    signal: np.ndarray, config: OODConfig | None = None
) -> OODResult:
    """Run the heuristic plausibility gate on a RAW signal window.

    Run this on the raw [2, N] (or [N, 2]) signal BEFORE z-score/high-pass
    preprocessing — z-scoring would erase the amplitude information the crest
    and std checks rely on. The spectral-flatness check is scale-invariant, so
    it is unaffected, but the gate is specified on raw units for clarity.

    Args:
        signal: raw vibration window, shape [2, N] or [N, 2].
        config: thresholds; defaults to ``OODConfig.from_env()``.

    Returns:
        OODResult. ``ood_flag=True`` means the input is implausible as bearing
        vibration and the caller should ABSTAIN (do not escalate the action).
    """
    if config is None:
        config = OODConfig.from_env()

    arr = np.asarray(signal)
    if arr.ndim == 2 and arr.shape[0] != 2 and arr.shape[1] == 2:
        arr = arr.T

    flatness = _spectral_flatness(arr)
    crest = _crest_factor(arr)
    min_std = _min_channel_std(arr)

    reason: str | None = None
    if min_std <= config.std_min:
        reason = (
            f"quasi-constant signal (min channel std={min_std:.2e} "
            f"<= {config.std_min:.0e}): implausible as live vibration"
        )
    elif crest <= config.crest_min:
        reason = (
            f"saturated/clipped sensor (crest factor={crest:.2f} "
            f"<= {config.crest_min:.2f}): peak/RMS collapsed toward a railed signal"
        )
    elif flatness >= config.flatness_max:
        reason = (
            f"broadband/white-noise-like spectrum (spectral flatness={flatness:.3f} "
            f">= {config.flatness_max:.2f}): real bearing vibration is spectrally "
            "structured (max observed on FEMTO ~0.28); input is out-of-distribution"
        )

    return OODResult(
        ood_flag=reason is not None,
        ood_score=flatness,
        ood_reason=reason,
        spectral_flatness=flatness,
        crest_factor=crest,
        min_channel_std=min_std,
    )
