"""Physics-grounded, RPM-invariant features + a model-agnostic second-opinion verifier.

Why this module exists (the architecture leap, 2026-06-16)
----------------------------------------------------------
A field-wide research sweep converged on one diagnosis: bearing-fault models
collapse cross-machine (FEMTO F1 0.884 in-distribution -> 0.352 leave-one-bearing-
out) largely because they learn from RAW time-domain windows and **ignore the
physics that is already available** — the shaft speed (rpm) and the bearing
geometry. A fault's vibration signature is rpm-COUPLED: the ball-pass frequencies
scale linearly with shaft speed, so the SAME fault at a different speed looks like
a different signal to a time-domain network. AION's models accept ``rpm`` and
``geometry`` in ``forward()`` and **discard them** — the single most quantifiable,
cheapest-to-fix, most verifier-aligned gap.

This module closes it with classical DSP and turns the physics into TWO things:

1. **An RPM-invariant representation.** In the ORDER domain (multiples of shaft
   rotation) the characteristic fault frequencies are FIXED constants that depend
   only on geometry, NOT on speed (see :meth:`BearingGeometry.fault_orders`). So an
   order-domain (squared-)envelope spectrum places a fault peak at the same order
   regardless of rpm — invariance *by construction*, not learned. This is the most
   leakage-robust input the field has measured (envelope SES >> time-domain under
   bearing-diversity stress).

2. **A model-agnostic second opinion (the verifier play).** :func:`physics_consistency`
   asks a question no learned model answers about itself: *is the spectral energy
   actually at the claimed defect order on THIS machine?* It does not care which
   model produced the claim — AION's, a customer's, or a foundation model's — so it
   composes with :func:`aion_nexus.verify.compose_certificates` as an independent
   check. A model that is confidently wrong on a new machine (the cross-machine
   failure mode) is caught when its claim and the physics disagree.

HONESTY (workspace 6.31). Order tracking removing the rpm (same-machine, cross-
SPEED) shift is deterministic DSP — near-certain. It does NOT by itself cross the
cross-MACHINE wall (sensor placement, transmission path, artificial-vs-real damage
remain); the literature puts physics-only LOBO around ~0.5-0.6, not a sellable
>0.85. The value is exactly that honesty: a physics check that CONFIRMS or
CONTRADICTS a verdict, and abstains when the evidence is weak, is worth more than a
black-box number — and the assurance tier here is EMPIRICAL (a thresholded SNR
heuristic), never a proof. ``scipy`` is required (a core dependency).

Kinematics reference (standard rolling-element bearing fault frequencies):
    fr   = rpm / 60                                    shaft rotation freq (Hz)
    BPFO = (n/2)·fr·(1 - (d/D)·cosφ)                   outer race  (Hz)
    BPFI = (n/2)·fr·(1 + (d/D)·cosφ)                   inner race  (Hz)
    BSF  = (D/2d)·fr·(1 - ((d/D)·cosφ)²)               rolling element (Hz)
    FTF  = (1/2)·fr·(1 - (d/D)·cosφ)                   cage        (Hz)
In ORDERS (divide by fr) the fr factor drops out -> geometry-only constants.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Fault families. Names are the physical defect location, NOT AION's positional
# degradation stages — this module is fault-type physics, used as a check.
FAULT_OUTER = "outer"   # BPFO
FAULT_INNER = "inner"   # BPFI
FAULT_BALL = "ball"     # BSF (often appears at 2xBSF in the envelope)
FAULT_CAGE = "cage"     # FTF

# Physics verdicts (kept distinct from the model's CERTIFIED/REVIEW/ABSTAIN).
PHYS_CONFIRM = "CONFIRM"            # energy clearly at the claimed/dominant fault order
PHYS_WEAK = "WEAK"                  # some energy, below the confidence threshold
PHYS_CONTRADICT = "CONTRADICT"      # claimed fault order is NOT where the energy is
PHYS_INDETERMINATE = "INDETERMINATE"  # no rpm/geometry, or signal too short -> cannot check


@dataclass(frozen=True)
class BearingGeometry:
    """Rolling-element bearing kinematics.

    ``ball_diameter`` (d) and ``pitch_diameter`` (D) only ever enter as the ratio
    d/D, so any consistent length unit is fine. ``contact_angle_deg`` (φ) is 0 for
    a deep-groove ball bearing.
    """

    n_rolling_elements: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_rolling_elements < 1:
            raise ValueError("n_rolling_elements must be >= 1")
        if not (0 < self.ball_diameter < self.pitch_diameter):
            raise ValueError("require 0 < ball_diameter < pitch_diameter")
        if not (-90.0 < self.contact_angle_deg < 90.0):
            raise ValueError("contact_angle_deg must be in (-90, 90)")

    @property
    def _ratio(self) -> float:
        return (self.ball_diameter / self.pitch_diameter) * math.cos(
            math.radians(self.contact_angle_deg))

    def fault_orders(self) -> dict[str, float]:
        """Characteristic fault frequencies in ORDERS (multiples of shaft rotation).

        These depend ONLY on geometry — NOT on rpm — which is the whole point: in
        the order domain a fault peak sits at a fixed location at every speed.
        """
        n = self.n_rolling_elements
        r = self._ratio
        return {
            FAULT_OUTER: (n / 2.0) * (1.0 - r),
            FAULT_INNER: (n / 2.0) * (1.0 + r),
            FAULT_BALL: (self.pitch_diameter / (2.0 * self.ball_diameter)) * (1.0 - r * r),
            FAULT_CAGE: 0.5 * (1.0 - r),
        }

    def fault_frequencies(self, rpm: float) -> dict[str, float]:
        """Characteristic fault frequencies in Hz at a given shaft speed (rpm)."""
        fr = shaft_frequency(rpm)
        return {k: v * fr for k, v in self.fault_orders().items()}


def shaft_frequency(rpm: float) -> float:
    """Shaft rotation frequency in Hz."""
    if rpm <= 0:
        raise ValueError("rpm must be > 0")
    return float(rpm) / 60.0


# --------------------------------------------------------------------------- #
# Envelope / order-domain spectra (the RPM-invariant representation)
# --------------------------------------------------------------------------- #

def _as_1d(signal: np.ndarray) -> np.ndarray:
    """Reduce a [C, N] or [N] signal to a 1-D analysis channel (mean over channels)."""
    arr = np.asarray(signal, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    if arr.ndim != 1:
        raise ValueError("signal must be 1-D [N] or 2-D [channels, N]")
    if arr.size < 16:
        raise ValueError("signal too short for envelope analysis")
    return arr


def envelope(signal: np.ndarray, fs: float, band: tuple[float, float] | None = None
             ) -> np.ndarray:
    """Amplitude envelope via the Hilbert transform (optionally band-passed first).

    Band-passing around a structural resonance before demodulation is the classic
    squared-envelope recipe; without a known resonance the full-band envelope still
    exposes the periodic impact modulation. The envelope is mean-removed so its
    spectrum shows the modulation lines, not a DC spike.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    x = _as_1d(signal)
    if band is not None:
        lo, hi = band
        nyq = fs / 2.0
        lo = max(lo, 1.0)
        hi = min(hi, nyq * 0.999)
        if lo < hi:
            sos = butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
            x = sosfiltfilt(sos, x)
    env = np.abs(hilbert(x))
    return env - env.mean()


def envelope_spectrum(signal: np.ndarray, fs: float,
                      band: tuple[float, float] | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(freqs_hz, amplitude) of the envelope — the squared-envelope spectrum (SES) magnitude."""
    env = envelope(signal, fs, band=band)
    n = env.size
    spec = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def order_resample(signal: np.ndarray, fs: float, rpm_profile: np.ndarray,
                   samples_per_rev: int = 64) -> np.ndarray:
    """Computed order tracking: resample a (possibly speed-varying) signal to a
    uniform SHAFT-ANGLE grid, so fault frequencies become rpm-invariant orders.

    ``rpm_profile`` is the instantaneous shaft speed per input sample (same length
    as the signal, or a scalar wrapped to that). The cumulative shaft angle
    ``theta(t) = ∫ fr dt`` is integrated, then the signal is interpolated onto
    equally-spaced angle points (``samples_per_rev`` per revolution). For constant
    rpm this reduces to plain resampling; the value is under VARYING rpm, where it
    de-smears peaks that a fixed-time FFT would blur.
    """
    x = _as_1d(signal)
    rpm_arr = np.atleast_1d(np.asarray(rpm_profile, dtype=np.float64))
    if rpm_arr.size == 1:
        rpm_arr = np.full(x.size, float(rpm_arr[0]))
    if rpm_arr.size != x.size:
        raise ValueError("rpm_profile must be a scalar or match the signal length")
    if np.any(rpm_arr <= 0):
        raise ValueError("rpm_profile must be strictly positive")
    fr = rpm_arr / 60.0                              # instantaneous shaft freq (Hz)
    dt = 1.0 / fs
    theta = np.concatenate([[0.0], np.cumsum(fr[:-1] * dt)])  # revolutions vs sample
    total_rev = float(theta[-1])
    n_out = max(samples_per_rev, int(total_rev * samples_per_rev))
    even_theta = np.linspace(0.0, total_rev, n_out, endpoint=False)
    return np.interp(even_theta, theta, x)


def order_spectrum(signal: np.ndarray, fs: float, rpm: float | np.ndarray,
                   band: tuple[float, float] | None = None,
                   samples_per_rev: int = 64
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Envelope spectrum with the x-axis in ORDERS (cycles per shaft revolution).

    Constant ``rpm`` (scalar): compute the Hz envelope spectrum and rescale the axis
    by 1/fr — the fault peak lands at its geometry-only order. Varying ``rpm``
    (array): angular-resample first (:func:`order_resample`), so the de-smeared
    envelope spectrum is natively in orders. Either way the peak location is
    rpm-INVARIANT — the property the whole leap rests on.
    """
    if np.isscalar(rpm) or (np.ndim(rpm) == 0):
        fr = shaft_frequency(float(rpm))
        freqs, spec = envelope_spectrum(signal, fs, band=band)
        return freqs / fr, spec
    # Speed-varying: resample to angle, demodulate (envelope), then FFT in orders.
    from scipy.signal import hilbert

    resampled = order_resample(signal, fs, rpm, samples_per_rev=samples_per_rev)
    env = np.abs(hilbert(resampled))
    env = env - env.mean()
    n = env.size
    spec = np.abs(np.fft.rfft(env * np.hanning(n))) / n
    orders = np.fft.rfftfreq(n, d=1.0 / samples_per_rev)  # cycles per revolution
    return orders, spec


# --------------------------------------------------------------------------- #
# Fault-order energy + the model-agnostic consistency verifier
# --------------------------------------------------------------------------- #

def _harmonic_snr(orders: np.ndarray, spec: np.ndarray, target: float, *,
                  tol: float, n_harmonics: int, local: float = 0.4) -> float:
    """Order-SNR of a fault family: geometric mean of its harmonics' LOCAL prominence.

    For each harmonic ``h·target`` the peak amplitude within ``±tol`` is divided by
    the LOCAL background (median amplitude in a ``±local``-order neighbourhood,
    EXCLUDING the peak window). Using a local background (not the global median) is
    what stops white noise scoring high — a noise spike is not prominent above its
    own neighbourhood. The GEOMETRIC MEAN across harmonics rewards a genuine
    harmonic comb (BPFO, 2·BPFO, 3·BPFO all present, as a real fault produces) and
    punishes a single isolated spike, so noise — which has no aligned comb — stays
    near 1.
    """
    snrs: list[float] = []
    top = orders[-1]
    for h in range(1, n_harmonics + 1):
        centre = target * h
        if centre >= top:
            break
        peak_mask = np.abs(orders - centre) <= tol
        if not np.any(peak_mask):
            continue
        peak = float(spec[peak_mask].max())
        local_mask = (np.abs(orders - centre) <= local) & (~peak_mask) & (orders > 0.1)
        bg = float(np.median(spec[local_mask])) if np.any(local_mask) else float(np.median(spec))
        snrs.append(peak / max(bg, 1e-12))
    if not snrs:
        return 0.0
    return float(np.exp(np.mean(np.log(np.maximum(snrs, 1e-6)))))


def fault_order_energy(signal: np.ndarray, fs: float, rpm: float | np.ndarray,
                       geometry: BearingGeometry, *, n_harmonics: int = 3,
                       tol: float = 0.05, band: tuple[float, float] | None = None
                       ) -> dict[str, float]:
    """Order-SNR (dimensionless) of each fault family at its characteristic order.

    For each of BPFO/BPFI/BSF/FTF, the local-prominence harmonic SNR (see
    :func:`_harmonic_snr`) — a speed-independent, geometry-aware feature that is high
    only when a genuine harmonic comb sits at that family's order. Higher = more
    evidence of that fault; white noise stays near 1 for every family.
    """
    orders, spec = order_spectrum(signal, fs, rpm, band=band)
    return {fault: _harmonic_snr(orders, spec, order, tol=tol, n_harmonics=n_harmonics)
            for fault, order in geometry.fault_orders().items()}


@dataclass(frozen=True)
class PhysicsVerdict:
    """Result of the model-agnostic physics second opinion."""

    verdict: str                       # CONFIRM | WEAK | CONTRADICT | INDETERMINATE
    dominant_fault: str | None         # fault family with the strongest order-SNR
    scores: dict[str, float]           # per-fault order-SNR (speed-invariant)
    claimed_fault: str | None          # the fault the upstream model claimed, if any
    detail: str
    # Assurance tier for composition: a thresholded SNR heuristic is EMPIRICAL,
    # never a proof — it composes (weakest-link) with a conformal certificate.
    assurance: str = "empirical"

    def as_component(self) -> dict:
        """A dict shaped for :func:`aion_nexus.verify.compose_certificates`.

        Maps the physics verdict onto the verdict vocabulary so a physics check can
        compose with a conformal certificate: CONFIRM -> CERTIFIED, WEAK/INDETERMINATE
        -> ABSTAIN (no escalation on weak physics), CONTRADICT -> REVIEW (a real
        disagreement a human must see — NOT silently certified).
        """
        mapping = {PHYS_CONFIRM: "CERTIFIED", PHYS_WEAK: "ABSTAIN",
                   PHYS_INDETERMINATE: "ABSTAIN", PHYS_CONTRADICT: "REVIEW"}
        return {"verdict": mapping[self.verdict], "assurance": self.assurance,
                "source": "physics", "detail": self.detail}


def physics_consistency(signal: np.ndarray, fs: float, rpm: float | np.ndarray | None,
                        geometry: BearingGeometry | None, *,
                        claimed_fault: str | None = None,
                        confirm_snr: float = 4.0, weak_snr: float = 2.0,
                        n_harmonics: int = 3, tol: float = 0.04,
                        band: tuple[float, float] | None = None) -> PhysicsVerdict:
    """A model-agnostic second opinion: is the energy at the claimed/dominant fault order?

    This is the verifier play. It asks a question no learned classifier answers
    about itself, using ONLY physics (rpm + geometry) — so it checks ANY model's
    claim (AION's, a customer's, a foundation model's) on THIS machine.

    - Without ``rpm``/``geometry`` it cannot run -> ``INDETERMINATE`` (honest: the
      physics check is unavailable, never a silent pass).
    - With a ``claimed_fault``: if that fault's order-SNR clears ``confirm_snr`` and
      is the dominant family -> ``CONFIRM``; if the energy is clearly at a DIFFERENT
      family -> ``CONTRADICT`` (the cross-machine failure mode caught); else ``WEAK``.
    - Without a claim: reports the dominant family and CONFIRM/WEAK by its SNR.

    Returns a :class:`PhysicsVerdict`; use :meth:`PhysicsVerdict.as_component` to
    compose it with a conformal certificate via ``compose_certificates``.
    """
    if rpm is None or geometry is None:
        return PhysicsVerdict(
            verdict=PHYS_INDETERMINATE, dominant_fault=None, scores={},
            claimed_fault=claimed_fault,
            detail="no rpm/geometry supplied: physics check unavailable (not a pass)")

    scores = fault_order_energy(signal, fs, rpm, geometry,
                                n_harmonics=n_harmonics, tol=tol, band=band)
    dominant = max(scores, key=scores.get)
    dom_snr = scores[dominant]

    if claimed_fault is not None:
        claim_snr = scores.get(claimed_fault, 0.0)
        if claim_snr >= confirm_snr and claimed_fault == dominant:
            v = PHYS_CONFIRM
            detail = (f"claimed '{claimed_fault}' confirmed: order-SNR {claim_snr:.1f} "
                      f">= {confirm_snr} and dominant")
        elif dom_snr >= confirm_snr and dominant != claimed_fault and \
                claim_snr < weak_snr:
            v = PHYS_CONTRADICT
            detail = (f"claimed '{claimed_fault}' (SNR {claim_snr:.1f}) but energy is at "
                      f"'{dominant}' (SNR {dom_snr:.1f}): physics disagrees with the model")
        elif claim_snr >= weak_snr:
            v = PHYS_WEAK
            detail = (f"claimed '{claimed_fault}' weakly supported: order-SNR "
                      f"{claim_snr:.1f} in [{weak_snr}, {confirm_snr})")
        else:
            v = PHYS_WEAK
            detail = (f"claimed '{claimed_fault}' not supported (SNR {claim_snr:.1f}) and "
                      f"no dominant fault clears {confirm_snr}: inconclusive")
        return PhysicsVerdict(v, dominant, scores, claimed_fault, detail)

    # No claim: report what the physics sees.
    if dom_snr >= confirm_snr:
        v, detail = PHYS_CONFIRM, f"dominant fault '{dominant}' at order-SNR {dom_snr:.1f}"
    elif dom_snr >= weak_snr:
        v, detail = PHYS_WEAK, f"weak '{dominant}' at order-SNR {dom_snr:.1f}"
    else:
        v, detail = PHYS_WEAK, f"no fault order stands out (max SNR {dom_snr:.1f})"
    return PhysicsVerdict(v, dominant, scores, None, detail)
