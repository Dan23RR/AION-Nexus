"""Tests for the physics front-end + model-agnostic second-opinion verifier.

The headline tests prove the two claims the architecture leap rests on:
  * test_kinematics_match_published_skf_6205 — the fault-order formulas reproduce
    the published SKF 6205 characteristic orders (3.585 / 5.415 / 2.357 / 0.398).
  * test_order_spectrum_is_rpm_invariant — the SAME fault at two different speeds
    lands at the SAME order (rpm-invariant BY CONSTRUCTION) while its Hz peak moves.
    This is the deterministic property the leap is built on.
  * the consistency verifier CONFIRMs a matching claim, CONTRADICTs a wrong claim
    (the cross-machine failure mode caught), and is INDETERMINATE without physics.
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.physics import (
    FAULT_INNER,
    FAULT_OUTER,
    PHYS_CONFIRM,
    PHYS_CONTRADICT,
    PHYS_INDETERMINATE,
    PHYS_WEAK,
    BearingGeometry,
    envelope_spectrum,
    fault_order_energy,
    order_spectrum,
    physics_consistency,
    shaft_frequency,
)

FS = 25_600
# SKF 6205 deep-groove ball bearing: 9 balls, ball d=7.94 mm, pitch D=39.04 mm, phi=0.
SKF_6205 = BearingGeometry(n_rolling_elements=9, ball_diameter=7.94, pitch_diameter=39.04)


def _bearing_fault_signal(fr_hz: float, fault_order: float, *, duration_s: float = 1.0,
                          resonance_hz: float = 3000.0, decay: float = 800.0,
                          amp: float = 1.0, noise: float = 0.05, seed: int = 0) -> np.ndarray:
    """Classic bearing-fault model: an impulse train at ``fault_order * fr`` where
    each impact rings a structural resonance (decaying sinusoid), plus noise.
    The envelope demodulates the resonance to reveal the fault frequency."""
    rng = np.random.default_rng(seed)
    n = int(FS * duration_s)
    t = np.arange(n) / FS
    sig = noise * rng.standard_normal(n)
    f_fault = fault_order * fr_hz
    period = 1.0 / f_fault
    for t0 in np.arange(0.0, duration_s, period):
        tt = t - t0
        idx = tt >= 0
        sig[idx] += amp * np.exp(-decay * tt[idx]) * np.sin(2 * np.pi * resonance_hz * tt[idx])
    return sig


def _dominant(x: np.ndarray, y: np.ndarray, floor: float) -> float:
    """x-location of the largest spectral peak above ``floor`` (skips near-DC)."""
    mask = x > floor
    return float(x[mask][np.argmax(y[mask])])


# --------------------------------------------------------------------------- #
# 1. Kinematics — reproduce published SKF 6205 fault orders
# --------------------------------------------------------------------------- #

def test_kinematics_match_published_skf_6205():
    orders = SKF_6205.fault_orders()
    assert orders[FAULT_OUTER] == pytest.approx(3.585, abs=0.01)   # BPFO
    assert orders[FAULT_INNER] == pytest.approx(5.415, abs=0.01)   # BPFI
    assert orders["ball"] == pytest.approx(2.357, abs=0.01)        # BSF
    assert orders["cage"] == pytest.approx(0.398, abs=0.01)        # FTF
    # BPFO + BPFI must equal n * (mean) — internal identity: BPFO+BPFI = n.
    assert orders[FAULT_OUTER] + orders[FAULT_INNER] == pytest.approx(9.0, abs=1e-9)


def test_fault_frequencies_scale_with_rpm():
    f1500 = SKF_6205.fault_frequencies(1500)
    f3000 = SKF_6205.fault_frequencies(3000)
    for k in f1500:
        assert f3000[k] == pytest.approx(2 * f1500[k], rel=1e-9)
    # BPFO at 1500 rpm = 3.585 * 25 Hz = 89.6 Hz
    assert f1500[FAULT_OUTER] == pytest.approx(3.585 * shaft_frequency(1500), abs=0.1)


def test_geometry_validation():
    with pytest.raises(ValueError):
        BearingGeometry(n_rolling_elements=0, ball_diameter=1, pitch_diameter=5)
    with pytest.raises(ValueError):
        BearingGeometry(n_rolling_elements=8, ball_diameter=10, pitch_diameter=5)


# --------------------------------------------------------------------------- #
# 2. THE headline: the order spectrum is RPM-invariant by construction
# --------------------------------------------------------------------------- #

def test_order_spectrum_is_rpm_invariant():
    bpfo_order = SKF_6205.fault_orders()[FAULT_OUTER]   # 3.585
    fr_lo, fr_hi = 25.0, 40.0                            # 1500 vs 2400 rpm

    sig_lo = _bearing_fault_signal(fr_lo, bpfo_order, seed=1)
    sig_hi = _bearing_fault_signal(fr_hi, bpfo_order, seed=2)

    # Order domain: the SAME fault lands at the SAME order at both speeds.
    o_lo, s_lo = order_spectrum(sig_lo, FS, rpm=fr_lo * 60)
    o_hi, s_hi = order_spectrum(sig_hi, FS, rpm=fr_hi * 60)
    peak_order_lo = _dominant(o_lo, s_lo, floor=0.5)
    peak_order_hi = _dominant(o_hi, s_hi, floor=0.5)
    assert peak_order_lo == pytest.approx(bpfo_order, abs=0.1)
    assert peak_order_hi == pytest.approx(bpfo_order, abs=0.1)
    assert abs(peak_order_lo - peak_order_hi) < 0.1     # rpm-INVARIANT

    # Hz domain: the SAME fault lands at DIFFERENT frequencies (the problem we fix).
    f_lo, a_lo = envelope_spectrum(sig_lo, FS)
    f_hi, a_hi = envelope_spectrum(sig_hi, FS)
    peak_hz_lo = _dominant(f_lo, a_lo, floor=10.0)
    peak_hz_hi = _dominant(f_hi, a_hi, floor=10.0)
    assert peak_hz_lo == pytest.approx(bpfo_order * fr_lo, abs=3.0)
    assert peak_hz_hi == pytest.approx(bpfo_order * fr_hi, abs=3.0)
    # The Hz peaks differ by the speed ratio — a time-domain model sees two signals.
    assert peak_hz_hi / peak_hz_lo == pytest.approx(fr_hi / fr_lo, rel=0.1)


def test_order_resample_handles_varying_rpm():
    # A linear speed sweep 1500 -> 1800 rpm. Impacts are placed at EXACT BPFO-order
    # positions in the ANGLE domain (every 1/BPFO revolutions), each ringing a
    # resonance — so a fixed-time FFT would smear the peak, but order tracking must
    # still recover a clean BPFO peak.
    bpfo_order = SKF_6205.fault_orders()[FAULT_OUTER]
    n = FS  # 1 s
    rng = np.random.default_rng(0)
    rpm_profile = np.linspace(1500.0, 1800.0, n)
    fr = rpm_profile / 60.0
    rev = np.cumsum(fr) / FS                       # cumulative shaft revolutions per sample
    t = np.arange(n) / FS
    sig = 0.05 * rng.standard_normal(n)
    ring_len = 512
    ring = np.exp(-800.0 * t[:ring_len]) * np.sin(2 * np.pi * 3000 * t[:ring_len])
    for impact_rev in np.arange(0.0, rev[-1], 1.0 / bpfo_order):
        i = int(np.searchsorted(rev, impact_rev))
        if i + ring_len <= n:
            sig[i:i + ring_len] += ring

    orders, spec = order_spectrum(sig, FS, rpm=rpm_profile)
    peak = _dominant(orders, spec, floor=0.5)
    assert peak == pytest.approx(bpfo_order, abs=0.3)


# --------------------------------------------------------------------------- #
# 3. Fault-order energy + the model-agnostic consistency verifier
# --------------------------------------------------------------------------- #

def test_fault_order_energy_localises_outer_race():
    sig = _bearing_fault_signal(30.0, SKF_6205.fault_orders()[FAULT_OUTER], seed=3)
    scores = fault_order_energy(sig, FS, rpm=1800, geometry=SKF_6205)
    assert scores[FAULT_OUTER] == max(scores.values())
    assert scores[FAULT_OUTER] > 4.0                       # clears the confirm threshold
    assert scores[FAULT_OUTER] > 2 * scores[FAULT_INNER]   # well-localised


def test_consistency_confirms_a_matching_claim():
    sig = _bearing_fault_signal(30.0, SKF_6205.fault_orders()[FAULT_OUTER], seed=4)
    v = physics_consistency(sig, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    assert v.verdict == PHYS_CONFIRM
    assert v.dominant_fault == FAULT_OUTER
    assert v.as_component()["verdict"] == "CERTIFIED"


def test_consistency_contradicts_a_wrong_claim():
    # An INNER-race signal, but the model claimed OUTER -> physics must CONTRADICT.
    sig = _bearing_fault_signal(30.0, SKF_6205.fault_orders()[FAULT_INNER], seed=5)
    v = physics_consistency(sig, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    assert v.verdict == PHYS_CONTRADICT
    assert v.dominant_fault == FAULT_INNER
    assert v.as_component()["verdict"] == "REVIEW"          # disagreement -> human, not certified


def test_consistency_weak_on_white_noise():
    sig = np.random.default_rng(6).standard_normal(FS)
    v = physics_consistency(sig, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    assert v.verdict in (PHYS_WEAK, PHYS_CONTRADICT)        # never CONFIRM on noise
    assert v.verdict != PHYS_CONFIRM


def test_consistency_indeterminate_without_physics():
    sig = _bearing_fault_signal(30.0, 3.585, seed=7)
    assert physics_consistency(sig, FS, rpm=None, geometry=SKF_6205).verdict == PHYS_INDETERMINATE
    assert physics_consistency(sig, FS, rpm=1800, geometry=None).verdict == PHYS_INDETERMINATE


# --------------------------------------------------------------------------- #
# 4. Composition with the conformal certificate (the verifier play)
# --------------------------------------------------------------------------- #

def test_physics_composes_with_certificate():
    from aion_nexus.verify import compose_certificates

    # Model is CERTIFIED on 'outer'; physics CONTRADICTs -> the composed system must
    # NOT remain CERTIFIED (the weakest link governs; a disagreement drops to REVIEW).
    sig = _bearing_fault_signal(30.0, SKF_6205.fault_orders()[FAULT_INNER], seed=8)
    phys = physics_consistency(sig, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    model_cert = {"verdict": "CERTIFIED", "assurance": "empirical"}
    composed = compose_certificates([model_cert, phys.as_component()], op="and")
    assert composed["verdict"] in ("REVIEW", "ABSTAIN")
    assert composed["verdict"] != "CERTIFIED"

    # When physics CONFIRMS, the composition can stay CERTIFIED.
    sig_ok = _bearing_fault_signal(30.0, SKF_6205.fault_orders()[FAULT_OUTER], seed=9)
    phys_ok = physics_consistency(sig_ok, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    composed_ok = compose_certificates([model_cert, phys_ok.as_component()], op="and")
    assert composed_ok["verdict"] == "CERTIFIED"
