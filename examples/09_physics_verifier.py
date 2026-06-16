"""Example 9: the PHYSICS front-end + model-agnostic second-opinion verifier.

The architecture leap (2026-06-16): bearing models collapse cross-machine largely
because they read RAW time-domain windows and IGNORE the physics already available
(shaft rpm + bearing geometry). This shows the two fixes, end-to-end, with no
checkpoint and no training:

    1. RPM-INVARIANCE BY CONSTRUCTION. The same fault at two speeds lands at the
       SAME order (geometry-only) but DIFFERENT Hz — so the order domain is the
       leakage-robust representation, for free.
    2. THE SECOND OPINION. physics_consistency() asks what no learned model answers
       about itself — "is the energy actually at the claimed defect order on THIS
       machine?" — and composes with the conformal certificate so a model that is
       confidently wrong cross-machine is CAUGHT (CERTIFIED + CONTRADICT -> REVIEW).

HONESTY (6.31): order tracking removes the cross-SPEED (same-machine) shift
deterministically; it does NOT by itself cross the cross-MACHINE wall (placement,
transmission path, artificial-vs-real damage remain). The value is the honest
second opinion, not a magic accuracy jump. Synthetic signals, for demonstration.

Run:
    python examples/09_physics_verifier.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus.physics import (
    FAULT_INNER,
    FAULT_OUTER,
    BearingGeometry,
    envelope_spectrum,
    order_spectrum,
    physics_consistency,
)
from aion_nexus.verify import compose_certificates

FS = 25_600
SKF_6205 = BearingGeometry(n_rolling_elements=9, ball_diameter=7.94, pitch_diameter=39.04)


def _fault_signal(fr_hz, fault_order, *, duration_s=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(FS * duration_s)
    t = np.arange(n) / FS
    sig = 0.05 * rng.standard_normal(n)
    for t0 in np.arange(0.0, duration_s, 1.0 / (fault_order * fr_hz)):
        tt = t - t0
        idx = tt >= 0
        sig[idx] += np.exp(-800.0 * tt[idx]) * np.sin(2 * np.pi * 3000 * tt[idx])
    return sig


def _peak(x, y, floor):
    m = x > floor
    return float(x[m][np.argmax(y[m])])


def main() -> int:
    bpfo = SKF_6205.fault_orders()[FAULT_OUTER]
    print(f"SKF 6205 fault ORDERS (geometry only, rpm-independent): "
          f"{ {k: round(v, 3) for k, v in SKF_6205.fault_orders().items()} }")

    # ---- 1. RPM invariance ----------------------------------------------------
    sig_1500 = _fault_signal(25.0, bpfo, seed=1)   # 1500 rpm
    sig_2400 = _fault_signal(40.0, bpfo, seed=2)   # 2400 rpm
    o1, s1 = order_spectrum(sig_1500, FS, rpm=1500)
    o2, s2 = order_spectrum(sig_2400, FS, rpm=2400)
    f1, a1 = envelope_spectrum(sig_1500, FS)
    f2, a2 = envelope_spectrum(sig_2400, FS)
    print("\n--- 1. The SAME outer-race fault at two speeds ---")
    print(f"  Hz domain   : peak {_peak(f1, a1, 10):.1f} Hz (1500 rpm) vs "
          f"{_peak(f2, a2, 10):.1f} Hz (2400 rpm)  -> DIFFERENT (a time-domain model sees two signals)")
    print(f"  ORDER domain: peak {_peak(o1, s1, 0.5):.2f} vs {_peak(o2, s2, 0.5):.2f}  "
          f"(BPFO={bpfo:.2f})  -> SAME, rpm-invariant by construction")

    # ---- 2. Second opinion: physics agrees with a correct model claim --------
    print("\n--- 2. Model claims 'outer' on an outer-race signal ---")
    v_ok = physics_consistency(sig_1500, FS, rpm=1500, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    print(f"  physics: {v_ok.verdict} ({v_ok.detail})")
    composed_ok = compose_certificates(
        [{"verdict": "CERTIFIED", "assurance": "empirical"}, v_ok.as_component()], op="and")
    print(f"  model CERTIFIED + physics {v_ok.verdict} -> composed: {composed_ok['verdict']}")

    # ---- 3. Second opinion: physics CATCHES a confidently-wrong model --------
    print("\n--- 3. Model claims 'outer' but the bearing has an INNER-race fault ---")
    sig_inner = _fault_signal(30.0, SKF_6205.fault_orders()[FAULT_INNER], seed=3)
    v_bad = physics_consistency(sig_inner, FS, rpm=1800, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    print(f"  physics: {v_bad.verdict} ({v_bad.detail})")
    composed_bad = compose_certificates(
        [{"verdict": "CERTIFIED", "assurance": "empirical"}, v_bad.as_component()], op="and")
    print(f"  model CERTIFIED + physics {v_bad.verdict} -> composed: {composed_bad['verdict']}  "
          "(the cross-machine failure CAUGHT — routed to a human, not certified)")

    # ---- 4. No physics -> honest abstention, never a silent pass --------------
    v_none = physics_consistency(sig_1500, FS, rpm=None, geometry=SKF_6205, claimed_fault=FAULT_OUTER)
    print(f"\n--- 4. No rpm/geometry -> {v_none.verdict} (the check is unavailable, not a pass) ---")

    assert composed_ok["verdict"] == "CERTIFIED"
    assert composed_bad["verdict"] != "CERTIFIED"
    print("\nThe physics is RPM-invariant by construction and is a model-agnostic second "
          "opinion that composes with the signed certificate. Honest scope: it closes the "
          "cross-SPEED gap deterministically and CATCHES disagreement; it is not a "
          "cross-machine accuracy cure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
