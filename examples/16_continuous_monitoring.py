"""Example 16: continuous monitoring — point-in-time certs -> a rolling SLO + drift.

The AI-assurance category watches models CONTINUOUSLY (drift, MTTD). AION already
emits per-decision certificates; the Monitor turns that stream into a live SLO
(certified/abstain rates, mean confidence) and a Population Stability Index drift
signal vs the calibration reference — flagging silent decay before labels exist.

Run:  python examples/16_continuous_monitoring.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus.monitoring import Monitor


def main() -> int:
    rng = np.random.default_rng(0)
    # calibration-time confidences (the reference the monitor compares against)
    reference = rng.beta(5, 2, 1000)
    mon = Monitor(window=400, reference_confidence=reference)

    # phase 1: a HEALTHY stream (same distribution as calibration)
    for c in rng.beta(5, 2, 400):
        mon.record(float(c), "CERTIFIED" if c > 0.8 else ("REVIEW" if c > 0.5 else "ABSTAIN"))
    s1 = mon.status()
    print(f"Phase 1 (healthy):  certified={s1['certified_rate']:.0%} "
          f"abstain={s1['abstain_rate']:.0%} drift={s1['drift_level']} "
          f"(PSI {s1['drift_psi']:.2f}) alerts={len(s1['alerts'])}")

    # phase 2: the input distribution DRIFTS (lower confidence) — silent decay
    mon.reset()
    for c in rng.beta(2, 5, 400):
        mon.record(float(c), "CERTIFIED" if c > 0.8 else ("REVIEW" if c > 0.5 else "ABSTAIN"))
    s2 = mon.status()
    print(f"Phase 2 (drifted):  certified={s2['certified_rate']:.0%} "
          f"abstain={s2['abstain_rate']:.0%} drift={s2['drift_level']} "
          f"(PSI {s2['drift_psi']:.2f}) alerts={len(s2['alerts'])}")
    for a in s2["alerts"]:
        print(f"   ALERT: {a}")

    # when delayed labels eventually arrive, the realized SLO is measurable
    realized = mon.realized_metrics(np.array([2, 3, 0, 1, 2]), np.array([0, 3, 0, 1, 2]))
    print(f"\nRealized (delayed labels): accuracy={realized['accuracy']:.2f}, "
          f"false-healthy rate={realized['false_healthy_rate']:.2f}")

    assert s1["drift_level"] == "none"
    assert s2["drift_level"] == "significant" and s2["alerts"]
    print("\nOK — the certificate stream is now a continuously-watchable SLO that flags "
          "distribution drift label-free. This is the continuous product the AI-assurance "
          "market expects, built over the existing per-decision verdicts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
