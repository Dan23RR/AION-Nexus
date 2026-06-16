"""Example 7: the FACTORY BRIDGE — a signed verdict on the factory's own bus.

The thesis (RD/18_PATH_TO_UNIGNORABLE): the one thing no incumbent ships is an
INDEPENDENT, third-party-checkable verdict that lives INSIDE the factory's own
protocol fabric. Substrate Core mints the signed certificate; ``aion_nexus.connect``
carries it onto Sparkplug B (the Unified-Namespace / MQTT language) and OPC UA
(the SCADA / historian language). An auditor pulls the certificate off the bus and
verifies it OFFLINE with the public key alone.

This runs with ZERO optional dependencies (no broker, no OPC UA server): it builds
the real Sparkplug B bytes in-process and decodes them as a third party would. With
the [factory] extra installed, the SAME payloads publish to a live MQTT broker /
OPC UA server unchanged.

    1. ISSUER certifies a decision and PUBLISHES it as a Sparkplug B payload.
    2. AUDITOR on the bus DECODES the payload and re-verifies the certificate
       offline with the public key alone                          -> trusted=True.
    3. A TAMPERED payload (hide the fault) is caught                -> trusted=False.
    4. HONESTY GATE: an ABSTAIN with an 'advanced' guess does NOT raise an alarm
       — severity is floored and OPC UA Quality = Uncertain.
    5. OPC UA Alarms & Conditions view of the certified verdict.

Run:
    python examples/07_factory_bridge.py
"""
from __future__ import annotations

import json

import numpy as np

from aion_nexus.connect import (
    QUALITY_UNCERTAIN,
    build_condition_model,
    build_sparkplug_payload,
    decode_payload,
    sparkplug_topic,
    to_factory_verdict,
)
from aion_nexus.connect.sparkplug import NDATA
from aion_nexus.verify import (
    Verifier,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)

CLASS_NAMES = ["normal", "early", "medium", "advanced"]


def _calibrated_verifier() -> Verifier:
    """Calibrate a conformal verifier on a TINY synthetic set (placeholder).

    HONESTY: synthetic probabilities are NOT exchangeable with any real bearing,
    so this calibration is not valid for deployment — it only makes the certify
    API runnable without a checkpoint. The signature/audit trail are real; the
    coverage NUMBER from this toy calibrator is a placeholder.
    """
    rng = np.random.default_rng(123)
    probs, labels = [], []
    for cls in range(len(CLASS_NAMES)):
        for _ in range(12):
            logits = rng.standard_normal(len(CLASS_NAMES))
            logits[cls] += 2.0                       # make class `cls` likely
            p = np.exp(logits) / np.exp(logits).sum()
            probs.append(p)
            labels.append(cls)
    v = Verifier(alpha=0.1, class_names=list(CLASS_NAMES))
    v.calibrate(np.vstack(probs), np.array(labels, dtype=int))
    return v


def _probs_for(target: int, sharpness: float = 6.0) -> np.ndarray:
    """A confident probability vector peaked on `target`."""
    logits = np.full(len(CLASS_NAMES), 0.0)
    logits[target] = sharpness
    return np.exp(logits) / np.exp(logits).sum()


def main() -> int:
    verifier = _calibrated_verifier()

    # The issuer's seed is the SECRET minting authority; only the public key ships.
    seed = generate_seed()
    issuer_pub = ed25519_pubkey_from_seed(seed)

    # ---- 1. ISSUER certifies 'advanced' and publishes it as Sparkplug B -------
    cert = verifier.certify(
        _probs_for(3), model_id="aion-nexus-demo", seed=seed,
        ttl_seconds=3600, key_id="line7-key-1")
    topic = sparkplug_topic("Plant1", NDATA, "edge-line7", "bearing-3")
    payload = build_sparkplug_payload(cert, seq=1)
    print("--- 1. Issuer published a CERTIFIED verdict on the factory bus ---")
    print(f"  topic:   {topic}")
    print(f"  payload: {len(payload)} bytes of Sparkplug B protobuf")
    print(f"  verdict: {cert.verdict}  auth: {cert.authentication}  "
          f"pubkey: {issuer_pub[:20]}...")

    # ---- 2. AUDITOR on the bus decodes + verifies OFFLINE ---------------------
    decoded = decode_payload(payload)
    metrics = {m["name"]: m["value"] for m in decoded["metrics"]}
    print("\n--- 2. An auditor on the bus reads the metrics ---")
    print(f"  AION/verdict={metrics['AION/verdict']}  "
          f"AION/condition_state={metrics['AION/condition_state']}  "
          f"AION/severity={metrics['AION/severity']}")
    cert_on_bus = json.loads(metrics["AION/certificate"])
    audit = verify_certificate(cert_on_bus, expected_pubkey=issuer_pub)
    print(f"  verify with PUBLIC KEY ALONE -> trusted={audit['trusted']} "
          f"({audit['authenticity']})")
    assert audit["trusted"] is True
    print("  -> The verdict on the factory's own bus is independently verifiable.")

    # ---- 3. TAMPER the payload (hide the fault) -> caught ---------------------
    cert_on_bus["predicted_name"] = "normal"   # attacker downgrades 'advanced'
    t = verify_certificate(cert_on_bus, expected_pubkey=issuer_pub)
    print("\n--- 3. A tampered verdict (advanced -> normal) is caught ---")
    print(f"  integrity_ok={t['integrity_ok']}  trusted={t['trusted']}")
    assert t["trusted"] is False
    print("  -> Hiding the fault breaks the content hash; the bus can't be spoofed.")

    # ---- 4. HONESTY GATE: an ABSTAIN never screams ALARM ---------------------
    abstain_cert = verifier.certify(
        np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES)),   # flat -> ambiguous/abstain
        model_id="aion-nexus-demo", seed=seed, ttl_seconds=3600)
    # Force the abstain case explicitly for the demo regardless of the toy qhat:
    abstain_view = to_factory_verdict({
        **abstain_cert.as_dict(), "verdict": "ABSTAIN",
        "predicted_name": "advanced", "conformal_set_names": CLASS_NAMES})
    print("\n--- 4. Honesty gate: ABSTAIN with an 'advanced' guess ---")
    print(f"  condition_state={abstain_view.condition_state}  "
          f"severity={abstain_view.severity}  active={abstain_view.active}  "
          f"actionable={abstain_view.actionable}")
    assert abstain_view.severity < 700 and abstain_view.active is False
    print("  -> No machine-stop alarm from an uncertain reading. It tells the truth.")

    # ---- 5. OPC UA Alarms & Conditions view ----------------------------------
    model = build_condition_model(cert)
    c = model["condition"]
    print("\n--- 5. OPC UA Alarms & Conditions view (CERTIFIED 'advanced') ---")
    print(f"  ActiveState={c['ActiveState']}  Severity={c['Severity']}  "
          f"Quality={c['QualityName']}  Retain={c['Retain']}")
    abstain_oa = build_condition_model(abstain_view)["condition"]
    print(f"  (ABSTAIN -> Quality={abstain_oa['QualityName']}, "
          f"Severity={abstain_oa['Severity']}: OPC UA's own 'I am not sure')")
    assert abstain_oa["Quality"] == QUALITY_UNCERTAIN

    print("\nAll checks passed. The signed verdict lives in the factory's own "
          "protocol fabric (Sparkplug B / OPC UA), is third-party verifiable, and "
          "stays honest when the model does not know.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
