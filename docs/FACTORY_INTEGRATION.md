# Factory integration — a signed verdict in the factory's own protocol fabric

`aion_nexus.connect` (v2.7.0) carries the Substrate Core **signed certificate**
onto the buses the factory already runs: **Sparkplug B** (the MQTT / Unified
Namespace language) and **OPC UA** (the SCADA / historian language). The point is
narrow and unignorable:

> A Siemens / Beckhoff / SKF integrator pulls AION's certificate off **their own**
> MQTT broker or OPC UA server and verifies it **offline with the public key
> alone**. No incumbent ships independent, third-party-checkable verification
> inside the factory fabric — they have no incentive to verify themselves.

This is the missing half of the thesis in
[`AION_NEXUS_RD/18_PATH_TO_UNIGNORABLE.md`](../../AION_NEXUS_RD/18_PATH_TO_UNIGNORABLE.md):
Substrate Core (v2.5/2.6) mints the certificate; the bridge (v2.7) makes it live
where the incumbents already are.

## What it is — and is NOT (workspace 6.31)

| Claim | Reality |
|---|---|
| Sparkplug B payloads | A faithful **Sparkplug B 3.0 `Payload`** protobuf: `timestamp`, `seq`, and `Metric`s with `Int64/Double/Boolean/String/Bytes` values. Wire-compatibility is **proven against Google's reference protobuf decoder** in `tests/test_connect.py`. NOT a full Eclipse Tahu reimplementation: `DataSet`/`Template`/`PropertySet` metric values and metadata are not emitted. |
| OPC UA | A **mapping onto OPC UA Alarms & Conditions (Part 9)** semantics (`ActiveState`, `Severity`, `Message`, `Quality`, `Retain`) plus AION extension variables, suitable for placement under an OPC UA for Machinery / Condition Monitoring machine node. NOT a certified companion-spec conformance. |
| The verdict | **Advisory** verification of a decision record. A valid certificate attests *this verifier, on this input, reached this verdict, untampered*. It is NOT a real-time safety interlock and does NOT prove the prediction is correct. |

## The honesty gate (the actual differentiator)

The diagnosis class (`normal…advanced`) and the **trust verdict**
(`CERTIFIED / REVIEW / ABSTAIN`) are different axes. An incumbent's alarm says
"advanced, severity 900" with no notion of its own uncertainty. The bridge
**refuses to**: severity is **capped by the trust verdict**, so an uncertain or
ambiguous reading can never present as a machine-stop alarm.

| Trust verdict | Sparkplug `AION/severity` / `condition_state` | OPC UA `Severity` / `Quality` / `ActiveState` |
|---|---|---|
| `CERTIFIED` | diagnosis-mapped (normal 1 → advanced 900) | diagnosis-mapped / **Good** / Active when a fault is present |
| `REVIEW` (ambiguous set) | capped to the warning band → `REVIEW` | capped / Good / `Retain=True` for human review |
| `ABSTAIN` (OOD / low confidence) | floored to 100 → `DATA_UNCERTAIN`, `active=false` | 100 / **Uncertain** / **not** an active alarm |
| any UNSIGNED cert | `actionable=false` | **Uncertain** (not tamper-evident) |

`ABSTAIN → OPC UA Quality = Uncertain` is the clean mapping: OPC UA already has a
vocabulary for "I am not sure", and the verdict uses it. The bus tells the truth
when the model does not know.

## Dependency posture

Building payloads/models is **dependency-free** — the in-package protobuf codec
(`aion_nexus/connect/_protobuf.py`) has no runtime dependency. The live transports
import their optional dependencies lazily, only when you connect/start:

```bash
pip install "aion-nexus[factory-mqtt]"    # Sparkplug B over MQTT (paho-mqtt; permissive licence)
pip install "aion-nexus[factory-opcua]"   # OPC UA live server (asyncua; LGPL-3.0 — see SECURITY.md)
pip install "aion-nexus[factory]"         # both
```

The split lets a licence-sensitive operator take only the permissive MQTT path.
`asyncua` is **LGPL-3.0**, an optional/dynamically-imported dependency outside the
Apache/BSD/MIT core guarantee — verify it against your policy before enabling the
OPC UA extra.

## Quickstart (no broker needed)

```python
from aion_nexus.verify import Verifier, generate_seed, ed25519_pubkey_from_seed, verify_certificate
from aion_nexus.connect import build_sparkplug_payload, decode_payload
import json

# 1. Issuer mints a SIGNED verdict (seed = secret minting authority).
seed = generate_seed()
cert = verifier.certify(probs, seed=seed, ttl_seconds=3600, key_id="line7-key-1")

# 2. Publish it as real Sparkplug B bytes (or use SparkplugPublisher for a live broker).
payload = build_sparkplug_payload(cert, seq=1)

# 3. A third party on the bus re-verifies it OFFLINE with the public key alone.
metrics = {m["name"]: m["value"] for m in decode_payload(payload)["metrics"]}
res = verify_certificate(json.loads(metrics["AION/certificate"]),
                         expected_pubkey=ed25519_pubkey_from_seed(seed))
assert res["trusted"] is True        # tamper any field -> trusted=False
```

Full runnable walkthrough (publish → decode-off-the-bus → verify → tamper →
honesty gate → OPC UA view): [`examples/07_factory_bridge.py`](../examples/07_factory_bridge.py).

## Live transports

### Sparkplug B over MQTT (Unified Namespace)

```python
from aion_nexus.connect import SparkplugPublisher

pub = SparkplugPublisher("Plant1", "edge-line7", device_id="bearing-3").connect("broker.local", 1883)
pub.birth()                          # NBIRTH/DBIRTH so the UNS establishes state
topic = pub.publish_verdict(cert)    # NDATA/DDATA: spBv1.0/Plant1/DDATA/edge-line7/bearing-3
pub.disconnect()
```

The publisher maintains the Sparkplug `seq` (wrapping 0–255), sets an `NDEATH`
last-will, and publishes the full signed certificate as the `AION/certificate`
metric.

### OPC UA Alarms & Conditions server

```python
from aion_nexus.connect import CertifiedConditionMonitoringServer

srv = CertifiedConditionMonitoringServer(endpoint="opc.tcp://0.0.0.0:4840")
await srv.start()
await srv.update(cert)               # refresh AionVerification node variables
# ... a SCADA client subscribes; reads the Certificate variable; verifies offline
await srv.stop()
```

## The AION Sparkplug metric set

| Metric | Type | Meaning |
|---|---|---|
| `AION/verdict` | String | `CERTIFIED` / `REVIEW` / `ABSTAIN` (trust) |
| `AION/health_class` | String | diagnosed bearing class (`normal…advanced`) |
| `AION/condition_state` | String | `NORMAL/ADVISORY/WARNING/ALARM/REVIEW/DATA_UNCERTAIN` |
| `AION/severity` | Int64 | OPC UA-scale severity 1–1000 (trust-capped) |
| `AION/active`, `AION/actionable`, `AION/verifiable` | Boolean | honest gates (`actionable` ⇔ CERTIFIED **and** signed) |
| `AION/assurance` | String | assurance tier (conformal ⇒ `empirical`, never proven) |
| `AION/conformal_set` | String | coverage-controlled label set |
| `AION/content_hash`, `AION/authentication`, `AION/pubkey`, `AION/key_id`, `AION/valid_until` | String | verification handles |
| `AION/certificate` | String | **the full signed certificate JSON** — what an auditor verifies |

## Standards context

- **Sparkplug B 3.0** (Eclipse Tahu) — MQTT payload + topic namespace for the UNS.
- **OPC UA Part 9** Alarms & Conditions; **OPC UA for Machinery / Condition
  Monitoring** companion spec — the placement target for the AION nodes.
- **ISO 13374** OSA-CBM: AION sits at *Advisory Generation*, outside the safety
  loop (see [`docs/COMPLIANCE_MAPPING.md`](COMPLIANCE_MAPPING.md)), which is why
  an OPC UA condition is advisory and the bridge never drives an autonomous stop.
