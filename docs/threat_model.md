# Threat Model — AION-NEXUS

STRIDE-based threat analysis of the AION-NEXUS deployment. Authored from a Siemens / industrial OT engineer's perspective. Originally written for v1.0; mitigation status updated for 2.2.0 (2026-06-11).

---

## System context

```
   ┌─────────────┐    HTTPS/REST    ┌────────────────┐
   │  Industrial │ ───────────────▶ │   AION-NEXUS   │
   │   plant     │                  │   container    │
   │  (sensors,  │                  │   (FastAPI +   │
   │   SCADA,    │ ◀─────────────── │    PyTorch)    │
   │   gateway)  │   prediction     │                │
   └─────────────┘                  └────────┬───────┘
                                              │
                                              ▼
                                     ┌────────────────┐
                                     │  Checkpoint    │
                                     │   (.pth)       │
                                     │   read-only    │
                                     └────────────────┘
```

Trust boundaries:

- **Plant network ↔ AION container**: untrusted on the input side; signal is validated.
- **AION container ↔ checkpoint volume**: trusted (operator places correct file).
- **AION container ↔ outside world**: NO outbound connectivity required.

---

## STRIDE analysis

### S — Spoofing

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Attacker spoofs sensor data to trigger false alarm | Medium | Low–Medium | In-app `AION_API_KEY` (X-API-Key, since 2.2.0) and/or auth at reverse proxy (mTLS); signal validation rejects malformed data |
| Attacker spoofs OPC UA client to inject prediction | Low | Medium | OPC UA security policies (Sign + Encrypt with X.509 certs) — operator config |
| Compromised checkpoint replaces production model | Low | High | SHA-256 verification; signed releases; immutable container |

### T — Tampering

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Tampering with input signal in transit | Medium | Low | TLS at reverse proxy; signature on signed payloads optional |
| Tampering with checkpoint file at rest | Low | High | Read-only volume; opt-in `expected_sha256` check at load time (`from_checkpoint`, since 2.2.0) |
| Tampering with response in transit | Low | Low–Medium | TLS at reverse proxy |
| Memory corruption attack against PyTorch | Very Low | High | Container isolation; non-root user; bounded request size |

### R — Repudiation

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Operator denies receiving "advanced" prediction | Medium | Medium | Append-only audit log of all predictions with `(timestamp, signal_hash, prediction, model_version)` |
| Audit log tampering | Low | Medium | Send log to write-only sink (Loki / SIEM); cryptographic chaining optional |

### I — Information disclosure

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Vibration signals contain sensitive process info | Low | Low | Container does not send signals anywhere; logs do not include raw signal (only hash + prediction) |
| Model weights leak intellectual property | Medium | Low | Apache 2.0 license — weights are open by design |
| Side-channel attack (timing, power) leaks input | Very Low | Low | Inference latency is signal-content-independent (no early-exit branches on input values) |

### D — Denial of service

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Huge payload DoS | High | Low | In-app body cap `AION_MAX_BODY_BYTES` (default 10 MB → `413`, since 2.2.0); reverse proxy rate limit |
| Computational DoS via repeated `/predict_long_signal` with massive signals | Medium | Medium | Bound `n_windows` to e.g., 1000 in `/predict_long_signal` |
| Slowloris attack | Low | Low | Reverse proxy timeout |
| Resource exhaustion (memory, file descriptors) | Low | Low | Container resource limits (in `docker-compose.yml`) |
| Adversarial input that triggers worst-case latency | Low | Low | Latency is bounded by architecture, not input |

### E — Elevation of privilege

| Threat | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Container escape → host root | Low | Critical | Non-root user `aion`; minimal base image; no `--privileged` flag |
| Pickle deserialization attack via checkpoint | Medium | Critical | `weights_only=True` is the default load path (v1/v6 since 2.0; v3 substrate loader since 2.2.0); unsafe fallback requires explicit opt-in. Only accept checkpoints from trusted sources |
| Code injection via uploaded file | Low | Critical | `validate_signal` parses CSV with `np.loadtxt` (no `eval`); no shell exec on filenames |

### Critical mitigation gaps — status (updated 2026-06-11)

- [x] **DONE (2.0)** — `torch.load(..., weights_only=True)` is the default in
  `InferenceEngine.from_checkpoint`; the unsafe `weights_only=False` path requires an
  explicit opt-in flag. The v3 substrate loader got the same hardening in 2.2.0.
  (An earlier revision of this document claimed v1.0 used `weights_only=False` by default —
  that is no longer the shipped behavior.)
- [x] **DONE (2.2.0)** — SHA-256 verification of the checkpoint at load time via the opt-in
  `expected_sha256` argument (load refused on mismatch).
- [x] **DONE (2.2.0)** — structured JSON logging with per-request `request_id`
  (`AION_LOG_JSON=1`), usable as the audit-log sink to stdout.
- [ ] Cap `/predict_long_signal` `n_windows` to prevent memory exhaustion (the in-app
  `AION_MAX_BODY_BYTES` cap bounds the practical signal size since 2.2.0, but an explicit
  window cap is still open).

---

## Adversarial input considerations

Bearing fault diagnosis is not a security-sensitive ML domain (unlike, e.g., face recognition or malware detection). However, anomaly mode could be evaded by:

- **Signal smoothing**: low-pass filter the input to remove fault frequencies. Detection rate drops.
- **Frequency masking**: notch out BPFO/BPFI bands before feeding to the model.
- **Adversarial perturbations**: small ε-bounded perturbations could shift the predicted class.

These are theoretical concerns in industrial PdM. **Not in v1.0 scope** because there is no rational adversary in the threat model: the operator wants accurate predictions, not to fool the model.

If deployed in adversarial settings (e.g., contracts where downtime liability is disputed and a third party has signal access), revisit.

---

## Compliance posture

Standards alignment for industrial deployment:

| Standard | Status |
|---|---|
| **IEC 62443-4-1** (Secure product development lifecycle) | Partial. v1.0 has structured release pipeline + threat model + secure defaults; lacks formal risk assessment. v3.0 target. |
| **IEC 62443-4-2** (Component security) | Partial. Container hardening + auth via reverse proxy. Full component scoring requires audit. |
| **ISO 27001** | Out of scope for v1.0 (would require organizational ISMS). |
| **NIST SP 800-53** | Selected controls applicable: AC-3, AU-2, IA-2, SI-3 (corresponds to access control, audit, identification, malicious code protection). Not formally certified. |
| **EU AI Act** (low-risk, industrial PdM) | Aligned. PdM is generally classified as low-risk under the Act; we provide model card + transparency disclosures (per `MODEL_CARD.md`). |
| **EU NIS2 Directive** | Operator-side responsibility (we are a tool, not a service). |

---

## Operator's responsibility checklist

Things AION-NEXUS does NOT do that the operator MUST handle:

- TLS termination
- Authentication (API key / mTLS / OAuth)
- Network segmentation (deploy in OT DMZ, not IT network)
- Firewall rules (allow only from authorized SCADA/historian endpoints)
- Backup of checkpoints + audit logs
- Patch management of the host OS / container runtime
- Incident response procedures
- User training on alarm management
- Periodic recalibration of confidence thresholds
- Sensor health monitoring (orthogonal to AION-NEXUS)

This checklist is reproduced in `DEPLOYMENT.md` Hardening Checklist for actionable use.
