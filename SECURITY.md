# Security Policy

## Reporting a vulnerability

**Do not file public GitHub issues for security vulnerabilities.**

Email **daniel.culotta@gmail.com** with subject `[SECURITY] AION-NEXUS`.

Include:
- AION-NEXUS version (`python -c "import aion_nexus; print(aion_nexus.__version__)"`)
- Affected component (model, server, scripts, etc.)
- Reproducer (minimal code/payload)
- Impact assessment

**Response SLA**:
- Acknowledgment within **48 hours**.
- Initial triage within **7 days**.
- Coordinated disclosure within **90 days** (per industry standard).

We will credit reporters in `CHANGELOG.md` unless you prefer anonymity.

## Threat model (summary)

The detailed threat model is in [`docs/threat_model.md`](./docs/threat_model.md). Operational summary:

### What we protect against

| Threat | Mitigation |
|---|---|
| Malformed input (DoS via huge payload) | FastAPI request size limit (configurable, default 10 MB) |
| Malformed signal (NaN, stuck sensor) | `validate_signal` rejects before model sees data |
| Container privilege escalation | Non-root user `aion` in Dockerfile |
| Inbound network exposure | Container has only port 8080; no outbound required |
| Supply chain (CVE in dependencies) | `scripts/audit_supply_chain.py` weekly in CI |
| Path traversal in CSV upload | FastAPI's `UploadFile.filename` is sanitized; we never write user-uploaded paths |
| Model poisoning at inference | N/A — inference does not modify the model |

### What we do NOT protect against (out of scope)

| Threat | Why out of scope |
|---|---|
| Training-time data poisoning | We assume the operator placed the correct trained checkpoint |
| Compromised checkpoint file | Verify SHA-256 of checkpoint against canonical hash |
| Compromised host environment | OS / container runtime security is the operator's responsibility |
| Side-channel attacks (timing, power analysis) | Not applicable to industrial PdM use cases |
| Adversarial example generation | Possible in principle but no documented practical attack path; not a v1.0 priority |

### Supply chain hardening

- All dependencies pinned in `requirements.txt`.
- Dependencies use only Apache-2.0 / BSD / MIT / Python-2.0 licenses.
- No GPL or AGPL dependencies (legal compatibility).
- Run `python -m scripts.audit_supply_chain` weekly.

### Data handling

- AION-NEXUS does NOT phone home, log to cloud, or transmit signal data anywhere.
- Inference is local. Logs written to stdout (operator captures).
- The container does not require any outbound network connectivity.

### Cryptographic posture

- Checkpoint integrity: validate SHA-256 against canonical hash in `checkpoints/README.md` after download.
- **Checkpoint pinning at serve time (v2.6.0)**: set `AION_CHECKPOINT_SHA256` to the expected hash; the server refuses to start (non-degraded) if the live checkpoint differs. `AION_REQUIRE_CHECKPOINT_PIN=1` refuses to start unless the pin is set. `/health` exposes the live and expected hashes.
- Network transport: deploy behind TLS-terminating reverse proxy (nginx, traefik). The bundled FastAPI server does NOT terminate TLS — that's the operator's responsibility.

### Certificate signing keys (v2.6.0 — the Substrate Core verification layer)

`/predict_certified` signs each verdict. **The signing seed is the authority to mint
certificates — treat it like a private key.**

- **Provenance & entropy.** Generate the seed with a CSPRNG:
  `python -c "from aion_nexus.verify import generate_seed; print(generate_seed())"`
  (32 random bytes, hex). The **product boundary enforces an entropy floor**: a seed
  below 32 bytes (e.g. a memorable passphrase or PIN) is **refused with 503** at
  `/predict_certified` — a guessable seed yields a brute-forceable key. The library
  primitives keep a backward-compatible permissive default; new minting surfaces must
  pass `strict=True` or call `assert_strong_seed`.
- **Asymmetric by default.** Prefer Ed25519 (`VERIFY_ED25519_SEED`): the verifier holds
  only the **public** key (`VERIFY_ED25519_PUBKEY` / the embedded `pubkey`) and **cannot
  forge**. HMAC (`VERIFY_HMAC_KEY`) is symmetric — whoever can verify can also forge.
- **Storage & rotation.** Keep the seed out of source/images; inject via a secret manager
  at runtime. Stamp a `key_id` (`AION_CERT_KEY_ID`) on every cert so you can rotate and
  attribute. For high-assurance deployments, sign via a **KMS/HSM**: implement the
  `aion_nexus.verify.Signer` interface (it signs a message, never exposing the key) — a
  pluggable backend is the intended path; no key should ever sit in application memory.
- **Expiry / anti-replay.** Certificates carry a signed validity window
  (`AION_CERT_TTL_SECONDS`, default 24h); an expired or replayed certificate fails
  verification. The window is bound into the signature, not the `content_hash`, so
  decision reproducibility (deterministic `content_hash`) is preserved.
- **Honest default.** With no key configured, certificates declare `authentication=NONE`
  (integrity hash only, **NOT tamper-evident**) and the response carries an explicit
  `warning`. `AION_REQUIRE_SIGNED_CERT=1` refuses to emit unsigned certificates.

### Supply-chain attestation (v2.6.0)

See [`docs/SUPPLY_CHAIN.md`](./docs/SUPPLY_CHAIN.md). `scripts/generate_sbom.py` emits a
CycloneDX SBOM; `scripts/audit_supply_chain.py --strict` is **fail-closed** (non-zero exit
on a high CVE or a missing `pip-audit`). Recommended for releases: hash-pinned lockfile
(`pip-compile --generate-hashes`) and signed release/container artifacts (cosign / SLSA).

### Factory bridge dependencies (v2.7.0 — `aion_nexus.connect`)

The bridge publishes SIGNED verdicts into the factory's protocol fabric. Its
security posture is deliberately conservative:

- **No new core dependency.** Building Sparkplug B payloads and OPC UA condition
  models uses an in-package protobuf codec (`aion_nexus/connect/_protobuf.py`)
  with no runtime dependency. The Apache/BSD/MIT core guarantee is unchanged.
- **Optional transports, split by licence.** `pip install "aion-nexus[factory-mqtt]"`
  pulls **paho-mqtt** (EPL-2.0 / EDL-1.0, BSD-style — permissive).
  `pip install "aion-nexus[factory-opcua]"` pulls **asyncua** (opcua-asyncio),
  which is **LGPL-3.0** — an optional, separately-installed, dynamically-imported,
  not-bundled dependency that is intentionally **outside** the Apache/BSD/MIT core
  guarantee. Verify it against your compliance policy before enabling the OPC UA
  extra; a licence-sensitive deployment can take only the permissive MQTT path.
- **The bridge does not weaken the certificate.** It carries the existing signed
  certificate verbatim; verification is unchanged (`verify_certificate` against
  the issuer's out-of-band public key). It does not introduce a new minting
  surface, and an UNSIGNED certificate on the bus is reported `actionable=false`
  / OPC UA `Quality = Uncertain` (never silently trusted).
- **Transport security is the operator's.** Put MQTT behind TLS + broker
  authn/authz; put OPC UA behind its security policies (Sign&Encrypt, user auth).
  The bridge publishes the verdict; it does not secure the wire.

## Hardening checklist for production deployments

- [ ] Run container as non-root (Dockerfile already enforces this).
- [ ] Mount `checkpoints/` read-only.
- [ ] Place behind TLS-terminating reverse proxy.
- [ ] Add authentication (API key / mTLS / OAuth) at the proxy layer.
- [ ] Rate-limit per source IP at the proxy layer.
- [ ] Set request size limit at proxy (default container handles via FastAPI; defense in depth).
- [ ] Enable structured logging (JSON, captured to ELK / Loki / Datadog).
- [ ] Run `pip-audit` weekly and patch within 7 days for high-severity CVEs.
- [ ] Verify SHA-256 of checkpoint after each update.
- [ ] Monitor `/health` for `running_avg_latency_ms` regression > 2× baseline.

## Coordinated disclosure

If you have a vulnerability that affects production deployments, the disclosure timeline:

1. Day 0: report received, acknowledged within 48h.
2. Day 7: triage complete, severity assigned (CVSS).
3. Day 30: patch ready (for high/critical) or 60 days (medium/low).
4. Day 90: public disclosure with credit to reporter.

We follow [Google Project Zero's 90-day policy](https://googleprojectzero.blogspot.com/p/vulnerability-disclosure-policy.html) as the upper bound.

## Contact

`daniel.culotta@gmail.com` — primary security contact.
