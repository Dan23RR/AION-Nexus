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
- Network transport: deploy behind TLS-terminating reverse proxy (nginx, traefik). The bundled FastAPI server does NOT terminate TLS — that's the operator's responsibility.

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
