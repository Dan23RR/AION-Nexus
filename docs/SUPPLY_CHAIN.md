# Supply-Chain Security Posture — AION-NEXUS

> The verification certificate is only worth as much as the binary that produced
> it. After `xz`/CVE-2024-3094, an OT buyer is right to ask: *how do I know the
> AION-NEXUS you shipped is the one you built?* This document is the honest
> answer — it separates what is **automated today** from what is a **documented
> recommendation** not yet wired into CI.

---

## TL;DR — what is real today

| Control | Status | Where |
|---|---|---|
| Dependency version bounds (tested upper bounds) | **Automated** | `requirements.txt`, `pyproject.toml` |
| CVE scan (pip-audit), **fail-closed** in CI | **Automated** | `scripts/audit_supply_chain.py --strict`, `.github/workflows/ci.yml` |
| License allow/block-list scan | **Automated** | `scripts/audit_supply_chain.py` |
| CycloneDX SBOM generated + uploaded as artifact per CI run | **Automated** | `scripts/generate_sbom.py`, CI `supply-chain-audit` job |
| Hash-pinned lockfile (`requirements.lock`, `--generate-hashes`) | **Best-effort in CI** (skipped if `pip-tools` unavailable) | CI step + manual command below |
| Signed SBOM / release / container (cosign, Sigstore) | **Recommendation — NOT automated** | § "Signing roadmap" below |
| SLSA provenance attestation | **Recommendation — NOT automated** | § "Signing roadmap" below |

Do **not** read this document as a claim that AION-NEXUS releases are
cryptographically signed end-to-end today. They are not yet. What *is* true: the
dependency tree is bounded, scanned fail-closed on every relevant CI run, and a
machine-readable SBOM is produced for every build.

---

## 1. Dependency pinning and bounds

Runtime dependencies live in `requirements.txt` (and mirror in
`pyproject.toml [project.dependencies]`). Every dependency carries a lower bound
(minimum tested) and an **upper bound at the next un-tested major** so a fresh
install cannot silently pull an unvetted major version into the codebase —
most importantly into the signing path:

```
cryptography>=43,<48   # we test and ship against 47.x; <48 blocks the next major
```

Bounds express *what we have tested*, not a guess. When we validate a new major,
we move the bound and re-run the full suite.

### Hash-pinned lockfile (defense against tampered/yanked artifacts)

Version bounds pin *which version*; they do **not** pin *which bytes*. To pin the
exact artifact hashes — so a re-published or tampered wheel on the index is
rejected at install time — generate a lockfile with hashes:

```bash
pip install pip-tools
pip-compile --generate-hashes --output-file requirements.lock requirements.txt
```

Install from it with hash enforcement:

```bash
pip install --require-hashes -r requirements.lock
```

`--require-hashes` makes pip refuse any artifact whose SHA-256 does not match the
locked hash, and refuses to install anything not present in the lockfile.

CI attempts to regenerate `requirements.lock` on the `supply-chain-audit` job and
uploads it as an artifact. That step is **best-effort**: if `pip-tools` is
unavailable it is skipped (not failed), and the lockfile is currently *generated*
and *published as an artifact* rather than *committed and enforced on every
install*. Enforcing `--require-hashes` in the production Docker build is the next
hardening step (see roadmap).

---

## 2. SBOM (Software Bill of Materials)

`scripts/generate_sbom.py` emits a **CycloneDX 1.5 JSON** document
(`sbom.cyclonedx.json`) listing every installed component with its version,
`pkg:pypi/...` package URL, and best-effort license.

It has two paths:

1. **`cyclonedx-bom` tool** (`cyclonedx-py env`) when installed — richer,
   spec-complete output.
2. **Pure-stdlib fallback** walking `importlib.metadata` — no third-party
   dependency required, so it always runs (including in a minimal CI step).

```bash
python scripts/generate_sbom.py                    # -> sbom.cyclonedx.json
python scripts/generate_sbom.py --out build/sbom.json
python scripts/generate_sbom.py --force-fallback   # force the stdlib path
```

**Honest limitation of the fallback:** the stdlib document is *minimal but
valid*. It does **not** include the dependency graph (edges), per-artifact
hashes, or fully-resolved SPDX license expressions for every package. For a
release-grade SBOM install `cyclonedx-bom` (the script uses it automatically).

The SBOM lets a customer feed AION-NEXUS into their own
Dependency-Track / Grype / Trivy pipeline and continuously re-scan our component
list against newly-disclosed CVEs — independent of us.

---

## 3. Fail-closed CVE + license audit

`scripts/audit_supply_chain.py` wraps `pip-audit` (CVEs) and `pip-licenses`
(license allow/block-list).

The important property is the **`--strict` gate used in CI is fail-closed**:

- **Without `--strict`** (local dev): missing tooling downgrades to a `WARN` and
  the run can still pass; only *found* vulnerabilities or blocked licenses fail.
- **With `--strict`** (CI): `pip-audit` **must** be installed and **must** run.
  If it is missing, times out, or returns unparsable output, the audit
  **fails** (non-zero exit) instead of passing with a silent warning. Any
  vulnerability or blocked/unknown license also fails.

This closes the previous fail-open hole, where a missing scanner returned a
silent warning and CI went green having never checked for CVEs.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | PASS — checks ran, nothing found |
| `1` | FAIL — real vulnerabilities and/or license issues found |
| `2` | WARN — non-strict, a check could not run (tool missing); not a hard fail |
| `3` | FAIL (strict) — the CVE scanner could not run; refused to certify clean |

**Honesty note on severity:** `pip-audit`'s default JSON report does not carry a
CVSS severity field, so the audit cannot reliably filter "HIGH only". The
deliberate choice is conservative for a security gate: in `--strict`, **any**
known vulnerability fails the build. If you need severity-graded triage, run
`pip-audit` with an OSV/vulnerability-database that returns severity and adjust
the gate — but a stricter gate is the safer default and is what ships.

```bash
python scripts/audit_supply_chain.py            # permissive (local)
python scripts/audit_supply_chain.py --strict   # fail-closed (CI)
python scripts/audit_supply_chain.py --strict --out supply_chain_report.json
```

### Known open finding (as of 2026-06-15)

Running `pip-audit -r requirements.txt` resolves `torch` to the newest version
inside our `>=2.0.0,<3.0.0` bound (`2.12.0`) and reports **CVE-2025-3000**, for
which `pip-audit` currently lists **no fix version** inside that major. This is
a real, open finding — not hidden:

- The fail-closed `--strict` gate **will** flag it (this is the gate working as
  designed, not a false positive).
- There is no patched `torch` in the `<3.0.0` range to pin to yet. Until one
  ships, the honest options are: (a) accept the risk with a documented,
  time-boxed `pip-audit --ignore-vuln CVE-2025-3000` waiver reviewed at each
  release, or (b) move to a fixed `torch` once published and update the bound.
  We have **not** added a blanket ignore — the gate stays red until a decision
  is recorded per release.

(Separately, a `pip-audit` scan of an *already-installed* CPU build such as
`torch 2.11.0+cpu` may report clean; the version pip-audit resolves from the
spec and the version actually installed can differ. The lockfile in § 1 removes
this ambiguity by pinning the exact resolved artifact.)

---

## 4. CI wiring

The `supply-chain-audit` job in `.github/workflows/ci.yml` runs on the weekly
cron and on push to `main`. It:

1. Installs runtime deps + `pip-audit`, `pip-licenses`, `cyclonedx-bom`.
2. Generates `sbom.cyclonedx.json` and uploads it as an artifact.
3. Best-effort generates and uploads `requirements.lock` (skipped if `pip-tools`
   is unavailable — non-fatal).
4. Runs `audit_supply_chain.py --strict` (fail-closed) and uploads the report.
5. Runs raw `pip-audit` for the human-readable log.

If any tracked CVE appears, the job goes red — by design.

---

## 5. Signing roadmap — recommended, NOT yet automated

These are the controls a post-`xz` OT buyer should ask for. We document the
intended flow honestly: **none of the steps in this section are automated in CI
today.** They are the recommended next increments.

### 5.1 Sign the SBOM and release artifacts (cosign / Sigstore)

Keyless signing with Sigstore binds an artifact to an OIDC identity and logs it
in the public Rekor transparency log:

```bash
# Sign the SBOM (keyless, OIDC identity recorded in Rekor)
cosign sign-blob --yes sbom.cyclonedx.json \
  --output-signature sbom.cyclonedx.json.sig \
  --output-certificate sbom.cyclonedx.json.pem

# A customer verifies it independently:
cosign verify-blob sbom.cyclonedx.json \
  --signature sbom.cyclonedx.json.sig \
  --certificate sbom.cyclonedx.json.pem \
  --certificate-identity-regexp '.*' \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

### 5.2 Sign the container image

```bash
cosign sign --yes ghcr.io/dan23rr/aion-nexus@sha256:<digest>
# Attach the SBOM as an attestation to the image:
cosign attest --yes --predicate sbom.cyclonedx.json --type cyclonedx \
  ghcr.io/dan23rr/aion-nexus@sha256:<digest>
```

### 5.3 SLSA provenance

Emit SLSA provenance for the build (e.g. via the
`slsa-framework/slsa-github-generator` reusable workflow) so a consumer can
verify *which workflow, from which commit, on which runner* produced the
artifact, and verify it with `slsa-verifier`. This raises the build to a
verifiable SLSA level rather than a claimed one.

### Why this matters (the AION-NEXUS thesis, applied to ourselves)

AION-NEXUS sells **independently verifiable** fault diagnoses: a signed
certificate a third party can check without trusting us. The same logic must
apply to our own supply chain — "verify, don't trust" turned inward. Signing our
releases is the consistency that earns OT trust. This document states plainly
where we are on that path and where we are not yet.

---

## Files

- `requirements.txt`, `pyproject.toml` — dependency bounds
- `scripts/generate_sbom.py` — CycloneDX SBOM generator (tool + stdlib fallback)
- `scripts/audit_supply_chain.py` — fail-closed CVE + license audit
- `.github/workflows/ci.yml` — `supply-chain-audit` job
