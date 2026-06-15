"""Compliance evidence mapping — the regulatory-evidence layer of Substrate Core.

A :class:`~aion_nexus.verify.certificate.Certificate` is a re-runnable record of
one decision. This module reads such a record and maps its fields to the
*technical evidence* they can supply toward specific regulatory obligations
(EU AI Act, ISO 13381-1, ISO/IEC 42001).

Read this before you quote anything from here
---------------------------------------------
This module PROVIDES EVIDENCE TOWARD obligations. It does **not** make a system
"compliant", and nothing it emits should be read as a compliance declaration or
a third-party conformity assessment. Whether a deployment is high-risk under the
EU AI Act, and which obligations actually bind, depends on the *use context* and
on a formal assessment performed by the deployer/operator and (where required) a
notified body. The certificate is an input to that process, not a substitute for
it. See :doc:`../docs/COMPLIANCE_MAPPING` for the full disclaimer and table.

Every mapped item carries three fields, and the third is mandatory:

- ``provides_evidence_for`` — the obligation the field speaks to, phrased as
  "provides evidence toward ..." (never "complies with").
- ``how`` — *which* certificate fields supply the evidence and how.
- ``limitation`` — what this evidence does **not** cover. This is never empty:
  if a control has no honest gap we still state the residual risk, because an
  evidence item with no stated limit is exactly the overclaim we refuse to ship.

Honesty constraints baked in (the brand is honesty):
- Conformal coverage is valid ONLY under exchangeability; cross-bearing /
  cross-machine serving breaks it. Every accuracy/robustness item says so.
- Tamper-evidence exists ONLY with an HMAC key set. Without it the record is an
  integrity hash, NOT tamper-evident against an adversary holding the source.
  The record-keeping item reads ``authentication`` and says which regime applies.
- The OOD signal-plausibility gate lives UPSTREAM of the verifier and is not a
  certificate field; the certificate alone does not prove the gate ran. We say
  so rather than implying the certificate carries it.
"""
from __future__ import annotations

from typing import Any

# The exact strings this module must never emit in deployer-facing evidence.
# A test asserts none of these appear in generated text; keep the language as
# "provides evidence toward" / "supports" / "maps to" instead.
_FORBIDDEN_CLAIMS = ("compliant", "conforme", "certified compliant", "conformità")

# Framework identifiers (kept as data so the card and the dict agree).
FRAMEWORK_EU_AI_ACT = "EU AI Act (Regulation (EU) 2024/1689)"
FRAMEWORK_ISO_13381 = "ISO 13381-1:2025"
FRAMEWORK_ISO_42001 = "ISO/IEC 42001:2023"


def _as_record(certificate: Any) -> dict:
    """Accept a Certificate, its ``as_dict()``, or a plain dict; return a dict."""
    if hasattr(certificate, "as_dict"):
        return certificate.as_dict()
    return dict(certificate)


def _auth_regime(record: dict) -> tuple[str, str]:
    """Return (regime_label, honest_description) from the ``authentication`` field.

    No silent assumption of tamper-evidence: only an HMAC-keyed record is
    forgery-resistant; everything else is an integrity hash.
    """
    auth = str(record.get("authentication", "NONE"))
    if auth == "HMAC-SHA256":
        return (
            "HMAC-SHA256 (keyed)",
            "records are HMAC-SHA256 signed: tamper-evident against an adversary "
            "who does not hold the key",
        )
    return (
        "NONE (integrity hash only)",
        "records carry a SHA-256 integrity hash but NO keyed signature: this "
        "proves internal consistency, NOT tamper-evidence against an adversary "
        "holding this source. Set VERIFY_HMAC_KEY to make the trail "
        "tamper-evident",
    )


def compliance_evidence(certificate: Any) -> dict:
    """Map one certificate to the regulatory evidence it can supply.

    Accepts a :class:`~aion_nexus.verify.certificate.Certificate`, its
    ``as_dict()``, or a compatible plain dict. Returns a structured dict::

        {
          "disclaimer": str,                 # the strong non-compliance caveat
          "certificate_ref": {...},          # cert_id / timestamp / verdict / auth
          "evidence": [ {                    # one entry per mapped obligation
              "framework": str,
              "reference": str,              # e.g. "Art. 12"
              "title": str,
              "provides_evidence_for": str,  # "provides evidence toward ..."
              "how": str,                    # which fields supply it
              "limitation": str,            # NON-EMPTY: what it does NOT cover
          }, ... ],
          "overall_limitation": str,
        }

    The returned text never asserts compliance; it states what the certificate
    *evidences* and, for every item, what it does not.
    """
    record = _as_record(certificate)
    auth_label, auth_desc = _auth_regime(record)
    verdict = str(record.get("verdict", "?"))
    has_input_hash = bool(record.get("input_sha256"))

    evidence: list[dict] = [
        {
            "framework": FRAMEWORK_EU_AI_ACT,
            "reference": "Art. 12",
            "title": "Record-keeping / automatic logging",
            "provides_evidence_for": (
                "provides evidence toward automatic, traceable record-keeping of "
                "each decision over the system's lifetime"
            ),
            "how": (
                "each decision emits a Certificate with a unique cert_id, a UTC "
                "timestamp, the input fingerprint input_sha256, and a "
                "content_hash over the decision payload; CertificateStore appends "
                "these to a hash-linked JSONL log (append-only) so the event "
                "sequence is reconstructable. Authentication regime: "
                f"{auth_label} — {auth_desc}."
            ),
            "limitation": (
                "logging captures the decision record only; it does NOT log the "
                "full input signal (only its SHA-256), nor upstream "
                "data-acquisition or operator context. "
                + (
                    "Because input_sha256 is absent on this record, the log does "
                    "NOT bind this decision to a specific input. "
                    if not has_input_hash else ""
                )
                + (
                    "Tamper-evidence is NOT in force on this record "
                    "(authentication=NONE): the log proves consistency but not "
                    "authenticity without VERIFY_HMAC_KEY."
                    if auth_label.startswith("NONE")
                    else "Retention duration and storage integrity over the legal "
                    "retention period are the deployer's responsibility."
                )
            ),
        },
        {
            "framework": FRAMEWORK_EU_AI_ACT,
            "reference": "Art. 14",
            "title": "Human oversight",
            "provides_evidence_for": (
                "provides evidence toward effective human oversight by routing "
                "uncertain decisions to a human instead of acting automatically"
            ),
            "how": (
                "the verdict field encodes a routing decision: CERTIFIED (singleton "
                "conformal set above the confidence floor) is safe to act on, "
                "REVIEW (conformal set with more than one label) is routed to human "
                "review, and ABSTAIN (below the confidence floor) withholds an "
                f"automatic action. This certificate's verdict is {verdict!r}. The "
                "verdict is bound into content_hash, so the routing decision shown "
                "cannot silently diverge from the one certified."
            ),
            "limitation": (
                "the system MAKES the human-in-the-loop hand-off available and "
                "records it; it does NOT prove a human actually reviewed the case, "
                "nor that the reviewer had the competence, authority or time to "
                "override. The existence and effectiveness of the downstream review "
                "process is an organisational control the deployer must implement "
                "and evidence separately."
            ),
        },
        {
            "framework": FRAMEWORK_EU_AI_ACT,
            "reference": "Art. 15",
            "title": "Accuracy, robustness and cybersecurity",
            "provides_evidence_for": (
                "provides evidence toward declared accuracy and robustness via a "
                "stated coverage target with an explicit validity condition, plus "
                "an upstream input-plausibility gate"
            ),
            "how": (
                "the certificate records the conformal miscoverage level alpha and "
                "the calibrated quantile qhat, so the declared coverage target "
                "(1 - alpha) is auditable per decision; the ABSTAIN verdict adds a "
                "confidence floor; and an out-of-distribution signal-plausibility "
                "gate (check_signal_plausibility) screens implausible inputs before "
                "inference. On the cybersecurity side, the HMAC chain (when keyed) "
                "evidences integrity of the decision log."
            ),
            "limitation": (
                "the conformal coverage guarantee holds ONLY under exchangeability "
                "of calibration and serving data; cross-bearing / cross-machine "
                "deployment breaks exchangeability and VOIDS the marginal "
                "1 - alpha guarantee — the declared target is then aspirational, "
                "not guaranteed. The OOD gate runs UPSTREAM of the verifier and is "
                "NOT a field of this certificate, so the certificate alone does NOT "
                "prove the gate ran on this input. Accuracy figures from a "
                "benchmark do not transfer to a different machine, sensor or regime "
                "without re-validation."
            ),
        },
        {
            "framework": FRAMEWORK_ISO_13381,
            "reference": "Clause 7 (prognostic stages)",
            "title": "Condition monitoring / prognostics stage reporting",
            "provides_evidence_for": (
                "provides evidence toward a documented prognostic-stage output with "
                "a quantified uncertainty set, supporting a prognostics process"
            ),
            "how": (
                "the predicted_name and the conformal_set_names report the estimated "
                "degradation stage together with the set of stages the coverage "
                "target cannot rule out, giving a stage estimate with an explicit "
                "uncertainty band rather than a bare point label."
            ),
            "limitation": (
                "the stages are POSITIONAL degradation/RUL stages, NOT fault-type "
                "diagnoses; a stage label does not name the failure mode. ISO "
                "13381-1 prognostics also expect remaining-useful-life estimates "
                "with confidence over a time horizon — this certificate reports a "
                "stage at a point in time, not a calibrated RUL trajectory. Full "
                "conformance requires the deployer's broader prognostic procedure."
            ),
        },
        {
            "framework": FRAMEWORK_ISO_42001,
            "reference": "AI management system",
            "title": "Traceability artefact for AI governance",
            "provides_evidence_for": (
                "provides evidence toward an AI management system by supplying a "
                "per-decision, re-runnable artefact (model_id, schema_version, "
                "auditable hash) that an AIMS can reference for traceability"
            ),
            "how": (
                "the certificate names the model (model_id), pins the record format "
                "(schema_version), and is independently re-verifiable via "
                "verify_certificate, so decisions can be tied to a model version and "
                "audited after the fact within a management system."
            ),
            "limitation": (
                "a single artefact is NOT a management system: ISO/IEC 42001 "
                "requires organisational scope, risk treatment, roles, monitoring "
                "and continual improvement. This module supplies one input to such "
                "a system and certifies nothing about the surrounding processes, "
                "data governance, or lifecycle management."
            ),
        },
    ]

    return {
        "disclaimer": (
            "This is NOT a declaration of conformity nor a third-party conformity "
            "assessment under the EU AI Act. It supplies technical evidence and "
            "traceability that SUPPORT the deployer/operator's own compliance "
            "process. High-risk classification and the final set of obligations "
            "depend on the use context and on a formal assessment."
        ),
        "certificate_ref": {
            "cert_id": record.get("cert_id"),
            "timestamp_utc": record.get("timestamp_utc"),
            "verdict": verdict,
            "authentication": record.get("authentication", "NONE"),
            "content_hash": record.get("content_hash"),
            "model_id": record.get("model_id"),
            "schema_version": record.get("schema_version"),
        },
        "evidence": evidence,
        "overall_limitation": (
            "Evidence is per-decision and technical. It does not assess the "
            "deployment context, does not establish high-risk classification, and "
            "does not replace the deployer's formal conformity process or any "
            "required third-party assessment."
        ),
    }


def evidence_card(certificate: Any) -> str:
    """Render :func:`compliance_evidence` as a human-readable plain-text / Markdown card.

    The card is deployer-facing and, by contract, never asserts compliance: it
    states what the certificate evidences and, for each item, what it does not.
    """
    data = compliance_evidence(certificate)
    ref = data["certificate_ref"]

    lines: list[str] = []
    lines.append("# Regulatory evidence card — Substrate Core certificate")
    lines.append("")
    lines.append(f"> {data['disclaimer']}")
    lines.append("")
    lines.append("## Certificate reference")
    lines.append("")
    lines.append(f"- cert_id: `{ref.get('cert_id')}`")
    lines.append(f"- timestamp_utc: `{ref.get('timestamp_utc')}`")
    lines.append(f"- verdict: **{ref.get('verdict')}**")
    lines.append(f"- authentication: `{ref.get('authentication')}`")
    lines.append(f"- model_id: `{ref.get('model_id')}`")
    lines.append(f"- content_hash: `{ref.get('content_hash')}`")
    lines.append("")
    lines.append("## Evidence mapping")
    lines.append("")

    for item in data["evidence"]:
        lines.append(f"### {item['framework']} — {item['reference']}: {item['title']}")
        lines.append("")
        lines.append(f"- Provides evidence for: {item['provides_evidence_for']}")
        lines.append(f"- How: {item['how']}")
        lines.append(f"- Limitation (NOT covered): {item['limitation']}")
        lines.append("")

    lines.append("## Overall limitation")
    lines.append("")
    lines.append(data["overall_limitation"])
    lines.append("")
    return "\n".join(lines)
