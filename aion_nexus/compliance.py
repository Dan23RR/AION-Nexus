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


# --------------------------------------------------------------------------- #
# EU AI Act Annex IV technical-documentation DOSSIER (Article 11)
# --------------------------------------------------------------------------- #
# Annex IV lists the nine points the technical documentation of a high-risk AI
# system must cover. This builds an honest EVIDENCE MAP: for each point, what
# AION-NEXUS artefacts can supply, and what the provider/deployer must own. It is
# a documentation SKELETON to accelerate the provider's own dossier, NOT a filled,
# signed technical documentation and NOT a statement that the system is documented
# to the standard. The strong disclaimer and per-section limitations enforce that.

# Per-section status (honest, never "done"):
STATUS_AION_EVIDENCE = "aion-evidence-available"   # AION supplies concrete artefacts
STATUS_PARTIAL = "partial-aion-plus-deployer"      # AION supplies some; deployer completes
STATUS_DEPLOYER_OWNED = "deployer-owned"           # only the provider/deployer can supply


def _system_ref(model_metadata: dict | None, record: dict | None) -> dict:
    """Resolve the system identity from caller metadata, the cert, and the package."""
    md = dict(model_metadata or {})
    try:
        from aion_nexus.version import __version__ as _pkg_version
    except Exception:  # noqa: BLE001 - version import must never break the dossier
        _pkg_version = None
    model_id = md.get("model_id") or (record or {}).get("model_id")
    return {
        "name": md.get("name", "AION-NEXUS"),
        "version": md.get("version") or _pkg_version,
        "model_id": model_id,
        "intended_purpose": md.get("intended_purpose"),
        "provider": md.get("provider"),
        "documentation_date": md.get("documentation_date"),  # caller stamps (no clock here)
    }


def annex_iv_dossier(model_metadata: dict | None = None, *,
                     certificate: Any = None) -> dict:
    """Build an EU AI Act Annex IV technical-documentation EVIDENCE MAP.

    Maps each of the nine Annex IV points (Article 11 technical documentation for
    high-risk AI systems) to: ``aion_provides`` (the concrete artefacts this
    package supplies), ``deployer_must_supply`` (what only the provider/deployer
    can author), a ``status`` and a NON-EMPTY ``limitation``.

    ``model_metadata`` (optional) lets the caller pass known facts — ``name``,
    ``version``, ``intended_purpose``, ``provider``, ``architecture``,
    ``datasets``, ``harmonised_standards``, ``documentation_date`` — which fill the
    corresponding sections; anything absent is reported as deployer-owned rather
    than invented. ``certificate`` (optional) ties the dossier to a concrete
    decision record and threads its authentication regime into the relevant points.

    HONESTY: the returned object is a documentation SKELETON to accelerate the
    provider's own Annex IV file. It is NOT the technical documentation itself, NOT
    a declaration of conformity, and NOT a statement that the system meets Annex IV.
    ``readiness`` counts which sections have AION-supplied evidence — it is a
    preparation metric, explicitly NOT a measure of regulatory conformity.
    """
    record = _as_record(certificate) if certificate is not None else None
    ref = _system_ref(model_metadata, record)
    md = dict(model_metadata or {})
    auth_label, auth_desc = _auth_regime(record or {})

    def _md_or_deployer(key: str, deployer_text: str) -> tuple[str, bool]:
        """Return (text, supplied_by_caller) — caller metadata wins, else deployer-owned."""
        val = md.get(key)
        return (str(val), True) if val else (deployer_text, False)

    arch_text, arch_supplied = _md_or_deployer(
        "architecture",
        "the provider must document the deployed system architecture and compute "
        "resources for their integration")
    data_text, data_supplied = _md_or_deployer(
        "datasets",
        "the provider must document the provenance, scope and governance of any "
        "calibration/serving data used in their deployment")
    std_text, std_supplied = _md_or_deployer(
        "harmonised_standards",
        "the provider selects and lists the harmonised standards they formally "
        "apply; as of the documentation date the CEN-CENELEC JTC 21 harmonised "
        "standards for the AI Act were not yet cited in the Official Journal")

    sections: list[dict] = [
        {
            "number": "1", "title": "General description of the AI system",
            "annex_iv_requirement": (
                "intended purpose, provider, versions, how it interacts with "
                "hardware/software, forms of distribution, and instructions for use"),
            "aion_provides": (
                f"package identity (name {ref['name']}, version {ref['version']}); "
                "distribution as a Python package + container (Dockerfile, "
                "docker-compose.yml); instructions for use (README.md, "
                "DEPLOYMENT.md); intended-purpose framing as ADVISORY "
                "degradation-stage estimation OUTSIDE the safety loop "
                "(MODEL_CARD.md, docs/COMPLIANCE_MAPPING.md)"),
            "deployer_must_supply": (
                ref.get("intended_purpose") or
                "the provider must state the intended purpose IN THEIR deployment "
                "context and how the system interacts with their hardware/software"),
            "status": STATUS_PARTIAL,
            "limitation": (
                "the package describes the GENERIC system; the intended purpose, "
                "deployment context and operator instructions for a specific "
                "installation are the provider's to author — they determine whether "
                "the use is high-risk at all."),
        },
        {
            "number": "2", "title": "Detailed description of elements and development process",
            "annex_iv_requirement": (
                "development methods, design specifications and logic, system "
                "architecture and compute, data requirements, human oversight, "
                "predetermined changes, validation/testing, and cybersecurity"),
            "aion_provides": (
                "development methodology and reproduction steps (MODEL_CARD.md, "
                "docs/reproduce.md, docs/architecture.md); data requirements and "
                f"contract (docs/data_contract.md); architecture: {arch_text}; "
                "human-oversight logic (the CERTIFIED/REVIEW/ABSTAIN verdict "
                "routing — see compliance_evidence Art.14); validation and testing "
                "(automated test suite, PERFORMANCE_BENCHMARKS.md, MODEL_CARD.md "
                "metrics WITH honest cross-bearing caveats); cybersecurity "
                "(SECURITY.md, Ed25519/HMAC signing, checkpoint pinning, SBOM)"),
            "deployer_must_supply": (
                "any provider-side fine-tuning/calibration methodology, the data "
                "governance of provider-supplied data, and predetermined-change / "
                "continuous-update plans for their deployment"),
            "status": STATUS_PARTIAL if (arch_supplied or data_supplied)
            else STATUS_PARTIAL,
            "limitation": (
                "the package documents the SHIPPED model and tooling; benchmark "
                "metrics do NOT transfer to a different machine/sensor/regime "
                "without re-validation (honest cross-bearing LOBO is materially "
                "lower than in-distribution), and any provider modification voids "
                "the shipped validation."),
        },
        {
            "number": "3", "title": "Monitoring, functioning and control of the system",
            "annex_iv_requirement": (
                "capabilities and limitations, expected accuracy (incl. for "
                "specific groups), foreseeable unintended outcomes, sources of "
                "risk, human oversight and required input data"),
            "aion_provides": (
                "documented capabilities AND limitations (MODEL_CARD.md, "
                "docs/negative_results.md, RETRACTIONS.md); the abstain/review "
                "behaviour and the upstream OOD signal-plausibility gate as control "
                "measures; required input specification (docs/data_contract.md); "
                "per-decision telemetry via the certificate + /metrics"),
            "deployer_must_supply": (
                "monitoring of the system in THEIR operating conditions, including "
                "accuracy for their asset population and their unintended-outcome "
                "analysis"),
            "status": STATUS_AION_EVIDENCE,
            "limitation": (
                "limitations are documented at the model level; accuracy for a "
                "specific deployed asset population, and outcomes specific to the "
                "deployment, can only be established by the provider on their data."),
        },
        {
            "number": "4", "title": "Appropriateness of the performance metrics",
            "annex_iv_requirement": "why the chosen metrics are appropriate for the system",
            "aion_provides": (
                "the rationale for the reported metrics and their honest scope "
                "(PERFORMANCE_BENCHMARKS.md, MODEL_CARD.md): conformal coverage as a "
                "calibrated-trust metric, F1 on degradation stages, and the explicit "
                "statement that labels are POSITIONAL degradation/RUL stages, not "
                "fault-type diagnoses"),
            "deployer_must_supply": (
                "justification that these metrics are appropriate for THEIR intended "
                "purpose and acceptance thresholds"),
            "status": STATUS_AION_EVIDENCE,
            "limitation": (
                "metric appropriateness is argued for the generic task; the "
                "provider must justify the metrics and pass/fail thresholds against "
                "their specific intended purpose."),
        },
        {
            "number": "5", "title": "Risk management system (Article 9)",
            "annex_iv_requirement": "the risk management system established and maintained",
            "aion_provides": (
                "a risk register as an INPUT (RISK_REGISTER.md) and the technical "
                "risk controls it references (OOD gate, abstain-on-uncertainty, "
                "signed audit trail, checkpoint pinning)"),
            "deployer_must_supply": (
                "the Article 9 risk management SYSTEM itself — a continuous, "
                "documented process with risk identification, estimation, "
                "evaluation and treatment over the lifecycle, owned by the provider"),
            "status": STATUS_DEPLOYER_OWNED,
            "limitation": (
                "a register and technical controls are inputs, NOT the risk "
                "management system: Article 9 requires an organisational, "
                "continuously-maintained process that only the provider can run."),
        },
        {
            "number": "6", "title": "Relevant changes through the lifecycle",
            "annex_iv_requirement": "a description of changes made to the system over its lifecycle",
            "aion_provides": (
                "a complete, dated change history (CHANGELOG.md), single-source "
                "versioning (aion_nexus.version), and a record of scientific "
                "corrections/retractions (RETRACTIONS.md) — an honest lifecycle trail"),
            "deployer_must_supply": (
                "the change log for THEIR deployment, including configuration, "
                "calibration and any provider-side modifications"),
            "status": STATUS_AION_EVIDENCE,
            "limitation": (
                "the package change history covers the SHIPPED artefact; changes "
                "introduced during integration and operation are the provider's to "
                "record."),
        },
        {
            "number": "7", "title": "List of harmonised standards applied",
            "annex_iv_requirement": (
                "harmonised standards applied in full or in part, or other "
                "solutions used to meet the requirements"),
            "aion_provides": (
                f"candidate/related standards referenced by the package: {std_text}; "
                "the package maps evidence toward ISO 13374/13381 (condition "
                "monitoring / prognostics), IEC 62443 (OT cybersecurity) and "
                "ISO/IEC 42001 (AI management) — see docs/COMPLIANCE_MAPPING.md"),
            "deployer_must_supply": (
                "the definitive list of standards the provider FORMALLY applies, and "
                "the rationale where a harmonised standard is not yet available"),
            "status": STATUS_PARTIAL if std_supplied else STATUS_DEPLOYER_OWNED,
            "limitation": (
                "referencing a standard is not applying it; formal application and "
                "the gap analysis where harmonised standards are absent (the JTC 21 "
                "standards were not yet in the Official Journal at the documentation "
                "date) are the provider's responsibility."),
        },
        {
            "number": "8", "title": "EU declaration of conformity",
            "annex_iv_requirement": "a copy of the EU declaration of conformity",
            "aion_provides": (
                "nothing for this point: a declaration of conformity can be issued "
                "ONLY by the provider placing the system on the market. The package "
                "supplies the technical EVIDENCE that supports such a process, never "
                "the declaration"),
            "deployer_must_supply": (
                "the EU declaration of conformity itself, issued and signed by the "
                "provider under their own responsibility after the required "
                "assessment"),
            "status": STATUS_DEPLOYER_OWNED,
            "limitation": (
                "this point is entirely provider-owned; nothing in this package "
                "constitutes, or substitutes for, a declaration of conformity or a "
                "third-party assessment."),
        },
        {
            "number": "9", "title": "Post-market monitoring system (Article 72)",
            "annex_iv_requirement": (
                "the system in place to evaluate AI system performance in the "
                "post-market phase"),
            "aion_provides": (
                "technical INPUTS to a post-market monitoring plan: the hash-linked "
                "certificate audit trail (CertificateStore), per-decision verdicts "
                "and uncertainty, /metrics telemetry, and the factory bridge that "
                "publishes signed verdicts to the operator's OPC UA / Sparkplug bus "
                "(aion_nexus.connect) for collection"),
            "deployer_must_supply": (
                "the Article 72 post-market monitoring PLAN and its operation — "
                "data collection, analysis, and the feedback loop into the risk "
                "management system, owned by the provider"),
            "status": STATUS_PARTIAL,
            "limitation": (
                "the package emits the telemetry; the monitoring plan, its "
                "thresholds, and the obligation to act on findings are the "
                "provider's."),
        },
    ]

    n_aion = sum(1 for s in sections if s["status"] == STATUS_AION_EVIDENCE)
    n_partial = sum(1 for s in sections if s["status"] == STATUS_PARTIAL)
    n_deployer = sum(1 for s in sections if s["status"] == STATUS_DEPLOYER_OWNED)

    return {
        "disclaimer": (
            "This is an Annex IV EVIDENCE MAP — a documentation skeleton that shows "
            "which technical artefacts this package can supply toward each Annex IV "
            "point and what the provider must author. It is NOT the technical "
            "documentation, NOT a declaration of conformity, and NOT a statement "
            "that the system meets Annex IV. High-risk classification and the duty "
            "to produce Annex IV documentation depend on the use context and a "
            "formal assessment by the provider, supported by legal review."),
        "regulation": "EU AI Act (Regulation (EU) 2024/1689), Article 11 and Annex IV",
        "system_ref": ref,
        "sections": sections,
        "readiness": {
            "sections_total": len(sections),
            "sections_with_aion_evidence": n_aion,
            "sections_partial": n_partial,
            "sections_deployer_owned": n_deployer,
            "note": (
                "Counts of where AION supplies evidence are a PREPARATION metric to "
                "help a provider assemble their dossier faster. They are explicitly "
                "NOT a measure of regulatory conformity or readiness to place the "
                "system on the market."),
        },
        "overall_limitation": (
            "Annex IV documentation is the provider's obligation. This map "
            "accelerates it by pointing to concrete artefacts, but the provider "
            "must author the context-specific content, run the Article 9 risk "
            "process, list and apply standards, issue the declaration of "
            "conformity, and operate post-market monitoring — none of which this "
            "package performs."),
    }


def annex_iv_card(model_metadata: dict | None = None, *, certificate: Any = None) -> str:
    """Render :func:`annex_iv_dossier` as a Markdown dossier skeleton.

    Provider-facing and, by contract, never asserts the system meets Annex IV: it
    states, per section, what AION supplies and what the provider must author.
    """
    data = annex_iv_dossier(model_metadata, certificate=certificate)
    ref = data["system_ref"]
    r = data["readiness"]

    lines: list[str] = []
    lines.append("# EU AI Act Annex IV — technical-documentation evidence map")
    lines.append("")
    lines.append(f"> {data['disclaimer']}")
    lines.append("")
    lines.append(f"**Regulation:** {data['regulation']}  ")
    lines.append(f"**System:** {ref.get('name')} v{ref.get('version')}"
                 + (f" (model_id `{ref.get('model_id')}`)" if ref.get('model_id') else ""))
    if ref.get("documentation_date"):
        lines.append(f"**Documentation date:** {ref.get('documentation_date')}")
    lines.append("")
    lines.append(
        f"**Preparation snapshot (NOT a conformity measure):** "
        f"{r['sections_with_aion_evidence']} of {r['sections_total']} points have "
        f"AION-supplied evidence, {r['sections_partial']} partial, "
        f"{r['sections_deployer_owned']} provider-owned.")
    lines.append("")
    for s in data["sections"]:
        lines.append(f"## {s['number']}. {s['title']}")
        lines.append("")
        lines.append(f"- **Annex IV asks:** {s['annex_iv_requirement']}")
        lines.append(f"- **AION provides:** {s['aion_provides']}")
        lines.append(f"- **Provider must supply:** {s['deployer_must_supply']}")
        lines.append(f"- **Status:** `{s['status']}`")
        lines.append(f"- **Limitation:** {s['limitation']}")
        lines.append("")
    lines.append("## Overall limitation")
    lines.append("")
    lines.append(data["overall_limitation"])
    lines.append("")
    return "\n".join(lines)


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
