"""Map a signed :class:`~aion_nexus.verify.Certificate` onto the factory's own
health/alarm semantics — transport-neutral, dependency-free.

Why this module exists (the kill-shot it closes)
------------------------------------------------
The Substrate Core verification layer mints a third-party-checkable certificate,
but a certificate is a JSON blob: a PLC, an OPC UA SCADA client, a Sparkplug B
Unified-Namespace broker do not know what it is. This module is the SPINE of the
factory bridge: it turns one certificate into a :class:`FactoryVerdict` — a
small, serialisable structure carrying (a) the diagnosis, (b) the TRUST verdict,
(c) an OPC UA-style severity, and (d) the WHOLE signed certificate so a third
party can verify it OFFLINE off their own bus. The concrete transports
(:mod:`aion_nexus.connect.sparkplug`, :mod:`aion_nexus.connect.opcua_cm`) are
thin renderers over this structure.

The honesty rule baked into the mapping (workspace 6.31)
-------------------------------------------------------
The diagnosis class (e.g. ``advanced``) and the TRUST verdict
(``CERTIFIED`` / ``REVIEW`` / ``ABSTAIN``) are DIFFERENT axes. An incumbent's
alarm says "advanced fault, severity 900" with no notion of its own uncertainty.
Ours refuses to: an ``ABSTAIN`` (out-of-distribution / low confidence) or a
``REVIEW`` (ambiguous conformal set) is NOT allowed to raise a high-severity,
machine-stop-grade alarm. Severity is CAPPED by the trust verdict:

- ``CERTIFIED`` -> the diagnosis drives severity (normal..advanced -> low..high).
- ``REVIEW``    -> severity capped to the WARNING band; state ``REVIEW`` (a
  human-in-the-loop condition, never an autonomous stop).
- ``ABSTAIN``   -> severity floored to ADVISORY; state ``DATA_UNCERTAIN``
  ("do NOT act on this as a fault alarm").

This mirrors the serving pipeline's no-escalation-on-abstain rule
(``aion_nexus.inference`` / the ``/predict_long_signal`` aggregate gate) and is
the actual differentiator: a verdict on the factory bus that tells the truth
when the model does not know, in the factory's own language.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Trust verdicts (mirror aion_nexus.verify.certificate; duplicated as plain
# strings so this module imports with ZERO dependency on the verify package —
# the bridge must be usable as a pure renderer over a certificate dict).
VERDICT_CERTIFIED = "CERTIFIED"
VERDICT_REVIEW = "REVIEW"
VERDICT_ABSTAIN = "ABSTAIN"

AUTH_NONE = "NONE"

# Condition states, named for the factory operator (not for us).
STATE_NORMAL = "NORMAL"               # in-distribution, healthy, certified
STATE_ADVISORY = "ADVISORY"           # early degradation, certified
STATE_WARNING = "WARNING"             # mid degradation, certified
STATE_ALARM = "ALARM"                 # advanced degradation, certified
STATE_REVIEW = "REVIEW"               # ambiguous conformal set -> human review
STATE_DATA_UNCERTAIN = "DATA_UNCERTAIN"  # abstain / OOD -> not a fault alarm

# OPC UA Alarms & Conditions severity is an integer in [1, 1000] (Part 9):
# 1 = lowest, 1000 = highest. These bands are the de-facto convention.
SEV_OK = 1            # condition inactive / healthy
SEV_ADVISORY = 250    # low band
SEV_WARNING = 600     # medium band
SEV_ALARM = 900       # high band
SEV_REVIEW_CAP = 500  # a REVIEW can never exceed the medium band
SEV_ABSTAIN = 100     # an ABSTAIN is a low-severity DATA quality condition

# Default diagnosis-class -> (state, severity) BEFORE the trust gate caps it.
# Keyed by the lower-cased predicted class name of the bearing model
# (normal/early/medium/advanced). Unknown names fall back to ADVISORY so an
# unrecognised class never silently maps to OK.
_CLASS_HEALTH: dict[str, tuple[str, int]] = {
    "normal": (STATE_NORMAL, SEV_OK),
    "early": (STATE_ADVISORY, SEV_ADVISORY),
    "medium": (STATE_WARNING, SEV_WARNING),
    "advanced": (STATE_ALARM, SEV_ALARM),
}
_UNKNOWN_CLASS_HEALTH = (STATE_ADVISORY, SEV_ADVISORY)


def _as_cert_dict(cert: Any) -> dict:
    """Accept a Certificate, its ``as_dict()``, or a plain dict; return a dict."""
    if hasattr(cert, "as_dict"):
        return cert.as_dict()
    if isinstance(cert, dict):
        return cert
    raise TypeError(
        "expected a Certificate, a certificate dict, or an object with as_dict(); "
        f"got {type(cert).__name__}")


@dataclass
class FactoryVerdict:
    """A transport-neutral view of one certified decision, ready for the factory.

    This is what the Sparkplug B and OPC UA renderers consume. It separates the
    DIAGNOSIS (``health_class``) from the TRUST verdict (``decision``) and folds
    the honesty rule into ``severity`` / ``condition_state`` so an uncertain
    verdict cannot masquerade as a confident alarm. The FULL signed certificate
    travels in ``certificate`` so any consumer can verify it independently.
    """

    # --- diagnosis (what) ---
    health_class: str                 # predicted_name, e.g. "advanced"
    conformal_set: list[str]          # the coverage-controlled label set
    # --- trust (how sure / how checkable) ---
    decision: str                     # CERTIFIED | REVIEW | ABSTAIN
    assurance: str                    # the assurance tier (e.g. "empirical")
    verifiable: bool                  # True iff signed (authentication != NONE)
    # --- factory-facing health/alarm (honesty-gated) ---
    condition_state: str              # NORMAL | ADVISORY | WARNING | ALARM | REVIEW | DATA_UNCERTAIN
    severity: int                     # OPC UA A&C severity [1, 1000], trust-capped
    active: bool                      # condition active (severity above OK floor)
    message: str                      # human-readable, operator-facing
    # --- provenance / verification handle ---
    content_hash: str = ""
    pubkey: str | None = None
    authentication: str = AUTH_NONE
    model_id: str | None = None
    valid_until: str | None = None
    key_id: str | None = None
    # Which conformal guarantee this verdict carries (v2.9.0). None = marginal/unstated.
    conformal_method: str | None = None
    coverage_guarantee: str | None = None
    certificate: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def actionable(self) -> bool:
        """A conservative gate: act autonomously ONLY on a verifiable CERTIFIED.

        This is intentionally strict and is the value the bridge advertises to
        the factory: a REVIEW (ambiguous), an ABSTAIN (uncertain) or an UNSIGNED
        certificate (not tamper-evident) is NEVER ``actionable`` — those route to
        a human. Callers may apply a looser policy, but the honest default is
        "do not let the line act on a verdict that is not both decisive and
        independently checkable".
        """
        return self.decision == VERDICT_CERTIFIED and self.verifiable

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def certificate_json(self) -> str:
        """The signed certificate as a compact JSON string (what a consumer
        feeds back into :func:`aion_nexus.verify.verify_certificate`)."""
        return json.dumps(self.certificate, sort_keys=True, separators=(",", ":"),
                          default=str)


def _severity_and_state(health_class: str, decision: str) -> tuple[int, str]:
    """Apply the diagnosis->health map, then CAP it by the trust verdict.

    The trust gate is the honesty mechanism: an uncertain or ambiguous verdict
    cannot escalate to a high-severity alarm regardless of the diagnosed class.
    """
    base_state, base_sev = _CLASS_HEALTH.get(
        health_class.lower().strip(), _UNKNOWN_CLASS_HEALTH)

    if decision == VERDICT_CERTIFIED:
        return base_sev, base_state

    if decision == VERDICT_REVIEW:
        # Ambiguous conformal set: surface as a human-review condition, capped to
        # the warning band so it can never drive an autonomous machine stop.
        return min(base_sev, SEV_REVIEW_CAP), STATE_REVIEW

    # ABSTAIN (or any unrecognised verdict -> fail safe): a data-quality
    # condition, never a fault alarm.
    return SEV_ABSTAIN, STATE_DATA_UNCERTAIN


def _message(health_class: str, decision: str, state: str, verifiable: bool,
             conformal_set: list[str]) -> str:
    """Compose the operator-facing message — honest about what the verdict means."""
    sig = "signed (third-party verifiable)" if verifiable else \
        "UNSIGNED (integrity-only, NOT tamper-evident)"
    if decision == VERDICT_CERTIFIED:
        head = f"CERTIFIED: bearing health '{health_class}'"
    elif decision == VERDICT_REVIEW:
        head = (f"REVIEW: ambiguous — conformal set {{{', '.join(conformal_set)}}} "
                "covers more than one class; route to a human")
    else:
        head = ("ABSTAIN: model is out-of-distribution or not confident enough — "
                "do NOT act on this as a fault alarm")
    return f"{head}; verdict→{state}; certificate {sig}"


def to_factory_verdict(cert: Any) -> FactoryVerdict:
    """Build a :class:`FactoryVerdict` from a certificate (object or dict).

    Pure mapping — no network, no optional dependency. The full certificate is
    carried verbatim in ``FactoryVerdict.certificate`` so any downstream consumer
    can re-verify it with :func:`aion_nexus.verify.verify_certificate` and the
    issuer's public key. The diagnosis/trust separation and the severity cap (see
    module docstring) are applied here, once, so every transport renders the same
    honest health state.
    """
    d = _as_cert_dict(cert)
    health_class = str(d.get("predicted_name", d.get("predicted_label", "unknown")))
    decision = str(d.get("verdict", VERDICT_ABSTAIN))
    auth = str(d.get("authentication", AUTH_NONE))
    verifiable = auth != AUTH_NONE
    cset = [str(n) for n in d.get("conformal_set_names", [])]

    severity, state = _severity_and_state(health_class, decision)
    # "active" asserts a real fault/attention condition is PRESENT. A DATA_UNCERTAIN
    # (abstain / OOD) reading asserts NO fault — it is surfaced via Quality=Uncertain
    # and Retain, not by raising an active alarm. This is the honest distinction
    # between "a fault is present" and "I cannot tell": never conflate them.
    active = severity > SEV_OK and state != STATE_DATA_UNCERTAIN

    return FactoryVerdict(
        health_class=health_class,
        conformal_set=cset,
        decision=decision,
        assurance=str(d.get("assurance", "")),
        verifiable=verifiable,
        condition_state=state,
        severity=severity,
        active=active,
        message=_message(health_class, decision, state, verifiable, cset),
        content_hash=str(d.get("content_hash", "")),
        pubkey=d.get("pubkey"),
        authentication=auth,
        model_id=d.get("model_id"),
        valid_until=d.get("valid_until"),
        key_id=d.get("key_id"),
        conformal_method=d.get("conformal_method"),
        coverage_guarantee=d.get("coverage_guarantee"),
        certificate=dict(d),
    )
