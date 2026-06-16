"""Expose a certified verdict as an **OPC UA Alarms & Conditions** condition plus
AION analysis variables — the language a Siemens / Beckhoff SCADA / historian
already speaks.

What this is, precisely (workspace 6.31 — no overclaim)
------------------------------------------------------
:func:`build_condition_model` maps a :class:`~aion_nexus.connect.mapping.FactoryVerdict`
onto **OPC UA Alarms & Conditions (Part 9)** semantics — ``ActiveState``,
``Severity`` (UInt16, 1..1000), ``Message``, ``Quality`` (a ``StatusCode``),
``Retain`` — and a set of AION extension variables (the verdict, the assurance
tier, and the WHOLE signed certificate) suitable for placement under an OPC UA
for Machinery / Condition Monitoring machine node. This is an information-model
MAPPING, dependency-free and testable; it is NOT a certified companion-spec
implementation and we do not claim conformance.

The honesty rule lands naturally in OPC UA terms
------------------------------------------------
OPC UA already has the vocabulary for "I am not sure": the ``Quality``
``StatusCode``. So an ``ABSTAIN`` maps to ``Quality = Uncertain`` with a LOW
severity — never a high-severity active alarm. A ``REVIEW`` stays ``Good`` but
capped to the warning band and ``Retain = True`` for human attention. Only a
``CERTIFIED`` verdict drives the diagnosis-mapped severity. An OPC UA engineer
reads this immediately — and no incumbent ships a verdict that flags its own
uncertainty as an OPC UA quality code.

Live server: :class:`CertifiedConditionMonitoringServer` builds a real address
space with ``asyncua`` when installed (``pip install aion-nexus[factory]``); the
model builder above needs nothing.
"""
from __future__ import annotations

from typing import Any

from .mapping import (
    STATE_DATA_UNCERTAIN,
    VERDICT_ABSTAIN,
    VERDICT_REVIEW,
    FactoryVerdict,
    to_factory_verdict,
)

# OPC UA StatusCode severities for the condition Quality field (Part 8 / Part 4).
QUALITY_GOOD = 0x00000000
QUALITY_UNCERTAIN = 0x40000000
QUALITY_BAD = 0x80000000

_QUALITY_NAME = {
    QUALITY_GOOD: "Good",
    QUALITY_UNCERTAIN: "Uncertain",
    QUALITY_BAD: "Bad",
}

# Default namespace URI for the AION analysis information model.
AION_NAMESPACE_URI = "http://aion-nexus.io/UA/Verification"


def _quality_for(verdict: FactoryVerdict) -> int:
    """Map the trust verdict to an OPC UA Quality StatusCode.

    ABSTAIN -> Uncertain (the model does not know — exactly what Quality is for).
    An UNSIGNED certificate is also Uncertain (not tamper-evident). Otherwise Good.
    """
    if verdict.decision == VERDICT_ABSTAIN or verdict.condition_state == STATE_DATA_UNCERTAIN:
        return QUALITY_UNCERTAIN
    if not verdict.verifiable:
        return QUALITY_UNCERTAIN
    return QUALITY_GOOD


def build_condition_model(cert: Any) -> dict:
    """Build a dependency-free OPC UA A&C condition model from a certificate.

    Returns a serialisable dict with two parts:

    - ``condition`` — the OPC UA Alarms & Conditions fields (ActiveState,
      Severity, Message, Quality, Retain, AckedState...).
    - ``analysis`` — the AION extension variables, including ``Certificate`` (the
      signed JSON an OPC UA client reads and re-verifies) and ``PublicKey``.

    The same honesty gate as the rest of the bridge applies: ABSTAIN/REVIEW
    cannot present as a high-severity active alarm.
    """
    verdict = cert if isinstance(cert, FactoryVerdict) else to_factory_verdict(cert)
    quality = _quality_for(verdict)
    # Retain semantics (Part 9): keep the condition visible while it needs
    # attention — an active alarm, a review, or an uncertain reading.
    retain = bool(verdict.active or verdict.decision in (VERDICT_REVIEW, VERDICT_ABSTAIN))

    condition = {
        "ConditionType": "AlarmConditionType",
        "SourceName": verdict.model_id or "AION-NEXUS",
        "ConditionName": "AION-NEXUS bearing health",
        "EnabledState": "Enabled",
        "ActiveState": "Active" if verdict.active else "Inactive",
        "ActiveStateId": bool(verdict.active),
        # AckedState: an autonomously-actionable CERTIFIED alarm starts Unacked;
        # uncertain/review conditions are advisory (no ack workflow forced).
        "AckedState": "Unacknowledged" if (verdict.active and verdict.actionable)
        else "Acknowledged",
        "Severity": int(verdict.severity),           # UInt16 [1, 1000]
        "Quality": quality,                          # StatusCode
        "QualityName": _QUALITY_NAME.get(quality, "Good"),
        "Message": verdict.message,
        "Retain": retain,
        "ConditionState": verdict.condition_state,
    }

    analysis = {
        "Verdict": verdict.decision,
        "HealthClass": verdict.health_class,
        "ConformalSet": ",".join(verdict.conformal_set),
        "Assurance": verdict.assurance,
        "Verifiable": verdict.verifiable,
        "Actionable": verdict.actionable,
        "Authentication": verdict.authentication,
        "ContentHash": verdict.content_hash,
        "PublicKey": verdict.pubkey,
        "KeyId": verdict.key_id,
        "ValidUntil": verdict.valid_until,
        "ModelId": verdict.model_id,
        # The crown jewel: the full signed certificate, on the OPC UA bus.
        "Certificate": verdict.certificate_json(),
    }
    return {"namespace_uri": AION_NAMESPACE_URI,
            "condition": condition, "analysis": analysis}


class CertifiedConditionMonitoringServer:
    """An OPC UA server that exposes AION certified verdicts as live nodes.

    Requires ``asyncua`` (``pip install aion-nexus[factory]``). The address space
    holds an ``AionVerification`` object whose variables mirror
    :func:`build_condition_model` (``Verdict``, ``Severity``, ``Quality``,
    ``Certificate`` ...). An OPC UA client subscribes to these like any other CM
    node; the ``Certificate`` variable carries the signed JSON for offline
    re-verification.

    Usage (async)::

        srv = CertifiedConditionMonitoringServer(endpoint="opc.tcp://0.0.0.0:4840")
        await srv.start()
        await srv.update(cert)          # push a fresh certified verdict
        ...
        await srv.stop()
    """

    def __init__(self, endpoint: str = "opc.tcp://0.0.0.0:4840", *,
                 server_name: str = "AION-NEXUS Verification Server",
                 namespace_uri: str = AION_NAMESPACE_URI) -> None:
        self.endpoint = endpoint
        self.server_name = server_name
        self.namespace_uri = namespace_uri
        self._server = None
        self._idx = None
        self._vars: dict = {}

    @staticmethod
    def _import_asyncua():
        try:
            from asyncua import Server
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "CertifiedConditionMonitoringServer requires asyncua. Install the "
                "factory extra: pip install aion-nexus[factory]") from exc
        return Server

    async def start(self) -> CertifiedConditionMonitoringServer:
        """Initialise the server and build the AION verification address space."""
        Server = self._import_asyncua()  # noqa: N806 — class object, conventionally capitalised
        server = Server()
        await server.init()
        server.set_endpoint(self.endpoint)
        server.set_server_name(self.server_name)
        idx = await server.register_namespace(self.namespace_uri)
        obj = await server.nodes.objects.add_object(idx, "AionVerification")
        # Seed the variables with empty values; update() refreshes them.
        var_specs = [
            ("Verdict", ""), ("HealthClass", ""), ("ConformalSet", ""),
            ("Assurance", ""), ("Verifiable", False), ("Actionable", False),
            ("Severity", 0), ("Quality", QUALITY_GOOD), ("QualityName", "Good"),
            ("ActiveState", "Inactive"), ("ConditionState", ""), ("Message", ""),
            ("Authentication", ""), ("ContentHash", ""), ("PublicKey", ""),
            ("KeyId", ""), ("ValidUntil", ""), ("ModelId", ""), ("Certificate", ""),
        ]
        for name, default in var_specs:
            self._vars[name] = await obj.add_variable(idx, name, default)
        await server.start()
        self._server = server
        self._idx = idx
        return self

    async def update(self, cert: Any) -> dict:
        """Push a fresh certified verdict to the address space; return the model."""
        if self._server is None:
            raise RuntimeError("call start() before update()")
        model = build_condition_model(cert)
        c, a = model["condition"], model["analysis"]
        flat = {
            "Verdict": a["Verdict"], "HealthClass": a["HealthClass"],
            "ConformalSet": a["ConformalSet"], "Assurance": a["Assurance"],
            "Verifiable": a["Verifiable"], "Actionable": a["Actionable"],
            "Severity": c["Severity"], "Quality": c["Quality"],
            "QualityName": c["QualityName"], "ActiveState": c["ActiveState"],
            "ConditionState": c["ConditionState"], "Message": c["Message"],
            "Authentication": a["Authentication"], "ContentHash": a["ContentHash"],
            "PublicKey": a["PublicKey"] or "", "KeyId": a["KeyId"] or "",
            "ValidUntil": a["ValidUntil"] or "", "ModelId": a["ModelId"] or "",
            "Certificate": a["Certificate"],
        }
        for name, value in flat.items():
            var = self._vars.get(name)
            if var is not None:
                await var.write_value(value)
        return model

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None
