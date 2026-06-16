"""``aion_nexus.connect`` — the factory bridge: a signed verdict in the factory's
own protocol fabric (Sparkplug B / Unified Namespace and OPC UA).

The thesis ([[18_PATH_TO_UNIGNORABLE]]) says the one thing that makes AION
unignorable is *a verdict that is cryptographically signed AND lives inside the
factory's own world*. Substrate Core mints the signed certificate; this package
carries it onto the bus the incumbents already speak — so a Siemens / Beckhoff /
SKF integrator pulls AION's certificate off **their** MQTT broker or **their**
OPC UA server and verifies it OFFLINE with the public key alone. No incumbent
ships independent, third-party-checkable verification inside the factory fabric.

Honesty (workspace 6.31): an ``ABSTAIN`` (out-of-distribution / low confidence)
or a ``REVIEW`` (ambiguous conformal set) is NEVER rendered as a high-severity,
machine-stop alarm — severity is capped by the trust verdict and an ABSTAIN maps
to OPC UA ``Quality = Uncertain``. The verdict on the bus tells the truth when
the model does not know. See :mod:`aion_nexus.connect.mapping`.

Dependency posture
------------------
Importing this package and BUILDING payloads/models is dependency-free (only the
in-package protobuf codec). The live transports — :class:`SparkplugPublisher`
(MQTT via ``paho-mqtt``) and :class:`CertifiedConditionMonitoringServer` (OPC UA
via ``asyncua``) — import their optional dependencies lazily, only when you
actually connect/start, and raise a clear ``pip install aion-nexus[factory]``
message otherwise.

Quickstart
----------
    >>> from aion_nexus.connect import to_factory_verdict, build_sparkplug_payload
    >>> fv = to_factory_verdict(cert)          # cert from Verifier.certify(...)
    >>> fv.actionable                          # True only if verifiable & CERTIFIED
    >>> payload = build_sparkplug_payload(cert) # real Sparkplug B protobuf bytes
"""
from __future__ import annotations

from .mapping import (
    AUTH_NONE,
    STATE_ADVISORY,
    STATE_ALARM,
    STATE_DATA_UNCERTAIN,
    STATE_NORMAL,
    STATE_REVIEW,
    STATE_WARNING,
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    FactoryVerdict,
    to_factory_verdict,
)
from .opcua_cm import (
    AION_NAMESPACE_URI,
    QUALITY_GOOD,
    QUALITY_UNCERTAIN,
    CertifiedConditionMonitoringServer,
    build_condition_model,
)
from .sparkplug import (
    SPARKPLUG_NAMESPACE,
    Metric,
    SparkplugPublisher,
    build_sparkplug_payload,
    decode_payload,
    encode_payload,
    sparkplug_topic,
    verdict_to_metrics,
)

__all__ = [
    # mapping (the spine)
    "FactoryVerdict",
    "to_factory_verdict",
    "VERDICT_CERTIFIED",
    "VERDICT_REVIEW",
    "VERDICT_ABSTAIN",
    "AUTH_NONE",
    "STATE_NORMAL",
    "STATE_ADVISORY",
    "STATE_WARNING",
    "STATE_ALARM",
    "STATE_REVIEW",
    "STATE_DATA_UNCERTAIN",
    # Sparkplug B
    "Metric",
    "SparkplugPublisher",
    "build_sparkplug_payload",
    "encode_payload",
    "decode_payload",
    "verdict_to_metrics",
    "sparkplug_topic",
    "SPARKPLUG_NAMESPACE",
    # OPC UA
    "build_condition_model",
    "CertifiedConditionMonitoringServer",
    "AION_NAMESPACE_URI",
    "QUALITY_GOOD",
    "QUALITY_UNCERTAIN",
]
