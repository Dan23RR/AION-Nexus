"""Render a :class:`~aion_nexus.connect.mapping.FactoryVerdict` as a **Sparkplug B**
payload and (optionally) publish it to an MQTT broker — the Unified-Namespace
language of the modern factory.

What this is, precisely (workspace 6.31 — no overclaim)
------------------------------------------------------
This emits a **Sparkplug B 3.0 ``Payload``** protobuf carrying the verdict as a
metric set, with the WHOLE signed certificate as the ``AION/certificate`` metric.
A Sparkplug-aware consumer (HiveMQ, Ignition, a UNS broker) decodes it like any
other Sparkplug payload; an auditor pulls the ``AION/certificate`` metric off the
bus and re-verifies it with :func:`aion_nexus.verify.verify_certificate` and the
issuer's public key — the unique property no incumbent ships.

Covered: ``Payload`` (timestamp, seq, metrics) and ``Metric`` (name, alias,
timestamp, datatype, is_null, and Int64/Double/Boolean/String/Bytes values),
encoded with the minimal in-package protobuf codec (:mod:`._protobuf`) so there
is NO runtime ``protobuf`` dependency. NOT emitted: DataSet/Template/PropertySet
metric values, metric metadata. This is a faithful payload for the metrics we
publish, not a full Tahu reimplementation — and we say so.

Transport: :class:`SparkplugPublisher` uses ``paho-mqtt`` when installed
(``pip install aion-nexus[factory]``); without it, payload BUILDING still works
(so the encoder is testable with zero dependencies) and only the live
``connect()/publish`` calls raise a clear, actionable error.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from . import _protobuf as pb
from .mapping import FactoryVerdict, to_factory_verdict

# --- Sparkplug B DataType enum (org.eclipse.tahu.protobuf.DataType) ---
DT_INT64 = 4
DT_DOUBLE = 10
DT_BOOLEAN = 11
DT_STRING = 12
DT_BYTES = 17

# --- Sparkplug B Metric field numbers + value-oneof field numbers ---
_M_NAME = 1
_M_ALIAS = 2
_M_TIMESTAMP = 3
_M_DATATYPE = 4
_M_IS_NULL = 7
_M_LONG_VALUE = 11      # int64/uint64
_M_DOUBLE_VALUE = 13
_M_BOOLEAN_VALUE = 14
_M_STRING_VALUE = 15
_M_BYTES_VALUE = 16

# --- Sparkplug B Payload field numbers ---
_P_TIMESTAMP = 1
_P_METRICS = 2
_P_SEQ = 3

# Sparkplug B message types (the ``message_type`` token in the topic namespace).
NBIRTH = "NBIRTH"   # edge node birth
NDATA = "NDATA"     # edge node data
DBIRTH = "DBIRTH"   # device birth
DDATA = "DDATA"     # device data
NDEATH = "NDEATH"   # edge node death (LWT)

SPARKPLUG_NAMESPACE = "spBv1.0"


@dataclass
class Metric:
    """One Sparkplug B metric (the subset of fields this package emits)."""
    name: str
    datatype: int
    value: Any = None
    alias: int | None = None
    timestamp_ms: int | None = None
    is_null: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)


def _encode_metric(m: Metric) -> bytes:
    parts: list[bytes] = []
    if m.name is not None:
        parts.append(pb.write_string_field(_M_NAME, m.name))
    if m.alias is not None:
        parts.append(pb.write_varint_field(_M_ALIAS, m.alias))
    if m.timestamp_ms is not None:
        parts.append(pb.write_varint_field(_M_TIMESTAMP, m.timestamp_ms))
    parts.append(pb.write_varint_field(_M_DATATYPE, m.datatype))

    if m.is_null or m.value is None:
        parts.append(pb.write_bool_field(_M_IS_NULL, True))
        return b"".join(parts)

    if m.datatype == DT_STRING:
        parts.append(pb.write_string_field(_M_STRING_VALUE, str(m.value)))
    elif m.datatype == DT_DOUBLE:
        parts.append(pb.write_double_field(_M_DOUBLE_VALUE, float(m.value)))
    elif m.datatype == DT_BOOLEAN:
        parts.append(pb.write_bool_field(_M_BOOLEAN_VALUE, bool(m.value)))
    elif m.datatype == DT_INT64:
        iv = int(m.value)
        if iv < 0:  # the minimal varint codec is unsigned; our ints are non-negative
            raise ValueError(
                f"metric {m.name!r}: this codec encodes non-negative Int64 only")
        parts.append(pb.write_varint_field(_M_LONG_VALUE, iv))
    elif m.datatype == DT_BYTES:
        parts.append(pb.write_bytes_field(_M_BYTES_VALUE, bytes(m.value)))
    else:
        raise ValueError(f"unsupported Sparkplug datatype {m.datatype} for "
                         f"metric {m.name!r} (this package emits "
                         "Int64/Double/Boolean/String/Bytes)")
    return b"".join(parts)


def encode_payload(metrics: list[Metric], *, seq: int, timestamp_ms: int | None = None
                   ) -> bytes:
    """Encode a Sparkplug B ``Payload`` (timestamp, seq, metrics) to protobuf bytes."""
    if not 0 <= seq <= 255:
        raise ValueError("Sparkplug seq must be in [0, 255] (it wraps per spec)")
    ts = _now_ms() if timestamp_ms is None else int(timestamp_ms)
    out: list[bytes] = [pb.write_varint_field(_P_TIMESTAMP, ts)]
    for m in metrics:
        out.append(pb.write_message_field(_P_METRICS, _encode_metric(m)))
    out.append(pb.write_varint_field(_P_SEQ, seq))
    return b"".join(out)


def decode_payload(buf: bytes) -> dict:
    """Decode a Sparkplug B ``Payload`` produced by :func:`encode_payload`.

    Returns ``{"timestamp": int, "seq": int, "metrics": [{"name", "datatype",
    "value", "is_null"}...]}``. Used by the edge-verifier example and the tests to
    prove the bytes round-trip — and to pull ``AION/certificate`` back off the bus.
    """
    timestamp = None
    seq = None
    metrics: list[dict] = []
    for field_number, _wire, value in pb.parse_fields(buf):
        if field_number == _P_TIMESTAMP:
            timestamp = value
        elif field_number == _P_SEQ:
            seq = value
        elif field_number == _P_METRICS:
            metrics.append(_decode_metric(value))
    return {"timestamp": timestamp, "seq": seq, "metrics": metrics}


def _decode_metric(buf: bytes) -> dict:
    name = None
    datatype = None
    is_null = False
    value: Any = None
    for field_number, _wire, raw in pb.parse_fields(buf):
        if field_number == _M_NAME:
            name = raw.decode("utf-8")
        elif field_number == _M_DATATYPE:
            datatype = raw
        elif field_number == _M_IS_NULL:
            is_null = bool(raw)
        elif field_number == _M_STRING_VALUE:
            value = raw.decode("utf-8")
        elif field_number == _M_DOUBLE_VALUE:
            value = raw
        elif field_number == _M_BOOLEAN_VALUE:
            value = bool(raw)
        elif field_number == _M_LONG_VALUE:
            value = raw
        elif field_number == _M_BYTES_VALUE:
            value = raw
    return {"name": name, "datatype": datatype,
            "value": None if is_null else value, "is_null": is_null}


def verdict_to_metrics(verdict: FactoryVerdict, *, timestamp_ms: int | None = None,
                       include_certificate: bool = True) -> list[Metric]:
    """Map a :class:`FactoryVerdict` to the AION Sparkplug B metric set.

    The metric names form a stable ``AION/...`` hierarchy. ``AION/certificate``
    carries the full signed certificate JSON — the metric an auditor verifies.
    """
    ts = timestamp_ms
    m: list[Metric] = [
        Metric("AION/verdict", DT_STRING, verdict.decision, timestamp_ms=ts),
        Metric("AION/health_class", DT_STRING, verdict.health_class, timestamp_ms=ts),
        Metric("AION/condition_state", DT_STRING, verdict.condition_state, timestamp_ms=ts),
        Metric("AION/severity", DT_INT64, int(verdict.severity), timestamp_ms=ts),
        Metric("AION/active", DT_BOOLEAN, verdict.active, timestamp_ms=ts),
        Metric("AION/actionable", DT_BOOLEAN, verdict.actionable, timestamp_ms=ts),
        Metric("AION/verifiable", DT_BOOLEAN, verdict.verifiable, timestamp_ms=ts),
        Metric("AION/assurance", DT_STRING, verdict.assurance, timestamp_ms=ts),
        Metric("AION/conformal_set", DT_STRING, ",".join(verdict.conformal_set),
               timestamp_ms=ts),
        Metric("AION/content_hash", DT_STRING, verdict.content_hash, timestamp_ms=ts),
        Metric("AION/authentication", DT_STRING, verdict.authentication, timestamp_ms=ts),
        Metric("AION/message", DT_STRING, verdict.message, timestamp_ms=ts),
    ]
    # Nullable provenance metrics — emitted as Sparkplug null when absent.
    m.append(Metric("AION/pubkey", DT_STRING, verdict.pubkey, timestamp_ms=ts,
                     is_null=verdict.pubkey is None))
    m.append(Metric("AION/key_id", DT_STRING, verdict.key_id, timestamp_ms=ts,
                     is_null=verdict.key_id is None))
    m.append(Metric("AION/valid_until", DT_STRING, verdict.valid_until, timestamp_ms=ts,
                     is_null=verdict.valid_until is None))
    m.append(Metric("AION/model_id", DT_STRING, verdict.model_id, timestamp_ms=ts,
                     is_null=verdict.model_id is None))
    if include_certificate:
        m.append(Metric("AION/certificate", DT_STRING, verdict.certificate_json(),
                        timestamp_ms=ts))
    return m


def build_sparkplug_payload(cert: Any, *, seq: int = 0, timestamp_ms: int | None = None,
                            include_certificate: bool = True) -> bytes:
    """One-shot: certificate -> :class:`FactoryVerdict` -> Sparkplug B ``Payload`` bytes."""
    verdict = cert if isinstance(cert, FactoryVerdict) else to_factory_verdict(cert)
    metrics = verdict_to_metrics(verdict, timestamp_ms=timestamp_ms,
                                 include_certificate=include_certificate)
    return encode_payload(metrics, seq=seq, timestamp_ms=timestamp_ms)


def sparkplug_topic(group_id: str, message_type: str, edge_node_id: str,
                    device_id: str | None = None) -> str:
    """Build the Sparkplug B topic: ``spBv1.0/{group}/{type}/{edge}[/{device}]``."""
    parts = [SPARKPLUG_NAMESPACE, group_id, message_type, edge_node_id]
    if device_id:
        parts.append(device_id)
    return "/".join(parts)


class SparkplugPublisher:
    """Publish AION certified verdicts to an MQTT broker as Sparkplug B payloads.

    The MQTT transport (``paho-mqtt``) is an OPTIONAL dependency: importing this
    module and BUILDING payloads needs nothing extra; only :meth:`connect` /
    :meth:`publish_verdict` require ``pip install aion-nexus[factory]``. The
    Sparkplug ``seq`` counter is maintained here (wrapping 0..255 per spec); call
    :meth:`birth` once before data messages so a UNS consumer can establish state.
    """

    def __init__(self, group_id: str, edge_node_id: str, device_id: str | None = None,
                 *, client_id: str | None = None) -> None:
        self.group_id = group_id
        self.edge_node_id = edge_node_id
        self.device_id = device_id
        self.client_id = client_id or f"aion-{edge_node_id}"
        self._client = None
        self._seq = 0

    # -- seq management (Sparkplug wraps the payload seq at 256) --
    def _next_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) % 256
        return s

    def _require_client(self):
        if self._client is None:
            raise RuntimeError(
                "not connected: call connect() first (and `pip install "
                "aion-nexus[factory]` to pull in paho-mqtt).")
        return self._client

    def connect(self, host: str = "localhost", port: int = 1883, *,
                keepalive: int = 60, username: str | None = None,
                password: str | None = None) -> SparkplugPublisher:
        """Connect to the MQTT broker. Requires ``paho-mqtt`` (the [factory] extra)."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SparkplugPublisher.connect requires paho-mqtt. Install the factory "
                "extra: pip install aion-nexus[factory]") from exc
        client = mqtt.Client(client_id=self.client_id)
        if username:
            client.username_pw_set(username, password)
        # Last-will: an NDEATH so the UNS knows if this edge node drops.
        death_topic = sparkplug_topic(self.group_id, NDEATH, self.edge_node_id)
        client.will_set(death_topic, payload=encode_payload([], seq=0), qos=1, retain=False)
        client.connect(host, port, keepalive)
        client.loop_start()
        self._client = client
        return self

    def birth(self) -> None:
        """Publish the Sparkplug birth certificate (NBIRTH, or DBIRTH if a device)."""
        client = self._require_client()
        mtype = DBIRTH if self.device_id else NBIRTH
        topic = sparkplug_topic(self.group_id, mtype, self.edge_node_id, self.device_id)
        # Birth carries a bdSeq + a seq reset to 0 (minimal: an empty-state birth).
        payload = encode_payload(
            [Metric("bdSeq", DT_INT64, 0)], seq=self._next_seq())
        client.publish(topic, payload, qos=0, retain=False)

    def publish_verdict(self, cert: Any, *, include_certificate: bool = True,
                        qos: int = 0) -> str:
        """Publish one certified verdict as a Sparkplug NDATA/DDATA payload.

        Returns the topic published to. ``cert`` may be a Certificate, a cert
        dict, or a pre-built :class:`FactoryVerdict`.
        """
        client = self._require_client()
        mtype = DDATA if self.device_id else NDATA
        topic = sparkplug_topic(self.group_id, mtype, self.edge_node_id, self.device_id)
        payload = build_sparkplug_payload(
            cert, seq=self._next_seq(), include_certificate=include_certificate)
        client.publish(topic, payload, qos=qos, retain=False)
        return topic

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
