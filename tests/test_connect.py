"""Tests for the factory bridge (``aion_nexus.connect``).

The headline test is :func:`test_signed_verdict_is_verifiable_off_the_bus`: a
REAL Ed25519-signed certificate is encoded to a Sparkplug B payload, decoded back
(as a third party on the bus would), and re-verified with the public key alone —
``trusted == True``. Tamper any field and trust collapses. That round-trip IS the
unignorable property: an independently-checkable verdict inside the factory's own
protocol fabric.

Everything here runs with ZERO optional dependencies (no paho-mqtt, no asyncua):
building payloads/models uses only the in-package protobuf codec.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json

import pytest

from aion_nexus.connect import (
    QUALITY_GOOD,
    QUALITY_UNCERTAIN,
    CertifiedConditionMonitoringServer,
    SparkplugPublisher,
    build_condition_model,
    build_sparkplug_payload,
    decode_payload,
    encode_payload,
    sparkplug_topic,
    to_factory_verdict,
)
from aion_nexus.connect import _protobuf as pb
from aion_nexus.connect.mapping import (
    SEV_ABSTAIN,
    SEV_ALARM,
    SEV_REVIEW_CAP,
    STATE_ALARM,
    STATE_DATA_UNCERTAIN,
    STATE_REVIEW,
)
from aion_nexus.connect.sparkplug import (
    DT_BOOLEAN,
    DT_BYTES,
    DT_DOUBLE,
    DT_INT64,
    DT_STRING,
    Metric,
)
from aion_nexus.verify import (
    Certificate,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)

_NOW = "2026-06-16T00:00:00+00:00"
_IN_WINDOW = "2026-06-16T00:10:00+00:00"   # 10 min later, inside a 1h ttl
_EXPIRED = "2026-06-16T02:00:00+00:00"     # 2h later, past a 1h ttl

_HAS_PAHO = importlib.util.find_spec("paho") is not None
_HAS_ASYNCUA = importlib.util.find_spec("asyncua") is not None
_HAS_PROTOBUF = importlib.util.find_spec("google.protobuf") is not None


def _signed_cert(verdict="CERTIFIED", predicted_name="advanced",
                 conformal=("advanced",), label=3, *, ttl=3600, key_id="bench-key-1"):
    """A real Ed25519-signed certificate with a strong seed and a validity window."""
    seed = generate_seed()
    cert = Certificate(
        predicted_label=label,
        predicted_name=predicted_name,
        conformal_set=[label],
        conformal_set_names=list(conformal),
        verdict=verdict,
        alpha=0.1,
        qhat=0.5,
        model_id="aion-nexus-test",
    )
    cert.seal(seed, scheme="ed25519", ttl_seconds=ttl, key_id=key_id, now_iso=_NOW)
    expected_pubkey = ed25519_pubkey_from_seed(seed)
    return cert, expected_pubkey


# --------------------------------------------------------------------------- #
# Protobuf codec — round-trips for every datatype we emit
# --------------------------------------------------------------------------- #

def test_protobuf_varint_roundtrip():
    for n in (0, 1, 127, 128, 300, 16384, 2**32, 2**63 - 1):
        encoded = pb.encode_varint(n)
        value, pos = pb.decode_varint(encoded, 0)
        assert value == n
        assert pos == len(encoded)


def test_protobuf_negative_varint_rejected():
    with pytest.raises(ValueError):
        pb.encode_varint(-1)


def test_metric_roundtrip_all_datatypes():
    metrics = [
        Metric("s", DT_STRING, "hello"),
        Metric("d", DT_DOUBLE, 3.5),
        Metric("b", DT_BOOLEAN, True),
        Metric("i", DT_INT64, 900),
        Metric("y", DT_BYTES, b"\x00\x01\x02"),
        Metric("n", DT_STRING, None, is_null=True),
    ]
    payload = encode_payload(metrics, seq=7, timestamp_ms=1234567890)
    decoded = decode_payload(payload)
    assert decoded["seq"] == 7
    assert decoded["timestamp"] == 1234567890
    by_name = {m["name"]: m for m in decoded["metrics"]}
    assert by_name["s"]["value"] == "hello"
    assert by_name["d"]["value"] == 3.5
    assert by_name["b"]["value"] is True
    assert by_name["i"]["value"] == 900
    assert by_name["y"]["value"] == b"\x00\x01\x02"
    assert by_name["n"]["is_null"] is True
    assert by_name["n"]["value"] is None


def test_seq_out_of_range_rejected():
    with pytest.raises(ValueError):
        encode_payload([], seq=256)


def test_negative_int64_metric_rejected():
    with pytest.raises(ValueError):
        encode_payload([Metric("x", DT_INT64, -5)], seq=0)


# --------------------------------------------------------------------------- #
# Mapping spine — diagnosis/trust separation and the honesty gate
# --------------------------------------------------------------------------- #

def test_certified_advanced_maps_to_high_alarm():
    cert, _ = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)
    fv = to_factory_verdict(cert)
    assert fv.condition_state == STATE_ALARM
    assert fv.severity == SEV_ALARM
    assert fv.active is True
    assert fv.actionable is True          # CERTIFIED + signed
    assert fv.verifiable is True


def test_certified_normal_is_not_an_alarm():
    cert, _ = _signed_cert("CERTIFIED", "normal", ("normal",), 0)
    fv = to_factory_verdict(cert)
    assert fv.active is False
    assert fv.severity == 1


def test_abstain_never_escalates_to_alarm():
    # Even with an 'advanced' diagnosis, an ABSTAIN must NOT raise a high alarm.
    cert, _ = _signed_cert("ABSTAIN", "advanced", ("normal", "early", "medium", "advanced"), 3)
    fv = to_factory_verdict(cert)
    assert fv.condition_state == STATE_DATA_UNCERTAIN
    assert fv.severity == SEV_ABSTAIN
    assert fv.severity < SEV_ALARM
    assert fv.actionable is False


def test_review_is_capped_to_warning_band():
    cert, _ = _signed_cert("REVIEW", "advanced", ("medium", "advanced"), 3)
    fv = to_factory_verdict(cert)
    assert fv.condition_state == STATE_REVIEW
    assert fv.severity <= SEV_REVIEW_CAP
    assert fv.actionable is False


def test_unsigned_certificate_is_not_actionable():
    cert = Certificate(
        predicted_label=3, predicted_name="advanced", conformal_set=[3],
        conformal_set_names=["advanced"], verdict="CERTIFIED")
    cert.seal(scheme="none")  # authentication = NONE
    fv = to_factory_verdict(cert)
    assert fv.verifiable is False
    assert fv.actionable is False         # CERTIFIED but NOT tamper-evident


# --------------------------------------------------------------------------- #
# THE headline: a signed verdict is verifiable off the bus
# --------------------------------------------------------------------------- #

def test_signed_verdict_is_verifiable_off_the_bus():
    cert, expected_pubkey = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)

    # Edge node encodes the certified verdict and publishes it as Sparkplug B.
    payload = build_sparkplug_payload(cert, seq=1)

    # A third party on the bus decodes the payload and pulls AION/certificate.
    decoded = decode_payload(payload)
    by_name = {m["name"]: m["value"] for m in decoded["metrics"]}
    assert by_name["AION/verdict"] == "CERTIFIED"
    assert by_name["AION/severity"] == SEV_ALARM
    cert_on_bus = json.loads(by_name["AION/certificate"])

    # ...and re-verifies it OFFLINE with the public key alone -> trusted.
    res = verify_certificate(cert_on_bus, expected_pubkey=expected_pubkey,
                             now_iso=_IN_WINDOW)
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "VERIFIED"
    assert res["trusted"] is True


def test_tampered_certificate_on_the_bus_is_rejected():
    cert, expected_pubkey = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)
    payload = build_sparkplug_payload(cert, seq=1)
    decoded = decode_payload(payload)
    by_name = {m["name"]: m["value"] for m in decoded["metrics"]}
    cert_on_bus = json.loads(by_name["AION/certificate"])

    # Forge the diagnosis: flip 'advanced' -> 'normal' (an attacker hiding a fault).
    cert_on_bus["predicted_name"] = "normal"
    res = verify_certificate(cert_on_bus, expected_pubkey=expected_pubkey,
                             now_iso=_IN_WINDOW)
    assert res["integrity_ok"] is False   # content_hash binds the label
    assert res["trusted"] is False


def test_expired_certificate_on_the_bus_is_not_trusted():
    cert, expected_pubkey = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3, ttl=3600)
    payload = build_sparkplug_payload(cert, seq=1)
    cert_on_bus = json.loads(
        {m["name"]: m["value"] for m in decode_payload(payload)["metrics"]}["AION/certificate"])
    res = verify_certificate(cert_on_bus, expected_pubkey=expected_pubkey, now_iso=_EXPIRED)
    assert res["expired"] is True
    assert res["trusted"] is False


def test_self_signed_without_expected_key_is_not_trusted():
    # Only the EMBEDDED pubkey available (no out-of-band expected key) -> SELF-SIGNED.
    cert, _ = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)
    cert_on_bus = json.loads(
        {m["name"]: m["value"]
         for m in decode_payload(build_sparkplug_payload(cert, seq=1))["metrics"]}["AION/certificate"])
    res = verify_certificate(cert_on_bus, now_iso=_IN_WINDOW)
    assert res["authenticity"] == "SELF-SIGNED"
    assert res["trusted"] is False


# --------------------------------------------------------------------------- #
# OPC UA Alarms & Conditions mapping
# --------------------------------------------------------------------------- #

def test_opcua_certified_advanced_is_active_good_alarm():
    cert, _ = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)
    model = build_condition_model(cert)
    c = model["condition"]
    assert c["Severity"] == SEV_ALARM
    assert c["ActiveStateId"] is True
    assert c["Quality"] == QUALITY_GOOD
    assert c["AckedState"] == "Unacknowledged"
    # The signed certificate rides as an OPC UA variable.
    assert json.loads(model["analysis"]["Certificate"])["verdict"] == "CERTIFIED"
    assert model["analysis"]["PublicKey"]


def test_opcua_abstain_is_uncertain_quality_low_severity():
    cert, _ = _signed_cert("ABSTAIN", "advanced", ("normal", "advanced"), 3)
    c = build_condition_model(cert)["condition"]
    assert c["Quality"] == QUALITY_UNCERTAIN   # OPC UA's own 'I am not sure'
    assert c["Severity"] == SEV_ABSTAIN
    assert c["ActiveStateId"] is False


def test_opcua_unsigned_is_uncertain_even_if_certified():
    cert = Certificate(
        predicted_label=3, predicted_name="advanced", conformal_set=[3],
        conformal_set_names=["advanced"], verdict="CERTIFIED")
    cert.seal(scheme="none")
    c = build_condition_model(cert)["condition"]
    assert c["Quality"] == QUALITY_UNCERTAIN   # not tamper-evident -> not Good


# --------------------------------------------------------------------------- #
# Topic namespace + graceful optional-dependency degradation
# --------------------------------------------------------------------------- #

def test_sparkplug_topic_namespace():
    assert sparkplug_topic("Plant1", "NDATA", "edge7") == "spBv1.0/Plant1/NDATA/edge7"
    assert sparkplug_topic("Plant1", "DDATA", "edge7", "bearing3") == \
        "spBv1.0/Plant1/DDATA/edge7/bearing3"


def test_publisher_requires_connect_before_publish():
    pub = SparkplugPublisher("Plant1", "edge7")
    cert, _ = _signed_cert()
    with pytest.raises(RuntimeError):
        pub.publish_verdict(cert)


def test_publisher_seq_wraps():
    pub = SparkplugPublisher("Plant1", "edge7")
    seqs = [pub._next_seq() for _ in range(258)]
    assert seqs[0] == 0
    assert seqs[255] == 255
    assert seqs[256] == 0   # wraps at 256 per Sparkplug spec


@pytest.mark.skipif(_HAS_PAHO, reason="paho-mqtt installed: ImportError path not exercised")
def test_publisher_connect_without_paho_gives_clear_error():
    pub = SparkplugPublisher("Plant1", "edge7")
    with pytest.raises(ImportError, match="aion-nexus\\[factory\\]"):
        pub.connect()


@pytest.mark.skipif(_HAS_ASYNCUA, reason="asyncua installed: ImportError path not exercised")
def test_opcua_server_without_asyncua_gives_clear_error():
    srv = CertifiedConditionMonitoringServer()
    with pytest.raises(ImportError, match="aion-nexus\\[factory\\]"):
        asyncio.run(srv.start())


# --------------------------------------------------------------------------- #
# Wire-compatibility: our bytes decode under the REFERENCE protobuf decoder.
# This is the "not JSON cosplay" proof — skipped if protobuf is not installed so
# it never becomes a hard test dependency (the codec itself needs nothing).
# --------------------------------------------------------------------------- #

def _reference_payload_class():
    """Build the Sparkplug B Payload/Metric subset as a runtime protobuf class."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "sparkplug_b_subset.proto"
    fdp.package = "spb_test"
    fdp.syntax = "proto2"
    payload = fdp.message_type.add()
    payload.name = "Payload"
    metric = fdp.message_type.add()
    metric.name = "Metric"
    F = descriptor_pb2.FieldDescriptorProto  # noqa: N806 — protobuf class alias

    def add(msg, name, number, ftype, label=F.LABEL_OPTIONAL, type_name=None):
        f = msg.field.add()
        f.name, f.number, f.label, f.type = name, number, label, ftype
        if type_name:
            f.type_name = type_name

    add(payload, "timestamp", 1, F.TYPE_UINT64)
    add(payload, "metrics", 2, F.TYPE_MESSAGE, F.LABEL_REPEATED, ".spb_test.Metric")
    add(payload, "seq", 3, F.TYPE_UINT64)
    add(metric, "name", 1, F.TYPE_STRING)
    add(metric, "alias", 2, F.TYPE_UINT64)
    add(metric, "timestamp", 3, F.TYPE_UINT64)
    add(metric, "datatype", 4, F.TYPE_UINT32)
    add(metric, "is_null", 7, F.TYPE_BOOL)
    add(metric, "long_value", 11, F.TYPE_UINT64)
    add(metric, "double_value", 13, F.TYPE_DOUBLE)
    add(metric, "boolean_value", 14, F.TYPE_BOOL)
    add(metric, "string_value", 15, F.TYPE_STRING)
    add(metric, "bytes_value", 16, F.TYPE_BYTES)

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName("spb_test.Payload"))


@pytest.mark.skipif(not _HAS_PROTOBUF, reason="protobuf not installed")
def test_sparkplug_bytes_are_wire_compatible_with_reference_protobuf():
    Payload = _reference_payload_class()  # noqa: N806 — protobuf message class
    cert, expected_pubkey = _signed_cert("CERTIFIED", "advanced", ("advanced",), 3)
    raw = build_sparkplug_payload(cert, seq=5, timestamp_ms=1234567890)

    msg = Payload()
    consumed = msg.ParseFromString(raw)     # reference Google protobuf decoder
    assert consumed == len(raw)             # every byte is valid protobuf
    assert msg.timestamp == 1234567890
    assert msg.seq == 5

    by_name = {m.name: m for m in msg.metrics}
    assert by_name["AION/verdict"].string_value == "CERTIFIED"
    assert by_name["AION/severity"].long_value == SEV_ALARM
    assert by_name["AION/active"].boolean_value is True

    # The certificate the REFERENCE decoder pulls out still verifies trusted.
    cert_on_bus = json.loads(by_name["AION/certificate"].string_value)
    res = verify_certificate(cert_on_bus, expected_pubkey=expected_pubkey,
                             now_iso=_IN_WINDOW)
    assert res["trusted"] is True
