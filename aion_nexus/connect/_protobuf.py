"""A tiny, self-contained Protocol Buffers wire codec — just the subset the
Sparkplug B payload needs, so the bridge emits REAL Sparkplug bytes with no
runtime ``protobuf`` dependency.

Scope (deliberately minimal, exactly the Sparkplug B Payload subset we use):
- varint (wire type 0): uint32/uint64/bool/enum
- 64-bit (wire type 1): IEEE-754 ``double`` (little-endian)
- length-delimited (wire type 2): ``string`` / ``bytes`` / embedded message

This is NOT a general protobuf library: no 32-bit fixed, no signed zig-zag, no
packed-repeated, no groups. It is enough to encode/decode the Sparkplug B
``Payload`` and ``Metric`` messages this package builds, and it round-trips with
itself (see ``tests/test_connect.py``). The field numbers and wire format follow
the published Protocol Buffers encoding spec, so the bytes are interoperable with
a real Sparkplug B / Eclipse Tahu decoder for the fields covered.

Wire format reference: a field is ``tag = (field_number << 3) | wire_type``
written as a varint, followed by the field's value encoded per its wire type.
"""
from __future__ import annotations

import struct

WIRE_VARINT = 0
WIRE_64BIT = 1
WIRE_LEN = 2


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def encode_varint(value: int) -> bytes:
    """Base-128 varint encoding of a non-negative integer."""
    if value < 0:
        raise ValueError("this minimal codec encodes non-negative varints only")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field_number: int, wire_type: int) -> bytes:
    return encode_varint((field_number << 3) | wire_type)


def write_varint_field(field_number: int, value: int) -> bytes:
    return _tag(field_number, WIRE_VARINT) + encode_varint(int(value))


def write_bool_field(field_number: int, value: bool) -> bytes:
    return write_varint_field(field_number, 1 if value else 0)


def write_double_field(field_number: int, value: float) -> bytes:
    # IEEE-754 binary64, little-endian (protobuf fixed64 / wire type 1).
    return _tag(field_number, WIRE_64BIT) + struct.pack("<d", float(value))


def write_bytes_field(field_number: int, value: bytes) -> bytes:
    return _tag(field_number, WIRE_LEN) + encode_varint(len(value)) + value


def write_string_field(field_number: int, value: str) -> bytes:
    return write_bytes_field(field_number, value.encode("utf-8"))


def write_message_field(field_number: int, message: bytes) -> bytes:
    """Embed a length-delimited sub-message (already-encoded bytes)."""
    return write_bytes_field(field_number, message)


# --------------------------------------------------------------------------- #
# Reader (round-trip / interop verification)
# --------------------------------------------------------------------------- #

def decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    """Decode a varint at ``pos``; return ``(value, new_pos)``."""
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("truncated varint")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long (>10 bytes)")


def parse_fields(buf: bytes) -> list[tuple[int, int, object]]:
    """Parse a message into a flat list of ``(field_number, wire_type, value)``.

    ``value`` is: an int for varints, a float for 64-bit, or raw ``bytes`` for
    length-delimited fields (the caller decides whether those bytes are a string,
    a packed value or an embedded message). Repeated fields appear multiple times.
    """
    pos = 0
    fields: list[tuple[int, int, object]] = []
    while pos < len(buf):
        tag, pos = decode_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == WIRE_VARINT:
            value, pos = decode_varint(buf, pos)
            fields.append((field_number, wire_type, value))
        elif wire_type == WIRE_64BIT:
            if pos + 8 > len(buf):
                raise ValueError("truncated 64-bit field")
            (value,) = struct.unpack_from("<d", buf, pos)
            pos += 8
            fields.append((field_number, wire_type, value))
        elif wire_type == WIRE_LEN:
            length, pos = decode_varint(buf, pos)
            if pos + length > len(buf):
                raise ValueError("truncated length-delimited field")
            chunk = buf[pos:pos + length]
            pos += length
            fields.append((field_number, wire_type, chunk))
        else:
            raise ValueError(f"unsupported wire type {wire_type} "
                             f"(this minimal codec handles 0/1/2 only)")
    return fields
