"""Unified signing primitives — separation of powers between minting and verifying.

This module gives ``aion_nexus.verify`` two signature schemes that sit side by
side, each making a DIFFERENT trust trade-off. Picking one is a deliberate
security decision, so the difference is stated up front:

Ed25519 (ASYMMETRIC) — the verifier CANNOT forge
-------------------------------------------------
A signing **seed** deterministically derives an Ed25519 key pair:

  - the PRIVATE key is the authority to MINT — only a holder of the seed can
    produce a signature that verifies;
  - the PUBLIC key is the capacity to VERIFY — a holder of the public key can
    check a signature but CANNOT produce a new valid one.

Separating the power to mint from the power to verify is the whole point. It
enables **independent public-key verification**: you publish the public key, and
a customer, an auditor, or an insurer verifies a certificate OFFLINE with the
public key ALONE — without ever receiving anything that would let them forge a
certificate in your name. This is the property that makes a signed certificate a
portable, third-party-checkable artifact.

HMAC-SHA256 (SYMMETRIC) — the verifier CAN forge
------------------------------------------------
HMAC uses ONE shared secret for both signing and verifying. It is fast and is
kept here for backward compatibility with the existing ``Certificate`` sealing
path (``authentication = "HMAC-SHA256"``). But note the trust ceiling: to let a
party VERIFY an HMAC you must give them the secret, and from that moment they can
FORGE signatures indistinguishable from yours. HMAC therefore proves authenticity
only *between mutually trusting parties who already share the secret* — it can
never support independent third-party verification the way Ed25519 does.

Honesty note (workspace rule 6.31)
----------------------------------
Tamper-evidence is real ONLY when a signature is verified against the EXPECTED
public key (Ed25519) or the EXPECTED shared secret (HMAC). A bare hash, or a
signature checked against an attacker-supplied key, proves nothing. The functions
below never raise on bad input — an invalid signature, malformed hex, or wrong
key yields ``False``, never an exception, so a verifier cannot be tricked into a
truthy result by feeding it garbage.

Determinism
-----------
Ed25519 (RFC 8032) is a DETERMINISTIC signature scheme: signing the same message
with the same seed always yields the byte-identical signature. This is a feature,
not a leak — it means a certificate's signature is reproducible and that two
honest signers of the same payload agree. ``test_signing.py`` pins this property.
"""
from __future__ import annotations

import hashlib
import hmac

_CRYPTOGRAPHY_HINT = (
    "Ed25519 signing requires the 'cryptography' package. Install it with "
    "`pip install cryptography` (>=41 ships Ed25519). HMAC signing "
    "(hmac_sign / hmac_verify) needs no extra dependency."
)


def _require_cryptography():
    """Import ``cryptography`` lazily with an actionable error if it is absent.

    Ed25519 support is an optional dependency: the HMAC path works with the
    standard library alone, so we do not import ``cryptography`` at module load.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as exc:  # pragma: no cover — exercised only without the dep
        raise RuntimeError(_CRYPTOGRAPHY_HINT) from exc
    return ed25519


def _privkey_from_seed(seed: bytes | str):
    """Derive the Ed25519 PRIVATE key deterministically from an arbitrary seed.

    The seed (bytes or str) is hashed to exactly 32 bytes with SHA-256 — the raw
    private-key size Ed25519 expects — so any seed length is accepted and the
    same seed always yields the same key. Holding this key is the authority to
    MINT signatures.
    """
    ed25519 = _require_cryptography()
    if isinstance(seed, str):
        seed = seed.encode("utf-8")
    raw = hashlib.sha256(bytes(seed)).digest()  # seed -> 32 deterministic bytes
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw)


def ed25519_pubkey_from_seed(seed: bytes | str) -> str:
    """Return the hex (raw 32-byte) Ed25519 PUBLIC key derived from ``seed``.

    The public key is safe to publish: it carries the capacity to VERIFY but NOT
    to mint. Ship it alongside a certificate (or publish it once) and any third
    party can verify your signatures with the public key alone.
    """
    from cryptography.hazmat.primitives import serialization

    pub = _privkey_from_seed(seed).public_key()
    return pub.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def ed25519_sign(message: str, seed: bytes | str) -> str:
    """Sign ``message`` with the PRIVATE key derived from ``seed``; return hex.

    Only a holder of the seed can produce a signature that verifies. ``message``
    is typically a certificate's ``content_hash``. Deterministic: same
    ``(seed, message)`` -> identical signature (RFC 8032).
    """
    sig = _privkey_from_seed(seed).sign(message.encode("utf-8"))
    return sig.hex()


def ed25519_verify(message: str, sig_hex: str, pubkey_hex: str) -> bool:
    """Verify ``sig_hex`` over ``message`` with the PUBLIC key ``pubkey_hex``.

    Verification confers NO power to mint. Returns ``True`` only when the
    signature is valid for this exact message under this exact public key; any
    invalid signature, malformed hex, or wrong key returns ``False`` and NEVER
    raises — a verifier cannot be coaxed into a truthy result with garbage input.
    """
    if not sig_hex or not pubkey_hex:
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )

        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pub.verify(bytes.fromhex(sig_hex), message.encode("utf-8"))
        return True
    except Exception:  # noqa: BLE001 — bad sig / wrong key / malformed hex -> False
        return False


def hmac_sign(message: str, key: bytes) -> str:
    """HMAC-SHA256 of ``message`` under shared secret ``key`` (hex).

    SYMMETRIC: anyone who can verify this can also forge it. Kept for backward
    compatibility with the certificate's HMAC sealing path. For independent
    third-party verification, use :func:`ed25519_sign` instead.
    """
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()


def hmac_verify(message: str, sig: str, key: bytes) -> bool:
    """Constant-time check of an HMAC-SHA256 signature; never raises.

    Uses :func:`hmac.compare_digest` to avoid timing side channels. Returns
    ``False`` on any malformed input rather than raising.
    """
    if not sig:
        return False
    try:
        expected = hmac.new(key, message.encode("utf-8"), hashlib.sha256).hexdigest()
        return hmac.compare_digest(str(sig), expected)
    except Exception:  # noqa: BLE001 — defensive: malformed input -> False, never raise
        return False
