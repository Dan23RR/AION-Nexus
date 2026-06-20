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

Entropy floor (red-team lesson — v2.6.0)
----------------------------------------
A signing seed IS the minting authority: anyone who can guess it can forge your
certificates. SHA-256-folding a short, low-entropy seed (``"1234"``, a word, a
PIN) to 32 bytes does NOT add entropy — the key is only as strong as the seed,
so a dictionary/brute-force attack over the seed recovers the private key. A red
team brute-forced ``"1234"`` in well under 10k tries and minted "valid"
certificates.

The closure has three parts:

  - :func:`generate_seed` — the BLESSED path: a full-entropy ``os.urandom(32)``
    seed (hex) that clears the floor. Mint identities this way.
  - ``strict=True`` on the primitives — REFUSE any seed below
    :data:`MIN_SEED_BYTES` (32). New code that takes an untrusted/operator-chosen
    seed should pass ``strict=True`` (or call :func:`assert_strong_seed`) so a
    weak seed fails loudly instead of producing a guessable key.
  - ``kdf=True`` — accept a memorable/low-entropy seed but stretch it with a slow,
    memory-hard KDF (scrypt). This raises the per-guess cost so brute force is
    expensive, but it can NEVER manufacture entropy the seed lacks — a truly
    guessable secret stays guessable. Prefer a real random seed.

HONESTY / backward-compat: the DEFAULT (``strict=False, kdf=False``) keeps the
legacy SHA-256 fold and does NOT raise, so existing callers that pass a memorable
seed (the certificate ``seal`` path, the store chain, the cheatbench harness)
remain byte-compatible. The floor is therefore a capability you OPT INTO, not a
silent breaking change — but it is the recommended posture for any new minting
surface, and the brute-force probe is genuinely closed for code that uses
``generate_seed`` or ``strict=True``.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import os
from collections import Counter
from typing import Protocol, runtime_checkable

# A raw Ed25519 private key is 32 bytes; we require at least that much seed
# material so the derived key carries full entropy rather than being a stretch of
# a guessable secret. ``generate_seed()`` returns exactly this many bytes (hex).
MIN_SEED_BYTES = 32

# Length alone is a weak proxy for entropy: a 32-byte constant (b"a"*32) or a
# repeated word clears the byte floor yet is trivially brute-forceable. The strict
# path therefore ALSO requires a minimum byte-alphabet size and Shannon entropy.
# Thresholds chosen so a full-entropy seed (os.urandom(32) raw, or its 64-char hex
# from generate_seed()) clears them comfortably, while b"a"*32, b"ab"*16 and a
# repeated passphrase do not.
MIN_SEED_DISTINCT = 8        # distinct byte values required
MIN_SEED_ENTROPY_BITS = 2.5  # Shannon bits per byte required

# scrypt cost parameters for the OPTIONAL kdf=True path. These are interactive-
# login-grade (RFC 7914 suggests n=2**14 for interactive use); they make each
# brute-force guess cost real CPU/memory without making honest signing slow.
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
# Fixed, NON-secret domain-separation salt. A KDF salt prevents precomputation
# across DOMAINS and need not be secret; pinning it keeps key derivation
# DETERMINISTIC (same seed -> same key -> reproducible signatures), which the
# certificate path relies on. It is NOT a substitute for seed entropy.
_KDF_SALT = b"aion-nexus.verify.signing/ed25519-kdf/v1"

_CRYPTOGRAPHY_HINT = (
    "Ed25519 signing requires the 'cryptography' package. Install it with "
    "`pip install cryptography` (>=41 ships Ed25519). HMAC signing "
    "(hmac_sign / hmac_verify) needs no extra dependency."
)

_WEAK_SEED_HINT = (
    "weak signing seed: %s. A guessable seed IS brute-forceable and a SHA-256 "
    "fold does NOT add entropy, so the derived Ed25519 private key would be "
    "guessable (a red team recovered '1234' in <10k tries). Use a full-entropy "
    "seed from generate_seed() (os.urandom(32) hex), or pass kdf=True to stretch "
    "a memorable seed with scrypt (slows brute force but CANNOT create entropy "
    "the seed lacks)."
)


def generate_seed() -> str:
    """Return a fresh, full-entropy Ed25519 signing seed as hex (32 random bytes).

    This is the RECOMMENDED way to mint a signer identity: ``os.urandom(32)`` is
    cryptographically secure and clears the :data:`MIN_SEED_BYTES` entropy floor.
    Treat the result as a SECRET (it is the authority to mint); publish only the
    public key from :func:`ed25519_pubkey_from_seed`.
    """
    return os.urandom(MIN_SEED_BYTES).hex()


def _seed_bytes(seed: bytes | str) -> bytes:
    """Normalise a seed to bytes (str -> utf-8) without changing its entropy."""
    return seed.encode("utf-8") if isinstance(seed, str) else bytes(seed)


def _weak_seed_reason(raw: bytes) -> str | None:
    """Return WHY ``raw`` is too weak to derive a key, or ``None`` if it is strong.

    Catches BOTH failure modes: (1) below the length floor, and (2) long but
    low-entropy (a constant, a repeated word/PIN) — which a length-only check
    misses even though such a seed is brute-forceable.
    """
    if len(raw) < MIN_SEED_BYTES:
        return f"{len(raw)} byte(s) < the {MIN_SEED_BYTES}-byte entropy floor"
    n = len(raw)
    distinct = len(set(raw))
    bits = -sum((c / n) * math.log2(c / n) for c in Counter(raw).values())
    if distinct < MIN_SEED_DISTINCT or bits < MIN_SEED_ENTROPY_BITS:
        return (f"below the entropy floor: low entropy ({distinct} distinct byte "
                f"value(s), {bits:.2f} bits/byte < the {MIN_SEED_ENTROPY_BITS} "
                f"floor) — a long but guessable seed (constant, repeated word, "
                f"PIN) is still brute-forceable")
    return None


def assert_strong_seed(seed: bytes | str) -> None:
    """Raise ``ValueError`` unless ``seed`` clears the entropy floor.

    The explicit gate behind ``strict=True``. Call it at any minting surface that
    accepts an untrusted or operator-chosen seed to fail loudly on a guessable
    secret BEFORE a key is derived from it. Checks BOTH the length floor
    (:data:`MIN_SEED_BYTES`) AND the entropy floor (:data:`MIN_SEED_DISTINCT`
    distinct bytes, :data:`MIN_SEED_ENTROPY_BITS` Shannon bits/byte) — so a
    32-byte constant or a repeated word is REJECTED, not just a short seed. A
    full-entropy seed from :func:`generate_seed` passes silently.
    """
    reason = _weak_seed_reason(_seed_bytes(seed))
    if reason is not None:
        raise ValueError(_WEAK_SEED_HINT % reason)


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


def _derive_raw32(seed: bytes | str, *, kdf: bool, strict: bool) -> bytes:
    """Map an arbitrary seed to the 32 raw private-key bytes Ed25519 expects.

    Selects the stretching function and (optionally) enforces the entropy floor:

    - ``kdf=True``: stretch the seed with scrypt (slow, memory-hard) under a fixed
      domain-separation salt — accepts a memorable seed, raises the per-guess
      cost of brute force, but does NOT add entropy the seed lacks. (``kdf`` wins
      over ``strict``: stretching is the deliberate way to use a short seed.)
    - else ``strict=True``: REQUIRE the seed to clear the entropy floor
      (:func:`_weak_seed_reason`: length AND distinct-byte/Shannon-entropy — so a
      32-byte constant or a repeated word is rejected, not just a short seed),
      then fold with one SHA-256 pass.
    - else (default ``strict=False, kdf=False``): the LEGACY behaviour — fold with
      SHA-256 and do NOT raise. Kept for byte-compatibility with existing callers;
      the floor is opt-in via ``strict`` / :func:`generate_seed`.

    All paths are DETERMINISTIC (same seed + same flags -> same key), which the
    certificate signing path relies on for reproducible signatures.
    """
    raw = _seed_bytes(seed)
    if kdf:
        # scrypt is memory-hard; cost params are module-level and fixed so the
        # derivation stays deterministic and reproducible across processes.
        return hashlib.scrypt(
            raw, salt=_KDF_SALT, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    if strict:
        reason = _weak_seed_reason(raw)
        if reason is not None:
            raise ValueError(_WEAK_SEED_HINT % reason)
    return hashlib.sha256(raw).digest()  # seed -> 32 deterministic bytes


def _privkey_from_seed(seed: bytes | str, *, kdf: bool = False, strict: bool = False):
    """Derive the Ed25519 PRIVATE key deterministically from a seed.

    Holding this key is the authority to MINT signatures. With ``strict=True`` the
    seed must clear the :data:`MIN_SEED_BYTES` entropy floor; ``kdf=True`` stretches
    a memorable seed with scrypt instead. See :func:`_derive_raw32`.
    """
    ed25519 = _require_cryptography()
    raw = _derive_raw32(seed, kdf=kdf, strict=strict)
    return ed25519.Ed25519PrivateKey.from_private_bytes(raw)


def ed25519_pubkey_from_seed(seed: bytes | str, *, kdf: bool = False,
                             strict: bool = False) -> str:
    """Return the hex (raw 32-byte) Ed25519 PUBLIC key derived from ``seed``.

    The public key is safe to publish: it carries the capacity to VERIFY but NOT
    to mint. Ship it alongside a certificate (or publish it once) and any third
    party can verify your signatures with the public key alone.

    Seed strength: by DEFAULT this derives with the legacy SHA-256 fold and does
    not enforce the floor (backward-compatible). Pass ``strict=True`` to REJECT a
    seed below :data:`MIN_SEED_BYTES`, or ``kdf=True`` to stretch a memorable seed
    with scrypt. The SAME ``kdf`` / ``strict`` flags must be used for signing and
    for deriving the matching pubkey, or the keys will not correspond.
    """
    from cryptography.hazmat.primitives import serialization

    pub = _privkey_from_seed(seed, kdf=kdf, strict=strict).public_key()
    return pub.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def ed25519_sign(message: str, seed: bytes | str, *, kdf: bool = False,
                 strict: bool = False) -> str:
    """Sign ``message`` with the PRIVATE key derived from ``seed``; return hex.

    Only a holder of the seed can produce a signature that verifies. ``message``
    is typically a certificate's ``content_hash`` (in v2.6.0, the per-certificate
    ``signing_payload`` that also binds expiry and identity). Deterministic: same
    ``(seed, message, kdf, strict)`` -> identical signature (RFC 8032).

    Seed strength: by DEFAULT derives with the legacy SHA-256 fold and does not
    raise (backward-compatible). Pass ``strict=True`` to REJECT a seed below
    :data:`MIN_SEED_BYTES` (closes the brute-forceable-seed probe — a weak seed
    raises ``ValueError``), or ``kdf=True`` to stretch a memorable seed with
    scrypt. Prefer a full-entropy seed from :func:`generate_seed`.
    """
    sig = _privkey_from_seed(
        seed, kdf=kdf, strict=strict).sign(message.encode("utf-8"))
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


# --------------------------------------------------------------------------- #
# Signer interface — pluggable signing for the enterprise / KMS bar
# --------------------------------------------------------------------------- #
#
# The certificate path currently calls ed25519_sign / hmac.new directly. That
# couples sealing to a LOCAL seed living in this process — fine for a single
# tenant, but it is exactly the coupling an enterprise / regulated deployment
# pushes back on: the minting key must live in a KMS or HSM, never in app memory.
#
# The Signer protocol decouples "what we sign" from "who holds the key". A signer
# is asked to SIGN A MESSAGE and to REPORT how a verifier should check it; it is
# never asked for the key itself. A future KmsSigner / HsmSigner implements the
# same two methods by delegating the signature to the external device — the seed
# never enters this process. Verification stays unchanged: an Ed25519 signature
# is still checked with the published public key by ed25519_verify, whether the
# private half lives in os.urandom bytes here or in a cloud HSM.


@runtime_checkable
class Signer(Protocol):
    """A pluggable signing authority. It signs messages; it never exposes the key.

    Implementations sign a message (typically a certificate's signing payload)
    and advertise the metadata a verifier needs:

    - :meth:`sign` returns the hex signature over ``message``.
    - :attr:`scheme` is the authentication label to stamp on the certificate
      (``"Ed25519"`` or ``"HMAC-SHA256"``), matching the constants in
      :mod:`aion_nexus.verify.certificate`.
    - :attr:`public_material` is what a verifier checks against: the hex PUBLIC
      key for Ed25519 (safe to embed/publish), or ``None`` for HMAC (the secret
      must NOT be embedded — verification needs the shared secret out-of-band).

    A KMS/HSM-backed signer implements this same surface by forwarding ``sign``
    to the external device; the private key never enters this process. That is
    why the interface is "sign a message", not "give me the key".
    """

    @property
    def scheme(self) -> str: ...

    @property
    def public_material(self) -> str | None: ...

    def sign(self, message: str) -> str: ...


class LocalEd25519Signer:
    """An :class:`Signer` backed by a LOCAL Ed25519 seed held in this process.

    This is the default, single-tenant signer: the seed lives in memory, so it is
    only as protected as the host. For a regulated deployment, swap in a
    KMS/HSM-backed signer with the same interface so the seed never enters the
    application.

    Seed strength: defaults to ``strict=True`` (this is NEW minting code, so it
    SHOULD reject a weak seed — a guessable signer key is the whole red-team
    finding). Pass ``strict=False`` to keep the legacy fold, or ``kdf=True`` to
    stretch a memorable seed with scrypt. Prefer :func:`generate_seed`.
    """

    scheme = "Ed25519"

    def __init__(self, seed: bytes | str, *, kdf: bool = False,
                 strict: bool = True) -> None:
        self._seed = seed
        self._kdf = kdf
        self._strict = strict
        # Derive eagerly so a weak seed fails fast at construction, not at sign().
        self._pubkey = ed25519_pubkey_from_seed(seed, kdf=kdf, strict=strict)

    @property
    def public_material(self) -> str | None:
        """The hex Ed25519 PUBLIC key — safe to embed in / publish with a cert."""
        return self._pubkey

    def sign(self, message: str) -> str:
        return ed25519_sign(message, self._seed, kdf=self._kdf, strict=self._strict)


class HmacSigner:
    """An :class:`Signer` backed by a shared HMAC-SHA256 secret.

    SYMMETRIC: whoever can verify can also forge (see module docstring). Kept for
    parity with the certificate's HMAC path. :attr:`public_material` is ``None``
    by design — the secret must NEVER be embedded in the certificate, since
    embedding it would hand every reader the power to mint.
    """

    scheme = "HMAC-SHA256"
    public_material = None  # the secret is NOT public material; do not embed it

    def __init__(self, key: bytes | str) -> None:
        self._key = key.encode("utf-8") if isinstance(key, str) else bytes(key)

    def sign(self, message: str) -> str:
        return hmac_sign(message, self._key)


class ExternalSigner:
    """A :class:`Signer` whose private key lives OUTSIDE this process (KMS / HSM).

    The custody fix: the application holds ONLY the PUBLIC key and a ``sign``
    callback that forwards the message to an external device (AWS KMS, Cloud HSM,
    Azure Key Vault via PKCS#11, ...). The private key NEVER enters the AION
    process, so a regulated deployment can mint signed certificates without
    in-process key material, with rotation/revocation managed by the device::

        signer = ExternalSigner(pubkey_hex, lambda msg: kms.sign(KEY_ID, msg))
        cert.seal_with(signer, key_id="kms-2026-q3", ttl_seconds=86400)

    The callback MUST return a hex Ed25519 signature over the EXACT message string,
    verifiable by ``ed25519_verify(msg, sig, pubkey_hex)``. Scheme defaults to
    Ed25519 (third-party-verifiable). The interface is "sign a message", never
    "give me the key" — which is the whole point.
    """

    def __init__(self, public_key: str | None, sign_callback, *,
                 scheme: str = "Ed25519") -> None:
        if not callable(sign_callback):
            raise TypeError("sign_callback must be callable (message -> hex signature)")
        if scheme == "Ed25519" and not public_key:
            raise ValueError("an Ed25519 ExternalSigner needs its public key (to embed/verify)")
        self._pub = str(public_key) if public_key is not None else None
        self._cb = sign_callback
        self._scheme = str(scheme)

    @property
    def scheme(self) -> str:
        return self._scheme

    @property
    def public_material(self) -> str | None:
        return self._pub

    def sign(self, message: str) -> str:
        sig = self._cb(str(message))
        if not isinstance(sig, str):
            raise TypeError("external sign callback must return a hex signature string")
        return sig
