"""Tests for aion_nexus.verify.signing — the unified signing primitives.

The contracts under test are the ones that make a signed certificate trustworthy
AND honest about its trust model:

  * Ed25519 round-trip: a signature minted from the seed verifies under the
    derived public key.
  * Wrong public key -> verification fails (a different identity cannot vouch).
  * Tampered signature -> verification fails (the signature binds the message).
  * Determinism (RFC 8032): the same (seed, message) yields the byte-identical
    signature.
  * SEPARATION OF POWERS: a party who holds ONLY the public key cannot produce a
    signature that verifies — minting requires the seed/private key. This is the
    property that distinguishes Ed25519 from HMAC and enables independent
    public-key verification.
  * HMAC round-trip + symmetry caveat: the shared secret both signs and verifies.
  * No verifier raises on malformed input — bad hex / wrong key -> False.
"""
from __future__ import annotations

from aion_nexus.verify.signing import (
    ed25519_pubkey_from_seed,
    ed25519_sign,
    ed25519_verify,
    hmac_sign,
    hmac_verify,
)

# A stand-in for the kind of message we sign in production: a content hash.
MESSAGE = "a" * 64  # 64 hex chars, the shape of a sha256 content_hash
SEED = b"aion-nexus signing seed v2.5.0"
OTHER_SEED = b"a different signer entirely"


# --------------------------------------------------------------------------- #
# Ed25519 — asymmetric: the verifier cannot forge
# --------------------------------------------------------------------------- #

def test_ed25519_round_trip_verifies():
    """A signature minted from the seed verifies under the derived public key."""
    pub = ed25519_pubkey_from_seed(SEED)
    sig = ed25519_sign(MESSAGE, SEED)
    assert ed25519_verify(MESSAGE, sig, pub) is True


def test_ed25519_wrong_pubkey_fails():
    """A signature does NOT verify under a different identity's public key."""
    sig = ed25519_sign(MESSAGE, SEED)
    wrong_pub = ed25519_pubkey_from_seed(OTHER_SEED)
    assert ed25519_verify(MESSAGE, sig, wrong_pub) is False


def test_ed25519_tampered_signature_fails():
    """Flipping a byte of the signature breaks verification."""
    pub = ed25519_pubkey_from_seed(SEED)
    sig = ed25519_sign(MESSAGE, SEED)
    # Mutate the first hex nibble to a definitely different value.
    flipped = ("f" if sig[0] != "f" else "0") + sig[1:]
    assert flipped != sig
    assert ed25519_verify(MESSAGE, flipped, pub) is False


def test_ed25519_tampered_message_fails():
    """The signature binds the exact message: a changed message no longer verifies."""
    pub = ed25519_pubkey_from_seed(SEED)
    sig = ed25519_sign(MESSAGE, SEED)
    assert ed25519_verify(MESSAGE + "x", sig, pub) is False


def test_ed25519_is_deterministic():
    """RFC 8032 determinism: same (seed, message) -> byte-identical signature."""
    sig_a = ed25519_sign(MESSAGE, SEED)
    sig_b = ed25519_sign(MESSAGE, SEED)
    assert sig_a == sig_b


def test_ed25519_pubkey_is_deterministic_and_raw32():
    """The derived public key is stable and is a raw 32-byte (64 hex) key."""
    pub_a = ed25519_pubkey_from_seed(SEED)
    pub_b = ed25519_pubkey_from_seed(SEED)
    assert pub_a == pub_b
    assert len(bytes.fromhex(pub_a)) == 32


def test_ed25519_seed_accepts_str_and_bytes_equivalently():
    """A str seed and its utf-8 bytes derive the same identity (str is encoded)."""
    pub_bytes = ed25519_pubkey_from_seed(b"shared-seed")
    pub_str = ed25519_pubkey_from_seed("shared-seed")
    assert pub_bytes == pub_str


# --------------------------------------------------------------------------- #
# SEPARATION OF POWERS — the heart of why Ed25519 enables third-party verify
# --------------------------------------------------------------------------- #

def test_public_key_holder_cannot_mint():
    """Holding ONLY the public key gives no power to forge a valid signature.

    We model an adversary who has captured the published public key (the verify
    capability) but not the seed (the mint capability). They have no API that
    takes a public key and emits a signature; the best they can do is fabricate
    bytes. We demonstrate that no public-key-derived material verifies as a
    signature — minting structurally requires the seed.
    """
    pub = ed25519_pubkey_from_seed(SEED)

    # The adversary cannot call ed25519_sign without a seed. Any signature they
    # forge from a DIFFERENT seed (the only seeds they could possibly hold) fails
    # under the genuine public key...
    forged = ed25519_sign(MESSAGE, OTHER_SEED)
    assert ed25519_verify(MESSAGE, forged, pub) is False

    # ...and naive attempts to reuse the public-key bytes as if they were a
    # signature do not verify either (they are not a signature over the message).
    assert ed25519_verify(MESSAGE, pub, pub) is False

    # Only the legitimate seed-holder produces something that verifies.
    legit = ed25519_sign(MESSAGE, SEED)
    assert ed25519_verify(MESSAGE, legit, pub) is True


def test_ed25519_verify_never_raises_on_garbage():
    """Malformed signature / public key returns False, never an exception."""
    pub = ed25519_pubkey_from_seed(SEED)
    assert ed25519_verify(MESSAGE, "not-hex", pub) is False
    assert ed25519_verify(MESSAGE, "ab", pub) is False          # valid hex, wrong length
    assert ed25519_verify(MESSAGE, ed25519_sign(MESSAGE, SEED), "zz") is False
    assert ed25519_verify(MESSAGE, "", pub) is False
    assert ed25519_verify(MESSAGE, "ab", "") is False


# --------------------------------------------------------------------------- #
# HMAC — symmetric: kept for backward compatibility (verifier CAN forge)
# --------------------------------------------------------------------------- #

def test_hmac_round_trip_verifies():
    """The shared secret both signs and verifies an HMAC."""
    key = b"shared-secret"
    sig = hmac_sign(MESSAGE, key)
    assert hmac_verify(MESSAGE, sig, key) is True


def test_hmac_wrong_key_fails():
    """A different secret does not verify the HMAC."""
    sig = hmac_sign(MESSAGE, b"shared-secret")
    assert hmac_verify(MESSAGE, sig, b"other-secret") is False


def test_hmac_tampered_message_fails():
    """The HMAC binds the message: a changed message no longer verifies."""
    key = b"shared-secret"
    sig = hmac_sign(MESSAGE, key)
    assert hmac_verify(MESSAGE + "x", sig, key) is False


def test_hmac_verify_never_raises_on_empty_sig():
    """An empty / falsy signature returns False rather than raising."""
    assert hmac_verify(MESSAGE, "", b"k") is False


def test_hmac_symmetry_means_verifier_can_forge():
    """Document the trust ceiling: with the shared secret, a verifier can forge.

    This is the precise sense in which HMAC is weaker than Ed25519 for
    third-party verification: the secret needed to VERIFY is the same secret
    needed to SIGN, so any verifier is also a minter.
    """
    key = b"shared-secret"
    # An auditor given the key to verify can mint a NEW signature over any
    # message and it verifies — exactly what Ed25519 prevents.
    forged = hmac_sign("attacker-chosen message", key)
    assert hmac_verify("attacker-chosen message", forged, key) is True
