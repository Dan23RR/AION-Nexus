"""Key custody — external signer + rotation + revocation (v2.20.0).

These prove the custody story that the offline-verifiable-certificate thesis
depends on: the private key can live OUTSIDE the process (KMS/HSM), and a
compromised key invalidates ONLY its own certificates, not the whole issuer.
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.verify import (
    ExternalSigner,
    KeyRing,
    Verifier,
    ed25519_pubkey_from_seed,
    ed25519_sign,
    generate_seed,
    verify_certificate,
    verify_with_keyring,
)

CLASSES = ["normal", "early", "medium", "advanced"]


def _verifier():
    v = Verifier(alpha=0.1, class_names=CLASSES)
    rng = np.random.default_rng(0)
    return v.calibrate(rng.dirichlet([1, 1, 1, 1], 200), np.array([0, 1, 2, 3] * 50))


def _kms_signer(seed):
    """An ExternalSigner whose 'device' holds the seed; AION only gets a callback."""
    return ExternalSigner(ed25519_pubkey_from_seed(seed), lambda m: ed25519_sign(m, seed))


def _cert(verifier, signer, key_id):
    return verifier.certify(np.array([0.05, 0.05, 0.1, 0.8]),
                            input_signal=np.zeros((2, 2560)),
                            signer=signer, key_id=key_id, ttl_seconds=3600)


def test_external_signer_signs_without_holding_the_key():
    seed = generate_seed()
    pub = ed25519_pubkey_from_seed(seed)
    cert = _cert(_verifier(), _kms_signer(seed), "k1")
    assert cert.authentication == "Ed25519"
    assert cert.pubkey == pub  # embedded public key, verifiable offline
    # the cert verifies against the expected (out-of-band) pubkey
    assert verify_certificate(cert.as_dict(), expected_pubkey=pub)["trusted"]


def test_active_key_is_trusted_via_keyring():
    seed = generate_seed()
    cert = _cert(_verifier(), _kms_signer(seed), "k1")
    ring = KeyRing().rotate("k1", ed25519_pubkey_from_seed(seed))
    res = verify_with_keyring(cert, ring)
    assert res["trusted"] and res["authenticity"] == "VERIFIED" and res["key_status"] == "active"


def test_rotation_retires_old_key_but_its_certs_stay_valid():
    seed = generate_seed()
    cert = _cert(_verifier(), _kms_signer(seed), "k1")
    ring = KeyRing().rotate("k1", ed25519_pubkey_from_seed(seed))
    ring.rotate("k2", ed25519_pubkey_from_seed(generate_seed()))  # k1 -> retired
    res = verify_with_keyring(cert, ring)
    assert res["key_status"] == "retired"
    assert res["trusted"]  # rotation does NOT retroactively invalidate


def test_revocation_invalidates_only_that_keys_certs():
    v = _verifier()
    s1, s2 = generate_seed(), generate_seed()
    cert1 = _cert(v, _kms_signer(s1), "k1")
    cert2 = _cert(v, _kms_signer(s2), "k2")
    ring = (KeyRing()
            .register("k1", ed25519_pubkey_from_seed(s1))
            .register("k2", ed25519_pubkey_from_seed(s2)))
    ring.revoke("k1", "seed exposed in incident-217")
    r1 = verify_with_keyring(cert1, ring)
    r2 = verify_with_keyring(cert2, ring)
    assert not r1["trusted"] and r1["authenticity"] == "REVOKED-KEY"
    assert "incident-217" in r1["key_note"]
    assert r2["trusted"]  # the other key is unaffected


def test_unknown_key_id_is_not_trusted():
    cert = _cert(_verifier(), _kms_signer(generate_seed()), "k1")
    res = verify_with_keyring(cert, KeyRing())  # empty registry
    assert not res["trusted"] and res["authenticity"] == "UNKNOWN-KEY"


def test_keyring_save_load_roundtrip(tmp_path):
    seed = generate_seed()
    ring = KeyRing().rotate("k1", ed25519_pubkey_from_seed(seed))
    ring.revoke("k1", "test")
    path = tmp_path / "keyring.json"
    ring.save(path)
    reloaded = KeyRing.load(path)
    assert reloaded.is_revoked("k1")
    assert reloaded.public_key_for("k1") == ed25519_pubkey_from_seed(seed)


def test_external_signer_validation():
    with pytest.raises(TypeError):
        ExternalSigner("abcd", "not-callable")
    with pytest.raises(ValueError):
        ExternalSigner(None, lambda m: "sig", scheme="Ed25519")  # Ed25519 needs a pubkey


def test_revoke_unknown_key_raises():
    with pytest.raises(KeyError):
        KeyRing().revoke("nope", "reason")
