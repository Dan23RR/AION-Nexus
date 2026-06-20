"""Certified-serving tests — the signed certificate wired into the product (v2.6.0).

Covers the /predict_certified + /verify endpoints, the checkpoint hash pin, and
the certify() ttl/key_id delegation. The contracts under test are the ones that
make the weapon REAL and HONEST (workspace 6.31):

  * /predict_certified emits a VERIFIABLE certificate; with VERIFY_ED25519_SEED in
    env it is Ed25519-signed and TRUSTED only against the EXPECTED issuer pubkey.
  * strict mode (AION_REQUIRE_SIGNED_CERT=1) without a key -> 503, never an
    unsigned cert silently passed off as authenticated.
  * no key, non-strict -> authentication=NONE + an explicit warning (honest).
  * /verify reproduces the offline auditor verdict (integrity / authenticity /
    trusted / expired).
  * an EXPIRED cert -> expired=True and trusted=False (anti-replay).
  * a cert signed by a DIFFERENT seed -> FORGED / not trusted.
  * the certificate is appended to the hash-chained store.

The server app is a process-level singleton; we use the same fresh-engine fixture
discipline as test_api_integration and inject a calibrated verifier per test.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from aion_nexus.config import CLASS_NAMES
from aion_nexus.verify import (
    AUTH_ED25519,
    AUTH_NONE,
    CertificateStore,
    Verifier,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)

# --------------------------------------------------------------------------- #
# Fixtures — calibrated verifier + a server client with verifier/store injected
# --------------------------------------------------------------------------- #

def _calibrated_verifier(engine) -> Verifier:
    """A conformal verifier calibrated on a tiny synthetic set (placeholder)."""
    rng = np.random.default_rng(123)
    probs_list, labels_list = [], []
    for cls in range(len(CLASS_NAMES)):
        for _ in range(8):
            sig = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
            res = engine.predict(sig)
            probs_list.append(
                np.array([res.probabilities[n] for n in CLASS_NAMES], dtype=np.float64))
            labels_list.append(cls)
    v = Verifier(alpha=0.1, class_names=list(CLASS_NAMES))
    v.calibrate(np.vstack(probs_list), np.array(labels_list, dtype=int))
    return v


@pytest.fixture
def app():
    from server.main import app as _app
    return _app


@pytest.fixture(autouse=True)
def fresh_state(app, tmp_path, monkeypatch):
    """Inject a fresh engine + calibrated verifier + temp-file store per test.

    A temp store path keeps the hash-chained audit log isolated and avoids
    polluting the repo with certificates.jsonl.
    """
    from aion_nexus import InferenceEngine
    from aion_nexus.model import create_aion_nexus

    torch.manual_seed(0)
    saved = (
        getattr(app.state, "engine", None),
        getattr(app.state, "startup_error", None),
        getattr(app.state, "verifier", None),
        getattr(app.state, "cert_store", None),
        getattr(app.state, "expected_checkpoint_sha256", None),
        getattr(app.state, "coverage_basis", None),
        getattr(app.state, "calibration_meta", None),
    )
    engine = InferenceEngine(create_aion_nexus())
    app.state.engine = engine
    app.state.startup_error = None
    app.state.verifier = _calibrated_verifier(engine)
    app.state.cert_store = CertificateStore(path=tmp_path / "certs.jsonl")
    app.state.expected_checkpoint_sha256 = None
    app.state.coverage_basis = "synthetic-placeholder"
    app.state.calibration_meta = None
    yield
    (app.state.engine, app.state.startup_error, app.state.verifier,
     app.state.cert_store, app.state.expected_checkpoint_sha256,
     app.state.coverage_basis, app.state.calibration_meta) = saved


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


def _signal():
    return np.random.default_rng(7).standard_normal((2, 2560)).tolist()


# --------------------------------------------------------------------------- #
# /predict_certified
# --------------------------------------------------------------------------- #

class TestPredictCertified:
    def test_signed_cert_is_verifiable_with_expected_pubkey(self, client, monkeypatch):
        """With VERIFY_ED25519_SEED set, the cert is Ed25519-signed and TRUSTED
        only against the EXPECTED issuer pubkey (the third-party-verifiable claim)."""
        seed = generate_seed()
        monkeypatch.setenv("VERIFY_ED25519_SEED", seed)
        r = client.post("/predict_certified", json={"signal": _signal()})
        assert r.status_code == 200, r.text
        body = r.json()
        cert = body["certificate"]
        assert cert["authentication"] == AUTH_ED25519
        # signed -> no UNSIGNED warning (a coverage-basis note may still be present
        # when the served verifier is on the synthetic placeholder, v2.16.0).
        assert "not tamper-evident" not in (body["warning"] or "").lower()
        assert body["pubkey"] == ed25519_pubkey_from_seed(seed)
        # Offline auditor verdict: trusted ONLY against the expected issuer key.
        audit = verify_certificate(cert, expected_pubkey=body["pubkey"])
        assert audit["integrity_ok"] is True
        assert audit["authenticity"] == "VERIFIED"
        assert audit["trusted"] is True

    def test_self_signed_without_expected_pubkey_is_not_trusted(self, client, monkeypatch):
        """An Ed25519 cert checked against only its EMBEDDED key is SELF-SIGNED,
        never trusted — the honesty rule: embedded != issuer-authenticated."""
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        cert = client.post("/predict_certified", json={"signal": _signal()}).json()["certificate"]
        audit = verify_certificate(cert)  # no expected_pubkey
        assert audit["authenticity"] == "SELF-SIGNED"
        assert audit["trusted"] is False

    def test_strict_mode_without_key_returns_503(self, client, monkeypatch):
        """AION_REQUIRE_SIGNED_CERT=1 and no signing key -> refuse with 503."""
        monkeypatch.delenv("VERIFY_ED25519_SEED", raising=False)
        monkeypatch.delenv("VERIFY_HMAC_KEY", raising=False)
        monkeypatch.setenv("AION_REQUIRE_SIGNED_CERT", "1")
        r = client.post("/predict_certified", json={"signal": _signal()})
        assert r.status_code == 503, r.text
        assert "signing key not configured" in r.json()["detail"]

    def test_weak_ed25519_seed_refused_at_product_boundary(self, client, monkeypatch):
        """SECURITY REGRESSION: a guessable Ed25519 seed (the red-team's '1234'
        kill-shot) must be REFUSED with 503 at the product boundary, never used to
        mint a brute-forceable 'trusted' certificate. The library stays
        back-compatible (legacy fold), so the server enforces the entropy floor."""
        monkeypatch.setenv("VERIFY_ED25519_SEED", "1234")
        monkeypatch.delenv("AION_REQUIRE_SIGNED_CERT", raising=False)
        r = client.post("/predict_certified", json={"signal": _signal()})
        assert r.status_code == 503, r.text
        assert "too weak" in r.json()["detail"].lower()
        assert "generate_seed" in r.json()["detail"]

    def test_no_key_non_strict_emits_none_with_warning(self, client, monkeypatch):
        """No key, non-strict -> authentication=NONE plus an explicit honest warning."""
        monkeypatch.delenv("VERIFY_ED25519_SEED", raising=False)
        monkeypatch.delenv("VERIFY_HMAC_KEY", raising=False)
        monkeypatch.delenv("AION_REQUIRE_SIGNED_CERT", raising=False)
        body = client.post("/predict_certified", json={"signal": _signal()}).json()
        assert body["certificate"]["authentication"] == AUTH_NONE
        assert body["pubkey"] is None
        assert body["warning"] is not None
        assert "not tamper-evident" in body["warning"].lower()

    def test_cert_has_validity_window_bound_by_ttl(self, client, monkeypatch):
        """The served cert carries a not_before/valid_until window and a jti."""
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        monkeypatch.setenv("AION_CERT_TTL_SECONDS", "3600")
        monkeypatch.setenv("AION_CERT_KEY_ID", "kid-42")
        cert = client.post("/predict_certified", json={"signal": _signal()}).json()["certificate"]
        assert cert["not_before"] and cert["valid_until"]
        assert cert["jti"]
        assert cert["key_id"] == "kid-42"

    def test_cert_appended_to_store(self, client, app, monkeypatch):
        """Every certified prediction is appended to the hash-chained audit store."""
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        before = sum(1 for _ in app.state.cert_store.iter_certs())
        assert client.post("/predict_certified", json={"signal": _signal()}).status_code == 200
        after = list(app.state.cert_store.iter_certs())
        assert len(after) == before + 1
        # The chain is intact (integrity holds regardless of auth scheme).
        assert app.state.cert_store.verify_chain().integrity_ok is True

    def test_ragged_signal_returns_400(self, client, monkeypatch):
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        r = client.post("/predict_certified", json={"signal": [[1.0, 2.0, 3.0], [1.0]]})
        assert r.status_code == 400, r.text

    def test_certified_requires_api_key_when_set(self, client, monkeypatch):
        monkeypatch.setenv("AION_API_KEY", "secret")
        assert client.post("/predict_certified", json={"signal": _signal()}).status_code == 401


# --------------------------------------------------------------------------- #
# /verify
# --------------------------------------------------------------------------- #

class TestVerifyEndpoint:
    def test_verify_trusted_with_expected_pubkey(self, client, monkeypatch):
        seed = generate_seed()
        monkeypatch.setenv("VERIFY_ED25519_SEED", seed)
        body = client.post("/predict_certified", json={"signal": _signal()}).json()
        r = client.post("/verify", json={
            "certificate": body["certificate"], "expected_pubkey": body["pubkey"]})
        assert r.status_code == 200, r.text
        v = r.json()
        assert v["integrity_ok"] is True
        assert v["authenticity"] == "VERIFIED"
        assert v["trusted"] is True

    def test_verify_tampered_field_not_trusted(self, client, monkeypatch):
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        body = client.post("/predict_certified", json={"signal": _signal()}).json()
        cert = dict(body["certificate"])
        forged = next(n for n in CLASS_NAMES if n != cert["predicted_name"])
        cert["predicted_name"] = forged  # tamper a display label
        v = client.post("/verify", json={
            "certificate": cert, "expected_pubkey": body["pubkey"]}).json()
        assert v["integrity_ok"] is False
        assert v["trusted"] is False

    def test_verify_wrong_seed_is_forged(self, client, monkeypatch):
        monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
        body = client.post("/predict_certified", json={"signal": _signal()}).json()
        other_pub = ed25519_pubkey_from_seed(generate_seed())
        v = client.post("/verify", json={
            "certificate": body["certificate"], "expected_pubkey": other_pub}).json()
        assert v["authenticity"] == "FORGED"
        assert v["trusted"] is False

    def test_verify_expired_cert(self, client, app, monkeypatch):
        """An expired validity window -> expired=True and trusted=False (anti-replay).

        We mint directly via the injected verifier with now pinned far in the past,
        then verify over HTTP — the signature is valid but the window is closed.
        """
        seed = generate_seed()
        pub = ed25519_pubkey_from_seed(seed)
        engine = app.state.engine
        res = engine.predict(np.asarray(_signal(), dtype=np.float32))
        probs = np.array([res.probabilities[n] for n in CLASS_NAMES], dtype=np.float64)
        cert = app.state.verifier.certify(
            probs, seed=seed, ttl_seconds=1, key_id="k",
            now_iso="2000-01-01T00:00:00.000+00:00")
        v = client.post("/verify", json={
            "certificate": cert.as_dict(), "expected_pubkey": pub}).json()
        assert v["expired"] is True
        assert v["trusted"] is False

    def test_verify_malformed_cert_returns_400(self, client):
        r = client.post("/verify", json={"certificate": {"not": "a cert"}})
        assert r.status_code == 400, r.text


# --------------------------------------------------------------------------- #
# Verifier.certify() ttl/key_id delegation (direct, no server)
# --------------------------------------------------------------------------- #

class TestCertifyTtlDelegation:
    def _verifier(self):
        rng = np.random.default_rng(0)
        n, k = 400, len(CLASS_NAMES)
        from aion_nexus.verify.conformal import softmax
        true = rng.integers(0, k, n)
        logits = rng.standard_normal((n, k))
        logits[np.arange(n), true] += 2.0
        probs = softmax(logits)
        v = Verifier(alpha=0.1, class_names=list(CLASS_NAMES))
        v.calibrate(probs, true)
        return v, probs[0]

    def test_ttl_and_key_id_are_set_and_signed(self):
        v, p = self._verifier()
        seed = generate_seed()
        cert = v.certify(p, seed=seed, ttl_seconds=3600, key_id="kid",
                         now_iso="2026-06-15T12:00:00.000+00:00")
        assert cert.not_before == "2026-06-15T12:00:00.000+00:00"
        assert cert.valid_until == "2026-06-15T13:00:00.000+00:00"
        assert cert.key_id == "kid"
        # The window/identity are tamper-evident: a valid in-window cert is trusted,
        # but editing valid_until breaks the signature.
        pub = ed25519_pubkey_from_seed(seed)
        ok = verify_certificate(cert, expected_pubkey=pub,
                                now_iso="2026-06-15T12:30:00.000+00:00")
        assert ok["trusted"] is True and ok["expired"] is False
        tampered = cert.as_dict()
        tampered["valid_until"] = "2099-01-01T00:00:00.000+00:00"
        bad = verify_certificate(tampered, expected_pubkey=pub,
                                 now_iso="2026-06-15T12:30:00.000+00:00")
        assert bad["authenticity"] == "FORGED"
        assert bad["trusted"] is False

    def test_content_hash_deterministic_despite_ttl(self):
        """The validity window must NOT enter content_hash: identical decisions
        keep an identical content_hash even with different windows (determinism)."""
        v, p = self._verifier()
        c1 = v.certify(p, ttl_seconds=3600, now_iso="2026-06-15T12:00:00.000+00:00")
        c2 = v.certify(p, ttl_seconds=99, now_iso="2020-01-01T00:00:00.000+00:00")
        assert c1.content_hash == c2.content_hash

    def test_no_ttl_is_timeless(self):
        """Backward-compat: no ttl -> no window fields, verify has no expired flag."""
        v, p = self._verifier()
        cert = v.certify(p, seed=generate_seed())
        assert cert.not_before is None and cert.valid_until is None
        audit = verify_certificate(cert, expected_pubkey=cert.pubkey)
        assert "expired" not in audit
