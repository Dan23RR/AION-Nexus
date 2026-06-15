"""Tests for Substrate Core — the model-agnostic verification layer (aion_nexus.verify).

Covers the contracts that make the certificates trustworthy AND honest:
  * conformal coverage on synthetic *exchangeable* data is >= 1 - alpha (tol)
  * content_hash is deterministic for identical inputs
  * no HMAC key -> authentication NONE, verify reports authenticity UNVERIFIED
  * with VERIFY_HMAC_KEY -> tampering ANY field (incl. the human-readable
    predicted_name) makes integrity_ok False; an untampered cert is VERIFIED
  * the certificate chain detects in-place tamper and a downgrade (HMAC->NONE)
  * verdict CERTIFIED / REVIEW / ABSTAIN is coherent with set size + confidence
"""
from __future__ import annotations

import numpy as np
import pytest

from aion_nexus.verify import (
    AUTH_ED25519,
    AUTH_HMAC,
    AUTH_NONE,
    EMPIRICAL,
    PROVEN,
    CertificateStore,
    ConformalCalibrator,
    Verifier,
    compose_certificates,
    ed25519_pubkey_from_seed,
    verify_certificate,
)
from aion_nexus.verify.certificate import (
    CERT_SCHEMA_VERSION,
    ENV_ED25519_PUBKEY,
    ENV_ED25519_SEED,
    ENV_HMAC_KEY,
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    Certificate,
    require_authenticated,
)
from aion_nexus.verify.conformal import softmax
from aion_nexus.verify.signing import generate_seed

CLASS_NAMES = ["normal", "early", "medium", "advanced"]


def _synthetic_exchangeable(n=8000, k=4, seed=0):
    """One i.i.d. pool -> split into calib/test. Exchangeable by construction."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k))
    logits[np.arange(n), true] += 2.0          # informative but noisy
    probs = softmax(logits)
    idx = rng.permutation(n)
    half = n // 2
    cal, test = idx[:half], idx[half:]
    return probs, true, cal, test


# ----------------------------------------------------------------------------- #
# 1. Conformal coverage on exchangeable data
# ----------------------------------------------------------------------------- #

@pytest.mark.parametrize("score", ["aps", "lac"])
def test_conformal_coverage_holds_under_exchangeability(score):
    alpha = 0.10
    probs, true, cal, test = _synthetic_exchangeable()
    cc = ConformalCalibrator(alpha=alpha, score=score)
    cc.fit(probs[cal], true[cal])
    res = cc.predict(probs[test])
    cov = ConformalCalibrator.empirical_coverage(res, true[test])
    # marginal coverage >= 1 - alpha (allow a small finite-sample slack)
    assert cov >= (1 - alpha) - 0.02, f"{score}: coverage {cov:.3f} < target {1 - alpha:.2f}"


def test_conformal_never_emits_empty_set():
    probs, true, cal, test = _synthetic_exchangeable()
    cc = ConformalCalibrator(alpha=0.5, score="lac")
    cc.fit(probs[cal], true[cal])
    res = cc.predict(probs[test])
    assert res.set_sizes.min() >= 1


def test_calibrator_exposes_exchangeability_caveat():
    cc = ConformalCalibrator()
    assert "exchangeab" in cc.coverage_valid_under.lower()
    assert "cross-bearing" in cc.coverage_valid_under.lower()


def test_calibrator_rejects_bad_alpha():
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=0.0)
    with pytest.raises(ValueError):
        ConformalCalibrator(alpha=1.0)


# ----------------------------------------------------------------------------- #
# 2. content_hash determinism
# ----------------------------------------------------------------------------- #

def _fit_verifier(monkeypatch, threshold=0.0, key=None):
    if key is None:
        monkeypatch.delenv(ENV_HMAC_KEY, raising=False)
    else:
        monkeypatch.setenv(ENV_HMAC_KEY, key)
    probs, true, cal, _ = _synthetic_exchangeable()
    v = Verifier(alpha=0.1, class_names=CLASS_NAMES, abstain_threshold=threshold)
    v.calibrate(probs[cal], true[cal])
    return v


def test_content_hash_is_deterministic(monkeypatch):
    v = _fit_verifier(monkeypatch)
    p = np.array([0.7, 0.1, 0.1, 0.1])
    c1 = v.certify(p, model_id="m")
    c2 = v.certify(p, model_id="m")
    # same decision -> identical content_hash (cert_id/timestamp differ)
    assert c1.content_hash == c2.content_hash
    assert c1.cert_id != c2.cert_id
    # binding the input also stays deterministic
    sig = np.arange(10, dtype="float32")
    assert v.certify(p, input_signal=sig).content_hash == \
        v.certify(p, input_signal=sig).content_hash


def test_content_hash_changes_with_decision(monkeypatch):
    v = _fit_verifier(monkeypatch)
    a = v.certify(np.array([0.7, 0.1, 0.1, 0.1]))
    b = v.certify(np.array([0.1, 0.7, 0.1, 0.1]))
    assert a.content_hash != b.content_hash


# ----------------------------------------------------------------------------- #
# 3. No key -> NONE / UNVERIFIED
# ----------------------------------------------------------------------------- #

def test_no_key_authentication_none_and_unverified(monkeypatch):
    v = _fit_verifier(monkeypatch, key=None)
    cert = v.certify(np.array([0.7, 0.1, 0.1, 0.1]))
    assert cert.authentication == AUTH_NONE
    assert cert.signature is None
    res = verify_certificate(cert)             # no key in env
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "UNVERIFIED"


# ----------------------------------------------------------------------------- #
# 4. With key -> tamper (incl. predicted_name) breaks integrity; clean = VERIFIED
# ----------------------------------------------------------------------------- #

def test_with_key_clean_cert_is_verified(monkeypatch):
    v = _fit_verifier(monkeypatch, key="s3cr3t-key")
    cert = v.certify(np.array([0.7, 0.1, 0.1, 0.1]))
    assert cert.authentication == AUTH_HMAC
    assert cert.signature is not None
    res = verify_certificate(cert)             # key from env
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "VERIFIED"


def test_tampered_predicted_name_breaks_integrity(monkeypatch):
    """Red-team lesson: editing ONLY the human-readable label must break the hash."""
    v = _fit_verifier(monkeypatch, key="s3cr3t-key")
    cert = v.certify(np.array([0.7, 0.1, 0.1, 0.1]))
    d = cert.as_dict()
    assert d["predicted_name"] == "normal"
    d["predicted_name"] = "advanced"           # forge the displayed class only
    res = verify_certificate(d)
    assert res["integrity_ok"] is False        # bound into content_hash


def test_tampered_verdict_detected_as_forged(monkeypatch):
    v = _fit_verifier(monkeypatch, key="s3cr3t-key")
    cert = v.certify(np.array([0.7, 0.1, 0.1, 0.1]))
    d = cert.as_dict()
    d["verdict"] = "CERTIFIED" if d["verdict"] != "CERTIFIED" else "ABSTAIN"
    res = verify_certificate(d)
    assert res["integrity_ok"] is False

    # forging content_hash too (recompute) without the key still can't sign
    d2 = cert.as_dict()
    d2["signature"] = "00" * 32
    res2 = verify_certificate(d2)
    assert res2["authenticity"] == "FORGED"


# ----------------------------------------------------------------------------- #
# 5. Certificate store chain
# ----------------------------------------------------------------------------- #

def test_chain_intact_keyed_is_verified(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HMAC_KEY, "chain-key")
    v = _fit_verifier(monkeypatch, key="chain-key")
    store = CertificateStore(path=tmp_path / "certs.jsonl")
    for p in ([0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]):
        store.append(v.certify(np.array(p)))
    integrity_ok, authenticity, broken_at = store.verify_chain()
    assert integrity_ok is True
    assert authenticity == "VERIFIED"
    assert broken_at is None


def test_chain_tamper_detected(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HMAC_KEY, "chain-key")
    v = _fit_verifier(monkeypatch, key="chain-key")
    path = tmp_path / "certs.jsonl"
    store = CertificateStore(path=path)
    for p in ([0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1]):
        store.append(v.certify(np.array(p)))

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = lines[1].replace('"normal"', '"advanced"') if '"normal"' in lines[1] \
        else lines[1].replace("REVIEW", "CERTIFIED")
    # ensure we actually changed a byte
    if tampered == lines[1]:
        tampered = lines[1][:-2] + ("0" if lines[1][-2] != "0" else "1") + lines[1][-1]
    lines[1] = tampered
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    store2 = CertificateStore(path=path)
    integrity_ok, authenticity, broken_at = store2.verify_chain()
    assert integrity_ok is False
    assert authenticity == "FORGED"
    assert broken_at == 1


def test_chain_downgrade_detected(monkeypatch, tmp_path):
    """A keyless attacker strips HMAC and re-links with plain SHA-256; a key-holder
    must detect the chain_auth=NONE downgrade as tamper."""
    path = tmp_path / "certs.jsonl"
    # 1) honest keyed store
    monkeypatch.setenv(ENV_HMAC_KEY, "chain-key")
    v = _fit_verifier(monkeypatch, key="chain-key")
    store = CertificateStore(path=path)
    store.append(v.certify(np.array([0.7, 0.1, 0.1, 0.1])))
    store.append(v.certify(np.array([0.1, 0.7, 0.1, 0.1])))

    # 2) attacker WITHOUT the key re-concatenates the chain as plain SHA-256
    monkeypatch.delenv(ENV_HMAC_KEY, raising=False)
    attacker = CertificateStore(path=tmp_path / "forged.jsonl")
    for rec in CertificateStore(path=path).iter_certs():
        clean = {k: x for k, x in rec.items()
                 if k not in ("prev_hash", "record_hash", "chain_auth")}
        attacker.append(clean)                 # keyless => chain_auth=NONE links
    # the forged store is internally consistent to a keyless auditor
    ig_keyless, auth_keyless, _ = attacker.verify_chain()
    assert ig_keyless is True and auth_keyless == "UNVERIFIED"

    # 3) the key-holder audits the forged store and detects the downgrade
    ig_keyed, auth_keyed, broken_at = attacker.verify_chain(key="chain-key")
    assert ig_keyed is False
    assert auth_keyed == "FORGED"
    assert broken_at == 0


def test_chain_resumes_across_restart(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_HMAC_KEY, "chain-key")
    v = _fit_verifier(monkeypatch, key="chain-key")
    path = tmp_path / "certs.jsonl"
    CertificateStore(path=path).append(v.certify(np.array([0.7, 0.1, 0.1, 0.1])))
    # new instance resumes the chain from disk
    CertificateStore(path=path).append(v.certify(np.array([0.1, 0.7, 0.1, 0.1])))
    integrity_ok, authenticity, broken_at = CertificateStore(path=path).verify_chain()
    assert integrity_ok and authenticity == "VERIFIED" and broken_at is None


# ----------------------------------------------------------------------------- #
# 6. Verdict coherence
# ----------------------------------------------------------------------------- #

def test_verdict_certified_on_confident_singleton(monkeypatch):
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))
    assert cert.verdict == "CERTIFIED"
    assert len(cert.conformal_set) == 1
    assert cert.conformal_set_names == ["normal"]


def test_verdict_review_on_ambiguous_set(monkeypatch):
    # tight alpha + near-uniform probs -> conformal set keeps >1 label
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.30, 0.28, 0.22, 0.20]))
    assert cert.verdict == "REVIEW"
    assert len(cert.conformal_set) > 1


def test_verdict_abstain_below_confidence_threshold(monkeypatch):
    v = _fit_verifier(monkeypatch, threshold=0.9)
    cert = v.certify(np.array([0.6, 0.2, 0.1, 0.1]))   # top prob 0.6 < 0.9
    assert cert.verdict == "ABSTAIN"


def test_verifier_requires_calibration_before_certify(monkeypatch):
    monkeypatch.delenv(ENV_HMAC_KEY, raising=False)
    v = Verifier(alpha=0.1, class_names=CLASS_NAMES)
    with pytest.raises(RuntimeError, match="calibrate"):
        v.certify(np.array([0.7, 0.1, 0.1, 0.1]))


def test_class_names_length_must_match_calibration(monkeypatch):
    monkeypatch.delenv(ENV_HMAC_KEY, raising=False)
    probs, true, cal, _ = _synthetic_exchangeable()
    v = Verifier(alpha=0.1, class_names=["a", "b"])     # 2 names, 4 classes
    with pytest.raises(ValueError, match="class_names"):
        v.calibrate(probs[cal], true[cal])


# ----------------------------------------------------------------------------- #
# 7. Assurance tier — conformal is EMPIRICAL, and the tier is HASHED
# ----------------------------------------------------------------------------- #

def _clean_env(monkeypatch):
    for k in (ENV_HMAC_KEY, ENV_ED25519_SEED, ENV_ED25519_PUBKEY):
        monkeypatch.delenv(k, raising=False)


def test_conformal_verdict_is_empirical(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))
    assert cert.assurance == EMPIRICAL          # never 'proven'
    assert "exchangeab" in v.assurance_caveat.lower()


def test_assurance_is_in_canonical_payload_and_hashed(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))
    assert "assurance" in cert.canonical_payload()
    # Two certs identical but for the tier must hash differently.
    c_emp = Certificate(
        predicted_label=0, predicted_name="normal", conformal_set=[0],
        conformal_set_names=["normal"], verdict=VERDICT_CERTIFIED, assurance=EMPIRICAL)
    c_pro = Certificate(
        predicted_label=0, predicted_name="normal", conformal_set=[0],
        conformal_set_names=["normal"], verdict=VERDICT_CERTIFIED, assurance=PROVEN)
    assert c_emp.compute_content_hash() != c_pro.compute_content_hash()


def test_assurance_overclaim_breaks_integrity(monkeypatch):
    """Upgrading the tier empirical->proven without re-sealing must break the hash."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    d = cert.as_dict()
    assert d["assurance"] == EMPIRICAL
    d["assurance"] = PROVEN                     # silent overclaim
    res = verify_certificate(d, expected_pubkey=cert.pubkey)
    assert res["integrity_ok"] is False
    assert res["trusted"] is False


# ----------------------------------------------------------------------------- #
# 8. Ed25519 — seal, verify against expected pubkey, self-signed, downgrade
# ----------------------------------------------------------------------------- #

def test_ed25519_seal_verified_and_trusted_against_expected_pubkey(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    assert cert.authentication == AUTH_ED25519
    assert cert.signature is not None
    assert cert.pubkey == ed25519_pubkey_from_seed("issuer-seed")
    res = verify_certificate(cert, expected_pubkey=cert.pubkey)
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "VERIFIED"
    assert res["trusted"] is True


def test_ed25519_without_expected_pubkey_is_self_signed_not_trusted(monkeypatch):
    """The embedded pubkey proves self-consistency only — issuer NOT verified."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    res = verify_certificate(cert)              # no expected pubkey
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "SELF-SIGNED"
    assert res["trusted"] is False              # CRITICAL: not trusted
    assert "issuer NOT verified" in res["detail"]


def test_ed25519_wrong_expected_pubkey_is_forged(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    attacker_pub = ed25519_pubkey_from_seed("attacker-seed")
    res = verify_certificate(cert, expected_pubkey=attacker_pub)
    assert res["authenticity"] == "FORGED"
    assert res["trusted"] is False


def test_ed25519_expected_pubkey_from_env(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    monkeypatch.setenv(ENV_ED25519_PUBKEY, cert.pubkey)
    res = verify_certificate(cert)              # expected key from env
    assert res["authenticity"] == "VERIFIED"
    assert res["trusted"] is True


def test_ed25519_downgrade_strip_sig_not_trusted(monkeypatch):
    """Strip signature + declare NONE -> verifier with expected key must not trust."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    expected_pub = cert.pubkey
    d = cert.as_dict()
    d["signature"] = None
    d["authentication"] = AUTH_NONE
    res = verify_certificate(d, expected_pubkey=expected_pub)
    assert res["trusted"] is False


def test_ed25519_tampered_label_breaks_integrity_not_trusted(monkeypatch):
    """Editing the label leaves the signature valid over the STALE content_hash,
    but the recomputed payload no longer matches it -> integrity_ok False ->
    NOT trusted. trusted (the conjunction) is the single safe flag."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    d = cert.as_dict()
    d["predicted_name"] = "advanced"            # tamper: payload no longer hashes to content_hash
    res = verify_certificate(d, expected_pubkey=cert.pubkey)
    assert res["integrity_ok"] is False         # recomputed hash != stored content_hash
    assert res["trusted"] is False              # CRITICAL: never trusted on a label tamper


def test_ed25519_tampered_content_hash_is_forged(monkeypatch):
    """If the attacker also recomputes content_hash to match the forged payload,
    the signature (which they cannot remake without the seed) no longer matches
    the new hash -> authenticity FORGED."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    d = cert.as_dict()
    d["predicted_name"] = "advanced"
    forged = Certificate(**{k: d[k] for k in (
        "predicted_label", "predicted_name", "conformal_set", "conformal_set_names",
        "verdict", "alpha", "qhat", "input_sha256", "model_id", "assurance")})
    d["content_hash"] = forged.compute_content_hash()   # rehash to pass integrity
    res = verify_certificate(d, expected_pubkey=cert.pubkey)
    assert res["integrity_ok"] is True          # hash now matches the forged payload
    assert res["authenticity"] == "FORGED"      # but the signature does not match the new hash
    assert res["trusted"] is False


# ----------------------------------------------------------------------------- #
# 9. seal() scheme precedence
# ----------------------------------------------------------------------------- #

def test_seal_precedence_explicit_seed_is_ed25519(monkeypatch):
    _clean_env(monkeypatch)
    # explicit arg (seed) wins even when an HMAC env key is also set
    monkeypatch.setenv(ENV_HMAC_KEY, "hmac-key")
    v = _fit_verifier(monkeypatch, key="hmac-key")
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed="issuer-seed")
    assert cert.authentication == AUTH_ED25519


def test_seal_precedence_env_ed25519_over_hmac(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv(ENV_HMAC_KEY, "hmac-key")
    monkeypatch.setenv(ENV_ED25519_SEED, "issuer-seed")
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))   # auto resolution
    assert cert.authentication == AUTH_ED25519
    assert cert.pubkey == ed25519_pubkey_from_seed("issuer-seed")


def test_seal_precedence_hmac_when_only_hmac(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv(ENV_HMAC_KEY, "hmac-key")
    v = _fit_verifier(monkeypatch, key="hmac-key")
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))
    assert cert.authentication == AUTH_HMAC
    assert cert.pubkey is None


def test_seal_scheme_none_forces_unsigned(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv(ENV_ED25519_SEED, "issuer-seed")
    v = _fit_verifier(monkeypatch)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), scheme="none")
    assert cert.authentication == AUTH_NONE
    assert cert.signature is None and cert.pubkey is None


# ----------------------------------------------------------------------------- #
# 10. compose_certificates — weakest link governs verdict AND assurance
# ----------------------------------------------------------------------------- #

def _cert(verdict, assurance=EMPIRICAL):
    return Certificate(
        predicted_label=0, predicted_name="normal", conformal_set=[0],
        conformal_set_names=["normal"], verdict=verdict, assurance=assurance)


def test_compose_and_propagates_review(monkeypatch):
    out = compose_certificates([_cert(VERDICT_CERTIFIED), _cert(VERDICT_REVIEW)], op="and")
    assert out["verdict"] == VERDICT_REVIEW     # weakest verdict propagates
    assert out["assurance"] == EMPIRICAL


def test_compose_and_propagates_abstain(monkeypatch):
    out = compose_certificates(
        [_cert(VERDICT_CERTIFIED), _cert(VERDICT_REVIEW), _cert(VERDICT_ABSTAIN)], op="and")
    assert out["verdict"] == VERDICT_ABSTAIN    # a single ABSTAIN drags it down


def test_compose_and_all_certified(monkeypatch):
    out = compose_certificates([_cert(VERDICT_CERTIFIED), _cert(VERDICT_CERTIFIED)], op="and")
    assert out["verdict"] == VERDICT_CERTIFIED


def test_compose_assurance_is_weakest_link(monkeypatch):
    out = compose_certificates(
        [_cert(VERDICT_CERTIFIED, PROVEN), _cert(VERDICT_CERTIFIED, EMPIRICAL)], op="and")
    # system assurance can never exceed the weakest evidence
    assert out["assurance"] == EMPIRICAL
    assert "weakest" in out["assurance_caveat"].lower()


def test_compose_or_takes_strongest_verdict(monkeypatch):
    out = compose_certificates([_cert(VERDICT_ABSTAIN), _cert(VERDICT_CERTIFIED)], op="or")
    assert out["verdict"] == VERDICT_CERTIFIED
    # but the assurance is STILL the weakest link (anti-overclaim invariant)
    assert out["assurance"] == EMPIRICAL


def test_compose_empty_is_abstain_none(monkeypatch):
    out = compose_certificates([], op="and")
    assert out["verdict"] == VERDICT_ABSTAIN
    assert out["n"] == 0


# ----------------------------------------------------------------------------- #
# 11. Certificate store chain with Ed25519
# ----------------------------------------------------------------------------- #

def test_chain_ed25519_verified_against_expected_pubkey(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    store = CertificateStore(path=tmp_path / "certs.jsonl", seed="chain-seed")
    for p in ([0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]):
        store.append(v.certify(np.array(p), seed="issuer-seed"))
    pub = ed25519_pubkey_from_seed("chain-seed")
    integrity_ok, authenticity, broken_at = CertificateStore(
        path=tmp_path / "certs.jsonl").verify_chain(pubkey=pub)
    assert integrity_ok is True
    assert authenticity == "VERIFIED"
    assert broken_at is None


def test_chain_ed25519_self_signed_when_no_expected_pubkey(monkeypatch, tmp_path):
    """Without an expected pubkey, the Ed25519 chain is integrity-only (UNVERIFIED)."""
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    store = CertificateStore(path=tmp_path / "certs.jsonl", seed="chain-seed")
    store.append(v.certify(np.array([0.7, 0.1, 0.1, 0.1]), seed="issuer-seed"))
    integrity_ok, authenticity, _ = CertificateStore(
        path=tmp_path / "certs.jsonl").verify_chain()       # no key/seed/pubkey
    assert integrity_ok is True
    assert authenticity == "UNVERIFIED"


def test_chain_ed25519_tamper_detected(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch)
    path = tmp_path / "certs.jsonl"
    store = CertificateStore(path=path, seed="chain-seed")
    for p in ([0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]):
        store.append(v.certify(np.array(p), seed="issuer-seed"))
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"normal"', '"advanced"', 1)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pub = ed25519_pubkey_from_seed("chain-seed")
    integrity_ok, authenticity, broken_at = CertificateStore(
        path=path).verify_chain(pubkey=pub)
    assert integrity_ok is False
    assert authenticity == "FORGED"
    assert broken_at == 0


# ----------------------------------------------------------------------------- #
# 12. Validity window (anti-replay) — expiry / not_before / jti / key_id
#     bound by the SIGNATURE, NOT by content_hash (determinism preserved).
# ----------------------------------------------------------------------------- #
#
# A fixed clock so every temporal assertion is deterministic (workspace rule:
# never read datetime.now in a test). T0 is "now" at seal time; T_BEFORE precedes
# not_before, T_AFTER is past valid_until for a short TTL.
T0 = "2026-06-15T12:00:00.000+00:00"
T_BEFORE = "2026-06-15T11:59:00.000+00:00"   # 1 min before T0
T_WITHIN = "2026-06-15T12:00:30.000+00:00"   # 30 s after T0 (TTL=60 -> inside)
T_AFTER = "2026-06-15T12:02:00.000+00:00"    # 2 min after T0 (TTL=60 -> expired)


def _ed_cert(monkeypatch):
    """A minimal Ed25519-signed cert helper (full-entropy seed, strict-clean)."""
    _clean_env(monkeypatch)
    seed = generate_seed()
    cert = Certificate(
        predicted_label=0, predicted_name="normal", conformal_set=[0],
        conformal_set_names=["normal"], verdict=VERDICT_CERTIFIED, alpha=0.1,
        qhat=0.5)
    return cert, seed


def test_schema_version_is_1_1():
    assert CERT_SCHEMA_VERSION == "1.1"


def test_ttl_sets_window_jti_and_signs_payload(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, key_id="kid-1", now_iso=T0)
    # window + identity fields are populated
    assert cert.not_before == T0
    assert cert.valid_until == "2026-06-15T12:01:00.000+00:00"
    assert cert.jti is not None and len(cert.jti) == 32   # uuid4 hex
    assert cert.key_id == "kid-1"
    # the signing payload binds all of them
    sp = cert.signing_payload()
    assert sp == f"{cert.content_hash}|{cert.not_before}|{cert.valid_until}|" \
                 f"{cert.jti}|{cert.key_id}"


def test_within_window_is_trusted_against_expected_pubkey(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, now_iso=T0)
    res = verify_certificate(cert, expected_pubkey=cert.pubkey, now_iso=T_WITHIN)
    assert res["integrity_ok"] is True
    assert res["authenticity"] == "VERIFIED"
    assert res["expired"] is False
    assert res["not_yet_valid"] is False
    assert res["trusted"] is True


def test_expired_cert_is_not_trusted(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, now_iso=T0)
    res = verify_certificate(cert, expected_pubkey=cert.pubkey, now_iso=T_AFTER)
    # signature still valid, but the cert is outside its window
    assert res["authenticity"] == "VERIFIED"
    assert res["expired"] is True
    assert res["trusted"] is False              # CRITICAL: expiry forces no-trust
    assert "expired" in res["detail"]


def test_not_yet_valid_cert_is_not_trusted(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, now_iso=T0)
    res = verify_certificate(cert, expected_pubkey=cert.pubkey, now_iso=T_BEFORE)
    assert res["not_yet_valid"] is True
    assert res["trusted"] is False
    assert "not yet valid" in res["detail"]


def test_tampering_valid_until_breaks_the_signature(monkeypatch):
    """Pushing valid_until into the future is TAMPER-EVIDENT: it changes the
    signing payload, so the signature no longer verifies -> FORGED, not trusted."""
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, now_iso=T0)
    d = cert.as_dict()
    d["valid_until"] = "2099-01-01T00:00:00.000+00:00"   # attacker extends validity
    # even checking "now" inside the forged window, the signature fails
    res = verify_certificate(d, expected_pubkey=cert.pubkey, now_iso=T_WITHIN)
    assert res["authenticity"] == "FORGED"      # signature covers valid_until
    assert res["trusted"] is False


def test_tampering_jti_or_key_id_breaks_the_signature(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, key_id="kid-1", now_iso=T0)
    for field_name, forged in (("jti", "deadbeef" * 4), ("key_id", "kid-evil")):
        d = cert.as_dict()
        d[field_name] = forged
        res = verify_certificate(d, expected_pubkey=cert.pubkey, now_iso=T_WITHIN)
        assert res["authenticity"] == "FORGED", field_name
        assert res["trusted"] is False, field_name


def test_content_hash_determinism_preserved_with_ttl(monkeypatch):
    """The headline invariant: temporal fields are NOT in content_hash, so two
    certs over the SAME decision share a content_hash even with different TTL
    windows / jti / key_id. Determinism + reproducibility survive anti-replay."""
    _clean_env(monkeypatch)
    seed = generate_seed()

    def _mk():
        return Certificate(
            predicted_label=0, predicted_name="normal", conformal_set=[0],
            conformal_set_names=["normal"], verdict=VERDICT_CERTIFIED,
            alpha=0.1, qhat=0.5)

    a = _mk().seal(seed, scheme="ed25519", ttl_seconds=60, key_id="k1", now_iso=T0)
    b = _mk().seal(seed, scheme="ed25519", ttl_seconds=3600, key_id="k2",
                   now_iso="2026-06-15T13:00:00.000+00:00")
    # identical decision -> identical content_hash despite different windows/jti
    assert a.content_hash == b.content_hash
    # but the signing payloads (and signatures) differ, because the windows differ
    assert a.signing_payload() != b.signing_payload()
    assert a.signature != b.signature
    assert a.jti != b.jti
    # and a cert with NO ttl over the same decision still matches the hash
    c = _mk().seal(seed, scheme="ed25519")
    assert c.content_hash == a.content_hash


def test_hmac_ttl_window_also_enforced(monkeypatch):
    """The validity window works on the HMAC path too (signature covers it)."""
    _clean_env(monkeypatch)
    monkeypatch.setenv(ENV_HMAC_KEY, "hmac-key")
    cert = Certificate(
        predicted_label=0, predicted_name="normal", conformal_set=[0],
        conformal_set_names=["normal"], verdict=VERDICT_CERTIFIED, alpha=0.1,
        qhat=0.5).seal(scheme="hmac", ttl_seconds=60, now_iso=T0)
    assert cert.authentication == AUTH_HMAC
    ok = verify_certificate(cert, now_iso=T_WITHIN)
    assert ok["authenticity"] == "VERIFIED" and ok["trusted"] is True
    gone = verify_certificate(cert, now_iso=T_AFTER)
    assert gone["expired"] is True and gone["trusted"] is False
    # tampering the window breaks the HMAC
    d = cert.as_dict()
    d["valid_until"] = "2099-01-01T00:00:00.000+00:00"
    forged = verify_certificate(d, now_iso=T_WITHIN)
    assert forged["authenticity"] == "FORGED" and forged["trusted"] is False


def test_timeless_cert_has_no_window_keys(monkeypatch):
    """A cert sealed WITHOUT ttl carries no window and verify omits the flags
    (back-compat: existing certs behave exactly as before)."""
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519")
    assert cert.not_before is None and cert.valid_until is None and cert.jti is None
    res = verify_certificate(cert, expected_pubkey=cert.pubkey)
    assert "expired" not in res and "not_yet_valid" not in res
    assert res["trusted"] is True


def test_malformed_window_fails_closed(monkeypatch):
    """An unparseable valid_until is treated as expired (fail-closed), never
    silently trusted."""
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519", ttl_seconds=60, now_iso=T0)
    d = cert.as_dict()
    d["valid_until"] = "not-a-timestamp"
    res = verify_certificate(d, expected_pubkey=cert.pubkey, now_iso=T_WITHIN)
    assert res["expired"] is True
    assert res["trusted"] is False


# ----------------------------------------------------------------------------- #
# 13. require_authenticated — safe-by-default strict gate for serving
# ----------------------------------------------------------------------------- #

def test_require_authenticated_raises_on_unsigned_certified(monkeypatch):
    _clean_env(monkeypatch)
    v = _fit_verifier(monkeypatch, key=None)
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]))   # CERTIFIED, no key
    assert cert.verdict == VERDICT_CERTIFIED
    assert cert.authentication == AUTH_NONE
    with pytest.raises(ValueError, match="unsigned CERTIFIED"):
        require_authenticated(cert)


def test_require_authenticated_passes_signed_certified(monkeypatch):
    cert, seed = _ed_cert(monkeypatch)
    cert.seal(seed, scheme="ed25519")
    assert require_authenticated(cert) is None         # signed -> ok


def test_require_authenticated_ignores_non_certified_verdicts(monkeypatch):
    """REVIEW / ABSTAIN do not authorise autonomous action, so an unsigned one is
    not the dangerous case the gate guards against."""
    for verdict in (VERDICT_REVIEW, VERDICT_ABSTAIN):
        cert = Certificate(
            predicted_label=0, predicted_name="normal", conformal_set=[0, 1],
            conformal_set_names=["normal", "early"], verdict=verdict).seal(scheme="none")
        assert cert.authentication == AUTH_NONE
        assert require_authenticated(cert) is None     # not gated
