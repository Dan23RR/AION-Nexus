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
    AUTH_HMAC,
    AUTH_NONE,
    CertificateStore,
    ConformalCalibrator,
    Verifier,
    verify_certificate,
)
from aion_nexus.verify.certificate import ENV_HMAC_KEY
from aion_nexus.verify.conformal import softmax

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
