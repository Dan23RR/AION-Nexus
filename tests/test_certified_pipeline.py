"""End-to-end certified-pipeline tests (v2.16.0) — the verification layer SERVED.

These pin the crack-#1 fix: the conformal certificate, the physics second opinion
and the compliance evidence are reachable as a SERVED product, and the synthetic-
vs-real calibration basis is a first-class, certificate-bound, honestly-surfaced
fact (workspace 6.31):

  * /predict_certified stamps coverage_basis into the cert (tamper-evident) and
    warns when it is the synthetic placeholder.
  * a real, leakage-checked calibration artifact flips coverage_basis to
    "real-holdout"; the leakage gate REFUSES a leaked real artifact.
  * physics CONTRADICT composes weakest-link so a confident-but-wrong model does
    NOT stay CERTIFIED; with no rpm/geometry the path is inert (backward-compatible).
  * /evidence and /annex_iv serve the EU AI Act mapping without forbidden claims.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aion_nexus.config import CLASS_NAMES
from aion_nexus.serving_calibration import (
    BASIS_REAL,
    BASIS_SYNTHETIC,
    load_calibration,
    save_calibration,
)
from aion_nexus.verify import CertificateStore, Verifier, generate_seed

FS = 25_600
SKF_6205 = {"n_rolling_elements": 9, "ball_diameter": 7.94, "pitch_diameter": 39.04}


def _signal() -> list:
    return (np.random.default_rng(0).standard_normal((2, 2560)) * 0.4).tolist()


def _inner_fault_signal(seed: int = 5) -> list:
    """An INNER-race bearing fault (so a claimed 'outer' must be CONTRADICTed)."""
    from aion_nexus.physics import FAULT_INNER, BearingGeometry
    geom = BearingGeometry(9, 7.94, 39.04)
    order = geom.fault_orders()[FAULT_INNER]
    rng = np.random.default_rng(seed)
    n, fr = FS, 1800.0 / 60.0
    t = np.arange(n) / FS
    sig = 0.05 * rng.standard_normal(n)
    period = 1.0 / (order * fr)
    for t0 in np.arange(0.0, 1.0, period):
        tt = t - t0
        idx = tt >= 0
        sig[idx] += np.exp(-800.0 * tt[idx]) * np.sin(2 * np.pi * 3000.0 * tt[idx])
    return np.vstack([sig, sig]).tolist()


def _calibrated_verifier(engine) -> Verifier:
    rng = np.random.default_rng(123)
    probs, labels = [], []
    for cls in range(len(CLASS_NAMES)):
        for _ in range(8):
            sig = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
            res = engine.predict(sig)
            probs.append(np.array([res.probabilities[n] for n in CLASS_NAMES], dtype=np.float64))
            labels.append(cls)
    return Verifier(alpha=0.1, class_names=list(CLASS_NAMES)).calibrate(
        np.vstack(probs), np.array(labels, dtype=int))


@pytest.fixture
def app():
    from server.main import app as _app
    return _app


@pytest.fixture(autouse=True)
def fresh_state(app, tmp_path, monkeypatch):
    """Inject a fresh engine + placeholder-calibrated verifier + temp store.

    Also sets a strong Ed25519 seed so certs are signed (the basis warning is then
    isolated from the unsigned warning) and points the calibration artifact env at
    an absent path so the default basis is deterministically the placeholder.
    """
    from aion_nexus import InferenceEngine
    from aion_nexus.model import create_aion_nexus

    torch.manual_seed(0)
    monkeypatch.setenv("VERIFY_ED25519_SEED", generate_seed())
    monkeypatch.setenv("AION_CALIBRATION_NPZ", str(tmp_path / "absent.npz"))
    monkeypatch.delenv("AION_REQUIRE_REAL_CALIBRATION", raising=False)
    monkeypatch.delenv("AION_RUL_ARTIFACT", raising=False)
    monkeypatch.delenv("AION_KEYRING", raising=False)
    monkeypatch.delenv("AION_CERT_KEY_ID", raising=False)
    app.state.rul_model = None
    app.state.keyring = None
    app.state.monitor = None
    saved = {k: getattr(app.state, k, None) for k in (
        "engine", "startup_error", "verifier", "cert_store",
        "expected_checkpoint_sha256", "coverage_basis", "calibration_meta")}
    engine = InferenceEngine(create_aion_nexus())
    app.state.engine = engine
    app.state.startup_error = None
    app.state.verifier = _calibrated_verifier(engine)
    app.state.cert_store = CertificateStore(path=tmp_path / "certs.jsonl")
    app.state.expected_checkpoint_sha256 = None
    app.state.coverage_basis = BASIS_SYNTHETIC
    app.state.calibration_meta = None
    yield
    for k, v in saved.items():
        setattr(app.state, k, v)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    return TestClient(app)


# --------------------------------------------------------------------------- #
# serving_calibration: the artifact format + the leakage gate
# --------------------------------------------------------------------------- #

def test_calibration_roundtrip(tmp_path):
    probs = np.full((12, 4), 0.25)
    labels = np.array([0, 1, 2, 3] * 3)
    path = tmp_path / "cal.npz"
    meta = save_calibration(path, probs, labels, CLASS_NAMES, basis="synthetic-demo")
    loaded = load_calibration(path)
    assert loaded["basis"] == "synthetic-demo"
    assert loaded["probs"].shape == (12, 4)
    assert list(loaded["class_names"]) == list(CLASS_NAMES)
    assert meta["n"] == 12 and meta["n_classes"] == 4


def test_leakage_gate_refuses_a_leaked_real_artifact(tmp_path):
    probs = np.full((8, 4), 0.25)
    labels = np.array([0, 1, 2, 3] * 2)
    with pytest.raises(ValueError, match="leak"):
        save_calibration(tmp_path / "leak.npz", probs, labels, CLASS_NAMES,
                         basis=BASIS_REAL,
                         train_groups=["B1", "B2"], calib_groups=["B1"] * 8)


def test_real_artifact_requires_groups(tmp_path):
    probs = np.full((8, 4), 0.25)
    labels = np.array([0, 1, 2, 3] * 2)
    with pytest.raises(ValueError, match="REQUIRES"):
        save_calibration(tmp_path / "x.npz", probs, labels, CLASS_NAMES, basis=BASIS_REAL)


def test_real_artifact_accepts_a_disjoint_split(tmp_path):
    probs = np.full((8, 4), 0.25)
    labels = np.array([0, 1, 2, 3] * 2)
    meta = save_calibration(tmp_path / "ok.npz", probs, labels, CLASS_NAMES,
                            basis=BASIS_REAL,
                            train_groups=["B1", "B2"], calib_groups=["B9"] * 8)
    assert meta["disjoint"] is True and meta["leakage_checked"] is True


# --------------------------------------------------------------------------- #
# coverage_basis: surfaced on the response AND bound into the certificate
# --------------------------------------------------------------------------- #

def test_placeholder_basis_is_surfaced_and_bound(client):
    body = client.post("/predict_certified", json={"signal": _signal()}).json()
    assert body["coverage_basis"] == BASIS_SYNTHETIC
    assert "synthetic-placeholder" in (body["warning"] or "")
    # bound (tamper-evidently) into the signed certificate's coverage_guarantee
    assert "calibration_basis=synthetic-placeholder" in body["certificate"]["coverage_guarantee"]


def test_real_artifact_flips_basis_to_real_holdout(app, client, tmp_path, monkeypatch):
    # Build a real, leakage-checked artifact and wire it in via _build_certifier.
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(4), size=24)
    labels = np.array([0, 1, 2, 3] * 6)
    art = tmp_path / "real_cal.npz"
    save_calibration(art, probs, labels, CLASS_NAMES, basis=BASIS_REAL,
                     train_groups=["TR1", "TR2"], calib_groups=["CAL"] * 24)
    monkeypatch.setenv("AION_CALIBRATION_NPZ", str(art))
    from server.main import _build_certifier
    _build_certifier(app)
    # _build_certifier creates a default-path store; redirect to tmp so the test
    # does not write a stray certificates.jsonl into the repo.
    app.state.cert_store = CertificateStore(path=tmp_path / "certs2.jsonl")
    assert app.state.coverage_basis == BASIS_REAL
    body = client.post("/predict_certified", json={"signal": _signal()}).json()
    assert body["coverage_basis"] == BASIS_REAL
    assert "calibration_basis=real-holdout" in body["certificate"]["coverage_guarantee"]
    assert "synthetic-placeholder" not in (body["warning"] or "")


def test_strict_real_calibration_refuses_placeholder(client, monkeypatch):
    monkeypatch.setenv("AION_REQUIRE_REAL_CALIBRATION", "1")
    r = client.post("/predict_certified", json={"signal": _signal()})
    assert r.status_code == 503
    assert "synthetic placeholder" in r.json()["detail"]


# --------------------------------------------------------------------------- #
# physics second opinion composed into the served certificate
# --------------------------------------------------------------------------- #

def test_physics_contradiction_drops_the_system_verdict(client):
    r = client.post("/predict_certified", json={
        "signal": _inner_fault_signal(),
        "rpm": 1800.0,
        "bearing": SKF_6205,
        "claimed_fault": "outer",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["physics"] is not None
    assert body["physics"]["verdict"] == "CONTRADICT"
    # weakest-link: a contradicted claim must NOT remain CERTIFIED
    assert body["composed"] is not None
    assert body["composed"]["verdict"] != "CERTIFIED"


def test_no_physics_fields_is_inert_and_backward_compatible(client):
    body = client.post("/predict_certified", json={"signal": _signal()}).json()
    assert body["physics"] is None
    assert body["composed"] is None
    assert body["coverage_basis"] == BASIS_SYNTHETIC


def test_physics_confirm_does_not_invent_a_contradiction(client):
    # A matching claim ('inner' on an inner-race fault) must not CONTRADICT.
    body = client.post("/predict_certified", json={
        "signal": _inner_fault_signal(),
        "rpm": 1800.0,
        "bearing": SKF_6205,
        "claimed_fault": "inner",
    }).json()
    assert body["physics"]["verdict"] != "CONTRADICT"


# --------------------------------------------------------------------------- #
# served compliance surface
# --------------------------------------------------------------------------- #

def test_evidence_endpoint_maps_a_certificate(client):
    cert = client.post("/predict_certified", json={"signal": _signal()}).json()["certificate"]
    r = client.post("/evidence", json={"certificate": cert})
    assert r.status_code == 200
    out = r.json()
    assert len(out["evidence"]) >= 5
    blob = json.dumps(out).lower()
    for forbidden in ("compliant", "conforme", "conformità"):
        assert forbidden not in blob


def test_annex_iv_endpoint_returns_nine_sections(client):
    r = client.post("/annex_iv", json={
        "model_metadata": {"architecture": "MultiScaleCNN+BiGRU"}, "markdown": True})
    assert r.status_code == 200
    out = r.json()
    assert len(out["dossier"]["sections"]) == 9
    assert "card_markdown" in out
    assert "compliant" not in out["card_markdown"].lower()


def test_evidence_endpoint_rejects_garbage(client):
    r = client.post("/evidence", json={"certificate": {"not": "a cert"}})
    assert r.status_code in (200, 400)  # tolerant map or clean 400, never 500


# --------------------------------------------------------------------------- #
# calibrated RUL served (v2.19.0)
# --------------------------------------------------------------------------- #

def test_verify_enforces_revocation_via_keyring(app, client, tmp_path, monkeypatch):
    from aion_nexus.verify import KeyRing, ed25519_pubkey_from_seed, generate_seed
    seed = generate_seed()
    monkeypatch.setenv("VERIFY_ED25519_SEED", seed)
    monkeypatch.setenv("AION_CERT_KEY_ID", "k1")
    cert = client.post("/predict_certified", json={"signal": _signal()}).json()["certificate"]
    assert cert["key_id"] == "k1"
    ring = KeyRing().rotate("k1", ed25519_pubkey_from_seed(seed))
    art = tmp_path / "keyring.json"
    ring.save(art)
    monkeypatch.setenv("AION_KEYRING", str(art))
    from server.main import _load_keyring
    _load_keyring(app)
    r1 = client.post("/verify", json={"certificate": cert}).json()
    assert r1["trusted"] and r1["key_status"] == "active"
    # revoke -> reload registry -> the same cert is no longer trusted
    ring.revoke("k1", "seed exposed incident-217")
    ring.save(art)
    _load_keyring(app)
    r2 = client.post("/verify", json={"certificate": cert}).json()
    assert not r2["trusted"] and r2["authenticity"] == "REVOKED-KEY"


def test_monitor_endpoint_tracks_the_certified_stream(app, client):
    from aion_nexus.monitoring import Monitor
    app.state.monitor = Monitor(window=100)
    for _ in range(3):
        assert client.post("/predict_certified", json={"signal": _signal()}).status_code == 200
    s = client.get("/monitor").json()
    assert s["n"] == 3
    assert 0.0 <= s["certified_rate"] <= 1.0 and "drift_level" in s


def test_predict_rul_503_when_not_configured(client):
    r = client.post("/predict_rul", json={"signal": _signal()})
    assert r.status_code == 503
    assert "RUL" in r.json()["detail"]


def test_predict_rul_returns_conformal_interval_when_configured(app, client, tmp_path, monkeypatch):
    from aion_nexus.rul import ConformalRUL, health_features_batch
    rng = np.random.default_rng(0)
    sigs = [rng.standard_normal((2, 2560)).astype(np.float32) for _ in range(200)]
    feats = health_features_batch(sigs)
    y = rng.uniform(0, 3600, 200)
    model = ConformalRUL(alpha=0.1).fit(feats[:120], y[:120]).calibrate(feats[120:], y[120:])
    art = tmp_path / "rul.joblib"
    model.save(art)
    monkeypatch.setenv("AION_RUL_ARTIFACT", str(art))
    from server.main import _load_rul_model
    _load_rul_model(app)
    assert app.state.rul_model is not None
    body = client.post("/predict_rul", json={"signal": _signal()}).json()
    assert body["lower"] >= 0.0
    assert body["lower"] <= body["point"] <= body["upper"]
    assert body["unit"] == "seconds" and body["method"] == "CQR"
    assert "exchangeability" in body["coverage_caveat"].lower()


# --------------------------------------------------------------------------- #
# temperature scaling (v2.17.0) — calibrate the over-confident model
# --------------------------------------------------------------------------- #

def test_fit_temperature_softens_an_overconfident_model():
    from aion_nexus.serving_calibration import apply_temperature, fit_temperature
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 4, n)
    # Over-confident: the model is right ~70% of the time but always near-one-hot.
    probs = np.full((n, 4), 0.02)
    for i in range(n):
        cls = y[i] if rng.random() < 0.7 else rng.integers(0, 4)
        probs[i] = 0.02
        probs[i, cls] = 0.94
    probs /= probs.sum(1, keepdims=True)
    temp = fit_temperature(probs, y)
    assert temp > 1.0, "an over-confident model should be softened (T > 1)"

    def _ece(p):
        conf, pred = p.max(1), p.argmax(1)
        acc = (pred == y).astype(float)
        e, bins = 0.0, np.linspace(0, 1, 11)
        for b in range(10):
            m = (conf > bins[b]) & (conf <= bins[b + 1])
            if m.sum():
                e += m.mean() * abs(acc[m].mean() - conf[m].mean())
        return e
    assert _ece(apply_temperature(probs, temp)) < _ece(probs)


def test_served_certificate_carries_risk_control(app, client, tmp_path, monkeypatch):
    rng = np.random.default_rng(4)
    probs = rng.dirichlet(np.ones(4), size=200)
    labels = np.array([0, 1, 2, 3] * 50)
    art = tmp_path / "rc_cal.npz"
    save_calibration(art, probs, labels, CLASS_NAMES, basis=BASIS_REAL,
                     train_groups=["TR"], calib_groups=["CAL"] * 200)
    monkeypatch.setenv("AION_CALIBRATION_NPZ", str(art))
    monkeypatch.setenv("AION_RISK_ALPHA", "0.05")
    from server.main import _build_certifier
    _build_certifier(app)
    app.state.cert_store = CertificateStore(path=tmp_path / "c.jsonl")
    assert app.state.risk_control is not None
    rc = client.post("/predict_certified", json={"signal": _signal()}).json()["risk_control"]
    assert rc is not None
    assert rc["method"] == "CRC" and rc["alpha"] == 0.05
    assert len(rc["set"]) >= 1 and isinstance(rc["flags_degraded"], bool)
    assert "<=" in rc["guarantee"] or "bounded" in rc["guarantee"].lower()


def test_risk_control_can_be_disabled(app, client, tmp_path, monkeypatch):
    monkeypatch.setenv("AION_RISK_ALPHA", "off")
    rng = np.random.default_rng(5)
    probs = rng.dirichlet(np.ones(4), size=40)
    labels = np.array([0, 1, 2, 3] * 10)
    art = tmp_path / "rc_off.npz"
    save_calibration(art, probs, labels, CLASS_NAMES, basis=BASIS_REAL,
                     train_groups=["TR"], calib_groups=["CAL"] * 40)
    monkeypatch.setenv("AION_CALIBRATION_NPZ", str(art))
    from server.main import _build_certifier
    _build_certifier(app)
    app.state.cert_store = CertificateStore(path=tmp_path / "c.jsonl")
    assert app.state.risk_control is None
    body = client.post("/predict_certified", json={"signal": _signal()}).json()
    assert body["risk_control"] is None


def test_apply_temperature_identity_and_renormalises():
    from aion_nexus.serving_calibration import apply_temperature
    p = np.array([[0.7, 0.2, 0.1]])
    assert np.allclose(apply_temperature(p, 1.0), p)
    out = apply_temperature(p, 2.5)
    assert np.isclose(out.sum(), 1.0)
    assert out.argmax() == p.argmax()  # temperature does not change the decision


def test_served_certificate_records_temperature(app, client, tmp_path, monkeypatch):
    # Wire a real artifact so _build_certifier fits + applies temperature, then the
    # served certificate's coverage_guarantee reflects the basis (temperature note
    # appears only when T != 1.0, which is data-dependent — assert it never lies).
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(4), size=40)
    labels = np.array([0, 1, 2, 3] * 10)
    art = tmp_path / "real_cal_temp.npz"
    save_calibration(art, probs, labels, CLASS_NAMES, basis=BASIS_REAL,
                     train_groups=["TR"], calib_groups=["CAL"] * 40)
    monkeypatch.setenv("AION_CALIBRATION_NPZ", str(art))
    from server.main import _build_certifier
    _build_certifier(app)
    app.state.cert_store = CertificateStore(path=tmp_path / "c.jsonl")
    assert isinstance(app.state.coverage_temperature, float)
    body = client.post("/predict_certified", json={"signal": _signal()}).json()
    cov = body["certificate"]["coverage_guarantee"]
    assert "calibration_basis=real-holdout" in cov
    # the stamped temperature, if present, must match the served factor
    if app.state.coverage_temperature != 1.0:
        assert f"T={app.state.coverage_temperature:.2f}" in cov
