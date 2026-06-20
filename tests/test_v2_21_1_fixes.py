"""Regression tests for the v2.21.1 fix batch (red-team audit 2026-06-20).

Each test pins a specific bug closed so it cannot silently regress:

- B1  few-shot must not drift the FROZEN encoder's BatchNorm running stats
- B2  a deployed keyring makes a cert WITHOUT a key_id untrusted (fail-closed)
- B3  the strict entropy floor rejects a long-but-low-entropy seed (b"a"*32)
- B4  (ATTEMPTED then REVERTED — making APS deterministic/consistent collapsed the
      CERTIFIED rate to full sets; the randomized-calibration APS is retained because
      it yields the useful singletons. See CHANGELOG / redteam doc. No test here.)
- B5  risk control reports honestly when the target bound is not achievable
- B6  verify_certificate rejects a verdict<->set inconsistency, even if signed
- B7  KeyRing round-trip preserves retired/revoked metadata (reason, not_after)
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# B1 — few-shot freezes the encoder's BatchNorm running stats
# --------------------------------------------------------------------------- #
def test_b1_fewshot_does_not_drift_encoder_batchnorm():
    import torch

    from aion_nexus.few_shot import FewShotAdapter
    from aion_nexus.inference import InferenceEngine
    from aion_nexus.model import AIONNexus

    engine = InferenceEngine(AIONNexus(), device="cpu", architecture_version="v1")
    adapter = FewShotAdapter(engine)

    head_roots = tuple(p.rstrip(".") for p in adapter._head_prefixes)

    def is_encoder(name: str) -> bool:
        return not any(name == r or name.startswith(r + ".") for r in head_roots)

    before = {
        n: m.running_mean.detach().clone()
        for n, m in adapter.model.named_modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and is_encoder(n)
    }
    assert before, "expected at least one encoder BatchNorm to guard"

    rng = np.random.default_rng(0)
    signals = [rng.standard_normal((2, 2560)).astype("float32") for _ in range(8)]
    labels = [0, 1, 2, 3, 0, 1, 2, 3]
    adapter.adapt(signals, labels, epochs=5, verbose=False)

    for n, m in adapter.model.named_modules():
        if n in before:
            assert torch.allclose(m.running_mean, before[n]), (
                f"encoder BatchNorm {n!r} running_mean drifted during adapt() — "
                "the frozen encoder must not absorb the target batch"
            )


# --------------------------------------------------------------------------- #
# B2 — keyring fail-closed: a cert without key_id cannot be trusted
# --------------------------------------------------------------------------- #
def test_b2_keyring_untrusts_cert_without_key_id():
    from aion_nexus.verify import KeyRing, Verifier
    from aion_nexus.verify.keyring import verify_with_keyring
    from aion_nexus.verify.signing import ed25519_pubkey_from_seed, generate_seed

    seed = generate_seed()
    pub = ed25519_pubkey_from_seed(seed)

    v = Verifier(alpha=0.1, class_names=["a", "b"])
    v.calibrate(
        np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]]),
        np.array([0, 1, 0, 1]),
    )
    cert = v.certify(np.array([0.95, 0.05]), seed=seed)  # signed, but NO key_id
    assert cert.key_id is None

    ring = KeyRing().register("k1", pub)
    res = verify_with_keyring(cert, ring)
    assert res["trusted"] is False
    assert res["authenticity"] == "UNKNOWN-KEY"


# --------------------------------------------------------------------------- #
# B3 — strict entropy floor rejects a long-but-guessable seed
# --------------------------------------------------------------------------- #
def test_b3_strict_rejects_long_low_entropy_seed():
    from aion_nexus.verify.signing import (
        LocalEd25519Signer,
        assert_strong_seed,
        ed25519_sign,
        generate_seed,
    )

    # 32-byte constant clears the LENGTH floor but is trivially brute-forceable.
    with pytest.raises(ValueError, match="entropy floor"):
        assert_strong_seed("a" * 32)
    with pytest.raises(ValueError, match="entropy floor"):
        ed25519_sign("msg", "a" * 40, strict=True)
    with pytest.raises(ValueError, match="entropy floor"):
        LocalEd25519Signer("ab" * 16)  # 2 distinct bytes

    # A full-entropy seed passes silently.
    assert_strong_seed(generate_seed())
    LocalEd25519Signer(generate_seed())  # must not raise


# --------------------------------------------------------------------------- #
# B5 — risk control is honest when the target bound is not achievable
# --------------------------------------------------------------------------- #
def test_b5_risk_control_honest_when_unachievable():
    from aion_nexus.verify.risk_control import conformal_risk_control

    # n = 1 -> finite-sample term B/(n+1) = 0.5 > alpha for any reasonable alpha,
    # so NO lambda can meet the bound: the result must say so, not keep claiming it.
    r = conformal_risk_control(np.array([[0.9, 0.05, 0.03, 0.02]]), np.array([3]),
                               alpha=0.05)
    assert r.bound_achieved is False
    assert "NOT achievable" in r.guarantee
    assert "<= 0.050 (conformal risk control" not in r.guarantee  # not the rosy claim

    # Large, well-separated calibration -> the bound IS achievable.
    m = 400
    y = np.array([2, 3] * (m // 2))
    p = np.full((m, 4), 0.02)
    p[np.arange(m), y] = 0.94
    p = p / p.sum(axis=1, keepdims=True)
    r2 = conformal_risk_control(p, y, alpha=0.05)
    assert r2.bound_achieved is True


# --------------------------------------------------------------------------- #
# B6 — verify_certificate rejects a signed-but-inconsistent verdict<->set
# --------------------------------------------------------------------------- #
def test_b6_verify_rejects_verdict_set_mismatch():
    from aion_nexus.verify.certificate import (
        VERDICT_CERTIFIED,
        Certificate,
        verify_certificate,
    )
    from aion_nexus.verify.signing import ed25519_pubkey_from_seed, generate_seed

    seed = generate_seed()
    pub = ed25519_pubkey_from_seed(seed)

    # A CERTIFIED verdict stamped on a NON-singleton conformal set. The cert is
    # internally hash-consistent and validly signed (we seal it ourselves), so the
    # ONLY thing that can catch the contradiction is the verdict<->set check.
    cert = Certificate(
        predicted_label=0, predicted_name="a",
        conformal_set=[0, 2, 3], conformal_set_names=["a", "c", "d"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
    )
    cert.seal(seed, scheme="ed25519")

    res = verify_certificate(cert, expected_pubkey=pub)
    assert res["integrity_ok"] is True          # hash matches the (bad) payload
    assert res["authenticity"] == "VERIFIED"    # signature is genuinely valid
    assert res["verdict_consistent"] is False   # ...but the verdict<->set lies
    assert res["trusted"] is False              # so it must NOT be trusted

    # A consistent CERTIFIED singleton verifies trusted (control).
    good = Certificate(
        predicted_label=1, predicted_name="b",
        conformal_set=[1], conformal_set_names=["b"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
    )
    good.seal(seed, scheme="ed25519")
    ok = verify_certificate(good, expected_pubkey=pub)
    assert ok["verdict_consistent"] is True
    assert ok["trusted"] is True


# --------------------------------------------------------------------------- #
# B7 — KeyRing round-trip preserves retired/revoked metadata
# --------------------------------------------------------------------------- #
def test_b7_keyring_roundtrip_preserves_metadata():
    from aion_nexus.verify.keyring import KeyRing

    ring = KeyRing().rotate("k1", "aa" * 32).rotate("k2", "bb" * 32)  # k1 retired
    ring.revoke("k2", "exposed in incident-9")

    ring2 = KeyRing.from_dict(ring.to_dict())

    k1 = ring2.get("k1")
    assert k1 is not None and k1.status == "retired"
    assert k1.reason == "rotated out"            # was being dropped on load

    k2 = ring2.get("k2")
    assert k2 is not None and k2.status == "revoked"
    assert k2.reason == "exposed in incident-9"
    assert ring2.is_revoked("k2") is True
