"""Example 6: CERTIFIED serving end-to-end — the weapon, wired in and honest.

The unique claim of AION-NEXUS / Verifier Labs is INDEPENDENT, third-party
verification: a decision ships as an Ed25519-SIGNED certificate that an outsider
— an auditor, a customer, an insurer who holds ONLY the public key — can verify
OFFLINE, without ever receiving anything that would let them forge a new one.

This example makes that real and shows its EDGES, not just its happy path:

    1. ISSUER mints a signed, time-boxed certificate (seed = secret authority).
    2. AUDITOR (public key ONLY, NO seed) verifies it offline  -> trusted=True.
    3. AUDITOR's check fails when a field is TAMPERED              -> trusted=False.
    4. AUDITOR's check fails against the WRONG issuer pubkey       -> trusted=False.
    5. An EXPIRED certificate fails the validity-window check      -> expired=True.

It runs WITHOUT a server (drives the Verifier directly). If the FastAPI server is
up with VERIFY_ED25519_SEED set, POST /predict_certified does exactly step 1 and
POST /verify does steps 2-5 over HTTP — same certificate, same verdicts.

HONEST LIMITS (workspace rule 6.31 — these are the brand, read them):

* This is ADVISORY verification of a DECISION RECORD, not loop safety. A valid
  certificate attests that THIS verifier, on THIS input, reached THIS verdict and
  that the record was not tampered. It does NOT prove the prediction is correct,
  and it is NOT a real-time safety interlock.
* Conformal coverage holds ONLY under exchangeability of calibration and serving
  data. The calibration set below is SYNTHETIC and is NOT valid for any real
  bearing — it exists only to make the API runnable. The signature/audit trail
  are real; the coverage NUMBER from this toy calibrator is a placeholder.
* "trusted" is the SINGLE safe flag. It is True ONLY when integrity holds AND the
  signature verifies against the EXPECTED (out-of-band) issuer key AND the cert is
  inside its validity window. A signature checked against the cert's OWN embedded
  key is merely SELF-SIGNED (trusted=False): an attacker can mint their own seed.

Run:
    python examples/06_certified_serving.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus import InferenceEngine
from aion_nexus.config import CLASS_NAMES
from aion_nexus.verify import (
    Verifier,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)


def _engine() -> InferenceEngine:
    """Load the v1 checkpoint if present, else random weights for an API smoke run."""
    from pathlib import Path
    ckpt = Path("checkpoints/aion_nexus_v1.pth")
    if ckpt.exists():
        return InferenceEngine.from_checkpoint(ckpt)
    print("Checkpoint not found — using RANDOM weights. Predictions are NOT "
          "meaningful; this run only exercises the certify/verify surface.")
    from aion_nexus.model import create_aion_nexus
    return InferenceEngine(create_aion_nexus())


def _calibrated_verifier(engine: InferenceEngine) -> Verifier:
    """Calibrate a conformal verifier on a TINY synthetic set (placeholder).

    HONESTY: random windows are NOT exchangeable with any real bearing, so this
    calibration is not valid for deployment — it only makes the API runnable.
    """
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


def main() -> int:
    engine = _engine()
    verifier = _calibrated_verifier(engine)

    # ---- 1. ISSUER mints a signed, time-boxed certificate --------------------
    # generate_seed() is the BLESSED full-entropy path (os.urandom(32) hex); it is
    # the SECRET minting authority. Only the PUBLIC key is published.
    seed = generate_seed()
    issuer_pub = ed25519_pubkey_from_seed(seed)

    rng = np.random.default_rng(7)
    signal = (rng.standard_normal((2, 2560)).astype(np.float32) * 0.5)
    pred = engine.predict(signal)
    probs = np.array([pred.probabilities[n] for n in CLASS_NAMES], dtype=np.float64)

    cert = verifier.certify(
        probs, input_signal=signal, model_id="aion-nexus-demo",
        seed=seed, ttl_seconds=3600, key_id="demo-key-1",
    )
    print("--- 1. Issuer minted a SIGNED certificate ---")
    print(f"  {cert.summary()}")
    print(f"  authentication: {cert.authentication}   key_id: {cert.key_id}")
    print(f"  valid:          {cert.not_before}  ..  {cert.valid_until}")
    print(f"  issuer pubkey:  {issuer_pub[:24]}...  (published; the seed is NOT)")

    # ---- 2. AUDITOR (public key ONLY) verifies OFFLINE -----------------------
    # The auditor never sees the seed. Passing expected_pubkey is what makes this
    # genuine issuer authentication (VERIFIED) rather than mere self-consistency.
    audit = verify_certificate(cert, expected_pubkey=issuer_pub)
    print("\n--- 2. Auditor verifies with the PUBLIC KEY ALONE ---")
    print(f"  integrity_ok={audit['integrity_ok']}  authenticity={audit['authenticity']}  "
          f"trusted={audit['trusted']}")
    assert audit["trusted"] is True, "a genuine cert must be trusted by the auditor"
    print("  -> trusted=True: the auditor confirmed the issuer WITHOUT the secret.")

    # ---- 3. TAMPER a field -> the signature no longer matches ----------------
    # Forge a DIFFERENT display label than the one certified (whatever it was), so
    # the edit is real regardless of which class the model happened to pick.
    tampered = cert.as_dict()
    forged_name = next(n for n in CLASS_NAMES if n != cert.predicted_name)
    tampered["predicted_name"] = forged_name
    t_audit = verify_certificate(tampered, expected_pubkey=issuer_pub)
    print("\n--- 3. Auditor catches a TAMPERED field ---")
    print(f"  integrity_ok={t_audit['integrity_ok']}  authenticity={t_audit['authenticity']}  "
          f"trusted={t_audit['trusted']}")
    assert t_audit["trusted"] is False, "a tampered cert must NOT be trusted"
    print("  -> trusted=False: editing even a display label breaks the hash.")

    # ---- 4. WRONG issuer key -> not the trusted issuer -----------------------
    other_pub = ed25519_pubkey_from_seed(generate_seed())   # a different identity
    w_audit = verify_certificate(cert, expected_pubkey=other_pub)
    print("\n--- 4. Auditor rejects the WRONG issuer key ---")
    print(f"  authenticity={w_audit['authenticity']}  trusted={w_audit['trusted']}")
    assert w_audit["trusted"] is False, "wrong issuer key must NOT be trusted"
    print("  -> trusted=False: a self-consistent cert from another seed is FORGED "
          "against the expected issuer.")

    # ---- 5. EXPIRED certificate -> validity-window check fails ----------------
    # Mint with now pinned in the past so the 1-second window is already closed;
    # the decision hash stays deterministic, only the (signed) window expires.
    expired_cert = verifier.certify(
        probs, input_signal=signal, model_id="aion-nexus-demo", seed=seed,
        ttl_seconds=1, key_id="demo-key-1",
        now_iso="2000-01-01T00:00:00.000+00:00",
    )
    e_audit = verify_certificate(expired_cert, expected_pubkey=issuer_pub)
    print("\n--- 5. Auditor rejects an EXPIRED certificate ---")
    print(f"  authenticity={e_audit['authenticity']}  expired={e_audit.get('expired')}  "
          f"trusted={e_audit['trusted']}")
    assert e_audit.get("expired") is True, "the window should be expired"
    assert e_audit["trusted"] is False, "an expired cert must NOT be trusted"
    print("  -> expired=True, trusted=False: a replayed/stale cert is refused even "
          "though its signature is otherwise valid (anti-replay).")

    print("\nAll checks passed. This is ADVISORY verification of a decision record "
          "(not loop safety); coverage holds only under exchangeability.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
