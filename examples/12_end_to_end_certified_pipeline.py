"""Example 12: the verification layer as ONE served product, end to end (v2.16.0).

This is the answer to "the layer is a library, not a product": a single,
reproducible chain from a REAL checkpoint to an offline-verifiable certificate
that composes the conformal verdict, the physics second opinion, the compliance
evidence and a signed evaluation report — with the synthetic-vs-real calibration
basis made an explicit, certificate-bound fact.

    raw window
       -> engine.predict (real v1 checkpoint)
       -> conformal certify  (basis stamped tamper-evidently into coverage_guarantee)
       -> physics second opinion (CONTRADICT a confident-but-wrong fault claim)
       -> compose (weakest link)  -> system verdict
       -> verify_certificate (public key only, offline)  -> trusted
       -> compliance evidence (EU AI Act Art.12/14/15) + Annex IV dossier
       -> signed, offline-verifiable leakage-free evaluation report

Every step hard-asserts. Honest scope: the calibration here is a labelled
SYNTHETIC-DEMO set (basis='synthetic-demo'), so the coverage number is a
placeholder — the chain shows exactly where a deployer drops in a real,
leakage-checked artifact (scripts/build_calibration) to make it real, and the
leakage gate is demonstrated live.

Run:
    python examples/12_end_to_end_certified_pipeline.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aion_nexus.compliance import annex_iv_dossier, compliance_evidence
from aion_nexus.config import CLASS_NAMES, SAMPLING_RATE_HZ
from aion_nexus.evaluation import evaluate_leave_one_group_out, verify_evaluation_report
from aion_nexus.inference import InferenceEngine
from aion_nexus.physics import (
    FAULT_INNER,
    FAULT_OUTER,
    PHYS_CONTRADICT,
    BearingGeometry,
    physics_consistency,
)
from aion_nexus.serving_calibration import (
    BASIS_DEMO,
    BASIS_REAL,
    coverage_guarantee_string,
    save_calibration,
    synthetic_demo_probs,
)
from aion_nexus.verify import (
    Verifier,
    compose_certificates,
    ed25519_pubkey_from_seed,
    generate_seed,
    verify_certificate,
)

_PKG = Path(__file__).resolve().parent.parent
_CKPT = _PKG / "checkpoints" / "aion_nexus_v1.pth"
# SKF 6205: 9 balls, ball d=7.94 mm, pitch D=39.04 mm, deep-groove (phi=0).
SKF_6205 = BearingGeometry(n_rolling_elements=9, ball_diameter=7.94, pitch_diameter=39.04)
RPM = 1800.0


def _bearing_fault_signal(fault_order: float, *, n: int = 25_600, seed: int = 5) -> np.ndarray:
    """A 2-channel impulse-train-rings-a-resonance bearing fault at ``fault_order``."""
    rng = np.random.default_rng(seed)
    fr = RPM / 60.0
    t = np.arange(n) / SAMPLING_RATE_HZ
    sig = 0.05 * rng.standard_normal(n)
    period = 1.0 / (fault_order * fr)
    for t0 in np.arange(0.0, n / SAMPLING_RATE_HZ, period):
        tt = t - t0
        idx = tt >= 0
        sig[idx] += np.exp(-800.0 * tt[idx]) * np.sin(2 * np.pi * 3000.0 * tt[idx])
    return np.vstack([sig, sig]).astype(np.float32)


def main() -> int:
    # ---- 0. real engine -----------------------------------------------------
    if _CKPT.exists():
        engine = InferenceEngine.from_checkpoint(str(_CKPT))
        src = f"real checkpoint {_CKPT.name}"
    else:  # fall back to a fresh model so the example still runs (clearly labelled)
        from aion_nexus.model import create_aion_nexus
        engine = InferenceEngine(create_aion_nexus())
        src = "fresh random-init model (checkpoint absent)"
    print(f"--- 0. Engine: {src} ---")

    # ---- 1. calibrate the served verifier (basis-honest) --------------------
    probs_cal, labels_cal = synthetic_demo_probs(
        lambda s: np.array([engine.predict(s).probabilities[n] for n in CLASS_NAMES],
                           dtype=np.float64),
        len(CLASS_NAMES))
    verifier = Verifier(alpha=0.1, class_names=list(CLASS_NAMES)).calibrate(probs_cal, labels_cal)
    basis = BASIS_DEMO
    cov = coverage_guarantee_string(basis, 0.1, leakage_checked=False)
    print(f"--- 1. Conformal verifier calibrated (basis={basis}) ---")
    print(f"    coverage_guarantee stamped into the cert: {cov[:72]}...")

    # ---- 1b. the leakage GATE is real: a leaked 'real' artifact is REFUSED ---
    leaked = False
    try:
        save_calibration(_PKG / "checkpoints" / "_tmp_leak.npz", probs_cal, labels_cal,
                         CLASS_NAMES, basis=BASIS_REAL,
                         train_groups=["B1", "B2"], calib_groups=["B1"] * len(labels_cal))
    except ValueError:
        leaked = True
    print(f"--- 1b. Leakage gate: 'real-holdout' with a leaked split REFUSED = {leaked} ---")

    # ---- 2. predict on an INNER-race fault, but DOMAIN expects OUTER ---------
    signal = _bearing_fault_signal(SKF_6205.fault_orders()[FAULT_INNER])
    result = engine.predict(signal)
    probs = np.array([result.probabilities[n] for n in CLASS_NAMES], dtype=np.float64)
    print(f"--- 2. Model verdict: {result.predicted_class_name} "
          f"(conf {result.confidence:.2f}) ---")

    # ---- 3. signed certificate (Ed25519), basis bound into the hash ----------
    seed = generate_seed()
    cert = verifier.certify(probs, input_signal=signal, model_id="aion-nexus-e2e",
                            seed=seed, ttl_seconds=3600, key_id="demo-key",
                            conformal_method="marginal-split-conformal",
                            coverage_guarantee=cov)
    print(f"--- 3. Certificate sealed: verdict={cert.verdict}, "
          f"auth={cert.authentication} ---")

    # ---- 4. physics second opinion: CONTRADICT the 'outer' claim -------------
    phys = physics_consistency(signal, fs=SAMPLING_RATE_HZ, rpm=RPM, geometry=SKF_6205,
                               claimed_fault=FAULT_OUTER)
    print(f"--- 4. Physics: {phys.verdict} (dominant={phys.dominant_fault}) "
          f"-> {phys.detail[:70]}... ---")

    # ---- 5. compose: a confident-but-wrong claim must NOT stay CERTIFIED ------
    composed = compose_certificates([cert, phys.as_component()], op="and")
    print(f"--- 5. Composed system verdict (weakest link): {composed['verdict']} "
          f"@ assurance {composed['assurance']} ---")

    # ---- 6. offline verification with the PUBLIC KEY only --------------------
    pub = ed25519_pubkey_from_seed(seed)
    vres = verify_certificate(cert.as_dict(), expected_pubkey=pub)
    print(f"--- 6. Offline verify (public key only): trusted={vres['trusted']} "
          f"({vres['authenticity']}) ---")

    # ---- 7. served compliance evidence + Annex IV ----------------------------
    evidence = compliance_evidence(cert)
    dossier = annex_iv_dossier({"architecture": "MultiScaleCNN+BiGRU (v1)"}, certificate=cert)
    print(f"--- 7. Evidence: {len(evidence['evidence'])} EU-AI-Act/ISO items; "
          f"Annex IV: {len(dossier['sections'])} sections ---")

    # ---- 8. signed, offline-verifiable leakage-free evaluation report --------
    rng = np.random.default_rng(0)
    ncls = len(CLASS_NAMES)
    cdir = rng.standard_normal((ncls, 8))
    cdir /= np.linalg.norm(cdir, axis=1, keepdims=True)
    bdir = rng.standard_normal((6, 8))
    bdir /= np.linalg.norm(bdir, axis=1, keepdims=True)
    fx, fy, fg = [], [], []
    for b in range(6):
        for _ in range(80):
            c = rng.integers(0, ncls)
            fx.append(3.0 * cdir[c] + 12.0 * bdir[b] + 0.3 * rng.standard_normal(8))
            fy.append(c)
            fg.append(b)

    def _knn(x_tr, y_tr, x_te, k=5):
        d = ((x_te[:, None, :] - x_tr[None, :, :]) ** 2).sum(-1)
        votes = y_tr[np.argsort(d, axis=1)[:, :k]]
        sc = np.stack([(votes == c).mean(1) for c in range(ncls)], axis=1)
        return sc.argmax(1), sc

    report = evaluate_leave_one_group_out(_knn, np.array(fx), np.array(fy), np.array(fg),
                                          n_classes=ncls, group_kind="bearing",
                                          model_id="knn-e2e")
    rseed = generate_seed()
    report.seal(rseed, scheme="ed25519")
    rres = verify_evaluation_report(report, expected_pubkey=ed25519_pubkey_from_seed(rseed))
    print(f"--- 8. Signed LOBO report: macro-F1 {report.f1_macro['mean']:.3f}"
          f"±{report.f1_macro['std']:.3f}, trusted={rres['trusted']} ---")

    # ---- hard invariants -----------------------------------------------------
    assert leaked, "the leakage gate must refuse a leaked real-holdout artifact"
    assert phys.verdict == PHYS_CONTRADICT, "physics must contradict the wrong claim"
    assert composed["verdict"] != "CERTIFIED", "contradicted system must not stay CERTIFIED"
    assert vres["trusted"], "a freshly signed cert must verify against its pubkey"
    assert "calibration_basis=synthetic-demo" in cert.as_dict().get("coverage_guarantee", "")
    assert len(evidence["evidence"]) >= 5 and len(dossier["sections"]) == 9
    assert rres["trusted"], "the signed evaluation report must verify offline"
    print("\nOK — one chain: real model -> conformal -> physics -> composed verdict -> "
          "offline-verifiable certificate -> EU-AI-Act evidence -> signed eval report. "
          "Drop a real calibration artifact in to turn coverage_basis -> real-holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
