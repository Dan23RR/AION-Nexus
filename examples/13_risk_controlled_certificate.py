"""Example 13: a SAFETY-bounded certificate on real FEMTO (v2.18.0).

Marginal conformal controls miscoverage; a predictive-maintenance buyer needs
more: a bound on the CATASTROPHIC miss — calling a degraded bearing healthy.
Conformal Risk Control (Angelopoulos et al. 2022) and RCPS (Bates et al. 2021)
pick a single threshold on a calibration set such that the EXPECTED false-healthy
rate is bounded, distribution-free and finite-sample.

This validates the guarantee on REAL FEMTO data and mints a risk-controlled
certificate verified offline. Run:

    python examples/13_risk_controlled_certificate.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aion_nexus.config import CLASS_NAMES
from aion_nexus.inference import InferenceEngine
from aion_nexus.verify import (
    Verifier,
    conformal_risk_control,
    ed25519_pubkey_from_seed,
    empirical_risk,
    generate_seed,
    rcps_threshold,
    verify_certificate,
)
from scripts.eval_real_femto import load_femto

_PKG = Path(__file__).resolve().parent.parent
_FEMTO = _PKG / "data" / "FEMTO+Bearing" / "10. FEMTO Bearing" / "FEMTOBearingDataSet"
_CKPT = _PKG / "checkpoints" / "aion_nexus_v1.pth"
ALPHA = 0.05


def main() -> int:
    if not _FEMTO.exists():
        print(f"FEMTO not found at {_FEMTO}; skipping.")
        return 0
    print("Loading real FEMTO (RAW)...")
    data = load_femto(_FEMTO, per_bearing=120)
    windows, y = data["windows"], data["labels"]
    engine = (InferenceEngine.from_checkpoint(str(_CKPT)) if _CKPT.exists()
              else InferenceEngine(__import__("aion_nexus.model", fromlist=["create_aion_nexus"]).create_aion_nexus()))
    res = engine.predict_batch(windows)
    probs = np.array([[r.probabilities[c] for c in CLASS_NAMES] for r in res], dtype=np.float64)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    ci, ti = perm[: len(y) // 2], perm[len(y) // 2:]

    base_miss = float(np.mean([
        (int(y[i]) in (2, 3)) and (int(probs[i].argmax()) in (0, 1)) for i in ti]))
    print(f"\nBaseline (point prediction) false-healthy rate: {base_miss:.3f}")

    crc = conformal_risk_control(probs[ci], y[ci], alpha=ALPHA)
    rcps = rcps_threshold(probs[ci], y[ci], alpha=ALPHA, delta=0.1)
    realized_crc = empirical_risk(probs[ti], y[ti], crc.lambda_hat)
    realized_rcps = empirical_risk(probs[ti], y[ti], rcps.lambda_hat)
    avg_set = float(np.mean([len(crc.prediction_set(probs[i])) for i in ti]))

    print(f"\nCRC  (alpha={ALPHA}): lambda={crc.lambda_hat:.2f}  "
          f"realized false-healthy on held-out = {realized_crc:.3f}  (<= {ALPHA}? "
          f"{realized_crc <= ALPHA + 0.02})  avg|set|={avg_set:.2f}")
    print(f"  {crc.guarantee}")
    print(f"RCPS (alpha={ALPHA}, delta=0.1): lambda={rcps.lambda_hat:.2f}  "
          f"realized = {realized_rcps:.3f}")
    print(f"  {rcps.guarantee}")

    # mint a certificate carrying the risk-control guarantee, verify offline
    seed = generate_seed()
    verifier = Verifier(alpha=0.1, class_names=list(CLASS_NAMES)).calibrate(probs[ci], y[ci])
    j = int(ti[0])
    cert = verifier.certify(probs[j], input_signal=np.asarray(windows[j]),
                            model_id="aion-v1-riskctrl", seed=seed, ttl_seconds=3600,
                            conformal_method="conformal-risk-control",
                            coverage_guarantee=crc.guarantee)
    vres = verify_certificate(cert.as_dict(), expected_pubkey=ed25519_pubkey_from_seed(seed))
    print(f"\nRisk-controlled certificate: verdict={cert.verdict}, offline trusted={vres['trusted']}")

    # HONESTY (6.31): the v1 model OVER-predicts 'advanced', so it rarely calls a
    # degraded bearing healthy -> the baseline miss rate is already near alpha. Here
    # the value of CRC is the distribution-free GUARANTEE (now certifiable), not a
    # large reduction; on a model that UNDER-flags, CRC tightens the miss rate
    # materially (tests/test_risk_control proves 0.08 -> 0.04 on synthetic data).
    if base_miss <= ALPHA:
        verdict = ("the model already over-flags (low miss rate); CRC now GUARANTEES "
                   "the bound, finite-sample and distribution-free")
    else:
        verdict = f"CRC reduced the miss rate from {base_miss:.3f} to {realized_crc:.3f}"
    print(f"\n-> {verdict}.")

    assert realized_crc <= ALPHA + 0.02, "CRC must control the risk on held-out real data"
    assert vres["trusted"], "the certificate must verify offline"
    print("OK — on REAL FEMTO the false-healthy rate carries a distribution-free bound "
          "(CRC), and the certificate that states it verifies offline. This is the "
          "safety claim an industrial buyer can act on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
