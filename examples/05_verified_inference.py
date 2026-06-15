"""Example 5: VERIFIED inference end-to-end — the Substrate Core pattern.

This is the value-capture layer of the AION-NEXUS / Verifier Labs thesis made
runnable: a classifier *proposes* a probability vector; a model-agnostic
*verifier* (conformal prediction set + abstain logic) *disposes* a verdict; and
the decision ships as an auditable, optionally tamper-evident **Certificate**
that maps to EU AI Act evidence.

    model proposes  ->  verifier + conformal dispose  ->  certificate (auditable)

Run:
    python examples/05_verified_inference.py [path/to/signal.csv]

Honest limits baked into this example (read them — they are the brand):

1. **Conformal coverage holds ONLY under exchangeability.** The 1 - alpha
   coverage guarantee is valid when the calibration data and the data being
   served are exchangeable (e.g. drawn i.i.d. from the same distribution). In
   bearing PdM, calibrating on bearing/machine A and serving on bearing/machine
   B BREAKS exchangeability and VOIDS the guarantee — the sets may under-cover.
   Calibrate per-bearing, or treat coverage as advisory and monitor it. Here we
   calibrate on a handful of synthetic samples purely to make the API runnable;
   that is NOT a valid calibration for any real deployment.

2. **Tamper-evidence requires a key.** Without ``VERIFY_HMAC_KEY`` set in the
   environment, the certificate carries ``authentication = NONE`` — an integrity
   hash only, NOT tamper-evident against an adversary who holds this source code.
   Set ``VERIFY_HMAC_KEY`` to get ``authentication = HMAC-SHA256`` (forgery-
   resistant). This example prints which mode you are in.

3. **The label is a degradation STAGE, not a fault type and not RUL in hours.**
   The 4 classes are a positional life-stage proxy
   (``degradation_pct = file_idx / (total - 1)`` quantized into 4 bins), NOT an
   independently diagnosed fault type and NOT a calibrated time-to-failure.

4. **The certificate provides EVIDENCE TOWARD EU AI Act articles, not
   compliance.** ``compliance_evidence`` maps the certificate to Art.12
   (logging), Art.14 (human oversight) and Art.15 (accuracy/robustness). It
   PROVIDES EVIDENCE; it does NOT make the system "EU AI Act compliant".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from aion_nexus import InferenceEngine
from aion_nexus.config import CLASS_NAMES
from aion_nexus.utils import load_signal_csv


def _load_signal(argv: list[str]) -> np.ndarray:
    """Load a 2-channel signal from CSV, or fall back to a synthetic window."""
    if len(argv) > 1:
        signal = load_signal_csv(argv[1])
        print(f"Loaded signal: {argv[1]}  shape={signal.shape}")
        return signal
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
    print("No CSV provided — using a synthetic random signal as demo "
          "(predictions are NOT meaningful on random input).")
    return signal


def _load_engine() -> InferenceEngine:
    """Load the v1 checkpoint, or random weights for an API-only smoke run."""
    ckpt = Path("checkpoints/aion_nexus_v1.pth")
    if ckpt.exists():
        return InferenceEngine.from_checkpoint(ckpt)
    print(f"Checkpoint not found at {ckpt}. See checkpoints/README.md.")
    print("Running with RANDOM weights — predictions are NOT meaningful; this "
          "run only exercises the verify/certify API surface.")
    from aion_nexus.model import create_aion_nexus
    return InferenceEngine(create_aion_nexus())


def _make_calibration_set(engine: InferenceEngine, n_per_class: int = 8):
    """Build a TINY synthetic calibration set (probs + labels).

    HONESTY: this is a placeholder so the API is runnable without shipping a
    private dataset. It is NOT a valid calibration for a real deployment —
    coverage holds only under exchangeability with the SERVING data, and random
    synthetic windows are not exchangeable with any real bearing. In production
    you calibrate on a held-out, in-distribution (ideally per-bearing) split.
    """
    rng = np.random.default_rng(123)
    probs_list: list[np.ndarray] = []
    labels_list: list[int] = []
    n_classes = len(CLASS_NAMES)
    for cls in range(n_classes):
        for _ in range(n_per_class):
            sig = rng.standard_normal((2, 2560)).astype(np.float32) * 0.5
            res = engine.predict(sig)
            p = np.array([res.probabilities[name] for name in CLASS_NAMES],
                         dtype=np.float64)
            probs_list.append(p)
            labels_list.append(cls)  # synthetic ground-truth label (placeholder)
    return np.vstack(probs_list), np.array(labels_list, dtype=int)


def main(argv: list[str]) -> int:
    signal = _load_signal(argv)
    engine = _load_engine()

    # 1) Model PROPOSES — raw classifier output (unchanged, fully preserved).
    pred = engine.predict(signal)
    probs = np.array([pred.probabilities[name] for name in CLASS_NAMES],
                     dtype=np.float64)
    print()
    print("--- 1. Model proposal (raw classifier output) ---")
    print(f"  point prediction: {pred.predicted_class_name} "
          f"(stage index {pred.predicted_class_index})")
    print(f"  confidence:       {pred.confidence:.3f} ({pred.confidence_band})")
    print("  NOTE: this is a degradation STAGE, not a fault type, not RUL in hours.")

    # 2) Verifier DISPOSES — calibrate a conformal verifier, then certify.
    #    The Verifier is model-agnostic: it only sees probability vectors, so the
    #    same code wraps v1 / v6 / v3 / any third-party classifier identically.
    from aion_nexus.verify import Verifier, verify_certificate

    print()
    print("--- 2. Calibrating the conformal verifier (alpha=0.1) ---")
    probs_calib, labels_calib = _make_calibration_set(engine)
    verifier = Verifier(alpha=0.1, class_names=CLASS_NAMES)
    verifier.calibrate(probs_calib, labels_calib)
    print(f"  calibrated on {len(labels_calib)} samples "
          f"(SYNTHETIC placeholder — not a real calibration set)")
    print(f"  coverage valid ONLY under: {verifier.coverage_valid_under}")

    # 3) Emit a Certificate. ``input_signal`` is hashed into the cert so the
    #    record is bound to the exact input. ``VERIFY_HMAC_KEY`` (if set) makes it
    #    tamper-evident; otherwise authentication = NONE (integrity hash only).
    print()
    print("--- 3. Certificate ---")
    cert = verifier.certify(probs, input_signal=signal, model_id="aion-nexus-v1")
    print(f"  verdict:        {cert.verdict}  "
          "(CERTIFIED = singleton set | REVIEW = ambiguous set | ABSTAIN = low conf)")
    print(f"  predicted:      {cert.predicted_name}")
    print(f"  conformal set:  {{{', '.join(cert.conformal_set_names)}}}  "
          f"(indices {cert.conformal_set})")
    print(f"  authentication: {cert.authentication}", end="")
    if cert.authentication == "NONE":
        print("  <-- integrity hash only; NOT tamper-evident. "
              "Set VERIFY_HMAC_KEY for HMAC-SHA256.")
    else:
        print("  <-- HMAC-SHA256 keyed: forgery-resistant.")
    print(f"  content_hash:   {cert.content_hash[:16]}...  (recomputable by anyone)")
    print(f"  input_sha256:   {(cert.input_sha256 or '(none)')[:16]}...")

    # Re-audit the certificate (integrity always; authenticity only with a key).
    audit = verify_certificate(cert)
    print(f"  re-audit:       integrity_ok={audit['integrity_ok']} "
          f"authenticity={audit['authenticity']}")
    if not os.environ.get("VERIFY_HMAC_KEY"):
        print("                  (authenticity UNVERIFIED is EXPECTED with no key — "
              "this is honest, not a failure.)")

    # 4) Map the certificate to EU AI Act EVIDENCE (not a compliance claim).
    print()
    print("--- 4. EU AI Act evidence (Art.12 / 14 / 15) ---")
    try:
        from aion_nexus.compliance import compliance_evidence
    except ImportError:
        print("  aion_nexus.compliance not available in this build — skipping.")
        print("  (When present, compliance_evidence(cert) returns the article-by-")
        print("   article EVIDENCE mapping. It PROVIDES EVIDENCE TOWARD the EU AI")
        print("   Act; it does NOT certify the system as 'compliant'.)")
        return 0

    evidence = compliance_evidence(cert)
    print("  This output PROVIDES EVIDENCE TOWARD the EU AI Act; it does NOT")
    print("  declare the system 'compliant'. Compliance is an org/process result,")
    print("  not something a single certificate can assert.")
    # Print the mapping generically so this stays correct regardless of the exact
    # return shape of compliance_evidence (dict of article -> evidence).
    if isinstance(evidence, dict):
        for article, detail in evidence.items():
            print(f"  - {article}: {detail}")
    else:
        print(f"  {evidence}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
