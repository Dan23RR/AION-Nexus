"""Substrate Core — the model-agnostic verification / certification layer.

This is the value-capture layer of the AION-NEXUS / Verifier thesis brought into
the public package: a calibrated-trust wrapper that sits above ANY classifier
(our v1 BiGRU, the v6 attention model, the v3 substrate encoder, or a
third-party model) and turns raw probabilities into auditable certificates.

What it provides — and what it does NOT
---------------------------------------
- **Conformal prediction sets** with a distribution-free, finite-sample coverage
  guarantee VALID ONLY UNDER EXCHANGEABILITY of calibration and serving data.
  Cross-bearing / cross-machine deployment breaks exchangeability and voids the
  marginal guarantee. Every calibrator carries this caveat in
  ``coverage_valid_under``.
- **Tamper-evidence ONLY with a key.** With ``VERIFY_HMAC_KEY`` set, certificates
  and the chain are HMAC-SHA256 signed (forgery-resistant). Without a key they
  carry an integrity hash only — ``authentication = NONE`` — which is honest
  about NOT being tamper-evident against an adversary holding this source.
- **Compliance EVIDENCE, not certification.** The audit trail + human-oversight
  verdicts map to EU AI Act articles (Art.12 logging, Art.14 human oversight,
  Art.15 accuracy/robustness). This package PROVIDES EVIDENCE TOWARD those
  articles; it does NOT make a system "EU AI Act compliant".

Public API
----------
- :class:`Verifier`            — the facade: ``.calibrate(...).certify(...)``
- :class:`Certificate`         — the auditable per-decision record
- :class:`ConformalCalibrator` — the split-conformal core
- :class:`CertificateStore`    — append-only JSONL audit log with optional chain
- :func:`verify_certificate`   — audit integrity + authenticity of a certificate
- :func:`compose_certificates` — compose verdicts; the weakest assurance governs
- :func:`run_cheatbench`       — MEASURE the cheating surface of the gate
- Ed25519 signing primitives and the assurance lattice (anti-overclaim tiers)
"""
from __future__ import annotations

from . import assurance
from .assurance import (
    BOUNDED,
    EMPIRICAL,
    NONE,
    PROVEN,
    PROVEN_SPEC,
    residual_risk_rule_of_three,
    strongest,
    weakest,
)
from .assurance import (
    describe as assurance_describe,
)
from .assurance import (
    rank as assurance_rank,
)
from .certificate import (
    AUTH_ED25519,
    AUTH_HMAC,
    AUTH_NONE,
    CERT_SCHEMA_VERSION,
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    Certificate,
    require_authenticated,
    sha256_signal,
    verify_certificate,
)
from .cheatbench import run_cheatbench
from .conformal import ConformalCalibrator, ConformalResult, softmax
from .signing import (
    HmacSigner,
    LocalEd25519Signer,
    Signer,
    assert_strong_seed,
    ed25519_pubkey_from_seed,
    ed25519_sign,
    ed25519_verify,
    generate_seed,
    hmac_sign,
    hmac_verify,
)
from .store import CertificateStore, ChainResult
from .verifier import Verifier, compose_certificates

__all__ = [
    "Verifier",
    "Certificate",
    "ConformalCalibrator",
    "ConformalResult",
    "CertificateStore",
    "ChainResult",
    "verify_certificate",
    "require_authenticated",
    "compose_certificates",
    "sha256_signal",
    "softmax",
    # cheatbench
    "run_cheatbench",
    # signing primitives
    "generate_seed",
    "assert_strong_seed",
    "ed25519_pubkey_from_seed",
    "ed25519_sign",
    "ed25519_verify",
    "hmac_sign",
    "hmac_verify",
    # pluggable Signer interface (KMS/HSM-ready)
    "Signer",
    "LocalEd25519Signer",
    "HmacSigner",
    # assurance lattice
    "assurance",
    "NONE",
    "EMPIRICAL",
    "BOUNDED",
    "PROVEN_SPEC",
    "PROVEN",
    "weakest",
    "strongest",
    "assurance_rank",
    "assurance_describe",
    "residual_risk_rule_of_three",
    # constants
    "AUTH_HMAC",
    "AUTH_NONE",
    "AUTH_ED25519",
    "CERT_SCHEMA_VERSION",
    "VERDICT_CERTIFIED",
    "VERDICT_REVIEW",
    "VERDICT_ABSTAIN",
]
