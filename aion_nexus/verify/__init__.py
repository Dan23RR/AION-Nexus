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
"""
from __future__ import annotations

from .certificate import (
    AUTH_HMAC,
    AUTH_NONE,
    CERT_SCHEMA_VERSION,
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    Certificate,
    sha256_signal,
    verify_certificate,
)
from .conformal import ConformalCalibrator, ConformalResult, softmax
from .store import CertificateStore, ChainResult
from .verifier import Verifier

__all__ = [
    "Verifier",
    "Certificate",
    "ConformalCalibrator",
    "ConformalResult",
    "CertificateStore",
    "ChainResult",
    "verify_certificate",
    "sha256_signal",
    "softmax",
    # constants
    "AUTH_HMAC",
    "AUTH_NONE",
    "CERT_SCHEMA_VERSION",
    "VERDICT_CERTIFIED",
    "VERDICT_REVIEW",
    "VERDICT_ABSTAIN",
]
