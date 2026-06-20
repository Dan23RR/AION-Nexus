"""Verifier — the model-agnostic facade of Substrate Core.

Wrap ANY classifier that emits class-probability vectors (numpy arrays) with a
calibrated-trust layer and get back a re-runnable :class:`Certificate`:

    >>> v = Verifier(alpha=0.1, class_names=["normal", "early", "medium", "advanced"])
    >>> v.calibrate(probs_calib, labels_calib)
    >>> cert = v.certify(probs_one_sample, input_signal=window, model_id="aion-v1")
    >>> cert.verdict        # CERTIFIED | REVIEW | ABSTAIN

There is NO torch dependency: the verifier operates purely on probability
arrays, so it sits above the BiGRU (v1), the v6 attention model, the v3 substrate
encoder, or any third-party classifier identically.

Verdict logic
-------------
- ``CERTIFIED`` — the conformal set is a singleton AND the top probability clears
  ``abstain_threshold`` (a confident, coverage-controlled label).
- ``REVIEW``    — the conformal set has more than one label (genuine ambiguity;
  the coverage guarantee still holds, but the model cannot single out one class).
- ``ABSTAIN``   — the top probability is below ``abstain_threshold`` (the model
  is not confident enough to act, even if the set happens to be a singleton).

Coverage caveat: the conformal guarantee is valid only under exchangeability of
calibration and serving data. Cross-bearing / cross-machine deployment breaks
exchangeability and voids the marginal 1 - alpha guarantee — see
:class:`~aion_nexus.verify.conformal.ConformalCalibrator`. The caveat travels on
every calibrator via its ``coverage_valid_under`` field.
"""
from __future__ import annotations

import numpy as np

from . import assurance as _assurance
from .certificate import (
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    Certificate,
    sha256_signal,
)
from .conformal import ConformalCalibrator

# A conformal verdict is EMPIRICAL — statistical, valid only under exchangeability,
# an ESTIMATE and NEVER a proof. The caveat travels on every certified verdict so
# the tier is never silently read as something stronger than it is.
CONFORMAL_ASSURANCE = _assurance.EMPIRICAL
ASSURANCE_CAVEAT = (
    "EMPIRICAL: the conformal coverage guarantee is statistical (marginal, "
    "1 - alpha) and holds ONLY under exchangeability of calibration and serving "
    "data; cross-bearing / cross-machine deployment breaks it. It bounds the "
    "MARGINAL miscoverage RATE, not the correctness of any single prediction. "
    "This is an estimate, NOT a proof — it can never exceed the EMPIRICAL tier."
)


class Verifier:
    """Model-agnostic calibrated-trust facade.

    Parameters
    ----------
    alpha:
        Conformal miscoverage level; coverage target ``1 - alpha``.
    score:
        Conformal score function, ``"aps"`` (default) or ``"lac"``.
    class_names:
        Optional human-readable label names. If omitted, names default to the
        stringified class index ("0", "1", ...). Binding these names into the
        certificate hash is the red-team lesson: a forged display label breaks
        the hash.
    abstain_threshold:
        Minimum top probability for a non-ABSTAIN verdict (default 0.0 = never
        abstain on confidence alone; raise to require a confidence floor).
    rng_seed:
        Seed for the APS randomization tie-break.
    """

    def __init__(self, alpha: float = 0.10, *, score: str = "aps",
                 class_names: list[str] | None = None,
                 abstain_threshold: float = 0.0, rng_seed: int = 0) -> None:
        if not 0.0 <= abstain_threshold < 1.0:
            raise ValueError("abstain_threshold must be in [0, 1)")
        self.calibrator = ConformalCalibrator(alpha=alpha, score=score, rng_seed=rng_seed)
        self.class_names = list(class_names) if class_names is not None else None
        self.abstain_threshold = float(abstain_threshold)

    # ---- calibration ----------------------------------------------------- #

    def calibrate(self, probs_calib: np.ndarray, labels_calib: np.ndarray) -> Verifier:
        """Fit the conformal quantile on a held-out calibration set. Returns self."""
        self.calibrator.fit(probs_calib, labels_calib)
        if self.class_names is not None and self.calibrator.n_classes is not None:
            if len(self.class_names) != self.calibrator.n_classes:
                raise ValueError(
                    f"class_names has {len(self.class_names)} entries but calibration "
                    f"data has {self.calibrator.n_classes} classes")
        return self

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.qhat is not None

    @property
    def coverage_valid_under(self) -> str:
        return self.calibrator.coverage_valid_under

    # ---- certification --------------------------------------------------- #

    def _name(self, idx: int) -> str:
        if self.class_names is not None and 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)

    @property
    def assurance_caveat(self) -> str:
        """The exchangeability caveat that every conformal (EMPIRICAL) verdict carries."""
        return ASSURANCE_CAVEAT

    def certify(self, probs: np.ndarray, *, input_signal=None,
                model_id: str | None = None,
                key: str | bytes | None = None,
                seed: str | bytes | None = None,
                signer=None,
                scheme: str = "auto",
                ttl_seconds: int | None = None,
                key_id: str | None = None,
                now_iso: str | None = None,
                conformal_method: str | None = None,
                coverage_guarantee: str | None = None) -> Certificate:
        """Certify ONE sample's probability vector into a sealed :class:`Certificate`.

        ``probs`` is a 1-D probability vector (or a single-row 2-D array).
        ``input_signal`` (optional) is hashed into ``input_sha256`` to bind the
        certificate to its exact input.

        Signing (delegated to :meth:`Certificate.seal`):

        - ``seed`` (optional) — an Ed25519 signing seed; when given the
          certificate is signed asymmetrically (``scheme`` forced to
          ``"ed25519"``) and ships an embedded public key for third-party
          verification. Equivalent to setting env ``VERIFY_ED25519_SEED``.
        - ``key`` (optional) — overrides env ``VERIFY_HMAC_KEY`` for HMAC signing
          (kept for backward compatibility; passed straight to ``seal``).
        - ``scheme`` — ``"auto"`` (default precedence: Ed25519 seed > HMAC key >
          NONE), or force ``"ed25519"`` / ``"hmac"`` / ``"none"``.

        Validity window & identity (anti-replay, v2.6.0; delegated to
        :meth:`Certificate.seal`):

        - ``ttl_seconds`` (optional) — when given, the sealed certificate carries
          a validity window (``not_before = now``, ``valid_until = now +
          ttl_seconds``) and a fresh ``jti``. These fields are NOT hashed into
          ``content_hash`` (so the decision hash stays DETERMINISTIC), but the
          SIGNATURE covers them via the signing payload, so any tamper with the
          window invalidates the signature. :func:`verify_certificate` rejects an
          expired / not-yet-valid certificate.
        - ``key_id`` (optional) — an opaque id of the signing key, for rotation.
        - ``now_iso`` (optional) — pin "now" for DETERMINISTIC tests (never read
          the wall clock in a test).

        The verdict's ``assurance`` is fixed to EMPIRICAL: a conformal guarantee
        is statistical and never a proof. See :pyattr:`assurance_caveat`.
        """
        if not self.is_calibrated:
            raise RuntimeError("call calibrate() before certify()")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 2:
            if probs.shape[0] != 1:
                raise ValueError("certify() handles ONE sample; pass a 1-D vector "
                                 "or a single-row array")
            probs = probs[0]
        if probs.ndim != 1:
            raise ValueError("probs must be a 1-D probability vector")

        result = self.calibrator.predict(probs[None])
        cset = sorted(int(c) for c in result.sets[0])
        point = int(np.argmax(probs))
        top_p = float(np.max(probs))

        if top_p < self.abstain_threshold:
            verdict = VERDICT_ABSTAIN
        elif len(cset) == 1:
            verdict = VERDICT_CERTIFIED
        else:
            verdict = VERDICT_REVIEW

        input_sha = sha256_signal(input_signal) if input_signal is not None else None
        cert = Certificate(
            predicted_label=point,
            predicted_name=self._name(point),
            conformal_set=cset,
            conformal_set_names=[self._name(c) for c in cset],
            verdict=verdict,
            alpha=float(self.calibrator.alpha),
            qhat=None if self.calibrator.qhat is None else float(self.calibrator.qhat),
            input_sha256=input_sha,
            model_id=model_id,
            assurance=CONFORMAL_ASSURANCE,   # conformal => EMPIRICAL, never proven
            # Optional conditional-conformal claim (v2.9.0). Bound into content_hash
            # only when set (see Certificate.canonical_payload) — a caller using a
            # class-conditional / Mondrian / weighted / ACI calibrator stamps the
            # method + guarantee here so the served verdict carries (tamper-evidently)
            # WHICH coverage guarantee it holds. Default None = marginal/unstated.
            conformal_method=conformal_method,
            coverage_guarantee=coverage_guarantee,
        )
        # An explicit seed forces the asymmetric (Ed25519) path; otherwise the seal
        # resolves by precedence (Ed25519 seed env > HMAC key env > NONE). The
        # validity-window / identity fields (ttl_seconds, key_id, now_iso) are
        # additive: passing None leaves the cert timeless (backward-compatible).
        # A pluggable Signer (e.g. a KMS/HSM-backed ExternalSigner) takes precedence:
        # the private key never enters this process. Otherwise fall back to the
        # in-process seed/HMAC path (dev / single-tenant).
        if signer is not None:
            return cert.seal_with(signer, ttl_seconds=ttl_seconds,
                                  key_id=key_id, now_iso=now_iso)
        if seed is not None:
            return cert.seal(seed, scheme="ed25519", ttl_seconds=ttl_seconds,
                             key_id=key_id, now_iso=now_iso)
        return cert.seal(key, scheme=scheme, ttl_seconds=ttl_seconds,
                         key_id=key_id, now_iso=now_iso)


# --------------------------------------------------------------------------- #
# Composition algebra — certificates compose, and the weakest link governs.
# --------------------------------------------------------------------------- #

# Verdict ordering for AND-composition: a composed system is only as safe to act
# on as its LEAST decisive component. CERTIFIED (act) is strongest; ABSTAIN (do
# not act) is weakest; REVIEW (human-in-the-loop) sits between. So a single
# ABSTAIN drags the whole AND to ABSTAIN, a single REVIEW to REVIEW.
_VERDICT_RANK = {VERDICT_ABSTAIN: 0, VERDICT_REVIEW: 1, VERDICT_CERTIFIED: 2}


def compose_certificates(certs, op: str = "and") -> dict:
    """Compose several certificates into ONE honest system verdict (sound, fail-safe).

    Ported from the substrate_core kernel's ``compose_and`` / ``weakest``: the
    weakest link governs. Returns a plain dict (NOT a sealed Certificate — a
    composition is a derived judgement, re-signing is the caller's choice)::

        {"verdict": str,            # composed CERTIFIED | REVIEW | ABSTAIN
         "assurance": str,          # the WEAKEST tier across the inputs
         "op": str,                 # "and" | "or"
         "n": int,                  # number of components
         "components": [...],       # per-cert (verdict, assurance)
         "assurance_caveat": str,   # the system can never exceed its weakest tier
         "detail": str}

    Semantics:

    - ``op="and"`` (a system holds IFF every part holds): the composed verdict is
      the WEAKEST component verdict — a single REVIEW propagates REVIEW, a single
      ABSTAIN propagates ABSTAIN. CERTIFIED only if ALL are CERTIFIED.
    - ``op="or"`` (any one path suffices): the composed verdict is the STRONGEST
      component verdict.

    In BOTH cases the system ``assurance`` is the WEAKEST tier present (the
    anti-overclaim invariant: a composition can never be stronger than its
    weakest evidence). Since a conformal certificate is EMPIRICAL, a system built
    only from conformal certificates is at most EMPIRICAL — never ``proven``.

    An empty list yields ABSTAIN / NONE (nothing composed -> no evidence).
    """
    if op not in ("and", "or"):
        raise ValueError(f"unknown op {op!r}; expected 'and' or 'or'")

    rows = []
    verdicts: list[str] = []
    assurances: list[str] = []
    for c in certs:
        d = c.as_dict() if hasattr(c, "as_dict") else dict(c)
        v = str(d.get("verdict", VERDICT_ABSTAIN))
        a = str(d.get("assurance", _assurance.NONE))
        verdicts.append(v)
        assurances.append(a)
        rows.append({"verdict": v, "assurance": a})

    if not verdicts:
        return {
            "verdict": VERDICT_ABSTAIN,
            "assurance": _assurance.NONE,
            "op": op,
            "n": 0,
            "components": [],
            "assurance_caveat": "empty composition: nothing to compose -> no evidence",
            "detail": "empty composition",
        }

    # Weakest link on BOTH axes for AND; strongest verdict for OR. Unknown verdicts
    # rank as ABSTAIN (fail-safe), mirroring assurance.weakest for unknown tiers.
    if op == "and":
        verdict = min(verdicts, key=lambda v: _VERDICT_RANK.get(v, 0))
    else:
        verdict = max(verdicts, key=lambda v: _VERDICT_RANK.get(v, 0))
    sys_assurance = _assurance.weakest(assurances)  # always the weakest tier

    return {
        "verdict": verdict,
        "assurance": sys_assurance,
        "op": op,
        "n": len(verdicts),
        "components": rows,
        "assurance_caveat": (
            f"system assurance = weakest link = {sys_assurance}; "
            f"{_assurance.describe(sys_assurance)} "
            "A composition can never exceed its weakest component; conformal "
            "evidence is EMPIRICAL, so a conformal-only system is never 'proven'."),
        "detail": (
            f"compose_{op}: {len(verdicts)} components -> verdict {verdict}, "
            f"assurance {sys_assurance} (weakest link)"),
    }
