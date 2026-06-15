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

from .certificate import (
    VERDICT_ABSTAIN,
    VERDICT_CERTIFIED,
    VERDICT_REVIEW,
    Certificate,
    sha256_signal,
)
from .conformal import ConformalCalibrator


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

    def certify(self, probs: np.ndarray, *, input_signal=None,
                model_id: str | None = None,
                key: str | bytes | None = None) -> Certificate:
        """Certify ONE sample's probability vector into a sealed :class:`Certificate`.

        ``probs`` is a 1-D probability vector (or a single-row 2-D array).
        ``input_signal`` (optional) is hashed into ``input_sha256`` to bind the
        certificate to its exact input. ``key`` (optional) overrides the env
        ``VERIFY_HMAC_KEY`` for signing.
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
        )
        return cert.seal(key)
