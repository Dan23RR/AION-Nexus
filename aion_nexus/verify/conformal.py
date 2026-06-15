"""Split (inductive) conformal prediction — model-agnostic calibrated trust.

Turns ANY classifier's probability vectors into PREDICTION SETS with a
distribution-free, finite-sample coverage guarantee::

    P(y_true in predict_set(probs)) >= 1 - alpha

The guarantee is *marginal* and holds ONLY under **exchangeability** of the
calibration and test data (e.g. both drawn i.i.d. from the same distribution).
It is the calibrated-trust layer of the Substrate / Verifier thesis: the model
is allowed to be wrong, but it must be *honest about when it does not know* — a
singleton set is a confident certified label; a larger set means ambiguity.

Exchangeability caveat (read before trusting coverage)
------------------------------------------------------
In bearing predictive maintenance the calibration data and the deployment data
are typically NOT exchangeable: calibrating on bearing A and serving on bearing
B (cross-bearing), or on a different machine (cross-machine), breaks
exchangeability. When that happens the 1 - alpha coverage is NO LONGER
guaranteed — the sets may under-cover. The calibrator records the assumption it
needs in :pyattr:`coverage_valid_under`; the caller is responsible for ensuring
calib/test are exchangeable (e.g. calibrate per-bearing) or for treating the
guarantee as advisory and monitoring empirical coverage.

Score functions
---------------
- **APS** (Romano, Sesia & Candes 2020): cumulative-probability score; better
  class-conditional coverage, slightly larger sets. Default.
- **LAC** (Sadinle, Lei & Wasserman 2019): score = 1 - p[y]; smallest sets,
  marginal coverage.

Never emits an empty set: if a score rule would yield one, it falls back to the
top-1 label, so downstream consumers always get an actionable prediction.

Pure numpy; no torch dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ScoreFn = Literal["aps", "lac"]


@dataclass
class ConformalResult:
    """Per-sample output of :meth:`ConformalCalibrator.predict`."""

    sets: list[list[int]]        # predicted label set per sample (never empty)
    abstain: np.ndarray          # bool[n]: True if set size != 1
    set_sizes: np.ndarray        # int[n]
    point_pred: np.ndarray       # int[n]: argmax (the action if not abstaining)


@dataclass
class ConformalCalibrator:
    """Split-conformal wrapper around precomputed class probabilities.

    Fit ``qhat`` on a held-out calibration set (probabilities + true labels),
    then turn fresh probability vectors into coverage-controlled prediction sets.

    Parameters
    ----------
    alpha:
        Target miscoverage; coverage target is ``1 - alpha``. Must be in (0, 1).
    score:
        ``"aps"`` (default) or ``"lac"``.
    rng_seed:
        Seed for the APS randomization tie-break (deterministic given the seed).
    """

    alpha: float = 0.10
    score: ScoreFn = "aps"
    rng_seed: int = 0
    qhat: float | None = field(default=None, init=False)
    n_cal: int = field(default=0, init=False)
    n_classes: int | None = field(default=None, init=False)
    # The assumption under which the 1 - alpha coverage guarantee is valid.
    coverage_valid_under: str = field(
        default="exchangeable calibration/test (i.i.d. same distribution); "
        "cross-bearing or cross-machine deployment BREAKS exchangeability and "
        "VOIDS the marginal coverage guarantee",
        init=False,
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.score not in ("aps", "lac"):
            raise ValueError("score must be 'aps' or 'lac'")
        self._rng = np.random.default_rng(self.rng_seed)

    # ---- scores ---------------------------------------------------------- #

    @staticmethod
    def _lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return 1.0 - probs[np.arange(len(labels)), labels]

    def _aps_scores(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        order = np.argsort(-probs, axis=1)
        sorted_p = np.take_along_axis(probs, order, axis=1)
        cum = np.cumsum(sorted_p, axis=1)
        ranks = np.argmax(order == labels[:, None], axis=1)
        rows = np.arange(len(labels))
        u = self._rng.uniform(size=len(labels))           # randomized APS tie-break
        return cum[rows, ranks] - u * sorted_p[rows, ranks]

    # ---- calibrate / predict -------------------------------------------- #

    def fit(self, probs_calib: np.ndarray, labels_calib: np.ndarray) -> float:
        """Calibrate ``qhat`` from calibration probabilities + true labels.

        Returns the computed ``qhat``. Uses the finite-sample-corrected quantile
        level ``ceil((n+1)(1-alpha)) / n`` so the coverage guarantee holds for
        the actual (finite) calibration size, not just asymptotically.
        """
        probs_calib = np.asarray(probs_calib, dtype=np.float64)
        labels_calib = np.asarray(labels_calib, dtype=int)
        if probs_calib.ndim != 2:
            raise ValueError("probs_calib must be a 2-D array [n_samples, n_classes]")
        n, k = probs_calib.shape
        if n == 0:
            raise ValueError("cannot calibrate on an empty calibration set")
        if labels_calib.shape != (n,):
            raise ValueError("labels_calib must have shape [n_samples]")
        if labels_calib.min() < 0 or labels_calib.max() >= k:
            raise ValueError("labels_calib contains out-of-range class indices")
        scores = (
            self._aps_scores(probs_calib, labels_calib)
            if self.score == "aps"
            else self._lac_scores(probs_calib, labels_calib)
        )
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self.qhat = float(np.quantile(scores, level, method="higher"))
        self.n_cal = int(n)
        self.n_classes = int(k)
        return self.qhat

    # Alias: some callers prefer scikit-style .calibrate(); keep both.
    calibrate = fit

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        """Return the prediction set (list of class indices) for each row.

        Never returns an empty set — falls back to top-1 if a score rule would.
        """
        return self.predict(probs).sets

    def predict(self, probs: np.ndarray) -> ConformalResult:
        """Full per-sample result (sets, abstain flags, sizes, point pred)."""
        if self.qhat is None:
            raise RuntimeError("call fit()/calibrate() before predict()")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.ndim != 2:
            raise ValueError("probs must be 1-D [n_classes] or 2-D [n_samples, n_classes]")
        n, k = probs.shape
        if self.n_classes is not None and k != self.n_classes:
            raise ValueError(
                f"probs has {k} classes but calibrated on {self.n_classes}")
        sets: list[list[int]] = []
        if self.score == "lac":
            keep = probs >= (1.0 - self.qhat)
            for i in range(n):
                s = [int(c) for c in np.where(keep[i])[0]]
                if not s:                                  # never empty: top-1 fallback
                    s = [int(np.argmax(probs[i]))]
                sets.append(s)
        else:  # APS
            order = np.argsort(-probs, axis=1)
            sorted_p = np.take_along_axis(probs, order, axis=1)
            cum = np.cumsum(sorted_p, axis=1)
            for i in range(n):
                stop = int(np.searchsorted(cum[i], self.qhat) + 1)
                stop = max(1, min(stop, k))                # >=1 (never empty), <=k
                sets.append([int(c) for c in order[i, :stop]])
        set_sizes = np.array([len(s) for s in sets], dtype=int)
        abstain = set_sizes != 1
        point = probs.argmax(axis=1)
        return ConformalResult(sets=sets, abstain=abstain, set_sizes=set_sizes,
                               point_pred=point)

    # ---- diagnostics ----------------------------------------------------- #

    @staticmethod
    def empirical_coverage(result: ConformalResult, labels: np.ndarray) -> float:
        """Fraction of samples whose true label is in the prediction set."""
        labels = np.asarray(labels, dtype=int)
        hits = [labels[i] in result.sets[i] for i in range(len(labels))]
        return float(np.mean(hits)) if hits else 0.0


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable row-wise softmax (helper for logits-only callers)."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
