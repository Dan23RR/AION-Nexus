"""Conditional conformal prediction — past the marginal guarantee.

The base :class:`~aion_nexus.verify.conformal.ConformalCalibrator` gives a
*marginal* coverage guarantee valid only under **exchangeability** of calibration
and serving data. The red-team's kill-shot #5 was exactly this: marginal coverage
is the floor, and cross-bearing / cross-machine deployment (non-exchangeable)
VOIDS it. This module turns that weakness into the claimable frontier — four
calibrators, each with a STRONGER or DIFFERENT guarantee, and each HONEST about
the assumption it needs:

1. :class:`ClassConditionalConformalCalibrator` — **per-class** coverage
   ``P(Y ∈ C(X) | Y = c) ≥ 1 − α`` for every class c (Sadinle, Lei & Wasserman
   2019). Catches the failure where marginal coverage is met on average while a
   rare-but-critical class is systematically under-covered.

2. :class:`MondrianConformalCalibrator` — **per-group** coverage
   ``P(Y ∈ C(X) | group(X) = g) ≥ 1 − α`` for every covariate-defined group g
   (e.g. per bearing / per operating regime). The group must be computable from
   the input at test time (NOT the label).

3. :class:`WeightedConformalCalibrator` — coverage under **covariate shift**
   (Tibshirani, Foygel Barber, Candès & Ramdas 2019). When calibration and test
   differ by a known/estimated likelihood ratio w(x) = dP_test/dP_cal, weighted
   quantiles recover the guarantee. With w ≡ 1 it reduces to standard split CP.

4. :class:`AdaptiveConformalGate` — **online** long-run coverage on a
   non-stationary stream WITHOUT exchangeability (Gibbs & Candès 2021, ACI). The
   realised miscoverage frequency converges to α for ANY (even adversarial) drift,
   given the realised labels as feedback.

HONESTY (workspace 6.31). None of these is a proof of correctness, and none
escapes its own assumption: class/group-conditional CP still needs *within*-class
/ *within*-group exchangeability; weighted CP is only as good as the weight
estimate; ACI guarantees a LONG-RUN AVERAGE, not per-step, coverage. Every
calibrator states its assumption in ``coverage_valid_under`` and its guarantee in
``guarantee``. The guarantees are not asserted in prose alone — they are PROVEN by
empirical-coverage simulation in ``tests/test_conformal_advanced.py``.

Pure numpy; no torch dependency. Scores reuse the base module's LAC / APS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conformal import ConformalCalibrator

# Tier label re-exported for callers that record which conformal method a verdict
# used; all conformal evidence is EMPIRICAL (statistical, never a proof).
CONDITIONAL_ASSURANCE = "empirical"


def finite_sample_level(n: int, alpha: float) -> float:
    """The finite-sample-corrected quantile level ``ceil((n+1)(1−α)) / n``.

    Capped at 1.0. This is the level that makes split-conformal coverage hold for
    the ACTUAL (finite) calibration size n, not just asymptotically. A group/class
    so small that the level would exceed 1 cannot give a non-trivial guarantee —
    the level saturates at 1.0 and the threshold becomes the max score (the set
    conservatively includes the class/everything). Callers should surface small n.
    """
    if n <= 0:
        return 1.0
    return float(min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0))


def _lac_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """LAC nonconformity score s_i = 1 − p_i[y_i]."""
    return 1.0 - probs[np.arange(len(labels)), labels]


# --------------------------------------------------------------------------- #
# 1. Class-conditional (classwise) conformal — per-class coverage
# --------------------------------------------------------------------------- #

@dataclass
class ClassConditionalConformalCalibrator:
    """Classwise LAC conformal: a SEPARATE threshold per class.

    Guarantee (finite-sample, under within-class exchangeability)::

        P(Y ∈ C(X) | Y = c) ≥ 1 − α   for every class c.

    A class c is included in C(x) iff ``1 − p_x[c] ≤ qhat_c`` i.e.
    ``p_x[c] ≥ 1 − qhat_c``, where ``qhat_c`` is the finite-sample (1−α) quantile
    of the calibration LAC scores OF CLASS c. Never empty (top-1 fallback).

    This is strictly stronger than marginal coverage where it matters most: a rare
    failure class cannot be silently under-covered to prop up the average. The cost
    is larger sets and a per-class calibration-size requirement — a class with few
    calibration points gets a conservative (often trivially full) threshold, which
    is surfaced in :pyattr:`small_classes`.
    """

    alpha: float = 0.10
    qhat_per_class: dict[int, float] = field(default_factory=dict, init=False)
    n_per_class: dict[int, int] = field(default_factory=dict, init=False)
    n_classes: int | None = field(default=None, init=False)
    small_classes: list[int] = field(default_factory=list, init=False)
    coverage_valid_under: str = field(
        default="within-class exchangeability (calibration and serving points of a "
        "given class are exchangeable); gives per-class coverage even when the "
        "overall marginal mix shifts",
        init=False)
    guarantee: str = field(
        default="P(Y in C(X) | Y=c) >= 1-alpha for each class c (class-conditional)",
        init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def fit(self, probs_calib: np.ndarray, labels_calib: np.ndarray
            ) -> ClassConditionalConformalCalibrator:
        probs = np.asarray(probs_calib, dtype=np.float64)
        labels = np.asarray(labels_calib, dtype=int)
        if probs.ndim != 2:
            raise ValueError("probs_calib must be 2-D [n_samples, n_classes]")
        n, k = probs.shape
        if labels.shape != (n,):
            raise ValueError("labels_calib must have shape [n_samples]")
        if n == 0:
            raise ValueError("cannot calibrate on an empty calibration set")
        if labels.min() < 0 or labels.max() >= k:
            raise ValueError("labels_calib contains out-of-range class indices")
        self.n_classes = int(k)
        self.qhat_per_class.clear()
        self.n_per_class.clear()
        self.small_classes.clear()
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            self.n_per_class[c] = n_c
            if n_c == 0:
                # No calibration data for class c: cannot bound its score. Fail
                # SAFE — include c whenever it is even slightly probable (qhat=1.0
                # => p_c >= 0). Flagged so the caller knows this class is uncovered.
                self.qhat_per_class[c] = 1.0
                self.small_classes.append(c)
                continue
            scores_c = 1.0 - probs[mask, c]
            level = finite_sample_level(n_c, self.alpha)
            if level >= 1.0:
                self.small_classes.append(c)  # too few points for a tight bound
            self.qhat_per_class[c] = float(np.quantile(scores_c, level, method="higher"))
        return self

    def predict_set(self, probs: np.ndarray) -> list[list[int]]:
        if self.n_classes is None:
            raise RuntimeError("call fit() before predict_set()")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} classes but calibrated on {self.n_classes}")
        sets: list[list[int]] = []
        for row in probs:
            s = [c for c in range(self.n_classes)
                 if (1.0 - row[c]) <= self.qhat_per_class[c]]
            if not s:                                   # never empty: top-1 fallback
                s = [int(np.argmax(row))]
            sets.append(s)
        return sets


# --------------------------------------------------------------------------- #
# 2. Mondrian (group-conditional) conformal — per-group coverage
# --------------------------------------------------------------------------- #

@dataclass
class MondrianConformalCalibrator:
    """Per-group split conformal: a SEPARATE base calibrator per covariate group.

    Guarantee (finite-sample, under within-group exchangeability)::

        P(Y ∈ C(X) | group(X) = g) ≥ 1 − α   for every group g seen in calibration.

    The ``group`` is any partition of the inputs computable AT TEST TIME (per
    bearing, per operating regime, per sensor) — NOT the label. This is the honest
    answer to the cross-bearing exchangeability break: calibrate per bearing and
    each bearing gets its own valid guarantee. A test point whose group was NOT in
    calibration has NO per-group guarantee; it falls back to a pooled global qhat,
    and ``fell_back_groups`` records that it happened.
    """

    alpha: float = 0.10
    score: str = "aps"
    rng_seed: int = 0
    _per_group: dict = field(default_factory=dict, init=False)
    _global: ConformalCalibrator | None = field(default=None, init=False)
    n_per_group: dict = field(default_factory=dict, init=False)
    n_classes: int | None = field(default=None, init=False)
    fell_back_groups: set = field(default_factory=set, init=False)
    coverage_valid_under: str = field(
        default="within-group exchangeability (calibration and serving points of a "
        "given group are exchangeable); per-group coverage holds even across groups "
        "that are NOT mutually exchangeable (e.g. different bearings)",
        init=False)
    guarantee: str = field(
        default="P(Y in C(X) | group(X)=g) >= 1-alpha for each calibrated group g",
        init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.score not in ("aps", "lac"):
            raise ValueError("score must be 'aps' or 'lac'")

    def fit(self, probs_calib: np.ndarray, labels_calib: np.ndarray,
            groups: np.ndarray) -> MondrianConformalCalibrator:
        probs = np.asarray(probs_calib, dtype=np.float64)
        labels = np.asarray(labels_calib, dtype=int)
        groups = np.asarray(groups)
        if probs.ndim != 2:
            raise ValueError("probs_calib must be 2-D [n_samples, n_classes]")
        n, k = probs.shape
        if labels.shape != (n,) or groups.shape != (n,):
            raise ValueError("labels and groups must each have shape [n_samples]")
        self.n_classes = int(k)
        self._per_group.clear()
        self.n_per_group.clear()
        # A pooled global calibrator for unseen-group fallback (no per-group claim).
        # NB: base ConformalCalibrator.fit() returns qhat (a float), not self, so we
        # construct, fit, then keep the OBJECT.
        glob = ConformalCalibrator(alpha=self.alpha, score=self.score, rng_seed=self.rng_seed)
        glob.fit(probs, labels)
        self._global = glob
        for g in np.unique(groups):
            mask = groups == g
            cal = ConformalCalibrator(alpha=self.alpha, score=self.score,
                                      rng_seed=self.rng_seed)
            cal.fit(probs[mask], labels[mask])
            self._per_group[_key(g)] = cal
            self.n_per_group[_key(g)] = int(mask.sum())
        return self

    def predict_set(self, probs: np.ndarray, groups: np.ndarray) -> list[list[int]]:
        if self._global is None:
            raise RuntimeError("call fit() before predict_set()")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        groups = np.atleast_1d(np.asarray(groups))
        if groups.shape[0] != probs.shape[0]:
            raise ValueError("groups must have one entry per probs row")
        sets: list[list[int]] = []
        for row, g in zip(probs, groups, strict=False):
            cal = self._per_group.get(_key(g))
            if cal is None:
                self.fell_back_groups.add(_key(g))   # unseen group: no per-group claim
                cal = self._global
            sets.append(cal.predict(row[None]).sets[0])
        return sets


def _key(g) -> object:
    """Hashable, stable key for a group value (numpy scalars -> python scalars)."""
    return g.item() if isinstance(g, np.generic) else g


# --------------------------------------------------------------------------- #
# 3. Weighted conformal — coverage under covariate shift
# --------------------------------------------------------------------------- #

@dataclass
class WeightedConformalCalibrator:
    """Weighted split conformal for covariate shift (Tibshirani et al. 2019).

    When the test distribution differs from calibration by a covariate-shift
    likelihood ratio ``w(x) = dP_test/dP_cal`` (up to a constant), the prediction
    set uses WEIGHTED quantiles of the calibration scores::

        C(x) = { y : 1 − p_x[y] ≤ Q_{1−α}( Σ_i p_i^w δ_{s_i} + p_{n+1}^w δ_{+∞} ) }

    with ``p_i^w = w(x_i) / (Σ_j w(x_j) + w(x))`` and ``p_{n+1}^w`` the test point's
    own normalised weight. Guarantee (finite-sample)::

        P(Y ∈ C(X)) ≥ 1 − α   under the covariate-shift model, IF the weights equal
        the true likelihood ratio.

    HONESTY: this does NOT estimate the weights — the caller supplies them (e.g.
    from a domain/probabilistic classifier). With ``w ≡ 1`` it reduces EXACTLY to
    standard split conformal (a useful sanity check, asserted in the tests). A
    wrong weight model degrades coverage gracefully toward the unweighted result;
    it is an approximation, never a guarantee beyond the weight quality.
    """

    alpha: float = 0.10
    _scores_sorted: np.ndarray | None = field(default=None, init=False)
    _w_sorted: np.ndarray | None = field(default=None, init=False)
    n_classes: int | None = field(default=None, init=False)
    coverage_valid_under: str = field(
        default="covariate shift with a KNOWN/estimated likelihood ratio "
        "w(x)=dP_test/dP_cal; exact only if the weights are correct (it is an "
        "approximation otherwise, reducing to unweighted CP at w=1)",
        init=False)
    guarantee: str = field(
        default="P(Y in C(X)) >= 1-alpha under covariate shift given correct weights",
        init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def fit(self, probs_calib: np.ndarray, labels_calib: np.ndarray,
            weight_calib: np.ndarray | None = None) -> WeightedConformalCalibrator:
        probs = np.asarray(probs_calib, dtype=np.float64)
        labels = np.asarray(labels_calib, dtype=int)
        if probs.ndim != 2:
            raise ValueError("probs_calib must be 2-D [n_samples, n_classes]")
        n, k = probs.shape
        if labels.shape != (n,):
            raise ValueError("labels_calib must have shape [n_samples]")
        if n == 0:
            raise ValueError("cannot calibrate on an empty calibration set")
        self.n_classes = int(k)
        scores = _lac_scores(probs, labels)
        if weight_calib is None:
            w = np.ones(n, dtype=np.float64)
        else:
            w = np.asarray(weight_calib, dtype=np.float64)
            if w.shape != (n,):
                raise ValueError("weight_calib must have shape [n_samples]")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
        # Pre-sort once so each per-test-point weighted quantile is O(n), not O(n log n).
        order = np.argsort(scores, kind="mergesort")
        self._scores_sorted = scores[order]
        self._w_sorted = w[order]
        return self

    def _weighted_qhat(self, w_test: float) -> float:
        """Weighted (1−α) quantile of calibration scores with a +∞ atom at w_test."""
        s_sorted = self._scores_sorted
        total = float(self._w_sorted.sum() + w_test)
        if total <= 0:
            return float("inf")
        cum = np.cumsum(self._w_sorted) / total
        target = 1.0 - self.alpha
        # If the +∞ atom carries more mass than α, the (1−α) quantile is +∞ ->
        # the set conservatively includes every label (full coverage). This is the
        # honest behaviour when the test point is heavily up-weighted vs calibration.
        if target > cum[-1] + 1e-12:
            return float("inf")
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(idx, len(s_sorted) - 1)
        return float(s_sorted[idx])

    def predict_set(self, probs: np.ndarray, weight_test=1.0) -> list[list[int]]:
        if self._scores_sorted is None:
            raise RuntimeError("call fit() before predict_set()")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs[None, :]
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} classes but calibrated on {self.n_classes}")
        n = probs.shape[0]
        w_test = np.broadcast_to(np.asarray(weight_test, dtype=np.float64), (n,))
        sets: list[list[int]] = []
        for row, wt in zip(probs, w_test, strict=False):
            if wt < 0:
                raise ValueError("weight_test must be non-negative")
            qhat = self._weighted_qhat(float(wt))
            s = [c for c in range(self.n_classes) if (1.0 - row[c]) <= qhat]
            if not s:                                   # never empty: top-1 fallback
                s = [int(np.argmax(row))]
            sets.append(s)
        return sets


# --------------------------------------------------------------------------- #
# 4. Adaptive Conformal Inference (online, non-stationary)
# --------------------------------------------------------------------------- #

@dataclass
class AdaptiveConformalGate:
    """Online conformal with a self-adjusting level (Gibbs & Candès 2021, ACI).

    For a NON-STATIONARY stream the marginal guarantee is hopeless, but ACI gives a
    LONG-RUN coverage guarantee with NO exchangeability assumption: maintain a
    running level ``α_t`` and, after seeing whether the realised label fell in the
    set, update::

        α_{t+1} = α_t + γ · (α − err_t),     err_t = 1[ y_t ∉ C_t ]

    Then for any (even adversarial) sequence::

        | (1/T) Σ_t err_t − α |  ≤  (1 + γ) / (γ · T)  → 0,

    i.e. the realised miscoverage frequency converges to α. The set at each step is
    formed from a fixed calibration score set at the current adaptive level.

    HONESTY: this is a LONG-RUN AVERAGE guarantee, not a per-step one — an
    individual step may over- or under-cover; and it needs the realised label as
    feedback (:meth:`step` with ``true_label``) to adapt. Without feedback it is a
    fixed-level gate. The miscoverage it drives to α is the empirical FREQUENCY.
    """

    alpha: float = 0.10
    gamma: float = 0.05
    _cal_scores: np.ndarray | None = field(default=None, init=False)
    n_classes: int | None = field(default=None, init=False)
    alpha_t: float = field(default=0.10, init=False)
    n_steps: int = field(default=0, init=False)
    n_errors: int = field(default=0, init=False)
    coverage_valid_under: str = field(
        default="none — long-run coverage holds for ARBITRARY (even adversarial) "
        "distribution drift, given the realised labels as online feedback",
        init=False)
    guarantee: str = field(
        default="|empirical miscoverage frequency - alpha| -> 0 as T grows (ACI, "
        "long-run average; NOT per-step)",
        init=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.gamma <= 0:
            raise ValueError("gamma (learning rate) must be > 0")
        self.alpha_t = self.alpha

    def fit(self, probs_calib: np.ndarray, labels_calib: np.ndarray
            ) -> AdaptiveConformalGate:
        probs = np.asarray(probs_calib, dtype=np.float64)
        labels = np.asarray(labels_calib, dtype=int)
        if probs.ndim != 2:
            raise ValueError("probs_calib must be 2-D [n_samples, n_classes]")
        n, k = probs.shape
        if labels.shape != (n,):
            raise ValueError("labels_calib must have shape [n_samples]")
        if n == 0:
            raise ValueError("cannot calibrate on an empty calibration set")
        self.n_classes = int(k)
        self._cal_scores = np.sort(_lac_scores(probs, labels))
        return self

    @property
    def empirical_miscoverage(self) -> float:
        return self.n_errors / self.n_steps if self.n_steps else 0.0

    def _set_for_level(self, row: np.ndarray) -> list[int]:
        level = float(np.clip(1.0 - self.alpha_t, 0.0, 1.0))
        if level <= 0.0:
            return [int(np.argmax(row))]                # α_t>=1: cover nothing -> top-1
        if level >= 1.0:
            return list(range(self.n_classes))          # α_t<=0: cover everything
        qhat = float(np.quantile(self._cal_scores, level, method="higher"))
        s = [c for c in range(self.n_classes) if (1.0 - row[c]) <= qhat]
        return s if s else [int(np.argmax(row))]

    def step(self, probs: np.ndarray, true_label: int | None = None) -> list[int]:
        """Emit the set for one streaming point and (if a label is given) adapt α_t.

        Returns the prediction set. Pass ``true_label`` to feed back the realised
        outcome — that is what drives the long-run coverage. Omit it to use the
        gate read-only at the current level (no adaptation).
        """
        if self._cal_scores is None:
            raise RuntimeError("call fit() before step()")
        row = np.asarray(probs, dtype=np.float64).reshape(-1)
        if row.shape[0] != self.n_classes:
            raise ValueError(
                f"probs has {row.shape[0]} classes but calibrated on {self.n_classes}")
        pred_set = self._set_for_level(row)
        if true_label is not None:
            err = 0 if int(true_label) in pred_set else 1
            self.n_steps += 1
            self.n_errors += err
            # ACI update; clip α_t to [0, 1] so the level stays valid.
            self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err),
                                         0.0, 1.0))
        return pred_set


# --------------------------------------------------------------------------- #
# Deploy-time covariate-shift weight estimation (makes weighted CP usable
# without oracle weights — estimate dP_target/dP_cal from UNLABELED target data)
# --------------------------------------------------------------------------- #

def _logreg_fit(x: np.ndarray, y: np.ndarray, *, l2: float, epochs: int,
                lr: float) -> tuple[np.ndarray, float]:
    """Tiny L2-regularised logistic regression (full-batch GD, pure numpy).

    Used only to estimate a density ratio by classification — no sklearn / torch
    dependency, keeping this module pure-numpy like the rest of the verify layer.
    """
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        z = np.clip(x @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y
        gw = x.T @ err / n + l2 * w / n
        gb = float(err.mean())
        w -= lr * gw
        b -= lr * gb
    return w, b


def estimate_covariate_shift_weights(
        cal_features: np.ndarray, target_features: np.ndarray, *,
        l2: float = 1.0, epochs: int = 400, lr: float = 0.5,
        weight_clip: tuple[float, float] = (1e-2, 1e2)):
    """Estimate the covariate-shift likelihood ratio ``w(x) = dP_target/dP_cal``
    from UNLABELED feature samples, by probabilistic classification.

    The deploy reality the base :class:`WeightedConformalCalibrator` leaves open: it
    needs the weights, but at install time you do NOT know them — you only have a
    pile of unlabeled windows from the new (target) machine. This estimates them the
    standard way (Bickel et al. 2007 / Sugiyama density-ratio-by-classification):
    label calibration features 0 and target features 1, fit a classifier, and read
    the density ratio off its odds, ``w(x) ∝ p(target|x) / (1 - p(target|x))`` (the
    constant class-prior factor cancels in the weighted-CP normalisation). Weights
    are clipped to ``weight_clip`` so a few extreme points can't destabilise the
    weighted quantile.

    ``cal_features`` / ``target_features`` are ``[n, d]`` arrays — use the model's
    penultimate embeddings, hand-crafted signal features, or the RPM-invariant
    physics order-SNR features (:func:`aion_nexus.physics.fault_order_energy`),
    which are exactly the condition-aware features a shift shows up in.

    Returns ``(weight_calib, weight_fn)``: ``weight_calib`` are the weights at the
    calibration points (feed to ``WeightedConformalCalibrator.fit(weight_calib=...)``),
    and ``weight_fn(features)`` returns the weights for new test points (feed to
    ``predict_set(weight_test=...)``).

    HONESTY (6.31): this is an APPROXIMATION. The recovered coverage is only as good
    as (a) the features actually capturing the shift and (b) the estimator fitting
    it; with no shift the weights collapse to ~uniform and it reduces to standard
    split conformal. It does NOT manufacture a guarantee the data cannot support.
    """
    cal_x = np.atleast_2d(np.asarray(cal_features, dtype=np.float64))
    tgt_x = np.atleast_2d(np.asarray(target_features, dtype=np.float64))
    if cal_x.ndim != 2 or tgt_x.ndim != 2 or cal_x.shape[1] != tgt_x.shape[1]:
        raise ValueError("cal_features and target_features must be [n, d] with equal d")
    x = np.vstack([cal_x, tgt_x])
    y = np.concatenate([np.zeros(len(cal_x)), np.ones(len(tgt_x))])
    mu = x.mean(axis=0)
    sd = x.std(axis=0) + 1e-8
    w, b = _logreg_fit((x - mu) / sd, y, l2=l2, epochs=epochs, lr=lr)
    lo, hi = weight_clip

    def weight_fn(features: np.ndarray) -> np.ndarray:
        xf = np.atleast_2d(np.asarray(features, dtype=np.float64))
        z = np.clip(((xf - mu) / sd) @ w + b, -30.0, 30.0)
        return np.clip(np.exp(z), lo, hi)            # odds = p/(1-p) = exp(logit)

    return weight_fn(cal_x), weight_fn


def deploy_weighted_calibrator(probs_cal: np.ndarray, labels_cal: np.ndarray,
                               cal_features: np.ndarray, target_features: np.ndarray,
                               *, alpha: float = 0.10, **weight_kwargs):
    """One call: estimate covariate-shift weights from unlabeled target features,
    then return a :class:`WeightedConformalCalibrator` fitted with them + the
    ``weight_fn`` to weight test points.

    Usage at deploy::

        cal, weight_fn = deploy_weighted_calibrator(
            probs_cal, labels_cal, cal_embeddings, target_embeddings, alpha=0.1)
        sets = cal.predict_set(probs_test, weight_test=weight_fn(test_embeddings))

    This is the rare item that converts the cross-machine wall INTO the product: a
    certified, honestly-WIDENED prediction set under shift instead of a hidden
    under-coverage — with the same honesty caveat as
    :func:`estimate_covariate_shift_weights` (only as good as the features + fit).
    """
    weight_cal, weight_fn = estimate_covariate_shift_weights(
        cal_features, target_features, **weight_kwargs)
    cal = WeightedConformalCalibrator(alpha=alpha).fit(
        probs_cal, labels_cal, weight_calib=weight_cal)
    return cal, weight_fn
