"""Degradation-stage estimation — the honest, first-class RUL-adjacent output.

WHAT THIS IS — AND IS NOT (read before using the numbers)
---------------------------------------------------------
The 4 FEMTO labels (``normal``, ``early``, ``medium``, ``advanced``) are NOT a
fault-*type* taxonomy. They are a **positional proxy of degradation stage**:
during the FEMTO PRONOSTIA run-to-failure recordings the labels were assigned by
file position (``degradation_pct = file_idx / (total_files - 1)``), bucketed into
four life-fraction bands (roughly 0-20 / 20-50 / 50-80 / 80-100 % of life). So
the classifier learned to read *how far along the degradation path* a window is,
which is exactly a (coarse) degradation-STAGE estimator.

This module makes that honest reframe a FIRST-CLASS product output:

- :class:`DegradationEstimate` exposes the ordinal stage (0-3), a human label
  (early/mid/advanced/critical), a continuous ``degradation_index`` in [0, 1],
  and an optional conformal *stage set* (the set of stages that are plausible at
  coverage 1 - alpha, via :mod:`aion_nexus.verify`).

- ``degradation_index`` is the probability-weighted expected stage ordinal,
  normalised to [0, 1]. It is **monotone in stage** and continuous, but it is a
  COARSE 4-level interpolation, NOT a calibrated time-to-failure. We deliberately
  do **not** call it "RUL" or "hours/cycles remaining".

WHAT WE DO NOT CLAIM
--------------------
- It is **NOT a RUL in hours/cycles**. A calibrated time-to-failure needs
  continuous run-to-failure labels (actual remaining life per window) and a
  regression head trained against them (the ``rul_head`` lives in R&D, not here).
  Mapping 4 ordinal buckets to wall-clock life is not valid without that.
- The conformal stage-set coverage is valid **only under exchangeability** of the
  calibration and serving data. Cross-bearing / cross-machine deployment breaks
  exchangeability and VOIDS the marginal 1 - alpha guarantee — the caveat travels
  on every :class:`~aion_nexus.verify.ConformalCalibrator` via
  ``coverage_valid_under`` and is surfaced in ``coverage_caveat`` below.
- Calling these stages "fault diagnosis" would be FALSE (the labels carry no
  fault-type information); "degradation stage" is the honest framing and the only
  one this module makes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from aion_nexus.config import CLASS_NAMES, NUM_CLASSES
from aion_nexus.verify import ConformalCalibrator

# Human-readable degradation-stage names, ordinal 0..3. These describe the STAGE
# OF DEGRADATION (life fraction), not a fault type. They are intentionally
# distinct from the classifier's training label names (normal/early/medium/
# advanced) so a reader cannot mistake the stage proxy for a fault taxonomy.
STAGE_LABELS: tuple[str, ...] = ("early", "mid", "advanced", "critical")

# Approximate life-fraction band each ordinal stage stands for, from the FEMTO
# positional labelling (degradation_pct = file_idx / (total - 1)). Reported for
# transparency; NOT a calibrated time mapping.
STAGE_LIFE_FRACTION_BANDS: tuple[tuple[float, float], ...] = (
    (0.00, 0.20),   # 0 early
    (0.20, 0.50),   # 1 mid
    (0.50, 0.80),   # 2 advanced
    (0.80, 1.00),   # 3 critical
)

# The single honest one-liner that must travel with every estimate.
DEGRADATION_DISCLAIMER = (
    "coarse degradation-stage proxy (4 positional life-fraction bands), "
    "NOT a calibrated time-to-failure / RUL in hours or cycles"
)


@dataclass
class DegradationEstimate:
    """First-class degradation-stage output derived from a classifier's probs.

    All numeric fields are honest about being a *coarse positional stage proxy*,
    never a calibrated remaining-useful-life.

    Attributes
    ----------
    stage_ordinal:
        Argmax stage index in ``0..NUM_CLASSES-1`` (0 = earliest, 3 = critical).
    stage_label:
        Human-readable stage name from :data:`STAGE_LABELS`.
    degradation_index:
        Continuous severity in ``[0, 1]`` = probability-weighted expected stage
        ordinal divided by ``NUM_CLASSES - 1``. Monotone in stage, but a COARSE
        4-anchor interpolation — NOT a time-to-failure. ``0`` ~ pristine,
        ``1`` ~ end-of-life *stage* (not "0 hours left").
    stage_probabilities:
        Per-stage probability map (the input classifier probabilities, re-keyed
        to the stage labels), preserved for transparency.
    confidence:
        Top stage probability (``max`` of the input probabilities).
    conformal_stage_set:
        Ordinal stages that are plausible at coverage ``1 - alpha`` (set from the
        conformal calibrator). ``None`` when no calibrator was supplied (point
        estimate only — see ``calibrated``).
    conformal_stage_labels:
        Human-readable names for ``conformal_stage_set`` (``None`` when uncalibrated).
    calibrated:
        True iff a fitted :class:`ConformalCalibrator` produced the stage set.
        When False, only the point estimate is meaningful and ``abstain``/
        ``conformal_*`` reflect the uncalibrated fallback.
    abstain:
        True when the estimate should not drive an automated action: either the
        OOD/plausibility gate fired upstream (``ood``), or — when calibrated — the
        conformal stage set is not a singleton (genuine stage ambiguity).
    abstain_reason:
        Human-readable reason for ``abstain`` (``None`` when not abstaining).
    coverage_caveat:
        The exchangeability caveat from the calibrator (``None`` when uncalibrated).
    disclaimer:
        :data:`DEGRADATION_DISCLAIMER` — the coarse-proxy / not-RUL statement.
    """

    stage_ordinal: int
    stage_label: str
    degradation_index: float
    stage_probabilities: dict[str, float]
    confidence: float
    conformal_stage_set: list[int] | None = None
    conformal_stage_labels: list[str] | None = None
    calibrated: bool = False
    abstain: bool = False
    abstain_reason: str | None = None
    coverage_caveat: str | None = None
    disclaimer: str = DEGRADATION_DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def _stage_label(ordinal: int) -> str:
    if 0 <= ordinal < len(STAGE_LABELS):
        return STAGE_LABELS[ordinal]
    return str(ordinal)


def _as_prob_vector(probs) -> np.ndarray:
    """Coerce probabilities to a 1-D float vector and validate shape.

    Accepts a 1-D array/sequence or a ``{name: prob}`` mapping keyed by the
    classifier's ``CLASS_NAMES`` (the shape produced by
    :class:`~aion_nexus.inference.PredictionResult`). Does NOT re-normalise: the
    input is assumed to already be a probability vector (softmax output).
    """
    if isinstance(probs, dict):
        try:
            vec = np.array([float(probs[name]) for name in CLASS_NAMES], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(
                f"probability dict is missing class {exc}; expected keys {CLASS_NAMES}"
            ) from exc
    else:
        vec = np.asarray(probs, dtype=np.float64).ravel()
    if vec.shape != (NUM_CLASSES,):
        raise ValueError(
            f"expected a probability vector of length {NUM_CLASSES} "
            f"(one per stage), got shape {vec.shape}"
        )
    if np.any(vec < 0) or not np.all(np.isfinite(vec)):
        raise ValueError("probabilities must be finite and non-negative")
    return vec


def degradation_index_from_probs(probs) -> float:
    """Continuous degradation index in ``[0, 1]`` from stage probabilities.

    Defined as the probability-weighted expected stage ordinal divided by
    ``NUM_CLASSES - 1``::

        index = (sum_k k * p_k) / (K - 1)

    This is **monotone**: shifting probability mass toward higher (more degraded)
    stages strictly increases the index. It is a COARSE 4-anchor interpolation of
    the positional life fraction, NOT a calibrated time-to-failure.
    """
    vec = _as_prob_vector(probs)
    total = float(vec.sum())
    if total <= 0.0:
        raise ValueError("probability vector sums to zero; cannot form an index")
    vec = vec / total
    ordinals = np.arange(NUM_CLASSES, dtype=np.float64)
    expected_stage = float(np.dot(vec, ordinals))
    return expected_stage / (NUM_CLASSES - 1)


def estimate_degradation(
    probs,
    *,
    calibrator: ConformalCalibrator | None = None,
    ood_flag: bool = False,
    ood_reason: str | None = None,
) -> DegradationEstimate:
    """Turn classifier stage probabilities into a :class:`DegradationEstimate`.

    Parameters
    ----------
    probs:
        A 1-D probability vector (length ``NUM_CLASSES``) or a ``{name: prob}``
        mapping keyed by :data:`~aion_nexus.config.CLASS_NAMES`. This is the raw
        softmax output of ANY of the classifiers (v1 / v6 / v3 head) — the
        estimate is model-agnostic, exactly like :mod:`aion_nexus.verify`.
    calibrator:
        Optional FITTED :class:`~aion_nexus.verify.ConformalCalibrator`. When
        supplied, ``conformal_stage_set`` is the coverage-controlled set of
        plausible stages and ``abstain`` fires on a non-singleton set (genuine
        stage ambiguity). When ``None``, only the point estimate is returned and
        ``calibrated`` is False — the honest "not calibrated" path.
    ood_flag:
        If True the upstream plausibility gate flagged the input as implausible
        bearing vibration; the estimate ABSTAINS regardless of the conformal set
        (an implausible window must never drive an automated maintenance action).
    ood_reason:
        Human-readable OOD reason to surface when ``ood_flag`` is True.

    Returns
    -------
    DegradationEstimate
    """
    vec = _as_prob_vector(probs)
    stage_ordinal = int(np.argmax(vec))
    confidence = float(np.max(vec))
    index = degradation_index_from_probs(vec)
    stage_probabilities = {
        STAGE_LABELS[k]: float(vec[k]) for k in range(NUM_CLASSES)
    }

    conformal_set: list[int] | None = None
    conformal_labels: list[str] | None = None
    coverage_caveat: str | None = None
    calibrated = False

    if calibrator is not None:
        if calibrator.qhat is None:
            raise ValueError(
                "calibrator must be fitted (call .fit()/.calibrate()) before use; "
                "pass calibrator=None for the uncalibrated point-estimate path"
            )
        if calibrator.n_classes is not None and calibrator.n_classes != NUM_CLASSES:
            raise ValueError(
                f"calibrator was fitted on {calibrator.n_classes} classes but the "
                f"degradation taxonomy has {NUM_CLASSES} stages"
            )
        cset = sorted(int(c) for c in calibrator.predict(vec[None]).sets[0])
        conformal_set = cset
        conformal_labels = [_stage_label(c) for c in cset]
        coverage_caveat = calibrator.coverage_valid_under
        calibrated = True

    # Abstain logic. OOD always wins (implausible input -> never act). Otherwise,
    # when calibrated, a non-singleton conformal set is genuine stage ambiguity
    # and we abstain from a point action. Uncalibrated: we never *fabricate* an
    # abstain from a single confidence number here — the point estimate stands,
    # and downstream consumers (engine/server) keep their own OOD/confidence gates.
    abstain = False
    abstain_reason: str | None = None
    if ood_flag:
        abstain = True
        abstain_reason = ood_reason or "input flagged out-of-distribution by the plausibility gate"
    elif calibrated and conformal_set is not None and len(conformal_set) != 1:
        abstain = True
        abstain_reason = (
            f"conformal stage set is ambiguous ({len(conformal_set)} plausible "
            "stages at the target coverage); not a single actionable stage"
        )

    return DegradationEstimate(
        stage_ordinal=stage_ordinal,
        stage_label=_stage_label(stage_ordinal),
        degradation_index=index,
        stage_probabilities=stage_probabilities,
        confidence=confidence,
        conformal_stage_set=conformal_set,
        conformal_stage_labels=conformal_labels,
        calibrated=calibrated,
        abstain=abstain,
        abstain_reason=abstain_reason,
        coverage_caveat=coverage_caveat,
    )
