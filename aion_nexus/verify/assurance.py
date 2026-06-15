"""Assurance lattice — a typed, strictly-ordered scale that makes overclaim structurally impossible.

This is the anti-overclaim spine of the verification layer, ported clean and self-contained from
the substrate_core kernel (no cross-package import). The principle is borrowed from safety-critical
certification — **Common Criteria EAL** (Evaluation Assurance Levels) and **DO-178C DAL** (Design
Assurance Levels): the *strength* of a claim is named on an explicit, totally-ordered scale, and the
weakest link governs composition. Once the scale is named, you cannot quietly sell sampled evidence as
a proof: the tier itself records the method and its coverage, so the gap between "found no
counterexample in N trials" and "holds for every input" is visible in the type, not buried in prose.

The ladder (strictly increasing strength)::

    NONE  <  EMPIRICAL  <  BOUNDED  <  PROVEN_SPEC  <  PROVEN

- ``NONE``        — no evidence at all (an ABSTAIN, or an unknown/unrecognised level: fail-safe).
- ``EMPIRICAL``   — SAMPLED (e.g. fuzz / N-trial): no counterexample in N draws yields an ESTIMATED
                    residual risk, NOT a proof. Always carries the exchangeability/coverage caveat.
- ``BOUNDED``     — EXHAUSTIVE over a DEFINED space (sweep / bound k): sound *inside* the bound, not general.
- ``PROVEN_SPEC`` — SYMBOLIC unsat within a declared bound k: all inputs up to k, with the bound stated.
- ``PROVEN``      — SOUND: an executed counterexample, or exhaustive over the whole declared domain.

Mapping for AION-NEXUS
----------------------
A conformal verdict (the prediction-set guarantee from :mod:`aion_nexus.verify.conformal`) is
**EMPIRICAL**: it is statistical and valid only under exchangeability of calibration and serving data —
it is an estimate, never a proof. A *physically executed refutation* (e.g. a physics-verifier envelope
that runs and produces a witness that the prediction violates a bearing-dynamics bound) is stronger:
it can reach **BOUNDED** (exhaustive within the declared operating envelope) or **PROVEN_SPEC**
(symbolic within a bound). Reporting the tier alongside every verdict is what keeps the conformal
guarantee from being dressed up as something it is not.
"""
from __future__ import annotations

# --- The ladder. Ordered constants; the integer rank below is the single source of truth. ---
NONE = "none"                # no evidence (ABSTAIN, or unknown level -> fail-safe weakest)
EMPIRICAL = "empirical"      # SAMPLED (fuzz / N-trial): no counterexample in N -> ESTIMATED risk, NOT proof
BOUNDED = "bounded"          # EXHAUSTIVE over a DEFINED space (sweep / bound k): sound within k, not general
PROVEN_SPEC = "proven-spec"  # SYMBOLIC unsat within bound k: all inputs up to k, with the bound declared
PROVEN = "proven"            # SOUND: executed counterexample, or exhaustive over the whole declared domain

# Strict total order. ``.get(level, 0)`` makes any unrecognised string rank as NONE — fail-safe:
# an unknown tier can never out-rank a known one, so it can never silently strengthen a claim.
_RANK: dict[str, int] = {NONE: 0, EMPIRICAL: 1, BOUNDED: 2, PROVEN_SPEC: 3, PROVEN: 4}

# Human-readable meaning of each tier: WHAT it asserts plus its method and coverage. Keyed by level.
_DESCRIBE: dict[str, str] = {
    NONE: (
        "NONE: no evidence. Either an ABSTAIN (the gate could not adjudicate) or an "
        "unrecognised level treated as weakest (fail-safe). Asserts nothing."
    ),
    EMPIRICAL: (
        "EMPIRICAL: SAMPLED evidence (fuzz / N-trial). No counterexample was found in N draws, "
        "which yields an ESTIMATED residual risk (see residual_risk_rule_of_three), NOT a proof. "
        "Validity depends on the sampling distribution / exchangeability; a hidden single-point "
        "failure outside the sample is not covered."
    ),
    BOUNDED: (
        "BOUNDED: EXHAUSTIVE over a DEFINED space (e.g. a parameter sweep or a bound k). Sound "
        "INSIDE that space; says nothing about inputs outside the declared bound."
    ),
    PROVEN_SPEC: (
        "PROVEN-SPEC: SYMBOLIC result (e.g. solver-unsat) within a declared bound k. Covers all "
        "inputs up to k, with the bound stated; not a claim beyond k."
    ),
    PROVEN: (
        "PROVEN: SOUND. An executed counterexample (the claim is refuted, witness re-runnable), or "
        "an exhaustive check over the WHOLE declared domain. The strongest tier."
    ),
}


def weakest(assurances: list[str]) -> str:
    """The weakest link governs: a composed system is only as strong as its weakest component.

    Returns the tier with the lowest rank. An empty list returns :data:`NONE` (nothing composed -> no
    evidence). Unknown strings rank as ``NONE`` (fail-safe), so an unrecognised tier drags the
    composition down rather than up. This is the ``compose_and`` semantics: every part must hold.
    """
    if not assurances:
        return NONE
    return min(assurances, key=lambda a: _RANK.get(a, 0))


def strongest(assurances: list[str]) -> str:
    """The strongest tier present — the ``compose_or`` semantics (any one path suffices).

    Returns the tier with the highest rank. An empty list returns :data:`NONE`. Unknown strings rank
    as ``NONE``. Use this only when a single satisfied alternative genuinely discharges the claim;
    for an all-must-hold composition use :func:`weakest`.
    """
    if not assurances:
        return NONE
    return max(assurances, key=lambda a: _RANK.get(a, 0))


def rank(level: str) -> int:
    """Integer rank of a tier (0..4). Unknown tiers rank 0 (NONE, fail-safe weakest)."""
    return _RANK.get(level, 0)


def describe(level: str) -> str:
    """Explain what a tier ASSERTS plus its method and coverage.

    An unrecognised level returns the :data:`NONE` description (fail-safe), so callers always get a
    non-empty, honest string rather than a misleading blank.
    """
    return _DESCRIBE.get(level, _DESCRIBE[NONE])


def residual_risk_rule_of_three(n_trials: int) -> float:
    """Estimate the residual failure probability after ``n_trials`` clean EMPIRICAL trials: ``3 / N``.

    The *rule of three*: if an event was not observed in N independent trials, an approximate upper
    bound on its probability is ``3 / N`` (the one-sided 95% bound when the true rate is small). For
    ``N = 300`` this gives ``0.01``.

    This is an ESTIMATE, not a proof, and it rests on assumptions:
      * the trials are independent and drawn from the SAME distribution the system will actually see;
      * the failure mode is not concentrated in a region the sampler avoids.
    When those fail — a black-box subject, a non-uniform input space, an adversarially-placed
    single-point bug — the true risk can be much higher: the estimate UNDERSTATES it. It only ever
    supports an EMPIRICAL tier, never a stronger one.

    ``n_trials <= 0`` returns ``1.0`` (no clean trials -> no evidence -> worst-case residual risk).
    """
    if n_trials <= 0:
        return 1.0
    return 3.0 / n_trials


__all__ = [
    "NONE",
    "EMPIRICAL",
    "BOUNDED",
    "PROVEN_SPEC",
    "PROVEN",
    "weakest",
    "strongest",
    "rank",
    "describe",
    "residual_risk_rule_of_three",
]
