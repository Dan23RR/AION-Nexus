"""Tests for the assurance lattice (aion_nexus.verify.assurance).

The lattice is the anti-overclaim spine: a strictly-ordered, typed scale where the weakest link
governs composition. These tests pin the contracts that make overclaim structurally impossible:
  * the order NONE < EMPIRICAL < BOUNDED < PROVEN_SPEC < PROVEN is total and respected;
  * weakest / strongest pick the right ends of mixed lists;
  * an empty composition is NONE (no evidence), not a crash;
  * an unknown/unrecognised tier ranks as NONE (fail-safe: it can never strengthen a claim);
  * the rule-of-three residual risk is 3/N (N=300 -> 0.01) and degrades safely;
  * every named tier has a non-empty, honest description.
"""
from __future__ import annotations

import math

import pytest

from aion_nexus.verify import assurance as asr

ALL_TIERS = [asr.NONE, asr.EMPIRICAL, asr.BOUNDED, asr.PROVEN_SPEC, asr.PROVEN]


def test_ladder_is_strictly_increasing():
    """rank() must be a strict total order NONE < EMPIRICAL < BOUNDED < PROVEN_SPEC < PROVEN."""
    ranks = [asr.rank(t) for t in ALL_TIERS]
    assert ranks == [0, 1, 2, 3, 4]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)  # strict: no ties


def test_weakest_picks_lowest_on_mixed_list():
    assert asr.weakest([asr.PROVEN, asr.EMPIRICAL, asr.BOUNDED]) == asr.EMPIRICAL
    assert asr.weakest([asr.PROVEN_SPEC, asr.PROVEN]) == asr.PROVEN_SPEC
    # NONE present -> NONE governs (an unverified component sinks the whole composition).
    assert asr.weakest([asr.PROVEN, asr.NONE, asr.BOUNDED]) == asr.NONE


def test_strongest_picks_highest_on_mixed_list():
    assert asr.strongest([asr.NONE, asr.EMPIRICAL, asr.BOUNDED]) == asr.BOUNDED
    assert asr.strongest([asr.EMPIRICAL, asr.PROVEN, asr.PROVEN_SPEC]) == asr.PROVEN
    assert asr.strongest([asr.NONE, asr.NONE]) == asr.NONE


def test_empty_list_is_none():
    """Nothing composed -> no evidence -> NONE (not an error)."""
    assert asr.weakest([]) == asr.NONE
    assert asr.strongest([]) == asr.NONE


def test_unknown_tier_treated_as_none_failsafe():
    """An unrecognised string ranks as NONE (0): fail-safe, it can never out-rank a known tier."""
    assert asr.rank("totally-made-up") == 0
    assert asr.rank("PROVEN") == 0  # case-sensitive: the constant is lowercase "proven"
    # In composition, an unknown tier drags weakest() down...
    assert asr.weakest([asr.PROVEN, "made-up"]) == "made-up"
    # ...and can never lift strongest() above a real tier.
    assert asr.strongest([asr.EMPIRICAL, "made-up"]) == asr.EMPIRICAL


def test_rule_of_three_n_300_is_one_percent():
    assert asr.residual_risk_rule_of_three(300) == pytest.approx(0.01)


@pytest.mark.parametrize("n,expected", [(3, 1.0), (30, 0.1), (300, 0.01), (3000, 0.001)])
def test_rule_of_three_scales_as_three_over_n(n, expected):
    assert asr.residual_risk_rule_of_three(n) == pytest.approx(expected)


@pytest.mark.parametrize("n", [0, -1, -1000])
def test_rule_of_three_non_positive_is_worst_case(n):
    """No clean trials -> no evidence -> residual risk 1.0 (fail-safe), never a divide-by-zero."""
    assert asr.residual_risk_rule_of_three(n) == 1.0


def test_rule_of_three_is_finite_and_in_unit_interval_for_positive_n():
    for n in (1, 2, 5, 100, 10_000):
        r = asr.residual_risk_rule_of_three(n)
        assert math.isfinite(r)
        assert 0.0 < r <= 3.0  # 3/N; <=1 once N>=3, but always positive and finite


def test_describe_non_empty_for_every_tier():
    for tier in ALL_TIERS:
        text = asr.describe(tier)
        assert isinstance(text, str)
        assert text.strip()  # non-empty


def test_describe_distinguishes_empirical_from_proven():
    """The honesty contract: EMPIRICAL names estimate/sampling, PROVEN names sound/executed."""
    emp = asr.describe(asr.EMPIRICAL).lower()
    assert "estimat" in emp or "sampl" in emp
    assert "not a proof" in emp or "not proof" in emp.replace(", ", " ")
    proven = asr.describe(asr.PROVEN).lower()
    assert "sound" in proven or "exhaustive" in proven or "executed" in proven


def test_describe_unknown_tier_falls_back_to_none_non_empty():
    """Unknown -> NONE description (fail-safe), still a non-empty honest string, never blank."""
    text = asr.describe("made-up-tier")
    assert text.strip()
    assert text == asr.describe(asr.NONE)
