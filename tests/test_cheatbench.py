"""Tests for aion_nexus.verify.cheatbench — the MEASURED cheating surface.

These are REGRESSION tests with teeth: the 4 closeable channels must stay CLOSED
(a silent re-opening is a security regression), and the honest residual
(``confident_singleton_unverified``) must stay reported as OPEN — never quietly
flipped to CLOSED, which would be the overclaim §6.31 forbids.
"""
from __future__ import annotations

from aion_nexus.verify import run_cheatbench
from aion_nexus.verify.cheatbench import _report


def test_run_cheatbench_shape():
    res = run_cheatbench()
    assert set(res) >= {"rate_closed", "n_closed", "n", "channels", "residual"}
    assert res["n"] == 5
    names = {c["name"] for c in res["channels"]}
    assert names == {
        "forge_without_key",
        "label_tamper",
        "assurance_overclaim",
        "downgrade_strip_sig",
        "confident_singleton_unverified",
    }


def test_four_channels_closed_regression():
    """The 4 closeable channels must MEASURE as CLOSED (no silent re-opening)."""
    res = run_cheatbench()
    by_name = {c["name"]: c for c in res["channels"]}
    for name in ("forge_without_key", "label_tamper",
                 "assurance_overclaim", "downgrade_strip_sig"):
        ch = by_name[name]
        assert ch["closed"] is True, f"{name} REGRESSED (now OPEN): {ch['detail']}"
        assert ch["expected_closed"] is True
        assert ch["regressed"] is False


def test_residual_is_declared_open():
    """The honest residual must stay OPEN — never dressed up as closed."""
    res = run_cheatbench()
    residual = res["residual"]
    assert residual["name"] == "confident_singleton_unverified"
    assert residual["closed"] is False          # OPEN, declared
    assert residual["expected_closed"] is False
    assert residual["regressed"] is False
    assert "marginal" in residual["detail"].lower()


def test_rate_closed_is_four_fifths():
    res = run_cheatbench()
    assert res["n_closed"] == 4
    assert abs(res["rate_closed"] - 0.8) < 1e-9


def test_no_channel_regressed():
    res = run_cheatbench()
    assert all(not c["regressed"] for c in res["channels"])


def test_report_is_printable():
    out = _report()
    assert "CHEATBENCH" in out
    assert "RESIDUAL" in out
    assert "CLOSED" in out and "OPEN" in out
