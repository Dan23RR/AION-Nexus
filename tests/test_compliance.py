"""Tests for the compliance-evidence layer (aion_nexus.compliance).

The contract these tests pin down is honesty-first:
  * compliance_evidence() returns the expected structure (disclaimer, certificate
    reference, a list of evidence items, an overall limitation)
  * every evidence item carries the three required keys, and `limitation` is
    NEVER empty (an evidence item with no stated limit is the overclaim we refuse)
  * the EU AI Act Art. 12 / 14 / 15 obligations are all mapped, plus the two ISO
    frameworks
  * NO forbidden claim string ("compliant" / "conforme" / "certified compliant" /
    "conformità") appears anywhere in the GENERATED text (dict or card) — the
    test fails loudly if it does
  * the human-readable card renders and stays free of forbidden claims
  * the auth regime in the evidence honestly tracks the certificate's
    `authentication` field (NONE => not tamper-evident; HMAC => tamper-evident)
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from aion_nexus.compliance import (
    _FORBIDDEN_CLAIMS,
    annex_iv_card,
    annex_iv_dossier,
    compliance_evidence,
    evidence_card,
)
from aion_nexus.verify import Verifier
from aion_nexus.verify.certificate import ENV_HMAC_KEY
from aion_nexus.verify.conformal import softmax

CLASS_NAMES = ["normal", "early", "medium", "advanced"]


def _synthetic_exchangeable(n=8000, k=4, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k))
    logits[np.arange(n), true] += 2.0
    probs = softmax(logits)
    idx = rng.permutation(n)
    half = n // 2
    return probs, true, idx[:half], idx[half:]


def _fit_verifier(monkeypatch, threshold=0.0, key=None):
    if key is None:
        monkeypatch.delenv(ENV_HMAC_KEY, raising=False)
    else:
        monkeypatch.setenv(ENV_HMAC_KEY, key)
    probs, true, cal, _ = _synthetic_exchangeable()
    v = Verifier(alpha=0.1, class_names=CLASS_NAMES, abstain_threshold=threshold)
    v.calibrate(probs[cal], true[cal])
    return v


def _certify(monkeypatch, probs, *, key=None, threshold=0.0, with_input=False):
    v = _fit_verifier(monkeypatch, threshold=threshold, key=key)
    sig = np.arange(2048, dtype="float32") if with_input else None
    return v.certify(np.asarray(probs), input_signal=sig, model_id="aion-v1")


def _all_generated_text(data: dict, card: str) -> str:
    """Concatenate every string the module emits, lower-cased, for claim scanning."""
    return (json.dumps(data, default=str) + "\n" + card).lower()


# --------------------------------------------------------------------------- #
# 1. Structure
# --------------------------------------------------------------------------- #

def test_structure_has_expected_top_level_keys(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    data = compliance_evidence(cert)
    for key in ("disclaimer", "certificate_ref", "evidence", "overall_limitation"):
        assert key in data, f"missing top-level key {key!r}"
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) >= 5


def test_certificate_ref_echoes_cert_fields(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    ref = compliance_evidence(cert)["certificate_ref"]
    assert ref["cert_id"] == cert.cert_id
    assert ref["verdict"] == cert.verdict
    assert ref["content_hash"] == cert.content_hash
    assert ref["model_id"] == "aion-v1"


def test_every_evidence_item_has_the_three_required_keys(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    for item in compliance_evidence(cert)["evidence"]:
        for key in ("framework", "reference", "title",
                    "provides_evidence_for", "how", "limitation"):
            assert key in item, f"evidence item missing {key!r}: {item}"


def test_limitation_fields_are_non_empty(monkeypatch):
    """The core honesty invariant: no evidence item without a stated limit."""
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    for item in compliance_evidence(cert)["evidence"]:
        lim = item["limitation"]
        assert isinstance(lim, str) and lim.strip(), \
            f"empty limitation for {item['reference']}"


def test_provides_evidence_phrasing_is_used(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    for item in compliance_evidence(cert)["evidence"]:
        assert "provides evidence toward" in item["provides_evidence_for"].lower()


# --------------------------------------------------------------------------- #
# 2. Coverage of the obligations
# --------------------------------------------------------------------------- #

def test_eu_ai_act_articles_are_mapped(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    refs = {(i["framework"], i["reference"]) for i in compliance_evidence(cert)["evidence"]}
    frameworks = {f for f, _ in refs}
    references = {r for _, r in refs}
    assert any("EU AI Act" in f for f in frameworks)
    assert "Art. 12" in references
    assert "Art. 14" in references
    assert "Art. 15" in references
    assert any("13381" in f for f in frameworks)
    assert any("42001" in f for f in frameworks)


def test_art15_states_exchangeability_caveat(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    art15 = next(i for i in compliance_evidence(cert)["evidence"]
                 if i["reference"] == "Art. 15")
    text = (art15["how"] + " " + art15["limitation"]).lower()
    assert "exchangeab" in text
    assert "cross-bearing" in text or "cross-machine" in text


def test_art15_flags_ood_gate_not_in_certificate(monkeypatch):
    """Honesty: the OOD gate is upstream, not a certificate field."""
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    art15 = next(i for i in compliance_evidence(cert)["evidence"]
                 if i["reference"] == "Art. 15")
    assert "ood" in art15["limitation"].lower() or \
        "upstream" in art15["limitation"].lower()


def test_iso13381_flags_positional_stage_not_fault_type(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    iso = next(i for i in compliance_evidence(cert)["evidence"]
               if "13381" in i["framework"])
    assert "positional" in iso["limitation"].lower()
    assert "fault" in iso["limitation"].lower()


# --------------------------------------------------------------------------- #
# 3. NO forbidden compliance-claim strings in generated text
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("verdict_probs,threshold", [
    ([0.97, 0.01, 0.01, 0.01], 0.0),   # CERTIFIED
    ([0.30, 0.28, 0.22, 0.20], 0.0),   # REVIEW
    ([0.60, 0.20, 0.10, 0.10], 0.9),   # ABSTAIN
])
def test_no_forbidden_claim_strings_in_generated_text(monkeypatch, verdict_probs, threshold):
    cert = _certify(monkeypatch, verdict_probs, threshold=threshold)
    data = compliance_evidence(cert)
    card = evidence_card(cert)
    haystack = _all_generated_text(data, card)
    for bad in _FORBIDDEN_CLAIMS:
        assert bad.lower() not in haystack, \
            f"forbidden claim string {bad!r} appeared in generated compliance text"


def test_no_forbidden_claim_strings_with_and_without_key(monkeypatch):
    for key in (None, "s3cr3t-key"):
        cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01], key=key)
        data = compliance_evidence(cert)
        haystack = _all_generated_text(data, evidence_card(cert))
        for bad in _FORBIDDEN_CLAIMS:
            assert bad.lower() not in haystack


# --------------------------------------------------------------------------- #
# 4. Human-readable card
# --------------------------------------------------------------------------- #

def test_card_renders_and_contains_disclaimer(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    card = evidence_card(cert)
    assert isinstance(card, str) and card.strip()
    assert "NOT a declaration of conformity" in card
    assert cert.cert_id in card
    # every framework reference shows up in the card
    for item in compliance_evidence(cert)["evidence"]:
        assert item["reference"] in card


def test_card_states_limitations_for_each_item(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    card = evidence_card(cert)
    assert card.lower().count("limitation") >= 5


# --------------------------------------------------------------------------- #
# 5. Auth regime honesty (NONE vs HMAC)
# --------------------------------------------------------------------------- #

def test_no_key_record_keeping_flags_not_tamper_evident(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01], key=None)
    assert cert.authentication == "NONE"
    art12 = next(i for i in compliance_evidence(cert)["evidence"]
                 if i["reference"] == "Art. 12")
    low = (art12["how"] + " " + art12["limitation"]).lower()
    assert "not" in low and "tamper" in low


def test_keyed_record_keeping_reports_tamper_evident(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01], key="s3cr3t-key")
    assert cert.authentication == "HMAC-SHA256"
    art12 = next(i for i in compliance_evidence(cert)["evidence"]
                 if i["reference"] == "Art. 12")
    assert "tamper-evident" in art12["how"].lower()


# --------------------------------------------------------------------------- #
# 6. Accepts a Certificate OR its dict
# --------------------------------------------------------------------------- #

def test_accepts_certificate_dict(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    from_obj = compliance_evidence(cert)
    from_dict = compliance_evidence(cert.as_dict())
    assert from_obj["certificate_ref"] == from_dict["certificate_ref"]
    assert len(from_obj["evidence"]) == len(from_dict["evidence"])


# --------------------------------------------------------------------------- #
# 7. Annex IV technical-documentation evidence map (v2.10.0)
# --------------------------------------------------------------------------- #

def _all_text(obj) -> str:
    return json.dumps(obj, default=str).lower()


def test_annex_iv_has_all_nine_points_with_required_keys():
    d = annex_iv_dossier()
    assert len(d["sections"]) == 9
    numbers = [s["number"] for s in d["sections"]]
    assert numbers == [str(i) for i in range(1, 10)]
    for s in d["sections"]:
        for key in ("title", "annex_iv_requirement", "aion_provides",
                    "deployer_must_supply", "status", "limitation"):
            assert s[key], f"section {s['number']} missing/empty {key}"


def test_annex_iv_never_emits_forbidden_claims():
    # Across several metadata shapes, neither the dict nor the card may claim conformity.
    for md in (None, {}, {"intended_purpose": "advisory only", "architecture": "BiGRU",
                         "datasets": "FEMTO", "harmonised_standards": "ISO 13374"}):
        blob = _all_text(annex_iv_dossier(md)) + annex_iv_card(md).lower()
        for claim in _FORBIDDEN_CLAIMS:
            assert claim.lower() not in blob, f"forbidden claim {claim!r} leaked"
        assert "compliant" not in blob          # the specific overclaim we refuse


def test_annex_iv_readiness_is_not_a_conformity_measure():
    d = annex_iv_dossier()
    r = d["readiness"]
    assert r["sections_total"] == 9
    assert (r["sections_with_aion_evidence"] + r["sections_partial"]
            + r["sections_deployer_owned"]) == 9
    # The note must explicitly disclaim that this is a conformity/readiness measure.
    assert "not a measure of regulatory conformity" in r["note"].lower()
    assert "not the technical documentation" in d["disclaimer"].lower()


def test_annex_iv_declaration_of_conformity_is_provider_owned():
    # Point 8 must never claim AION supplies the declaration.
    sec8 = next(s for s in annex_iv_dossier()["sections"] if s["number"] == "8")
    assert sec8["status"] == "deployer-owned"
    assert "only by the provider" in sec8["aion_provides"].lower()


def test_annex_iv_uses_caller_metadata_else_marks_provider_owned():
    with_md = annex_iv_dossier({"architecture": "BiGRU 1.06M params"})
    sec2_with = next(s for s in with_md["sections"] if s["number"] == "2")
    assert "BiGRU 1.06M params" in sec2_with["aion_provides"]
    # Absent metadata -> the standards point is provider-owned (honest), not invented.
    without = annex_iv_dossier()
    sec7 = next(s for s in without["sections"] if s["number"] == "7")
    assert sec7["status"] == "deployer-owned"


def test_annex_iv_threads_certificate_identity(monkeypatch):
    cert = _certify(monkeypatch, [0.97, 0.01, 0.01, 0.01])
    d = annex_iv_dossier({"name": "AION-NEXUS"}, certificate=cert)
    assert d["system_ref"]["model_id"] == cert.model_id


def test_annex_iv_card_renders_markdown():
    card = annex_iv_card({"version": "2.10.0", "documentation_date": "2026-06-16"})
    assert card.startswith("# EU AI Act Annex IV")
    assert "## 8. EU declaration of conformity" in card
    assert "Limitation:" in card
