"""cheatbench — MEASURE the cheating surface of the certificate gate (do NOT assume it ~0).

Ported in spirit from ``substrate_core.cheatbench``: the honest version of the
pitch "tamper-evident by construction". The question is not "is the gate secure?"
but "an adversary who WANTS a trusted-CERTIFIED certificate they did not earn —
which channels are still open AFTER the design, MEASURED by a real run?".

For each channel we construct an adversarial certificate, push it through the
public verification surface (``verify_certificate`` / the store / the assurance
hash), and MEASURE the outcome. A channel is:

  * CLOSED — the attack is detected: ``trusted`` is ``False`` (or ``integrity_ok``
    is ``False``). The gate did its job, MEASURED, not assumed.
  * OPEN   — the attack succeeds, OR it is an irreducible residual we report
    openly rather than hide.

The honest residual (channel 5, ``confident_singleton_unverified``) mirrors
``gen_evasion_point`` in substrate_core: the conformal guarantee is MARGINAL
COVERAGE, not per-instance correctness. A confidently-wrong but in-distribution
input can still yield a singleton CERTIFIED set whose crypto is perfectly valid.
That is NOT a tamper bug — the certificate is authentic and its tier is honestly
EMPIRICAL — it is the declared limit of the method. We surface it as OPEN with its
mitigation (an OOD gate covers OOD drift, NOT the confident in-distribution
error), because §6.31 forbids dressing an empirical guarantee as more than it is.

``run_cheatbench()`` returns a dict; ``_report()`` prints a human-readable table.
Every number comes from a real run — nothing here is asserted a priori.
"""
from __future__ import annotations

import numpy as np

from . import assurance as _assurance
from .certificate import (
    AUTH_NONE,
    VERDICT_CERTIFIED,
    Certificate,
    verify_certificate,
)
from .signing import ed25519_pubkey_from_seed
from .verifier import Verifier

# A fixed adversary seed (the attacker's OWN key — they can always mint one).
_ATTACKER_SEED = b"cheatbench-attacker-seed"
# The legitimate issuer seed whose PUBLIC key a verifier would trust out-of-band.
_ISSUER_SEED = b"cheatbench-issuer-seed"

CLASS_NAMES = ["normal", "early", "medium", "advanced"]


def _fit_verifier(threshold: float = 0.0) -> Verifier:
    """A calibrated verifier on synthetic exchangeable data (no env / no key)."""
    rng = np.random.default_rng(0)
    n, k = 4000, 4
    true = rng.integers(0, k, n)
    logits = rng.standard_normal((n, k))
    logits[np.arange(n), true] += 2.0
    # softmax
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= probs.sum(axis=1, keepdims=True)
    v = Verifier(alpha=0.1, class_names=CLASS_NAMES, abstain_threshold=threshold)
    v.calibrate(probs, true)
    return v


# --------------------------------------------------------------------------- #
# Channels — each returns (closed: bool, detail: str). MEASURED, never assumed.
# --------------------------------------------------------------------------- #

def _forge_without_key() -> tuple[bool, str]:
    """(1) Forge a CERTIFIED certificate WITHOUT any signing key.

    An attacker builds a clean-looking CERTIFIED certificate, seals it with NO key
    (``authentication=NONE``), and ships it. A verifier that demands issuer
    authentication (``expected_pubkey``) must NOT trust it. Expected CLOSED.
    """
    cert = Certificate(
        predicted_label=0, predicted_name="normal",
        conformal_set=[0], conformal_set_names=["normal"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
    ).seal(scheme="none")
    assert cert.authentication == AUTH_NONE  # sanity: no key was used
    expected_pub = ed25519_pubkey_from_seed(_ISSUER_SEED)
    res = verify_certificate(cert, expected_pubkey=expected_pub)
    closed = res["trusted"] is False
    return closed, (
        f"unsigned CERTIFIED cert -> authentication={cert.authentication}; "
        f"verify(expected_pubkey) -> authenticity={res['authenticity']}, "
        f"trusted={res['trusted']} (a NONE cert can never be trusted)")


def _label_tamper() -> tuple[bool, str]:
    """(2) Seal a cert, then edit ``predicted_name`` WITHOUT re-sealing.

    The human-readable label is bound into ``content_hash``, so editing only the
    display label breaks integrity. Expected CLOSED.
    """
    cert = Certificate(
        predicted_label=0, predicted_name="normal",
        conformal_set=[0], conformal_set_names=["normal"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
    ).seal(_ISSUER_SEED, scheme="ed25519")
    d = cert.as_dict()
    d["predicted_name"] = "advanced"  # forge the displayed class only
    res = verify_certificate(d, expected_pubkey=cert.pubkey)
    closed = (res["integrity_ok"] is False) and (res["trusted"] is False)
    return closed, (
        f"edited predicted_name normal->advanced post-seal -> "
        f"integrity_ok={res['integrity_ok']}, trusted={res['trusted']} "
        "(label is in the hashed payload)")


def _assurance_overclaim() -> tuple[bool, str]:
    """(3) Upgrade the assurance tier empirical->proven WITHOUT re-signing.

    The tier is part of the canonical payload, so it is hashed. Silently claiming
    a stronger guarantee breaks ``content_hash`` and the signature. Expected
    CLOSED — this is the structural anti-overclaim guarantee.
    """
    cert = Certificate(
        predicted_label=0, predicted_name="normal",
        conformal_set=[0], conformal_set_names=["normal"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
        assurance=_assurance.EMPIRICAL,
    ).seal(_ISSUER_SEED, scheme="ed25519")
    d = cert.as_dict()
    assert d["assurance"] == _assurance.EMPIRICAL
    d["assurance"] = _assurance.PROVEN  # overclaim: empirical sold as proven
    res = verify_certificate(d, expected_pubkey=cert.pubkey)
    closed = (res["integrity_ok"] is False) and (res["trusted"] is False)
    return closed, (
        f"upgraded assurance empirical->proven post-seal -> "
        f"integrity_ok={res['integrity_ok']}, trusted={res['trusted']} "
        "(tier is hashed; overclaim breaks content_hash + signature)")


def _downgrade_strip_sig() -> tuple[bool, str]:
    """(4) Strip the signature from an Ed25519 cert and declare it NONE.

    An attacker without the seed removes ``signature`` and sets
    ``authentication=NONE``, hoping a verifier degrades to integrity-only and
    trusts it. Verified against the EXPECTED pubkey, it must NOT be trusted.
    Expected CLOSED.
    """
    cert = Certificate(
        predicted_label=0, predicted_name="normal",
        conformal_set=[0], conformal_set_names=["normal"],
        verdict=VERDICT_CERTIFIED, alpha=0.1, qhat=0.5,
    ).seal(_ISSUER_SEED, scheme="ed25519")
    expected_pub = cert.pubkey
    d = cert.as_dict()
    d["signature"] = None
    d["authentication"] = AUTH_NONE          # downgrade the declared scheme
    res = verify_certificate(d, expected_pubkey=expected_pub)
    # Under expected_pubkey, a NONE cert routes to the HMAC/NONE leg with no HMAC
    # key -> UNVERIFIED, never trusted. Either way trusted must be False.
    closed = res["trusted"] is False
    return closed, (
        f"stripped signature + authentication->NONE -> "
        f"authenticity={res['authenticity']}, trusted={res['trusted']} "
        "(a stripped cert authenticates to nothing under the expected key)")


def _confident_singleton_unverified() -> tuple[bool, str]:
    """(5) HONEST RESIDUAL — a confident, in-distribution singleton the
    certificate cannot verify per-instance.

    We feed the calibrated verifier a confident in-distribution probability
    vector. The conformal gate emits a singleton CERTIFIED set and the
    certificate is perfectly authentic. Crucially, *whether that singleton is
    correct for this instance is a fact the marginal guarantee does not speak
    to* — we do NOT claim the prediction is wrong (we cannot, from probabilities
    alone), only that a confident error WOULD pass here indistinguishably from a
    correct one.

    This is NOT a tamper bug and NOT closeable by the crypto: conformal coverage
    is MARGINAL (a rate over the population), NOT per-instance correctness. We
    report it OPEN, with the mitigation, exactly as substrate_core reports
    ``gen_evasion_point``. Hiding it would be the overclaim §6.31 forbids.
    """
    v = _fit_verifier(threshold=0.0)
    # Confident singleton on class 0, sealed honestly. Whether it is "wrong" is a
    # per-instance fact the marginal guarantee does not speak to.
    cert = v.certify(np.array([0.97, 0.01, 0.01, 0.01]), seed=_ISSUER_SEED)
    res = verify_certificate(cert, expected_pubkey=cert.pubkey)
    emitted_certified = (cert.verdict == VERDICT_CERTIFIED)
    # The certificate is trusted (authentic) AND its tier is honestly EMPIRICAL.
    honest_tier = (cert.assurance == _assurance.EMPIRICAL)
    # OPEN: a valid, trusted, CERTIFIED cert carries no per-instance correctness
    # guarantee — a confident error would be indistinguishable from a correct one.
    closed = False
    detail = (
        f"verdict={cert.verdict}, |set|={len(cert.conformal_set)}, "
        f"assurance={cert.assurance}, trusted={res['trusted']} -- "
        "RESIDUAL (OPEN, declared): conformal coverage is MARGINAL, not "
        "per-instance correctness; a confident in-distribution error would pass as "
        "a valid singleton CERTIFIED, indistinguishable from a correct one. "
        "Mitigation: an OOD gate covers DISTRIBUTION drift, NOT the confident "
        "in-distribution error; the EMPIRICAL tier + exchangeability caveat "
        "already signal this honestly.")
    # Sanity that the certificate is genuinely authentic and honestly-tiered.
    assert emitted_certified and honest_tier and res["trusted"]
    return closed, detail


# Ordered channels: 1-4 are CLOSED guarantees (regression), 5 is the OPEN residual.
_CHANNELS = [
    ("forge_without_key", _forge_without_key, True),
    ("label_tamper", _label_tamper, True),
    ("assurance_overclaim", _assurance_overclaim, True),
    ("downgrade_strip_sig", _downgrade_strip_sig, True),
    ("confident_singleton_unverified", _confident_singleton_unverified, False),
]


def run_cheatbench() -> dict:
    """Run every channel and MEASURE the cheating surface. Returns::

        {"rate_closed": float,         # fraction of channels measured CLOSED
         "n_closed": int, "n": int,
         "channels": [{"name", "closed", "expected_closed", "detail"}],
         "residual": {...}}            # the declared OPEN residual (channel 5)

    Each channel's ``closed`` flag is the MEASURED outcome of a real run, compared
    against ``expected_closed`` so a regression (a channel that was closed silently
    opening) is visible.
    """
    rows = []
    n_closed = 0
    for name, fn, expected_closed in _CHANNELS:
        closed, detail = fn()
        n_closed += int(closed)
        rows.append({
            "name": name,
            "closed": closed,
            "expected_closed": expected_closed,
            "regressed": closed != expected_closed,
            "detail": detail,
        })
    residual = next(r for r in rows if r["name"] == "confident_singleton_unverified")
    return {
        "rate_closed": n_closed / len(_CHANNELS),
        "n_closed": n_closed,
        "n": len(_CHANNELS),
        "channels": rows,
        "residual": residual,
    }


def _report() -> str:
    """Human-readable cheatbench table. Returns the string (also printed)."""
    res = run_cheatbench()
    lines = [
        "=" * 78,
        "CHEATBENCH — cheating surface MEASURED (trusted-CERTIFIED unearned = cheat)",
        "=" * 78,
        f"  {'channel':34} {'status':10} {'expected':10}",
        "-" * 78,
    ]
    for r in res["channels"]:
        status = "CLOSED" if r["closed"] else "OPEN"
        exp = "CLOSED" if r["expected_closed"] else "OPEN (residual)"
        flag = "  <REGRESSED>" if r["regressed"] else ""
        lines.append(f"  {r['name']:34} {status:10} {exp:10}{flag}")
    lines.append("-" * 78)
    lines.append(
        f"  closed: {res['n_closed']}/{res['n']} channels  "
        f"(rate_closed = {res['rate_closed']:.2f})")
    lines.append(
        "  RESIDUAL (OPEN, declared): confident_singleton_unverified. Conformal")
    lines.append(
        "  coverage is MARGINAL, not per-instance correctness. An OOD gate covers")
    lines.append(
        "  distribution drift, NOT the confident in-distribution error. Reported,")
    lines.append("  not hidden (the EMPIRICAL tier already signals it).")
    lines.append("=" * 78)
    out = "\n".join(lines)
    print(out)
    return out


if __name__ == "__main__":
    _report()
