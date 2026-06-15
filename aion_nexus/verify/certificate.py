"""Certificate — a re-runnable, optionally tamper-evident record of one decision.

Every certified prediction ships a :class:`Certificate`: the verdict, the
conformal prediction set, the calibration parameters, and a ``content_hash``
that anyone can recompute to confirm the record is internally consistent.

Integrity vs. authenticity — read this before trusting a certificate
--------------------------------------------------------------------
``content_hash`` is a plain SHA-256 over the *canonical public payload*. It
proves INTERNAL CONSISTENCY (the fields were not corrupted) and gives
reproducibility — anyone with this code can recompute it. It does NOT prove
AUTHENTICITY: an adversary who has this source can fabricate any payload and a
matching hash. So ``content_hash`` alone is NOT tamper-evident against an
attacker.

Authenticity comes from the keyed signature. Set env ``VERIFY_HMAC_KEY`` and
every certificate carries ``authentication = "HMAC-SHA256"`` plus
``signature = HMAC-SHA256(key, content_hash)``; without the secret an attacker
cannot forge that. With no key set, the certificate honestly declares
``authentication = "NONE"`` (integrity hash only — NOT tamper-evident against an
adversary with the source). Verify with :func:`verify_certificate`.

Ed25519 (asymmetric) — third-party-checkable authenticity
---------------------------------------------------------
Set env ``VERIFY_ED25519_SEED`` (or pass a seed to :meth:`Certificate.seal`) and
the certificate is signed with an Ed25519 PRIVATE key derived from the seed:
``authentication = "Ed25519"``, ``signature = ed25519_sign(content_hash, seed)``,
and the matching PUBLIC key is embedded in ``pubkey``. Because verifying needs
only the public key, anyone — an auditor, a customer, an insurer — can check the
certificate OFFLINE without ever receiving anything that would let them forge a
new one (the property HMAC cannot offer; see :mod:`aion_nexus.verify.signing`).

HONESTY — what the EMBEDDED pubkey does and does NOT prove
----------------------------------------------------------
Verifying a signature against the pubkey *embedded in the certificate* proves
only INTERNAL CONSISTENCY (the signature matches that key over this payload). It
does NOT prove the certificate came from the EXPECTED issuer: an attacker can
mint their own seed, sign a forged payload, and embed their own pubkey — wholly
self-consistent. Genuine issuer authentication requires checking against an
EXPECTED pubkey obtained out-of-band (arg ``expected_pubkey`` or env
``VERIFY_ED25519_PUBKEY``). :func:`verify_certificate` therefore reports
``"SELF-SIGNED"`` (NOT ``"VERIFIED"``) when only the embedded key is available,
and ``trusted`` is ``True`` ONLY for ``"VERIFIED"``.

Assurance tier (anti-overclaim)
-------------------------------
Every certificate names the STRENGTH of its verdict via ``assurance`` (see
:mod:`aion_nexus.verify.assurance`). A conformal verdict is ``EMPIRICAL`` —
statistical, valid only under exchangeability, an estimate and NEVER a proof.
``assurance`` is part of the canonical payload, so it is HASHED: silently
upgrading the tier (``empirical`` -> ``proven``) after sealing breaks
``content_hash`` and the signature. The tier cannot be overclaimed in place.

Red-team lesson baked in
------------------------
The canonical payload binds the **human-readable labels** (``predicted_name``,
``conformal_set_names``) as well as the numeric fields. An attacker editing only
a display label would otherwise produce a record whose dashboard shows a forged
class name while the crypto still says OK. Here, any such edit breaks
``content_hash`` (and therefore the HMAC), so the displayed verdict cannot
diverge from the certified one.

Expiry & identity WITHOUT breaking determinism (v2.6.0)
-------------------------------------------------------
Anti-replay needs a certificate to carry a validity window (``not_before`` /
``valid_until``), a unique id (``jti``) and a signing-key id (``key_id``). Naively
hashing those into ``content_hash`` would destroy the determinism property —
identical decisions must keep producing the identical ``content_hash`` for
reproducibility. So those four fields are NOT in :meth:`canonical_payload`.
Instead the SIGNATURE covers a wider :meth:`signing_payload` =
``content_hash | not_before | valid_until | jti | key_id``. The decision hash
stays deterministic, while the temporal/identity fields are TAMPER-EVIDENT: any
edit to them changes the signing payload, so the signature no longer verifies.
:func:`verify_certificate` recomputes the signing payload, checks the signature
over THAT, and additionally rejects an expired or not-yet-valid window.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from . import assurance as _assurance
from . import signing as _signing

CERT_SCHEMA_VERSION = "1.1"  # 1.1: signing_payload binds expiry/identity (v2.6.0)

# Authentication levels (the certificate's `authentication` field).
AUTH_NONE = "NONE"            # integrity hash only — NOT tamper-evident vs. an adversary
AUTH_HMAC = "HMAC-SHA256"     # keyed symmetric signature — forgery-resistant only between
#                              parties who share the secret (and can therefore both forge)
AUTH_ED25519 = "Ed25519"      # keyed asymmetric signature — independently third-party-checkable
#                              with the PUBLIC key alone (the verifier cannot forge)

# Verdicts.
VERDICT_CERTIFIED = "CERTIFIED"   # singleton conformal set -> safe to act on
VERDICT_REVIEW = "REVIEW"         # conformal set has >1 label -> human-in-the-loop
VERDICT_ABSTAIN = "ABSTAIN"       # low confidence -> do not act

ENV_HMAC_KEY = "VERIFY_HMAC_KEY"
ENV_ED25519_SEED = "VERIFY_ED25519_SEED"      # signing seed -> Ed25519 private key (MINTS)
ENV_ED25519_PUBKEY = "VERIFY_ED25519_PUBKEY"  # EXPECTED issuer public key (VERIFIES)


def _hmac_key(explicit: str | bytes | None = None) -> bytes | None:
    """Resolve the HMAC key: explicit arg first, then env ``VERIFY_HMAC_KEY``."""
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    key = os.environ.get(ENV_HMAC_KEY)
    return key.encode() if key else None


def _ed25519_seed(explicit: str | bytes | None = None) -> bytes | None:
    """Resolve the Ed25519 signing seed: explicit arg first, then env ``VERIFY_ED25519_SEED``.

    Returns the raw seed bytes (any length — :mod:`signing` hashes it to 32) or
    ``None`` if no seed is configured. The seed is the MINTING authority; keep it
    secret. The matching public key (safe to publish) comes from
    :func:`signing.ed25519_pubkey_from_seed`.
    """
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    seed = os.environ.get(ENV_ED25519_SEED)
    return seed.encode() if seed else None


def _expected_pubkey(explicit: str | None = None) -> str | None:
    """Resolve the EXPECTED issuer public key: explicit arg, then env ``VERIFY_ED25519_PUBKEY``.

    This is the key obtained OUT-OF-BAND that identifies the trusted issuer.
    Verifying against it (not the embedded key) is what separates genuine
    issuer-authentication from mere internal self-consistency.
    """
    if explicit is not None:
        return explicit
    return os.environ.get(ENV_ED25519_PUBKEY) or None


@dataclass
class Certificate:
    """An auditable record of a single certified prediction.

    ``content_hash`` is computed over a canonical payload that includes the
    human-readable labels. ``cert_id`` and ``timestamp_utc`` are provenance only
    and are NOT hashed, so the same inputs on the same calibrated verifier yield
    an identical ``content_hash`` (determinism), while each emission stays
    uniquely identifiable.
    """

    # --- decision content (all hashed) ---
    predicted_label: int
    predicted_name: str
    conformal_set: list[int]
    conformal_set_names: list[str]
    verdict: str                       # CERTIFIED | REVIEW | ABSTAIN
    alpha: float | None = None
    qhat: float | None = None
    input_sha256: str | None = None    # sha256 of the input signal, if provided
    model_id: str | None = None        # opaque model identifier, if provided
    # Assurance TIER — the STRENGTH of the verdict (see aion_nexus.verify.assurance).
    # Default EMPIRICAL: a conformal verdict is statistical, never a proof. HASHED
    # (in canonical_payload) so a silent overclaim of the tier breaks content_hash.
    assurance: str = _assurance.EMPIRICAL
    # --- provenance (NOT hashed) ---
    schema_version: str = CERT_SCHEMA_VERSION
    cert_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    # --- integrity / authenticity ---
    content_hash: str = ""
    authentication: str = AUTH_NONE
    signature: str | None = None
    # Ed25519 PUBLIC key embedded for convenience (provenance, like ``signature``;
    # NOT in canonical_payload). Verifying against THIS key proves only internal
    # consistency, NOT issuer identity — see verify_certificate / module docstring.
    pubkey: str | None = None
    # --- validity window & identity (v2.6.0; NOT in canonical_payload) ---
    # These are NOT hashed into content_hash (that would break determinism), but
    # the SIGNATURE covers them via signing_payload(), so they are tamper-evident.
    not_before: str | None = None     # ISO-8601: cert invalid before this instant
    valid_until: str | None = None    # ISO-8601: cert invalid (expired) after this
    jti: str | None = None            # unique cert id for anti-replay (uuid4 hex)
    key_id: str | None = None         # opaque id of the signing key (for rotation)

    def canonical_payload(self) -> dict:
        """The exact, order-independent dict that ``content_hash`` is taken over.

        Includes the human-readable labels (red-team lesson) AND the
        ``assurance`` tier (so an in-place overclaim of the verdict's strength
        breaks the hash). Excludes the provenance fields (``cert_id``,
        ``timestamp_utc``, ``schema_version``), the embedded ``pubkey`` (a
        verification aid, not a decided fact), and the integrity/authenticity
        fields themselves, so the hash is deterministic for identical decisions.
        """
        return {
            "predicted_label": int(self.predicted_label),
            "predicted_name": str(self.predicted_name),
            "conformal_set": [int(c) for c in self.conformal_set],
            "conformal_set_names": [str(n) for n in self.conformal_set_names],
            "verdict": str(self.verdict),
            "alpha": None if self.alpha is None else round(float(self.alpha), 10),
            "qhat": None if self.qhat is None else round(float(self.qhat), 10),
            "input_sha256": self.input_sha256,
            "model_id": self.model_id,
            # Tier is HASHED: empirical->proven without re-signing breaks the hash.
            "assurance": str(self.assurance),
        }

    def compute_content_hash(self) -> str:
        """SHA-256 over the canonical payload (deterministic for identical inputs)."""
        return _sha256_canonical(self.canonical_payload())

    def signing_payload(self) -> str:
        """The exact string the SIGNATURE covers (decision hash + expiry + identity).

        ``content_hash | not_before | valid_until | jti | key_id`` (missing fields
        render as empty). This binds the validity window and identity into the
        signature WITHOUT putting them in :meth:`canonical_payload`, so

        - ``content_hash`` stays DETERMINISTIC for identical decisions, while
        - any tamper with ``not_before`` / ``valid_until`` / ``jti`` / ``key_id``
          changes this string and therefore breaks signature verification.

        The separator ``|`` is unambiguous here: ``content_hash`` is hex and the
        timestamps are ISO-8601, neither of which contains ``|``.
        """
        return "|".join((
            self.content_hash,
            self.not_before or "",
            self.valid_until or "",
            self.jti or "",
            self.key_id or "",
        ))

    def seal(self, key: str | bytes | None = None, *, scheme: str = "auto",
             ttl_seconds: int | None = None, key_id: str | None = None,
             now_iso: str | None = None) -> Certificate:
        """Fill ``content_hash`` and (if a key is available) the signature.

        Mutates and returns ``self``. ``scheme`` selects the signature algorithm:

        - ``"auto"`` (default) — resolve by PRECEDENCE: an explicit ``key`` arg
          (treated as an Ed25519 seed) wins; then env ``VERIFY_ED25519_SEED``
          (Ed25519); then env ``VERIFY_HMAC_KEY`` (HMAC-SHA256); else NONE.
        - ``"ed25519"`` — force Ed25519. Seed = explicit ``key`` or
          ``VERIFY_ED25519_SEED``; if neither is set, fall back to NONE (honest:
          we never claim a signature we did not make).
        - ``"hmac"`` — force HMAC. Key = explicit ``key`` or ``VERIFY_HMAC_KEY``;
          if neither is set, NONE.
        - ``"none"`` — integrity hash only, no signature.

        Validity window (anti-replay) — ``ttl_seconds``: when given, set
        ``not_before = now``, ``valid_until = now + ttl_seconds``, and a fresh
        ``jti`` (uuid4). ``key_id`` labels the signing key (for rotation). Pass
        ``now_iso`` to pin "now" for DETERMINISTIC tests (never read the clock in
        a test). These fields are NOT hashed into ``content_hash`` (determinism is
        preserved) but ARE covered by the signature via :meth:`signing_payload`,
        so altering any of them invalidates the signature.

        Ed25519 path: ``authentication=Ed25519``, ``signature`` over the
        SIGNING PAYLOAD (not just ``content_hash``), and the matching ``pubkey``
        embedded so a third party can verify with the public key alone. With no
        key/seed the certificate honestly declares ``authentication=NONE``
        (integrity-only — NOT tamper-evident against an adversary holding this
        source).

        Seed strength: ``seal`` accepts memorable issuer seeds (the existing
        behaviour) so it does NOT enforce the raw-primitive entropy floor — it
        derives the key with the same SHA-256 fold as before, keeping signatures
        and embedded pubkeys byte-compatible with prior releases. To MINT with a
        full-entropy seed instead, generate one with
        :func:`aion_nexus.verify.signing.generate_seed`. The strict floor that
        closes the brute-forceable-seed probe lives on the raw
        :func:`signing.ed25519_sign` / :func:`signing.ed25519_pubkey_from_seed`
        primitives, which reject a weak seed unless ``kdf=True`` is passed.
        """
        self.content_hash = self.compute_content_hash()
        scheme = (scheme or "auto").lower()
        if scheme not in ("auto", "ed25519", "hmac", "none"):
            raise ValueError(
                f"unknown scheme {scheme!r}; expected auto|ed25519|hmac|none")

        # Validity window / identity — set BEFORE signing so the signature covers
        # them. ttl_seconds is the trigger; key_id may be set independently.
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValueError("ttl_seconds must be a positive number of seconds")
            now = _parse_iso(now_iso) if now_iso else datetime.now(timezone.utc)
            self.not_before = _iso(now)
            self.valid_until = _iso(now + timedelta(seconds=int(ttl_seconds)))
            self.jti = uuid.uuid4().hex
        if key_id is not None:
            self.key_id = str(key_id)

        # Reset signature provenance; the chosen path fills back in what it sets.
        self.signature = None
        self.pubkey = None

        if scheme == "none":
            self.authentication = AUTH_NONE
            return self

        seed = _ed25519_seed(key) if scheme in ("auto", "ed25519") else None
        hmac_k = _hmac_key(key) if scheme in ("auto", "hmac") else None

        if scheme == "auto":
            # Explicit precedence: Ed25519 seed (arg or env) > HMAC key (env) > NONE.
            if seed is not None:
                hmac_k = None
            elif hmac_k is not None:
                seed = None
        elif scheme == "ed25519":
            hmac_k = None
        elif scheme == "hmac":
            seed = None

        # The signature covers the FULL signing payload (decision + expiry + id),
        # which is why the temporal/identity fields above are tamper-evident.
        msg = self.signing_payload()
        if seed is not None:
            self.authentication = AUTH_ED25519
            # Legacy-compatible derivation (strict=False is the primitive default):
            # a memorable issuer seed keeps producing the same key as prior
            # releases, so signatures and embedded pubkeys stay byte-compatible.
            # To MINT with full entropy, generate a seed via signing.generate_seed
            # (which clears the floor) and pass strict=True at the call site.
            self.signature = _signing.ed25519_sign(msg, seed)
            self.pubkey = _signing.ed25519_pubkey_from_seed(seed)
        elif hmac_k is not None:
            self.authentication = AUTH_HMAC
            self.signature = hmac.new(
                hmac_k, msg.encode(), hashlib.sha256).hexdigest()
        else:
            self.authentication = AUTH_NONE
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        sset = "{" + ",".join(self.conformal_set_names) + "}"
        return (f"[{self.verdict:9s}] pred={self.predicted_name} "
                f"set={sset} assurance={self.assurance} auth={self.authentication} "
                f"hash={self.content_hash[:10]}")


def _sha256_canonical(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Time helpers — all UTC, ISO-8601, with an injectable "now" for determinism
# --------------------------------------------------------------------------- #

def _iso(dt: datetime) -> str:
    """Render a datetime as a UTC ISO-8601 string (millisecond precision)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _parse_iso(s: str) -> datetime:
    """Parse an ISO-8601 timestamp to an aware UTC datetime.

    Accepts a trailing ``Z`` (treated as ``+00:00``). A naive timestamp is
    assumed to be UTC. Used to compare a certificate's validity window against
    "now"; raises ``ValueError`` on a malformed string (callers pass machine-
    generated ISO, so a bad value is a real error, not silently ignored).
    """
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else \
        dt.astimezone(timezone.utc)


def _signing_payload_from_dict(d: dict, content_hash: str) -> str:
    """Reconstruct :meth:`Certificate.signing_payload` from a plain cert dict.

    MUST mirror the dataclass method exactly so verification recomputes the same
    string the signer covered: ``content_hash | not_before | valid_until | jti |
    key_id`` (missing -> empty). This is how the temporal/identity fields become
    tamper-evident WITHOUT entering ``content_hash`` (which stays deterministic).
    """
    return "|".join((
        content_hash,
        str(d.get("not_before") or ""),
        str(d.get("valid_until") or ""),
        str(d.get("jti") or ""),
        str(d.get("key_id") or ""),
    ))


def _validity_window(d: dict, now_iso: str | None) -> tuple[bool, bool, str]:
    """Return ``(expired, not_yet_valid, note)`` for a cert's validity window.

    Compares ``now`` (``now_iso`` if given, else the real clock — pass ``now_iso``
    in tests for determinism) against ``valid_until`` / ``not_before``. A cert
    with neither field is timeless: ``(False, False, "")``. A malformed window
    timestamp is treated as a FAIL-CLOSED condition (expired=True) rather than
    silently trusted.
    """
    nb = d.get("not_before")
    vu = d.get("valid_until")
    if not nb and not vu:
        return False, False, ""
    now = _parse_iso(now_iso) if now_iso else datetime.now(timezone.utc)
    try:
        if vu and now > _parse_iso(str(vu)):
            return True, False, f"expired (now > valid_until={vu})"
        if nb and now < _parse_iso(str(nb)):
            return False, True, f"not yet valid (now < not_before={nb})"
    except ValueError:
        # Unparseable window -> fail closed: never trust a cert we cannot time-check.
        return True, False, "malformed validity window -> treated as expired"
    return False, False, f"within validity window [{nb or '-'} .. {vu or '-'}]"


def verify_certificate(cert, key: str | bytes | None = None, *,
                       expected_pubkey: str | None = None,
                       now_iso: str | None = None) -> dict:
    """Audit a certificate's integrity, authenticity AND validity window.

    Accepts a :class:`Certificate` or its ``as_dict()``. Returns::

        {"integrity_ok": bool,     # content_hash recomputes from the public payload
         "authenticity": str,      # "VERIFIED" | "SELF-SIGNED" | "UNVERIFIED" | "FORGED"
         "trusted": bool,          # integrity_ok AND VERIFIED AND in-window
         "expired": bool,          # (present only when a validity window was set)
         "not_yet_valid": bool,    # (present only when a validity window was set)
         "detail": str}

    Always gate trust on ``trusted`` (or the explicit conjunction), never on
    ``authenticity`` alone: a label-only tamper that leaves ``content_hash``
    unchanged would still authenticate, so ``authenticity`` in isolation is a
    foot-gun. ``trusted`` is the single safe flag, and it is ``True`` ONLY for
    ``"VERIFIED"`` — never for ``"SELF-SIGNED"``, ``"UNVERIFIED"`` or ``"FORGED"``.

    The authentication scheme is read from the certificate's ``authentication``
    field (NONE / HMAC-SHA256 / Ed25519):

    - **NONE** -> ``"UNVERIFIED"`` (bytes internally consistent only; an adversary
      with the source could have produced them — honestly NOT tamper-evidence).
    - **HMAC-SHA256** -> recompute the HMAC over the SIGNING PAYLOAD with ``key``
      (arg or env ``VERIFY_HMAC_KEY``). No key -> ``"UNVERIFIED"``. Match ->
      ``"VERIFIED"``; mismatch / missing signature -> ``"FORGED"``.
    - **Ed25519** -> verify the signature over the SIGNING PAYLOAD:

      * against the EXPECTED issuer key (``expected_pubkey`` arg or env
        ``VERIFY_ED25519_PUBKEY``) when one is supplied: valid -> ``"VERIFIED"``;
        invalid -> ``"FORGED"``.
      * when NO expected key is supplied, fall back to the certificate's EMBEDDED
        ``pubkey``: a valid signature proves only INTERNAL CONSISTENCY (the cert
        is self-consistent), NOT that it came from the trusted issuer — an
        attacker can mint their own seed and embed their own pubkey. This returns
        ``"SELF-SIGNED"`` (NOT ``"VERIFIED"``), and ``trusted`` stays ``False``.
        Supply ``expected_pubkey`` to authenticate the issuer.

    Validity window (anti-replay, v2.6.0): the signature covers the SIGNING
    PAYLOAD = ``content_hash | not_before | valid_until | jti | key_id``, so those
    fields are tamper-evident (altering ``valid_until`` breaks the signature ->
    ``"FORGED"``) even though they are NOT in ``content_hash`` (which stays
    deterministic). Independently, if ``now`` is past ``valid_until`` the result
    carries ``expired=True`` and ``trusted=False``; if before ``not_before``,
    ``not_yet_valid=True`` and ``trusted=False``. Pass ``now_iso`` to pin "now"
    for deterministic tests; the default reads the real UTC clock.

    ``integrity_ok`` also fails if the human-readable labels OR the ``assurance``
    tier were tampered: the payload that ``content_hash`` is taken over includes
    ``predicted_name``, ``conformal_set_names`` and ``assurance``, so editing only
    a label — or silently upgrading the tier — breaks the hash.
    """
    d = cert.as_dict() if hasattr(cert, "as_dict") else dict(cert)
    payload = {
        "predicted_label": int(d["predicted_label"]),
        "predicted_name": str(d["predicted_name"]),
        "conformal_set": [int(c) for c in d["conformal_set"]],
        "conformal_set_names": [str(n) for n in d["conformal_set_names"]],
        "verdict": str(d["verdict"]),
        "alpha": None if d.get("alpha") is None else round(float(d["alpha"]), 10),
        "qhat": None if d.get("qhat") is None else round(float(d["qhat"]), 10),
        "input_sha256": d.get("input_sha256"),
        "model_id": d.get("model_id"),
        # Tier is part of the hashed payload (anti-overclaim). Default EMPIRICAL
        # keeps pre-2.5 certificates (no field) recomputing to the same hash only
        # if they were sealed with that default — older certs carry it explicitly.
        "assurance": str(d.get("assurance", _assurance.EMPIRICAL)),
    }
    integrity_ok = _sha256_canonical(payload) == d.get("content_hash")
    content_hash = str(d.get("content_hash", ""))
    # The signature covers the FULL signing payload (decision + expiry + identity).
    signed_msg = _signing_payload_from_dict(d, content_hash)
    expired, not_yet_valid, window_note = _validity_window(d, now_iso)
    in_window = not (expired or not_yet_valid)
    sig = d.get("signature")
    auth = d.get("authentication", AUTH_NONE)

    if auth == AUTH_ED25519:
        res = _verify_ed25519(
            d, integrity_ok, signed_msg, sig, expected_pubkey)
        return _apply_window(res, d, expired, not_yet_valid, in_window, window_note)

    # ---- HMAC-SHA256 / NONE path ----
    k = _hmac_key(key)
    if k is None:
        res = {
            "integrity_ok": integrity_ok,
            "authenticity": "UNVERIFIED",
            "trusted": False,  # no key -> never tamper-evident, so never trusted
            "detail": ("no key available: integrity %s, authenticity UNVERIFIED "
                       "(content_hash proves consistency, NOT authenticity; set "
                       "VERIFY_HMAC_KEY to authenticate)"
                       % ("OK" if integrity_ok else "FAILED")),
        }
        return _apply_window(res, d, expired, not_yet_valid, in_window, window_note)

    # HMAC now covers the SIGNING PAYLOAD (decision + expiry + identity), so a
    # tamper with the validity window breaks the HMAC -> FORGED.
    expected = hmac.new(k, signed_msg.encode(), hashlib.sha256).hexdigest()
    if sig and hmac.compare_digest(str(sig), expected):
        authenticity = "VERIFIED"
    else:
        authenticity = "FORGED"
    res = {
        "integrity_ok": integrity_ok,
        "authenticity": authenticity,
        # Single safe flag: a label-only tamper keeps the HMAC valid but breaks
        # integrity, so trust must require BOTH (and an in-window cert; see
        # _apply_window).
        "trusted": bool(integrity_ok and authenticity == "VERIFIED"),
        "detail": f"integrity {'OK' if integrity_ok else 'FAILED'}, "
                  f"HMAC authenticity {authenticity}",
    }
    return _apply_window(res, d, expired, not_yet_valid, in_window, window_note)


def _apply_window(res: dict, d: dict, expired: bool, not_yet_valid: bool,
                  in_window: bool, window_note: str) -> dict:
    """Fold the validity-window verdict into a base authenticity result.

    A certificate with no window is unchanged. Otherwise ``expired`` /
    ``not_yet_valid`` are surfaced as explicit flags and FORCE ``trusted=False``
    (a signature can be perfectly valid yet the cert out of its allowed window —
    the anti-replay guard). The note is appended to ``detail`` for auditability.
    """
    if not d.get("not_before") and not d.get("valid_until"):
        return res  # timeless certificate — nothing to add
    res["expired"] = expired
    res["not_yet_valid"] = not_yet_valid
    if not in_window:
        res["trusted"] = False
    res["detail"] = f"{res['detail']}; {window_note}"
    return res


def _verify_ed25519(d: dict, integrity_ok: bool, signed_msg: str,
                    sig: str | None, expected_pubkey: str | None) -> dict:
    """Ed25519 leg of :func:`verify_certificate` — issuer-aware, fail-safe.

    Verifies the signature over the SIGNING PAYLOAD (``signed_msg`` = content_hash
    + expiry + identity), so a tamper with the validity window or key_id breaks
    the signature.

    Honesty rule (workspace 6.31): a signature checked against the EMBEDDED key
    only proves self-consistency. Genuine issuer authentication requires an
    EXPECTED key obtained out-of-band, so the two cases return DIFFERENT verdicts
    and only the expected-key match yields ``trusted=True``.
    """
    expected = _expected_pubkey(expected_pubkey)
    embedded = d.get("pubkey")

    if expected is not None:
        ok = _signing.ed25519_verify(signed_msg, str(sig or ""), expected)
        authenticity = "VERIFIED" if ok else "FORGED"
        detail = (
            f"integrity {'OK' if integrity_ok else 'FAILED'}, Ed25519 authenticity "
            f"{authenticity} against EXPECTED issuer pubkey")
        return {
            "integrity_ok": integrity_ok,
            "authenticity": authenticity,
            "trusted": bool(integrity_ok and authenticity == "VERIFIED"),
            "detail": detail,
        }

    # No expected key: we can only check self-consistency against the embedded key.
    if embedded and _signing.ed25519_verify(signed_msg, str(sig or ""), embedded):
        return {
            "integrity_ok": integrity_ok,
            "authenticity": "SELF-SIGNED",
            # NOT trusted: self-consistent, issuer NOT verified.
            "trusted": False,
            "detail": (
                f"integrity {'OK' if integrity_ok else 'FAILED'}, Ed25519 signature "
                "self-consistent against the EMBEDDED pubkey, but issuer NOT verified "
                "— provide expected_pubkey (or set VERIFY_ED25519_PUBKEY) to "
                "authenticate the issuer"),
        }

    # Missing/invalid signature, or no embedded key to check against.
    return {
        "integrity_ok": integrity_ok,
        "authenticity": "FORGED",
        "trusted": False,
        "detail": (
            f"integrity {'OK' if integrity_ok else 'FAILED'}, Ed25519 signature "
            "missing or invalid against the embedded pubkey"),
    }


def require_authenticated(cert) -> None:
    """Raise if a CERTIFIED certificate is unsigned (``authentication == NONE``).

    A safe-by-default GATE for callers that must refuse to act on an unsigned
    "CERTIFIED" verdict — exactly the strict mode a serving layer enables so an
    integrity-only (NOT tamper-evident) certificate can never drive an automated
    action. This is a HELPER, not forced behaviour: ``seal``/``verify`` still
    return honest results for unsigned certs; this just lets a caller turn the
    NONE case into a hard error.

    Raises ``ValueError`` only for the dangerous combination ``verdict ==
    CERTIFIED`` AND ``authentication == NONE``. REVIEW / ABSTAIN verdicts (which
    do not authorise autonomous action) and any signed certificate pass through.
    Note this checks the DECLARED scheme, not the signature's validity — pair it
    with :func:`verify_certificate` (gate on ``trusted``) for full assurance.
    """
    d = cert.as_dict() if hasattr(cert, "as_dict") else dict(cert)
    verdict = str(d.get("verdict", ""))
    auth = d.get("authentication", AUTH_NONE)
    if verdict == VERDICT_CERTIFIED and auth == AUTH_NONE:
        raise ValueError(
            "strict mode: refusing an unsigned CERTIFIED certificate "
            "(authentication=NONE is integrity-only, NOT tamper-evident). Sign it "
            "with an Ed25519 seed or VERIFY_HMAC_KEY, or do not act on it.")


def sha256_signal(signal) -> str:
    """Deterministic SHA-256 over an input signal (numpy array or bytes-like).

    Used to bind a certificate to the exact input it certifies.
    """
    h = hashlib.sha256()
    try:
        import numpy as np

        arr = np.ascontiguousarray(np.asarray(signal))
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    except Exception:  # noqa: BLE001 — last-resort fallback for non-array inputs
        h.update(repr(signal).encode())
    return h.hexdigest()
