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

Red-team lesson baked in
------------------------
The canonical payload binds the **human-readable labels** (``predicted_name``,
``conformal_set_names``) as well as the numeric fields. An attacker editing only
a display label would otherwise produce a record whose dashboard shows a forged
class name while the crypto still says OK. Here, any such edit breaks
``content_hash`` (and therefore the HMAC), so the displayed verdict cannot
diverge from the certified one.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

CERT_SCHEMA_VERSION = "1.0"

# Authentication levels (the certificate's `authentication` field).
AUTH_NONE = "NONE"            # integrity hash only — NOT tamper-evident vs. an adversary
AUTH_HMAC = "HMAC-SHA256"     # keyed signature present — forgery-resistant

# Verdicts.
VERDICT_CERTIFIED = "CERTIFIED"   # singleton conformal set -> safe to act on
VERDICT_REVIEW = "REVIEW"         # conformal set has >1 label -> human-in-the-loop
VERDICT_ABSTAIN = "ABSTAIN"       # low confidence -> do not act

ENV_HMAC_KEY = "VERIFY_HMAC_KEY"


def _hmac_key(explicit: str | bytes | None = None) -> bytes | None:
    """Resolve the HMAC key: explicit arg first, then env ``VERIFY_HMAC_KEY``."""
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    key = os.environ.get(ENV_HMAC_KEY)
    return key.encode() if key else None


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
    # --- provenance (NOT hashed) ---
    schema_version: str = CERT_SCHEMA_VERSION
    cert_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
    # --- integrity / authenticity ---
    content_hash: str = ""
    authentication: str = AUTH_NONE
    signature: str | None = None

    def canonical_payload(self) -> dict:
        """The exact, order-independent dict that ``content_hash`` is taken over.

        Includes the human-readable labels (red-team lesson). Excludes the
        provenance fields (``cert_id``, ``timestamp_utc``, ``schema_version``)
        and the integrity/authenticity fields themselves, so the hash is
        deterministic for identical decisions.
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
        }

    def compute_content_hash(self) -> str:
        """SHA-256 over the canonical payload (deterministic for identical inputs)."""
        return _sha256_canonical(self.canonical_payload())

    def seal(self, key: str | bytes | None = None) -> Certificate:
        """Fill ``content_hash`` and (if a key is available) the HMAC signature.

        Mutates and returns ``self``. With a key (explicit or ``VERIFY_HMAC_KEY``)
        sets ``authentication=HMAC-SHA256`` and ``signature``; otherwise
        ``authentication=NONE`` and ``signature=None`` (honest: integrity-only).
        """
        self.content_hash = self.compute_content_hash()
        k = _hmac_key(key)
        if k is None:
            self.authentication = AUTH_NONE
            self.signature = None
        else:
            self.authentication = AUTH_HMAC
            self.signature = hmac.new(k, self.content_hash.encode(), hashlib.sha256).hexdigest()
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        sset = "{" + ",".join(self.conformal_set_names) + "}"
        return (f"[{self.verdict:9s}] pred={self.predicted_name} "
                f"set={sset} auth={self.authentication} hash={self.content_hash[:10]}")


def _sha256_canonical(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def verify_certificate(cert, key: str | bytes | None = None) -> dict:
    """Audit a certificate's integrity and authenticity. Two honest outcomes.

    Accepts a :class:`Certificate` or its ``as_dict()``. ``key`` defaults to env
    ``VERIFY_HMAC_KEY``. Returns::

        {"integrity_ok": bool,     # content_hash recomputes from the public payload
         "authenticity": str,      # "VERIFIED" | "UNVERIFIED" | "FORGED"
         "trusted": bool,          # integrity_ok AND authenticity == "VERIFIED"
         "detail": str}

    Always gate trust on ``trusted`` (or the explicit conjunction), never on
    ``authenticity`` alone: a label-only tamper that leaves ``content_hash``
    unchanged would still authenticate, so ``authenticity`` in isolation is a
    foot-gun. ``trusted`` is the single safe flag.

    - No key available -> authenticity ``"UNVERIFIED"`` (we can only confirm the
      bytes are internally consistent; an adversary with the source could have
      produced them — this is honestly NOT tamper-evidence).
    - Key available -> recompute the HMAC over ``content_hash`` with
      ``compare_digest``: match -> ``"VERIFIED"``; mismatch / missing /
      declared-NONE signature -> ``"FORGED"``.

    ``integrity_ok`` also fails if the human-readable labels were tampered: the
    payload that ``content_hash`` is taken over includes ``predicted_name`` and
    ``conformal_set_names``, so editing only a label breaks the hash.
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
    }
    integrity_ok = _sha256_canonical(payload) == d.get("content_hash")

    k = _hmac_key(key)
    if k is None:
        return {
            "integrity_ok": integrity_ok,
            "authenticity": "UNVERIFIED",
            "trusted": False,  # no key -> never tamper-evident, so never trusted
            "detail": ("no key available: integrity %s, authenticity UNVERIFIED "
                       "(content_hash proves consistency, NOT authenticity; set "
                       "VERIFY_HMAC_KEY to authenticate)"
                       % ("OK" if integrity_ok else "FAILED")),
        }

    expected = hmac.new(k, str(d.get("content_hash", "")).encode(),
                        hashlib.sha256).hexdigest()
    sig = d.get("signature")
    if sig and hmac.compare_digest(str(sig), expected):
        authenticity = "VERIFIED"
    else:
        authenticity = "FORGED"
    return {
        "integrity_ok": integrity_ok,
        "authenticity": authenticity,
        # Single safe flag: a label-only tamper keeps the HMAC valid but breaks
        # integrity, so trust must require BOTH.
        "trusted": bool(integrity_ok and authenticity == "VERIFIED"),
        "detail": f"integrity {'OK' if integrity_ok else 'FAILED'}, "
                  f"HMAC authenticity {authenticity}",
    }


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
