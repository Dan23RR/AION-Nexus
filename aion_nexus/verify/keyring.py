"""Key registry — rotation + revocation for the signing keys (custody, v2.20.0).

Offline-verifiable certificates are only as trustworthy as the CUSTODY of their
signing keys. The signing module already lets the private key live in a KMS/HSM
(:class:`~aion_nexus.verify.signing.ExternalSigner`); this module closes the other
half: a publishable **key registry** (no secrets) that supports ROTATION (retire a
key, activate a successor) and REVOCATION (a compromised/retired key invalidates
ONLY its own certificates, not the whole issuer). It is a transparency log: the
``KeyRecord`` set is safe to publish so any third party can resolve a cert's
``key_id`` to the expected public key and check it was not revoked.

    ring = KeyRing().rotate("2026-q3", pub_q3)          # active key
    cert = verifier.certify(..., signer=ExternalSigner(pub_q3, kms_sign), key_id="2026-q3")
    verify_with_keyring(cert, ring)                      # trusted iff key active + sig valid
    ring.revoke("2026-q3", "seed exposed in incident-217")
    verify_with_keyring(cert, ring)                      # now trusted=False (REVOKED-KEY)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .certificate import verify_certificate

KEY_ACTIVE = "active"
KEY_RETIRED = "retired"      # rotated out; its existing certs remain valid unless revoked
KEY_REVOKED = "revoked"      # compromised/withdrawn; ALL its certs are untrusted


@dataclass(frozen=True)
class KeyRecord:
    """One signing key's public half + lifecycle status (no secret material)."""

    key_id: str
    public_key: str            # hex Ed25519 public key (publishable)
    status: str = KEY_ACTIVE
    created: str | None = None  # ISO-8601, optional
    not_after: str | None = None
    reason: str | None = None   # revocation/retirement reason


class KeyRing:
    """A publishable registry of signing keys with rotation + revocation."""

    def __init__(self) -> None:
        self._keys: dict[str, KeyRecord] = {}
        self._order: list[str] = []   # registration order (for active resolution)

    # ---- registration / rotation / revocation --------------------------- #

    def register(self, key_id: str, public_key: str, *, status: str = KEY_ACTIVE,
                 created: str | None = None) -> KeyRing:
        if not key_id or not public_key:
            raise ValueError("key_id and public_key are required")
        if status not in (KEY_ACTIVE, KEY_RETIRED, KEY_REVOKED):
            raise ValueError(f"unknown status {status!r}")
        if key_id not in self._keys:
            self._order.append(key_id)
        self._keys[key_id] = KeyRecord(key_id, str(public_key), status, created)
        return self

    def rotate(self, key_id: str, public_key: str, *, created: str | None = None) -> KeyRing:
        """Retire every currently-active key and register ``key_id`` as the new active."""
        for kid, rec in list(self._keys.items()):
            if rec.status == KEY_ACTIVE:
                self._keys[kid] = KeyRecord(rec.key_id, rec.public_key, KEY_RETIRED,
                                            rec.created, rec.not_after, "rotated out")
        return self.register(key_id, public_key, status=KEY_ACTIVE, created=created)

    def revoke(self, key_id: str, reason: str) -> KeyRing:
        rec = self._keys.get(key_id)
        if rec is None:
            raise KeyError(f"unknown key_id {key_id!r}")
        self._keys[key_id] = KeyRecord(rec.key_id, rec.public_key, KEY_REVOKED,
                                       rec.created, rec.not_after, str(reason))
        return self

    # ---- lookup --------------------------------------------------------- #

    def get(self, key_id: str) -> KeyRecord | None:
        return self._keys.get(key_id)

    def public_key_for(self, key_id: str) -> str | None:
        rec = self._keys.get(key_id)
        return rec.public_key if rec else None

    def is_revoked(self, key_id: str) -> bool:
        rec = self._keys.get(key_id)
        return rec is not None and rec.status == KEY_REVOKED

    def active(self) -> KeyRecord | None:
        for kid in reversed(self._order):
            rec = self._keys[kid]
            if rec.status == KEY_ACTIVE:
                return rec
        return None

    # ---- persistence (publishable transparency log; no secrets) --------- #

    def to_dict(self) -> dict:
        return {"keys": [asdict(self._keys[k]) for k in self._order]}

    @classmethod
    def from_dict(cls, data: dict) -> KeyRing:
        """Reconstruct a ring FAITHFULLY from its ``to_dict()`` form.

        Restores EVERY persisted field (status, created, ``not_after``, ``reason``)
        for ALL keys — not just revoked ones — so a retired key keeps the
        why/when of its retirement. A transparency log that dropped those on
        save/load would not be auditable, which is the whole point of the registry.
        """
        ring = cls()
        valid = {f.name for f in fields(KeyRecord)}
        for k in data.get("keys", []):
            kid = k["key_id"]
            if kid not in ring._keys:
                ring._order.append(kid)
            ring._keys[kid] = KeyRecord(**{n: v for n, v in k.items() if n in valid})
        return ring

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> KeyRing:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def verify_with_keyring(cert, keyring: KeyRing, *, now_iso: str | None = None) -> dict:
    """Verify a certificate against a KEY REGISTRY, enforcing rotation + revocation.

    Resolves the cert's ``key_id`` to its registered public key and verifies the
    signature against it (genuine issuer authentication — not the self-signed
    fallback). Then DOWNGRADES trust on a custody failure:

    - ``key_id`` absent from the ring  -> ``authenticity="UNKNOWN-KEY"``, trusted=False
      (cannot authenticate against a registered issuer).
    - key REVOKED                       -> ``authenticity="REVOKED-KEY"``, trusted=False
      (only THIS key's certs are invalidated; other keys stay valid).

    A RETIRED key still trusts its previously-minted certs (rotation does not
    retroactively invalidate). Returns the :func:`verify_certificate` dict plus
    ``key_id`` / ``key_status`` / ``key_note``.
    """
    d = cert.as_dict() if hasattr(cert, "as_dict") else dict(cert)
    key_id = d.get("key_id") or None
    rec = keyring.get(key_id) if key_id else None

    if rec is None:
        res = verify_certificate(cert, now_iso=now_iso)   # falls back to embedded pubkey
        res["authenticity"] = "UNKNOWN-KEY"
        res["trusted"] = False
        res["key_id"] = key_id
        res["key_status"] = None
        res["key_note"] = ("key_id is not in the registry; cannot authenticate against a "
                           "known issuer (resolved only the cert's embedded pubkey).")
        return res

    res = verify_certificate(cert, expected_pubkey=rec.public_key, now_iso=now_iso)
    res["key_id"] = key_id
    res["key_status"] = rec.status
    res["key_note"] = ""
    if rec.status == KEY_REVOKED:
        res["authenticity"] = "REVOKED-KEY"
        res["trusted"] = False
        res["key_note"] = f"signing key REVOKED: {rec.reason or 'no reason given'}"
    elif rec.status == KEY_RETIRED:
        res["key_note"] = "signing key retired (rotated out); existing certs remain valid"
    return res
