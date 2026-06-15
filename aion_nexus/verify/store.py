"""Append-only JSONL certificate store with an optional tamper-evident chain.

One JSON line per certificate (an audit trail for enterprise / EU AI Act Art.12
logging evidence). Each record carries ``prev_hash`` (the previous record's
``record_hash``, genesis = 64 zeros) and ``record_hash`` (a digest over the
canonical record including ``prev_hash``), forming a hash chain.

Integrity vs. authenticity — what ``verify_chain()`` actually proves
--------------------------------------------------------------------
The chain link uses one of THREE schemes, resolved by precedence at append time
and recorded per-record in ``chain_auth`` so the store is honest about its own
guarantee:

* ``chain_auth=NONE`` (no key/seed): plain SHA-256. ``verify_chain()`` only
  detects ACCIDENTAL corruption (a flipped byte, a truncated line). It is NOT
  tamper-evident against an adversary who has this source — they can edit a
  record and recompute the whole chain. Do not claim tamper-evidence here.
* ``chain_auth=HMAC-SHA256`` (env ``VERIFY_HMAC_KEY`` or explicit key): the link
  is keyed symmetrically. An adversary without the secret cannot re-concatenate a
  forged record. Detects in-place edits, reorders and deletions — but to AUDIT
  the store you need the same secret, with which you could also forge it.
* ``chain_auth=Ed25519`` (env ``VERIFY_ED25519_SEED`` or explicit seed): the link
  is an Ed25519 signature over the canonical record. The store embeds the PUBLIC
  key once (genesis ``chain_pubkey``); a third party VERIFIES the whole chain
  with the public key ALONE — without any secret that would let them forge it.
  This is the asymmetric analogue of the HMAC path.

Precedence at append: explicit seed (Ed25519) > env ``VERIFY_ED25519_SEED`` >
explicit key / env ``VERIFY_HMAC_KEY`` (HMAC) > NONE.

DOWNGRADE GUARD (unchanged and extended): when a key/seed IS supplied to
``verify_chain``, a record declaring a WEAKER ``chain_auth`` than the configured
scheme fails closed — ``chain_auth`` is unauthenticated, so a NONE record under a
configured key/seed IS the strip-and-re-link attack.

Path resolution: explicit ``path`` arg > env ``VERIFY_CERT_STORE`` >
``./certificates.jsonl``. Appends are serialized with a lock; the store resumes
an existing chain on restart.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from collections.abc import Iterator
from pathlib import Path

from . import signing as _signing

GENESIS_HASH = "0" * 64
ENV_HMAC_KEY = "VERIFY_HMAC_KEY"
ENV_ED25519_SEED = "VERIFY_ED25519_SEED"
ENV_CERT_STORE = "VERIFY_CERT_STORE"

# chain_auth values.
CHAIN_NONE = "NONE"
CHAIN_HMAC = "HMAC-SHA256"
CHAIN_ED25519 = "Ed25519"


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _chain_key(explicit: str | bytes | None = None) -> bytes | None:
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    key = os.environ.get(ENV_HMAC_KEY)
    return key.encode() if key else None


def _chain_seed(explicit: str | bytes | None = None) -> bytes | None:
    """Resolve the Ed25519 chain seed: explicit arg, then env ``VERIFY_ED25519_SEED``."""
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    seed = os.environ.get(ENV_ED25519_SEED)
    return seed.encode() if seed else None


def _link_digest(canonical: str, key: bytes | None) -> str:
    """Keyed (HMAC-SHA256) link if a key is present, else plain SHA-256."""
    blob = canonical.encode()
    if key is None:
        return hashlib.sha256(blob).hexdigest()
    return hmac.new(key, blob, hashlib.sha256).hexdigest()


class ChainResult:
    """Result of :meth:`CertificateStore.verify_chain`.

    Truthy iff the chain links are intact (so ``if store.verify_chain():`` keeps
    working). Attributes:

    - ``integrity_ok``: links recompute (no accidental corruption / break).
    - ``authenticity``: "VERIFIED" (keyed, forgery-resistant), "UNVERIFIED"
      (integrity-only — no key, or HMAC records that could not be key-checked),
      or "FORGED" (a keyed mismatch / downgrade was detected).
    - ``broken_at``: index of the first bad record, else None.
    """

    __slots__ = ("integrity_ok", "authenticity", "broken_at", "detail")

    def __init__(self, integrity_ok: bool, authenticity: str,
                 broken_at: int | None, detail: str) -> None:
        self.integrity_ok = integrity_ok
        self.authenticity = authenticity
        self.broken_at = broken_at
        self.detail = detail

    def __bool__(self) -> bool:
        return self.integrity_ok

    def as_tuple(self) -> tuple[bool, str, int | None]:
        """``(integrity_ok, authenticity, broken_at)`` — convenient for tests."""
        return self.integrity_ok, self.authenticity, self.broken_at

    def __iter__(self):
        return iter(self.as_tuple())

    def __repr__(self) -> str:
        return (f"ChainResult(integrity_ok={self.integrity_ok}, "
                f"authenticity={self.authenticity!r}, broken_at={self.broken_at}, "
                f"detail={self.detail!r})")


class CertificateStore:
    """Append-only JSONL certificate log with an optional tamper-evident chain."""

    def __init__(self, path: str | os.PathLike | None = None, chain: bool = True, *,
                 seed: str | bytes | None = None) -> None:
        self.path = Path(path or os.environ.get(ENV_CERT_STORE, "./certificates.jsonl"))
        self.chain = chain
        # An Ed25519 seed (explicit or env) makes the chain link asymmetric and
        # third-party-verifiable. None falls back to the HMAC / SHA-256 path.
        self.seed = seed
        self._lock = threading.Lock()
        self._prev_hash = self._resume_chain() if chain else None

    def _resume_chain(self) -> str:
        """Continue an existing chain across restarts (last record_hash on disk)."""
        last_line = None
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line
        if last_line is None:
            return GENESIS_HASH
        return json.loads(last_line).get("record_hash", GENESIS_HASH)

    def append(self, cert, key: str | bytes | None = None,
               seed: str | bytes | None = None) -> dict:
        """Append one certificate (a ``Certificate`` or a plain dict).

        Returns the stored record (certificate fields + ``prev_hash`` /
        ``record_hash`` / ``chain_auth`` if chained). The chain link scheme is
        resolved by PRECEDENCE: an Ed25519 seed (arg ``seed``, the store's
        configured ``seed``, or env ``VERIFY_ED25519_SEED``) > an HMAC key (arg
        ``key`` or env ``VERIFY_HMAC_KEY``) > plain SHA-256 (NONE). For the
        Ed25519 path the record also carries ``chain_pubkey`` (the public key,
        safe to publish) so the chain is verifiable by a third party with the
        public key alone.
        """
        record = dict(cert.as_dict()) if hasattr(cert, "as_dict") else dict(cert)
        with self._lock:
            if self.chain:
                sd = _chain_seed(seed if seed is not None else self.seed)
                k = None if sd is not None else _chain_key(key)
                record["prev_hash"] = self._prev_hash
                if sd is not None:
                    record["chain_auth"] = CHAIN_ED25519
                    record["chain_pubkey"] = _signing.ed25519_pubkey_from_seed(sd)
                elif k is not None:
                    record["chain_auth"] = CHAIN_HMAC
                else:
                    record["chain_auth"] = CHAIN_NONE
                canonical = _canonical(
                    {kk: vv for kk, vv in record.items() if kk != "record_hash"})
                if sd is not None:
                    record["record_hash"] = _signing.ed25519_sign(canonical, sd)
                else:
                    record["record_hash"] = _link_digest(canonical, k)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(_canonical(record) + "\n")
            if self.chain:
                self._prev_hash = record["record_hash"]
        return record

    def iter_certs(self) -> Iterator[dict]:
        """Yield stored records in append order (empty if no file yet)."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def verify_chain(self, key: str | bytes | None = None, *,
                     seed: str | bytes | None = None,
                     pubkey: str | None = None) -> ChainResult:
        """Verify the chain links back to genesis. Returns a :class:`ChainResult`
        (truthy iff intact).

        Each record's link is recomputed with the scheme its own ``chain_auth``
        declares (NONE / HMAC-SHA256 / Ed25519). The configured verification
        secret is resolved by precedence: an Ed25519 ``seed`` (arg, store, or env
        ``VERIFY_ED25519_SEED``) OR an expected ``pubkey`` (arg) > an HMAC ``key``
        (arg or env ``VERIFY_HMAC_KEY``) > keyless (integrity-only).

        Ed25519 records:
          * If an expected ``pubkey`` is supplied, every Ed25519 link is verified
            against THAT key -> "VERIFIED" (genuine third-party authentication) or
            "FORGED". This is the honest mode for an external auditor.
          * Else if a ``seed`` is configured, the link is verified against the
            public key derived from the seed (the issuer auditing its own store).
          * Else (neither): the link is verified against each record's EMBEDDED
            ``chain_pubkey``. A valid signature then proves only self-consistency,
            NOT issuer identity, so the chain is reported "UNVERIFIED" (integrity
            only) — exactly the certificate's SELF-SIGNED honesty, at chain level.

        DOWNGRADE GUARD: when a key/seed/pubkey IS supplied, a record declaring a
        weaker ``chain_auth`` than the configured scheme fails closed.
        ``chain_auth`` is unauthenticated, so under a configured secret a NONE (or
        cross-scheme) record IS the strip-and-re-link attack. Consequence
        (intended): a store legitimately created WITHOUT a secret cannot later be
        audited by passing one — audit it keyless (integrity-only by construction).
        """
        k = _chain_key(key)
        sd = _chain_seed(seed if seed is not None else self.seed)
        # The public key the auditor verifies AGAINST: an explicit expected key
        # wins; otherwise derive it from the configured seed (issuer self-audit).
        expected_pub = pubkey or (
            _signing.ed25519_pubkey_from_seed(sd) if sd is not None else None)
        asymmetric = expected_pub is not None

        prev = GENESIS_HASH
        all_keyed_checked = True
        any_record = False
        for i, record in enumerate(self.iter_certs()):
            any_record = True
            stored_hash = record.pop("record_hash", None)
            if record.get("prev_hash") != prev:
                bad = (k is not None) or asymmetric
                return ChainResult(
                    False, "FORGED" if bad else "UNVERIFIED", i,
                    f"record {i}: prev_hash does not link to record {i - 1}")
            declared = record.get("chain_auth", "NONE")
            canonical = _canonical(record)

            if declared == CHAIN_ED25519:
                # Verify against the EXPECTED key if we have one; else fall back to
                # the record's embedded key (self-consistent, NOT issuer-verified).
                verify_pub = expected_pub or record.get("chain_pubkey")
                if not _signing.ed25519_verify(canonical, str(stored_hash or ""),
                                               str(verify_pub or "")):
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: Ed25519 signature mismatch (tamper detected)")
                if not asymmetric:
                    # Embedded-key only: self-consistent, issuer NOT authenticated.
                    all_keyed_checked = False
            elif declared == CHAIN_HMAC:
                if asymmetric:
                    # Downgrade: an Ed25519 auditor never accepts an HMAC record.
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: chain_auth=HMAC-SHA256 but Ed25519 verification "
                        "is configured -- scheme downgrade (tamper detected)")
                if k is None:
                    all_keyed_checked = False           # cannot key-check; structure only
                    prev = stored_hash
                    continue
                expected = hmac.new(k, canonical.encode(), hashlib.sha256).hexdigest()
                if stored_hash is None or not hmac.compare_digest(stored_hash, expected):
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: HMAC record_hash mismatch (tamper detected)")
            else:  # plain SHA-256 link (chain_auth=NONE)
                all_keyed_checked = False
                expected = hashlib.sha256(canonical.encode()).hexdigest()
                if stored_hash is None or expected != stored_hash:
                    return ChainResult(
                        False, "UNVERIFIED", i,
                        f"record {i}: SHA-256 record_hash mismatch (corruption)")
                if k is not None or asymmetric:
                    # Downgrade attack: a secret-holder never writes NONE records, so
                    # a NONE record means the keyed link was stripped and re-linked.
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: chain_auth=NONE but a key/seed is configured -- "
                        "keyed link stripped / chain re-concatenated (downgrade tamper detected)")
            prev = stored_hash

        if not any_record:
            return ChainResult(True, "UNVERIFIED", None, "empty store")
        if all_keyed_checked:
            scheme = "Ed25519-authenticated" if asymmetric else "HMAC-authenticated"
            return ChainResult(True, "VERIFIED", None,
                               f"chain intact, {scheme} (forgery-resistant)")
        return ChainResult(True, "UNVERIFIED", None,
                           "chain intact, but NOT authenticated (integrity-only; set "
                           "VERIFY_HMAC_KEY or VERIFY_ED25519_SEED / pass an expected "
                           "pubkey for forgery resistance)")
