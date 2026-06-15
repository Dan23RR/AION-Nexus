"""Append-only JSONL certificate store with an optional tamper-evident chain.

One JSON line per certificate (an audit trail for enterprise / EU AI Act Art.12
logging evidence). Each record carries ``prev_hash`` (the previous record's
``record_hash``, genesis = 64 zeros) and ``record_hash`` (a digest over the
canonical record including ``prev_hash``), forming a hash chain.

Integrity vs. authenticity — what ``verify_chain()`` actually proves
--------------------------------------------------------------------
The chain link is **HMAC-SHA256 keyed by env ``VERIFY_HMAC_KEY`` when that key
is set, and plain SHA-256 otherwise**. Each record records ``chain_auth``
("HMAC-SHA256" or "NONE") so the store is honest about its own guarantee:

* No key (``chain_auth=NONE``): ``verify_chain()`` only detects ACCIDENTAL
  corruption (a flipped byte, a truncated line). It is NOT tamper-evident
  against an adversary who has this source — they can edit a record and
  recompute the whole chain. Do not claim tamper-evidence in this mode.
* Key set (``chain_auth=HMAC-SHA256``): the link is keyed, so an adversary
  without the secret cannot re-concatenate a forged record. ``verify_chain()``
  then detects in-place edits, reorders and deletions.

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

GENESIS_HASH = "0" * 64
ENV_HMAC_KEY = "VERIFY_HMAC_KEY"
ENV_CERT_STORE = "VERIFY_CERT_STORE"


def _canonical(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)


def _chain_key(explicit: str | bytes | None = None) -> bytes | None:
    if explicit is not None:
        return explicit if isinstance(explicit, bytes) else explicit.encode()
    key = os.environ.get(ENV_HMAC_KEY)
    return key.encode() if key else None


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

    def __init__(self, path: str | os.PathLike | None = None, chain: bool = True) -> None:
        self.path = Path(path or os.environ.get(ENV_CERT_STORE, "./certificates.jsonl"))
        self.chain = chain
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

    def append(self, cert, key: str | bytes | None = None) -> dict:
        """Append one certificate (a ``Certificate`` or a plain dict).

        Returns the stored record (certificate fields + ``prev_hash`` /
        ``record_hash`` / ``chain_auth`` if chained). The chain link is HMAC-keyed
        when a key is available (explicit arg or ``VERIFY_HMAC_KEY``).
        """
        record = dict(cert.as_dict()) if hasattr(cert, "as_dict") else dict(cert)
        with self._lock:
            if self.chain:
                k = _chain_key(key)
                record["prev_hash"] = self._prev_hash
                record["chain_auth"] = "HMAC-SHA256" if k is not None else "NONE"
                canonical = _canonical(
                    {kk: vv for kk, vv in record.items() if kk != "record_hash"})
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

    def verify_chain(self, key: str | bytes | None = None) -> ChainResult:
        """Verify the chain links back to genesis. Returns a :class:`ChainResult`
        (truthy iff intact). ``key`` defaults to env ``VERIFY_HMAC_KEY``.

        Each record's link is recomputed with the digest its own ``chain_auth``
        declares. An HMAC-keyed record only counts as *authenticated*
        (forgery-resistant) when a matching key is supplied.

        DOWNGRADE GUARD: when a key IS supplied, a record declaring
        ``chain_auth=NONE`` fails closed. ``chain_auth`` is unauthenticated, so a
        keyless attacker could strip the HMAC from a real record and re-link it
        with plain SHA-256; an honest keyed store never writes NONE records, so
        to a key-holder a NONE record IS the attack. Consequence (intended): a
        store legitimately created WITHOUT a key cannot later be audited by
        passing a key — audit it keyless (it is integrity-only by construction).
        """
        k = _chain_key(key)

        prev = GENESIS_HASH
        all_keyed_checked = True
        any_record = False
        for i, record in enumerate(self.iter_certs()):
            any_record = True
            stored_hash = record.pop("record_hash", None)
            if record.get("prev_hash") != prev:
                return ChainResult(
                    False, "FORGED" if k is not None else "UNVERIFIED", i,
                    f"record {i}: prev_hash does not link to record {i - 1}")
            declared = record.get("chain_auth", "NONE")
            canonical = _canonical(record)
            if declared == "HMAC-SHA256":
                if k is None:
                    all_keyed_checked = False           # cannot key-check; structure only
                    prev = stored_hash
                    continue
                expected = hmac.new(k, canonical.encode(), hashlib.sha256).hexdigest()
                if stored_hash is None or not hmac.compare_digest(stored_hash, expected):
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: HMAC record_hash mismatch (tamper detected)")
            else:  # plain SHA-256 link
                all_keyed_checked = False
                expected = hashlib.sha256(canonical.encode()).hexdigest()
                if stored_hash is None or expected != stored_hash:
                    return ChainResult(
                        False, "UNVERIFIED", i,
                        f"record {i}: SHA-256 record_hash mismatch (corruption)")
                if k is not None:
                    # Downgrade attack: a key-holder never writes NONE records, so
                    # a NONE record means the HMAC was stripped and re-concatenated.
                    return ChainResult(
                        False, "FORGED", i,
                        f"record {i}: chain_auth=NONE but a key is configured -- "
                        "HMAC stripped / chain re-concatenated (downgrade tamper detected)")
            prev = stored_hash

        if not any_record:
            return ChainResult(True, "UNVERIFIED", None, "empty store")
        if all_keyed_checked:
            return ChainResult(True, "VERIFIED", None,
                               "chain intact, HMAC-authenticated (forgery-resistant)")
        return ChainResult(True, "UNVERIFIED", None,
                           "chain intact, but NOT authenticated (integrity-only; set "
                           "VERIFY_HMAC_KEY for forgery resistance)")
