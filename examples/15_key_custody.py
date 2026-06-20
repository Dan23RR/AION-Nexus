"""Example 15: key custody — KMS/HSM signing + rotation + revocation (v2.20.0).

Offline-verifiable certificates are only as trustworthy as the custody of the
signing key. This shows the two halves of the fix:

  1. The private key NEVER enters the AION process: an ExternalSigner forwards
     signing to an external device (here a stand-in for AWS KMS / Cloud HSM /
     Azure Key Vault). AION holds only the PUBLIC key + a sign callback.
  2. A publishable KeyRing supports ROTATION (retire a key, activate a successor)
     and REVOCATION (a compromised key invalidates ONLY its own certificates).

Run:  python examples/15_key_custody.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus.verify import (
    ExternalSigner,
    KeyRing,
    Verifier,
    ed25519_pubkey_from_seed,
    ed25519_sign,
    generate_seed,
    verify_with_keyring,
)

CLASSES = ["normal", "early", "medium", "advanced"]


def main() -> int:
    # ---- a stand-in KMS/HSM: the DEVICE holds the seed; AION gets a callback ---
    device_seed = generate_seed()                 # lives in the "device", not AION
    pub = ed25519_pubkey_from_seed(device_seed)

    def kms_sign(message: str) -> str:            # AION calls this; never sees the key
        return ed25519_sign(message, device_seed)

    signer = ExternalSigner(pub, kms_sign)
    print("1. ExternalSigner: AION holds the PUBLIC key + a callback; the private "
          "key stays in the device.")

    # ---- mint a certificate via the external signer ---------------------------
    v = Verifier(alpha=0.1, class_names=CLASSES)
    v.calibrate(np.random.default_rng(0).dirichlet([1, 1, 1, 1], 200), np.array([0, 1, 2, 3] * 50))
    cert = v.certify(np.array([0.05, 0.05, 0.1, 0.8]), input_signal=np.zeros((2, 2560)),
                     signer=signer, key_id="kms-2026-q3", ttl_seconds=86400)
    print(f"2. Sealed with KMS key: auth={cert.authentication}, key_id={cert.key_id}")

    # ---- publishable registry; rotation + revocation --------------------------
    ring = KeyRing().rotate("kms-2026-q3", pub)
    r_active = verify_with_keyring(cert, ring)
    print(f"3. Active key  -> trusted={r_active['trusted']} ({r_active['authenticity']})")

    ring.rotate("kms-2027-q1", ed25519_pubkey_from_seed(generate_seed()))   # q3 retired
    r_retired = verify_with_keyring(cert, ring)
    print(f"4. After rotation, old cert -> trusted={r_retired['trusted']} "
          f"(key_status={r_retired['key_status']}; rotation does NOT invalidate)")

    ring.revoke("kms-2026-q3", "seed exposed in incident-217")
    r_revoked = verify_with_keyring(cert, ring)
    print(f"5. After revocation -> trusted={r_revoked['trusted']} "
          f"({r_revoked['authenticity']}: {r_revoked['key_note'][:46]}...)")

    r_unknown = verify_with_keyring(cert, KeyRing())   # empty registry
    print(f"6. Unknown key_id -> trusted={r_unknown['trusted']} ({r_unknown['authenticity']})")

    assert r_active["trusted"] and r_active["authenticity"] == "VERIFIED"
    assert r_retired["trusted"] and r_retired["key_status"] == "retired"
    assert not r_revoked["trusted"] and r_revoked["authenticity"] == "REVOKED-KEY"
    assert not r_unknown["trusted"] and r_unknown["authenticity"] == "UNKNOWN-KEY"
    print("\nOK — the private key never entered this process, and a compromised key "
          "invalidates only its own certificates. This is the custody story a regulated "
          "buyer's security team requires before trusting signed certificates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
