"""Example 10: leakage-free evaluation, attested — the honest number as a feature.

Evaluation leakage is endemic in bearing-fault benchmarking (Hendriks et al. 2022:
40/41 CWRU studies leaked). A random window-level split puts windows from the SAME
bearing on both sides, so the model can key on bearing identity and posts 95-99% —
which collapses to 35-60% under a bearing-disjoint (leave-one-bearing-out) split.
Buyers can't trust vendor numbers; that's the opening for an independent verifier.

This shows AION shipping honest evaluation AS a feature, with no real data:

    1. A leaky random split looks great...
    2. ...but the leakage detector FLAGS it (a machine-checkable audit of any split).
    3. Leave-one-bearing-out gives the HONEST interval (mean +/- std), much lower.
    4. The result is sealed into a signed report a third party verifies offline.

Run:
    python examples/10_leakage_free_eval.py
"""
from __future__ import annotations

import numpy as np

from aion_nexus.evaluation import (
    check_group_disjoint,
    evaluate_leave_one_group_out,
    macro_f1,
    verify_evaluation_report,
)
from aion_nexus.verify import ed25519_pubkey_from_seed, generate_seed

K = 4


def _knn(x_tr, y_tr, x_te, k=5):
    d = ((x_te[:, None, :] - x_tr[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1)[:, :k]
    votes = y_tr[nn]
    scores = np.stack([(votes == c).mean(1) for c in range(K)], axis=1)
    return scores.argmax(1), scores


def main() -> int:
    # 8 bearings: a clear class signal + a DOMINANT per-bearing identity offset.
    rng = np.random.default_rng(0)
    cdir = rng.standard_normal((K, 8))
    cdir /= np.linalg.norm(cdir, axis=1, keepdims=True)
    bdir = rng.standard_normal((8, 8))
    bdir /= np.linalg.norm(bdir, axis=1, keepdims=True)
    feats, lab, grp = [], [], []
    for b in range(8):
        for _ in range(120):
            c = rng.integers(0, K)
            feats.append(3.0 * cdir[c] + 12.0 * bdir[b] + 0.3 * rng.standard_normal(8))
            lab.append(c)
            grp.append(b)
    x, y, g = np.array(feats), np.array(lab), np.array(grp)

    # 1-2. Random split looks great, but the detector flags the leakage.
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    pred, _ = _knn(x[tr], y[tr], x[te])
    leak = check_group_disjoint(g[tr], g[te])
    print("--- 1-2. Random window-level split ---")
    print(f"  macro-F1 = {macro_f1(y[te], pred, K):.3f}  (looks great)")
    print(f"  leakage detector: disjoint={leak.disjoint}  -> {leak.detail}")

    # 3. Leave-one-bearing-out: the honest interval.
    report = evaluate_leave_one_group_out(_knn, x, y, g, n_classes=K,
                                          group_kind="bearing", model_id="knn-demo")
    f = report.f1_macro
    print("\n--- 3. Leave-one-bearing-out (honest) ---")
    print(f"  macro-F1 = {f['mean']:.3f} +/- {f['std']:.3f}  (min {f['min']:.3f}, "
          f"max {f['max']:.3f}, {report.n_folds} folds, all disjoint={report.all_folds_disjoint})")
    print(f"  -> the honest number is far lower than the leaky one. {report.summary()}")

    # 4. Seal + third-party verify.
    seed = generate_seed()
    report.seal(seed, scheme="ed25519")
    res = verify_evaluation_report(report, expected_pubkey=ed25519_pubkey_from_seed(seed))
    print("\n--- 4. Signed attestation ---")
    print(f"  verify (public key only): trusted={res['trusted']} ({res['authenticity']})")
    tampered = report.as_dict()
    tampered["f1_macro"] = {**f, "mean": 0.99}
    res2 = verify_evaluation_report(tampered, expected_pubkey=ed25519_pubkey_from_seed(seed))
    print(f"  forge a better number -> trusted={res2['trusted']} (integrity_ok={res2['integrity_ok']})")

    assert leak.disjoint is False
    assert f["mean"] < 0.6 and res["trusted"] and not res2["trusted"]
    print("\nThe leakage-free protocol + honest interval are bound into a signed report: "
          "a buyer audits the split and re-verifies the number offline. Honest scope: the "
          "check proves disjointness of the GROUP IDS supplied, not leakage they don't capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
