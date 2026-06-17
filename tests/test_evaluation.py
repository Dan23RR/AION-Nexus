"""Tests for leakage-free evaluation (aion_nexus.evaluation).

The headline test reproduces, in miniature, the field's defining failure: the SAME
model and data give an inflated score under a random (leaky) split and a much lower
HONEST score under leave-one-bearing-out — and the harness exposes it. Plus: the
leakage detector catches a leaked split, the metrics are correct, and the signed
report is tamper-evident.
"""
from __future__ import annotations

import numpy as np

from aion_nexus.evaluation import (
    EvaluationReport,
    check_group_disjoint,
    evaluate_leave_one_group_out,
    honest_interval,
    leave_one_group_out,
    macro_auroc,
    macro_f1,
    per_class_recall,
    verify_evaluation_report,
)
from aion_nexus.verify import ed25519_pubkey_from_seed, generate_seed

K = 4


def _knn(x_tr, y_tr, x_te, *, k=5, n_classes=K):
    """A k-NN classifier — the classic leakage EXPLOITER (memorises neighbours)."""
    d = ((x_te[:, None, :] - x_tr[None, :, :]) ** 2).sum(-1)     # [n_te, n_tr]
    nn = np.argsort(d, axis=1)[:, :k]
    votes = y_tr[nn]
    scores = np.stack([(votes == c).mean(1) for c in range(n_classes)], axis=1)
    return scores.argmax(1), scores


def _grouped_dataset(n_bearings=8, per_bearing=120, d=8, seed=0):
    """Each sample = a clear shared CLASS signal + a DOMINANT per-bearing IDENTITY
    offset + small noise. Within a bearing the class is easily separable, so a
    random split lets k-NN key on same-bearing neighbours (leakage -> inflated);
    leave-one-bearing-out removes that crutch and the bearing offset dominates the
    distance (honest -> collapse)."""
    rng = np.random.default_rng(seed)
    class_dir = rng.standard_normal((K, d))
    class_dir /= np.linalg.norm(class_dir, axis=1, keepdims=True)
    bearing_dir = rng.standard_normal((n_bearings, d))
    bearing_dir /= np.linalg.norm(bearing_dir, axis=1, keepdims=True)
    feats, lab, grp = [], [], []
    for b in range(n_bearings):
        for _ in range(per_bearing):
            c = rng.integers(0, K)
            feat = 3.0 * class_dir[c] + 12.0 * bearing_dir[b] + 0.3 * rng.standard_normal(d)
            feats.append(feat)
            lab.append(c)
            grp.append(b)
    return np.array(feats), np.array(lab), np.array(grp)


# --------------------------------------------------------------------------- #
# 1. The leakage detector
# --------------------------------------------------------------------------- #

def test_leakage_detector_catches_a_leaked_split():
    leaked = check_group_disjoint([1, 1, 2, 3], [3, 4, 5])     # group 3 on both sides
    assert leaked.disjoint is False
    assert 3 in leaked.leaked_groups
    assert "LEAKAGE" in leaked.detail

    clean = check_group_disjoint([1, 1, 2, 3], [4, 5, 6])
    assert clean.disjoint is True
    assert clean.leaked_groups == []


def test_leave_one_group_out_folds_are_disjoint():
    groups = np.array([0, 0, 1, 1, 2, 2])
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    for train_idx, test_idx, held in folds:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert set(groups[test_idx]) == {held}


# --------------------------------------------------------------------------- #
# 2. THE headline: random (leaky) split inflates vs leave-one-bearing-out
# --------------------------------------------------------------------------- #

def test_lobo_exposes_the_inflation_of_a_random_split():
    x, y, g = _grouped_dataset(seed=1)
    rng = np.random.default_rng(2)

    # Random (window-level) split: train/test SHARE bearings -> leakage.
    idx = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    leak = check_group_disjoint(g[tr], g[te])
    assert leak.disjoint is False                      # the random split IS leaky
    pred, _ = _knn(x[tr], y[tr], x[te])
    random_f1 = macro_f1(y[te], pred, K)

    # Leave-one-bearing-out: disjoint by construction -> the honest number.
    report = evaluate_leave_one_group_out(_knn, x, y, g, n_classes=K, group_kind="bearing")
    assert report.all_folds_disjoint is True
    honest_f1 = report.f1_macro["mean"]

    # The leaky split is dramatically inflated vs the honest LOBO number.
    assert random_f1 > 0.85, f"leaky split should look great: {random_f1:.3f}"
    assert honest_f1 < random_f1 - 0.3, \
        f"LOBO must expose the collapse: honest {honest_f1:.3f} vs leaky {random_f1:.3f}"
    assert report.n_folds == 8


# --------------------------------------------------------------------------- #
# 3. Prevalence-independent metrics
# --------------------------------------------------------------------------- #

def test_macro_auroc_perfect_and_chance():
    rng = np.random.default_rng(3)
    y = rng.integers(0, K, size=400)
    perfect = np.eye(K)[y] + 0.01 * rng.standard_normal((400, K))  # true class scored highest
    assert macro_auroc(y, perfect) > 0.98
    chance = rng.standard_normal((400, K))
    assert 0.4 < macro_auroc(y, chance) < 0.6


def test_macro_f1_and_recall():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 2, 2, 2])              # class1 one miss, class2 over-predicted
    assert per_class_recall(y_true, y_pred, 3)[0] == 1.0
    assert per_class_recall(y_true, y_pred, 3)[1] == 0.5
    assert 0.0 < macro_f1(y_true, y_pred, 3) < 1.0


def test_honest_interval_reports_spread_not_a_point():
    iv = honest_interval([0.3, 0.5, 0.1, 0.4])
    assert iv["n"] == 4
    assert iv["min"] == 0.1 and iv["max"] == 0.5
    assert iv["std"] > 0                               # the spread the field hides


# --------------------------------------------------------------------------- #
# 4. The signed attestation
# --------------------------------------------------------------------------- #

def test_report_seals_and_verifies_and_is_tamper_evident():
    x, y, g = _grouped_dataset(seed=4)
    report = evaluate_leave_one_group_out(_knn, x, y, g, n_classes=K,
                                          group_kind="bearing", model_id="knn-demo")
    seed = generate_seed()
    pub = ed25519_pubkey_from_seed(seed)
    report.seal(seed, scheme="ed25519")

    res = verify_evaluation_report(report, expected_pubkey=pub)
    assert res["integrity_ok"] is True
    assert res["trusted"] is True

    # Forge a better honest number -> integrity breaks (the claim is bound).
    forged = report.as_dict()
    forged["f1_macro"] = dict(forged["f1_macro"])
    forged["f1_macro"]["mean"] = 0.99
    res2 = verify_evaluation_report(forged, expected_pubkey=pub)
    assert res2["integrity_ok"] is False
    assert res2["trusted"] is False


def test_report_without_expected_key_is_self_signed_not_trusted():
    x, y, g = _grouped_dataset(seed=5)
    report = evaluate_leave_one_group_out(_knn, x, y, g, n_classes=K, group_kind="bearing")
    report.seal(generate_seed(), scheme="ed25519")
    res = verify_evaluation_report(report)             # only the embedded key
    assert res["authenticity"] == "SELF-SIGNED"
    assert res["trusted"] is False


def test_unsigned_report_is_unverified():
    iv = honest_interval([0.4, 0.5])
    rep = EvaluationReport(protocol="leave-one-group-out", group_kind="bearing",
                           n_folds=2, n_samples=10, all_folds_disjoint=True,
                           f1_macro=iv, auroc_macro=iv, per_fold=[]).seal(scheme="none")
    res = verify_evaluation_report(rep)
    assert res["authenticity"] == "UNVERIFIED"
    assert res["trusted"] is False
    assert res["integrity_ok"] is True                 # hash still recomputes
