"""Leakage-free evaluation as a product feature — the honest number, attested.

Why this module exists (the verifier brand, in the evaluation itself)
---------------------------------------------------------------------
A field-wide research finding: evaluation leakage is endemic in bearing-fault
benchmarking. Hendriks et al. (2022) found 40/41 CWRU studies leaked; only ~6/55
bearing papers used rigorous splits. Random window-level splits put windows from
the SAME recording / bearing / operating condition on BOTH sides, producing
95-99% accuracy that collapses to 35-60% under recording- or bearing-disjoint
splits (arXiv:2509.22267). So buyers literally cannot trust vendor accuracy
numbers — which is the precise opening for an INDEPENDENT VERIFIER.

This module ships honest evaluation AS a feature:

1. **A machine-checkable leakage detector** (:func:`check_group_disjoint`). Given
   the bearing / recording / machine id of every sample, it proves whether a claimed
   train/test split is group-disjoint. This audits ANYONE's split — including a
   vendor's claimed protocol — with a yes/no a procurement team can verify.
2. **A leave-one-group-out harness** (:func:`evaluate_leave_one_group_out`) that
   reports an HONEST INTERVAL (mean ± std across folds), not a single stratified
   number, using prevalence-independent metrics (macro-AUROC, macro-F1, per-class
   recall) — the metrics a certificate needs.
3. **A signed attestation** (:class:`EvaluationReport`) that binds the protocol +
   the leakage check + the honest intervals into a content hash, signable with the
   same Ed25519 / HMAC primitives as the per-decision certificate, so the
   "measured leakage-free" claim is itself tamper-evident and third-party-checkable.

HONESTY (workspace 6.31). The leakage detector verifies disjointness of the GROUP
IDS the caller supplies — it CANNOT detect leakage those ids do not capture (e.g.
overlapping sliding windows within a recording if you only grouped by bearing, or
operating-condition leakage if you only grouped by machine). The report attests
the PROTOCOL and reports the numbers honestly; it does NOT claim the numbers are
good — only that they were measured the way it says. Pure numpy; no torch.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# 1. The leakage detector — the core verifier capability
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LeakageCheck:
    """Result of auditing a train/test split for group leakage."""

    disjoint: bool                 # True iff NO group appears in both train and test
    leaked_groups: list            # the groups present on both sides (empty if disjoint)
    n_train_groups: int
    n_test_groups: int
    detail: str


def check_group_disjoint(train_groups: Sequence, test_groups: Sequence) -> LeakageCheck:
    """Audit whether a train/test split is group-disjoint (the leakage check).

    ``train_groups`` / ``test_groups`` are the grouping id (bearing / recording /
    machine) of each train / test sample. A split is leakage-free *with respect to
    that grouping* iff no id appears on both sides. This is the machine-checkable
    property a verifier exposes: a vendor's "99% accuracy" can be audited by asking
    for the sample group ids and running this.
    """
    train_set = set(train_groups)
    test_set = set(test_groups)
    leaked = [_scalar(v) for v in sorted(train_set & test_set, key=str)]
    disjoint = not leaked
    detail = (f"group-disjoint over {len(train_set)} train / {len(test_set)} test groups"
              if disjoint else
              f"LEAKAGE: {len(leaked)} group(s) appear in BOTH train and test "
              f"(e.g. {leaked[:3]}) — accuracy measured on this split is inflated")
    return LeakageCheck(disjoint, leaked, len(train_set), len(test_set), detail)


def leave_one_group_out(groups: Sequence) -> list[tuple[np.ndarray, np.ndarray, object]]:
    """Leave-one-group-out folds: for each distinct group, (train_idx, test_idx, held_group).

    Each fold holds out one entire group (one bearing / machine) for test and
    trains on the rest — disjoint BY CONSTRUCTION, the honest cross-machine protocol.
    """
    groups_arr = np.asarray(groups)
    folds = []
    for g in _unique_stable(groups_arr):
        test_idx = np.where(groups_arr == g)[0]
        train_idx = np.where(groups_arr != g)[0]
        folds.append((train_idx, test_idx, _scalar(g)))
    return folds


def _unique_stable(arr: np.ndarray) -> list:
    seen, out = set(), []
    for v in arr.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _scalar(g):
    return g.item() if isinstance(g, np.generic) else g


# --------------------------------------------------------------------------- #
# 2. Prevalence-independent metrics + honest intervals (pure numpy)
# --------------------------------------------------------------------------- #

def _binary_auroc(y_bin: np.ndarray, score: np.ndarray) -> float | None:
    """One-vs-rest AUROC via the Mann-Whitney rank statistic. None if degenerate."""
    pos = y_bin == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None                                   # class absent -> undefined
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties (so AUROC is exact under ties)
    _assign_tie_ranks(score, ranks)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _assign_tie_ranks(score: np.ndarray, ranks: np.ndarray) -> None:
    order = np.argsort(score, kind="mergesort")
    s_sorted = score[order]
    i = 0
    n = len(s_sorted)
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0               # average of 1-based ranks
            ranks[order[i:j + 1]] = avg
        i = j + 1


def macro_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUROC — prevalence-independent (classes absent
    from ``y_true`` are skipped, not counted as 0)."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be [n_samples, n_classes]")
    aucs = [a for c in range(scores.shape[1])
            if (a := _binary_auroc((y_true == c).astype(int), scores[:, c])) is not None]
    return float(np.mean(aucs)) if aucs else float("nan")


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Recall per class (sensitivity); a class absent from ``y_true`` maps to None."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {}
    for c in range(n_classes):
        sup = int((y_true == c).sum())
        out[c] = float(((y_pred == c) & (y_true == c)).sum() / sup) if sup else None
    return out


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over classes present in ``y_true`` (prevalence-independent)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if (y_true == c).sum() == 0:
            continue
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def honest_interval(values: Sequence[float]) -> dict:
    """The honest cross-fold summary: mean, std, min, max, n — NOT a single number."""
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)],
                     dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                "max": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()), "n": int(arr.size)}


# --------------------------------------------------------------------------- #
# 3. The signed attestation
# --------------------------------------------------------------------------- #

REPORT_SCHEMA_VERSION = "1.0"
AUTH_NONE = "NONE"
AUTH_HMAC = "HMAC-SHA256"
AUTH_ED25519 = "Ed25519"


@dataclass
class EvaluationReport:
    """A signable attestation of a leakage-free evaluation protocol + honest numbers.

    Binds the protocol, the leakage check, and the honest metric intervals into a
    ``content_hash`` so the "measured leakage-free" claim is tamper-evident; sign it
    with :meth:`seal` (Ed25519 / HMAC) so a third party verifies it offline.
    """

    protocol: str                  # e.g. "leave-one-group-out"
    group_kind: str                # "bearing" | "machine" | "recording" | ...
    n_folds: int
    n_samples: int
    all_folds_disjoint: bool       # every fold's train/test was group-disjoint
    f1_macro: dict                 # honest_interval
    auroc_macro: dict              # honest_interval
    per_fold: list                 # [{held_group, f1_macro, auroc_macro, n_test}]
    model_id: str | None = None
    schema_version: str = REPORT_SCHEMA_VERSION
    content_hash: str = ""
    authentication: str = AUTH_NONE
    signature: str | None = None
    pubkey: str | None = None

    def canonical_payload(self) -> dict:
        """The order-independent dict ``content_hash`` is taken over (provenance
        fields and the signature/pubkey are excluded, so it is deterministic)."""
        return {
            "protocol": self.protocol,
            "group_kind": self.group_kind,
            "n_folds": int(self.n_folds),
            "n_samples": int(self.n_samples),
            "all_folds_disjoint": bool(self.all_folds_disjoint),
            "f1_macro": _round_interval(self.f1_macro),
            "auroc_macro": _round_interval(self.auroc_macro),
            "model_id": self.model_id,
        }

    def compute_content_hash(self) -> str:
        blob = json.dumps(self.canonical_payload(), sort_keys=True,
                          separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def seal(self, key: str | bytes | None = None, *, scheme: str = "auto") -> EvaluationReport:
        """Fill ``content_hash`` and (if a key is available) sign it.

        ``scheme`` "auto": an explicit ``key`` (Ed25519 seed) wins, else NONE.
        Reuses :mod:`aion_nexus.verify.signing`, so a sealed report is verified by
        the same offline, public-key-only path as a per-decision certificate.
        """
        from aion_nexus.verify import signing

        self.content_hash = self.compute_content_hash()
        scheme = (scheme or "auto").lower()
        self.signature = None
        self.pubkey = None
        if scheme in ("auto", "ed25519") and key is not None:
            self.authentication = AUTH_ED25519
            self.signature = signing.ed25519_sign(self.content_hash, key)
            self.pubkey = signing.ed25519_pubkey_from_seed(key)
        elif scheme == "hmac" and key is not None:
            self.authentication = AUTH_HMAC
            kb = key if isinstance(key, bytes) else str(key).encode()
            self.signature = signing.hmac_sign(self.content_hash, kb)
        else:
            self.authentication = AUTH_NONE
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        f, a = self.f1_macro, self.auroc_macro
        return (f"[{self.protocol}/{self.group_kind}] {self.n_folds} folds, "
                f"disjoint={self.all_folds_disjoint} | macro-F1 {f['mean']:.3f}±{f['std']:.3f} "
                f"| macro-AUROC {a['mean']:.3f}±{a['std']:.3f} | auth={self.authentication}")


def _round_interval(d: dict) -> dict:
    return {k: (None if v is None else (int(v) if k == "n" else round(float(v), 6)))
            for k, v in d.items()}


def verify_evaluation_report(report, key: str | bytes | None = None, *,
                             expected_pubkey: str | None = None) -> dict:
    """Audit an :class:`EvaluationReport` (or its ``as_dict()``): integrity + authenticity.

    Returns ``{integrity_ok, authenticity, trusted, detail}`` with the same honesty
    rule as the certificate: ``trusted`` is True only when integrity holds AND the
    signature verifies against the EXPECTED issuer key (a self-embedded Ed25519 key
    is SELF-SIGNED, not trusted).
    """
    from aion_nexus.verify import signing

    d = report.as_dict() if hasattr(report, "as_dict") else dict(report)
    rep = EvaluationReport(**{k: d[k] for k in (
        "protocol", "group_kind", "n_folds", "n_samples", "all_folds_disjoint",
        "f1_macro", "auroc_macro", "per_fold")}, model_id=d.get("model_id"))
    integrity_ok = rep.compute_content_hash() == d.get("content_hash")
    auth = d.get("authentication", AUTH_NONE)
    sig = d.get("signature")
    ch = str(d.get("content_hash", ""))

    if auth == AUTH_ED25519:
        if expected_pubkey is not None:
            ok = signing.ed25519_verify(ch, str(sig or ""), expected_pubkey)
            authenticity = "VERIFIED" if ok else "FORGED"
            trusted = bool(integrity_ok and ok)
        elif d.get("pubkey") and signing.ed25519_verify(ch, str(sig or ""), d["pubkey"]):
            authenticity, trusted = "SELF-SIGNED", False
        else:
            authenticity, trusted = "FORGED", False
    elif auth == AUTH_HMAC:
        if key is None:
            authenticity, trusted = "UNVERIFIED", False
        else:
            kb = key if isinstance(key, bytes) else str(key).encode()
            ok = signing.hmac_verify(ch, str(sig or ""), kb)
            authenticity = "VERIFIED" if ok else "FORGED"
            trusted = bool(integrity_ok and ok)
    else:
        authenticity, trusted = "UNVERIFIED", False
    return {"integrity_ok": integrity_ok, "authenticity": authenticity,
            "trusted": trusted,
            "detail": f"integrity {'OK' if integrity_ok else 'FAILED'}, {auth} {authenticity}"}


# --------------------------------------------------------------------------- #
# 4. The harness
# --------------------------------------------------------------------------- #

# predict_fn(X_train, y_train, X_test) -> (y_pred[int], scores[n_test, n_classes])
PredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def evaluate_leave_one_group_out(predict_fn: PredictFn, x: np.ndarray, y: np.ndarray,
                                 groups: Sequence, *, n_classes: int | None = None,
                                 group_kind: str = "group", model_id: str | None = None
                                 ) -> EvaluationReport:
    """Run a leave-one-group-out evaluation and return an honest, sealable report.

    ``predict_fn(X_train, y_train, X_test) -> (y_pred, scores)`` is the caller's
    model (trained per fold) — model-agnostic by design (it can wrap AION, a
    customer's model, or a foundation model). Each fold holds out one entire group;
    disjointness is checked (always true here by construction) and recorded. Metrics
    are prevalence-independent and reported as an interval across folds.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=int)
    if n_classes is None:
        n_classes = int(y.max()) + 1
    folds = leave_one_group_out(groups)
    if not folds:
        raise ValueError("no groups to evaluate")

    per_fold, f1s, aurocs = [], [], []
    all_disjoint = True
    groups_arr = np.asarray(groups)
    for train_idx, test_idx, held in folds:
        leak = check_group_disjoint(groups_arr[train_idx], groups_arr[test_idx])
        all_disjoint = all_disjoint and leak.disjoint
        y_pred, scores = predict_fn(x[train_idx], y[train_idx], x[test_idx])
        y_pred = np.asarray(y_pred, dtype=int)
        scores = np.asarray(scores, dtype=np.float64)
        f1 = macro_f1(y[test_idx], y_pred, n_classes)
        auc = macro_auroc(y[test_idx], scores)
        f1s.append(f1)
        aurocs.append(auc)
        per_fold.append({"held_group": _scalar(held), "f1_macro": f1,
                         "auroc_macro": auc, "n_test": int(len(test_idx))})

    return EvaluationReport(
        protocol="leave-one-group-out", group_kind=group_kind, n_folds=len(folds),
        n_samples=int(len(y)), all_folds_disjoint=all_disjoint,
        f1_macro=honest_interval(f1s), auroc_macro=honest_interval(aurocs),
        per_fold=per_fold, model_id=model_id)
