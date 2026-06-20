"""Real conformal-calibration artifacts for certified serving (v2.16.0).

Crack-#1 fix (the verification layer was a library, not a served product). The
served ``/predict_certified`` verifier used to be calibrated on a SYNTHETIC
placeholder, so the certificate's *coverage number* was meaningless on real
bearings — honest in a docstring, but not STRUCTURAL. This module makes the
synthetic-vs-real distinction a first-class, leakage-checked, certificate-bound
fact:

  * A deployer builds a calibration artifact from a HELD-OUT, GROUP-DISJOINT
    split of their real labelled data (:func:`save_calibration`). The
    group-disjointness LEAKAGE CHECK (:mod:`aion_nexus.evaluation`) is baked into
    the artifact's metadata AND enforced: a ``real-holdout`` artifact REFUSES to
    be written if the calibration groups leak into the training groups.
  * The server loads it (:func:`load_calibration`) and stamps
    ``coverage_basis = "real-holdout"`` into the signed certificate (tamper-
    evident, via the certificate's ``coverage_guarantee`` hash channel). With no
    artifact present it falls back to ``"synthetic-placeholder"`` and says so on
    every response.

HONESTY (workspace 6.31). This module does NOT make the coverage guarantee
stronger: a conformal guarantee is EMPIRICAL and valid only under exchangeability
of calibration and serving data — cross-bearing / cross-machine deployment still
breaks it. What it changes is whether the calibration set is REAL, and it makes
that fact visible and verifiable instead of buried in a placeholder.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from aion_nexus.evaluation import check_group_disjoint

# Calibration-basis vocabulary stamped (tamper-evidently) into the certificate.
BASIS_REAL = "real-holdout"           # REAL data, calibration group-disjoint from TRAINING
BASIS_REAL_INDIST = "real-indistribution"  # REAL data, but NOT certified disjoint from training
BASIS_SYNTHETIC = "synthetic-placeholder"  # the server's runnable fallback
BASIS_DEMO = "synthetic-demo"         # a clearly-labelled demo artifact (not real)

ARTIFACT_VERSION = "1.0"

# real-holdout is the ONLY basis whose leakage gate is ENFORCED (refuses a leaked
# split). real-indistribution is honest about real data whose calibration set is
# NOT proven disjoint from the model's training distribution (e.g. a globally-
# stratified checkpoint): the coverage is measured on real data, but it is not a
# clean cross-machine/cross-bearing guarantee — the basis says so.
_VALID_BASES = (BASIS_REAL, BASIS_REAL_INDIST, BASIS_SYNTHETIC, BASIS_DEMO)


def coverage_guarantee_string(basis: str, alpha: float, *, leakage_checked: bool,
                              temperature: float = 1.0) -> str:
    """The exact string stamped into the certificate's ``coverage_guarantee`` hash.

    Binding the basis (and the temperature-scaling factor) into the certificate
    hash is what makes the synthetic-vs-real distinction TAMPER-EVIDENT: a
    real-holdout cert and a placeholder cert hash differently, and silently
    upgrading the claim breaks the signature.
    """
    leak = "leakage-checked" if leakage_checked else "leakage-UNCHECKED"
    temp = "" if float(temperature) == 1.0 else f" temperature-scaled T={float(temperature):.2f};"
    return (f"marginal split-conformal, target coverage {1.0 - float(alpha):.3f} "
            f"(alpha={float(alpha):.3f}); calibration_basis={basis}; {leak};{temp} "
            "EMPIRICAL — valid only under exchangeability of calibration and "
            "serving data (cross-bearing/cross-machine breaks it).")


def save_calibration(
    path: str | Path,
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    *,
    basis: str,
    train_groups: Sequence | None = None,
    calib_groups: Sequence | None = None,
    source: str | None = None,
    created: str | None = None,
) -> dict:
    """Write a conformal-calibration artifact (``.npz``) the server can load.

    Parameters
    ----------
    probs, labels:
        ``probs`` is the model's class-probability matrix ``(N, K)`` on the
        held-out calibration windows; ``labels`` the integer ground truth ``(N,)``.
    class_names:
        The ``K`` human-readable class names, in column order of ``probs``.
    basis:
        One of :data:`BASIS_REAL`, :data:`BASIS_DEMO`, :data:`BASIS_SYNTHETIC`.
    train_groups, calib_groups:
        The group ids (bearing / recording / machine) of the TRAINING set and of
        the CALIBRATION set. For a :data:`BASIS_REAL` artifact BOTH are required
        and the function REFUSES to write if they are not group-disjoint (the
        leakage gate). For demo/synthetic artifacts they are optional.

    Returns
    -------
    dict
        The metadata written into the artifact.

    Raises
    ------
    ValueError
        On shape/label mismatch, an unknown ``basis``, or — for a ``real-holdout``
        artifact — missing groups or a leaked split.
    """
    if basis not in _VALID_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {_VALID_BASES}")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    class_names = [str(c) for c in class_names]
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (N, K); got shape {probs.shape}")
    n, k = probs.shape
    if labels.shape != (n,):
        raise ValueError(f"labels must be 1-D length {n}; got shape {labels.shape}")
    if len(class_names) != k:
        raise ValueError(
            f"class_names has {len(class_names)} entries but probs has {k} columns")
    if n < k + 1:
        raise ValueError(
            f"need at least n_classes+1={k + 1} calibration samples for a finite "
            f"conformal quantile; got {n}")
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError(
            f"labels must be in [0, {k}); got range [{labels.min()}, {labels.max()}]")

    leakage_checked = False
    disjoint: bool | None = None
    leaked_groups: list = []
    if basis == BASIS_REAL:
        if train_groups is None or calib_groups is None:
            raise ValueError(
                "a 'real-holdout' calibration artifact REQUIRES train_groups and "
                "calib_groups so the leakage gate can prove the calibration split "
                "does not leak into training (use basis='synthetic-demo' otherwise).")
    if train_groups is not None and calib_groups is not None:
        if len(calib_groups) != n:
            raise ValueError(
                f"calib_groups length {len(calib_groups)} != n_samples {n}")
        leak = check_group_disjoint(train_groups, calib_groups)
        leakage_checked = True
        disjoint = bool(leak.disjoint)
        leaked_groups = [str(g) for g in leak.leaked_groups]
        if basis == BASIS_REAL and not disjoint:
            raise ValueError(
                "REFUSING to write a 'real-holdout' calibration artifact: the "
                f"calibration groups leak into training ({leaked_groups}). A leaked "
                "calibration set inflates apparent coverage. Re-split so calibration "
                "bearings are disjoint from training, or label this basis honestly.")

    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "basis": basis,
        "n": int(n),
        "n_classes": int(k),
        "class_names": class_names,
        "alpha_hint": 0.1,
        "leakage_checked": leakage_checked,
        "disjoint": disjoint,
        "leaked_groups": leaked_groups,
        "source": source,
        "created": created,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        probs=probs,
        labels=labels,
        class_names=np.array(class_names),
        meta=np.array(json.dumps(meta)),
    )
    return meta


def load_calibration(path: str | Path) -> dict:
    """Load a calibration artifact written by :func:`save_calibration`.

    Returns ``{"probs", "labels", "class_names", "basis", "meta"}``. Raises
    ``FileNotFoundError`` if absent, ``ValueError`` on a malformed artifact.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"calibration artifact not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "probs" not in data or "labels" not in data:
            raise ValueError(f"{path} is not a calibration artifact (missing probs/labels)")
        probs = np.asarray(data["probs"], dtype=np.float64)
        labels = np.asarray(data["labels"], dtype=np.int64)
        class_names = [str(c) for c in data["class_names"]] if "class_names" in data else None
        try:
            meta = json.loads(str(data["meta"])) if "meta" in data else {}
        except (ValueError, TypeError):
            meta = {}
    if probs.ndim != 2 or labels.shape != (probs.shape[0],):
        raise ValueError(
            f"malformed calibration artifact: probs {probs.shape}, labels {labels.shape}")
    return {
        "probs": probs,
        "labels": labels,
        "class_names": class_names,
        "basis": meta.get("basis", BASIS_REAL),
        "meta": meta,
    }


def fit_temperature(probs: np.ndarray, labels: np.ndarray,
                    *, grid: int = 36, lo: float = 0.5, hi: float = 4.0) -> float:
    """Temperature scaling (Guo et al., ICML 2017): the single T that minimises NLL.

    Operates on PROBABILITIES (re-tempers ``p ** (1/T)`` and renormalises), so it
    needs no logits. ``T > 1`` softens an over-confident model (the common case),
    ``T < 1`` sharpens. A well-calibrated posterior gives the conformal score
    function honest inputs -> tighter, more trustworthy prediction sets. Returns
    1.0 (no-op) if the calibration set is degenerate.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or len(labels) != probs.shape[0] or probs.shape[0] < 2:
        return 1.0
    eps = 1e-12
    best_t, best_nll = 1.0, float("inf")
    idx = np.arange(len(labels))
    for temp in np.linspace(lo, hi, grid):
        p = np.power(probs + eps, 1.0 / float(temp))
        p = p / p.sum(1, keepdims=True)
        nll = float(-np.mean(np.log(p[idx, labels] + eps)))
        if nll < best_nll:
            best_nll, best_t = nll, float(temp)
    return best_t


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Re-temper a probability vector/matrix by ``temperature`` and renormalise."""
    if temperature == 1.0:
        return np.asarray(probs, dtype=np.float64)
    p = np.power(np.asarray(probs, dtype=np.float64) + 1e-12, 1.0 / float(temperature))
    return p / p.sum(axis=-1, keepdims=True)


def synthetic_demo_probs(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    n_classes: int,
    *,
    per_class: int = 8,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the SAME synthetic placeholder the server falls back to, as a reusable
    function so server, CLI and demo share one honest source.

    ``predict_fn`` maps a raw ``[2, 2560]`` window to a length-``n_classes``
    probability vector. The result is a (``n_classes*per_class``, ``n_classes``)
    probability matrix with round-robin synthetic labels — NOT exchangeable with
    real bearings (that is the whole point of labelling it a placeholder).
    """
    rng = np.random.default_rng(seed)
    probs_list, labels_list = [], []
    for cls in range(n_classes):
        for _ in range(per_class):
            sig = rng.standard_normal((2, 2560)).astype("float32") * 0.5
            probs_list.append(np.asarray(predict_fn(sig), dtype=np.float64))
            labels_list.append(cls)
    return np.vstack(probs_list), np.array(labels_list, dtype=np.int64)
