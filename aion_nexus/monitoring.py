"""Continuous production monitoring — point-in-time certification -> rolling SLO + drift.

The AI-assurance category (Fiddler / Arize / WhyLabs / NannyML) has converged on
CONTINUOUS monitoring (MTTD / MTTR), not point-in-time checks. AION has the shift
MATH (weighted conformal, ACI, conformal risk control) but emitted per-decision
certificates only. This turns the certificate stream into a rolling monitor:

  * a live SLO over the last N decisions (certified / review / abstain rates, mean
    confidence, and — when delayed labels arrive — realized accuracy / miss rate);
  * distribution DRIFT detection via the Population Stability Index (PSI) on the
    confidence/score stream vs a calibration reference;
  * a LABEL-FREE performance signal (mean confidence as a calibrated-accuracy proxy,
    plus the abstention trend) that flags silent decay before labels exist.

HONESTY (workspace 6.31): label-free performance estimation ASSUMES the model is
calibrated (use it with temperature scaling), and confidence-drift is a proxy for
true performance drift — it flags "something changed", not "performance dropped by
X". Bearing PdM is the canonical delayed/absent-label problem; that is exactly why
these label-free signals matter. Pure numpy; no torch.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# Standard PSI interpretation thresholds (Population Stability Index).
PSI_NONE = 0.10        # < 0.10: no meaningful shift
PSI_MODERATE = 0.25    # 0.10-0.25: moderate shift (watch); > 0.25: significant (act)


def population_stability_index(reference: np.ndarray, current: np.ndarray, *,
                               bins: int = 10) -> float:
    """PSI between a REFERENCE and a CURRENT 1-D sample (e.g. confidence streams).

    Bins by the reference's quantile edges so each reference bin holds ~equal mass,
    then PSI = sum( (cur_frac - ref_frac) * ln(cur_frac / ref_frac) ). Returns 0.0
    if either sample is too small to bin.
    """
    ref = np.asarray(reference, dtype=np.float64).ravel()
    cur = np.asarray(current, dtype=np.float64).ravel()
    if ref.size < bins or cur.size < 2:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    eps = 1e-6
    ref_frac = np.histogram(ref, edges)[0] / ref.size + eps
    cur_frac = np.histogram(cur, edges)[0] / cur.size + eps
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _drift_level(psi: float) -> str:
    if psi < PSI_NONE:
        return "none"
    if psi < PSI_MODERATE:
        return "moderate"
    return "significant"


@dataclass
class DecisionRecord:
    confidence: float
    verdict: str
    certified: bool
    timestamp: str | None = None


class Monitor:
    """A rolling monitor over the certificate/decision stream.

    Parameters
    ----------
    window:
        Max number of recent decisions to retain (a ring buffer).
    reference_confidence:
        The calibration-time confidence distribution; PSI is measured against it.
    """

    def __init__(self, *, window: int = 500,
                 reference_confidence: np.ndarray | None = None) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._buf: deque[DecisionRecord] = deque(maxlen=self.window)
        self._ref = (np.asarray(reference_confidence, dtype=np.float64).ravel()
                     if reference_confidence is not None else None)

    def record(self, confidence: float, verdict: str, *,
               certified: bool | None = None, timestamp: str | None = None) -> None:
        cert = bool(certified) if certified is not None else (str(verdict) == "CERTIFIED")
        self._buf.append(DecisionRecord(float(confidence), str(verdict), cert, timestamp))

    def __len__(self) -> int:
        return len(self._buf)

    def reset(self) -> None:
        self._buf.clear()

    def status(self) -> dict:
        """Rolling SLO + drift + label-free performance signal over the window."""
        n = len(self._buf)
        if n == 0:
            return {"n": 0, "alerts": ["no decisions recorded yet"]}
        conf = np.array([r.confidence for r in self._buf], dtype=np.float64)
        verdicts = [r.verdict for r in self._buf]
        certified_rate = float(np.mean([r.certified for r in self._buf]))
        review_rate = float(np.mean([v == "REVIEW" for v in verdicts]))
        abstain_rate = float(np.mean([v == "ABSTAIN" for v in verdicts]))
        psi = (population_stability_index(self._ref, conf)
               if self._ref is not None and self._ref.size else 0.0)
        level = _drift_level(psi)
        alerts = []
        if level == "significant":
            alerts.append(f"input/confidence drift SIGNIFICANT (PSI={psi:.2f}) — "
                          "re-calibrate; coverage guarantee may be void")
        elif level == "moderate":
            alerts.append(f"input/confidence drift moderate (PSI={psi:.2f}) — watch")
        if abstain_rate > 0.5:
            alerts.append(f"abstain rate {abstain_rate:.0%} > 50% — the model is "
                          "increasingly unsure (possible silent performance decay)")
        return {
            "n": n,
            "certified_rate": round(certified_rate, 4),
            "review_rate": round(review_rate, 4),
            "abstain_rate": round(abstain_rate, 4),
            "mean_confidence": round(float(np.mean(conf)), 4),
            # label-free accuracy proxy: under calibration, mean confidence ~ accuracy
            "estimated_accuracy_labelfree": round(float(np.mean(conf)), 4),
            "drift_psi": round(psi, 4),
            "drift_level": level,
            "alerts": alerts,
            "caveat": ("label-free signals assume a calibrated model (use temperature "
                       "scaling); drift_psi flags a change, not a quantified accuracy drop"),
        }

    def realized_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, *,
                         degraded=(2, 3)) -> dict:
        """When DELAYED labels arrive, the actually-realized accuracy + false-healthy rate."""
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)
        if y_true.shape != y_pred.shape or y_true.size == 0:
            raise ValueError("y_true and y_pred must be equal-length, non-empty")
        deg = set(int(d) for d in degraded)
        false_healthy = np.array([
            (int(t) in deg) and (int(p) not in deg)
            for t, p in zip(y_true, y_pred, strict=False)])
        return {
            "accuracy": round(float(np.mean(y_true == y_pred)), 4),
            "false_healthy_rate": round(float(np.mean(false_healthy)), 4),
            "n": int(y_true.size),
        }
