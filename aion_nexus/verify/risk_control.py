"""Conformal Risk Control + RCPS — bound a SAFETY risk, not just miscoverage.

Marginal conformal prediction controls the miscoverage *rate*. Industrial
predictive maintenance needs more: the cost is ASYMMETRIC — calling a failing
bearing healthy is catastrophic, a false alarm is cheap. Risk control bounds the
EXPECTATION of an arbitrary monotone loss with a finite-sample, distribution-free
guarantee, by choosing a single threshold ``lambda`` on a calibration set.

Two guarantees, both distribution-free under exchangeability of calibration and
serving data (the same caveat conformal coverage carries):

- **Conformal Risk Control** (Angelopoulos, Bates, Candès, Jordan, Lei, Malik,
  2022): ``E[L_lambda_hat(X, Y)] <= alpha``. The expectation bound.
- **RCPS** (Bates, Angelopoulos, Lei, Malik, Jordan, 2021):
  ``P(R(lambda_hat) <= alpha) >= 1 - delta``. The high-probability bound
  (Hoeffding upper confidence bound).

The loss family ``L_lambda`` must be **monotone non-increasing in lambda** (a
larger lambda gives a larger/safer prediction set, hence a smaller loss). The
PdM loss provided here is the **false-healthy rate**: the bearing is degraded but
the prediction set flags nothing beyond 'healthy'. Pure numpy; no torch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FEMTO 4-class degradation taxonomy (config.CLASS_NAMES order):
#   0 normal, 1 early, 2 medium, 3 advanced.
HEALTHY_CLASSES = (0, 1)
DEGRADED_CLASSES = (2, 3)


@dataclass(frozen=True)
class RiskControlResult:
    """The chosen threshold and the guarantee it carries."""

    lambda_hat: float            # the calibrated set-inclusion threshold (1 - p cutoff)
    alpha: float                 # the target risk bound
    method: str                  # "CRC" | "RCPS"
    calibrated_risk: float       # empirical risk on the calibration set at lambda_hat
    n_calibration: int
    loss_name: str
    delta: float | None = None   # RCPS confidence parameter (None for CRC)
    guarantee: str = ""          # human-readable statement of what is bounded
    coverage_valid_under: str = (
        "exchangeability of calibration and serving data; cross-bearing / "
        "cross-machine deployment breaks it and voids the risk bound")

    def prediction_set(self, probs_row: np.ndarray) -> list[int]:
        """The risk-controlled prediction set for one probability vector."""
        return _set_at(np.asarray(probs_row, dtype=np.float64), self.lambda_hat)


def _set_at(probs: np.ndarray, lam: float) -> list[int]:
    """Set C_lambda(x) = {k : p_k >= 1 - lambda}; grows monotonically with lambda.

    Never empty: falls back to the argmax so a consumer always gets an action.
    """
    keep = np.where(probs >= 1.0 - lam)[0]
    if keep.size == 0:
        return [int(np.argmax(probs))]
    return [int(k) for k in keep]


def false_healthy_loss(probs: np.ndarray, labels: np.ndarray, lam: float, *,
                       degraded: tuple[int, ...] = DEGRADED_CLASSES) -> np.ndarray:
    """Per-sample loss: 1 if the bearing IS degraded but C_lambda flags only healthy.

    L_lambda(x, y) = 1{ y in degraded AND C_lambda(x) contains no degraded class }.
    Monotone NON-INCREASING in lambda (a larger set is more likely to flag
    degradation), bounded in [0, 1]. This is the catastrophic miss an industrial
    operator must bound: a degraded bearing the system did not flag.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    degraded_set = set(int(d) for d in degraded)
    thresh = 1.0 - lam
    flags_degraded = (probs[:, list(degraded_set)] >= thresh).any(axis=1)
    is_degraded = np.array([int(y) in degraded_set for y in labels])
    return (is_degraded & ~flags_degraded).astype(np.float64)


def empirical_risk(probs: np.ndarray, labels: np.ndarray, lam: float, *,
                   loss=false_healthy_loss) -> float:
    """Mean loss on (probs, labels) at threshold lambda — for validation."""
    return float(np.mean(loss(probs, labels, lam)))


def _risk_curve(probs, labels, lambdas, loss) -> np.ndarray:
    return np.array([np.mean(loss(probs, labels, float(lam))) for lam in lambdas])


def conformal_risk_control(probs: np.ndarray, labels: np.ndarray, *,
                           alpha: float = 0.05,
                           lambdas: np.ndarray | None = None,
                           loss=false_healthy_loss,
                           loss_name: str = "false-healthy rate",
                           bound: float = 1.0) -> RiskControlResult:
    """Pick the SMALLEST safe set whose expected loss is bounded by ``alpha`` (CRC).

    Angelopoulos et al. 2022:  lambda_hat = inf{ lambda :
        (n / (n + 1)) * R_hat(lambda) + B / (n + 1) <= alpha }
    where ``R_hat`` is the empirical mean loss on calibration and ``B`` the loss
    upper bound. Guarantees ``E[L_lambda_hat(X, Y)] <= alpha`` on an exchangeable
    test point. Scans ``lambdas`` ascending and returns the first that satisfies
    the bound (the loss is monotone non-increasing, so the satisfying set is an
    upper interval and its infimum is that first lambda).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n < 1 or probs.ndim != 2 or probs.shape[0] != n:
        raise ValueError("probs must be (n, k) and labels (n,) with n >= 1")
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, 101)
    rc = _risk_curve(probs, labels, lambdas, loss)
    adjusted = (n / (n + 1.0)) * rc + bound / (n + 1.0)
    ok = np.where(adjusted <= alpha)[0]
    # The full set (lambda = max) drives the loss to 0, so a solution exists once
    # n is large enough (B/(n+1) <= alpha). If none qualifies, fall back to the
    # largest lambda (the safest set) and report the residual risk honestly.
    idx = int(ok[0]) if ok.size else int(len(lambdas) - 1)
    lam = float(lambdas[idx])
    return RiskControlResult(
        lambda_hat=lam, alpha=float(alpha), method="CRC",
        calibrated_risk=float(rc[idx]), n_calibration=n, loss_name=loss_name,
        guarantee=(f"E[{loss_name}] <= {alpha:.3f} (conformal risk control, "
                   f"distribution-free, finite-sample) — the expected rate of "
                   f"failing to flag a degraded bearing is bounded by {alpha:.1%}"))


def rcps_threshold(probs: np.ndarray, labels: np.ndarray, *,
                   alpha: float = 0.05, delta: float = 0.1,
                   lambdas: np.ndarray | None = None,
                   loss=false_healthy_loss,
                   loss_name: str = "false-healthy rate") -> RiskControlResult:
    """High-probability risk control (RCPS, Bates et al. 2021) via Hoeffding UCB.

    Returns lambda_hat such that ``P(R(lambda_hat) <= alpha) >= 1 - delta``. Uses
    the Hoeffding upper confidence bound ``R_hat + sqrt(log(1/delta) / (2n))`` and
    picks the smallest lambda whose UCB is at most alpha. Stronger than CRC (bounds
    the risk with high probability, not just in expectation), slightly larger sets.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < delta < 1.0:
        raise ValueError("alpha and delta must be in (0, 1)")
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if lambdas is None:
        lambdas = np.linspace(0.0, 1.0, 101)
    rc = _risk_curve(probs, labels, lambdas, loss)
    ucb = rc + np.sqrt(np.log(1.0 / delta) / (2.0 * n))
    ok = np.where(ucb <= alpha)[0]
    idx = int(ok[0]) if ok.size else int(len(lambdas) - 1)
    lam = float(lambdas[idx])
    return RiskControlResult(
        lambda_hat=lam, alpha=float(alpha), method="RCPS", delta=float(delta),
        calibrated_risk=float(rc[idx]), n_calibration=n, loss_name=loss_name,
        guarantee=(f"P({loss_name} <= {alpha:.3f}) >= {1 - delta:.2f} (RCPS, "
                   f"Hoeffding) — with probability {1 - delta:.0%} the rate of "
                   f"failing to flag a degraded bearing is at most {alpha:.1%}"))
