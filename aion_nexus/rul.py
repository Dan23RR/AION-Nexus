"""Calibrated Remaining Useful Life (RUL) with conformal prediction intervals.

The :mod:`aion_nexus.degradation` module gives a coarse STAGE and is explicit that
it is *not* a time-to-failure. This module fills that gap honestly: a calibrated
RUL (time-to-failure in seconds/cycles) with a distribution-free interval via
**Conformalized Quantile Regression** (Romano, Patterson & Candès, NeurIPS 2019).

    fit quantile regressors q_lo, q_hi on training run-to-failure data
    -> conformalize the interval on a held-out CALIBRATION set
    -> P(RUL_true in [lower, upper]) >= 1 - alpha   (marginal, finite-sample)

The guarantee holds under **exchangeability** of calibration and serving data
(the same caveat conformal coverage carries — cross-bearing / cross-machine
deployment breaks it). The honest behaviour when the model is uncertain (e.g. a
never-seen bearing) is a WIDE interval that still covers, not a confident wrong
number. This is the trustworthy-RUL play: we do not pretend to know the exact
time-to-failure; we emit a calibrated interval that is honest about uncertainty.

Run-to-failure labels MUST be the ACTUAL remaining life per window (e.g. FEMTO
``(n_files - 1 - i) * 10 s``), NEVER the positional ``file_idx / total`` proxy
the degradation stage uses — that is the §6.31 line this module must not cross.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# FEMTO/PRONOSTIA records a 0.1 s snapshot every 10 s -> one acquisition = 10 s.
FEMTO_DT_SECONDS = 10.0


@dataclass(frozen=True)
class RULEstimate:
    """A calibrated remaining-useful-life estimate with a conformal interval."""

    point: float                 # median time-to-failure
    lower: float                 # lower bound of the 1 - alpha interval (>= 0)
    upper: float                 # upper bound
    alpha: float                 # miscoverage level; coverage target 1 - alpha
    unit: str                    # "seconds" | "cycles" | ...
    method: str = "CQR"          # Conformalized Quantile Regression
    coverage_caveat: str = (
        "marginal 1-alpha coverage VALID ONLY under exchangeability of calibration "
        "and serving data; cross-bearing / cross-machine deployment breaks it and "
        "voids the guarantee (the interval may then under-cover)")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def as_dict(self) -> dict:
        return {"point": self.point, "lower": self.lower, "upper": self.upper,
                "width": self.width, "alpha": self.alpha, "unit": self.unit,
                "method": self.method, "coverage_caveat": self.coverage_caveat}


def rul_labels_for_run(n_files: int, dt_seconds: float = FEMTO_DT_SECONDS) -> np.ndarray:
    """True RUL (seconds) per acquisition of a COMPLETE run-to-failure: (N-1-i)*dt."""
    if n_files < 1:
        raise ValueError("n_files must be >= 1")
    return (np.arange(n_files - 1, -1, -1).astype(np.float64)) * float(dt_seconds)


def _stat_features(x: np.ndarray) -> list[float]:
    """Robust health-indicator features for one channel (no torch, no sklearn)."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)) + 1e-12)
    peak = float(np.max(np.abs(x)))
    p2p = float(np.ptp(x))
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    z = (x - mean) / std
    kurt = float(np.mean(z ** 4))           # kurtosis (impulsiveness grows with damage)
    skew = float(np.mean(z ** 3))
    crest = peak / rms                       # crest factor
    # spectral flatness: geomean(psd) / mean(psd) — tonal (low) vs broadband (high)
    psd = np.abs(np.fft.rfft(x - mean)) ** 2 + 1e-12
    flat = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))
    return [rms, peak, p2p, kurt, skew, crest, flat]


def health_features(signal: np.ndarray) -> np.ndarray:
    """Map a [C, N] (or [N]) window to a fixed health-indicator feature vector."""
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig[None, :]
    return np.concatenate([_stat_features(ch) for ch in sig])


def health_features_batch(signals) -> np.ndarray:
    """Stack :func:`health_features` over an iterable of windows -> (n, d)."""
    return np.vstack([health_features(s) for s in signals])


class ConformalRUL:
    """Conformalized Quantile Regression for RUL (Romano et al. 2019).

    Two-step, mirroring the :class:`~aion_nexus.verify.Verifier` discipline:
    ``fit`` trains the quantile regressors; ``calibrate`` computes the conformal
    correction on a HELD-OUT set so the interval is honestly sized. The regressors
    default to scikit-learn gradient-boosted quantile models; any object with the
    sklearn ``fit``/``predict`` API can be injected for testing.
    """

    def __init__(self, alpha: float = 0.1, *, unit: str = "seconds",
                 n_estimators: int = 200, max_depth: int = 3,
                 learning_rate: float = 0.05, random_state: int = 0) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.unit = unit
        self._lo_q = alpha / 2.0
        self._hi_q = 1.0 - alpha / 2.0
        self._gbr_kwargs = dict(n_estimators=n_estimators, max_depth=max_depth,
                                learning_rate=learning_rate, random_state=random_state)
        self._lo = self._mid = self._hi = None
        self._correction: float | None = None

    def _make(self, q: float):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(loss="quantile", alpha=q, **self._gbr_kwargs)

    def fit(self, features: np.ndarray, rul: np.ndarray) -> ConformalRUL:
        """Fit the lower / median / upper quantile regressors on run-to-failure data."""
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(rul, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("features must be (n, d) and rul (n,)")
        self._lo = self._make(self._lo_q).fit(x, y)
        self._mid = self._make(0.5).fit(x, y)
        self._hi = self._make(self._hi_q).fit(x, y)
        self._correction = 0.0   # uncalibrated until calibrate() is called
        return self

    def _raw_bounds(self, x: np.ndarray):
        lo = self._lo.predict(x)
        hi = self._hi.predict(x)
        # enforce non-crossing quantiles (gradient boosting can invert them)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return lo, hi

    def calibrate(self, features: np.ndarray, rul: np.ndarray) -> ConformalRUL:
        """CQR conformity: correction Q = quantile of max(qlo - y, y - qhi)."""
        if self._lo is None:
            raise RuntimeError("call fit() before calibrate()")
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(rul, dtype=np.float64)
        lo, hi = self._raw_bounds(x)
        scores = np.maximum(lo - y, y - hi)
        n = len(y)
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self._correction = float(np.quantile(scores, level, method="higher"))
        return self

    @property
    def is_calibrated(self) -> bool:
        return self._correction is not None

    def predict(self, features: np.ndarray) -> list[RULEstimate]:
        if self._lo is None:
            raise RuntimeError("call fit() (and calibrate()) before predict()")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        lo, hi = self._raw_bounds(x)
        mid = self._mid.predict(x)
        q = self._correction or 0.0
        out = []
        for i in range(len(x)):
            low = max(0.0, float(lo[i] - q))           # RUL is non-negative
            up = max(low, float(hi[i] + q))
            point = float(min(max(mid[i], low), up))
            out.append(RULEstimate(point=point, lower=low, upper=up,
                                   alpha=self.alpha, unit=self.unit))
        return out

    def predict_one(self, features: np.ndarray) -> RULEstimate:
        return self.predict(np.asarray(features, dtype=np.float64)[None, :])[0]

    def save(self, path) -> None:
        """Persist the fitted model (joblib). Reload with :func:`load_rul`."""
        import joblib
        joblib.dump({
            "lo": self._lo, "mid": self._mid, "hi": self._hi,
            "correction": self._correction, "alpha": self.alpha, "unit": self.unit,
        }, path)

    def coverage(self, features: np.ndarray, rul: np.ndarray) -> dict:
        """Empirical coverage + mean interval width on a labelled set (validation)."""
        ests = self.predict(features)
        y = np.asarray(rul, dtype=np.float64)
        covered = np.array([e.lower <= y[i] <= e.upper for i, e in enumerate(ests)])
        widths = np.array([e.width for e in ests])
        preds = np.array([e.point for e in ests])
        return {
            "coverage": float(np.mean(covered)),
            "target": 1.0 - self.alpha,
            "mean_width": float(np.mean(widths)),
            "median_width": float(np.median(widths)),
            "mae": float(np.mean(np.abs(preds - y))),
            "n": int(len(y)),
        }


def load_rul(path) -> ConformalRUL:
    """Reload a fitted :class:`ConformalRUL` saved with :meth:`ConformalRUL.save`.

    SECURITY: joblib uses pickle — load ONLY from a trusted path you produced. The
    server loads it solely from the operator-configured ``AION_RUL_ARTIFACT`` env.
    """
    import joblib
    d = joblib.load(path)
    m = ConformalRUL(alpha=float(d["alpha"]), unit=str(d.get("unit", "seconds")))
    m._lo, m._mid, m._hi = d["lo"], d["mid"], d["hi"]
    m._correction = d["correction"]
    return m
