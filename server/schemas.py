"""Pydantic schemas for the FastAPI server."""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Pydantic-level cap on the outer signal list ([2, N] or [N, 2] rows). Kept in
# sync with server.main.MAX_SIGNAL_ROWS; duplicated here to avoid importing the
# heavy server module from the schema module (circular-import safe). 262144
# samples is ~10.2 s at the FEMTO 25.6 kHz sampling rate.
MAX_SIGNAL_ROWS = 262_144

# Default cap on the number of windows /predict_long_signal will evaluate in one
# request. Each window is a full model forward; without a cap a small body with
# stride=1 expands into tens of thousands of forwards and pins a CPU worker for
# minutes (DoS). Overridable via AION_MAX_WINDOWS. 5000 windows is ~3.5 minutes
# of FEMTO recording at non-overlapping 0.1 s windows — far beyond any normal
# single request, while still cheap to reject.
MAX_WINDOWS_ENV = "AION_MAX_WINDOWS"
DEFAULT_MAX_WINDOWS = 5000


def _max_windows() -> int:
    raw = os.environ.get(MAX_WINDOWS_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_WINDOWS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_WINDOWS
    return value if value > 0 else DEFAULT_MAX_WINDOWS


class PredictResponse(BaseModel):
    predicted_class_index: int = Field(..., ge=0, le=3)
    predicted_class_name: Literal["normal", "early", "medium", "advanced"]
    description: str
    probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_band: Literal["low", "medium", "high"]
    recommended_action: dict
    latency_ms: float
    model_version: str
    # Additive plausibility-gate fields (backward-compatible: default to the
    # in-distribution / no-abstain case). When ood_flag is True the input is
    # implausible as bearing vibration, the engine has ABSTAINED, and
    # recommended_action is forced to no-escalation. The raw classifier output
    # (predicted_class_*, probabilities, confidence) is always preserved.
    ood_flag: bool = False
    ood_score: float = 0.0
    ood_reason: str | None = None
    abstain: bool = False


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n_inputs: int
    total_latency_ms: float


class DegradationEstimateSchema(BaseModel):
    """First-class degradation-STAGE estimate (honest coarse positional proxy).

    HONESTY CONTRACT (mirrors aion_nexus.degradation.DegradationEstimate): the
    stage is a COARSE positional proxy of the FEMTO run-to-failure life fraction
    (4 bands), NOT a calibrated time-to-failure / RUL in hours or cycles. The
    ``disclaimer`` field always carries this statement. ``conformal_stage_set``
    coverage is valid ONLY under exchangeability — cross-bearing/cross-machine
    deployment breaks it and voids the marginal guarantee (``coverage_caveat``).
    """

    stage_ordinal: int = Field(..., ge=0, le=3)
    stage_label: str
    degradation_index: float = Field(..., ge=0.0, le=1.0)
    stage_probabilities: dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    conformal_stage_set: list[int] | None = None
    conformal_stage_labels: list[str] | None = None
    calibrated: bool = False
    abstain: bool = False
    abstain_reason: str | None = None
    coverage_caveat: str | None = None
    disclaimer: str


class PredictDegradationResponse(PredictResponse):
    """``/predict_degradation`` response: the full predict contract plus the
    additive first-class ``degradation`` estimate. Inheriting from
    :class:`PredictResponse` keeps the prediction fields byte-identical to
    ``/predict`` (additive, non-breaking)."""

    degradation: DegradationEstimateSchema


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "down"]
    version: str
    architecture_version: str
    device: str
    model_param_count: int
    inference_count: int
    running_avg_latency_ms: float
    # Checkpoint hash pin (v2.6.0): the SHA-256 of the live checkpoint file
    # (``None`` if the engine was built from an in-memory model) and the expected
    # pin from ``AION_CHECKPOINT_SHA256`` (``None`` if unset). When both are
    # present they MUST match (the server refuses to start otherwise); exposing
    # them lets an operator / auditor confirm WHICH weights are serving.
    checkpoint_sha256: str | None = None
    expected_checkpoint_sha256: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


# ---- Certified serving (v2.6.0): /predict_certified and /verify --------------

class PredictCertifiedResponse(BaseModel):
    """``/predict_certified`` response: the prediction PLUS a sealed certificate.

    The certificate is returned as a plain dict (``Certificate.as_dict()``) rather
    than a fixed schema so the response stays byte-compatible with the dataclass
    as it evolves (additive fields like ``not_before`` / ``jti`` do not break the
    contract). ``pubkey`` is the Ed25519 PUBLIC key to verify against (``None`` for
    HMAC / NONE). ``verdict`` mirrors ``certificate["verdict"]`` for convenience.

    HONESTY: ``warning`` is set (non-null) ONLY when the certificate is UNSIGNED
    (``authentication = NONE``) because no signing key is configured — it states
    plainly that the certificate is integrity-only and NOT tamper-evident. A
    signed certificate carries ``warning = None``. The server NEVER silently
    pretends an unsigned cert is authenticated.
    """

    prediction: PredictResponse
    certificate: dict
    pubkey: str | None = None
    verdict: str
    warning: str | None = None


class VerifyRequest(BaseModel):
    """``/verify`` body: a certificate dict to audit, plus the EXPECTED issuer key.

    ``expected_pubkey`` is the out-of-band Ed25519 public key that identifies the
    trusted issuer. Supplying it is what separates genuine issuer authentication
    (``authenticity = VERIFIED``, ``trusted = True``) from mere self-consistency
    (``SELF-SIGNED``, ``trusted = False``) — see :func:`verify_certificate`.
    ``now_iso`` optionally pins "now" for deterministic validity-window checks
    (default: the server's real UTC clock).
    """

    certificate: dict
    expected_pubkey: str | None = None
    now_iso: str | None = None


class VerifyResponse(BaseModel):
    """``/verify`` result: the full :func:`verify_certificate` verdict.

    ``trusted`` is the SINGLE safe flag to gate on (True ONLY when integrity holds
    AND the issuer is VERIFIED against the expected key AND the cert is in its
    validity window). ``expired`` / ``not_yet_valid`` are present only when the
    certificate carried a validity window.
    """

    integrity_ok: bool
    authenticity: str
    trusted: bool
    detail: str
    expired: bool | None = None
    not_yet_valid: bool | None = None


class LongSignalRequest(BaseModel):
    """Request body for /predict_long_signal (window-then-aggregate).

    Hardened against CPU-DoS (red-team kill-shot #1): the outer signal list is
    capped at MAX_SIGNAL_ROWS, window and stride are bounded to positive values
    with a sane minimum, and a model-level validator rejects any request whose
    implied window count exceeds AION_MAX_WINDOWS — computed from list lengths
    BEFORE any numpy allocation or model forward, so an attack body is rejected
    in well under a second.
    """

    signal: list[list[float]] = Field(
        ...,
        max_length=MAX_SIGNAL_ROWS,
        description="Long signal, shape [2, N] or [N, 2]",
    )
    aggregation: str = Field(
        "mean", description="'mean', 'majority', or 'max_class'"
    )
    # window must be >= SIGNAL_LENGTH (2560); a smaller window cannot be fed to
    # the model. ge=2560 also blocks window=0/1 stride-explosion attacks.
    window: int = Field(
        2560, ge=2560, le=MAX_SIGNAL_ROWS,
        description="Window size in samples (>= 2560)",
    )
    # stride must be strictly positive; stride=0 would loop forever / divide by
    # zero. None means non-overlapping (stride = window).
    stride: int | None = Field(
        None, gt=0, le=MAX_SIGNAL_ROWS,
        description="Stride in samples (> 0); None = non-overlapping",
    )

    @model_validator(mode="after")
    def _bound_window_count(self) -> LongSignalRequest:
        """Reject requests whose window count would exceed AION_MAX_WINDOWS.

        Computed from the longest axis of the (possibly [N, 2]) signal list,
        purely from Python list lengths — no numpy, no model forward — so the
        worst-case attack (huge body, stride=1) is rejected cheaply.
        """
        if not self.signal:
            return self  # downstream validation handles the empty/short case
        n_rows = len(self.signal)
        n_cols = len(self.signal[0]) if self.signal[0] is not None else 0
        # Signal length is the longer axis ([2, N] -> N, or [N, 2] -> N).
        n = max(n_rows, n_cols)
        stride = self.stride if self.stride is not None else self.window
        if n < self.window:
            return self  # too-short signal is rejected later with a clear error
        n_windows = (n - self.window) // stride + 1
        cap = _max_windows()
        if n_windows > cap:
            raise ValueError(
                f"request would evaluate {n_windows} windows (signal length "
                f"~{n}, window {self.window}, stride {stride}), exceeding the "
                f"limit of {cap} (configure via {MAX_WINDOWS_ENV}). Reduce the "
                "signal length, increase the window/stride, or split the request."
            )
        return self
