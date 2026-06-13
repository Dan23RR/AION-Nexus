"""FastAPI service for AION-NEXUS bearing-fault diagnosis.

Endpoints:
    GET  /health               — liveness + monitoring snapshot (never requires API key)
    GET  /version              — model + API version
    GET  /metrics              — Prometheus-format metrics
    POST /predict              — single signal as JSON body
    POST /predict_csv          — single signal as CSV upload (FEMTO format)
    POST /predict_batch        — multiple CSVs in one request
    POST /predict_long_signal  — window-then-aggregate over multi-second signal

Environment:
    AION_CHECKPOINT       — checkpoint path (default checkpoints/aion_nexus_v1.pth)
    AION_DEVICE           — torch device (default cpu)
    AION_API_KEY          — if set, all endpoints except /health require the
                            X-API-Key header to match it
    AION_MAX_BODY_BYTES   — request body size limit (default 10485760 = 10 MiB)
    AION_CORS_ORIGINS     — comma-separated allowed origins; unset = no CORS
                            middleware (same-origin only). Wildcard "*" disables
                            credentials (never wildcard + credentials).
    AION_LOG_JSON         — "1" enables structured JSON logging with request_id

Run:
    AION_CHECKPOINT=checkpoints/aion_nexus_v1.pth \\
      uvicorn server.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from aion_nexus.inference import InferenceEngine
from aion_nexus.preprocessing import SignalValidationError
from aion_nexus.utils import aggregate_window_predictions, segment_signal
from aion_nexus.version import __version__
from server.schemas import (
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    LongSignalRequest,
    PredictResponse,
)

# ---- Configuration ----------------------------------------------------------

CHECKPOINT_ENV = "AION_CHECKPOINT"
DEFAULT_CHECKPOINT = "checkpoints/aion_nexus_v1.pth"
API_KEY_ENV = "AION_API_KEY"
MAX_BODY_ENV = "AION_MAX_BODY_BYTES"
DEFAULT_MAX_BODY_BYTES = 10_485_760  # 10 MiB
CORS_ENV = "AION_CORS_ORIGINS"
LOG_JSON_ENV = "AION_LOG_JSON"
MAX_BATCH_FILES_ENV = "AION_MAX_BATCH_FILES"
DEFAULT_MAX_BATCH_FILES = 256  # cap the number of uploads per /predict_batch call
# Pydantic-level cap on the outer signal list ([2, N] or [N, 2] rows). Combined
# with the byte-level body cap this bounds worst-case parse cost. 262144 samples
# is ~10.2 s at the FEMTO 25.6 kHz sampling rate.
MAX_SIGNAL_ROWS = 262_144

# No-escalation action returned when the aggregated plausibility gate abstains on
# /predict_long_signal (kept structurally identical to the engine's per-window
# abstain action in aion_nexus.inference). An implausible recording must never
# escalate to an automated stop/inspection/replacement.
_AGGREGATE_ABSTAIN_ACTION: dict = {
    "alert_level": 0,
    "stop_machine": False,
    "schedule_inspection": False,
    "abstain": True,
}


# ---- Logging (optional structured JSON with request_id) ---------------------

class _JsonLogFormatter(logging.Formatter):
    """One JSON object per log line; carries request_id when present."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _configure_logging() -> None:
    if os.environ.get(LOG_JSON_ENV, "") == "1":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonLogFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )


_configure_logging()
_logger = logging.getLogger("aion_nexus.server")


# ---- Metrics (Prometheus if available, minimal text fallback otherwise) -----

class _Metrics:
    """Prediction/latency/error metrics, exposed at /metrics in Prometheus text
    format. Uses prometheus_client when importable; otherwise falls back to a
    minimal hand-rendered exposition (same metric names) so /metrics always works.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fb_predictions: dict[str, int] = {}
        self._fb_errors: dict[tuple[str, int], int] = {}
        self._fb_latency_sum = 0.0
        self._fb_latency_count = 0
        try:
            from prometheus_client import (
                CONTENT_TYPE_LATEST,
                CollectorRegistry,
                Counter,
                Histogram,
                generate_latest,
            )
            # Dedicated registry: avoids duplicate-metric errors on re-import.
            self._registry = CollectorRegistry()
            self._predictions = Counter(
                "aion_predictions_total", "Predictions served, by predicted class",
                ["predicted_class"], registry=self._registry,
            )
            self._errors = Counter(
                "aion_request_errors_total",
                "HTTP responses with status >= 400, by endpoint and status",
                ["endpoint", "status"], registry=self._registry,
            )
            self._latency = Histogram(
                "aion_inference_latency_ms", "Model inference latency per signal (ms)",
                registry=self._registry,
                buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000),
            )
            self._generate_latest = generate_latest
            self.content_type = CONTENT_TYPE_LATEST
            self._prom = True
        except ImportError:
            self._prom = False
            self.content_type = "text/plain; version=0.0.4; charset=utf-8"

    def observe_prediction(self, class_name: str, latency_ms: float) -> None:
        if self._prom:
            self._predictions.labels(predicted_class=class_name).inc()
            self._latency.observe(latency_ms)
            return
        with self._lock:
            self._fb_predictions[class_name] = self._fb_predictions.get(class_name, 0) + 1
            self._fb_latency_sum += latency_ms
            self._fb_latency_count += 1

    def observe_error(self, endpoint: str, status: int) -> None:
        if self._prom:
            self._errors.labels(endpoint=endpoint, status=str(status)).inc()
            return
        with self._lock:
            key = (endpoint, status)
            self._fb_errors[key] = self._fb_errors.get(key, 0) + 1

    def render(self) -> bytes:
        if self._prom:
            return self._generate_latest(self._registry)
        with self._lock:
            lines = ["# TYPE aion_predictions_total counter"]
            for name, count in sorted(self._fb_predictions.items()):
                lines.append(f'aion_predictions_total{{predicted_class="{name}"}} {count}')
            lines.append("# TYPE aion_request_errors_total counter")
            for (endpoint, status), count in sorted(self._fb_errors.items()):
                lines.append(
                    f'aion_request_errors_total{{endpoint="{endpoint}",status="{status}"}} {count}'
                )
            lines.append("# TYPE aion_inference_latency_ms summary")
            lines.append(f"aion_inference_latency_ms_sum {self._fb_latency_sum}")
            lines.append(f"aion_inference_latency_ms_count {self._fb_latency_count}")
        return ("\n".join(lines) + "\n").encode()


METRICS = _Metrics()


# ---- Engine bootstrap --------------------------------------------------------

def _load_engine(app: FastAPI) -> None:
    """Load the model checkpoint (called once from the lifespan handler)."""
    checkpoint = os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
    device = os.environ.get("AION_DEVICE", "cpu")
    if not Path(checkpoint).exists():
        _logger.error(
            "Checkpoint not found at %s. Set %s env var or place file in %s.",
            checkpoint, CHECKPOINT_ENV, DEFAULT_CHECKPOINT,
        )
        # Allow server to start in 'degraded' mode for /health checks
        app.state.engine = None
        app.state.startup_error = f"Checkpoint not found: {checkpoint}"
        return
    try:
        app.state.engine = InferenceEngine.from_checkpoint(checkpoint, device=device)
        app.state.startup_error = None
        _logger.info("AION-NEXUS engine ready on %s", device)
    except Exception as exc:
        _logger.exception("Failed to load engine")
        app.state.engine = None
        app.state.startup_error = str(exc)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown handler (replaces the deprecated @app.on_event)."""
    _load_engine(app)
    yield


app = FastAPI(
    title="AION-NEXUS",
    version=__version__,
    description="Production bearing-fault diagnosis from raw vibration signals.",
    contact={"name": "AION NEXUS", "email": "daniel.culotta@gmail.com"},
    lifespan=_lifespan,
)


# ---- Security: optional API key ----------------------------------------------

def _require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """If AION_API_KEY is set, require a matching X-API-Key header.

    Applied to every endpoint except /health (liveness probes must not need
    credentials). Constant-time comparison; no-op when the env var is unset.
    """
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return
    # Compare bytes: str compare_digest raises TypeError on non-ASCII header
    # values (Starlette decodes headers as latin-1), turning a bad key into a 500.
    supplied = (x_api_key or "").encode("utf-8", errors="surrogateescape")
    if not secrets.compare_digest(supplied, expected.encode("utf-8")):
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key (X-API-Key header)"
        )


# ---- Middleware ---------------------------------------------------------------
# Starlette executes middleware in reverse registration order: the body-size
# limit is registered first (innermost), the request-context middleware second
# (so it also logs/counts 413 rejections), CORS last (outermost).

@app.middleware("http")
async def _limit_body_size(request: Request, call_next):
    """Reject requests whose declared body exceeds AION_MAX_BODY_BYTES with 413."""
    max_bytes = int(os.environ.get(MAX_BODY_ENV, str(DEFAULT_MAX_BODY_BYTES)))
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"error": "invalid Content-Length header", "detail": None},
            )
        if declared > max_bytes:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "request body too large",
                    "detail": f"Body of {declared} bytes exceeds the limit of "
                              f"{max_bytes} bytes (configure via {MAX_BODY_ENV}).",
                },
            )
    return await call_next(request)


@app.middleware("http")
async def _request_context(request: Request, call_next):
    """Request-ID propagation + access log + error-rate metrics."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    response.headers["X-Request-ID"] = request_id
    if response.status_code >= 400:
        METRICS.observe_error(request.url.path, response.status_code)
    _logger.info(
        "%s %s -> %d (%.1f ms)",
        request.method, request.url.path, response.status_code, duration_ms,
        extra={"request_id": request_id},
    )
    return response


# CORS: opt-in only. Default (env unset) = no CORS middleware at all.
_cors_origins = [o.strip() for o in os.environ.get(CORS_ENV, "").split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        # Never combine wildcard origins with credentials (spec-invalid and unsafe).
        allow_credentials="*" not in _cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    )


def _require_engine() -> InferenceEngine:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Engine unavailable: {getattr(app.state, 'startup_error', 'not loaded')}",
        )
    return engine


# ---- Endpoints ------------------------------------------------------------

@app.get("/version", dependencies=[Depends(_require_api_key)])
def version() -> dict:
    return {"model": __version__, "api": __version__}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    engine = getattr(app.state, "engine", None)
    if engine is None:
        return HealthResponse(
            status="down",
            version=__version__,
            architecture_version="none",
            device="none",
            model_param_count=0,
            inference_count=0,
            running_avg_latency_ms=0.0,
        )
    info = engine.get_health()
    return HealthResponse(status="healthy", **info)


@app.get("/metrics", dependencies=[Depends(_require_api_key)])
def metrics() -> Response:
    """Prometheus-format metrics (prediction counts, latency, error counts)."""
    return Response(content=METRICS.render(), media_type=METRICS.content_type)


class JsonSignalRequest(BaseModel):
    signal: list[list[float]] = Field(
        ...,
        max_length=MAX_SIGNAL_ROWS,
        description="Signal as nested list, shape [2, N] or [N, 2], N >= 2560",
    )


def _read_upload_capped(upload: UploadFile) -> bytes:
    """Read an upload enforcing AION_MAX_BODY_BYTES even for chunked requests.

    The body-size middleware only sees a declared Content-Length; a chunked
    upload bypasses it, so the cap is enforced again here at read time.
    """
    max_bytes = int(os.environ.get(MAX_BODY_ENV, str(DEFAULT_MAX_BODY_BYTES)))
    contents = upload.file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds the limit of {max_bytes} bytes "
                   f"(configure via {MAX_BODY_ENV}).",
        )
    return contents


def _csv_to_signal(contents: bytes) -> np.ndarray:
    """Parse a FEMTO acc_*.csv (6 cols, channels at idx 4,5) or a 2-col CSV."""
    arr = np.loadtxt(io.BytesIO(contents), delimiter=",")
    if arr.ndim != 2:
        raise ValueError(f"CSV must be 2D, got {arr.ndim}D shape {arr.shape}")
    if arr.shape[1] >= 6:
        return arr[:, [4, 5]].T  # FEMTO format
    return arr.T if arr.shape[0] != 2 else arr


# NOTE: the predict endpoints are deliberately sync (`def`, not `async def`):
# FastAPI runs them in its threadpool, so a long CPU-bound model forward does
# not block the event loop (keeps /health responsive under load).

@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict(body: JsonSignalRequest) -> PredictResponse:
    """Predict on a single 2-channel vibration window via JSON.

    For CSV upload, use `/predict_csv` instead.

    Body:
        signal: nested list of shape [2, N] or [N, 2], N >= 2560.
    """
    engine = _require_engine()
    # A ragged list (e.g. [[1,2,3],[1]]) makes np.asarray(dtype=float32) raise
    # ValueError BEFORE engine.predict's try/except — which previously escaped as
    # an unhandled 500 with a stacktrace (kill-shot #2). Catch it here -> 400.
    try:
        signal = np.asarray(body.signal, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            400, f"Malformed signal (ragged or non-numeric rows): {exc}"
        ) from exc

    try:
        result = engine.predict(signal)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    METRICS.observe_prediction(result.predicted_class_name, result.latency_ms)
    return PredictResponse(**result.to_dict())


@app.post(
    "/predict_csv",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict_csv(file: Annotated[UploadFile, File(description="CSV file")]
                ) -> PredictResponse:
    """Predict on a single 2-channel vibration window from a CSV upload.

    Accepts either FEMTO acc_*.csv format (channels at columns 4, 5) or a
    plain 2-column CSV.
    """
    engine = _require_engine()
    contents = _read_upload_capped(file)
    try:
        signal = _csv_to_signal(contents)
    except Exception as exc:
        raise HTTPException(400, f"CSV parse error: {exc}") from exc

    try:
        result = engine.predict(signal)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    METRICS.observe_prediction(result.predicted_class_name, result.latency_ms)
    return PredictResponse(**result.to_dict())


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict_batch(files: Annotated[list[UploadFile], File()]) -> BatchPredictResponse:
    """Predict on multiple uploaded CSV files in one request."""
    if not files:
        raise HTTPException(400, "No files uploaded")
    max_files = int(os.environ.get(MAX_BATCH_FILES_ENV, str(DEFAULT_MAX_BATCH_FILES)))
    if len(files) > max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} > limit {max_files} "
                   f"(configure via {MAX_BATCH_FILES_ENV}).",
        )
    engine = _require_engine()

    signals = []
    for f in files:
        contents = _read_upload_capped(f)
        try:
            arr = np.loadtxt(io.BytesIO(contents), delimiter=",")
            if arr.shape[1] >= 6:
                signals.append(arr[:, [4, 5]].T)
            else:
                signals.append(arr.T if arr.shape[0] != 2 else arr)
        except Exception as exc:
            raise HTTPException(400, f"CSV parse error in {f.filename}: {exc}") from exc

    try:
        results = engine.predict_batch(signals)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    for r in results:
        METRICS.observe_prediction(r.predicted_class_name, r.latency_ms)
    total_latency = sum(r.latency_ms for r in results)
    return BatchPredictResponse(
        predictions=[PredictResponse(**r.to_dict()) for r in results],
        n_inputs=len(results),
        total_latency_ms=total_latency,
    )


@app.post("/predict_long_signal", dependencies=[Depends(_require_api_key)])
def predict_long_signal(req: LongSignalRequest) -> dict:
    """Window-then-aggregate prediction over a long signal (multi-second).

    Useful for bearing diagnosis from a multi-second recording rather than a
    single 0.1-second window.

    DoS hardening (kill-shot #1) is enforced at the schema layer
    (``LongSignalRequest``): the signal list is length-capped, window/stride are
    bounded, and the implied window count is checked against ``AION_MAX_WINDOWS``
    BEFORE this handler runs — so an attack body is rejected with 422 before any
    numpy allocation or model forward.

    Plausibility hardening (kill-shot #4) is propagated to the AGGREGATED verdict:
    per-window OOD flags from ``predict_batch`` are aggregated, and if a majority
    of windows are implausible (white noise / saturated / quasi-constant) the
    response ABSTAINS — ``recommended_action`` is forced to no-escalation and the
    ``ood_*`` fields are surfaced — so an implausible recording cannot escalate.
    """
    engine = _require_engine()
    # Guard the ndarray conversion: a ragged list ([[1,2,3],[1]]) makes
    # np.asarray raise ValueError, which would otherwise escape as a 500
    # (kill-shot #2, same root cause as /predict).
    try:
        signal = np.asarray(req.signal, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            400, f"Malformed signal (ragged or non-numeric rows): {exc}"
        ) from exc

    try:
        windows = segment_signal(signal, window=req.window, stride=req.stride)
        results = engine.predict_batch(windows)
    except (SignalValidationError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    probs = [np.array([r.probabilities[c] for c in ["normal", "early", "medium", "advanced"]])
             for r in results]
    try:
        idx, agg = aggregate_window_predictions(probs, method=req.aggregation)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    from aion_nexus.config import CLASS_ACTIONS, CLASS_DESCRIPTIONS, CLASS_NAMES
    name = CLASS_NAMES[idx]

    # Propagate the per-window plausibility gate to the AGGREGATED verdict
    # (kill-shot #4 on the long-signal path): predict_batch already abstains
    # per window, but the aggregation above ignores those per-window actions and
    # would otherwise re-derive an escalating action straight from CLASS_ACTIONS.
    # That let a whole recording of white noise / saturated / quasi-constant
    # windows escalate to alert_level >= 1 here even though every single window
    # was individually flagged OOD and abstained. We aggregate the OOD verdict:
    # if a majority of windows are implausible, the recording as a whole is
    # implausible -> ABSTAIN (no escalation), and the OOD fields are surfaced.
    n_ood = sum(1 for r in results if r.ood_flag)
    ood_fraction = n_ood / len(results) if results else 0.0
    aggregate_ood = ood_fraction > 0.5
    if aggregate_ood:
        recommended_action = dict(_AGGREGATE_ABSTAIN_ACTION)
        # First OOD reason is representative (all share the same gate).
        ood_reason = next((r.ood_reason for r in results if r.ood_flag), None)
    else:
        recommended_action = CLASS_ACTIONS[name]
        ood_reason = None

    METRICS.observe_prediction(name, sum(r.latency_ms for r in results))
    return {
        "predicted_class_index": idx,
        "predicted_class_name": name,
        "description": CLASS_DESCRIPTIONS[name],
        "aggregated_probabilities": {n: float(agg[i]) for i, n in enumerate(CLASS_NAMES)},
        "recommended_action": recommended_action,
        "n_windows": len(windows),
        "aggregation_method": req.aggregation,
        "model_version": __version__,
        # Aggregated plausibility gate (additive, backward-compatible).
        "ood_flag": aggregate_ood,
        "ood_windows": n_ood,
        "ood_fraction": round(ood_fraction, 4),
        "ood_reason": ood_reason,
        "abstain": aggregate_ood,
    }
