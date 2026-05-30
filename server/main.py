"""FastAPI service for AION-NEXUS bearing-fault diagnosis.

Endpoints:
    GET  /health               — liveness + monitoring snapshot
    GET  /version              — model + API version
    POST /predict              — single signal as JSON body
    POST /predict_csv          — single signal as CSV upload (FEMTO format)
    POST /predict_batch        — multiple CSVs in one request
    POST /predict_long_signal  — window-then-aggregate over multi-second signal

Run:
    AION_CHECKPOINT=checkpoints/aion_nexus_v1.pth \\
      uvicorn server.main:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aion_nexus.inference import InferenceEngine
from aion_nexus.preprocessing import SignalValidationError
from aion_nexus.utils import load_signal_csv, segment_signal, aggregate_window_predictions
from aion_nexus.version import __version__

from server.schemas import (
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_logger = logging.getLogger("aion_nexus.server")


# ---- Engine bootstrap ------------------------------------------------------

CHECKPOINT_ENV = "AION_CHECKPOINT"
DEFAULT_CHECKPOINT = "checkpoints/aion_nexus_v1.pth"

app = FastAPI(
    title="AION-NEXUS",
    version=__version__,
    description="Production bearing-fault diagnosis from raw vibration signals.",
    contact={"name": "AION NEXUS", "email": "daniel.culotta@gmail.com"},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_engine() -> None:
    """Load the model checkpoint at server startup."""
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


def _require_engine() -> InferenceEngine:
    if app.state.engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Engine unavailable: {app.state.startup_error}",
        )
    return app.state.engine


# ---- Endpoints ------------------------------------------------------------

@app.get("/version")
def version() -> dict:
    return {"model": __version__, "api": __version__}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if app.state.engine is None:
        return HealthResponse(
            status="down",
            version=__version__,
            device="none",
            model_param_count=0,
            inference_count=0,
            running_avg_latency_ms=0.0,
        )
    info = app.state.engine.get_health()
    return HealthResponse(status="healthy", **info)


class JsonSignalRequest(BaseModel):
    signal: list[list[float]] = Field(
        ..., description="Signal as nested list, shape [2, N] or [N, 2], N >= 2560"
    )


def _csv_to_signal(contents: bytes) -> np.ndarray:
    """Parse a FEMTO acc_*.csv (6 cols, channels at idx 4,5) or a 2-col CSV."""
    arr = np.loadtxt(io.BytesIO(contents), delimiter=",")
    if arr.ndim != 2:
        raise ValueError(f"CSV must be 2D, got {arr.ndim}D shape {arr.shape}")
    if arr.shape[1] >= 6:
        return arr[:, [4, 5]].T  # FEMTO format
    return arr.T if arr.shape[0] != 2 else arr


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(body: JsonSignalRequest) -> PredictResponse:
    """Predict on a single 2-channel vibration window via JSON.

    For CSV upload, use `/predict_csv` instead.

    Body:
        signal: nested list of shape [2, N] or [N, 2], N >= 2560.
    """
    engine = _require_engine()
    signal = np.asarray(body.signal, dtype=np.float32)

    try:
        result = engine.predict(signal)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc))

    return PredictResponse(**result.to_dict())


@app.post(
    "/predict_csv",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_csv(file: Annotated[UploadFile, File(description="CSV file")]
                      ) -> PredictResponse:
    """Predict on a single 2-channel vibration window from a CSV upload.

    Accepts either FEMTO acc_*.csv format (channels at columns 4, 5) or a
    plain 2-column CSV.
    """
    engine = _require_engine()
    contents = await file.read()
    try:
        signal = _csv_to_signal(contents)
    except Exception as exc:
        raise HTTPException(400, f"CSV parse error: {exc}")

    try:
        result = engine.predict(signal)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc))

    return PredictResponse(**result.to_dict())


@app.post(
    "/predict_batch",
    response_model=BatchPredictResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch(files: Annotated[list[UploadFile], File()]) -> BatchPredictResponse:
    """Predict on multiple uploaded CSV files in one request."""
    if not files:
        raise HTTPException(400, "No files uploaded")
    engine = _require_engine()

    signals = []
    for f in files:
        contents = await f.read()
        try:
            arr = np.loadtxt(io.BytesIO(contents), delimiter=",")
            if arr.shape[1] >= 6:
                signals.append(arr[:, [4, 5]].T)
            else:
                signals.append(arr.T if arr.shape[0] != 2 else arr)
        except Exception as exc:
            raise HTTPException(400, f"CSV parse error in {f.filename}: {exc}")

    try:
        results = engine.predict_batch(signals)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc))

    total_latency = sum(r.latency_ms for r in results)
    return BatchPredictResponse(
        predictions=[PredictResponse(**r.to_dict()) for r in results],
        n_inputs=len(results),
        total_latency_ms=total_latency,
    )


class LongSignalRequest(BaseModel):
    signal: list[list[float]] = Field(..., description="Long signal, shape [2, N]")
    aggregation: str = Field("mean", description="'mean', 'majority', or 'max_class'")
    window: int = Field(2560, description="Window size in samples")
    stride: int | None = Field(None, description="Stride; None = non-overlapping")


@app.post("/predict_long_signal")
def predict_long_signal(req: LongSignalRequest) -> dict:
    """Window-then-aggregate prediction over a long signal (multi-second).

    Useful for bearing diagnosis from a multi-second recording rather than a
    single 0.1-second window.
    """
    engine = _require_engine()
    signal = np.asarray(req.signal, dtype=np.float32)

    try:
        windows = segment_signal(signal, window=req.window, stride=req.stride)
        results = engine.predict_batch(windows)
    except (SignalValidationError, ValueError) as exc:
        raise HTTPException(400, str(exc))

    probs = [np.array([r.probabilities[c] for c in ["normal", "early", "medium", "advanced"]])
             for r in results]
    try:
        idx, agg = aggregate_window_predictions(probs, method=req.aggregation)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    from aion_nexus.config import CLASS_NAMES, CLASS_DESCRIPTIONS, CLASS_ACTIONS
    name = CLASS_NAMES[idx]
    return {
        "predicted_class_index": idx,
        "predicted_class_name": name,
        "description": CLASS_DESCRIPTIONS[name],
        "aggregated_probabilities": {n: float(agg[i]) for i, n in enumerate(CLASS_NAMES)},
        "recommended_action": CLASS_ACTIONS[name],
        "n_windows": len(windows),
        "aggregation_method": req.aggregation,
        "model_version": __version__,
    }
