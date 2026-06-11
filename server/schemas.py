"""Pydantic schemas for the FastAPI server."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    n_inputs: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "down"]
    version: str
    architecture_version: str
    device: str
    model_param_count: int
    inference_count: int
    running_avg_latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
