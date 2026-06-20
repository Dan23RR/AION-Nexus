"""FastAPI service for AION-NEXUS bearing-fault diagnosis.

Endpoints:
    GET  /health               — liveness + monitoring snapshot (never requires API key)
    GET  /version              — model + API version
    GET  /metrics              — Prometheus-format metrics
    POST /predict              — single signal as JSON body
    POST /predict_certified    — single signal -> SIGNED, auditable Certificate
    POST /verify               — audit a certificate against an expected pubkey
    POST /predict_csv          — single signal as CSV upload (FEMTO format)
    POST /predict_batch        — multiple CSVs in one request
    POST /predict_long_signal  — window-then-aggregate over multi-second signal

Environment:
    AION_CHECKPOINT       — checkpoint path (default checkpoints/aion_nexus_v1.pth)
    AION_DEVICE           — torch device (default cpu)
    AION_API_KEY          — if set, all endpoints except /health require the
                            X-API-Key header to match it
    AION_MAX_BODY_BYTES   — request body size limit (default 10485760 = 10 MiB)
    AION_MAX_BATCH_FILES  — max uploads per /predict_batch call (default 256)
    AION_MAX_BATCH_BYTES  — aggregate byte budget per /predict_batch call
                            (default 52428800 = 50 MiB)
    AION_CORS_ORIGINS     — comma-separated allowed origins; unset = no CORS
                            middleware (same-origin only). Wildcard "*" disables
                            credentials (never wildcard + credentials).
    AION_LOG_JSON         — "1" enables structured JSON logging with request_id

Certified serving (v2.6.0 — wires the signed certificate into the product):
    VERIFY_ED25519_SEED       — Ed25519 signing seed; when set, /predict_certified
                                emits an Ed25519-SIGNED certificate (third-party
                                verifiable with the public key alone). Without it
                                the cert is authentication=NONE (integrity-only).
    VERIFY_HMAC_KEY           — fallback HMAC signing key (symmetric; verifier can
                                also forge — see aion_nexus.verify.signing).
    AION_CERT_TTL_SECONDS     — certificate validity window in seconds (default
                                86400 = 24h). The signature covers the window, so
                                an expired cert fails verification (anti-replay).
    AION_CERT_KEY_ID          — opaque id of the signing key, stamped on the cert
                                (for key rotation / audit).
    AION_REQUIRE_SIGNED_CERT  — "1" = STRICT: /predict_certified refuses (503) to
                                emit an unsigned certificate when no key is set.
    VERIFY_CERT_STORE         — path to the append-only certificate audit log
                                (default ./certificates.jsonl). Every certified
                                prediction is appended to the hash-chained store.

Checkpoint pinning (v2.6.0 — attest WHICH weights are serving):
    AION_CHECKPOINT_SHA256       — expected SHA-256 of the checkpoint file. When
                                   set, the server refuses to start (non-degraded)
                                   if the live checkpoint's hash differs.
    AION_REQUIRE_CHECKPOINT_PIN  — "1" = STRICT: refuse to start unless
                                   AION_CHECKPOINT_SHA256 is set.

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
from typing import Annotated, Literal

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
    AnnexIVRequest,
    BatchPredictResponse,
    BearingGeometrySchema,
    ErrorResponse,
    EvidenceRequest,
    HealthResponse,
    LongSignalRequest,
    PredictCertifiedResponse,
    PredictDegradationResponse,
    PredictResponse,
    PredictRULResponse,
    VerifyRequest,
    VerifyResponse,
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
# Aggregate BYTE budget across all files in one /predict_batch call. The per-file
# cap (AION_MAX_BODY_BYTES) and the file-count cap each bound one axis, but 256
# files just under the body cap could still total gigabytes; this caps the SUM so
# a batch cannot exhaust memory by staying under both per-axis limits.
MAX_BATCH_BYTES_ENV = "AION_MAX_BATCH_BYTES"
DEFAULT_MAX_BATCH_BYTES = 52_428_800  # 50 MiB
# Pydantic-level cap on the outer signal list ([2, N] or [N, 2] rows). Combined
# with the byte-level body cap this bounds worst-case parse cost. 262144 samples
# is ~10.2 s at the FEMTO 25.6 kHz sampling rate.
MAX_SIGNAL_ROWS = 262_144

# ---- Certified serving config (v2.6.0) --------------------------------------
ED25519_SEED_ENV = "VERIFY_ED25519_SEED"
HMAC_KEY_ENV = "VERIFY_HMAC_KEY"
CERT_TTL_ENV = "AION_CERT_TTL_SECONDS"
DEFAULT_CERT_TTL_SECONDS = 86_400  # 24h validity window for a served certificate
CERT_KEY_ID_ENV = "AION_CERT_KEY_ID"
REQUIRE_SIGNED_CERT_ENV = "AION_REQUIRE_SIGNED_CERT"  # "1" -> strict: refuse unsigned
# Real conformal-calibration artifact (v2.16.0). When this points at a calibration
# .npz built from a held-out, group-disjoint REAL split (scripts/build_calibration),
# the served verifier calibrates on real data and the certificate's coverage_basis
# becomes "real-holdout". Absent -> the runnable synthetic placeholder, surfaced
# honestly on every response. Default resolves next to the loaded checkpoint.
CALIBRATION_NPZ_ENV = "AION_CALIBRATION_NPZ"
DEFAULT_CALIBRATION_NPZ = "checkpoints/calibration_v1.npz"
REQUIRE_REAL_CALIBRATION_ENV = "AION_REQUIRE_REAL_CALIBRATION"  # "1" -> 503 if placeholder
# Conformal Risk Control (v2.18.0): bound the expected false-healthy rate (failing
# to flag a degraded bearing) at this level. Set AION_RISK_ALPHA="off" to disable.
RISK_ALPHA_ENV = "AION_RISK_ALPHA"
DEFAULT_RISK_ALPHA = 0.05
# Calibrated RUL model (v2.19.0): a fitted ConformalRUL artifact (joblib, produced
# by the deployer from run-to-failure data). Loaded ONLY from this trusted path.
RUL_ARTIFACT_ENV = "AION_RUL_ARTIFACT"
# Key registry (v2.20.0): a publishable KeyRing JSON (no secrets) enabling rotation
# + revocation enforcement at /verify — a revoked key_id makes its certs untrusted.
KEYRING_ARTIFACT_ENV = "AION_KEYRING"

# ---- Checkpoint pin config (v2.6.0) -----------------------------------------
CHECKPOINT_SHA256_ENV = "AION_CHECKPOINT_SHA256"           # expected file hash
REQUIRE_CHECKPOINT_PIN_ENV = "AION_REQUIRE_CHECKPOINT_PIN"  # "1" -> require the pin

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
    """Load the model checkpoint (called once from the lifespan handler).

    Checkpoint pinning (v2.6.0): if ``AION_CHECKPOINT_SHA256`` is set, the file's
    hash is verified before loading and a mismatch is a HARD failure (the engine
    is NOT loaded — degraded mode — so the server cannot silently serve the wrong
    weights). In STRICT mode (``AION_REQUIRE_CHECKPOINT_PIN=1``) a MISSING pin is
    itself a hard failure. Default (no pin) is unchanged for backward compat.
    """
    checkpoint = os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
    device = os.environ.get("AION_DEVICE", "cpu")
    expected_sha = os.environ.get(CHECKPOINT_SHA256_ENV) or None
    require_pin = os.environ.get(REQUIRE_CHECKPOINT_PIN_ENV, "") == "1"

    app.state.expected_checkpoint_sha256 = expected_sha

    if require_pin and not expected_sha:
        msg = (f"strict checkpoint pinning: {REQUIRE_CHECKPOINT_PIN_ENV}=1 but "
               f"{CHECKPOINT_SHA256_ENV} is not set. Refusing to start without a "
               "checkpoint hash to verify.")
        _logger.error(msg)
        app.state.engine = None
        app.state.startup_error = msg
        return

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
        # expected_sha (when set) makes from_checkpoint RAISE on a hash mismatch,
        # so a wrong/tampered checkpoint fails closed -> degraded mode, never served.
        app.state.engine = InferenceEngine.from_checkpoint(
            checkpoint, device=device, expected_sha256=expected_sha)
        app.state.startup_error = None
        if expected_sha:
            _logger.info("Checkpoint hash pin OK (%s)", expected_sha)
        _logger.info("AION-NEXUS engine ready on %s", device)
    except Exception as exc:
        _logger.exception("Failed to load engine")
        app.state.engine = None
        app.state.startup_error = str(exc)


def _calibration_artifact_path() -> Path:
    """Resolve the calibration-artifact path (env override, else next to the package)."""
    raw = os.environ.get(CALIBRATION_NPZ_ENV)
    if raw and raw.strip():
        return Path(raw).expanduser()
    # Default resolves relative to the package root (parent of this server/ dir).
    return Path(__file__).resolve().parent.parent / DEFAULT_CALIBRATION_NPZ


def _build_certifier(app: FastAPI) -> None:
    """Build the calibrated Verifier + certificate store for /predict_certified.

    Calibration basis (v2.16.0 — crack-#1 fix). The Verifier is calibrated on a
    REAL held-out artifact when one is present (``AION_CALIBRATION_NPZ`` or
    ``checkpoints/calibration_v1.npz``, built by ``scripts/build_calibration`` from
    a leakage-checked split), in which case ``app.state.coverage_basis =
    "real-holdout"``. Absent, it falls back to a SMALL synthetic placeholder so the
    conformal API stays runnable, and ``coverage_basis = "synthetic-placeholder"``.

    HONESTY (workspace 6.31): a synthetic calibration is NOT valid for any real
    deployment — conformal coverage holds only under exchangeability with the
    SERVING data, and random windows are not exchangeable with real bearings. The
    basis is stamped (tamper-evidently) into every certificate's
    ``coverage_guarantee`` and surfaced on every response, so the placeholder is
    never silently passed off as a real coverage number. The certificate's verdict
    logic, signature and audit trail are real regardless of basis.
    """
    engine = getattr(app.state, "engine", None)
    app.state.coverage_basis = None
    app.state.calibration_meta = None
    app.state.coverage_temperature = 1.0
    app.state.risk_control = None
    app.state.monitor = None
    if engine is None:
        app.state.verifier = None
        app.state.cert_store = None
        return
    try:
        import numpy as _np

        from aion_nexus.config import CLASS_NAMES
        from aion_nexus.serving_calibration import (
            BASIS_SYNTHETIC,
            apply_temperature,
            fit_temperature,
            load_calibration,
            synthetic_demo_probs,
        )
        from aion_nexus.verify import CertificateStore, Verifier

        class_names = list(CLASS_NAMES)
        basis = BASIS_SYNTHETIC
        meta: dict | None = None
        probs = labels = None

        artifact = _calibration_artifact_path()
        if artifact.exists():
            try:
                loaded = load_calibration(artifact)
                a_names = loaded["class_names"] or class_names
                if list(a_names) != class_names:
                    raise ValueError(
                        f"calibration artifact class_names {a_names} != server "
                        f"class_names {class_names}")
                probs, labels = loaded["probs"], loaded["labels"]
                basis, meta = loaded["basis"], loaded["meta"]
                _logger.info("Loaded REAL calibration artifact (%s, n=%d, basis=%s)",
                             artifact, len(labels), basis)
            except Exception:
                # A broken artifact must not take down certified serving — fall back
                # to the placeholder, loudly.
                _logger.exception("Calibration artifact unusable; using placeholder")
                probs = labels = None

        if probs is None:
            probs, labels = synthetic_demo_probs(
                lambda sig: _np.array(
                    [engine.predict(sig).probabilities[n] for n in class_names],
                    dtype=_np.float64),
                len(class_names))
            basis = BASIS_SYNTHETIC

        # Temperature scaling (Guo et al. 2017): the v1 model is over-confident on
        # real data (ECE ~0.22 -> ~0.06 after scaling). Fit T on the calibration set
        # and re-temper BOTH calibration and serving probabilities consistently, so
        # the conformal score transform is identical on both sides and the coverage
        # guarantee is preserved while the sets become honestly calibrated.
        probs_arr = _np.asarray(probs, dtype=float)
        labels_arr = _np.asarray(labels, dtype=int)
        temperature = fit_temperature(probs_arr, labels_arr)
        verifier = Verifier(alpha=0.1, class_names=class_names)
        tempered = apply_temperature(probs_arr, temperature)
        verifier.calibrate(tempered, labels_arr)
        app.state.verifier = verifier
        app.state.coverage_basis = basis
        app.state.calibration_meta = meta
        app.state.coverage_temperature = temperature

        # Continuous monitor seeded with the calibration confidence distribution, so
        # PSI drift is measured against what the verifier was calibrated on.
        try:
            from aion_nexus.monitoring import Monitor
            app.state.monitor = Monitor(reference_confidence=tempered.max(axis=1))
        except Exception:
            app.state.monitor = None

        # Conformal Risk Control: bound the expected false-healthy rate on the SAME
        # (temperature-scaled) calibration the conformal layer uses. Opt-out with
        # AION_RISK_ALPHA="off". The guarantee inherits the calibration basis caveat.
        app.state.risk_control = None
        raw_alpha = os.environ.get(RISK_ALPHA_ENV, "")
        if raw_alpha.strip().lower() not in ("off", "0", "none", "false"):
            try:
                from aion_nexus.verify import conformal_risk_control
                ra = float(raw_alpha) if raw_alpha.strip() else DEFAULT_RISK_ALPHA
                app.state.risk_control = conformal_risk_control(
                    tempered, labels_arr, alpha=ra)
            except Exception:
                _logger.exception("Risk control calibration failed; disabling it")
                app.state.risk_control = None
        # The store path resolves from VERIFY_CERT_STORE (env) by default; the
        # chain link scheme (Ed25519 / HMAC / NONE) is resolved at append time by
        # the same env precedence the certificate uses.
        app.state.cert_store = CertificateStore()
        _logger.info("Certified-serving verifier calibrated (basis=%s).", basis)
    except Exception:
        # Never let an optional cert facility block the core prediction service.
        _logger.exception("Failed to build certified-serving verifier")
        app.state.verifier = None
        app.state.cert_store = None
        app.state.coverage_basis = None


def _load_rul_model(app: FastAPI) -> None:
    """Load a fitted ConformalRUL artifact (joblib) from AION_RUL_ARTIFACT, if set.

    Loaded ONLY from the operator-configured trusted path (joblib uses pickle). A
    missing/broken artifact disables /predict_rul (503), never crashes startup.
    """
    app.state.rul_model = None
    path = os.environ.get(RUL_ARTIFACT_ENV)
    if not path or not path.strip():
        return
    try:
        from aion_nexus.rul import load_rul
        app.state.rul_model = load_rul(path.strip())
        _logger.info("Loaded calibrated RUL model from %s", path)
    except Exception:
        _logger.exception("Failed to load RUL artifact; /predict_rul disabled")
        app.state.rul_model = None


def _load_keyring(app: FastAPI) -> None:
    """Load a publishable KeyRing (rotation + revocation) from AION_KEYRING, if set."""
    app.state.keyring = None
    path = os.environ.get(KEYRING_ARTIFACT_ENV)
    if not path or not path.strip():
        return
    try:
        from aion_nexus.verify import KeyRing
        app.state.keyring = KeyRing.load(path.strip())
        _logger.info("Loaded key registry from %s", path)
        # With a registry deployed, /verify is fail-CLOSED: a cert WITHOUT a key_id
        # resolves to UNKNOWN-KEY (untrusted). So the minting side MUST stamp one, or
        # every served certificate would be untrusted by its own auditor.
        if not (os.environ.get(CERT_KEY_ID_ENV) or "").strip():
            _logger.warning(
                "%s is set but %s is NOT: served certificates will carry no key_id "
                "and /verify (fail-closed under a keyring) will mark them UNKNOWN-KEY "
                "/ untrusted. Set %s to the active key's id.",
                KEYRING_ARTIFACT_ENV, CERT_KEY_ID_ENV, CERT_KEY_ID_ENV)
    except Exception:
        _logger.exception("Failed to load key registry; revocation enforcement off")
        app.state.keyring = None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup/shutdown handler (replaces the deprecated @app.on_event)."""
    _load_engine(app)
    _build_certifier(app)
    _load_rul_model(app)
    _load_keyring(app)
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
    expected_sha = getattr(app.state, "expected_checkpoint_sha256", None)
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
            checkpoint_sha256=None,
            expected_checkpoint_sha256=expected_sha,
        )
    info = engine.get_health()
    return HealthResponse(
        status="healthy", expected_checkpoint_sha256=expected_sha, **info)


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


class CertifiedSignalRequest(JsonSignalRequest):
    """``/predict_certified`` body: a signal plus an OPTIONAL physics second opinion.

    When ``rpm`` and ``bearing`` are supplied, the server runs the model-agnostic
    physics verifier (envelope/order analysis) on the SAME window and composes its
    verdict (weakest-link) with the conformal certificate. ``claimed_fault`` (one
    of ``outer`` / ``inner`` / ``ball`` / ``cage``) is the fault family domain
    knowledge expects for this machine; with it, physics can CONTRADICT a confident
    model whose energy sits at a different family. Omit all three to get the plain
    certified prediction (fully backward-compatible with the old body).
    """

    rpm: float | None = Field(
        None, gt=0.0, description="Shaft speed (rpm) for the physics second opinion")
    bearing: BearingGeometrySchema | None = Field(
        None, description="Bearing geometry for the physics second opinion")
    claimed_fault: Literal["outer", "inner", "ball", "cage"] | None = Field(
        None, description="Expected fault family (enables CONTRADICT detection)")


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


# ---- Certified serving (v2.6.0): wire the signed certificate into the product

def _require_verifier():
    """Return the calibrated Verifier, or 503 if certified serving is unavailable."""
    verifier = getattr(app.state, "verifier", None)
    if verifier is None:
        raise HTTPException(
            status_code=503,
            detail="Certified serving unavailable: the conformal verifier failed "
                   "to calibrate at startup (see server logs).",
        )
    return verifier


def _coverage_basis() -> str:
    """How the served verifier was calibrated: 'real-holdout' or 'synthetic-placeholder'."""
    from aion_nexus.serving_calibration import BASIS_SYNTHETIC
    return getattr(app.state, "coverage_basis", None) or BASIS_SYNTHETIC


def _coverage_temperature() -> float:
    """The temperature-scaling factor the served verifier was calibrated with (1.0 = none)."""
    t = getattr(app.state, "coverage_temperature", None)
    return float(t) if t else 1.0


def _compose_physics(signal, rpm, bearing, claimed_fault, cert):
    """Run the model-agnostic physics second opinion and compose it with the cert.

    Returns ``(physics_dict_or_None, composed_dict_or_None)``.

    The physics check runs only when ``rpm`` and ``bearing`` are supplied. It is
    composed into the system verdict (weakest-link AND) ONLY when it returns a
    DEFINITE opinion (CONFIRM / CONTRADICT): a WEAK / INDETERMINATE physics result
    carries no information and must NOT drag a CERTIFIED conformal verdict down to
    ABSTAIN — it is reported for transparency but left out of the composition. A
    CONTRADICT drops the system below CERTIFIED (a confident-but-wrong model
    caught); a CONFIRM corroborates without raising (AND can only lower).
    """
    if rpm is None or bearing is None:
        return None, None
    try:
        from aion_nexus.config import SAMPLING_RATE_HZ
        from aion_nexus.physics import (
            PHYS_CONFIRM,
            PHYS_CONTRADICT,
            BearingGeometry,
            physics_consistency,
        )
        from aion_nexus.verify import compose_certificates

        geometry = BearingGeometry(
            n_rolling_elements=bearing.n_rolling_elements,
            ball_diameter=bearing.ball_diameter,
            pitch_diameter=bearing.pitch_diameter,
            contact_angle_deg=bearing.contact_angle_deg,
        )
        verdict = physics_consistency(
            np.asarray(signal, dtype=np.float64),
            fs=SAMPLING_RATE_HZ,
            rpm=float(rpm),
            geometry=geometry,
            claimed_fault=claimed_fault,
        )
        physics_dict = {
            "verdict": verdict.verdict,
            "dominant_fault": verdict.dominant_fault,
            "claimed_fault": verdict.claimed_fault,
            "scores": {k: float(v) for k, v in verdict.scores.items()},
            "assurance": verdict.assurance,
            "detail": verdict.detail,
        }
        composed = None
        if verdict.verdict in (PHYS_CONFIRM, PHYS_CONTRADICT):
            composed = compose_certificates([cert, verdict.as_component()], op="and")
        return physics_dict, composed
    except (ValueError, KeyError) as exc:
        # A malformed geometry / unsupported claim must not 500 a prediction the
        # caller already paid for — report the failure, skip composition.
        _logger.warning("Physics second opinion skipped: %s", exc)
        return {"verdict": "INDETERMINATE", "detail": f"physics unavailable: {exc}"}, None


def _cert_ttl_seconds() -> int:
    """Resolve the certificate validity window (seconds) from AION_CERT_TTL_SECONDS."""
    raw = os.environ.get(CERT_TTL_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_CERT_TTL_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_CERT_TTL_SECONDS
    return value if value > 0 else DEFAULT_CERT_TTL_SECONDS


def _signing_key_configured() -> bool:
    """True iff an Ed25519 seed OR an HMAC key is set (so we can SIGN a cert)."""
    return bool(os.environ.get(ED25519_SEED_ENV) or os.environ.get(HMAC_KEY_ENV))


def _weak_ed25519_seed_error() -> str | None:
    """Return an error string if a configured Ed25519 seed is below the entropy
    floor, else None.

    The library's ``seal()`` derives keys with the legacy SHA-256 fold for
    backward compatibility and does NOT reject a weak seed — so the PRODUCT
    boundary must. A guessable minting seed (the red-team's '1234' kill-shot)
    would yield a brute-forceable private key and let an attacker mint
    "trusted" certificates. Here we refuse to sign with it rather than emit a
    forgeable-looking certificate.
    """
    seed = os.environ.get(ED25519_SEED_ENV)
    if not seed:
        return None
    from aion_nexus.verify import assert_strong_seed
    try:
        assert_strong_seed(seed)
    except ValueError as exc:
        return str(exc)
    return None


@app.post(
    "/predict_certified",
    response_model=PredictCertifiedResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict_certified(body: CertifiedSignalRequest) -> PredictCertifiedResponse:
    """Predict AND emit a SIGNED, auditable :class:`Certificate` for the decision.

    Reuses the exact ``/predict`` pipeline (preprocess + OOD/plausibility gate +
    classifier), then runs the calibrated conformal verifier to produce a sealed
    certificate, APPENDS it to the hash-chained certificate store (audit trail),
    and returns ``{prediction, certificate, pubkey, verdict}``.

    Signing & honesty (workspace 6.31):

    - With ``VERIFY_ED25519_SEED`` set, the certificate is Ed25519-SIGNED and ships
      its public key so a third party (auditor / customer / insurer) can verify it
      OFFLINE with the public key alone — the unique weapon, now wired into the
      product. ``VERIFY_HMAC_KEY`` is the symmetric fallback.
    - With NO key configured: in STRICT mode (``AION_REQUIRE_SIGNED_CERT=1``) this
      refuses with 503 ("signing key not configured"); otherwise it emits a cert
      with ``authentication = NONE`` and sets an explicit ``warning`` that the cert
      is integrity-only and NOT tamper-evident. We never silently pretend.

    The certificate carries a validity window (``AION_CERT_TTL_SECONDS``, default
    24h) bound into the signature, so a replayed/expired cert fails verification.
    Auth, body-size limit and the OOD gate are inherited from the shared pipeline.
    """
    strict = os.environ.get(REQUIRE_SIGNED_CERT_ENV, "") == "1"
    if strict and not _signing_key_configured():
        raise HTTPException(
            status_code=503,
            detail=("signing key not configured: AION_REQUIRE_SIGNED_CERT=1 but "
                    f"neither {ED25519_SEED_ENV} nor {HMAC_KEY_ENV} is set. Refusing "
                    "to emit an unsigned (NOT tamper-evident) certificate."),
        )

    # Secure-at-the-product-boundary: never MINT with a guessable Ed25519 seed.
    # The library stays back-compatible (legacy fold), so we enforce the entropy
    # floor here and refuse rather than hand out a brute-forceable signature.
    weak_seed_err = _weak_ed25519_seed_error()
    if weak_seed_err is not None:
        raise HTTPException(
            status_code=503,
            detail=(f"{ED25519_SEED_ENV} is too weak to mint with: {weak_seed_err} "
                    "Generate a full-entropy seed: "
                    "python -c \"from aion_nexus.verify import generate_seed; print(generate_seed())\""),
        )

    # Optional strict gate: refuse to emit a placeholder-coverage certificate. A
    # deployer who has wired a real calibration artifact can set this to guarantee
    # no cert ever ships a synthetic coverage number (the honesty gate, hard mode).
    from aion_nexus.serving_calibration import BASIS_SYNTHETIC
    if (os.environ.get(REQUIRE_REAL_CALIBRATION_ENV, "") == "1"
            and _coverage_basis() == BASIS_SYNTHETIC):
        raise HTTPException(
            status_code=503,
            detail=(f"{REQUIRE_REAL_CALIBRATION_ENV}=1 but the served verifier is "
                    "calibrated on the synthetic placeholder. Provide a real "
                    f"calibration artifact via {CALIBRATION_NPZ_ENV} (build it with "
                    "scripts/build_calibration) before emitting certificates."),
        )

    engine = _require_engine()
    verifier = _require_verifier()
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

    # Build the probability vector in the verifier's class order and certify. The
    # signing scheme resolves by env precedence inside seal (Ed25519 > HMAC > NONE);
    # the TTL binds a validity window into the signature (anti-replay).
    from aion_nexus.config import CLASS_NAMES
    probs = np.array([result.probabilities[name] for name in CLASS_NAMES],
                     dtype=np.float64)
    key_id = os.environ.get(CERT_KEY_ID_ENV) or None
    # Bind the calibration basis (real-holdout vs synthetic-placeholder) and the
    # temperature factor into the certificate's coverage_guarantee, which IS hashed
    # into content_hash — so the synthetic-vs-real distinction is TAMPER-EVIDENT,
    # not just a docstring caveat. Temperature-scale the serving probabilities with
    # the SAME factor the conformal calibrator was fit under (consistent score
    # transform -> coverage preserved, sets honestly calibrated).
    from aion_nexus.serving_calibration import (
        apply_temperature,
        coverage_guarantee_string,
    )
    basis = _coverage_basis()
    temperature = _coverage_temperature()
    cov_meta = getattr(app.state, "calibration_meta", None) or {}
    cov_guarantee = coverage_guarantee_string(
        basis, float(verifier.calibrator.alpha),
        leakage_checked=bool(cov_meta.get("leakage_checked")),
        temperature=temperature)
    cert = verifier.certify(
        apply_temperature(probs, temperature),
        input_signal=signal,
        model_id=f"aion-nexus-{__version__}",
        ttl_seconds=_cert_ttl_seconds(),
        key_id=key_id,
        conformal_method="marginal-split-conformal",
        coverage_guarantee=cov_guarantee,
    )

    # Append to the hash-chained audit store (best-effort: a store failure must not
    # drop the prediction the caller already paid for).
    store = getattr(app.state, "cert_store", None)
    if store is not None:
        try:
            store.append(cert)
        except Exception:
            _logger.exception("Failed to append certificate to the store")

    # Optional model-agnostic physics second opinion, composed weakest-link with
    # the conformal cert (a CONTRADICT drops the system below CERTIFIED).
    physics_dict, composed = _compose_physics(
        signal, body.rpm, body.bearing, body.claimed_fault, cert)

    # Conformal Risk Control: the risk-controlled prediction set whose expected
    # false-healthy rate is bounded (the safety-critical guarantee for PdM).
    rc_dict = None
    rc = getattr(app.state, "risk_control", None)
    if rc is not None:
        from aion_nexus.verify import DEGRADED_CLASSES
        rc_set = rc.prediction_set(apply_temperature(probs, temperature))
        rc_dict = {
            "method": rc.method,
            "alpha": rc.alpha,
            "lambda_hat": rc.lambda_hat,
            "guarantee": rc.guarantee,
            "set": rc_set,
            "set_names": [CLASS_NAMES[i] for i in rc_set],
            "flags_degraded": any(i in DEGRADED_CLASSES for i in rc_set),
            "calibrated_risk": rc.calibrated_risk,
            "coverage_basis": basis,
        }

    warnings: list[str] = []
    from aion_nexus.verify import AUTH_NONE
    if cert.authentication == AUTH_NONE:
        warnings.append(
            "certificate is UNSIGNED (authentication=NONE): integrity hash only, "
            "NOT tamper-evident against an adversary holding the source. Set "
            f"{ED25519_SEED_ENV} (Ed25519, third-party verifiable) or {HMAC_KEY_ENV} "
            "(HMAC) to sign it.")
    if basis == BASIS_SYNTHETIC:
        warnings.append(
            "coverage_basis=synthetic-placeholder: the conformal coverage number is "
            "a PLACEHOLDER (verifier calibrated on synthetic windows, not real "
            "bearings). Build a real, leakage-checked calibration artifact "
            f"(scripts/build_calibration -> {CALIBRATION_NPZ_ENV}) to make coverage "
            "meaningful. Verdict logic, signature and audit trail are real regardless.")
    warning = " | ".join(warnings) if warnings else None

    # Feed the continuous monitor (rolling SLO + drift over the certificate stream).
    mon = getattr(app.state, "monitor", None)
    if mon is not None:
        try:
            mon.record(float(result.confidence), cert.verdict)
        except Exception:
            _logger.exception("Monitor record failed (non-fatal)")

    METRICS.observe_prediction(result.predicted_class_name, result.latency_ms)
    return PredictCertifiedResponse(
        prediction=PredictResponse(**result.to_dict()),
        certificate=cert.as_dict(),
        pubkey=cert.pubkey,
        verdict=cert.verdict,
        warning=warning,
        coverage_basis=basis,
        physics=physics_dict,
        composed=composed,
        risk_control=rc_dict,
    )


@app.get("/monitor", dependencies=[Depends(_require_api_key)])
def monitor() -> dict:
    """Rolling SLO + drift over the recent certificate stream (continuous monitoring).

    Returns certified/review/abstain rates, mean confidence (a label-free accuracy
    proxy under calibration), the Population Stability Index of the confidence
    distribution vs the calibration reference, a drift level, and any alerts. The
    point-in-time certificate becomes a continuously-watchable SLO.
    """
    mon = getattr(app.state, "monitor", None)
    if mon is None:
        return {"n": 0, "alerts": ["monitor unavailable (certified serving not calibrated)"]}
    return mon.status()


@app.post(
    "/verify",
    response_model=VerifyResponse,
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def verify(body: VerifyRequest) -> VerifyResponse:
    """Audit a certificate's integrity, authenticity AND validity window.

    Delegates to :func:`aion_nexus.verify.verify_certificate`. Supply
    ``expected_pubkey`` (the issuer's out-of-band Ed25519 public key) to get
    genuine issuer authentication (``trusted = True`` only then); without it an
    Ed25519 cert can at most be ``SELF-SIGNED`` (``trusted = False``). The HMAC key
    (if the cert is HMAC-signed) is read from ``VERIFY_HMAC_KEY`` server-side.
    ``now_iso`` optionally pins "now" for the validity-window check.

    This endpoint is pure verification: it needs no engine and no secret beyond
    what the caller / env supplies, so an auditor can call it directly.
    """
    from aion_nexus.verify import verify_certificate
    keyring = getattr(app.state, "keyring", None)
    try:
        if keyring is not None:
            # A key registry is deployed -> ENFORCE rotation + revocation for EVERY
            # certificate, fail-CLOSED. Routing through verify_with_keyring regardless
            # of whether the cert carries a key_id is the fix for the prior fail-OPEN
            # gap: a cert WITHOUT a key_id can no longer slip past the registry and be
            # trusted via its embedded pubkey — verify_with_keyring marks a missing /
            # unknown key_id as UNKNOWN-KEY (trusted=False), and a revoked key as
            # REVOKED-KEY. When a keyring is loaded the deployer MUST mint with
            # AION_CERT_KEY_ID set (_load_keyring warns if it is not).
            from aion_nexus.verify import verify_with_keyring
            res = verify_with_keyring(body.certificate, keyring, now_iso=body.now_iso)
        else:
            res = verify_certificate(
                body.certificate,
                expected_pubkey=body.expected_pubkey,
                now_iso=body.now_iso,
            )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(
            400, f"Malformed certificate (cannot verify): {exc}") from exc
    return VerifyResponse(
        integrity_ok=res["integrity_ok"],
        authenticity=res["authenticity"],
        trusted=res["trusted"],
        detail=res["detail"],
        expired=res.get("expired"),
        not_yet_valid=res.get("not_yet_valid"),
        key_status=res.get("key_status"),
        key_note=res.get("key_note") or None,
    )


@app.post(
    "/evidence",
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def evidence(body: EvidenceRequest) -> dict:
    """Map a certificate onto an EU AI Act / ISO evidence map (served, v2.16.0).

    Turns a signed :class:`Certificate` into the structured evidence dict
    (:func:`aion_nexus.compliance.compliance_evidence`): EU AI Act Art. 12/14/15 +
    ISO 13381-1 + ISO/IEC 42001, each item carrying an explicit limitation. Pure
    standards-mapping — needs no engine and no secret. HONESTY (6.31): this is
    evidence scaffolding, NOT a conformity assessment; the module is contractually
    barred from emitting 'compliant'/'conforme'.
    """
    from aion_nexus.compliance import compliance_evidence
    try:
        return compliance_evidence(body.certificate)
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"Cannot map certificate to evidence: {exc}") from exc


@app.post(
    "/annex_iv",
    responses={400: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def annex_iv(body: AnnexIVRequest) -> dict:
    """Generate the EU AI Act Annex IV (Article 11) documentation skeleton (served).

    Returns the 9-section dossier (:func:`aion_nexus.compliance.annex_iv_dossier`),
    filling sections only from caller-supplied ``model_metadata`` and marking the
    rest provider-owned. With ``markdown=true`` a rendered card is included.
    HONESTY (6.31): readiness is NOT conformity; the dossier never claims the
    system is compliant.
    """
    from aion_nexus.compliance import annex_iv_card, annex_iv_dossier
    try:
        dossier = annex_iv_dossier(body.model_metadata, certificate=body.certificate)
        out: dict = {"dossier": dossier}
        if body.markdown:
            out["card_markdown"] = annex_iv_card(
                body.model_metadata, certificate=body.certificate)
        return out
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(400, f"Cannot build Annex IV dossier: {exc}") from exc


@app.post(
    "/predict_rul",
    response_model=PredictRULResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict_rul(body: JsonSignalRequest) -> PredictRULResponse:
    """Calibrated Remaining Useful Life with a conformal interval (v2.19.0).

    Requires a fitted ConformalRUL artifact configured via ``AION_RUL_ARTIFACT``
    (the deployer builds it from their run-to-failure data). Returns a median
    time-to-failure plus a ``1 - alpha`` conformal interval whose coverage holds
    under exchangeability of the deployer's calibration asset and the serving
    asset — the ``coverage_caveat`` states the cross-bearing/cross-machine limit.
    HONESTY: this is a TIME-TO-FAILURE (unlike /predict_degradation's coarse stage).
    """
    model = getattr(app.state, "rul_model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=("calibrated RUL unavailable: set AION_RUL_ARTIFACT to a fitted "
                    "ConformalRUL artifact built from run-to-failure data "
                    "(see examples/14_calibrated_rul.py)."))
    try:
        signal = np.asarray(body.signal, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Malformed signal: {exc}") from exc
    try:
        from aion_nexus.rul import health_features
        est = model.predict_one(health_features(signal))
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"Cannot estimate RUL: {exc}") from exc
    return PredictRULResponse(**est.as_dict())


@app.post(
    "/predict_degradation",
    response_model=PredictDegradationResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    dependencies=[Depends(_require_api_key)],
)
def predict_degradation(body: JsonSignalRequest) -> PredictDegradationResponse:
    """Predict on a single window AND return a first-class degradation-STAGE estimate.

    Reuses the exact ``/predict`` pipeline (preprocess + classifier + plausibility
    gate) and adds the additive ``degradation`` object — the honest reframe made a
    product output. Auth, body-size limit and the OOD/abstain gate are inherited
    from the shared pipeline (an OOD-flagged window abstains in ``degradation``).

    HONESTY: the degradation stage is a COARSE positional proxy of the FEMTO life
    fraction (4 bands), NOT a calibrated time-to-failure / RUL in hours or cycles
    — see the ``degradation.disclaimer`` field. No conformal stage set is returned
    here (the served engine carries no fitted calibrator): ``degradation.calibrated``
    is False and only the point estimate is provided. For coverage-controlled
    stage sets, call ``InferenceEngine.predict_degradation(signal, calibrator=...)``
    with a calibrator fitted on EXCHANGEABLE (e.g. per-bearing) calibration data.
    """
    engine = _require_engine()
    try:
        signal = np.asarray(body.signal, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            400, f"Malformed signal (ragged or non-numeric rows): {exc}"
        ) from exc

    try:
        result = engine.predict_degradation(signal)
    except SignalValidationError as exc:
        raise HTTPException(400, str(exc)) from exc

    METRICS.observe_prediction(result.predicted_class_name, result.latency_ms)
    return PredictDegradationResponse(**result.to_dict())


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
    """Predict on multiple uploaded CSV files in one request.

    Three independent DoS caps bound the request: the file-COUNT cap
    (``AION_MAX_BATCH_FILES``), the per-file BYTE cap (``AION_MAX_BODY_BYTES``,
    enforced in ``_read_upload_capped``) and the aggregate BYTE budget across all
    files (``AION_MAX_BATCH_BYTES``) — the last closes the gap where many files,
    each just under the per-file cap, sum to gigabytes.
    """
    if not files:
        raise HTTPException(400, "No files uploaded")
    max_files = int(os.environ.get(MAX_BATCH_FILES_ENV, str(DEFAULT_MAX_BATCH_FILES)))
    if len(files) > max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Too many files: {len(files)} > limit {max_files} "
                   f"(configure via {MAX_BATCH_FILES_ENV}).",
        )
    max_batch_bytes = int(
        os.environ.get(MAX_BATCH_BYTES_ENV, str(DEFAULT_MAX_BATCH_BYTES)))
    engine = _require_engine()

    signals = []
    total_bytes = 0
    for f in files:
        contents = _read_upload_capped(f)
        # Aggregate byte budget: enforced as files stream in, BEFORE parsing, so
        # an over-budget batch is rejected without doing the parse work.
        total_bytes += len(contents)
        if total_bytes > max_batch_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Batch payload too large: aggregate {total_bytes} bytes "
                       f"exceeds the limit of {max_batch_bytes} bytes "
                       f"(configure via {MAX_BATCH_BYTES_ENV}).",
            )
        try:
            arr = np.loadtxt(io.BytesIO(contents), delimiter=",")
            # EXPLICIT 2D guard, aligned with _csv_to_signal. A 1-column CSV parses
            # to a 1-D array, so the old `arr.shape[1]` raised IndexError that was
            # only caught by accident below; check ndim up front for a clear error.
            if arr.ndim != 2:
                raise ValueError(
                    f"CSV must be 2D, got {arr.ndim}D shape {arr.shape}")
            if arr.shape[1] >= 6:
                signals.append(arr[:, [4, 5]].T)  # FEMTO format
            else:
                signals.append(arr.T if arr.shape[0] != 2 else arr)
        except HTTPException:
            raise
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
