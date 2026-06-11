# REST API Reference

The AION-NEXUS server exposes a small REST API. All responses are JSON (except `/metrics`, which uses the Prometheus exposition format). Applies to package version 2.2.0.

Base URL (default): `http://localhost:8080`

## Authentication

Since 2.2.0 the server supports optional in-app API-key authentication:

- Set the `AION_API_KEY` environment variable to enable it. When set, every request must
  carry the key in the `X-API-Key` header; requests without it (or with a wrong key) get
  `401 Unauthorized`.
- **`GET /health` is exempt** so orchestrator liveness probes keep working without secrets.
- When `AION_API_KEY` is unset (default), no auth is enforced — for production either set it
  or place the server behind a reverse proxy (nginx, traefik) that handles TLS + auth.
  TLS termination is always the operator's responsibility (the bundled server does not do TLS).

## Server configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `AION_CHECKPOINT` | `checkpoints/aion_nexus_v1.pth` | Checkpoint to serve |
| `AION_DEVICE` | `cpu` | Inference device |
| `AION_API_KEY` | unset (auth off) | Enables `X-API-Key` auth (see above) |
| `AION_MAX_BODY_BYTES` | `10485760` (10 MB) | Request body cap; larger requests are rejected with `413` before parsing |
| `AION_CORS_ORIGINS` | unset (no CORS headers) | Comma-separated allowlist of origins |
| `AION_LOG_JSON` | unset (plain text logs) | `1` switches to structured JSON logs with a per-request `request_id` |

## Endpoints

### `GET /health`

Liveness probe with operational telemetry. Exempt from API-key auth.

**Response**

```json
{
  "status": "healthy",
  "version": "2.2.0",
  "architecture_version": "v1",
  "device": "cpu",
  "model_param_count": 1061724,
  "inference_count": 1234,
  "running_avg_latency_ms": 12.34
}
```

`status` may be `healthy`, `degraded`, or `down`. A `down` status means the checkpoint failed to load — set the `AION_CHECKPOINT` env var or fix the path. `architecture_version` reports which architecture was auto-detected from the checkpoint (`v1`, `v3`, or `v6`).

### `GET /version`

Returns the model + API version.

```json
{ "model": "2.2.0", "api": "2.2.0" }
```

### `GET /metrics`

Prometheus exposition format (text), powered by `prometheus-client`. Exposes request counts
and latency histograms for scraping. Point your Prometheus scrape config at this endpoint.

### `POST /predict`

Single-window prediction. **JSON body only.** For CSV upload, use `/predict_csv`.

- **JSON body** with `signal` field (required)

#### Request (JSON)

```json
{
  "signal": [
    [0.012, 0.011, ...],   // 2560 floats — channel 0 (horizontal)
    [-0.003, 0.005, ...]   // 2560 floats — channel 1 (vertical)
  ]
}
```

#### Request (CSV upload — separate endpoint)

```bash
curl -X POST http://localhost:8080/predict_csv -F "file=@acc_00321.csv"
```

`/predict_csv` accepts FEMTO PRONOSTIA format (6 columns; channels 4–5 are horizontal/vertical accelerometers) or a plain 2-column CSV. Same response schema as `/predict`.

#### Response (200)

```json
{
  "predicted_class_index": 2,
  "predicted_class_name": "medium",
  "description": "Progressive degradation — plan replacement before next major maintenance.",
  "probabilities": {
    "normal":   0.04,
    "early":    0.13,
    "medium":   0.74,
    "advanced": 0.09
  },
  "confidence": 0.74,
  "confidence_band": "medium",
  "recommended_action": {
    "alert_level": 2,
    "stop_machine": false,
    "plan_replacement": true
  },
  "latency_ms": 12.4,
  "model_version": "2.2.0"
}
```

#### Errors

- `400 Bad Request` — signal validation failed (wrong shape, NaN, stuck sensor, too short)
- `401 Unauthorized` — `AION_API_KEY` is set and the `X-API-Key` header is missing or wrong
- `413 Payload Too Large` — request body exceeds `AION_MAX_BODY_BYTES` (default 10 MB)
- `503 Service Unavailable` — checkpoint not loaded; set `AION_CHECKPOINT` env var

### `POST /predict_batch`

Batch prediction over multiple uploaded CSVs.

```bash
curl -X POST http://localhost:8080/predict_batch \
     -F "files=@acc1.csv" -F "files=@acc2.csv" -F "files=@acc3.csv"
```

Response is a list of per-file `PredictResponse` objects plus aggregate latency.

### `POST /predict_long_signal`

Window-then-aggregate prediction over a multi-second recording. The server segments the signal into non-overlapping 0.1-second windows, predicts each, and aggregates.

```json
{
  "signal": [[...], [...]],     // [2, N] with N >> 2560
  "aggregation": "mean",         // 'mean' | 'majority' | 'max_class'
  "window": 2560,
  "stride": null                 // null = non-overlapping
}
```

Response includes the aggregated decision and the number of windows used:

```json
{
  "predicted_class_index": 1,
  "predicted_class_name": "early",
  "description": "...",
  "aggregated_probabilities": {...},
  "recommended_action": {...},
  "n_windows": 50,
  "aggregation_method": "mean",
  "model_version": "2.2.0"
}
```

## Confidence-band semantics

| Band | Probability | Meaning |
|---|---|---|
| `high` | ≥ 0.85 | Automated action permitted |
| `medium` | 0.65 ≤ p < 0.85 | Standard reporting; review on safety-critical machines |
| `low` | < 0.65 | Send to human expert; ambiguous prediction |

These thresholds are configurable in `aion_nexus/config.py` — tune per deployment.

## Rate limits

The default deployment has no rate limiting (body size is capped in-app via
`AION_MAX_BODY_BYTES`). For multi-tenant production:

- Use a reverse proxy (nginx / traefik) for IP-based rate limiting.
- Use FastAPI middleware (e.g., `slowapi`) for per-user/per-API-key limits.

## OpenAPI / interactive docs

FastAPI auto-generates docs at:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI spec: `http://localhost:8080/openapi.json`

## Client libraries

Minimal Python client provided in `examples/04_api_client.py`. For high-volume production, use an HTTP client with connection pooling (`httpx`, `aiohttp`).
