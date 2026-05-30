# REST API Reference

The AION-NEXUS server exposes a small REST API. All responses are JSON.

Base URL (default): `http://localhost:8080`

## Authentication

The default deployment has no authentication. For production, place behind a reverse proxy (nginx, traefik) that handles TLS + auth.

## Endpoints

### `GET /health`

Liveness probe with operational telemetry.

**Response**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cpu",
  "model_param_count": 1061724,
  "inference_count": 1234,
  "running_avg_latency_ms": 12.34
}
```

`status` may be `healthy`, `degraded`, or `down`. A `down` status means the checkpoint failed to load — set the `AION_CHECKPOINT` env var or fix the path.

### `GET /version`

Returns the model + API version.

```json
{ "model": "1.0.0", "api": "1.0.0" }
```

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
  "model_version": "1.0.0"
}
```

#### Errors

- `400 Bad Request` — signal validation failed (wrong shape, NaN, stuck sensor, too short)
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
  "model_version": "1.0.0"
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

The default deployment has no rate limiting. For multi-tenant production:

- Use a reverse proxy (nginx / traefik) for IP-based rate limiting.
- Use FastAPI middleware (e.g., `slowapi`) for per-user/per-API-key limits.

## OpenAPI / interactive docs

FastAPI auto-generates docs at:

- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`
- OpenAPI spec: `http://localhost:8080/openapi.json`

## Client libraries

Minimal Python client provided in `examples/04_api_client.py`. For high-volume production, use an HTTP client with connection pooling (`httpx`, `aiohttp`).
