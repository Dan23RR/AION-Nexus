# Data Contract — AION-NEXUS v1.0

Formal specification of the input/output contract. Any change to this contract requires a major version bump (semver). Any deviation from this spec is a bug.

---

## Input contract

### Vibration signal

| Property | Required | Value |
|---|---|---|
| Type | yes | `numpy.ndarray` or `list[list[float]]` |
| Dtype | yes | float (cast to `float32` internally) |
| Dimensions | yes | exactly 2 |
| Shape | yes | `[2, N]` or `[N, 2]` (auto-transposed) |
| Channels | yes | exactly 2 (e.g., horizontal + vertical accelerometers) |
| Length N | yes | ≥ 2,560 samples |
| Sampling rate | recommended | 25,600 Hz (model trained at this rate) |
| Units | recommended | g (acceleration in units of 9.81 m/s²) or m/s² |
| Pre-processing | none | model expects raw signal; z-score normalization applied internally |
| Centering | none | model expects unfiltered signal (mean removed internally) |
| NaN / Inf | rejected | `SignalValidationError` raised before any computation |
| Stuck channel (std < 1e-9) | rejected | `SignalValidationError` raised |

### Optional metadata (currently unused)

| Field | Type | Notes |
|---|---|---|
| `rpm` | float | Shaft rotation speed in RPM. Reserved for v2.0 order-tracking preprocessor. |
| `bearing_geometry` | dict | Bearing dimensions for fault frequency calculation. Reserved for v2.0 attribution. |

These fields are accepted by `InferenceEngine.predict(signal, rpm=..., geometry=...)` for forward compatibility but are not used by the v1.0 model.

---

## Output contract

### `PredictionResult` schema

```python
@dataclass
class PredictionResult:
    predicted_class_index: int           # 0..3
    predicted_class_name: str            # one of {"normal", "early", "medium", "advanced"}
    description: str                     # human-readable description
    probabilities: dict[str, float]      # 4 entries; values sum to 1.0 within ±1e-5
    confidence: float                    # [0.0, 1.0]; max(probabilities)
    confidence_band: str                 # one of {"high", "medium", "low"}
    recommended_action: dict             # operational action recommendation
    latency_ms: float                    # > 0
    model_version: str                   # SemVer (e.g., "1.0.0")
```

### Class taxonomy (frozen in v1.0)

| Index | Name | Description | Action |
|---|---|---|---|
| 0 | normal | Healthy bearing | None |
| 1 | early | Initial defect | Schedule inspection in next maintenance cycle |
| 2 | medium | Progressive degradation | Plan replacement before next major maintenance |
| 3 | advanced | Imminent failure | Stop machine and replace bearing |

The mapping `(index, name, description, action)` is a frozen contract. Adding classes requires a major version bump (e.g., adding a `critical` class for industries with finer alerting needs would be v2.0).

### Confidence bands

| Band | Confidence range | Operational meaning |
|---|---|---|
| `high` | ≥ 0.85 | Automated action permitted |
| `medium` | 0.65 ≤ p < 0.85 | Standard reporting; review on safety-critical assets |
| `low` | < 0.65 | Send to human expert; ambiguous prediction |

Thresholds are configurable in `aion_nexus.config` (`HIGH_CONFIDENCE_THRESHOLD`, `LOW_CONFIDENCE_THRESHOLD`).

### Recommended action contract

```python
{
  "alert_level": int,           # 0..3, monotonic with severity
  "stop_machine": bool,         # True for class=advanced only in v1.0
  "schedule_inspection": bool,  # True for class=early
  "plan_replacement": bool,     # True for class=medium
  "replace_immediately": bool,  # True for class=advanced
}
```

The customer's CMMS (SAP PM, IBM Maximo, etc.) consumes this directly.

---

## Backward / forward compatibility

### Backward compatibility (v1.X)

- `PredictionResult` may add NEW fields (e.g., `rul_estimate_days` in v1.5). Consumers MUST tolerate unknown fields.
- `recommended_action` may add NEW keys with default-False semantics. Consumers MUST tolerate unknown keys.
- `confidence_band` thresholds may shift slightly; consumers should NOT hard-code thresholds.

### Forward compatibility

- API consumers SHOULD NOT remove fields from request/response based on absence — server may add fields.
- API consumers SHOULD NOT assume specific dict iteration order.
- API consumers SHOULD validate `model_version` matches their tested version.

### Major version (v2.0) breaking changes

The following changes will trigger a major version bump:

- Class taxonomy change (adding/removing classes, renaming).
- Input shape change (e.g., 4 channels instead of 2).
- Sampling rate change (e.g., 50 kHz instead of 25.6 kHz).
- Removal of any field from `PredictionResult`.
- Changes to confidence-band semantics that flip operational decisions.

---

## Preprocessing pipeline contract

Pipeline order (must match training in `aion_data._process_csv_file`):

1. **Validate** shape (auto-transpose [N,2] to [2,N]), reject NaN/Inf, reject stuck sensors (per-channel std < 1e-7).
2. **Crop or pad** to exactly `SIGNAL_LENGTH = 2560`:
   - If `n < 2560`: reject (`SignalValidationError`).
   - If `n > 2560`: center-crop to middle 2560 samples.
   - Note: training-time loader truncates from start (`signal[:2560]`); production uses center-crop. For canonical FEMTO data (most files exactly 2560), behavior is identical. For longer recordings, center-crop is more stable (avoids edge transients). If exact training-equivalence is required for pre-release verification, use the original loader via `--aion-data-repo` flag of `verify_checkpoint.py`.
3. **Z-score per channel**: `(x - mean) / (std + 1e-8)` per channel per sample.
4. **High-pass Butterworth filter**: 2nd order, 1 Hz cutoff at 25.6 kHz sampling rate, applied via `scipy.signal.sosfilt` per channel. Required for byte-equivalent inference vs training; without it, F1 drops from 0.884 to 0.279 (verified empirically 2026-04-27).
5. **Cast** to float32 and wrap as `torch.Tensor` of shape `[1, 2, 2560]` (or `[B, 2, 2560]` for batched).

**Frozen contract**: changing the order, adding/removing steps, or modifying parameters (cutoff, filter order) is a major version bump because the trained checkpoint becomes incompatible.

## Stability guarantees

| Element | Stability |
|---|---|
| `PredictionResult` field names + types | Strong: stable across v1.x; major version only |
| `PredictionResult` field semantics | Strong: stable across v1.x |
| Class taxonomy (4 classes, names) | Strong: stable across v1.x |
| Confidence-band thresholds | Weak: configurable, may default-change in v1.X |
| Architecture parameter count | Strong: stable across v1.x (1,061,724 params) |
| Architecture layer-wise spec | Strong: described in `docs/architecture.md` |
| F1 numbers on FEMTO/MFPT | Strong: same checkpoint = same numbers ±1e-3 |
| API endpoints (`/predict`, `/health`, etc.) | Strong: stable across v1.x; new endpoints additive |
| OPC UA / MQTT topics (when added) | Strong: documented in `docs/api_reference.md` |

---

## Validation tooling

To verify a deployment satisfies this contract:

```bash
# Schema-level validation
pytest tests/test_smoke.py::test_inference_engine_synthetic_no_checkpoint

# F1 verification (requires checkpoint + data)
python -m scripts.verify_checkpoint

# API contract test (requires running server)
pytest tests/test_api_integration.py
```

Any deviation from this contract is a release-blocking bug. File via `SECURITY.md` if security-relevant or normal channels otherwise.
