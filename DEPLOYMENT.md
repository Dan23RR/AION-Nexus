# Deployment Guide

Three supported deployment modes, increasing in complexity.

## 1. Local Python (development / testing)

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt

# Place checkpoint
mkdir -p checkpoints
cp /path/to/best_model.pth checkpoints/aion_nexus_v1.pth

# Smoke test (no checkpoint needed)
pytest tests/test_smoke.py -v

# Inference example
python examples/01_basic_inference.py
```

## 2. Containerized server (recommended for production)

```bash
# Build
docker build -t aion-nexus:1.0.0 .

# Run with checkpoint mounted as a read-only volume
docker run -d \
    --name aion-nexus \
    -p 8080:8080 \
    -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
    aion-nexus:1.0.0

# Or with docker-compose
docker compose up -d

# Smoke test the API
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d "$(python -c 'import json,numpy as np; print(json.dumps({\"signal\": np.random.randn(2,2560).tolist()}))')"
```

The container uses CPU by default. For GPU, use `--gpus all` and set `AION_DEVICE=cuda`.

### Resource sizing

The container's defaults (1G RAM, 2 CPU) fit on a small VM. For high throughput:

| Throughput target | CPUs | RAM | Notes |
|---|---|---|---|
| < 50 req/s | 2 | 1 GB | docker-compose default |
| < 500 req/s | 8 | 2 GB | scale `--workers 4` |
| > 500 req/s | GPU | 4 GB | T4 or better |

## 3. Edge deployment (ONNX Runtime)

For Raspberry Pi / NVIDIA Jetson / industrial gateways:

```bash
# 1. Export ONNX
python -m scripts.export_onnx --checkpoint checkpoints/aion_nexus_v1.pth --dynamic-batch

# 2. Deploy aion_nexus.onnx + aion_nexus/ package to edge device
# 3. Use onnxruntime for inference (lighter than full PyTorch)
```

Minimal edge inference snippet:

```python
import onnxruntime as ort
import numpy as np
from aion_nexus.preprocessing import preprocess_signal
from aion_nexus.config import CLASS_NAMES

session = ort.InferenceSession("aion_nexus.onnx")
signal = ...  # acquired from sensor, shape [2, 2560]
x = preprocess_signal(signal).numpy()  # [1, 2, 2560]
logits, _ = session.run(["logits", "features"], {"input": x})
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
print(CLASS_NAMES[probs.argmax()])
```

## Operational concerns

### Monitoring
- `/health` endpoint exposes inference count, running latency, model version.
- Log all predictions with timestamp + confidence + signal hash for audit.
- Alert on confidence_band == "low" rate above threshold (signals model drift).

### Versioning
- Embed model version in every prediction response (`model_version` field).
- Rotate models with a canary deploy (5% traffic for 24h before full rollout).
- Keep at least 2 prior versions deployable for fast rollback.

### Sensor health
- AION-NEXUS detects stuck sensors via `SignalValidationError` (zero-std channel).
- Other failure modes (drift, calibration error) are NOT detected by the model alone — pair with sensor-health monitoring.

### Few-shot adaptation in production
- New machine type → collect 10 samples × 4 classes (40 total).
- Run `python examples/03_few_shot_adaptation.py` adapted to your data loader.
- Save adapted checkpoint with machine ID in filename.
- Deploy as a new version, do NOT replace the v1 checkpoint.

### Safety guardrails

For machines whose failure has safety implications:

1. Never auto-action `class=advanced` predictions without human review unless `confidence > 0.95`.
2. Use `confidence_band == "low"` as a "send to expert" signal, not a "predict normal" fallback.
3. Run multiple windows of inference and aggregate via `predict_long_signal` instead of trusting a single 0.1-second window.
4. Pair predictions with redundant sensors (vibration + temperature + acoustic) when criticality justifies it.

## Disaster recovery

- Checkpoints + ONNX models stored in object storage with versioned bucket policy.
- Container images tagged by semver and stored in private registry.
- Automated rebuild from git tag → image (CI/CD template at `.github/workflows/release.yml` — TODO).
