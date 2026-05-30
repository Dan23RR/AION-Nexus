# Troubleshooting

Common issues and their resolution.

## Setup

### "ImportError: No module named aion_nexus"

The package is not installed. Run from project root:

```bash
pip install -e .
```

Or set `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

### "ValueError: Architecture drift: got X params, expected 1,061,724"

The model code (`aion_nexus/model.py`) has been modified in a way that changes parameter count. This breaks checkpoint compatibility. Either:

- Revert the changes (use `git diff aion_nexus/model.py`).
- Bump the major version and retrain from scratch.

The strict check is in `create_aion_nexus()`; bypass via direct instantiation only for development.

## Inference

### "FileNotFoundError: Checkpoint not found at checkpoints/aion_nexus_v1.pth"

Place the trained checkpoint at the expected path. See `checkpoints/README.md` for instructions on obtaining `best_model.pth` from the source training repository.

### "SignalValidationError: Channel 0 appears stuck (std=0)"

Your accelerometer signal has constant values on at least one channel. Either:

- Sensor failure / disconnection — diagnose hardware.
- Recording was zero-padded — re-acquire.
- Pre-processing zeroed channel — check upstream pipeline.

The model refuses to predict on stuck sensors to prevent confidently-wrong outputs.

### "SignalValidationError: Signal too short"

You provided fewer than 2,560 samples. The model needs 0.1 seconds at 25.6 kHz. Either acquire a longer signal or upsample.

### Predictions are "low confidence" on most samples

Possible causes:

1. **Domain shift** — your machine differs significantly from FEMTO training data. Solution: collect 10 labeled samples per class and run `examples/03_few_shot_adaptation.py`.
2. **Sampling rate mismatch** — your sensor samples at != 25.6 kHz. Resample to 25.6 kHz before inference.
3. **Sensor calibration drift** — recalibrate accelerometers.
4. **Genuine borderline degradation** — class boundaries are inherently fuzzy. Aggregate over multiple windows via `/predict_long_signal`.

### Predictions concentrate on one class regardless of input

Almost always indicates either:

- Model checkpoint did not load (random weights). Check `engine.get_health()` — `model_param_count` should be 1,061,724 and the predictions should differ across diverse inputs.
- Signals are not z-score-normalized. The preprocessing pipeline does this; if you bypass it, normalize manually.

### Latency is much higher than reported (12 ms on CPU)

- First few inferences include warmup overhead. Skip the first ~20 calls when measuring.
- Other processes competing for CPU. Pin to dedicated cores in production.
- Heavy logging in tight loop. Reduce log verbosity.
- If still slow on CPU, export to ONNX and use ONNX Runtime — typically 30% faster.

## Few-shot adaptation

### Adaptation loss does not decrease

- **Insufficient samples**: need at least 4, recommended 10 per class.
- **Label noise**: re-verify your labels.
- **Same class for all samples**: adapter expects multi-class data.
- **Learning rate too high**: try `lr=5e-5` instead of `1e-4`.

### Adapted model performs worse on target data than base zero-shot

- **Sample distribution mismatch**: your 10 samples may not be representative. Sample from across the operating envelope.
- **Catastrophic forgetting**: the encoder is frozen by design; if you also unfreeze it, the model over-fits to the few samples. Keep encoder frozen.
- **Class imbalance**: 10 normal + 0 advanced will collapse the head. Stratify samples.

### Adapter modifies the source engine

It shouldn't — the adapter deep-copies the model. If it does (predictions on the source change after `adapter.adapt()`), that's a bug. Open an issue.

## Server / Docker

### "Engine unavailable: Checkpoint not found"

Set the `AION_CHECKPOINT` environment variable:

```bash
AION_CHECKPOINT=/path/to/aion_nexus_v1.pth uvicorn server.main:app
```

Or in Docker:

```bash
docker run -e AION_CHECKPOINT=/app/checkpoints/aion_nexus_v1.pth -v ./checkpoints:/app/checkpoints aion-nexus
```

### Container exits immediately

Check logs: `docker logs aion-nexus`. Most common: missing checkpoint, missing dependency, or port conflict.

### "Address already in use" on port 8080

Another process is using the port. Either kill it or use a different port:

```bash
uvicorn server.main:app --port 9090
```

### Health probe fails on Kubernetes / orchestrator

Increase `start_period` in healthcheck — the model takes ~3-5 seconds to load on first request. The provided `Dockerfile` uses `start-period=10s`.

## Performance

### F1 on my dataset is much lower than 0.898

Expected if your dataset is not FEMTO. The model was trained on FEMTO and verified at:

- F1=0.898 on FEMTO (in-distribution)
- F1=0.615 on MFPT (zero-shot cross-domain)

For a new dataset:

1. Run zero-shot — expect 0.5–0.7 F1 depending on similarity to FEMTO.
2. If unacceptable, run few-shot adaptation with 10 samples/class.
3. If still unacceptable, the gap may be too large — retrain from scratch on combined data (out of scope of this v1.0 release).

### F1 on FEMTO test is below 0.880

Possible deviation sources:

- Different test split — the released number used the specific 60/20/20 stratified split documented in `MODEL_CARD.md`.
- Different random seed — the released number was deterministic from seed 0.
- Modified preprocessing — verify `preprocess_signal` produces z-scored output.

If deviation > 0.005, run `python -m scripts.verify_checkpoint` to compare directly against published numbers.

## Reporting bugs

Include:

1. AION-NEXUS version (`python -c "import aion_nexus; print(aion_nexus.__version__)"`)
2. Python + PyTorch version
3. OS + hardware
4. Minimal reproducer (input shape, expected behavior, actual behavior)
5. Log output

Open an issue or email daniel.culotta@gmail.com.
