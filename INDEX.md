# AION-NEXUS — Package Navigation

Production-grade bearing-fault diagnosis. Start here.

## Read in this order

1. [`README.md`](./README.md) — what it does, quickstart, performance summary.
2. [`MODEL_CARD.md`](./MODEL_CARD.md) — intended use, training data, ethical considerations, limitations.
3. [`PERFORMANCE_BENCHMARKS.md`](./PERFORMANCE_BENCHMARKS.md) — full verified numbers, ablations, SOTA comparison.
4. [`DEPLOYMENT.md`](./DEPLOYMENT.md) — local / Docker / edge.
5. [`CHANGELOG.md`](./CHANGELOG.md) — release log + roadmap.

## Documentation

- [`docs/architecture.md`](./docs/architecture.md) — layer-by-layer model spec.
- [`docs/api_reference.md`](./docs/api_reference.md) — REST API (auth, env vars, `/metrics`).
- [`docs/reproduce.md`](./docs/reproduce.md) — how to reproduce the published F1 numbers (with caveats).
- [`docs/troubleshooting.md`](./docs/troubleshooting.md) — common issues.
- [`docs/negative_results.md`](./docs/negative_results.md) — what didn't work and why.
- [`docs/task_mismatch.md`](./docs/task_mismatch.md) — CWRU case study (label semantics).
- [`docs/threat_model.md`](./docs/threat_model.md) — STRIDE analysis + mitigation status.
- [`docs/FAQ.md`](./docs/FAQ.md) — anticipated reviewer/customer questions.

## Code

- `aion_nexus/` — installable Python package (model, inference, preprocessing, few-shot).
- `server/` — FastAPI service.
- `tests/` — pytest suite (no checkpoint required for smoke tests).
- `scripts/` — operational scripts (verify checkpoint, benchmark inference, ONNX export).
- `examples/` — quickstart usage.

## Operational artifacts

- `Dockerfile` + `docker-compose.yml` — container deployment.
- `.github/workflows/ci.yml` — CI on push/PR.
- `pyproject.toml` + `requirements.txt` — Python packaging.

## Where the trained checkpoint goes

Place `aion_nexus_v1.pth` in `checkpoints/` (the file is NOT committed; see `checkpoints/README.md` for how to obtain it from the source training repository).

## Independence verification

If you want to re-verify the published F1 numbers against your own data, run:

```bash
python -m scripts.verify_checkpoint \
    --checkpoint checkpoints/aion_nexus_v1.pth \
    --femto-test data/femto_test \
    --mfpt data/mfpt
```

The script asserts:
- FEMTO test F1 within ±0.005 of 0.884 → VERIFIED
- MFPT zero-shot F1 within ±0.010 of 0.615 → VERIFIED
- Otherwise → flagged as DEVIATION with delta reported.

> **Caveat (2026-06-04)**: the MFPT zero-shot 0.615 is a logged October-2025 result that is
> **not currently reproducible** from the shipped code (current loader: 0.5546 @ n=224 vs
> 0.615 @ n=94; original windowing recipe lost). See `docs/reproduce.md`.

## Verified delivery

The package was originally assembled and statically verified on 2026-04-27; the current
release is **2.2.0** (see [`CHANGELOG.md`](./CHANGELOG.md)).

Verification status (cross-source agreement against 4 independent JSON result files in the source repository):

| Claim | Verified |
|---|---|
| F1 = 0.898 FEMTO validation | ✓ `final_results.json` |
| F1 = 0.884 FEMTO test | ✓ `cross_validation_results.json` |
| F1 = 0.615 MFPT zero-shot | ✓ `cross_validation_results.json` |
| F1 = 0.672 MFPT few-shot 10 | ✓ `mfpt_sensitivity_results.json` (mean of 3 runs) |
| 1,061,724 model parameters | ✓ `aion_nexus_training.log` line 9 |
| Architecture spec | ✓ AST-verified against source `aion_nexus.py` |

The package is structurally complete; linting (`ruff check`) and the coverage gate are enforced in CI on every push/PR (`.github/workflows/ci.yml`). Runtime smoke test (`pytest tests/test_smoke.py`) requires `pip install -r requirements.txt` on the host machine.

## Path to commercial production

Recommended next steps once you've placed the checkpoint and confirmed `pytest` passes:

1. **Smoke test in your environment**: `pytest tests/test_smoke.py -v`.
2. **Verify F1**: run `scripts/verify_checkpoint.py` against your data copies of FEMTO + MFPT.
3. **Build container**: `docker build -t aion-nexus:2.2.0 .` then `docker run -p 8080:8080 -v $(pwd)/checkpoints:/app/checkpoints:ro aion-nexus:2.2.0`.
4. **Hit the API**: `curl http://localhost:8080/health` then `python examples/04_api_client.py`.
5. **Few-shot adapt** to a target machine: replace `make_target_dataset()` in `examples/03_few_shot_adaptation.py` with your data loader, run, save adapted checkpoint.
6. **Edge deploy** (optional): `python -m scripts.export_onnx --checkpoint checkpoints/aion_nexus_v1.pth --dynamic-batch`.

For commercial use cases requiring SLA / safety certification (e.g., ISO 13374, IEC 61508):
- Pair predictions with redundant sensor health monitoring.
- Add per-tenant rate limiting + auth in front of the FastAPI server.
- Maintain a rollback path (keep N-1 and N-2 checkpoints versioned).
- Continuously monitor `confidence_band == "low"` rate as a model-drift signal.

## License

Apache 2.0 — see [`LICENSE`](./LICENSE). Commercial use permitted with attribution.

## Contact

Daniel Culotta — daniel.culotta@gmail.com
