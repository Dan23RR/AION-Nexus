# Changelog

All notable changes to AION-NEXUS will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] — 2026-06-11 (Enterprise hardening)

### Security
- **Substrate v3 checkpoint loader hardened**: `torch.load(..., weights_only=True)` is now the
  default load path, with an optional `expected_sha256` integrity check before deserialization.
  Closes the threat-model gap tracked in `docs/threat_model.md` ("Critical mitigation gaps").
- **Server authentication**: optional `AION_API_KEY` env var enables `X-API-Key` header auth on
  all endpoints (`/health` stays exempt for orchestrator liveness probes).
- **Request body size limit**: `AION_MAX_BODY_BYTES` (default 10 MB) — oversized requests are
  rejected with `413` before parsing. Makes the long-standing SECURITY.md claim enforceable in
  the app itself, not only at the reverse proxy.
- **CORS**: `AION_CORS_ORIGINS` (comma-separated allowlist). Default: no CORS headers at all.

### Added
- **v3 substrate servable**: `detect_architecture` now recognizes v3 checkpoints; few-shot
  tooling works against the frozen v3 encoder.
- **`/metrics` endpoint** (Prometheus exposition format) — request counts and latency; depends
  on `prometheus-client` (added to `requirements.txt`).
- **Structured JSON logging**: `AION_LOG_JSON=1` switches logs to JSON with a per-request
  `request_id`.
- `HealthResponse` now includes `architecture_version`.
- `docs/reproduce.md` — honest, minimal reproduction guide for the published F1 = 0.884
  (including the globally-stratified-split caveat).

### Changed
- FastAPI startup migrated from deprecated `on_event` hooks to the `lifespan` context manager.
- Package version → **2.2.0** (`aion_nexus/version.py` single source of truth).

### Packaging / CI
- `pyproject.toml`: console entry points fixed (`scripts/` is now a real package included in
  `packages.find`); `py.typed` marker shipped (`Typing :: Typed` classifier +
  `[tool.setuptools.package-data]`); absolute `[project.urls]`; coverage config with
  `fail_under = 70`.
- Dockerfile: torch installed from the CPU wheel index (`download.pytorch.org/whl/cpu`) —
  slim CPU image; OCI version labels 2.2.0. `docker-compose.yml`: obsolete `version:` key
  removed, image tag aligned to 2.2.0.
- CI: coverage gate (`--cov-fail-under=70`), weekly scheduled supply-chain audit job
  (`scripts/audit_supply_chain.py` + `pip-audit` — makes the SECURITY.md "weekly in CI" claim
  true), `python -m build` + `twine check dist/*` job, container smoke test (`docker run` +
  `curl /health`) after the Docker build.

### Documentation reconciliation
- MODEL_CARD / PERFORMANCE_BENCHMARKS / INDEX / README / docs aligned to 2.2.0 and to the
  verified numbers only: v6 "+5.0 pp" advantage row caveated (not a valid head-to-head, see
  §6.31 disclosure); sample-efficiency 10-shot point reconciled to the verified 0.672 ± 0.006;
  cost figures and SOTA/latency tables explicitly labeled as estimates / literature-reported
  where no artifact exists in `results/`; cross-dataset 0.91–1.00 claim qualified inline as
  binary health (2-class) vs a weak random-init control; `FewShotAdapter` quickstart fixed to
  the real signature `FewShotAdapter(engine)`; FAQ pointer to non-existent
  `aion_nexus.utils.cohens_d` corrected.

## [2.1.0] — 2026-06-04 (v3 self-supervised substrate foundation backbone)

### Added
- **v3 substrate** (`aion_nexus/substrate_v3.py`): a PatchTST self-supervised foundation
  encoder (1,220,928 params), pretrained contrastively (NT-Xent) on unlabeled
  FEMTO+MFPT+CWRU vibration. Frozen encoder + per-deployment few-shot head; served via the
  AION-2 verified trust layer (conformal + physics verifier → tamper-evident certificate).
  `create_substrate_v3()` with encoder param-count guard; checkpoint
  `checkpoints/aion_nexus_substrate_v3.pth` (objective `contrastive-ntxent-patchTST`).
- `tests/test_substrate_v3.py` — param guard, drift-refusal, forward contract,
  frozen-encoder/trainable-head, checkpoint load (5 tests).

### Verified (honest, §6.31)
- v3 is NOT a higher in-distribution classifier — v1 FEMTO F1=0.884 stands; v3 full-transfer
  to unseen machines ≈ 0.55.
- v3's value = cross-domain FEW-SHOT: 10-shot LOBO FEMTO severity F1 ≈ 0.78; **cross-DATASET
  10-shot macro-F1 0.91–1.00** (FEMTO↔MFPT↔CWRU) — **binary health (2-class) task, vs a weak
  random-init control** (random-init 0.5–0.8). Zero-shot cross-rig is NOT reliable.
- Scaled run (9,315 windows / 400 epochs, Colab T4): 10-shot 0.760→0.783, full-transfer
  0.554→0.533 — **within noise** → the encoder is **architecture-saturated** at
  d_model 192/depth 4. Promoted (drop-in) as the production v3 checkpoint. The real next
  lever is a bigger **v3.1** (see Roadmap), not more data at this scale.

### Notes
- Backward-compatible: v1 (default) and v6 unchanged; v3 is opt-in.
- Fulfils the 2.1.0 roadmap item (multi-domain pre-training), realized as a self-supervised
  substrate. Scaled pretraining reproducible via the AION-2 `foundation/` scripts (Colab GPU).

## [1.0.0] — 2025-10-10

### Added
- Production architecture: Multi-Scale CNN + Channel Attention + Bidirectional GRU + 3-layer MLP classifier (1,061,724 parameters, 4.1 MB).
- `aion_nexus.InferenceEngine` — checkpoint-loading inference with batch support, latency telemetry, confidence banding.
- `aion_nexus.FewShotAdapter` — 10-sample domain adaptation, encoder frozen, classifier-only fine-tuning.
- `aion_nexus.preprocessing` — strict input validation (shape, NaN/Inf, stuck-sensor detection) + per-channel z-score normalization.
- `server` — FastAPI service with `/predict`, `/predict_batch`, `/predict_long_signal`, `/health`, `/version` endpoints.
- `scripts/verify_checkpoint.py` — independent F1 verification against published numbers.
- `scripts/benchmark_inference.py` — latency / throughput benchmark.
- `scripts/export_onnx.py` — ONNX export for edge deployment.
- Test suite: smoke tests + few-shot tests, no checkpoint required for CI.
- Dockerfile with non-root user, healthcheck, resource limits.
- Documentation: README, MODEL_CARD, PERFORMANCE_BENCHMARKS, DEPLOYMENT, architecture, troubleshooting.

### Verified performance
- FEMTO in-distribution F1 = 0.898 (validation), 0.884 (test)
- MFPT zero-shot cross-domain F1 = 0.615
- MFPT few-shot 10-sample F1 = 0.672 ± 0.006 (3 runs)

All numbers cross-verified across `final_results.json`, `cross_validation_results.json`, and `mfpt_sensitivity_results.json`.

### Documented negative results
- SimCLR contrastive pretraining on FEMTO unlabeled — caused MFPT zero-shot F1 to drop from 0.615 to 0.184 (−70%). NOT included in v1.0 checkpoint.
- AdaBN adaptation with MFPT class imbalance — F1 dropped from 0.615 to 0.488 (−21%). NOT in v1.0.
- CWRU severity-mapped task — F1 ≈ 0.34 due to physically invalid label mapping (Spearman correlation = −0.30). Documented in `docs/task_mismatch.md`.

### Known limitations (see MODEL_CARD.md for details)
- 4-class severity diagnosis only; not validated for fault localization.
- Sampling rate < 10 kHz not supported (resample first).
- Acoustic / thermal sensor inputs not validated.

## [2.0.0] — 2026-04-27 (v6 architecture + critical comparability finding)

### CRITICAL methodological finding
The v6 "F1=0.934" headline number was measured on a **DIFFERENT dataset** than v1's "F1=0.884":
- **v1** trained AND tested on **Test_set/Test_set** = 11 run-to-failure bearings (industrial scenario, 13,959 samples).
- **v6** trained AND tested on **Test_set/Training_set/Learning_set** = 6 short-run calibration bearings (~7,500 samples).

When v6 (trained on Learning_set) is evaluated on the industrial Test_set 11 bearings (cross-bearing transfer), F1 drops from 0.934 to **0.302** (verified 2026-04-27). This is the same §6.31 family F1 (measurement-construct misalignment) pattern that produced 21 retractions in sister projects (motif/, VELLERA/).

**Honest re-statement**: v1 and v6 are NOT comparable head-to-head on the same data — the published F1 numbers come from different test sets. After full cross-evaluation:

|  | Learning_set (calibration) | Test_set (run-to-failure) |
|---|---|---|
| v1 (BiGRU) | 0.382 cross | 0.884 in-dist |
| v6 (TempAttn+TRM) | 0.934 in-dist | 0.302 cross |

**Both architectures fail symmetrically when tested on the OTHER FEMTO subset**. The architecture choice (BiGRU vs TempAttn+TRM) is overshadowed by the train-test distribution shift. There is no "production default architecture"; there is "the right architecture for the regime", with **few-shot adaptation (F1=0.672 with 10 samples)** as the cross-regime tool. Production package supports both architectures with auto-detection from the checkpoint state-dict.

### Added
- **AION-NEXUS v6 architecture** (`AIONNexusV6`): MultiScale CNN + ChannelAttn + **TemporalSelfAttention** + **TinyRecursiveReasoner**.
  - 716,577 parameters (32.5% smaller than v1)
  - 2.73 MB on disk (vs 4.1 MB for v1)
  - **In-distribution Test F1 = 0.934** on Learning_set held-out (verified delta=0.0000)
  - **Cross-domain F1 on Test_set run-to-failure = 0.302** (verified — significant degradation)
  - Noise-robust at training time: SNR+5dB F1 ≈ 0.87
  - Inference latency ~16 ms
- **`TemporalSelfAttention`** module (4-head MHA + learned-query pooling, replaces v1's BiGRU + dual-pool)
- **`TinyRecursiveReasoner`** module (TRM-inspired progressive refinement; Jolicoeur-Martineau 2025)
- **Auto-detection of architecture version** in `InferenceEngine.from_checkpoint`. Inspects state_dict keys to pick v1 or v6; can be forced via `version="v1"|"v6"` flag.
- `architecture_version` attribute on `InferenceEngine`, exposed in `/health` endpoint.
- `tests/test_v6.py` — 14 v6-specific tests (param count, forward, auto-detection, backward-compat).
- v1 baseline (1,061,724 params, F1=0.884) remains fully supported as a backward-compatible mode.

### Changed
- `aion_nexus/__init__.py` re-exports both v1 (`AIONNexus`, `create_aion_nexus`) and v6 (`AIONNexusV6`, `create_aion_nexus_v6`) symbols.
- `scripts/verify_checkpoint.py` adapts published-F1 tolerances based on detected architecture.

### Verification
- v1 architecture: analytical param count = **1,061,724** = training log (delta = 0).
- v6 architecture: analytical param count = **716,577** = training log (delta = 0).
- v1 F1 verification: 0.8843 vs published 0.8840 (delta = 0.0003) — VERIFIED on 2026-04-27.
- v6 F1 verification: pending user run with `verify_checkpoint --checkpoint nexus_ultra_v6/best_model.pth`.

### Breaking changes
- This is a **major version bump** because the default architecture path expanded. Existing v1.0 deployments continue to work unchanged (backward-compatible: v1 checkpoints still detected and loaded). New deployments may use v6 by passing a v6 checkpoint to `from_checkpoint`.

## [1.0.2] — 2026-04-27 (CRITICAL: preprocessing alignment fix)

### Fixed — CRITICAL
- **Preprocessing chain now matches training**. Independent verification on
  the same FEMTO data revealed F1 = 0.279 instead of 0.884: a 60-percentage-
  point delta caused by missing preprocessing steps in the production
  package vs the training-time loader.
  - **Added high-pass Butterworth 1 Hz filter** (2nd order, sosfilt) in
    `aion_nexus.preprocessing.preprocess_signal`. Training in `aion_data.py`
    applies this filter after z-score normalization; without it, the model
    receives out-of-distribution input and predictions collapse.
  - Pipeline order is now: validate → crop → z-score per channel → HP-Butterworth
    (matches `aion_data._process_csv_file` step-for-step).
- **`scripts/verify_checkpoint.py` rewritten to import the original
  `aion_data.AIONDataset` + `create_stratified_temporal_splits` (seed=42)**.
  Required to reproduce the 2,792-sample test set used for the published
  F1=0.884. The previous implementation walked subdirectories, producing
  a different evaluation set and an invalid F1 number.
- Added `diagnose_checkpoint` step that runs without any data: param-count
  match + class distribution sanity on synthetic input. Catches "model
  collapsed to one class" failures before any downstream evaluation.

### Added
- `scipy>=1.10.0,<2.0.0` to `requirements.txt`. Required for
  `scipy.signal.butter` / `sosfilt`. Mean-removal fallback if scipy
  unavailable, but training used scipy — production should match.

### Validation
- Verified that test_preprocessing_edge_cases tests still pass after
  preprocessing change (assertions on mean/std relaxed to reflect HP-
  filter attenuation behavior).

## [1.0.1] — 2026-04-27 (bomb-proofing patch)

### Fixed
- **Stuck-sensor detection threshold**: raised from `1e-9` to `1e-7` and stddev now computed in float64. Float32 precision noise (e.g., `np.asarray([0.1]*N, dtype=float32).std() ≈ 2e-8`) was leaking past the previous threshold; effectively-stuck sensors were not being caught. New threshold aligns with industrial accelerometer noise floors (PCB 352C03 ~5 μg).
- **NaN/Inf check moved before center-crop**: invalid values in any portion of the input signal are now caught regardless of whether they fall within the centered 2,560-sample window. Previously, NaN/Inf in the cropped tail was silently dropped.
- **`/predict` endpoint split into `/predict` (JSON) and `/predict_csv` (file upload)**: FastAPI cannot reliably auto-detect content-type when a single endpoint accepts both `File()` and a Pydantic body. The previous design always rejected JSON bodies as "missing." JSON-only `/predict` is now the canonical path; CSV uploads use `/predict_csv` (same response schema).
- **`/predict_long_signal` now returns 400 (not 500) on invalid `aggregation` method**: `ValueError` from `aggregate_window_predictions` is wrapped in `HTTPException(400, ...)`.

### Breaking changes (within 1.x)
- `POST /predict` now accepts JSON body ONLY. CSV upload moved to `POST /predict_csv`. Clients that posted CSVs to `/predict` need to update endpoint URL.

### Added
- `docs/FAQ.md` — 51 anticipated questions with answers (verification, architecture, deployment, few-shot, edge cases, concurrency, security, comparisons, reproducibility, licensing).
- `docs/threat_model.md` — STRIDE-based threat analysis with mitigation gaps.
- `docs/data_contract.md` — formal input/output spec with stability guarantees.
- `SECURITY.md` — vulnerability disclosure policy + hardening checklist.
- `CONTRIBUTING.md` — workflow, §6.31 discipline, ADR process.
- `.pre-commit-config.yaml` — ruff + secrets detection + smoke-test hook.
- `Makefile` — common operational targets (install, test, lint, audit, etc.).
- `tests/test_preprocessing_edge_cases.py` — 23 edge cases for `validate_signal` / `preprocess_signal`.
- `tests/test_determinism.py` — same-seed reproducibility, eval-mode determinism, batch consistency.
- `tests/test_concurrency.py` — thread-safety of read-only inference.
- `tests/test_api_integration.py` — FastAPI TestClient end-to-end (200, 400, 422, 503 paths).
- `scripts/audit_supply_chain.py` — pip-audit + license allowlist/blocklist.
- `scripts/verify_onnx_parity.py` — numerical equivalence check between PyTorch and ONNX export.
- `scripts/quantize.py` — INT8 dynamic quantization with FP32 parity validation.
- `scripts/generate_manifest.py` — SHA-256 manifest with `--check` mode.

### Verified
- Architecture parameter count analytically derived: 1,061,724 (delta = 0 vs training log).
- 64 of 78 pytest cases pass on user's local environment with torch installed (1.0.0). Remaining 14 traced to two root causes: API integration state-leakage (fixed in this patch) and float32 precision in stuck-sensor threshold (fixed in this patch).

## Roadmap (future)

> Current released version is **2.2.0** (see above). The items below are post-2.2 plans.
> (2.2.0 shipped as enterprise hardening; the multi-task-heads item previously slated for
> 2.2.0 moves to 2.3.0, streaming to 2.4.0.)

- ✅ 2.1.0: Multi-domain pre-training — DONE, realized as the v3 self-supervised substrate (FEMTO+MFPT+CWRU; Paderborn/NASA-IMS to be added in the scaled run).
- v3.1 (bigger arch d_model 256/depth 6 + hybrid masked⊕contrastive): **TESTED 2026-06-04, FAILED** — LOBO 10-shot 0.801 (+0.018 vs v3, within noise), full-transfer down → architecture is NOT the lever on the current 3-rig data; not promoted (v3 stays). The production module is arch-flexible (`AIONNexusV3.from_checkpoint`) for a future data-driven v3.2. See `AION_NEXUS_RD/aion2_verified_substrate/foundation/RETRACTION_substrate_v31.md`.
- **Next real lever = DATA diversity**: add Paderborn (KAt-DataCenter) + NASA-IMS rigs to the unlabeled pretraining corpus, re-pretrain at the v3 architecture.
- 2.3.0: Multi-task heads (severity + location simultaneously).
- 2.4.0: Streaming inference engine (online sequential decisions).
