# AION-NEXUS

**Production-grade bearing degradation-stage estimation from raw vibration signals.**

> **What the labels actually are (read this first).** The 4 classes are a **positional
> life-stage** label — `degradation_pct = file_idx / (total_files − 1)` quantized into 4 bins —
> **not an independently diagnosed fault type**. This system performs **degradation-stage / RUL
> estimation, not fault-type diagnosis**. Wherever this document says "fault diagnosis" or
> "severity class", read it through this caveat. We previously labelled this "fault diagnosis";
> that name is a misnomer and is being corrected.

A multi-scale temporal deep-learning system that estimates bearing degradation stage from 2-channel accelerometer data. Honest, reproducible cross-*dataset* numbers: F1 0.615 zero-shot and 0.672 few-shot (10 samples/class) on MFPT — never seen in training. Cross-*bearing* generalization without adaptation collapses below 0.4 (LOBO); we measure and publish this rather than hide it. See [MODEL_CARD.md](MODEL_CARD.md) and [PERFORMANCE_BENCHMARKS.md](PERFORMANCE_BENCHMARKS.md).

> **Where we publish our failures.** Retracted / corrected claims live in
> [RETRACTIONS.md](RETRACTIONS.md). Exact reproduction steps (and the known non-reproducible
> numbers) live in [docs/reproduce.md](docs/reproduce.md). Read both before quoting any number.

[![Status](https://img.shields.io/badge/status-production-green)]() [![Version](https://img.shields.io/badge/version-2.2.0-blue)]() [![Python](https://img.shields.io/badge/python-3.10+-blue)]() [![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange)]() [![License](https://img.shields.io/badge/license-Apache--2.0-blue)]()

---

## What it does

Given 0.1 seconds of 2-channel vibration signal sampled at 25.6 kHz from a rolling-element bearing, AION-NEXUS predicts one of four degradation-stage classes. **These classes are a positional life-stage proxy (RUL), not a diagnosed fault type** (see headline caveat):

| Class | Meaning | Recommended action |
|---|---|---|
| 0 — Normal | Healthy bearing | No action |
| 1 — Early | Initial defect, low impact | Schedule inspection |
| 2 — Medium | Progressive degradation | Plan replacement |
| 3 — Advanced | Imminent failure | Stop machine, replace immediately |

## Verified performance

Numbers cross-validated against 4 independent result-log files in the source repository AND independently reproduced via `scripts/verify_checkpoint.py` on 2026-04-27.

### Production-default: v1 (BiGRU, recommended for industrial deployment)

| Benchmark | F1 (macro) | n samples | Verified |
|---|---|---|---|
| **FEMTO in-distribution** (11 run-to-failure bearings, **stratified-random split**) | **0.884** | 2,792 | delta 0.0003 ✓ |
| **FEMTO validation** (same bearings) | **0.898** | 2,792 | source `final_results.json` |
| **Honest leave-one-bearing-out (LOBO)** — generalization to an unseen bearing | **v6: 0.352 ± 0.112** ² | — | verified (see benchmarks) |
| **MFPT zero-shot cross-domain** | **0.615** ¹ | 94 | source `cross_validation_results.json` |
| **MFPT few-shot, 10 samples** | **0.672 ± 0.006** | 94 | source `mfpt_sensitivity_results.json` (3 runs) |

The headline **0.884 is a stratified-random split** (windows from every bearing appear in both
train and test). The honest generalization signal is the LOBO row: when a whole bearing is held
out, F1 collapses to **0.352 ± 0.112** (v6). A clean LOBO for the v1 checkpoint is still pending
(only a weaker per-bearing breakdown of the existing checkpoint exists — see benchmarks). Treat
0.884 as in-distribution, not as evidence of generalization to a new bearing.

¹ **Reproducibility caveat (2026-06-04)**: 0.615 is a logged October-2025 result, not currently
reproducible from the shipped code (current MFPT loader: F1 = 0.5546 @ n=224 vs 0.615 @ n=94;
original windowing recipe lost). See `docs/reproduce.md`.

² **LOBO caveat**: 0.352 ± 0.112 is the verified v6 leave-one-bearing-out result (full retrain
holding one bearing out). A clean v1 LOBO is **pending**. The per-bearing breakdown in the
benchmarks (0.9218 ± 0.0426) is **NOT** LOBO — it evaluates the existing checkpoint, which saw
samples from every bearing in training, so it is an optimistic proxy, not a generalization number.

### Optional architecture: v6 (TempAttn+TRM)

| Benchmark | F1 (macro) | n samples | Verified |
|---|---|---|---|
| **FEMTO in-distribution** (6 Learning_set calibration bearings) | **0.934** | 1,507 | delta 0.0000 ✓ |
| **Cross-domain on 11 run-to-failure bearings** | **0.302** | 2,792 | verified — significant collapse |

**Important**: v1 and v6 were trained on DIFFERENT FEMTO subsets. The "v6 better than v1 by +5pp" claim that may circulate in older docs is NOT a valid head-to-head: v6's high F1 is on the simpler Learning_set, and it does not transfer to the industrial run-to-failure regime. See [`MODEL_CARD.md`](./MODEL_CARD.md) section "v1 vs v6 — IMPORTANT comparability disclosure".

**Recommendation for industrial deployment**: use v1 (default). v6 is documented as an architectural exploration with cross-domain limitation.

## Why it works

Three architectural choices, each empirically validated by ablation:

1. **Multi-scale CNN** (kernels 3, 7, 15, 31, 63, 127) — captures fault signatures across the high/medium/low frequency bands where outer-race, inner-race, and cage faults respectively manifest. Removing this drops cross-domain F1 by **−20 pp**.
2. **Bidirectional GRU** (hidden=128, 2 layers) — temporal context for fault evolution. Removing it drops cross-domain F1 by **−12.7 pp**.
3. **Channel attention** (SE-style) — dynamically weights time-scales per sample. Removing it drops cross-domain F1 by **−5.5 pp**.

Total: 1,061,724 trainable parameters, 4.1 MB on disk.

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Place the trained checkpoint
mkdir -p checkpoints
cp /path/to/best_model.pth checkpoints/aion_nexus_v1.pth

# 3. Smoke test (synthetic input, no data required)
python -m pytest tests/test_smoke.py -v

# 4. Predict on your own signal
python examples/01_basic_inference.py path/to/your_signal.csv

# 5. Run as a service
uvicorn server.main:app --host 0.0.0.0 --port 8080
curl -X POST http://localhost:8080/predict_csv -F "file=@path/to/your_signal.csv"
```

## Hardware requirements

> **Estimated, not benchmarked in `results/`** — engineering estimates; measure on your target
> hardware with `python -m scripts.benchmark_inference`.

| Mode | Latency (per sample) | Throughput | RAM | Notes |
|---|---|---|---|---|
| CPU inference (1 thread) | ~12 ms | ~80 samples/sec | < 200 MB | tested on x86_64 |
| GPU inference (T4) | ~1.5 ms | ~700 samples/sec | < 1 GB | batch size 32 |
| Edge (ARM Cortex-A72) | ~45 ms | ~22 samples/sec | < 200 MB | Raspberry Pi 4 class |

The 4.1 MB model fits comfortably on edge devices. ONNX export available via `scripts/export_onnx.py`.

## Deployment

- **Local Python**: `pip install .`
- **Docker**: `docker build -t aion-nexus . && docker run -p 8080:8080 aion-nexus`
- **REST API**: FastAPI server exposes `/predict`, `/predict_csv`, `/predict_batch`, `/predict_long_signal`, `/health`, `/version`, `/metrics` (Prometheus). See `docs/api_reference.md`.
- **Edge / ONNX**: `python scripts/export_onnx.py` produces `aion_nexus.onnx` runnable on ONNX Runtime.

### Server configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `AION_CHECKPOINT` | `checkpoints/aion_nexus_v1.pth` | Checkpoint to serve |
| `AION_DEVICE` | `cpu` | Inference device |
| `AION_API_KEY` | unset (auth off) | Enables `X-API-Key` header auth on all endpoints (`/health` stays open) |
| `AION_MAX_BODY_BYTES` | `10485760` (10 MB) | Request body cap — oversized requests get `413` |
| `AION_CORS_ORIGINS` | unset (no CORS) | Comma-separated allowed origins |
| `AION_LOG_JSON` | unset (plain logs) | `1` switches to structured JSON logs with `request_id` |

## Few-shot domain adaptation

When deploying to a new machine type, you can adapt with **10 labeled samples per class** (40 total) for a +5.7 pp F1 lift (delta over the 0.615 zero-shot baseline — see caveat ¹ above on its provenance). This is engineered to minimize annotation cost.

```python
from aion_nexus import InferenceEngine, FewShotAdapter

engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")
adapter = FewShotAdapter(engine)
adapter.adapt(target_signals, target_labels, epochs=5, lr=1e-4)
adapter.save("checkpoints/aion_nexus_v1_machine42.pth")
```

See `examples/03_few_shot_adaptation.py` for a complete walk-through.

## Substrate v3 — few-shot cross-domain backbone (since 2.1.0)

For onboarding a **new machine/rig with minimal labels**, the package ships a self-supervised
PatchTST foundation encoder (`aion_nexus/substrate_v3.py`, 1,220,928 params, pretrained
NT-Xent on unlabeled FEMTO+MFPT+CWRU — 9,315 windows, 400 epochs). The encoder stays frozen;
a small head is trained per deployment with ~10 labels/class.

Verified numbers (§6.31 honest framing):

- **10-shot F1 = 0.783 ± 0.041** on held-out FEMTO bearings — **TRANSDUCTIVE, not clean LOBO.**
  The SSL encoder is contrastively pretrained on a corpus (`cache_corpus_full.npz`) that
  **includes the held-out bearings (Bearing1_5 / 2_5 / 3_3) as unlabeled data**
  (`substrate_corpus.py` iterates all 11 RTF bearings with no exclusion). So the encoder has
  seen the test bearings' signals; only their labels were withheld. A nested-LOBO with clean SSL
  (re-pretrain excluding the held-out bearing) has **not** been run, so 0.783 must not be quoted
  as leave-one-bearing-out. LOBO full-transfer (no target labels) = 0.533.
- Cross-dataset 10-shot macro-F1 0.91–1.00 on all FEMTO↔MFPT↔CWRU pairs — **binary health
  (2-class) task, vs a weak random-init control**; not a 4-class severity result. **This task is
  matched by a non-ML threshold on kurtosis** calibrated on the same 10 samples (≈ 0.911 mean
  macro-F1, 1.000 on FEMTO→CWRU), so the 0.91–1.00 number does **not** demonstrate the
  substrate's value. The value must be shown on 4-class severity, not on binary health.
- **Zero-shot cross-rig is NOT reliable** (mean lift −0.03): always collect the ~10 labels/class.
- v3 does NOT beat v1 in-distribution (FEMTO F1 0.884 stands). Bigger (v3.1) and more-data
  (v3.2) variants did not move the 10-shot ceiling (~0.78) and were not promoted.

See `MODEL_CARD.md` ("v3 substrate backbone") and `PERFORMANCE_BENCHMARKS.md`
("Substrate v3") for the full tables and caveats.

## Verified inference (Substrate Core)

The accurate classifier is the commodity; the **scarce, defensible layer is the
independent verification of that prediction** (see `AION_NEXUS_RD/14_MARKET_VISION_2032.md`).
`aion_nexus.verify` (Substrate Core) brings that layer into the public package: a
**model-agnostic** wrapper that sits above ANY classifier (our v1 BiGRU, v6, the v3
substrate encoder, or a third-party model) and turns raw probabilities into an
**auditable certificate**.

The pattern is **model proposes, verifier + conformal dispose, certificate records**:

1. **The model proposes** a probability vector (raw classifier output, preserved unchanged).
2. **A conformal verifier disposes** a verdict via a distribution-free prediction set:
   - `CERTIFIED` — the conformal set is a singleton and the top probability clears the
     abstain threshold (a confident, coverage-controlled label).
   - `REVIEW` — the set has more than one label (genuine ambiguity; a human decides).
   - `ABSTAIN` — the top probability is below the threshold (do not act).
3. **A `Certificate` records** the decision: verdict, conformal set, calibration
   parameters, a typed **assurance tier**, a `content_hash` anyone can recompute, an
   `input_sha256` binding it to the exact signal, and an `authentication` field
   (`NONE`, `HMAC-SHA256`, or `Ed25519`).

```python
import numpy as np
from aion_nexus import InferenceEngine
from aion_nexus.config import CLASS_NAMES
from aion_nexus.verify import Verifier, verify_certificate

engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")

# Calibrate ONCE on a held-out, in-distribution split (probs + true labels).
verifier = Verifier(alpha=0.1, class_names=CLASS_NAMES)
verifier.calibrate(probs_calib, labels_calib)

# Per prediction: model proposes -> verifier disposes -> certificate.
pred  = engine.predict(signal)
probs = np.array([pred.probabilities[n] for n in CLASS_NAMES])
cert  = verifier.certify(probs, input_signal=signal, model_id="aion-nexus-v1")

print(cert.verdict)              # CERTIFIED | REVIEW | ABSTAIN
print(cert.conformal_set_names)  # e.g. ['medium'] (singleton) or ['early','medium']
print(cert.authentication)       # NONE (integrity only) | HMAC-SHA256 (tamper-evident)
verify_certificate(cert)         # re-audit integrity (+ authenticity if a key is set)
```

A complete, runnable end-to-end walk-through is in
[`examples/05_verified_inference.py`](./examples/05_verified_inference.py).

**Wired into the running product (v2.6.0).** The signed certificate is not just a library:
`POST /predict_certified` returns a signed `Certificate` + public key (appended to a
hash-chained audit store), and `POST /verify` lets an auditor confirm it **offline with the
public key alone** — the property no incumbent offers. Set `VERIFY_ED25519_SEED` (generate one
with `python -c "from aion_nexus.verify import generate_seed; print(generate_seed())"`) to
sign; a weak seed is refused at the boundary, and each certificate carries a signed validity
window (anti-replay). See [`examples/06_certified_serving.py`](./examples/06_certified_serving.py)
and the [security policy](./SECURITY.md#certificate-signing-keys-v260--the-substrate-core-verification-layer).

**In the factory's own protocol fabric (v2.7.0).** `aion_nexus.connect` carries the signed
certificate onto **Sparkplug B** (MQTT / Unified Namespace) and **OPC UA** (Alarms &
Conditions / Condition Monitoring), so a Siemens / Beckhoff / SKF integrator pulls AION's
verdict off **their own bus** and verifies it offline with the public key alone — and the
Sparkplug bytes are proven wire-compatible against the reference protobuf decoder. The bridge
stays honest: an `ABSTAIN`/`REVIEW` is **never** rendered as a machine-stop alarm (severity is
capped by the trust verdict; `ABSTAIN → OPC UA Quality = Uncertain`). Building payloads needs
no extra dependency; live transports use `pip install "aion-nexus[factory]"`. See
[`examples/07_factory_bridge.py`](./examples/07_factory_bridge.py) and
[`docs/FACTORY_INTEGRATION.md`](./docs/FACTORY_INTEGRATION.md).

### Honest caveats (read before quoting any of this)

These are not fine print — they are the product. A false claim here destroys the
credibility the verifier exists to provide.

- **Conformal coverage holds ONLY under exchangeability.** The `1 - alpha` coverage
  guarantee is valid only when calibration and serving data are exchangeable.
  **Cross-bearing / cross-machine deployment breaks exchangeability and VOIDS the
  guarantee** — the sets may under-cover. Calibrate per-bearing, or treat coverage as
  advisory and monitor empirical coverage. The assumption travels on every certificate
  via the calibrator's `coverage_valid_under` field.
- **Authentication has three honest levels.** `NONE` = integrity hash only, **NOT
  tamper-evident** against an adversary with this source. `HMAC-SHA256` (env
  `VERIFY_HMAC_KEY`) = forgery-resistant, but symmetric: **whoever can verify can also
  forge**. `Ed25519` (env `VERIFY_ED25519_SEED`) = **asymmetric — the verifier holds only
  the public key and CANNOT forge.** This is what makes *independent* verification real: a
  customer, insurer, or auditor verifies a certificate with the public key alone, without
  receiving the power to mint fakes. Verifying against the **embedded** public key only
  proves self-consistency (`authenticity = SELF-SIGNED`, **not** trusted); pass the
  issuer's **expected** public key to get `VERIFIED` + `trusted`.
- **Typed assurance — overclaim is structurally impossible.** Every certificate carries an
  `assurance` tier on the ordered lattice `none < empirical < bounded < proven-spec <
  proven`. A conformal verdict is **always `empirical`** (statistical, exchangeability-
  dependent — never a proof). Composing certificates takes the **weakest link**, so an
  empirical component can never be laundered into a "proven" system claim. The tier is in
  the hashed payload, so tampering with it breaks the signature.
- **We measure the cheating surface, we don't assume it.** `aion_nexus.verify.run_cheatbench()`
  runs real attacks against the certificate gate and reports each channel CLOSED or OPEN.
  Four channels (forge-without-key, label-tamper, assurance-overclaim, downgrade) measure as
  CLOSED; the honest **OPEN** residual is that conformal coverage is *marginal*, not
  per-instance — a confident in-distribution error would pass as a valid singleton,
  indistinguishable from a correct one. We surface it rather than hide it.
- **Degradation-stage, not time-to-failure.** `aion_nexus.degradation` reports a
  **positional degradation stage** (the same 4 life-stage bins) with a conformal set —
  **NOT a calibrated RUL in hours**. See the headline caveat on positional labels.
- **Evidence toward EU AI Act, NOT compliance.** `aion_nexus.compliance.compliance_evidence(cert)`
  maps the certificate to Art.12 (logging / reconstructability), Art.14 (human oversight),
  and Art.15 (accuracy / robustness). It **provides evidence toward** those articles; it
  does **NOT** make the system "EU AI Act compliant" or "certified compliant" — compliance
  is an organizational/process result, not something a single certificate can assert. The
  `/predict_degradation` endpoint returns the degradation-stage estimate with its conformal
  set under the same caveats.

> All headline performance caveats above still apply: the 4 classes are **positional
> life-stage labels** (not diagnosed fault types); `0.884` is an **in-distribution
> stratified-random split**, with honest generalization at **LOBO 0.352 ± 0.112 (v6)**;
> and the substrate `0.783 ± 0.041` 10-shot number is **transductive (SSL leakage of the
> held-out bearings), not a clean LOBO**. Verification does not change any of these — it
> makes the model **honest about when not to trust it**.

## Limitations and scope

This product is intended for **rolling-element bearing diagnosis on rotating machinery**. Out-of-scope tasks where performance has been documented to degrade significantly:

- **Different sensor modalities** (e.g., acoustic emission, thermal). Not validated.
- **Different fault types** outside the standard four-class severity (normal/early/medium/advanced).
- **CWRU fault-location task** (ball/inner/outer/cage classification) — F1 ≈ 0.34 due to task semantic mismatch (severity ≠ location). See `docs/task_mismatch.md`.
- **Sampling rates < 10 kHz** — fault signatures fall outside the model's receptive field. Resample or retrain.

## Documentation

- [`MODEL_CARD.md`](./MODEL_CARD.md) — what the model does, training data, intended use, ethical considerations.
- [`PERFORMANCE_BENCHMARKS.md`](./PERFORMANCE_BENCHMARKS.md) — full benchmark tables, comparisons to SOTA, ablations.
- [`DEPLOYMENT.md`](./DEPLOYMENT.md) — how to deploy on cloud, edge, on-prem.
- [`docs/architecture.md`](./docs/architecture.md) — layer-by-layer architecture.
- [`docs/api_reference.md`](./docs/api_reference.md) — full REST API spec.
- [`docs/troubleshooting.md`](./docs/troubleshooting.md) — common issues.
- [`examples/05_verified_inference.py`](./examples/05_verified_inference.py) — end-to-end
  **Verified inference (Substrate Core)**: model proposes, conformal verifier disposes,
  auditable certificate + EU AI Act evidence, with the honest caveats inline.

## License

Apache License 2.0 — see [`LICENSE`](./LICENSE). Commercial use permitted with attribution.

## Citation

If this work supports a paper, please cite:

```bibtex
@article{aion_nexus_2025,
  title  = {NEXUS: Multi-Scale Temporal Deep Learning for Zero-Shot and Few-Shot
            Cross-Domain Bearing Degradation-Stage Estimation},
  author = {Culotta, Daniel},
  year   = {2025},
  note   = {Manuscript in preparation. Target: IEEE Trans. Industrial Informatics.
            NB: labels are positional life-stage (RUL proxy), not diagnosed fault type.}
}
```

## Contact

Daniel Culotta — daniel.culotta@gmail.com
AION NEXUS — predictive maintenance for rotating machinery.
