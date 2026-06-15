# Model Card — AION-NEXUS v2.2

> **What this model actually estimates (read first).** The 4 output classes are a **positional
> life-stage** label: `degradation_pct = file_idx / (total_files − 1)`, quantized into 4 bins.
> They are **NOT an independently diagnosed fault type**. This is **degradation-stage / RUL
> estimation, not fault-type diagnosis.** Every use of "severity", "fault diagnosis", or "class"
> below should be read through this caveat. "Fault diagnosis" in older copy is a misnomer.
>
> **Honesty pointers:** retracted/corrected claims → [RETRACTIONS.md](RETRACTIONS.md);
> exact reproduction steps and known-non-reproducible numbers → [docs/reproduce.md](docs/reproduce.md).

## Model details

- **Name**: AION-NEXUS
- **Version**: 2.2.0 (2026-06-11; enterprise hardening — server auth/limits/metrics, hardened v3 loader, packaging/CI fixes; see `CHANGELOG.md`). 2.1.0 (2026-06-04) added the **v3 self-supervised substrate** foundation backbone. 2.0.0 (2026-04-27) added v6 + the §6.31 comparability finding. The v1 BiGRU below remains the production-recommended IN-DISTRIBUTION architecture; v3 is the cross-domain FEW-SHOT backbone (see "v3 substrate backbone" below).
- **Architecture**: Multi-Scale Temporal CNN + Channel Attention + Bidirectional GRU + 3-layer MLP classifier
- **Parameters**: 1,061,724 (4.1 MB FP32)
- **Input**: 2-channel vibration signal, 2,560 samples (0.1 s @ 25.6 kHz)
- **Output**: 4-class softmax probability distribution over **positional life-stages** (normal, early, medium, advanced = degradation-stage bins, **not diagnosed fault types** — see headline caveat)
- **License**: Apache 2.0
- **Author**: Daniel Culotta
- **Contact**: daniel.culotta@gmail.com

## v3 substrate backbone (added 2.1.0) — honest positioning

v3 is a **self-supervised PatchTST foundation encoder** (~1.22M params), pretrained
contrastively (NT-Xent) on unlabeled FEMTO+MFPT+CWRU vibration. It is **frozen**; a small
head is trained per-deployment with ~10 labels/class (few-shot), then served through the
AION-2 verified trust layer (conformal prediction sets + closed-form physics verifier →
tamper-evident certificate).

**What v3 is NOT (§6.31):** a higher in-distribution classifier. v1's FEMTO in-distribution
F1=0.884 is unbeaten by v3 for that task (v3 full-transfer to unseen machines ≈ 0.55).

**What v3 IS:** the cross-domain few-shot backbone, evaluated under a (transductive) held-out-bearing
protocol and **cross-DATASET** (FEMTO↔MFPT↔CWRU):
- 10-shot macro-F1 **0.91–1.00** across all dataset pairs — **binary health (2-class) task, vs a weak random-init control** (random-init 0.5–0.8); not a 4-class severity result.
  **Triviality caveat (new):** this binary cross-rig task is **matched by a non-ML threshold on
  kurtosis** calibrated on the same 10 samples (≈ 0.911 mean macro-F1; 1.000 on FEMTO→CWRU).
  So 0.91–1.00 does **NOT** demonstrate the substrate's value — the value must be proven on
  4-class severity, not on binary health.
- 10-shot held-out-bearing severity F1 ≈ 0.78 (precisely 0.783 ± 0.041, see benchmarks) on
  held-out FEMTO bearings. **TRANSDUCTIVE, not clean LOBO (new caveat):** the SSL encoder is
  contrastively pretrained on a corpus (`cache_corpus_full.npz`) that **includes the held-out
  bearings (Bearing1_5 / 2_5 / 3_3) as unlabeled data** — `substrate_corpus.py` iterates all 11
  RTF bearings with no exclusion. The encoder has therefore seen the held-out bearings' signals;
  only their labels were withheld. A clean nested-LOBO SSL (re-pretrain excluding the held-out
  bearing) has **not** been run. Do **not** quote 0.783 as leave-one-bearing-out without the
  word "transductive". (Scaled run 9,315/400ep: within
  noise → architecture-saturated. **v3.1** — a 2.6× bigger net + hybrid masked⊕contrastive —
  was tested and ALSO did not improve beyond noise (10-shot 0.801, +0.018) → the lever is
  **data diversity** (more rigs: Paderborn/NASA-IMS), NOT model size. v3.1 not promoted; v3
  stays. See `AION_NEXUS_RD/aion2_verified_substrate/foundation/RETRACTION_substrate_v31.md`.)
- zero-shot cross-rig is **NOT reliable** — collect the ~10 labels.

Use v3 to onboard a NEW machine/rig with minimal labels, and as the trustworthy backbone
behind certified inference. Checkpoint: `checkpoints/aion_nexus_substrate_v3.pth` (objective
`contrastive-ntxent-patchTST`). Trust layer + few-shot tooling live in the AION-2 package
(`AION_NEXUS_RD/aion2_verified_substrate/`).

## Intended use

Predictive maintenance of rolling-element bearings on rotating machinery, in two
complementary modes:

1. **Degradation-stage estimation.** From vibration accelerometer data, estimate the
   bearing's **degradation stage** — a positional life-stage / RUL proxy, **not
   fault-type diagnosis** (see headline caveat) — to inform maintenance scheduling.
   Surfaced as a first-class capability via `aion_nexus.degradation` (a
   `DegradationEstimate`: the positional stage plus a conformal set) and the
   `/predict_degradation` endpoint.

2. **Verified inference (Substrate Core).** Wrap the classifier in the model-agnostic
   verification layer (`aion_nexus.verify`): a conformal prediction set + abstain logic
   turn each raw probability vector into an auditable `Certificate` with a
   `CERTIFIED / REVIEW / ABSTAIN` verdict. The defensible value is **independent
   verification — telling the operator when NOT to trust the model** — not a higher
   accuracy number. The certificate maps to EU AI Act **evidence** (Art.12/14/15) via
   `aion_nexus.compliance.compliance_evidence` (evidence toward, NOT a compliance claim).

### Primary use cases

- Industrial condition monitoring systems
- Fleet-wide bearing health assessment
- Predictive maintenance scheduling
- Anomaly screening on streaming sensor data
- **Auditable, certified maintenance decisions** — the certificate (CERTIFIED/REVIEW/ABSTAIN)
  as the deliverable, with an append-only audit trail for Art.12-style logging evidence

### Out of scope

This system is explicitly NOT, and must not be presented as:

- **Fault-type diagnosis.** The 4 classes are a positional degradation stage
  (`degradation_pct = file_idx / (total − 1)` quantized into 4 bins), **not an
  independently diagnosed fault type**. It does not identify *which* defect is present.
- **RUL in calibrated time units.** `aion_nexus.degradation` reports a degradation
  **stage** with a conformal set, **NOT a remaining-useful-life in hours/cycles**. Any
  time-to-failure read-off would be uncalibrated and is not supported.
- **A declaration of EU AI Act (or any) compliance.** The certificate and
  `compliance_evidence` **provide evidence toward** Art.12/14/15; they do **NOT** make the
  system "compliant" or "certified compliant". Compliance is an organizational/process
  result, not something a model or a single certificate asserts.
- **A coverage guarantee outside exchangeability.** The conformal `1 - alpha` coverage
  holds ONLY under exchangeability of calibration and serving data; cross-bearing /
  cross-machine deployment **breaks it** and the guarantee is void (sets may under-cover).
- **Tamper-evidence without a key.** Certificates are tamper-evident (HMAC-SHA256) ONLY
  when `VERIFY_HMAC_KEY` is set; otherwise `authentication = NONE` (integrity hash only).
- Diagnosis of other rotating-machinery faults (gears, shafts, couplings, lubrication)
- Acoustic emission or thermal sensor inputs
- Sampling rates below 10 kHz
- Fault localization (ball vs inner-race vs outer-race classification — see Limitations)
- Safety-critical decisions without human review (the model predicts probabilities, not certainties)

## Training data

### Source dataset

**FEMTO** (PRONOSTIA platform) — public benchmark for bearing prognostics.

- **Bearings**: 11 rolling-element bearings run to failure under three operating conditions
- **Sensors**: Two accelerometers per bearing (horizontal + vertical axes)
- **Sampling rate**: 25.6 kHz
- **Total samples (after segmentation)**: 13,959 vibration windows of 2,560 points each
- **Segmentation**: Sliding 0.1-second windows
- **Class assignment**: Temporal degradation percentage of bearing remaining life
  - Normal (class 0): 0–20%
  - Early (class 1): 20–50%
  - Medium (class 2): 50–80%
  - Advanced (class 3): 80–100%

### Training split

- Globally stratified split: 60% train (8,375), 20% validation (2,792), 20% test (2,792)
- Class balance after stratification: approximately equal across splits
- Note: original design called for purely temporal split with 5% gap; the released training run used global stratification (see training log) — this is a documented divergence from the design spec

### Preprocessing

- Per-signal z-score normalization (zero mean, unit std per channel per sample)
- No artificial augmentation in the training pipeline used to produce the released checkpoint
- Optional augmentations for fine-tuning runs: Gaussian noise (σ=0.01), amplitude scaling (0.9–1.1), time-warp, mixup

## Evaluation data

### In-distribution: FEMTO test split

- 2,792 samples held out from training distribution
- Same operating conditions as training

### Cross-domain: MFPT bearing dataset

- **Source**: Machinery Fault Prevention Technology, public benchmark
- **Resampled** from native 97.6 kHz to 25.6 kHz to match training distribution
- **94 samples** in evaluation set
- **Zero-shot**: Model evaluated without seeing any MFPT samples in training
- **Few-shot**: Model evaluated after 10 labeled MFPT samples per class (40 total) used for classifier fine-tuning, encoder frozen

## Verified performance

| Benchmark | F1 macro | Accuracy | Std (over runs) | Source file |
|---|---|---|---|---|
| FEMTO validation | 0.898 | 89.5% | n/a (single split) | `final_results.json` |
| FEMTO test (**stratified-random split**) | 0.884 | 88.1% | n/a (single split) | `cross_validation_results.json` |
| **Honest LOBO (held-out bearing)** — v6 | **0.352** | — | **0.112** (LOBO folds) | see `PERFORMANCE_BENCHMARKS.md` |
| MFPT zero-shot | 0.615 | 79.8% | n/a (single eval) | `cross_validation_results.json` |
| MFPT few-shot (10 samples/class) | 0.672 | 87.3% | 0.006 (3 runs) | `mfpt_sensitivity_results.json` |

These numbers were extracted from result-log JSON files generated by the original training and evaluation runs. Cross-source agreement confirms the numbers are not transcription artifacts.

> **Read the 0.884 honestly.** FEMTO test 0.884 is a **stratified-random split** — windows from
> every bearing appear in both train and test, so it measures **in-distribution** performance,
> not generalization to a new bearing. The honest generalization signal is the **LOBO row:
> v6 = 0.352 ± 0.112** when a whole bearing is held out. **A clean v1 LOBO is still pending**
> (only a weaker per-bearing breakdown of the existing v1 checkpoint exists — 0.9218 ± 0.0426 —
> which is NOT LOBO because that checkpoint saw samples from every bearing during training). Do
> not present 0.884 as evidence the model generalizes to an unseen bearing.

> **MFPT zero-shot 0.615 — reproducibility caveat (2026-06-04)**: this number is from the
> October 2025 evaluation logs and is **not currently reproducible from the shipped code**:
> the current MFPT loader yields F1 = 0.5546 on n = 224 windows vs the logged 0.615 on n = 94;
> the original windowing/selection recipe was lost. Documented in
> `AION_NEXUS_RD/experiments/substrate_F1/PREREG_DEVIATION_2026-06-04.md`. Treat 0.615 as a
> logged historical result, not a currently verifiable one.

### Per-class breakdown (FEMTO test)

Regenerated 2026-05-30 from the v1 checkpoint on the reproduced 2,792-sample test split
(aggregate macro-F1 = 0.8843, matches the published 0.884). Source: `results/per_class_f1.json`;
reproduce with `python -m scripts.verify_per_class`. (The previous hand-entered table was
arithmetically inconsistent — e.g. F1 below both precision and recall — and has been replaced.)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — Normal | 0.953 | 0.904 | 0.928 | 560 |
| 1 — Early | 0.872 | 0.902 | 0.887 | 836 |
| 2 — Medium | 0.851 | 0.846 | 0.848 | 836 |
| 3 — Advanced | 0.871 | 0.879 | 0.875 | 560 |

Confusion matrix is concentrated on the diagonal. Most residual error is between adjacent severity classes (Early ↔ Medium), which is expected given that the underlying degradation is continuous and class boundaries are necessarily fuzzy.

## Comparison to state of the art

> **Provenance caveat**: the prior-method numbers below are **literature-reported, not
> independently reproduced** by this project; citations needed. Settings (datasets, splits,
> preprocessing) may not be exactly comparable. Treat this table as indicative, not as a
> verified head-to-head benchmark.

| Method | Year | Approach | Zero-shot F1 (MFPT) | Few-shot F1 (10 samples) |
|---|---|---|---|---|
| Transfer Learning fine-tune | 2020 | full fine-tune | n/a | 0.52 |
| DANN (adversarial) | 2021 | domain-adversarial | 0.52 | 0.61 |
| Prototypical Networks | 2022 | metric learning | 0.54 | 0.61 |
| MAML | 2022 | meta-learning | 0.58 | 0.61 |
| Deep CORAL | 2023 | feature correlation | 0.51 | n/a |
| **AION-NEXUS** | **2025** | **multi-scale + temporal + attention** | **0.615** | **0.672** |

Subject to the provenance caveat above, AION-NEXUS reports the best zero-shot F1 on MFPT and the best 10-sample few-shot F1 among the listed methods, with no adversarial training, no meta-training, and no target-domain data exposure during pre-training.

## v1 vs v6 — IMPORTANT comparability disclosure (2026-04-27)

This package supports two architectures: v1 (BiGRU, default) and v6 (TempAttn+TRM, opt-in).

The two were trained on DIFFERENT FEMTO subsets:

- **v1 (1,061,724 params, F1=0.884)** trained on FEMTO Test_set/Test_set = 11 run-to-failure bearings (industrial deployment regime).
- **v6 (716,577 params, F1=0.934)** trained on FEMTO Test_set/Training_set/Learning_set = 6 short-run calibration bearings.

These F1 values are NOT directly comparable. Independent verification on 2026-04-27 showed:

- v1 on its own test set: F1 = 0.884 (verified, delta 0.0003 vs published)
- v6 on its own test set: F1 = 0.934 (verified, delta 0.0000 vs published)
- v6 cross-domain on 11 run-to-failure bearings: F1 = 0.302 (cross-bearing transfer collapse)

### Cross-evaluation matrix (2×2 verified 2026-04-27)

|  | Eval on Learning_set (calibration) | Eval on Test_set (run-to-failure) |
|---|---|---|
| Train on Test_set RTF (v1) | F1 0.382 (cross) | F1 0.884 (in-dist) |
| Train on Learning_set (v6) | F1 0.934 (in-dist) | F1 0.302 (cross) |

**Both architectures fail symmetrically on cross-bearing transfer.** Neither generalizes to the other's regime. The architecture choice matters less than the train/test distribution match.

### Deployment recommendation by regime

| Customer scenario | Bearings | Cycle | Recommended architecture | Expected F1 |
|---|---|---|---|---|
| Industrial PdM (typical) | run-to-failure type | long, progressive | **v1** | 0.88 |
| OEM testing / quality control | calibration type | short, cyclic | **v6** | 0.93 |
| Unknown / new machine type | any | any | **few-shot adaptation** with 10 samples/class | 0.67+ |

**Key insight for sales/positioning**: the production wedge is the **few-shot adaptation pipeline**, not architecture choice. Cross-regime deployment without few-shot will collapse F1 below 0.4 for either architecture. Use few-shot whenever the deployment domain doesn't exactly match a training regime.

**For short-cycle calibration scenarios** matching v6 training: `InferenceEngine.from_checkpoint(ckpt, version="v6")`.
**For run-to-failure long-cycle scenarios**: `InferenceEngine.from_checkpoint(ckpt, version="v1")` (default).
**For anything else**: load v1 or v6 (whichever is closer to your deployment), then `FewShotAdapter` with 10 labeled samples per class.

## Limitations

1. **Class imbalance on advanced faults across domains**. On MFPT zero-shot, advanced-class recall is 60% vs 100% on normal/medium. Operating decisions on advanced predictions should be reviewed by a domain expert.

2. **Task semantic mismatch on CWRU**. The Case Western Reserve University dataset uses fault-size labels (0.007", 0.014", 0.021"). Mapping these to FEMTO severity classes is physically invalid (Spearman correlation of fault size to vibration severity = −0.30). On CWRU, the model performs at F1 ≈ 0.34 (location) / 0.36 (severity). Documented in `docs/task_mismatch.md`.

3. **Sampling rate sensitivity**. The model was trained at 25.6 kHz. Inputs sampled below 10 kHz miss the high-frequency receptive field of the short-term CNN branch. Resample target signals to 25.6 kHz before inference.

4. **Self-supervised pretraining harms cross-domain**. SimCLR contrastive pretraining on FEMTO unlabeled data caused MFPT zero-shot F1 to drop from 0.615 to 0.184 (−70%). The released checkpoint does NOT include SimCLR pretraining. If you fine-tune, do not use single-domain contrastive pretraining.

5. **AdaBN harms under MFPT class imbalance**. MFPT has zero "early" samples and 50% "advanced". Adapting BatchNorm statistics to MFPT unlabeled data drops F1 from 0.615 to 0.488. The released model uses FEMTO BN statistics.

## Ethical considerations

- **Decision criticality**: A misclassified "advanced" bearing fault can lead to machine failure with safety implications (e.g., catastrophic vibration in heavy machinery). Always pair the model's prediction with domain-expert review for critical machines.
- **Continuous monitoring vs spot diagnosis**: The model is designed for spot diagnosis on a 0.1-second window. Robust deployment integrates predictions across consecutive windows (majority-vote, calibrated probabilities, hysteresis).
- **Bias in training population**: Training data is from one industrial bearing fault simulator (PRONOSTIA). Performance on different bearing geometries, lubrication regimes, or operating temperatures may degrade. Validate before deployment on a new machine class.
- **Sensor calibration drift**: The model assumes calibrated accelerometers. Sensor failure or drift may produce confidently-wrong predictions. Pair with sensor health monitoring.

## How the released checkpoint was produced

- Single training run, October 1–2 2025, ~17.6 hours on CPU
- Optimizer: Adam (lr=5e-4, weight_decay=1e-4)
- Batch size: 8
- Best epoch: 17 (selected by validation F1)
- Loss: cross-entropy
- Early stopping patience: 15 epochs
- Hardware: x86_64 CPU
- Random seed: documented in `aion_nexus_training.log`

## Versioning

- v1.0.0 — initial release with verified F1 numbers above
- Future versions will be tracked in `CHANGELOG.md`
- Backward-compatibility policy: any breaking change to the input contract bumps the major version

## Reporting issues

For performance regressions, unexpected behavior, or safety concerns, open an issue in the project repository or contact `daniel.culotta@gmail.com`. Include: input signal sample, model version, expected vs actual prediction, and operating context.
