# Performance Benchmarks — AION-NEXUS v2.2 (updated 2026-06-11)

> **Scope of every F1 below.** The task is **degradation-stage / RUL estimation**, not fault-type
> diagnosis: the 4 labels are a positional life-stage proxy (`degradation_pct = file_idx /
> (total−1)`, 4 bins), not independently diagnosed fault types. Read "severity"/"class"
> accordingly. Two split regimes appear below and are **not interchangeable**: **stratified-random**
> (in-distribution, e.g. FEMTO test 0.884) vs **held-out-bearing / LOBO** (generalization, e.g.
> v6 0.352 ± 0.112). Retracted/corrected claims: [RETRACTIONS.md](RETRACTIONS.md). Reproduction
> steps and known-non-reproducible numbers: [docs/reproduce.md](docs/reproduce.md).

Core v1 numbers verified against 4 independent JSON result-log files in the source repository (`final_results.json`, `cross_validation_results.json`, `mfpt_sensitivity_results.json`, `aion_nexus_training.log`). Cross-source agreement confirms numbers are not transcription artifacts. Tables that are estimates or literature-reported (not backed by artifacts in `results/`) are explicitly labeled as such.

## In-distribution: FEMTO bearing dataset

| Architecture | Train data | Test data | n samples test | F1 macro | Status (2026-04-27) |
|---|---|---|---|---|---|
| **v1 (BiGRU)** | Test_set/Test_set (11 RTF bearings) | held-out same | 2,792 | **0.884** | VERIFIED delta=0.0003 |
| **v1 (BiGRU)** | Test_set/Test_set (11 RTF bearings) | validation | 2,792 | **0.898** | source: `nexus_results/final_results.json` |
| **v6 (TempAttn+TRM)** | Learning_set (6 calib bearings) | held-out same | 1,507 | **0.934** | VERIFIED delta=0.0000 |
| **v6 (TempAttn+TRM)** | Learning_set (6 calib bearings) | **cross-domain** Test_set RTF | 2,792 | **0.302** | VERIFIED |

**Critical reading note** (added after 2026-04-27 cross-domain verification):

The v1 F1=0.884 and v6 F1=0.934 headline numbers are **NOT directly comparable**: they are measured on DIFFERENT FEMTO subsets.

- **v1 train+test = Test_set/Test_set** = 11 run-to-failure bearings (Bearing1_3 ... Bearing3_3). This is the industrial deployment regime: long degradation cycles, progressive failure.
- **v6 train+test = Test_set/Training_set/Learning_set** = 6 short-run calibration bearings (Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2). FEMTO PRONOSTIA "calibration" set, designed for parameter tuning, not run-to-failure prediction.

Cross-evaluation of v6 (trained on Learning_set) on the 11 Test_set bearings drops F1 from 0.934 to **0.302** — a 0.633 collapse caused by domain shift between calibration bearings and run-to-failure bearings.

For industrial deployment on long run-to-failure data, **v1 is the recommended architecture**: F1=0.884 on RTF bearings, while v6 generalizes poorly. v6 may still be appropriate for short-cycle calibration scenarios where its training distribution matches.

**Independent verification (2026-04-27, all reproduced delta < 0.001 vs published)**:
- v1 in-distribution: `verify_checkpoint --checkpoint v1.pth --femto-root .../Test_set/Test_set` → **F1 = 0.8843**
- v1 cross-domain: `verify_checkpoint --checkpoint v1.pth --femto-root .../Learning_set` → **F1 = 0.3819**
- v6 in-distribution: `verify_checkpoint --checkpoint v6.pth --femto-root .../Learning_set` → **F1 = 0.9343**
- v6 cross-domain: `verify_checkpoint --checkpoint v6.pth --femto-root .../Test_set/Test_set` → **F1 = 0.3017**

### Cross-evaluation matrix (full 2×2 — SYMMETRIC FAILURE)

|  | Eval on Learning_set (calibration, 1507 samples) | Eval on Test_set (run-to-failure, 2792 samples) |
|---|---|---|
| Train on Test_set RTF (v1) | **0.3819** ✗ (cross) | **0.8843** ✓ (in-dist) |
| Train on Learning_set (v6) | **0.9343** ✓ (in-dist) | **0.3017** ✗ (cross) |

**Both models fail catastrophically when tested on the OTHER FEMTO subset.** v1 drops 0.502 F1 going RTF → Learning. v6 drops 0.632 F1 going Learning → RTF. The architecture difference (BiGRU vs TempAttn+TRM) is dwarfed by the train-test distribution shift.

**Methodological implication**: there is no "best architecture" in the v1-vs-v6 comparison; there is a "best architecture per regime". Cross-regime deployment requires few-shot adaptation, not architecture switching alone.

### Per-bearing F1 breakdown of v1 checkpoint (2026-05-25)

To address the question "is the F1=0.884 stratified-random number inflated by bearing-identity leakage?", we evaluated the v1 production checkpoint independently on each of the 11 run-to-failure bearings in Test_set:

| Bearing | n samples | F1 macro | Accuracy |
|---|---:|---:|---:|
| Bearing1_3 | 1,802 | 0.9623 | 0.963 |
| Bearing1_4 | 1,139 | 0.9315 | 0.928 |
| Bearing1_5 | 2,302 | 0.9407 | 0.939 |
| Bearing1_6 | 2,302 | 0.9390 | 0.935 |
| Bearing1_7 | 1,502 | 0.9615 | 0.959 |
| Bearing2_3 | 1,202 | 0.9286 | 0.932 |
| Bearing2_4 | 612 | 0.8917 | 0.887 |
| Bearing2_5 | 2,002 | 0.9641 | 0.964 |
| Bearing2_6 | 572 | 0.8785 | 0.872 |
| Bearing2_7 | 172 | 0.8242 | 0.820 |
| Bearing3_3 | 352 | 0.9173 | 0.909 |
| **Mean ± Std** | — | **0.9218 ± 0.0426** | — |
| **Range** (max − min) | — | **0.1400** (0.8242 → 0.9641) | — |

**Interpretation**: low cross-bearing variance (std 0.043, range 0.14) suggests the v1 checkpoint produces **uniform predictions across the 11 run-to-failure bearings**, NOT concentrated on a subset. This is consistent with the headline F1=0.884 NOT being heavily inflated by bearing-identity leakage from the stratified-random split.

**Caveat (full honesty)**: this is NOT a true leave-one-bearing-out (LOBO) measurement. True LOBO would require retraining the model 11 times, holding out one bearing each time, to measure generalization to a completely unseen bearing. This is **scheduled for the next iteration**. The number above is the per-bearing F1 breakdown of the EXISTING checkpoint, which has seen some samples from each bearing during training. It is a cheaper but weaker proxy. The honest cross-bearing generalization signal remains the **MFPT zero-shot F1=0.615** number below — MFPT is a different bearing dataset, never seen during training.

**Reproduce**:
```bash
python -m scripts.per_bearing_f1_breakdown \
    --checkpoint checkpoints/aion_nexus_v1.pth \
    --femto-root "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet" \
    --bearing-subset "Test_set/Test_set" \
    --out-dir results
```

Full JSON output: `results/per_bearing_f1.json`. Markdown report: `results/per_bearing_f1.md`.

### Per-class F1 in cross-domain failure (both models share the same failure pattern)

- v1 cross-domain (RTF→Learning): [0.62, 0.12, 0.39, 0.40] — collapses class 1 ("Early")
- v6 cross-domain (Learning→RTF): [0.62, 0.17, 0.04, 0.38] — collapses classes 1+2 ("Early/Medium")

Both models predict extremes (Normal class 0 + Advanced class 3) reliably across regimes; both lose intermediate severity classification when the test distribution differs from training.

### v6 architectural properties (vs v1)

| Property | v1 | v6 | Δ |
|---|---|---|---|
| Test F1 macro (on its OWN test set) | 0.884 | 0.934 | NOT comparable — different test sets (see Critical reading note above); the apparent "+5.0 pp" is NOT a valid head-to-head |
| Parameters | 1,061,724 | 716,577 | **−32.5%** |
| Disk size (FP32) | 4.1 MB | 2.73 MB | −33% |
| CPU inference (per sample) | ~12 ms | ~16 ms | +33% (still real-time) |
| GPU inference (per sample) | ~1.5 ms | ~1.5 ms | parity |
| Noise-robust SNR+5dB F1 | not measured | ≈0.87 | new capability |
| Architecture | MultiScaleCNN + ChannelAttn + **BiGRU** + 3-layer MLP | MultiScaleCNN + ChannelAttn + **TemporalSelfAttention (4-head MHA)** + **TinyRecursiveReasoner** | refactored aggregation + reasoning

### Per-class FEMTO test breakdown

Regenerated 2026-05-30 from the v1 checkpoint on the reproduced 2,792-sample test split
(aggregate macro-F1 = 0.8843, matches published 0.884). Source `results/per_class_f1.json`;
reproduce with `python -m scripts.verify_per_class`. Replaces an earlier hand-entered table
that was arithmetically inconsistent (F1 below both P and R for class 0).

| Class | Support | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|
| 0 — Normal | 560 | 0.953 | 0.904 | 0.928 | Highest precision |
| 1 — Early | 836 | 0.872 | 0.902 | 0.887 | Boundary fuzzy with Medium |
| 2 — Medium | 836 | 0.851 | 0.846 | 0.848 | Boundary fuzzy with Early |
| 3 — Advanced | 560 | 0.871 | 0.879 | 0.875 | Adjacent-class confusion dominates error |

Diagonal-dominated confusion matrix: most residual error is Early↔Medium misclassification, which is physically expected since these are arbitrary cut-points on a continuous degradation process.

## Cross-domain: MFPT bearing dataset (zero-shot)

No MFPT samples in training. Resampled from 97.6 kHz to 25.6 kHz to match training-distribution sampling rate.

| Metric | Value | Source |
|---|---|---|
| F1 macro | **0.615** | `cross_validation_results.json` |
| Accuracy | 79.8% | `cross_validation_results.json` |
| Normal recall | 100% (24/24) | `cross_validation_results.json` |
| Medium recall | 100% (23/23) | `cross_validation_results.json` |
| Advanced recall | 60% (28/47) | `cross_validation_results.json` |

> **Reproducibility caveat (2026-06-04)**: the 0.615 zero-shot number comes from the
> October 2025 evaluation logs and is **not currently reproducible from the shipped code**:
> the current MFPT loader yields F1 = 0.5546 on n = 224 windows vs the logged 0.615 on
> n = 94; the original windowing/selection recipe was lost. Documented in
> `AION_NEXUS_RD/experiments/substrate_F1/PREREG_DEVIATION_2026-06-04.md`. Treat 0.615 as a
> logged historical result, not a currently verifiable one.

**Cross-domain gap from FEMTO test (0.884) to MFPT zero-shot (0.615): −26.9 percentage points.**

## Cross-domain: MFPT few-shot adaptation

10 labeled MFPT samples per class (40 total). Encoder frozen; classifier head fine-tuned for 5 epochs at lr=1e-4.

| Run | F1 macro | Notes |
|---|---|---|
| Run 1 | 0.6729 | seed varied |
| Run 2 | 0.6795 | seed varied |
| Run 3 | 0.6641 | seed varied |
| **Mean ± std** | **0.6722 ± 0.006** | mean of 3 runs |

**Few-shot lift over zero-shot: +5.7 percentage points F1 with 40 labeled samples.**

### Sample efficiency curve

> **Provenance note**: the intermediate points (1–5 samples/class) come from the October 2025
> sensitivity analysis and have **not been independently re-verified**. The 10-samples/class
> point is reconciled to the verified value **0.672 ± 0.006** (3 runs, table above); an
> earlier revision of this table showed an unexplained 0.702 at this point. The dollar
> "run cost" figures ($5 … $15,000) and the "224× cheaper than full retraining" ratio shown
> in earlier revisions were **unmeasured estimates** and have been removed.

| Samples/class | Total samples | F1 macro | Std | Provenance |
|---|---|---|---|---|
| 0 (zero-shot) | 0 | 0.615 | n/a | logged Oct-2025 — see reproducibility caveat above |
| 1 | 4 | 0.543 | 0.082 | sensitivity analysis, not independently verified |
| 2 | 8 | 0.621 | 0.047 | sensitivity analysis, not independently verified |
| 3 | 12 | 0.658 | 0.031 | sensitivity analysis, not independently verified |
| 5 | 20 | 0.681 | 0.024 | sensitivity analysis, not independently verified |
| 10 | 40 | **0.672** | 0.006 | **verified** (3 runs, few-shot table above) |
| Full retrain | 13,959 | 0.898 | n/a | FEMTO validation F1 — different dataset/regime, not an MFPT number |

**Recommended operating point**: 10 samples per class — the only point on this curve with a
verified, low-variance value (0.672 ± 0.006).

## Substrate v3 (few-shot cross-domain) — added 2.1.0

Self-supervised **PatchTST** foundation encoder (1,220,928 params), pretrained contrastively
(NT-Xent) on an unlabeled FEMTO+MFPT+CWRU corpus (9,315 windows, 400 epochs). Frozen encoder +
small per-deployment few-shot head. Promoted to production in 2.1.0 (2026-06-04).

| Benchmark (FEMTO held-out bearing) | F1 macro | Status |
|---|---|---|
| 10-shot (10 labels/class on the held-out bearing) — **TRANSDUCTIVE, not clean LOBO** ³ | **0.783 ± 0.041** | verified (transductive) |
| Full-transfer (no target labels) | 0.533 | verified |

³ **SSL-leakage caveat (new, critical).** The 0.783 number is **transductive, not
leave-one-bearing-out.** The frozen PatchTST encoder is contrastively pretrained on
`cache_corpus_full.npz`, a corpus that **includes the held-out bearings (Bearing1_5 / 2_5 / 3_3)
as unlabeled windows** (~664 windows/bearing for all 11 RTF bearings;
`AION_NEXUS_RD/aion2_verified_substrate/foundation/substrate_corpus.py` line ~58 iterates every
RTF bearing dir with no held-out exclusion). The encoder has therefore *seen* the held-out
bearings' signals; only their labels were withheld at head-training time. A clean **nested-LOBO
SSL** (re-pretrain the encoder excluding the held-out bearing) has **not** been run. Until it is,
**do not call 0.783 "LOBO" without the qualifier "transductive"**, and do not present it as
clean leave-one-bearing-out generalization.

**Honest positioning (§6.31):**

- v3 is **NOT** a better in-distribution classifier — v1's FEMTO test F1 = 0.884 stands.
- Cross-**dataset** 10-shot macro-F1 0.91–1.00 (FEMTO↔MFPT↔CWRU, all pairs; lift +0.277 vs
  random-init) — **binary health (2-class) task, vs a weak random-init control**; not a
  4-class severity result. **Triviality caveat (new):** this binary cross-rig health task is
  **matched by a non-ML threshold on kurtosis** calibrated on the same 10 samples (≈ 0.911 mean
  macro-F1; 1.000 on FEMTO→CWRU). So 0.91–1.00 does **NOT** demonstrate the substrate's value —
  the substrate's value has to be shown on the **4-class severity** task, not on binary health.
- Zero-shot cross-rig is **NOT reliable** (mean lift −0.03 vs random-init) — collect the
  ~10 labels/class.
- Scaling did not move the 10-shot ceiling (≈ 0.78, stable): **v3.1** (3.2M params, hybrid
  NT-Xent⊕MAE) reached 10-shot 0.801 ± 0.036 = +0.018, below the pre-registered +0.03
  threshold — retracted, NOT promoted (`RETRACTION_substrate_v31.md`). **v3.2** (+Paderborn
  data, IMS = 0 windows) 10-shot 0.778 ± 0.024 (flat), full-transfer 0.565 (best) — NOT
  promoted.

## Ablation study: component contribution

Each row removes ONE component from full NEXUS, holds everything else constant.

| Configuration | FEMTO F1 | MFPT zero-shot F1 | ΔFEMTO | ΔMFPT |
|---|---|---|---|---|
| Full NEXUS | 0.898 | 0.615 | reference | reference |
| − Multi-scale CNN (single kernel) | 0.778 | 0.492 | −13.4% | **−20.0%** |
| − Bidirectional GRU | 0.821 | 0.537 | −8.6% | −12.7% |
| − Channel attention | 0.852 | 0.581 | −5.1% | −5.5% |
| Single-scale CNN only | 0.756 | 0.445 | −15.8% | −27.6% |

**Key finding: multi-scale architecture is the largest contributor to cross-domain generalization (−20 pp when removed).** Scale-invariance is critical for transferring across bearings with different fault-frequency profiles.

## Comparison to state of the art

> **Provenance caveat**: the prior-method numbers below are **literature-reported, not
> independently reproduced** by this project; citations needed. Settings (datasets, splits,
> preprocessing) may not be exactly comparable. Treat this table as indicative, not as a
> verified head-to-head benchmark.

| Method | Year | Zero-shot F1 (MFPT) | Few-shot 10-sample F1 | Approach |
|---|---|---|---|---|
| Transfer Learning fine-tune | 2020 | n/a | 0.52 | Full fine-tune |
| DANN (adversarial domain) | 2021 | 0.52 | 0.61 | Adversarial alignment |
| Prototypical Networks | 2022 | 0.54 | 0.61 | Metric learning |
| MAML | 2022 | 0.58 | 0.61 | Meta-learning |
| Deep CORAL | 2023 | 0.51 | n/a | Feature correlation |
| **AION-NEXUS** | **2025** | **0.615** | **0.672** | **Multi-scale + temporal + attention** |

Subject to the provenance caveat above, AION-NEXUS reports the best number in both zero-shot and 10-sample few-shot configurations on MFPT among the listed methods, and does so without adversarial training, meta-training, or target-domain pre-training.

## Inference performance

> **Estimated, not benchmarked in `results/`**: the figures below are engineering estimates —
> no latency artifact in `results/` backs them. Measure on your target hardware with
> `python -m scripts.benchmark_inference` before relying on them.

Single forward pass on a 2,560-sample 2-channel signal.

| Hardware | Latency (per sample) | Throughput (batch=1) | Throughput (batch=32) | Memory |
|---|---|---|---|---|
| Intel Xeon (1 thread) | 12 ms | 83 samples/s | 230 samples/s | 180 MB |
| Intel Xeon (8 threads) | 4 ms | 250 samples/s | 580 samples/s | 200 MB |
| NVIDIA T4 GPU | 1.5 ms | 670 samples/s | 12,000 samples/s | 800 MB |
| ARM Cortex-A72 (Pi 4) | 45 ms | 22 samples/s | 65 samples/s | 180 MB |

ONNX export reduces latency by approximately 30% via constant folding and operator fusion (also an estimate — verify with `scripts/verify_onnx_parity.py` and your own timing).

## Negative results documented

These experiments did NOT produce improvements and are documented to inform future work and prevent re-running of dead-end approaches.

| Experiment | Hypothesis | Outcome | Drop |
|---|---|---|---|
| SimCLR contrastive pretraining on FEMTO | "Self-supervised features generalize better cross-domain" | Catastrophic cross-domain regression | MFPT F1: 0.615 → 0.184 (−70%) |
| Adaptive BatchNorm with MFPT unlabeled | "Re-estimate BN stats on target distribution" | Class imbalance corrupts BN | MFPT F1: 0.615 → 0.488 (−21%) |
| CWRU severity-mapped task | "FEMTO severity labels transfer to CWRU fault sizes" | Spearman correlation of mappings = −0.30 | CWRU F1: 0.34 |

These negative results are documented in `docs/negative_results.md`.

## Reproducibility

- **Checkpoint**: `checkpoints/aion_nexus_v1.pth` (4.1 MB, SHA-256 verified)
- **Architecture code**: `aion_nexus/model.py` exactly matches the training-time architecture
- **Verification script**: `scripts/verify_checkpoint.py` re-runs evaluation on FEMTO test and MFPT zero-shot
- **Run instructions**: see `docs/reproduce.md`

If you reproduce these numbers and observe a deviation greater than ±0.005 F1, please open an issue with your environment and dataset checksums.
