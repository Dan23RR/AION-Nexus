# LOBO Cross-Validation — AION-NEXUS v6 (measured result, published)

> **This is our own model's worst number, measured and published on purpose.** It is the substance behind the §6.31 honesty hallmark: we falsify our own claims under the methodologically-correct protocol instead of hiding behind a favorable split.

## What this is

A **6-fold leave-one-bearing-out (LOBO)** cross-validation of the AION-NEXUS **v6** architecture (TempAttn + TinyRecursiveReasoner, 716,577 params) on the FEMTO PRONOSTIA bearing benchmark. Each fold holds out one bearing entirely from training and tests on it — the correct protocol for measuring *cross-bearing* generalization, as opposed to the globally-stratified random split that leaks bearing identity into both train and test.

## Headline result

| Protocol | F1 macro | Notes |
|---|---:|---|
| Stratified-random (original) | **0.9343** | Standard but methodologically optimistic for per-bearing benchmarks |
| **Leave-one-bearing-out (LOBO)** | **0.3523 ± 0.1116** | 6-fold, the honest cross-bearing estimate |
| **Collapse** | **−0.582 absolute** | The model learns bearing identity, not fault physics, under stratified split |

Per-fold F1 macro: 0.250, 0.486, 0.368, 0.323, 0.193, 0.494 (range 0.19–0.49, high variance is itself diagnostic of poor cross-bearing transfer).

- **Date run**: 2026-02-11
- **Compute**: ~39 h CPU total (141,189 s across 6 folds)
- **Training protocol per fold**: 3-stage progressive (5+20+15 epochs), identical to the original v6 training.

## Why this matters (and how to read it honestly)

1. **This is the v6 architecture, not the shipped v1 checkpoint.** The production-recommended model for run-to-failure deployment is **v1** (`aion_nexus_v1.pth`). The **true LOBO of the shipped v1 checkpoint is still scheduled** (11× retraining on the run-to-failure bearings, ~30–60 h CPU). The per-bearing F1 *breakdown* of the existing v1 checkpoint (mean 0.922 ± 0.043, see `../per_bearing_f1.md`) is a cheaper, weaker proxy that suggests v1's 0.884 stratified number is not heavily inflated — but it is **not** a true LOBO and must not be quoted as one.

2. **The defensible cross-domain numbers are the MFPT ones** (zero-shot F1 0.615, few-shot 10/class 0.672). MFPT is an entirely different bearing dataset never seen in training — LOBO-equivalent in practice. Those, not 0.884, are the wedge.

3. **The PHM 2026 Oslo paper §5.3** reports this exact result as *"the same multi-scale CNN architecture ... F1 dropped to 0.35 ± 0.11 under LOBO"* and cites `culotta2026bearing`. It is our result, not a third-party citation. Any commercial material that attributes this collapse to "the literature" or "similar architectures" is **wrong** and must be corrected (single source of truth: `_v2_post_correction/NARRATIVE.md` §5).

## Files

| File | Content |
|---|---|
| `lobo_cv_results.json` | Full per-fold results: F1 macro/weighted, accuracy, confusion matrices, train/val/test bearings, timing. |
| `lobo_cv.log` | Run log of the 6-fold validation. |
| `lobo_cv_validation.py` | The validation script that produced the result (provenance; paths are from the source R&D workspace). |

## Provenance

Copied verbatim 2026-05-30 from the R&D workspace `DefinitiveAION/clean/lobo_cv_results/` into the production package so the evidence has a public home alongside the claim. Source files unchanged.
