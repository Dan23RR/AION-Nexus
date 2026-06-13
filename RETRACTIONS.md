# Retractions & Corrections

This project publishes the claims it has retracted. Self-correction is part of the
methodology, not an embarrassment hidden in a commit message. If you are doing due
diligence, this file is the short version of "what did they once claim that turned out
to be wrong, and how did they find out."

The detailed, dated technical record is in [`CHANGELOG.md`](CHANGELOG.md) (the §6.31
correction, v2.0.0) and [`MODEL_CARD.md`](MODEL_CARD.md). This page is the index.

## Retracted claims (do not cite these — they are wrong)

| Retracted claim | Reality | How it was caught |
|---|---|---|
| "CWRU cross-domain F1 = 0.9495 / 94.5%" | It was few-shot **in-domain** on CWRU, not cross-domain zero-shot — a measurement-construct mislabel. | Independent re-evaluation, Oct 2025 – Apr 2026 (§6.31). |
| "v6 cross-domain validated, F1 = 0.934" | v6 reaches 0.934 **in-distribution** but **collapses to 0.302 cross-bearing**. The two numbers must always travel together. | Symmetric cross-evaluation matrix v1↔v6. |
| "ECG cross-domain 74.4% F1" | Single best epoch, no replicate — not a validated result. | Replication attempt failed. |
| "Context is 7.3× more predictive than AI" | Manipulated framing; withdrawn. | Internal review. |
| "Patent pending" | **Zero patents have been filed.** This phrase should never have been published. | Self-audit, 2026. |
| "EU AI Act compliant" | No formal conformity assessment was ever performed. Withdrawn. | Self-audit, 2026. |
| Inflated ROI figures (e.g. "ROI 67–83%", "ROI 194%") | Not measured. The project is pre-revenue with zero pilots; any ROI figure is a model, not a result. | Self-audit, 2026. |

## What this means for the numbers we *do* publish

Every headline number in this repository is reproducible from the shipped checkpoints,
or carries an explicit caveat where it is not:

- **FEMTO test F1 = 0.884** — real and independently reproducible (see [`docs/reproduce.md`](docs/reproduce.md)), but on a **globally-stratified split** (same bearing in train and test). The honest leave-one-bearing-out (LOBO) number is lower; for v6 it is **0.352 ± 0.112**. A true LOBO run of the shipped v1 checkpoint is still pending.
- **MFPT zero-shot F1 = 0.615** — carries a **provenance caveat**: it is not reproducible from the current loader (which yields 0.555); the original Oct-2025 recipe was lost. Do not cite 0.615 without this caveat.
- **Substrate v3 LOBO 10-shot ≈ 0.783** — see the honesty note in the model card: the SSL encoder was pretrained on a corpus that included the held-out bearings (unlabeled), so this figure is **transductive, not a clean nested-LOBO** result.

## On the labels (the most important caveat)

The 4-class FEMTO labels in AION-1 are derived **positionally** — `degradation_pct =
file_index / (total_files − 1)`, quantized into four bins — not from independent fault-type
analysis. This system estimates **how far through a run-to-failure sequence a signal is**
(a degradation-stage / RUL proxy), which is **not the same as diagnosing the fault type**.
We are explicit about this so no one mistakes the task for fault-type classification.

---

*If you find a claim in this repository that contradicts this file, the claim is the bug —
open an issue. Last updated: 2026-06-12.*
