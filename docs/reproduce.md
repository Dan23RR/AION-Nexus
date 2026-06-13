# Reproducing the published numbers

Minimal, honest guide to reproducing the headline F1 = 0.884 (FEMTO test, v1 checkpoint) and
what can — and cannot — currently be reproduced from this package.

## TL;DR — what runs from a clean public-repo clone vs what needs external data

| Step | Clean clone of the public repo? | What it proves |
|---|---|---|
| `pytest tests/` (smoke + unit suite) | ✅ yes, no dataset | Checkpoint loads; architecture = 1,061,724 params; no class collapse; determinism |
| `python examples/01_basic_inference.py examples/sample_signal.csv` | ✅ yes, no dataset | Inference path works on a real bundled FEMTO window; produces a life-stage prediction |
| **Reproduce F1 = 0.884 (Path A)** | ❌ **no** — needs the external FEMTO dataset **and** the training-time loader `aion_data.py` | The exact published test-set F1 |

**The public repo does NOT ship the FEMTO dataset** (license + size). So **0.884 is not "out of
the box"** from a clone — it requires you to bring the FEMTO data and the source loader. The smoke
suite and the single-window demo are what a reviewer can run in ~5 minutes with zero external data.

> **What the model predicts (read this first).** The 4 classes are a **life-stage / severity
> index** (0 = normal … 3 = advanced), derived from each window's *position* in a run-to-failure
> sequence (`degradation_pct = file_idx / (total − 1)`, binned into 4). It is **RUL/severity-stage
> estimation, not fault-*type* diagnosis**. "Fault diagnosis" elsewhere is a misnomer for this task.

## What you need to reproduce F1 = 0.884 (Path A)

1. **Checkpoint**: `checkpoints/aion_nexus_v1.pth` (see `checkpoints/README.md` for provenance
   and SHA-256). Shipped in this package.
2. **FEMTO data** (external, NOT in the public repo): the PRONOSTIA `Test_set/Test_set` subset
   (11 run-to-failure bearings) plus a `metadata.json` describing the bearings.
3. **The original training-time loader** (preferred, "Path A"): the source repo containing
   `aion_data.py` (`AIONDataset` + `create_stratified_temporal_splits`, seed = 42). This is
   required to regenerate the *exact* test split on which 0.884 was published.

## Reproduce FEMTO test F1 = 0.884

```bash
python -m scripts.verify_checkpoint \
    --checkpoint checkpoints/aion_nexus_v1.pth \
    --aion-data-repo /path/to/source-repo \
    --femto-root "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set" \
    --metadata data/FEMTO+Bearing/metadata.json \
    --out results/verification.json
```

- Expected: **F1 ≈ 0.884** (published; verified 2026-04-27 with delta 0.0003).
- Tolerance: the script flags a **DEVIATION** (exit code 1) if |F1 − 0.884| > **0.01**
  (the tolerance encoded in `scripts/verify_checkpoint.py`).
- **A PASS means something was actually verified.** If no dataset is provided, or only
  `.mat`/unlabelled data is found, the script prints **"NO VERIFICATION PERFORMED" / "NO SAMPLES
  EVALUATED"** and **exits non-zero (2 or 3) — never "PASSED"**. See "Honest exit codes" below.
- Without the original repo + `metadata.json`, Path A is **skipped** (not silently faked). A
  best-effort native FEMTO scan (`--femto <root>`) or `_classN.csv` directory scan (Path B) is
  **not** bit-equivalent to the published test set; treat it as a load/sanity check.

A no-data static check (param count + collapse detection on synthetic input) runs first via
`diagnose_checkpoint`; the v1 parameter count must be exactly **1,061,724**.

## Honest exit codes (`scripts/verify_checkpoint.py`)

| Exit | Meaning |
|---|---|
| `0` | **VERIFICATION PASSED** — ≥1 published number was checked against real data and matched within tolerance |
| `1` | **VERIFICATION FAILED** — a published number deviated beyond tolerance |
| `2` | **NO SAMPLES EVALUATED** — nothing loaded; nothing verified (provide a real dataset) |
| `3` | **NO VERIFICATION PERFORMED** — data loaded but none comparable to a published number (e.g. MFPT `.mat`, whose fault-type labels don't map to the severity classes) |

The previous version of this script could print "VERIFICATION PASSED" with **zero** samples
evaluated (e.g. pointing `--mfpt` at a directory of `.mat` files, which the old `.csv` scanner
silently skipped). That false-pass is fixed: a PASS now requires a real comparison.

## MFPT note (`--mfpt`)

The MFPT distribution ships `.mat` files. The verifier now **reads them** (`scipy.io.loadmat`,
`bearing.gs` single channel duplicated to 2 channels) and runs the model on them as a **load
check** — it reports the prediction distribution and proves the data flows end-to-end. It emits
**no F1 and no PASS**, because MFPT filenames encode **fault type** (baseline / inner-race /
outer-race), which does **not** map to the model's severity classes. The historical MFPT
zero-shot F1 = 0.615 used a separate, now-lost windowing recipe — see below.

## Caveats — read before quoting 0.884

- **Split caveat (mandatory)**: 0.884 is measured on a **globally stratified** split. The
  model saw windows from every one of the 11 bearings during training. It is *not* a
  leave-one-bearing-out (LOBO) number; an honest LOBO estimate is expected to be lower. True
  LOBO retraining for v1 is scheduled, not yet run.
- A weaker proxy exists: per-bearing evaluation of the existing checkpoint gives mean
  F1 = 0.9218 ± 0.0426 over the 11 bearings (`scripts/per_bearing_f1_breakdown.py`,
  `results/per_bearing_f1.json`) — uniform across bearings, but still not LOBO.

## What is currently NOT reproducible

- **MFPT zero-shot F1 = 0.615**: logged in October 2025 on n = 94 windows. The current MFPT
  loader yields F1 = 0.5546 on n = 224 windows; the original windowing/selection recipe was
  lost. Documented in
  `AION_NEXUS_RD/experiments/substrate_F1/PREREG_DEVIATION_2026-06-04.md`. Treat 0.615 as a
  logged historical result.
- **MFPT few-shot 10/class = 0.672 ± 0.006** (3 runs) remains the verified few-shot
  reference; rerun via `examples/03_few_shot_adaptation.py` with your MFPT copy.

## Environment

- Python ≥ 3.10, `pip install -r requirements.txt` (scipy is required — the HP-Butterworth
  filter is part of the preprocessing contract; without it F1 collapses to ≈ 0.28).
- CPU is sufficient; results are deterministic in eval mode (see `tests/test_determinism.py`).

If you reproduce these numbers and observe a deviation greater than ±0.005 F1, open an issue
with your environment and dataset checksums.
