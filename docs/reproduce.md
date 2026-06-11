# Reproducing the published numbers

Minimal, honest guide to reproducing the headline F1 = 0.884 (FEMTO test, v1 checkpoint) and
what can — and cannot — currently be reproduced from this package.

## What you need

1. **Checkpoint**: `checkpoints/aion_nexus_v1.pth` (see `checkpoints/README.md` for provenance
   and SHA-256).
2. **FEMTO data**: the PRONOSTIA `Test_set/Test_set` subset (11 run-to-failure bearings,
   13,959 windows after segmentation).
3. **The original training-time loader** (preferred, "Path A"): the source repo containing
   `aion_data.py` (`AIONDataset` + `create_stratified_temporal_splits`, seed = 42). This is
   required to regenerate the *exact* 2,792-sample test split on which 0.884 was published.

## Reproduce FEMTO test F1 = 0.884

```bash
python -m scripts.verify_checkpoint \
    --checkpoint checkpoints/aion_nexus_v1.pth \
    --aion-data-repo /path/to/source-repo \
    --femto-root "data/FEMTO+Bearing/10. FEMTO Bearing/FEMTOBearingDataSet/Test_set/Test_set" \
    --out results/verification.json
```

- Expected: **F1 = 0.8843** (published 0.884; verified 2026-04-27 with delta 0.0003).
- Tolerance: the script flags a DEVIATION if |F1 − 0.884| > 0.005.
- Without the original repo (`--skip-original` / fallback "Path B" directory scan) the split
  is **not** bit-equivalent to the published test set; treat the result as a sanity check, not
  a reproduction.

A no-data static check (param count + collapse detection on synthetic input) runs first via
`diagnose_checkpoint`; the v1 parameter count must be exactly **1,061,724**.

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
