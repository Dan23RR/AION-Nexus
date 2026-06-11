# Checkpoints

This directory holds trained model weights. **Do not commit `.pth` or `.onnx` files to version control** (`.gitignore` already excludes them).

## v1.0 release checkpoint

**File expected here**: `aion_nexus_v1.pth`
**SHA-256**: `6c1859a52d3d4f82253d4073e773eac7df97dce14e37cd94a024349c816e53be`
**Size**: ~4.1 MB (FP32)
**Trained on**: FEMTO PRONOSTIA bearing dataset
**Verified F1**:
- FEMTO validation: 0.898
- FEMTO test: 0.884 (globally-stratified split; LOBO honest estimate is lower)
- MFPT zero-shot: 0.615

The released checkpoint is `best_model.pth` from the source training repository
(Oct 2025 run). It is byte-identical to `aion_nexus_v1.pth` — same SHA-256.

## v6 architecture checkpoint (opt-in, NOT the production default)

**File**: `aion_nexus_v6.pth`
**SHA-256**: `0d24e93fde556ecf3a6f6464aa4c763dddad495fcb212f5d1de113bbf2a60f47`
**Size**: ~2.73 MB (FP32)
**Note**: F1=0.934 is on the FEMTO **Learning_set** calibration subset — a different
subset, NOT comparable to v1's test number. v6 collapses cross-bearing (0.302) and
under LOBO (0.352 ± 0.112, see `results/lobo_cv_v6/`). Use only for short-cycle
calibration regimes.

## v3 substrate checkpoint (self-supervised foundation encoder)

**File**: `aion_nexus_substrate_v3.pth`
**SHA-256**: `be0e5b146e5e0fa28d9dfb10bf3015c5f2d7ca217896f7866fd793454c5505b6`
**Size**: ~4.9 MB (FP32; pretrained encoder 1,220,928 params + `cfg`/`objective` metadata)
**Pretraining**: PatchTST-style patch transformer, self-supervised contrastive NT-Xent,
400 epochs on a pooled unlabeled vibration corpus (9,315 windows: FEMTO + MFPT + CWRU).
Promoted to production in v2.1.0 (2026-06-04).
**Verified (LOBO on FEMTO)**: 10-shot macro-F1 0.783 ± 0.041; full-transfer 0.533.
**Note**: this file contains the pretrained ENCODER only — the classification head is
trained per-deployment (few-shot). `InferenceEngine.from_checkpoint` refuses to serve it
directly (the head would be untrained); adapt it with `FewShotAdapter` and serve the
adapted checkpoint instead. Zero-shot cross-rig is NOT reliable (mean lift −0.03) —
collect ~10 labels/class on the target machine.

## Adapted checkpoints

Few-shot adaptation produces additional checkpoints (e.g., `aion_nexus_v1_machine42.pth`) which inherit from `aion_nexus_v1.pth`. Each adapted checkpoint should be tracked with:

- The base checkpoint hash it was adapted from
- The 10–50 samples used for adaptation (or a hash of them)
- The validation F1 on a held-out target set

## Format

Checkpoints are saved either as a raw state dict or as a dict with key `model_state_dict`. The `InferenceEngine.from_checkpoint` factory handles both.

## Loading

```python
from aion_nexus import InferenceEngine

# Loads with weights_only=True by default (blocks pickle-based RCE). Pass the
# registered SHA-256 to also verify integrity at load time:
engine = InferenceEngine.from_checkpoint(
    "checkpoints/aion_nexus_v1.pth",
    expected_sha256="6c1859a52d3d4f82253d4073e773eac7df97dce14e37cd94a024349c816e53be",
)
```

## ONNX export

For edge deployment, export to ONNX:

```bash
python -m scripts.export_onnx --checkpoint checkpoints/aion_nexus_v1.pth --dynamic-batch
```

This produces `checkpoints/aion_nexus.onnx` runnable on ONNX Runtime, OpenVINO, or any ONNX-compatible runtime.
