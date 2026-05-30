# Checkpoints

This directory holds trained model weights. **Do not commit `.pth` or `.onnx` files to version control** (`.gitignore` already excludes them).

## v1.0 release checkpoint

**File expected here**: `aion_nexus_v1.pth`
**SHA-256**: (run `sha256sum aion_nexus_v1.pth` after placing the file and record the hash here)
**Size**: ~4.1 MB (FP32)
**Trained on**: FEMTO PRONOSTIA bearing dataset
**Verified F1**:
- FEMTO validation: 0.898
- FEMTO test: 0.884
- MFPT zero-shot: 0.615

The released checkpoint is `results/nexus_results/best_model.pth` from the source training repository (commit hash to be added on release).

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
engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")
```

## ONNX export

For edge deployment, export to ONNX:

```bash
python -m scripts.export_onnx --checkpoint checkpoints/aion_nexus_v1.pth --dynamic-batch
```

This produces `checkpoints/aion_nexus.onnx` runnable on ONNX Runtime, OpenVINO, or any ONNX-compatible runtime.
