# Architecture Reference

Layer-by-layer specification of AION-NEXUS v1.0. **This document is the source of truth for the model contract.** Any divergence from `aion_nexus/model.py` is a bug in one of the two.

## Pipeline overview

```
Raw signal [B, 2, 2560]
        │
        ▼
Multi-Scale Temporal CNN  (3 parallel branches)
        │
        ▼   [B, 192, 640]
Channel Attention (SE-style, avg + max pool)
        │
        ▼   [B, 192, 640]
Bidirectional GRU (hidden=128, 2 layers, dropout=0.2)
        │
        ▼   [B, 256, 640]  →  avg + max pool
        │
        ▼   [B, 512]
Classification Head (3-layer MLP with BN + Dropout)
        │
        ▼
Logits [B, num_classes]   +   Features [B, 512]
```

## Component-by-component

### 1. MultiScaleTemporalCNN

Three parallel 1D-CNN branches operating on the 2-channel raw input.

**Short-term branch** — captures high-frequency outer-race fault signatures.

| Layer | Spec | Output |
|---|---|---|
| Conv1d | 2→32, k=3, p=1 | [B, 32, 2560] |
| BN + ReLU | | |
| Conv1d | 32→32, k=7, p=3 | [B, 32, 2560] |
| BN + ReLU | | |
| MaxPool1d(4) | stride=4 | [B, 32, 640] |
| Conv1d | 32→64, k=3, p=1 | [B, 64, 640] |
| BN + ReLU | | |

**Medium-term branch** — captures mid-frequency inner-race fault signatures.

| Layer | Spec | Output |
|---|---|---|
| Conv1d | 2→32, k=15, p=7 | [B, 32, 2560] |
| BN + ReLU | | |
| MaxPool1d(4) | stride=4 | [B, 32, 640] |
| Conv1d | 32→64, k=31, p=15 | [B, 64, 640] |
| BN + ReLU | | |

**Long-term branch** — captures low-frequency cage / rolling-element signatures.

| Layer | Spec | Output |
|---|---|---|
| Conv1d | 2→32, k=63, p=31 | [B, 32, 2560] |
| BN + ReLU | | |
| MaxPool1d(4) | stride=4 | [B, 32, 640] |
| Conv1d | 32→64, k=127, p=63 | [B, 64, 640] |
| BN + ReLU | | |

Output of MultiScaleCNN: concat along channel dim → **[B, 192, 640]**.

### 2. AttentionFusion (SE-style channel attention)

Squeeze-and-Excitation channel attention with concurrent avg + max pooling.

| Layer | Spec |
|---|---|
| AdaptiveAvgPool1d(1) | [B, 192, 1] → squeeze → [B, 192] |
| AdaptiveMaxPool1d(1) | [B, 192, 1] → squeeze → [B, 192] |
| Linear(192 → 24) + ReLU | reduction = 8 |
| Linear(24 → 192) + Sigmoid | gate per channel |
| Combine | `attn = avg_attn + max_attn` |
| Apply | `out = x * attn.unsqueeze(-1)` |

Output: [B, 192, 640] (re-weighted along channel dim).

### 3. TemporalEncoder (bidirectional GRU + dual pooling)

| Layer | Spec | Output |
|---|---|---|
| Transpose | [B, 192, 640] → [B, 640, 192] | |
| GRU | hidden=128, layers=2, bidirectional, dropout=0.2 | [B, 640, 256] |
| Transpose | back → [B, 256, 640] | |
| AdaptiveAvgPool1d(1) | [B, 256, 1] → [B, 256] | avg |
| AdaptiveMaxPool1d(1) | [B, 256, 1] → [B, 256] | max |
| Concat | [B, 512] | output |

The 512-dim feature vector is exposed via `engine.extract_features(signal)` for embedding / clustering / few-shot use cases.

### 4. ClassificationHead

3-layer MLP with BatchNorm and graded dropout.

| Layer | Spec | Output |
|---|---|---|
| Linear | 512 → 256 | |
| BatchNorm1d + ReLU | | |
| Dropout | p=0.3 | |
| Linear | 256 → 128 | |
| BatchNorm1d + ReLU | | |
| Dropout | p=0.201 (= 0.3 * 0.67) | |
| Linear | 128 → num_classes (4) | logits |

**No softmax inside the head** — the engine applies softmax at inference time; the head returns raw logits.

## Parameter inventory

```
MultiScaleTemporalCNN   ~62K
  short_conv             6,624
  medium_conv           42,016
  long_conv            114,016  (long kernels dominate)
AttentionFusion          9,432
TemporalEncoder        623,616  (BiGRU dominates total)
ClassificationHead     165,956
                       ───────
Total              1,061,724  (4.1 MB FP32)
```

The architecture's runtime cost is dominated by the BiGRU (~60% of params, ~70% of forward latency on CPU).

## Receptive field analysis

After the multi-scale CNN with MaxPool(4), each output timestep covers:

| Branch | Effective receptive field (samples) | Time @25.6 kHz |
|---|---|---|
| Short | ~33 | 1.3 ms |
| Medium | ~196 | 7.7 ms |
| Long | ~764 | 29.8 ms |

The bidirectional GRU then extends context across all 640 timesteps in both directions, giving a global receptive field equal to the input length (100 ms).

## Why the design works (ablation-grounded)

Every component drops cross-domain F1 if removed (see PERFORMANCE_BENCHMARKS.md):

| Removal | FEMTO ΔF1 | MFPT ΔF1 |
|---|---|---|
| Multi-scale (single kernel) | −13% | **−20%** |
| Bidirectional GRU | −9% | **−13%** |
| Channel attention | −5% | **−5%** |
| All (single CNN only) | −16% | **−28%** |

The multi-scale CNN is the largest contributor to cross-domain generalization — scale-invariance is what allows the model to transfer from FEMTO bearings to MFPT bearings with different fault-frequency profiles.

## Frozen contract

Architectural changes that BREAK checkpoint compatibility (require retraining):

- Number of CNN branches, channels, kernels, pooling factors
- GRU hidden dim, num_layers, bidirectionality, dropout
- ClassificationHead layer dims
- BN / Dropout placement

Architectural changes that PRESERVE checkpoint compatibility (refactor only):

- Code organization (split into more files, rename variables)
- Type annotations
- Logging / docstrings
- `forward` return-dict key naming (if the engine adapts)

When in doubt, run `pytest tests/test_smoke.py::test_model_param_count` — it asserts exactly 1,061,724 parameters.
