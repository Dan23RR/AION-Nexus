# Ride a foundation encoder, don't out-pretrain one

`aion_nexus.foundation` (v2.14.0) wraps **any frozen encoder** that maps a vibration
window `[B, C, N] → [B, D]` into AION's certified pipeline (inference engine,
few-shot head, conformal verification, physics second opinion). The same adapter
rides UniFault, MOMENT, Mantis, Moirai, a customer's encoder, or AION's own.

## Why (the binding constraint)

The architecture-leap research found the wall to cross-machine bearing diagnosis is
**data diversity** (the number of distinct physical bearings seen in pretraining),
**not** model capacity (`AION_NEXUS_RD/20_ARCHITECTURE_LEAP.md`). A solo founder
cannot out-pretrain UniFault (>9B points, 10 datasets, MIT). So: **ride the encoder,
own the verification + adaptation layer.** Verify / adapt ANY model, don't compete.

## The adapter

```python
from aion_nexus.foundation import wrap_foundation_encoder
from aion_nexus.inference import InferenceEngine
from aion_nexus.few_shot import FewShotAdapter

adapter = wrap_foundation_encoder(
    encoder,            # frozen: a torch module or callable [B,C,N] -> [B,D]
    embed_dim=128,
    num_classes=4,
    input_length=1024,  # what the ENCODER expects (the adapter resamples to it)
    input_channels=2)
engine = InferenceEngine(adapter, architecture_version="ext")
FewShotAdapter(engine).adapt(signals, labels)   # trains the HEAD only; encoder stays frozen
```

The adapter conforms to the v1/v3/v6 forward contract (`{"logits","features"}`),
harmonises AION's `[2,2560]`@25.6 kHz window to the encoder's expected length /
channels (and an optional encoder-specific `normalize` hook), and keeps the frozen
encoder in eval (BatchNorm running stats never drift) even while the head trains.

## UniFault — the verified recipe (the primary target)

Facts confirmed from the repo + paper (arXiv:2504.01373; github.com/emadeldeen24/UniFault):

- **License: MIT** (plain, no academic-use restriction).
- **Variants**: only the **Tiny / Lite** checkpoint (`embed_dim=128`, 823K) is
  confirmably **downloadable** (a Dropbox link in the README; `pretrained_models/Tiny/pretrain-epoch=<id>.ckpt`,
  a PyTorch-Lightning checkpoint). The larger variants are not confirmably released.
- **Input contract is NOT AION's.** UniFault wants **univariate**, **1024 timesteps**
  (resample to a 0.1 s temporal resolution → standardise to 1024), **per-channel
  min-max** normalisation — **not** AION's z-score + 1 Hz high-pass. `patch_size=64` → 16 patches.
- **Embedding extraction**: `Transformer_bkbone(args).forward(x)` returns the
  representation `[B, C·num_patches, embed_dim]` (no classifier); **mean-pool over the
  token axis** → `[B, embed_dim]` (what its own `predict()` does before the head).

```python
import torch
# 1. build the backbone with the Tiny config, load the frozen pretrained weights
backbone = Transformer_bkbone(args)            # embed_dim=128, heads=4, depth=4, patch_size=64,
                                               #   seq_len=1024, num_channels=2, num_classes=4
ckpt = torch.load("pretrained_models/Tiny/pretrain-epoch=<id>.ckpt", map_location="cpu")
matched = {k: v for k, v in ckpt["state_dict"].items()
           if k in backbone.state_dict() and backbone.state_dict()[k].size() == v.size()}
backbone.load_state_dict(matched, strict=False)   # patch-embed + transformer load; head ignored
backbone.eval().requires_grad_(False)

# 2. wrap: pool to [B,128]; UniFault wants per-channel min-max (not z-score)
def unifault_encoder(x):                        # x: [B, 2, 1024]
    return backbone(x).mean(dim=1)              # -> [B, 128]

def minmax(x):
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)

from aion_nexus.foundation import ExternalEncoderAdapter
adapter = ExternalEncoderAdapter(unifault_encoder, embed_dim=128, num_classes=4,
                                 target_length=1024, normalize=minmax)
```

> ⚠️ **Verify before production.** The loading recipe was derived from a repo
> *summary*, not byte-exact source. Open `model/model.py` and confirm the exact
> `forward` return shape and pooling axis before shipping a real UniFault integration.
> Also note AION's `InferenceEngine.predict()` applies z-score + 1 Hz HP *first*; for
> a UniFault benchmark feed appropriately-preprocessed (raw → min-max) windows
> directly (e.g. via the eval harness), not the production z-score+HP path.

## 🔴 The UniFault leakage trap (read before quoting any number)

UniFault was **pretrained on FEMTO, MFPT, CWRU, XJTU-SY, IMS, PU, KAIST, HIT-SM, CNC**.
So a "leave-one-bearing-out" number computed on **FEMTO or MFPT bearings** using the
UniFault encoder is **transductive at the encoder level** — those bearings were in
the encoder's pretraining — which is *exactly* the flaw (v3's `0.783` leak) this was
meant to fix, one level up. A **clean inductive** LOBO with a foundation encoder
requires the held-out bearings to be absent from BOTH the few-shot split AND the
encoder's pretraining corpus. Most public bearing datasets are in UniFault's corpus,
so an honest benchmark needs a dataset it never pretrained on. UniFault's own paper
respects this ("the fine-tuning samples are excluded from the pretraining data") — so
must any number AION reports. Use `evaluate_leave_one_group_out` (v2.13.0) and state
the encoder's pretraining corpus alongside the result.

## Alternative frozen encoders (the same adapter wraps them)

| Encoder | Input | Notes |
|---|---|---|
| **MOMENT** (AutonLab) | length 512, univariate, RevIN internal | HF `AutonLab/MOMENT-1-{small,base,large}`; general TS, weak on raw vibration — a cheap baseline. |
| **Mantis** (Paris-Noah/Huawei) | length 512 | Purpose-built for TS **classification**; HF `paris-noah/Mantis-8M`, `pip mantis-tsfm`; calibrated — pairs with AION's conformal UQ. |
| **Moirai** (Salesforce) | any-variate, patch 64, configurable context | Forecast-centric; verification target more than diagnostic core. |

For each: set `input_length` to the encoder's expected length and supply its
normalisation via `normalize`. None of these is vibration-pretrained the way UniFault
is — they are baselines and verification targets, not the diagnostic core.

*Code: `aion_nexus/foundation.py`, `tests/test_foundation.py`, `examples/11_foundation_encoder.py`.*
