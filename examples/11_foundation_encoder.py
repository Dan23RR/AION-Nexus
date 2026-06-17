"""Example 11: ride a frozen foundation encoder + few-shot head, end-to-end.

The cross-machine wall is a DATA-DIVERSITY problem, not a capacity one — so a solo
team rides a foundation encoder (UniFault, >9B points, MIT) instead of out-pretraining
one, and owns the verification + adaptation layer. This shows the flow with a synthetic
stand-in encoder (no weights to download):

    1. Wrap a FROZEN encoder + a few-shot head (the encoder stays frozen).
    2. Adapt the HEAD to a new machine with a handful of labels (encoder untouched).
    3. Serve through the certified inference engine.

To use the REAL UniFault encoder, swap the synthetic encoder for the loaded backbone
(see docs/FOUNDATION_ENCODERS.md): input_length=1024, embed_dim=128, normalize=min-max,
and READ the leakage-trap note — UniFault pretrained on FEMTO/MFPT, so a clean inductive
LOBO needs a dataset it never saw.

Run:
    python examples/11_foundation_encoder.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aion_nexus.few_shot import FewShotAdapter
from aion_nexus.foundation import wrap_foundation_encoder
from aion_nexus.inference import InferenceEngine

K = 4


class _SynthFoundationEncoder(nn.Module):
    """Stand-in for UniFault/MOMENT/Mantis: [B, C, 1024] -> [B, 128]."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(2, 32, 7, padding=3), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, embed_dim))

    def forward(self, x):
        return self.net(x)


def _minmax(x):
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    return (x - lo) / (hi - lo + 1e-8)


def main() -> int:
    # 1. Wrap a frozen encoder; AION's [2,2560] is harmonised to the encoder's 1024.
    adapter = wrap_foundation_encoder(
        _SynthFoundationEncoder(128), embed_dim=128, num_classes=K,
        input_length=1024, input_channels=2)
    adapter.normalize = _minmax                    # encoder-specific norm (UniFault = min-max)
    print("--- 1. Wrapped a frozen foundation encoder ---")
    print(f"  total params {adapter.get_num_params():,}, trainable (head only) "
          f"{adapter.get_trainable_params():,}  -> we ride the encoder, adapt the head")

    engine = InferenceEngine(adapter, architecture_version="ext")
    rng = np.random.default_rng(0)
    enc_before = next(adapter.encoder.parameters()).detach().clone()

    # 2. Few-shot adapt the HEAD to a new machine (10 windows/class), encoder frozen.
    sigs = [rng.standard_normal((2, 2560)).astype("float32") for _ in range(40)]
    labels = [i % K for i in range(40)]
    fs = FewShotAdapter(engine)
    out = fs.adapt(sigs, labels, epochs=5, verbose=False)
    enc_after = next(fs.model.encoder.parameters()).detach()
    print("\n--- 2. Few-shot adapted the head ---")
    print(f"  head loss {out['epoch_losses'][0]:.3f} -> {out['final_loss']:.3f}; "
          f"encoder unchanged = {torch.allclose(enc_before, enc_after)}")

    # 3. Serve the adapted model through the certified engine.
    result = fs.to_engine().predict(rng.standard_normal((2, 2560)).astype("float32"))
    print("\n--- 3. Certified serving ---")
    print(f"  predicted '{result.predicted_class_name}' conf {result.confidence:.2f}; "
          f"probs sum {sum(result.probabilities.values()):.3f}")

    assert torch.allclose(enc_before, enc_after)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
    print("\nThe same adapter wraps UniFault/MOMENT/Mantis or a customer's model. Honest "
          "scope: it adds no pretraining diversity of its own, and a clean inductive LOBO "
          "needs the encoder to NOT have seen the held-out bearings (docs/FOUNDATION_ENCODERS.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
