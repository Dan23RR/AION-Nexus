"""Tests for the external foundation-encoder adapter (aion_nexus.foundation).

Verifies the adapter rides a FROZEN encoder + few-shot head correctly, harmonises
AION's input to the encoder, keeps the frozen encoder (incl. BatchNorm) untouched
during few-shot adaptation, and drops into the existing InferenceEngine + few-shot
path. A synthetic encoder stands in for UniFault/MOMENT/etc. (no weights needed).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from aion_nexus.foundation import ExternalEncoderAdapter, wrap_foundation_encoder

K = 4


class _SynthEncoder(nn.Module):
    """A stand-in foundation encoder [B, C, N] -> [B, D], with BatchNorm so we can
    prove a frozen encoder's running stats never drift during head adaptation."""

    def __init__(self, embed_dim=16, in_ch=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.proj = nn.Linear(in_ch * 4, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        h = self.pool(x).flatten(1)          # [B, C*4]
        return self.bn(self.proj(h))         # [B, D]


def _adapter(embed_dim=16, in_ch=2, **kw):
    return ExternalEncoderAdapter(_SynthEncoder(embed_dim, in_ch), embed_dim,
                                  num_classes=K, **kw)


# --------------------------------------------------------------------------- #
# 1. Forward contract + freezing
# --------------------------------------------------------------------------- #

def test_forward_contract_matches_v1_v3():
    a = _adapter()
    out = a(torch.randn(5, 2, 2560))
    assert set(out) == {"logits", "features"}
    assert out["logits"].shape == (5, K)
    assert out["features"].shape == (5, 16)


def test_encoder_is_frozen_head_is_trainable():
    a = _adapter()
    enc_trainable = sum(p.requires_grad for p in a.encoder.parameters())
    head_trainable = sum(p.requires_grad for p in a.head.parameters())
    assert enc_trainable == 0                      # encoder frozen
    assert head_trainable == 2                      # head weight + bias
    assert a.get_trainable_params() == sum(p.numel() for p in a.head.parameters())


def test_frozen_encoder_stays_eval_even_in_train_mode():
    a = _adapter()
    a.train()                                       # put the adapter in train mode
    assert a.encoder.training is False              # ...but the frozen encoder stays eval
    assert a.head.training is True


# --------------------------------------------------------------------------- #
# 2. Input harmonisation
# --------------------------------------------------------------------------- #

def test_harmonise_resamples_length():
    a = _adapter(target_length=512)
    # encoder expects whatever; the adapter resamples 2560 -> 512 before the encoder.
    out = a(torch.randn(3, 2, 2560))
    assert out["logits"].shape == (3, K)


def test_harmonise_reduces_channels_to_one():
    a = _adapter(in_ch=1, target_channels=1, channel_reduce="mean")
    out = a(torch.randn(3, 2, 2560))                # 2-channel input -> reduced to 1
    assert out["features"].shape == (3, 16)


def test_harmonise_rejects_channel_expansion():
    a = _adapter(in_ch=2, target_channels=4)
    with pytest.raises(ValueError, match="fabricate"):
        a(torch.randn(2, 2, 2560))


def test_normalize_hook_is_applied():
    seen = {}

    def minmax(x):                                  # UniFault-style per-channel min-max
        seen["called"] = True
        lo = x.amin(dim=-1, keepdim=True)
        hi = x.amax(dim=-1, keepdim=True)
        return (x - lo) / (hi - lo + 1e-8)

    a = _adapter(target_length=512, normalize=minmax)
    out = a(torch.randn(3, 2, 2560) * 10 + 5)       # off-range input
    assert seen.get("called") is True               # the encoder-specific norm ran
    assert out["logits"].shape == (3, K)


def test_bad_embed_dim_is_caught():
    enc = _SynthEncoder(embed_dim=16)
    a = ExternalEncoderAdapter(enc, embed_dim=32, num_classes=K)   # wrong embed_dim
    with pytest.raises(ValueError, match="encoder must return"):
        a(torch.randn(2, 2, 2560))


# --------------------------------------------------------------------------- #
# 3. Few-shot adapts ONLY the head; the frozen encoder is untouched
# --------------------------------------------------------------------------- #

def test_few_shot_adapts_head_only_and_leaves_encoder_intact():
    from aion_nexus.few_shot import FewShotAdapter
    from aion_nexus.inference import InferenceEngine

    a = _adapter()
    engine = InferenceEngine(a, architecture_version="ext")
    bn_mean_before = a.encoder.bn.running_mean.clone()
    enc_w_before = a.encoder.proj.weight.detach().clone()
    head_w_before = a.head.weight.detach().clone()

    adapter = FewShotAdapter(engine)               # selects head.* via "ext"
    sigs = [np.random.default_rng(i).standard_normal((2, 2560)).astype("float32") for i in range(8)]
    labels = [i % K for i in range(8)]
    adapter.adapt(sigs, labels, epochs=3, verbose=False)

    adapted = adapter.model
    # The head moved; the frozen encoder weights AND its BatchNorm running stats did not.
    assert not torch.allclose(adapted.head.weight, head_w_before)
    assert torch.allclose(adapted.encoder.proj.weight, enc_w_before)
    assert torch.allclose(adapted.encoder.bn.running_mean, bn_mean_before)


# --------------------------------------------------------------------------- #
# 4. Integrates with the inference engine (certified-serving-ready)
# --------------------------------------------------------------------------- #

def test_integrates_with_inference_engine():
    from aion_nexus.inference import InferenceEngine

    engine = InferenceEngine(_adapter(), architecture_version="ext")
    sig = np.random.default_rng(0).standard_normal((2, 2560)).astype("float32")
    result = engine.predict(sig)
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
    assert result.predicted_class_name in result.probabilities


def test_wrap_foundation_encoder_factory():
    a = wrap_foundation_encoder(_SynthEncoder(24), embed_dim=24, num_classes=K,
                                input_length=512, input_channels=2)
    assert isinstance(a, ExternalEncoderAdapter)
    assert a(torch.randn(2, 2, 2560))["logits"].shape == (2, K)
    # The encoder is frozen by the factory (ride it, adapt the head).
    assert a.get_trainable_params() == sum(p.numel() for p in a.head.parameters())


def test_callable_encoder_also_works():
    # The encoder may be a plain callable, not only an nn.Module.
    def enc(x):                                     # [B, C, N] -> [B, 8]
        return x.mean(dim=2).repeat(1, 4)[:, :8]
    a = ExternalEncoderAdapter(enc, embed_dim=8, num_classes=K, freeze=True)
    assert a(torch.randn(3, 2, 2560))["logits"].shape == (3, K)
