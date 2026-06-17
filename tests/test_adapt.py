"""Tests for source-free test-time adaptation (aion_nexus.adapt).

The headline test proves the value: a model trained on a SOURCE machine loses
accuracy on a covariate-shifted TARGET, and AdaBN (re-estimating BatchNorm running
stats on UNLABELED target windows) recovers it — no target labels used. Plus: TENT
reduces prediction entropy, the collapse guard fires on a degenerate target, and a
BatchNorm-free model is returned unchanged with no false adaptation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aion_nexus.adapt import (
    prediction_entropy,
    recalibrate_batchnorm,
    source_free_adapt,
    tent_adapt,
)

K = 3
N = 256


class _TinyBN(nn.Module):
    """time-pool -> Linear -> BatchNorm -> ReLU -> Linear. The BN running stats are
    fitted on the source machine, so a target mean-shift mis-normalises until AdaBN."""

    def __init__(self, k=K):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, k)

    def forward(self, x, rpm=None, geometry=None):
        h = x.mean(dim=2)                       # [B, 2] pooled feature
        h = torch.relu(self.bn(self.fc1(h)))
        return {"logits": self.fc2(h)}


_CLASS_MEAN = torch.tensor([[-2.0, 1.5], [1.8, -1.2], [0.2, 2.4]])   # 3 well-separated classes


def _gen(n, rng, shift=0.0):
    """Signals whose TIME-MEAN encodes the class (+ an optional covariate shift)."""
    y = rng.integers(0, K, size=n)
    x = torch.empty(n, 2, N)
    for i in range(n):
        base = _CLASS_MEAN[y[i]] + shift
        x[i] = base[:, None] + 0.3 * torch.from_numpy(rng.standard_normal((2, N)).astype("float32"))
    return x, torch.from_numpy(y)


def _accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)["logits"].argmax(dim=-1)
    return float((pred == y).float().mean())


def _train_on_source(seed=0, epochs=60):
    rng = np.random.default_rng(seed)
    x, y = _gen(600, rng, shift=0.0)
    model = _TinyBN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x)["logits"], y)
        loss.backward()
        opt.step()
    model.eval()
    return model, rng


# --------------------------------------------------------------------------- #
# 1. THE headline: AdaBN recovers target accuracy with NO labels
# --------------------------------------------------------------------------- #

def test_adabn_recovers_target_accuracy_without_labels():
    model, rng = _train_on_source()
    # Source accuracy is high (sanity).
    xs, ys = _gen(400, rng, shift=0.0)
    assert _accuracy(model, xs, ys) > 0.9

    # Target = same classes, a covariate mean-shift -> the source BN mis-normalises.
    xt, yt = _gen(600, rng, shift=3.0)
    acc_before = _accuracy(model, xt, yt)

    # AdaBN on UNLABELED target windows (xt only — labels yt are NOT used).
    adapted = recalibrate_batchnorm(model, xt)
    acc_after = _accuracy(adapted, xt, yt)

    assert acc_before < 0.75, f"target should be degraded pre-adaptation: {acc_before:.2f}"
    assert acc_after > acc_before + 0.1, \
        f"AdaBN must recover accuracy label-free: {acc_before:.2f} -> {acc_after:.2f}"
    # The source model is never mutated.
    assert _accuracy(model, xt, yt) == acc_before


# --------------------------------------------------------------------------- #
# 2. TENT reduces entropy; collapse guard; affine-only
# --------------------------------------------------------------------------- #

def test_tent_reduces_entropy_on_balanced_target():
    model, rng = _train_on_source()
    xt, _ = _gen(400, rng, shift=2.0)
    adapted, res = tent_adapt(model, xt, lr=1e-3, steps=3)
    assert res.method == "tent" and res.n_bn_layers == 1
    assert res.post_entropy <= res.pre_entropy + 1e-6      # entropy did not increase
    assert not res.collapsed                                # balanced target -> no collapse


def test_tent_collapse_guard_fires_on_degenerate_target():
    model, rng = _train_on_source()
    # A single-state target (all class 0) drives TENT to one class -> guard fires.
    xt = torch.stack([( _CLASS_MEAN[0] + 2.0)[:, None]
                      + 0.3 * torch.from_numpy(rng.standard_normal((2, N)).astype("float32"))
                      for _ in range(200)])
    _, res = tent_adapt(model, xt, lr=5e-3, steps=5)
    assert res.collapsed is True
    assert res.warning is not None and "AdaBN" in res.warning


# --------------------------------------------------------------------------- #
# 3. Honest no-op on a BatchNorm-free model
# --------------------------------------------------------------------------- #

class _NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(2)
        self.fc = nn.Linear(2, K)

    def forward(self, x, rpm=None, geometry=None):
        return {"logits": self.fc(self.ln(x.mean(dim=2)))}


def test_no_batchnorm_returns_unchanged():
    model = _NoBN()
    x = torch.randn(20, 2, N)
    before = model(x)["logits"].detach().clone()
    adapted = recalibrate_batchnorm(model, x)
    assert torch.allclose(adapted(x)["logits"], before)    # unchanged: no false adaptation
    _, res = tent_adapt(model, x)
    assert res.n_bn_layers == 0 and "no-op" in res.detail


# --------------------------------------------------------------------------- #
# 4. Helpers / convenience
# --------------------------------------------------------------------------- #

def test_prediction_entropy_bounds():
    uniform = torch.zeros(4, K)                            # uniform softmax -> max entropy ln K
    assert abs(float(prediction_entropy(uniform)) - np.log(K)) < 1e-5
    peaked = torch.tensor([[10.0, 0.0, 0.0]])              # near-one-hot -> ~0 entropy
    assert float(prediction_entropy(peaked)) < 0.01


def test_source_free_adapt_dispatch():
    model, rng = _train_on_source(epochs=20)
    xt, _ = _gen(120, rng, shift=2.0)
    assert isinstance(source_free_adapt(model, xt, method="adabn"), nn.Module)
    out = source_free_adapt(model, xt, method="tent", steps=1)
    assert isinstance(out, tuple) and out[1].method == "tent"
    try:
        source_free_adapt(model, xt, method="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
