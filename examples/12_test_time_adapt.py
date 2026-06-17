"""Example 12: source-free test-time adaptation — close the loop, no labels.

A model trained on one machine loses accuracy on a new (covariate-shifted) machine.
Source-free TTA adapts it at install time using ONLY the unlabeled windows the new
machine already produces — no target labels, no source data. This shows the safe
default (AdaBN: re-estimate BatchNorm stats) recovering accuracy on a shifted
target, with no labels used.

HONESTY (6.31): AdaBN is BatchNorm-based and cannot collapse; TENT is more powerful
but can degenerate (guarded). TTA adapts the shift the stats capture — it is not a
cross-machine cure, and any conformal coverage must be re-certified after adapting.

Run:
    python examples/12_test_time_adapt.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from aion_nexus.adapt import recalibrate_batchnorm, tent_adapt

K, N = 3, 256
CLASS_MEAN = torch.tensor([[-2.0, 1.5], [1.8, -1.2], [0.2, 2.4]])


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.bn = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, K)

    def forward(self, x, rpm=None, geometry=None):
        return {"logits": self.fc2(torch.relu(self.bn(self.fc1(x.mean(dim=2)))))}


def _gen(n, rng, shift=0.0):
    y = rng.integers(0, K, size=n)
    x = torch.empty(n, 2, N)
    for i in range(n):
        x[i] = (CLASS_MEAN[y[i]] + shift)[:, None] + 0.3 * torch.from_numpy(
            rng.standard_normal((2, N)).astype("float32"))
    return x, torch.from_numpy(y)


def _acc(model, x, y):
    model.eval()
    with torch.no_grad():
        return float((model(x)["logits"].argmax(-1) == y).float().mean())


def main() -> int:
    rng = np.random.default_rng(0)
    model = TinyBN()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    xs, ys = _gen(600, rng)
    model.train()
    for _ in range(60):
        opt.zero_grad()
        nn.functional.cross_entropy(model(xs)["logits"], ys).backward()
        opt.step()
    model.eval()

    xt, yt = _gen(600, rng, shift=3.0)          # a new machine: covariate mean-shift
    before = _acc(model, xt, yt)
    print("--- A model trained on machine A, deployed to machine B ---")
    print(f"  source accuracy: {_acc(model, *_gen(400, rng)):.2f}")
    print(f"  target accuracy BEFORE adaptation: {before:.2f}  (the cross-machine drop)")

    adapted = recalibrate_batchnorm(model, xt)   # AdaBN on UNLABELED target windows
    after = _acc(adapted, xt, yt)
    print("\n--- AdaBN: re-estimate BatchNorm stats on unlabeled target windows ---")
    print(f"  target accuracy AFTER adaptation: {after:.2f}  (no target labels used)")

    _, res = tent_adapt(model, xt, steps=3)
    print("\n--- TENT (entropy-min, guarded) ---")
    print(f"  {res.detail}; collapsed={res.collapsed}")

    assert after > before + 0.1
    print("\nSource-free TTA recovered accuracy on a new machine with NO labels. Honest "
          "scope: AdaBN cannot collapse; TENT is guarded; re-certify conformal coverage "
          "after adapting (TTA gives plasticity, the certificate gives the guarantee).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
