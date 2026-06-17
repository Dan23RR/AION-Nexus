"""Source-free test-time adaptation — adapt ANY model to a new machine, label-free.

The architecture-leap roadmap's #4. A model trained on one set of bearings sees a
shifted input distribution on a new machine (operating conditions, sensor mount,
transmission path). Source-free TTA closes part of that gap at INSTALL time using
ONLY the abundant unlabeled data every running machine produces — no target labels,
no access to the source training data. It is model-agnostic: it adapts AION's model,
a customer's, or a wrapped foundation encoder.

Two methods, safest first
-------------------------
- :func:`recalibrate_batchnorm` (**AdaBN**, the robust default). Re-estimate the
  BatchNorm running mean/var on unlabeled target windows. It cannot collapse (no
  optimisation, no labels), and it fixes the most common shift: a model whose BN
  statistics, fitted on the source machine, mis-normalise the target activations.
- :func:`tent_adapt` (**TENT**, Wang et al. ICLR 2021). Minimise the prediction
  entropy on unlabeled target windows by updating ONLY the BN affine parameters
  (γ, β). More powerful, but it CAN collapse to a degenerate single-class predictor
  under class imbalance or heavy shift — so it ships with a collapse guard that
  aborts and recommends AdaBN, and it is never the silent default.

HONESTY (workspace 6.31)
------------------------
- These are BatchNorm-based. A model with no BatchNorm (e.g. the LayerNorm/transformer
  v3 substrate, or a frozen foundation encoder) has no running stats to re-estimate —
  the functions detect that and return the model unchanged with a warning, never a
  silent no-op pretending to adapt.
- TTA adapts to the shift the BN stats / entropy CAPTURE; it cannot repair a
  representation that fundamentally does not transfer. The ceiling is the model.
- It assumes the unlabeled target window pool is predominantly NORMAL/healthy at
  install (the usual case) — adapting BN stats to a pool dominated by an unusual
  fault would shift them wrongly. State the assumption; the caller owns it.
- Adaptation changes the model, so any conformal coverage must be RE-CERTIFIED after
  adaptation (recalibrate the conformal layer on a post-adaptation split). TTA gives
  plasticity; the certificate gives the guarantee — keep both honest.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


@dataclass
class TTAResult:
    """Outcome of a source-free adaptation run."""

    method: str                    # "adabn" | "tent"
    n_bn_layers: int
    n_target_windows: int
    pre_entropy: float | None      # mean prediction entropy before (tent only)
    post_entropy: float | None     # ...and after
    collapsed: bool                # TENT degenerated to a near-single-class predictor
    warning: str | None
    detail: str


def _logits(out) -> torch.Tensor:
    """Extract logits from a model that returns {'logits',...} or a bare tensor."""
    return out["logits"] if isinstance(out, dict) else out


def _bn_layers(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, _BN_TYPES)]


def _as_batches(data, *, batch_size: int = 64, device: str = "cpu") -> list[torch.Tensor]:
    """Normalise target data to a list of model-ready tensor batches ``[B, C, N]``.

    Accepts: a single tensor ``[B, C, N]``; an iterable of tensors; or an iterable of
    raw signals (numpy ``[C, N]``) which are preprocessed exactly like serving
    (z-score + 1 Hz high-pass) before batching.
    """
    if isinstance(data, torch.Tensor):
        t = data if data.dim() == 3 else data.unsqueeze(0)
        return [t[i:i + batch_size].to(device) for i in range(0, t.shape[0], batch_size)]
    items = list(data)
    if items and isinstance(items[0], torch.Tensor) and items[0].dim() == 3:
        return [b.to(device) for b in items]            # already batched tensors
    # raw signals -> preprocess like serving
    from aion_nexus.preprocessing import preprocess_batch
    sigs = [np.asarray(s) for s in items]
    out = []
    for i in range(0, len(sigs), batch_size):
        out.append(preprocess_batch(sigs[i:i + batch_size]).to(device))
    return out


def prediction_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean Shannon entropy of the softmax over a logits batch (the TENT objective)."""
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return -(p * logp).sum(dim=-1).mean()


def recalibrate_batchnorm(model: nn.Module, target_data, *, batch_size: int = 64,
                          device: str = "cpu", reset: bool = True) -> nn.Module:
    """AdaBN: re-estimate BatchNorm running stats on UNLABELED target windows.

    The safe, label-free, non-collapsing default. Returns a deep-copied, adapted
    model (the source model is never mutated). If the model has no BatchNorm, it is
    returned unchanged with a warning — never a silent no-op.
    """
    adapted = copy.deepcopy(model).to(device)
    bns = _bn_layers(adapted)
    if not bns:
        _logger.warning("recalibrate_batchnorm: model has no BatchNorm; returning "
                        "unchanged (this TTA method needs BN running stats).")
        return adapted
    adapted.eval()
    for bn in bns:
        if reset:
            bn.reset_running_stats()
        bn.momentum = None            # cumulative moving average over the target pool
        bn.train()                    # only BN tracks stats; the rest stays eval
    batches = _as_batches(target_data, batch_size=batch_size, device=device)
    with torch.no_grad():
        for xb in batches:
            _logits(adapted(xb))
    adapted.eval()
    _logger.info("AdaBN: recalibrated %d BN layers on %d target windows",
                 len(bns), sum(b.shape[0] for b in batches))
    return adapted


def tent_adapt(model: nn.Module, target_data, *, lr: float = 1e-3, steps: int = 1,
               batch_size: int = 64, device: str = "cpu",
               collapse_dominance: float = 0.92) -> tuple[nn.Module, TTAResult]:
    """TENT: entropy-minimisation TTA on the BN affine params (Wang et al. 2021).

    Updates ONLY the BatchNorm affine parameters (γ, β) to minimise the mean
    prediction entropy on unlabeled target windows. More powerful than AdaBN, but it
    can COLLAPSE to a degenerate predictor (always one class) under class imbalance
    or heavy shift — so after adaptation the predicted-class distribution is checked:
    if a single class dominates beyond ``collapse_dominance``, the run is flagged
    ``collapsed`` with a warning to fall back to :func:`recalibrate_batchnorm`.

    Returns ``(adapted_model, TTAResult)``. The source model is never mutated.
    """
    adapted = copy.deepcopy(model).to(device)
    bns = _bn_layers(adapted)
    batches = _as_batches(target_data, batch_size=batch_size, device=device)
    n_windows = sum(b.shape[0] for b in batches)
    if not bns:
        msg = "model has no BatchNorm; TENT cannot run (needs BN affine params)"
        _logger.warning("tent_adapt: %s", msg)
        return adapted, TTAResult("tent", 0, n_windows, None, None, False, msg,
                                  "no-op: no BatchNorm")

    # Train mode + only BN affine params require grad (the TENT recipe).
    adapted.train()
    for p in adapted.parameters():
        p.requires_grad_(False)
    params: list[torch.nn.Parameter] = []
    for bn in bns:
        bn.train()
        if bn.affine:
            bn.weight.requires_grad_(True)
            bn.bias.requires_grad_(True)
            params += [bn.weight, bn.bias]
    if not params:
        msg = "BatchNorm layers have affine=False; no TENT parameters to optimise"
        return adapted, TTAResult("tent", len(bns), n_windows, None, None, False,
                                  msg, "no-op: BN affine disabled")

    with torch.no_grad():
        pre_ent = float(torch.stack([prediction_entropy(_logits(adapted(xb)))
                                     for xb in batches]).mean())

    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(max(1, steps)):
        for xb in batches:
            opt.zero_grad(set_to_none=True)
            loss = prediction_entropy(_logits(adapted(xb)))
            loss.backward()
            opt.step()

    adapted.eval()
    with torch.no_grad():
        all_logits = torch.cat([_logits(adapted(xb)) for xb in batches], dim=0)
        post_ent = float(prediction_entropy(all_logits))
        preds = all_logits.argmax(dim=-1)
        _, counts = torch.unique(preds, return_counts=True)
        dominance = float(counts.max()) / float(len(preds)) if len(preds) else 1.0

    collapsed = dominance > collapse_dominance
    warning = None
    if collapsed:
        warning = (f"TENT predictions collapsed to one class ({dominance:.0%} of target "
                   f"windows, > {collapse_dominance:.0%}). This is EITHER entropy-min "
                   "degenerating under shift/imbalance OR a genuinely single-state target; "
                   "if unsure, prefer recalibrate_batchnorm (AdaBN), which cannot collapse.")
        _logger.warning(warning)
    return adapted, TTAResult(
        "tent", len(bns), n_windows, pre_ent, post_ent, collapsed, warning,
        f"entropy {pre_ent:.3f} -> {post_ent:.3f} on {n_windows} windows; "
        f"class dominance {dominance:.0%}")


def source_free_adapt(model: nn.Module, target_data, *, method: str = "adabn",
                      device: str = "cpu", **kwargs):
    """Convenience: adapt ``model`` to ``target_data`` (unlabeled) by ``method``.

    ``"adabn"`` (default, safe) -> :func:`recalibrate_batchnorm` (returns the model).
    ``"tent"`` -> :func:`tent_adapt` (returns ``(model, TTAResult)``). Re-certify any
    conformal coverage on a post-adaptation split — TTA changes the model.
    """
    method = method.lower()
    if method == "adabn":
        return recalibrate_batchnorm(model, target_data, device=device, **kwargs)
    if method == "tent":
        return tent_adapt(model, target_data, device=device, **kwargs)
    raise ValueError(f"unknown method {method!r}; expected 'adabn' or 'tent'")
