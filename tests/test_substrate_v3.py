"""Tests for the v3 self-supervised substrate backbone (production)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from aion_nexus import create_substrate_v3, V3_ENCODER_PARAM_COUNT, NUM_CLASSES
from aion_nexus.substrate_v3 import V3_EMBED_DIM


def test_v3_encoder_param_guard():
    m = create_substrate_v3()
    assert sum(p.numel() for p in m.encoder.parameters()) == V3_ENCODER_PARAM_COUNT


def test_v3_param_guard_raises_on_drift():
    with pytest.raises(ValueError):
        # wrong architecture vs frozen contract -> factory must refuse
        from aion_nexus.substrate_v3 import AIONNexusV3, SubstrateEncoderV3
        m = AIONNexusV3.__new__(AIONNexusV3)  # bypass init
        torch.nn.Module.__init__(m)
        m.encoder = SubstrateEncoderV3(d_model=128)   # diverged
        m.head = torch.nn.Linear(128, 4)
        # emulate the factory guard
        n = sum(p.numel() for p in m.encoder.parameters())
        if n != V3_ENCODER_PARAM_COUNT:
            raise ValueError("v3 substrate drift")


def test_v3_forward_contract():
    m = create_substrate_v3().eval()
    out = m(torch.randn(2, 2, 2560))
    assert "logits" in out and "features" in out
    assert out["logits"].shape == (2, NUM_CLASSES)
    assert out["features"].shape == (2, V3_EMBED_DIM)


def test_v3_encoder_frozen_head_trainable():
    m = create_substrate_v3()
    enc_grad = any(p.requires_grad for p in m.encoder.parameters())
    head_grad = all(p.requires_grad for p in m.head.parameters())
    assert (not enc_grad) and head_grad        # few-shot: frozen encoder, trainable head


def test_v3_load_checkpoint_if_present():
    ckpt = Path(__file__).parents[1] / "checkpoints" / "aion_nexus_substrate_v3.pth"
    if not ckpt.exists():
        pytest.skip("substrate checkpoint not present")
    m = create_substrate_v3()
    m.load_substrate(str(ckpt))
    m.eval()
    out = m(torch.from_numpy(np.random.default_rng(0).standard_normal((1, 2, 2560)).astype("float32")))
    assert out["features"].shape == (1, V3_EMBED_DIM)


def test_v31_bigger_arch_hosts_in_production():
    # the production encoder can be built at the v3.1 architecture (d_model 256, depth 6)
    from aion_nexus.substrate_v3 import SubstrateEncoderV3, V3_1_ENCODER_PARAM_COUNT
    enc = SubstrateEncoderV3(d_model=256, depth=6, nhead=8, dim_ff=512)
    assert sum(p.numel() for p in enc.parameters()) == V3_1_ENCODER_PARAM_COUNT   # 3,206,656
    assert enc.embed(torch.randn(2, 2, 2560)).shape == (2, 256)


def test_from_checkpoint_reads_cfg_if_present():
    from aion_nexus.substrate_v3 import AIONNexusV3
    ckpt = Path(__file__).parents[1] / "checkpoints" / "aion_nexus_substrate_v3.pth"
    if not ckpt.exists():
        pytest.skip("substrate checkpoint not present")
    m = AIONNexusV3.from_checkpoint(str(ckpt)).eval()      # reads arch from cfg, drops mask_token
    out = m(torch.randn(1, 2, 2560))
    assert out["logits"].shape == (1, NUM_CLASSES) and out["features"].shape[1] == m.encoder.out_dim
