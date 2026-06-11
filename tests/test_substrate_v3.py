"""Tests for the v3 self-supervised substrate backbone (production)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from aion_nexus import NUM_CLASSES, V3_ENCODER_PARAM_COUNT, create_substrate_v3
from aion_nexus.substrate_v3 import V3_EMBED_DIM


def test_v3_encoder_param_guard():
    m = create_substrate_v3()
    assert sum(p.numel() for p in m.encoder.parameters()) == V3_ENCODER_PARAM_COUNT


def test_v3_param_guard_raises_on_drift(monkeypatch):
    """The REAL factory guard must refuse an encoder that diverged from the
    frozen contract. Simulate drift by patching the expected count the factory
    checks against, then verify create_substrate_v3 itself raises."""
    import aion_nexus.substrate_v3 as sv3
    monkeypatch.setattr(sv3, "V3_ENCODER_PARAM_COUNT", V3_ENCODER_PARAM_COUNT + 1)
    with pytest.raises(ValueError, match="drift"):
        sv3.create_substrate_v3()
    # opting out of the guard must not raise
    sv3.create_substrate_v3(strict_param_check=False)


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
    from aion_nexus.substrate_v3 import V3_1_ENCODER_PARAM_COUNT, SubstrateEncoderV3
    enc = SubstrateEncoderV3(d_model=256, depth=6, nhead=8, dim_ff=512)
    # 3,206,400 (production encoder; the PRETRAINING encoder is +256 for the mask_token)
    assert sum(p.numel() for p in enc.parameters()) == V3_1_ENCODER_PARAM_COUNT
    assert enc.embed(torch.randn(2, 2, 2560)).shape == (2, 256)


def test_from_checkpoint_reads_cfg_if_present():
    from aion_nexus.substrate_v3 import AIONNexusV3
    ckpt = Path(__file__).parents[1] / "checkpoints" / "aion_nexus_substrate_v3.pth"
    if not ckpt.exists():
        pytest.skip("substrate checkpoint not present")
    m = AIONNexusV3.from_checkpoint(str(ckpt)).eval()      # reads arch from cfg, drops mask_token
    out = m(torch.randn(1, 2, 2560))
    assert out["logits"].shape == (1, NUM_CLASSES) and out["features"].shape[1] == m.encoder.out_dim


# ---- serving-path integration (InferenceEngine + FewShotAdapter) -------------

def test_detect_architecture_v3_full_model():
    from aion_nexus.inference import InferenceEngine
    sd = create_substrate_v3().state_dict()     # encoder.* + head.*
    assert InferenceEngine.detect_architecture(sd) == "v3"


def test_detect_architecture_v3_encoder_only():
    from aion_nexus.inference import InferenceEngine
    enc_sd = create_substrate_v3().encoder.state_dict()
    raw = {"encoder": enc_sd, "cfg": {"d_model": 192}, "objective": "contrastive-ntxent-patchTST"}
    assert InferenceEngine.detect_architecture(raw) == "v3"


def test_detect_architecture_still_rejects_unknown():
    from aion_nexus.inference import InferenceEngine
    with pytest.raises(ValueError, match="Unrecognized"):
        InferenceEngine.detect_architecture({"some.layer.weight": torch.zeros(1)})


def test_few_shot_v3_head_prefix():
    """FewShotAdapter on a v3 engine must train ONLY the linear head."""
    from aion_nexus import FewShotAdapter, InferenceEngine
    engine = InferenceEngine(create_substrate_v3(), architecture_version="v3")
    adapter = FewShotAdapter(engine)
    trainable = [n for n, p in adapter.model.named_parameters() if p.requires_grad]
    assert trainable, "v3 adaptation froze everything (silent no-op)"
    assert all(n.startswith("head.") for n in trainable)


def test_from_checkpoint_v3_full_model_serves(tmp_path):
    """A v3 checkpoint WITH a trained head is rebuilt and served end-to-end."""
    from aion_nexus.inference import InferenceEngine
    ckpt = tmp_path / "v3_adapted.pth"
    torch.save({"model_state_dict": create_substrate_v3().state_dict()}, ckpt)
    engine = InferenceEngine.from_checkpoint(ckpt)
    assert engine.architecture_version == "v3"
    sig = np.random.default_rng(0).standard_normal((2, 2560)).astype("float32")
    result = engine.predict(sig)
    assert result.predicted_class_name in ("normal", "early", "medium", "advanced")
    assert engine.get_health()["architecture_version"] == "v3"


def test_from_checkpoint_v3_encoder_only_refused_actionably(tmp_path):
    """The raw pretrained substrate (no trained head) must be refused with an
    actionable error, not served with a random head and not crash opaquely."""
    from aion_nexus.inference import InferenceEngine
    ckpt = tmp_path / "v3_encoder_only.pth"
    torch.save(
        {"encoder": create_substrate_v3().encoder.state_dict(),
         "cfg": {"patch_len": 64, "d_model": 192, "depth": 4, "nhead": 4},
         "objective": "contrastive-ntxent-patchTST"},
        ckpt,
    )
    with pytest.raises(ValueError, match="head"):
        InferenceEngine.from_checkpoint(ckpt)
