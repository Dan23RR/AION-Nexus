"""Few-shot adapter tests (synthetic data, no checkpoint required)."""
import numpy as np
import pytest
import torch

from aion_nexus import (
    InferenceEngine,
    FewShotAdapter,
    create_aion_nexus,
    NUM_CHANNELS,
    SIGNAL_LENGTH,
)


def _make_dummy_data(n_per_class=10, n_classes=4):
    sigs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per_class):
            sigs.append(np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32))
            labels.append(c)
    return sigs, labels


def test_freeze_encoder_correct_param_count():
    model = create_aion_nexus()
    engine = InferenceEngine(model)
    adapter = FewShotAdapter(engine)

    n_train = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in adapter.model.parameters())
    # Only the classifier head should be trainable (~150K of ~1.06M)
    assert 100_000 < n_train < 250_000
    assert n_total - n_train > 800_000


def test_adapt_runs_and_decreases_loss():
    torch.manual_seed(0)
    model = create_aion_nexus()
    engine = InferenceEngine(model)
    adapter = FewShotAdapter(engine)

    sigs, labels = _make_dummy_data(n_per_class=5)
    history = adapter.adapt(sigs, labels, epochs=5, lr=1e-3, verbose=False)

    assert len(history["epoch_losses"]) == 5
    # Loss should generally decrease over 5 epochs on consistent data
    assert history["epoch_losses"][-1] <= history["epoch_losses"][0] + 0.1


def test_adapt_too_few_samples():
    model = create_aion_nexus()
    engine = InferenceEngine(model)
    adapter = FewShotAdapter(engine)
    sigs, labels = _make_dummy_data(n_per_class=1, n_classes=2)
    with pytest.raises(ValueError, match="at least 4 samples"):
        adapter.adapt(sigs[:2], labels[:2], epochs=1)


def test_adapter_does_not_mutate_source(tmp_path):
    """Adapter deep-copies the model; adapting must not change the source engine."""
    torch.manual_seed(0)
    model = create_aion_nexus()
    engine = InferenceEngine(model)
    sig = np.random.randn(NUM_CHANNELS, SIGNAL_LENGTH).astype(np.float32)
    pred_before = engine.predict(sig)

    adapter = FewShotAdapter(engine)
    sigs, labels = _make_dummy_data(n_per_class=5)
    adapter.adapt(sigs, labels, epochs=3, lr=1e-3, verbose=False)

    pred_after = engine.predict(sig)
    # Source engine prediction must be unchanged
    for cls in pred_before.probabilities:
        assert pred_before.probabilities[cls] == pytest.approx(
            pred_after.probabilities[cls], abs=1e-5
        )


def test_adapter_save_and_reload(tmp_path):
    model = create_aion_nexus()
    engine = InferenceEngine(model)
    adapter = FewShotAdapter(engine)
    sigs, labels = _make_dummy_data(n_per_class=5)
    adapter.adapt(sigs, labels, epochs=2, lr=1e-3, verbose=False)

    path = tmp_path / "adapted.pth"
    adapter.save(path)
    assert path.exists()

    loaded_engine = InferenceEngine.from_checkpoint(path, strict=False)
    assert loaded_engine.model.get_num_params() == model.get_num_params()


# --- v6 coverage (regression guard for the silent no-op / empty-optimizer bug) ---

def _make_v6_engine():
    from aion_nexus.model_v6 import create_aion_nexus_v6
    model = create_aion_nexus_v6()
    return InferenceEngine(model, architecture_version="v6")


def test_v6_freeze_unfreezes_reasoning_head():
    """v6's head is `recursive_reasoner.*`, NOT `classifier.*`. The old code froze
    100% of v6 params (the bug); the fix must leave the reasoning head trainable."""
    engine = _make_v6_engine()
    adapter = FewShotAdapter(engine)
    n_train = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in adapter.model.parameters())
    assert n_train > 0, "v6 few-shot froze all params (regression of the no-op bug)"
    assert n_train < n_total, "encoder must stay frozen"


def test_v6_adapt_actually_updates_head_weights():
    """The bug made v6 adaptation a no-op. Assert the head weights actually move."""
    torch.manual_seed(0)
    engine = _make_v6_engine()
    adapter = FewShotAdapter(engine)

    before = [p.detach().clone() for p in adapter.model.parameters() if p.requires_grad]
    sigs, labels = _make_dummy_data(n_per_class=5)
    history = adapter.adapt(sigs, labels, epochs=5, lr=1e-3, verbose=False)
    after = [p.detach().clone() for p in adapter.model.parameters() if p.requires_grad]

    assert len(history["epoch_losses"]) == 5
    moved = any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert moved, "v6 few-shot did not update any weights (no-op bug regression)"


def test_unknown_architecture_raises():
    model = create_aion_nexus()
    engine = InferenceEngine(model, architecture_version="v99")
    with pytest.raises(ValueError, match="Unknown architecture_version"):
        FewShotAdapter(engine)
