"""Example 3: few-shot domain adaptation for a new machine type.

Recipe:
1. Collect 10 samples per severity class on the new machine (40 total).
2. Label them.
3. Run this script — encoder frozen, classifier head fine-tunes in ~5 minutes.
4. Save the adapted checkpoint, deploy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aion_nexus import FewShotAdapter, InferenceEngine
from aion_nexus.model import create_aion_nexus


def make_target_dataset(n_per_class: int = 10) -> tuple[list[np.ndarray], list[int]]:
    """REPLACE THIS: load your target-machine labeled samples here.

    For each sample:
      - Acquire a 0.1-second 2-channel 25.6-kHz vibration recording.
      - Save it as a [2, >=2560] numpy array.
      - Tag it with the class index (0=normal, 1=early, 2=medium, 3=advanced).
    """
    rng = np.random.default_rng(42)
    sigs, labels = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            sigs.append(rng.standard_normal((2, 2560)).astype(np.float32) * 0.5)
            labels.append(cls)
    return sigs, labels


def main() -> int:
    src_ckpt = Path("checkpoints/aion_nexus_v1.pth")
    if src_ckpt.exists():
        engine = InferenceEngine.from_checkpoint(src_ckpt)
    else:
        print("WARNING: no source checkpoint, using random weights for demo.")
        engine = InferenceEngine(create_aion_nexus())

    target_sigs, target_labels = make_target_dataset(n_per_class=10)
    print(f"Target dataset: {len(target_sigs)} samples across "
          f"{len(set(target_labels))} classes.")

    adapter = FewShotAdapter(engine)
    history = adapter.adapt(
        signals=target_sigs,
        labels=target_labels,
        epochs=5,
        lr=1e-4,
        verbose=True,
    )

    print()
    print("Adaptation complete.")
    print(f"  Initial loss: {history['epoch_losses'][0]:.4f}")
    print(f"  Final loss:   {history['epoch_losses'][-1]:.4f}")

    out_path = Path("checkpoints/aion_nexus_v1_machine42.pth")
    adapter.save(out_path)
    print(f"  Saved adapted model to: {out_path}")

    # Use the adapted engine for inference
    adapted_engine = adapter.to_engine()
    test_sig = target_sigs[0]
    result = adapted_engine.predict(test_sig)
    print()
    print("Sanity prediction on first target sample:")
    print(f"  predicted: {result.predicted_class_name}  confidence: {result.confidence:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
