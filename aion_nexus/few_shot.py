"""Few-shot domain adaptation for AION-NEXUS.

Adapt a trained model to a new bearing type / operating condition with as few
as 10 labeled samples per class. Encoder is frozen; only the classifier head is
fine-tuned. ~5 minutes on CPU per adaptation cycle.

Verified lift: +5.7 percentage points F1 on MFPT zero-shot → 10-sample protocol.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 — standard PyTorch convention

from aion_nexus.inference import InferenceEngine
from aion_nexus.preprocessing import preprocess_batch
from aion_nexus.version import __version__

_logger = logging.getLogger(__name__)


# Which submodule is the trainable "head" that few-shot adaptation fine-tunes,
# per architecture. The encoder (everything else) stays frozen. v1's MLP head is
# `classifier.*`; v6's reasoning head that maps features -> logits is
# `recursive_reasoner.*`; v3's (PatchTST substrate) linear few-shot head is
# `head.*`. Filtering the wrong prefix freezes 100% of parameters,
# which makes adaptation a silent no-op AND crashes the optimizer with an empty
# parameter list. _freeze_encoder() guards against exactly that.
_HEAD_PREFIXES: dict[str, tuple[str, ...]] = {
    "v1": ("classifier.",),
    "v6": ("recursive_reasoner.",),
    "v3": ("head.",),
    "ext": ("head.",),   # ExternalEncoderAdapter: frozen foundation encoder + linear head
}


class FewShotAdapter:
    """Fine-tune only the classifier/reasoning head on a small target-domain dataset.

    Supports all three architectures: v1 (BiGRU + MLP ``classifier`` head), v6
    (TemporalSelfAttention + ``recursive_reasoner`` head) and v3 (PatchTST
    substrate + linear ``head``). The head to unfreeze is selected from the source
    engine's ``architecture_version`` — passing the wrong one would freeze every
    parameter and make adaptation a no-op, so it is checked.

    Usage:
        engine = InferenceEngine.from_checkpoint("checkpoints/aion_nexus_v1.pth")
        adapter = FewShotAdapter(engine)
        adapter.adapt(target_signals, target_labels, epochs=5, lr=1e-4)
        adapter.save("checkpoints/aion_nexus_v1_machine42.pth")
    """

    def __init__(self, engine: InferenceEngine) -> None:
        # Deep-copy the model so adaptation does not mutate the source engine
        self.model = copy.deepcopy(engine.model)
        self.device = engine.device
        self.architecture_version = getattr(engine, "architecture_version", "v1")
        self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        """Freeze the encoder; leave only this architecture's head trainable.

        Raises:
            ValueError: if the architecture version is unknown.
            RuntimeError: if no parameter matches the head prefix (which would make
                adaptation a no-op and crash the optimizer with an empty param list).
        """
        prefixes = _HEAD_PREFIXES.get(self.architecture_version)
        if prefixes is None:
            raise ValueError(
                f"Unknown architecture_version {self.architecture_version!r}; "
                f"expected one of {sorted(_HEAD_PREFIXES)}"
            )
        for name, p in self.model.named_parameters():
            p.requires_grad = any(name.startswith(pre) for pre in prefixes)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        if n_train == 0:
            raise RuntimeError(
                f"Few-shot froze ALL {n_total:,} parameters for architecture "
                f"{self.architecture_version!r}: no head matched prefixes {prefixes}. "
                "Adaptation would be a no-op; check _HEAD_PREFIXES against the "
                "model's module names."
            )
        _logger.info(
            "FewShot[%s]: %d / %d params trainable (%.1f%%)",
            self.architecture_version, n_train, n_total, 100 * n_train / n_total,
        )

    def adapt(
        self,
        signals: Iterable[np.ndarray],
        labels: Iterable[int],
        epochs: int = 5,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        verbose: bool = True,
    ) -> dict:
        """Run few-shot fine-tuning.

        Args:
            signals: iterable of [2, >=2560] arrays.
            labels: matching iterable of int class indices.
            epochs: number of fine-tuning epochs (5 is the verified default).
            lr: learning rate (1e-4 is the verified default).
            weight_decay: L2 regularization.
            label_smoothing: cross-entropy label smoothing (helps few-shot stability).
            verbose: log per-epoch loss.

        Returns:
            dict with 'epoch_losses' and 'final_loss' for monitoring.
        """
        signals_l = list(signals)
        labels_l = list(labels)
        if len(signals_l) != len(labels_l):
            raise ValueError("signals and labels must have the same length")
        if len(signals_l) < 4:
            raise ValueError(
                f"Need at least 4 samples for stable adaptation; got {len(signals_l)}"
            )

        x = preprocess_batch(signals_l).to(self.device)
        y = torch.tensor(labels_l, dtype=torch.long, device=self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.model.train()

        epoch_losses = []
        for epoch in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = F.cross_entropy(out["logits"], y, label_smoothing=label_smoothing)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            if verbose:
                _logger.info("Few-shot epoch %d/%d: loss=%.4f", epoch + 1, epochs, loss.item())

        self.model.eval()
        return {"epoch_losses": epoch_losses, "final_loss": epoch_losses[-1]}

    def to_engine(self) -> InferenceEngine:
        """Wrap the adapted model into an InferenceEngine for serving."""
        return InferenceEngine(self.model, device=self.device,
                               architecture_version=self.architecture_version)

    def save(self, path: str | Path) -> None:
        """Save adapted model state-dict (compatible with from_checkpoint)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_version": __version__,
            "adapted": True,
        }, path)
        _logger.info("Adapted model saved: %s", path)
