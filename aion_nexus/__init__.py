"""AION-NEXUS — production bearing-fault diagnosis from raw vibration signals.

Two architecture versions supported, both verified against published numbers:

- **v1** (BiGRU-based, 1,061,724 params, 4.1 MB): F1=0.884 on the FEMTO Test_set
  run-to-failure bearings (globally-stratified split). Recommended for industrial
  run-to-failure deployment.

- **v6** (TemporalSelfAttention + TRM, 716,577 params, 2.73 MB): F1=0.934 on the
  FEMTO Learning_set calibration bearings — a DIFFERENT subset, NOT comparable to
  v1's number. v6 collapses to 0.302 cross-bearing and 0.352 +/- 0.112 under LOBO
  (see results/lobo_cv_v6/). Smaller/faster/noise-robust; opt-in for short-cycle
  calibration regimes, not a drop-in accuracy upgrade. See MODEL_CARD.md.

`InferenceEngine.from_checkpoint()` auto-detects the architecture from the
checkpoint's state_dict keys; no flag required for the common case.
"""
from aion_nexus.version import __version__
from aion_nexus.model import AIONNexus, create_aion_nexus
from aion_nexus.model_v6 import AIONNexusV6, create_aion_nexus_v6, V6_PARAM_COUNT_4CLASS
from aion_nexus.temporal_attention import TemporalSelfAttention
from aion_nexus.recursive_reasoner import TinyRecursiveReasoner
from aion_nexus.preprocessing import preprocess_signal, validate_signal
from aion_nexus.inference import InferenceEngine, PredictionResult
from aion_nexus.few_shot import FewShotAdapter
from aion_nexus.config import (
    CLASS_NAMES,
    CLASS_DESCRIPTIONS,
    NUM_CLASSES,
    SIGNAL_LENGTH,
    NUM_CHANNELS,
    SAMPLING_RATE_HZ,
    MODEL_PARAM_COUNT,
)

__all__ = [
    "__version__",
    # v1 architecture
    "AIONNexus", "create_aion_nexus",
    # v6 architecture
    "AIONNexusV6", "create_aion_nexus_v6", "V6_PARAM_COUNT_4CLASS",
    "TemporalSelfAttention", "TinyRecursiveReasoner",
    # Engine + utilities
    "InferenceEngine", "PredictionResult", "FewShotAdapter",
    "preprocess_signal", "validate_signal",
    # Constants
    "CLASS_NAMES", "CLASS_DESCRIPTIONS", "NUM_CLASSES",
    "SIGNAL_LENGTH", "NUM_CHANNELS", "SAMPLING_RATE_HZ", "MODEL_PARAM_COUNT",
]
