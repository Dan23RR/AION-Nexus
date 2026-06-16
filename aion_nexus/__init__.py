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

- **v3 substrate** (PatchTST self-supervised foundation encoder, ~1.22M params):
  a FROZEN cross-domain FEW-SHOT backbone — NOT a higher in-distribution classifier
  (v1's 0.884 stands). Adapt to a NEW machine with ~10 labels/class; served via the
  AION-2 verified trust layer (conformal + physics verifier -> certificate).
  Cross-dataset 10-shot health macro-F1 0.91-1.00 (vs random-init 0.5-0.8).
  See `substrate_v3.py` + MODEL_CARD.md.

`InferenceEngine.from_checkpoint()` auto-detects the architecture from the
checkpoint's state_dict keys; no flag required for the common case.
"""
from aion_nexus.compliance import (
    annex_iv_card,
    annex_iv_dossier,
    compliance_evidence,
    evidence_card,
)
from aion_nexus.config import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    MODEL_PARAM_COUNT,
    NUM_CHANNELS,
    NUM_CLASSES,
    SAMPLING_RATE_HZ,
    SIGNAL_LENGTH,
)
from aion_nexus.degradation import DegradationEstimate, estimate_degradation
from aion_nexus.few_shot import FewShotAdapter
from aion_nexus.inference import InferenceEngine, PredictionResult
from aion_nexus.model import AIONNexus, create_aion_nexus
from aion_nexus.model_v6 import V6_PARAM_COUNT_4CLASS, AIONNexusV6, create_aion_nexus_v6
from aion_nexus.physics import (
    BearingGeometry,
    PhysicsVerdict,
    fault_order_energy,
    order_spectrum,
    physics_consistency,
)
from aion_nexus.preprocessing import preprocess_signal, validate_signal
from aion_nexus.recursive_reasoner import TinyRecursiveReasoner
from aion_nexus.substrate_v3 import (
    V3_ENCODER_PARAM_COUNT,
    AIONNexusV3,
    SubstrateEncoderV3,
    create_substrate_v3,
)
from aion_nexus.temporal_attention import TemporalSelfAttention
from aion_nexus.verify import (
    Certificate,
    ConformalCalibrator,
    Verifier,
    verify_certificate,
)
from aion_nexus.version import __version__

__all__ = [
    "__version__",
    # v1 architecture
    "AIONNexus", "create_aion_nexus",
    # v6 architecture
    "AIONNexusV6", "create_aion_nexus_v6", "V6_PARAM_COUNT_4CLASS",
    "TemporalSelfAttention", "TinyRecursiveReasoner",
    # v3 substrate (self-supervised foundation backbone)
    "AIONNexusV3", "SubstrateEncoderV3", "create_substrate_v3", "V3_ENCODER_PARAM_COUNT",
    # Engine + utilities
    "InferenceEngine", "PredictionResult", "FewShotAdapter",
    "preprocess_signal", "validate_signal",
    # Degradation-stage (honest coarse positional proxy, NOT calibrated RUL)
    "DegradationEstimate", "estimate_degradation",
    # Physics front-end + model-agnostic second-opinion verifier (RPM-invariant orders)
    "BearingGeometry", "PhysicsVerdict", "physics_consistency",
    "fault_order_energy", "order_spectrum",
    # Substrate Core: model-agnostic verification & certification layer
    "Verifier", "Certificate", "ConformalCalibrator", "verify_certificate",
    "compliance_evidence", "evidence_card",
    "annex_iv_dossier", "annex_iv_card",
    # Constants
    "CLASS_NAMES", "CLASS_DESCRIPTIONS", "NUM_CLASSES",
    "SIGNAL_LENGTH", "NUM_CHANNELS", "SAMPLING_RATE_HZ", "MODEL_PARAM_COUNT",
]
