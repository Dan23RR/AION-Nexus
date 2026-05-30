"""Production constants for AION-NEXUS.

These define the input contract: signal length, sampling rate, number of channels,
and class taxonomy. Changing any of these requires retraining and a major version bump.
"""

# Input contract
NUM_CHANNELS: int = 2          # 2 accelerometers (horizontal, vertical)
SIGNAL_LENGTH: int = 2560      # 0.1 seconds at 25.6 kHz
SAMPLING_RATE_HZ: int = 25_600

# Output taxonomy (4-class severity)
NUM_CLASSES: int = 4
CLASS_NAMES: list[str] = ["normal", "early", "medium", "advanced"]
CLASS_DESCRIPTIONS: dict[str, str] = {
    "normal":   "Healthy bearing — no maintenance action required.",
    "early":    "Initial defect detected — schedule inspection within next maintenance cycle.",
    "medium":   "Progressive degradation — plan replacement before next major maintenance.",
    "advanced": "Imminent failure — stop machine and replace bearing immediately.",
}

# Recommended actions per class (machine-readable)
CLASS_ACTIONS: dict[str, dict] = {
    "normal":   {"alert_level": 0, "stop_machine": False, "schedule_inspection": False},
    "early":    {"alert_level": 1, "stop_machine": False, "schedule_inspection": True},
    "medium":   {"alert_level": 2, "stop_machine": False, "plan_replacement": True},
    "advanced": {"alert_level": 3, "stop_machine": True,  "replace_immediately": True},
}

# Model architecture (frozen — informs validation)
MODEL_PARAM_COUNT: int = 1_061_724
MODEL_SIZE_MB: float = 4.1
PENULTIMATE_FEATURE_DIM: int = 512

# Confidence thresholds (operational defaults; tune per deployment)
LOW_CONFIDENCE_THRESHOLD: float = 0.65   # below this, flag for human review
HIGH_CONFIDENCE_THRESHOLD: float = 0.85  # above this, automated action permitted
