"""Utilities for loading events, engineering features, and training models."""

from .data import EventData, load_events  # noqa: F401

# Active modules
from .rich_features import RichFeatureExtractor, RichFeatures  # noqa: F401
from .rich_dataset import RichFeatureDataset, collate_rich_features  # noqa: F401
from .ema_adaptive_decay import EMAAdaptiveDecayModel  # noqa: F401
from .position_velocity_model import SmartHomeModel, MultiTaskLoss  # noqa: F401
from .pv_dataset import PVDataset, collate_pv_features  # noqa: F401

# Deprecated modules moved to deprecated/ folder
# from .features import FeatureSet, build_feature_set
# from .sequence_dataset import SequenceSamples, build_sequence_samples
