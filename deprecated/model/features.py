from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .data import EventData


@dataclass
class FeatureSet:
    """Feature matrix and targets for downstream training."""

    features: np.ndarray
    labels: np.ndarray
    sample_indices: List[int]


@dataclass
class FeatureMetadata:
    """Describes how the feature matrix was constructed."""

    window_size: int
    feature_dim: int
    sensor_vocab: List[str]
    state_vocab: List[str]
    value_type_vocab: List[str]
    label_names: List[str]


def build_feature_set(
    events: EventData,
    window_size: int,
    max_samples: Optional[int] = None,
) -> Tuple[FeatureSet, FeatureMetadata]:
    """Construct the supervised dataset from the streaming events."""

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    num_events = len(events.timestamps)
    label_indices: List[int] = []
    for idx in range(window_size - 1, num_events):
        if events.activity_ids[idx] != 0:
            label_indices.append(idx)
    if not label_indices:
        raise RuntimeError("No labeled events found. Verify the processed CSV.")

    if max_samples is not None:
        label_indices = label_indices[:max_samples]

    num_samples = len(label_indices)
    num_sensors = len(events.sensor_vocab)
    num_states = len(events.state_vocab)
    num_value_types = len(events.value_type_vocab)
    feature_dim = 2 * num_sensors + num_states + num_value_types + 7

    X = np.zeros((num_samples, feature_dim), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int32)

    for row_idx, center in enumerate(label_indices):
        start = center - window_size + 1
        window_ids = events.sensor_ids[start : center + 1]

        freq = np.bincount(window_ids, minlength=num_sensors).astype(np.float32)
        freq /= window_size

        offset = 0
        X[row_idx, offset : offset + num_sensors] = freq
        offset += num_sensors

        current_sensor = events.sensor_ids[center]
        X[row_idx, offset + current_sensor] = 1.0
        offset += num_sensors

        current_state = events.state_ids[center]
        X[row_idx, offset + current_state] = 1.0
        offset += num_states

        current_value_type = events.value_type_ids[center]
        X[row_idx, offset + current_value_type] = 1.0
        offset += num_value_types

        numeric_value = events.numeric_values[center]
        X[row_idx, offset] = numeric_value
        offset += 1

        X[row_idx, offset] = events.has_numeric[center]
        offset += 1

        timestamp = events.timestamps[center]
        minute_of_day = (
            timestamp.hour * 60
            + timestamp.minute
            + timestamp.second / 60.0
            + timestamp.microsecond / 60_000_000.0
        )
        tod_angle = 2 * math.pi * minute_of_day / (24 * 60)
        X[row_idx, offset] = math.sin(tod_angle)
        X[row_idx, offset + 1] = math.cos(tod_angle)
        offset += 2

        dow_angle = 2 * math.pi * timestamp.weekday() / 7.0
        X[row_idx, offset] = math.sin(dow_angle)
        X[row_idx, offset + 1] = math.cos(dow_angle)
        offset += 2

        window_duration = (
            (events.timestamps[center] - events.timestamps[start]).total_seconds() / 60.0
        )
        X[row_idx, offset] = max(window_duration, 0.0)

        label = events.activity_ids[center]
        if label <= 0:
            raise AssertionError("Encountered unlabeled sample after filtering.")
        y[row_idx] = label - 1

    metadata = FeatureMetadata(
        window_size=window_size,
        feature_dim=feature_dim,
        sensor_vocab=list(events.sensor_vocab),
        state_vocab=list(events.state_vocab),
        value_type_vocab=list(events.value_type_vocab),
        label_names=events.activity_vocab[1:],
    )

    feature_set = FeatureSet(features=X, labels=y, sample_indices=label_indices)
    return feature_set, metadata
