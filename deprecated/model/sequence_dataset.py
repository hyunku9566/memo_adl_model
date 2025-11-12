from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .data import EventData


@dataclass
class SequenceSamples:
    """Holds sliding-window sequences for deep models."""

    sensor_seq: np.ndarray
    state_seq: np.ndarray
    value_type_seq: np.ndarray
    numeric_seq: np.ndarray
    numeric_mask_seq: np.ndarray
    time_features_seq: np.ndarray
    labels: np.ndarray
    sample_indices: np.ndarray
    window_size: int
    sensor_vocab: List[str]
    state_vocab: List[str]
    value_type_vocab: List[str]
    label_names: List[str]
    numeric_mean: float
    numeric_std: float


def build_sequence_samples(
    events: EventData,
    window_size: int,
    max_samples: Optional[int] = None,
) -> SequenceSamples:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    num_events = len(events.timestamps)
    if num_events == 0:
        raise ValueError("No events available.")

    sensor_ids = np.asarray(events.sensor_ids, dtype=np.int64)
    state_ids = np.asarray(events.state_ids, dtype=np.int64)
    value_type_ids = np.asarray(events.value_type_ids, dtype=np.int64)
    numeric_values = np.asarray(events.numeric_values, dtype=np.float32)
    numeric_mask = np.asarray(events.has_numeric, dtype=np.float32)
    activity_ids = np.asarray(events.activity_ids, dtype=np.int64)

    tod_sin, tod_cos, dow_sin, dow_cos = _compute_time_features(events)

    numeric_mean, numeric_std = _compute_numeric_stats(numeric_values, numeric_mask)
    numeric_std_values = numeric_values.copy()
    has_numeric = numeric_mask > 0.5
    if has_numeric.any():
        numeric_std_values[has_numeric] = (numeric_std_values[has_numeric] - numeric_mean) / numeric_std
    else:
        numeric_std_values[:] = 0.0

    label_indices: List[int] = []
    for idx in range(window_size - 1, num_events):
        if activity_ids[idx] > 0:
            label_indices.append(idx)
    if not label_indices:
        raise RuntimeError(
            "No labeled samples found. Ensure the processed CSV contains activity annotations."
        )
    if max_samples is not None:
        label_indices = label_indices[:max_samples]

    num_samples = len(label_indices)
    sensor_seq = np.zeros((num_samples, window_size), dtype=np.int64)
    state_seq = np.zeros((num_samples, window_size), dtype=np.int64)
    value_type_seq = np.zeros((num_samples, window_size), dtype=np.int64)
    numeric_seq = np.zeros((num_samples, window_size), dtype=np.float32)
    numeric_mask_seq = np.zeros((num_samples, window_size), dtype=np.float32)
    time_features_seq = np.zeros((num_samples, window_size, 4), dtype=np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)

    for row_idx, idx in enumerate(label_indices):
        start = idx - window_size + 1
        stop = idx + 1
        sensor_seq[row_idx] = sensor_ids[start:stop]
        state_seq[row_idx] = state_ids[start:stop]
        value_type_seq[row_idx] = value_type_ids[start:stop]
        numeric_seq[row_idx] = numeric_std_values[start:stop]
        numeric_mask_seq[row_idx] = numeric_mask[start:stop]
        time_features_seq[row_idx, :, 0] = tod_sin[start:stop]
        time_features_seq[row_idx, :, 1] = tod_cos[start:stop]
        time_features_seq[row_idx, :, 2] = dow_sin[start:stop]
        time_features_seq[row_idx, :, 3] = dow_cos[start:stop]
        labels[row_idx] = activity_ids[idx] - 1

    return SequenceSamples(
        sensor_seq=sensor_seq,
        state_seq=state_seq,
        value_type_seq=value_type_seq,
        numeric_seq=numeric_seq,
        numeric_mask_seq=numeric_mask_seq,
        time_features_seq=time_features_seq,
        labels=labels,
        sample_indices=np.asarray(label_indices, dtype=np.int64),
        window_size=window_size,
        sensor_vocab=events.sensor_vocab,
        state_vocab=events.state_vocab,
        value_type_vocab=events.value_type_vocab,
        label_names=events.activity_vocab[1:],
        numeric_mean=float(numeric_mean),
        numeric_std=float(numeric_std),
    )


def _compute_time_features(events: EventData):
    minutes = np.array(
        [
            ts.hour * 60
            + ts.minute
            + ts.second / 60.0
            + ts.microsecond / 60_000_000.0
            for ts in events.timestamps
        ],
        dtype=np.float32,
    )
    tod_angle = 2 * math.pi * minutes / (24 * 60)
    tod_sin = np.sin(tod_angle).astype(np.float32)
    tod_cos = np.cos(tod_angle).astype(np.float32)

    weekdays = np.array([ts.weekday() for ts in events.timestamps], dtype=np.float32)
    dow_angle = 2 * math.pi * weekdays / 7.0
    dow_sin = np.sin(dow_angle).astype(np.float32)
    dow_cos = np.cos(dow_angle).astype(np.float32)
    return tod_sin, tod_cos, dow_sin, dow_cos


def _compute_numeric_stats(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = mask > 0.5
    if not valid.any():
        return 0.0, 1.0
    subset = values[valid]
    mean = float(subset.mean())
    std = float(subset.std())
    if std < 1e-6:
        std = 1.0
    return mean, std
