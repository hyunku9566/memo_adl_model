from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

VALUE_TYPE_VOCAB = ["missing", "state", "numeric", "string"]


@dataclass
class EventData:
    """Stores the parsed event stream along with vocabularies."""

    timestamps: List[dt.datetime]
    sensor_ids: List[int]
    state_ids: List[int]
    value_type_ids: List[int]
    numeric_values: List[float]
    has_numeric: List[int]
    activity_ids: List[int]
    sensor_vocab: List[str]
    state_vocab: List[str]
    value_type_vocab: List[str]
    activity_vocab: List[str]


def load_events(csv_path: Path) -> EventData:
    """Load processed events from disk and build vocabularies on the fly."""

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"{csv_path} does not exist")

    timestamps: List[dt.datetime] = []
    sensor_ids: List[int] = []
    state_ids: List[int] = []
    value_type_ids: List[int] = []
    numeric_values: List[float] = []
    has_numeric: List[int] = []
    activity_ids: List[int] = []

    sensor_vocab: List[str] = []
    sensor_to_idx: Dict[str, int] = {}

    state_vocab: List[str] = ["<NONE>"]
    state_to_idx: Dict[str, int] = {"<NONE>": 0}

    value_type_vocab = VALUE_TYPE_VOCAB.copy()
    value_type_to_idx = {name: idx for idx, name in enumerate(value_type_vocab)}

    activity_vocab: List[str] = ["<NONE>"]
    activity_to_idx: Dict[str, int] = {"<NONE>": 0}

    with path.open() as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            timestamp = dt.datetime.fromisoformat(row["timestamp"])
            timestamps.append(timestamp)

            sensor_name = row["sensor"]
            sensor_idx = sensor_to_idx.setdefault(sensor_name, len(sensor_vocab))
            if sensor_idx == len(sensor_vocab):
                sensor_vocab.append(sensor_name)
            sensor_ids.append(sensor_idx)

            state_token = (row.get("value_state") or "").strip().upper()
            if not state_token:
                state_idx = 0
            else:
                state_idx = state_to_idx.setdefault(state_token, len(state_vocab))
                if state_idx == len(state_vocab):
                    state_vocab.append(state_token)
            state_ids.append(state_idx)

            value_type = (row.get("value_type") or "missing").strip().lower()
            value_type_idx = value_type_to_idx.get(value_type)
            if value_type_idx is None:
                value_type_idx = value_type_to_idx["missing"]
            value_type_ids.append(value_type_idx)

            numeric_raw = (row.get("value_numeric") or "").strip()
            if numeric_raw:
                try:
                    numeric_value = float(numeric_raw)
                except ValueError:
                    numeric_value = 0.0
                    numeric_present = 0
                else:
                    numeric_present = 1
            else:
                numeric_value = 0.0
                numeric_present = 0
            numeric_values.append(numeric_value)
            has_numeric.append(numeric_present)

            activity_name = (row.get("activity") or "").strip()
            if not activity_name:
                activity_idx = 0
            else:
                activity_idx = activity_to_idx.setdefault(activity_name, len(activity_vocab))
                if activity_idx == len(activity_vocab):
                    activity_vocab.append(activity_name)
            activity_ids.append(activity_idx)

    return EventData(
        timestamps=timestamps,
        sensor_ids=sensor_ids,
        state_ids=state_ids,
        value_type_ids=value_type_ids,
        numeric_values=numeric_values,
        has_numeric=has_numeric,
        activity_ids=activity_ids,
        sensor_vocab=sensor_vocab,
        state_vocab=state_vocab,
        value_type_vocab=value_type_vocab,
        activity_vocab=activity_vocab,
    )
