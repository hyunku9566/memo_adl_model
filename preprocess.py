#!/usr/bin/env python3
"""Combine and normalize the raw CASAS-style sensor logs.

The script reads every CSV file found under `data/raw` (configurable),
parses the heterogeneous records, and emits a single chronologically
sorted CSV with a consistent schema so that downstream training code can
work with a predictable layout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import heapq
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

STATE_TOKENS = {
    "ON",
    "OFF",
    "OPEN",
    "CLOSE",
    "OPENED",
    "CLOSED",
    "PRESENT",
    "ABSENT",
}

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/processed/events.csv")
WARNING_SAMPLE_LIMIT = 15


@dataclass
class Event:
    """Normalized representation of a single sensor event."""

    timestamp: dt.datetime
    sensor: str
    raw_value: str
    value_type: str
    numeric_value: Optional[float]
    state: str
    activity: str
    activity_phase: str
    source_file: str

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp.isoformat(timespec="microseconds"),
            self.sensor,
            self.raw_value,
            self.value_type,
            format_numeric(self.numeric_value),
            self.state,
            self.activity,
            self.activity_phase,
            self.source_file,
        ]


@dataclass
class Stats:
    """Accumulates bookkeeping information while parsing."""

    raw_rows: int = 0
    emitted_rows: int = 0
    skipped_rows: int = 0
    skipped_by_reason: Counter = field(default_factory=Counter)
    rows_by_file: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    warnings_emitted: int = 0

    def record_skip(self, reason: str) -> None:
        self.skipped_rows += 1
        self.skipped_by_reason[reason] += 1

    def should_log_warning(self) -> bool:
        if self.warnings_emitted < WARNING_SAMPLE_LIMIT:
            self.warnings_emitted += 1
            return True
        return False

    def report(self) -> str:
        parts = [
            f"Raw rows: {self.raw_rows}",
            f"Emitted rows: {self.emitted_rows}",
        ]
        if self.skipped_rows:
            parts.append(
                f"Skipped rows: {self.skipped_rows} "
                f"({', '.join(f'{k}={v}' for k, v in self.skipped_by_reason.most_common())})"
            )
        return "\n".join(parts)


class FileCursor:
    """Streams normalized events from a single CSV file."""

    def __init__(self, path: Path, stats: Stats):
        self.path = path
        self.stats = stats
        self.handle = path.open(newline="")
        self.reader = csv.reader(self.handle)
        self.line_no = 0

    def __iter__(self) -> "FileCursor":
        return self

    def __next__(self) -> Event:
        for row in self.reader:
            self.line_no += 1
            event = build_event(row, self.path.name, self.line_no, self.stats)
            if event is None:
                continue
            return event
        self.close()
        raise StopIteration

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw sensor CSV logs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory that holds the raw CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the processed CSV (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of events to emit (useful for quick tests).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_dir = args.input_dir
    if not input_dir.is_dir():
        logging.error("Input directory %s does not exist or is not a directory.", input_dir)
        return 1

    csv_files = sorted(p for p in input_dir.iterdir() if p.suffix == ".csv" and p.is_file())
    if not csv_files:
        logging.error("No CSV files found under %s", input_dir)
        return 1

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    events = iter_sorted_events(csv_files, stats)
    write_events(events, output_path, args.limit, stats)

    logging.info("Wrote %s rows to %s", stats.emitted_rows, output_path)
    if stats.skipped_rows:
        logging.info(
            "Skipped %s rows (%s)",
            stats.skipped_rows,
            ", ".join(f"{k}={v}" for k, v in stats.skipped_by_reason.most_common()),
        )
    return 0


def iter_sorted_events(files: Sequence[Path], stats: Stats) -> Iterator[Event]:
    """Yield events from every file in chronological order."""

    cursors: List[FileCursor] = []
    heap: List[Tuple[dt.datetime, int, Event]] = []

    try:
        for idx, path in enumerate(files):
            cursor = FileCursor(path, stats)
            cursors.append(cursor)
            try:
                event = next(cursor)
            except StopIteration:
                continue
            heapq.heappush(heap, (event.timestamp, idx, event))

        while heap:
            _, idx, event = heapq.heappop(heap)
            yield event
            cursor = cursors[idx]
            try:
                next_event = next(cursor)
            except StopIteration:
                continue
            heapq.heappush(heap, (next_event.timestamp, idx, next_event))
    finally:
        for cursor in cursors:
            cursor.close()


def write_events(events: Iterator[Event], output_path: Path, limit: Optional[int], stats: Stats) -> None:
    header = [
        "timestamp",
        "sensor",
        "value_raw",
        "value_type",
        "value_numeric",
        "value_state",
        "activity",
        "activity_phase",
        "source_file",
    ]
    with output_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for event in events:
            writer.writerow(event.to_csv_row())
            stats.emitted_rows += 1
            if limit is not None and stats.emitted_rows >= limit:
                break


def build_event(row: Sequence[str], source: str, line_no: int, stats: Stats) -> Optional[Event]:
    stats.raw_rows += 1
    stats.rows_by_file[source] += 1

    if len(row) < 2:
        log_skip("too_few_columns", source, line_no, stats)
        return None

    date_str = row[0].strip()
    time_str = row[1].strip()
    if not date_str or not time_str:
        log_skip("missing_timestamp", source, line_no, stats)
        return None

    try:
        timestamp = parse_timestamp(date_str, time_str)
    except ValueError:
        log_skip("bad_timestamp", source, line_no, stats)
        return None

    sensor = row[2].strip() if len(row) >= 3 else ""
    if not sensor:
        log_skip("missing_sensor", source, line_no, stats)
        return None

    value = row[3].strip() if len(row) >= 4 else ""
    activity_token = row[4].strip() if len(row) >= 5 else ""
    raw_value, value_type, numeric_value, state = normalize_value(value)
    activity, activity_phase = normalize_activity(activity_token)
    
    # Extract activity from filename if not provided in row
    # Filename format: p<person>.<task>.csv (e.g., p01.t1.csv -> task="t1")
    if not activity:
        activity = _extract_activity_from_filename(source)

    return Event(
        timestamp=timestamp,
        sensor=sensor,
        raw_value=raw_value,
        value_type=value_type,
        numeric_value=numeric_value,
        state=state,
        activity=activity,
        activity_phase=activity_phase,
        source_file=source,
    )


def log_skip(reason: str, source: str, line_no: int, stats: Stats) -> None:
    stats.record_skip(reason)
    if stats.should_log_warning():
        logging.warning("Skipping row (%s) at %s:%s", reason, source, line_no)


def parse_timestamp(date_str: str, time_str: str) -> dt.datetime:
    combined = f"{date_str} {time_str}"
    for fmt in TIMESTAMP_FORMATS:
        try:
            return dt.datetime.strptime(combined, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp {combined!r}")


def normalize_value(value: str) -> Tuple[str, str, Optional[float], str]:
    if not value:
        return "", "missing", None, ""

    raw = value.strip()
    if not raw:
        return "", "missing", None, ""

    upper = raw.upper()
    if upper in STATE_TOKENS:
        return raw, "state", None, upper

    try:
        numeric = float(raw)
    except ValueError:
        return raw, "string", None, ""
    return raw, "numeric", numeric, ""


def normalize_activity(token: str) -> Tuple[str, str]:
    if not token:
        return "", ""

    cleaned = token.strip()
    if not cleaned:
        return "", ""
    if "=" not in cleaned:
        return cleaned, ""

    name, phase = cleaned.split("=", 1)
    phase = phase.strip().strip('"')
    return name.strip(), phase


def format_numeric(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:g}"


def _extract_activity_from_filename(source: str) -> str:
    """
    Extract activity label from filename.
    
    Expected format: p<person>.<task>.csv (e.g., p01.t1.csv)
    Returns: task identifier (e.g., "t1") or empty string if not found.
    """
    import re
    # Match pattern like "p01.t1.csv" or "p01.t1"
    match = re.search(r'\.([tp]\d+)', source)
    if match:
        return match.group(1)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
