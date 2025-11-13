#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IHS 데이터셋 전처리 스크립트
============================

IHS 센서 데이터를 기존 CASAS 형식(events.csv)으로 변환합니다.

입력 형식 (IHS CSV):
    date,time,sensor,value[,activity[=phase]]
    예: 2015-10-08,00:10:03.360213,MainDoor,20.5
        2015-10-08,01:24:41.157990,BedroomABed,ON,Sleep
        2015-10-08,01:24:43.783981,BedroomABed,OFF,Sleep="begin"

출력 형식 (events.csv):
    timestamp,sensor,raw_value,value_type,numeric_value,state,activity,activity_phase,source_file
    예: 2015-10-08T00:10:03.360213,MainDoor,20.5,numeric,20.5,,,ihs06.csv

사용법:
    python preprocess_ihs.py \
        --input-dir data/ihsdata/raw \
        --output data/ihsdata/processed/events.csv
"""

import argparse
import csv
import datetime as dt
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# State tokens (ON/OFF 등)
STATE_TOKENS = {
    "ON", "OFF", "OPEN", "CLOSE", "OPENED", "CLOSED",
    "PRESENT", "ABSENT", "TRUE", "FALSE"
}

# Activity name normalization (대소문자 통일)
ACTIVITY_NORMALIZE = {
    "Sleep_Out_of_Bed": "Sleep_Out_Of_Bed",  # 소문자 o -> 대문자 O
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Event:
    """정규화된 이벤트"""
    timestamp: dt.datetime
    sensor: str
    raw_value: str
    value_type: str  # "state", "numeric", "unknown"
    numeric_value: Optional[float]
    state: str
    activity: str
    activity_phase: str  # "begin", "end", "" (empty)
    source_file: str

    def to_csv_row(self) -> List[str]:
        return [
            self.timestamp.isoformat(timespec="microseconds"),
            self.sensor,
            self.raw_value,
            self.value_type,
            f"{self.numeric_value:.6f}" if self.numeric_value is not None else "",
            self.state,  # value_state column for compatibility
            self.activity,
            self.activity_phase,
            self.source_file,
        ]


@dataclass
class Stats:
    """통계 정보"""
    raw_rows: int = 0
    emitted_rows: int = 0
    skipped_rows: int = 0
    skipped_by_reason: Counter = field(default_factory=Counter)
    rows_by_file: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record_skip(self, reason: str) -> None:
        self.skipped_rows += 1
        self.skipped_by_reason[reason] += 1

    def report(self) -> str:
        parts = [
            f"Raw rows: {self.raw_rows:,}",
            f"Emitted rows: {self.emitted_rows:,}",
        ]
        if self.skipped_rows:
            parts.append(
                f"Skipped rows: {self.skipped_rows:,} "
                f"({', '.join(f'{k}={v}' for k, v in self.skipped_by_reason.most_common())})"
            )
        parts.append("\nRows by file:")
        for fname, count in sorted(self.rows_by_file.items()):
            parts.append(f"  {fname}: {count:,}")
        return "\n".join(parts)


def parse_timestamp(date_str: str, time_str: str) -> Optional[dt.datetime]:
    """
    IHS 타임스탬프 파싱: date,time 두 칼럼을 합쳐 datetime으로 변환
    
    예:
        date_str = "2015-10-08"
        time_str = "00:10:03.360213"
    """
    try:
        ts_str = f"{date_str} {time_str}"
        # Try with microseconds first
        try:
            return dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            return dt.datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"Failed to parse timestamp '{date_str} {time_str}': {e}")
        return None


def parse_activity_field(activity_field: str) -> tuple[str, str]:
    """
    활동 필드 파싱
    
    형식:
        "Sleep" -> ("Sleep", "")
        "Sleep=\"begin\"" -> ("Sleep", "begin")
        "Sleep=\"end\"" -> ("Sleep", "end")
        "Bed_Toilet_Transition" -> ("Bed_Toilet_Transition", "")
    
    Returns:
        (activity_name, phase)
    """
    if not activity_field or activity_field.strip() == "":
        return "", ""
    
    # Check for phase annotation (activity="phase")
    if "=" in activity_field:
        parts = activity_field.split("=", 1)
        activity = parts[0].strip()
        phase = parts[1].strip().strip('"\'')  # Remove quotes
    else:
        activity = activity_field.strip()
        phase = ""
    
    # Normalize activity name (대소문자 통일)
    activity = ACTIVITY_NORMALIZE.get(activity, activity)
    
    return activity, phase


def classify_value(raw_value: str) -> tuple[str, Optional[float], str]:
    """
    센서 값 분류
    
    Returns:
        (value_type, numeric_value, state)
        - value_type: "state", "numeric", "unknown"
        - numeric_value: float or None
        - state: "ON", "OFF", etc. or ""
    """
    val_upper = raw_value.strip().upper()
    
    # State token
    if val_upper in STATE_TOKENS:
        return "state", None, val_upper
    
    # Try numeric
    try:
        num = float(raw_value)
        return "numeric", num, ""
    except ValueError:
        pass
    
    # Unknown
    return "unknown", None, ""


def parse_ihs_row(row: List[str], source_file: str, stats: Stats) -> Optional[Event]:
    """
    IHS CSV 행 파싱
    
    형식: date,time,sensor,value[,activity[=phase]]
    
    Returns:
        Event or None (skip)
    """
    stats.raw_rows += 1
    
    # Minimum 4 columns required
    if len(row) < 4:
        stats.record_skip("too_few_columns")
        return None
    
    date_str = row[0].strip()
    time_str = row[1].strip()
    sensor = row[2].strip()
    raw_value = row[3].strip()
    
    # Parse timestamp
    timestamp = parse_timestamp(date_str, time_str)
    if timestamp is None:
        stats.record_skip("invalid_timestamp")
        return None
    
    # Parse activity (optional 5th column)
    activity = ""
    activity_phase = ""
    if len(row) >= 5 and row[4].strip():
        activity, activity_phase = parse_activity_field(row[4])
    
    # Classify value
    value_type, numeric_value, state = classify_value(raw_value)
    
    # Skip empty sensor names
    if not sensor:
        stats.record_skip("empty_sensor")
        return None
    
    event = Event(
        timestamp=timestamp,
        sensor=sensor,
        raw_value=raw_value,
        value_type=value_type,
        numeric_value=numeric_value,
        state=state,
        activity=activity,
        activity_phase=activity_phase,
        source_file=source_file
    )
    
    stats.emitted_rows += 1
    stats.rows_by_file[source_file] += 1
    
    return event


def process_ihs_files(input_dir: Path, output_path: Path) -> None:
    """
    IHS 데이터 파일들을 읽어 통합된 events.csv 생성
    
    Args:
        input_dir: IHS CSV 파일들이 있는 디렉토리
        output_path: 출력 events.csv 경로
    """
    logger.info(f"Scanning input directory: {input_dir}")
    
    # Find all CSV files
    csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return
    
    logger.info(f"Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")
    
    stats = Stats()
    events: List[Event] = []
    
    # Parse all files
    for csv_file in csv_files:
        logger.info(f"Processing {csv_file.name}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    event = parse_ihs_row(row, csv_file.name, stats)
                    if event:
                        events.append(event)
        except Exception as e:
            logger.error(f"Failed to read {csv_file.name}: {e}")
            continue
    
    logger.info(f"Parsed {len(events):,} events total")
    
    # Sort by timestamp
    logger.info("Sorting events by timestamp...")
    events.sort(key=lambda e: e.timestamp)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing to {output_path}...")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header (use value_state for compatibility with rich_features.py)
        writer.writerow([
            "timestamp", "sensor", "value_raw", "value_type",
            "value_numeric", "value_state", "activity", "activity_phase", "source_file"
        ])
        
        # Data
        for event in events:
            writer.writerow(event.to_csv_row())
    
    logger.info(f"✅ Output written to {output_path}")
    logger.info("\n" + stats.report())


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess IHS dataset into unified events.csv format"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/ihsdata/raw'),
        help='Directory containing IHS CSV files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/ihsdata/processed/events.csv'),
        help='Output events.csv path'
    )
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1
    
    process_ihs_files(args.input_dir, args.output)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
