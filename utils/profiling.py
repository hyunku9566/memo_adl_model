from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Timer:
    """Simple wall-clock timer for profiling code sections."""

    label: str
    start_time: float = field(default_factory=time.perf_counter)
    elapsed: Optional[float] = None

    def stop(self) -> float:
        if self.elapsed is None:
            self.elapsed = time.perf_counter() - self.start_time
        return self.elapsed

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class TimerGroup:
    """Collects multiple timers for aggregate reporting."""

    def __init__(self) -> None:
        self.timers: Dict[str, Timer] = {}

    def start(self, label: str) -> Timer:
        timer = Timer(label)
        self.timers[label] = timer
        return timer

    def stop(self, label: str) -> float:
        timer = self.timers.get(label)
        if timer is None:
            raise KeyError(f"No timer named {label!r}")
        return timer.stop()

    def summary(self) -> str:
        lines = []
        for label, timer in self.timers.items():
            if timer.elapsed is None:
                continue
            lines.append(f"{label}: {timer.elapsed:.2f}s")
        return "\n".join(lines)
