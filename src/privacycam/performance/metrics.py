"""Pipeline performance metrics collection."""

import collections
import threading
from typing import Dict, Any


class PipelineMetrics:
    """Thread-safe collection of metrics for pipeline stages.

    Uses string stage names instead of enum references to avoid
    circular imports with the pipeline package.
    """

    STAGES = ("CAPTURE", "DETECTION", "TRACKING", "MASKING", "ANONYMIZATION", "OUTPUT")

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._stage_times: Dict[str, collections.deque] = {
            stage: collections.deque(maxlen=window_size) for stage in self.STAGES
        }
        self._lock = threading.Lock()

    def record(self, stage, duration_ms: float) -> None:
        """Records a duration for a specific stage.

        Args:
            stage: A PipelineStage enum value or a string stage name.
            duration_ms: Duration in milliseconds.
        """
        key = stage.value.upper() if hasattr(stage, "value") else str(stage).upper()
        with self._lock:
            if key not in self._stage_times:
                self._stage_times[key] = collections.deque(maxlen=self.window_size)
            self._stage_times[key].append(duration_ms)

    def get_stage_avg(self, stage) -> float:
        key = stage.value.upper() if hasattr(stage, "value") else str(stage).upper()
        with self._lock:
            times = list(self._stage_times.get(key, []))
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_stage_p95(self, stage) -> float:
        key = stage.value.upper() if hasattr(stage, "value") else str(stage).upper()
        with self._lock:
            times = sorted(self._stage_times.get(key, []))
        if not times:
            return 0.0
        idx = int(len(times) * 0.95)
        return times[min(idx, len(times) - 1)]

    def get_stage_p99(self, stage) -> float:
        key = stage.value.upper() if hasattr(stage, "value") else str(stage).upper()
        with self._lock:
            times = sorted(self._stage_times.get(key, []))
        if not times:
            return 0.0
        idx = int(len(times) * 0.99)
        return times[min(idx, len(times) - 1)]

    def get_total_avg(self) -> float:
        total = 0.0
        with self._lock:
            for times in self._stage_times.values():
                t = list(times)
                if t:
                    total += sum(t) / len(t)
        return total

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all metrics."""
        summary = {}
        for stage_name in self.STAGES:
            summary[stage_name] = {
                "avg": self.get_stage_avg(stage_name),
                "p95": self.get_stage_p95(stage_name),
                "p99": self.get_stage_p99(stage_name),
            }
        summary["total_avg"] = self.get_total_avg()
        return summary

    def reset(self) -> None:
        """Resets all metrics."""
        with self._lock:
            for deq in self._stage_times.values():
                deq.clear()
