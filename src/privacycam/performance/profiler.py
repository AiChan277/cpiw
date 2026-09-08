import time
from typing import List, Dict, Any
import numpy as np

class Profiler:
    """Context manager for profiling code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self._start_ns: int = 0
        self._end_ns: int = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_ns = time.perf_counter_ns()

    @property
    def duration_ms(self) -> float:
        """Returns the profiled duration in milliseconds."""
        return (self._end_ns - self._start_ns) / 1_000_000.0

    @staticmethod
    def percentile(values: List[float], p: float) -> float:
        """Computes the given percentile from a list of values."""
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    @staticmethod
    def statistics(values: List[float]) -> Dict[str, Any]:
        """Computes basic statistics for a list of values."""
        if not values:
            return {"avg": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        
        return {
            "avg": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": Profiler.percentile(values, 95),
            "p99": Profiler.percentile(values, 99),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values)
        }

class ProfilerSession:
    """Collects and aggregates multiple profiler results."""
    
    def __init__(self, name: str):
        self.name = name
        self._durations: List[float] = []

    def record(self, duration_ms: float) -> None:
        """Records a new duration."""
        self._durations.append(duration_ms)

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics for the recorded durations."""
        return Profiler.statistics(self._durations)

    def reset(self) -> None:
        """Clears recorded durations."""
        self._durations.clear()
