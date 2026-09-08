from enum import Enum
from dataclasses import dataclass
import time

class PipelineStage(Enum):
    CAPTURE = "capture"
    DETECTION = "detection"
    TRACKING = "tracking"
    MASKING = "masking"
    ANONYMIZATION = "anonymization"
    OUTPUT = "output"

@dataclass
class StageResult:
    stage: PipelineStage
    duration_ms: float
    success: bool
    error: str | None = None

class StageTiming:
    """Context manager for timing pipeline stages."""
    
    def __init__(self, stage: PipelineStage):
        self.stage = stage
        self._start_ns: int = 0
        self._end_ns: int = 0

    def __enter__(self):
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end_ns = time.perf_counter_ns()

    @property
    def duration_ms(self) -> float:
        """Returns the duration of the stage in milliseconds."""
        return (self._end_ns - self._start_ns) / 1_000_000.0
