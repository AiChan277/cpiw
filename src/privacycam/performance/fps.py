import time
import collections
from typing import Dict, Any

class FPSCounter:
    """Calculates frames per second over a moving window."""
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._timestamps = collections.deque(maxlen=window_size)
        self._dropped = 0

    def tick(self) -> None:
        """Records a frame timestamp."""
        self._timestamps.append(time.time())

    @property
    def fps(self) -> float:
        """Calculates current FPS based on timestamps in the window."""
        if len(self._timestamps) < 2:
            return 0.0
        
        time_diff = self._timestamps[-1] - self._timestamps[0]
        if time_diff == 0:
            return 0.0
            
        return (len(self._timestamps) - 1) / time_diff

    def record_drop(self) -> None:
        """Increments the dropped frame counter."""
        self._dropped += 1

    @property
    def dropped_count(self) -> int:
        """Returns the number of dropped frames."""
        return self._dropped

    def reset(self) -> None:
        """Clears timestamps and drops."""
        self._timestamps.clear()
        self._dropped = 0


class PipelineFPS:
    """Manages FPS counters for different pipeline stages."""

    def __init__(self, window_size: int = 60):
        self.input_fps = FPSCounter(window_size)
        self.processing_fps = FPSCounter(window_size)
        self.output_fps = FPSCounter(window_size)

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of FPS across the pipeline."""
        return {
            "input": {
                "fps": self.input_fps.fps,
                "dropped": self.input_fps.dropped_count
            },
            "processing": {
                "fps": self.processing_fps.fps,
                "dropped": self.processing_fps.dropped_count
            },
            "output": {
                "fps": self.output_fps.fps,
                "dropped": self.output_fps.dropped_count
            }
        }
