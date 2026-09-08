from typing import List, Optional
from privacycam.tracking.track import Track

class DetectionScheduler:
    """Schedules face detection to run at specific intervals or conditions."""
    
    def __init__(self, detection_interval: int = 3, confidence_threshold: float = 0.3):
        self.detection_interval = detection_interval
        self.confidence_threshold = confidence_threshold
        self._frame_count: int = 0
        self._force_detection: bool = True

    def should_detect(self, tracks: Optional[List[Track]] = None) -> bool:
        """
        Determines whether detection should run for the current frame.
        """
        if self._force_detection:
            self._force_detection = False
            return True

        if not tracks:
            return True

        if any(track.confidence < self.confidence_threshold for track in tracks):
            return True

        if self._frame_count % self.detection_interval == 0:
            return True

        return False

    def tick(self) -> None:
        """Increments the internal frame counter."""
        self._frame_count += 1

    def force_next_detection(self) -> None:
        """Forces detection to run on the next frame."""
        self._force_detection = True

    def reset(self) -> None:
        """Resets counters and flags."""
        self._frame_count = 0
        self._force_detection = True

    def set_interval(self, interval: int) -> None:
        """Updates the detection interval."""
        self.detection_interval = interval
