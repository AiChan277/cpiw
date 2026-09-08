from enum import Enum
from dataclasses import dataclass, field
from privacycam.detection.detection import BoundingBox, FaceDetection

class TrackState(Enum):
    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    DEAD = "dead"

@dataclass
class Track:
    track_id: int
    bounding_box: BoundingBox
    confidence: float
    age: int = 0
    frames_since_seen: int = 0
    velocity: tuple[float, float] = (0.0, 0.0)
    state: TrackState = TrackState.NEW
    hit_count: int = 0
    smoothing_alpha: float = 0.70

    def predict(self) -> BoundingBox:
        """Zero-drift position hold. In webcam scenarios, keeping the last confirmed
        position completely eliminates bounding boxes drifting into background furniture."""
        self.velocity = (0.0, 0.0)
        return self.bounding_box

    def update(self, detection: FaceDetection, min_hits: int = 1) -> None:
        """Smoothly updates track with matched detection using Exponential Moving Average."""
        # Anti-jitter: Exponential Moving Average coordinate smoothing
        # If newly matched after being lost, snap faster (min 0.95)
        # Otherwise use configured smoothing_alpha
        alpha = self.smoothing_alpha if self.frames_since_seen <= 1 else min(0.95, self.smoothing_alpha + 0.20)
        b1 = detection.bounding_box
        b0 = self.bounding_box
        smooth_x1 = alpha * b1.x1 + (1.0 - alpha) * b0.x1
        smooth_y1 = alpha * b1.y1 + (1.0 - alpha) * b0.y1
        smooth_x2 = alpha * b1.x2 + (1.0 - alpha) * b0.x2
        smooth_y2 = alpha * b1.y2 + (1.0 - alpha) * b0.y2

        self.bounding_box = BoundingBox(smooth_x1, smooth_y1, smooth_x2, smooth_y2)
        self.confidence = detection.confidence
        self.frames_since_seen = 0
        self.hit_count += 1
        self.age += 1
        self.velocity = (0.0, 0.0)
        self.state = TrackState.ACTIVE

    def mark_lost(self) -> None:
        self.frames_since_seen += 1
        self.state = TrackState.LOST
        self.age += 1

    def mark_dead(self) -> None:
        self.state = TrackState.DEAD

    @property
    def is_confirmed(self) -> bool:
        return self.hit_count >= 1

    @property
    def is_dead(self) -> bool:
        return self.state == TrackState.DEAD
