import dataclasses
from typing import Tuple, List

@dataclasses.dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    @property
    def area(self) -> float:
        if not self.is_valid():
            return 0.0
        return self.width * self.height

    def clip(self, frame_width: float, frame_height: float) -> "BoundingBox":
        return BoundingBox(
            x1=max(0.0, min(self.x1, frame_width)),
            y1=max(0.0, min(self.y1, frame_height)),
            x2=max(0.0, min(self.x2, frame_width)),
            y2=max(0.0, min(self.y2, frame_height)),
        )

    def expand(self, ratio: float) -> "BoundingBox":
        dw = self.width * ratio
        dh = self.height * ratio
        return BoundingBox(
            x1=self.x1 - dw / 2.0,
            y1=self.y1 - dh / 2.0,
            x2=self.x2 + dw / 2.0,
            y2=self.y2 + dh / 2.0,
        )

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def iou(self, other: "BoundingBox") -> float:
        if not self.is_valid() or not other.is_valid():
            return 0.0

        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        i_width = max(0.0, ix2 - ix1)
        i_height = max(0.0, iy2 - iy1)
        intersection = i_width * i_height

        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1

@dataclasses.dataclass
class FaceLandmarks:
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]
    mouth_left: tuple[float, float]
    mouth_right: tuple[float, float]

    def to_list(self) -> List[Tuple[float, float]]:
        return [
            self.left_eye,
            self.right_eye,
            self.nose,
            self.mouth_left,
            self.mouth_right,
        ]

@dataclasses.dataclass
class FaceDetection:
    bounding_box: BoundingBox
    confidence: float
    landmarks: FaceLandmarks | None = None
    timestamp: float = 0.0

    @property
    def is_valid(self) -> bool:
        return self.confidence > 0 and self.bounding_box.is_valid()

    def __repr__(self) -> str:
        return f"FaceDetection(bbox={self.bounding_box.to_int_tuple()}, conf={self.confidence:.2f})"
