import numpy as np
import cv2
from typing import List
from privacycam.detection.detection import FaceDetection
from privacycam.masking.geometry import expand_bbox

class MaskGenerator:
    """Generates optimized binary and feathered masks for detected face regions."""

    def __init__(self, expansion: float = 0.20, feather_radius: int = 15, use_ellipse: bool = True):
        self.expansion = expansion
        self.feather_radius = feather_radius
        self.use_ellipse = use_ellipse

    def generate(self, detection: FaceDetection, frame_height: int, frame_width: int) -> np.ndarray:
        return self.generate_multi([detection], frame_height, frame_width)

    def generate_multi(self, detections: List[FaceDetection], frame_height: int, frame_width: int) -> np.ndarray:
        """Generates a combined uint8 feathered mask (0-255) for fast integer blending."""
        if not detections:
            return np.zeros((frame_height, frame_width), dtype=np.uint8)

        binary_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        for det in detections:
            bbox = det.bounding_box
            expanded = expand_bbox(bbox, self.expansion, frame_width, frame_height)

            if self.use_ellipse:
                cx = int((expanded.x1 + expanded.x2) / 2)
                cy = int((expanded.y1 + expanded.y2) / 2)
                ax = max(1, int((expanded.x2 - expanded.x1) / 2))
                ay = max(1, int((expanded.y2 - expanded.y1) / 2))
                cv2.ellipse(binary_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
            else:
                x1, y1 = int(expanded.x1), int(expanded.y1)
                x2, y2 = int(expanded.x2), int(expanded.y2)
                cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255, -1)

        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1
            pad = int(self.feather_radius * 2.5)
            min_x = max(0, min(int(det.bounding_box.x1) for det in detections) - pad)
            min_y = max(0, min(int(det.bounding_box.y1) for det in detections) - pad)
            max_x = min(frame_width, max(int(det.bounding_box.x2) for det in detections) + pad)
            max_y = min(frame_height, max(int(det.bounding_box.y2) for det in detections) + pad)

            if max_x > min_x and max_y > min_y:
                roi = binary_mask[min_y:max_y, min_x:max_x]
                binary_mask[min_y:max_y, min_x:max_x] = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        return binary_mask

    def set_expansion(self, ratio: float) -> None:
        self.expansion = max(0.0, float(ratio))

    def set_feather(self, radius: int) -> None:
        self.feather_radius = max(0, int(radius))
