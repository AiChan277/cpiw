import numpy as np
from privacycam.anonymization.anonymizer import Anonymizer
from privacycam.anonymization.modes import AnonymizationMode
from privacycam.detection.detection import BoundingBox

class SolidAnonymizer(Anonymizer):
    """Anonymizes face regions with a solid color overlay."""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)):
        self.set_color(color)

    def anonymize(self, frame: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))

        margin_w = max(int((x2 - x1) * 0.4), 16)
        margin_h = max(int((y2 - y1) * 0.4), 16)
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2]
        solid_roi = np.full_like(roi, self.color)
        mask_roi = mask[y1:y2, x1:x2]

        if mask_roi.shape[:2] != roi.shape[:2]:
            return frame

        # Vectorized C++ OpenCV alpha blend (<0.3ms)
        import cv2
        if mask_roi.ndim == 2:
            m3 = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        else:
            m3 = mask_roi

        if m3.dtype != np.uint8:
            m3 = (m3 * 255.0).clip(0, 255).astype(np.uint8)

        inv_m3 = cv2.bitwise_not(m3)
        fg = cv2.multiply(solid_roi, m3, scale=1.0 / 255.0)
        bg = cv2.multiply(roi, inv_m3, scale=1.0 / 255.0)
        frame[y1:y2, x1:x2] = cv2.add(fg, bg)
        return frame

    @property
    def mode(self) -> AnonymizationMode:
        return AnonymizationMode.SOLID

    def set_color(self, color: tuple[int, int, int]) -> None:
        self.color = color
