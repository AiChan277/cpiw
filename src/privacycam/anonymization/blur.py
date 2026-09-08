import numpy as np
import cv2
from privacycam.anonymization.anonymizer import Anonymizer
from privacycam.anonymization.modes import AnonymizationMode
from privacycam.detection.detection import BoundingBox

class GaussianBlurAnonymizer(Anonymizer):
    """Anonymizes face regions using optimized Gaussian blur with pyramidal acceleration."""

    def __init__(self, strength: int = 51, fast_mode: bool = True):
        self.fast_mode = fast_mode
        self.set_strength(strength)

    def anonymize(self, frame: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))

        # Margin expansion to cover feathered boundaries
        margin_w = max(int((x2 - x1) * 0.35), 14)
        margin_h = max(int((y2 - y1) * 0.35), 14)
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)

        rw, rh = x2 - x1, y2 - y1
        if rw <= 4 or rh <= 4:
            return frame

        roi = frame[y1:y2, x1:x2]

        # 1. Fast Pyramidal Blur (<0.2ms) or Standard Gaussian Blur
        if self.fast_mode:
            scale = max(4, min(rw // 16, rh // 16, 8))
            sw, sh = max(4, rw // scale), max(4, rh // scale)
            small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
            ksize = max(5, (self.strength // scale) | 1)
            small_blurred = cv2.GaussianBlur(small, (ksize, ksize), 0)
            roi_blurred = cv2.resize(small_blurred, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            ksize = self.strength if self.strength % 2 == 1 else self.strength + 1
            roi_blurred = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # 2. Extract mask ROI
        mask_roi = mask[y1:y2, x1:x2]
        if mask_roi.shape[:2] != (rh, rw):
            return frame

        # 3. Vectorized C++ OpenCV alpha blend (<0.3ms)
        if mask_roi.ndim == 2:
            m3 = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        else:
            m3 = mask_roi

        if m3.dtype != np.uint8:
            m3 = (m3 * 255.0).clip(0, 255).astype(np.uint8)

        inv_m3 = cv2.bitwise_not(m3)
        fg = cv2.multiply(roi_blurred, m3, scale=1.0 / 255.0)
        bg = cv2.multiply(roi, inv_m3, scale=1.0 / 255.0)
        frame[y1:y2, x1:x2] = cv2.add(fg, bg)
        return frame

    @property
    def mode(self) -> AnonymizationMode:
        return AnonymizationMode.BLUR

    def set_strength(self, value: int) -> None:
        val = max(3, int(value))
        self.strength = val if val % 2 == 1 else val + 1

    def set_fast_mode(self, enabled: bool) -> None:
        self.fast_mode = enabled
