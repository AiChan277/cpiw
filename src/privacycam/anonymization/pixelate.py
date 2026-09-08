import numpy as np
import cv2
from privacycam.anonymization.anonymizer import Anonymizer
from privacycam.anonymization.modes import AnonymizationMode
from privacycam.detection.detection import BoundingBox

class PixelateAnonymizer(Anonymizer):
    """Anonymizes face regions using accelerated pixelation."""

    def __init__(self, blocks: int = 12):
        self.set_blocks(blocks)

    def anonymize(self, frame: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
        x2, y2 = min(w, int(bbox.x2)), min(h, int(bbox.y2))

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
        blocks = max(2, min(self.blocks, rw, rh))
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

        mask_roi = mask[y1:y2, x1:x2]
        if mask_roi.shape[:2] != (rh, rw):
            return frame

        # Vectorized C++ OpenCV alpha blend (<0.3ms)
        if mask_roi.ndim == 2:
            m3 = cv2.cvtColor(mask_roi, cv2.COLOR_GRAY2BGR)
        else:
            m3 = mask_roi

        if m3.dtype != np.uint8:
            m3 = (m3 * 255.0).clip(0, 255).astype(np.uint8)

        inv_m3 = cv2.bitwise_not(m3)
        fg = cv2.multiply(pixelated_roi, m3, scale=1.0 / 255.0)
        bg = cv2.multiply(roi, inv_m3, scale=1.0 / 255.0)
        frame[y1:y2, x1:x2] = cv2.add(fg, bg)
        return frame

    @property
    def mode(self) -> AnonymizationMode:
        return AnonymizationMode.PIXELATE

    def set_blocks(self, value: int) -> None:
        self.blocks = max(2, int(value))
