import numpy as np
import cv2
from privacycam.detection.detection import BoundingBox

def feather_mask(mask: np.ndarray, radius: int = 15) -> np.ndarray:
    if radius <= 0:
        return (mask.astype(np.float32) / 255.0)
        
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return (blurred.astype(np.float32) / 255.0)

def adaptive_feather(mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    radius = int(max(w, h) * 0.1)
    return feather_mask(mask, radius)
