import cv2
import numpy as np
from typing import Tuple, List, Optional
from privacycam.detection.detection import FaceDetection, BoundingBox

class Preprocessor:
    """Image preprocessing for detection models."""

    def __init__(self, target_width: int = 672, target_height: int = 384):
        self.target_width = target_width
        self.target_height = target_height

    def resize(self, image: np.ndarray, width: int, height: int, keep_aspect: bool = False) -> np.ndarray:
        if keep_aspect:
            h, w = image.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            
            result = np.zeros((height, width, 3), dtype=np.uint8)
            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2
            result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            return result
        else:
            return cv2.resize(image, (width, height))

    def letterbox(self, image: np.ndarray, target_w: int, target_h: int) -> Tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        result = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        result[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        return result, scale, pad_x, pad_y

    def normalize(self, image: np.ndarray, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> np.ndarray:
        img_float = image.astype(np.float32)
        if mean is not None:
            img_float -= mean
        if std is not None:
            img_float /= std
        return img_float

    def to_chw(self, image: np.ndarray) -> np.ndarray:
        return image.transpose((2, 0, 1))

    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """Converts an image to an NCHW float32 tensor."""
        if image.shape[1] == self.target_width and image.shape[0] == self.target_height:
            resized = image
        else:
            resized = self.resize(image, self.target_width, self.target_height, keep_aspect=False)
        chw = self.to_chw(resized)
        batch = np.expand_dims(chw, axis=0).astype(np.float32)
        return batch

    def scale_detections(self, detections: List[FaceDetection], original_w: int, original_h: int, input_w: int, input_h: int) -> List[FaceDetection]:
        scaled_detections = []
        for det in detections:
            bbox = det.bounding_box
            x1 = bbox.x1 * original_w / input_w
            y1 = bbox.y1 * original_h / input_h
            x2 = bbox.x2 * original_w / input_w
            y2 = bbox.y2 * original_h / input_h
            
            scaled_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            scaled_det = FaceDetection(
                bounding_box=scaled_bbox,
                confidence=det.confidence,
                landmarks=det.landmarks,
                timestamp=det.timestamp
            )
            scaled_detections.append(scaled_det)
        return scaled_detections
