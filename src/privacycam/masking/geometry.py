import numpy as np
import cv2
from privacycam.detection.detection import BoundingBox

def expand_bbox(bbox: BoundingBox, ratio: float, frame_width: int, frame_height: int) -> BoundingBox:
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    cx = bbox.x1 + w / 2
    cy = bbox.y1 + h / 2
    
    new_w = w * (1.0 + ratio)
    new_h = h * (1.0 + ratio)
    
    return clip_bbox(BoundingBox(
        x1=cx - new_w / 2,
        y1=cy - new_h / 2,
        x2=cx + new_w / 2,
        y2=cy + new_h / 2
    ), frame_width, frame_height)

def clip_bbox(bbox: BoundingBox, frame_width: int, frame_height: int) -> BoundingBox:
    return BoundingBox(
        x1=max(0.0, min(float(frame_width), bbox.x1)),
        y1=max(0.0, min(float(frame_height), bbox.y1)),
        x2=max(0.0, min(float(frame_width), bbox.x2)),
        y2=max(0.0, min(float(frame_height), bbox.y2))
    )

def create_elliptical_mask(bbox: BoundingBox, frame_height: int, frame_width: int) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    center = (int((bbox.x1 + bbox.x2) / 2), int((bbox.y1 + bbox.y2) / 2))
    axes = (int((bbox.x2 - bbox.x1) / 2), int((bbox.y2 - bbox.y1) / 2))
    if axes[0] > 0 and axes[1] > 0:
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask

def create_rectangular_mask(bbox: BoundingBox, frame_height: int, frame_width: int) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    cv2.rectangle(mask, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), 255, -1)
    return mask

def bbox_aspect_ratio(bbox: BoundingBox) -> float:
    w = bbox.x2 - bbox.x1
    h = bbox.y2 - bbox.y1
    if h == 0:
        return 0.0
    return w / h

def scale_bbox(bbox: BoundingBox, scale_x: float, scale_y: float) -> BoundingBox:
    return BoundingBox(
        x1=bbox.x1 * scale_x,
        y1=bbox.y1 * scale_y,
        x2=bbox.x2 * scale_x,
        y2=bbox.y2 * scale_y
    )
