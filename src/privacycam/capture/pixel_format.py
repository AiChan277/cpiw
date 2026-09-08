"""Pixel format definitions and conversions."""

import enum
import cv2
import numpy as np
from typing import Any

class PixelFormat(enum.Enum):
    """Supported pixel formats."""
    BGR = "bgr"
    RGB = "rgb"
    RGBA = "rgba"
    YUV = "yuv"
    NV12 = "nv12"

def convert(image: np.ndarray, from_format: PixelFormat, to_format: PixelFormat) -> np.ndarray:
    """Convert an image between pixel formats."""
    if from_format == to_format:
        return image.copy()
        
    conversions = {
        (PixelFormat.BGR, PixelFormat.RGB): cv2.COLOR_BGR2RGB,
        (PixelFormat.BGR, PixelFormat.RGBA): cv2.COLOR_BGR2RGBA,
        (PixelFormat.RGB, PixelFormat.BGR): cv2.COLOR_RGB2BGR,
        (PixelFormat.RGB, PixelFormat.RGBA): cv2.COLOR_RGB2RGBA,
        (PixelFormat.RGBA, PixelFormat.BGR): cv2.COLOR_RGBA2BGR,
        (PixelFormat.RGBA, PixelFormat.RGB): cv2.COLOR_RGBA2RGB,
    }
    
    code = conversions.get((from_format, to_format))
    if code is not None:
        return cv2.cvtColor(image, code)
        
    raise ValueError(f"Unsupported conversion: {from_format} to {to_format}")
