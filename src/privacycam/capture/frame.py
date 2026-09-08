"""Frame representation."""

import dataclasses
import numpy as np
import time
from privacycam.capture.pixel_format import PixelFormat, convert

@dataclasses.dataclass
class Frame:
    """A video frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_id: int
    width: int
    height: int
    pixel_format: PixelFormat
    
    @classmethod
    def from_ndarray(cls, image: np.ndarray, frame_id: int, pixel_format: PixelFormat = PixelFormat.BGR) -> "Frame":
        """Create a Frame from a numpy array."""
        height, width = image.shape[:2]
        return cls(
            image=image,
            timestamp=time.time(),
            frame_id=frame_id,
            width=width,
            height=height,
            pixel_format=pixel_format
        )
        
    def copy(self) -> "Frame":
        """Create a deep copy of this frame."""
        return dataclasses.replace(self, image=self.image.copy())
        
    def to_rgb(self) -> "Frame":
        """Return a new frame in RGB format."""
        if self.pixel_format == PixelFormat.RGB:
            return self.copy()
        new_image = convert(self.image, self.pixel_format, PixelFormat.RGB)
        return dataclasses.replace(self, image=new_image, pixel_format=PixelFormat.RGB)
        
    def to_bgr(self) -> "Frame":
        """Return a new frame in BGR format."""
        if self.pixel_format == PixelFormat.BGR:
            return self.copy()
        new_image = convert(self.image, self.pixel_format, PixelFormat.BGR)
        return dataclasses.replace(self, image=new_image, pixel_format=PixelFormat.BGR)
        
    @property
    def is_valid(self) -> bool:
        """Check if the frame contains valid image data."""
        return self.image is not None and self.image.size > 0
