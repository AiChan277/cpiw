"""Manager for multiple cameras."""

import threading
import logging
from typing import Optional, List, Dict
import cv2

from privacycam.config import CameraConfig
from privacycam.capture.camera import Camera

logger = logging.getLogger(__name__)

class CameraManager:
    """Manages enumeration and switching of cameras."""
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self._lock = threading.Lock()
        self._current_camera: Optional[Camera] = None
        
    def enumerate_cameras(self) -> List[Dict[str, any]]:
        """Probe indices 0-9 to find available cameras."""
        cameras = []
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    "index": index,
                    "name": f"Camera {index}",
                    "resolution": (width, height)
                })
                cap.release()
        return cameras
        
    def get_camera(self, index: int) -> Camera:
        """Get a camera instance for the given index."""
        config = self.config
        config.device_index = index
        return Camera(index, config)
        
    def get_default_camera(self) -> Camera:
        """Get a camera instance for the default index."""
        return self.get_camera(self.config.device_index)
        
    def switch_camera(self, index: int) -> Camera:
        """Switch to a new camera index."""
        with self._lock:
            if self._current_camera is not None:
                self._current_camera.close()
                
            self._current_camera = self.get_camera(index)
            self._current_camera.open()
            return self._current_camera
            
    def reconnect(self) -> bool:
        """Attempt to reconnect to the current camera."""
        with self._lock:
            if self._current_camera is None:
                return False
                
            idx = self._current_camera.device_index
            self._current_camera.close()
            
            try:
                self._current_camera = self.get_camera(idx)
                self._current_camera.open()
                return True
            except Exception as e:
                logger.error(f"Failed to reconnect camera {idx}: {e}")
                return False
                
    @property
    def current_camera(self) -> Optional[Camera]:
        """Get the currently active camera."""
        with self._lock:
            return self._current_camera
            
    @property
    def available_cameras(self) -> List[Dict[str, any]]:
        """Get list of available cameras."""
        return self.enumerate_cameras()
        
    def close(self) -> None:
        """Close current camera."""
        with self._lock:
            if self._current_camera is not None:
                self._current_camera.close()
                self._current_camera = None
