"""Camera wrapper with asynchronous, zero-latency background frame grabbing."""

import cv2
import logging
import platform
import threading
import time
from typing import Optional, Tuple, List
from privacycam.config import CameraConfig
from privacycam.exceptions import CameraInitializationError, CameraDisconnectedError
from privacycam.capture.frame import Frame
from privacycam.capture.pixel_format import PixelFormat

logger = logging.getLogger(__name__)


class Camera:
    """High-performance threaded camera capture wrapper.

    Runs a background daemon thread that continuously grabs the freshest
    hardware frame from the sensor. This eliminates USB driver buffer lag
    and ensures reading from the camera takes < 0.1 ms without blocking.
    """

    def __init__(self, device_index: int, config: CameraConfig):
        self.device_index = device_index
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_counter = 0

        # Threaded capture state
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[Frame] = None
        self._new_frame_event = threading.Event()

    def open(self) -> bool:
        """Open the camera with configured settings and launch capture thread."""
        logger.info(f"Opening camera {self.device_index}")

        # On Windows, cv2.CAP_DSHOW provides significantly faster startup and lower lag
        if platform.system() == "Windows":
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            # Request MJPG for high framerates at 720p/1080p
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        else:
            self._cap = cv2.VideoCapture(self.device_index)

        if not self._cap.isOpened():
            # Fallback to default backend if DSHOW fails
            self._cap = cv2.VideoCapture(self.device_index)
            if not self._cap.isOpened():
                raise CameraInitializationError(f"Failed to open camera {self.device_index}")

        # Set resolution & framerate
        self.set_resolution(self.config.width, self.config.height)
        self.set_fps(self.config.fps)

        # Start background capture thread
        self._running = True
        self._new_frame_event.clear()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

        # Wait up to 1.5 seconds for first frame
        self._new_frame_event.wait(timeout=1.5)
        return True

    def _capture_worker(self) -> None:
        """Continuous background loop grabbing latest frames from hardware sensor."""
        fmt = PixelFormat(self.config.pixel_format)

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.01)
                continue

            ret, image = self._cap.read()
            if ret and image is not None and image.size > 0:
                self._frame_counter += 1
                frame = Frame.from_ndarray(
                    image=image,
                    frame_id=self._frame_counter,
                    pixel_format=fmt,
                )
                with self._lock:
                    self._latest_frame = frame
                self._new_frame_event.set()
            else:
                time.sleep(0.005)

    def read(self, wait_for_new: bool = False, timeout: float = 0.04) -> Optional[Frame]:
        """Reads the latest camera frame with zero lag.

        Args:
            wait_for_new: If True, waits for a new sensor frame up to timeout.
                         If False, immediately returns the most recent frame (<0.1ms).
            timeout: Maximum seconds to wait if wait_for_new is True.
        """
        if not self.is_opened:
            raise CameraDisconnectedError("Camera is not opened")

        if wait_for_new:
            self._new_frame_event.wait(timeout=timeout)
            self._new_frame_event.clear()

        with self._lock:
            return self._latest_frame

    def close(self) -> None:
        """Release background thread and camera hardware."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Closed camera {self.device_index}")

    @property
    def is_opened(self) -> bool:
        """Check if the camera is opened."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the current resolution."""
        if not self.is_opened:
            return (0, 0)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        """Get the current FPS."""
        if not self.is_opened:
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    def get_supported_resolutions(self) -> List[Tuple[int, int, str]]:
        """Returns standard high-definition and native resolutions for this camera."""
        return [
            (2560, 1440, "2560x1440 (2K QHD - Native Max)"),
            (1920, 1080, "1920x1080 (Full HD 1080p)"),
            (1280, 720, "1280x720 (HD 720p - Fast)"),
            (960, 540, "960x540 (qHD 540p)"),
            (640, 480, "640x480 (SD 480p - Eco)"),
        ]

    def set_resolution(self, width: int, height: int) -> Tuple[int, int]:
        """Set the camera resolution with FourCC preservation for high framerate."""
        if self.is_opened:
            with self._lock:
                if platform.system() == "Windows":
                    self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.config.width = actual_w
                self.config.height = actual_h
                logger.info(f"Resolution set to {actual_w}x{actual_h}")
                return (actual_w, actual_h)
        return (0, 0)

    def set_fps(self, fps: int) -> None:
        """Set the camera FPS."""
        self.config.fps = fps
        if self.is_opened:
            with self._lock:
                self._cap.set(cv2.CAP_PROP_FPS, fps)
                logger.info(f"Target FPS set to {fps}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

