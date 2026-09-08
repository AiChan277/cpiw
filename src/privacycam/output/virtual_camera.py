import logging
import threading
from typing import Optional

try:
    import pyvirtualcam
    VIRTUAL_CAM_AVAILABLE = True
except ImportError:
    VIRTUAL_CAM_AVAILABLE = False

from privacycam.output.output import VideoOutput
from privacycam.capture.frame import Frame
from privacycam.capture.pixel_format import PixelFormat
from privacycam.exceptions import VirtualCameraError
import cv2

logger = logging.getLogger(__name__)


class VirtualCameraOutput(VideoOutput):
    """Outputs frames to a virtual camera device with non-blocking asynchronous dispatch."""

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._cam: Optional[pyvirtualcam.Camera] = None
        self._active = False

        # Non-blocking async queue
        self._thread: Optional[threading.Thread] = None
        self._pending_image = None
        self._lock = threading.Lock()
        self._event = threading.Event()

    def start(self) -> None:
        if not VIRTUAL_CAM_AVAILABLE:
            logger.warning("pyvirtualcam is not installed.")
            raise VirtualCameraError("pyvirtualcam is not installed or available.")

        try:
            self._cam = pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps)
            logger.info(f"Virtual camera started: {self.device_name}")
            self._active = True

            # Start non-blocking sender thread
            self._thread = threading.Thread(target=self._sender_loop, daemon=True)
            self._thread.start()
        except Exception as e:
            raise VirtualCameraError(f"Failed to start virtual camera: {e}")

    def _sender_loop(self) -> None:
        """Background thread sending frames to OBS Virtual Camera without blocking main loop."""
        while self._active:
            self._event.wait(timeout=0.05)
            self._event.clear()

            with self._lock:
                img = self._pending_image
                self._pending_image = None

            if img is not None and self._cam is not None:
                try:
                    self._cam.send(img)
                except Exception as e:
                    logger.error(f"Error sending to virtual camera: {e}")

    def write(self, frame: Frame) -> None:
        """Queues a frame for non-blocking virtual camera delivery."""
        if not self._active or not self._cam:
            return

        image = frame.image

        # Virtual camera expects RGB
        if frame.pixel_format == PixelFormat.BGR:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.shape[1] != self.width or image.shape[0] != self.height:
            image = cv2.resize(image, (self.width, self.height))

        # Atomic non-blocking push
        with self._lock:
            self._pending_image = image
        self._event.set()

    def stop(self) -> None:
        self._active = False
        self._event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
            self._thread = None

        if self._cam:
            self._cam.close()
            self._cam = None

    def is_active(self) -> bool:
        return self._active

    @property
    def device_name(self) -> Optional[str]:
        if self._cam:
            return self._cam.device
        return None
