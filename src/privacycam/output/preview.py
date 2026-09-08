import cv2
import numpy as np
from typing import List, Dict, Optional

from privacycam.output.output import VideoOutput
from privacycam.capture.frame import Frame
from privacycam.tracking.track import Track
from privacycam.performance.fps import FPSCounter

class PreviewOutput(VideoOutput):
    """Output stage for displaying video with OpenCV."""
    
    def __init__(self, window_name: str = "PrivacyCam Preview", width: int = 1280, height: int = 720, 
                 show_fps: bool = True, show_detections: bool = False):
        self.window_name = window_name
        self.width = width
        self.height = height
        self.show_fps = show_fps
        self.show_detections = show_detections
        self._active = False
        self._fps_counter = FPSCounter()

    def start(self) -> None:
        self._active = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)

    def write(self, frame: Frame) -> None:
        if not self._active:
            return

        display_image = frame.image.copy()

        # Resize if dimensions differ
        if display_image.shape[1] != self.width or display_image.shape[0] != self.height:
            display_image = cv2.resize(display_image, (self.width, self.height))

        self._fps_counter.tick()

        if self.show_fps:
            fps = self._fps_counter.fps
            cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.window_name, display_image)
        cv2.waitKey(1)

    def stop(self) -> None:
        if self._active:
            cv2.destroyWindow(self.window_name)
            self._active = False

    def is_active(self) -> bool:
        return self._active

    def draw_debug_overlay(self, frame: np.ndarray, tracks: List[Track], metrics: Dict) -> np.ndarray:
        """Draws debugging information like bounding boxes and timings onto the image."""
        img = frame.copy()
        
        # Draw tracks
        for track in tracks:
            bbox = track.bounding_box
            x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"ID:{track.track_id} Conf:{track.confidence:.2f}", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw metrics
        y_offset = 60
        for stage_name, stage_stats in metrics.items():
            if isinstance(stage_stats, dict) and 'avg' in stage_stats:
                text = f"{stage_name}: {stage_stats['avg']:.1f}ms"
                cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                y_offset += 25

        return img
