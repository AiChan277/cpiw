import threading
import logging
from typing import List, Dict

from privacycam.capture.frame import Frame
from privacycam.capture.camera import Camera
from privacycam.detection.detector import FaceDetector
from privacycam.detection.detection import FaceDetection
from privacycam.tracking.tracker import FaceTracker
from privacycam.tracking.track import Track
from privacycam.masking.mask_generator import MaskGenerator
from privacycam.anonymization.anonymizer import Anonymizer
from privacycam.output.output import VideoOutput
from privacycam.config import AppConfig

from privacycam.pipeline.frame_queue import FrameQueue
from privacycam.pipeline.scheduler import DetectionScheduler
from privacycam.pipeline.stages import PipelineStage, StageTiming
from privacycam.performance.metrics import PipelineMetrics

logger = logging.getLogger(__name__)

class PrivacyPipeline:
    """Main application pipeline managing capture, processing, and output."""

    def __init__(self, camera: Camera, detector: FaceDetector, tracker: FaceTracker,
                 mask_generator: MaskGenerator, anonymizer: Anonymizer,
                 outputs: List[VideoOutput], config: AppConfig):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.mask_generator = mask_generator
        self.anonymizer = anonymizer
        self.outputs = outputs
        self.config = config

        self._running = False
        self._capture_thread = None
        self._process_thread = None
        
        # Max queue size of 2 typically avoids latency buildup while smoothing jitter
        self._capture_queue = FrameQueue(maxsize=2)
        self._scheduler = DetectionScheduler()
        self._metrics = PipelineMetrics()
        
        self._current_tracks: List[Track] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """Starts the capture and processing threads."""
        if self._running:
            return

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)

        self._capture_thread.start()
        self._process_thread.start()
        
        logger.info("Pipeline started.")

    def stop(self) -> None:
        """Stops the pipeline and joins threads."""
        if not self._running:
            return

        self._running = False
        if self._capture_thread:
            self._capture_thread.join()
        if self._process_thread:
            self._process_thread.join()
            
        logger.info("Pipeline stopped.")

    @property
    def is_running(self) -> bool:
        """Checks if the pipeline is running."""
        return self._running

    def get_metrics(self) -> Dict:
        """Returns current performance metrics."""
        return self._metrics.get_summary()

    def get_current_tracks(self) -> List[Track]:
        """Returns the current list of active tracks."""
        with self._lock:
            return list(self._current_tracks)

    def _capture_loop(self) -> None:
        """Thread loop for capturing frames from the camera."""
        while self._running:
            try:
                frame = self.camera.read()
                if frame:
                    self._capture_queue.put(frame)
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                # Optional: Handle camera disconnection, reconnect logic, etc.
                self._running = False

    def _tracks_to_detections(self, tracks: List[Track]) -> List[FaceDetection]:
        """Converts tracks to face detections for mask generation."""
        return [
            FaceDetection(
                bounding_box=t.bounding_box,
                confidence=t.confidence,
                timestamp=0.0 # Will use current frame timestamp ideally
            ) for t in tracks
        ]

    def _process_loop(self) -> None:
        """Thread loop for processing frames (detect, track, mask, anonymize)."""
        while self._running:
            try:
                frame = self._capture_queue.get(timeout=0.1)
                if not frame:
                    continue

                detections = []
                # 1. Detection
                with StageTiming(PipelineStage.DETECTION) as stage:
                    if self._scheduler.should_detect(self._current_tracks):
                        detections = self.detector.detect(frame)
                self._metrics.record(PipelineStage.DETECTION, stage.duration_ms)

                # 2. Tracking
                with StageTiming(PipelineStage.TRACKING) as stage:
                    self._current_tracks = self.tracker.update(detections, frame.frame_id)
                self._metrics.record(PipelineStage.TRACKING, stage.duration_ms)

                # 3. Masking
                with StageTiming(PipelineStage.MASKING) as stage:
                    tracks_dets = self._tracks_to_detections(self._current_tracks)
                    mask = self.mask_generator.generate_multi(tracks_dets, frame.height, frame.width)
                self._metrics.record(PipelineStage.MASKING, stage.duration_ms)

                # 4. Anonymization
                with StageTiming(PipelineStage.ANONYMIZATION) as stage:
                    for det in tracks_dets:
                        # Depending on anonymizer design, might anonymize all at once or per bbox.
                        # Using the required API:
                        frame.image = self.anonymizer.anonymize(frame.image, mask, det.bounding_box)
                self._metrics.record(PipelineStage.ANONYMIZATION, stage.duration_ms)

                # 5. Output
                with StageTiming(PipelineStage.OUTPUT) as stage:
                    for output in self.outputs:
                        if output.is_active():
                            output.write(frame)
                self._metrics.record(PipelineStage.OUTPUT, stage.duration_ms)

                self._scheduler.tick()

            except Exception as e:
                logger.error(f"Error in process loop: {e}", exc_info=True)
