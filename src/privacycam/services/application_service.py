"""Application service — bridge between PySide6 UI and the processing pipeline.

Manages real-time multi-threaded frame capture, AI inference (NPU/GPU/CPU),
tracking, anonymization, and virtual camera streaming with performance profiles.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer

from privacycam.config import AppConfig

logger = logging.getLogger(__name__)


class ApplicationService(QObject):
    """Manages the full lifecycle of the privacy pipeline."""

    frame_ready = Signal(np.ndarray)
    metrics_updated = Signal(dict)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    mode_changed = Signal(str)

    def __init__(self, config: AppConfig):
        super().__init__()
        self._config = config

        self._camera_manager = None
        self._camera = None
        self._detector = None
        self._tracker = None
        self._mask_generator = None
        self._anonymizer = None
        self._virtual_camera_output = None

        self._performance_mode = getattr(config, "performance_mode", "realtime")

        self._is_running = False
        self._process_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_metrics: dict = {}
        self._lock = threading.Lock()

        # Asynchronous inference shared state
        self._infer_lock = threading.Lock()
        self._infer_input_frame = None
        self._infer_event = threading.Event()
        self._latest_detections: Optional[list] = None
        self._latest_infer_ms: float = 0.0

        # UI frame state & metrics timer (100ms = 10Hz text metrics updates)
        self._ui_frame_pending = False
        self._ui_timer = QTimer(self)
        self._ui_timer.timeout.connect(self._push_metrics_to_ui)
        self._ui_timer.setInterval(100)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create all subsystems based on current config."""
        logger.info("ApplicationService initializing...")

        # 1. Camera Manager & Threaded Camera
        from privacycam.capture.camera_manager import CameraManager
        self._camera_manager = CameraManager(self._config.camera)

        try:
            self._camera = self._camera_manager.switch_camera(
                self._config.camera.device_index
            )
            res = self._camera.resolution
            logger.info(f"Camera opened: {res[0]}x{res[1]}")
        except Exception as e:
            logger.error(f"Camera init failed: {e}")
            self.error_occurred.emit(f"Camera error: {e}")
            return

        # 2. AI Detector (NPU/GPU/CPU)
        self._detector = self._create_detector()

        # 3. Face Tracker
        from privacycam.tracking.tracker import FaceTracker
        self._tracker = FaceTracker(
            max_lost_frames=self._config.tracking.max_lost_frames,
            iou_threshold=self._config.tracking.iou_threshold,
            min_hits=self._config.tracking.min_hits,
            smoothing_alpha=self._config.tracking.smoothing_factor,
            distance_threshold=self._config.tracking.distance_threshold,
        )

        # 4. Mask Generator
        from privacycam.masking.mask_generator import MaskGenerator
        self._mask_generator = MaskGenerator(
            expansion=self._config.anonymization.expansion,
            feather_radius=self._config.anonymization.feather,
            use_ellipse=True,
        )

        # 5. Anonymizer
        self._rebuild_anonymizer()

        # 6. Apply initial performance profile
        self.set_performance_mode(self._performance_mode)

        # 6.5. Apply initial tracking preset
        if hasattr(self._config, "tracking") and hasattr(self._config.tracking, "preset"):
            self.apply_tracking_preset(self._config.tracking.preset)

        # 7. Virtual Camera
        if self._config.output.virtual_camera:
            self._start_virtual_camera()

        logger.info("ApplicationService ready.")

    def _create_detector(self):
        """Try OpenVINO on resolved device, fall back gracefully."""
        model_path = self._find_model()
        if model_path:
            try:
                from privacycam.detection.openvino_detector import OpenVINODetector
                from privacycam.services.device_service import DeviceService
                svc = DeviceService()
                resolved = svc.resolve_device(self._config.detection.device)

                det = OpenVINODetector(
                    model_path=model_path,
                    device=resolved,
                    confidence_threshold=self._config.detection.confidence_threshold,
                    input_width=self._config.detection.input_width,
                    input_height=self._config.detection.input_height,
                )
                det.warmup()
                logger.info(f"Detector ready on {det.device}")
                return det
            except Exception as e:
                logger.warning(f"OpenVINO detector failed on {self._config.detection.device}: {e}")
                self.error_occurred.emit(f"Model init failed on {self._config.detection.device}: {e}")

        logger.warning("No model found — running in passthrough mode.")
        self.error_occurred.emit(
            "No face-detection model found.\nRun scripts/download_model.py to download one."
        )
        return _PassthroughDetector()

    def _find_model(self) -> Optional[Path]:
        for d in [
            Path(__file__).resolve().parent.parent.parent / "models",
            Path.cwd() / "models",
        ]:
            if not d.is_dir():
                continue
            for xml in d.rglob("*.xml"):
                if xml.with_suffix(".bin").exists():
                    return xml
        return None

    def _rebuild_anonymizer(self) -> None:
        from privacycam.anonymization.anonymizer import AnonymizerFactory
        from privacycam.anonymization.modes import AnonymizationMode
        mode = AnonymizationMode.from_string(self._config.anonymization.mode)
        new_anon = AnonymizerFactory.create(
            mode,
            strength=self._config.anonymization.blur_strength,
            blocks=self._config.anonymization.pixelate_blocks,
            color=self._config.anonymization.solid_color,
        )
        if hasattr(new_anon, "set_fast_mode"):
            new_anon.set_fast_mode(self._performance_mode in ("realtime", "low"))

        with self._lock:
            self._anonymizer = new_anon

    def _start_virtual_camera(self) -> None:
        try:
            from privacycam.output.virtual_camera import VirtualCameraOutput
            self._virtual_camera_output = VirtualCameraOutput(
                width=self._config.camera.width,
                height=self._config.camera.height,
                fps=self._config.camera.fps,
            )
            self._virtual_camera_output.start()
            logger.info("Virtual camera started.")
        except Exception as e:
            logger.warning(f"Virtual camera unavailable: {e}")
            self._virtual_camera_output = None

    # ------------------------------------------------------------------
    # Performance Modes (Realtime < 20ms, Low, Medium, High)
    # ------------------------------------------------------------------

    def set_performance_mode(self, mode: str) -> None:
        """Applies configuration preset for the selected performance profile."""
        mode_key = mode.lower().strip()
        logger.info(f"Applying Performance Profile: {mode_key.upper()}")

        with self._lock:
            self._performance_mode = mode_key

            if "realtime" in mode_key:
                # Target: < 20 ms total latency, maximum responsiveness
                if self._mask_generator:
                    self._mask_generator.set_feather(11)
                if hasattr(self._anonymizer, "set_fast_mode"):
                    self._anonymizer.set_fast_mode(True)

            elif "low" in mode_key:
                # Target: Minimum power consumption / battery saver
                if self._mask_generator:
                    self._mask_generator.set_feather(7)
                if hasattr(self._anonymizer, "set_fast_mode"):
                    self._anonymizer.set_fast_mode(True)

            elif "medium" in mode_key:
                # Target: Balanced quality and speed
                if self._mask_generator:
                    self._mask_generator.set_feather(15)
                if hasattr(self._anonymizer, "set_fast_mode"):
                    self._anonymizer.set_fast_mode(False)

            elif "high" in mode_key:
                # Target: Maximum visual fidelity
                if self._mask_generator:
                    self._mask_generator.set_feather(21)
                if hasattr(self._anonymizer, "set_fast_mode"):
                    self._anonymizer.set_fast_mode(False)

        self.mode_changed.emit(self._performance_mode)
        self.status_changed.emit(f"Mode: {self._performance_mode.upper()}")

    @property
    def performance_mode(self) -> str:
        return self._performance_mode

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the processing and async inference loops."""
        if self._is_running:
            return

        if self._camera is None:
            self.initialize()

        self._is_running = True

        # 1. Asynchronous inference worker thread
        self._infer_thread = threading.Thread(
            target=self._async_inference_worker, daemon=True
        )
        self._infer_thread.start()

        # 2. Main real-time render thread
        self._process_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._process_thread.start()

        self._ui_timer.start()
        self.status_changed.emit("running")
        logger.info(f"Pipeline started in {self._performance_mode.upper()} mode.")

    def stop(self) -> None:
        """Stop the processing loops."""
        if not self._is_running:
            return

        self._is_running = False
        self._ui_timer.stop()
        self._infer_event.set()

        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.5)

        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.5)

        self.status_changed.emit("stopped")
        logger.info("Pipeline stopped.")

    @property
    def is_running(self) -> bool:
        return self._is_running

    # ------------------------------------------------------------------
    # Asynchronous Inference Worker (Runs on NPU / GPU / CPU in background)
    # ------------------------------------------------------------------

    def _async_inference_worker(self) -> None:
        """Continuously runs face detection as frames arrive from the camera."""
        while self._is_running:
            self._infer_event.wait(timeout=0.05)
            self._infer_event.clear()

            with self._infer_lock:
                item = self._infer_input_frame
                self._infer_input_frame = None

            if item is None or not self._is_running:
                continue

            small_frame, orig_w, orig_h = item

            with self._lock:
                detector = self._detector

            if detector and detector.is_ready:
                t0 = time.perf_counter()
                try:
                    if hasattr(detector, "detect"):
                        import inspect
                        sig = inspect.signature(detector.detect)
                        if "orig_width" in sig.parameters:
                            dets = detector.detect(small_frame, orig_width=orig_w, orig_height=orig_h)
                        else:
                            dets = detector.detect(small_frame)
                    else:
                        dets = []
                    t_ms = (time.perf_counter() - t0) * 1000
                    with self._infer_lock:
                        self._latest_detections = dets
                        self._latest_infer_ms = t_ms
                except Exception as e:
                    logger.debug(f"Async detection error: {e}")

    # ------------------------------------------------------------------
    # Real-Time Rendering Loop (Zero-latency camera -> tracking -> blur -> UI)
    # ------------------------------------------------------------------

    def _processing_loop(self) -> None:
        """High-framerate rendering loop achieving < 20 ms turnaround latency."""
        from privacycam.capture.frame import Frame
        from privacycam.detection.detection import FaceDetection
        from privacycam.performance.fps import FPSCounter

        fps_counter = FPSCounter(window_size=60)
        current_tracks = []
        frame_idx = 0

        while self._is_running:
            with self._lock:
                cam = self._camera
            if not cam or not cam.is_opened:
                time.sleep(0.01)
                continue

            frame = cam.read(wait_for_new=True, timeout=0.033)
            if frame is None or not frame.is_valid:
                time.sleep(0.002)
                continue

            # Start timing pure pipeline processing latency
            t_start = time.perf_counter()

            frame_idx += 1
            interval = max(1, self._config.detection.interval)
            h, w = frame.image.shape[:2]

            # Dispatch lightweight downscaled frame to async AI detector
            if frame_idx % interval == 0:
                det_w = getattr(self._detector, "input_width", 672)
                det_h = getattr(self._detector, "input_height", 384)
                small_img = cv2.resize(frame.image, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
                small_frame = Frame.from_ndarray(small_img, frame.frame_id, frame.pixel_format)
                with self._infer_lock:
                    self._infer_input_frame = (small_frame, w, h)
                self._infer_event.set()

            # Zero-copy direct image buffer
            image = frame.image

            # 3. Retrieve latest detections from worker and update tracker
            with self._infer_lock:
                new_dets = self._latest_detections
                self._latest_detections = None
                infer_time_ms = self._latest_infer_ms

            if self._config.tracking.enabled and self._tracker:
                current_tracks = self._tracker.update(new_dets, frame.frame_id)
            else:
                if new_dets is not None and self._tracker:
                    self._last_raw_tracks = [self._tracker._create_track(d) for d in new_dets]
                current_tracks = getattr(self, "_last_raw_tracks", [])

            # 4. Masking + Anonymization (Optimized in-place)
            if current_tracks:
                track_detections = [
                    FaceDetection(
                        bounding_box=t.bounding_box,
                        confidence=t.confidence,
                    )
                    for t in current_tracks
                ]
                with self._lock:
                    masker = self._mask_generator
                    anonymizer = self._anonymizer

                if masker and anonymizer:
                    mask = masker.generate_multi(track_detections, h, w)
                    for td in track_detections:
                        image = anonymizer.anonymize(image, mask, td.bounding_box)

            # 5. Debug overlay (optional)
            if self._config.ui.debug_overlay:
                image = self._draw_debug(image, current_tracks)

            # 6. Virtual Camera Output (Asynchronous non-blocking push)
            if self._virtual_camera_output and self._virtual_camera_output.is_active():
                from privacycam.capture.frame import Frame
                from privacycam.capture.pixel_format import PixelFormat
                out_frame = Frame.from_ndarray(image, frame.frame_id, PixelFormat.BGR)
                self._virtual_camera_output.write(out_frame)

            # 7. Metrics & Timing
            fps_counter.tick()
            process_latency_ms = (time.perf_counter() - t_start) * 1000
            active_device = self.current_device

            with self._lock:
                should_emit = not self._ui_frame_pending
                if should_emit:
                    self._ui_frame_pending = True
                self._latest_frame = image
                self._latest_metrics = {
                    "fps": fps_counter.fps,
                    "latency": process_latency_ms,
                    "latency_avg": process_latency_ms,
                    "latency_p95": process_latency_ms,
                    "inference_time": infer_time_ms,
                    "faces": len(current_tracks),
                    "dropped_frames": fps_counter.dropped_count,
                    "device": active_device,
                    "mode": self._performance_mode.upper(),
                }

            if should_emit:
                self.frame_ready.emit(image)

    def notify_ui_rendered(self) -> None:
        """Called by UI thread once frame paint is complete to allow next frame delivery."""
        with self._lock:
            self._ui_frame_pending = False

    def _draw_debug(self, image: np.ndarray, tracks) -> np.ndarray:
        """Draw bounding boxes and track labels on the frame."""
        for t in tracks:
            x1, y1, x2, y2 = t.bounding_box.to_int_tuple()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"#{t.track_id} ({t.confidence:.2f})"
            cv2.putText(
                image, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
            )
        return image

    # ------------------------------------------------------------------
    # UI Metrics Push (10 Hz periodic)
    # ------------------------------------------------------------------

    def _push_metrics_to_ui(self) -> None:
        """Periodically emits lightweight metrics to UI on Qt thread."""
        with self._lock:
            metrics = self._latest_metrics.copy()

        if metrics:
            self.metrics_updated.emit(metrics)

    # ------------------------------------------------------------------
    # Live Settings Updates
    # ------------------------------------------------------------------

    @property
    def current_device(self) -> str:
        if self._detector and hasattr(self._detector, "device"):
            return self._detector.device
        return self._config.detection.device

    def update_anonymization_mode(self, mode: str) -> None:
        self._config.anonymization.mode = mode
        self._rebuild_anonymizer()
        logger.info(f"Anonymization mode -> {mode}")

    def update_blur_strength(self, strength: int) -> None:
        self._config.anonymization.blur_strength = strength
        self._rebuild_anonymizer()

    def update_pixelate_blocks(self, blocks: int) -> None:
        self._config.anonymization.pixelate_blocks = blocks
        self._rebuild_anonymizer()

    def update_solid_color(self, color: tuple) -> None:
        self._config.anonymization.solid_color = color
        self._rebuild_anonymizer()

    def update_expansion(self, ratio: float) -> None:
        self._config.anonymization.expansion = ratio
        with self._lock:
            if self._mask_generator:
                self._mask_generator.set_expansion(ratio)

    def update_feather(self, radius: int) -> None:
        self._config.anonymization.feather = radius
        with self._lock:
            if self._mask_generator:
                self._mask_generator.set_feather(radius)

    def update_confidence(self, threshold: float) -> None:
        self._config.detection.confidence_threshold = threshold
        with self._lock:
            if self._detector and hasattr(self._detector, "confidence_threshold"):
                self._detector.confidence_threshold = threshold

    def update_detection_interval(self, interval: int) -> None:
        self._config.detection.interval = interval
        logger.info(f"Detection interval -> {interval}")

    def update_tracking_enabled(self, enabled: bool) -> None:
        self._config.tracking.enabled = enabled
        logger.info(f"Tracking enabled -> {enabled}")

    def update_tracking_smoothing(self, alpha: float) -> None:
        self._config.tracking.smoothing_factor = alpha
        with self._lock:
            if self._tracker and hasattr(self._tracker, "set_smoothing_alpha"):
                self._tracker.set_smoothing_alpha(alpha)
        logger.info(f"Tracking smoothing -> {alpha:.2f}")

    def update_tracking_persistence(self, frames: int) -> None:
        self._config.tracking.max_lost_frames = frames
        with self._lock:
            if self._tracker:
                self._tracker.max_lost_frames = frames
        logger.info(f"Tracking persistence -> {frames} frames")

    def update_tracking_distance(self, factor: float) -> None:
        self._config.tracking.distance_threshold = factor
        with self._lock:
            if self._tracker:
                self._tracker.distance_threshold = factor
        logger.info(f"Tracking distance fallback -> {factor:.2f}x")

    def update_tracking_iou(self, iou: float) -> None:
        self._config.tracking.iou_threshold = iou
        with self._lock:
            if self._tracker:
                self._tracker.iou_threshold = iou
        logger.info(f"Tracking IoU threshold -> {iou:.2f}")

    def apply_tracking_preset(self, preset_key: str) -> None:
        from privacycam.tracking.presets import TRACKING_PRESETS
        preset = TRACKING_PRESETS.get(preset_key)
        if not preset:
            return
        self._config.tracking.preset = preset_key
        self.update_detection_interval(preset["interval"])
        self.update_tracking_smoothing(preset["smoothing"])
        self.update_tracking_persistence(preset["persistence"])
        self.update_tracking_distance(preset["distance"])
        self.update_tracking_iou(preset["iou"])
        logger.info(f"Applied tracking preset: {preset_key}")

    def update_debug_overlay(self, enabled: bool) -> None:
        self._config.ui.debug_overlay = enabled

    def update_device(self, device: str) -> None:
        """Swaps the inference device live."""
        logger.info(f"Updating inference device to {device}...")
        self._config.detection.device = device

        old_detector = self._detector
        try:
            new_detector = self._create_detector()
            with self._lock:
                self._detector = new_detector
            if old_detector and hasattr(old_detector, "close"):
                old_detector.close()
            logger.info(f"Successfully running on {new_detector.device}")
            self.status_changed.emit(f"Running on {new_detector.device}")
        except Exception as e:
            logger.error(f"Failed to switch device to {device}: {e}")
            self.error_occurred.emit(f"Device switch error: {e}")

    def update_camera(self, index: int) -> None:
        """Hot-swaps physical camera device."""
        self._config.camera.device_index = index
        try:
            with self._lock:
                new_cam = self._camera_manager.switch_camera(index)
                self._camera = new_cam
            logger.info(f"Switched to camera {index}")
            self.status_changed.emit(f"Switched to camera {index}")
        except Exception as e:
            self.error_occurred.emit(f"Camera switch failed: {e}")
            logger.error(f"Camera switch failed: {e}")

    def update_resolution(self, width: int, height: int) -> None:
        self._config.camera.width = width
        self._config.camera.height = height
        with self._lock:
            if self._camera and self._camera.is_opened:
                actual_w, actual_h = self._camera.set_resolution(width, height)
                logger.info(f"Updated camera resolution to {actual_w}x{actual_h}")
                self.status_changed.emit(f"Resolution: {actual_w}x{actual_h}")

    def update_fps(self, fps: int) -> None:
        self._config.camera.fps = fps
        with self._lock:
            if self._camera and self._camera.is_opened:
                self._camera.set_fps(fps)
                self.status_changed.emit(f"Camera FPS: {fps}")

    def update_virtual_camera(self, enabled: bool) -> None:
        if enabled and self._virtual_camera_output is None:
            self._start_virtual_camera()
        elif not enabled and self._virtual_camera_output is not None:
            self._virtual_camera_output.stop()
            self._virtual_camera_output = None

    # ------------------------------------------------------------------
    # Queries & Cleanup
    # ------------------------------------------------------------------

    def get_supported_resolutions(self) -> List[Tuple[int, int, str]]:
        """Query supported resolutions from current camera."""
        with self._lock:
            if self._camera and hasattr(self._camera, "get_supported_resolutions"):
                return self._camera.get_supported_resolutions()
        return [
            (2560, 1440, "2560x1440 (2K QHD - Native Max)"),
            (1920, 1080, "1920x1080 (Full HD 1080p)"),
            (1280, 720, "1280x720 (HD 720p - Fast)"),
            (960, 540, "960x540 (qHD 540p)"),
            (640, 480, "640x480 (SD 480p - Eco)"),
        ]

    def get_metrics(self) -> dict:
        with self._lock:
            return self._latest_metrics.copy()

    def get_available_cameras(self) -> List[dict]:
        if self._camera_manager:
            return self._camera_manager.enumerate_cameras()
        return []

    def get_available_devices(self) -> List[str]:
        try:
            from privacycam.services.device_service import DeviceService
            svc = DeviceService()
            return svc.get_available_devices()
        except Exception:
            return ["NPU", "GPU.1", "GPU.0", "CPU", "AUTO"]

    def cleanup(self) -> None:
        """Gracefully release all hardware and thread resources."""
        self.stop()
        if self._virtual_camera_output:
            try:
                self._virtual_camera_output.stop()
            except Exception:
                pass
        if self._detector and hasattr(self._detector, "close"):
            self._detector.close()
        if self._camera_manager:
            self._camera_manager.close()
        logger.info("ApplicationService cleaned up.")


class _PassthroughDetector:
    """Dummy detector that returns no faces."""
    def detect(self, frame):
        return []
    def warmup(self):
        pass
    @property
    def device(self) -> str:
        return "NONE"
    @property
    def is_ready(self) -> bool:
        return True
    def close(self):
        pass
