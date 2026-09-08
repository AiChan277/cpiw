"""Main PrivacyCam application — wires all subsystems together."""

import sys
import logging
import signal
from pathlib import Path
from typing import Optional

from privacycam.config import AppConfig
from privacycam.logging_config import setup_logging

logger = logging.getLogger(__name__)


class PrivacyCamApp:
    """Main application class that initializes and runs the full pipeline."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._is_running = False

        # Subsystem references (initialized in _setup_components)
        self.camera = None
        self.camera_manager = None
        self.detector = None
        self.tracker = None
        self.mask_generator = None
        self.anonymizer = None
        self.pipeline = None
        self.outputs: list = []

    # ------------------------------------------------------------------
    # Component initialization
    # ------------------------------------------------------------------

    def _setup_components(self) -> None:
        """Initialize all subsystems based on config."""
        logger.info("Initializing PrivacyCam components...")

        # ---------- Camera ----------
        from privacycam.capture.camera_manager import CameraManager
        self.camera_manager = CameraManager(self.config.camera)
        self.camera = self.camera_manager.switch_camera(
            self.config.camera.device_index
        )
        actual_res = self.camera.resolution
        logger.info(
            f"Camera {self.config.camera.device_index} opened — "
            f"{actual_res[0]}×{actual_res[1]} @ {self.camera.fps:.0f} FPS"
        )

        # ---------- Detection ----------
        self.detector = self._create_detector()

        # ---------- Tracking ----------
        from privacycam.tracking.tracker import FaceTracker
        self.tracker = FaceTracker(
            max_lost_frames=self.config.tracking.max_lost_frames,
            iou_threshold=self.config.tracking.iou_threshold,
            min_hits=self.config.tracking.min_hits,
        )

        # ---------- Mask generation ----------
        from privacycam.masking.mask_generator import MaskGenerator
        self.mask_generator = MaskGenerator(
            expansion=self.config.anonymization.expansion,
            feather_radius=self.config.anonymization.feather,
            use_ellipse=True,
        )

        # ---------- Anonymization ----------
        from privacycam.anonymization.anonymizer import AnonymizerFactory
        from privacycam.anonymization.modes import AnonymizationMode
        mode = AnonymizationMode.from_string(self.config.anonymization.mode)
        self.anonymizer = AnonymizerFactory.create(
            mode,
            strength=self.config.anonymization.blur_strength,
            blocks=self.config.anonymization.pixelate_blocks,
            color=self.config.anonymization.solid_color,
        )

        # ---------- Outputs ----------
        self.outputs = []
        if self.config.output.preview:
            from privacycam.output.preview import PreviewOutput
            self.outputs.append(
                PreviewOutput(
                    window_name="PrivacyCam Preview",
                    width=self.config.output.preview_width,
                    height=self.config.output.preview_height,
                    show_fps=self.config.ui.show_fps,
                    show_detections=self.config.ui.debug_overlay,
                )
            )

        if self.config.output.virtual_camera:
            try:
                from privacycam.output.virtual_camera import VirtualCameraOutput
                vcam = VirtualCameraOutput(
                    width=self.config.camera.width,
                    height=self.config.camera.height,
                    fps=self.config.camera.fps,
                )
                vcam.start()
                self.outputs.append(vcam)
            except Exception as e:
                logger.warning(f"Virtual camera unavailable: {e}")

        # ---------- Pipeline ----------
        from privacycam.pipeline.pipeline import PrivacyPipeline
        self.pipeline = PrivacyPipeline(
            camera=self.camera,
            detector=self.detector,
            tracker=self.tracker,
            mask_generator=self.mask_generator,
            anonymizer=self.anonymizer,
            outputs=self.outputs,
            config=self.config,
        )
        logger.info("All components initialized.")

    def _create_detector(self):
        """Create a face detector — try OpenVINO, fall back to stub."""
        model_path = self._find_model()
        if model_path:
            try:
                from privacycam.detection.openvino_detector import OpenVINODetector
                detector = OpenVINODetector(
                    model_path=model_path,
                    device=self.config.detection.device,
                    confidence_threshold=self.config.detection.confidence_threshold,
                    input_width=self.config.detection.input_width,
                    input_height=self.config.detection.input_height,
                )
                detector.warmup()
                logger.info(f"OpenVINO detector ready on {self.config.detection.device}")
                return detector
            except Exception as e:
                logger.warning(f"OpenVINO detector failed: {e}")

        logger.warning(
            "No face-detection model found. Running in passthrough mode. "
            "Run  scripts/download_model.py  to download a model."
        )
        return _PassthroughDetector()

    def _find_model(self) -> Optional[Path]:
        """Search common locations for a face-detection .xml model."""
        search_dirs = [
            Path(__file__).resolve().parent.parent.parent / "models",
            Path.cwd() / "models",
        ]
        for d in search_dirs:
            if not d.is_dir():
                continue
            for xml in d.rglob("*.xml"):
                bin_path = xml.with_suffix(".bin")
                if bin_path.exists():
                    logger.info(f"Model found: {xml}")
                    return xml
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the application (GUI mode by default)."""
        self._is_running = True
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            self.run_gui()
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            self.stop()
            raise

    def run_gui(self) -> None:
        """Start the PySide6 GUI application."""
        logger.info("Starting PrivacyCam GUI...")

        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer

        qt_app = QApplication.instance() or QApplication(sys.argv)

        # Build the service layer (bridges UI ↔ pipeline)
        from privacycam.services.application_service import ApplicationService
        service = ApplicationService(self.config)

        # Build the main window
        from privacycam.ui.main_window import MainWindow
        window = MainWindow(self.config, service)
        window.show()

        # Allow Ctrl-C in terminal to close the app
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(200)

        exit_code = qt_app.exec()
        service.cleanup()
        sys.exit(exit_code)

    def run_headless(self) -> None:
        """Run without GUI — OpenCV preview only."""
        logger.info("Starting headless mode...")
        signal.signal(signal.SIGINT, self._signal_handler)

        self._is_running = True
        self._setup_components()

        # Start all outputs
        for out in self.outputs:
            if not out.is_active():
                try:
                    out.start()
                except Exception as e:
                    logger.warning(f"Output start failed: {e}")

        self.pipeline.start()
        logger.info("Pipeline running. Press Ctrl+C to stop.")

        import time
        try:
            while self._is_running and self.pipeline.is_running:
                time.sleep(0.05)
                # Check for OpenCV 'q' key
                import cv2
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop all subsystems."""
        if not self._is_running:
            return
        logger.info("Shutting down PrivacyCam...")
        self._is_running = False

        if self.pipeline:
            self.pipeline.stop()
        for out in self.outputs:
            try:
                out.stop()
            except Exception:
                pass
        if self.camera_manager:
            self.camera_manager.close()
        logger.info("PrivacyCam stopped.")

    def _signal_handler(self, sig, frame):
        logger.info("Ctrl+C received — shutting down.")
        self.stop()


# ------------------------------------------------------------------
# Passthrough detector (used when no model is available)
# ------------------------------------------------------------------

class _PassthroughDetector:
    """Dummy detector that returns no faces — lets the app run without a model."""

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
