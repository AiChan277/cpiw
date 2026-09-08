import logging
import time
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import numpy as np

from privacycam.capture.frame import Frame
from privacycam.detection.detection import BoundingBox, FaceDetection
from privacycam.detection.detector import FaceDetector
from privacycam.detection.model_manager import ModelManager
from privacycam.detection.preprocessing import Preprocessor

class InferenceError(Exception):
    pass

logger = logging.getLogger(__name__)

class OpenVINODetector(FaceDetector):
    """OpenVINO-based face detector with hardware acceleration for NPU, GPU, and CPU."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "NPU",
        confidence_threshold: float = 0.5,
        input_width: int = 672,
        input_height: int = 384,
    ):
        self.model_path = Path(model_path)
        self.target_device = device
        self.actual_device = device
        self.confidence_threshold = confidence_threshold
        self.input_width = input_width
        self.input_height = input_height

        self._model_manager = ModelManager(self.model_path.parent)
        self._preprocessor = Preprocessor(target_width=input_width, target_height=input_height)

        self._compiled_model = None
        self._input_layer = None
        self._output_layer = None
        self._ready = False

        self._load_and_compile()

    def _get_device_config(self, device: str) -> Dict[str, Any]:
        """Returns maximum-performance compilation hints tailored to the accelerator."""
        cfg = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
            "EXECUTION_MODE_HINT": "PERFORMANCE",
        }

        # Specific optimization flags to MAXIMIZE Intel AI Boost NPU
        if "NPU" in device.upper():
            cfg.update({
                "MODEL_PRIORITY": "HIGH",
                "NPU_TURBO": "YES",
                "NPU_TILES": 2,
            })

        return cfg

    def _load_and_compile(self) -> None:
        """Loads and compiles the model with accelerator turbo optimizations."""
        try:
            logger.info(f"Loading OpenVINO model from {self.model_path}")
            t0 = time.time()
            model = self._model_manager.load_model(self.model_path)

            from privacycam.services.device_service import DeviceService
            svc = DeviceService()
            resolved_target = svc.resolve_device(self.target_device)
            dev_cfg = self._get_device_config(resolved_target)

            try:
                # Compile with turbo performance parameters
                self._compiled_model = self._model_manager._core.compile_model(
                    model, device_name=resolved_target, config=dev_cfg
                )
                self.actual_device = resolved_target
            except Exception as dev_err:
                logger.warning(
                    f"Optimized compile on '{resolved_target}' failed ({dev_err}), trying standard compile..."
                )
                try:
                    self._compiled_model = self._model_manager._core.compile_model(
                        model, device_name=resolved_target
                    )
                    self.actual_device = resolved_target
                except Exception as fallback_err:
                    logger.warning(
                        f"Failed to compile model on '{resolved_target}': {fallback_err}. Falling back to CPU."
                    )
                    self._compiled_model = self._model_manager._core.compile_model(model, "CPU")
                    self.actual_device = "CPU"

            # Extract hardware execution device
            try:
                exec_dev = self._compiled_model.get_property("EXECUTION_DEVICES")
                if isinstance(exec_dev, list) and len(exec_dev) > 0:
                    self.actual_device = exec_dev[0].replace("(", "").replace(")", "")
                elif isinstance(exec_dev, str):
                    self.actual_device = exec_dev.replace("(", "").replace(")", "")
            except Exception:
                pass

            self._input_layer = self._compiled_model.input(0)
            self._output_layer = self._compiled_model.output(0)

            # Dynamically determine the model's actual required input shape
            shape = list(self._input_layer.shape)
            if len(shape) == 4:
                if shape[1] in (1, 3):  # NCHW
                    self.input_height = int(shape[2])
                    self.input_width = int(shape[3])
                else:  # NHWC
                    self.input_height = int(shape[1])
                    self.input_width = int(shape[2])
            else:
                self.input_height = 384
                self.input_width = 672

            self._preprocessor.target_width = self.input_width
            self._preprocessor.target_height = self.input_height

            self._ready = True
            logger.info(
                f"Model compiled on {self.actual_device} in {time.time()-t0:.2f}s "
                f"(Input resolution: {self.input_width}x{self.input_height}, Turbo: Active)"
            )
        except Exception as e:
            logger.error(f"Failed to load OpenVINO model: {e}")
            raise InferenceError(f"Model load failed: {e}")

    def detect(
        self,
        frame: Frame,
        orig_width: Optional[int] = None,
        orig_height: Optional[int] = None,
    ) -> List[FaceDetection]:
        """Detects faces in a video frame."""
        if not self._ready:
            raise InferenceError("Detector is not ready")

        t0 = time.time()

        try:
            image = frame.image
            cur_h, cur_w = image.shape[:2]
            target_w = orig_width if orig_width is not None else cur_w
            target_h = orig_height if orig_height is not None else cur_h

            input_tensor = self._preprocessor.to_tensor(image)
            results = self._compiled_model([input_tensor])[self._output_layer]

            detections = []

            # Shape is [1, 1, N, 7] -> [image_id, label, conf, x_min, y_min, x_max, y_max]
            if len(results.shape) == 4:
                results = results[0][0]

            for res in results:
                conf = float(res[2])
                if conf < self.confidence_threshold:
                    continue

                x_min = float(res[3])
                y_min = float(res[4])
                x_max = float(res[5])
                y_max = float(res[6])

                if x_max <= x_min or y_max <= y_min:
                    continue

                bbox = BoundingBox(
                    x1=x_min * target_w,
                    y1=y_min * target_h,
                    x2=x_max * target_w,
                    y2=y_max * target_h,
                ).clip(target_w, target_h)

                det = FaceDetection(
                    bounding_box=bbox,
                    confidence=conf,
                    timestamp=frame.timestamp,
                )

                if det.is_valid:
                    detections.append(det)

            return detections

        except Exception as e:
            logger.error(f"Inference error on {self.actual_device}: {e}")
            raise InferenceError(f"Inference error: {e}")

    def warmup(self) -> None:
        """Warms up inference engine with dummy inputs."""
        if not self._ready:
            return

        logger.info(f"Warming up OpenVINO detector on {self.actual_device}...")
        try:
            shape = [1, 3, self.input_height, self.input_width]
            dummy_input = np.zeros(shape, dtype=np.float32)
            for _ in range(3):
                self._compiled_model([dummy_input])
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup warning: {e}")

    @property
    def device(self) -> str:
        return self.actual_device

    @property
    def is_ready(self) -> bool:
        return self._ready

    def close(self) -> None:
        self._compiled_model = None
        self._ready = False
