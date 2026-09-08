import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
try:
    import openvino as ov
except ImportError:
    ov = None

class ModelLoadError(Exception): pass
class ModelNotFoundError(Exception): pass
class ModelValidationError(Exception): pass
class DeviceNotAvailableError(Exception): pass

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir: Union[Path, str]):
        self.models_dir = Path(models_dir)
        self._core_instance = None

    @property
    def _core(self):
        if self._core_instance is None:
            if ov is None:
                raise ImportError("OpenVINO is not installed")
            self._core_instance = ov.Core()
        return self._core_instance

    def find_model(self, model_name: str) -> Optional[Path]:
        xml_path = self.models_dir / f"{model_name}.xml"
        if xml_path.exists():
            return xml_path
        
        for p in self.models_dir.rglob("*.xml"):
            if p.stem == model_name:
                return p
        return None

    def validate_model(self, model_path: Path) -> bool:
        if not model_path.exists():
            return False
        bin_path = model_path.with_suffix(".bin")
        return bin_path.exists()

    def load_model(self, model_path: Path):
        try:
            logger.info(f"Loading model from {model_path}")
            if not self.validate_model(model_path):
                raise ModelValidationError(f"Invalid model path: {model_path}")
            return self._core.read_model(model=str(model_path))
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")

    def compile_model(self, model, device: str):
        try:
            from privacycam.services.device_service import DeviceService
            svc = DeviceService()
            resolved = svc.resolve_device(device)
            logger.info(f"Compiling model for device {resolved} (requested: {device})")
            if resolved != "AUTO" and not self.is_device_available(resolved):
                raise DeviceNotAvailableError(f"Device {resolved} is not available")
            return self._core.compile_model(model, device_name=resolved)
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            raise ModelLoadError(f"Failed to compile model: {e}")

    def get_available_devices(self) -> List[str]:
        try:
            return self._core.available_devices
        except Exception as e:
            logger.warning(f"Failed to get available devices: {e}")
            return []

    def is_device_available(self, device: str) -> bool:
        if device == "AUTO":
            return True
        from privacycam.services.device_service import DeviceService
        svc = DeviceService()
        resolved = svc.resolve_device(device)
        available = self.get_available_devices()
        return resolved in available

    def get_device_info(self, device: str) -> Dict:
        info = {}
        if not self.is_device_available(device) and device != "AUTO":
            return info
        try:
            info["name"] = device
            if device in self.get_available_devices():
                info["full_name"] = self._core.get_property(device, "FULL_DEVICE_NAME")
        except Exception as e:
            logger.warning(f"Failed to get device info: {e}")
        return info

    def get_model_info(self, model_path: Path) -> Dict:
        info = {}
        try:
            model = self.load_model(model_path)
            info["inputs"] = [{"name": i.any_name, "shape": list(i.shape)} for i in model.inputs]
            info["outputs"] = [{"name": o.any_name, "shape": list(o.shape)} for o in model.outputs]
        except Exception as e:
            logger.warning(f"Failed to get model info: {e}")
        return info
