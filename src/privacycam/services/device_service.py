import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DeviceService:
    """Service to discover, query, and resolve AI inference hardware (NPU, GPU, CPU)."""

    def __init__(self):
        self._core = None
        try:
            import openvino as ov
            self._core = ov.Core()
        except ImportError:
            try:
                from openvino.runtime import Core
                self._core = Core()
            except Exception as e:
                logger.warning(f"OpenVINO Core could not be initialized: {e}")

    def get_available_devices(self) -> List[str]:
        """Returns raw device IDs available in OpenVINO."""
        if not self._core:
            return ["CPU"]
        return self._core.available_devices

    def resolve_device(self, device_str: str) -> str:
        """Resolves generic aliases ('NVIDIA', 'GPU', 'NPU', 'INTEL') to exact device IDs."""
        if not self._core:
            return "CPU"

        av = self.get_available_devices()
        qu = device_str.strip().upper()

        if qu in ("AUTO", "CPU"):
            return qu

        # Exact match
        for d in av:
            if d.upper() == qu:
                return d

        # Check NPU
        if qu == "NPU":
            if "NPU" in av:
                return "NPU"
            for d in av:
                if "NPU" in d:
                    return d

        # Check discrete NVIDIA GPU
        if any(k in qu for k in ("NVIDIA", "RTX", "GTX", "GEFORCE", "DGPU")):
            for d in av:
                if "GPU" in d:
                    try:
                        name = self._core.get_property(d, "FULL_DEVICE_NAME").upper()
                        if "NVIDIA" in name or "GEFORCE" in name:
                            return d
                    except Exception:
                        pass
            if "GPU.1" in av:
                return "GPU.1"

        # Check generic GPU (prefer NVIDIA dGPU if present, else first GPU)
        if qu == "GPU":
            for d in av:
                if "GPU" in d:
                    try:
                        name = self._core.get_property(d, "FULL_DEVICE_NAME").upper()
                        if "NVIDIA" in name or "GEFORCE" in name:
                            return d
                    except Exception:
                        pass
            for d in av:
                if "GPU" in d:
                    return d

        # Check Intel integrated GPU
        if any(k in qu for k in ("INTEL", "IGPU")):
            for d in av:
                if "GPU" in d:
                    try:
                        name = self._core.get_property(d, "FULL_DEVICE_NAME").upper()
                        if "INTEL" in name:
                            return d
                    except Exception:
                        pass
            if "GPU.0" in av:
                return "GPU.0"

        return device_str

    def get_device_options(self) -> List[Dict[str, str]]:
        """Returns friendly device options with descriptive labels for the UI."""
        options = []
        if not self._core:
            options.append({"id": "CPU", "name": "⚙️ CPU (Default)"})
            return options

        av = self._core.available_devices

        # 1. Dedicated NPU (top priority for PrivacyCam)
        for d in av:
            if "NPU" in d:
                try:
                    full_name = self._core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = "Intel AI Boost"
                options.append({"id": d, "name": f"NPU: {full_name}"})

        # 2. Discrete NVIDIA GPU
        for d in av:
            if "GPU" in d:
                try:
                    full_name = self._core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = d
                if "NVIDIA" in full_name.upper() or "GEFORCE" in full_name.upper():
                    options.append({"id": d, "name": f"NVIDIA GPU: {full_name} ({d})"})

        # 3. Integrated Intel GPU
        for d in av:
            if "GPU" in d:
                try:
                    full_name = self._core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = d
                if "INTEL" in full_name.upper():
                    options.append({"id": d, "name": f"Intel GPU: {full_name} ({d})"})

        # 4. CPU
        for d in av:
            if d == "CPU":
                try:
                    full_name = self._core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    full_name = "Intel CPU"
                options.append({"id": "CPU", "name": f"CPU: {full_name}"})

        # 5. AUTO option
        options.append({"id": "AUTO", "name": "AUTO: Automatic Selection"})

        return options

    def is_npu_available(self) -> bool:
        return any("NPU" in d for d in self.get_available_devices())

    def is_gpu_available(self) -> bool:
        return any("GPU" in d for d in self.get_available_devices())

    def get_device_info(self, device: str) -> dict:
        if not self._core:
            return {"name": device, "type": "CPU", "full_name": "CPU Mock"}

        resolved = self.resolve_device(device)
        try:
            full_name = self._core.get_property(resolved, "FULL_DEVICE_NAME")
        except Exception:
            full_name = resolved

        dev_type = "CPU"
        if "NPU" in resolved:
            dev_type = "NPU"
        elif "GPU" in resolved:
            dev_type = "GPU"

        return {
            "name": resolved,
            "type": dev_type,
            "full_name": full_name
        }

    def get_optimal_device(self) -> str:
        """Prefers NPU for energy efficiency and offloading, then NVIDIA dGPU, then iGPU, then CPU."""
        devices = self.get_available_devices()
        if any("NPU" in d for d in devices):
            return "NPU"
        for d in devices:
            if "GPU" in d:
                try:
                    name = self._core.get_property(d, "FULL_DEVICE_NAME").upper()
                    if "NVIDIA" in name or "GEFORCE" in name:
                        return d
                except Exception:
                    pass
        for d in devices:
            if "GPU" in d:
                return d
        return "CPU"

    def validate_device(self, device: str) -> bool:
        if device == "AUTO":
            return True
        resolved = self.resolve_device(device)
        return resolved in self.get_available_devices()
