"""Configuration classes for PrivacyCam."""

import dataclasses
from typing import Any
import yaml
from pathlib import Path
from privacycam.exceptions import ConfigurationError

@dataclasses.dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    pixel_format: str = "bgr"
    auto_reconnect: bool = True
    reconnect_delay: float = 2.0

@dataclasses.dataclass
class DetectionConfig:
    device: str = "AUTO"
    model_path: str = ""
    confidence_threshold: float = 0.5
    interval: int = 1
    input_width: int = 672
    input_height: int = 384

@dataclasses.dataclass
class TrackingConfig:
    enabled: bool = True
    max_lost_frames: int = 25
    iou_threshold: float = 0.20
    min_hits: int = 1
    smoothing_factor: float = 0.70
    distance_threshold: float = 1.50
    preset: str = "smooth"

@dataclasses.dataclass
class AnonymizationConfig:
    mode: str = "blur"
    blur_strength: int = 51
    pixelate_blocks: int = 12
    solid_color: tuple[int, int, int] = (0, 0, 0)
    expansion: float = 0.20
    feather: int = 15

@dataclasses.dataclass
class OutputConfig:
    virtual_camera: bool = True
    preview: bool = True
    preview_width: int = 1280
    preview_height: int = 720

@dataclasses.dataclass
class PerformanceConfig:
    max_queue_size: int = 2
    enable_metrics: bool = True
    warmup_iterations: int = 3
    drop_frames: bool = True

@dataclasses.dataclass
class UIConfig:
    theme: str = "system"
    debug_overlay: bool = False
    show_fps: bool = True
    show_detections: bool = False

@dataclasses.dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str | None = None
    verbose: bool = False

@dataclasses.dataclass
class AppConfig:
    camera: CameraConfig = dataclasses.field(default_factory=CameraConfig)
    detection: DetectionConfig = dataclasses.field(default_factory=DetectionConfig)
    tracking: TrackingConfig = dataclasses.field(default_factory=TrackingConfig)
    anonymization: AnonymizationConfig = dataclasses.field(default_factory=AnonymizationConfig)
    output: OutputConfig = dataclasses.field(default_factory=OutputConfig)
    performance: PerformanceConfig = dataclasses.field(default_factory=PerformanceConfig)
    ui: UIConfig = dataclasses.field(default_factory=UIConfig)
    logging: LoggingConfig = dataclasses.field(default_factory=LoggingConfig)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {path}: {e}")
            
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Construct configuration from a dictionary."""
        config = cls()
        config.merge(data)
        return config
        
    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        return dataclasses.asdict(self)
        
    def merge(self, overrides: dict[str, Any]) -> None:
        """Deep merge another configuration dictionary."""
        for section_name, section_data in overrides.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        
    @classmethod
    def default(cls) -> "AppConfig":
        """Return default configuration."""
        return cls()
