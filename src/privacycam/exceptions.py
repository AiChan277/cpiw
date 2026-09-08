"""Exception hierarchy for PrivacyCam."""

class PrivacyCamError(Exception):
    """Base exception for PrivacyCam."""
    pass

class CameraError(PrivacyCamError):
    """Base exception for camera-related errors."""
    pass

class CameraInitializationError(CameraError):
    """Raised when a camera fails to initialize."""
    pass

class CameraNotFoundError(CameraError):
    """Raised when a requested camera is not found."""
    pass

class CameraDisconnectedError(CameraError):
    """Raised when a camera unexpectedly disconnects."""
    pass

class CameraPermissionError(CameraError):
    """Raised when there is a permission error accessing the camera."""
    pass

class ModelError(PrivacyCamError):
    """Base exception for model-related errors."""
    pass

class ModelLoadError(ModelError):
    """Raised when a model fails to load."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a model file is not found."""
    pass

class ModelValidationError(ModelError):
    """Raised when a model validation fails."""
    pass

class InferenceError(PrivacyCamError):
    """Base exception for inference-related errors."""
    pass

class DeviceNotAvailableError(InferenceError):
    """Raised when the requested inference device is not available."""
    pass

class VirtualCameraError(PrivacyCamError):
    """Raised when virtual camera operations fail."""
    pass

class ConfigurationError(PrivacyCamError):
    """Raised when there is a configuration error."""
    pass

class PipelineError(PrivacyCamError):
    """Base exception for pipeline-related errors."""
    pass

class PipelineOverloadError(PipelineError):
    """Raised when the processing pipeline is overloaded."""
    pass
