import abc
from privacycam.capture.frame import Frame

class VideoOutput(abc.ABC):
    """Abstract base class for video outputs."""
    
    @abc.abstractmethod
    def start(self) -> None:
        """Initializes and starts the video output."""
        pass

    @abc.abstractmethod
    def write(self, frame: Frame) -> None:
        """Writes a frame to the output."""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stops the output and cleans up resources."""
        pass

    @abc.abstractmethod
    def is_active(self) -> bool:
        """Returns True if the output is currently active."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
