import abc
from typing import List
from privacycam.capture.frame import Frame
from privacycam.detection.detection import FaceDetection

class FaceDetector(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame: Frame) -> List[FaceDetection]:
        pass

    @abc.abstractmethod
    def warmup(self) -> None:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def is_ready(self) -> bool:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> "FaceDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
