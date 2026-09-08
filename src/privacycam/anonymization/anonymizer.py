from abc import ABC, abstractmethod
import numpy as np
from privacycam.detection.detection import BoundingBox
from privacycam.anonymization.modes import AnonymizationMode

class Anonymizer(ABC):
    @abstractmethod
    def anonymize(self, frame: np.ndarray, mask: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def mode(self) -> AnonymizationMode:
        pass

class AnonymizerFactory:
    @staticmethod
    def create(mode: AnonymizationMode, **kwargs) -> Anonymizer:
        if mode == AnonymizationMode.BLUR:
            from privacycam.anonymization.blur import GaussianBlurAnonymizer
            return GaussianBlurAnonymizer(strength=kwargs.get('strength', 31))
        elif mode == AnonymizationMode.PIXELATE:
            from privacycam.anonymization.pixelate import PixelateAnonymizer
            return PixelateAnonymizer(blocks=kwargs.get('blocks', 10))
        elif mode == AnonymizationMode.SOLID:
            from privacycam.anonymization.solid import SolidAnonymizer
            return SolidAnonymizer(color=kwargs.get('color', (0, 0, 0)))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
