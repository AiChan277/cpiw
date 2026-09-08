from enum import Enum

class AnonymizationMode(Enum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID = "solid"
    
    @classmethod
    def from_string(cls, value: str) -> "AnonymizationMode":
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Unknown anonymization mode: {value}. Supported: {[m.value for m in cls]}")
