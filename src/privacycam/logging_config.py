"""Logging configuration for PrivacyCam."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from privacycam.constants import LOG_FORMAT, LOG_DATE_FORMAT

class ColorFormatter(logging.Formatter):
    """Formatter to add colors to log levels."""
    
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(level: str = "INFO", log_file: str | None = None, verbose: bool = False) -> None:
    """Setup logging for the application."""
    
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger("privacycam")
    logger.setLevel(log_level)
    logger.propagate = False
    
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    color_formatter = ColorFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    # Suppress noisy third-party loggers
    logging.getLogger("openvino").setLevel(logging.WARNING)
    logging.getLogger("PySide6").setLevel(logging.WARNING)
