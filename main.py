"""PrivacyCam — Run with: py main.py"""

import sys
import os

# Add src to path so privacycam package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from privacycam.app import PrivacyCamApp
from privacycam.config import AppConfig
from privacycam.logging_config import setup_logging


def main():
    setup_logging(level="INFO")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
    if os.path.exists(config_path):
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig.default()
    app = PrivacyCamApp(config)
    try:
        app.run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
