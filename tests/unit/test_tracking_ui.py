import os
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from privacycam.ui.tracking_panel import TrackingPanel, TRACKING_PRESETS
from privacycam.config import AppConfig
from privacycam.services.application_service import ApplicationService


@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tracking_panel_init_and_signals(qapp):
    panel = TrackingPanel()
    
    received = {}
    panel.tracking_toggled.connect(lambda val: received.setdefault("tracking", val))
    panel.interval_changed.connect(lambda val: received.setdefault("interval", val))
    panel.smoothing_changed.connect(lambda val: received.setdefault("smoothing", val))
    panel.persistence_changed.connect(lambda val: received.setdefault("persistence", val))
    panel.distance_changed.connect(lambda val: received.setdefault("distance", val))
    panel.iou_changed.connect(lambda val: received.setdefault("iou", val))
    panel.preset_changed.connect(lambda val: received.setdefault("preset", val))

    # Test slider modifications
    panel._interval_slider.setValue(3)
    assert received.get("interval") == 3
    assert panel._preset_combo.currentData() == "custom"

    panel._smoothing_slider.setValue(80)
    assert abs(received.get("smoothing") - 0.80) < 1e-3

    panel._persistence_slider.setValue(30)
    assert received.get("persistence") == 30

    panel._distance_slider.setValue(20)
    assert abs(received.get("distance") - 2.0) < 1e-3

    panel._iou_slider.setValue(15)
    assert abs(received.get("iou") - 0.15) < 1e-3

    # Test preset selection
    panel._preset_combo.setCurrentIndex(panel._preset_combo.findData("responsive"))
    assert received.get("preset") == "responsive"
    assert panel._interval_slider.value() == TRACKING_PRESETS["responsive"]["interval"]
    assert panel._smoothing_slider.value() == int(round(TRACKING_PRESETS["responsive"]["smoothing"] * 100))


def test_application_service_tracking_methods(qapp):
    config = AppConfig()
    service = ApplicationService(config)
    
    # Test tracking update proxy methods
    service.update_tracking_smoothing(0.55)
    assert config.tracking.smoothing_factor == 0.55

    service.update_tracking_persistence(40)
    assert config.tracking.max_lost_frames == 40

    service.update_tracking_distance(2.5)
    assert config.tracking.distance_threshold == 2.5

    service.update_tracking_iou(0.18)
    assert config.tracking.iou_threshold == 0.18

    service.apply_tracking_preset("stationary")
    assert config.tracking.preset == "stationary"
    assert config.detection.interval == TRACKING_PRESETS["stationary"]["interval"]
    assert config.tracking.smoothing_factor == TRACKING_PRESETS["stationary"]["smoothing"]
    assert config.tracking.max_lost_frames == TRACKING_PRESETS["stationary"]["persistence"]
