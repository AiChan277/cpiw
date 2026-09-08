from typing import List, Dict
from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QComboBox, QSpinBox, 
    QCheckBox, QPushButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import Signal, Slot

class SettingsPanel(QWidget):
    device_changed = Signal(str)
    mode_changed = Signal(str)
    interval_changed = Signal(int)
    tracking_changed = Signal(bool)
    debug_changed = Signal(bool)
    virtual_camera_changed = Signal(bool)
    log_level_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group_box = QGroupBox("Settings & Performance", self)
        gb_layout = QVBoxLayout(group_box)
        
        # Performance Profile
        pl = QVBoxLayout()
        pl.addWidget(QLabel("Performance Profile:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("⚡ Realtime (< 20 ms)", "realtime")
        self._mode_combo.addItem("🍃 Low (Eco / Battery Saver)", "low")
        self._mode_combo.addItem("⚖️ Medium (Balanced)", "medium")
        self._mode_combo.addItem("💎 High (Studio Quality)", "high")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        pl.addWidget(self._mode_combo)
        gb_layout.addLayout(pl)

        # Device Selector
        dl = QVBoxLayout()
        dl.addWidget(QLabel("Inference Device:"))
        self._device_combo = QComboBox()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        dl.addWidget(self._device_combo)
        gb_layout.addLayout(dl)
        
        # Debug overlay
        self._debug_cb = QCheckBox("Debug overlay (show bboxes)")
        self._debug_cb.setChecked(False)
        self._debug_cb.toggled.connect(self.debug_changed.emit)
        gb_layout.addWidget(self._debug_cb)
        
        self._vcam_cb = QCheckBox("Virtual camera (OBS/Discord/Zoom)")
        self._vcam_cb.setChecked(True)
        self._vcam_cb.toggled.connect(self.virtual_camera_changed.emit)
        gb_layout.addWidget(self._vcam_cb)
        
        layout.addWidget(group_box)

    def populate_devices(self, devices: List[Dict[str, str]], current_device: str = "AUTO") -> None:
        """Populates the device selector with actual hardware devices."""
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        
        from privacycam.services.device_service import DeviceService
        svc = DeviceService()
        resolved_curr = svc.resolve_device(current_device)

        selected_idx = 0
        for i, d in enumerate(devices):
            self._device_combo.addItem(d["name"], d["id"])
            if d["id"] == current_device or d["id"] == resolved_curr:
                selected_idx = i
        self._device_combo.setCurrentIndex(selected_idx)
        self._device_combo.blockSignals(False)

    @Slot(int)
    def _on_device_changed(self, index: int) -> None:
        device_id = self._device_combo.currentData()
        if device_id:
            self.device_changed.emit(device_id)

    @Slot(int)
    def _on_mode_changed(self, index: int) -> None:
        mode_id = self._mode_combo.currentData()
        if mode_id:
            self.mode_changed.emit(mode_id)
