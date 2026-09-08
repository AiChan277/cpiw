"""TrackingPanel — UI panel for adjusting face tracking and anti-jitter parameters."""

from typing import Dict, Any
from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QCheckBox
)
from PySide6.QtCore import Qt, Signal, Slot

from privacycam.tracking.presets import TRACKING_PRESETS


class TrackingPanel(QWidget):
    """Control panel for real-time face tracking parameters and presets."""

    preset_changed = Signal(str)
    tracking_toggled = Signal(bool)
    interval_changed = Signal(int)
    smoothing_changed = Signal(float)
    persistence_changed = Signal(int)
    distance_changed = Signal(float)
    iou_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._updating_ui = False

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        group_box = QGroupBox("Tracking & Anti-Jitter", self)
        gb_layout = QVBoxLayout(group_box)
        gb_layout.setSpacing(6)

        # 1. Enable Face Tracking Checkbox
        self._enable_cb = QCheckBox("Enable Face Tracking", group_box)
        self._enable_cb.setChecked(True)
        self._enable_cb.toggled.connect(self._on_tracking_toggled)
        gb_layout.addWidget(self._enable_cb)

        # 2. Preset Selector
        preset_layout = QVBoxLayout()
        preset_layout.addWidget(QLabel("Tracking Preset:"))
        self._preset_combo = QComboBox(group_box)
        for key, p in TRACKING_PRESETS.items():
            self._preset_combo.addItem(p["name"], key)
        self._preset_combo.addItem("⚙️ Custom", "custom")
        self._preset_combo.currentIndexChanged.connect(self._on_preset_combo_changed)
        preset_layout.addWidget(self._preset_combo)
        gb_layout.addLayout(preset_layout)

        # 3. Detection Interval (Polling cadence)
        int_header = QHBoxLayout()
        int_header.addWidget(QLabel("Detection Polling:"))
        self._interval_label = QLabel("Every 2 frames")
        self._interval_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        int_header.addWidget(self._interval_label)
        gb_layout.addLayout(int_header)

        self._interval_slider = QSlider(Qt.Orientation.Horizontal, group_box)
        self._interval_slider.setRange(1, 6)
        self._interval_slider.setValue(2)
        self._interval_slider.valueChanged.connect(self._on_interval_slider_changed)
        gb_layout.addWidget(self._interval_slider)

        # 4. Anti-Jitter Smoothing Alpha (0.10 to 1.00)
        sm_header = QHBoxLayout()
        sm_header.addWidget(QLabel("Coordinate Smoothing:"))
        self._smoothing_label = QLabel("65% (Smooth)")
        self._smoothing_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        sm_header.addWidget(self._smoothing_label)
        gb_layout.addLayout(sm_header)

        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal, group_box)
        self._smoothing_slider.setRange(10, 100)
        self._smoothing_slider.setValue(65)
        self._smoothing_slider.valueChanged.connect(self._on_smoothing_slider_changed)
        gb_layout.addWidget(self._smoothing_slider)

        # 5. Persistence / Grace Period (frames)
        per_header = QHBoxLayout()
        per_header.addWidget(QLabel("Persistence (Grace):"))
        self._persistence_label = QLabel("25 frames (~0.8s)")
        self._persistence_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        per_header.addWidget(self._persistence_label)
        gb_layout.addLayout(per_header)

        self._persistence_slider = QSlider(Qt.Orientation.Horizontal, group_box)
        self._persistence_slider.setRange(5, 60)
        self._persistence_slider.setValue(25)
        self._persistence_slider.valueChanged.connect(self._on_persistence_slider_changed)
        gb_layout.addWidget(self._persistence_slider)

        # 6. Fast-Motion Reach (distance fallback factor: 0.5x to 3.0x)
        dist_header = QHBoxLayout()
        dist_header.addWidget(QLabel("Fast-Motion Reach:"))
        self._distance_label = QLabel("1.5x")
        self._distance_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        dist_header.addWidget(self._distance_label)
        gb_layout.addLayout(dist_header)

        self._distance_slider = QSlider(Qt.Orientation.Horizontal, group_box)
        self._distance_slider.setRange(5, 30)  # maps to 0.5x .. 3.0x
        self._distance_slider.setValue(15)     # 1.5x
        self._distance_slider.valueChanged.connect(self._on_distance_slider_changed)
        gb_layout.addWidget(self._distance_slider)

        # 7. Match Strictness / IoU Threshold (0.05 to 0.50)
        iou_header = QHBoxLayout()
        iou_header.addWidget(QLabel("IoU Match Strictness:"))
        self._iou_label = QLabel("20%")
        self._iou_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        iou_header.addWidget(self._iou_label)
        gb_layout.addLayout(iou_header)

        self._iou_slider = QSlider(Qt.Orientation.Horizontal, group_box)
        self._iou_slider.setRange(5, 50)       # maps to 0.05 .. 0.50
        self._iou_slider.setValue(20)          # 0.20
        self._iou_slider.valueChanged.connect(self._on_iou_slider_changed)
        gb_layout.addWidget(self._iou_slider)

        root_layout.addWidget(group_box)

    # ------------------------------------------------------------------
    # Public setters for synchronizing with AppConfig / presets
    # ------------------------------------------------------------------

    def set_tracking_enabled(self, enabled: bool) -> None:
        self._enable_cb.blockSignals(True)
        self._enable_cb.setChecked(enabled)
        self._enable_cb.blockSignals(False)

    def set_preset(self, preset_key: str) -> None:
        idx = self._preset_combo.findData(preset_key)
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def set_values(
        self,
        interval: int,
        smoothing: float,
        persistence: int,
        distance: float,
        iou: float,
    ) -> None:
        """Sets slider values programmatically without switching preset to Custom."""
        self._updating_ui = True
        try:
            self._interval_slider.setValue(interval)
            self._update_interval_label(interval)

            self._smoothing_slider.setValue(int(round(smoothing * 100)))
            self._update_smoothing_label(smoothing)

            self._persistence_slider.setValue(persistence)
            self._update_persistence_label(persistence)

            self._distance_slider.setValue(int(round(distance * 10)))
            self._update_distance_label(distance)

            self._iou_slider.setValue(int(round(iou * 100)))
            self._update_iou_label(iou)
        finally:
            self._updating_ui = False

    # ------------------------------------------------------------------
    # Internal Slots & Event Handlers
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_tracking_toggled(self, checked: bool) -> None:
        self.tracking_toggled.emit(checked)

    @Slot(int)
    def _on_preset_combo_changed(self, index: int) -> None:
        preset_key = self._preset_combo.currentData()
        if not preset_key or preset_key == "custom":
            return

        preset = TRACKING_PRESETS.get(preset_key)
        if not preset:
            return

        self._updating_ui = True
        try:
            self._interval_slider.setValue(preset["interval"])
            self._update_interval_label(preset["interval"])

            self._smoothing_slider.setValue(int(round(preset["smoothing"] * 100)))
            self._update_smoothing_label(preset["smoothing"])

            self._persistence_slider.setValue(preset["persistence"])
            self._update_persistence_label(preset["persistence"])

            self._distance_slider.setValue(int(round(preset["distance"] * 10)))
            self._update_distance_label(preset["distance"])

            self._iou_slider.setValue(int(round(preset["iou"] * 100)))
            self._update_iou_label(preset["iou"])
        finally:
            self._updating_ui = False

        self.preset_changed.emit(preset_key)

    def _mark_custom(self) -> None:
        """Switches preset dropdown to 'Custom' when user drags any slider."""
        if self._updating_ui:
            return
        idx = self._preset_combo.findData("custom")
        if idx >= 0 and self._preset_combo.currentIndex() != idx:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentIndex(idx)
            self._preset_combo.blockSignals(False)

    @Slot(int)
    def _on_interval_slider_changed(self, val: int) -> None:
        self._update_interval_label(val)
        self._mark_custom()
        self.interval_changed.emit(val)

    @Slot(int)
    def _on_smoothing_slider_changed(self, val: int) -> None:
        alpha = val / 100.0
        self._update_smoothing_label(alpha)
        self._mark_custom()
        self.smoothing_changed.emit(alpha)

    @Slot(int)
    def _on_persistence_slider_changed(self, val: int) -> None:
        self._update_persistence_label(val)
        self._mark_custom()
        self.persistence_changed.emit(val)

    @Slot(int)
    def _on_distance_slider_changed(self, val: int) -> None:
        dist = val / 10.0
        self._update_distance_label(dist)
        self._mark_custom()
        self.distance_changed.emit(dist)

    @Slot(int)
    def _on_iou_slider_changed(self, val: int) -> None:
        iou = val / 100.0
        self._update_iou_label(iou)
        self._mark_custom()
        self.iou_changed.emit(iou)

    # ------------------------------------------------------------------
    # Label formatting helpers
    # ------------------------------------------------------------------

    def _update_interval_label(self, val: int) -> None:
        if val == 1:
            text = "Every frame (Max CPU/NPU)"
        else:
            text = f"Every {val} frames (~{int(30/val)} Hz)"
        self._interval_label.setText(text)

    def _update_smoothing_label(self, alpha: float) -> None:
        pct = int(round(alpha * 100))
        if pct <= 50:
            desc = "Heavy damping"
        elif pct <= 75:
            desc = "Smooth"
        else:
            desc = "Snappy"
        self._smoothing_label.setText(f"{pct}% ({desc})")

    def _update_persistence_label(self, frames: int) -> None:
        seconds = frames / 30.0
        self._persistence_label.setText(f"{frames} frames (~{seconds:.1f}s)")

    def _update_distance_label(self, dist: float) -> None:
        self._distance_label.setText(f"{dist:.1f}x face size")

    def _update_iou_label(self, iou: float) -> None:
        pct = int(round(iou * 100))
        desc = "Lenient" if pct <= 15 else ("Balanced" if pct <= 25 else "Strict")
        self._iou_label.setText(f"{pct}% ({desc})")
