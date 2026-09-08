from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QComboBox, QPushButton
from PySide6.QtCore import Signal, Slot

class CameraPanel(QWidget):
    camera_changed = Signal(int)
    resolution_changed = Signal(int, int)
    fps_changed = Signal(int)
    refresh_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group_box = QGroupBox("Camera", self)
        gb_layout = QVBoxLayout(group_box)
        
        self._camera_combo = QComboBox(group_box)
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        gb_layout.addWidget(self._camera_combo)
        
        self._resolution_combo = QComboBox(group_box)
        self.populate_resolutions()
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        gb_layout.addWidget(self._resolution_combo)
        
        self._fps_combo = QComboBox(group_box)
        self._fps_combo.addItems(["60", "30", "24", "15"])
        self._fps_combo.currentIndexChanged.connect(self._on_fps_changed)
        gb_layout.addWidget(self._fps_combo)
        
        self._refresh_btn = QPushButton("Refresh Cameras", group_box)
        self._refresh_btn.clicked.connect(self.refresh_requested)
        gb_layout.addWidget(self._refresh_btn)
        
        layout.addWidget(group_box)

    def populate_resolutions(self, resolutions=None) -> None:
        """Populate camera resolution options."""
        self._resolution_combo.blockSignals(True)
        self._resolution_combo.clear()
        
        default_resolutions = [
            (2560, 1440, "2560x1440 (2K QHD - Native Max)"),
            (1920, 1080, "1920x1080 (Full HD 1080p)"),
            (1280, 720, "1280x720 (HD 720p - Fast)"),
            (960, 540, "960x540 (qHD 540p)"),
            (640, 480, "640x480 (SD 480p - Eco)"),
        ]
        items = resolutions if resolutions else default_resolutions
        for item in items:
            if isinstance(item, tuple) and len(item) == 3:
                w, h, label = item
            elif isinstance(item, tuple) and len(item) == 2:
                w, h = item
                label = f"{w}x{h}"
            else:
                continue
            self._resolution_combo.addItem(label, (w, h))
        self._resolution_combo.blockSignals(False)

    def set_current_resolution(self, width: int, height: int) -> None:
        """Select the matching resolution in combobox."""
        self._resolution_combo.blockSignals(True)
        for i in range(self._resolution_combo.count()):
            data = self._resolution_combo.itemData(i)
            if data == (width, height):
                self._resolution_combo.setCurrentIndex(i)
                break
        self._resolution_combo.blockSignals(False)

    def set_current_fps(self, fps: int) -> None:
        """Select the matching FPS in combobox."""
        self._fps_combo.blockSignals(True)
        idx = self._fps_combo.findText(str(int(fps)))
        if idx >= 0:
            self._fps_combo.setCurrentIndex(idx)
        self._fps_combo.blockSignals(False)

    @Slot(int)
    def _on_camera_changed(self, index: int):
        data = self._camera_combo.currentData()
        dev_idx = data if data is not None else index
        self.camera_changed.emit(dev_idx)

    @Slot(int)
    def _on_resolution_changed(self, index: int):
        data = self._resolution_combo.currentData()
        if data and isinstance(data, (tuple, list)):
            w, h = data
            self.resolution_changed.emit(w, h)
        else:
            raw = self._resolution_combo.currentText().split()[0]
            if "x" in raw:
                try:
                    w, h = map(int, raw.split("x"))
                    self.resolution_changed.emit(w, h)
                except ValueError:
                    pass

    @Slot(int)
    def _on_fps_changed(self, index: int):
        fps_str = self._fps_combo.currentText()
        if fps_str:
            try:
                self.fps_changed.emit(int(float(fps_str)))
            except ValueError:
                pass

