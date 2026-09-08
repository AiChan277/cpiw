from PySide6.QtWidgets import QWidget, QGroupBox, QVBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import QTimer

class PerformancePanel(QWidget):
    """Telemetry display showing live FPS, latency, inference, and hardware."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group_box = QGroupBox("Performance Telemetry", self)
        gb_layout = QGridLayout(group_box)
        
        self._mode_label = QLabel("Profile: REALTIME")
        self._mode_label.setStyleSheet("font-weight: bold; color: #38bdf8;")
        self._fps_label = QLabel("FPS: -")
        self._latency_label = QLabel("Latency: -")
        self._inference_label = QLabel("AI Inference: -")
        self._faces_label = QLabel("Faces: 0")
        self._device_label = QLabel("Device: -")
        
        gb_layout.addWidget(self._mode_label, 0, 0)
        gb_layout.addWidget(self._fps_label, 0, 1)
        gb_layout.addWidget(self._latency_label, 1, 0)
        gb_layout.addWidget(self._inference_label, 1, 1)
        gb_layout.addWidget(self._faces_label, 2, 0)
        gb_layout.addWidget(self._device_label, 2, 1)
        
        layout.addWidget(group_box)
        
        self._latest_metrics = {}
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._refresh_display)
        self._update_timer.start(250)

    def update_metrics(self, metrics: dict) -> None:
        self._latest_metrics.update(metrics)

    def _refresh_display(self):
        m = self._latest_metrics
        mode = m.get("mode", "REALTIME")
        lat = m.get("latency", 0.0)
        fps = m.get("fps", 0.0)
        infer = m.get("inference_time", 0.0)
        faces = m.get("faces", 0)
        device = m.get("device", "N/A")

        self._mode_label.setText(f"Profile: {mode}")
        self._fps_label.setText(f"FPS: {fps:.1f}")

        # Color-code latency: Green when < 20 ms!
        if lat < 20.0:
            color = "#22c55e" # Green
        elif lat < 35.0:
            color = "#eab308" # Yellow
        else:
            color = "#ef4444" # Red

        self._latency_label.setText(f"Latency: {lat:.1f} ms")
        self._latency_label.setStyleSheet(f"font-weight: bold; color: {color};")
        self._inference_label.setText(f"AI: {infer:.1f} ms")
        self._faces_label.setText(f"Faces: {faces}")
        self._device_label.setText(f"Dev: {device}")
