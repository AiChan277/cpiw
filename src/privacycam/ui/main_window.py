"""PrivacyCam main window — PySide6 UI fully wired to ApplicationService."""

import logging
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QStatusBar, QPushButton, QSizePolicy, QScrollArea, QMessageBox,
)
from PySide6.QtCore import Qt, Slot, QRect
from PySide6.QtGui import QImage, QPainter, QColor, QCloseEvent, QFont

from privacycam.config import AppConfig
from privacycam.services.application_service import ApplicationService
from privacycam.services.device_service import DeviceService
from privacycam.ui.camera_panel import CameraPanel
from privacycam.ui.privacy_panel import PrivacyPanel
from privacycam.ui.tracking_panel import TrackingPanel
from privacycam.ui.performance_panel import PerformancePanel
from privacycam.ui.settings_panel import SettingsPanel

logger = logging.getLogger(__name__)


class VideoWidget(QWidget):
    """Zero-overhead hardware-accelerated video rendering widget.

    Paints frames directly using QPainter without intermediate software
    CPU pixmap scaling, eliminating video lag and stutter at high resolutions.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._image: Optional[QImage] = None
        self._placeholder_text = "Camera Feed Ready\nClick '▶  Start Pipeline' to begin"
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #0f172a; border-radius: 8px;")

    def set_frame(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return

        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1

        if ch == 3:
            fmt = QImage.Format.Format_BGR888
            bpl = 3 * w
        elif ch == 4:
            fmt = QImage.Format.Format_RGBA8888
            bpl = 4 * w
        else:
            return

        self._image = QImage(frame.data, w, h, bpl, fmt).copy()
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder_text = text
        self._image = None
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        rect = self.rect()

        if self._image is None or self._image.isNull():
            painter.fillRect(rect, QColor("#0f172a"))
            painter.setPen(QColor("#94a3b8"))
            font = painter.font()
            font.setPointSize(15)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._placeholder_text)
            return

        img_w = self._image.width()
        img_h = self._image.height()
        widget_w = rect.width()
        widget_h = rect.height()

        if widget_w <= 0 or widget_h <= 0 or img_w <= 0 or img_h <= 0:
            return

        scale = min(widget_w / img_w, widget_h / img_h)
        target_w = int(img_w * scale)
        target_h = int(img_h * scale)
        target_x = (widget_w - target_w) // 2
        target_y = (widget_h - target_h) // 2
        target_rect = QRect(target_x, target_y, target_w, target_h)

        # Clear letterbox background margins
        painter.fillRect(rect, QColor("#0f172a"))
        # Render frame (Direct3D/OpenGL hardware accelerated scaling)
        painter.drawImage(target_rect, self._image)


class MainWindow(QMainWindow):
    """Main window for PrivacyCam."""

    def __init__(
        self,
        config: AppConfig,
        service: ApplicationService,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._config = config
        self._service = service

        self.setWindowTitle("PrivacyCam - AI Privacy Filter")
        self.setMinimumSize(1080, 640)

        self._build_ui()
        self._connect_service_signals()
        self._connect_panel_signals()
        self._populate_devices()
        self._populate_cameras()

        # Initialize the service
        self._service.initialize()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        # ---------- Left: video preview ----------
        self._video_widget = VideoWidget(self)
        root.addWidget(self._video_widget, stretch=72)

        # ---------- Right: control panel (scrollable) ----------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumWidth(340)
        scroll.setMinimumWidth(280)

        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)
        ctrl_layout.setSpacing(6)

        # Start / Stop button
        self._start_btn = QPushButton("▶  Start Pipeline")
        self._start_btn.setCheckable(True)
        self._start_btn.setMinimumHeight(44)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #16a34a; color: white; "
            "font-size: 15px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #15803d; }"
            "QPushButton:checked { background-color: #dc2626; }"
            "QPushButton:checked:hover { background-color: #b91c1c; }"
        )
        self._start_btn.clicked.connect(self._on_start_stop)
        ctrl_layout.addWidget(self._start_btn)

        # Subsystem Panels
        self._camera_panel = CameraPanel(self)
        self._privacy_panel = PrivacyPanel(self)
        self._tracking_panel = TrackingPanel(self)
        self._settings_panel = SettingsPanel(self)
        self._performance_panel = PerformancePanel(self)

        ctrl_layout.addWidget(self._camera_panel)
        ctrl_layout.addWidget(self._privacy_panel)
        ctrl_layout.addWidget(self._tracking_panel)
        ctrl_layout.addWidget(self._settings_panel)
        ctrl_layout.addWidget(self._performance_panel)
        ctrl_layout.addStretch()

        # Initialize tracking panel controls from config
        self._tracking_panel.set_tracking_enabled(self._config.tracking.enabled)
        self._tracking_panel.set_values(
            interval=self._config.detection.interval,
            smoothing=self._config.tracking.smoothing_factor,
            persistence=self._config.tracking.max_lost_frames,
            distance=self._config.tracking.distance_threshold,
            iou=self._config.tracking.iou_threshold,
        )
        self._tracking_panel.set_preset(self._config.tracking.preset)

        scroll.setWidget(controls)
        root.addWidget(scroll, stretch=28)

        # ---------- Status bar ----------
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._status_label.setFont(QFont("Consolas", 9))
        self._status_bar.addPermanentWidget(self._status_label)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_service_signals(self) -> None:
        """Connect ApplicationService signals -> UI update slots."""
        self._service.frame_ready.connect(self._on_frame_ready)
        self._service.metrics_updated.connect(self._on_metrics_updated)
        self._service.status_changed.connect(self._on_status_changed)
        self._service.error_occurred.connect(self._on_error)

    def _connect_panel_signals(self) -> None:
        """Connect UI panel signals -> ApplicationService methods."""
        # Camera panel
        self._camera_panel.camera_changed.connect(self._service.update_camera)
        self._camera_panel.resolution_changed.connect(self._service.update_resolution)
        self._camera_panel.fps_changed.connect(self._service.update_fps)
        self._camera_panel.refresh_requested.connect(self._populate_cameras)

        # Privacy panel
        self._privacy_panel.mode_changed.connect(self._service.update_anonymization_mode)
        self._privacy_panel.strength_changed.connect(self._service.update_blur_strength)
        self._privacy_panel.blocks_changed.connect(self._service.update_pixelate_blocks)
        self._privacy_panel.color_changed.connect(self._service.update_solid_color)
        self._privacy_panel.expansion_changed.connect(self._service.update_expansion)
        self._privacy_panel.feather_changed.connect(self._service.update_feather)
        self._privacy_panel.confidence_changed.connect(self._service.update_confidence)

        # Tracking panel
        self._tracking_panel.tracking_toggled.connect(self._service.update_tracking_enabled)
        self._tracking_panel.interval_changed.connect(self._service.update_detection_interval)
        self._tracking_panel.smoothing_changed.connect(self._service.update_tracking_smoothing)
        self._tracking_panel.persistence_changed.connect(self._service.update_tracking_persistence)
        self._tracking_panel.distance_changed.connect(self._service.update_tracking_distance)
        self._tracking_panel.iou_changed.connect(self._service.update_tracking_iou)
        self._tracking_panel.preset_changed.connect(self._service.apply_tracking_preset)

        # Settings panel
        self._settings_panel.device_changed.connect(self._service.update_device)
        self._settings_panel.mode_changed.connect(self._service.set_performance_mode)
        self._settings_panel.debug_changed.connect(self._service.update_debug_overlay)
        self._settings_panel.virtual_camera_changed.connect(self._service.update_virtual_camera)

    def _populate_devices(self) -> None:
        """Populates available inference hardware (CPU, Intel GPU, NVIDIA GPU, NPU)."""
        device_service = DeviceService()
        options = device_service.get_device_options()
        self._settings_panel.populate_devices(
            options, current_device=self._config.detection.device
        )

    def _populate_cameras(self) -> None:
        """Refresh camera list and resolution options in the camera panel."""
        cameras = self._service.get_available_cameras()
        self._camera_panel._camera_combo.blockSignals(True)
        self._camera_panel._camera_combo.clear()
        for cam in cameras:
            self._camera_panel._camera_combo.addItem(
                f"[{cam['index']}] {cam['name']}  ({cam['resolution'][0]}x{cam['resolution'][1]})",
                cam["index"],
            )
        self._camera_panel._camera_combo.blockSignals(False)

        # Populate supported native resolutions
        resolutions = self._service.get_supported_resolutions()
        self._camera_panel.populate_resolutions(resolutions)
        self._camera_panel.set_current_resolution(self._config.camera.width, self._config.camera.height)
        self._camera_panel.set_current_fps(self._config.camera.fps)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_start_stop(self) -> None:
        if self._start_btn.isChecked():
            self._start_btn.setText("■  Stop Pipeline")
            self._video_widget.set_placeholder("Starting pipeline...\nStreaming camera")
            self._service.start()
        else:
            self._start_btn.setText("▶  Start Pipeline")
            self._service.stop()
            self._video_widget.set_placeholder("Pipeline Stopped\nClick '▶  Start Pipeline' to begin")

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Display a BGR frame on the hardware-accelerated video widget."""
        try:
            self._video_widget.set_frame(frame)
        finally:
            self._service.notify_ui_rendered()

    @Slot(dict)
    def _on_metrics_updated(self, metrics: dict) -> None:
        fps = metrics.get("fps", 0.0)
        latency = metrics.get("latency", 0.0)
        infer = metrics.get("inference_time", 0.0)
        faces = metrics.get("faces", 0)
        device = metrics.get("device", "—")
        mode = metrics.get("mode", "REALTIME")

        badge = "[<20ms OK]" if latency < 20.0 else "[OPTIMIZING]"
        self._status_label.setText(
            f"Mode: {mode} | Latency: {latency:4.1f}ms {badge} | "
            f"FPS: {fps:4.1f} | AI: {infer:4.1f}ms | Faces: {faces} | Device: {device}"
        )
        self._performance_panel.update_metrics(metrics)

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        logger.info(f"Status -> {status}")
        self._status_bar.showMessage(status, 4000)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        logger.error(f"Error: {message}")
        QMessageBox.warning(self, "PrivacyCam", message)

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        """Graceful shutdown on window close."""
        self._service.cleanup()
        event.accept()
