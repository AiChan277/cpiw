from PySide6.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QComboBox, QSlider, QLabel, 
    QPushButton, QHBoxLayout, QColorDialog
)
from PySide6.QtCore import Qt, Signal, Slot

class PrivacyPanel(QWidget):
    mode_changed = Signal(str)
    strength_changed = Signal(int)
    blocks_changed = Signal(int)
    color_changed = Signal(tuple)
    expansion_changed = Signal(float)
    feather_changed = Signal(int)
    confidence_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group_box = QGroupBox("Privacy", self)
        gb_layout = QVBoxLayout(group_box)
        
        self._mode_combo = QComboBox(group_box)
        self._mode_combo.addItems(["Blur", "Pixelate", "Solid"])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        gb_layout.addWidget(QLabel("Mode:"))
        gb_layout.addWidget(self._mode_combo)
        
        self._controls_widget = QWidget()
        self._controls_layout = QVBoxLayout(self._controls_widget)
        self._controls_layout.setContentsMargins(0, 0, 0, 0)
        gb_layout.addWidget(self._controls_widget)
        
        self._blur_widget = QWidget()
        bl = QHBoxLayout(self._blur_widget)
        bl.addWidget(QLabel("Blur:"))
        self._blur_slider = QSlider(Qt.Orientation.Horizontal)
        self._blur_slider.setRange(5, 101)
        self._blur_slider.setSingleStep(2)
        self._blur_slider.setValue(31)
        self._blur_slider.valueChanged.connect(self._on_blur_changed)
        self._blur_label = QLabel("31")
        bl.addWidget(self._blur_slider)
        bl.addWidget(self._blur_label)
        self._controls_layout.addWidget(self._blur_widget)
        
        self._pixel_widget = QWidget()
        pl = QHBoxLayout(self._pixel_widget)
        pl.addWidget(QLabel("Blocks:"))
        self._pixel_slider = QSlider(Qt.Orientation.Horizontal)
        self._pixel_slider.setRange(3, 30)
        self._pixel_slider.setValue(10)
        self._pixel_slider.valueChanged.connect(self._on_pixel_changed)
        self._pixel_label = QLabel("10")
        pl.addWidget(self._pixel_slider)
        pl.addWidget(self._pixel_label)
        self._controls_layout.addWidget(self._pixel_widget)
        
        self._color_widget = QWidget()
        cl = QHBoxLayout(self._color_widget)
        cl.addWidget(QLabel("Color:"))
        self._color_btn = QPushButton("Select Color")
        self._color_btn.clicked.connect(self._on_color_btn_clicked)
        cl.addWidget(self._color_btn)
        self._controls_layout.addWidget(self._color_widget)
        
        el = QHBoxLayout()
        el.addWidget(QLabel("Expansion:"))
        self._expansion_slider = QSlider(Qt.Orientation.Horizontal)
        self._expansion_slider.setRange(0, 50)
        self._expansion_slider.setValue(15)
        self._expansion_slider.valueChanged.connect(self._on_expansion_changed)
        self._expansion_label = QLabel("0.15")
        el.addWidget(self._expansion_slider)
        el.addWidget(self._expansion_label)
        gb_layout.addLayout(el)
        
        fl = QHBoxLayout()
        fl.addWidget(QLabel("Feather:"))
        self._feather_slider = QSlider(Qt.Orientation.Horizontal)
        self._feather_slider.setRange(0, 50)
        self._feather_slider.setValue(15)
        self._feather_slider.valueChanged.connect(self._on_feather_changed)
        self._feather_label = QLabel("15")
        fl.addWidget(self._feather_slider)
        fl.addWidget(self._feather_label)
        gb_layout.addLayout(fl)
        
        confl = QHBoxLayout()
        confl.addWidget(QLabel("Confidence:"))
        self._conf_slider = QSlider(Qt.Orientation.Horizontal)
        self._conf_slider.setRange(10, 100)
        self._conf_slider.setValue(50)
        self._conf_slider.valueChanged.connect(self._on_conf_changed)
        self._conf_label = QLabel("0.5")
        confl.addWidget(self._conf_slider)
        confl.addWidget(self._conf_label)
        gb_layout.addLayout(confl)
        
        layout.addWidget(group_box)
        self._update_controls_visibility()

    @Slot(str)
    def _on_mode_changed(self, mode: str):
        self._update_controls_visibility()
        self.mode_changed.emit(mode.lower())

    def _update_controls_visibility(self):
        mode = self._mode_combo.currentText()
        self._blur_widget.setVisible(mode == "Blur")
        self._pixel_widget.setVisible(mode == "Pixelate")
        self._color_widget.setVisible(mode == "Solid")

    @Slot(int)
    def _on_blur_changed(self, value: int):
        if value % 2 == 0:
            value += 1
            self._blur_slider.setValue(value)
        self._blur_label.setText(str(value))
        self.strength_changed.emit(value)

    @Slot(int)
    def _on_pixel_changed(self, value: int):
        self._pixel_label.setText(str(value))
        self.blocks_changed.emit(value)

    @Slot()
    def _on_color_btn_clicked(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.color_changed.emit((r, g, b))

    @Slot(int)
    def _on_expansion_changed(self, value: int):
        val_f = value / 100.0
        self._expansion_label.setText(f"{val_f:.2f}")
        self.expansion_changed.emit(val_f)

    @Slot(int)
    def _on_feather_changed(self, value: int):
        self._feather_label.setText(str(value))
        self.feather_changed.emit(value)

    @Slot(int)
    def _on_conf_changed(self, value: int):
        val_f = value / 100.0
        self._conf_label.setText(f"{val_f:.2f}")
        self.confidence_changed.emit(val_f)
