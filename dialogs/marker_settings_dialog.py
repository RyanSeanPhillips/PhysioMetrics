"""
Event Marker Settings Dialog.

Allows users to configure default display settings for event markers
including line widths and fill opacity.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QSlider, QPushButton, QGroupBox,
    QDialogButtonBox, QCheckBox,
)
from PyQt6.QtCore import Qt


class MarkerSettingsDialog(QDialog):
    """Dialog for configuring event marker display settings."""

    def __init__(self, parent=None, single_width=0, paired_width=0, fill_alpha=30):
        super().__init__(parent)
        self.setWindowTitle("Event Marker Settings")
        self.setFixedWidth(380)

        self._single_width = single_width
        self._paired_width = paired_width
        self._fill_alpha = fill_alpha

        self._init_ui()
        self._apply_dark_theme()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Line Width Settings
        width_group = QGroupBox("Line Width (px)")
        width_layout = QGridLayout(width_group)

        width_layout.addWidget(QLabel("Single markers:"), 0, 0)
        self._spin_single = QSpinBox()
        self._spin_single.setRange(0, 6)
        self._spin_single.setValue(self._single_width)
        self._spin_single.setToolTip("Line width for single (point) markers\n0 = cosmetic (always 1px, thinnest possible)")
        self._spin_single.setSpecialValueText("0 (thinnest)")
        width_layout.addWidget(self._spin_single, 0, 1)

        width_layout.addWidget(QLabel("Paired marker edges:"), 1, 0)
        self._spin_paired = QSpinBox()
        self._spin_paired.setRange(0, 6)
        self._spin_paired.setValue(self._paired_width)
        self._spin_paired.setToolTip("Edge line width for paired (region) markers\n0 = cosmetic (always 1px, thinnest possible)")
        self._spin_paired.setSpecialValueText("0 (thinnest)")
        width_layout.addWidget(self._spin_paired, 1, 1)

        layout.addWidget(width_group)

        # Fill Opacity
        fill_group = QGroupBox("Region Fill Opacity")
        fill_layout = QHBoxLayout(fill_group)

        self._slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self._slider_alpha.setRange(0, 80)
        self._slider_alpha.setValue(self._fill_alpha)
        self._slider_alpha.setToolTip("Fill opacity for paired marker regions (0 = transparent, 80 = opaque)")

        self._lbl_alpha = QLabel(f"{self._fill_alpha}")
        self._lbl_alpha.setFixedWidth(25)
        self._slider_alpha.valueChanged.connect(lambda v: self._lbl_alpha.setText(str(v)))

        fill_layout.addWidget(self._slider_alpha)
        fill_layout.addWidget(self._lbl_alpha)

        layout.addWidget(fill_group)

        # Apply to all existing markers checkbox
        self._chk_apply_all = QCheckBox("Apply to all existing markers (reset per-marker overrides)")
        self._chk_apply_all.setToolTip(
            "When checked, all existing markers will use these defaults.\n"
            "Clears any per-marker line width overrides set via right-click menu."
        )
        layout.addWidget(self._chk_apply_all)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @property
    def single_width(self) -> int:
        return self._spin_single.value()

    @property
    def paired_width(self) -> int:
        return self._spin_paired.value()

    @property
    def fill_alpha(self) -> int:
        return self._slider_alpha.value()

    @property
    def apply_to_all(self) -> bool:
        return self._chk_apply_all.isChecked()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox {
                font-weight: bold; border: 1px solid #3a3a3a;
                border-radius: 4px; margin-top: 8px; padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #88aaff;
            }
            QLabel { color: #e0e0e0; background: transparent; }
            QSpinBox {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px;
            }
            QSlider::groove:horizontal {
                height: 6px; background: #3a3a3a; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -4px 0;
                background: #2a7fff; border-radius: 7px;
            }
            QPushButton {
                background-color: #3a3a3a; color: #e0e0e0;
                border: 1px solid #555; border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #2a7fff; }
            QCheckBox { color: #e0e0e0; background: transparent; spacing: 6px; }
        """)
