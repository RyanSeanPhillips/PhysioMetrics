"""
Outlier Metrics Selection Dialog for PhysioMetrics.

This dialog allows users to select which breath metrics should be used for
outlier detection during breathing analysis.
"""

import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QCheckBox,
    QPushButton, QScrollArea, QWidget, QLineEdit
)
from PyQt6.QtCore import Qt

from dialogs.export_mixin import ExportMixin


class OutlierMetricsDialog(ExportMixin, QDialog):
    def __init__(self, parent=None, available_metrics=None, selected_metrics=None, outlier_sd=3.0):
        from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QCheckBox,
                                    QPushButton, QScrollArea, QWidget)
        from PyQt6.QtCore import Qt

        super().__init__(parent)
        self.setWindowTitle("Select Outlier Detection Metrics")
        self.resize(1000, 450)  # Wider for 3-column layout

        # Apply dark theme
        self._apply_dark_theme()
        self._enable_dark_title_bar()
        self.setup_export_menu()

        # Store available metrics
        self.available_metrics = available_metrics or []
        self.selected_metrics = set(selected_metrics or [])
        self.outlier_sd = outlier_sd

        # Metric descriptions
        self.metric_descriptions = {
            "if": "Instantaneous Frequency (Hz) - breath rate",
            "amp_insp": "Inspiratory Amplitude - peak height",
            "amp_exp": "Expiratory Amplitude - trough depth",
            "ti": "Inspiratory Time (s) - inhalation duration",
            "te": "Expiratory Time (s) - exhalation duration",
            "area_insp": "Inspiratory Area - integral during inhalation",
            "area_exp": "Expiratory Area - integral during exhalation",
            "vent_proxy": "Ventilation Proxy - breathing effort estimate",
            "d1": "D1 - duty cycle (Ti/Ttot)",
            "d2": "D2 - expiratory fraction (Te/Ttot)"
        }

        self._setup_ui()

    def _apply_dark_theme(self):
        """Apply dark theme styling to match main application."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
            }
            QCheckBox {
                color: #d4d4d4;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #505050;
            }
            QCheckBox::indicator:checked {
                background-color: #0e639c;
                border: 1px solid #1177bb;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px 15px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border: 1px solid #505050;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QScrollArea {
                border: 1px solid #3e3e42;
                border-radius: 4px;
                background-color: #1e1e1e;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1e1e1e;
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass

    def _setup_ui(self):
        """Build the dialog UI."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Title label
        title = QLabel("Select which metrics to use for outlier detection:")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Info label
        info = QLabel("Breaths with values beyond ±N standard deviations (set below) "
                     "for ANY selected metric will be flagged as outliers.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #B0B0B0; margin-bottom: 10px;")
        main_layout.addWidget(info)

        # Outlier SD threshold input
        sd_layout = QHBoxLayout()
        sd_label = QLabel("Outlier Threshold (SD):")
        sd_label.setStyleSheet("font-weight: bold;")
        sd_layout.addWidget(sd_label)

        self.sd_input = QLineEdit()
        self.sd_input.setText(str(self.outlier_sd))
        self.sd_input.setFixedWidth(100)
        self.sd_input.setPlaceholderText("3.0")
        self.sd_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
            }
            QLineEdit:focus {
                border: 1px solid #0e639c;
            }
        """)
        sd_layout.addWidget(self.sd_input)

        sd_help = QLabel("(Number of standard deviations from mean)")
        sd_help.setStyleSheet("color: #B0B0B0; font-style: italic;")
        sd_layout.addWidget(sd_help)
        sd_layout.addStretch()

        main_layout.addLayout(sd_layout)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()

        # Use grid layout for multi-column display (2 columns)
        scroll_layout = QGridLayout(scroll_content)
        scroll_layout.setSpacing(10)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        # Create checkboxes in 3-column grid
        self.checkboxes = {}
        num_columns = 3
        for idx, metric in enumerate(self.available_metrics):
            checkbox = QCheckBox(metric)
            checkbox.setChecked(metric in self.selected_metrics)

            # Add description if available
            if metric in self.metric_descriptions:
                checkbox.setText(f"{metric} - {self.metric_descriptions[metric]}")

            self.checkboxes[metric] = checkbox

            # Calculate row and column
            row = idx // num_columns
            col = idx % num_columns
            scroll_layout.addWidget(checkbox, row, col)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # Quick selection buttons
        quick_buttons = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all)
        quick_buttons.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self.deselect_all)
        quick_buttons.addWidget(deselect_all_btn)

        default_btn = QPushButton("Reset to Default")
        default_btn.clicked.connect(self.reset_to_default)
        quick_buttons.addWidget(default_btn)

        main_layout.addLayout(quick_buttons)

        # Dialog buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)

        main_layout.addLayout(button_layout)

    def select_all(self):
        """Check all metric checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all(self):
        """Uncheck all metric checkboxes."""
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def reset_to_default(self):
        """Reset to default metric selection."""
        default_metrics = ["if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"]
        for metric, checkbox in self.checkboxes.items():
            checkbox.setChecked(metric in default_metrics)

    def get_selected_metrics(self):
        """Return list of selected metric keys."""
        return [metric for metric, checkbox in self.checkboxes.items() if checkbox.isChecked()]

    def get_outlier_sd(self):
        """Return the outlier SD threshold value."""
        try:
            return float(self.sd_input.text())
        except (ValueError, AttributeError):
            return 3.0  # Default value
