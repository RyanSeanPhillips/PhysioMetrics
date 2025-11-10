"""
Outlier Metrics Selection Dialog for PhysioMetrics.

This dialog allows users to select which breath metrics should be used for
outlier detection during breathing analysis.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt


class OutlierMetricsDialog(QDialog):
    def __init__(self, parent=None, available_metrics=None, selected_metrics=None):
        from PyQt6.QtWidgets import (QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
                                    QPushButton, QScrollArea, QWidget)
        from PyQt6.QtCore import Qt

        super().__init__(parent)
        self.setWindowTitle("Select Outlier Detection Metrics")
        self.resize(500, 600)

        # Store available metrics
        self.available_metrics = available_metrics or []
        self.selected_metrics = set(selected_metrics or [])

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

        # Main layout
        main_layout = QVBoxLayout(self)

        # Title label
        title = QLabel("Select which metrics to use for outlier detection:")
        title.setStyleSheet("font-size: 12pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Info label
        info = QLabel("Breaths with values beyond ±N standard deviations (set in SD field) "
                     "for ANY selected metric will be flagged as outliers.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #B0B0B0; margin-bottom: 15px;")
        main_layout.addWidget(info)

        # Scroll area for checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Create checkboxes
        self.checkboxes = {}
        for metric in self.available_metrics:
            checkbox = QCheckBox(metric)
            checkbox.setChecked(metric in self.selected_metrics)

            # Add description if available
            if metric in self.metric_descriptions:
                checkbox.setText(f"{metric} - {self.metric_descriptions[metric]}")

            checkbox.setStyleSheet("margin: 5px 0px;")
            self.checkboxes[metric] = checkbox
            scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
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
