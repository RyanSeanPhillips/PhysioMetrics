"""
Export Options Dialog

Allows users to select which metrics to include in data exports.
Metrics are organized into logical groups for easy navigation.
"""

import sys

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QCheckBox,
    QPushButton, QScrollArea, QWidget, QLabel, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

from dialogs.export_mixin import ExportMixin


class ExportOptionsDialog(ExportMixin, QDialog):
    """Dialog for selecting which metrics to export."""

    # Signal emitted when user saves changes
    options_changed = pyqtSignal(dict)

    # Metric groups with descriptions
    # NOTE: These match the Y2 plot dropdown metrics from core.metrics.METRIC_SPECS
    METRIC_GROUPS = {
        'Basic Breathing Metrics': {
            'description': 'Fundamental breathing measurements',
            'metrics': ['if', 'amp_insp', 'amp_exp', 'area_insp', 'area_exp',
                       'ti', 'te', 'vent_proxy', 'regularity']
        },
        'Region Detection & Probabilities': {
            'description': 'Automated breath pattern detection and model probabilities',
            'metrics': ['eupnic', 'apnea', 'sniff_conf', 'eupnea_conf', 'p_noise', 'p_breath']
        },
        'Derivative Metrics': {
            'description': 'Rate of change measurements',
            'metrics': ['d1', 'd2', 'max_dinsp', 'max_dexp']
        },
        'Peak Candidate Metrics': {
            'description': 'ML features for merge detection and noise classification',
            'metrics': ['gap_to_next_norm', 'trough_ratio_next', 'onset_height_ratio',
                       'prom_asymmetry', 'amplitude_normalized']
        },
        'Width Features': {
            'description': 'Peak width measurements at different heights',
            'metrics': ['fwhm', 'width_25', 'width_75', 'width_ratio']
        },
        'Sigh Detection Features': {
            'description': 'Advanced features for identifying sighs',
            'metrics': ['n_inflections', 'rise_variability', 'n_shoulder_peaks',
                       'shoulder_prominence', 'rise_autocorr', 'peak_sharpness',
                       'trough_sharpness', 'skewness', 'kurtosis']
        },
        'Shape & Ratio Metrics': {
            'description': 'Breath shape and timing ratios',
            'metrics': ['peak_to_trough', 'amp_ratio', 'ti_te_ratio',
                       'area_ratio', 'total_area', 'ibi']
        },
        'Normalized Metrics': {
            'description': 'Z-scored relative measurements',
            'metrics': ['amp_insp_norm', 'amp_exp_norm', 'peak_to_trough_norm',
                       'prominence_norm', 'ibi_norm', 'ti_norm', 'te_norm']
        }
    }

    # Default export settings (what to include by default)
    # Start with commonly used metrics, user can enable advanced features as needed
    DEFAULT_ENABLED = {
        'Basic Breathing Metrics',
        'Region Detection & Probabilities'
    }

    def __init__(self, parent=None, current_options=None):
        """
        Initialize the export options dialog.

        Args:
            parent: Parent widget
            current_options: Dict of currently enabled metrics (None = use defaults)
        """
        super().__init__(parent)
        self.setWindowTitle("Export Options - Select Metrics")
        self.resize(700, 600)

        # Store checkboxes for each metric
        self.group_checkboxes = {}  # group_name -> QCheckBox (select all for group)
        self.metric_checkboxes = {}  # metric_name -> QCheckBox

        # Load current options or use defaults
        if current_options is None:
            self.current_options = self._get_default_options()
        else:
            self.current_options = current_options.copy()

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()
        self._load_current_settings()
        self.setup_export_menu()

    def _get_default_options(self):
        """Get default enabled metrics based on DEFAULT_ENABLED groups."""
        enabled = set()
        for group_name, group_data in self.METRIC_GROUPS.items():
            if group_name in self.DEFAULT_ENABLED:
                enabled.update(group_data['metrics'])
        return {metric: True for metric in enabled}

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Header label
        header = QLabel(
            "Select which metrics to include in ML training exports.\n"
            "These are the same metrics available in the Y2 plot dropdown."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Scrollable area for metric groups
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Create a group box for each metric category
        for group_name, group_data in self.METRIC_GROUPS.items():
            group_box = self._create_group_box(group_name, group_data)
            scroll_layout.addWidget(group_box)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Buttons at bottom
        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_none)
        button_layout.addWidget(select_none_btn)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _create_group_box(self, group_name, group_data):
        """Create a group box for a metric category."""
        group_box = QGroupBox(group_name)
        layout = QVBoxLayout()

        # Description label
        desc = QLabel(group_data['description'])
        desc.setStyleSheet("color: #888; font-style: italic;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # "Select All" checkbox for this group
        select_all_cb = QCheckBox(f"Select all in {group_name}")
        select_all_cb.setStyleSheet("font-weight: bold;")
        select_all_cb.stateChanged.connect(
            lambda state, gn=group_name: self._on_group_select_all(gn, state)
        )
        self.group_checkboxes[group_name] = select_all_cb
        layout.addWidget(select_all_cb)

        # Individual metric checkboxes in 3-column grid
        grid = QGridLayout()
        grid.setSpacing(5)
        metrics = group_data['metrics']
        for i, metric in enumerate(metrics):
            row = i // 3
            col = i % 3
            cb = QCheckBox(metric)
            cb.stateChanged.connect(self._on_metric_changed)
            self.metric_checkboxes[metric] = cb
            grid.addWidget(cb, row, col)
        layout.addLayout(grid)

        group_box.setLayout(layout)
        return group_box

    def _load_current_settings(self):
        """Load current settings into checkboxes."""
        for metric, checkbox in self.metric_checkboxes.items():
            is_enabled = self.current_options.get(metric, False)
            checkbox.setChecked(is_enabled)

        # Update group "select all" checkboxes
        self._update_group_checkboxes()

    def _update_group_checkboxes(self):
        """Update the state of group 'select all' checkboxes."""
        for group_name, group_data in self.METRIC_GROUPS.items():
            group_cb = self.group_checkboxes.get(group_name)
            if group_cb is None:
                continue

            # Check if all metrics in this group are enabled
            all_checked = all(
                self.metric_checkboxes[m].isChecked()
                for m in group_data['metrics']
                if m in self.metric_checkboxes
            )

            # Block signals to avoid recursion
            group_cb.blockSignals(True)
            group_cb.setChecked(all_checked)
            group_cb.blockSignals(False)

    def _on_group_select_all(self, group_name, state):
        """Handle group 'select all' checkbox."""
        is_checked = (state == Qt.CheckState.Checked.value)
        group_data = self.METRIC_GROUPS[group_name]

        for metric in group_data['metrics']:
            cb = self.metric_checkboxes.get(metric)
            if cb:
                cb.setChecked(is_checked)

    def _on_metric_changed(self):
        """Handle individual metric checkbox change."""
        self._update_group_checkboxes()

    def _select_all(self):
        """Select all metrics."""
        for cb in self.metric_checkboxes.values():
            cb.setChecked(True)

    def _select_none(self):
        """Deselect all metrics."""
        for cb in self.metric_checkboxes.values():
            cb.setChecked(False)

    def _reset_to_defaults(self):
        """Reset to default metric selection."""
        defaults = self._get_default_options()
        for metric, cb in self.metric_checkboxes.items():
            cb.setChecked(defaults.get(metric, False))

    def _save_and_close(self):
        """Save selections and close dialog."""
        # Collect enabled metrics
        enabled_metrics = {}
        for metric, cb in self.metric_checkboxes.items():
            enabled_metrics[metric] = cb.isChecked()

        # Emit signal with new options
        self.options_changed.emit(enabled_metrics)
        self.accept()

    def get_enabled_metrics(self):
        """
        Get dictionary of enabled metrics.

        Returns:
            dict: {metric_name: bool} for all metrics
        """
        return {
            metric: cb.isChecked()
            for metric, cb in self.metric_checkboxes.items()
        }

    def _apply_dark_theme(self):
        """Apply dark theme styling to the dialog."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2a7fff;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QCheckBox::indicator:hover {
                border-color: #2a7fff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2a7fff;
            }
            QPushButton:pressed {
                background-color: #2a7fff;
            }
            QPushButton:default {
                border-color: #2a7fff;
            }
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #555;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int

                # DWMWA_USE_IMMERSIVE_DARK_MODE
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20

                # Get window handle
                hwnd = int(self.winId())

                # Set dark mode (1 = dark, 0 = light)
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    byref(value),
                    sizeof(value)
                )
            except Exception:
                # Silently fail if not supported (Windows 10 older builds)
                pass
