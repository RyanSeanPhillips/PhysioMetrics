"""
Advanced Peak Editor Dialog for PhysioMetrics

Provides efficient review of edge cases and model disagreements:
- Tab 1: Breath vs Noise edge case review (using P_edge, ML disagreements, ML probabilities)
- Tab 2: Eupnea vs Sniff classification (future)
- Tab 3: Sigh vs Normal breath detection (future)

Features embedded matplotlib plotting with light/dark theme support.
"""

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QRadioButton, QButtonGroup, QCheckBox,
    QWidget, QMessageBox, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QBrush

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import theme manager
from core.plot_themes import PlotThemeManager


class AdvancedPeakEditorDialog(QDialog):
    """Dialog for reviewing edge cases in peak classification."""

    def __init__(self, main_window, parent=None):
        """
        Initialize the Advanced Peak Editor dialog.

        Args:
            main_window: Reference to MainWindow instance
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.main_window = main_window

        # Edge case data
        self.edge_cases = []  # List of edge case dictionaries
        self.current_edge_idx = -1  # Index in edge_cases list
        self.reviewed_peaks = set()  # Set of (sweep, peak_idx) tuples that were reviewed

        self.setWindowTitle("Advanced Peak Editor")
        self.resize(1200, 800)

        # Apply dark theme to entire dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                color: #cccccc;
            }
            QPushButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #6a6a6a;
                border-color: #2d2d30;
            }
            QRadioButton {
                color: #cccccc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #666;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
            QRadioButton::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QRadioButton::indicator:hover {
                border-color: #2a7fff;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #666;
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
            QComboBox {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 3px 6px;
                border-radius: 3px;
            }
            QComboBox:hover {
                background-color: #2d2d30;
            }
            QComboBox::drop-down {
                background: #2d2d30;
                border-left: 1px solid #3e3e42;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background: #252526;
                color: #cccccc;
                selection-background-color: #094771;
                selection-color: #ffffff;
                border: 1px solid #3e3e42;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d30;
                color: #cccccc;
                padding: 8px 16px;
                border: 1px solid #3e3e42;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background: #3e3e42;
            }
        """)

        self._init_ui()
        self._connect_signals()
        self._enable_dark_title_bar()

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

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Tab widget for different review types
        self.tab_widget = QTabWidget()

        # Tab 1: Breath/Noise
        self.breath_noise_tab = self._create_breath_noise_tab()
        self.tab_widget.addTab(self.breath_noise_tab, "Breath/Noise")

        # Tab 2: Eupnea/Sniff (placeholder for now)
        self.eupnea_sniff_tab = QWidget()
        eupnea_layout = QVBoxLayout()
        eupnea_layout.addWidget(QLabel("Eupnea/Sniff review - Coming soon"))
        self.eupnea_sniff_tab.setLayout(eupnea_layout)
        self.tab_widget.addTab(self.eupnea_sniff_tab, "Eupnea/Sniff")

        # Tab 3: Sigh/Normal (placeholder for now)
        self.sigh_normal_tab = QWidget()
        sigh_layout = QVBoxLayout()
        sigh_layout.addWidget(QLabel("Sigh/Normal review - Coming soon"))
        self.sigh_normal_tab.setLayout(sigh_layout)
        self.tab_widget.addTab(self.sigh_normal_tab, "Sigh/Normal")

        main_layout.addWidget(self.tab_widget)

        self.setLayout(main_layout)

    def _create_breath_noise_tab(self) -> QWidget:
        """Create the Breath/Noise review tab (Tab 1)."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Main splitter (left controls, right plot)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (15% width)
        left_panel = self._create_left_panel_tab1()
        splitter.addWidget(left_panel)

        # Right panel (85% width)
        right_panel = self._create_plot_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (15% left, 85% right)
        splitter.setSizes([150, 850])

        layout.addWidget(splitter)

        # Bottom action bar
        action_bar = self._create_action_bar_tab1()
        layout.addWidget(action_bar)

        tab.setLayout(layout)
        return tab

    def _create_left_panel_tab1(self) -> QWidget:
        """Create the left control panel for Tab 1."""
        widget = QWidget()
        layout = QVBoxLayout()

        # === Edge Case Type ===
        edge_group = QGroupBox("Edge Case Type")
        edge_layout = QVBoxLayout()

        self.edge_type_group = QButtonGroup()

        self.edge_p_edge_radio = QRadioButton("Threshold P_edge")
        self.edge_p_edge_radio.setChecked(True)
        self.edge_type_group.addButton(self.edge_p_edge_radio, 0)
        edge_layout.addWidget(self.edge_p_edge_radio)

        self.edge_ml_disagree_radio = QRadioButton("ML Disagreement")
        self.edge_type_group.addButton(self.edge_ml_disagree_radio, 1)
        edge_layout.addWidget(self.edge_ml_disagree_radio)

        # Model comparison dropdowns for disagreement detection
        ml_compare_layout = QHBoxLayout()
        ml_compare_layout.addWidget(QLabel("  Compare:"))
        self.ml_model_a_combo = QComboBox()
        self.ml_model_a_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])
        ml_compare_layout.addWidget(self.ml_model_a_combo)
        ml_compare_layout.addWidget(QLabel("vs"))
        self.ml_model_b_combo = QComboBox()
        self.ml_model_b_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])
        self.ml_model_b_combo.setCurrentIndex(1)  # Default to XGBoost
        ml_compare_layout.addWidget(self.ml_model_b_combo)
        ml_compare_layout.addStretch()
        edge_layout.addLayout(ml_compare_layout)

        self.edge_ml_prob_radio = QRadioButton("ML Probability")
        self.edge_type_group.addButton(self.edge_ml_prob_radio, 2)
        edge_layout.addWidget(self.edge_ml_prob_radio)

        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)

        # === Sort Order ===
        sort_group = QGroupBox("Sort")
        sort_layout = QVBoxLayout()

        self.sort_order_group = QButtonGroup()

        self.sort_asc_radio = QRadioButton("Ascending")
        self.sort_order_group.addButton(self.sort_asc_radio, 0)
        sort_layout.addWidget(self.sort_asc_radio)

        self.sort_desc_radio = QRadioButton("Descending")
        self.sort_desc_radio.setChecked(True)
        self.sort_order_group.addButton(self.sort_desc_radio, 1)
        sort_layout.addWidget(self.sort_desc_radio)

        sort_group.setLayout(sort_layout)
        layout.addWidget(sort_group)

        # === Display Options ===
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()

        self.show_events_checkbox = QCheckBox("Show Events")
        self.show_events_checkbox.setChecked(True)
        display_layout.addWidget(self.show_events_checkbox)

        self.show_peaks_checkbox = QCheckBox("Show All Peaks")
        self.show_peaks_checkbox.setChecked(True)
        display_layout.addWidget(self.show_peaks_checkbox)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # === Plot Controls ===
        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout()

        plot_layout.addWidget(QLabel("Window:"))
        self.window_size_combo = QComboBox()
        self.window_size_combo.addItems(["±0.5s", "±1s", "±2s", "±5s", "±10s"])
        self.window_size_combo.setCurrentText("±2s")
        plot_layout.addWidget(self.window_size_combo)

        plot_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        plot_layout.addWidget(self.theme_combo)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # === Detect Button ===
        self.detect_btn = QPushButton("Detect Edge Cases")
        self.detect_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.detect_btn)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _create_plot_panel(self) -> QWidget:
        """Create the plot panel with matplotlib canvas."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Matplotlib canvas
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Initialize theme manager and apply default theme
        self.theme_manager = PlotThemeManager(default_theme='dark')
        self.theme_manager.apply_theme(self.ax, self.fig, 'dark')
        self.canvas.draw()

        # Info label below plot
        self.info_label = QLabel("No edge cases loaded. Click 'Detect Edge Cases' to begin.")
        self.info_label.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        widget.setLayout(layout)
        return widget

    def _create_action_bar_tab1(self) -> QGroupBox:
        """Create the bottom action bar for Tab 1."""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()

        # Navigation row
        nav_layout = QHBoxLayout()

        self.first_btn = QPushButton("◀◀ First")
        self.first_btn.setEnabled(False)
        nav_layout.addWidget(self.first_btn)

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        self.last_btn = QPushButton("Last ▶▶")
        self.last_btn.setEnabled(False)
        nav_layout.addWidget(self.last_btn)

        nav_layout.addStretch()

        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        nav_layout.addWidget(self.progress_label)

        layout.addLayout(nav_layout)

        # Classification row
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Classification:"))

        self.mark_breath_btn = QPushButton("Mark as BREATH")
        self.mark_breath_btn.setEnabled(False)
        self.mark_breath_btn.setStyleSheet("background-color: #2d5016; padding: 8px;")
        class_layout.addWidget(self.mark_breath_btn)

        self.mark_noise_btn = QPushButton("Mark as NOISE")
        self.mark_noise_btn.setEnabled(False)
        self.mark_noise_btn.setStyleSheet("background-color: #5a1a1a; padding: 8px;")
        class_layout.addWidget(self.mark_noise_btn)

        self.skip_btn = QPushButton("Skip/Uncertain")
        self.skip_btn.setEnabled(False)
        self.skip_btn.setStyleSheet("padding: 8px;")
        class_layout.addWidget(self.skip_btn)

        class_layout.addStretch()

        layout.addLayout(class_layout)

        # Utility row
        util_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Edge Cases CSV")
        self.export_btn.setEnabled(False)
        util_layout.addWidget(self.export_btn)

        self.jump_main_btn = QPushButton("Jump to Main Window")
        self.jump_main_btn.setEnabled(False)
        util_layout.addWidget(self.jump_main_btn)

        util_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        util_layout.addWidget(close_btn)

        layout.addLayout(util_layout)

        group.setLayout(layout)
        return group

    def _connect_signals(self):
        """Connect all signal handlers."""
        # Detect button
        self.detect_btn.clicked.connect(self._detect_edge_cases)

        # Navigation buttons
        self.first_btn.clicked.connect(self._navigate_first)
        self.prev_btn.clicked.connect(self._navigate_previous)
        self.next_btn.clicked.connect(self._navigate_next)
        self.last_btn.clicked.connect(self._navigate_last)

        # Classification buttons
        self.mark_breath_btn.clicked.connect(self._mark_as_breath)
        self.mark_noise_btn.clicked.connect(self._mark_as_noise)
        self.skip_btn.clicked.connect(self._skip_uncertain)

        # Jump to main
        self.jump_main_btn.clicked.connect(self._jump_to_main_window)

        # Export
        self.export_btn.clicked.connect(self._export_edge_cases)

        # Plot controls
        self.window_size_combo.currentTextChanged.connect(self._refresh_plot)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.show_events_checkbox.stateChanged.connect(self._refresh_plot)
        self.show_peaks_checkbox.stateChanged.connect(self._refresh_plot)

    def _detect_edge_cases(self):
        """Detect edge cases based on selected method."""
        print("\n===== Detecting Edge Cases =====")

        # Check if peaks have been detected
        if not hasattr(self.main_window.state, 'all_peaks_by_sweep') or not self.main_window.state.all_peaks_by_sweep:
            error_msg = "Please detect peaks first using 'Find Peaks & Events' button in the main window.\n\nThe Advanced Peak Editor needs detected peaks to find edge cases."
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("No Peaks Detected")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            print("[Edge Detection] No peaks detected yet")
            print("===== Detection Aborted =====\n")
            return

        # Get selected edge case type
        edge_type_id = self.edge_type_group.checkedId()

        if edge_type_id == 0:
            print("Method: Threshold P_edge")
            self._detect_p_edge_cases()
        elif edge_type_id == 1:
            print("Method: ML Disagreement")
            self._detect_ml_disagreements()
        elif edge_type_id == 2:
            print("Method: ML Probability")
            self._detect_ml_probability_edges()

        # Sort results
        self._sort_edge_cases()

        # Update UI
        self._update_ui_state()

        # Show first edge case
        if self.edge_cases:
            self.current_edge_idx = 0
            self._update_plot(0)
        else:
            # No edge cases found - show message in plot
            self._show_no_edge_cases_message()

        print(f"Total edge cases found: {len(self.edge_cases)}")
        print("===== Detection Complete =====\n")

    def _detect_p_edge_cases(self):
        """Detect edge cases using threshold P_edge metric."""
        self.edge_cases = []

        # Get sample rate
        sr_hz = self.main_window.state.sr_hz
        if sr_hz is None or sr_hz == 0:
            print("Error: Sample rate not available")
            return

        # Check if ANY sweep has P_edge data
        has_p_edge = any('P_edge' in peak_data for peak_data in self.main_window.state.all_peaks_by_sweep.values())

        if not has_p_edge:
            error_msg = ("P_edge data is not available in the current detection results.\n\n"
                        "P_edge is only computed by threshold-based detection.\n\n"
                        "Try using 'ML Disagreement' instead to find edge cases.")
            print(f"[P_edge Detection] {error_msg}")

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("P_edge Not Available")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        for sweep_idx, peak_data in self.main_window.state.all_peaks_by_sweep.items():
            if 'P_edge' not in peak_data:
                print(f"Skipping sweep {sweep_idx}: No P_edge data")
                continue

            p_edge = peak_data['P_edge']
            indices = peak_data['indices']  # Sample indices

            # Convert indices to times
            times = indices / sr_hz

            # Find peaks with high P_edge (close to decision boundary)
            # Lower threshold to 0.3 to catch more edge cases
            threshold = 0.3
            edge_mask = p_edge > threshold

            edge_indices = np.where(edge_mask)[0]
            print(f"Sweep {sweep_idx}: {len(edge_indices)} edge cases (P_edge > {threshold})")

            for peak_idx in edge_indices:
                self.edge_cases.append({
                    'sweep': int(sweep_idx),
                    'peak_idx': int(peak_idx),
                    'time': float(times[peak_idx]),
                    'sample_idx': int(indices[peak_idx]),
                    'p_edge': float(p_edge[peak_idx]),
                    'method': 'p_edge',
                    'current_label': int(peak_data['labels'][peak_idx]),
                })

    def _detect_ml_disagreements(self):
        """Detect edge cases where ML models disagree."""
        self.edge_cases = []

        # Get selected models
        model_a_name = self.ml_model_a_combo.currentText()
        model_b_name = self.ml_model_b_combo.currentText()

        if model_a_name == model_b_name:
            error_msg = "Please select two different models to compare."
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Same Model Selected")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        # Map names to array keys
        model_map = {
            "Threshold": "labels_threshold_ro",
            "XGBoost": "labels_xgboost_ro",
            "Random Forest": "labels_rf_ro",
            "MLP": "labels_mlp_ro"
        }

        key_a = model_map[model_a_name]
        key_b = model_map[model_b_name]

        print(f"Comparing {model_a_name} vs {model_b_name}")

        # Get sample rate
        sr_hz = self.main_window.state.sr_hz
        if sr_hz is None or sr_hz == 0:
            print("Error: Sample rate not available")
            return

        for sweep_idx, peak_data in self.main_window.state.all_peaks_by_sweep.items():
            # Check if predictions exist
            if key_a not in peak_data or key_b not in peak_data:
                print(f"Skipping sweep {sweep_idx}: Missing predictions for {model_a_name} or {model_b_name}")
                continue

            labels_a = peak_data[key_a]
            labels_b = peak_data[key_b]
            indices = peak_data['indices']  # Sample indices

            # Convert indices to times
            times = indices / sr_hz

            # Find disagreements
            disagree_mask = (labels_a != labels_b)
            disagree_indices = np.where(disagree_mask)[0]

            print(f"Sweep {sweep_idx}: {len(disagree_indices)} disagreements")

            for peak_idx in disagree_indices:
                self.edge_cases.append({
                    'sweep': int(sweep_idx),
                    'peak_idx': int(peak_idx),
                    'time': float(times[peak_idx]),
                    'sample_idx': int(indices[peak_idx]),
                    'model_a_name': model_a_name,
                    'model_b_name': model_b_name,
                    'model_a_pred': int(labels_a[peak_idx]),
                    'model_b_pred': int(labels_b[peak_idx]),
                    'method': 'ml_disagree',
                    'current_label': int(peak_data['labels'][peak_idx]),
                })

    def _detect_ml_probability_edges(self):
        """Detect edge cases using ML model probability scores."""
        # TODO: Implement when probability scores are available
        error_msg = ("ML Probability edge detection requires probability scores.\n\n"
                    "This feature will be implemented when models output probabilities.\n\n"
                    "Try using 'ML Disagreement' instead to find edge cases.")

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Not Implemented")
        msg_box.setText(error_msg)
        msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_box.exec()

        self.edge_cases = []

    def _sort_edge_cases(self):
        """Sort edge cases based on selected order."""
        if not self.edge_cases:
            return

        sort_desc = self.sort_desc_radio.isChecked()

        # Sort by appropriate metric
        if self.edge_cases[0]['method'] == 'p_edge':
            self.edge_cases.sort(key=lambda x: x['p_edge'], reverse=sort_desc)
        elif self.edge_cases[0]['method'] == 'ml_disagree':
            # For disagreements, just keep original order or sort by time
            self.edge_cases.sort(key=lambda x: x['time'], reverse=sort_desc)

    def _update_ui_state(self):
        """Update UI button states based on edge cases."""
        has_cases = len(self.edge_cases) > 0

        # Enable/disable navigation
        self.first_btn.setEnabled(has_cases)
        self.prev_btn.setEnabled(has_cases)
        self.next_btn.setEnabled(has_cases)
        self.last_btn.setEnabled(has_cases)

        # Enable/disable classification
        self.mark_breath_btn.setEnabled(has_cases)
        self.mark_noise_btn.setEnabled(has_cases)
        self.skip_btn.setEnabled(has_cases)

        # Enable/disable utilities
        self.export_btn.setEnabled(has_cases)
        self.jump_main_btn.setEnabled(has_cases)

        # Update progress
        reviewed = len(self.reviewed_peaks)
        total = len(self.edge_cases)
        self.progress_label.setText(f"Progress: {reviewed}/{total}")

    def _update_plot(self, edge_idx):
        """
        Update the plot to show the selected edge case.

        Args:
            edge_idx: Index in self.edge_cases list
        """
        if edge_idx < 0 or edge_idx >= len(self.edge_cases):
            return

        edge = self.edge_cases[edge_idx]
        sweep_idx = edge['sweep']
        peak_idx = edge['peak_idx']
        peak_time = edge['time']

        # Get data
        peak_data = self.main_window.state.all_peaks_by_sweep[sweep_idx]

        # Get sweep data using the main window's method
        # Switch to the correct sweep first
        old_sweep = self.main_window.state.sweep_idx
        self.main_window.state.sweep_idx = sweep_idx

        # Get processed trace data
        times, trace = self.main_window._current_trace()

        # Restore original sweep
        self.main_window.state.sweep_idx = old_sweep

        if times is None or trace is None:
            # Can't plot without sweep data
            error_msg = f"Cannot display plot: Sweep {sweep_idx} data not available.\n\nPlease ensure data is loaded and the analyze channel is selected."
            print(f"Error: {error_msg}")

            # Create a copyable error dialog
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("No Sweep Data")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        # Get window bounds - center on peak
        window_text = self.window_size_combo.currentText()
        half_window = float(window_text.replace('±', '').replace('s', ''))

        t_min = peak_time - half_window
        t_max = peak_time + half_window

        # Find indices within window
        mask = (times >= t_min) & (times <= t_max)
        window_times = times[mask]
        window_trace = trace[mask]

        # Clear and redraw
        self.ax.clear()

        # Apply theme
        theme_name = 'dark' if self.theme_combo.currentText() == 'Dark' else 'light'
        self.theme_manager.apply_theme(self.ax, self.fig, theme_name)

        # Plot trace
        trace_color = self.theme_manager.get_color('trace_color')
        self.ax.plot(window_times, window_trace,
                     color=trace_color,
                     linewidth=self.theme_manager.get_value('trace_linewidth'),
                     label='Respiratory Signal', zorder=1)

        # Plot disagreement peak (highlighted)
        # Use transparency if marked as noise by active classifier
        disagree_color = self.theme_manager.get_color('disagreement_peak_color')
        disagree_size = self.theme_manager.get_value('disagreement_peak_size')

        # Find y-value at peak time
        peak_time_idx = np.argmin(np.abs(times - peak_time))
        peak_y_value = trace[peak_time_idx]

        # Check if current peak is marked as breath or noise by active classifier
        current_peak_label = peak_data['labels'][peak_idx]  # 0=noise, 1=breath
        is_breath = current_peak_label == 1

        # Use transparency for noise peaks
        peak_alpha = 1.0 if is_breath else 0.4
        peak_label = 'Edge Case Peak (Breath)' if is_breath else 'Edge Case Peak (Noise)'

        self.ax.plot(peak_time, peak_y_value,
                     marker=self.theme_manager.get_value('disagreement_peak_marker'),
                     color=disagree_color,
                     markersize=disagree_size,
                     alpha=peak_alpha,
                     label=peak_label,
                     zorder=5)

        # Plot other peaks in window (if enabled)
        # Only show peaks that the ACTIVE classifier marks as breaths
        if self.show_peaks_checkbox.isChecked():
            # Convert peak indices to times
            sr_hz = self.main_window.state.sr_hz
            all_peak_indices = peak_data['indices']
            all_peak_times = all_peak_indices / sr_hz

            # Get active classifier labels (only show peaks marked as breath by active classifier)
            active_labels = peak_data['labels']  # This is the currently displayed labels

            # Filter to window AND breaths only
            peak_mask = (all_peak_times >= t_min) & (all_peak_times <= t_max) & (active_labels == 1)
            other_peak_times = all_peak_times[peak_mask]

            # Remove current peak (use small tolerance for float comparison)
            tolerance = 0.001  # 1ms tolerance
            other_peak_times = other_peak_times[np.abs(other_peak_times - peak_time) > tolerance]

            if len(other_peak_times) > 0:
                # Find y-values for other peaks
                other_peak_y = []
                for pt in other_peak_times:
                    idx = np.argmin(np.abs(times - pt))
                    other_peak_y.append(trace[idx])

                peak_color = self.theme_manager.get_color('peak_color')
                self.ax.plot(other_peak_times, other_peak_y,
                             marker=self.theme_manager.get_value('peak_marker'),
                             color=peak_color,
                             linestyle='',
                             markersize=self.theme_manager.get_value('peak_size'),
                             label='Breaths (active classifier)',
                             zorder=3)

        # Plot event markers (onset/offset/expmin/expoff) if enabled
        # Only show events for peaks that active classifier marks as breaths
        if self.show_events_checkbox.isChecked():
            breath_data = self.main_window.state.all_breaths_by_sweep.get(sweep_idx)
            if breath_data:
                sr_hz = self.main_window.state.sr_hz

                # Get active classifier labels and peak times for filtering
                active_labels = peak_data['labels']  # Currently displayed labels
                all_peak_indices = peak_data['indices']
                all_peak_times = all_peak_indices / sr_hz

                # Helper function: Check if event belongs to a breath peak
                def is_breath_event(event_time):
                    """Check if event time is associated with a breath peak."""
                    # Find nearest peak to this event time
                    peak_dists = np.abs(all_peak_times - event_time)
                    nearest_peak_idx = np.argmin(peak_dists)

                    # Only show event if:
                    # 1. There's a peak nearby (within 0.5s)
                    # 2. That peak is marked as breath by active classifier
                    if peak_dists[nearest_peak_idx] < 0.5:
                        return active_labels[nearest_peak_idx] == 1
                    return False

                # Inspiratory onsets (green upward triangles)
                if 'onsets' in breath_data and len(breath_data['onsets']) > 0:
                    onset_samples = breath_data['onsets']
                    onset_times = onset_samples / sr_hz
                    onset_mask = (onset_times >= t_min) & (onset_times <= t_max)
                    onset_times_in_window = onset_times[onset_mask]

                    # Filter by breath classification
                    onset_y_vals = []
                    onset_times_filtered = []
                    for ot in onset_times_in_window:
                        if is_breath_event(ot):
                            idx = np.argmin(np.abs(times - ot))
                            onset_y_vals.append(trace[idx])
                            onset_times_filtered.append(ot)

                    if len(onset_times_filtered) > 0:
                        self.ax.scatter(onset_times_filtered, onset_y_vals,
                                       marker='^',
                                       color=self.theme_manager.get_color('onset_color'),
                                       s=50,
                                       zorder=4)

                # Inspiratory offsets (orange downward triangles)
                if 'offsets' in breath_data and len(breath_data['offsets']) > 0:
                    offset_samples = breath_data['offsets']
                    offset_times = offset_samples / sr_hz
                    offset_mask = (offset_times >= t_min) & (offset_times <= t_max)
                    offset_times_in_window = offset_times[offset_mask]

                    # Filter by breath classification
                    offset_y_vals = []
                    offset_times_filtered = []
                    for ot in offset_times_in_window:
                        if is_breath_event(ot):
                            idx = np.argmin(np.abs(times - ot))
                            offset_y_vals.append(trace[idx])
                            offset_times_filtered.append(ot)

                    if len(offset_times_filtered) > 0:
                        self.ax.scatter(offset_times_filtered, offset_y_vals,
                                       marker='v',
                                       color=self.theme_manager.get_color('offset_color'),
                                       s=50,
                                       zorder=4)

                # Expiratory minima (blue squares)
                if 'expmins' in breath_data and len(breath_data['expmins']) > 0:
                    expmin_samples = breath_data['expmins']
                    expmin_times = expmin_samples / sr_hz
                    expmin_mask = (expmin_times >= t_min) & (expmin_times <= t_max)
                    expmin_times_in_window = expmin_times[expmin_mask]

                    # Filter by breath classification
                    expmin_y_vals = []
                    expmin_times_filtered = []
                    for et in expmin_times_in_window:
                        if is_breath_event(et):
                            idx = np.argmin(np.abs(times - et))
                            expmin_y_vals.append(trace[idx])
                            expmin_times_filtered.append(et)

                    if len(expmin_times_filtered) > 0:
                        self.ax.scatter(expmin_times_filtered, expmin_y_vals,
                                       marker='s',
                                       color=self.theme_manager.get_color('expmin_color'),
                                       s=50,
                                       zorder=4)

                # Expiratory offsets (purple diamonds)
                if 'expoffs' in breath_data and len(breath_data['expoffs']) > 0:
                    expoff_samples = breath_data['expoffs']
                    expoff_times = expoff_samples / sr_hz
                    expoff_mask = (expoff_times >= t_min) & (expoff_times <= t_max)
                    expoff_times_in_window = expoff_times[expoff_mask]

                    # Filter by breath classification
                    expoff_y_vals = []
                    expoff_times_filtered = []
                    for et in expoff_times_in_window:
                        if is_breath_event(et):
                            idx = np.argmin(np.abs(times - et))
                            expoff_y_vals.append(trace[idx])
                            expoff_times_filtered.append(et)

                    if len(expoff_times_filtered) > 0:
                        self.ax.scatter(expoff_times_filtered, expoff_y_vals,
                                       marker='D',
                                       color=self.theme_manager.get_color('expoff_color'),
                                       s=50,
                                       zorder=4)

        # Highlight disagreement span
        span_color = self.theme_manager.get_color('disagreement_span_color')
        self.ax.axvspan(peak_time - 0.1, peak_time + 0.1,
                        alpha=0.3,
                        color=span_color,
                        zorder=2)

        # Labels and title
        self.ax.set_xlabel('Time (s)', color=self.theme_manager.get_color('label_color'))
        self.ax.set_ylabel('Amplitude', color=self.theme_manager.get_color('label_color'))

        # Build title
        if edge['method'] == 'p_edge':
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s | P_edge: {edge['p_edge']:.3f}"
        elif edge['method'] == 'ml_disagree':
            # Use dynamic model names
            pred_a_text = "Breath" if edge['model_a_pred'] == 1 else "Noise"
            pred_b_text = "Breath" if edge['model_b_pred'] == 1 else "Noise"
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s | {edge['model_a_name']}: {pred_a_text} vs {edge['model_b_name']}: {pred_b_text}"
        else:
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s"

        self.ax.set_title(title, color=self.theme_manager.get_color('text_color'))

        # Legend
        self.ax.legend(loc='upper right')

        # Ensure window is exactly centered on peak
        # Set explicit x-axis limits AFTER all plotting is done
        self.ax.set_xlim(t_min, t_max)

        # Refresh canvas
        self.canvas.draw()

        # Update info label
        current_label_text = "BREATH" if edge['current_label'] == 1 else "NOISE"
        reviewed_text = "✓ Reviewed" if (edge['sweep'], edge['peak_idx']) in self.reviewed_peaks else "Pending"

        info_text = (f"Peak #{edge['peak_idx']} / {len(self.edge_cases)} edge cases | "
                    f"Sweep: {edge['sweep']} | Time: {edge['time']:.2f}s | "
                    f"Current: {current_label_text} | {reviewed_text}")

        if edge['method'] == 'p_edge':
            info_text += f" | P_edge: {edge['p_edge']:.3f}"

        self.info_label.setText(info_text)

    def _show_no_edge_cases_message(self):
        """Show a message in the plot when no edge cases are found."""
        self.ax.clear()

        # Apply theme
        theme_name = 'dark' if self.theme_combo.currentText() == 'Dark' else 'light'
        self.theme_manager.apply_theme(self.ax, self.fig, theme_name)

        # Add centered text message
        self.ax.text(0.5, 0.5, 'No edge cases found.\n\nTry:\n• Lowering the P_edge threshold\n• Using a different detection method\n• Loading ML models for disagreement detection',
                    ha='center', va='center',
                    transform=self.ax.transAxes,
                    fontsize=14,
                    color=self.theme_manager.get_color('text_color'))

        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas.draw()

        self.info_label.setText("No edge cases found with current settings.")

    def _refresh_plot(self):
        """Refresh the current plot with new settings."""
        if self.current_edge_idx >= 0:
            self._update_plot(self.current_edge_idx)

    def _on_theme_changed(self, theme_text):
        """Handle theme dropdown change."""
        self._refresh_plot()

    def _navigate_first(self):
        """Navigate to first edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = 0
        self._update_plot(self.current_edge_idx)

    def _navigate_previous(self):
        """Navigate to previous edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = max(0, self.current_edge_idx - 1)
        self._update_plot(self.current_edge_idx)

    def _navigate_next(self):
        """Navigate to next edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = min(len(self.edge_cases) - 1, self.current_edge_idx + 1)
        self._update_plot(self.current_edge_idx)

    def _navigate_last(self):
        """Navigate to last edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = len(self.edge_cases) - 1
        self._update_plot(self.current_edge_idx)

    def _mark_as_breath(self):
        """Mark current peak as BREATH."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self._apply_classification(edge, label=1, label_name="BREATH")

    def _mark_as_noise(self):
        """Mark current peak as NOISE."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self._apply_classification(edge, label=0, label_name="NOISE")

    def _apply_classification(self, edge: dict, label: int, label_name: str):
        """
        Apply classification to a peak.

        Args:
            edge: Edge case dictionary
            label: 0=noise, 1=breath
            label_name: Human-readable label
        """
        sweep = edge['sweep']
        peak_idx = edge['peak_idx']

        print(f"\n===== Applying Classification =====")
        print(f"Sweep: {sweep}, Peak: {peak_idx}")
        print(f"Marking as: {label_name} (label={label})")

        peak_data = self.main_window.state.all_peaks_by_sweep[sweep]
        peak_data['labels'][peak_idx] = label
        peak_data['label_source'][peak_idx] = 'user'

        # Mark as reviewed
        self.reviewed_peaks.add((sweep, peak_idx))

        # Update edge case current_label
        edge['current_label'] = label

        # Update progress
        self._update_ui_state()

        # Refresh plot if on current sweep
        if self.main_window.current_sweep == sweep:
            self.main_window.plot_sweep()

        print("===== Classification Applied =====\n")

        # Auto-advance to next
        self._navigate_next()

    def _skip_uncertain(self):
        """Skip current peak (mark as reviewed but don't change label)."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self.reviewed_peaks.add((edge['sweep'], edge['peak_idx']))

        # Update progress
        self._update_ui_state()

        # Auto-advance
        self._navigate_next()

    def _jump_to_main_window(self):
        """Jump to current peak in main window."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]

        print(f"\n===== Jumping to Main Window =====")
        print(f"Sweep: {edge['sweep']}, Time: {edge['time']:.2f}s")

        # Switch to sweep
        self.main_window.CurrentSweep.setValue(edge['sweep'])

        # Center view on peak
        window_half_width = 2.0
        xlim_left = max(0, edge['time'] - window_half_width)
        xlim_right = edge['time'] + window_half_width

        self.main_window.ax.set_xlim(xlim_left, xlim_right)
        self.main_window.canvas.draw()

        print("===== Jump Complete =====\n")

    def _export_edge_cases(self):
        """Export edge cases to CSV."""
        from PyQt6.QtWidgets import QFileDialog
        import csv

        if not self.edge_cases:
            QMessageBox.warning(self, "No Data", "No edge cases to export.")
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Edge Cases",
            "edge_cases.csv",
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'Sweep', 'PeakIndex', 'Time_s', 'Method',
                    'P_edge', 'Current_Label', 'Reviewed'
                ])

                # Data rows
                for edge in self.edge_cases:
                    reviewed = "Yes" if (edge['sweep'], edge['peak_idx']) in self.reviewed_peaks else "No"
                    p_edge_val = edge.get('p_edge', '')

                    writer.writerow([
                        edge['sweep'],
                        edge['peak_idx'],
                        f"{edge['time']:.3f}",
                        edge['method'],
                        f"{p_edge_val:.3f}" if p_edge_val else '',
                        edge['current_label'],
                        reviewed
                    ])

            QMessageBox.information(self, "Export Complete",
                                  f"Edge cases exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                               f"Failed to export edge cases:\n{str(e)}")
