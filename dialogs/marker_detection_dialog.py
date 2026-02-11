"""
Marker Detection Dialog.

Dialog for configuring and running automatic event detection
on signal channels with live preview visualization.
"""

import sys
from typing import Optional, Callable, List, Tuple, Dict, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QPushButton, QGroupBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QRadioButton,
    QButtonGroup, QFrame, QScrollArea, QWidget,
    QSizePolicy, QMessageBox, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QKeySequence, QShortcut
import numpy as np
import pyqtgraph as pg

from core.detection import DetectorRegistry, EventDetector, DetectionResult
from core.detection.base import ParamSpec, ParamType
from viewmodels.event_marker_viewmodel import EventMarkerViewModel

from dialogs.export_mixin import ExportMixin


class MarkerDetectionDialog(ExportMixin, QDialog):
    """
    Dialog for automatic event detection with live preview.

    Features:
    - Preview window showing detected events
    - Navigation through detected events
    - Live threshold line visualization
    - Support for multiple detection methods
    - Single/Paired marker type selection
    """

    # Signals
    detection_complete = pyqtSignal(int)  # Number of events detected
    preview_requested = pyqtSignal(list)  # List of (start, end) tuples for preview
    threshold_changed = pyqtSignal(float, str)  # (threshold_value, channel_name) for live preview
    threshold_cleared = pyqtSignal()  # Clear threshold line from plot

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        channel_names: List[str],
        get_signal_data: Callable[[str], Tuple[np.ndarray, np.ndarray, float]],
        sweep_idx: int,
        parent: Optional[QWidget] = None,
        initial_channel: Optional[str] = None,
        time_offset: float = 0.0,
    ):
        """
        Initialize the detection dialog.

        Args:
            viewmodel: Event marker viewmodel for creating markers
            channel_names: List of available channel names
            get_signal_data: Callback to get (time, signal, sample_rate) for a channel
            sweep_idx: Current sweep index
            parent: Parent widget
            initial_channel: Channel to pre-select (optional)
            time_offset: Time offset applied in main display (for stimulus normalization)
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._channel_names = channel_names
        self._get_signal_data = get_signal_data
        self._sweep_idx = sweep_idx
        self._initial_channel = initial_channel
        self._time_offset = time_offset

        self._current_detector: Optional[EventDetector] = None
        self._param_widgets: Dict[str, QWidget] = {}
        self._last_result: Optional[DetectionResult] = None

        # Preview state
        self._current_preview_idx = 0
        self._preview_events: List[Tuple[float, float]] = []
        self._signal_data: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

        # Auto-detection control
        self._auto_detect_enabled = True
        self._detection_pending = False

        # Preview plot items
        self._threshold_line: Optional[pg.InfiniteLine] = None
        self._event_region: Optional[pg.LinearRegionItem] = None
        self._lookback_region: Optional[pg.LinearRegionItem] = None
        self._baseline_region: Optional[pg.LinearRegionItem] = None
        self._onset_line: Optional[pg.InfiniteLine] = None
        self._signal_plot_item = None

        self._setup_ui()
        self._enable_dark_title_bar()
        self.setup_export_menu()
        self._connect_signals()
        self._populate_detectors()
        self._populate_channels()
        self._populate_marker_types()

        # Pre-select channel if provided
        if initial_channel and initial_channel in channel_names:
            idx = self._channel_combo.findText(initial_channel)
            if idx >= 0:
                self._channel_combo.setCurrentIndex(idx)

        # Load initial signal data
        self._load_signal_data()

        # Run initial detection after a brief delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(200, self._run_auto_detection)

    def reject(self) -> None:
        """Handle dialog rejection (Cancel/close) - clear threshold line."""
        self.threshold_cleared.emit()
        super().reject()

    def accept(self) -> None:
        """Handle dialog acceptance - clear threshold line."""
        self.threshold_cleared.emit()
        super().accept()

    def _setup_ui(self) -> None:
        """Set up the dialog UI with preview on left, options on right."""
        self.setWindowTitle("Auto-Detect Events")
        self.setMinimumWidth(700)
        self.setMinimumHeight(450)
        self.resize(950, 650)  # Default size, but fully resizable

        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(12)

        # =====================================================================
        # LEFT SIDE: Preview
        # =====================================================================
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Preview plot
        preview_group = QGroupBox("Preview")
        preview_group_layout = QVBoxLayout(preview_group)

        # Main signal plot with dual Y-axes
        self._preview_plot = pg.PlotWidget()
        self._preview_plot.setBackground('#1e1e1e')
        self._preview_plot.setMinimumSize(300, 250)
        self._preview_plot.showGrid(x=True, y=True, alpha=0.3)
        self._preview_plot.setLabel('bottom', 'Time', units='s')
        self._preview_plot.setLabel('left', 'Signal', color='#4fc3f7')

        # Lock Y-axis - only allow X-axis zoom/pan
        self._preview_plot.setMouseEnabled(x=True, y=False)
        self._preview_plot.getViewBox().setMouseMode(pg.ViewBox.RectMode)

        # Style the axes
        for axis in ['left', 'bottom']:
            ax = self._preview_plot.getAxis(axis)
            ax.setTextPen(pg.mkPen('#d4d4d4'))
            ax.setPen(pg.mkPen('#3e3e42'))

        # Add secondary Y-axis for derivative
        self._derivative_viewbox = pg.ViewBox()
        self._preview_plot.scene().addItem(self._derivative_viewbox)
        self._preview_plot.getAxis('right').linkToView(self._derivative_viewbox)
        self._derivative_viewbox.setXLink(self._preview_plot)

        # Style right axis for derivative
        self._preview_plot.showAxis('right')
        self._preview_plot.setLabel('right', 'dV/dt', color='#ff7043')
        right_axis = self._preview_plot.getAxis('right')
        right_axis.setTextPen(pg.mkPen('#ff7043'))
        right_axis.setPen(pg.mkPen('#ff7043', width=1))

        # Keep viewboxes synchronized on resize
        def update_views():
            self._derivative_viewbox.setGeometry(self._preview_plot.getViewBox().sceneBoundingRect())
            self._derivative_viewbox.linkedViewChanged(self._preview_plot.getViewBox(), self._derivative_viewbox.XAxis)

        self._preview_plot.getViewBox().sigResized.connect(update_views)

        preview_group_layout.addWidget(self._preview_plot)

        # Navigation controls
        nav_layout = QHBoxLayout()
        nav_layout.addStretch()

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.setToolTip("Previous detection")
        nav_layout.addWidget(self._prev_btn)

        self._nav_label = QLabel("No detections")
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav_label.setMinimumWidth(120)
        nav_layout.addWidget(self._nav_label)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(40)
        self._next_btn.setToolTip("Next detection")
        nav_layout.addWidget(self._next_btn)

        nav_layout.addStretch()
        preview_group_layout.addLayout(nav_layout)

        # View controls - Row 1: View mode and window size
        row1_layout = QHBoxLayout()

        # Full view / Event view toggle
        self._full_view_radio = QRadioButton("Full")
        self._event_view_radio = QRadioButton("Event")
        self._event_view_radio.setChecked(True)
        self._view_mode_group = QButtonGroup(self)
        self._view_mode_group.addButton(self._full_view_radio, 0)
        self._view_mode_group.addButton(self._event_view_radio, 1)
        row1_layout.addWidget(QLabel("View:"))
        row1_layout.addWidget(self._full_view_radio)
        row1_layout.addWidget(self._event_view_radio)

        row1_layout.addSpacing(16)

        # Window size controls (only for event view)
        row1_layout.addWidget(QLabel("Before:"))
        self._window_before_spin = QDoubleSpinBox()
        self._window_before_spin.setRange(0.5, 30.0)
        self._window_before_spin.setValue(5.0)
        self._window_before_spin.setSuffix(" s")
        self._window_before_spin.setSingleStep(1.0)
        self._window_before_spin.setFixedWidth(70)
        row1_layout.addWidget(self._window_before_spin)

        row1_layout.addWidget(QLabel("After:"))
        self._window_after_spin = QDoubleSpinBox()
        self._window_after_spin.setRange(1.0, 60.0)
        self._window_after_spin.setValue(15.0)
        self._window_after_spin.setSuffix(" s")
        self._window_after_spin.setSingleStep(1.0)
        self._window_after_spin.setFixedWidth(70)
        row1_layout.addWidget(self._window_after_spin)

        row1_layout.addStretch()
        preview_group_layout.addLayout(row1_layout)

        # View controls - Row 2: Derivative options
        row2_layout = QHBoxLayout()

        # Derivative show/hide and smoothing
        self._show_derivative_cb = QCheckBox("Show dV/dt")
        self._show_derivative_cb.setChecked(True)
        self._show_derivative_cb.setToolTip("Show/hide derivative trace (orange, right Y-axis)")
        row2_layout.addWidget(self._show_derivative_cb)

        row2_layout.addSpacing(8)

        row2_layout.addWidget(QLabel("Smooth:"))
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(1, 501)
        self._smooth_spin.setSingleStep(10)
        self._smooth_spin.setValue(51)  # Default window size
        self._smooth_spin.setToolTip(
            "Smoothing window size for derivative calculation.\n\n"
            "Higher values = smoother derivative (less noise)\n"
            "Lower values = more responsive to quick changes\n\n"
            "• 11-21: Minimal smoothing\n"
            "• 51: Default (moderate smoothing)\n"
            "• 101-201: Heavy smoothing for noisy signals\n\n"
            "Must be odd number (will be adjusted automatically)"
        )
        self._smooth_spin.setFixedWidth(60)
        row2_layout.addWidget(self._smooth_spin)

        row2_layout.addStretch()
        preview_group_layout.addLayout(row2_layout)

        # Legend for Hargreaves visualization
        self._legend_label = QLabel("")
        self._legend_label.setWordWrap(True)
        self._legend_label.setStyleSheet("color: #888; font-size: 10px;")
        preview_group_layout.addWidget(self._legend_label)

        preview_layout.addWidget(preview_group)

        # =====================================================================
        # RIGHT SIDE: Options
        # =====================================================================
        options_container = QWidget()
        options_layout = QVBoxLayout(options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)

        # Line 1: Channel and Category
        source_group = QGroupBox("Source && Marker")
        source_layout = QHBoxLayout(source_group)
        source_layout.setSpacing(12)

        # Channel
        source_layout.addWidget(QLabel("Channel:"))
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(100)
        source_layout.addWidget(self._channel_combo)

        source_layout.addSpacing(16)

        # Category
        source_layout.addWidget(QLabel("Category:"))
        self._category_combo = QComboBox()
        self._category_combo.setMinimumWidth(120)
        source_layout.addWidget(self._category_combo)

        source_layout.addStretch()
        options_layout.addWidget(source_group)

        # Condition selector (for grouping markers by experimental phase)
        condition_group = QGroupBox("Condition")
        condition_layout = QHBoxLayout(condition_group)

        condition_layout.addWidget(QLabel("Condition:"))
        self._condition_combo = QComboBox()
        self._condition_combo.setEditable(True)
        self._condition_combo.setMinimumWidth(120)
        self._condition_combo.setToolTip(
            "Assign a condition to group these markers.\n\n"
            "Examples:\n"
            "• 'baseline' vs 'treatment'\n"
            "• 'iso' vs 'awake'\n"
            "• 'pre' vs 'post'\n\n"
            "Leave empty for no condition grouping.\n"
            "Type a custom condition or select from presets."
        )
        # Add common presets
        self._condition_combo.addItem("")  # Empty = no condition
        self._condition_combo.addItem("baseline")
        self._condition_combo.addItem("treatment")
        self._condition_combo.addItem("iso")
        self._condition_combo.addItem("awake")
        self._condition_combo.addItem("pre")
        self._condition_combo.addItem("post")
        condition_layout.addWidget(self._condition_combo)

        condition_layout.addStretch()
        options_layout.addWidget(condition_group)

        # Line 2: Detection Method and Single/Paired
        method_group = QGroupBox("Detection")
        method_layout = QHBoxLayout(method_group)
        method_layout.setSpacing(12)

        # Detection method dropdown
        method_layout.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        self._method_combo.setMinimumWidth(140)
        method_layout.addWidget(self._method_combo)

        method_layout.addSpacing(16)

        # Single/Paired
        method_layout.addWidget(QLabel("Markers:"))
        self._paired_radio = QRadioButton("Paired")
        self._single_radio = QRadioButton("Single")
        self._paired_radio.setChecked(True)
        self._marker_type_group = QButtonGroup(self)
        self._marker_type_group.addButton(self._paired_radio, 0)
        self._marker_type_group.addButton(self._single_radio, 1)
        method_layout.addWidget(self._paired_radio)
        method_layout.addWidget(self._single_radio)

        method_layout.addStretch()
        options_layout.addWidget(method_group)

        # Parameters
        params_group = QGroupBox("Parameters")
        self._params_group = params_group
        params_layout = QVBoxLayout(params_group)

        self._params_container = QWidget()
        self._params_layout = QGridLayout(self._params_container)
        self._params_layout.setColumnStretch(1, 1)
        self._params_layout.setVerticalSpacing(6)
        params_layout.addWidget(self._params_container)
        params_layout.addStretch()

        options_layout.addWidget(params_group)

        # Scope selection
        scope_group = QGroupBox("Scope")
        scope_layout = QVBoxLayout(scope_group)
        self._scope_current = QRadioButton("Current Sweep Only")
        self._scope_all = QRadioButton("All Sweeps")
        self._scope_current.setChecked(True)
        scope_layout.addWidget(self._scope_current)
        scope_layout.addWidget(self._scope_all)
        options_layout.addWidget(scope_group)

        # Time Range filtering
        time_range_group = QGroupBox("Time Range")
        time_range_layout = QGridLayout(time_range_group)

        # Enable checkbox
        self._time_range_enabled = QCheckBox("Limit detection to time range")
        self._time_range_enabled.setChecked(False)
        self._time_range_enabled.setToolTip(
            "Enable to detect events only within a specific time window.\n\n"
            "Useful for separating experimental phases:\n"
            "• Baseline vs treatment periods\n"
            "• Isoflurane vs awake periods\n"
            "• Different stimulus conditions"
        )
        self._time_range_enabled.toggled.connect(self._on_time_range_toggled)
        time_range_layout.addWidget(self._time_range_enabled, 0, 0, 1, 4)

        # Start time
        time_range_layout.addWidget(QLabel("Start:"), 1, 0)
        self._time_start_spin = QDoubleSpinBox()
        self._time_start_spin.setRange(-3600.0, 36000.0)  # Large range for long recordings
        self._time_start_spin.setValue(0.0)
        self._time_start_spin.setSuffix(" s")
        self._time_start_spin.setDecimals(1)
        self._time_start_spin.setSingleStep(10.0)
        self._time_start_spin.setEnabled(False)
        self._time_start_spin.setToolTip("Start time for detection window (in seconds)")
        self._time_start_spin.valueChanged.connect(self._on_time_range_changed)
        time_range_layout.addWidget(self._time_start_spin, 1, 1)

        # End time
        time_range_layout.addWidget(QLabel("End:"), 1, 2)
        self._time_end_spin = QDoubleSpinBox()
        self._time_end_spin.setRange(-3600.0, 36000.0)
        self._time_end_spin.setValue(300.0)  # Default 5 minutes
        self._time_end_spin.setSuffix(" s")
        self._time_end_spin.setDecimals(1)
        self._time_end_spin.setSingleStep(10.0)
        self._time_end_spin.setEnabled(False)
        self._time_end_spin.setToolTip("End time for detection window (in seconds)")
        self._time_end_spin.valueChanged.connect(self._on_time_range_changed)
        time_range_layout.addWidget(self._time_end_spin, 1, 3)

        options_layout.addWidget(time_range_group)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888;")
        options_layout.addWidget(self._status_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip("Run detection and show preview without creating markers")
        button_layout.addWidget(self._preview_btn)

        self._detect_btn = QPushButton("Detect")
        self._detect_btn.setToolTip("Create markers for detected events")
        self._detect_btn.setDefault(True)
        button_layout.addWidget(self._detect_btn)

        self._cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self._cancel_btn)

        options_layout.addLayout(button_layout)

        # Add both sides to main layout
        main_layout.addWidget(preview_container, stretch=1)
        main_layout.addWidget(options_container, stretch=1)

        # Apply dark theme styling
        self._apply_styling()

    def _enable_dark_title_bar(self) -> None:
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

    def _apply_styling(self) -> None:
        """Apply dark theme styling."""
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
                font-weight: bold;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #252526;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #d4d4d4;
            }
            QComboBox {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                selection-background-color: #094771;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
                min-height: 20px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #007acc;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                background-color: #3e3e42;
                border: none;
                width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover,
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #505050;
            }
            QPushButton {
                background-color: #0e639c;
                border: none;
                border-radius: 3px;
                padding: 6px 16px;
                color: white;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8c;
            }
            QRadioButton {
                spacing: 8px;
                color: #d4d4d4;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 7px;
                background-color: #2d2d30;
            }
            QRadioButton::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QRadioButton::indicator:hover {
                border-color: #007acc;
            }
            QCheckBox {
                spacing: 8px;
                color: #d4d4d4;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #3e3e42;
                border-radius: 2px;
                background-color: #2d2d30;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QCheckBox::indicator:hover {
                border-color: #007acc;
            }
            QLabel {
                color: #d4d4d4;
                background-color: transparent;
            }
        """)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._preview_btn.clicked.connect(self._on_preview)
        self._detect_btn.clicked.connect(self._on_detect)
        self._cancel_btn.clicked.connect(self.reject)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._method_combo.currentIndexChanged.connect(self._on_detector_changed)
        self._prev_btn.clicked.connect(self._on_prev_detection)
        self._next_btn.clicked.connect(self._on_next_detection)

        # View mode and window size
        self._full_view_radio.toggled.connect(self._on_view_mode_changed)
        self._event_view_radio.toggled.connect(self._on_view_mode_changed)
        self._window_before_spin.valueChanged.connect(self._on_window_size_changed)
        self._window_after_spin.valueChanged.connect(self._on_window_size_changed)

        # Derivative controls
        self._show_derivative_cb.toggled.connect(self._on_derivative_visibility_changed)
        self._smooth_spin.valueChanged.connect(self._on_smooth_changed)

        # Ctrl+R shortcut to close dialog (allows hot reload to proceed)
        ctrl_r_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        ctrl_r_shortcut.activated.connect(self._on_hot_reload_requested)

    def _on_hot_reload_requested(self) -> None:
        """Handle Ctrl+R - close dialog to allow hot reload."""
        self.reject()  # Close dialog, main window shortcut will fire

    def _on_view_mode_changed(self) -> None:
        """Handle view mode toggle."""
        # Enable/disable window size controls based on view mode
        is_event_view = self._event_view_radio.isChecked()
        self._window_before_spin.setEnabled(is_event_view)
        self._window_after_spin.setEnabled(is_event_view)

        # Update the view
        if self._full_view_radio.isChecked():
            self._show_full_view()
        elif self._preview_events:
            self._show_current_detection()

    def _on_window_size_changed(self) -> None:
        """Handle window size change."""
        if self._event_view_radio.isChecked() and self._preview_events:
            self._show_current_detection()

    def _on_time_range_toggled(self, enabled: bool) -> None:
        """Handle time range checkbox toggle."""
        self._time_start_spin.setEnabled(enabled)
        self._time_end_spin.setEnabled(enabled)

        # Initialize end time to signal duration if enabling
        if enabled and self._signal_data is not None:
            time, signal, sample_rate = self._signal_data
            display_time = time - self._time_offset
            # Set default range
            if self._time_start_spin.value() == 0.0 and self._time_end_spin.value() == 300.0:
                self._time_end_spin.setValue(display_time[-1])

        # Re-run detection with new settings
        self._schedule_auto_detection()

    def _on_time_range_changed(self) -> None:
        """Handle time range value changes."""
        # Re-run detection with new time range
        self._schedule_auto_detection()

    def _get_time_range_bounds(self) -> Optional[Tuple[float, float]]:
        """Get the time range bounds for filtering, or None if disabled.

        Returns times in signal coordinates (with offset applied back).
        """
        if not self._time_range_enabled.isChecked():
            return None

        # These are display times, convert back to signal times
        start_display = self._time_start_spin.value()
        end_display = self._time_end_spin.value()

        # Add offset back to get signal time
        start_time = start_display + self._time_offset
        end_time = end_display + self._time_offset

        return (start_time, end_time)

    def _on_derivative_visibility_changed(self, visible: bool) -> None:
        """Handle derivative plot visibility toggle."""
        self._smooth_spin.setEnabled(visible)
        # Show/hide the right axis and derivative trace
        self._preview_plot.showAxis('right', visible)
        self._update_derivative_plot()

    def _on_smooth_changed(self) -> None:
        """Handle smoothing value change - recalculate derivative."""
        self._update_derivative_plot()

    def _calculate_derivative(self, signal: np.ndarray, time: np.ndarray) -> np.ndarray:
        """Calculate smoothed derivative of signal using Savitzky-Golay filter."""
        from scipy.signal import savgol_filter

        # Get smoothing window size (must be odd)
        window = self._smooth_spin.value()
        if window % 2 == 0:
            window += 1
        if window < 3:
            window = 3

        # Ensure window doesn't exceed signal length
        if window >= len(signal):
            window = len(signal) - 1 if len(signal) % 2 == 0 else len(signal) - 2
            if window < 3:
                # Signal too short, use simple diff
                dt = time[1] - time[0] if len(time) > 1 else 1.0
                return np.gradient(signal, dt)

        # Calculate derivative using Savitzky-Golay filter
        # deriv=1 means first derivative, polyorder=2 or 3 works well
        try:
            dt = time[1] - time[0] if len(time) > 1 else 1.0
            derivative = savgol_filter(signal, window, polyorder=3, deriv=1, delta=dt)
            return derivative
        except Exception:
            # Fallback to simple gradient
            dt = time[1] - time[0] if len(time) > 1 else 1.0
            return np.gradient(signal, dt)

    def _update_derivative_plot(self) -> None:
        """Update the derivative trace on the secondary Y-axis."""
        # Clear previous derivative items from the viewbox
        self._derivative_viewbox.clear()

        if self._signal_data is None or not self._show_derivative_cb.isChecked():
            return

        time, signal, sample_rate = self._signal_data
        display_time = time - self._time_offset

        # Calculate smoothed derivative
        derivative = self._calculate_derivative(signal, time)

        # Create plot data item for derivative
        pen = pg.mkPen(color='#ff7043', width=1)  # Orange-red color
        derivative_curve = pg.PlotDataItem(display_time, derivative, pen=pen)
        self._derivative_viewbox.addItem(derivative_curve)

        # Add zero line for reference
        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen('#555555', width=1, style=Qt.PenStyle.DotLine)
        )
        self._derivative_viewbox.addItem(zero_line)

        # Auto-scale the derivative Y-axis
        self._derivative_viewbox.autoRange()

    def _show_full_view(self) -> None:
        """Show full signal view."""
        if self._signal_data is None:
            return

        time, signal, sample_rate = self._signal_data
        display_time = time - self._time_offset

        # Set view to full signal range
        self._preview_plot.setXRange(display_time[0], display_time[-1], padding=0.02)

        # Auto-scale Y
        y_min, y_max = signal.min(), signal.max()
        y_range = y_max - y_min
        if y_range > 0:
            self._preview_plot.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    def _on_channel_changed(self) -> None:
        """Handle channel selection change."""
        self._load_signal_data()
        self._update_preview_plot()
        self._emit_threshold_update()

    def _load_signal_data(self) -> None:
        """Load signal data for the selected channel."""
        channel = self._channel_combo.currentText()
        if not channel:
            self._signal_data = None
            return

        try:
            self._signal_data = self._get_signal_data(channel)
            self._update_preview_plot()
        except Exception as e:
            print(f"[MarkerDetectionDialog] Error loading signal data: {e}")
            self._signal_data = None

    def _update_preview_plot(self) -> None:
        """Update the preview plot with current signal data."""
        self._preview_plot.clear()
        self._threshold_line = None
        self._event_region = None
        self._lookback_region = None
        self._baseline_region = None
        self._onset_line = None
        self._signal_plot_item = None

        if self._signal_data is None:
            return

        time, signal, sample_rate = self._signal_data

        # Apply time offset to match main display (subtract offset so t=0 is at stim)
        display_time = time - self._time_offset

        # Plot signal - use simple default settings first
        pen = pg.mkPen(color='#4fc3f7', width=1)
        self._signal_plot_item = self._preview_plot.plot(
            display_time, signal, pen=pen,
        )

        # Add draggable threshold line
        self._add_threshold_line()

        # Update derivative plot
        self._update_derivative_plot()

        # If we have preview events, show the current one in event view
        if self._preview_events and self._event_view_radio.isChecked():
            self._show_current_detection()
        elif self._full_view_radio.isChecked():
            self._show_full_view()

    def _add_threshold_line(self) -> None:
        """Add draggable threshold line to preview plot."""
        if self._current_detector is None:
            return

        threshold = self._current_detector.get_param('threshold')
        if threshold is None:
            return

        # Remove existing threshold line
        if self._threshold_line is not None:
            try:
                self._threshold_line.sigPositionChanged.disconnect()
            except:
                pass
            self._preview_plot.removeItem(self._threshold_line)

        # Add new DRAGGABLE threshold line
        pen = pg.mkPen(color='#ff9800', width=2, style=Qt.PenStyle.DashLine)
        hover_pen = pg.mkPen(color='#ffb74d', width=3, style=Qt.PenStyle.DashLine)
        self._threshold_line = pg.InfiniteLine(
            pos=threshold,
            angle=0,
            pen=pen,
            hoverPen=hover_pen,
            movable=True,  # Make it draggable!
            label=f'Threshold: {threshold:.3f}',
            labelOpts={'position': 0.95, 'color': (255, 152, 0), 'fill': (30, 30, 30, 200)}
        )
        self._threshold_line.sigPositionChanged.connect(self._on_threshold_dragged)
        self._preview_plot.addItem(self._threshold_line)

    def _on_threshold_dragged(self, line: pg.InfiniteLine) -> None:
        """Handle threshold line being dragged."""
        new_threshold = line.value()

        # Update the label
        line.label.setFormat(f'Threshold: {new_threshold:.3f}')

        # Update detector parameter
        if self._current_detector:
            self._current_detector.set_param('threshold', new_threshold)

            # Update the spinbox widget to reflect new value
            if 'threshold' in self._param_widgets:
                widget = self._param_widgets['threshold']
                widget.blockSignals(True)
                widget.setValue(new_threshold)
                widget.blockSignals(False)

            # Emit threshold update for main plot
            self._emit_threshold_update()

            # Schedule auto-detection
            self._schedule_auto_detection()

    def _show_current_detection(self) -> None:
        """Show the current detection in the preview plot."""
        if not self._preview_events or self._signal_data is None:
            return

        # Clamp index
        self._current_preview_idx = max(0, min(self._current_preview_idx, len(self._preview_events) - 1))

        start_time, end_time = self._preview_events[self._current_preview_idx]
        time, signal, sample_rate = self._signal_data

        # Convert to display time (with offset applied)
        display_start = start_time - self._time_offset
        display_end = end_time - self._time_offset
        display_time = time - self._time_offset

        # Clear previous visualization items
        for item in [self._event_region, self._lookback_region, self._baseline_region, self._onset_line]:
            if item is not None:
                try:
                    self._preview_plot.removeItem(item)
                except:
                    pass

        # Get window size from spinboxes
        window_before = self._window_before_spin.value()
        window_after = self._window_after_spin.value()

        # Calculate view window using user-specified sizes
        view_start = max(display_time[0], display_start - window_before)
        view_end = min(display_time[-1], display_start + window_after)

        # Set view range
        self._preview_plot.setXRange(view_start, view_end, padding=0.02)

        # Auto-scale Y to visible data
        mask = (display_time >= view_start) & (display_time <= view_end)
        if np.any(mask):
            y_min, y_max = signal[mask].min(), signal[mask].max()
            y_range = y_max - y_min
            if y_range > 0:
                self._preview_plot.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Add event region (green) - use display times
        self._event_region = pg.LinearRegionItem(
            values=[display_start, display_end],
            brush=pg.mkBrush(0, 200, 100, 50),
            pen=pg.mkPen(0, 200, 100, 200),
            movable=False,
        )
        self._preview_plot.addItem(self._event_region)

        # For Hargreaves, show lookback window and baseline region
        if self._current_detector and 'Hargreaves' in self._current_detector.name:
            self._show_hargreaves_visualization(display_start, display_end, display_time, signal)

        # Update navigation label
        self._nav_label.setText(f"Detection {self._current_preview_idx + 1} of {len(self._preview_events)}")

        # Update button states
        self._prev_btn.setEnabled(self._current_preview_idx > 0)
        self._next_btn.setEnabled(self._current_preview_idx < len(self._preview_events) - 1)

    def _show_hargreaves_visualization(self, start_time: float, end_time: float,
                                        time: np.ndarray, signal: np.ndarray) -> None:
        """Show Hargreaves-specific visualization (lookback window, baseline region)."""
        lookback_time = self._current_detector.get_param('lookback_time') or 1.0
        baseline_frac = self._current_detector.get_param('baseline_fraction') or 0.2

        # Find the threshold crossing point (approximate)
        threshold = self._current_detector.get_param('threshold') or 0.5

        # Lookback window starts from threshold crossing and goes backward
        # For visualization, we'll show from (start_time - small_offset) to (start_time + lookback_region)
        lookback_start = start_time - lookback_time
        lookback_end = start_time

        # Baseline region is first N% of lookback window
        baseline_start = lookback_start
        baseline_end = lookback_start + lookback_time * baseline_frac

        # Add lookback region (blue, very transparent)
        if lookback_start >= time[0]:
            self._lookback_region = pg.LinearRegionItem(
                values=[lookback_start, lookback_end],
                brush=pg.mkBrush(33, 150, 243, 30),  # Light blue
                pen=pg.mkPen(33, 150, 243, 100),
                movable=False,
            )
            self._preview_plot.addItem(self._lookback_region)

        # Add baseline region (yellow, more visible)
        if baseline_start >= time[0]:
            self._baseline_region = pg.LinearRegionItem(
                values=[baseline_start, baseline_end],
                brush=pg.mkBrush(255, 235, 59, 50),  # Yellow
                pen=pg.mkPen(255, 235, 59, 150),
                movable=False,
            )
            self._preview_plot.addItem(self._baseline_region)

        # Add onset marker line (vertical line at detected onset)
        self._onset_line = pg.InfiniteLine(
            pos=start_time,
            angle=90,
            pen=pg.mkPen(0, 255, 0, 200, width=2),
            label='Onset',
            labelOpts={'position': 0.9, 'color': (0, 255, 0), 'fill': (30, 30, 30, 200)}
        )
        self._preview_plot.addItem(self._onset_line)

        # Update legend
        self._legend_label.setText(
            "Legend: Green = Detected Event | Blue = Lookback Window | "
            "Yellow = Baseline Region | Orange = Threshold"
        )

    def _on_prev_detection(self) -> None:
        """Navigate to previous detection."""
        if self._current_preview_idx > 0:
            self._current_preview_idx -= 1
            self._update_preview_plot()
            self._show_current_detection()

    def _on_next_detection(self) -> None:
        """Navigate to next detection."""
        if self._current_preview_idx < len(self._preview_events) - 1:
            self._current_preview_idx += 1
            self._update_preview_plot()
            self._show_current_detection()

    def _populate_detectors(self) -> None:
        """Populate detection method dropdown."""
        detectors = DetectorRegistry.all()

        self._method_combo.clear()
        for detector_class in detectors:
            self._method_combo.addItem(detector_class.name, detector_class.name)
            # Set tooltip for the item
            idx = self._method_combo.count() - 1
            self._method_combo.setItemData(idx, detector_class.description, Qt.ItemDataRole.ToolTipRole)

        # Select first detector by default - this triggers _on_detector_changed
        if self._method_combo.count() > 0:
            self._method_combo.setCurrentIndex(0)
            self._on_detector_changed()

    def _populate_channels(self) -> None:
        """Populate channel dropdown."""
        self._channel_combo.clear()
        for name in self._channel_names:
            self._channel_combo.addItem(name)

    def _populate_marker_types(self) -> None:
        """Populate marker type dropdown."""
        self._category_combo.clear()
        categories = self._viewmodel.get_categories()

        for cat in sorted(categories, key=lambda c: c.name):
            self._category_combo.addItem(cat.display_name, cat.name)

    def _on_detector_changed(self, index: int = None) -> None:
        """Handle detector method selection change."""
        name = self._method_combo.currentData()
        if not name:
            return

        self._current_detector = DetectorRegistry.create(name)
        self._rebuild_param_widgets()

        # Auto-select Hargreaves category if Hargreaves method selected
        if 'Hargreaves' in name:
            self._auto_select_hargreaves_category()
            self._paired_radio.setChecked(True)  # Hargreaves uses paired markers
            self._legend_label.setText(
                "Legend: Green = Detected Event | Blue = Lookback Window | "
                "Yellow = Baseline Region | Orange = Threshold"
            )
        else:
            self._legend_label.setText("")

        # Update preview
        self._update_preview_plot()

    def _auto_select_hargreaves_category(self) -> None:
        """Auto-select Hargreaves category."""
        # Find Hargreaves category in combo
        for i in range(self._category_combo.count()):
            if self._category_combo.itemData(i) == 'hargreaves':
                self._category_combo.setCurrentIndex(i)
                break

    def _rebuild_param_widgets(self) -> None:
        """Rebuild parameter widgets for current detector."""
        # Clear existing widgets
        for widget in self._param_widgets.values():
            widget.deleteLater()
        self._param_widgets.clear()

        # Clear layout
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self._current_detector:
            return

        # Create widgets for each parameter
        specs = self._current_detector.get_param_specs()
        for row, spec in enumerate(specs):
            # Label
            label_text = spec.label
            if spec.unit:
                label_text += f" ({spec.unit})"
            label = QLabel(label_text + ":")
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            self._params_layout.addWidget(label, row, 0)

            # Widget based on type
            widget = self._create_param_widget(spec)
            self._params_layout.addWidget(widget, row, 1)
            self._param_widgets[spec.name] = widget

    def _create_param_widget(self, spec: ParamSpec) -> QWidget:
        """Create appropriate widget for a parameter specification."""
        if spec.param_type == ParamType.FLOAT:
            widget = QDoubleSpinBox()
            widget.setDecimals(3)
            if spec.min_value is not None:
                widget.setMinimum(spec.min_value)
            if spec.max_value is not None:
                widget.setMaximum(spec.max_value)
            if spec.step is not None:
                widget.setSingleStep(spec.step)
            widget.setValue(spec.default)
            widget.valueChanged.connect(
                lambda v, n=spec.name: self._on_param_changed(n, v)
            )

        elif spec.param_type == ParamType.INT:
            widget = QSpinBox()
            if spec.min_value is not None:
                widget.setMinimum(int(spec.min_value))
            if spec.max_value is not None:
                widget.setMaximum(int(spec.max_value))
            if spec.step is not None:
                widget.setSingleStep(int(spec.step))
            widget.setValue(int(spec.default))
            widget.valueChanged.connect(
                lambda v, n=spec.name: self._on_param_changed(n, v)
            )

        elif spec.param_type == ParamType.BOOL:
            widget = QCheckBox()
            widget.setChecked(bool(spec.default))
            widget.stateChanged.connect(
                lambda s, n=spec.name: self._on_param_changed(n, s == Qt.CheckState.Checked.value)
            )

        elif spec.param_type == ParamType.CHOICE:
            widget = QComboBox()
            for display, value in spec.choices:
                widget.addItem(display, value)
            # Set default
            for i in range(widget.count()):
                if widget.itemData(i) == spec.default:
                    widget.setCurrentIndex(i)
                    break
            widget.currentIndexChanged.connect(
                lambda idx, w=widget, n=spec.name: self._on_param_changed(n, w.currentData())
            )

        else:
            widget = QLabel(f"Unknown type: {spec.param_type}")

        if spec.tooltip:
            widget.setToolTip(spec.tooltip)

        return widget

    def _on_param_changed(self, name: str, value: Any) -> None:
        """Handle parameter value change."""
        if self._current_detector:
            self._current_detector.set_param(name, value)
            # If threshold changed, update the threshold line on plot
            if name == 'threshold':
                self._emit_threshold_update()
                # Update threshold line position without recreating
                if self._threshold_line is not None:
                    self._threshold_line.blockSignals(True)
                    self._threshold_line.setValue(value)
                    self._threshold_line.label.setFormat(f'Threshold: {value:.3f}')
                    self._threshold_line.blockSignals(False)

            # Schedule auto-detection for any parameter change
            self._schedule_auto_detection()

    def _schedule_auto_detection(self) -> None:
        """Schedule auto-detection with debouncing."""
        if not self._auto_detect_enabled:
            return

        # Use a timer to debounce rapid changes
        if not hasattr(self, '_auto_detect_timer'):
            from PyQt6.QtCore import QTimer
            self._auto_detect_timer = QTimer()
            self._auto_detect_timer.setSingleShot(True)
            self._auto_detect_timer.timeout.connect(self._run_auto_detection)

        # Reset timer on each call (debounce)
        self._auto_detect_timer.stop()
        self._auto_detect_timer.start(300)  # 300ms debounce

    def _run_auto_detection(self) -> None:
        """Run detection and update preview automatically."""
        if self._signal_data is None or self._current_detector is None:
            return

        result = self._run_detection(warn_on_many=False)  # Don't warn during auto-detect
        if result:
            # Check for too many events
            if result.count > 100:
                self._status_label.setText(f"⚠ {result.count} events (try higher min_gap)")
                self._status_label.setStyleSheet("color: #ff9800;")  # Orange warning
                # Still show preview but limit to first 100
                self._preview_events = result.events[:100]
            else:
                self._status_label.setStyleSheet("color: #888;")
                self._preview_events = result.events

            # Keep current index if still valid, otherwise go to first
            if self._current_preview_idx >= len(self._preview_events):
                self._current_preview_idx = 0

            if len(self._preview_events) > 0:
                self._show_current_detection()
                shown = len(self._preview_events)
                total = result.count
                if total > shown:
                    self._nav_label.setText(f"Detection {self._current_preview_idx + 1} of {shown} (of {total})")
                else:
                    self._nav_label.setText(f"Detection {self._current_preview_idx + 1} of {total}")
            else:
                self._nav_label.setText("No detections")
                self._preview_events = []
                # Clear visualization items
                for item in [self._event_region, self._lookback_region, self._baseline_region, self._onset_line]:
                    if item is not None:
                        try:
                            self._preview_plot.removeItem(item)
                        except:
                            pass
                self._event_region = None
                self._lookback_region = None
                self._baseline_region = None
                self._onset_line = None

    def _emit_threshold_update(self) -> None:
        """Emit threshold update signal for live preview on plot."""
        if self._current_detector:
            threshold = self._current_detector.get_param('threshold')
            if threshold is not None:
                channel = self._channel_combo.currentText()
                if channel:
                    self.threshold_changed.emit(threshold, channel)

    def _get_current_signal_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Get signal data for currently selected channel."""
        channel = self._channel_combo.currentText()
        if not channel:
            return None
        return self._get_signal_data(channel)

    def _run_detection(self, warn_on_many: bool = True) -> Optional[DetectionResult]:
        """Run detection with current settings."""
        if not self._current_detector:
            self._status_label.setText("No detector selected")
            return None

        data = self._get_current_signal_data()
        if data is None:
            self._status_label.setText("No signal data available")
            return None

        time, signal, sample_rate = data

        try:
            result = self._current_detector.detect(signal, time, sample_rate)

            # Apply time range filtering if enabled
            time_bounds = self._get_time_range_bounds()
            if time_bounds is not None:
                start_bound, end_bound = time_bounds
                # Filter events to only those within the time range
                # Event start time must be within the range
                filtered_events = [
                    (start, end) for (start, end) in result.events
                    if start >= start_bound and start <= end_bound
                ]
                # Create new result with filtered events
                from core.detection.base import DetectionResult
                result = DetectionResult(events=filtered_events, metadata=result.metadata)

            self._last_result = result

            # Build status message
            if time_bounds is not None:
                range_str = f" (in range)"
            else:
                range_str = ""

            if result.count > 100:
                self._status_label.setText(f"⚠ Detected {result.count} events{range_str}")
                self._status_label.setStyleSheet("color: #ff9800;")
            else:
                self._status_label.setText(f"Detected {result.count} events{range_str}")
                self._status_label.setStyleSheet("color: #888;")

            return result
        except Exception as e:
            self._status_label.setText(f"Detection error: {e}")
            self._status_label.setStyleSheet("color: #f44336;")
            return None

    def _on_preview(self) -> None:
        """Handle preview button click."""
        result = self._run_detection()
        if result:
            self._preview_events = result.events
            self._current_preview_idx = 0

            if result.count > 0:
                self._update_preview_plot()
                self._show_current_detection()
                self._nav_label.setText(f"Detection 1 of {result.count}")
            else:
                self._nav_label.setText("No detections")
                self._preview_events = []

            self.preview_requested.emit(result.events)

    def _on_detect(self) -> None:
        """Handle detect button click."""
        result = self._run_detection()
        if not result or result.count == 0:
            QMessageBox.information(
                self,
                "No Events",
                "No events were detected with the current settings.\n"
                "Try adjusting the threshold or other parameters."
            )
            return

        # Warn if too many events
        if result.count > 100:
            reply = QMessageBox.warning(
                self,
                "Many Events Detected",
                f"Detected {result.count} events. This seems like a lot.\n\n"
                f"This may indicate:\n"
                f"• Threshold is too sensitive\n"
                f"• Min Gap is too small (try 0.5-1.0s)\n"
                f"• Signal is very noisy\n\n"
                f"Do you want to create all {result.count} markers anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Get selected marker type and condition
        category = self._category_combo.currentData()
        is_paired = self._paired_radio.isChecked()
        condition = self._condition_combo.currentText().strip() or None

        if not category:
            QMessageBox.warning(
                self,
                "Select Category",
                "Please select a category for the detected events."
            )
            return

        # Get first label from category
        cat = self._viewmodel.service.registry.get(category)
        label = cat.labels[0] if cat and cat.labels else 'default'

        # Show selection dialog if more than 1 event detected
        events_to_create = result.events
        if len(result.events) > 1:
            from dialogs.event_selection_dialog import EventSelectionDialog
            sel_dialog = EventSelectionDialog(result.events, parent=self)
            if sel_dialog.exec() != EventSelectionDialog.DialogCode.Accepted:
                return
            events_to_create = sel_dialog.get_selected_events()
            if not events_to_create:
                self._status_label.setText("No events selected")
                return

        # Create markers for each selected event
        created = 0
        for start_time, end_time in events_to_create:
            if is_paired:
                # Create paired marker for each event
                marker = self._viewmodel.add_paired_marker(
                    start_time=start_time,
                    end_time=end_time,
                    sweep_idx=self._sweep_idx,
                    category=category,
                    label=label,
                )
            else:
                # Create single marker at start time only
                marker = self._viewmodel.add_single_marker(
                    time=start_time,
                    sweep_idx=self._sweep_idx,
                )
                # Update category/label
                if marker:
                    self._viewmodel.update_marker(
                        marker.id,
                        category=category,
                        label=label,
                    )

            # Set condition on marker if specified
            if marker and condition:
                self._viewmodel.update_marker(marker.id, condition=condition)

            if marker:
                created += 1

        self._status_label.setText(f"Created {created} markers")
        self.detection_complete.emit(created)
        self.accept()

    def get_last_result(self) -> Optional[DetectionResult]:
        """Get the last detection result."""
        return self._last_result
