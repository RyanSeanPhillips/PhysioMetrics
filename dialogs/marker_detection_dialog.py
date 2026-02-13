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
    QMenu, QWidgetAction,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QPen, QKeySequence, QShortcut
import numpy as np
import pyqtgraph as pg

from core.detection import DetectorRegistry, EventDetector, DetectionResult
from core.detection.base import ParamSpec, ParamType
from viewmodels.event_marker_viewmodel import EventMarkerViewModel

from dialogs.export_mixin import ExportMixin
from dialogs.event_selection_dialog import CONDITION_PRESETS


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
        initial_events: Optional[List[Tuple[float, float]]] = None,
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
            initial_events: Pre-loaded events to show in preview (e.g., from existing markers)
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._channel_names = channel_names
        self._get_signal_data = get_signal_data
        self._sweep_idx = sweep_idx
        self._initial_channel = initial_channel
        self._time_offset = time_offset
        self._initial_events = initial_events

        self._current_detector: Optional[EventDetector] = None
        self._param_widgets: Dict[str, QWidget] = {}
        self._last_result: Optional[DetectionResult] = None

        # Preview state
        self._current_preview_idx = 0
        self._preview_events: List[Tuple[float, float]] = []
        self._preview_conditions: List[Optional[str]] = []  # Per-event conditions from loaded markers
        self._signal_data: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

        # Auto-detection control
        self._auto_detect_enabled = True
        self._detection_pending = False

        # Event list panel state
        self._event_checkboxes: List[QCheckBox] = []
        self._event_condition_combos: List[QComboBox] = []
        self._marker_groups: Dict[Tuple[str, str], list] = {}  # (source_channel, category) -> [EventMarker]
        self._current_default_condition: str = ""  # Track for condition propagation
        self._loading_marker_group = False  # Prevent auto-detection when loading a group
        self._preview_marker_ids: List[Optional[str]] = []  # Marker IDs when editing existing
        self._preview_original_times: List[Optional[Tuple[float, float]]] = []  # Original times for change detection

        # Preview plot items
        self._threshold_line: Optional[pg.InfiniteLine] = None
        self._event_region: Optional[pg.LinearRegionItem] = None
        self._lookback_region: Optional[pg.LinearRegionItem] = None
        self._baseline_region: Optional[pg.LinearRegionItem] = None
        self._onset_line: Optional[pg.InfiniteLine] = None
        self._signal_plot_item = None
        self._intersection_items: List = []  # Temp items drawn during edge drag

        # Undo/redo for preview edge drags
        self._edge_drag_history: List[Tuple[int, Tuple[float, float]]] = []
        self._edge_drag_redo: List[Tuple[int, Tuple[float, float]]] = []

        self._setup_ui()
        self._enable_dark_title_bar()
        self._set_window_icon()
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

        # Load initial signal data and set threshold to match signal range
        self._load_signal_data()
        self._auto_set_threshold()
        self._update_preview_plot()  # Rebuild plot with the corrected threshold

        # Scan existing markers and populate group dropdown
        self._scan_marker_groups()
        self._populate_marker_group_combo()

        # If initial events are provided, load them into preview
        if initial_events:
            self._preview_events = list(initial_events)
            self._current_preview_idx = 0
            self._auto_detect_enabled = False  # Don't overwrite with auto-detection
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, self._show_initial_events)
        elif self._marker_group_combo.currentIndex() > 0:
            # Existing marker group was auto-selected — load it
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, lambda: self._on_marker_group_changed(self._marker_group_combo.currentIndex()))
        else:
            # No existing markers or "(New Detection)" selected — run auto-detection
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
        self.setMinimumWidth(900)
        self.setMinimumHeight(450)
        self.resize(1200, 650)  # Wider to accommodate event list panel

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)

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
        self._preview_plot.setLabel('left', 'Signal', color='#d4d4d4')

        # Disable SI prefix auto-scaling on Y-axis so values match threshold labels
        # (prevents pyqtgraph from showing 0.5 instead of 500 with a ×0.001 multiplier)
        self._preview_plot.getAxis('left').enableAutoSIPrefix(False)

        # Lock Y-axis - only allow X-axis scroll-wheel zoom and pan (no rect drag zoom)
        self._preview_plot.setMouseEnabled(x=True, y=False)
        self._preview_plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)

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

        # Style right axis for derivative (hidden by default since dV/dt starts unchecked)
        right_axis = self._preview_plot.getAxis('right')
        right_axis.setLabel('dV/dt', color='#ff7043')
        right_axis.setTextPen(pg.mkPen('#ff7043'))
        right_axis.setPen(pg.mkPen('#ff7043', width=1))
        self._preview_plot.showAxis('right', False)

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

        # Legend for Hargreaves visualization
        self._legend_label = QLabel("")
        self._legend_label.setWordWrap(True)
        self._legend_label.setStyleSheet("color: #888; font-size: 10px;")
        preview_group_layout.addWidget(self._legend_label)

        # --- Hidden widgets (state only, accessed via right-click context menu) ---
        # View mode
        self._full_view_radio = QRadioButton("Full")
        self._event_view_radio = QRadioButton("Event")
        self._event_view_radio.setChecked(True)
        self._view_mode_group = QButtonGroup(self)
        self._view_mode_group.addButton(self._full_view_radio, 0)
        self._view_mode_group.addButton(self._event_view_radio, 1)
        # Window size
        self._window_before_spin = QDoubleSpinBox()
        self._window_before_spin.setRange(0.5, 30.0)
        self._window_before_spin.setValue(5.0)
        self._window_after_spin = QDoubleSpinBox()
        self._window_after_spin.setRange(1.0, 60.0)
        self._window_after_spin.setValue(15.0)
        # Derivative controls
        self._show_derivative_cb = QCheckBox()
        self._show_derivative_cb.setChecked(False)
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(1, 501)
        self._smooth_spin.setSingleStep(10)
        self._smooth_spin.setValue(51)

        # Build the right-click context menu for the preview plot
        self._build_preview_context_menu()

        preview_layout.addWidget(preview_group)

        # =====================================================================
        # RIGHT SIDE: Options
        # =====================================================================
        options_container = QWidget()
        options_container.setMaximumWidth(340)
        options_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        options_layout = QVBoxLayout(options_container)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)

        # Detection Setup: Channel, Category, Method, Markers, Condition
        setup_group = QGroupBox("Detection Setup")
        setup_grid = QGridLayout(setup_group)
        setup_grid.setSpacing(6)
        setup_grid.setColumnStretch(1, 1)
        setup_grid.setColumnStretch(3, 1)

        # Row 0: Channel + Category
        setup_grid.addWidget(QLabel("Channel:"), 0, 0)
        self._channel_combo = QComboBox()
        self._channel_combo.setToolTip("Signal channel to run detection on")
        setup_grid.addWidget(self._channel_combo, 0, 1)

        setup_grid.addWidget(QLabel("Category:"), 0, 2)
        self._category_combo = QComboBox()
        self._category_combo.setToolTip("Marker category for detected events (e.g., Hargreaves, Threshold)")
        setup_grid.addWidget(self._category_combo, 0, 3)

        # Row 1: Method + Markers (Paired/Single)
        setup_grid.addWidget(QLabel("Method:"), 1, 0)
        self._method_combo = QComboBox()
        self._method_combo.setToolTip("Detection algorithm to use")
        setup_grid.addWidget(self._method_combo, 1, 1)

        markers_widget = QWidget()
        markers_layout = QHBoxLayout(markers_widget)
        markers_layout.setContentsMargins(0, 0, 0, 0)
        markers_layout.setSpacing(8)
        markers_label = QLabel("Markers:")
        markers_layout.addWidget(markers_label)
        self._paired_radio = QRadioButton("Paired")
        self._paired_radio.setToolTip("Create paired markers with start and end edges for each event")
        self._single_radio = QRadioButton("Single")
        self._single_radio.setToolTip("Create single point markers at each event onset")
        self._paired_radio.setChecked(True)
        self._marker_type_group = QButtonGroup(self)
        self._marker_type_group.addButton(self._paired_radio, 0)
        self._marker_type_group.addButton(self._single_radio, 1)
        markers_layout.addWidget(self._paired_radio)
        markers_layout.addWidget(self._single_radio)
        markers_layout.addStretch()
        setup_grid.addWidget(markers_widget, 1, 2, 1, 2)

        # Row 2: Condition
        setup_grid.addWidget(QLabel("Condition:"), 2, 0)
        self._condition_combo = QComboBox()
        self._condition_combo.setEditable(True)
        self._condition_combo.setToolTip(
            "Assign a condition to group these markers.\n\n"
            "Examples:\n"
            "• 'baseline' vs 'treatment'\n"
            "• 'iso' vs 'awake'\n"
            "• 'pre' vs 'post'\n\n"
            "Leave empty for no condition grouping.\n"
            "Type a custom condition or select from presets."
        )
        self._condition_combo.addItem("")  # Empty = no condition
        self._condition_combo.addItem("baseline")
        self._condition_combo.addItem("treatment")
        self._condition_combo.addItem("iso")
        self._condition_combo.addItem("awake")
        self._condition_combo.addItem("pre")
        self._condition_combo.addItem("post")
        setup_grid.addWidget(self._condition_combo, 2, 1)

        options_layout.addWidget(setup_group)

        # Parameters — fits content snugly, stretches when many params
        params_group = QGroupBox("Parameters")
        self._params_group = params_group
        params_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        params_layout = QVBoxLayout(params_group)
        params_layout.setContentsMargins(8, 14, 8, 6)

        self._params_container = QWidget()
        self._params_layout = QGridLayout(self._params_container)
        self._params_layout.setColumnStretch(1, 1)
        self._params_layout.setVerticalSpacing(6)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(self._params_container)

        options_layout.addWidget(params_group)

        # Scope & Time Range (merged)
        scope_time_group = QGroupBox("Scope && Time Range")
        scope_time_layout = QGridLayout(scope_time_group)
        scope_time_layout.setSpacing(6)

        # Row 0: Scope radios (horizontal)
        self._scope_current = QRadioButton("Current Sweep")
        self._scope_current.setToolTip("Detect events in current sweep only")
        self._scope_all = QRadioButton("All Sweeps")
        self._scope_all.setToolTip("Detect events across all sweeps")
        self._scope_current.setChecked(True)
        scope_time_layout.addWidget(self._scope_current, 0, 0)
        scope_time_layout.addWidget(self._scope_all, 0, 1)

        # Row 1: Time range checkbox + spinboxes
        self._time_range_enabled = QCheckBox("Limit to time range")
        self._time_range_enabled.setChecked(False)
        self._time_range_enabled.setToolTip(
            "Enable to detect events only within a specific time window.\n\n"
            "Useful for separating experimental phases:\n"
            "• Baseline vs treatment periods\n"
            "• Isoflurane vs awake periods\n"
            "• Different stimulus conditions"
        )
        self._time_range_enabled.toggled.connect(self._on_time_range_toggled)
        scope_time_layout.addWidget(self._time_range_enabled, 1, 0)

        time_range_widget = QWidget()
        time_range_hl = QHBoxLayout(time_range_widget)
        time_range_hl.setContentsMargins(0, 0, 0, 0)
        time_range_hl.setSpacing(6)

        time_range_hl.addWidget(QLabel("Start:"))
        self._time_start_spin = QDoubleSpinBox()
        self._time_start_spin.setRange(-3600.0, 36000.0)
        self._time_start_spin.setValue(0.0)
        self._time_start_spin.setSuffix(" s")
        self._time_start_spin.setDecimals(1)
        self._time_start_spin.setSingleStep(10.0)
        self._time_start_spin.setEnabled(False)
        self._time_start_spin.setToolTip("Start time for detection window (in seconds)")
        self._time_start_spin.valueChanged.connect(self._on_time_range_changed)
        time_range_hl.addWidget(self._time_start_spin)

        time_range_hl.addWidget(QLabel("End:"))
        self._time_end_spin = QDoubleSpinBox()
        self._time_end_spin.setRange(-3600.0, 36000.0)
        self._time_end_spin.setValue(300.0)
        self._time_end_spin.setSuffix(" s")
        self._time_end_spin.setDecimals(1)
        self._time_end_spin.setSingleStep(10.0)
        self._time_end_spin.setEnabled(False)
        self._time_end_spin.setToolTip("End time for detection window (in seconds)")
        self._time_end_spin.valueChanged.connect(self._on_time_range_changed)
        time_range_hl.addWidget(self._time_end_spin)

        scope_time_layout.addWidget(time_range_widget, 1, 1, 1, 2)

        options_layout.addWidget(scope_time_group)

        # Push remaining space below the option widgets
        options_layout.addStretch(1)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888;")
        options_layout.addWidget(self._status_label)

        # =====================================================================
        # RIGHT SIDE: Event List Panel
        # =====================================================================
        event_list_container = self._build_event_list_panel()

        # Add all three panels to splitter
        self._splitter.addWidget(preview_container)
        self._splitter.addWidget(options_container)
        self._splitter.addWidget(event_list_container)
        self._splitter.setSizes([400, 280, 280])
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, False)
        self._splitter.setCollapsible(2, True)

        main_layout.addWidget(self._splitter, stretch=1)

        # Button row at bottom (outside splitter, spans full width)
        button_layout = QHBoxLayout()

        self._load_existing_btn = QPushButton("Reload from Plot")
        self._load_existing_btn.setToolTip(
            "Re-scan markers from the plot and refresh the marker group dropdown.\n"
            "Useful after creating markers in another dialog or after undo/redo."
        )
        button_layout.addWidget(self._load_existing_btn)

        button_layout.addStretch()

        self._detect_btn = QPushButton("Save to Plot")
        self._detect_btn.setToolTip("Create markers for checked events in the event list")
        self._detect_btn.setDefault(True)
        button_layout.addWidget(self._detect_btn)

        self._cancel_btn = QPushButton("Cancel")
        button_layout.addWidget(self._cancel_btn)

        main_layout.addLayout(button_layout)

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

    def _set_window_icon(self) -> None:
        """Set the PhysioMetrics icon on the dialog title bar."""
        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "assets" / "plethapp_thumbnail_dark_round.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

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
            QTableWidget {
                background-color: #1e1e1e;
                alternate-background-color: #252526;
                border: 1px solid #3e3e42;
                gridline-color: #3e3e42;
                color: #d4d4d4;
            }
            QTableWidget::item {
                padding: 2px 4px;
            }
            QTableWidget::item:selected {
                background-color: #094771;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #d4d4d4;
                padding: 4px;
                border: 1px solid #3e3e42;
                font-weight: bold;
            }
            QSplitter::handle {
                background-color: #3e3e42;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #007acc;
            }
        """)

    def _build_event_list_panel(self) -> QWidget:
        """Build the event list panel (right column) with marker group dropdown and event table."""
        container = QGroupBox("Detected Events")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 14, 8, 8)
        layout.setSpacing(6)

        # Marker group dropdown
        self._marker_group_combo = QComboBox()
        self._marker_group_combo.setToolTip(
            "Select existing markers to view/edit, or choose New Detection"
        )
        self._marker_group_combo.addItem("(New Detection)", None)
        layout.addWidget(self._marker_group_combo)

        # Toolbar: Select All, Deselect All, count
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.setFixedHeight(24)
        self._select_all_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        toolbar.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("Deselect All")
        self._deselect_all_btn.setFixedHeight(24)
        self._deselect_all_btn.setStyleSheet("padding: 2px 8px; font-size: 11px;")
        toolbar.addWidget(self._deselect_all_btn)

        toolbar.addStretch()

        self._event_count_label = QLabel("0 events")
        self._event_count_label.setStyleSheet("color: #888; font-size: 11px;")
        toolbar.addWidget(self._event_count_label)

        layout.addLayout(toolbar)

        # Event table
        self._event_table = QTableWidget()
        self._event_table.setColumnCount(5)
        self._event_table.setHorizontalHeaderLabels(["#", "Start", "End", "\u0394", "Condition"])
        header = self._event_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._event_table.setColumnWidth(0, 50)   # "#" — checkbox + "  99"
        self._event_table.setColumnWidth(1, 58)   # Start — "XXXX.X"
        self._event_table.setColumnWidth(2, 58)   # End
        self._event_table.setColumnWidth(3, 42)   # Δ — "XX.X"
        self._event_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._event_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._event_table.verticalHeader().setDefaultSectionSize(25)
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.setAlternatingRowColors(True)

        layout.addWidget(self._event_table)

        return container

    def _build_preview_context_menu(self) -> None:
        """Build right-click context menu for the preview plot."""
        # Disable pyqtgraph's default context menu
        self._preview_plot.setMenuEnabled(False)
        self._preview_plot.getViewBox().setMenuEnabled(False)

        self._preview_plot.scene().sigMouseClicked.connect(self._on_plot_right_click)

    def _on_plot_right_click(self, evt) -> None:
        """Show context menu on right-click in preview plot."""
        if evt.button() != Qt.MouseButton.RightButton:
            return
        # Accept the event to prevent pyqtgraph's native context menu
        evt.accept()

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; }
            QMenu::item:selected { background-color: #094771; }
            QMenu::separator { background-color: #3e3e42; height: 1px; margin: 4px 8px; }
            QWidget { background-color: #252526; color: #d4d4d4; }
            QDoubleSpinBox, QSpinBox {
                background-color: #2d2d30; border: 1px solid #3e3e42;
                border-radius: 3px; padding: 2px 4px; color: #d4d4d4;
            }
        """)

        # Reset View
        reset_action = menu.addAction("Reset View")
        reset_action.triggered.connect(self._on_reset_view)

        menu.addSeparator()

        # View Mode
        view_menu = menu.addMenu("View Mode")
        full_action = view_menu.addAction("Full Signal")
        full_action.setCheckable(True)
        full_action.setChecked(self._full_view_radio.isChecked())
        full_action.triggered.connect(lambda: self._full_view_radio.setChecked(True))
        event_action = view_menu.addAction("Event View")
        event_action.setCheckable(True)
        event_action.setChecked(self._event_view_radio.isChecked())
        event_action.triggered.connect(lambda: self._event_view_radio.setChecked(True))

        # Window Before/After (as inline widget actions)
        menu.addSeparator()

        before_widget = QWidget()
        before_layout = QHBoxLayout(before_widget)
        before_layout.setContentsMargins(16, 2, 16, 2)
        before_label = QLabel("Before:")
        before_spin = QDoubleSpinBox()
        before_spin.setRange(0.5, 30.0)
        before_spin.setSuffix(" s")
        before_spin.setSingleStep(1.0)
        before_spin.setValue(self._window_before_spin.value())
        before_spin.valueChanged.connect(self._window_before_spin.setValue)
        before_layout.addWidget(before_label)
        before_layout.addWidget(before_spin)
        before_wa = QWidgetAction(menu)
        before_wa.setDefaultWidget(before_widget)
        menu.addAction(before_wa)

        after_widget = QWidget()
        after_layout = QHBoxLayout(after_widget)
        after_layout.setContentsMargins(16, 2, 16, 2)
        after_label = QLabel("After:")
        after_spin = QDoubleSpinBox()
        after_spin.setRange(1.0, 60.0)
        after_spin.setSuffix(" s")
        after_spin.setSingleStep(1.0)
        after_spin.setValue(self._window_after_spin.value())
        after_spin.valueChanged.connect(self._window_after_spin.setValue)
        after_layout.addWidget(after_label)
        after_layout.addWidget(after_spin)
        after_wa = QWidgetAction(menu)
        after_wa.setDefaultWidget(after_widget)
        menu.addAction(after_wa)

        menu.addSeparator()

        # Show dV/dt
        dvdt_action = menu.addAction("Show dV/dt")
        dvdt_action.setCheckable(True)
        dvdt_action.setChecked(self._show_derivative_cb.isChecked())
        dvdt_action.triggered.connect(self._show_derivative_cb.setChecked)

        # Smooth spinner (only meaningful when dV/dt is shown)
        smooth_widget = QWidget()
        smooth_layout = QHBoxLayout(smooth_widget)
        smooth_layout.setContentsMargins(16, 2, 16, 2)
        smooth_label = QLabel("Smooth:")
        smooth_spin = QSpinBox()
        smooth_spin.setRange(1, 501)
        smooth_spin.setSingleStep(10)
        smooth_spin.setValue(self._smooth_spin.value())
        smooth_spin.valueChanged.connect(self._smooth_spin.setValue)
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(smooth_spin)
        smooth_wa = QWidgetAction(menu)
        smooth_wa.setDefaultWidget(smooth_widget)
        menu.addAction(smooth_wa)

        menu.exec(evt.screenPos().toPoint() if hasattr(evt.screenPos(), 'toPoint') else evt.screenPos())

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._detect_btn.clicked.connect(self._on_detect)
        self._cancel_btn.clicked.connect(self.reject)
        self._load_existing_btn.clicked.connect(self._on_reload_from_plot)
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        self._method_combo.currentIndexChanged.connect(self._on_detector_changed)
        self._prev_btn.clicked.connect(self._on_prev_detection)
        self._next_btn.clicked.connect(self._on_next_detection)

        # Event list panel signals
        self._marker_group_combo.currentIndexChanged.connect(self._on_marker_group_changed)
        self._event_table.cellClicked.connect(self._on_event_list_clicked)
        self._select_all_btn.clicked.connect(self._on_select_all)
        self._deselect_all_btn.clicked.connect(self._on_deselect_all)

        # Condition default propagation
        self._condition_combo.currentTextChanged.connect(self._on_default_condition_changed)

        # Category change → auto-reset marker group if it no longer matches
        self._category_combo.currentIndexChanged.connect(self._on_category_changed)

        # Paired/Single toggle → update preview visual and re-run auto-detection
        self._paired_radio.toggled.connect(self._on_marker_type_changed)

        # View mode and window size (hidden widgets, triggered from context menu)
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

        # Ctrl+Z / Ctrl+Y for undo/redo of edge drags in preview
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self._on_undo_edge_drag)
        redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        redo_shortcut.activated.connect(self._on_redo_edge_drag)

        # Ctrl+Shift+D for developer tuning dialog
        dev_shortcut = QShortcut(QKeySequence("Ctrl+Shift+D"), self)
        dev_shortcut.activated.connect(self._open_dev_tools)

    def _show_initial_events(self) -> None:
        """Show pre-loaded initial events in the preview."""
        if self._preview_events:
            self._populate_event_list()
            self._update_preview_plot()
            self._show_current_detection()
            self._nav_label.setText(f"Detection 1 of {len(self._preview_events)}")
            self._status_label.setText(f"Loaded {len(self._preview_events)} existing events")
            # Re-enable auto-detection for parameter changes
            self._auto_detect_enabled = True

    def _on_reload_from_plot(self) -> None:
        """Re-scan markers from plot and refresh the marker group dropdown."""
        self._scan_marker_groups()
        self._populate_marker_group_combo()
        self._status_label.setText(f"Refreshed — {len(self._marker_groups)} marker group(s) found")

    # ------------------------------------------------------------------
    # Marker group scanning and event list management
    # ------------------------------------------------------------------

    def _scan_marker_groups(self) -> None:
        """Scan existing markers and group by (source_channel, category)."""
        self._marker_groups.clear()
        markers = self._viewmodel.get_markers_for_sweep(self._sweep_idx)
        for m in markers:
            channel = m.source_channel or "(unknown channel)"
            key = (channel, m.category)
            if key not in self._marker_groups:
                self._marker_groups[key] = []
            self._marker_groups[key].append(m)
        # Sort each group by start_time
        for key in self._marker_groups:
            self._marker_groups[key].sort(key=lambda m: m.start_time)

    def _populate_marker_group_combo(self) -> None:
        """Populate the marker group dropdown from scanned groups."""
        self._marker_group_combo.blockSignals(True)
        self._marker_group_combo.clear()
        self._marker_group_combo.addItem("(New Detection)", None)

        # Get category display names
        categories = {c.name: c.display_name for c in self._viewmodel.get_categories()}

        # Sort groups by event count descending
        sorted_groups = sorted(
            self._marker_groups.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        best_match_idx = -1
        current_channel = self._channel_combo.currentText()
        current_category = self._category_combo.currentData()

        for i, ((channel, category), marker_list) in enumerate(sorted_groups):
            cat_display = categories.get(category, category)
            label = f"{cat_display} on {channel} ({len(marker_list)} events)"
            self._marker_group_combo.addItem(label, (channel, category))

            # Check if this matches current channel + category
            if best_match_idx < 0 and channel == current_channel and category == current_category:
                best_match_idx = i + 1  # +1 because index 0 is "(New Detection)"

        # Auto-select best match, or largest group, or stay on New Detection
        if best_match_idx > 0:
            self._marker_group_combo.setCurrentIndex(best_match_idx)
        elif len(sorted_groups) > 0 and not self._initial_events:
            # Select largest group (index 1) if no better match
            self._marker_group_combo.setCurrentIndex(1)

        self._marker_group_combo.blockSignals(False)

    def _on_marker_group_changed(self, index: int) -> None:
        """Handle marker group dropdown selection."""
        data = self._marker_group_combo.currentData()

        if data is None:
            # "(New Detection)" selected — re-enable auto-detection
            self._loading_marker_group = False
            self._auto_detect_enabled = True
            self._preview_conditions.clear()
            self._preview_marker_ids.clear()
            self._preview_original_times.clear()
            self._detect_btn.setText("Save to Plot")
            self._detect_btn.setToolTip("Create markers for checked events in the event list")
            self._run_auto_detection()
            return

        channel, category = data
        self._loading_marker_group = True
        self._auto_detect_enabled = False

        # Switch channel combo to match the group's source_channel
        ch_idx = self._channel_combo.findText(channel)
        if ch_idx >= 0 and ch_idx != self._channel_combo.currentIndex():
            self._channel_combo.blockSignals(True)
            self._channel_combo.setCurrentIndex(ch_idx)
            self._channel_combo.blockSignals(False)
            self._load_signal_data()

        # Switch category combo to match
        for i in range(self._category_combo.count()):
            if self._category_combo.itemData(i) == category:
                self._category_combo.blockSignals(True)
                self._category_combo.setCurrentIndex(i)
                self._category_combo.blockSignals(False)
                break

        # Load markers into preview — track IDs for edit-mode save
        marker_list = self._marker_groups.get((channel, category), [])
        self._preview_events = []
        self._preview_conditions = []
        self._preview_marker_ids = []
        self._preview_original_times = []
        for m in marker_list:
            if m.is_paired and m.end_time is not None:
                times = (m.start_time, m.end_time)
            else:
                times = (m.start_time, m.start_time + 0.5)
            self._preview_events.append(times)
            self._preview_original_times.append(times)
            self._preview_conditions.append(m.condition)
            self._preview_marker_ids.append(m.id)

        self._current_preview_idx = 0
        self._edge_drag_history.clear()
        self._edge_drag_redo.clear()

        self._populate_event_list()
        self._update_preview_plot()
        if self._preview_events:
            self._show_current_detection()
            self._nav_label.setText(f"Detection 1 of {len(self._preview_events)}")
        self._status_label.setText(f"Loaded {len(self._preview_events)} existing markers — uncheck to remove, drag edges to adjust")
        self._detect_btn.setText("Update Plot")
        self._detect_btn.setToolTip("Sync plot markers: remove unchecked, update moved edges, keep the rest")
        self._loading_marker_group = False

    def _populate_event_list(self) -> None:
        """Populate the event table from _preview_events."""
        self._event_table.setRowCount(0)
        self._event_checkboxes.clear()
        self._event_condition_combos.clear()

        if not self._preview_events:
            self._event_count_label.setText("0 events")
            return

        default_condition = self._condition_combo.currentText().strip()

        self._event_table.setRowCount(len(self._preview_events))

        for i, (start, end) in enumerate(self._preview_events):
            # Column 0: Checkbox with event number
            cb = QCheckBox(f"  {i + 1}")
            cb.setChecked(True)
            self._event_checkboxes.append(cb)
            self._event_table.setCellWidget(i, 0, cb)

            # Column 1: Start time (display time with offset)
            display_start = start - self._time_offset
            item_start = QTableWidgetItem(f"{display_start:.1f}")
            item_start.setFlags(item_start.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._event_table.setItem(i, 1, item_start)

            # Column 2: End time
            display_end = end - self._time_offset
            item_end = QTableWidgetItem(f"{display_end:.1f}")
            item_end.setFlags(item_end.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._event_table.setItem(i, 2, item_end)

            # Column 3: Duration
            duration = end - start
            item_dur = QTableWidgetItem(f"{duration:.1f}")
            item_dur.setFlags(item_dur.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._event_table.setItem(i, 3, item_dur)

            # Column 4: Condition combo
            cond_combo = QComboBox()
            cond_combo.setEditable(True)
            for preset in CONDITION_PRESETS:
                cond_combo.addItem(preset if preset else "(none)")
            # Set condition: per-event condition from loaded markers, or default
            per_event_cond = self._preview_conditions[i] if i < len(self._preview_conditions) else None
            if per_event_cond:
                idx = cond_combo.findText(per_event_cond)
                if idx >= 0:
                    cond_combo.setCurrentIndex(idx)
                else:
                    cond_combo.setCurrentText(per_event_cond)
            elif default_condition:
                idx = cond_combo.findText(default_condition)
                if idx >= 0:
                    cond_combo.setCurrentIndex(idx)
                else:
                    cond_combo.setCurrentText(default_condition)
            self._event_condition_combos.append(cond_combo)
            self._event_table.setCellWidget(i, 4, cond_combo)

        self._event_count_label.setText(f"{len(self._preview_events)} events")

        # Highlight current preview row
        if 0 <= self._current_preview_idx < len(self._preview_events):
            self._event_table.selectRow(self._current_preview_idx)

    def _on_event_list_clicked(self, row: int, col: int) -> None:
        """Handle click on event list row — navigate preview to that event."""
        if 0 <= row < len(self._preview_events):
            self._current_preview_idx = row
            self._update_preview_plot()
            self._show_current_detection()

    def _on_select_all(self) -> None:
        """Check all events in the event list."""
        for cb in self._event_checkboxes:
            cb.setChecked(True)

    def _on_deselect_all(self) -> None:
        """Uncheck all events in the event list."""
        for cb in self._event_checkboxes:
            cb.setChecked(False)

    def _on_default_condition_changed(self, text: str) -> None:
        """Propagate default condition change to event list combos that still have the old default."""
        old_default = self._current_default_condition
        new_default = text.strip()
        self._current_default_condition = new_default

        for combo in self._event_condition_combos:
            current = combo.currentText().strip()
            # Update if combo still has old default or is empty/(none)
            if current == old_default or current in ("", "(none)"):
                if new_default:
                    idx = combo.findText(new_default)
                    if idx >= 0:
                        combo.setCurrentIndex(idx)
                    else:
                        combo.setCurrentText(new_default)
                else:
                    combo.setCurrentIndex(0)  # "(none)"

    def _on_category_changed(self) -> None:
        """Handle category combo change — reset marker group if it no longer matches."""
        current_category = self._category_combo.currentData()
        group_data = self._marker_group_combo.currentData()

        if group_data is not None:
            # Currently viewing an existing marker group — check if category still matches
            _channel, group_category = group_data
            if group_category != current_category:
                # Mismatch → switch to "(New Detection)" and re-run auto-detection
                self._marker_group_combo.blockSignals(True)
                self._marker_group_combo.setCurrentIndex(0)  # "(New Detection)"
                self._marker_group_combo.blockSignals(False)
                self._auto_detect_enabled = True
                self._preview_conditions.clear()
                self._schedule_auto_detection()

    def _on_marker_type_changed(self, checked: bool) -> None:
        """Handle Paired/Single toggle — update preview visualization."""
        if not checked:
            return  # Only handle the "checked" signal, not the unchecked
        # Refresh the preview to show single line vs paired region
        if self._preview_events:
            self._show_current_detection()

    def _on_hot_reload_requested(self) -> None:
        """Handle Ctrl+R - close dialog to allow hot reload."""
        self.reject()  # Close dialog, main window shortcut will fire

    def _on_view_mode_changed(self) -> None:
        """Handle view mode toggle."""
        if self._full_view_radio.isChecked():
            self._show_full_view()
        elif self._preview_events:
            self._show_current_detection()

    def _on_reset_view(self) -> None:
        """Reset view to default range (current event or full signal)."""
        if self._full_view_radio.isChecked():
            self._show_full_view()
        elif self._preview_events:
            self._show_current_detection()
        else:
            self._show_full_view()

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
        """Handle derivative plot visibility toggle (preserves current X range)."""
        self._smooth_spin.setEnabled(visible)
        # Save current X range before updating
        x_range = self._preview_plot.getViewBox().viewRange()[0]
        self._update_derivative_plot()
        # Restore X range so toggling dV/dt doesn't jump the view
        self._preview_plot.setXRange(*x_range, padding=0)

    def _on_smooth_changed(self) -> None:
        """Handle smoothing value change - recalculate derivative (preserves X range)."""
        x_range = self._preview_plot.getViewBox().viewRange()[0]
        self._update_derivative_plot()
        self._preview_plot.setXRange(*x_range, padding=0)

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

        show = self._show_derivative_cb.isChecked()
        self._preview_plot.showAxis('right', show)

        if self._signal_data is None or not show:
            return

        time, signal, sample_rate = self._signal_data
        display_time = time - self._time_offset

        # Calculate smoothed derivative
        derivative = self._calculate_derivative(signal, time)

        # Create plot data item for derivative
        pen = pg.mkPen(color='#ff7043', width=1)  # Orange-red color
        derivative_curve = pg.PlotDataItem(display_time, derivative, pen=pen)
        derivative_curve.setDownsampling(auto=True, method='peak')
        derivative_curve.setClipToView(True)
        self._derivative_viewbox.addItem(derivative_curve)

        # Add zero line for reference
        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen('#555555', width=1, style=Qt.PenStyle.DotLine)
        )
        self._derivative_viewbox.addItem(zero_line)

        # Auto-scale derivative Y-axis only (not X)
        self._derivative_viewbox.enableAutoRange(axis=pg.ViewBox.YAxis)

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
        self._auto_set_threshold()
        self._update_preview_plot()
        self._emit_threshold_update()

    def _auto_set_threshold(self) -> None:
        """Auto-set threshold to a sensible value based on the current signal.

        Uses the signal midpoint as a starting point. This is called when
        the channel changes so the threshold isn't stuck at a value from a
        completely different signal range.
        """
        if self._signal_data is None or self._current_detector is None:
            return

        # Only auto-set if the detector has a threshold parameter
        if self._current_detector.get_param('threshold') is None:
            return

        time, signal, sample_rate = self._signal_data
        sig_min = float(np.min(signal))
        sig_max = float(np.max(signal))
        sig_range = sig_max - sig_min
        # Start at the midpoint of the signal range
        new_threshold = (sig_min + sig_max) / 2.0

        # Update detector
        self._current_detector.set_param('threshold', new_threshold)

        # Update spinbox widget — also expand its range to cover this signal
        if 'threshold' in self._param_widgets:
            widget = self._param_widgets['threshold']
            widget.blockSignals(True)
            # Expand spinbox range to cover signal with headroom
            widget.setMinimum(sig_min - sig_range)
            widget.setMaximum(sig_max + sig_range)
            # Set step size relative to signal range (1% of range)
            widget.setSingleStep(max(0.001, sig_range * 0.01))
            widget.setValue(new_threshold)
            widget.blockSignals(False)

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

        # Plot signal - match main plot trace color
        # Use auto-downsampling with peak mode to avoid visual ringing artifacts
        # from full-resolution filtered data (same approach as main plot)
        pen = pg.mkPen(color='#d4d4d4', width=1)
        self._signal_plot_item = self._preview_plot.plot(
            display_time, signal, pen=pen,
        )
        self._signal_plot_item.setDownsampling(auto=True, method='peak')
        self._signal_plot_item.setClipToView(True)

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
            labelOpts={
                'position': 0.80,
                'color': (255, 152, 0),
                'fill': (30, 30, 30, 200),
                'anchors': [(1, 1), (1, 1)],  # Right-justify the label
            }
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

    def _on_single_onset_dragged(self, line: pg.InfiniteLine) -> None:
        """Handle single-mode onset line drag — update stored start time."""
        if not self._preview_events:
            return
        idx = self._current_preview_idx
        if 0 <= idx < len(self._preview_events):
            display_pos = line.value()
            new_start = display_pos + self._time_offset
            _old_start, old_end = self._preview_events[idx]
            self._preview_events[idx] = (new_start, old_end)

    def _on_event_region_dragged(self, region: pg.LinearRegionItem) -> None:
        """Handle event region edge drag — update stored times and draw intersection lines."""
        if not self._preview_events:
            return

        idx = self._current_preview_idx
        if 0 <= idx < len(self._preview_events):
            # Capture pre-drag state on first move (before updating)
            if not hasattr(self, '_edge_drag_pre_drag_state') or self._edge_drag_pre_drag_state is None:
                self._edge_drag_pre_drag_state = self._preview_events[idx]

        display_start, display_end = region.getRegion()
        # Convert display times back to signal times
        new_start = display_start + self._time_offset
        new_end = display_end + self._time_offset
        if 0 <= idx < len(self._preview_events):
            self._preview_events[idx] = (new_start, new_end)

        # Draw intersection lines at both edges
        self._draw_intersection_lines(display_start, display_end)

    def _on_event_region_drag_finished(self, region: pg.LinearRegionItem) -> None:
        """Record undo entry and clear intersection lines when edge drag finishes."""
        self._clear_intersection_lines()

        # Push to undo history (record current state AFTER drag)
        idx = self._current_preview_idx
        if 0 <= idx < len(self._preview_events):
            # The _edge_drag_pre_drag_state was saved before the drag started
            if hasattr(self, '_edge_drag_pre_drag_state') and self._edge_drag_pre_drag_state is not None:
                self._edge_drag_history.append((idx, self._edge_drag_pre_drag_state))
                # Cap at 20 entries
                if len(self._edge_drag_history) > 20:
                    self._edge_drag_history.pop(0)
                # Clear redo stack on new action
                self._edge_drag_redo.clear()
                self._edge_drag_pre_drag_state = None

    def _on_undo_edge_drag(self) -> None:
        """Undo the last edge drag adjustment."""
        if not self._edge_drag_history:
            return
        idx, old_times = self._edge_drag_history.pop()
        if 0 <= idx < len(self._preview_events):
            # Save current state for redo
            self._edge_drag_redo.append((idx, self._preview_events[idx]))
            # Restore old state
            self._preview_events[idx] = old_times
            # If we're viewing this event, refresh the region display
            if idx == self._current_preview_idx:
                self._show_current_detection()

    def _on_redo_edge_drag(self) -> None:
        """Redo the last undone edge drag adjustment."""
        if not self._edge_drag_redo:
            return
        idx, new_times = self._edge_drag_redo.pop()
        if 0 <= idx < len(self._preview_events):
            # Save current state for undo
            self._edge_drag_history.append((idx, self._preview_events[idx]))
            # Apply redo state
            self._preview_events[idx] = new_times
            # If we're viewing this event, refresh the region display
            if idx == self._current_preview_idx:
                self._show_current_detection()

    def _draw_intersection_lines(self, display_start: float, display_end: float) -> None:
        """Draw horizontal dashed lines where region edges intersect the signal."""
        self._clear_intersection_lines()

        if self._signal_data is None:
            return

        time, signal, sample_rate = self._signal_data
        display_time = time - self._time_offset

        edge_color = QColor(0, 200, 100, 200)

        for x_pos in (display_start, display_end):
            try:
                y_value = float(np.interp(x_pos, display_time, signal))
                if np.isnan(y_value):
                    continue
            except Exception:
                continue

            # Horizontal dashed line at intersection Y
            h_pen = QPen(edge_color)
            h_pen.setWidth(0)  # Cosmetic pen
            h_pen.setStyle(Qt.PenStyle.DashLine)
            h_line = pg.InfiniteLine(pos=y_value, angle=0, pen=h_pen, movable=False)
            h_line.setZValue(1050)
            self._preview_plot.addItem(h_line, ignoreBounds=True)
            self._intersection_items.append(h_line)

            # Small circle at intersection point
            circle = pg.ScatterPlotItem(
                [x_pos], [y_value], size=8,
                pen=pg.mkPen(edge_color, width=1),
                brush=pg.mkBrush(edge_color.lighter(150)),
                symbol='o',
            )
            circle.setZValue(1060)
            self._preview_plot.addItem(circle, ignoreBounds=True)
            self._intersection_items.append(circle)

    def _clear_intersection_lines(self) -> None:
        """Remove all intersection line items from the preview plot."""
        for item in self._intersection_items:
            try:
                self._preview_plot.removeItem(item)
            except Exception:
                pass
        self._intersection_items.clear()

    def _show_current_detection(self) -> None:
        """Show the current detection in the preview plot."""
        if not self._preview_events or self._signal_data is None:
            return

        # Reset pre-drag state when showing a new/refreshed detection
        self._edge_drag_pre_drag_state = None

        # Clamp index
        self._current_preview_idx = max(0, min(self._current_preview_idx, len(self._preview_events) - 1))

        start_time, end_time = self._preview_events[self._current_preview_idx]
        time, signal, sample_rate = self._signal_data

        # Convert to display time (with offset applied)
        display_start = start_time - self._time_offset
        display_end = end_time - self._time_offset
        display_time = time - self._time_offset

        # Clear previous visualization items
        self._clear_intersection_lines()
        single_line = getattr(self, '_single_onset_line', None)
        for item in [self._event_region, self._lookback_region, self._baseline_region, self._onset_line, single_line]:
            if item is not None:
                try:
                    self._preview_plot.removeItem(item)
                except:
                    pass
        self._single_onset_line = None

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

        # Visualization depends on Single vs Paired mode
        is_single_mode = self._single_radio.isChecked()
        edge_color = QColor(0, 200, 100, 200)

        if is_single_mode:
            # Single mode: prominent vertical line at onset, faint region for context
            self._event_region = pg.LinearRegionItem(
                values=[display_start, display_end],
                brush=pg.mkBrush(0, 200, 100, 20),  # Very faint fill
                pen=pg.mkPen(0, 200, 100, 60),  # Dim edges
                movable=False,
                swapMode='none',
            )
            for edge_line in self._event_region.lines:
                edge_line.setMovable(False)
            self._preview_plot.addItem(self._event_region)

            # Bold onset line — this is what becomes the single marker
            self._single_onset_line = pg.InfiniteLine(
                pos=display_start, angle=90,
                pen=pg.mkPen(0, 255, 100, 255, width=2),
                movable=True,
                label='Onset',
                labelOpts={'position': 0.9, 'color': (0, 255, 100), 'fill': (30, 30, 30, 200)},
            )
            self._single_onset_line.sigPositionChanged.connect(self._on_single_onset_dragged)
            self._preview_plot.addItem(self._single_onset_line)
        else:
            # Paired mode: full draggable region
            hover_color = edge_color.lighter(150)
            edge_pen = QPen(edge_color)
            edge_pen.setWidth(0)  # Cosmetic pen (1px regardless of zoom)
            self._event_region = pg.LinearRegionItem(
                values=[display_start, display_end],
                brush=pg.mkBrush(0, 200, 100, 50),
                pen=edge_pen,
                movable=False,  # Don't allow whole-region drag, only edges
                swapMode='none',
            )
            for edge_line in self._event_region.lines:
                edge_line.setMovable(True)
                line_pen = QPen(edge_color)
                line_pen.setWidth(0)
                edge_line.setPen(line_pen)
                hover_pen = QPen(hover_color)
                hover_pen.setWidth(0)
                edge_line.setHoverPen(hover_pen)
                edge_line.setBounds([None, None])
                edge_line._maxMarkerSize = 20
                edge_line.setCursor(Qt.CursorShape.SizeHorCursor)
            self._event_region.sigRegionChanged.connect(self._on_event_region_dragged)
            self._event_region.sigRegionChangeFinished.connect(self._on_event_region_drag_finished)
            self._preview_plot.addItem(self._event_region)

        # For Hargreaves, show lookback window and baseline region
        if self._current_detector and 'Hargreaves' in self._current_detector.name:
            self._show_hargreaves_visualization(display_start, display_end, display_time, signal)

        # Update navigation label
        self._nav_label.setText(f"Detection {self._current_preview_idx + 1} of {len(self._preview_events)}")

        # Update button states
        self._prev_btn.setEnabled(self._current_preview_idx > 0)
        self._next_btn.setEnabled(self._current_preview_idx < len(self._preview_events) - 1)

        # Sync event list row highlight
        if self._event_table.rowCount() > 0 and self._current_preview_idx < self._event_table.rowCount():
            self._event_table.selectRow(self._current_preview_idx)
            self._event_table.scrollTo(
                self._event_table.model().index(self._current_preview_idx, 0)
            )

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
        """Rebuild parameter widgets for current detector in 2-column layout."""
        # Clear existing widgets
        for widget in self._param_widgets.values():
            widget.deleteLater()
        self._param_widgets.clear()

        # Clear layout
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Reset column stretches for 4-column grid
        self._params_layout.setColumnStretch(0, 0)  # Label left
        self._params_layout.setColumnStretch(1, 1)  # Widget left
        self._params_layout.setColumnStretch(2, 0)  # Label right
        self._params_layout.setColumnStretch(3, 1)  # Widget right
        self._params_layout.setHorizontalSpacing(10)

        if not self._current_detector:
            return

        # Create widgets for each parameter in 2-column layout
        specs = self._current_detector.get_param_specs()
        n = len(specs)
        rows_left = (n + 1) // 2  # Left column gets extra if odd

        for i, spec in enumerate(specs):
            # Fill left column first, then right
            if i < rows_left:
                row = i
                col_label, col_widget = 0, 1
            else:
                row = i - rows_left
                col_label, col_widget = 2, 3

            # Label
            label_text = spec.label
            if spec.unit:
                label_text += f" ({spec.unit})"
            label = QLabel(label_text + ":")
            if spec.tooltip:
                label.setToolTip(spec.tooltip)
            self._params_layout.addWidget(label, row, col_label)

            # Widget based on type
            widget = self._create_param_widget(spec)
            self._params_layout.addWidget(widget, row, col_widget)
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

        # Don't run auto-detection while loading a marker group
        if self._loading_marker_group:
            return

        # Clear edge drag undo/redo history on new detection
        self._edge_drag_history.clear()
        self._edge_drag_redo.clear()
        self._edge_drag_pre_drag_state = None

        # Clear per-event state for fresh detection
        self._preview_conditions.clear()
        self._preview_marker_ids.clear()
        self._preview_original_times.clear()

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

            # Populate event list table
            self._populate_event_list()

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

    def _on_detect(self) -> None:
        """Save events to plot — creates new markers or syncs existing ones.

        Edit mode (loaded existing markers): deletes unchecked, updates moved, keeps rest.
        Create mode (fresh detection): creates checked events as new markers.
        """
        if not self._preview_events:
            self._status_label.setText("No events to save")
            return

        category = self._category_combo.currentData()
        is_paired = self._paired_radio.isChecked()
        source_channel = self._channel_combo.currentText()

        if not category:
            QMessageBox.warning(
                self, "Select Category",
                "Please select a category for the detected events.",
            )
            return

        cat = self._viewmodel.service.registry.get(category)
        label = cat.labels[0] if cat and cat.labels else 'default'

        # Determine if we're in edit mode (loaded existing markers with IDs)
        editing_existing = bool(self._preview_marker_ids) and len(self._preview_marker_ids) == len(self._preview_events)

        if editing_existing:
            self._save_edit_mode(category, label, is_paired, source_channel)
        else:
            self._save_create_mode(category, label, is_paired, source_channel)

    def _save_edit_mode(self, category: str, label: str, is_paired: bool, source_channel: str) -> None:
        """Save in edit mode — sync plot markers with the event list state."""
        deleted = 0
        updated = 0
        kept = 0

        for i in range(len(self._preview_events)):
            marker_id = self._preview_marker_ids[i] if i < len(self._preview_marker_ids) else None
            is_checked = self._event_checkboxes[i].isChecked() if i < len(self._event_checkboxes) else True
            start, end = self._preview_events[i]

            # Get per-event condition
            condition = ""
            if i < len(self._event_condition_combos):
                condition = self._event_condition_combos[i].currentText().strip()
                if condition == "(none)":
                    condition = ""

            if marker_id:
                if not is_checked:
                    # Unchecked existing marker → delete from plot
                    self._viewmodel.delete_marker(marker_id)
                    deleted += 1
                else:
                    # Checked existing marker — check if times changed (edge drag)
                    original = self._preview_original_times[i] if i < len(self._preview_original_times) else None
                    if original and (abs(start - original[0]) > 0.001 or abs(end - original[1]) > 0.001):
                        # Times changed → move marker
                        if is_paired:
                            self._viewmodel.service.move_marker(marker_id, start, end)
                        else:
                            self._viewmodel.service.move_marker(marker_id, start)
                        updated += 1
                    else:
                        kept += 1

                    # Update condition if changed
                    self._viewmodel.update_marker(marker_id, condition=condition or None)
            else:
                # No marker ID but checked → new event, create it
                if is_checked:
                    self._create_single_marker(start, end, category, label, is_paired, source_channel, condition)
                    updated += 1

        parts = []
        if deleted:
            parts.append(f"{deleted} removed")
        if updated:
            parts.append(f"{updated} updated")
        if kept:
            parts.append(f"{kept} kept")
        status = ", ".join(parts) if parts else "No changes"

        self._status_label.setText(status)
        self.detection_complete.emit(deleted + updated)
        self.accept()

    def _save_create_mode(self, category: str, label: str, is_paired: bool, source_channel: str) -> None:
        """Save in create mode — create new markers for all checked events."""
        # Collect checked events
        events_to_create = []
        for i, cb in enumerate(self._event_checkboxes):
            if cb.isChecked() and i < len(self._preview_events):
                start, end = self._preview_events[i]
                condition = self._event_condition_combos[i].currentText().strip()
                if condition == "(none)":
                    condition = ""
                events_to_create.append((start, end, condition or None))

        if not events_to_create:
            self._status_label.setText("No events selected")
            return

        # Warn if many events
        if len(events_to_create) > 100:
            reply = QMessageBox.warning(
                self, "Many Events Selected",
                f"{len(events_to_create)} events selected.\n\n"
                f"Create all {len(events_to_create)} markers?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        created = 0
        for start_time, end_time, event_condition in events_to_create:
            if self._create_single_marker(start_time, end_time, category, label, is_paired, source_channel, event_condition):
                created += 1

        self._status_label.setText(f"Created {created} markers")
        self.detection_complete.emit(created)
        self.accept()

    def _create_single_marker(
        self, start: float, end: float, category: str, label: str,
        is_paired: bool, source_channel: str, condition: Optional[str],
    ) -> bool:
        """Create one marker on the plot. Returns True on success."""
        if is_paired:
            marker = self._viewmodel.add_paired_marker(
                start_time=start, end_time=end,
                sweep_idx=self._sweep_idx, category=category, label=label,
            )
        else:
            marker = self._viewmodel.add_single_marker(
                time=start, sweep_idx=self._sweep_idx,
            )
            if marker:
                self._viewmodel.update_marker(marker.id, category=category, label=label)

        if marker:
            marker.source_channel = source_channel
            if condition:
                self._viewmodel.update_marker(marker.id, condition=condition)
            return True
        return False

    def get_last_result(self) -> Optional[DetectionResult]:
        """Get the last detection result."""
        return self._last_result

    def _open_dev_tools(self) -> None:
        """Open the developer tuning dialog (Ctrl+Shift+D)."""
        if hasattr(self, '_dev_tools') and self._dev_tools is not None:
            self._dev_tools.raise_()
            self._dev_tools.activateWindow()
            return
        self._dev_tools = DetectionDevToolsDialog(self)
        self._dev_tools.show()


class DetectionDevToolsDialog(QDialog):
    """
    Developer tuning dialog for live-adjusting visual parameters
    of the MarkerDetectionDialog. Opens via Ctrl+Shift+D.
    """

    def __init__(self, detection_dialog: MarkerDetectionDialog, parent=None):
        super().__init__(parent or detection_dialog)
        self._dd = detection_dialog
        self.setWindowTitle("Dev Tools — Detection Dialog")
        self.setMinimumWidth(320)
        self.resize(340, 560)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        # Dark theme
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #d4d4d4; }
            QGroupBox {
                font-weight: bold; border: 1px solid #3e3e42; border-radius: 4px;
                margin-top: 12px; padding-top: 12px; background-color: #252526; color: #d4d4d4;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #d4d4d4; }
            QLabel { color: #d4d4d4; background-color: transparent; }
            QSpinBox {
                background-color: #2d2d30; border: 1px solid #3e3e42; border-radius: 3px;
                padding: 2px 4px; color: #d4d4d4; min-height: 18px;
            }
            QSpinBox::up-button, QSpinBox::down-button { background-color: #3e3e42; border: none; width: 14px; }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover { background-color: #505050; }
            QPushButton {
                background-color: #0e639c; border: none; border-radius: 3px;
                padding: 4px 12px; color: white;
            }
            QPushButton:hover { background-color: #1177bb; }
        """)

        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, byref(value), sizeof(value))
            except Exception:
                pass

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(6)

        # --- Splitter Sizes ---
        splitter_group = QGroupBox("Splitter Sizes")
        sg = QGridLayout(splitter_group)
        sg.setSpacing(4)
        sizes = self._dd._splitter.sizes()
        self._splitter_spins = []
        for i, (label, val) in enumerate([("Preview:", sizes[0] if len(sizes) > 0 else 400),
                                           ("Options:", sizes[1] if len(sizes) > 1 else 280),
                                           ("Event List:", sizes[2] if len(sizes) > 2 else 280)]):
            sg.addWidget(QLabel(label), i, 0)
            spin = QSpinBox()
            spin.setRange(50, 1200)
            spin.setValue(val)
            spin.setSingleStep(10)
            spin.valueChanged.connect(self._apply_splitter_sizes)
            sg.addWidget(spin, i, 1)
            self._splitter_spins.append(spin)
        scroll_layout.addWidget(splitter_group)

        # --- Options Column ---
        opts_group = QGroupBox("Options Column")
        og = QGridLayout(opts_group)
        og.setSpacing(4)
        og.addWidget(QLabel("Max Width:"), 0, 0)
        self._opts_max_w = QSpinBox()
        self._opts_max_w.setRange(150, 600)
        self._opts_max_w.setValue(self._dd._splitter.widget(1).maximumWidth())
        self._opts_max_w.setSingleStep(10)
        self._opts_max_w.valueChanged.connect(lambda v: self._dd._splitter.widget(1).setMaximumWidth(v))
        og.addWidget(self._opts_max_w, 0, 1)
        scroll_layout.addWidget(opts_group)

        # --- Table Column Widths ---
        table_group = QGroupBox("Event Table")
        tg = QGridLayout(table_group)
        tg.setSpacing(4)
        col_labels = ["# col:", "Start col:", "End col:", "\u0394 col:"]
        col_indices = [0, 1, 2, 3]
        self._col_width_spins = []
        for row, (label, col_idx) in enumerate(zip(col_labels, col_indices)):
            tg.addWidget(QLabel(label), row, 0)
            spin = QSpinBox()
            spin.setRange(25, 200)
            spin.setValue(self._dd._event_table.columnWidth(col_idx))
            spin.setSingleStep(2)
            spin.valueChanged.connect(lambda v, c=col_idx: self._dd._event_table.setColumnWidth(c, v))
            tg.addWidget(spin, row, 1)
            self._col_width_spins.append(spin)

        tg.addWidget(QLabel("Row Height:"), len(col_labels), 0)
        self._row_height_spin = QSpinBox()
        self._row_height_spin.setRange(16, 50)
        self._row_height_spin.setValue(self._dd._event_table.verticalHeader().defaultSectionSize())
        self._row_height_spin.setSingleStep(1)
        self._row_height_spin.valueChanged.connect(
            lambda v: self._dd._event_table.verticalHeader().setDefaultSectionSize(v)
        )
        tg.addWidget(self._row_height_spin, len(col_labels), 1)
        scroll_layout.addWidget(table_group)

        # --- Dialog Size ---
        dialog_group = QGroupBox("Dialog Size")
        dg = QGridLayout(dialog_group)
        dg.setSpacing(4)
        dg.addWidget(QLabel("Width:"), 0, 0)
        self._dlg_w = QSpinBox()
        self._dlg_w.setRange(600, 2400)
        self._dlg_w.setValue(self._dd.width())
        self._dlg_w.setSingleStep(20)
        self._dlg_w.valueChanged.connect(lambda v: self._dd.resize(v, self._dd.height()))
        dg.addWidget(self._dlg_w, 0, 1)

        dg.addWidget(QLabel("Height:"), 1, 0)
        self._dlg_h = QSpinBox()
        self._dlg_h.setRange(300, 1200)
        self._dlg_h.setValue(self._dd.height())
        self._dlg_h.setSingleStep(20)
        self._dlg_h.valueChanged.connect(lambda v: self._dd.resize(self._dd.width(), v))
        dg.addWidget(self._dlg_h, 1, 1)

        dg.addWidget(QLabel("Min Width:"), 2, 0)
        self._dlg_min_w = QSpinBox()
        self._dlg_min_w.setRange(400, 2000)
        self._dlg_min_w.setValue(self._dd.minimumWidth())
        self._dlg_min_w.setSingleStep(20)
        self._dlg_min_w.valueChanged.connect(lambda v: self._dd.setMinimumWidth(v))
        dg.addWidget(self._dlg_min_w, 2, 1)
        scroll_layout.addWidget(dialog_group)

        # --- Spacing / Margins ---
        spacing_group = QGroupBox("Spacing / Margins")
        spg = QGridLayout(spacing_group)
        spg.setSpacing(4)

        spg.addWidget(QLabel("Main Margin:"), 0, 0)
        self._main_margin = QSpinBox()
        self._main_margin.setRange(0, 30)
        margins = self._dd.layout().contentsMargins()
        self._main_margin.setValue(margins.left())
        self._main_margin.setSingleStep(1)
        self._main_margin.valueChanged.connect(
            lambda v: self._dd.layout().setContentsMargins(v, v, v, v)
        )
        spg.addWidget(self._main_margin, 0, 1)

        spg.addWidget(QLabel("Main Spacing:"), 1, 0)
        self._main_spacing = QSpinBox()
        self._main_spacing.setRange(0, 20)
        self._main_spacing.setValue(self._dd.layout().spacing())
        self._main_spacing.setSingleStep(1)
        self._main_spacing.valueChanged.connect(lambda v: self._dd.layout().setSpacing(v))
        spg.addWidget(self._main_spacing, 1, 1)

        spg.addWidget(QLabel("Options Spacing:"), 2, 0)
        self._opts_spacing = QSpinBox()
        self._opts_spacing.setRange(0, 20)
        opts_widget = self._dd._splitter.widget(1)
        self._opts_spacing.setValue(opts_widget.layout().spacing() if opts_widget.layout() else 8)
        self._opts_spacing.setSingleStep(1)
        self._opts_spacing.valueChanged.connect(
            lambda v: opts_widget.layout().setSpacing(v) if opts_widget.layout() else None
        )
        spg.addWidget(self._opts_spacing, 2, 1)
        scroll_layout.addWidget(spacing_group)

        # --- Preview Window ---
        preview_group = QGroupBox("Preview Window")
        pvg = QGridLayout(preview_group)
        pvg.setSpacing(4)

        pvg.addWidget(QLabel("Before (s):"), 0, 0)
        self._before_spin = QSpinBox()
        self._before_spin.setRange(1, 60)
        self._before_spin.setValue(int(self._dd._window_before_spin.value()))
        self._before_spin.valueChanged.connect(lambda v: self._dd._window_before_spin.setValue(float(v)))
        pvg.addWidget(self._before_spin, 0, 1)

        pvg.addWidget(QLabel("After (s):"), 1, 0)
        self._after_spin = QSpinBox()
        self._after_spin.setRange(1, 120)
        self._after_spin.setValue(int(self._dd._window_after_spin.value()))
        self._after_spin.valueChanged.connect(lambda v: self._dd._window_after_spin.setValue(float(v)))
        pvg.addWidget(self._after_spin, 1, 1)
        scroll_layout.addWidget(preview_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Print values button
        print_btn = QPushButton("Print Current Values")
        print_btn.clicked.connect(self._print_values)
        layout.addWidget(print_btn)

    def _apply_splitter_sizes(self) -> None:
        """Apply splitter sizes from spinboxes."""
        sizes = [s.value() for s in self._splitter_spins]
        self._dd._splitter.setSizes(sizes)

    def _print_values(self) -> None:
        """Print all current values to console for easy copy-paste into code."""
        dd = self._dd
        sizes = dd._splitter.sizes()
        print("\n" + "=" * 50)
        print("Detection Dialog Dev Tool — Current Values")
        print("=" * 50)
        print(f"Dialog size: resize({dd.width()}, {dd.height()})")
        print(f"Dialog min width: setMinimumWidth({dd.minimumWidth()})")
        print(f"Splitter sizes: setSizes({sizes})")
        print(f"Options max width: setMaximumWidth({dd._splitter.widget(1).maximumWidth()})")
        print(f"Table column widths: [0]={dd._event_table.columnWidth(0)}, "
              f"[1]={dd._event_table.columnWidth(1)}, "
              f"[2]={dd._event_table.columnWidth(2)}, "
              f"[3]={dd._event_table.columnWidth(3)}")
        print(f"Table row height: {dd._event_table.verticalHeader().defaultSectionSize()}")
        margins = dd.layout().contentsMargins()
        print(f"Main margins: ({margins.left()}, {margins.top()}, {margins.right()}, {margins.bottom()})")
        print(f"Main spacing: {dd.layout().spacing()}")
        print(f"Preview before/after: {dd._window_before_spin.value():.1f}s / {dd._window_after_spin.value():.1f}s")
        print("=" * 50)

    def closeEvent(self, event):
        """Clear reference on parent when closed."""
        self._dd._dev_tools = None
        super().closeEvent(event)
