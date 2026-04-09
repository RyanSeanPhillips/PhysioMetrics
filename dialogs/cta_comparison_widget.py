"""
CTA Comparison Widget.

Standalone QWidget extracted from PhotometryCTADialog for hosting in a QTabWidget.
Provides all CTA configuration, generation, preview, and export functionality
with support for both event marker and breath event trigger sources.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QProgressBar, QFileDialog,
    QListWidget, QListWidgetItem, QSplitter, QTabWidget,
    QMessageBox, QSizePolicy, QScrollArea, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QRadioButton, QStackedWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QBrush

from viewmodels.cta_viewmodel import CTAViewModel
from core.domain.events import EventMarker

# Default color palette for metrics
_METRIC_COLORS = [
    '#4488ff', '#44cc66', '#ff6644', '#cc44ff', '#ffcc44',
    '#44cccc', '#ff44aa', '#88cc44', '#ff8844', '#4466ff',
]


class CTAComparisonWidget(QWidget):
    """
    Widget for generating and previewing photometry CTAs.

    Allows users to:
    - Select trigger source (event markers or breath events)
    - Select which event marker categories to include
    - Configure time windows (before/after event)
    - Select which metrics to include (with per-metric colors)
    - Preview CTA plots
    - Export via matplotlib toolbar save icon or CSV
    """

    # Signals
    name_changed = pyqtSignal(str)
    cta_generated = pyqtSignal()

    def __init__(
        self,
        parent=None,
        viewmodel: Optional[CTAViewModel] = None,
        markers: Optional[List[EventMarker]] = None,
        signals: Optional[Dict[str, np.ndarray]] = None,
        time_array: Optional[np.ndarray] = None,
        metric_labels: Optional[Dict[str, str]] = None,
        channel_colors: Optional[Dict[str, str]] = None,
        breath_data: Optional[Dict[str, Any]] = None,
        source_stem: str = "",
    ):
        super().__init__(parent)

        # Store data
        self._markers = markers or []
        self._signals = signals or {}
        self._time_array = time_array
        self._metric_labels = metric_labels or {}
        self._channel_colors = channel_colors or {}
        self._breath_data = breath_data
        self._source_stem = source_stem  # e.g. "3_30_2026_B__recovered_photometry"

        # Per-metric colors (assigned during populate)
        self._metric_colors: Dict[str, QColor] = {}
        self._active_metric_row: int = -1  # Tracked for up/down reordering
        self._checkbox_just_toggled = False

        # Optional default export directory (set externally)
        self._default_export_dir: Optional[str] = None

        # Create or use provided ViewModel
        self._viewmodel = viewmodel or CTAViewModel(self)

        if metric_labels:
            self._viewmodel.set_available_metrics(metric_labels)

        # Connect ViewModel signals
        self._viewmodel.calculation_started.connect(self._on_calculation_started)
        self._viewmodel.calculation_progress.connect(self._on_calculation_progress)
        self._viewmodel.calculation_complete.connect(self._on_calculation_complete)
        self._viewmodel.preview_ready.connect(self._on_preview_ready)
        self._viewmodel.error_occurred.connect(self._on_error)
        self._viewmodel.export_complete.connect(self._on_export_complete)

        # Build UI
        self._init_ui()

        # Populate
        self._populate_marker_categories()
        self._populate_metrics()
        self._update_breath_info()

    def _init_ui(self):
        """Build the widget UI."""
        main_layout = QHBoxLayout(self)

        # Left panel: Configuration (scrollable for smaller windows)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(370)
        left_scroll.setStyleSheet("QScrollArea { border: none; }")
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # --- Trigger Source (compact) ---
        trigger_group = QGroupBox("Trigger Source")
        trigger_layout = QVBoxLayout(trigger_group)
        trigger_layout.setContentsMargins(6, 2, 6, 2)
        trigger_layout.setSpacing(2)

        trigger_radio_layout = QHBoxLayout()
        self._radio_event_markers = QRadioButton("Event Markers")
        self._radio_breath_events = QRadioButton("Breath Events")
        self._radio_stim_events = QRadioButton("Stim Events")
        self._radio_event_markers.setChecked(True)
        trigger_radio_layout.addWidget(self._radio_event_markers)
        trigger_radio_layout.addWidget(self._radio_breath_events)
        trigger_radio_layout.addWidget(self._radio_stim_events)
        trigger_layout.addLayout(trigger_radio_layout)

        # Disable breath events if no breath data available
        if self._breath_data is None:
            self._radio_breath_events.setEnabled(False)
            self._radio_breath_events.setToolTip("No breath data available (run peak detection first)")

        # Disable stim events if no stim data available
        stim_spans = self._get_stim_spans()
        if not stim_spans:
            self._radio_stim_events.setEnabled(False)
            self._radio_stim_events.setToolTip("No stim channel detected")
        else:
            n_stim = sum(len(spans) for spans in stim_spans.values())
            self._radio_stim_events.setToolTip(
                f"Use laser/stimulus onset and offset as CTA triggers ({n_stim} stim events)"
            )

        # Stacked widget: page 0 = marker categories, page 1 = breath config
        self._trigger_stack = QStackedWidget()
        self._trigger_stack.setMaximumHeight(180)

        # Page 0: marker categories (without QGroupBox wrapper)
        markers_widget = QWidget()
        markers_inner = QVBoxLayout(markers_widget)
        markers_inner.setContentsMargins(0, 0, 0, 0)
        self._marker_list = QListWidget()
        self._marker_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._marker_list.setMaximumHeight(80)
        markers_inner.addWidget(self._marker_list)
        self._trigger_stack.addWidget(markers_widget)

        # Page 1: breath events config
        breath_widget = QWidget()
        breath_layout = QGridLayout(breath_widget)
        breath_layout.setContentsMargins(0, 0, 0, 0)

        # Lazy import to avoid circular/heavy imports at module level
        from core.services.breath_event_extractor import BreathEventExtractor

        breath_layout.addWidget(QLabel("Trigger:"), 0, 0)
        self._breath_trigger_combo = QComboBox()
        for key, display in BreathEventExtractor.TRIGGER_POINTS.items():
            self._breath_trigger_combo.addItem(display, key)
        breath_layout.addWidget(self._breath_trigger_combo, 0, 1)

        breath_layout.addWidget(QLabel("Type:"), 1, 0)
        self._breath_type_combo = QComboBox()
        for key, display in BreathEventExtractor.BREATH_TYPES.items():
            self._breath_type_combo.addItem(display, key)
        breath_layout.addWidget(self._breath_type_combo, 1, 1)

        breath_layout.addWidget(QLabel("Max events:"), 2, 0)
        self._breath_max_events = QSpinBox()
        self._breath_max_events.setRange(0, 99999)
        self._breath_max_events.setValue(100)
        self._breath_max_events.setSpecialValueText("All")
        breath_layout.addWidget(self._breath_max_events, 2, 1)

        breath_layout.addWidget(QLabel("Min bout:"), 3, 0)
        self._breath_min_bout = QSpinBox()
        self._breath_min_bout.setRange(1, 999)
        self._breath_min_bout.setValue(1)
        self._breath_min_bout.setToolTip("Minimum consecutive breaths of same type")
        breath_layout.addWidget(self._breath_min_bout, 3, 1)

        # Time region filter — always visible start/end with marker or manual entry
        # Start time row
        start_row = QWidget()
        start_layout = QHBoxLayout(start_row)
        start_layout.setContentsMargins(0, 0, 0, 0)
        start_layout.setSpacing(3)
        start_layout.addWidget(QLabel("From:"))
        self._breath_time_start = QDoubleSpinBox()
        self._breath_time_start.setRange(0.0, 999999.0)
        self._breath_time_start.setValue(0.0)
        self._breath_time_start.setSingleStep(10.0)
        self._breath_time_start.setDecimals(1)
        self._breath_time_start.setSuffix("s")
        self._breath_time_start.setSpecialValueText("Start")
        self._breath_time_start.valueChanged.connect(self._update_breath_info)
        start_layout.addWidget(self._breath_time_start, 1)
        self._breath_start_marker = QComboBox()
        self._breath_start_marker.setMaximumWidth(90)
        self._breath_start_marker.setToolTip("Pick a marker's start time")
        self._breath_start_marker.addItem("manual", None)
        self._breath_start_marker.currentIndexChanged.connect(
            lambda: self._on_marker_time_picked('start'))
        start_layout.addWidget(self._breath_start_marker)
        breath_layout.addWidget(start_row, 4, 0, 1, 2)

        # End time row
        end_row = QWidget()
        end_layout = QHBoxLayout(end_row)
        end_layout.setContentsMargins(0, 0, 0, 0)
        end_layout.setSpacing(3)
        end_layout.addWidget(QLabel("  To:"))
        self._breath_time_end = QDoubleSpinBox()
        self._breath_time_end.setRange(0.0, 999999.0)
        self._breath_time_end.setValue(0.0)
        self._breath_time_end.setSingleStep(10.0)
        self._breath_time_end.setDecimals(1)
        self._breath_time_end.setSuffix("s")
        self._breath_time_end.setSpecialValueText("End")
        self._breath_time_end.valueChanged.connect(self._update_breath_info)
        end_layout.addWidget(self._breath_time_end, 1)
        self._breath_end_marker = QComboBox()
        self._breath_end_marker.setMaximumWidth(90)
        self._breath_end_marker.setToolTip("Pick a marker's end time")
        self._breath_end_marker.addItem("manual", None)
        self._breath_end_marker.currentIndexChanged.connect(
            lambda: self._on_marker_time_picked('end'))
        end_layout.addWidget(self._breath_end_marker)
        breath_layout.addWidget(end_row, 5, 0, 1, 2)

        # Show info about breath data availability
        self._breath_auto_conditions = None  # Kept for compat, unused
        self._breath_info_label = QLabel("")
        self._breath_info_label.setStyleSheet("color: #888; font-size: 10px;")
        self._breath_info_label.setWordWrap(True)
        breath_layout.addWidget(self._breath_info_label, 6, 0, 1, 2)

        self._trigger_stack.addWidget(breath_widget)

        # Page 2: stim events info
        stim_widget = QWidget()
        stim_layout = QVBoxLayout(stim_widget)
        stim_layout.setContentsMargins(0, 0, 0, 0)
        stim_info = QLabel(
            "CTA will be generated around detected laser/stimulus events.\n"
            "Both onset and offset alignments will be included."
        )
        stim_info.setWordWrap(True)
        stim_info.setStyleSheet("color: #999; font-size: 11px; padding: 4px;")
        stim_layout.addWidget(stim_info)
        stim_layout.addStretch()
        self._trigger_stack.addWidget(stim_widget)

        trigger_layout.addWidget(self._trigger_stack)
        left_layout.addWidget(trigger_group)

        # Connect radio buttons to switch stacked widget pages
        def _update_trigger_page():
            if self._radio_event_markers.isChecked():
                self._trigger_stack.setCurrentIndex(0)
            elif self._radio_breath_events.isChecked():
                self._trigger_stack.setCurrentIndex(1)
            elif self._radio_stim_events.isChecked():
                self._trigger_stack.setCurrentIndex(2)

        self._radio_event_markers.toggled.connect(_update_trigger_page)
        self._radio_breath_events.toggled.connect(self._on_trigger_source_changed)
        self._radio_stim_events.toggled.connect(_update_trigger_page)
        self._breath_trigger_combo.currentIndexChanged.connect(self._on_breath_trigger_changed)

        # --- Time Windows (single line) ---
        window_group = QGroupBox("Time Windows")
        window_layout = QHBoxLayout(window_group)
        window_layout.setContentsMargins(8, 4, 8, 4)

        window_layout.addWidget(QLabel("Before (s):"))
        self._spin_before = QDoubleSpinBox()
        self._spin_before.setRange(0.1, 99999.9)
        self._spin_before.setSingleStep(5.0)
        self._spin_before.setDecimals(1)
        self._spin_before.setToolTip("Time window before event (seconds)")
        window_layout.addWidget(self._spin_before)

        window_layout.addWidget(QLabel("After (s):"))
        self._spin_after = QDoubleSpinBox()
        self._spin_after.setRange(0.1, 99999.9)
        self._spin_after.setSingleStep(5.0)
        self._spin_after.setDecimals(1)
        self._spin_after.setToolTip("Time window after event (seconds)")
        window_layout.addWidget(self._spin_after)

        # Set values and connect signals after both spinboxes exist
        self._spin_before.setValue(self._viewmodel.config.window_before)
        self._spin_after.setValue(self._viewmodel.config.window_after)
        self._spin_before.valueChanged.connect(self._on_window_changed)
        self._spin_after.valueChanged.connect(self._on_window_changed)

        left_layout.addWidget(window_group)

        # --- Z-Score Normalization (checkbox is the group title) ---
        self._zscore_group = QGroupBox("Z-Score Normalization")
        self._zscore_group.setCheckable(True)
        self._zscore_group.setChecked(self._viewmodel.config.zscore_baseline)
        self._zscore_group.toggled.connect(self._on_zscore_changed)
        zscore_layout = QHBoxLayout(self._zscore_group)
        zscore_layout.setContentsMargins(8, 4, 8, 4)

        zscore_layout.addWidget(QLabel("Start (s):"))
        self._spin_baseline_start = QDoubleSpinBox()
        self._spin_baseline_start.setRange(-99999.9, 0.0)
        self._spin_baseline_start.setSingleStep(0.5)
        self._spin_baseline_start.setDecimals(2)
        self._spin_baseline_start.setValue(self._viewmodel.config.baseline_start)
        self._spin_baseline_start.setToolTip("Start of baseline period (negative = before event)")
        self._spin_baseline_start.setKeyboardTracking(True)
        self._spin_baseline_start.lineEdit().setReadOnly(False)
        self._spin_baseline_start.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._spin_baseline_start.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_start)

        zscore_layout.addWidget(QLabel("End (s):"))
        self._spin_baseline_end = QDoubleSpinBox()
        self._spin_baseline_end.setRange(-99999.9, 99999.9)
        self._spin_baseline_end.setSingleStep(0.25)
        self._spin_baseline_end.setDecimals(2)
        self._spin_baseline_end.setValue(self._viewmodel.config.baseline_end)
        self._spin_baseline_end.setToolTip("End of baseline period (0 = event onset)")
        self._spin_baseline_end.setKeyboardTracking(True)
        self._spin_baseline_end.lineEdit().setReadOnly(False)
        self._spin_baseline_end.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._spin_baseline_end.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_end)

        left_layout.addWidget(self._zscore_group)

        # --- Condition Mode ---
        self._condition_group = QGroupBox("Conditions")
        self._condition_group.setMaximumHeight(120)
        condition_group = self._condition_group
        condition_layout = QVBoxLayout(condition_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._condition_mode_combo = QComboBox()
        self._condition_mode_combo.addItem("Combined", "combined")
        self._condition_mode_combo.addItem("Separate", "separate")
        self._condition_mode_combo.addItem("Overlay", "overlay")
        self._condition_mode_combo.setCurrentIndex(2)  # Default to Overlay
        self._condition_mode_combo.setToolTip(
            "Combined: Pool all events into one CTA\n"
            "Separate: Independent CTA per condition (stacked plots)\n"
            "Overlay: Conditions on same axes with different colors"
        )
        mode_layout.addWidget(self._condition_mode_combo)
        condition_layout.addLayout(mode_layout)

        # Condition checklist
        self._condition_list = QListWidget()
        self._condition_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._condition_list.setMaximumHeight(60)
        condition_layout.addWidget(self._condition_list)

        left_layout.addWidget(condition_group)

        # --- Metrics Table (color + name with checkboxes, up/down arrows) ---
        metrics_group = QGroupBox("Metrics to Include")
        metrics_layout = QVBoxLayout(metrics_group)

        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Color", "Metric"])
        self._metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._metrics_table.setColumnWidth(0, 40)
        self._metrics_table.verticalHeader().setVisible(False)
        self._metrics_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._metrics_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._metrics_table.setMinimumHeight(80)
        self._metrics_table.cellClicked.connect(self._on_metrics_cell_clicked)
        metrics_layout.addWidget(self._metrics_table)

        # Up/Down arrow buttons for reordering
        _arrow_style = "QPushButton { border: none; background: transparent; } QPushButton:hover { background: #3a3a3a; }"
        arrow_layout = QHBoxLayout()
        self._btn_move_up = QPushButton()
        self._btn_move_up.setFixedSize(24, 20)
        self._btn_move_up.setToolTip("Move selected metric up")
        self._btn_move_up.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ArrowUp))
        self._btn_move_up.setStyleSheet(_arrow_style)
        self._btn_move_up.clicked.connect(self._move_metric_up)
        arrow_layout.addWidget(self._btn_move_up)

        self._btn_move_down = QPushButton()
        self._btn_move_down.setFixedSize(24, 20)
        self._btn_move_down.setToolTip("Move selected metric down")
        self._btn_move_down.setIcon(self.style().standardIcon(self.style().StandardPixmap.SP_ArrowDown))
        self._btn_move_down.setStyleSheet(_arrow_style)
        self._btn_move_down.clicked.connect(self._move_metric_down)
        arrow_layout.addWidget(self._btn_move_down)
        arrow_layout.addStretch()
        metrics_layout.addLayout(arrow_layout)

        left_layout.addWidget(metrics_group, 1)  # stretch=1 so metrics gets remaining space

        # --- Generate button ---
        self._btn_generate = QPushButton("Generate CTA")
        self._btn_generate.setMinimumHeight(40)
        self._btn_generate.setStyleSheet(
            "QPushButton { background-color: #1a6b1a; border: 1px solid #2a8a2a; }"
            "QPushButton:hover { background-color: #228b22; border-color: #33aa33; }"
            "QPushButton:pressed { background-color: #0d4d0d; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._btn_generate.clicked.connect(self._on_generate_clicked)
        left_layout.addWidget(self._btn_generate)

        # Export CSV button
        self._btn_export_csv = QPushButton("Export CSV")
        self._btn_export_csv.setMinimumHeight(32)
        self._btn_export_csv.setEnabled(False)
        self._btn_export_csv.setStyleSheet(
            "QPushButton { background-color: #1a4a6b; border: 1px solid #2a6a8a; }"
            "QPushButton:hover { background-color: #22628b; border-color: #33a; }"
            "QPushButton:pressed { background-color: #0d3a4d; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        left_layout.addWidget(self._btn_export_csv)

        # Show paired marker lines toggle
        self._chk_show_paired_lines = QCheckBox("Show paired marker lines")
        self._chk_show_paired_lines.setChecked(True)
        self._chk_show_paired_lines.setToolTip(
            "Show faint lines where the paired event (onset/withdrawal) occurs on each trace"
        )
        self._chk_show_paired_lines.toggled.connect(lambda: self._update_preview_plots())
        left_layout.addWidget(self._chk_show_paired_lines)

        # Show baseline zone toggle
        self._chk_show_stim = QCheckBox("Show stim period")
        self._chk_show_stim.setChecked(True)
        self._chk_show_stim.setToolTip(
            "Shade the laser/stimulus on period on the CTA plot.\n"
            "Uses stim onset/offset from the current file's detected stim channel."
        )
        self._chk_show_stim.toggled.connect(lambda: self._update_preview_plots())
        left_layout.addWidget(self._chk_show_stim)

        self._chk_show_baseline = QCheckBox("Show baseline zone")
        self._chk_show_baseline.setChecked(True)
        self._chk_show_baseline.setToolTip(
            "Show the z-score normalization window as a shaded region on plots"
        )
        self._chk_show_baseline.toggled.connect(lambda: self._update_preview_plots())
        left_layout.addWidget(self._chk_show_baseline)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)


        left_scroll.setWidget(left_panel)
        main_layout.addWidget(left_scroll)

        # Right panel: Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self._plot_tab = QScrollArea()
        self._plot_tab.setWidgetResizable(True)
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_tab.setWidget(self._plot_container)
        right_layout.addWidget(self._plot_tab)

        self._placeholder_label = QLabel("Configure options and click 'Generate CTA'")
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setStyleSheet("color: #888; font-size: 14px;")
        self._plot_layout.addWidget(self._placeholder_label)

        main_layout.addWidget(right_panel, 1)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def tab_name(self) -> str:
        """Return a name suitable for a tab label based on trigger source."""
        if self._radio_breath_events.isChecked():
            trigger_display = self._breath_trigger_combo.currentText()
            breath_type = self._breath_type_combo.currentText()
            if breath_type and breath_type != 'All breaths':
                return f"{trigger_display} ({breath_type})"
            return trigger_display or "Breath CTA"
        else:
            # Event markers: first checked category name
            for i in range(self._marker_list.count()):
                item = self._marker_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    return item.text()
            return "Event CTA"

    # -------------------------------------------------------------------------
    # Populate
    # -------------------------------------------------------------------------

    def _populate_marker_categories(self):
        """Populate marker categories from available markers.

        For paired markers, shows both onset and withdrawal entries.
        """
        self._marker_list.clear()

        print(f"[CTA Markers] Populating from {len(self._markers)} markers")
        cat_label_pairs = {}  # (category, label) -> has_paired
        for marker in self._markers:
            key = (marker.category, marker.label)
            if key not in cat_label_pairs:
                cat_label_pairs[key] = False
            if hasattr(marker, 'is_paired') and marker.is_paired:
                cat_label_pairs[key] = True
        print(f"[CTA Markers] Categories: {cat_label_pairs}")

        has_any_paired = False
        for (category, label), has_paired in sorted(cat_label_pairs.items()):
            cat_str = f"{category}:{label}"
            if has_paired:
                has_any_paired = True
                for alignment in ('onset', 'withdrawal'):
                    item = QListWidgetItem(f"{cat_str} ({alignment})")
                    item.setData(Qt.ItemDataRole.UserRole, {
                        'category': category, 'label': label, 'alignment': alignment
                    })
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
                    self._marker_list.addItem(item)
            else:
                item = QListWidgetItem(cat_str)
                item.setData(Qt.ItemDataRole.UserRole, {
                    'category': category, 'label': label, 'alignment': 'onset'
                })
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                self._marker_list.addItem(item)

        # Auto-enable withdrawal if paired markers exist
        if has_any_paired:
            self._viewmodel.include_withdrawal = True

        # Populate condition filter and breath time sources
        self._populate_conditions()
        self._update_breath_time_sources()

    def _populate_conditions(self):
        """Populate condition filter from marker conditions."""
        self._condition_list.clear()

        conditions = set()
        for marker in self._markers:
            if getattr(marker, 'visible', True) and marker.condition:
                conditions.add(marker.condition)

        if not conditions:
            item = QListWidgetItem("(no conditions set)")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._condition_list.addItem(item)
            return

        # "All" option
        item = QListWidgetItem("All conditions")
        item.setData(Qt.ItemDataRole.UserRole, '__all__')
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self._condition_list.addItem(item)

        for cond in sorted(conditions):
            item = QListWidgetItem(cond)
            item.setData(Qt.ItemDataRole.UserRole, cond)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._condition_list.addItem(item)

    def _populate_conditions_from_markers(self, markers):
        """Populate condition filter from a list of markers (e.g., breath events)."""
        self._condition_list.clear()
        conditions = set(m.condition for m in markers if m.condition)

        if not conditions:
            item = QListWidgetItem("(no conditions set)")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._condition_list.addItem(item)
            return

        item = QListWidgetItem("All conditions")
        item.setData(Qt.ItemDataRole.UserRole, '__all__')
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked)
        self._condition_list.addItem(item)

        for cond in sorted(conditions):
            item = QListWidgetItem(cond)
            item.setData(Qt.ItemDataRole.UserRole, cond)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self._condition_list.addItem(item)

    def _get_selected_conditions(self):
        """Get list of selected conditions, or None for 'all'."""
        selected = []
        for i in range(self._condition_list.count()):
            item = self._condition_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data == '__all__' and item.checkState() == Qt.CheckState.Checked:
                return None  # All conditions
            if data and data != '__all__' and item.checkState() == Qt.CheckState.Checked:
                selected.append(data)
        return selected if selected else None

    def _populate_metrics(self):
        """Populate metrics table with checkboxes and color swatches."""
        self._metrics_table.blockSignals(True)
        self._metrics_table.setRowCount(0)
        self._metric_colors.clear()

        items = list(self._metric_labels.items()) if self._metric_labels else [
            (k, k) for k in self._signals.keys()
        ]

        self._metrics_table.setRowCount(len(items))
        for row, (key, label) in enumerate(items):
            # Use channel color from main plot if available, otherwise palette
            if key in self._channel_colors:
                color = QColor(self._channel_colors[key])
            else:
                color = QColor(_METRIC_COLORS[row % len(_METRIC_COLORS)])
            self._metric_colors[key] = color

            # Color swatch cell
            color_item = QTableWidgetItem()
            color_item.setBackground(QBrush(color))
            color_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            color_item.setData(Qt.ItemDataRole.UserRole, key)
            self._metrics_table.setItem(row, 0, color_item)

            # Metric name cell with checkbox
            display = f"{label} ({key})" if label != key else key
            name_item = QTableWidgetItem(display)
            name_item.setFlags(
                Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled |
                Qt.ItemFlag.ItemIsUserCheckable
            )
            name_item.setCheckState(Qt.CheckState.Unchecked)
            name_item.setData(Qt.ItemDataRole.UserRole, key)
            self._metrics_table.setItem(row, 1, name_item)

        self._metrics_table.blockSignals(False)
        # Connect cellChanged only once (disconnect first to avoid duplicates)
        try:
            self._metrics_table.cellChanged.disconnect(self._on_metric_check_changed)
        except (TypeError, RuntimeError):
            pass
        self._metrics_table.cellChanged.connect(self._on_metric_check_changed)
        self._update_metric_row_styles()

    def _on_metrics_cell_clicked(self, row: int, col: int):
        """Handle click on metrics table - toggle checkbox or open color picker."""
        self._active_metric_row = row
        if col == 1:
            # Toggle checkbox on click in the metric name column.
            # Qt handles clicks directly on the checkbox indicator via cellChanged,
            # so we only manually toggle for clicks on the text area (not the indicator).
            # We detect this by checking if the checkbox state already changed via cellChanged
            # since the last click -- if so, skip the manual toggle to avoid double-toggle.
            name_item = self._metrics_table.item(row, 1)
            if name_item and not getattr(self, '_checkbox_just_toggled', False):
                self._metrics_table.blockSignals(True)
                new_state = (Qt.CheckState.Unchecked
                             if name_item.checkState() == Qt.CheckState.Checked
                             else Qt.CheckState.Checked)
                name_item.setCheckState(new_state)
                self._metrics_table.blockSignals(False)
                self._update_metric_row_styles()
            self._checkbox_just_toggled = False
            return
        if col == 0:
            item = self._metrics_table.item(row, 0)
            if item:
                key = item.data(Qt.ItemDataRole.UserRole)
                current_color = self._metric_colors.get(key, QColor('#4488ff'))
                color = QColorDialog.getColor(current_color, self, f"Color for {key}")
                if color.isValid():
                    self._metric_colors[key] = color
                    item.setBackground(QBrush(color))

    def _on_metric_check_changed(self, row: int, col: int):
        """Handle checkbox toggle - update row styling."""
        if col == 1:
            self._checkbox_just_toggled = True
            self._update_metric_row_styles()

    def _update_metric_row_styles(self):
        """Gray out unchecked metric rows."""
        for row in range(self._metrics_table.rowCount()):
            name_item = self._metrics_table.item(row, 1)
            color_item = self._metrics_table.item(row, 0)
            if not name_item:
                continue
            checked = name_item.checkState() == Qt.CheckState.Checked
            fg = QColor('#e0e0e0') if checked else QColor('#555555')
            name_item.setForeground(QBrush(fg))
            if color_item:
                if checked:
                    key = color_item.data(Qt.ItemDataRole.UserRole)
                    color_item.setBackground(QBrush(self._metric_colors.get(key, QColor('#4488ff'))))
                else:
                    color_item.setBackground(QBrush(QColor('#3a3a3a')))

    def _move_metric_up(self):
        """Move the active metric row up one position."""
        row = self._active_metric_row
        if row <= 0:
            return
        self._swap_metric_rows(row, row - 1)
        self._active_metric_row = row - 1

    def _move_metric_down(self):
        """Move the active metric row down one position."""
        row = self._active_metric_row
        if row < 0 or row >= self._metrics_table.rowCount() - 1:
            return
        self._swap_metric_rows(row, row + 1)
        self._active_metric_row = row + 1

    def _swap_metric_rows(self, row_a: int, row_b: int):
        """Swap two rows in the metrics table, preserving all data."""
        self._metrics_table.blockSignals(True)

        def _take_row(r):
            color_item = self._metrics_table.takeItem(r, 0)
            name_item = self._metrics_table.takeItem(r, 1)
            return color_item, name_item

        items_a = _take_row(row_a)
        items_b = _take_row(row_b)

        self._metrics_table.setItem(row_a, 0, items_b[0])
        self._metrics_table.setItem(row_a, 1, items_b[1])
        self._metrics_table.setItem(row_b, 0, items_a[0])
        self._metrics_table.setItem(row_b, 1, items_a[1])

        self._metrics_table.blockSignals(False)

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------

    def _get_selected_categories(self) -> List[str]:
        """Get unique selected category:label strings."""
        categories = []
        seen = set()
        for i in range(self._marker_list.count()):
            item = self._marker_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    cat_str = f"{data['category']}:{data['label']}"
                else:
                    cat_str = item.text()
                if cat_str not in seen:
                    categories.append(cat_str)
                    seen.add(cat_str)
        return categories

    def _get_selected_alignments(self) -> Dict[str, List[str]]:
        """Get selected alignments per category:label."""
        alignments = {}
        for i in range(self._marker_list.count()):
            item = self._marker_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                data = item.data(Qt.ItemDataRole.UserRole)
                if data:
                    cat_str = f"{data['category']}:{data['label']}"
                    if cat_str not in alignments:
                        alignments[cat_str] = []
                    alignments[cat_str].append(data.get('alignment', 'onset'))
        return alignments

    def _get_selected_metrics(self) -> List[str]:
        """Get checked metric keys in table order."""
        metrics = []
        for row in range(self._metrics_table.rowCount()):
            item = self._metrics_table.item(row, 1)
            if item and item.checkState() == Qt.CheckState.Checked:
                metrics.append(item.data(Qt.ItemDataRole.UserRole))
        return metrics

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def _on_trigger_source_changed(self, checked: bool):
        """Handle trigger source radio button change."""
        is_breath = self._radio_breath_events.isChecked()
        if is_breath:
            self._update_breath_info()
            # Hide conditions unless auto-assign will populate them
            self._condition_group.setVisible(False)
        else:
            self._condition_group.setVisible(True)
        self.name_changed.emit(self.tab_name)

    def _on_breath_trigger_changed(self):
        """Disable type filter for bout on/offset triggers (type is implied)."""
        from core.services.breath_event_extractor import BreathEventExtractor
        trigger = self._breath_trigger_combo.currentData()
        is_bout = trigger in BreathEventExtractor.BOUT_TRIGGERS
        self._breath_type_combo.setEnabled(not is_bout)
        if is_bout:
            self._breath_type_combo.setCurrentIndex(0)  # Reset to "All"
        self.name_changed.emit(self.tab_name)

    def _on_marker_time_picked(self, which: str):
        """Handle marker selection in the start/end time picker."""
        if which == 'start':
            combo = self._breath_start_marker
            spin = self._breath_time_start
        else:
            combo = self._breath_end_marker
            spin = self._breath_time_end

        marker_id = combo.currentData()
        if marker_id is None:
            return  # "manual" selected, keep spinbox as-is

        for m in self._markers:
            if m.id == marker_id:
                if which == 'start':
                    spin.setValue(m.start_time)
                else:
                    # Use end_time if paired, else start_time
                    spin.setValue(m.end_time if m.end_time is not None else m.start_time)
                break

    def _update_breath_time_sources(self):
        """Populate marker picker combos with available markers."""
        if not hasattr(self, '_breath_start_marker'):
            return

        # Gather all markers sorted by time
        sorted_markers = sorted(self._markers, key=lambda m: m.start_time)

        for combo in (self._breath_start_marker, self._breath_end_marker):
            combo.blockSignals(True)
            current_data = combo.currentData()
            # Keep "manual" entry, remove rest
            while combo.count() > 1:
                combo.removeItem(1)

            for m in sorted_markers:
                if not getattr(m, 'visible', True):
                    continue
                cond = f" [{m.condition}]" if m.condition else ""
                t = m.start_time
                display = f"{m.category}{cond} @{t:.0f}s"
                combo.addItem(display, m.id)

            # Restore selection
            if current_data:
                for i in range(combo.count()):
                    if combo.itemData(i) == current_data:
                        combo.setCurrentIndex(i)
                        break
            combo.blockSignals(False)

    def _on_window_changed(self):
        self._viewmodel.window_before = self._spin_before.value()
        self._viewmodel.window_after = self._spin_after.value()

    def _on_zscore_changed(self, checked: bool):
        self._viewmodel.config.zscore_baseline = checked

    def _on_baseline_changed(self):
        self._viewmodel.config.baseline_start = self._spin_baseline_start.value()
        self._viewmodel.config.baseline_end = self._spin_baseline_end.value()

    def _on_generate_clicked(self):
        """Handle Generate button click."""
        metrics = self._get_selected_metrics()
        if not metrics:
            QMessageBox.warning(self, "No Selection", "Please select at least one metric.")
            return

        if self._time_array is None or len(self._signals) == 0:
            QMessageBox.warning(self, "No Data", "No signal data available for CTA calculation.")
            return

        # Warn about large windows (traces are serialized to JSON -- very large
        # windows can use significant memory). Future: downsample traces for storage.
        total_window = self._spin_before.value() + self._spin_after.value()
        if total_window > 120:
            reply = QMessageBox.question(
                self, "Large Time Window",
                f"Total window is {total_window:.0f}s. Large windows may use "
                f"significant memory and slow down saving.\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Determine trigger source and get markers
        if self._radio_breath_events.isChecked() and self._breath_data:
            from core.services.breath_event_extractor import BreathEventExtractor
            extractor = BreathEventExtractor()
            breath_markers = extractor.extract(
                trigger_point=self._breath_trigger_combo.currentData(),
                breath_type_filter=self._breath_type_combo.currentData(),
                max_events=self._breath_max_events.value(),
                min_bout_length=self._breath_min_bout.value(),
                auto_assign_conditions=False,
                all_peaks=self._breath_data['all_peaks'],
                all_breaths=self._breath_data['all_breaths'],
                sigh_indices=self._breath_data.get('sigh_indices', np.array([])),
                time_array=self._time_array,
                sr_hz=self._breath_data.get('sr_hz', 1000.0),
                sweep_idx=self._breath_data.get('sweep_idx', 0),
                existing_markers=self._markers,
                time_window_start=self._breath_time_start.value(),
                time_window_end=self._breath_time_end.value(),
            )
            filtered_markers = breath_markers  # These are already the events
            print(f"[CTA Breath] Extracted {len(filtered_markers)} breath events")

            if not filtered_markers:
                QMessageBox.warning(
                    self, "No Events",
                    "No breath events found with the current settings."
                )
                return

            # Warn about large event counts (each event = window extraction + interpolation)
            if len(filtered_markers) > 500:
                reply = QMessageBox.question(
                    self, "Many Events",
                    f"{len(filtered_markers)} breath events selected.\n\n"
                    f"This may take a while and use significant memory.\n"
                    f"Consider using a time range or max events filter.\n\n"
                    f"Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return

            # For breath events, derive categories from the generated markers
            categories = list(set(
                f"{m.category}:{m.label}" for m in filtered_markers
            ))
            self._viewmodel.selected_categories = categories
            self._viewmodel.selected_metrics = metrics
            self._viewmodel.include_withdrawal = False  # Single markers, no withdrawal
            self._condition_group.setVisible(False)

        elif self._radio_stim_events.isChecked():
            # Stim onset/offset as trigger events
            stim_markers = self._build_stim_markers()
            if not stim_markers:
                QMessageBox.warning(self, "No Stim Data", "No stim events detected.")
                return

            filtered_markers = stim_markers
            categories = ['stimulus:laser']
            self._viewmodel.selected_categories = categories
            self._viewmodel.selected_metrics = metrics
            self._viewmodel.include_withdrawal = True  # Include both onset and offset CTAs
            print(f"[CTA Stim] Using {len(filtered_markers)} stim events as triggers")

        else:
            # Original event marker path
            categories = self._get_selected_categories()
            if not categories:
                QMessageBox.warning(self, "No Selection", "Please select at least one marker category.")
                return

            # Check if any withdrawal alignment is selected
            alignments = self._get_selected_alignments()
            has_withdrawal = any('withdrawal' in aligns for aligns in alignments.values())
            self._viewmodel.include_withdrawal = has_withdrawal

            self._viewmodel.selected_categories = categories
            self._viewmodel.selected_metrics = metrics

            # Filter markers by condition and visibility
            selected_conditions = self._get_selected_conditions()
            filtered_markers = [
                m for m in self._markers
                if getattr(m, 'visible', True)
                and (selected_conditions is None or not m.condition or m.condition in selected_conditions)
            ]

        # Get condition mode — breath events always use combined
        condition_mode = self._condition_mode_combo.currentData() or "combined"
        if self._radio_breath_events.isChecked():
            condition_mode = "combined"

        self._viewmodel.generate_preview(
            markers=filtered_markers,
            signals=self._signals,
            time_array=self._time_array,
            metric_labels=self._metric_labels,
            condition_mode=condition_mode,
        )

        # Emit name change since generation may reflect the current config
        self.name_changed.emit(self.tab_name)

    def _on_calculation_started(self):
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_generate.setEnabled(False)
        self._btn_generate.setText("Calculating...")

    def _on_calculation_progress(self, value: int):
        self._progress.setValue(value)

    def _on_calculation_complete(self):
        self._progress.setVisible(False)
        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("Generate CTA")

    def _on_preview_ready(self):
        """Handle preview ready."""
        collection = self._viewmodel.current_collection
        if collection:
            print(f"[CTA] Collection has {len(collection.results)} results")
            for key, r in collection.results.items():
                print(f"[CTA]   {key}: {r.n_events} events, traces={len(r.traces)}")
        else:
            print("[CTA] No collection after generation")
            cond_colls = self._viewmodel.condition_collections
            if cond_colls:
                print(f"[CTA] But have {len(cond_colls)} condition collections")
                for cname, coll in cond_colls.items():
                    print(f"[CTA]   {cname}: {len(coll.results)} results")

        self._btn_export_csv.setEnabled(True)
        self._update_preview_plots()
        self.cta_generated.emit()

    def _on_error(self, message: str):
        self._progress.setVisible(False)
        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("Generate CTA")
        QMessageBox.critical(self, "CTA Error", message)

    def _on_export_complete(self, filepath: str):
        QMessageBox.information(self, "Export Complete", f"CTA data exported to:\n{filepath}")

    def _on_export_csv_clicked(self):
        """Export CTA data to wide-format CSV (all metrics in one file)."""
        # Build smart default filename from source stem + tab name
        tab_suffix = self.tab_name.replace(' ', '_').replace('/', '_').replace(':', '-')
        if self._source_stem:
            default_name = f"{self._source_stem}_CTA_{tab_suffix}.csv"
        else:
            default_name = f"CTA_{tab_suffix}.csv"
        default_dir = self._default_export_dir or ""

        start_path = str(Path(default_dir) / default_name) if default_dir else default_name

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CTA Data", start_path,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filepath:
            self._viewmodel.export_to_csv_wide(filepath, metadata=self._build_export_metadata())

    def _build_export_metadata(self) -> dict:
        """Build metadata dict for CSV header."""
        meta = {}

        # Source file info
        if self._source_stem:
            meta['source_file'] = self._source_stem

        # Get parent state info
        parent = self.parent()
        if parent and hasattr(parent, '_source_stem'):
            meta['source_file'] = parent._source_stem
        if parent and hasattr(parent, '_source_dir'):
            meta['source_dir'] = parent._source_dir

        # App version
        try:
            from version_info import VERSION_STRING
            meta['app_version'] = VERSION_STRING
        except ImportError:
            pass

        # Parent state details
        mw = self.window()
        if mw and hasattr(mw, 'state'):
            st = mw.state
            if hasattr(st, 'in_path') and st.in_path:
                meta['source_file'] = str(st.in_path)
            if hasattr(st, 'sr_hz'):
                meta['sample_rate'] = st.sr_hz
            if hasattr(st, 'analyze_chan'):
                meta['analyze_channel'] = st.analyze_chan
            exp_idx = getattr(st, 'photometry_experiment_index', None)
            if exp_idx is not None:
                meta['experiment_index'] = exp_idx

        # Animal ID from master list
        if mw and hasattr(mw, '_active_master_list_row') and hasattr(mw, '_master_file_list'):
            row = mw._active_master_list_row
            if row is not None and row < len(mw._master_file_list):
                animal_id = mw._master_file_list[row].get('animal_id', '')
                if animal_id:
                    meta['animal_id'] = animal_id

        # Trigger source
        if self._radio_breath_events.isChecked():
            trigger = self._breath_trigger_combo.currentText()
            btype = self._breath_type_combo.currentText()
            meta['trigger_source'] = f'Breath Events: {trigger} ({btype})'
            meta['max_events'] = self._breath_max_events.value()
            t_start = self._breath_time_start.value()
            t_end = self._breath_time_end.value()
            if t_start > 0 or t_end > 0:
                meta['time_window'] = f'{t_start:.1f}s - {t_end:.1f}s' if t_end > 0 else f'{t_start:.1f}s - end'
        else:
            cats = self._get_selected_categories()
            meta['trigger_source'] = f'Event Markers: {", ".join(cats)}' if cats else 'Event Markers'

        # Condition mode
        meta['condition_mode'] = self._condition_mode_combo.currentText()

        # Selected conditions
        selected = self._get_selected_conditions()
        if selected:
            meta['conditions'] = selected

        # Selected metrics
        metrics = self._get_selected_metrics()
        if metrics:
            meta['metrics'] = metrics

        return meta

    # -------------------------------------------------------------------------
    # Breath Info
    # -------------------------------------------------------------------------

    def _update_breath_info(self):
        """Update the breath info label showing data availability for current time window."""
        if not hasattr(self, '_breath_info_label'):
            return

        if self._breath_data is None:
            self._breath_info_label.setText("No breath data available")
            return

        all_peaks = self._breath_data.get('all_peaks', {})
        peak_indices = all_peaks.get('indices', np.array([]))
        labels = all_peaks.get('labels', np.array([]))

        n_total = len(peak_indices)
        if n_total == 0:
            self._breath_info_label.setText("No peaks detected")
            return

        # Apply time window filter
        time_mask = np.ones(n_total, dtype=bool)
        if self._time_array is not None and len(peak_indices) > 0:
            t_start = self._breath_time_start.value()
            t_end = self._breath_time_end.value()
            # Convert peak sample indices to times
            peak_times = np.array([
                self._time_array[int(idx)] if 0 <= int(idx) < len(self._time_array) else 0
                for idx in peak_indices
            ])
            if t_start > 0:
                time_mask &= (peak_times >= t_start)
            if t_end > 0:
                time_mask &= (peak_times <= t_end)

        # Count breath vs noise within time window
        breath_mask = (labels == 1) & time_mask if len(labels) == n_total else time_mask.copy()
        n_breaths = int(np.sum(breath_mask))

        # Count breath types if available
        breath_type_class = all_peaks.get('breath_type_class', np.array([]))
        info_parts = [f"{n_breaths} breaths"]
        if len(breath_type_class) == n_total:
            n_eupnea = int(np.sum((breath_type_class == 0) & breath_mask))
            n_sniffing = int(np.sum((breath_type_class == 1) & breath_mask))
            info_parts.append(f"({n_eupnea} eup, {n_sniffing} sniff)")

        # Count sighs in window
        sigh_indices = self._breath_data.get('sigh_indices', np.array([]))
        if len(sigh_indices) > 0:
            if self._time_array is not None:
                sigh_times = np.array([
                    self._time_array[int(idx)] if 0 <= int(idx) < len(self._time_array) else 0
                    for idx in sigh_indices
                ])
                t_start = self._breath_time_start.value()
                t_end = self._breath_time_end.value()
                sigh_mask = np.ones(len(sigh_indices), dtype=bool)
                if t_start > 0:
                    sigh_mask &= (sigh_times >= t_start)
                if t_end > 0:
                    sigh_mask &= (sigh_times <= t_end)
                n_sighs = int(np.sum(sigh_mask))
            else:
                n_sighs = len(sigh_indices)
            info_parts.append(f", {n_sighs} sighs")

        window_note = ""
        t_s = self._breath_time_start.value()
        t_e = self._breath_time_end.value()
        if t_s > 0 or t_e > 0:
            end_str = f"{t_e:.0f}" if t_e > 0 else "..."
            window_note = f" in [{t_s:.0f}-{end_str}s]"

        self._breath_info_label.setText(" ".join(info_parts) + window_note)

    # -------------------------------------------------------------------------
    # Preview Plots
    # -------------------------------------------------------------------------

    def _update_preview_plots(self):
        """Update preview plots with CTA results."""
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        collection = self._viewmodel.current_collection
        has_condition_data = bool(self._viewmodel.condition_collections)
        if (not collection or not collection.results) and not has_condition_data:
            label = QLabel("No CTA results to display")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_layout.addWidget(label)
            return

        try:
            self._create_matplotlib_preview()
        except Exception as e:
            label = QLabel(f"Error creating preview: {e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_layout.addWidget(label)
            print(f"[CTA Preview] Error: {e}")
            import traceback
            traceback.print_exc()

    def _create_matplotlib_preview(self):
        """Create matplotlib preview of CTA results -- dispatches by condition mode."""
        mode = self._viewmodel.condition_mode

        if mode == 'overlay':
            self._create_overlay_preview()
        elif mode == 'separate':
            self._create_separate_preview()
        else:
            self._create_combined_preview()

    def _add_figure_to_layout(self, fig, n_metrics: int):
        """Add a matplotlib figure + toolbar to the plot layout."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        canvas = FigureCanvas(fig)
        canvas.setMinimumHeight(int(3.5 * max(1, n_metrics) * fig.dpi))
        canvas.setStyleSheet("background-color: #1e1e1e;")

        scroll_area = self._plot_tab
        def _forward_wheel(event, _sa=scroll_area):
            _sa.verticalScrollBar().setValue(
                _sa.verticalScrollBar().value() - event.angleDelta().y()
            )
        canvas.wheelEvent = _forward_wheel

        toolbar = NavigationToolbar2QT(canvas, self)
        toolbar.setStyleSheet("""
            QToolBar { background-color: #1e1e1e; border: none; spacing: 3px; padding: 0px; margin: 0px; }
            QToolButton {
                background-color: transparent; border: none;
                padding: 3px; color: #d4d4d4;
            }
            QToolButton:hover { background-color: #3d3d3d; }
            QToolButton:pressed { background-color: #5d5d5d; }
            QLabel { color: #d4d4d4; }
        """)

        container = QWidget()
        container.setStyleSheet("background-color: #1e1e1e; border: none;")
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addWidget(canvas)
        container_layout.addWidget(toolbar)

        self._plot_layout.addWidget(container)
        canvas._figure = fig

    def _get_metrics_and_types(self, collection):
        """Get ordered metrics and marker types for a collection."""
        # When breath events are active, ignore the marker list — read from collection
        if self._radio_breath_events.isChecked():
            types = set()
            for r in collection.results.values():
                types.add(f"{r.category}:{r.label}")
            return sorted(types)

        marker_types = self._get_selected_categories()
        if not marker_types:
            types = set()
            for r in collection.results.values():
                types.add(f"{r.category}:{r.label}")
            marker_types = sorted(types)
        return marker_types

    def _create_combined_preview(self):
        """Create combined CTA preview (original behavior)."""
        import matplotlib.pyplot as plt

        collection = self._viewmodel.current_collection
        if not collection:
            return

        is_zscored = self._viewmodel.config.zscore_baseline
        marker_types = self._get_metrics_and_types(collection)

        for marker_type in marker_types:
            category, label = marker_type.split(":", 1)
            type_results = collection.get_results_for_marker_type(category, label)
            if not type_results:
                continue

            selected_metrics_ordered = self._get_selected_metrics()
            available_metrics = set(r.metric_key for r in type_results.values())
            metrics = [m for m in selected_metrics_ordered if m in available_metrics]
            if not metrics:
                metrics = list(available_metrics)
            n_metrics = len(metrics)

            has_withdrawal = any(r.alignment == 'withdrawal' for r in type_results.values())
            n_cols = 2 if has_withdrawal else 1

            fig, axes = plt.subplots(
                n_metrics, n_cols,
                figsize=(5 * n_cols, 3.5 * n_metrics),
                squeeze=False, facecolor='#1e1e1e'
            )
            fig.suptitle(f"CTA: {marker_type}", fontsize=12, fontweight='bold',
                         color='#e0e0e0', y=1.0 - 0.01 / max(1, n_metrics))

            show_paired = self._chk_show_paired_lines.isChecked()
            for idx, metric_key in enumerate(metrics):
                metric_color = self._metric_colors.get(metric_key, QColor('#4488ff')).name()

                onset_key = f"{category}:{label}:onset:{metric_key}"
                onset_result = type_results.get(onset_key)
                withdrawal_key = f"{category}:{label}:withdrawal:{metric_key}"
                withdrawal_result = type_results.get(withdrawal_key)

                ax_onset = axes[idx, 0]
                if onset_result and onset_result.traces:
                    self._plot_cta_on_axis(ax_onset, onset_result, 'Onset', metric_color, is_zscored, show_paired)
                else:
                    ax_onset.text(0.5, 0.5, 'No onset data', ha='center', va='center',
                                  transform=ax_onset.transAxes, color='#888')
                    ax_onset.set_facecolor('#252525')
                    ax_onset.set_title(
                        f"{onset_result.metric_label if onset_result else metric_key} - Onset CTA",
                        color='#e0e0e0'
                    )

                if has_withdrawal:
                    ax_withdrawal = axes[idx, 1]
                    if withdrawal_result and withdrawal_result.traces:
                        self._plot_cta_on_axis(ax_withdrawal, withdrawal_result, 'Withdrawal', metric_color, is_zscored, show_paired)
                    else:
                        ax_withdrawal.text(0.5, 0.5, 'No withdrawal data', ha='center', va='center',
                                           transform=ax_withdrawal.transAxes, color='#888')
                        ax_withdrawal.set_facecolor('#252525')
                        ax_withdrawal.set_title(f"{metric_key} - Withdrawal CTA", color='#e0e0e0')

            fig.tight_layout(h_pad=3.0, w_pad=2.0, rect=[0, 0, 1, 0.97])
            self._add_figure_to_layout(fig, n_metrics)

    def _create_separate_preview(self):
        """Create separate CTA preview -- one figure per condition."""
        import matplotlib.pyplot as plt

        condition_collections = self._viewmodel.condition_collections
        if not condition_collections:
            return

        is_zscored = self._viewmodel.config.zscore_baseline
        show_paired = self._chk_show_paired_lines.isChecked()

        for cond_name, collection in sorted(condition_collections.items()):
            marker_types = self._get_metrics_and_types(collection)

            for marker_type in marker_types:
                category, label = marker_type.split(":", 1)
                type_results = collection.get_results_for_marker_type(category, label)
                if not type_results:
                    continue

                selected_metrics_ordered = self._get_selected_metrics()
                available_metrics = set(r.metric_key for r in type_results.values())
                metrics = [m for m in selected_metrics_ordered if m in available_metrics]
                if not metrics:
                    metrics = list(available_metrics)
                n_metrics = len(metrics)

                has_withdrawal = any(r.alignment == 'withdrawal' for r in type_results.values())
                n_cols = 2 if has_withdrawal else 1

                fig, axes = plt.subplots(
                    n_metrics, n_cols,
                    figsize=(5 * n_cols, 3.5 * n_metrics),
                    squeeze=False, facecolor='#1e1e1e'
                )
                fig.suptitle(f"CTA: {marker_type} [{cond_name}]", fontsize=12,
                             fontweight='bold', color='#e0e0e0',
                             y=1.0 - 0.01 / max(1, n_metrics))

                for idx, metric_key in enumerate(metrics):
                    metric_color = self._metric_colors.get(metric_key, QColor('#4488ff')).name()

                    onset_result = type_results.get(f"{category}:{label}:onset:{metric_key}")
                    withdrawal_result = type_results.get(f"{category}:{label}:withdrawal:{metric_key}")

                    ax_onset = axes[idx, 0]
                    if onset_result and onset_result.traces:
                        self._plot_cta_on_axis(ax_onset, onset_result, 'Onset', metric_color, is_zscored, show_paired)
                    else:
                        ax_onset.set_facecolor('#252525')
                        ax_onset.set_title(f"{metric_key} - Onset CTA", color='#e0e0e0')
                        ax_onset.text(0.5, 0.5, 'No data', ha='center', va='center',
                                      transform=ax_onset.transAxes, color='#888')

                    if has_withdrawal:
                        ax_w = axes[idx, 1]
                        if withdrawal_result and withdrawal_result.traces:
                            self._plot_cta_on_axis(ax_w, withdrawal_result, 'Withdrawal', metric_color, is_zscored, show_paired)
                        else:
                            ax_w.set_facecolor('#252525')
                            ax_w.set_title(f"{metric_key} - Withdrawal CTA", color='#e0e0e0')
                            ax_w.text(0.5, 0.5, 'No data', ha='center', va='center',
                                      transform=ax_w.transAxes, color='#888')

                fig.tight_layout(h_pad=3.0, w_pad=2.0, rect=[0, 0, 1, 0.97])
                self._add_figure_to_layout(fig, n_metrics)

    def _plot_overlay_condition(self, ax, result, cond_color: str, cond_name: str,
                                show_paired: bool = True):
        """Plot one condition's CTA data on an overlay axis."""
        if not result or not result.traces:
            return
        # Individual traces
        for trace in result.traces:
            ax.plot(trace.time, trace.values, color=cond_color, alpha=0.12, linewidth=0.5)
        # Mean + SEM
        if result.time_common is not None and result.mean is not None:
            ax.plot(result.time_common, result.mean, color=cond_color, linewidth=2,
                    label=f'{cond_name} (n={result.n_events})')
            if result.sem is not None:
                ax.fill_between(
                    result.time_common,
                    result.mean - result.sem, result.mean + result.sem,
                    alpha=0.2, color=cond_color,
                )
        # Paired marker lines
        if show_paired:
            added = False
            for trace in result.traces:
                if trace.paired_event_offset is not None:
                    ax.axvline(trace.paired_event_offset, color=cond_color,
                               alpha=0.15, linewidth=1.0, linestyle=':',
                               label=f'{cond_name} withdrawal' if not added else None)
                    added = True

    def _create_overlay_preview(self):
        """Create overlay CTA preview -- conditions on same axes, different colors."""
        import matplotlib.pyplot as plt

        condition_collections = self._viewmodel.condition_collections
        if not condition_collections:
            return

        is_zscored = self._viewmodel.config.zscore_baseline
        show_paired = self._chk_show_paired_lines.isChecked()

        # Assign a color per condition — use user's metric colors when only
        # one metric is selected; use condition palette when multiple metrics
        # would cause color ambiguity
        _CONDITION_COLORS = [
            '#4488ff', '#ff6644', '#44cc66', '#cc44ff', '#ffcc44',
            '#44cccc', '#ff44aa', '#88cc44', '#ff8844', '#4466ff',
        ]
        cond_names = sorted(condition_collections.keys())
        cond_color_map = {
            name: _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            for i, name in enumerate(cond_names)
        }
        # Build per-metric condition color maps: if the user set a metric color,
        # derive condition variants from it (darker/lighter) so the user's
        # color choice is respected while conditions remain distinguishable
        _metric_cond_colors = {}  # {metric_key: {cond_name: color_str}}
        for mk in self._get_selected_metrics():
            user_color = self._metric_colors.get(mk)
            if user_color is not None:
                from PyQt6.QtGui import QColor
                base = QColor(user_color) if isinstance(user_color, str) else user_color
                h, s, l, _ = base.getHslF()
                per_cond = {}
                for ci, cn in enumerate(cond_names):
                    # Shift lightness per condition: first=original, then lighter/darker
                    if len(cond_names) == 1:
                        per_cond[cn] = base.name()
                    else:
                        shift = (ci / max(len(cond_names) - 1, 1)) * 0.4 - 0.2  # -0.2 to +0.2
                        new_l = max(0.15, min(0.85, l + shift))
                        c = QColor.fromHslF(h, min(s * 1.1, 1.0), new_l)
                        per_cond[cn] = c.name()
                _metric_cond_colors[mk] = per_cond

        # Build "All combined" collection if there are multiple conditions
        all_combined = None
        if len(condition_collections) > 1:
            # Merge all results by recalculating from combined markers
            # We already have per-condition results, so merge traces manually
            all_combined = self._build_combined_collection(condition_collections)

        # Gather all marker types across all conditions
        all_marker_types = set()
        for collection in condition_collections.values():
            for r in collection.results.values():
                all_marker_types.add(f"{r.category}:{r.label}")
        selected_cats = self._get_selected_categories()
        marker_types = sorted(selected_cats if selected_cats else all_marker_types)

        for marker_type in marker_types:
            category, label = marker_type.split(":", 1)

            # Gather metrics across all conditions for this marker type
            all_available = set()
            has_withdrawal = False
            for collection in condition_collections.values():
                type_results = collection.get_results_for_marker_type(category, label)
                for r in type_results.values():
                    all_available.add(r.metric_key)
                    if r.alignment == 'withdrawal':
                        has_withdrawal = True

            selected_metrics_ordered = self._get_selected_metrics()
            metrics = [m for m in selected_metrics_ordered if m in all_available]
            if not metrics:
                metrics = sorted(all_available)
            n_metrics = len(metrics)
            if n_metrics == 0:
                continue

            n_cols = 2 if has_withdrawal else 1

            fig, axes = plt.subplots(
                n_metrics, n_cols,
                figsize=(5 * n_cols, 3.5 * n_metrics),
                squeeze=False, facecolor='#1e1e1e'
            )
            fig.suptitle(f"CTA: {marker_type} (Overlay)", fontsize=12,
                         fontweight='bold', color='#e0e0e0',
                         y=1.0 - 0.01 / max(1, n_metrics))

            for idx, metric_key in enumerate(metrics):
                ax_onset = axes[idx, 0]
                ax_onset.set_facecolor('#252525')
                ax_onset.tick_params(colors='#d4d4d4', which='both')
                for spine in ax_onset.spines.values():
                    spine.set_color('#3a3a3a')
                self._draw_baseline_zone(ax_onset)
                self._draw_stim_zone(ax_onset)
                ax_onset.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5)

                if has_withdrawal:
                    ax_w = axes[idx, 1]
                    ax_w.set_facecolor('#252525')
                    ax_w.tick_params(colors='#d4d4d4', which='both')
                    for spine in ax_w.spines.values():
                        spine.set_color('#3a3a3a')
                    self._draw_baseline_zone(ax_w)
                    self._draw_stim_zone(ax_w)
                    ax_w.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5)

                metric_label_text = metric_key

                # Plot "All combined" first (behind per-condition traces) in gray
                if all_combined:
                    combined_results = all_combined.get_results_for_marker_type(category, label)
                    onset_r = combined_results.get(f"{category}:{label}:onset:{metric_key}")
                    if onset_r:
                        metric_label_text = onset_r.metric_label
                        self._plot_overlay_condition(ax_onset, onset_r, '#888888', 'All combined', show_paired)
                    if has_withdrawal:
                        w_r = combined_results.get(f"{category}:{label}:withdrawal:{metric_key}")
                        if w_r:
                            self._plot_overlay_condition(ax_w, w_r, '#888888', 'All combined', show_paired)

                # Plot per-condition traces on top
                for cond_name in cond_names:
                    collection = condition_collections[cond_name]
                    type_results = collection.get_results_for_marker_type(category, label)
                    # Use per-metric condition color if available, fall back to palette
                    if metric_key in _metric_cond_colors and cond_name in _metric_cond_colors[metric_key]:
                        cond_color = _metric_cond_colors[metric_key][cond_name]
                    else:
                        cond_color = cond_color_map[cond_name]

                    onset_result = type_results.get(f"{category}:{label}:onset:{metric_key}")
                    if onset_result:
                        metric_label_text = onset_result.metric_label
                    self._plot_overlay_condition(ax_onset, onset_result, cond_color, cond_name, show_paired)

                    if has_withdrawal:
                        w_result = type_results.get(f"{category}:{label}:withdrawal:{metric_key}")
                        self._plot_overlay_condition(ax_w, w_result, cond_color, cond_name, show_paired)

                ylabel = metric_label_text
                if is_zscored:
                    ylabel = f'{ylabel} (z-scored)'

                ax_onset.set_xlabel('Time from Onset (s)', color='#d4d4d4')
                ax_onset.set_ylabel(ylabel, color='#d4d4d4')
                ax_onset.set_title(f'{metric_label_text} - Onset CTA', color='#e0e0e0')
                ax_onset.legend(fontsize=8, loc='best', facecolor='#2a2a2a',
                                edgecolor='#3a3a3a', labelcolor='#d4d4d4')
                ax_onset.grid(True, alpha=0.2, color='#3a3a3a')

                if has_withdrawal:
                    ax_w.set_xlabel('Time from Withdrawal (s)', color='#d4d4d4')
                    ax_w.set_ylabel(ylabel, color='#d4d4d4')
                    ax_w.set_title(f'{metric_label_text} - Withdrawal CTA', color='#e0e0e0')
                    ax_w.legend(fontsize=8, loc='best', facecolor='#2a2a2a',
                                edgecolor='#3a3a3a', labelcolor='#d4d4d4')
                    ax_w.grid(True, alpha=0.2, color='#3a3a3a')

            fig.tight_layout(h_pad=3.0, w_pad=2.0, rect=[0, 0, 1, 0.97])
            self._add_figure_to_layout(fig, n_metrics)

    def _build_combined_collection(self, condition_collections):
        """Merge per-condition collections into a single 'All combined' collection."""
        from core.domain.cta import CTACollection, CTAResult
        combined = CTACollection(config=self._viewmodel.config)

        # Gather all result keys across conditions
        all_keys = set()
        for coll in condition_collections.values():
            all_keys.update(coll.results.keys())

        for key in all_keys:
            # Merge traces from all conditions for this key
            merged_traces = []
            template_result = None
            for coll in condition_collections.values():
                r = coll.results.get(key)
                if r:
                    template_result = r
                    merged_traces.extend(r.traces)

            if not template_result or not merged_traces:
                continue

            # Recompute mean/SEM from merged traces
            time_common = template_result.time_common
            if time_common is not None and len(merged_traces) > 0:
                from scipy import interpolate as _interp
                # Interpolate all traces onto common time base
                all_values = []
                for trace in merged_traces:
                    if len(trace.time) < 2:
                        continue
                    f = _interp.interp1d(trace.time, trace.values, kind='linear',
                                         bounds_error=False, fill_value=np.nan)
                    all_values.append(f(time_common))

                if all_values:
                    stacked = np.array(all_values)
                    mean = np.nanmean(stacked, axis=0)
                    sem = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(len(all_values)) if len(all_values) > 1 else np.zeros_like(mean)
                else:
                    mean = template_result.mean
                    sem = template_result.sem

                merged_result = CTAResult(
                    metric_key=template_result.metric_key,
                    metric_label=template_result.metric_label,
                    alignment=template_result.alignment,
                    category=template_result.category,
                    label=template_result.label,
                    config=template_result.config,
                    traces=merged_traces,
                    time_common=time_common,
                    mean=mean,
                    sem=sem,
                    n_events=len(merged_traces),
                )
                combined.add_result(merged_result)

        return combined

    def _get_stim_spans(self) -> dict:
        """Get stim spans from parent MainWindow's state."""
        mw = self.window()
        if mw and hasattr(mw, 'state'):
            return getattr(mw.state, 'stim_spans_by_sweep', {})
        return {}

    def _build_stim_markers(self) -> list:
        """Convert detected stim spans into EventMarker objects for CTA calculation."""
        from core.domain.events.models import EventMarker
        spans = self._get_stim_spans()
        markers = []
        for sweep_idx, sweep_spans in spans.items():
            for i, (start, end) in enumerate(sweep_spans):
                markers.append(EventMarker(
                    start_time=start,
                    end_time=end,
                    sweep_idx=sweep_idx,
                    category='stimulus',
                    label='laser',
                    is_paired=True,
                ))
        return markers

    def _get_stim_duration(self) -> Optional[float]:
        """Get stim duration from the parent MainWindow's state (seconds)."""
        mw = self.window()
        if mw and hasattr(mw, 'state'):
            spans = getattr(mw.state, 'stim_spans_by_sweep', {})
            if spans:
                # Use first sweep's first span
                first_sweep_spans = next(iter(spans.values()), [])
                if first_sweep_spans:
                    start, end = first_sweep_spans[0]
                    return float(end - start)
        return None

    def _draw_stim_zone(self, ax, event_time_offset: float = 0.0):
        """Draw stim on/off shading on a CTA plot.

        For stim-triggered CTAs (event = stim onset), stim starts at t=0.
        For other triggers, we'd need per-event stim offsets (future work).
        For now, draws a shaded region from 0 to stim_duration.
        """
        if not self._chk_show_stim.isChecked():
            return
        stim_dur = self._get_stim_duration()
        if stim_dur is None or stim_dur <= 0:
            return
        ax.axvspan(
            event_time_offset, event_time_offset + stim_dur,
            alpha=0.08, color='#4488ff', zorder=0,
            label='Stim on',
        )

    def _draw_baseline_zone(self, ax):
        """Draw the z-score baseline zone as a shaded region if enabled."""
        if not self._chk_show_baseline.isChecked():
            return
        config = self._viewmodel.config
        if not config.zscore_baseline:
            return
        ax.axvspan(
            config.baseline_start, config.baseline_end,
            alpha=0.12, color='#44aaff', zorder=0,
            label='Baseline zone',
        )

    def _plot_cta_on_axis(self, ax, result, alignment_label: str, color: str,
                          is_zscored: bool = False, show_paired_lines: bool = True):
        """Plot a single CTA result on the given axis."""
        ax.set_facecolor('#252525')
        ax.tick_params(colors='#d4d4d4', which='both')
        for spine in ax.spines.values():
            spine.set_color('#3a3a3a')

        # Baseline zone and stim zone (behind data)
        self._draw_baseline_zone(ax)
        self._draw_stim_zone(ax)

        for trace in result.traces:
            ax.plot(trace.time, trace.values, color=color, alpha=0.2, linewidth=0.5)

        if result.time_common is not None and result.mean is not None:
            ax.plot(result.time_common, result.mean, color=color, linewidth=2,
                    label=f'Mean (n={result.n_events})')
            if result.sem is not None:
                ax.fill_between(
                    result.time_common,
                    result.mean - result.sem,
                    result.mean + result.sem,
                    alpha=0.3, color=color, label='SEM'
                )

        ax.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5, label=alignment_label)

        # Draw paired event marker lines (e.g., withdrawal lines on onset CTA)
        if show_paired_lines:
            paired_label = 'Withdrawal' if alignment_label == 'Onset' else 'Onset'
            added_legend = False
            for trace in result.traces:
                if trace.paired_event_offset is not None:
                    ax.axvline(
                        trace.paired_event_offset,
                        color='#ff5555', alpha=0.15, linewidth=1.0, linestyle=':',
                        label=paired_label if not added_legend else None,
                    )
                    added_legend = True

        ax.set_xlabel(f'Time from {alignment_label} (s)', color='#d4d4d4')

        ylabel = result.metric_label
        if is_zscored:
            ylabel = f'{ylabel} (z-scored)'
        ax.set_ylabel(ylabel, color='#d4d4d4')

        ax.set_title(f'{result.metric_label} - {alignment_label} CTA', color='#e0e0e0')
        ax.legend(fontsize=8, loc='best', facecolor='#2a2a2a', edgecolor='#3a3a3a', labelcolor='#d4d4d4')
        ax.grid(True, alpha=0.2, color='#3a3a3a')

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_data(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
    ):
        """Set or update the data for CTA calculation."""
        self._markers = markers
        self._signals = signals
        self._time_array = time_array
        self._metric_labels = metric_labels or {}

        if metric_labels:
            self._viewmodel.set_available_metrics(metric_labels)

        self._populate_marker_categories()
        self._populate_metrics()

    def refresh_data(self):
        """Re-populate UI from current data (call after external data changes)."""
        self._populate_marker_categories()
        self._populate_metrics()
        self._update_breath_info()

    # -------------------------------------------------------------------------
    # Config Serialization (for workspace round-trip save/restore)
    # -------------------------------------------------------------------------

    def get_config(self) -> dict:
        """Serialize all widget state + viewmodel collections for save."""
        from core.domain.cta.models import CTATabConfig

        config = CTATabConfig(
            tab_name=self.tab_name,
            trigger_source='breath_events' if self._radio_breath_events.isChecked() else 'event_markers',
            breath_trigger_point=self._breath_trigger_combo.currentData() or 'inspiratory_onset',
            breath_type_filter=self._breath_type_combo.currentData() or 'all',
            breath_max_events=self._breath_max_events.value(),
            breath_min_bout=self._breath_min_bout.value(),
            breath_time_start=self._breath_time_start.value(),
            breath_time_end=self._breath_time_end.value(),
            window_before=self._spin_before.value(),
            window_after=self._spin_after.value(),
            zscore_enabled=self._zscore_group.isChecked(),
            baseline_start=self._spin_baseline_start.value(),
            baseline_end=self._spin_baseline_end.value(),
            condition_mode=self._condition_mode_combo.currentData() or 'overlay',
            selected_categories=self._get_selected_categories(),
            selected_conditions=self._get_selected_conditions() or [],
            selected_metrics=self._get_selected_metrics(),
            metric_colors={k: v.name() for k, v in self._metric_colors.items()},
            show_paired_lines=self._chk_show_paired_lines.isChecked(),
            show_baseline_zone=self._chk_show_baseline.isChecked(),
            show_stim_period=self._chk_show_stim.isChecked(),
        ).to_dict()

        # Add CTA results (collection + condition_collections)
        result = {'config': config}

        vm = self._viewmodel
        if vm.current_collection and vm.current_collection.results:
            result['collection'] = vm.current_collection.to_dict()

        if vm.condition_collections:
            result['condition_collections'] = {
                cond: coll.to_dict()
                for cond, coll in vm.condition_collections.items()
            }

        result['condition_mode_actual'] = vm.condition_mode

        return result

    def apply_config(self, tab_data: dict) -> None:
        """Restore widget state from a saved config dict and render plots."""
        from core.domain.cta.models import CTATabConfig, CTACollection

        config_dict = tab_data.get('config', {})
        cfg = CTATabConfig.from_dict(config_dict)

        # --- Block signals during restore ---
        widgets_to_block = [
            self._spin_before, self._spin_after,
            self._spin_baseline_start, self._spin_baseline_end,
            self._zscore_group,
            self._condition_mode_combo,
            self._breath_trigger_combo, self._breath_type_combo,
            self._breath_max_events, self._breath_min_bout,
            self._breath_time_start, self._breath_time_end,
            self._chk_show_paired_lines, self._chk_show_baseline,
            self._radio_event_markers, self._radio_breath_events,
        ]
        for w in widgets_to_block:
            w.blockSignals(True)

        try:
            # Trigger source
            if cfg.trigger_source == 'breath_events':
                self._radio_breath_events.setChecked(True)
                self._trigger_stack.setCurrentIndex(1)
                self._condition_group.setVisible(False)
            else:
                self._radio_event_markers.setChecked(True)
                self._trigger_stack.setCurrentIndex(0)
                self._condition_group.setVisible(True)

            # Breath config
            idx = self._breath_trigger_combo.findData(cfg.breath_trigger_point)
            if idx >= 0:
                self._breath_trigger_combo.setCurrentIndex(idx)
            idx = self._breath_type_combo.findData(cfg.breath_type_filter)
            if idx >= 0:
                self._breath_type_combo.setCurrentIndex(idx)
            self._breath_max_events.setValue(cfg.breath_max_events)
            self._breath_min_bout.setValue(cfg.breath_min_bout)
            self._breath_time_start.setValue(cfg.breath_time_start)
            self._breath_time_end.setValue(cfg.breath_time_end)

            # Time windows
            self._spin_before.setValue(cfg.window_before)
            self._spin_after.setValue(cfg.window_after)

            # Z-score
            self._zscore_group.setChecked(cfg.zscore_enabled)
            self._spin_baseline_start.setValue(cfg.baseline_start)
            self._spin_baseline_end.setValue(cfg.baseline_end)

            # Condition mode
            idx = self._condition_mode_combo.findData(cfg.condition_mode)
            if idx >= 0:
                self._condition_mode_combo.setCurrentIndex(idx)

            # Update viewmodel config to match
            self._viewmodel.config.window_before = cfg.window_before
            self._viewmodel.config.window_after = cfg.window_after
            self._viewmodel.config.zscore_baseline = cfg.zscore_enabled
            self._viewmodel.config.baseline_start = cfg.baseline_start
            self._viewmodel.config.baseline_end = cfg.baseline_end

            # Selected categories (match by text against populated marker list)
            if cfg.selected_categories:
                saved_set = set(cfg.selected_categories)
                for i in range(self._marker_list.count()):
                    item = self._marker_list.item(i)
                    data = item.data(Qt.ItemDataRole.UserRole)
                    if data:
                        cat_str = f"{data['category']}:{data['label']}"
                    else:
                        cat_str = item.text()
                    item.setCheckState(
                        Qt.CheckState.Checked if cat_str in saved_set
                        else Qt.CheckState.Unchecked
                    )

            # Selected conditions (match by text)
            if cfg.selected_conditions:
                saved_set = set(cfg.selected_conditions)
                for i in range(self._condition_list.count()):
                    item = self._condition_list.item(i)
                    cond = item.data(Qt.ItemDataRole.UserRole)
                    if cond == '__all__':
                        item.setCheckState(Qt.CheckState.Unchecked)
                    elif cond and cond in saved_set:
                        item.setCheckState(Qt.CheckState.Checked)
                    elif cond:
                        item.setCheckState(Qt.CheckState.Unchecked)

            # Selected metrics (check/uncheck and apply colors)
            if cfg.selected_metrics:
                saved_set = set(cfg.selected_metrics)
                self._metrics_table.blockSignals(True)
                for row in range(self._metrics_table.rowCount()):
                    name_item = self._metrics_table.item(row, 1)
                    if name_item:
                        key = name_item.data(Qt.ItemDataRole.UserRole)
                        name_item.setCheckState(
                            Qt.CheckState.Checked if key in saved_set
                            else Qt.CheckState.Unchecked
                        )
                        # Restore color
                        if key in cfg.metric_colors:
                            color = QColor(cfg.metric_colors[key])
                            self._metric_colors[key] = color
                            color_item = self._metrics_table.item(row, 0)
                            if color_item:
                                color_item.setBackground(QBrush(color))
                self._metrics_table.blockSignals(False)
                self._update_metric_row_styles()

            # Display toggles
            self._chk_show_paired_lines.setChecked(cfg.show_paired_lines)
            self._chk_show_baseline.setChecked(cfg.show_baseline_zone)
            if hasattr(cfg, 'show_stim_period'):
                self._chk_show_stim.setChecked(cfg.show_stim_period)

        finally:
            for w in widgets_to_block:
                w.blockSignals(False)

        # --- Restore CTA results into viewmodel ---
        collection_dict = tab_data.get('collection')
        if collection_dict:
            try:
                self._viewmodel._current_collection = CTACollection.from_dict(collection_dict)
            except Exception as e:
                print(f"[CTA restore] Failed to restore collection: {e}")

        cond_colls = tab_data.get('condition_collections')
        if cond_colls:
            try:
                self._viewmodel._condition_collections = {
                    cond: CTACollection.from_dict(coll_dict)
                    for cond, coll_dict in cond_colls.items()
                }
            except Exception as e:
                print(f"[CTA restore] Failed to restore condition collections: {e}")

        actual_mode = tab_data.get('condition_mode_actual', cfg.condition_mode)
        self._viewmodel._condition_mode = actual_mode

        # Render plots
        if self._viewmodel.current_collection or self._viewmodel.condition_collections:
            self._update_preview_plots()
            self._btn_export_csv.setEnabled(True)
