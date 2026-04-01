"""
Photometry CTA (Condition-Triggered Average) Dialog.

Provides a UI for configuring CTA parameters, previewing results,
and exporting CTA data aligned to event markers.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QCheckBox, QComboBox,
    QPushButton, QGroupBox, QProgressBar, QFileDialog,
    QListWidget, QListWidgetItem, QSplitter, QTabWidget,
    QWidget, QMessageBox, QSizePolicy, QScrollArea, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QBrush

from viewmodels.cta_viewmodel import CTAViewModel
from core.domain.events import EventMarker

from dialogs.export_mixin import ExportMixin

# Default color palette for metrics
_METRIC_COLORS = [
    '#4488ff', '#44cc66', '#ff6644', '#cc44ff', '#ffcc44',
    '#44cccc', '#ff44aa', '#88cc44', '#ff8844', '#4466ff',
]


class PhotometryCTADialog(ExportMixin, QDialog):
    """
    Dialog for generating and previewing photometry CTAs.

    Allows users to:
    - Select which event marker categories to include
    - Configure time windows (before/after event)
    - Select which metrics to include (with per-metric colors)
    - Preview CTA plots
    - Export via matplotlib toolbar save icon
    """

    # Signals
    cta_generated = pyqtSignal()  # Emitted when CTA generation completes

    def __init__(
        self,
        parent=None,
        viewmodel: Optional[CTAViewModel] = None,
        markers: Optional[List[EventMarker]] = None,
        signals: Optional[Dict[str, np.ndarray]] = None,
        time_array: Optional[np.ndarray] = None,
        metric_labels: Optional[Dict[str, str]] = None,
        channel_colors: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Photometry CTA Generator")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        self.setSizeGripEnabled(True)
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )

        # Store data
        self._markers = markers or []
        self._signals = signals or {}
        self._time_array = time_array
        self._metric_labels = metric_labels or {}
        self._channel_colors = channel_colors or {}

        # Per-metric colors (assigned during populate)
        self._metric_colors: Dict[str, QColor] = {}
        self._active_metric_row: int = -1  # Tracked for up/down reordering

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

        # Apply dark theme
        self._apply_dark_theme()
        self.setup_export_menu()

    def _init_ui(self):
        """Build the dialog UI."""
        main_layout = QHBoxLayout(self)

        # Left panel: Configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(350)

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
        self._spin_baseline_start.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_start)

        zscore_layout.addWidget(QLabel("End (s):"))
        self._spin_baseline_end = QDoubleSpinBox()
        self._spin_baseline_end.setRange(-99999.9, 99999.9)
        self._spin_baseline_end.setSingleStep(0.25)
        self._spin_baseline_end.setDecimals(2)
        self._spin_baseline_end.setValue(self._viewmodel.config.baseline_end)
        self._spin_baseline_end.setToolTip("End of baseline period (0 = event onset)")
        self._spin_baseline_end.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_end)

        left_layout.addWidget(self._zscore_group)

        # --- Event Marker Categories (checkbox list) ---
        markers_group = QGroupBox("Event Marker Categories")
        markers_layout = QVBoxLayout(markers_group)

        self._marker_list = QListWidget()
        self._marker_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._marker_list.setMaximumHeight(120)
        markers_layout.addWidget(self._marker_list)

        left_layout.addWidget(markers_group)

        # --- Condition Mode ---
        condition_group = QGroupBox("Conditions")
        condition_layout = QVBoxLayout(condition_group)

        # Mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._condition_mode_combo = QComboBox()
        self._condition_mode_combo.addItem("Combined", "combined")
        self._condition_mode_combo.addItem("Separate", "separate")
        self._condition_mode_combo.addItem("Overlay", "overlay")
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
        self._condition_list.setMaximumHeight(80)
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
        self._metrics_table.setMaximumHeight(200)
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

        left_layout.addWidget(metrics_group)

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

        left_layout.addStretch()

        # Close button
        self._btn_close = QPushButton("Close")
        self._btn_close.setStyleSheet(
            "QPushButton { background-color: #6b1a1a; border: 1px solid #8a2a2a; }"
            "QPushButton:hover { background-color: #8b2222; border-color: #aa3333; }"
            "QPushButton:pressed { background-color: #4d0d0d; }"
        )
        self._btn_close.clicked.connect(self.close)
        left_layout.addWidget(self._btn_close)

        main_layout.addWidget(left_panel)

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

        # Populate condition filter
        self._populate_conditions()

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
            # since the last click — if so, skip the manual toggle to avoid double-toggle.
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
        categories = self._get_selected_categories()
        if not categories:
            QMessageBox.warning(self, "No Selection", "Please select at least one marker category.")
            return

        metrics = self._get_selected_metrics()
        if not metrics:
            QMessageBox.warning(self, "No Selection", "Please select at least one metric.")
            return

        if self._time_array is None or len(self._signals) == 0:
            QMessageBox.warning(self, "No Data", "No signal data available for CTA calculation.")
            return

        # Warn about large windows (traces are serialized to JSON — very large
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

        # Get condition mode
        condition_mode = self._condition_mode_combo.currentData() or "combined"

        self._viewmodel.generate_preview(
            markers=filtered_markers,
            signals=self._signals,
            time_array=self._time_array,
            metric_labels=self._metric_labels,
            condition_mode=condition_mode,
        )

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
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        # Build a smart default filename from the source data path
        default_name = "CTA_export.csv"
        default_dir = ""
        st = getattr(self, '_state', None) or getattr(self.parent(), 'state', None) if self.parent() else None
        if st and hasattr(st, 'in_path') and st.in_path:
            source_path = Path(st.in_path)
            default_dir = str(source_path.parent)
            default_name = f"{source_path.stem}_CTA.csv"

        start_path = str(Path(default_dir) / default_name) if default_dir else default_name

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CTA Data", start_path,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filepath:
            self._viewmodel.export_to_csv_wide(filepath)

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
        """Create matplotlib preview of CTA results — dispatches by condition mode."""
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
        """Create separate CTA preview — one figure per condition."""
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
        """Create overlay CTA preview — conditions on same axes, different colors."""
        import matplotlib.pyplot as plt

        condition_collections = self._viewmodel.condition_collections
        if not condition_collections:
            return

        is_zscored = self._viewmodel.config.zscore_baseline
        show_paired = self._chk_show_paired_lines.isChecked()

        # Assign a color per condition (+ white for "All combined")
        _CONDITION_COLORS = [
            '#4488ff', '#ff6644', '#44cc66', '#cc44ff', '#ffcc44',
            '#44cccc', '#ff44aa', '#88cc44', '#ff8844', '#4466ff',
        ]
        cond_names = sorted(condition_collections.keys())
        cond_color_map = {
            name: _CONDITION_COLORS[i % len(_CONDITION_COLORS)]
            for i, name in enumerate(cond_names)
        }

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
                ax_onset.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5)

                if has_withdrawal:
                    ax_w = axes[idx, 1]
                    ax_w.set_facecolor('#252525')
                    ax_w.tick_params(colors='#d4d4d4', which='both')
                    for spine in ax_w.spines.values():
                        spine.set_color('#3a3a3a')
                    self._draw_baseline_zone(ax_w)
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

        # Baseline zone (behind data)
        self._draw_baseline_zone(ax)

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
    # Theme
    # -------------------------------------------------------------------------

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QGroupBox {
                font-weight: bold; border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 8px; padding-top: 12px; background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #88aaff;
            }
            QGroupBox::indicator { width: 14px; height: 14px; }
            QGroupBox::indicator:unchecked {
                border: 1px solid #555; border-radius: 3px; background-color: #2a2a2a;
            }
            QGroupBox::indicator:checked {
                border: 1px solid #2a7fff; border-radius: 3px; background-color: #2a7fff;
            }
            QLabel { color: #e0e0e0; background-color: transparent; }
            QListWidget {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
            }
            QListWidget::item:selected { background-color: #2a7fff; color: white; }
            QTableWidget {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
                gridline-color: #3a3a3a;
            }
            QTableWidget::item:selected { background-color: transparent; color: #e0e0e0; }
            QHeaderView::section {
                background-color: #333; color: #ccc; border: 1px solid #3a3a3a;
                padding: 3px; font-size: 11px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px;
            }
            QCheckBox { color: #e0e0e0; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border: 1px solid #555;
                border-radius: 3px; background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked { background-color: #2a7fff; border-color: #2a7fff; }
            QPushButton {
                background-color: #3a3a3a; color: #e0e0e0;
                border: 1px solid #555; border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #2a7fff; }
            QPushButton:pressed { background-color: #2a7fff; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QProgressBar {
                border: 1px solid #3a3a3a; border-radius: 4px;
                background-color: #2a2a2a; text-align: center;
            }
            QProgressBar::chunk { background-color: #2a7fff; border-radius: 3px; }
            QScrollArea { background-color: #1e1e1e; border: none; }
            QScrollArea > QWidget > QWidget { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QToolBar { background-color: #2d2d2d; border: none; spacing: 3px; padding: 2px; }
            QToolButton {
                background-color: #3d3d3d; border: 1px solid #4d4d4d;
                border-radius: 3px; padding: 4px; color: #d4d4d4;
            }
            QToolButton:hover { background-color: #4d4d4d; border: 1px solid #5d5d5d; }
            QToolButton:pressed { background-color: #5d5d5d; }
            QToolButton:checked { background-color: #2a7fff; border-color: #2a7fff; }
        """)

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
