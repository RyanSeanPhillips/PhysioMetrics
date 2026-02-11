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
    QWidget, QMessageBox, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor

from viewmodels.cta_viewmodel import CTAViewModel
from core.domain.events import EventMarker

from dialogs.export_mixin import ExportMixin


class PhotometryCTADialog(ExportMixin, QDialog):
    """
    Dialog for generating and previewing photometry CTAs.

    Allows users to:
    - Select which event marker categories to include
    - Configure time windows (before/after event)
    - Select which metrics to include
    - Preview CTA plots
    - Export to CSV, NPZ, or PDF
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
    ):
        """
        Initialize the CTA dialog.

        Args:
            parent: Parent widget
            viewmodel: Optional CTAViewModel (creates one if not provided)
            markers: List of EventMarkers to use
            signals: Dictionary of signal arrays keyed by metric name
            time_array: Time array for signals
            metric_labels: Optional metric labels for display
        """
        super().__init__(parent)
        self.setWindowTitle("Photometry CTA Generator")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        # Make dialog resizable with size grip
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

        # Create or use provided ViewModel
        self._viewmodel = viewmodel or CTAViewModel(self)

        # Set available metrics
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

        # Populate marker categories
        self._populate_marker_categories()

        # Populate metrics
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

        # Window Configuration
        window_group = QGroupBox("Time Windows")
        window_layout = QGridLayout(window_group)

        # Window before
        window_layout.addWidget(QLabel("Before event (s):"), 0, 0)
        self._spin_before = QDoubleSpinBox()
        self._spin_before.setRange(0.1, 30.0)
        self._spin_before.setValue(self._viewmodel.window_before)
        self._spin_before.setSingleStep(0.5)
        self._spin_before.setDecimals(1)
        self._spin_before.valueChanged.connect(self._on_window_changed)
        window_layout.addWidget(self._spin_before, 0, 1)

        # Window after
        window_layout.addWidget(QLabel("After event (s):"), 1, 0)
        self._spin_after = QDoubleSpinBox()
        self._spin_after.setRange(0.1, 30.0)
        self._spin_after.setValue(self._viewmodel.window_after)
        self._spin_after.setSingleStep(0.5)
        self._spin_after.setDecimals(1)
        self._spin_after.valueChanged.connect(self._on_window_changed)
        window_layout.addWidget(self._spin_after, 1, 1)

        # Include withdrawal
        self._chk_withdrawal = QCheckBox("Include withdrawal CTAs")
        self._chk_withdrawal.setChecked(self._viewmodel.include_withdrawal)
        self._chk_withdrawal.setToolTip(
            "For paired markers (with start and end times),\n"
            "also generate CTAs aligned to the end/withdrawal time"
        )
        self._chk_withdrawal.toggled.connect(self._on_withdrawal_changed)
        window_layout.addWidget(self._chk_withdrawal, 2, 0, 1, 2)

        left_layout.addWidget(window_group)

        # Z-Score Baseline Configuration
        zscore_group = QGroupBox("Z-Score Normalization")
        zscore_layout = QGridLayout(zscore_group)

        # Enable z-score checkbox
        self._chk_zscore = QCheckBox("Z-score to baseline")
        self._chk_zscore.setChecked(self._viewmodel.config.zscore_baseline)
        self._chk_zscore.setToolTip(
            "Normalize each trial to its baseline period.\n"
            "z = (signal - baseline_mean) / baseline_std\n\n"
            "Essential for comparing across trials with different baseline levels."
        )
        self._chk_zscore.toggled.connect(self._on_zscore_changed)
        zscore_layout.addWidget(self._chk_zscore, 0, 0, 1, 2)

        # Baseline start
        zscore_layout.addWidget(QLabel("Baseline start (s):"), 1, 0)
        self._spin_baseline_start = QDoubleSpinBox()
        self._spin_baseline_start.setRange(-30.0, 0.0)
        self._spin_baseline_start.setValue(self._viewmodel.config.baseline_start)
        self._spin_baseline_start.setSingleStep(0.5)
        self._spin_baseline_start.setDecimals(1)
        self._spin_baseline_start.setToolTip("Start of baseline period (relative to event, negative = before)")
        self._spin_baseline_start.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_start, 1, 1)

        # Baseline end
        zscore_layout.addWidget(QLabel("Baseline end (s):"), 2, 0)
        self._spin_baseline_end = QDoubleSpinBox()
        self._spin_baseline_end.setRange(-30.0, 30.0)
        self._spin_baseline_end.setValue(self._viewmodel.config.baseline_end)
        self._spin_baseline_end.setSingleStep(0.5)
        self._spin_baseline_end.setDecimals(1)
        self._spin_baseline_end.setToolTip("End of baseline period (typically 0 = event onset)")
        self._spin_baseline_end.valueChanged.connect(self._on_baseline_changed)
        zscore_layout.addWidget(self._spin_baseline_end, 2, 1)

        left_layout.addWidget(zscore_group)

        # Marker Categories
        markers_group = QGroupBox("Event Marker Categories")
        markers_layout = QVBoxLayout(markers_group)

        self._marker_list = QListWidget()
        self._marker_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._marker_list.setMaximumHeight(150)
        markers_layout.addWidget(self._marker_list)

        # Select all/none buttons
        btn_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all_markers)
        btn_select_none = QPushButton("Select None")
        btn_select_none.clicked.connect(self._select_no_markers)
        btn_layout.addWidget(btn_select_all)
        btn_layout.addWidget(btn_select_none)
        markers_layout.addLayout(btn_layout)

        # Move Up/Down buttons for reordering plot panel order
        order_layout = QHBoxLayout()
        btn_move_up = QPushButton("\u25B2 Move Up")
        btn_move_up.setToolTip("Move selected category up (changes plot panel order)")
        btn_move_up.clicked.connect(self._move_marker_up)
        btn_move_down = QPushButton("\u25BC Move Down")
        btn_move_down.setToolTip("Move selected category down (changes plot panel order)")
        btn_move_down.clicked.connect(self._move_marker_down)
        order_layout.addWidget(btn_move_up)
        order_layout.addWidget(btn_move_down)
        markers_layout.addLayout(order_layout)

        left_layout.addWidget(markers_group)

        # Metrics Selection
        metrics_group = QGroupBox("Metrics to Include")
        metrics_layout = QVBoxLayout(metrics_group)

        self._metrics_list = QListWidget()
        self._metrics_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._metrics_list.setMaximumHeight(200)
        metrics_layout.addWidget(self._metrics_list)

        # Select all/none buttons for metrics
        btn_layout2 = QHBoxLayout()
        btn_select_all_m = QPushButton("Select All")
        btn_select_all_m.clicked.connect(self._select_all_metrics)
        btn_select_none_m = QPushButton("Select None")
        btn_select_none_m.clicked.connect(self._select_no_metrics)
        btn_layout2.addWidget(btn_select_all_m)
        btn_layout2.addWidget(btn_select_none_m)
        metrics_layout.addLayout(btn_layout2)

        left_layout.addWidget(metrics_group)

        # Generate button
        self._btn_generate = QPushButton("Generate CTA Preview")
        self._btn_generate.setMinimumHeight(40)
        self._btn_generate.clicked.connect(self._on_generate_clicked)
        left_layout.addWidget(self._btn_generate)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        left_layout.addWidget(self._progress)

        # Export buttons
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)

        self._btn_export_csv = QPushButton("Export to CSV...")
        self._btn_export_csv.clicked.connect(self._on_export_csv)
        self._btn_export_csv.setEnabled(False)
        export_layout.addWidget(self._btn_export_csv)

        self._btn_export_npz = QPushButton("Export to NPZ...")
        self._btn_export_npz.clicked.connect(self._on_export_npz)
        self._btn_export_npz.setEnabled(False)
        export_layout.addWidget(self._btn_export_npz)

        self._btn_export_pdf = QPushButton("Export to PDF...")
        self._btn_export_pdf.clicked.connect(self._on_export_pdf)
        self._btn_export_pdf.setEnabled(False)
        export_layout.addWidget(self._btn_export_pdf)

        # Include individual traces checkbox
        self._chk_include_traces = QCheckBox("Include individual traces in export")
        self._chk_include_traces.setChecked(True)
        export_layout.addWidget(self._chk_include_traces)

        left_layout.addWidget(export_group)

        left_layout.addStretch()

        # Close button
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.close)
        left_layout.addWidget(self._btn_close)

        main_layout.addWidget(left_panel)

        # Right panel: Preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Plot area (scrollable)
        self._plot_tab = QScrollArea()
        self._plot_tab.setWidgetResizable(True)
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_tab.setWidget(self._plot_container)
        right_layout.addWidget(self._plot_tab)

        # Initial placeholder
        self._placeholder_label = QLabel("Configure options and click 'Generate CTA Preview'")
        self._placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_label.setStyleSheet("color: #888; font-size: 14px;")
        self._plot_layout.addWidget(self._placeholder_label)

        main_layout.addWidget(right_panel, 1)  # Stretch

    def _populate_marker_categories(self):
        """Populate the marker categories list from available markers."""
        self._marker_list.clear()

        # Get unique category:label combinations
        categories = set()
        for marker in self._markers:
            categories.add(f"{marker.category}:{marker.label}")

        for cat in sorted(categories):
            item = QListWidgetItem(cat)
            item.setSelected(True)  # Select all by default
            self._marker_list.addItem(item)

    def _populate_metrics(self):
        """Populate the metrics list."""
        self._metrics_list.clear()

        for key, label in self._metric_labels.items():
            item = QListWidgetItem(f"{label} ({key})")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setSelected(True)  # Select all by default
            self._metrics_list.addItem(item)

        # If no metric labels provided, add signals keys
        if not self._metric_labels and self._signals:
            for key in self._signals.keys():
                item = QListWidgetItem(key)
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setSelected(True)
                self._metrics_list.addItem(item)

    def _select_all_markers(self):
        """Select all marker categories."""
        for i in range(self._marker_list.count()):
            self._marker_list.item(i).setSelected(True)

    def _select_no_markers(self):
        """Deselect all marker categories."""
        for i in range(self._marker_list.count()):
            self._marker_list.item(i).setSelected(False)

    def _move_marker_up(self):
        """Move the current marker category up in the list."""
        row = self._marker_list.currentRow()
        if row <= 0:
            return
        item = self._marker_list.takeItem(row)
        was_selected = item.isSelected()
        self._marker_list.insertItem(row - 1, item)
        item.setSelected(was_selected)
        self._marker_list.setCurrentRow(row - 1)

    def _move_marker_down(self):
        """Move the current marker category down in the list."""
        row = self._marker_list.currentRow()
        if row < 0 or row >= self._marker_list.count() - 1:
            return
        item = self._marker_list.takeItem(row)
        was_selected = item.isSelected()
        self._marker_list.insertItem(row + 1, item)
        item.setSelected(was_selected)
        self._marker_list.setCurrentRow(row + 1)

    def _select_all_metrics(self):
        """Select all metrics."""
        for i in range(self._metrics_list.count()):
            self._metrics_list.item(i).setSelected(True)

    def _select_no_metrics(self):
        """Deselect all metrics."""
        for i in range(self._metrics_list.count()):
            self._metrics_list.item(i).setSelected(False)

    def _on_window_changed(self):
        """Handle window parameter changes."""
        self._viewmodel.window_before = self._spin_before.value()
        self._viewmodel.window_after = self._spin_after.value()

    def _on_withdrawal_changed(self, checked: bool):
        """Handle withdrawal checkbox change."""
        self._viewmodel.include_withdrawal = checked

    def _on_zscore_changed(self, checked: bool):
        """Handle z-score checkbox change."""
        self._viewmodel.config.zscore_baseline = checked
        # Enable/disable baseline spinboxes
        self._spin_baseline_start.setEnabled(checked)
        self._spin_baseline_end.setEnabled(checked)

    def _on_baseline_changed(self):
        """Handle baseline period change."""
        self._viewmodel.config.baseline_start = self._spin_baseline_start.value()
        self._viewmodel.config.baseline_end = self._spin_baseline_end.value()

    def _get_selected_categories(self) -> List[str]:
        """Get list of selected marker categories."""
        categories = []
        for i in range(self._marker_list.count()):
            item = self._marker_list.item(i)
            if item.isSelected():
                categories.append(item.text())
        return categories

    def _get_selected_metrics(self) -> List[str]:
        """Get list of selected metric keys."""
        metrics = []
        for i in range(self._metrics_list.count()):
            item = self._metrics_list.item(i)
            if item.isSelected():
                metrics.append(item.data(Qt.ItemDataRole.UserRole))
        return metrics

    def _on_generate_clicked(self):
        """Handle Generate button click."""
        # Validate selections
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

        # Update ViewModel selections
        self._viewmodel.selected_categories = categories
        self._viewmodel.selected_metrics = metrics

        # Generate preview
        self._viewmodel.generate_preview(
            markers=self._markers,
            signals=self._signals,
            time_array=self._time_array,
            metric_labels=self._metric_labels,
        )

    def _on_calculation_started(self):
        """Handle calculation start."""
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_generate.setEnabled(False)
        self._btn_generate.setText("Calculating...")

    def _on_calculation_progress(self, value: int):
        """Handle calculation progress update."""
        self._progress.setValue(value)

    def _on_calculation_complete(self):
        """Handle calculation completion."""
        self._progress.setVisible(False)
        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("Generate CTA Preview")

    def _on_preview_ready(self):
        """Handle preview ready - update the plots."""
        # Enable export buttons
        self._btn_export_csv.setEnabled(True)
        self._btn_export_npz.setEnabled(True)
        self._btn_export_pdf.setEnabled(True)

        # Debug: Print what's in the collection
        collection = self._viewmodel.current_collection
        if collection:
            print(f"[CTA Debug] Collection has {len(collection.results)} results")
            print(f"[CTA Debug] include_withdrawal setting: {self._viewmodel.config.include_withdrawal}")
            for key, result in collection.results.items():
                print(f"[CTA Debug]   {key}: alignment={result.alignment}, n_events={result.n_events}")

        # Generate preview plots
        self._update_preview_plots()

        # Emit signal
        self.cta_generated.emit()

    def _update_preview_plots(self):
        """Update the preview plots with CTA results."""
        # Clear existing plots
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        collection = self._viewmodel.current_collection
        if not collection or not collection.results:
            label = QLabel("No CTA results to display")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_layout.addWidget(label)
            return

        # Create matplotlib figure for preview
        try:
            self._create_matplotlib_preview()
        except Exception as e:
            label = QLabel(f"Error creating preview: {e}")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_layout.addWidget(label)
            print(f"[CTA Preview] Error: {e}")

    def _create_matplotlib_preview(self):
        """Create matplotlib preview of CTA results."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        collection = self._viewmodel.current_collection
        if not collection:
            return

        # Check if z-scoring is enabled for y-axis label
        is_zscored = self._viewmodel.config.zscore_baseline

        # Use UI list order for plot panel ordering (respects user reordering)
        marker_types = self._get_selected_categories()
        if not marker_types:
            marker_types = self._viewmodel.get_marker_types()

        for marker_type in marker_types:
            # Get results for this marker type
            category, label = marker_type.split(":", 1)
            type_results = collection.get_results_for_marker_type(category, label)

            if not type_results:
                continue

            # Determine layout
            metrics = list(set(r.metric_key for r in type_results.values()))
            n_metrics = len(metrics)

            # 2 columns: onset, withdrawal (if applicable) - NO histogram
            has_withdrawal = any(r.alignment == 'withdrawal' for r in type_results.values())
            n_cols = 2 if has_withdrawal else 1

            # Create figure with dark theme
            fig, axes = plt.subplots(
                n_metrics, n_cols,
                figsize=(5 * n_cols, 3 * n_metrics),
                squeeze=False,
                facecolor='#1e1e1e'
            )
            fig.suptitle(f"CTA: {marker_type}", fontsize=12, fontweight='bold', color='#e0e0e0')

            for idx, metric_key in enumerate(metrics):
                # Get onset result
                onset_key = f"{category}:{label}:onset:{metric_key}"
                onset_result = type_results.get(onset_key)

                # Get withdrawal result (if exists)
                withdrawal_key = f"{category}:{label}:withdrawal:{metric_key}"
                withdrawal_result = type_results.get(withdrawal_key)

                # Plot onset CTA
                ax_onset = axes[idx, 0]
                if onset_result and onset_result.traces:
                    self._plot_cta_on_axis(ax_onset, onset_result, 'Onset', 'blue', is_zscored)
                else:
                    ax_onset.text(0.5, 0.5, 'No onset data', ha='center', va='center',
                                  transform=ax_onset.transAxes)
                    ax_onset.set_title(f"{onset_result.metric_label if onset_result else metric_key} - Onset CTA")

                # Plot withdrawal CTA (if applicable)
                if has_withdrawal:
                    ax_withdrawal = axes[idx, 1]
                    if withdrawal_result and withdrawal_result.traces:
                        self._plot_cta_on_axis(ax_withdrawal, withdrawal_result, 'Withdrawal', 'green', is_zscored)
                    else:
                        ax_withdrawal.text(0.5, 0.5, 'No withdrawal data', ha='center', va='center',
                                           transform=ax_withdrawal.transAxes)
                        ax_withdrawal.set_title(f"{metric_key} - Withdrawal CTA")

            plt.tight_layout()

            # Add to layout with dark theme styling
            canvas = FigureCanvas(fig)
            canvas.setStyleSheet("background-color: #1e1e1e;")

            toolbar = NavigationToolbar2QT(canvas, self)
            # Style the toolbar for dark theme
            toolbar.setStyleSheet("""
                QToolBar {
                    background-color: #2d2d2d;
                    border: none;
                    spacing: 3px;
                    padding: 2px;
                }
                QToolButton {
                    background-color: #3d3d3d;
                    border: 1px solid #4d4d4d;
                    border-radius: 3px;
                    padding: 3px;
                    color: #d4d4d4;
                }
                QToolButton:hover {
                    background-color: #4d4d4d;
                    border: 1px solid #5d5d5d;
                }
                QToolButton:pressed {
                    background-color: #5d5d5d;
                }
                QLabel {
                    color: #d4d4d4;
                }
            """)

            container = QWidget()
            container.setStyleSheet("background-color: #1e1e1e; border: none;")
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(toolbar)
            container_layout.addWidget(canvas)

            self._plot_layout.addWidget(container)

            # Keep reference to prevent garbage collection
            canvas._figure = fig

    def _plot_cta_on_axis(self, ax, result, alignment_label: str, color: str, is_zscored: bool = False):
        """Plot a single CTA result on the given axis.

        Args:
            ax: Matplotlib axis to plot on
            result: CTAResult object with traces and statistics
            alignment_label: 'Onset' or 'Withdrawal'
            color: Plot color
            is_zscored: Whether data has been z-score normalized
        """
        # Apply dark theme to axis
        ax.set_facecolor('#252525')
        ax.tick_params(colors='#d4d4d4', which='both')
        ax.spines['bottom'].set_color('#3a3a3a')
        ax.spines['top'].set_color('#3a3a3a')
        ax.spines['left'].set_color('#3a3a3a')
        ax.spines['right'].set_color('#3a3a3a')

        # Plot individual traces (faint)
        for trace in result.traces:
            ax.plot(trace.time, trace.values, color=color, alpha=0.2, linewidth=0.5)

        # Plot mean and SEM
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

        # Add event line
        ax.axvline(0, color='#ff5555', linestyle='--', linewidth=1.5, label=f'{alignment_label}')

        ax.set_xlabel(f'Time from {alignment_label} (s)', color='#d4d4d4')

        # Y-axis label - indicate z-scored if applicable
        ylabel = result.metric_label
        if is_zscored:
            ylabel = f'{ylabel} (z-scored)'
        ax.set_ylabel(ylabel, color='#d4d4d4')

        ax.set_title(f'{result.metric_label} - {alignment_label} CTA', color='#e0e0e0')
        ax.legend(fontsize=8, loc='best', facecolor='#2a2a2a', edgecolor='#3a3a3a', labelcolor='#d4d4d4')
        ax.grid(True, alpha=0.2, color='#3a3a3a')

    def _on_error(self, message: str):
        """Handle error from ViewModel."""
        self._progress.setVisible(False)
        self._btn_generate.setEnabled(True)
        self._btn_generate.setText("Generate CTA Preview")
        QMessageBox.critical(self, "CTA Error", message)

    def _on_export_csv(self):
        """Export CTA data to CSV."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CTA to CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            include_traces = self._chk_include_traces.isChecked()
            self._viewmodel.export_to_csv(filepath, include_traces=include_traces)

    def _on_export_npz(self):
        """Export CTA data to NPZ."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CTA to NPZ", "", "NPZ Files (*.npz)"
        )
        if filepath:
            self._viewmodel.export_to_npz(filepath)

    def _on_export_pdf(self):
        """Export CTA plots to PDF."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export CTA to PDF", "", "PDF Files (*.pdf)"
        )
        if filepath:
            self._export_to_pdf(filepath)

    def _export_to_pdf(self, filepath: str):
        """Generate and save PDF with CTA plots."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        collection = self._viewmodel.current_collection
        if not collection:
            QMessageBox.warning(self, "No Data", "No CTA data to export.")
            return

        # Check if z-scoring is enabled
        is_zscored = self._viewmodel.config.zscore_baseline

        try:
            with PdfPages(filepath) as pdf:
                marker_types = self._viewmodel.get_marker_types()

                for marker_type in marker_types:
                    category, label = marker_type.split(":", 1)
                    type_results = collection.get_results_for_marker_type(category, label)

                    if not type_results:
                        continue

                    metrics = list(set(r.metric_key for r in type_results.values()))
                    n_metrics = len(metrics)
                    has_withdrawal = any(r.alignment == 'withdrawal' for r in type_results.values())
                    # 2 columns: onset, withdrawal (no histogram)
                    n_cols = 2 if has_withdrawal else 1

                    # Create figure with dark theme
                    fig, axes = plt.subplots(
                        n_metrics, n_cols,
                        figsize=(5 * n_cols, 3 * n_metrics),
                        squeeze=False,
                        facecolor='#1e1e1e'
                    )
                    fig.suptitle(f"CTA: {marker_type}", fontsize=12, fontweight='bold', color='#e0e0e0')

                    for idx, metric_key in enumerate(metrics):
                        onset_key = f"{category}:{label}:onset:{metric_key}"
                        onset_result = type_results.get(onset_key)

                        withdrawal_key = f"{category}:{label}:withdrawal:{metric_key}"
                        withdrawal_result = type_results.get(withdrawal_key)

                        ax_onset = axes[idx, 0]
                        if onset_result and onset_result.traces:
                            self._plot_cta_on_axis(ax_onset, onset_result, 'Onset', 'blue', is_zscored)
                        else:
                            ax_onset.text(0.5, 0.5, 'No onset data', ha='center', va='center',
                                          transform=ax_onset.transAxes)

                        if has_withdrawal:
                            ax_withdrawal = axes[idx, 1]
                            if withdrawal_result and withdrawal_result.traces:
                                self._plot_cta_on_axis(ax_withdrawal, withdrawal_result, 'Withdrawal', 'green', is_zscored)
                            else:
                                ax_withdrawal.text(0.5, 0.5, 'No withdrawal data', ha='center', va='center',
                                                   transform=ax_withdrawal.transAxes)

                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

            self._viewmodel.export_complete.emit(filepath)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export PDF: {e}")

    def _on_export_complete(self, filepath: str):
        """Handle export completion."""
        QMessageBox.information(self, "Export Complete", f"CTA data exported to:\n{filepath}")

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #88aaff;
            }
            QLabel {
                color: #e0e0e0;
                background-color: transparent;
            }
            QListWidget {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #2a7fff;
                color: white;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 4px;
            }
            QCheckBox {
                color: #e0e0e0;
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
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2a7fff;
            }
            QPushButton:pressed {
                background-color: #2a7fff;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QProgressBar {
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                background-color: #2a2a2a;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2a7fff;
                border-radius: 3px;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2a2a2a;
                color: #e0e0e0;
                padding: 8px 16px;
                border: 1px solid #3a3a3a;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 3px;
                padding: 2px;
            }
            QToolButton {
                background-color: #3d3d3d;
                border: 1px solid #4d4d4d;
                border-radius: 3px;
                padding: 4px;
                color: #d4d4d4;
            }
            QToolButton:hover {
                background-color: #4d4d4d;
                border: 1px solid #5d5d5d;
            }
            QToolButton:pressed {
                background-color: #5d5d5d;
            }
            QToolButton:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
        """)

    def set_data(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Set or update the data for CTA calculation.

        Args:
            markers: List of EventMarkers to use
            signals: Dictionary of signal arrays keyed by metric name
            time_array: Time array for signals
            metric_labels: Optional metric labels for display
        """
        self._markers = markers
        self._signals = signals
        self._time_array = time_array
        self._metric_labels = metric_labels or {}

        # Update ViewModel
        if metric_labels:
            self._viewmodel.set_available_metrics(metric_labels)

        # Refresh UI
        self._populate_marker_categories()
        self._populate_metrics()
