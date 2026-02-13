"""
Data Assembly Widget - Photometry Import Dialog

Handles:
- FP_data and AI_data file selection
- Column mapping (time, LED state, fiber columns)
- Raw signal preview
- Channel assignment for multi-fiber/multi-experiment data
- dF/F computation and visualization via ExperimentPlotter
- Save to NPZ file
"""

import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout,
    QComboBox, QCheckBox, QLabel, QTableWidgetItem, QHeaderView, QPushButton,
    QMenu, QSplitter, QGroupBox, QDoubleSpinBox, QSpinBox, QFrame, QScrollArea
)
from PyQt6.QtCore import pyqtSignal, Qt, QEvent
from PyQt6.QtGui import QAction
from PyQt6.uic import loadUi

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

from core import photometry
from .experiment_plotter import ExperimentPlotter

# Configure PyQtGraph for dark theme and performance
pg.setConfigOptions(
    antialias=True,
    useOpenGL=True,
    background='#1e1e1e',
    foreground='#cccccc',
)

# Minimum height per plot panel in the preview
MIN_HEIGHT_PER_PANEL = 120

# Subsample factor for AI data during loading (1 = full resolution)
# Preview will use smart subsampling based on visible range
AI_SUBSAMPLE = 1  # Full resolution for accurate display


def _safe_clear_graphics_layout(layout):
    """Safely clear PyQtGraph GraphicsLayoutWidget, handling inconsistent item tracking.

    PyQtGraph's GraphicsLayoutWidget.clear() can raise KeyError or ValueError when
    items have been removed or reparented in ways that leave the internal tracking
    inconsistent. This wrapper handles those cases gracefully.

    Args:
        layout: GraphicsLayoutWidget to clear
    """
    try:
        layout.clear()
    except (KeyError, ValueError, RuntimeError) as e:
        # Fallback: manually remove items
        try:
            for item in list(layout.items.keys()) if hasattr(layout, 'items') else []:
                try:
                    layout.removeItem(item)
                except (KeyError, ValueError, RuntimeError):
                    pass
        except (AttributeError, RuntimeError):
            pass  # Layout may have been deleted


class WheelEventFilter(QWidget):
    """Event filter to forward wheel events to parent scroll area."""

    def __init__(self, target_scroll_area: QWidget, parent=None):
        super().__init__(parent)
        self._scroll_area = target_scroll_area

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            # Forward wheel event to scroll area
            if self._scroll_area:
                from PyQt6.QtWidgets import QApplication
                QApplication.sendEvent(self._scroll_area.verticalScrollBar(), event)
                return True  # Event handled
        return False


class DataAssemblyWidget(QWidget):
    """
    Data Assembly tab for photometry import.

    Signals:
        data_loaded: Emitted when data is successfully loaded, with raw data dict
        npz_saved: Emitted when NPZ is saved, with path
    """

    data_loaded = pyqtSignal(dict)
    npz_saved = pyqtSignal(Path)
    load_into_app_requested = pyqtSignal(Path)  # Emitted when user clicks "Save & Load"
    load_different_requested = pyqtSignal()  # Emitted when user clicks "Load Different Experiment"

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load UI from .ui file
        ui_path = Path(__file__).parent.parent.parent / "ui" / "photometry_data_assembly.ui"
        if ui_path.exists():
            loadUi(ui_path, self)
            # Set column stretch for source files grid (2:1 ratio - left 2/3, right 1/3)
            if hasattr(self, 'source_files_group'):
                grid = self.source_files_group.layout()
                if grid is not None:
                    grid.setColumnStretch(0, 2)
                    grid.setColumnStretch(1, 1)
        else:
            # Fallback: create minimal layout if .ui file not found
            self._create_fallback_ui()

        # Initialize state
        self.file_paths: Dict[str, Optional[Path]] = {
            'fp_data': None,
            'ai_data': None,
            'timestamps': None,
            'notes': None
        }

        # Data storage (raw - preserved for reference)
        self._fp_data = None  # DataFrame from FP CSV
        self._ai_data = None  # DataFrame from AI CSV
        self._timestamps = None  # numpy array (raw timestamps in ms)

        # Preprocessed data (computed once after loading, used for all plotting/analysis)
        self._preprocessed = None  # Will be dict with common_time, fibers, ai_channels, etc.
        self._processed_results = {}  # dF/F results keyed by 'exp_{idx}'

        # Track the NPZ path we loaded from (for save without prompt)
        self._loaded_npz_path: Optional[Path] = None

        # Store original state hash for detecting changes (more robust than tracking individual changes)
        self._original_state_hash: Optional[str] = None

        # Column mapping
        self.fiber_columns: Dict[str, dict] = {}  # fiber_col -> {'checkbox': QCheckBox, ...}
        self.ai_columns: Dict[int, dict] = {}  # index -> {'checkbox': QCheckBox, 'column': str, ...}

        # Experiment plotter for dF/F visualization
        self._plotter = ExperimentPlotter()

        # Setup PyQtGraph for preview plots
        self._setup_pyqtgraph()

        # Connect signals
        self._setup_connections()

    def _create_fallback_ui(self):
        """Create minimal UI if .ui file not found."""
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Error: photometry_data_assembly.ui not found"))

    def _setup_pyqtgraph(self):
        """Setup PyQtGraph graphics layout for high-performance plotting."""
        # Main graphics layout widget
        self.graphics_layout = GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#1e1e1e')

        # Set size policy to expand in both directions
        from PyQt6.QtWidgets import QSizePolicy
        self.graphics_layout.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

        # Track plot items for each panel
        self._plot_items: List[pg.PlotItem] = []

        # Track experiment graphics layouts for multi-experiment tabs
        self._experiment_layouts: Dict[int, GraphicsLayoutWidget] = {}

        # Map plot items to channel names for context menu
        self._plot_to_channel: Dict[pg.PlotItem, str] = {}

        # Add to canvas container (defined in .ui file)
        if hasattr(self, 'canvas_container'):
            layout = QVBoxLayout(self.canvas_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.graphics_layout)

        # Install event filter to forward wheel events to parent scroll area
        if hasattr(self, 'scroll_area'):
            self._wheel_filter = WheelEventFilter(self.scroll_area, self)
            self.graphics_layout.installEventFilter(self._wheel_filter)

        # Add Reset View button on the left side of the controls row
        # We need to find the QHBoxLayout that contains chk_normalize_time and btn_update_preview
        if hasattr(self, 'btn_update_preview') and hasattr(self, 'chk_normalize_time'):
            from PyQt6.QtWidgets import QHBoxLayout

            # Search recursively for the layout containing the checkbox
            def find_layout_containing_widget(layout, widget):
                if layout is None:
                    return None
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    if item.widget() == widget:
                        return layout
                    if item.layout():
                        result = find_layout_containing_widget(item.layout(), widget)
                        if result:
                            return result
                return None

            # Start from preview_section's layout
            if hasattr(self, 'preview_section'):
                controls_layout = find_layout_containing_widget(
                    self.preview_section.layout(),
                    self.chk_normalize_time
                )

                if controls_layout and isinstance(controls_layout, QHBoxLayout):
                    self.btn_reset_view = QPushButton("Reset View")
                    self.btn_reset_view.setToolTip("Reset all plots to show full data range")
                    self.btn_reset_view.clicked.connect(self._reset_all_plots)
                    self.btn_reset_view.setStyleSheet("""
                        QPushButton {
                            background-color: #2ea043;
                            border: none;
                            border-radius: 3px;
                            padding: 8px 16px;
                            color: white;
                            font-weight: bold;
                        }
                        QPushButton:hover { background-color: #3fb950; }
                        QPushButton:pressed { background-color: #238636; }
                    """)
                    # Insert at position 0 (left side, before the spacer)
                    controls_layout.insertWidget(0, self.btn_reset_view)

    def _setup_connections(self):
        """Connect UI signals to handlers."""
        # File browse buttons
        if hasattr(self, 'btn_fp_browse'):
            self.btn_fp_browse.clicked.connect(lambda: self._browse_file('fp_data'))
        if hasattr(self, 'btn_ai_browse'):
            self.btn_ai_browse.clicked.connect(lambda: self._browse_file('ai_data'))
        if hasattr(self, 'btn_ts_browse'):
            self.btn_ts_browse.clicked.connect(lambda: self._browse_file('timestamps'))
        if hasattr(self, 'btn_notes_browse'):
            self.btn_notes_browse.clicked.connect(lambda: self._browse_file('notes'))

        # Notes view button
        if hasattr(self, 'btn_notes_view'):
            self.btn_notes_view.clicked.connect(self._show_notes_popout)

        # Update preview button
        if hasattr(self, 'btn_update_preview'):
            self.btn_update_preview.clicked.connect(self._update_preview_plot)

        # Save buttons
        if hasattr(self, 'btn_save_data'):
            self.btn_save_data.clicked.connect(lambda: self._on_save_data_file(load_after=False))
        if hasattr(self, 'btn_save_and_load'):
            self.btn_save_and_load.clicked.connect(lambda: self._on_save_data_file(load_after=True))

        # Number of experiments
        if hasattr(self, 'spin_num_experiments'):
            self.spin_num_experiments.valueChanged.connect(self._on_num_experiments_changed)

        # Preview tab changes - update toolbar to work with active canvas
        if hasattr(self, 'preview_tab_widget'):
            self.preview_tab_widget.currentChanged.connect(self._on_preview_tab_changed)

        # Track edit mode state
        self._edit_mode = False
        self._current_exp_idx = 0
        self._n_experiments = 1
        self._load_different_btn = None

    def set_edit_mode(self, current_exp_idx: int, n_experiments: int):
        """
        Enable edit mode with current experiment tracking.

        In edit mode:
        - Tracks which experiment is currently loaded
        - Shows "Load Different Experiment" button if n_experiments > 1

        Args:
            current_exp_idx: Currently loaded experiment (0-indexed)
            n_experiments: Total number of experiments in the NPZ
        """
        self._edit_mode = True
        self._current_exp_idx = current_exp_idx
        self._n_experiments = n_experiments

        # Update button text for clarity
        if hasattr(self, 'btn_save_and_load'):
            self.btn_save_and_load.setText("Save && Reload")
            self.btn_save_and_load.setToolTip(
                f"Save changes and reload experiment {current_exp_idx + 1}"
            )

        # Add "Load Different Experiment" button if multiple experiments
        if n_experiments > 1 and self._load_different_btn is None:
            self._add_load_different_button()

    def _add_load_different_button(self):
        """Add 'Load Different Experiment' button in edit mode."""
        # Find the button layout (horizontal layout containing save buttons)
        if not hasattr(self, 'btn_save_and_load'):
            return

        parent_layout = self.btn_save_and_load.parent()
        if parent_layout is None:
            return

        # Find the layout containing the buttons
        layout = None
        if hasattr(parent_layout, 'layout') and callable(parent_layout.layout):
            layout = parent_layout.layout()

        if layout is None:
            return

        # Create the "Load Different Experiment" button
        self._load_different_btn = QPushButton("Load Different Experiment")
        self._load_different_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #094771; }
        """)
        self._load_different_btn.setToolTip(
            f"Switch to a different experiment ({self._n_experiments} available)"
        )
        self._load_different_btn.clicked.connect(self.load_different_requested.emit)

        # Insert before the "Save & Reload" button
        # Find the index of btn_save_and_load
        for i in range(layout.count()):
            item = layout.itemAt(i)
            if item and item.widget() == self.btn_save_and_load:
                layout.insertWidget(i, self._load_different_btn)
                break

    def _browse_file(self, file_type: str):
        """
        Simple file browser using QFileDialog.
        No recent files - that's handled by main window.
        """
        # Determine filter based on file type
        if file_type == 'fp_data':
            filter_str = "CSV Files (*.csv);;All Files (*)"
            title = "Select Photometry Data File (FP_data)"
        elif file_type == 'ai_data':
            filter_str = "CSV Files (*.csv);;All Files (*)"
            title = "Select Analog Inputs File (AI_data)"
        elif file_type == 'timestamps':
            filter_str = "CSV Files (*.csv);;All Files (*)"
            title = "Select Timestamps File"
        elif file_type == 'notes':
            filter_str = "Notes Files (*.txt *.csv *.xlsx *.xls *.docx);;All Files (*)"
            title = "Select Notes File"
        else:
            return

        # Get start directory
        start_dir = ""
        if self.file_paths.get('fp_data'):
            start_dir = str(self.file_paths['fp_data'].parent)

        file_path, _ = QFileDialog.getOpenFileName(self, title, start_dir, filter_str)

        if file_path:
            self.file_paths[file_type] = Path(file_path)
            self._update_file_edits()

            # For FP data, also look for companion files
            if file_type == 'fp_data':
                companions = photometry.find_companion_files(Path(file_path))
                if companions.get('ai_data') and not self.file_paths.get('ai_data'):
                    self.file_paths['ai_data'] = companions['ai_data']
                if companions.get('timestamps') and not self.file_paths.get('timestamps'):
                    self.file_paths['timestamps'] = companions['timestamps']
                self._update_file_edits()

            # Load and preview data
            if file_type in ('fp_data', 'ai_data', 'timestamps'):
                self._load_and_preview_data()

            # Update notes info
            if file_type == 'notes':
                self._update_notes_info()

    def _update_file_edits(self):
        """Update file path display fields."""
        if hasattr(self, 'fp_data_edit') and self.file_paths.get('fp_data'):
            self.fp_data_edit.setText(str(self.file_paths['fp_data']))

        if hasattr(self, 'ai_data_edit'):
            if self.file_paths.get('ai_data'):
                self.ai_data_edit.setText(str(self.file_paths['ai_data']))
            else:
                self.ai_data_edit.clear()

        if hasattr(self, 'timestamps_edit'):
            if self.file_paths.get('timestamps'):
                self.timestamps_edit.setText(str(self.file_paths['timestamps']))
                self.timestamps_edit.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: #d4d4d4; }")
            else:
                self.timestamps_edit.clear()
                self.timestamps_edit.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: #888888; }")

        if hasattr(self, 'notes_edit'):
            if self.file_paths.get('notes'):
                self.notes_edit.setText(str(self.file_paths['notes']))
            else:
                self.notes_edit.clear()

        # Enable notes view button
        if hasattr(self, 'btn_notes_view'):
            self.btn_notes_view.setEnabled(self.file_paths.get('notes') is not None)

    def _load_and_preview_data(self):
        """Load data files in background thread and update preview when done."""
        if not self.file_paths.get('fp_data'):
            return

        from PyQt6.QtWidgets import QProgressDialog
        from core.file_load_worker import FileLoadWorker

        # Show a proper modal progress dialog
        self._load_progress = QProgressDialog(
            "Loading photometry data...\nThis may take a moment for large files.",
            None, 0, 0, self
        )
        self._load_progress.setWindowTitle("Loading Photometry Files")
        self._load_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._load_progress.setMinimumDuration(0)
        self._load_progress.setCancelButton(None)
        self._load_progress.show()

        # Bundle all I/O into a single function for the worker
        file_paths = dict(self.file_paths)  # snapshot

        def _load_all_csv_files(progress_callback=None):
            """Load all CSV files (runs in background thread)."""
            results = {'fp_data': None, 'ai_data': None, 'timestamps': None, 'ts_path': None}

            # 1. Load FP data (the big one)
            if progress_callback:
                progress_callback(0, 3, "Loading photometry CSV...")
            t0 = time.perf_counter()
            results['fp_data'] = photometry.load_photometry_csv(file_paths['fp_data'])
            print(f"[Timing] Load FP data ({len(results['fp_data'])} rows): {time.perf_counter() - t0:.2f}s")

            # 2. Load AI data if available
            if file_paths.get('ai_data'):
                if progress_callback:
                    progress_callback(1, 3, "Loading AI data CSV...")
                try:
                    t0 = time.perf_counter()
                    results['ai_data'] = photometry.load_ai_data_csv(
                        file_paths['ai_data'], subsample=AI_SUBSAMPLE
                    )
                    print(f"[Timing] Load AI data (1/{AI_SUBSAMPLE} subsampled): {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error loading AI data: {e}")

            # 3. Load timestamps
            if progress_callback:
                progress_callback(2, 3, "Loading timestamps...")
            ts_path = file_paths.get('timestamps')
            if not ts_path:
                ts_path = photometry.find_timestamps_file(file_paths['fp_data'])

            if ts_path:
                try:
                    t0 = time.perf_counter()
                    results['timestamps'] = photometry.load_timestamps_csv(ts_path, subsample=AI_SUBSAMPLE)
                    results['ts_path'] = ts_path
                    print(f"[Timing] Load timestamps (1/{AI_SUBSAMPLE} subsampled): {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error loading timestamps: {e}")

            return results

        # Create and start worker
        self._csv_load_worker = FileLoadWorker(_load_all_csv_files)
        self._csv_load_worker.progress.connect(lambda c, t, m: (
            self._load_progress.setLabelText(f"{m}\n{Path(file_paths['fp_data']).name}")
        ))
        self._csv_load_worker.finished.connect(self._on_csv_data_loaded)
        self._csv_load_worker.error.connect(self._on_csv_load_error)
        self._csv_load_worker.start()

    def _on_csv_load_error(self, msg):
        """Handle CSV loading errors."""
        self._load_progress.close()
        error_msg = msg.split('\n\n')[0] if '\n\n' in msg else msg
        print(f"[Photometry] Error loading FP data: {error_msg}")
        QMessageBox.warning(self, "Load Error", f"Failed to load FP data:\n{error_msg}")

    def _on_csv_data_loaded(self, results):
        """Completion handler after CSV files loaded in background (runs on main thread)."""
        self._load_progress.close()

        # Store loaded data
        self._fp_data = results['fp_data']
        if self._fp_data is None:
            QMessageBox.warning(self, "Load Error", "Failed to load FP data.")
            return

        # Populate column dropdowns
        self._populate_fp_column_combos()

        # Update preview table
        self._update_fp_preview_table()

        # Store AI data if loaded
        if results['ai_data'] is not None:
            self._ai_data = results['ai_data']
            self._populate_ai_column_controls()
            self._update_ai_preview_table()

        # Store timestamps if loaded
        if results['timestamps'] is not None:
            self._timestamps = results['timestamps']
            if results['ts_path']:
                self.file_paths['timestamps'] = results['ts_path']
                self._update_file_edits()
                self._update_timestamps_info()

        # Update channel table
        self._update_channel_table()

        # Update output file label (shows auto-generated filename)
        self._update_output_file_list()

        # Preprocess all data once (creates common time base, interpolates signals)
        self._preprocess_all_data()

        # Create experiment tabs based on spinner value (ensures tabs exist even when n_experiments=1)
        if hasattr(self, 'spin_num_experiments'):
            self._update_experiment_tabs(self.spin_num_experiments.value())

        # Pre-calculate number of panels for initial resize
        n_fiber_panels = len([c for c, info in self.fiber_columns.items()
                             if info['checkbox'].isChecked()])
        n_ai_panels = len(self.ai_columns) if self._ai_data is not None else 0
        self._resize_preview_for_panels(n_fiber_panels + n_ai_panels)

        # Draw preview
        self._update_preview_plot()

        self._hide_progress()

        # Focus spinbox so arrow keys work immediately
        if hasattr(self, 'spin_num_experiments'):
            self.spin_num_experiments.setFocus()

    def _show_progress(self, message: str):
        """Show loading progress message."""
        if hasattr(self, 'loading_label'):
            self.loading_label.setText(message)
            self.loading_label.setVisible(True)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def _hide_progress(self):
        """Hide loading progress message."""
        if hasattr(self, 'loading_label'):
            self.loading_label.setVisible(False)

    def _populate_fp_column_combos(self):
        """Populate FP column mapping dropdowns."""
        if self._fp_data is None:
            return

        columns = list(self._fp_data.columns)

        # Time column combo
        if hasattr(self, 'fp_columns_container'):
            # Create combo boxes dynamically
            # This would need more detailed implementation
            pass

        # Detect fiber columns
        fiber_cols = photometry.detect_fiber_columns(self._fp_data)
        self._populate_fiber_column_controls(fiber_cols)

    def _populate_fiber_column_controls(self, fiber_cols: List[str]):
        """Create checkboxes for fiber columns."""
        self.fiber_columns.clear()

        if not hasattr(self, 'fp_columns_container'):
            return

        # Clear existing widgets
        layout = self.fp_columns_container.layout()
        if layout is None:
            layout = QHBoxLayout(self.fp_columns_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

        # Remove old widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add fiber checkboxes
        for col in fiber_cols:
            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.setStyleSheet("color: #d4d4d4;")
            cb.stateChanged.connect(self._on_fiber_column_changed)
            layout.addWidget(cb)
            self.fiber_columns[col] = {'checkbox': cb}

        layout.addStretch()

        # Auto-set number of experiments to match number of fiber channels
        if hasattr(self, 'spin_num_experiments') and len(fiber_cols) > 0:
            self.spin_num_experiments.setValue(len(fiber_cols))

    def _populate_ai_column_controls(self):
        """Create checkboxes for AI columns."""
        self.ai_columns.clear()

        if self._ai_data is None or not hasattr(self, 'ai_columns_container'):
            return

        layout = self.ai_columns_container.layout()
        if layout is None:
            layout = QHBoxLayout(self.ai_columns_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

        # Remove old widgets
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add AI column checkboxes
        for i, col in enumerate(self._ai_data.columns):
            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.setStyleSheet("color: #d4d4d4;")
            layout.addWidget(cb)
            self.ai_columns[i] = {'checkbox': cb, 'column': col}

        layout.addStretch()

    def _update_fp_preview_table(self):
        """Update FP data preview table."""
        if self._fp_data is None or not hasattr(self, 'fp_preview_table'):
            return

        col_names, rows = photometry.get_file_preview(self.file_paths['fp_data'], n_rows=5)

        table = self.fp_preview_table
        table.setColumnCount(len(col_names))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(col_names)

        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(val)))

    def _update_ai_preview_table(self):
        """Update AI data preview table."""
        if self._ai_data is None or not hasattr(self, 'ai_preview_table'):
            return

        col_names, rows = photometry.get_file_preview(self.file_paths['ai_data'], n_rows=5)

        table = self.ai_preview_table
        table.setColumnCount(len(col_names))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(col_names)

        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                table.setItem(r, c, QTableWidgetItem(str(val)))

    def _update_timestamps_info(self):
        """Update timestamps info label."""
        if not hasattr(self, 'ts_info_label'):
            return

        if self._timestamps is not None and len(self._timestamps) > 0:
            t_min = self._timestamps.min()
            t_max = self._timestamps.max()
            n_samples = len(self._timestamps)
            self.ts_info_label.setText(
                f"{n_samples:,} samples, {t_min:.1f} - {t_max:.1f} ms"
            )
        else:
            self.ts_info_label.setText("No timestamps loaded")

    def _update_notes_info(self):
        """Update notes preview label."""
        if not hasattr(self, 'notes_preview_label'):
            return

        notes_path = self.file_paths.get('notes')
        if not notes_path:
            self.notes_preview_label.setText("No notes file selected")
            return

        try:
            # Read first few lines
            with open(notes_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()[:3]
            preview = ''.join(lines).strip()
            if len(preview) > 100:
                preview = preview[:100] + "..."
            self.notes_preview_label.setText(preview if preview else "(empty file)")
        except Exception as e:
            self.notes_preview_label.setText(f"Error reading: {e}")

    def _show_notes_popout(self):
        """Show notes file in a popout dialog."""
        notes_path = self.file_paths.get('notes')
        if not notes_path:
            QMessageBox.warning(self, "No Notes File", "Please select a notes file first.")
            return

        # Create simple preview dialog
        from PyQt6.QtWidgets import QDialog, QTextEdit

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Notes: {notes_path.name}")
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        try:
            with open(notes_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_edit.setPlainText(f.read())
        except Exception as e:
            text_edit.setPlainText(f"Error reading file: {e}")

        layout.addWidget(text_edit)
        dialog.exec()

    def _on_num_experiments_changed(self, value: int):
        """Handle number of experiments change."""
        self._update_experiment_tabs(value)
        self._update_channel_table()
        self._update_output_file_list()
        self._update_preview_plot()

    def _update_experiment_tab_name(self, exp_idx: int, animal_id: str):
        """Update the tab name when animal ID changes."""
        # Mark that changes have been made
        self._has_unsaved_changes = True

        if not hasattr(self, 'preview_tab_widget'):
            return

        tab_widget = self.preview_tab_widget
        tab_index = exp_idx + 1  # +1 because "All Channels" is at index 0

        if tab_index < tab_widget.count():
            if animal_id.strip():
                tab_widget.setTabText(tab_index, f"Exp {exp_idx + 1}: {animal_id.strip()}")
            else:
                tab_widget.setTabText(tab_index, f"Exp {exp_idx + 1}")

    def _update_experiment_tabs(self, n_experiments: int):
        """Create or remove experiment preview tabs based on experiment count."""
        if not hasattr(self, 'preview_tab_widget'):
            return

        tab_widget = self.preview_tab_widget

        # Count current experiment tabs (excluding "All Channels")
        current_exp_tabs = tab_widget.count() - 1

        if n_experiments < 1:
            # Remove all experiment tabs, keep only "All Channels"
            while tab_widget.count() > 1:
                tab_widget.removeTab(1)
            self._experiment_layouts.clear()
            # Clean up all experiment controls and secondary viewboxes
            if hasattr(self, '_dff_controls'):
                self._dff_controls.clear()
            if hasattr(self, '_secondary_viewboxes'):
                self._secondary_viewboxes.clear()
            return

        # Add new experiment tabs if needed
        for i in range(current_exp_tabs, n_experiments):
            exp_widget = QWidget()
            exp_main_layout = QHBoxLayout(exp_widget)
            exp_main_layout.setContentsMargins(0, 0, 0, 0)
            exp_main_layout.setSpacing(0)

            # Create splitter for controls (left) and plots (right)
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.setStyleSheet("""
                QSplitter::handle {
                    background-color: #3e3e42;
                    width: 3px;
                }
                QSplitter::handle:hover {
                    background-color: #007acc;
                }
            """)

            # Left panel: dF/F controls
            controls_panel = self._create_dff_controls_panel(i)
            splitter.addWidget(controls_panel)

            # Right panel: Plots
            plot_container = QWidget()
            plot_layout = QVBoxLayout(plot_container)
            plot_layout.setContentsMargins(0, 0, 0, 0)

            graphics_layout = GraphicsLayoutWidget()
            graphics_layout.setBackground('#1e1e1e')

            # Set size policy to expand
            from PyQt6.QtWidgets import QSizePolicy
            graphics_layout.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            graphics_layout.setMinimumHeight(400)

            plot_layout.addWidget(graphics_layout)
            splitter.addWidget(plot_container)

            # Set initial splitter sizes (controls: 280px, plots: rest)
            splitter.setSizes([280, 800])

            exp_main_layout.addWidget(splitter)

            # Install wheel filter
            if hasattr(self, 'scroll_area'):
                wheel_filter = WheelEventFilter(self.scroll_area, self)
                graphics_layout.installEventFilter(wheel_filter)

            self._experiment_layouts[i] = graphics_layout

            tab_widget.addTab(exp_widget, f"Exp {i + 1}")

        # Remove excess tabs
        while tab_widget.count() > n_experiments + 1:
            idx = tab_widget.count() - 1
            tab_widget.removeTab(idx)
            exp_idx = idx - 1
            if exp_idx in self._experiment_layouts:
                del self._experiment_layouts[exp_idx]
            # Clean up controls for removed experiment
            controls_key = f'exp_{exp_idx}'
            if hasattr(self, '_dff_controls') and controls_key in self._dff_controls:
                del self._dff_controls[controls_key]
            # Clean up secondary viewboxes
            if hasattr(self, '_secondary_viewboxes') and exp_idx in self._secondary_viewboxes:
                del self._secondary_viewboxes[exp_idx]

    def _create_dff_controls_panel(self, exp_idx: int) -> QWidget:
        """Create the dF/F processing controls panel for an experiment tab.

        Args:
            exp_idx: Experiment index (0-based)

        Returns:
            QWidget containing all the dF/F controls
        """
        from PyQt6.QtWidgets import QLineEdit

        # Create scrollable container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: #252526;
                border: none;
            }
        """)

        # Main container widget
        container = QWidget()
        container.setStyleSheet("background-color: #252526;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Store references to controls for this experiment
        controls_key = f'exp_{exp_idx}'
        if not hasattr(self, '_dff_controls'):
            self._dff_controls = {}
        self._dff_controls[controls_key] = {}

        # --- Animal/Experiment ID Group ---
        id_group = QGroupBox("Experiment Info")
        id_group.setStyleSheet(self._get_group_style())
        id_layout = QVBoxLayout(id_group)
        id_layout.setSpacing(4)

        id_row = QHBoxLayout()
        id_row.addWidget(QLabel("Animal ID:"))
        animal_id_edit = QLineEdit()
        animal_id_edit.setPlaceholderText(f"e.g., Mouse_{exp_idx + 1}")
        animal_id_edit.setStyleSheet("""
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QLineEdit:focus { border: 1px solid #007acc; }
        """)
        id_row.addWidget(animal_id_edit)
        id_layout.addLayout(id_row)
        self._dff_controls[controls_key]['animal_id_edit'] = animal_id_edit

        # Update tab name when animal ID changes
        animal_id_edit.textChanged.connect(lambda text, idx=exp_idx: self._update_experiment_tab_name(idx, text))

        layout.addWidget(id_group)

        # --- Processing Group (combines dF/F + Detrending) ---
        processing_group = QGroupBox("Processing")
        processing_group.setStyleSheet(self._get_group_style())
        processing_layout = QVBoxLayout(processing_group)
        processing_layout.setSpacing(6)

        # dF/F Method selection
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("dF/F Method:"))
        method_combo = QComboBox()
        method_combo.addItems(["Fitted (regression)", "Simple (subtraction)"])
        method_combo.setStyleSheet(self._get_combo_style())
        method_row.addWidget(method_combo)
        processing_layout.addLayout(method_row)
        self._dff_controls[controls_key]['combo_dff_method'] = method_combo

        # Detrend Method selection
        detrend_row = QHBoxLayout()
        detrend_row.addWidget(QLabel("Detrend:"))
        detrend_method = QComboBox()
        detrend_method.addItems(["None", "Linear", "Exponential", "Biexponential"])
        detrend_method.setStyleSheet(self._get_combo_style())
        detrend_row.addWidget(detrend_method)
        processing_layout.addLayout(detrend_row)
        self._dff_controls[controls_key]['combo_detrend_method'] = detrend_method

        # Fit window range (always visible and interactive)
        fit_label = QLabel("Fit Window:")
        fit_label.setStyleSheet("color: #888888; font-size: 10px; margin-top: 4px;")
        processing_layout.addWidget(fit_label)

        fit_range_row = QHBoxLayout()
        fit_range_row.addWidget(QLabel("Start:"))
        spin_fit_start = QDoubleSpinBox()
        spin_fit_start.setRange(0.0, 1000.0)
        spin_fit_start.setSingleStep(0.5)
        spin_fit_start.setDecimals(1)
        spin_fit_start.setValue(0.0)  # Default to start
        spin_fit_start.setSuffix(" min")
        spin_fit_start.setStyleSheet(self._get_spinbox_style())
        fit_range_row.addWidget(spin_fit_start)
        self._dff_controls[controls_key]['spin_fit_start'] = spin_fit_start

        fit_range_row.addWidget(QLabel("End:"))
        spin_fit_end = QDoubleSpinBox()
        spin_fit_end.setRange(0.0, 1000.0)
        spin_fit_end.setSingleStep(0.5)
        spin_fit_end.setDecimals(1)
        spin_fit_end.setValue(0.0)  # Default to 0 = use all data
        spin_fit_end.setSuffix(" min")
        spin_fit_end.setStyleSheet(self._get_spinbox_style())
        fit_range_row.addWidget(spin_fit_end)
        self._dff_controls[controls_key]['spin_fit_end'] = spin_fit_end

        processing_layout.addLayout(fit_range_row)

        # Hint about fit window
        fit_hint = QLabel("Drag region on plot to adjust. Set both to 0 to use all data.")
        fit_hint.setStyleSheet("color: #888888; font-size: 9px;")
        fit_hint.setWordWrap(True)
        processing_layout.addWidget(fit_hint)

        layout.addWidget(processing_group)

        # --- Filtering Group ---
        filter_group = QGroupBox("Filtering")
        filter_group.setStyleSheet(self._get_group_style())
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setSpacing(4)

        # Low-pass row
        lowpass_row = QHBoxLayout()
        chk_lowpass = QCheckBox("Low-pass filter:")
        chk_lowpass.setStyleSheet("color: #d4d4d4;")
        lowpass_row.addWidget(chk_lowpass)
        self._dff_controls[controls_key]['chk_lowpass'] = chk_lowpass

        spin_lowpass_hz = QDoubleSpinBox()
        spin_lowpass_hz.setRange(0.1, 50.0)
        spin_lowpass_hz.setValue(2.0)
        spin_lowpass_hz.setSuffix(" Hz")
        spin_lowpass_hz.setEnabled(False)
        spin_lowpass_hz.setStyleSheet(self._get_spinbox_style())
        lowpass_row.addWidget(spin_lowpass_hz)
        self._dff_controls[controls_key]['spin_lowpass_hz'] = spin_lowpass_hz

        # Connect checkbox to enable/disable spinbox
        chk_lowpass.toggled.connect(spin_lowpass_hz.setEnabled)

        filter_layout.addLayout(lowpass_row)

        # Exclude start row
        exclude_row = QHBoxLayout()
        exclude_row.addWidget(QLabel("Exclude start:"))
        spin_exclude_start = QDoubleSpinBox()
        spin_exclude_start.setRange(0.0, 60.0)
        spin_exclude_start.setSingleStep(0.5)
        spin_exclude_start.setSuffix(" min")
        spin_exclude_start.setDecimals(1)
        spin_exclude_start.setStyleSheet(self._get_spinbox_style())
        exclude_row.addWidget(spin_exclude_start)
        self._dff_controls[controls_key]['spin_exclude_start'] = spin_exclude_start

        filter_layout.addLayout(exclude_row)
        layout.addWidget(filter_group)

        # --- Display Options Group ---
        display_group = QGroupBox("Display Options")
        display_group.setStyleSheet(self._get_group_style())
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        chk_show_intermediates = QCheckBox("Show intermediate signals")
        chk_show_intermediates.setChecked(False)  # Default OFF for performance (rendering fewer traces)
        chk_show_intermediates.setStyleSheet("color: #d4d4d4;")
        chk_show_intermediates.setToolTip("Show raw signals, fitted isosbestic, and detrend curve")
        display_layout.addWidget(chk_show_intermediates)
        self._dff_controls[controls_key]['chk_show_intermediates'] = chk_show_intermediates

        layout.addWidget(display_group)

        # --- Connect signals for auto-update ---
        # Use debounce timer to avoid rapid re-computation
        from PyQt6.QtCore import QTimer

        # Create a debounce timer for this experiment's controls
        debounce_timer = QTimer()
        debounce_timer.setSingleShot(True)
        debounce_timer.setInterval(400)  # 400ms debounce delay
        self._dff_controls[controls_key]['_debounce_timer'] = debounce_timer

        def make_auto_update(idx, timer):
            def do_update():
                # Only auto-update if we have data loaded
                if hasattr(self, '_fp_data') and self._fp_data is not None:
                    print(f"[Photometry] Auto-update triggered for Exp {idx + 1}")
                    self._compute_dff(idx)
                elif self._preprocessed is not None:
                    # Also handle cached data case
                    print(f"[Photometry] Auto-update triggered for Exp {idx + 1} (cached)")
                    self._compute_dff(idx)

            # Connect timer once
            timer.timeout.connect(do_update)

            def auto_update():
                # Mark that changes have been made
                self._has_unsaved_changes = True
                # Mark this experiment as needing recompute
                setattr(self, f'_exp_{idx}_dirty', True)
                # Restart the debounce timer
                timer.stop()
                timer.start()

            return auto_update

        auto_update_fn = make_auto_update(exp_idx, debounce_timer)

        # Connect combo boxes (changes trigger auto-update and mark unsaved)
        method_combo.currentIndexChanged.connect(auto_update_fn)
        detrend_method.currentIndexChanged.connect(auto_update_fn)

        # Connect checkboxes
        chk_lowpass.toggled.connect(auto_update_fn)
        chk_show_intermediates.toggled.connect(auto_update_fn)

        # Connect spinboxes with debounced valueChanged for responsive interaction
        spin_fit_start.valueChanged.connect(auto_update_fn)
        spin_fit_end.valueChanged.connect(auto_update_fn)
        spin_lowpass_hz.valueChanged.connect(auto_update_fn)
        spin_exclude_start.valueChanged.connect(auto_update_fn)

        # Spacer
        layout.addStretch()

        # --- Fit Parameters Group ---
        params_group = QGroupBox("Fit Parameters")
        params_group.setStyleSheet(self._get_group_style())
        params_layout = QVBoxLayout(params_group)

        fit_params_label = QLabel("No fit performed yet")
        fit_params_label.setStyleSheet("color: #888888; font-size: 10px; font-family: monospace;")
        fit_params_label.setWordWrap(True)
        params_layout.addWidget(fit_params_label)
        self._dff_controls[controls_key]['fit_params_label'] = fit_params_label

        layout.addWidget(params_group)

        scroll.setWidget(container)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)

        return scroll

    def _get_group_style(self) -> str:
        """Get consistent group box style."""
        return """
            QGroupBox {
                background-color: #2d2d30;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #569cd6;
            }
            QLabel {
                color: #d4d4d4;
                background: transparent;
            }
        """

    def _get_combo_style(self) -> str:
        """Get consistent combo box style."""
        return """
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px;
                color: #d4d4d4;
                min-width: 100px;
            }
            QComboBox:hover { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #d4d4d4;
                selection-background-color: #094771;
            }
        """

    def _get_spinbox_style(self) -> str:
        """Get consistent spinbox style."""
        return """
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px;
                color: #d4d4d4;
                min-width: 70px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover { border-color: #007acc; }
        """

    def _compute_dff(self, exp_idx: int):
        """Compute and display dF/F for an experiment using ExperimentPlotter.

        Uses preprocessed data (already interpolated to common time base) for speed.
        Only the dF/F fitting and optional detrending is computed here.
        """
        import time as time_module

        t_start = time_module.perf_counter()
        print(f"[Photometry] Computing dF/F for Exp {exp_idx + 1}")

        # Check for preprocessed data
        if self._preprocessed is None:
            print(f"[Photometry] No preprocessed data available")
            return

        # Get controls for this experiment
        controls_key = f'exp_{exp_idx}'
        if not hasattr(self, '_dff_controls') or controls_key not in self._dff_controls:
            print(f"[Photometry] No controls found for {controls_key}")
            return

        controls = self._dff_controls[controls_key]

        # Get processing parameters from controls
        params = self._get_dff_params(exp_idx)

        # Get experiment assignments
        exp_assignments = self._get_experiment_assignments()

        # Find fiber channels assigned to this experiment
        # Handle both GCaMP (green) and Red (560nm) signal channels
        exp_fibers = []
        for channel_name, assigned_exp in exp_assignments.items():
            if assigned_exp == exp_idx or assigned_exp == -1:
                # Check for both GCaMP and Red signal channels
                if '-GCaMP' in channel_name:
                    fiber_col = channel_name.replace('-GCaMP', '')
                    if fiber_col not in exp_fibers:
                        exp_fibers.append(fiber_col)
                elif '-Red' in channel_name:
                    fiber_col = channel_name.replace('-Red', '')
                    if fiber_col not in exp_fibers:
                        exp_fibers.append(fiber_col)

        if not exp_fibers:
            print(f"[Photometry] No fiber channels assigned to Exp {exp_idx + 1}")
            return

        # Get graphics layout for this experiment
        if exp_idx not in self._experiment_layouts:
            return

        graphics_layout = self._experiment_layouts[exp_idx]
        _safe_clear_graphics_layout(graphics_layout)

        # Clear fit regions for this experiment
        self._plotter.clear_fit_regions(exp_idx)

        # Store processed results
        if not hasattr(self, '_processed_results'):
            self._processed_results = {}
        self._processed_results[controls_key] = {}

        show_intermediates = controls['chk_show_intermediates'].isChecked()

        # Get AI channels for this experiment
        ai_channels = self._get_ai_channels_for_experiment(exp_idx)

        # Get common time from preprocessed data (in SECONDS, normalized to start at 0)
        common_time_sec = self._preprocessed['common_time']
        common_time_min = common_time_sec / 60.0  # For dF/F functions that expect minutes

        row = 0
        all_plot_items = []

        for fiber_col in exp_fibers:
            # Use preprocessed fiber data (already interpolated to common time)
            if fiber_col not in self._preprocessed['fibers']:
                print(f"[Photometry] Skipping {fiber_col}: not in preprocessed data")
                continue

            fiber_preproc = self._preprocessed['fibers'][fiber_col]
            iso_signal = fiber_preproc['iso']
            gcamp_signal = fiber_preproc['gcamp']

            # Skip channels with all-NaN values
            if np.all(np.isnan(iso_signal)) or np.all(np.isnan(gcamp_signal)):
                print(f"[Photometry] Skipping {fiber_col}: all NaN values")
                continue

            # Compute dF/F using FAST path (data already aligned, no interpolation needed)
            t0 = time_module.perf_counter()
            try:
                # Use the simple/fitted dF/F functions directly (no interpolation)
                if params['method'] == 'simple':
                    dff, fit_params = photometry.compute_dff_simple(gcamp_signal, iso_signal)
                    fitted_iso = iso_signal  # No fitting for simple method
                else:
                    dff, fit_params = photometry.compute_dff_fitted(
                        gcamp_signal, iso_signal, common_time_min,
                        fit_start=params['fit_start'],
                        fit_end=params['fit_end']
                    )
                    fitted_iso = fit_params.get('fitted_iso', iso_signal)

                # Apply detrending if requested
                dff_raw = dff.copy()
                detrend_curve = None
                if params['detrend_method'] != 'none':
                    dff, detrend_curve, detrend_params = photometry.detrend_signal(
                        dff, common_time_min,
                        method=params['detrend_method'],
                        fit_start=params['fit_start'],
                        fit_end=params['fit_end']
                    )
                    fit_params.update(detrend_params)

                # Apply lowpass filter if requested
                if params['lowpass_hz'] is not None and params['lowpass_hz'] > 0:
                    sample_rate = self._preprocessed['sample_rate']
                    dff = photometry.lowpass_filter(dff, params['lowpass_hz'], sample_rate)

                print(f"[Timing] dF/F for {fiber_col}: {time_module.perf_counter() - t0:.3f}s (fast path)")

            except Exception as e:
                print(f"[Photometry] Error computing dF/F for {fiber_col}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Build intermediates dict for plotting (times in SECONDS for plotter)
            intermediates = {
                'time': common_time_sec,
                'iso_aligned': iso_signal,
                'gcamp_aligned': gcamp_signal,
                'fitted_iso': fitted_iso,
                'dff_raw': dff_raw,
                'detrend_curve': detrend_curve,
                'fit_params': fit_params
            }

            # Store results (times in SECONDS)
            dff_results = {
                'time': common_time_sec,
                'dff': dff,
                'intermediates': intermediates
            }
            self._processed_results[controls_key][fiber_col] = dff_results

            # Fit region is always enabled (draggable on plots)
            fit_region_enabled = True
            fit_start = controls['spin_fit_start'].value() if 'spin_fit_start' in controls else 0.0
            fit_end = controls['spin_fit_end'].value() if 'spin_fit_end' in controls else 10.0

            # Initialize fit window to sensible defaults if both are 0
            if fit_start == 0 and fit_end == 0:
                t_min_min = common_time_min[0]  # Start time in minutes
                t_max_min = common_time_min[-1]  # End time in minutes
                duration_min = t_max_min - t_min_min

                # Default: start at 5 min, end at 1 min before recording end
                # Fallback for short recordings (< 7 min): use first 20% of recording
                if duration_min >= 7.0:
                    fit_start = t_min_min + 5.0  # Start at 5 min
                    fit_end = t_max_min - 1.0    # End 1 min before end
                else:
                    # Short recording fallback: first 20% of recording
                    fit_start = t_min_min
                    fit_end = t_min_min + duration_min * 0.2
                # Update the spinboxes with these defaults
                if 'spin_fit_start' in controls:
                    controls['spin_fit_start'].blockSignals(True)
                    controls['spin_fit_start'].setValue(fit_start)
                    controls['spin_fit_start'].blockSignals(False)
                if 'spin_fit_end' in controls:
                    controls['spin_fit_end'].blockSignals(True)
                    controls['spin_fit_end'].setValue(fit_end)
                    controls['spin_fit_end'].blockSignals(False)

            # Create callback for fit region changes
            def on_fit_region_changed(start, end, idx=exp_idx):
                ctrl_key = f'exp_{idx}'
                if hasattr(self, '_dff_controls') and ctrl_key in self._dff_controls:
                    ctrls = self._dff_controls[ctrl_key]
                    if 'spin_fit_start' in ctrls:
                        ctrls['spin_fit_start'].blockSignals(True)
                        ctrls['spin_fit_start'].setValue(start)
                        ctrls['spin_fit_start'].blockSignals(False)
                    if 'spin_fit_end' in ctrls:
                        ctrls['spin_fit_end'].blockSignals(True)
                        ctrls['spin_fit_end'].setValue(end)
                        ctrls['spin_fit_end'].blockSignals(False)
                self._compute_dff(idx)

            # Use preprocessed AI data (already on common time base)
            ai_data_for_plot = self._preprocessed.get('ai_channels', {})

            # Create fiber_data for plotter (times in SECONDS, on common time base)
            fdata_for_plot = {
                'iso_time': common_time_sec,
                'iso': iso_signal,
                'gcamp_time': common_time_sec,
                'gcamp': gcamp_signal,
                'label': fiber_col
            }

            # Use the plotter to create all plots for this fiber
            plot_items, next_row = self._plotter.plot_experiment(
                exp_idx=exp_idx,
                fiber_col=fiber_col,
                fiber_data=fdata_for_plot,
                dff_results=dff_results,
                ai_data=ai_data_for_plot,
                ai_time=common_time_sec,
                ai_channels=ai_channels if row == 0 else [],
                show_intermediates=show_intermediates,
                graphics_layout=graphics_layout,
                fit_region_enabled=fit_region_enabled,
                fit_start=fit_start,
                fit_end=fit_end,
                on_region_changed=on_fit_region_changed
            )

            all_plot_items.extend(plot_items)

        # Update fit parameters display
        self._update_fit_params_label(exp_idx)

        # Mark this experiment as clean (not needing recompute)
        setattr(self, f'_exp_{exp_idx}_dirty', False)

        # Force layout update
        from PyQt6.QtWidgets import QApplication
        graphics_layout.update()
        QApplication.processEvents()

        total_time = time_module.perf_counter() - t_start
        print(f"[Timing] Total _compute_dff for Exp {exp_idx + 1}: {total_time:.3f}s ({len(all_plot_items)} plots)")

    def _get_ai_channels_for_experiment(self, exp_idx: int) -> List[Tuple[int, str]]:
        """Get list of AI channels assigned to this experiment.

        Args:
            exp_idx: Experiment index

        Returns:
            List of (column_index, channel_name) tuples
        """
        ai_channels = []

        # Check if we have AI data (either raw DataFrame or preprocessed)
        has_raw_ai = self._ai_data is not None
        has_preprocessed_ai = (self._preprocessed is not None and
                               'ai_channels' in self._preprocessed and
                               len(self._preprocessed['ai_channels']) > 0)

        if not has_raw_ai and not has_preprocessed_ai:
            return ai_channels

        # Build column name to index mapping from ai_columns (populated during load)
        col_name_to_idx = {}
        if hasattr(self, 'ai_columns'):
            for col_idx, info in self.ai_columns.items():
                col_name_to_idx[info['column']] = col_idx

        # Get experiment assignments
        exp_assignments = self._get_experiment_assignments()

        # Find AI channels for this experiment from assignments
        for channel_name, assigned_exp in exp_assignments.items():
            if channel_name.startswith('AI-') and (assigned_exp == exp_idx or assigned_exp == -1):
                col_name = channel_name[3:]  # Remove "AI-" prefix
                col_idx = col_name_to_idx.get(col_name)
                if col_idx is not None:
                    ai_channels.append((col_idx, channel_name))
                elif has_raw_ai and col_name in self._ai_data.columns:
                    col_idx = list(self._ai_data.columns).index(col_name)
                    ai_channels.append((col_idx, channel_name))

        # Also add checked AI columns not in assignments
        if hasattr(self, 'ai_columns'):
            for col_idx, info in self.ai_columns.items():
                if info['checkbox'].isChecked():
                    channel_name = f"AI-{info['column']}"
                    if channel_name not in exp_assignments:
                        ai_channels.append((col_idx, channel_name))

        # If we loaded from cached/NPZ data and have no AI column widgets yet,
        # use preprocessed AI channel indices directly
        if not ai_channels and has_preprocessed_ai:
            for col_idx in self._preprocessed['ai_channels'].keys():
                channel_name = f"AI-{col_idx}"
                ai_channels.append((col_idx, channel_name))

        return ai_channels

    def _add_ai_channels_to_experiment(self, exp_idx: int, graphics_layout, start_row: int, first_plot) -> list:
        """Add AI channels assigned to this experiment.

        Args:
            exp_idx: Experiment index
            graphics_layout: GraphicsLayoutWidget to add plots to
            start_row: Starting row for AI plots
            first_plot: First plot for X-axis linking

        Returns:
            List of created plot items
        """
        ai_plots = []

        # Check if we have AI data and timestamps
        if self._ai_data is None or self._timestamps is None:
            return ai_plots

        # Build a mapping from column names to indices
        col_name_to_idx = {}
        if hasattr(self, 'ai_columns'):
            for col_idx, info in self.ai_columns.items():
                col_name_to_idx[info['column']] = col_idx

        # Get experiment assignments to find AI channels for this experiment
        exp_assignments = self._get_experiment_assignments()

        # Find AI channels assigned to this experiment or unassigned (-1)
        ai_channels_for_exp = []
        for channel_name, assigned_exp in exp_assignments.items():
            # Show AI channels that are either assigned to this exp OR unassigned (-1)
            if channel_name.startswith('AI-') and (assigned_exp == exp_idx or assigned_exp == -1):
                # Extract column name from channel name (format is "AI-<column_name>")
                col_name = channel_name[3:]  # Remove "AI-" prefix
                # Look up the column index from our mapping
                col_idx = col_name_to_idx.get(col_name)
                if col_idx is not None:
                    ai_channels_for_exp.append((col_idx, channel_name))
                else:
                    # Try direct lookup if column name is in DataFrame
                    if col_name in self._ai_data.columns:
                        col_idx = list(self._ai_data.columns).index(col_name)
                        ai_channels_for_exp.append((col_idx, channel_name))

        # Also add AI channels that aren't in the assignments yet (new channels)
        if hasattr(self, 'ai_columns'):
            for col_idx, info in self.ai_columns.items():
                if info['checkbox'].isChecked():
                    channel_name = f"AI-{info['column']}"
                    if channel_name not in exp_assignments:
                        ai_channels_for_exp.append((col_idx, channel_name))

        if not ai_channels_for_exp:
            return ai_plots

        # Get AI timestamps in minutes
        # Timestamps file uses ComputerTimestamp format (milliseconds)
        ai_time = self._timestamps.flatten() if hasattr(self._timestamps, 'flatten') else self._timestamps
        ai_time_min = ai_time / 60000.0  # Always ms to minutes

        row = start_row
        for col_idx, channel_name in ai_channels_for_exp:
            if col_idx >= self._ai_data.shape[1]:
                continue

            ai_signal = self._ai_data.iloc[:, col_idx].values

            plot = graphics_layout.addPlot(row=row, col=0)
            ai_plots.append(plot)
            self._style_dff_plot(plot, first_plot)
            if first_plot:
                plot.setXLink(first_plot)

            plot.plot(ai_time_min, ai_signal, pen=pg.mkPen('#ce9178', width=1))
            plot.setLabel('left', channel_name, color='#ce9178')
            plot.setTitle(f'{channel_name}', color='#888888', size='8pt')
            row += 1

        return ai_plots

    def _add_fit_region_to_plots(self, exp_idx: int, plot_items: list):
        """Add interactive fit region selector to ALL photometry plots.

        Creates a region on each plot that stays synchronized. Dragging any
        region updates all others and the spinboxes.

        Args:
            exp_idx: Experiment index
            plot_items: List of plot items to add region to (photometry only)
        """
        controls_key = f'exp_{exp_idx}'
        controls = self._dff_controls.get(controls_key, {})

        if not hasattr(self, '_fit_regions'):
            self._fit_regions = {}

        # Remove existing regions if any
        if exp_idx in self._fit_regions:
            old_regions = self._fit_regions[exp_idx]
            if isinstance(old_regions, list):
                for old_region, plot in zip(old_regions, plot_items):
                    try:
                        plot.removeItem(old_region)
                    except Exception:
                        pass
            else:
                # Handle old single-region format
                for plot in plot_items:
                    try:
                        plot.removeItem(old_regions)
                    except Exception:
                        pass

        if not plot_items:
            return

        # Get current fit range values
        fit_start = controls.get('spin_fit_start')
        fit_end = controls.get('spin_fit_end')

        start_val = fit_start.value() if fit_start else 0
        end_val = fit_end.value() if fit_end else 10

        # Create a region for each photometry plot
        regions = []
        for i, plot in enumerate(plot_items):
            region = pg.LinearRegionItem(
                values=[start_val, end_val],
                brush=pg.mkBrush(100, 100, 255, 50),
                pen=pg.mkPen('#6464ff', width=2),
                movable=True
            )
            plot.addItem(region)
            regions.append(region)

        # Store all regions
        self._fit_regions[exp_idx] = regions

        # Flag to prevent recursive updates
        self._updating_regions = False

        # Connect first region to control all others
        def on_region_changed():
            if self._updating_regions:
                return
            self._updating_regions = True

            min_val, max_val = regions[0].getRegion()

            # Update spinboxes
            if fit_start:
                fit_start.blockSignals(True)
                fit_start.setValue(min_val)
                fit_start.blockSignals(False)
            if fit_end:
                fit_end.blockSignals(True)
                fit_end.setValue(max_val)
                fit_end.blockSignals(False)

            # Sync all other regions
            for region in regions[1:]:
                region.blockSignals(True)
                region.setRegion([min_val, max_val])
                region.blockSignals(False)

            self._updating_regions = False

        def on_region_change_finished():
            # Trigger recompute with new fit window
            self._compute_dff(exp_idx)

        # Connect only the first region to handlers (it controls all others)
        regions[0].sigRegionChanged.connect(on_region_changed)
        regions[0].sigRegionChangeFinished.connect(on_region_change_finished)

        # Make other regions sync back to first when dragged
        for region in regions[1:]:
            def sync_to_first(r=region):
                if self._updating_regions:
                    return
                self._updating_regions = True
                min_val, max_val = r.getRegion()
                regions[0].setRegion([min_val, max_val])
                self._updating_regions = False
            region.sigRegionChanged.connect(sync_to_first)
            region.sigRegionChangeFinished.connect(on_region_change_finished)

    def _style_dff_plot(self, plot, first_plot):
        """Apply consistent styling to a dF/F plot."""
        plot.setMouseEnabled(x=True, y=True)
        plot.vb.setMouseMode(pg.ViewBox.PanMode)
        plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)
        plot.getAxis('left').setPen('#3e3e42')
        plot.getAxis('left').setTextPen('#cccccc')
        plot.getAxis('bottom').setPen('#3e3e42')
        plot.getAxis('bottom').setTextPen('#cccccc')
        plot.showGrid(x=False, y=False)

    def _update_secondary_viewboxes(self, exp_idx: int):
        """Update all secondary Y-axis ViewBox geometries for an experiment.

        This ensures the secondary ViewBoxes (for dual Y-axis plots) are properly
        sized after the layout is complete. Called after all plots are created.

        Args:
            exp_idx: Experiment index
        """
        if not hasattr(self, '_secondary_viewboxes') or exp_idx not in self._secondary_viewboxes:
            return

        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication

        # Process events to ensure layout is settled
        QApplication.processEvents()

        def update_all():
            if not hasattr(self, '_secondary_viewboxes') or exp_idx not in self._secondary_viewboxes:
                return
            for plot, vb in self._secondary_viewboxes[exp_idx]:
                try:
                    vb.setGeometry(plot.vb.sceneBoundingRect())
                    vb.linkedViewChanged(plot.vb, vb.XAxis)
                    vb.enableAutoRange(axis=pg.ViewBox.YAxis)
                except Exception as e:
                    print(f"[Photometry] Error updating secondary ViewBox: {e}")

        # Update immediately
        update_all()

        # Also update after a delay to catch any late layout changes
        QTimer.singleShot(100, update_all)
        QTimer.singleShot(300, update_all)

    def _preprocess_all_data(self):
        """Preprocess all data once after loading.

        Creates a common time base and interpolates all signals to it.
        This is called once after files are loaded, not per-experiment.

        Stores result in self._preprocessed with structure:
        {
            'common_time': np.ndarray,  # Normalized time in seconds, starts at 0
            'sample_rate': float,       # Hz (from photometry)
            'time_offset': float,       # Original start time (for reference)
            'duration': float,          # Total duration in seconds
            'fibers': {
                'G0': {'iso': np.ndarray, 'gcamp': np.ndarray},
                'G1': {'iso': np.ndarray, 'gcamp': np.ndarray},
            },
            'ai_channels': {
                0: np.ndarray,  # Interpolated to common_time
                1: np.ndarray,
            },
        }
        """
        import time as time_module
        from scipy import interpolate

        if self._fp_data is None:
            self._preprocessed = None
            return

        t0 = time_module.perf_counter()
        print("[Photometry] Preprocessing all data...")

        # Find time and LED columns
        # Prefer ComputerTimestamp (milliseconds) to match AI timestamps file
        time_col = None
        led_col = None
        for col in self._fp_data.columns:
            col_lower = col.lower()
            # Prefer ComputerTimestamp which matches the timestamps file format
            if 'computertimestamp' in col_lower:
                time_col = col
            elif time_col is None and ('timestamp' in col_lower or col_lower == 'time'):
                time_col = col
            if led_col is None and ('ledstate' in col_lower or col_lower == 'led'):
                led_col = col

        if not time_col or not led_col:
            print(f"[Photometry] Cannot preprocess: time_col={time_col}, led_col={led_col}")
            self._preprocessed = None
            return

        print(f"[Photometry] Using time column: {time_col}")

        data = self._fp_data

        # Separate by LED state: 1 = isosbestic, 2 = GCaMP
        iso_mask = data[led_col] == 1
        gcamp_mask = data[led_col] == 2

        # Get raw timestamps
        iso_time_raw = data.loc[iso_mask, time_col].values
        gcamp_time_raw = data.loc[gcamp_mask, time_col].values

        if len(iso_time_raw) == 0 or len(gcamp_time_raw) == 0:
            print("[Photometry] No iso or gcamp data found")
            self._preprocessed = None
            return

        # Determine time unit based on column name (most reliable method)
        # ComputerTimestamp is in milliseconds, SystemTimestamp is in seconds
        # This matches the approach used in _update_preview_plot() which works correctly
        if 'system' in time_col.lower():
            # SystemTimestamp: already in seconds
            iso_time_sec = iso_time_raw
            gcamp_time_sec = gcamp_time_raw
            # Calculate photometry sample rate from data
            time_range = iso_time_raw[-1] - iso_time_raw[0]
            fp_sample_rate = len(iso_time_raw) / time_range if time_range > 0 else 30.0
            print(f"[Photometry] Using SystemTimestamp (seconds), FP SR ~{fp_sample_rate:.1f} Hz")
        else:
            # ComputerTimestamp: milliseconds, convert to seconds
            iso_time_sec = iso_time_raw / 1000.0
            gcamp_time_sec = gcamp_time_raw / 1000.0
            # Calculate photometry sample rate from data (in seconds)
            time_range_sec = (iso_time_raw[-1] - iso_time_raw[0]) / 1000.0
            fp_sample_rate = len(iso_time_raw) / time_range_sec if time_range_sec > 0 else 30.0
            print(f"[Photometry] Using ComputerTimestamp (ms->s), FP SR ~{fp_sample_rate:.1f} Hz")

        # Find common time range (overlap of iso and gcamp)
        fp_t_start = max(iso_time_sec[0], gcamp_time_sec[0])
        fp_t_end = min(iso_time_sec[-1], gcamp_time_sec[-1])

        # Check if AI data is available - if so, use its higher sample rate
        use_ai_timebase = False
        ai_time_sec = None
        ai_sample_rate = None

        if self._ai_data is not None and self._timestamps is not None:
            # Get AI timestamps and convert to seconds
            ai_time_raw = self._timestamps.flatten()
            ai_time_sec = ai_time_raw / 1000.0  # Always ms to seconds
            ai_duration = ai_time_sec[-1] - ai_time_sec[0]
            ai_sample_rate = len(ai_time_raw) / ai_duration if ai_duration > 0 else 1000.0
            print(f"[Photometry] AI data available: {len(ai_time_raw)} samples, "
                  f"{ai_duration:.1f}s duration, SR ~{ai_sample_rate:.1f} Hz")

            # Use AI timebase if it has significantly higher sample rate
            if ai_sample_rate > fp_sample_rate * 2:
                use_ai_timebase = True
                print(f"[Photometry] Using AI timebase ({ai_sample_rate:.1f} Hz) - upsampling photometry")
            else:
                print(f"[Photometry] AI sample rate not higher, using photometry timebase")

        # Determine the common time array based on available data
        if use_ai_timebase and ai_time_sec is not None:
            # Use AI timebase - normalize to start at 0
            ai_time_norm = ai_time_sec - ai_time_sec[0]

            # Find overlap between AI and photometry time ranges
            # Convert photometry times to same reference (normalized to AI start)
            fp_offset = fp_t_start - ai_time_sec[0]  # How far into AI recording FP starts

            # Find the overlap region
            t_start = max(0, fp_offset)  # Start at AI time 0 or FP start, whichever is later
            t_end = min(ai_time_norm[-1], fp_t_end - ai_time_sec[0])  # End at earlier of AI or FP end

            # Use AI sample rate for common time
            sample_rate = ai_sample_rate
            duration = t_end - t_start
            n_points = int(duration * sample_rate)

            # Create common time at AI sample rate
            common_time = np.linspace(t_start, t_end, n_points)

            # Store original time offset (relative to AI start)
            time_offset = ai_time_sec[0] + t_start

            print(f"[Photometry] Common time (AI-based): {duration:.1f}s, {n_points} points at {sample_rate:.1f} Hz")
        else:
            # Fall back to photometry timebase
            sample_rate = fp_sample_rate
            t_start = fp_t_start
            t_end = fp_t_end
            time_offset = t_start
            duration = t_end - t_start
            n_points = int(duration * sample_rate)
            common_time = np.linspace(0, duration, n_points)

            print(f"[Photometry] Common time (FP-based): {duration:.1f}s, {n_points} points at {sample_rate:.1f} Hz")

        # Get ALL fiber columns (not just selected - preprocess everything)
        fiber_columns = [col for col in data.columns
                        if col.startswith('G') and col[1:].isdigit()]

        # Batch extract all fiber data (one .loc[] call per mask instead of per-fiber)
        iso_matrix = data.loc[iso_mask, fiber_columns].values    # shape: (n_iso, n_fibers)
        gcamp_matrix = data.loc[gcamp_mask, fiber_columns].values  # shape: (n_gcamp, n_fibers)

        # Normalize photometry time for interpolation (computed once, shared by all fibers)
        if use_ai_timebase and ai_time_sec is not None:
            iso_time_norm = iso_time_sec - ai_time_sec[0]
            gcamp_time_norm = gcamp_time_sec - ai_time_sec[0]
        else:
            iso_time_norm = iso_time_sec - fp_t_start
            gcamp_time_norm = gcamp_time_sec - fp_t_start

        # Identify valid fibers (not all NaN) before parallel interpolation
        valid_indices = []
        for idx, fiber_col in enumerate(fiber_columns):
            if not (np.all(np.isnan(iso_matrix[:, idx])) or np.all(np.isnan(gcamp_matrix[:, idx]))):
                valid_indices.append(idx)

        # Parallel interpolation across fibers
        def interp_fiber(idx):
            """Interpolate a single fiber's iso and gcamp signals to common time."""
            try:
                iso_interp = np.interp(common_time, iso_time_norm, iso_matrix[:, idx])
                gcamp_interp = np.interp(common_time, gcamp_time_norm, gcamp_matrix[:, idx])
                return (fiber_columns[idx], iso_interp, gcamp_interp)
            except Exception as e:
                print(f"[Photometry] Error interpolating {fiber_columns[idx]}: {e}")
                return None

        fibers = {}
        if len(valid_indices) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(len(valid_indices), 4)) as exe:
                results = list(exe.map(interp_fiber, valid_indices))
            for result in results:
                if result is not None:
                    fiber_col, iso_interp, gcamp_interp = result
                    fibers[fiber_col] = {'iso': iso_interp, 'gcamp': gcamp_interp}
        else:
            # Single fiber — no threading overhead
            for idx in valid_indices:
                result = interp_fiber(idx)
                if result is not None:
                    fiber_col, iso_interp, gcamp_interp = result
                    fibers[fiber_col] = {'iso': iso_interp, 'gcamp': gcamp_interp}

        if use_ai_timebase:
            print(f"[Photometry] Upsampled {len(fibers)} fiber signals from ~{fp_sample_rate:.1f} Hz to {sample_rate:.1f} Hz")
        else:
            print(f"[Photometry] Interpolated {len(fibers)} fiber signals at {sample_rate:.1f} Hz")

        # Process AI channels
        ai_channels = {}
        if self._ai_data is not None and self._timestamps is not None:
            # ai_time_sec already computed above if AI data available
            if ai_time_sec is None:
                ai_time_raw = self._timestamps.flatten()
                ai_time_sec = ai_time_raw / 1000.0

            # Normalize AI time to start at 0
            ai_time_norm = ai_time_sec - ai_time_sec[0]

            for col_idx, col_name in enumerate(self._ai_data.columns):
                ai_signal = self._ai_data[col_name].values

                # Handle length mismatch between timestamps and AI data
                min_len = min(len(ai_time_norm), len(ai_signal))
                if min_len < 10:
                    continue

                try:
                    if use_ai_timebase:
                        # Using AI timebase - just extract the samples in the common time range
                        # Find indices that fall within common_time range
                        start_idx = np.searchsorted(ai_time_norm[:min_len], common_time[0])
                        end_idx = np.searchsorted(ai_time_norm[:min_len], common_time[-1])

                        # Extract and resample to exact common_time length if needed
                        ai_segment = ai_signal[start_idx:end_idx]
                        ai_time_segment = ai_time_norm[start_idx:end_idx]

                        # Interpolate to exact common_time points (should be very close, minimal interpolation)
                        if len(ai_segment) > 0:
                            ai_interp = np.interp(common_time, ai_time_segment, ai_segment)
                        else:
                            continue
                    else:
                        # Using FP timebase - downsample AI to match
                        ai_interp = np.interp(
                            common_time,
                            ai_time_norm[:min_len],
                            ai_signal[:min_len]
                        )

                    ai_channels[col_idx] = ai_interp
                except Exception as e:
                    print(f"[Photometry] Error processing AI channel {col_idx}: {e}")
                    continue

            if use_ai_timebase:
                print(f"[Photometry] Extracted {len(ai_channels)} AI channels at native {ai_sample_rate:.1f} Hz")
            else:
                print(f"[Photometry] Downsampled {len(ai_channels)} AI channels to {sample_rate:.1f} Hz")

        # Store preprocessed data
        self._preprocessed = {
            'common_time': common_time,
            'sample_rate': sample_rate,
            'time_offset': time_offset,
            'duration': duration,
            'fibers': fibers,
            'ai_channels': ai_channels
        }

        print(f"[Timing] Preprocessing complete: {time_module.perf_counter() - t0:.3f}s")
        print(f"[Photometry] Preprocessed {len(fibers)} fibers, {len(ai_channels)} AI channels")

    def _prepare_fiber_data(self) -> Dict:
        """Prepare RAW fiber data for dF/F computation.

        Returns:
            Dict mapping fiber column names to their iso/gcamp signals and times.
            Time is in MINUTES (as required by compute_dff_full).

        This returns the original raw data (not interpolated) so that
        compute_dff_full() can do its own proper interpolation.
        """
        if self._fp_data is None:
            print("[Photometry] No FP data available")
            return {}

        # Get selected fiber columns
        selected_fibers = [col for col, info in self.fiber_columns.items()
                          if info['checkbox'].isChecked()]
        if not selected_fibers:
            return {}

        # Find time and LED columns (prefer ComputerTimestamp to match AI timestamps)
        time_col = None
        led_col = None
        for col in self._fp_data.columns:
            col_lower = col.lower()
            if 'computertimestamp' in col_lower:
                time_col = col
            elif time_col is None and ('timestamp' in col_lower or col_lower == 'time'):
                time_col = col
            if led_col is None and ('ledstate' in col_lower or col_lower == 'led'):
                led_col = col

        if not time_col or not led_col:
            print(f"[Photometry] Cannot prepare fiber data: time_col={time_col}, led_col={led_col}")
            return {}

        data = self._fp_data

        # Separate by LED state: 1 = isosbestic, 2 = GCaMP
        iso_mask = data[led_col] == 1
        gcamp_mask = data[led_col] == 2

        # Get raw timestamps
        iso_time_raw = data.loc[iso_mask, time_col].values
        gcamp_time_raw = data.loc[gcamp_mask, time_col].values

        if len(iso_time_raw) == 0 or len(gcamp_time_raw) == 0:
            print("[Photometry] No iso or gcamp data found")
            return {}

        # Normalize time to start at 0 and convert to MINUTES
        # (compute_dff_full expects time in minutes)
        t_min = min(iso_time_raw[0], gcamp_time_raw[0])

        # Determine timestamp units based on column name (most reliable method)
        # ComputerTimestamp is in milliseconds, SystemTimestamp is in seconds
        # This matches the approach used in _update_preview_plot() which works correctly
        if 'system' in time_col.lower():
            time_divisor = 60  # SystemTimestamp: seconds to minutes
        else:
            time_divisor = 60000  # ComputerTimestamp: milliseconds to minutes

        iso_time_min = (iso_time_raw - t_min) / time_divisor
        gcamp_time_min = (gcamp_time_raw - t_min) / time_divisor

        print(f"[Photometry] Raw fiber data: time_col={time_col}, divisor={time_divisor}, range 0 to {iso_time_min[-1]:.1f} min")

        fiber_data = {}
        for fiber_col in selected_fibers:
            iso_signal = data.loc[iso_mask, fiber_col].values
            gcamp_signal = data.loc[gcamp_mask, fiber_col].values

            # Skip if all NaN
            if np.all(np.isnan(iso_signal)) or np.all(np.isnan(gcamp_signal)):
                continue

            fiber_data[fiber_col] = {
                'iso_time': iso_time_min,
                'iso': iso_signal,
                'gcamp_time': gcamp_time_min,
                'gcamp': gcamp_signal,
                'label': fiber_col
            }

        return fiber_data

    def _get_dff_params(self, exp_idx: int) -> Dict:
        """Get dF/F processing parameters from UI controls for an experiment."""
        controls_key = f'exp_{exp_idx}'
        if not hasattr(self, '_dff_controls') or controls_key not in self._dff_controls:
            return {
                'method': 'fitted',
                'detrend_method': 'none',
                'lowpass_hz': None,
                'exclude_start_min': 0.0,
                'fit_start': 0.0,
                'fit_end': 0.0
            }

        controls = self._dff_controls[controls_key]
        params = {
            'method': 'fitted',
            'detrend_method': 'none',
            'lowpass_hz': None,
            'exclude_start_min': 0.0,
            'fit_start': 0.0,
            'fit_end': 0.0
        }

        # dF/F method
        if 'combo_dff_method' in controls:
            method_text = controls['combo_dff_method'].currentText()
            params['method'] = 'fitted' if 'Fitted' in method_text else 'simple'

        # Detrend method
        if 'combo_detrend_method' in controls:
            detrend_text = controls['combo_detrend_method'].currentText().lower()
            if 'linear' in detrend_text:
                params['detrend_method'] = 'linear'
            elif 'biexp' in detrend_text:
                params['detrend_method'] = 'biexponential'
            elif 'exp' in detrend_text:
                params['detrend_method'] = 'exponential'
            else:
                params['detrend_method'] = 'none'

        # Low-pass filter
        if 'chk_lowpass' in controls and controls['chk_lowpass'].isChecked():
            if 'spin_lowpass_hz' in controls:
                params['lowpass_hz'] = controls['spin_lowpass_hz'].value()

        # Exclude start
        if 'spin_exclude_start' in controls:
            params['exclude_start_min'] = controls['spin_exclude_start'].value()

        # Fit range (always used)
        if 'spin_fit_start' in controls:
            params['fit_start'] = controls['spin_fit_start'].value()
        if 'spin_fit_end' in controls:
            params['fit_end'] = controls['spin_fit_end'].value()

        return params

    def _update_fit_params_label(self, exp_idx: int):
        """Update the fit parameters label with computation results."""
        controls_key = f'exp_{exp_idx}'
        if not hasattr(self, '_dff_controls') or controls_key not in self._dff_controls:
            return

        controls = self._dff_controls[controls_key]
        if 'fit_params_label' not in controls:
            return

        label = controls['fit_params_label']

        if not hasattr(self, '_processed_results') or controls_key not in self._processed_results:
            label.setText("No fit performed yet")
            return

        results = self._processed_results[controls_key]
        if not results:
            label.setText("No fit performed yet")
            return

        lines = []
        for fiber_col, fiber_results in results.items():
            intermediates = fiber_results.get('intermediates', {})
            fit_params = intermediates.get('fit_params', {}) if intermediates else {}

            if fit_params:
                lines.append(f"--- {fiber_col} ---")
                method = fit_params.get('method', 'unknown')
                if method == 'fitted':
                    lines.append(f"  Method: Fitted regression")
                    lines.append(f"  Slope: {fit_params.get('slope', 0):.4f}")
                    lines.append(f"  R²: {fit_params.get('r_squared', 0):.4f}")
                else:
                    lines.append(f"  Method: Simple subtraction")
                    lines.append(f"  Iso mean: {fit_params.get('iso_mean', 0):.2f}")

                detrend = fit_params.get('detrend_method', 'none')
                lines.append(f"  Detrend: {detrend}")

                if 'exp_tau' in fit_params:
                    lines.append(f"  Tau: {fit_params['exp_tau']:.2f} min")

        label.setText("\n".join(lines) if lines else "No parameters")

    def _resize_preview_for_panels(self, n_panels: int):
        """Dynamically resize the preview area based on number of plot panels."""
        if n_panels < 1:
            n_panels = 1

        # Calculate required height
        required_height = max(400, n_panels * MIN_HEIGHT_PER_PANEL)

        # Update canvas container minimum height
        if hasattr(self, 'canvas_container'):
            self.canvas_container.setMinimumHeight(required_height)

        # Update preview section minimum height
        if hasattr(self, 'preview_section'):
            self.preview_section.setMinimumHeight(required_height + 100)

        # Process events to apply layout changes immediately
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def _resize_channel_table(self, n_rows: int):
        """Resize channel table to fit all rows."""
        if not hasattr(self, 'channel_table'):
            return

        row_height = 30
        header_height = 30
        padding = 20

        # Calculate required height (min 130, max 400)
        required_height = min(400, max(130, header_height + n_rows * row_height + padding))

        self.channel_table.setMinimumHeight(required_height)

        # Also update parent group box
        if hasattr(self, 'channel_group'):
            self.channel_group.setMinimumHeight(required_height + 60)

    def _get_fiber_type(self, fiber_col: str) -> str:
        """
        Determine if a fiber column is green (GCaMP) or red (560nm).

        Args:
            fiber_col: Column name like 'G0', 'R1', etc.

        Returns:
            'green' for G0/G1/etc, 'red' for R0/R1/etc, 'unknown' otherwise
        """
        col_upper = str(fiber_col).upper()
        if col_upper.startswith('G'):
            return 'green'
        elif col_upper.startswith('R'):
            return 'red'
        return 'unknown'

    def _get_signal_channel_name(self, fiber_col: str) -> str:
        """
        Get the signal channel name for a fiber (GCaMP or Red based on fiber type).

        Args:
            fiber_col: Column name like 'G0', 'R1', etc.

        Returns:
            Channel name like 'G0-GCaMP' or 'R0-Red'
        """
        fiber_type = self._get_fiber_type(fiber_col)
        if fiber_type == 'red':
            return f"{fiber_col}-Red"
        return f"{fiber_col}-GCaMP"

    def _update_channel_table(self):
        """Update channel assignment table with separated iso/gcamp channels."""
        if not hasattr(self, 'channel_table'):
            return

        table = self.channel_table
        table.setRowCount(0)

        n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1

        # Get list of fiber columns for auto-assignment
        fiber_col_list = [col for col, info in self.fiber_columns.items()
                          if info['checkbox'].isChecked()]

        # Add rows for each fiber column - separated into Signal and Iso
        # Signal type depends on fiber: G0/G1 = GCaMP (470nm), R0/R1 = Red (560nm)
        for fiber_idx, fiber_col in enumerate(fiber_col_list):
            info = self.fiber_columns[fiber_col]
            if info['checkbox'].isChecked():
                # Auto-assign: Fiber N -> Exp N (if enough experiments)
                auto_exp_idx = fiber_idx + 1 if fiber_idx < n_exp else 0  # 0 = "All"

                # Detect fiber type from column name (G = green/GCaMP, R = red/560nm)
                fiber_type = self._get_fiber_type(fiber_col)
                if fiber_type == 'red':
                    signal_label = "Red"
                    signal_name = f"{fiber_col}-Red"
                    signal_display = f"{fiber_col} (560nm)"
                else:
                    signal_label = "GCaMP"
                    signal_name = f"{fiber_col}-GCaMP"
                    signal_display = f"{fiber_col} (GCaMP)"

                # Add Signal row (GCaMP or Red depending on fiber type)
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(signal_name))
                table.setItem(row, 1, QTableWidgetItem(signal_display))

                combo = QComboBox()
                combo.addItem("None")  # Skip this channel
                combo.addItem("All")   # Option to assign to all experiments
                for i in range(n_exp):
                    combo.addItem(f"Exp {i+1}")
                # Auto-assign: +1 offset for "None" option
                combo.setCurrentIndex(auto_exp_idx + 1)  # Auto-assign (+1 for None option)
                combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
                table.setCellWidget(row, 2, combo)

                type_combo = QComboBox()
                type_combo.addItems(["GCaMP", "Isosbestic", "Red", "Other"])
                type_combo.setCurrentText(signal_label)
                table.setCellWidget(row, 3, type_combo)

                # Add Isosbestic row
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QTableWidgetItem(f"{fiber_col}-Iso"))
                table.setItem(row, 1, QTableWidgetItem(f"{fiber_col} (Iso)"))

                combo = QComboBox()
                combo.addItem("None")  # Skip this channel
                combo.addItem("All")   # Option to assign to all experiments
                for i in range(n_exp):
                    combo.addItem(f"Exp {i+1}")
                combo.setCurrentIndex(auto_exp_idx + 1)  # Auto-assign (same as signal, +1 for None)
                combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
                table.setCellWidget(row, 2, combo)

                type_combo = QComboBox()
                type_combo.addItems(["GCaMP", "Isosbestic", "Red", "Other"])
                type_combo.setCurrentText("Isosbestic")
                table.setCellWidget(row, 3, type_combo)

        # Add rows for AI channels
        if self._ai_data is not None:
            for idx, info in self.ai_columns.items():
                if info['checkbox'].isChecked():
                    col = info['column']
                    row = table.rowCount()
                    table.insertRow(row)
                    table.setItem(row, 0, QTableWidgetItem(f"AI-{col}"))
                    table.setItem(row, 1, QTableWidgetItem(f"AI {col}"))

                    combo = QComboBox()
                    combo.addItem("None")  # Skip this channel
                    combo.addItem("All")   # Option to assign to all experiments
                    for i in range(n_exp):
                        combo.addItem(f"Exp {i+1}")
                    combo.setCurrentIndex(1)  # Default to "All"
                    combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
                    table.setCellWidget(row, 2, combo)

                    type_combo = QComboBox()
                    type_combo.addItems(["Pleth", "Thermal", "Stim", "Other"])
                    table.setCellWidget(row, 3, type_combo)

        # Resize table to fit content
        self._resize_channel_table(table.rowCount())

    def _update_output_file_list(self):
        """Update the output files label to show the single NPZ file that will be created."""
        if not hasattr(self, 'output_files_label'):
            return

        # Get the folder name from FP data path
        fp_path = self.file_paths.get('fp_data')
        if fp_path:
            # Use the SAME logic as save_to_npz() to show accurate filename
            # Format: {grandparent}_{experiment_folder}_photometry.npz
            if fp_path.parent.name.lower().startswith('fp_data'):
                # FP_data_0/FP_data_0.csv structure
                experiment_folder = fp_path.parent.parent.name
                grandparent = fp_path.parent.parent.parent.name if fp_path.parent.parent.parent else ""
                save_location = fp_path.parent  # Same folder as fp_data file
            else:
                experiment_folder = fp_path.parent.name
                grandparent = fp_path.parent.parent.name if fp_path.parent.parent else ""
                save_location = fp_path.parent

            # Build filename matching save_to_npz() logic
            if grandparent and grandparent != experiment_folder:
                grandparent_short = grandparent[:20] if len(grandparent) > 20 else grandparent
                npz_filename = f"{grandparent_short}_{experiment_folder}_photometry.npz"
            else:
                npz_filename = f"{experiment_folder}_photometry.npz"

            n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1
            exp_text = f"Contains {n_exp} experiment(s)" if n_exp > 1 else "Contains 1 experiment"

            self.output_files_label.setText(
                f"📁 {npz_filename}\n"
                f"   Location: {save_location}\n"
                f"   {exp_text}"
            )
        else:
            self.output_files_label.setText("Load data to see output file...")

    def _on_fiber_column_changed(self):
        """Handle fiber column selection change."""
        self._update_channel_table()
        self._update_preview_plot()

    def _on_experiment_assignment_changed(self):
        """Handle experiment assignment change - update experiment tabs.

        Also links GCaMP and Iso channels from the same fiber so they're
        always assigned to the same experiment.
        """
        # Mark that changes have been made
        self._has_unsaved_changes = True

        # Get the combo that triggered this
        sender = self.sender()
        if sender is None or not hasattr(self, 'channel_table'):
            self._update_preview_plot()
            return

        # Find which row this combo is in
        table = self.channel_table
        sender_row = -1
        for row in range(table.rowCount()):
            if table.cellWidget(row, 2) is sender:
                sender_row = row
                break

        if sender_row < 0:
            self._update_preview_plot()
            return

        # Get channel name and find its paired channel
        channel_item = table.item(sender_row, 0)
        if channel_item is None:
            self._update_preview_plot()
            return

        channel_name = channel_item.text()
        new_exp = sender.currentIndex()  # 0 = All, 1 = Exp 1, etc.

        # Determine the paired channel name
        # Handle both GCaMP (green) and Red (560nm) signal channels
        paired_name = None
        if '-GCaMP' in channel_name:
            fiber_col = channel_name.replace('-GCaMP', '')
            paired_name = f"{fiber_col}-Iso"
        elif '-Red' in channel_name:
            fiber_col = channel_name.replace('-Red', '')
            paired_name = f"{fiber_col}-Iso"
        elif '-Iso' in channel_name:
            fiber_col = channel_name.replace('-Iso', '')
            # Check if this is a red or green fiber
            fiber_type = self._get_fiber_type(fiber_col)
            if fiber_type == 'red':
                paired_name = f"{fiber_col}-Red"
            else:
                paired_name = f"{fiber_col}-GCaMP"

        # Find and update the paired channel's combo
        if paired_name:
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item and item.text() == paired_name:
                    paired_combo = table.cellWidget(row, 2)
                    if paired_combo and paired_combo.currentIndex() != new_exp:
                        # Block signals to prevent infinite recursion
                        paired_combo.blockSignals(True)
                        paired_combo.setCurrentIndex(new_exp)
                        paired_combo.blockSignals(False)
                    break

        self._update_preview_plot()

        # Auto-compute dF/F for affected experiment(s)
        if self._fp_data is not None:
            if new_exp > 0:  # Assigned to specific experiment (1 = Exp 1, etc.)
                exp_idx = new_exp - 1
                if exp_idx in self._experiment_layouts:
                    print(f"[Photometry] Auto-computing dF/F for Exp {exp_idx + 1} after assignment change")
                    self._compute_dff(exp_idx)
            else:  # Assigned to "All" - compute for all experiments
                for exp_idx in self._experiment_layouts.keys():
                    print(f"[Photometry] Auto-computing dF/F for Exp {exp_idx + 1} after assignment change")
                    self._compute_dff(exp_idx)

    def _on_preview_tab_changed(self, index: int):
        """Handle preview tab change - render All Channels or compute dF/F as needed."""
        if index == 0:
            # "All Channels" tab - render preview if not already done
            if self._preprocessed is not None and not self._plot_items:
                print("[Photometry] Rendering All Channels preview (first view)")
                self._update_preview_plot()
        elif index > 0 and self._preprocessed is not None:
            # Experiment tabs
            exp_idx = index - 1  # Convert tab index to experiment index

            if exp_idx in self._experiment_layouts:
                # Only compute if this experiment hasn't been computed yet
                # or if it's marked as needing recompute
                if not hasattr(self, '_processed_results'):
                    self._processed_results = {}
                controls_key = f'exp_{exp_idx}'
                needs_compute = (
                    controls_key not in self._processed_results or
                    not self._processed_results[controls_key] or
                    getattr(self, f'_exp_{exp_idx}_dirty', True)
                )

                if needs_compute:
                    print(f"[Photometry] Computing dF/F for Exp {exp_idx + 1} (first view or params changed)")
                    self._compute_dff(exp_idx)
                    setattr(self, f'_exp_{exp_idx}_dirty', False)

    def _subsample_for_preview(self, time_arr: np.ndarray, signal_arr: np.ndarray,
                                max_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Subsample data for faster preview plotting."""
        if len(time_arr) <= max_points:
            return time_arr, signal_arr
        step = len(time_arr) // max_points
        return time_arr[::step], signal_arr[::step]

    def _handle_wheel_event(self, event, plot, axis=None):
        """Handle wheel events: shift+scroll = X zoom, otherwise pass to scroll area.

        Args:
            event: The wheel event
            plot: The plot item
            axis: Optional axis parameter (passed by PyQtGraph AxisItem, ignored)
        """
        from PyQt6.QtCore import Qt
        modifiers = event.modifiers()

        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            # Shift+scroll = zoom X axis
            delta = event.delta()
            scale_factor = 1.1 if delta > 0 else 0.9
            plot.vb.scaleBy((scale_factor, 1.0))
            event.accept()
        else:
            # Pass to parent scroll area
            event.ignore()

    def _add_context_menu_to_plot(self, plot, channel_name: str):
        """Add custom menu items to PyQtGraph's native ViewBox context menu."""
        vb = plot.vb
        menu = vb.menu

        # Add separator before our custom items
        menu.addSeparator()

        # Experiment assignment submenu
        exp_menu = menu.addMenu("Assign to Experiment")
        exp_menu.setProperty('channel_name', channel_name)

        # We need to rebuild this menu each time it's shown to get current state
        exp_menu.aboutToShow.connect(lambda m=exp_menu, ch=channel_name: self._populate_exp_menu(m, ch))

        # Signal type submenu
        type_menu = menu.addMenu("Signal Type")
        type_menu.aboutToShow.connect(lambda m=type_menu, ch=channel_name: self._populate_type_menu(m, ch))

    def _populate_exp_menu(self, menu, channel_name: str):
        """Populate experiment assignment submenu with current state."""
        menu.clear()

        n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1
        current_exp = self._get_channel_experiment(channel_name)

        action_all = menu.addAction("All Experiments")
        action_all.setCheckable(True)
        action_all.setChecked(current_exp == -1)
        action_all.triggered.connect(lambda: self._set_channel_experiment(channel_name, -1))

        for i in range(n_exp):
            action = menu.addAction(f"Exp {i + 1}")
            action.setCheckable(True)
            action.setChecked(current_exp == i)
            action.triggered.connect(lambda checked, idx=i: self._set_channel_experiment(channel_name, idx))

    def _populate_type_menu(self, menu, channel_name: str):
        """Populate signal type submenu with current state."""
        menu.clear()

        current_type = self._get_channel_type(channel_name)

        # Different options for fiber vs AI channels
        if '-GCaMP' in channel_name or '-Red' in channel_name or '-Iso' in channel_name:
            type_options = ["GCaMP", "Isosbestic", "Red", "Other"]
        elif channel_name.startswith('AI-'):
            type_options = ["Pleth", "Thermal", "Stim", "Other"]
        else:
            type_options = ["Other"]

        for signal_type in type_options:
            action = menu.addAction(signal_type)
            action.setCheckable(True)
            action.setChecked(current_type == signal_type)
            action.triggered.connect(lambda checked, t=signal_type: self._set_channel_type(channel_name, t))

    def _get_channel_experiment(self, channel_name: str) -> int:
        """Get current experiment assignment for a channel. Returns -1 for 'All'."""
        if not hasattr(self, 'channel_table'):
            return -1

        table = self.channel_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.text() == channel_name:
                combo = table.cellWidget(row, 2)
                if combo:
                    idx = combo.currentIndex()
                    return -1 if idx == 0 else idx - 1
        return -1

    def _set_channel_experiment(self, channel_name: str, exp_idx: int):
        """Set experiment assignment for a channel. -1 means 'All'."""
        if not hasattr(self, 'channel_table'):
            return

        table = self.channel_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.text() == channel_name:
                combo = table.cellWidget(row, 2)
                if combo:
                    # 0 = "All", 1 = Exp 1, etc.
                    combo_idx = 0 if exp_idx == -1 else exp_idx + 1
                    combo.setCurrentIndex(combo_idx)
                break

    def _get_channel_type(self, channel_name: str) -> str:
        """Get current signal type for a channel."""
        if not hasattr(self, 'channel_table'):
            return ""

        table = self.channel_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.text() == channel_name:
                type_combo = table.cellWidget(row, 3)
                if type_combo:
                    return type_combo.currentText()
        return ""

    def _set_channel_type(self, channel_name: str, signal_type: str):
        """Set signal type for a channel."""
        if not hasattr(self, 'channel_table'):
            return

        table = self.channel_table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.text() == channel_name:
                type_combo = table.cellWidget(row, 3)
                if type_combo:
                    type_combo.setCurrentText(signal_type)
                break

    def _reset_all_plots(self):
        """Reset all plots to show full data range."""
        for plot in self._plot_items:
            plot.enableAutoRange()
            plot.autoRange()
        # Also reset experiment tab plots
        for graphics_layout in self._experiment_layouts.values():
            for item in graphics_layout.items():
                if hasattr(item, 'autoRange'):
                    item.enableAutoRange()
                    item.autoRange()

    def _clear_all_plots(self):
        """Clear all plots and reset tracking lists.

        Call this at the START of load_from_cached_data() and load_from_npz()
        to prevent plot duplication when reopening the dialog.
        """
        print("[Photometry] Clearing all plots...")

        # Clear Tab 1 (raw preview) - use safe clear to handle inconsistent state
        if hasattr(self, 'graphics_layout') and self.graphics_layout is not None:
            _safe_clear_graphics_layout(self.graphics_layout)

        # Clear Tab 2 (experiment plots) - clear each layout then reset the dict
        for exp_idx, graphics_layout in list(self._experiment_layouts.items()):
            if graphics_layout is not None:
                _safe_clear_graphics_layout(graphics_layout)

        # CRITICAL: Reset plot tracking lists to prevent duplication
        self._plot_items = []
        self._plot_to_channel = {}

        # Clear fit regions from the plotter
        if hasattr(self, '_plotter') and self._plotter is not None:
            for exp_idx in list(self._plotter._fit_regions.keys()):
                self._plotter.clear_fit_regions(exp_idx)
            self._plotter._fit_regions = {}

        # Clear processed results
        if hasattr(self, '_processed_results'):
            self._processed_results = {}

        # Reset dirty flags for all experiments (force recompute on next view)
        for exp_idx in range(10):  # Support up to 10 experiments
            setattr(self, f'_exp_{exp_idx}_dirty', True)

        # Clear secondary viewboxes (for AI channels with separate Y-axis)
        if hasattr(self, '_secondary_viewboxes'):
            self._secondary_viewboxes = {}

        # Clear the dF/F debounce timers to prevent stale callbacks
        if hasattr(self, '_dff_controls'):
            for controls_key, controls in self._dff_controls.items():
                if '_debounce_timer' in controls:
                    try:
                        controls['_debounce_timer'].stop()
                    except:
                        pass

        print("[Photometry] All plots cleared")

    def _interpolate_to_timestamps(self, source_time: np.ndarray, source_signal: np.ndarray,
                                    target_time: np.ndarray) -> np.ndarray:
        """Interpolate source signal to match target timestamps.

        Args:
            source_time: Original timestamps (e.g., photometry at 20 Hz)
            source_signal: Original signal values
            target_time: Target timestamps to interpolate to (e.g., AI at 1000 Hz)

        Returns:
            Interpolated signal at target timestamps
        """
        if len(source_time) < 2 or len(source_signal) < 2:
            return np.full(len(target_time), np.nan)

        try:
            # np.interp is much faster than scipy.interp1d for linear interpolation
            # (no object creation overhead, C implementation)
            result = np.interp(target_time, source_time, source_signal,
                              left=np.nan, right=np.nan)
            return result
        except Exception as e:
            print(f"[Photometry] Interpolation error: {e}")
            return np.full(len(target_time), np.nan)

    def _update_preview_plot(self):
        """Update the raw signals preview plot with separated iso/gcamp channels using PyQtGraph."""
        # Check if we have data - either raw FP data or preprocessed
        has_raw_data = self._fp_data is not None
        has_preprocessed = self._preprocessed is not None and 'fibers' in self._preprocessed

        if not has_raw_data and not has_preprocessed:
            return

        # If we only have preprocessed data, use the faster path
        if not has_raw_data and has_preprocessed:
            self._update_preview_plot_from_preprocessed()
            return

        self._show_progress("Drawing raw signals preview...")

        try:
            # Clear existing plots
            _safe_clear_graphics_layout(self.graphics_layout)
            self._plot_items.clear()
            self._plot_to_channel.clear()

            # Get selected fiber columns
            selected_fibers = [col for col, info in self.fiber_columns.items()
                              if info['checkbox'].isChecked()]

            if not selected_fibers:
                # Show message in empty plot
                plot = self.graphics_layout.addPlot(row=0, col=0)
                plot.setTitle("No fiber columns selected", color='#888888')
                return

            # Auto-detect time and LED columns
            time_col = None
            led_col = None
            for col in self._fp_data.columns:
                col_lower = col.lower()
                if 'computertimestamp' in col_lower:
                    time_col = col
                elif time_col is None and ('timestamp' in col_lower or col_lower == 'time'):
                    time_col = col
                if led_col is None and ('ledstate' in col_lower or col_lower == 'led'):
                    led_col = col

            if not time_col or not led_col:
                plot = self.graphics_layout.addPlot(row=0, col=0)
                plot.setTitle(f"Could not detect time/LED columns", color='#ff6b6b')
                return

            # Determine timestamp units
            time_divisor = 60000  # Default: milliseconds
            if 'system' in time_col.lower():
                time_divisor = 60  # seconds

            # Separate isosbestic and GCaMP data
            data = self._fp_data.copy()
            iso_mask = data[led_col] == 1
            gcamp_mask = data[led_col] == 2

            iso_time_raw = data.loc[iso_mask, time_col].values
            gcamp_time_raw = data.loc[gcamp_mask, time_col].values

            # Check if we have AI timestamps for common time base
            normalize = hasattr(self, 'chk_normalize_time') and self.chk_normalize_time.isChecked()
            use_ai_timebase = (self._ai_data is not None and self._timestamps is not None
                               and len(self._timestamps) > 0)

            if use_ai_timebase:
                common_time_raw = self._timestamps.copy()
                if normalize:
                    t_min = np.min(common_time_raw)
                    common_time = (common_time_raw - t_min) / 60000
                    iso_time_for_interp = iso_time_raw - t_min
                    gcamp_time_for_interp = gcamp_time_raw - t_min
                else:
                    common_time = common_time_raw / 60000
                    iso_time_for_interp = iso_time_raw
                    gcamp_time_for_interp = gcamp_time_raw

                fiber_data = {}
                for fiber_col in selected_fibers:
                    iso_signal_raw = data.loc[iso_mask, fiber_col].values
                    gcamp_signal_raw = data.loc[gcamp_mask, fiber_col].values

                    iso_interp = self._interpolate_to_timestamps(
                        iso_time_for_interp, iso_signal_raw,
                        common_time_raw - (t_min if normalize else 0)
                    )
                    gcamp_interp = self._interpolate_to_timestamps(
                        gcamp_time_for_interp, gcamp_signal_raw,
                        common_time_raw - (t_min if normalize else 0)
                    )

                    fiber_data[fiber_col] = {
                        'iso_time': common_time, 'iso': iso_interp,
                        'gcamp_time': common_time, 'gcamp': gcamp_interp,
                        'label': fiber_col
                    }
                ai_time = common_time
            else:
                if normalize and len(iso_time_raw) > 0 and len(gcamp_time_raw) > 0:
                    all_times = np.concatenate([iso_time_raw, gcamp_time_raw])
                    t_min = np.min(all_times)
                    iso_time = (iso_time_raw - t_min) / time_divisor
                    gcamp_time = (gcamp_time_raw - t_min) / time_divisor
                else:
                    iso_time = iso_time_raw / time_divisor
                    gcamp_time = gcamp_time_raw / time_divisor

                fiber_data = {}
                for fiber_col in selected_fibers:
                    fiber_data[fiber_col] = {
                        'iso_time': iso_time, 'iso': data.loc[iso_mask, fiber_col].values,
                        'gcamp_time': gcamp_time, 'gcamp': data.loc[gcamp_mask, fiber_col].values,
                        'label': fiber_col
                    }
                ai_time = None

            # Count AI channels - include column index for consistent colors
            ai_channels = []
            if self._ai_data is not None and self._timestamps is not None:
                for col_idx, info in self.ai_columns.items():
                    if info['checkbox'].isChecked():
                        ai_channels.append((col_idx, info['column'], f"AI {info['column']}"))

            n_plots = len(fiber_data) + len(ai_channels)
            if n_plots == 0:
                return

            # Resize preview area
            self._resize_preview_for_panels(n_plots)

            # Create linked plots (shared X axis)
            row = 0
            first_plot = None

            # Plot each fiber
            for fiber_col, fdata in fiber_data.items():
                plot = self.graphics_layout.addPlot(row=row, col=0)
                self._plot_items.append(plot)

                # Store channel mapping for context menu (use GCaMP as primary)
                channel_name = f"{fiber_col}-GCaMP"
                self._plot_to_channel[plot] = channel_name

                # Configure mouse interaction
                plot.setMouseEnabled(x=True, y=True)
                plot.vb.setMouseMode(pg.ViewBox.PanMode)
                # Custom wheel event: shift+scroll = X zoom, otherwise ignore
                plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)

                # Add custom items to PyQtGraph's native context menu
                self._add_context_menu_to_plot(plot, channel_name)

                # Style the plot
                plot.setLabel('left', f'{fiber_col}', color='#ffffff')
                plot.setTitle(f'Raw: {fiber_col}', color='#cccccc', size='9pt')
                plot.getAxis('left').setPen('#3e3e42')
                plot.getAxis('left').setTextPen('#cccccc')
                plot.getAxis('bottom').setPen('#3e3e42')
                plot.getAxis('bottom').setTextPen('#cccccc')
                plot.getAxis('bottom').setStyle(showValues=False)  # Hide X ticks (not bottom)
                plot.showGrid(x=False, y=False)

                # Link X axis to first plot
                if first_plot is None:
                    first_plot = plot
                else:
                    plot.setXLink(first_plot)

                # Plot GCaMP (green)
                if len(fdata['gcamp_time']) > 0:
                    t_plot, s_plot = self._subsample_for_preview(fdata['gcamp_time'], fdata['gcamp'])
                    # Filter out NaN values
                    valid = ~np.isnan(s_plot)
                    plot.plot(t_plot[valid], s_plot[valid], pen=pg.mkPen('#00cc00', width=1), name='GCaMP')

                # Plot Isosbestic (blue) - use ViewBox for second Y axis
                if len(fdata['iso_time']) > 0:
                    t_plot, s_plot = self._subsample_for_preview(fdata['iso_time'], fdata['iso'])
                    valid = ~np.isnan(s_plot)
                    plot.plot(t_plot[valid], s_plot[valid], pen=pg.mkPen('#5555ff', width=1), name='Iso')

                row += 1

            # Plot AI channels
            if ai_channels and ai_time is not None:
                ai_colors = ['#ff9900', '#00cccc', '#ff66ff', '#ffff00', '#ff6666']

                for col_idx, col, label in ai_channels:
                    plot = self.graphics_layout.addPlot(row=row, col=0)
                    self._plot_items.append(plot)

                    # Store channel mapping for context menu
                    channel_name = f"AI-{col}"
                    self._plot_to_channel[plot] = channel_name

                    # Configure mouse interaction
                    plot.setMouseEnabled(x=True, y=True)
                    plot.vb.setMouseMode(pg.ViewBox.PanMode)
                    plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)

                    # Add custom items to PyQtGraph's native context menu
                    self._add_context_menu_to_plot(plot, channel_name)

                    # Use column index for consistent colors across all views
                    color = ai_colors[col_idx % len(ai_colors)]
                    plot.setLabel('left', label, color='#ffffff')
                    plot.setTitle(label, color='#cccccc', size='9pt')
                    plot.getAxis('left').setPen('#3e3e42')
                    plot.getAxis('left').setTextPen('#cccccc')
                    plot.getAxis('bottom').setPen('#3e3e42')
                    plot.getAxis('bottom').setTextPen('#cccccc')
                    plot.getAxis('bottom').setStyle(showValues=False)  # Hide X ticks (not bottom)
                    plot.showGrid(x=False, y=False)

                    if first_plot:
                        plot.setXLink(first_plot)

                    ai_signal = self._ai_data[col].values
                    min_len = min(len(ai_time), len(ai_signal))
                    t_arr = ai_time[:min_len]
                    s_arr = ai_signal[:min_len]

                    t_plot, s_plot = self._subsample_for_preview(t_arr, s_arr)
                    plot.plot(t_plot, s_plot, pen=pg.mkPen(color, width=1))

                    row += 1

            # Show X tick labels only on bottom plot
            if self._plot_items:
                self._plot_items[-1].getAxis('bottom').setStyle(showValues=True)
                self._plot_items[-1].setLabel('bottom', 'Time (minutes)', color='#cccccc')

            # Auto-range all plots to show full data
            for plot in self._plot_items:
                plot.enableAutoRange()
                plot.autoRange()

            # Note: Experiment tabs are NOT updated here.
            # They use _compute_dff() which is triggered by _on_preview_tab_changed()
            # when the user switches to an experiment tab.

        except Exception as e:
            print(f"[Photometry] Error drawing preview: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._hide_progress()

    def _update_preview_plot_from_preprocessed(self):
        """Update the raw signals preview plot from preprocessed data (when raw FP data is not available).

        This is used when reopening the dialog from the gear icon, where we have preprocessed
        fiber data but not the original FP_data DataFrame.
        """
        if self._preprocessed is None or 'fibers' not in self._preprocessed:
            return

        self._show_progress("Drawing preview from cached data...")

        try:
            # Clear existing plots
            _safe_clear_graphics_layout(self.graphics_layout)
            self._plot_items.clear()
            self._plot_to_channel.clear()

            fibers = self._preprocessed['fibers']
            common_time = self._preprocessed['common_time']
            ai_channels_data = self._preprocessed.get('ai_channels', {})

            if len(fibers) == 0:
                plot = self.graphics_layout.addPlot(row=0, col=0)
                plot.setTitle("No fiber data available", color='#888888')
                return

            # Convert time to minutes
            common_time_min = common_time / 60.0

            # Plot each fiber
            row = 0
            first_plot = None
            iso_colors = ['#9d4edd', '#c77dff', '#e0aaff']  # Purple shades
            gcamp_colors = ['#2e8b57', '#3cb371', '#66cdaa']  # Green shades

            for i, (fiber_col, fiber_data) in enumerate(fibers.items()):
                iso_signal = fiber_data.get('iso')
                gcamp_signal = fiber_data.get('gcamp')

                if iso_signal is None or gcamp_signal is None:
                    continue

                # Create Iso plot
                plot_iso = self.graphics_layout.addPlot(row=row, col=0)
                self._plot_items.append(plot_iso)
                self._plot_to_channel[plot_iso] = f"{fiber_col}-Iso"

                if first_plot is None:
                    first_plot = plot_iso
                else:
                    plot_iso.setXLink(first_plot)

                plot_iso.setMouseEnabled(x=True, y=True)
                plot_iso.vb.setMouseMode(pg.ViewBox.PanMode)
                plot_iso.vb.wheelEvent = lambda ev, p=plot_iso, axis=None: self._handle_wheel_event(ev, p, axis)

                color = iso_colors[i % len(iso_colors)]
                plot_iso.setLabel('left', f'{fiber_col}\nIso', color=color)
                plot_iso.setTitle(f'{fiber_col} Isosbestic', color='#cccccc', size='9pt')
                plot_iso.getAxis('left').setPen('#3e3e42')
                plot_iso.getAxis('left').setTextPen(color)
                plot_iso.getAxis('bottom').setStyle(showValues=False)
                plot_iso.showGrid(x=False, y=False)

                t_plot, s_plot = self._subsample_for_preview(common_time_min, iso_signal)
                plot_iso.plot(t_plot, s_plot, pen=pg.mkPen(color, width=1),
                             clipToView=True, autoDownsample=True, downsampleMethod='subsample')
                row += 1

                # Create GCaMP plot
                plot_gcamp = self.graphics_layout.addPlot(row=row, col=0)
                self._plot_items.append(plot_gcamp)
                self._plot_to_channel[plot_gcamp] = f"{fiber_col}-GCaMP"
                plot_gcamp.setXLink(first_plot)

                plot_gcamp.setMouseEnabled(x=True, y=True)
                plot_gcamp.vb.setMouseMode(pg.ViewBox.PanMode)
                plot_gcamp.vb.wheelEvent = lambda ev, p=plot_gcamp, axis=None: self._handle_wheel_event(ev, p, axis)

                color = gcamp_colors[i % len(gcamp_colors)]
                plot_gcamp.setLabel('left', f'{fiber_col}\nGCaMP', color=color)
                plot_gcamp.setTitle(f'{fiber_col} GCaMP', color='#cccccc', size='9pt')
                plot_gcamp.getAxis('left').setPen('#3e3e42')
                plot_gcamp.getAxis('left').setTextPen(color)
                plot_gcamp.getAxis('bottom').setStyle(showValues=False)
                plot_gcamp.showGrid(x=False, y=False)

                t_plot, s_plot = self._subsample_for_preview(common_time_min, gcamp_signal)
                plot_gcamp.plot(t_plot, s_plot, pen=pg.mkPen(color, width=1),
                               clipToView=True, autoDownsample=True, downsampleMethod='subsample')
                row += 1

            # Plot AI channels
            if ai_channels_data:
                ai_colors = ['#ff9900', '#00cccc', '#ff66ff', '#ffff00', '#ff6666']

                for col_idx, ai_signal in ai_channels_data.items():
                    # Normalize col_idx
                    col_idx_int = int(col_idx) if isinstance(col_idx, str) else col_idx

                    plot = self.graphics_layout.addPlot(row=row, col=0)
                    self._plot_items.append(plot)
                    channel_name = f"AI-{col_idx}"
                    self._plot_to_channel[plot] = channel_name

                    if first_plot:
                        plot.setXLink(first_plot)

                    plot.setMouseEnabled(x=True, y=True)
                    plot.vb.setMouseMode(pg.ViewBox.PanMode)
                    plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)

                    color = ai_colors[col_idx_int % len(ai_colors)]
                    plot.setLabel('left', channel_name, color=color)
                    plot.setTitle(channel_name, color='#cccccc', size='9pt')
                    plot.getAxis('left').setPen('#3e3e42')
                    plot.getAxis('left').setTextPen(color)
                    plot.getAxis('bottom').setStyle(showValues=False)
                    plot.showGrid(x=False, y=False)

                    t_plot, s_plot = self._subsample_for_preview(common_time_min, ai_signal)
                    plot.plot(t_plot, s_plot, pen=pg.mkPen(color, width=1),
                             clipToView=True, autoDownsample=True, downsampleMethod='subsample')
                    row += 1

            # Show X tick labels only on bottom plot
            if self._plot_items:
                self._plot_items[-1].getAxis('bottom').setStyle(showValues=True)
                self._plot_items[-1].setLabel('bottom', 'Time (minutes)', color='#cccccc')

            # Auto-range all plots
            for plot in self._plot_items:
                plot.enableAutoRange()
                plot.autoRange()

        except Exception as e:
            print(f"[Photometry] Error drawing preview from preprocessed: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._hide_progress()

    def _update_experiment_preview_plots(self, fiber_data, ai_channels, ai_time, normalize):
        """Update per-experiment preview tabs based on channel assignments using PyQtGraph."""
        if not self._experiment_layouts:
            return

        # Get experiment assignments
        exp_assignments = self._get_experiment_assignments()
        n_experiments = len(self._experiment_layouts)

        for exp_idx in range(n_experiments):
            if exp_idx not in self._experiment_layouts:
                continue

            graphics_layout = self._experiment_layouts[exp_idx]
            _safe_clear_graphics_layout(graphics_layout)

            # Find channels assigned to this experiment (or "All" = -1)
            exp_fibers = []
            exp_ai = []

            for channel_name, assigned_exp in exp_assignments.items():
                if assigned_exp == exp_idx or assigned_exp == -1:
                    if '-GCaMP' in channel_name or '-Red' in channel_name or '-Iso' in channel_name:
                        fiber_col = channel_name.split('-')[0]
                        if fiber_col not in exp_fibers:
                            exp_fibers.append(fiber_col)
                    elif channel_name.startswith('AI-'):
                        ai_col = channel_name.replace('AI-', '')
                        if ai_col not in exp_ai:
                            exp_ai.append(ai_col)

            n_panels = len(exp_fibers) + len(exp_ai)
            if n_panels == 0:
                plot = graphics_layout.addPlot(row=0, col=0)
                plot.setTitle(f"No channels assigned to Exp {exp_idx + 1}", color='#888888')
                continue

            row = 0
            first_plot = None
            plot_items = []

            # Plot fibers
            for fiber_col in exp_fibers:
                if fiber_col not in fiber_data:
                    continue

                fdata = fiber_data[fiber_col]
                plot = graphics_layout.addPlot(row=row, col=0)
                plot_items.append(plot)

                # Store channel mapping and configure mouse interaction
                channel_name = f"{fiber_col}-GCaMP"
                plot.setMouseEnabled(x=True, y=True)
                plot.vb.setMouseMode(pg.ViewBox.PanMode)
                plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)

                # Add context menu to PyQtGraph's native menu
                self._add_context_menu_to_plot(plot, channel_name)

                # Style the plot
                plot.setLabel('left', f'{fiber_col}', color='#ffffff')
                plot.setTitle(f'Exp {exp_idx + 1}: {fiber_col}', color='#cccccc', size='9pt')
                plot.getAxis('left').setPen('#3e3e42')
                plot.getAxis('left').setTextPen('#cccccc')
                plot.getAxis('bottom').setPen('#3e3e42')
                plot.getAxis('bottom').setTextPen('#cccccc')
                plot.getAxis('bottom').setStyle(showValues=False)  # Hide X ticks (not bottom)
                plot.showGrid(x=False, y=False)

                if first_plot is None:
                    first_plot = plot
                else:
                    plot.setXLink(first_plot)

                # GCaMP
                if len(fdata['gcamp_time']) > 0:
                    t_plot, s_plot = self._subsample_for_preview(fdata['gcamp_time'], fdata['gcamp'])
                    valid = ~np.isnan(s_plot)
                    plot.plot(t_plot[valid], s_plot[valid], pen=pg.mkPen('#00cc00', width=1))

                # Iso
                if len(fdata['iso_time']) > 0:
                    t_plot, s_plot = self._subsample_for_preview(fdata['iso_time'], fdata['iso'])
                    valid = ~np.isnan(s_plot)
                    plot.plot(t_plot[valid], s_plot[valid], pen=pg.mkPen('#5555ff', width=1))

                row += 1

            # Plot AI channels
            if ai_time is not None:
                ai_colors = ['#ff9900', '#00cccc', '#ff66ff', '#ffff00', '#ff6666']
                for ai_col in exp_ai:
                    if ai_col not in self._ai_data.columns:
                        continue

                    # Find column index for consistent colors
                    col_idx = 0
                    for idx, info in self.ai_columns.items():
                        if info['column'] == ai_col:
                            col_idx = idx
                            break

                    plot = graphics_layout.addPlot(row=row, col=0)
                    plot_items.append(plot)

                    # Store channel mapping and configure mouse interaction
                    channel_name = f"AI-{ai_col}"
                    plot.setMouseEnabled(x=True, y=True)
                    plot.vb.setMouseMode(pg.ViewBox.PanMode)
                    plot.vb.wheelEvent = lambda ev, p=plot, axis=None: self._handle_wheel_event(ev, p, axis)

                    # Add context menu to PyQtGraph's native menu
                    self._add_context_menu_to_plot(plot, channel_name)

                    # Style the plot - use column index for consistent colors
                    color = ai_colors[col_idx % len(ai_colors)]
                    plot.setLabel('left', f'AI {ai_col}', color='#ffffff')
                    plot.setTitle(f'Exp {exp_idx + 1}: AI {ai_col}', color='#cccccc', size='9pt')
                    plot.getAxis('left').setPen('#3e3e42')
                    plot.getAxis('left').setTextPen('#cccccc')
                    plot.getAxis('bottom').setPen('#3e3e42')
                    plot.getAxis('bottom').setTextPen('#cccccc')
                    plot.getAxis('bottom').setStyle(showValues=False)  # Hide X ticks (not bottom)
                    plot.showGrid(x=False, y=False)

                    if first_plot:
                        plot.setXLink(first_plot)

                    ai_signal = self._ai_data[ai_col].values
                    min_len = min(len(ai_time), len(ai_signal))
                    t_arr = ai_time[:min_len]
                    s_arr = ai_signal[:min_len]
                    t_plot, s_plot = self._subsample_for_preview(t_arr, s_arr)
                    plot.plot(t_plot, s_plot, pen=pg.mkPen(color, width=1))

                    row += 1

            # Show X tick labels only on bottom plot and add axis label
            if plot_items:
                plot_items[-1].getAxis('bottom').setStyle(showValues=True)
                plot_items[-1].setLabel('bottom', 'Time (minutes)', color='#cccccc')

            # Auto-range all plots to show full data
            for plot in plot_items:
                plot.enableAutoRange()
                plot.autoRange()

    def _on_save_data_file(self, load_after: bool = False):
        """Handle save button click - saves all experiments to a single NPZ file."""
        # Compare current state to original to detect changes (more robust than tracking)
        if self._loaded_npz_path is not None and self._original_state_hash is not None:
            current_hash = self._compute_state_hash()
            if current_hash == self._original_state_hash:
                print("[Photometry] No changes detected (state unchanged) - skipping save")
                if load_after:
                    self.load_into_app_requested.emit(self._loaded_npz_path)
                return

        # If we loaded from an existing NPZ, skip the file dialog and save to same path
        prompt_for_path = self._loaded_npz_path is None
        saved_path = self.save_to_npz(prompt_for_path=prompt_for_path)

        if saved_path:
            # Update the original hash to current state after save
            self._original_state_hash = self._compute_state_hash()
            if load_after:
                # Signal that user wants to load this data into the app
                self.load_into_app_requested.emit(saved_path)

    def get_raw_data(self) -> Optional[Dict]:
        """Get raw data for passing to processing tab."""
        if self._fp_data is None:
            return None

        # Get selected fiber columns
        selected_fibers = [col for col, info in self.fiber_columns.items()
                          if info['checkbox'].isChecked()]

        if not selected_fibers:
            return None

        # Auto-detect time and LED columns
        # Prefer ComputerTimestamp (milliseconds) over SystemTimestamp (seconds)
        time_col = None
        led_col = None
        for col in self._fp_data.columns:
            col_lower = col.lower()
            # Prefer ComputerTimestamp which is in milliseconds
            if 'computertimestamp' in col_lower:
                time_col = col
            elif time_col is None and ('timestamp' in col_lower or col_lower == 'time'):
                time_col = col
            # Match LED state column
            if led_col is None and ('ledstate' in col_lower or col_lower == 'led'):
                led_col = col

        if not time_col or not led_col:
            return None

        # Separate channels
        fiber_data = photometry.separate_channels_multi_fiber(
            self._fp_data, led_col, time_col, selected_fibers
        )

        # Include experiment assignments from channel table
        exp_assignments = self._get_experiment_assignments()

        return {
            'fiber_data': fiber_data,
            'ai_data': self._ai_data,
            'timestamps': self._timestamps,
            'file_paths': self.file_paths.copy(),
            'experiment_assignments': exp_assignments
        }

    def get_data_for_main_app(self, exp_idx: int = 0, npz_path: Path = None) -> Optional[Dict]:
        """Get processed data for a specific experiment in format expected by main app.

        Args:
            exp_idx: Which experiment to load (0-indexed)
            npz_path: Path to NPZ file (for reopening dialog later)

        Returns:
            Dict with keys:
            - 'sweeps': Dict[str, np.ndarray] - channel_name -> (n_samples, 1) array
            - 'channel_names': List[str] - ordered list of channel names
            - 'channel_visibility': Dict[str, bool] - which channels visible by default
            - 't': np.ndarray - time vector in seconds
            - 'sr_hz': float - sample rate
            - 'photometry_raw': Dict - raw data for recalculation
            - 'photometry_params': Dict - current dF/F parameters
            - 'photometry_npz_path': Path - source file
            - 'dff_channel_name': str - name of the dF/F channel
            - 'experiment_index': int - which experiment this is
            - 'n_experiments': int - total number of experiments
            - 'animal_id': str - animal ID for this experiment
        """
        if self._preprocessed is None:
            return None

        # Build sweeps dict and channel names
        sweeps = {}
        channel_names = []
        channel_visibility = {}  # True = visible by default, False = hidden

        # Get common time from preprocessed data
        common_time = self._preprocessed['common_time']  # Already in seconds
        sample_rate = self._preprocessed['sample_rate']

        # Get experiment assignments to find fibers for this experiment
        exp_assignments = self._get_experiment_assignments()

        # Find fibers assigned to this experiment (or "All")
        # Handle both GCaMP (green) and Red (560nm) signal channels
        experiment_fibers = []
        for channel_name, assigned_exp in exp_assignments.items():
            if assigned_exp == exp_idx or assigned_exp == -1:  # -1 means "All"
                # Extract fiber column from channel name (e.g., "G0-GCaMP" -> "G0", "R0-Red" -> "R0")
                if '-GCaMP' in channel_name:
                    fiber_col = channel_name.replace('-GCaMP', '')
                    if fiber_col not in experiment_fibers:
                        experiment_fibers.append(fiber_col)
                elif '-Red' in channel_name:
                    fiber_col = channel_name.replace('-Red', '')
                    if fiber_col not in experiment_fibers:
                        experiment_fibers.append(fiber_col)

        # If no assignments found, fall back to all selected fibers
        if not experiment_fibers:
            experiment_fibers = [col for col, info in self.fiber_columns.items()
                                if info['checkbox'].isChecked()]

        if not experiment_fibers:
            print(f"[Photometry] No fibers found for experiment {exp_idx}")
            return None

        # Get dF/F parameters for this experiment
        params = self._get_dff_params(exp_idx)

        # Get animal ID for this experiment
        animal_id = ""
        controls_key = f'exp_{exp_idx}'
        if hasattr(self, '_dff_controls') and controls_key in self._dff_controls:
            controls = self._dff_controls[controls_key]
            if 'animal_id_edit' in controls:
                animal_id = controls['animal_id_edit'].text()

        # Get number of experiments
        n_experiments = 1
        if hasattr(self, 'spin_num_experiments'):
            n_experiments = self.spin_num_experiments.value()

        # Process each fiber
        dff_channel_name = None
        common_time_min = common_time / 60.0

        for fiber_col in experiment_fibers:
            if fiber_col not in self._preprocessed['fibers']:
                continue

            fiber_data = self._preprocessed['fibers'][fiber_col]
            iso_signal = fiber_data['iso']
            gcamp_signal = fiber_data['gcamp']

            # Compute dF/F using the same approach as _compute_dff
            try:
                method = params.get('method', 'fitted')
                fit_start = params.get('fit_start', 0)
                fit_end = params.get('fit_end', common_time_min[-1] if len(common_time_min) > 0 else 0)

                # Compute dF/F based on method
                if method == 'simple':
                    dff, fit_params = photometry.compute_dff_simple(gcamp_signal, iso_signal)
                    fitted_iso = iso_signal  # No fitting for simple method
                else:
                    dff, fit_params = photometry.compute_dff_fitted(
                        gcamp_signal, iso_signal, common_time_min,
                        fit_start=fit_start,
                        fit_end=fit_end
                    )
                    fitted_iso = fit_params.get('fitted_iso', iso_signal)

                # Apply detrending if requested
                detrend_curve = None
                detrend_method = params.get('detrend_method', 'none')
                if detrend_method != 'none':
                    dff, detrend_curve, _ = photometry.detrend_signal(
                        dff, common_time_min,
                        method=detrend_method,
                        fit_start=fit_start,
                        fit_end=fit_end
                    )

                # Apply lowpass filter if requested
                lowpass_hz = params.get('lowpass_hz')
                if lowpass_hz is not None and lowpass_hz > 0:
                    dff = photometry.lowpass_filter(dff, lowpass_hz, sample_rate)

                # Add dF/F channel (VISIBLE by default)
                dff_name = f'{fiber_col}-dF/F'
                sweeps[dff_name] = dff.reshape(-1, 1)
                channel_names.append(dff_name)
                channel_visibility[dff_name] = True
                if dff_channel_name is None:
                    dff_channel_name = dff_name

                # Add raw signals (HIDDEN by default)
                iso_name = f'{fiber_col}-Iso'
                gcamp_name = f'{fiber_col}-GCaMP'
                sweeps[iso_name] = iso_signal.reshape(-1, 1)
                sweeps[gcamp_name] = gcamp_signal.reshape(-1, 1)
                channel_names.extend([iso_name, gcamp_name])
                channel_visibility[iso_name] = False
                channel_visibility[gcamp_name] = False

                # Add fitted isosbestic (HIDDEN by default)
                if fitted_iso is not None:
                    fitted_name = f'{fiber_col}-Fitted'
                    sweeps[fitted_name] = fitted_iso.reshape(-1, 1)
                    channel_names.append(fitted_name)
                    channel_visibility[fitted_name] = False

                # Add detrend curve if available (HIDDEN by default)
                if detrend_curve is not None:
                    detrend_name = f'{fiber_col}-Detrend'
                    sweeps[detrend_name] = detrend_curve.reshape(-1, 1)
                    channel_names.append(detrend_name)
                    channel_visibility[detrend_name] = False

            except Exception as e:
                print(f"[Photometry] Error computing dF/F for {fiber_col}: {e}")
                import traceback
                traceback.print_exc()

        # Add AI channels (VISIBLE by default)
        for col_idx, ai_signal in self._preprocessed.get('ai_channels', {}).items():
            ai_name = f'AI-{col_idx}'
            sweeps[ai_name] = ai_signal.reshape(-1, 1)
            channel_names.append(ai_name)
            channel_visibility[ai_name] = True

        if not sweeps:
            print(f"[Photometry] No channels created for experiment {exp_idx}")
            return None

        print(f"[Photometry] get_data_for_main_app: Exp {exp_idx + 1} ({animal_id})")
        print(f"[Photometry]   {len(channel_names)} channels: {channel_names}")
        print(f"[Photometry]   Visible: {[c for c, v in channel_visibility.items() if v]}")
        print(f"[Photometry]   dff_channel_name={dff_channel_name}")

        # Build raw data for recalculation
        raw_photometry_data = {
            'fibers': self._preprocessed['fibers'],
            'common_time': common_time,
            'sample_rate': sample_rate,
            'ai_channels': self._preprocessed.get('ai_channels', {}),
        }

        return {
            'sweeps': sweeps,
            'channel_names': channel_names,
            'channel_visibility': channel_visibility,
            't': common_time,
            'sr_hz': sample_rate,
            'photometry_raw': raw_photometry_data,
            'photometry_params': params,
            'photometry_npz_path': npz_path,
            'dff_channel_name': dff_channel_name,
            'experiment_index': exp_idx,
            'n_experiments': n_experiments,
            'animal_id': animal_id,
        }

    def _get_experiment_assignments(self) -> Dict[str, int]:
        """Get channel-to-experiment assignments from the channel table.

        Returns:
            Dict mapping channel names to experiment index.
            -1 means "All" experiments.
        """
        assignments = {}
        if not hasattr(self, 'channel_table'):
            return assignments

        table = self.channel_table
        for row in range(table.rowCount()):
            channel_item = table.item(row, 0)
            if channel_item:
                channel_name = channel_item.text()
                combo = table.cellWidget(row, 2)
                if combo:
                    exp_text = combo.currentText()
                    if exp_text == "None":
                        assignments[channel_name] = -2  # -2 means skip this channel
                    elif exp_text == "All":
                        assignments[channel_name] = -1  # -1 means all experiments
                    else:
                        try:
                            exp_num = int(exp_text.replace("Exp ", "")) - 1  # 0-indexed
                            assignments[channel_name] = exp_num
                        except ValueError:
                            assignments[channel_name] = -1
        return assignments

    def _compute_state_hash(self) -> str:
        """Compute a hash of the current saveable state for change detection.

        This compares settings that affect the output (dF/F params, assignments, etc.)
        but not the raw data (which doesn't change).
        """
        import hashlib

        # Gather the key settings that matter for detecting changes
        hash_parts = []

        # Number of experiments
        n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1
        hash_parts.append(f"n_exp:{n_exp}")

        # dF/F parameters for each experiment
        for exp_idx in range(n_exp):
            params = self._get_dff_params(exp_idx)
            hash_parts.append(f"exp{exp_idx}:{params}")

        # Channel assignments
        assignments = self._get_experiment_assignments()
        hash_parts.append(f"assignments:{sorted(assignments.items())}")

        # Animal IDs
        animal_ids = {}
        for exp_idx in range(n_exp):
            controls_key = f'exp_{exp_idx}'
            if hasattr(self, '_dff_controls') and controls_key in self._dff_controls:
                controls = self._dff_controls[controls_key]
                if 'animal_id_edit' in controls:
                    animal_ids[exp_idx] = controls['animal_id_edit'].text()
        hash_parts.append(f"animal_ids:{animal_ids}")

        # Selected fibers
        selected_fibers = sorted([col for col, info in self.fiber_columns.items()
                                  if info['checkbox'].isChecked()])
        hash_parts.append(f"fibers:{selected_fibers}")

        # Create hash
        state_str = "|".join(hash_parts)
        return hashlib.md5(state_str.encode()).hexdigest()

    def save_to_npz(self, prompt_for_path: bool = True) -> Optional[Path]:
        """Save all dialog state to NPZ file for reopening later.

        The NPZ file contains:
        - File paths (FP, AI, timestamps)
        - Preprocessed data (common_time, fibers, ai_channels)
        - Channel assignments
        - dF/F parameters for each experiment
        - Number of experiments
        - Selected columns

        Args:
            prompt_for_path: If False and we loaded from an existing NPZ, save to
                           that path without prompting. Default True for backwards compat.

        Returns:
            Path to saved file, or None if save failed/cancelled
        """
        if self._preprocessed is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Data",
                "Please load photometry data before saving."
            )
            return None

        # If we loaded from an existing NPZ and prompt_for_path is False, use that path
        if not prompt_for_path and self._loaded_npz_path is not None:
            save_path = self._loaded_npz_path
            print(f"[Photometry] Saving to existing file: {save_path}")
        else:
            # Generate default filename with path context for clarity
            # Format: {grandparent}_{parent}_photometry.npz (e.g., "Project_251212 Awake Hargreaves_photometry.npz")
            fp_path = self.file_paths.get('fp_data')
            if fp_path:
                # Build filename from folder hierarchy for better identification
                if fp_path.parent.name.lower().startswith('fp_data'):
                    # FP_data_0/FP_data_0.csv structure
                    experiment_folder = fp_path.parent.parent.name
                    grandparent = fp_path.parent.parent.parent.name if fp_path.parent.parent.parent else ""
                else:
                    experiment_folder = fp_path.parent.name
                    grandparent = fp_path.parent.parent.name if fp_path.parent.parent else ""

                # Combine names for clarity (truncate if too long)
                if grandparent and grandparent != experiment_folder:
                    # Truncate grandparent to 20 chars max
                    grandparent_short = grandparent[:20] if len(grandparent) > 20 else grandparent
                    filename = f"{grandparent_short}_{experiment_folder}_photometry.npz"
                else:
                    filename = f"{experiment_folder}_photometry.npz"

                save_dir = fp_path.parent  # Same folder as fp_data file
                default_path = save_dir / filename
            elif self._loaded_npz_path is not None:
                # Use the loaded path as default
                default_path = self._loaded_npz_path
            else:
                default_path = Path.home() / "photometry.npz"

            # Ask user for save location
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Photometry Data",
                str(default_path),
                "NumPy Archive (*.npz);;All Files (*)"
            )

            if not save_path:
                return None  # User cancelled

            save_path = Path(save_path)

        try:
            # Gather all state to save
            state = self._gather_state_for_save()

            # Save to NPZ
            np.savez_compressed(save_path, **state)

            print(f"[Photometry] Saved dialog state to: {save_path}")

            # Register in NPZ registry for auto-discovery
            # This allows the app to remember that this source file has a processed NPZ
            fp_path = self.file_paths.get('fp_data')
            if fp_path:
                try:
                    from core import config as app_config
                    n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1

                    # Collect animal IDs
                    animal_ids = []
                    if hasattr(self, '_dff_controls'):
                        for exp_idx in range(n_exp):
                            controls_key = f'exp_{exp_idx}'
                            if controls_key in self._dff_controls:
                                controls = self._dff_controls[controls_key]
                                if 'animal_id_edit' in controls:
                                    animal_ids.append(controls['animal_id_edit'].text())

                    app_config.register_npz_file(
                        source_file_path=fp_path,
                        npz_path=save_path,
                        n_experiments=n_exp,
                        animal_ids=animal_ids
                    )
                except Exception as reg_err:
                    print(f"[Photometry] Warning: Failed to register NPZ: {reg_err}")

            # Emit signal
            self.npz_saved.emit(save_path)

            return save_path

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save photometry data:\n{e}"
            )
            import traceback
            traceback.print_exc()
            return None

    def _gather_state_for_save(self) -> Dict:
        """Gather all dialog state for saving to NPZ.

        Returns dict with all arrays and metadata needed to restore the dialog.
        """
        state = {}

        # File paths (as strings for NPZ compatibility)
        file_paths = {}
        for key, path in self.file_paths.items():
            file_paths[key] = str(path) if path else ''
        state['file_paths'] = np.array([str(file_paths)], dtype=object)

        # Preprocessed data
        if self._preprocessed:
            state['common_time'] = self._preprocessed['common_time']
            state['sample_rate'] = np.array([self._preprocessed['sample_rate']])
            state['duration'] = np.array([self._preprocessed['duration']])
            state['time_offset'] = np.array([self._preprocessed['time_offset']])

            # Fiber data (iso and gcamp for each fiber)
            for fiber_col, data in self._preprocessed['fibers'].items():
                state[f'fiber_{fiber_col}_iso'] = data['iso']
                state[f'fiber_{fiber_col}_gcamp'] = data['gcamp']

            # AI channels
            for col_idx, data in self._preprocessed['ai_channels'].items():
                state[f'ai_channel_{col_idx}'] = data

            # Store fiber column names
            state['fiber_columns'] = np.array(list(self._preprocessed['fibers'].keys()), dtype=object)
            state['ai_column_indices'] = np.array(list(self._preprocessed['ai_channels'].keys()))

        # Channel assignments
        assignments = self._get_experiment_assignments()
        state['experiment_assignments'] = np.array([str(assignments)], dtype=object)

        # Number of experiments
        n_experiments = 1
        if hasattr(self, 'spin_num_experiments'):
            n_experiments = self.spin_num_experiments.value()
        state['n_experiments'] = np.array([n_experiments])

        # dF/F parameters for each experiment
        dff_params = {}
        for exp_idx in range(n_experiments):
            params = self._get_dff_params(exp_idx)
            dff_params[exp_idx] = params
        state['dff_params'] = np.array([str(dff_params)], dtype=object)

        # Animal IDs for each experiment
        animal_ids = {}
        for exp_idx in range(n_experiments):
            controls_key = f'exp_{exp_idx}'
            if hasattr(self, '_dff_controls') and controls_key in self._dff_controls:
                controls = self._dff_controls[controls_key]
                if 'animal_id_edit' in controls:
                    animal_ids[exp_idx] = controls['animal_id_edit'].text()
        state['animal_ids'] = np.array([str(animal_ids)], dtype=object)

        # Selected fiber columns (checkboxes)
        selected_fibers = [col for col, info in self.fiber_columns.items()
                          if info['checkbox'].isChecked()]
        state['selected_fibers'] = np.array(selected_fibers, dtype=object)

        # Version for future compatibility
        state['version'] = np.array([1])

        return state

    def load_from_npz(self, npz_path: Path) -> bool:
        """Load dialog state from an NPZ file using a background thread.

        Args:
            npz_path: Path to the NPZ file

        Returns:
            True (loading starts asynchronously; UI updated on completion)
        """
        from PyQt6.QtWidgets import QProgressDialog
        from core.file_load_worker import FileLoadWorker

        # Clear existing plots first to prevent duplication on reopen
        self._clear_all_plots()

        # Show progress dialog
        self._npz_load_progress = QProgressDialog(
            f"Loading photometry session...\n{npz_path.name}",
            None, 0, 0, self
        )
        self._npz_load_progress.setWindowTitle("Loading Photometry NPZ")
        self._npz_load_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self._npz_load_progress.setMinimumDuration(0)
        self._npz_load_progress.setCancelButton(None)
        self._npz_load_progress.show()

        self._loading_npz_path = npz_path

        # Background function: read NPZ and extract all data into plain dicts/arrays
        def _read_npz_data(progress_callback=None):
            """Read and parse NPZ file (runs in background thread)."""
            parsed = {
                'file_paths': {},
                'preprocessed': None,
                'n_experiments': 1,
                'dff_params': None,
                'animal_ids': None,
                'selected_fibers': None,
                'experiment_assignments': None,
            }

            with np.load(npz_path, allow_pickle=True) as data:
                # File paths
                if 'file_paths' in data:
                    try:
                        parsed['file_paths'] = eval(str(data['file_paths'][0]))
                    except Exception as e:
                        print(f"[Photometry] Error loading file paths: {e}")

                # Preprocessed data
                if 'common_time' in data:
                    common_time = np.array(data['common_time'])
                    sample_rate = float(data['sample_rate'][0]) if 'sample_rate' in data else 100.0
                    duration = float(data['duration'][0]) if 'duration' in data else 0
                    time_offset = float(data['time_offset'][0]) if 'time_offset' in data else 0

                    fiber_columns = list(data['fiber_columns']) if 'fiber_columns' in data else []

                    fibers = {}
                    for fiber_col in fiber_columns:
                        iso_key = f'fiber_{fiber_col}_iso'
                        gcamp_key = f'fiber_{fiber_col}_gcamp'
                        if iso_key in data and gcamp_key in data:
                            fibers[fiber_col] = {
                                'iso': np.array(data[iso_key]),
                                'gcamp': np.array(data[gcamp_key])
                            }

                    ai_channels = {}
                    for key in data.files:
                        if key.startswith('ai_channel_'):
                            col_idx = key.replace('ai_channel_', '')
                            ai_channels[col_idx] = np.array(data[key])

                    parsed['preprocessed'] = {
                        'common_time': common_time,
                        'sample_rate': sample_rate,
                        'duration': duration,
                        'time_offset': time_offset,
                        'fibers': fibers,
                        'ai_channels': ai_channels,
                    }

                # Experiment count
                parsed['n_experiments'] = int(data['n_experiments'][0]) if 'n_experiments' in data else 1

                # dF/F params
                if 'dff_params' in data:
                    try:
                        parsed['dff_params'] = eval(str(data['dff_params'][0]))
                    except Exception:
                        pass

                # Animal IDs
                if 'animal_ids' in data:
                    try:
                        parsed['animal_ids'] = eval(str(data['animal_ids'][0]))
                    except Exception:
                        pass

                # Selected fibers
                if 'selected_fibers' in data:
                    parsed['selected_fibers'] = list(data['selected_fibers'])

                # Experiment assignments
                if 'experiment_assignments' in data:
                    try:
                        parsed['experiment_assignments'] = eval(str(data['experiment_assignments'][0]))
                    except Exception:
                        pass

            return parsed

        self._npz_read_worker = FileLoadWorker(_read_npz_data)
        self._npz_read_worker.finished.connect(self._on_npz_data_loaded)
        self._npz_read_worker.error.connect(self._on_npz_read_error)
        self._npz_read_worker.start()
        return True

    def _on_npz_read_error(self, msg):
        """Handle NPZ read errors."""
        self._npz_load_progress.close()
        error_msg = msg.split('\n\n')[0] if '\n\n' in msg else msg
        print(f"[Photometry] Error loading NPZ: {error_msg}")
        QMessageBox.warning(self, "Load Error", f"Failed to load NPZ:\n{error_msg}")

    def _on_npz_data_loaded(self, parsed):
        """Apply parsed NPZ data to the UI (runs on main thread)."""
        self._npz_load_progress.close()
        npz_path = self._loading_npz_path

        try:
            # Apply file paths
            for key, path_str in parsed['file_paths'].items():
                if path_str:
                    self.file_paths[key] = Path(path_str)

            self._update_file_edits()

            # Apply preprocessed data
            if parsed['preprocessed'] is not None:
                self._preprocessed = parsed['preprocessed']
                self._populate_fiber_columns_from_data(parsed['preprocessed']['fibers'])
                self._populate_ai_columns_from_data(parsed['preprocessed']['ai_channels'])

            # Experiment count
            n_experiments = parsed['n_experiments']
            if hasattr(self, 'spin_num_experiments'):
                self.spin_num_experiments.blockSignals(True)
                self.spin_num_experiments.setValue(n_experiments)
                self.spin_num_experiments.blockSignals(False)

            self._update_experiment_tabs(n_experiments)

            # dF/F params
            if parsed['dff_params'] is not None:
                try:
                    self._apply_dff_params(parsed['dff_params'])
                except Exception as e:
                    print(f"[Photometry] Error loading dF/F params: {e}")

            # Animal IDs
            if parsed['animal_ids'] is not None:
                try:
                    self._apply_animal_ids(parsed['animal_ids'])
                except Exception as e:
                    print(f"[Photometry] Error loading animal IDs: {e}")

            # Selected fibers
            if parsed['selected_fibers'] is not None:
                for fiber_col, info in self.fiber_columns.items():
                    info['checkbox'].setChecked(fiber_col in parsed['selected_fibers'])

            # Populate channel table
            self._populate_channel_table_from_preprocessed()

            # Experiment assignments (AFTER table is populated)
            if parsed['experiment_assignments'] is not None:
                try:
                    self._apply_experiment_assignments(parsed['experiment_assignments'])
                except Exception as e:
                    print(f"[Photometry] Error loading assignments: {e}")

            # Render preview
            self._update_preview_plot()

            # Compute dF/F for each experiment
            for exp_idx in range(n_experiments):
                self._compute_dff(exp_idx)

            # Switch to first experiment tab
            if hasattr(self, 'preview_tab_widget') and n_experiments > 0:
                self.preview_tab_widget.setCurrentIndex(1)

            # Track the loaded path
            self._loaded_npz_path = npz_path

            # Store original state hash for change detection
            self._original_state_hash = self._compute_state_hash()

            print(f"[Photometry] Loaded state from: {npz_path}")

            # Reload raw CSV data in background to populate preview tables
            # (NPZ only stores preprocessed data, not raw CSV content)
            if self.file_paths.get('fp_data') and Path(self.file_paths['fp_data']).exists():
                self._reload_raw_csvs_for_preview()

        except Exception as e:
            print(f"[Photometry] Error applying NPZ data: {e}")
            import traceback
            traceback.print_exc()

    def _reload_raw_csvs_for_preview(self):
        """Reload raw CSV files in background to populate preview tables after NPZ load."""
        from core.file_load_worker import FileLoadWorker
        from core import photometry

        file_paths = dict(self.file_paths)

        def _load_raw_csvs(progress_callback=None):
            """Load raw CSV files for preview (runs in background thread)."""
            results = {'fp_data': None, 'ai_data': None, 'timestamps': None}

            t0 = time.perf_counter()
            results['fp_data'] = photometry.load_photometry_csv(file_paths['fp_data'])
            print(f"[Timing] Reload FP data for preview ({len(results['fp_data'])} rows): {time.perf_counter() - t0:.2f}s")

            if file_paths.get('ai_data') and Path(file_paths['ai_data']).exists():
                try:
                    t0 = time.perf_counter()
                    results['ai_data'] = photometry.load_ai_data_csv(
                        file_paths['ai_data'], subsample=AI_SUBSAMPLE
                    )
                    print(f"[Timing] Reload AI data for preview: {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error reloading AI data: {e}")

            ts_path = file_paths.get('timestamps')
            if ts_path and Path(ts_path).exists():
                try:
                    t0 = time.perf_counter()
                    results['timestamps'] = photometry.load_timestamps_csv(ts_path, subsample=AI_SUBSAMPLE)
                    print(f"[Timing] Reload timestamps for preview: {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error reloading timestamps: {e}")

            return results

        self._raw_csv_worker = FileLoadWorker(_load_raw_csvs)
        self._raw_csv_worker.finished.connect(self._on_raw_csvs_reloaded)
        self._raw_csv_worker.error.connect(lambda msg: print(f"[Photometry] Raw CSV reload failed: {msg.split(chr(10))[0]}"))
        self._raw_csv_worker.start()

    def _on_raw_csvs_reloaded(self, results):
        """Apply reloaded raw CSV data to preview tables (runs on main thread)."""
        if results['fp_data'] is not None:
            self._fp_data = results['fp_data']
            self._populate_fp_column_combos()
            self._update_fp_preview_table()

        if results['ai_data'] is not None:
            self._ai_data = results['ai_data']
            self._populate_ai_column_controls()
            self._update_ai_preview_table()

        if results['timestamps'] is not None:
            self._timestamps = results['timestamps']
            self._update_timestamps_info()

        print("[Photometry] Raw CSV data reloaded for preview tables")

    def load_from_cached_data(self, cached_data: Dict, npz_path: Path = None,
                               initial_params: Optional[Dict] = None,
                               session_state: Optional[Dict] = None) -> bool:
        """Load dialog state from cached in-memory data (for instant reopening).

        This uses cached signal data for speed while loading metadata (file paths,
        experiment assignments, animal IDs) from the NPZ file if available.

        Args:
            cached_data: Dict with 'fibers', 'common_time', 'sample_rate', 'ai_channels'
            npz_path: Optional path to NPZ file for loading metadata (file paths, etc.)
            initial_params: Optional dict of dF/F parameters to restore (fit_start, fit_end, etc.)
            session_state: Optional dict with 'n_experiments', 'experiment_index', 'animal_id'
                          from the main app state

        Returns:
            True if loaded successfully, False otherwise
        """
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        # Show progress dialog for user feedback
        progress = QProgressDialog("Restoring photometry session...", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(5)
        QApplication.processEvents()

        try:
            # Clear existing plots first to prevent duplication on reopen
            progress.setLabelText("Preparing workspace...")
            progress.setValue(10)
            QApplication.processEvents()
            self._clear_all_plots()

            # Extract data from cached format
            fibers = cached_data.get('fibers', {})
            common_time = cached_data.get('common_time', np.array([]))
            sample_rate = cached_data.get('sample_rate', 100.0)
            ai_channels = cached_data.get('ai_channels', {})

            if len(common_time) == 0 or len(fibers) == 0:
                print("[Photometry] Cached data is empty or invalid")
                progress.close()
                return False

            # Calculate duration and time offset
            duration = common_time[-1] - common_time[0] if len(common_time) > 1 else 0
            time_offset = common_time[0] if len(common_time) > 0 else 0

            # Store preprocessed data
            self._preprocessed = {
                'common_time': common_time,
                'sample_rate': sample_rate,
                'duration': duration,
                'time_offset': time_offset,
                'fibers': fibers,
                'ai_channels': ai_channels,
            }

            progress.setLabelText("Setting up fiber columns...")
            progress.setValue(20)
            QApplication.processEvents()

            # Populate fiber columns UI
            self._populate_fiber_columns_from_data(fibers)

            # Populate AI columns UI
            self._populate_ai_columns_from_data(ai_channels)

            # Load metadata from NPZ file if available (file paths, n_experiments, etc.)
            # This gives us full state restoration while using fast cached signal data
            npz_metadata = None
            if npz_path and Path(npz_path).exists():
                progress.setLabelText("Restoring session settings...")
                progress.setValue(30)
                QApplication.processEvents()
                npz_metadata = self._load_npz_metadata(npz_path)

            # Determine n_experiments from session_state, NPZ, or default
            n_experiments = 1
            if session_state and 'n_experiments' in session_state:
                n_experiments = session_state['n_experiments']
            elif npz_metadata and 'n_experiments' in npz_metadata:
                n_experiments = npz_metadata['n_experiments']

            print(f"[Photometry] Using n_experiments={n_experiments} from session/NPZ")

            if hasattr(self, 'spin_num_experiments'):
                self.spin_num_experiments.blockSignals(True)
                self.spin_num_experiments.setValue(n_experiments)
                self.spin_num_experiments.blockSignals(False)

            progress.setLabelText("Setting up experiments...")
            progress.setValue(40)
            QApplication.processEvents()

            # Create experiment tabs
            self._update_experiment_tabs(n_experiments)

            # Populate the channel table with fiber and AI channels
            self._populate_channel_table_from_preprocessed()

            # Apply file paths from NPZ metadata (for display in UI)
            if npz_metadata and 'file_paths' in npz_metadata:
                for key, path_str in npz_metadata['file_paths'].items():
                    if path_str:
                        self.file_paths[key] = Path(path_str)
                self._update_file_edits()

            # Apply experiment assignments from NPZ
            if npz_metadata and 'experiment_assignments' in npz_metadata:
                self._apply_experiment_assignments(npz_metadata['experiment_assignments'])

            # Apply animal IDs from NPZ
            if npz_metadata and 'animal_ids' in npz_metadata:
                self._apply_animal_ids(npz_metadata['animal_ids'])

            # Apply dF/F parameters - prefer initial_params (current state), fallback to NPZ
            progress.setLabelText("Restoring parameters...")
            progress.setValue(50)
            QApplication.processEvents()

            if initial_params:
                self._apply_dff_params(initial_params)
            elif npz_metadata and 'dff_params' in npz_metadata:
                self._apply_dff_params(npz_metadata['dff_params'])

            # Render All Channels preview from preprocessed data
            progress.setLabelText("Rendering signal overview...")
            progress.setValue(52)
            QApplication.processEvents()
            self._update_preview_plot()

            # Generate plot previews for each experiment
            for exp_idx in range(n_experiments):
                progress.setLabelText(f"Generating plots for experiment {exp_idx + 1}...")
                progress.setValue(55 + int(40 * (exp_idx + 1) / n_experiments))
                QApplication.processEvents()
                self._compute_dff(exp_idx)

            progress.setLabelText("Ready")
            progress.setValue(95)
            QApplication.processEvents()

            # Switch to first experiment tab
            if hasattr(self, 'preview_tab_widget') and n_experiments > 0:
                self.preview_tab_widget.setCurrentIndex(1)

            # Track the NPZ path for saving back to same file
            if npz_path:
                self._loaded_npz_path = npz_path

            # Store original state hash for change detection
            self._original_state_hash = self._compute_state_hash()

            progress.setValue(100)
            progress.close()

            print(f"[Photometry] Loaded from cached data with {n_experiments} experiment(s)")
            return True

        except Exception as e:
            print(f"[Photometry] Error loading cached data: {e}")
            import traceback
            traceback.print_exc()
            progress.close()
            return False

    def _load_npz_metadata(self, npz_path: Path) -> Optional[Dict]:
        """Load only metadata from NPZ file (fast, doesn't load signal arrays).

        Returns dict with file_paths, n_experiments, experiment_assignments,
        dff_params, animal_ids - or None if loading fails.
        """
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                metadata = {}

                # Load file paths
                if 'file_paths' in data:
                    try:
                        metadata['file_paths'] = eval(str(data['file_paths'][0]))
                    except:
                        metadata['file_paths'] = {}

                # Load n_experiments
                if 'n_experiments' in data:
                    metadata['n_experiments'] = int(data['n_experiments'][0])

                # Load experiment assignments
                if 'experiment_assignments' in data:
                    try:
                        metadata['experiment_assignments'] = eval(str(data['experiment_assignments'][0]))
                    except:
                        metadata['experiment_assignments'] = {}

                # Load dF/F params
                if 'dff_params' in data:
                    try:
                        metadata['dff_params'] = eval(str(data['dff_params'][0]))
                    except:
                        metadata['dff_params'] = {}

                # Load animal IDs
                if 'animal_ids' in data:
                    try:
                        metadata['animal_ids'] = eval(str(data['animal_ids'][0]))
                    except:
                        metadata['animal_ids'] = {}

                return metadata

        except Exception as e:
            print(f"[Photometry] Error loading NPZ metadata: {e}")
            return None

    def _populate_fiber_columns_from_data(self, fibers: Dict):
        """Populate fiber columns UI from loaded data."""
        # Clear existing
        self.fiber_columns.clear()
        if hasattr(self, 'fiber_select_table'):
            self.fiber_select_table.setRowCount(0)

        for fiber_col in fibers.keys():
            # Create checkbox for selection
            checkbox = QCheckBox()
            checkbox.setChecked(True)

            self.fiber_columns[fiber_col] = {
                'checkbox': checkbox,
            }

            # Add to table if it exists
            if hasattr(self, 'fiber_select_table'):
                row = self.fiber_select_table.rowCount()
                self.fiber_select_table.insertRow(row)
                self.fiber_select_table.setCellWidget(row, 0, checkbox)
                self.fiber_select_table.setItem(row, 1, QTableWidgetItem(fiber_col))

    def _populate_ai_columns_from_data(self, ai_channels: Dict):
        """Populate AI columns UI from loaded data."""
        self.ai_columns.clear()

        for col_idx in ai_channels.keys():
            checkbox = QCheckBox()
            checkbox.setChecked(True)

            self.ai_columns[col_idx] = {
                'checkbox': checkbox,
                'column': f'AI-{col_idx}',
            }

    def _apply_experiment_assignments(self, assignments: Dict):
        """Apply experiment assignments to the channel table.

        Assignment values:
            -2: None (skip this channel)
            -1: All experiments
            0+: Specific experiment index (0-based)
        """
        if not hasattr(self, 'channel_table'):
            return

        table = self.channel_table
        n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1

        print(f"[Photometry] Applying {len(assignments)} experiment assignments (n_exp={n_exp})")

        for row in range(table.rowCount()):
            channel_item = table.item(row, 0)
            if channel_item:
                channel_name = channel_item.text()
                if channel_name in assignments:
                    assigned_exp = assignments[channel_name]
                    combo = table.cellWidget(row, 2)
                    if combo:
                        old_text = combo.currentText()
                        if assigned_exp == -2:
                            combo.setCurrentText("None")
                        elif assigned_exp == -1:
                            combo.setCurrentText("All")
                        elif assigned_exp < n_exp:
                            # Only set if the experiment exists
                            combo.setCurrentText(f"Exp {assigned_exp + 1}")
                        else:
                            # Saved experiment index exceeds current n_experiments
                            # Keep the auto-assignment (don't change)
                            print(f"[Photometry] Warning: {channel_name} was assigned to Exp {assigned_exp + 1} "
                                  f"but only {n_exp} experiments exist, keeping auto-assignment")
                            continue

                        new_text = combo.currentText()
                        if old_text != new_text:
                            print(f"[Photometry] {channel_name}: {old_text} -> {new_text}")

    def _apply_dff_params(self, dff_params: Dict):
        """Apply dF/F parameters to experiment controls.

        Note: Control keys must match those used when creating controls:
        - combo_dff_method (not method_combo)
        - combo_detrend_method (not detrend_combo)
        - spin_fit_start / spin_fit_end (not fit_start_spin / fit_end_spin)
        - chk_lowpass (not lowpass_check)
        - spin_lowpass_hz (not lowpass_spin)
        """
        if not hasattr(self, '_dff_controls'):
            return

        for exp_idx, params in dff_params.items():
            controls_key = f'exp_{exp_idx}'
            if controls_key not in self._dff_controls:
                continue

            controls = self._dff_controls[controls_key]

            # Method - key is 'combo_dff_method'
            # Items are "Fitted (regression)", "Simple (subtraction)" so use MatchStartsWith
            if 'combo_dff_method' in controls and 'method' in params:
                idx = controls['combo_dff_method'].findText(
                    params['method'].title(), Qt.MatchFlag.MatchStartsWith)
                if idx >= 0:
                    controls['combo_dff_method'].setCurrentIndex(idx)

            # Detrend method - key is 'combo_detrend_method'
            if 'combo_detrend_method' in controls and 'detrend_method' in params:
                idx = controls['combo_detrend_method'].findText(params['detrend_method'].title())
                if idx >= 0:
                    controls['combo_detrend_method'].setCurrentIndex(idx)

            # Fit range - keys are 'spin_fit_start' and 'spin_fit_end'
            if 'spin_fit_start' in controls and 'fit_start' in params:
                controls['spin_fit_start'].setValue(params['fit_start'])
            if 'spin_fit_end' in controls and 'fit_end' in params:
                controls['spin_fit_end'].setValue(params['fit_end'])

            # Lowpass - keys are 'chk_lowpass' and 'spin_lowpass_hz'
            if 'chk_lowpass' in controls:
                has_lowpass = params.get('lowpass_hz') is not None
                controls['chk_lowpass'].setChecked(has_lowpass)
                if 'spin_lowpass_hz' in controls and has_lowpass:
                    controls['spin_lowpass_hz'].setValue(params.get('lowpass_hz', 0.1))

            # Exclude start - key is 'spin_exclude_start'
            if 'spin_exclude_start' in controls and 'exclude_start_min' in params:
                controls['spin_exclude_start'].setValue(params['exclude_start_min'])

    def _apply_animal_ids(self, animal_ids: Dict):
        """Apply animal IDs to experiment controls."""
        if not hasattr(self, '_dff_controls'):
            return

        for exp_idx, animal_id in animal_ids.items():
            controls_key = f'exp_{exp_idx}'
            if controls_key in self._dff_controls:
                controls = self._dff_controls[controls_key]
                if 'animal_id_edit' in controls:
                    controls['animal_id_edit'].setText(animal_id)

    def _populate_channel_table_from_preprocessed(self):
        """Populate channel table from preprocessed data (used when loading from NPZ).

        Only includes fibers that have their checkboxes checked (matching the behavior
        of _populate_channel_table when loading from raw data). This ensures experiment
        assignments are preserved correctly on reload.
        """
        if not hasattr(self, 'channel_table') or self._preprocessed is None:
            return

        table = self.channel_table
        table.setRowCount(0)

        n_exp = self.spin_num_experiments.value() if hasattr(self, 'spin_num_experiments') else 1

        # Get list of fiber columns that are checked (matching _populate_channel_table behavior)
        # This ensures consistency between loading from raw data vs NPZ
        all_fiber_cols = list(self._preprocessed.get('fibers', {}).keys())
        fiber_col_list = []
        for fiber_col in all_fiber_cols:
            if fiber_col in self.fiber_columns:
                if self.fiber_columns[fiber_col]['checkbox'].isChecked():
                    fiber_col_list.append(fiber_col)
            else:
                # Fiber not in UI checkboxes yet - include it (will be checked by default)
                fiber_col_list.append(fiber_col)

        print(f"[Photometry] Populating channel table: {len(fiber_col_list)} checked fibers out of {len(all_fiber_cols)}")

        for fiber_idx, fiber_col in enumerate(fiber_col_list):
            # Auto-assign: Fiber N -> Exp N (if enough experiments)
            # Use index within CHECKED fibers only, not all fibers
            auto_exp_idx = fiber_idx + 1 if fiber_idx < n_exp else 0  # 0 = "All"

            # Detect fiber type from column name (G = green/GCaMP, R = red/560nm)
            fiber_type = self._get_fiber_type(fiber_col)
            if fiber_type == 'red':
                signal_label = "Red"
                signal_name = f"{fiber_col}-Red"
                signal_display = f"{fiber_col} (560nm)"
            else:
                signal_label = "GCaMP"
                signal_name = f"{fiber_col}-GCaMP"
                signal_display = f"{fiber_col} (GCaMP)"

            # Add Signal row (GCaMP or Red depending on fiber type)
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(signal_name))
            table.setItem(row, 1, QTableWidgetItem(signal_display))

            combo = QComboBox()
            combo.addItem("None")  # Skip this channel
            combo.addItem("All")
            for i in range(n_exp):
                combo.addItem(f"Exp {i+1}")
            combo.setCurrentIndex(auto_exp_idx + 1)  # +1 for None option
            combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
            table.setCellWidget(row, 2, combo)

            type_combo = QComboBox()
            type_combo.addItems(["GCaMP", "Isosbestic", "Red", "Other"])
            type_combo.setCurrentText(signal_label)
            table.setCellWidget(row, 3, type_combo)

            # Add Isosbestic row
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(f"{fiber_col}-Iso"))
            table.setItem(row, 1, QTableWidgetItem(f"{fiber_col} (Iso)"))

            combo = QComboBox()
            combo.addItem("None")  # Skip this channel
            combo.addItem("All")
            for i in range(n_exp):
                combo.addItem(f"Exp {i+1}")
            combo.setCurrentIndex(auto_exp_idx + 1)  # +1 for None option
            combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
            table.setCellWidget(row, 2, combo)

            type_combo = QComboBox()
            type_combo.addItems(["GCaMP", "Isosbestic", "Red", "Other"])
            type_combo.setCurrentText("Isosbestic")
            table.setCellWidget(row, 3, type_combo)

        # Add rows for AI channels from preprocessed data
        for col_idx in self._preprocessed.get('ai_channels', {}).keys():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(f"AI-{col_idx}"))
            table.setItem(row, 1, QTableWidgetItem(f"AI {col_idx}"))

            combo = QComboBox()
            combo.addItem("None")  # Skip this channel
            combo.addItem("All")
            for i in range(n_exp):
                combo.addItem(f"Exp {i+1}")
            combo.setCurrentIndex(1)  # Default to "All"
            combo.currentIndexChanged.connect(self._on_experiment_assignment_changed)
            table.setCellWidget(row, 2, combo)

            type_combo = QComboBox()
            type_combo.addItems(["Pleth", "Thermal", "Stim", "Other"])
            table.setCellWidget(row, 3, type_combo)

        # Resize table to fit content
        self._resize_channel_table(table.rowCount())
