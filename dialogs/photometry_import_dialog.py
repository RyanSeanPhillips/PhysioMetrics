"""
Photometry Import Dialog

Multi-tab wizard for importing and processing fiber photometry data.
Guides user through file selection, column mapping, time windowing,
processing options, and preview before loading into PhysioMetrics.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QTabWidget,
    QLabel, QPushButton, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QGroupBox, QCheckBox,
    QMessageBox, QHeaderView, QSizePolicy, QComboBox, QSplitter,
    QDoubleSpinBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

# Matplotlib imports for embedded plot
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector

from core import photometry


# Dark theme stylesheet
DARK_STYLESHEET = """
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
        background-color: transparent;
    }
    QLabel[class="header"] {
        font-size: 12px;
        font-weight: bold;
        color: #ffffff;
        padding: 4px 0px;
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
    QLineEdit {
        background-color: #252526;
        color: #cccccc;
        border: 1px solid #3e3e42;
        padding: 4px;
        border-radius: 3px;
    }
    QLineEdit:read-only {
        background-color: #1e1e1e;
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
    QDoubleSpinBox {
        background-color: #252526;
        color: #cccccc;
        border: 1px solid #3e3e42;
        padding: 3px;
        border-radius: 3px;
    }
    QTableWidget {
        background-color: #252526;
        color: #cccccc;
        border: 1px solid #3e3e42;
        gridline-color: #3e3e42;
    }
    QTableWidget::item {
        padding: 2px;
    }
    QHeaderView::section {
        background-color: #2d2d30;
        color: #cccccc;
        padding: 4px;
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
    QSplitter::handle {
        background-color: #3e3e42;
    }
"""


class PhotometryImportDialog(QDialog):
    """
    Two-tab dialog for importing and processing fiber photometry data.

    Tab 1 - Data Assembly:
    - Select FP_data and AI_data files
    - Map columns (time, LED state, signal, AI channels)
    - Preview raw data
    - Save to *_photometry.npz file

    Tab 2 - Processing:
    - Configure ΔF/F method (fitted vs simple)
    - Configure detrending (none, linear, exponential, biexponential)
    - Set fit range
    - Preview processed signals
    - Load into main application
    """

    # Signal emitted when data is ready to load into main app
    data_ready = pyqtSignal(dict)

    def __init__(self, parent=None, initial_path: Optional[Path] = None,
                 photometry_npz_path: Optional[Path] = None):
        """
        Initialize the photometry import dialog.

        Args:
            parent: Parent widget
            initial_path: Path to raw FP_data CSV file (opens Tab 1)
            photometry_npz_path: Path to existing *_photometry.npz file (opens Tab 2)
        """
        super().__init__(parent)
        self.setWindowTitle("Import Photometry Data")
        self.setMinimumSize(1100, 700)
        self.resize(1350, 800)

        # Add standard minimize/maximize/close buttons to title bar
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.WindowMinMaxButtonsHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        # Store span selectors for multiple axes (Tab 1)
        self._span_selectors = []
        self._photometry_axes = []  # Track which axes are photometry (for fit range display)

        # Tab 2 span selectors
        self._span_selectors_tab2 = []
        self._photometry_axes_tab2 = []

        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)

        # Store file paths (for Tab 1 - raw data)
        self.file_paths: Dict[str, Optional[Path]] = {
            'fp_data': None,
            'ai_data': None
        }

        # Store column mappings
        self.fp_columns = {
            'time': None,
            'led_state': None,
            'signal': None
        }
        self.ai_columns: Dict[int, Dict] = {}  # col_idx -> {'enabled': bool, 'label': str}

        # Store loaded data for preview
        self._fp_data = None
        self._ai_data = None
        self._timestamps = None

        # Photometry NPZ data (for Tab 2 - from saved file)
        self._photometry_data = None  # Dict from NPZ file
        self._photometry_npz_path = Path(photometry_npz_path) if photometry_npz_path else None

        # Store initial path for deferred loading
        self._initial_path = initial_path

        # Track which mode we're in
        self._start_on_tab2 = False

        # If photometry NPZ provided, we'll go directly to Tab 2
        if photometry_npz_path and Path(photometry_npz_path).exists():
            self._photometry_npz_path = Path(photometry_npz_path)
            self._start_on_tab2 = True
        elif initial_path:
            # Raw CSV file - set up for Tab 1
            self.file_paths['fp_data'] = initial_path
            # Auto-detect companion files (fast - just checks paths)
            companions = photometry.find_companion_files(initial_path)
            self.file_paths.update(companions)

        self._setup_ui()
        self._enable_dark_title_bar()

        # Switch to Tab 2 if loading existing NPZ
        if self._start_on_tab2:
            self.tab_widget.setCurrentIndex(1)

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

    def showEvent(self, event):
        """Override showEvent to defer data loading until dialog is visible."""
        super().showEvent(event)
        # Load data after dialog is shown (deferred to avoid blocking)
        if self._start_on_tab2 and self._photometry_npz_path:
            # Tab 2: Load existing NPZ file
            QTimer.singleShot(50, lambda: self._load_photometry_npz(self._photometry_npz_path))
        elif self._initial_path and self._fp_data is None:
            # Tab 1: Load raw CSV files
            QTimer.singleShot(50, self._load_and_preview_data)

    def _setup_ui(self):
        """Set up the dialog UI with two tabs."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tab_widget)

        # Tab 1: Data Assembly
        self._create_tab1_data_assembly()

        # Tab 2: Processing
        self._create_tab2_processing()

        # Note: Data loading is deferred to showEvent()

    # =========================================================================
    # Tab 1: Data Assembly
    # =========================================================================

    def _create_tab1_data_assembly(self):
        """Create Tab 1: File Selection, Column Mapping, and Raw Data Preview."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        main_layout = QHBoxLayout()

        # Create splitter for resizable left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ===== LEFT PANEL: File selection, column mapping, and processing options =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(4)

        # Header
        header = QLabel("Step 1: Select Files & Map Columns")
        header.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; padding: 4px 0px;")
        left_layout.addWidget(header)

        # ----- Photometry Data Section -----
        fp_group = QGroupBox("Photometry Data (FP_data)")
        fp_layout = QVBoxLayout(fp_group)
        fp_layout.setSpacing(3)

        # File selection row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("File:"))
        self.fp_data_edit = QLineEdit()
        self.fp_data_edit.setReadOnly(True)
        self.fp_data_edit.setPlaceholderText("Select FP_data CSV file...")
        file_row.addWidget(self.fp_data_edit, 1)
        btn_fp_browse = QPushButton("Browse")
        btn_fp_browse.setFixedWidth(60)
        btn_fp_browse.clicked.connect(lambda: self._browse_file('fp_data'))
        file_row.addWidget(btn_fp_browse)
        fp_layout.addLayout(file_row)

        # Column mapping (more compact)
        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("Time:"))
        self.combo_time = QComboBox()
        self.combo_time.setFixedWidth(70)
        self.combo_time.currentIndexChanged.connect(self._on_column_changed)
        col_row.addWidget(self.combo_time)
        col_row.addWidget(QLabel("LED:"))
        self.combo_led_state = QComboBox()
        self.combo_led_state.setFixedWidth(70)
        self.combo_led_state.currentIndexChanged.connect(self._on_column_changed)
        col_row.addWidget(self.combo_led_state)
        col_row.addWidget(QLabel("Sig:"))
        self.combo_signal = QComboBox()
        self.combo_signal.setFixedWidth(70)
        self.combo_signal.currentIndexChanged.connect(self._on_column_changed)
        col_row.addWidget(self.combo_signal)
        col_row.addStretch()
        fp_layout.addLayout(col_row)

        # Preview table with horizontal scrolling
        self.fp_preview_table = QTableWidget()
        self.fp_preview_table.setMaximumHeight(95)
        self.fp_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.fp_preview_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        fp_layout.addWidget(self.fp_preview_table)

        left_layout.addWidget(fp_group)

        # ----- Analog Inputs Section -----
        ai_group = QGroupBox("Analog Inputs (AI data)")
        ai_layout = QVBoxLayout(ai_group)
        ai_layout.setSpacing(3)

        # File selection row
        ai_file_row = QHBoxLayout()
        ai_file_row.addWidget(QLabel("File:"))
        self.ai_data_edit = QLineEdit()
        self.ai_data_edit.setReadOnly(True)
        self.ai_data_edit.setPlaceholderText("Optional: AI data CSV")
        ai_file_row.addWidget(self.ai_data_edit, 1)
        btn_ai_browse = QPushButton("Browse")
        btn_ai_browse.setFixedWidth(60)
        btn_ai_browse.clicked.connect(lambda: self._browse_file('ai_data'))
        ai_file_row.addWidget(btn_ai_browse)
        ai_layout.addLayout(ai_file_row)

        # Column checkboxes container (will be populated dynamically - horizontal layout)
        self.ai_columns_widget = QWidget()
        self.ai_columns_layout = QHBoxLayout(self.ai_columns_widget)
        self.ai_columns_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_columns_layout.setSpacing(8)
        ai_layout.addWidget(self.ai_columns_widget)

        # Preview table with horizontal scrolling
        self.ai_preview_table = QTableWidget()
        self.ai_preview_table.setMaximumHeight(95)
        self.ai_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ai_preview_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        ai_layout.addWidget(self.ai_preview_table)

        left_layout.addWidget(ai_group)

        # ----- Processing Options Section -----
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setSpacing(4)

        # Row 1: Time normalization
        time_row = QHBoxLayout()
        self.chk_normalize_time = QCheckBox("Normalize time (t=0)")
        self.chk_normalize_time.setChecked(True)
        self.chk_normalize_time.setToolTip("Start time at 0 from first data point")
        self.chk_normalize_time.stateChanged.connect(self._update_preview_plot)
        time_row.addWidget(self.chk_normalize_time)
        time_row.addStretch()
        proc_layout.addLayout(time_row)

        # Row 2: ΔF/F method
        dff_row = QHBoxLayout()
        dff_row.addWidget(QLabel("ΔF/F:"))
        self.combo_dff_method = QComboBox()
        self.combo_dff_method.addItem("Fitted (regression)", "fitted")
        self.combo_dff_method.addItem("Simple subtraction", "simple")
        self.combo_dff_method.setToolTip(
            "Fitted: Linear regression to scale isosbestic to GCaMP\n"
            "Simple: Normalize then subtract"
        )
        self.combo_dff_method.currentIndexChanged.connect(self._update_preview_plot)
        dff_row.addWidget(self.combo_dff_method, 1)
        proc_layout.addLayout(dff_row)

        # Row 3: Detrend method
        detrend_row = QHBoxLayout()
        detrend_row.addWidget(QLabel("Detrend:"))
        self.combo_detrend_method = QComboBox()
        self.combo_detrend_method.addItem("None", "none")
        self.combo_detrend_method.addItem("Linear", "linear")
        self.combo_detrend_method.addItem("Exponential", "exponential")
        self.combo_detrend_method.addItem("Biexponential", "biexponential")
        self.combo_detrend_method.setCurrentIndex(2)  # Default to exponential
        self.combo_detrend_method.setToolTip(
            "None: No detrending\n"
            "Linear: Subtract best-fit line\n"
            "Exponential: a*exp(-t/tau) + b\n"
            "Biexponential: fast + slow decay"
        )
        self.combo_detrend_method.currentIndexChanged.connect(self._update_preview_plot)
        self.combo_detrend_method.currentIndexChanged.connect(self._update_detrend_range_enabled)
        detrend_row.addWidget(self.combo_detrend_method, 1)
        proc_layout.addLayout(detrend_row)

        # Row 4: Fit range
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit range:"))
        self.spin_detrend_start = QDoubleSpinBox()
        self.spin_detrend_start.setRange(0, 1000)
        self.spin_detrend_start.setDecimals(1)
        self.spin_detrend_start.setValue(5.0)  # Default 5 min
        self.spin_detrend_start.setSuffix(" min")
        self.spin_detrend_start.setFixedWidth(70)
        self.spin_detrend_start.valueChanged.connect(self._update_preview_plot)
        fit_row.addWidget(self.spin_detrend_start)
        fit_row.addWidget(QLabel("-"))
        self.spin_detrend_end = QDoubleSpinBox()
        self.spin_detrend_end.setRange(0, 1000)
        self.spin_detrend_end.setDecimals(1)
        self.spin_detrend_end.setValue(10.0)  # Default 10 min
        self.spin_detrend_end.setSuffix(" min")
        self.spin_detrend_end.setFixedWidth(70)
        self.spin_detrend_end.valueChanged.connect(self._update_preview_plot)
        fit_row.addWidget(self.spin_detrend_end)
        proc_layout.addLayout(fit_row)

        # Row 5: Select range checkbox
        select_row = QHBoxLayout()
        self.chk_select_range = QCheckBox("Select fit range on plot")
        self.chk_select_range.setChecked(False)
        self.chk_select_range.setToolTip("Click and drag on any photometry plot to select fit range")
        self.chk_select_range.stateChanged.connect(self._toggle_span_selector)
        select_row.addWidget(self.chk_select_range)
        select_row.addStretch()
        proc_layout.addLayout(select_row)

        # Row 6: Exclude and low-pass
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Exclude:"))
        self.spin_exclude_start = QDoubleSpinBox()
        self.spin_exclude_start.setRange(0, 60)
        self.spin_exclude_start.setDecimals(1)
        self.spin_exclude_start.setValue(0)
        self.spin_exclude_start.setSuffix(" min")
        self.spin_exclude_start.setFixedWidth(70)
        self.spin_exclude_start.setToolTip("Exclude initial transient")
        self.spin_exclude_start.valueChanged.connect(self._update_preview_plot)
        filter_row.addWidget(self.spin_exclude_start)
        filter_row.addStretch()
        proc_layout.addLayout(filter_row)

        lowpass_row = QHBoxLayout()
        self.chk_lowpass = QCheckBox("Low-pass:")
        self.chk_lowpass.setChecked(False)
        self.chk_lowpass.stateChanged.connect(lambda s: self.spin_lowpass.setEnabled(s))
        self.chk_lowpass.stateChanged.connect(self._update_preview_plot)
        lowpass_row.addWidget(self.chk_lowpass)
        self.spin_lowpass = QDoubleSpinBox()
        self.spin_lowpass.setRange(0.1, 50)
        self.spin_lowpass.setDecimals(1)
        self.spin_lowpass.setValue(2.0)
        self.spin_lowpass.setSuffix(" Hz")
        self.spin_lowpass.setFixedWidth(70)
        self.spin_lowpass.setEnabled(False)
        self.spin_lowpass.valueChanged.connect(self._update_preview_plot)
        lowpass_row.addWidget(self.spin_lowpass)
        lowpass_row.addStretch()
        proc_layout.addLayout(lowpass_row)

        # Row 7: Show intermediates
        self.chk_show_intermediates = QCheckBox("Show intermediate steps")
        self.chk_show_intermediates.setChecked(False)
        self.chk_show_intermediates.setToolTip("Show fitted signals and detrend curves")
        self.chk_show_intermediates.stateChanged.connect(self._update_preview_plot)
        proc_layout.addWidget(self.chk_show_intermediates)

        # Row 8: Update button (standout color)
        btn_update = QPushButton("Update Preview")
        btn_update.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
        """)
        btn_update.clicked.connect(self._update_preview_plot)
        proc_layout.addWidget(btn_update)

        left_layout.addWidget(proc_group)

        # Stretch at bottom of left panel
        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # ===== RIGHT PANEL: Preview Plot =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)

        # Plot header with toolbar on right
        header_row = QHBoxLayout()
        plot_header = QLabel("Signal Preview")
        plot_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; padding: 4px 0px;")
        header_row.addWidget(plot_header)

        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #ffcc00; font-style: italic;")
        self.loading_label.setVisible(False)
        header_row.addWidget(self.loading_label)

        header_row.addStretch()

        # Navigation toolbar for zoom/pan (transparent background, right-aligned)
        self.preview_figure = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e1e')
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.nav_toolbar = NavigationToolbar(self.preview_canvas, right_widget)
        self.nav_toolbar.setStyleSheet("""
            QToolBar {
                background: transparent;
                border: none;
                spacing: 2px;
            }
            QToolButton {
                background: transparent;
                color: #888888;
                border: none;
                padding: 4px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #3e3e42;
                color: #ffffff;
            }
            QToolButton:pressed {
                background-color: #094771;
                color: #ffffff;
            }
        """)
        header_row.addWidget(self.nav_toolbar)

        right_layout.addLayout(header_row)

        # Matplotlib canvas
        self.preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.preview_canvas, 1)

        # Hidden spinboxes for time range (used internally but not shown)
        self.spin_time_start = QDoubleSpinBox()
        self.spin_time_start.setRange(0, 100000)
        self.spin_time_start.setDecimals(1)
        self.spin_time_start.setVisible(False)
        self.spin_time_end = QDoubleSpinBox()
        self.spin_time_end.setRange(0, 100000)
        self.spin_time_end.setDecimals(1)
        self.spin_time_end.setVisible(False)

        splitter.addWidget(right_widget)

        # Set splitter sizes (left panel narrower)
        splitter.setSizes([320, 1080])

        main_layout.addWidget(splitter)
        tab_layout.addLayout(main_layout)

        # Bottom buttons for Tab 1
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # Save button
        self.btn_save_data = QPushButton("Save Data File")
        self.btn_save_data.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
        """)
        self.btn_save_data.clicked.connect(self._on_save_data_file)
        btn_layout.addWidget(self.btn_save_data)

        tab_layout.addLayout(btn_layout)

        # Add tab to tab widget
        self.tab_widget.addTab(tab, "1. Data Assembly")

        # Initialize displays
        self._update_file_edits()

    # =========================================================================
    # Tab 2: Processing
    # =========================================================================

    def _create_tab2_processing(self):
        """Create Tab 2: ΔF/F Processing and Preview."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        main_layout = QHBoxLayout()

        # Create splitter for resizable left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # =====================================================================
        # Left panel: Processing options
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Hidden label for tracking source (used internally)
        self.lbl_source_file = QLabel("")
        self.lbl_source_file.setVisible(False)

        # ΔF/F Method
        method_group = QGroupBox("ΔF/F Calculation")
        method_layout = QVBoxLayout(method_group)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.combo_dff_method_tab2 = QComboBox()
        self.combo_dff_method_tab2.addItem("Fitted (regression)", "fitted")
        self.combo_dff_method_tab2.addItem("Simple (subtraction)", "simple")
        self.combo_dff_method_tab2.currentIndexChanged.connect(self._on_processing_changed)
        method_row.addWidget(self.combo_dff_method_tab2)
        method_layout.addLayout(method_row)

        left_layout.addWidget(method_group)

        # Detrending options
        detrend_group = QGroupBox("Detrending")
        detrend_layout = QVBoxLayout(detrend_group)

        detrend_row = QHBoxLayout()
        detrend_row.addWidget(QLabel("Method:"))
        self.combo_detrend_method_tab2 = QComboBox()
        self.combo_detrend_method_tab2.addItem("None", "none")
        self.combo_detrend_method_tab2.addItem("Linear", "linear")
        self.combo_detrend_method_tab2.addItem("Exponential", "exponential")
        self.combo_detrend_method_tab2.addItem("Biexponential", "biexponential")
        self.combo_detrend_method_tab2.setCurrentIndex(2)  # Default to exponential
        self.combo_detrend_method_tab2.currentIndexChanged.connect(self._on_processing_changed)
        detrend_row.addWidget(self.combo_detrend_method_tab2)
        detrend_layout.addLayout(detrend_row)

        # Fit range
        fit_range_layout = QHBoxLayout()
        fit_range_layout.addWidget(QLabel("Fit range:"))
        self.spin_fit_start_tab2 = QDoubleSpinBox()
        self.spin_fit_start_tab2.setRange(0, 1000)
        self.spin_fit_start_tab2.setDecimals(1)
        self.spin_fit_start_tab2.setSuffix(" min")
        self.spin_fit_start_tab2.setValue(5.0)
        self.spin_fit_start_tab2.valueChanged.connect(self._on_processing_changed)
        fit_range_layout.addWidget(self.spin_fit_start_tab2)
        fit_range_layout.addWidget(QLabel("-"))
        self.spin_fit_end_tab2 = QDoubleSpinBox()
        self.spin_fit_end_tab2.setRange(0, 1000)
        self.spin_fit_end_tab2.setDecimals(1)
        self.spin_fit_end_tab2.setSuffix(" min")
        self.spin_fit_end_tab2.setValue(30.0)
        self.spin_fit_end_tab2.valueChanged.connect(self._on_processing_changed)
        fit_range_layout.addWidget(self.spin_fit_end_tab2)
        detrend_layout.addLayout(fit_range_layout)

        # Select range on plot checkbox
        self.chk_select_range_tab2 = QCheckBox("Select fit range on plot")
        self.chk_select_range_tab2.setChecked(False)
        self.chk_select_range_tab2.setToolTip("Click and drag on any photometry plot to select fit range")
        self.chk_select_range_tab2.stateChanged.connect(self._toggle_span_selector_tab2)
        detrend_layout.addWidget(self.chk_select_range_tab2)

        left_layout.addWidget(detrend_group)

        # Filtering options
        filter_group = QGroupBox("Filtering")
        filter_layout = QVBoxLayout(filter_group)

        self.chk_lowpass_tab2 = QCheckBox("Low-pass filter")
        self.chk_lowpass_tab2.stateChanged.connect(self._on_processing_changed)
        filter_layout.addWidget(self.chk_lowpass_tab2)

        lowpass_row = QHBoxLayout()
        lowpass_row.addWidget(QLabel("Cutoff:"))
        self.spin_lowpass_tab2 = QDoubleSpinBox()
        self.spin_lowpass_tab2.setRange(0.1, 50)
        self.spin_lowpass_tab2.setDecimals(1)
        self.spin_lowpass_tab2.setSuffix(" Hz")
        self.spin_lowpass_tab2.setValue(2.0)
        self.spin_lowpass_tab2.valueChanged.connect(self._on_processing_changed)
        lowpass_row.addWidget(self.spin_lowpass_tab2)
        lowpass_row.addStretch()
        filter_layout.addLayout(lowpass_row)

        left_layout.addWidget(filter_group)

        # Preview options
        preview_group = QGroupBox("Preview Options")
        preview_layout = QVBoxLayout(preview_group)

        self.chk_show_intermediates_tab2 = QCheckBox("Show intermediate steps")
        self.chk_show_intermediates_tab2.setChecked(True)
        self.chk_show_intermediates_tab2.stateChanged.connect(self._on_processing_changed)
        preview_layout.addWidget(self.chk_show_intermediates_tab2)

        left_layout.addWidget(preview_group)

        # Update button
        btn_update = QPushButton("Update Preview")
        btn_update.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        btn_update.clicked.connect(self._update_tab2_preview)
        left_layout.addWidget(btn_update)

        left_layout.addStretch()

        splitter.addWidget(left_widget)

        # =====================================================================
        # Right panel: Preview plot
        # =====================================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure for Tab 2
        self.fig_tab2 = Figure(figsize=(10, 8), facecolor='#1e1e1e')
        self.canvas_tab2 = FigureCanvas(self.fig_tab2)
        self.canvas_tab2.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self.canvas_tab2)

        # Navigation toolbar
        self.toolbar_tab2 = NavigationToolbar(self.canvas_tab2, right_widget)
        self.toolbar_tab2.setStyleSheet("""
            QToolBar { background: transparent; border: none; spacing: 2px; }
            QToolButton { background: transparent; color: #888888; border: none; padding: 2px; }
            QToolButton:hover { background-color: #3e3e42; color: #ffffff; }
        """)
        right_layout.addWidget(self.toolbar_tab2)

        splitter.addWidget(right_widget)

        # Set splitter sizes
        splitter.setSizes([280, 1070])

        main_layout.addWidget(splitter)
        tab_layout.addLayout(main_layout)

        # Bottom buttons for Tab 2
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        # Cancel button
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        # Load into App button
        self.btn_load_app = QPushButton("Load into App")
        self.btn_load_app.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
        """)
        self.btn_load_app.clicked.connect(self._on_load_into_app)
        btn_layout.addWidget(self.btn_load_app)

        tab_layout.addLayout(btn_layout)

        # Add tab to tab widget
        self.tab_widget.addTab(tab, "2. Processing")

    def _on_tab_changed(self, index: int):
        """Handle tab change - populate Tab 2 with Tab 1 data when switching."""
        if index == 1:  # Switched to Tab 2 (Processing)
            # If we have raw data from Tab 1 but no NPZ loaded, build the data structure
            if self._fp_data is not None and self._photometry_data is None:
                self._build_photometry_data_from_tab1()
            # Update Tab 2 preview
            if self._photometry_data is not None:
                self._update_tab2_preview()

    def _build_photometry_data_from_tab1(self):
        """Build photometry data dict from Tab 1's raw data for use in Tab 2."""
        if self._fp_data is None:
            return

        # Get column mappings
        time_col = self.combo_time.currentData()
        led_col = self.combo_led_state.currentData()
        signal_col = self.combo_signal.currentData()

        if not all([time_col, led_col, signal_col]):
            print("[Photometry] Cannot build data - columns not mapped")
            return

        # Process FP data (same logic as _save_photometry_data)
        data = self._fp_data.copy()

        # Handle header row if present
        if not np.issubdtype(data[led_col].dtype, np.number):
            data = data.iloc[1:].copy()
            for col in [time_col, led_col, signal_col]:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Separate channels
        iso_mask = data[led_col] == 1
        gcamp_mask = data[led_col] == 2

        iso_time_raw = data.loc[iso_mask, time_col].values
        iso_signal = data.loc[iso_mask, signal_col].values
        gcamp_time_raw = data.loc[gcamp_mask, time_col].values
        gcamp_signal = data.loc[gcamp_mask, signal_col].values

        if len(iso_time_raw) == 0 or len(gcamp_signal) == 0:
            print("[Photometry] No valid channel data found")
            return

        # Find global t_min for normalization
        all_times = [iso_time_raw, gcamp_time_raw]
        if self._timestamps is not None:
            all_times.append(self._timestamps)
        global_t_min = min(np.min(t) for t in all_times if len(t) > 0)

        # Normalize FP time to minutes, starting at 0
        fp_time = (iso_time_raw - global_t_min) / 60000

        # Estimate sample rate
        if len(fp_time) > 1:
            fp_dt = np.median(np.diff(fp_time)) * 60
            fp_sample_rate = 1.0 / fp_dt if fp_dt > 0 else 0
        else:
            fp_sample_rate = 0

        # Build data dict (same format as NPZ file)
        self._photometry_data = {
            'source_fp_path': str(self.file_paths['fp_data']) if self.file_paths['fp_data'] else '',
            'source_ai_path': str(self.file_paths['ai_data']) if self.file_paths['ai_data'] else '',
            'fp_time': fp_time,
            'iso': iso_signal,
            'gcamp': gcamp_signal,
        }

        # Add AI data if available
        if self._ai_data is not None and self._timestamps is not None:
            ai_time = (self._timestamps - global_t_min) / 60000

            if len(ai_time) > 1:
                ai_dt = np.median(np.diff(ai_time)) * 60
                ai_sample_rate = 1.0 / ai_dt if ai_dt > 0 else 0
            else:
                ai_sample_rate = 0

            self._photometry_data['ai_time'] = ai_time

            # Get enabled AI channels
            ai_channels = {}
            for i, col_info in self.ai_columns.items():
                if col_info['checkbox'].isChecked():
                    col_name = col_info['column']
                    label = col_info['label_edit'].text() or f"ai_{col_name}"
                    label = label.replace(' ', '_')
                    ai_channels[label] = self._ai_data[col_name].values

            self._photometry_data['ai_channels'] = ai_channels
        else:
            ai_sample_rate = 0

        # Add metadata
        self._photometry_data['metadata'] = {
            'format_version': 1,
            'fp_sample_rate': fp_sample_rate,
            'ai_sample_rate': ai_sample_rate,
            'source': 'tab1_raw_data',  # Indicate this came from raw data, not saved file
        }

        # Update source label in Tab 2
        fp_name = self.file_paths['fp_data'].name if self.file_paths['fp_data'] else 'Unknown'
        self.lbl_source_file.setText(f"From Tab 1: {fp_name}")
        self.lbl_source_file.setStyleSheet("color: #00cc00; font-size: 10px;")

        # Copy processing settings from Tab 1 to Tab 2
        self._sync_settings_tab1_to_tab2()

        print(f"[Photometry] Built data from Tab 1: {len(fp_time)} FP samples")

    def _sync_settings_tab1_to_tab2(self):
        """Copy processing settings from Tab 1 controls to Tab 2 controls."""
        # Block signals to prevent triggering updates during sync
        self.combo_dff_method_tab2.blockSignals(True)
        self.combo_detrend_method_tab2.blockSignals(True)
        self.spin_fit_start_tab2.blockSignals(True)
        self.spin_fit_end_tab2.blockSignals(True)
        self.chk_lowpass_tab2.blockSignals(True)
        self.spin_lowpass_tab2.blockSignals(True)

        # Sync ΔF/F method
        self.combo_dff_method_tab2.setCurrentIndex(self.combo_dff_method.currentIndex())

        # Sync detrend method
        self.combo_detrend_method_tab2.setCurrentIndex(self.combo_detrend_method.currentIndex())

        # Sync fit range
        self.spin_fit_start_tab2.setValue(self.spin_detrend_start.value())
        self.spin_fit_end_tab2.setValue(self.spin_detrend_end.value())

        # Sync low-pass filter
        self.chk_lowpass_tab2.setChecked(self.chk_lowpass.isChecked())
        self.spin_lowpass_tab2.setValue(self.spin_lowpass.value())

        # Restore signals
        self.combo_dff_method_tab2.blockSignals(False)
        self.combo_detrend_method_tab2.blockSignals(False)
        self.spin_fit_start_tab2.blockSignals(False)
        self.spin_fit_end_tab2.blockSignals(False)
        self.chk_lowpass_tab2.blockSignals(False)
        self.spin_lowpass_tab2.blockSignals(False)

    def _browse_photometry_npz(self):
        """Browse for an existing *_photometry.npz file."""
        start_dir = ""
        if self._photometry_npz_path:
            start_dir = str(self._photometry_npz_path.parent)
        elif self.file_paths.get('fp_data'):
            fp_path = self.file_paths['fp_data']
            if fp_path.parent.name.lower().startswith('fp_data'):
                start_dir = str(fp_path.parent.parent)
            else:
                start_dir = str(fp_path.parent)

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Photometry Data File",
            start_dir,
            "Photometry Files (*_photometry.npz);;NPZ Files (*.npz);;All Files (*)"
        )

        if path:
            self._load_photometry_npz(Path(path))

    def _load_photometry_npz(self, path: Path):
        """Load a *_photometry.npz file for processing."""
        try:
            data = np.load(path, allow_pickle=True)
            self._photometry_data = {key: data[key] for key in data.files}
            self._photometry_npz_path = path

            # Update source label
            self.lbl_source_file.setText(f"Loaded: {path.name}")
            self.lbl_source_file.setStyleSheet("color: #00cc00; font-size: 10px;")

            # Update preview
            self._update_tab2_preview()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load photometry data:\n{str(e)}"
            )

    def _on_processing_changed(self):
        """Handle changes to processing options - auto-update preview."""
        # Only update if we have data loaded
        if self._photometry_data is not None:
            self._update_tab2_preview()

    def _update_tab2_preview(self):
        """Update the preview plot on Tab 2 with processed data.

        Styling matches Tab 1 exactly for consistency.
        """
        if self._photometry_data is None:
            # No data loaded yet
            self.fig_tab2.clear()
            ax = self.fig_tab2.add_subplot(111)
            ax.set_facecolor('#252526')
            ax.text(0.5, 0.5, 'No data loaded.\nSwitch to Tab 1 to load data first.',
                   ha='center', va='center', color='#888888', fontsize=12,
                   transform=ax.transAxes)
            self.canvas_tab2.draw()
            return

        # Get processing parameters from Tab 2 controls
        dff_method = self.combo_dff_method_tab2.currentData()
        detrend_method = self.combo_detrend_method_tab2.currentData()
        fit_start = self.spin_fit_start_tab2.value()
        fit_end = self.spin_fit_end_tab2.value()
        lowpass_enabled = self.chk_lowpass_tab2.isChecked()
        lowpass_hz = self.spin_lowpass_tab2.value() if lowpass_enabled else None
        show_intermediates = self.chk_show_intermediates_tab2.isChecked()

        # Extract RAW data from Tab 1
        fp_time = self._photometry_data.get('fp_time', np.array([]))
        iso = self._photometry_data.get('iso', np.array([]))
        gcamp = self._photometry_data.get('gcamp', np.array([]))

        if len(fp_time) == 0 or len(iso) == 0 or len(gcamp) == 0:
            self.fig_tab2.clear()
            ax = self.fig_tab2.add_subplot(111)
            ax.set_facecolor('#252526')
            ax.text(0.5, 0.5, 'Invalid data format.',
                   ha='center', va='center', color='#ff6666', fontsize=12,
                   transform=ax.transAxes)
            self.canvas_tab2.draw()
            return

        # Compute ΔF/F locally in Tab 2 with Tab 2's settings
        try:
            dff_time, dff_signal, intermediates = self._compute_dff(
                fp_time, iso, fp_time, gcamp,
                method=dff_method,
                detrend_method=detrend_method,
                lowpass_hz=lowpass_hz,
                exclude_start_min=0,
                detrend_fit_start=fit_start,
                detrend_fit_end=fit_end,
                return_intermediates=True
            )
        except Exception as e:
            self.fig_tab2.clear()
            ax = self.fig_tab2.add_subplot(111)
            ax.set_facecolor('#252526')
            ax.text(0.5, 0.5, f'Error computing ΔF/F:\n{str(e)}',
                   ha='center', va='center', color='#ff6666', fontsize=12,
                   transform=ax.transAxes)
            self.canvas_tab2.draw()
            return

        # Plot the results
        self.fig_tab2.clear()

        # Helper to add fit range highlight
        def add_fit_range(ax):
            if fit_end > fit_start:
                ax.axvspan(fit_start, fit_end, alpha=0.15, color='yellow', zorder=0)

        # Determine number of plots
        n_plots = 2  # Raw signals + Final ΔF/F
        if show_intermediates and intermediates:
            if dff_method == 'fitted':
                n_plots += 1  # Fitted comparison
            if detrend_method != 'none':
                n_plots += 1  # Detrend curve

        # Check for AI channels
        ai_time = self._photometry_data.get('ai_time')
        ai_channels = self._photometry_data.get('ai_channels')
        if ai_channels is not None:
            ai_channels = ai_channels.item() if hasattr(ai_channels, 'item') else ai_channels
            n_plots += len(ai_channels)

        axes = self.fig_tab2.subplots(n_plots, 1, sharex=True)
        if n_plots == 1:
            axes = [axes]

        # Style axes (matching Tab 1 exactly)
        for ax in axes:
            ax.set_facecolor('#252526')
            ax.tick_params(colors='#cccccc', labelsize=8)
            ax.xaxis.label.set_color('#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            for spine in ax.spines.values():
                spine.set_color('#3e3e42')

        ax_idx = 0
        int_time = intermediates.get('time', dff_time) if intermediates else dff_time
        fit_params = intermediates.get('fit_params', {}) if intermediates else {}

        # Track photometry axes for span selector
        photometry_axes = []

        # Plot 1: Raw aligned signals (dual y-axis)
        ax = axes[ax_idx]
        iso_aligned = intermediates.get('iso_aligned', iso) if intermediates else iso
        gcamp_aligned = intermediates.get('gcamp_aligned', gcamp) if intermediates else gcamp

        # Subsample for performance
        t_plot, s_plot = self._subsample_for_preview(int_time, gcamp_aligned)
        ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP (470nm)')
        ax.set_ylabel('GCaMP', fontsize=8, color='#00cc00')
        ax.tick_params(axis='y', labelcolor='#00cc00')

        ax_iso = ax.twinx()
        t_plot, s_plot = self._subsample_for_preview(int_time, iso_aligned)
        ax_iso.plot(t_plot, s_plot, color='#5555ff', linewidth=0.5, alpha=0.8, label='Iso (415nm)')
        ax_iso.set_ylabel('Iso', fontsize=8, color='#5555ff')
        ax_iso.tick_params(axis='y', labelcolor='#5555ff')
        ax_iso.set_facecolor('#252526')
        for spine in ax_iso.spines.values():
            spine.set_color('#3e3e42')

        add_fit_range(ax)
        photometry_axes.append(ax)
        ax_idx += 1

        # Intermediate plots
        if show_intermediates and intermediates:
            # Fitted iso vs GCaMP (same scale) - only for fitted method
            if dff_method == 'fitted' and intermediates.get('fitted_iso') is not None:
                ax = axes[ax_idx]
                fitted_iso = intermediates['fitted_iso']

                r_sq = fit_params.get('r_squared', 0)
                slope = fit_params.get('slope', 0)
                quality = "low artifacts" if r_sq < 0.1 else ("moderate" if r_sq < 0.5 else "high artifacts")

                t_plot, s_plot = self._subsample_for_preview(int_time, gcamp_aligned)
                ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP')

                t_plot, s_plot = self._subsample_for_preview(int_time, fitted_iso)
                ax.plot(t_plot, s_plot, color='#ff6666', linewidth=0.8, alpha=0.9,
                       label=f'Fitted Iso (slope={slope:.3f}, R²={r_sq:.3f}, {quality})')

                ax.set_ylabel('Fitted', fontsize=8)
                ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
                add_fit_range(ax)
                photometry_axes.append(ax)
                ax_idx += 1

            # Raw ΔF/F with detrend curve
            if detrend_method != 'none' and intermediates.get('dff_raw') is not None:
                ax = axes[ax_idx]
                dff_raw = intermediates['dff_raw']
                detrend_curve = intermediates.get('detrend_curve')

                t_plot, s_plot = self._subsample_for_preview(int_time, dff_raw)
                ax.plot(t_plot, s_plot, color='#888888', linewidth=0.5, alpha=0.7, label='Raw ΔF/F')

                if detrend_curve is not None and len(detrend_curve) == len(int_time):
                    detrend_name = fit_params.get('detrend_method', 'trend')
                    t_plot, s_plot = self._subsample_for_preview(int_time, detrend_curve)
                    ax.plot(t_plot, s_plot, color='#ff5555', linewidth=1.5, alpha=0.9, label=f'{detrend_name}')

                ax.set_ylabel('Raw ΔF/F (%)', fontsize=8)
                ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
                add_fit_range(ax)
                photometry_axes.append(ax)
                ax_idx += 1

        # Final ΔF/F
        ax = axes[ax_idx]
        t_plot, s_plot = self._subsample_for_preview(dff_time, dff_signal)
        ax.plot(t_plot, s_plot, color='#ff9900', linewidth=0.5, alpha=0.8)
        ax.set_ylabel('ΔF/F (%)', fontsize=8)
        ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
        add_fit_range(ax)
        photometry_axes.append(ax)
        ax.relim()
        ax.autoscale_view(scaley=True)
        ax_idx += 1

        # AI channels
        if ai_channels is not None and ai_time is not None:
            for label, signal in ai_channels.items():
                if ax_idx < len(axes):
                    ax = axes[ax_idx]

                    # Color based on label (matching Tab 1)
                    label_lower = label.lower()
                    if 'therm' in label_lower or 'stim' in label_lower or 'temp' in label_lower:
                        color = '#ff4444'
                    else:
                        color = '#cccccc'

                    t_plot, s_plot = self._subsample_for_preview(ai_time, signal)
                    ax.plot(t_plot, s_plot, color=color, linewidth=0.5, alpha=0.8)
                    ax.set_ylabel(label, fontsize=8)
                    ax_idx += 1

        # Set x-label on bottom plot
        axes[-1].set_xlabel('Time (minutes)', fontsize=9)

        self.fig_tab2.tight_layout()

        # Create span selectors on photometry axes
        self._create_span_selectors_tab2(photometry_axes)

        self.canvas_tab2.draw()

    def _on_load_into_app(self):
        """Load processed photometry data into the main application."""
        if self._photometry_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load a photometry data file first."
            )
            return

        # Get processing parameters
        processing_params = {
            'dff_method': self.combo_dff_method_tab2.currentData(),
            'detrend_method': self.combo_detrend_method_tab2.currentData(),
            'fit_range_start': self.spin_fit_start_tab2.value(),
            'fit_range_end': self.spin_fit_end_tab2.value(),
            'lowpass_enabled': self.chk_lowpass_tab2.isChecked(),
            'lowpass_hz': self.spin_lowpass_tab2.value(),
        }

        # Store for main app to retrieve
        self._processing_params = processing_params
        self._saved_path = self._photometry_npz_path

        # TODO: Emit signal or set result for main app
        QMessageBox.information(
            self,
            "Load into App",
            f"Loading data with settings:\n"
            f"  Method: {processing_params['dff_method']}\n"
            f"  Detrend: {processing_params['detrend_method']}\n"
            f"  Fit range: {processing_params['fit_range_start']}-{processing_params['fit_range_end']} min\n\n"
            f"(Main app integration coming soon)"
        )

        self.accept()

    # =========================================================================
    # Tab 2 Span Selector Methods
    # =========================================================================

    def _toggle_span_selector_tab2(self, state):
        """Enable/disable the span selector for Tab 2."""
        self._update_span_selector_state_tab2()

    def _update_span_selector_state_tab2(self):
        """Update span selector active state for Tab 2."""
        is_active = self.chk_select_range_tab2.isChecked()
        for ss in self._span_selectors_tab2:
            try:
                ss.set_active(is_active)
            except Exception:
                pass
        # Change cursor to indicate selection mode
        if is_active:
            self.canvas_tab2.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.canvas_tab2.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_span_selected_tab2(self, xmin, xmax):
        """Handle span selection on Tab 2 photometry plot."""
        if xmax <= xmin:
            return

        # Update spinboxes (block signals to avoid recursive update)
        self.spin_fit_start_tab2.blockSignals(True)
        self.spin_fit_end_tab2.blockSignals(True)

        self.spin_fit_start_tab2.setValue(xmin)
        self.spin_fit_end_tab2.setValue(xmax)

        self.spin_fit_start_tab2.blockSignals(False)
        self.spin_fit_end_tab2.blockSignals(False)

        print(f"[Photometry Tab2] Selected fit range: {xmin:.1f} - {xmax:.1f} min")

        # Store selection state before update
        was_selecting = self.chk_select_range_tab2.isChecked()

        # Trigger preview update
        self._update_tab2_preview()

        # Restore cursor state after plot update
        if was_selecting:
            self.canvas_tab2.setCursor(Qt.CursorShape.CrossCursor)

    def _create_span_selectors_tab2(self, axes_list):
        """Create span selectors on Tab 2 photometry axes."""
        # Clean up old span selectors
        for ss in self._span_selectors_tab2:
            try:
                ss.set_active(False)
                ss.disconnect_events()
            except Exception:
                pass
        self._span_selectors_tab2 = []
        self._photometry_axes_tab2 = axes_list

        # Create span selector for each axis
        for ax in axes_list:
            ss = SpanSelector(
                ax,
                self._on_span_selected_tab2,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='cyan'),
                interactive=True,
                drag_from_anywhere=True
            )
            self._span_selectors_tab2.append(ss)

        # Set state based on checkbox
        self._update_span_selector_state_tab2()

    # =========================================================================
    # Tab 1 Helper Functions
    # =========================================================================

    def _browse_file(self, file_type: str):
        """Open file browser for selecting a file."""
        # Show wait cursor
        self.setCursor(Qt.CursorShape.WaitCursor)

        try:
            # Determine starting directory
            start_dir = ""
            if self.file_paths['fp_data']:
                start_dir = str(self.file_paths['fp_data'].parent)
                # If FP data is in subfolder, go up to parent
                if self.file_paths['fp_data'].parent.name.lower().startswith('fp_data'):
                    start_dir = str(self.file_paths['fp_data'].parent.parent)

            # File filter
            filter_str = "CSV Files (*.csv);;All Files (*)"

            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select {file_type.replace('_', ' ').title()} File",
                start_dir,
                filter_str
            )

            if path:
                self.file_paths[file_type] = Path(path)
                self._update_file_edits()

                # If FP data selected, auto-detect companion
                if file_type == 'fp_data':
                    companions = photometry.find_companion_files(Path(path))
                    if companions.get('ai_data') and not self.file_paths['ai_data']:
                        self.file_paths['ai_data'] = companions['ai_data']
                        self._update_file_edits()

                # Reload data
                self._load_and_preview_data()
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _update_file_edits(self):
        """Update the file path line edits."""
        if self.file_paths['fp_data']:
            self.fp_data_edit.setText(str(self.file_paths['fp_data']))
        else:
            self.fp_data_edit.clear()

        if self.file_paths['ai_data']:
            self.ai_data_edit.setText(str(self.file_paths['ai_data']))
        else:
            self.ai_data_edit.clear()

    def _show_progress(self, message: str):
        """Show progress message and force UI update."""
        from PyQt6.QtWidgets import QApplication
        self.loading_label.setText(message)
        self.loading_label.setVisible(True)
        QApplication.processEvents()

    def _load_and_preview_data(self):
        """Load data files and update all previews with progress feedback."""
        import time
        from PyQt6.QtWidgets import QApplication

        # Invalidate Tab 2 data since we're loading new files
        if self._photometry_data is not None:
            metadata = self._photometry_data.get('metadata', {})
            if metadata.get('source') == 'tab1_raw_data':
                self._photometry_data = None

        # Show wait cursor
        self.setCursor(Qt.CursorShape.WaitCursor)
        total_start = time.perf_counter()

        try:
            # Stage 1: Update preview tables (fast - only reads first few rows)
            t0 = time.perf_counter()
            self._show_progress("Loading file previews...")
            self._update_fp_preview_table()
            self._update_ai_preview_table()
            print(f"[Timing] Preview tables: {time.perf_counter() - t0:.2f}s")

            # Stage 2: Load FP data (slow - reads entire file)
            if self.file_paths['fp_data'] and self.file_paths['fp_data'].exists():
                t0 = time.perf_counter()
                self._show_progress("Loading photometry data...")
                try:
                    self._fp_data = photometry.load_photometry_csv(self.file_paths['fp_data'])
                    print(f"[Timing] Load FP data ({len(self._fp_data)} rows): {time.perf_counter() - t0:.2f}s")
                    t0 = time.perf_counter()
                    self._populate_fp_column_combos()
                    print(f"[Timing] Populate FP combos: {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error loading FP data: {e}")
                    self._fp_data = None

            # Stage 3: Load AI data (subsampled for preview - every 10th row)
            # This reduces 1.4M rows to ~140K rows, much faster to load
            AI_SUBSAMPLE = 10  # Load every 10th row for preview
            if self.file_paths['ai_data'] and self.file_paths['ai_data'].exists():
                t0 = time.perf_counter()
                self._show_progress("Loading analog inputs (subsampled)...")
                try:
                    self._ai_data = photometry.load_ai_data_csv(self.file_paths['ai_data'], subsample=AI_SUBSAMPLE)
                    print(f"[Timing] Load AI data ({len(self._ai_data)} rows, 1/{AI_SUBSAMPLE} subsampled): {time.perf_counter() - t0:.2f}s")
                    t0 = time.perf_counter()
                    self._populate_ai_column_controls()
                    print(f"[Timing] Populate AI controls: {time.perf_counter() - t0:.2f}s")
                except Exception as e:
                    print(f"[Photometry] Error loading AI data: {e}")
                    self._ai_data = None

            # Stage 4: Load timestamps (subsampled to match AI data)
            if self.file_paths['fp_data']:
                ts_path = photometry.find_timestamps_file(self.file_paths['fp_data'])
                if ts_path and ts_path.exists():
                    t0 = time.perf_counter()
                    self._show_progress("Loading timestamps (subsampled)...")
                    try:
                        self._timestamps = photometry.load_timestamps_csv(ts_path, subsample=AI_SUBSAMPLE)
                        print(f"[Timing] Load timestamps ({len(self._timestamps)} rows, 1/{AI_SUBSAMPLE} subsampled): {time.perf_counter() - t0:.2f}s")
                        print(f"[Photometry] Timestamps range: {self._timestamps.min():.1f} - {self._timestamps.max():.1f} ms")
                    except Exception as e:
                        print(f"[Photometry] Error loading timestamps: {e}")
                        self._timestamps = None
                else:
                    print(f"[Photometry] No timestamps file found")
                    self._timestamps = None

            # Stage 5: Generate preview plot
            t0 = time.perf_counter()
            self._show_progress("Generating preview plot...")
            self._update_preview_plot()
            print(f"[Timing] Update preview plot: {time.perf_counter() - t0:.2f}s")

            print(f"[Timing] === TOTAL LOAD TIME: {time.perf_counter() - total_start:.2f}s ===")

        finally:
            # Reset cursor and hide loading
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.loading_label.setVisible(False)

    def _populate_fp_column_combos(self):
        """Populate FP data column dropdowns based on loaded data."""
        if self._fp_data is None:
            return

        columns = list(self._fp_data.columns)

        # Block signals while populating
        self.combo_time.blockSignals(True)
        self.combo_led_state.blockSignals(True)
        self.combo_signal.blockSignals(True)

        self.combo_time.clear()
        self.combo_led_state.clear()
        self.combo_signal.clear()

        # Check if first row is header
        first_row = self._fp_data.iloc[0]
        has_header = False
        header_names = {}

        # Check for known header names
        for i, val in enumerate(first_row):
            val_str = str(val).lower().strip()
            if val_str in ['ledstate', 'led_state', 'computertimestamp', 'computer_timestamp',
                           'systemtimestamp', 'system_timestamp', 'framecounter', 'g0', 'g1']:
                has_header = True
                header_names[i] = str(first_row.iloc[i])

        # Populate combos with column names (and header if found)
        for i, col in enumerate(columns):
            if has_header and i in header_names:
                display_name = f"{col} ({header_names[i]})"
            else:
                display_name = col

            self.combo_time.addItem(display_name, col)
            self.combo_led_state.addItem(display_name, col)
            self.combo_signal.addItem(display_name, col)

        # Auto-select based on Neurophotometrics format
        # col3 = LedState, col4 = ComputerTimestamp, col5 = G0
        if len(columns) >= 5:
            # Try to find by header name first
            time_idx = None
            led_idx = None
            signal_idx = None

            for i, name in header_names.items():
                name_lower = name.lower()
                if 'computertimestamp' in name_lower or 'timestamp' in name_lower:
                    time_idx = i
                elif 'ledstate' in name_lower:
                    led_idx = i
                elif name_lower in ['g0', 'g1', 'g2']:
                    signal_idx = i

            # Fall back to default positions if not found
            if time_idx is None:
                time_idx = 3  # col4
            if led_idx is None:
                led_idx = 2  # col3
            if signal_idx is None:
                signal_idx = 4  # col5

            self.combo_time.setCurrentIndex(time_idx)
            self.combo_led_state.setCurrentIndex(led_idx)
            self.combo_signal.setCurrentIndex(signal_idx)

        self.combo_time.blockSignals(False)
        self.combo_led_state.blockSignals(False)
        self.combo_signal.blockSignals(False)

    def _populate_ai_column_controls(self):
        """Create checkbox + label controls for AI data columns (horizontal layout)."""
        # Clear existing controls
        while self.ai_columns_layout.count():
            item = self.ai_columns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.ai_columns.clear()

        if self._ai_data is None:
            return

        # Default labels based on notebook
        default_labels = {
            0: "Thermal",
            1: "Pleth"
        }

        # Add each column as checkbox + small label edit (horizontal)
        for i, col in enumerate(self._ai_data.columns):
            checkbox = QCheckBox(f"{col}:")
            checkbox.setChecked(i < 2)  # Enable first 2 by default
            self.ai_columns_layout.addWidget(checkbox)

            label_edit = QLineEdit()
            label_edit.setPlaceholderText("Label")
            label_edit.setText(default_labels.get(i, ""))
            label_edit.setFixedWidth(60)
            self.ai_columns_layout.addWidget(label_edit)

            # Store references
            self.ai_columns[i] = {
                'checkbox': checkbox,
                'label_edit': label_edit,
                'column': col
            }

            # Connect signals
            checkbox.stateChanged.connect(self._on_ai_column_changed)
            label_edit.textChanged.connect(self._on_ai_column_changed)

        # Add stretch at end
        self.ai_columns_layout.addStretch()

    def _update_fp_preview_table(self):
        """Update FP data preview table with RAW data including headers."""
        if not self.file_paths['fp_data']:
            self.fp_preview_table.clear()
            self.fp_preview_table.setRowCount(0)
            self.fp_preview_table.setColumnCount(0)
            return

        # Use raw file preview to show original headers
        col_names, rows = photometry.get_file_preview(self.file_paths['fp_data'], n_rows=5)
        if not rows:
            return

        self.fp_preview_table.setColumnCount(len(col_names))
        self.fp_preview_table.setRowCount(len(rows))
        self.fp_preview_table.setHorizontalHeaderLabels(col_names)

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val)[:15])  # Truncate long values
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.fp_preview_table.setItem(i, j, item)

    def _update_ai_preview_table(self):
        """Update AI data preview table with RAW data including headers."""
        if not self.file_paths['ai_data']:
            self.ai_preview_table.clear()
            self.ai_preview_table.setRowCount(0)
            self.ai_preview_table.setColumnCount(0)
            return

        # Use raw file preview to show original data
        col_names, rows = photometry.get_file_preview(self.file_paths['ai_data'], n_rows=5)
        if not rows:
            return

        self.ai_preview_table.setColumnCount(len(col_names))
        self.ai_preview_table.setRowCount(len(rows))
        self.ai_preview_table.setHorizontalHeaderLabels(col_names)
        # Force header to be visible
        self.ai_preview_table.horizontalHeader().setVisible(True)

        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val)[:15])
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.ai_preview_table.setItem(i, j, item)

    def _update_detrend_range_enabled(self):
        """Enable/disable detrend range spinboxes based on detrend method."""
        method = self.combo_detrend_method.currentData()
        enabled = method != 'none'
        self.spin_detrend_start.setEnabled(enabled)
        self.spin_detrend_end.setEnabled(enabled)
        self.chk_select_range.setEnabled(enabled)

    def _toggle_span_selector(self, state):
        """Enable/disable the span selector for interactive range selection."""
        self._update_span_selector_state()

    def _update_span_selector_state(self):
        """Update span selector active state and cursor based on checkbox."""
        is_active = self.chk_select_range.isChecked()
        for ss in self._span_selectors:
            try:
                ss.set_active(is_active)
            except Exception:
                pass
        # Change cursor to indicate selection mode
        if is_active:
            self.preview_canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.preview_canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_span_selected(self, xmin, xmax):
        """Handle span selection on any photometry plot."""
        # Ensure we have a valid range
        if xmax <= xmin:
            return

        # Update spinboxes (block signals to avoid recursive update)
        self.spin_detrend_start.blockSignals(True)
        self.spin_detrend_end.blockSignals(True)

        self.spin_detrend_start.setValue(xmin)
        self.spin_detrend_end.setValue(xmax)

        self.spin_detrend_start.blockSignals(False)
        self.spin_detrend_end.blockSignals(False)

        print(f"[Photometry] Selected detrend fit range: {xmin:.1f} - {xmax:.1f} min")

        # Store selection state before update
        was_selecting = self.chk_select_range.isChecked()

        # Trigger preview update (this will recreate the span selectors)
        self._update_preview_plot()

        # Restore cursor state after plot update
        if was_selecting:
            self.preview_canvas.setCursor(Qt.CursorShape.CrossCursor)

    def _create_span_selectors(self, axes_list):
        """Create span selectors on all specified axes."""
        # Clean up old span selectors
        for ss in self._span_selectors:
            try:
                ss.set_active(False)
                ss.disconnect_events()
            except Exception:
                pass
        self._span_selectors = []
        self._photometry_axes = axes_list

        # Create span selector for each axis
        for ax in axes_list:
            ss = SpanSelector(
                ax,
                self._on_span_selected,
                'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='cyan'),
                interactive=True,
                drag_from_anywhere=True
            )
            self._span_selectors.append(ss)

        # Set state based on checkbox and restore cursor
        self._update_span_selector_state()

    def _on_column_changed(self):
        """Handle FP column selection change."""
        # Invalidate Tab 2 data so it gets rebuilt on next tab switch
        if self._photometry_data is not None:
            metadata = self._photometry_data.get('metadata', {})
            if metadata.get('source') == 'tab1_raw_data':
                self._photometry_data = None
        self._update_preview_plot()

    def _on_ai_column_changed(self):
        """Handle AI column checkbox/label change."""
        # Invalidate Tab 2 data so it gets rebuilt on next tab switch
        if self._photometry_data is not None:
            metadata = self._photometry_data.get('metadata', {})
            if metadata.get('source') == 'tab1_raw_data':
                self._photometry_data = None
        self._update_preview_plot()

    def _subsample_for_preview(self, time_arr: np.ndarray, signal_arr: np.ndarray,
                                max_points: int = 5000) -> tuple:
        """
        Subsample data using min-max decimation to preserve signal shape.

        For each bucket, keeps both min and max values to preserve peaks/valleys.
        This is much better than simple decimation for preserving waveform shape.
        """
        n = len(time_arr)
        if n <= max_points:
            return time_arr, signal_arr

        # Number of buckets (each bucket produces 2 points: min and max)
        n_buckets = max_points // 2
        bucket_size = n // n_buckets

        # Pre-allocate output arrays
        out_time = []
        out_signal = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < n_buckets - 1 else n

            bucket_signal = signal_arr[start:end]
            bucket_time = time_arr[start:end]

            if len(bucket_signal) == 0:
                continue

            # Find min and max indices
            min_idx = np.argmin(bucket_signal)
            max_idx = np.argmax(bucket_signal)

            # Add in time order (min first if it comes before max)
            if min_idx <= max_idx:
                out_time.extend([bucket_time[min_idx], bucket_time[max_idx]])
                out_signal.extend([bucket_signal[min_idx], bucket_signal[max_idx]])
            else:
                out_time.extend([bucket_time[max_idx], bucket_time[min_idx]])
                out_signal.extend([bucket_signal[max_idx], bucket_signal[min_idx]])

        return np.array(out_time), np.array(out_signal)

    def _compute_dff(self, iso_time: np.ndarray, iso_signal: np.ndarray,
                      gcamp_time: np.ndarray, gcamp_signal: np.ndarray,
                      method: str = 'fitted', detrend_method: str = 'linear',
                      lowpass_hz: float = None, exclude_start_min: float = 0.0,
                      detrend_fit_start: float = 0.0, detrend_fit_end: float = 0.0,
                      return_intermediates: bool = False) -> tuple:
        """
        Compute ΔF/F using specified method.

        Processing pipeline:
        1. Interpolate both channels to common time base
        2. (Optional) Exclude initial transient period
        3. (Optional) Low-pass filter both signals
        4. Compute ΔF/F using selected method
        5. (Optional) Detrend ΔF/F to remove drift

        Args:
            iso_time, iso_signal: Isosbestic channel data
            gcamp_time, gcamp_signal: Calcium channel data
            method: 'fitted' (regression) or 'simple' (direct subtraction)
            detrend_method: 'none', 'linear', 'exponential', or 'biexponential'
            lowpass_hz: If provided, apply low-pass filter at this frequency
            exclude_start_min: Exclude this many minutes from the start
            detrend_fit_start, detrend_fit_end: Time window for fitting detrend curve
            return_intermediates: If True, return dict with intermediate processing data

        Returns:
            If return_intermediates=False: (common_time, dff_signal)
            If return_intermediates=True: (common_time, dff_signal, intermediates_dict)
        """
        from scipy import interpolate, stats, signal
        from scipy.optimize import curve_fit

        # Initialize intermediates dict
        intermediates = {
            'time': None,        # Common time vector
            'iso_aligned': None,
            'gcamp_aligned': None,
            'iso_normalized': None,
            'gcamp_normalized': None,
            'fitted_iso': None,  # For fitted method
            'dff_raw': None,     # ΔF/F before detrending
            'detrend_curve': None,  # The curve being subtracted
            'fit_params': {}     # Fit parameters for display
        }

        if len(iso_time) == 0 or len(gcamp_time) == 0:
            if return_intermediates:
                return np.array([]), np.array([]), intermediates
            return np.array([]), np.array([])

        # Find common time range
        t_min = max(np.min(iso_time), np.min(gcamp_time))
        t_max = min(np.max(iso_time), np.max(gcamp_time))

        # Apply exclusion of initial transient
        if exclude_start_min > 0:
            t_min = t_min + exclude_start_min

        if t_max <= t_min:
            print("[Photometry] Warning: No data remaining after exclusion")
            if return_intermediates:
                return np.array([]), np.array([]), intermediates
            return np.array([]), np.array([])

        # Create common time vector (use the shorter signal's sample rate)
        n_points = min(len(iso_time), len(gcamp_time))
        common_time = np.linspace(t_min, t_max, n_points)

        # Store time in intermediates
        intermediates['time'] = common_time.copy()

        # Calculate sample rate (samples per minute -> Hz)
        if len(common_time) > 1:
            dt_min = (common_time[-1] - common_time[0]) / (len(common_time) - 1)
            fs = 1.0 / (dt_min * 60)  # Convert to Hz
        else:
            fs = 20.0  # Default assumption

        # Interpolate both signals to common time base
        iso_interp = interpolate.interp1d(iso_time, iso_signal, kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
        gcamp_interp = interpolate.interp1d(gcamp_time, gcamp_signal, kind='linear',
                                            bounds_error=False, fill_value='extrapolate')

        iso_aligned = iso_interp(common_time)
        gcamp_aligned = gcamp_interp(common_time)

        # Store aligned signals
        intermediates['iso_aligned'] = iso_aligned.copy()
        intermediates['gcamp_aligned'] = gcamp_aligned.copy()

        # Optional: Low-pass filter both signals
        if lowpass_hz is not None and fs > 2 * lowpass_hz:
            nyq = fs / 2
            normalized_cutoff = lowpass_hz / nyq
            b, a = signal.butter(2, normalized_cutoff, btype='low')
            iso_aligned = signal.filtfilt(b, a, iso_aligned)
            gcamp_aligned = signal.filtfilt(b, a, gcamp_aligned)
            print(f"[Photometry] Applied {lowpass_hz} Hz low-pass filter (fs={fs:.1f} Hz)")

        # Compute ΔF/F based on method
        if method == 'fitted':
            # Linear regression: fit isosbestic to GCaMP
            # Use the same fit window as detrending (if specified) to avoid wake-up artifacts
            t_normalized = common_time - common_time[0]  # Time starting at 0
            if detrend_fit_end > detrend_fit_start:
                fit_mask = (t_normalized >= detrend_fit_start) & (t_normalized <= detrend_fit_end)
                if np.sum(fit_mask) < 10:  # Need at least 10 points
                    fit_mask = np.ones(len(t_normalized), dtype=bool)
                    print(f"[Photometry] Iso fit window too narrow, using all data")
                else:
                    print(f"[Photometry] Fitting iso regression to {detrend_fit_start:.1f}-{detrend_fit_end:.1f} min window")
            else:
                fit_mask = np.ones(len(t_normalized), dtype=bool)

            # Fit regression only on the selected window
            iso_for_fit = iso_aligned[fit_mask]
            gcamp_for_fit = gcamp_aligned[fit_mask]
            slope, intercept, r_value, p_value, std_err = stats.linregress(iso_for_fit, gcamp_for_fit)

            # Apply fit to ALL data
            fitted_iso = slope * iso_aligned + intercept
            r_squared = r_value**2

            # Log fitting results with interpretation
            print(f"[Photometry] ΔF/F fitting: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_squared:.4f}")
            if r_squared < 0.1:
                print(f"[Photometry] Low R² indicates minimal shared artifacts - good signal quality!")
            elif r_squared > 0.5:
                print(f"[Photometry] High R² indicates significant shared artifacts - fitting will help")

            # Store fitted isosbestic
            intermediates['fitted_iso'] = fitted_iso.copy()
            intermediates['fit_params']['method'] = 'fitted'
            intermediates['fit_params']['slope'] = slope
            intermediates['fit_params']['intercept'] = intercept
            intermediates['fit_params']['r_squared'] = r_squared

            # ΔF/F = (GCaMP - fitted_iso) / fitted_iso
            epsilon = np.abs(fitted_iso).mean() * 1e-6
            dff = (gcamp_aligned - fitted_iso) / (fitted_iso + epsilon) * 100

        else:  # simple subtraction
            # Normalize each signal by its mean first
            iso_mean = np.mean(iso_aligned)
            gcamp_mean = np.mean(gcamp_aligned)

            iso_norm = iso_aligned / iso_mean
            gcamp_norm = gcamp_aligned / gcamp_mean

            # Store normalized signals
            intermediates['iso_normalized'] = iso_norm.copy()
            intermediates['gcamp_normalized'] = gcamp_norm.copy()
            intermediates['fit_params']['method'] = 'simple'
            intermediates['fit_params']['iso_mean'] = iso_mean
            intermediates['fit_params']['gcamp_mean'] = gcamp_mean

            # ΔF/F = (GCaMP_norm - Iso_norm) / Iso_norm * 100
            epsilon = np.abs(iso_norm).mean() * 1e-6
            dff = (gcamp_norm - iso_norm) / (iso_norm + epsilon) * 100
            print(f"[Photometry] Simple subtraction: iso_mean={iso_mean:.2f}, gcamp_mean={gcamp_mean:.2f}")

        # Store raw ΔF/F before detrending
        intermediates['dff_raw'] = dff.copy()

        # Apply detrending
        # Determine fit window (0 = use all data)
        t_normalized = common_time - common_time[0]  # Time starting at 0
        if detrend_fit_end > detrend_fit_start:
            fit_mask = (t_normalized >= detrend_fit_start) & (t_normalized <= detrend_fit_end)
            if np.sum(fit_mask) < 10:  # Need at least 10 points
                fit_mask = np.ones(len(t_normalized), dtype=bool)
                print(f"[Photometry] Fit window too narrow, using all data")
            else:
                print(f"[Photometry] Fitting detrend to {detrend_fit_start:.1f}-{detrend_fit_end:.1f} min")
        else:
            fit_mask = np.ones(len(t_normalized), dtype=bool)

        t_for_fit = t_normalized[fit_mask]
        dff_for_fit = dff[fit_mask]

        if detrend_method == 'linear':
            # Use polyfit for linear detrend (matches notebook approach)
            coeffs = np.polyfit(t_for_fit, dff_for_fit, deg=1)
            trend = np.polyval(coeffs, t_normalized)  # Apply to ALL data
            dff_detrended = dff - trend
            print(f"[Photometry] Linear detrend: slope={coeffs[0]:.4f}/min, intercept={coeffs[1]:.2f}%")

            # Store detrend curve
            intermediates['detrend_curve'] = trend.copy()
            intermediates['fit_params']['detrend_method'] = 'linear'
            intermediates['fit_params']['detrend_slope'] = coeffs[0]
            intermediates['fit_params']['detrend_intercept'] = coeffs[1]

            dff = dff_detrended

        elif detrend_method == 'exponential':
            # Fit single exponential decay: y = a * exp(-t/tau) + b
            def exp_decay(t, a, tau, b):
                return a * np.exp(-t / tau) + b

            try:
                # Use fit window data for initial guesses
                n_pts = len(dff_for_fit)
                early_mean = np.mean(dff_for_fit[:max(1, n_pts//10)])
                late_mean = np.mean(dff_for_fit[-max(1, n_pts//10):])

                a0 = early_mean - late_mean
                tau0 = (t_for_fit[-1] - t_for_fit[0]) / 2 if len(t_for_fit) > 1 else 1.0
                b0 = late_mean

                if abs(a0) < 0.1:
                    print(f"[Photometry] No exponential decay detected (Δ={a0:.3f}%), skipping")
                    intermediates['fit_params']['detrend_method'] = 'none (no decay)'
                else:
                    popt, pcov = curve_fit(exp_decay, t_for_fit, dff_for_fit,
                                           p0=[a0, tau0, b0],
                                           maxfev=10000,
                                           bounds=([-100, 0.1, -100], [100, t_normalized[-1]*10, 100]))

                    # Apply fit to ALL data
                    exp_fit = exp_decay(t_normalized, *popt)
                    dff_detrended = dff - exp_fit + popt[2]
                    print(f"[Photometry] Exponential detrend: a={popt[0]:.2f}%, tau={popt[1]:.2f} min, b={popt[2]:.2f}%")

                    # Store detrend curve
                    intermediates['detrend_curve'] = exp_fit.copy()
                    intermediates['fit_params']['detrend_method'] = 'exponential'
                    intermediates['fit_params']['exp_a'] = popt[0]
                    intermediates['fit_params']['exp_tau'] = popt[1]
                    intermediates['fit_params']['exp_b'] = popt[2]

                    dff = dff_detrended

            except Exception as e:
                print(f"[Photometry] Exponential fit failed: {e}, using linear instead")
                coeffs = np.polyfit(t_for_fit, dff_for_fit, deg=1)
                trend = np.polyval(coeffs, t_normalized)
                intermediates['detrend_curve'] = trend.copy()
                intermediates['fit_params']['detrend_method'] = 'linear (exp failed)'
                dff = dff - trend

        elif detrend_method == 'biexponential':
            # Fit biexponential decay: y = a1*exp(-t/tau1) + a2*exp(-t/tau2) + b
            def biexp_decay(t, a1, tau1, a2, tau2, b):
                return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + b

            try:
                n_pts = len(dff_for_fit)
                early_mean = np.mean(dff_for_fit[:max(1, n_pts//10)])
                late_mean = np.mean(dff_for_fit[-max(1, n_pts//10):])
                total_decay = early_mean - late_mean

                t_range = (t_for_fit[-1] - t_for_fit[0]) if len(t_for_fit) > 1 else 1.0
                a1_0 = total_decay * 0.7
                tau1_0 = max(0.1, t_range / 10)  # Ensure positive tau
                a2_0 = total_decay * 0.3
                tau2_0 = max(0.5, t_range / 2)   # Ensure positive tau
                b0 = late_mean

                if abs(total_decay) < 0.1:
                    print(f"[Photometry] No decay detected (Δ={total_decay:.3f}%), skipping biexp")
                    intermediates['fit_params']['detrend_method'] = 'none (no decay)'
                else:
                    popt, pcov = curve_fit(biexp_decay, t_for_fit, dff_for_fit,
                                           p0=[a1_0, tau1_0, a2_0, tau2_0, b0],
                                           maxfev=20000,
                                           bounds=([-100, 0.01, -100, 0.1, -100],
                                                   [100, t_normalized[-1]*5, 100, t_normalized[-1]*10, 100]))

                    # Apply fit to ALL data
                    biexp_fit = biexp_decay(t_normalized, *popt)
                    dff_detrended = dff - biexp_fit + popt[4]
                    print(f"[Photometry] Biexponential detrend: a1={popt[0]:.2f}%, tau1={popt[1]:.2f}min, "
                          f"a2={popt[2]:.2f}%, tau2={popt[3]:.2f}min, b={popt[4]:.2f}%")

                    # Store detrend curve
                    intermediates['detrend_curve'] = biexp_fit.copy()
                    intermediates['fit_params']['detrend_method'] = 'biexponential'
                    intermediates['fit_params']['biexp_a1'] = popt[0]
                    intermediates['fit_params']['biexp_tau1'] = popt[1]
                    intermediates['fit_params']['biexp_a2'] = popt[2]
                    intermediates['fit_params']['biexp_tau2'] = popt[3]
                    intermediates['fit_params']['biexp_b'] = popt[4]

                    dff = dff_detrended

            except Exception as e:
                print(f"[Photometry] Biexponential fit failed: {e}, using linear instead")
                coeffs = np.polyfit(t_for_fit, dff_for_fit, deg=1)
                trend = np.polyval(coeffs, t_normalized)
                intermediates['detrend_curve'] = trend.copy()
                intermediates['fit_params']['detrend_method'] = 'linear (biexp failed)'
                dff = dff - trend

        # Clip extreme values to prevent y-scaling issues (keep within reasonable range)
        dff_clipped = np.clip(dff, -50, 50)  # Clip to ±50%
        if not np.allclose(dff, dff_clipped):
            n_clipped = np.sum(dff != dff_clipped)
            print(f"[Photometry] Warning: Clipped {n_clipped} extreme values to ±50%")
            dff = dff_clipped

        if return_intermediates:
            return common_time, dff, intermediates
        return common_time, dff

    def _update_preview_plot(self):
        """Update the matplotlib preview plot with current data and settings."""
        import time
        from PyQt6.QtWidgets import QApplication

        plot_start = time.perf_counter()

        # Show loading indicator
        self.loading_label.setText("Generating preview...")
        self.loading_label.setVisible(True)
        QApplication.processEvents()

        self.preview_figure.clear()

        if self._fp_data is None:
            self.loading_label.setVisible(False)
            self.preview_canvas.draw()
            return

        # Get selected columns
        time_col = self.combo_time.currentData()
        led_col = self.combo_led_state.currentData()
        signal_col = self.combo_signal.currentData()

        if not all([time_col, led_col, signal_col]):
            self.loading_label.setVisible(False)
            self.preview_canvas.draw()
            return

        try:
            # Separate isosbestic (LED=1) and GCaMP (LED=2)
            t0 = time.perf_counter()
            # Skip first row if it's header
            data = self._fp_data.copy()
            if not np.issubdtype(data[led_col].dtype, np.number):
                data = data.iloc[1:].copy()
                for col in [time_col, led_col, signal_col]:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            iso_mask = data[led_col] == 1
            gcamp_mask = data[led_col] == 2

            # Get raw time values in ms
            iso_time_raw = data.loc[iso_mask, time_col].values
            iso_signal = data.loc[iso_mask, signal_col].values
            gcamp_time_raw = data.loc[gcamp_mask, time_col].values
            gcamp_signal = data.loc[gcamp_mask, signal_col].values
            print(f"[Timing]   Data separation: {time.perf_counter() - t0:.3f}s")

            # Normalize time if checkbox is checked
            t0 = time.perf_counter()
            if self.chk_normalize_time.isChecked():
                # Find the minimum time across both channels
                all_times_raw = np.concatenate([iso_time_raw, gcamp_time_raw]) if len(iso_time_raw) > 0 and len(gcamp_time_raw) > 0 else iso_time_raw
                t_min = np.min(all_times_raw) if len(all_times_raw) > 0 else 0
                iso_time = (iso_time_raw - t_min) / 60000  # Convert to minutes, starting at 0
                gcamp_time = (gcamp_time_raw - t_min) / 60000
            else:
                # Use raw computer timestamps converted to minutes
                iso_time = iso_time_raw / 60000
                gcamp_time = gcamp_time_raw / 60000
            print(f"[Timing]   Time normalization: {time.perf_counter() - t0:.3f}s")

            # Compute ΔF/F
            t0 = time.perf_counter()
            self.loading_label.setText("Computing ΔF/F...")
            QApplication.processEvents()

            # Get processing options from UI controls
            dff_method = self.combo_dff_method.currentData()
            detrend_method = self.combo_detrend_method.currentData()
            exclude_start = self.spin_exclude_start.value()
            lowpass_hz = self.spin_lowpass.value() if self.chk_lowpass.isChecked() else None
            detrend_fit_start = self.spin_detrend_start.value()
            detrend_fit_end = self.spin_detrend_end.value()
            show_intermediates = self.chk_show_intermediates.isChecked()

            # Always get intermediates for the raw aligned panel
            result = self._compute_dff(
                iso_time, iso_signal, gcamp_time, gcamp_signal,
                method=dff_method,
                detrend_method=detrend_method,
                lowpass_hz=lowpass_hz,
                exclude_start_min=exclude_start,
                detrend_fit_start=detrend_fit_start,
                detrend_fit_end=detrend_fit_end,
                return_intermediates=True
            )

            dff_time, dff_signal, intermediates = result
            print(f"[Timing]   ΔF/F computation: {time.perf_counter() - t0:.3f}s")

            # Count plots needed
            # Order: Iso, GCaMP, [intermediates if enabled], final ΔF/F, AI channels
            ai_plots = []
            if self._ai_data is not None:
                for i, col_info in self.ai_columns.items():
                    if col_info['checkbox'].isChecked():
                        label = col_info['label_edit'].text() or f"AI {col_info['column']}"
                        ai_plots.append((col_info['column'], label))

            # Calculate number of panels
            # Always show: Raw aligned signals (dual y-axis) + final ΔF/F + AI
            # Optional intermediates: Fitted iso vs GCaMP, Raw ΔF/F with detrend curve
            n_intermediate_plots = 0
            if show_intermediates and intermediates is not None:
                if dff_method == 'fitted':
                    n_intermediate_plots += 1  # Fitted iso vs GCaMP (same scale)
                if detrend_method != 'none':
                    n_intermediate_plots += 1  # Raw ΔF/F with detrend curve

            # Total: Raw aligned + intermediates + final ΔF/F + AI
            n_plots = 1 + n_intermediate_plots + 1 + len(ai_plots)

            # Update time range spinboxes
            all_times = np.concatenate([iso_time, gcamp_time]) if len(iso_time) > 0 and len(gcamp_time) > 0 else iso_time
            if len(all_times) > 0:
                self.spin_time_start.setRange(0, float(np.max(all_times)))
                self.spin_time_end.setRange(0, float(np.max(all_times)))
                if self.spin_time_end.value() == 0:
                    self.spin_time_end.setValue(float(np.max(all_times)))

            # Get time range and fit range for plotting
            t_start = self.spin_time_start.value()
            t_end = self.spin_time_end.value()
            fit_start = self.spin_detrend_start.value()
            fit_end = self.spin_detrend_end.value()

            t0 = time.perf_counter()
            self.loading_label.setText("Drawing plots...")
            QApplication.processEvents()

            # Create subplots
            axes = self.preview_figure.subplots(n_plots, 1, sharex=True)
            if n_plots == 1:
                axes = [axes]

            # Apply dark theme to axes
            for ax in axes:
                ax.set_facecolor('#252526')
                ax.tick_params(colors='#cccccc', labelsize=8)
                ax.xaxis.label.set_color('#cccccc')
                ax.yaxis.label.set_color('#cccccc')
                for spine in ax.spines.values():
                    spine.set_color('#3e3e42')
            print(f"[Timing]   Create subplots + styling: {time.perf_counter() - t0:.3f}s")

            # Track photometry axes for span selector
            photometry_axes = []
            ax_idx = 0
            t0 = time.perf_counter()

            # Helper to add fit range highlight to photometry axes
            def add_fit_range(ax):
                if fit_end > fit_start:
                    ax.axvspan(fit_start, fit_end, alpha=0.15, color='yellow', zorder=0)

            # Get intermediates data for the first panel
            if intermediates is not None:
                int_time = dff_time
                int_mask = (int_time >= t_start) & (int_time <= t_end) if len(dff_time) > 0 else None
                fit_params = intermediates.get('fit_params', {})
                method_name = fit_params.get('method', 'unknown')
            else:
                int_time = dff_time
                int_mask = (int_time >= t_start) & (int_time <= t_end) if len(dff_time) > 0 else None
                fit_params = {}
                method_name = dff_method

            # Plot 1: Raw aligned signals (dual y-axes) - ALWAYS shown
            raw_ax = axes[ax_idx]
            if intermediates is not None and len(dff_time) > 0:
                iso_aligned = intermediates.get('iso_aligned')
                gcamp_aligned = intermediates.get('gcamp_aligned')

                if gcamp_aligned is not None and len(gcamp_aligned) == len(int_time):
                    t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], gcamp_aligned[int_mask])
                    raw_ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP (470nm)')
                    raw_ax.set_ylabel('GCaMP', fontsize=8, color='#00cc00')
                    raw_ax.tick_params(axis='y', labelcolor='#00cc00')

                if iso_aligned is not None and len(iso_aligned) == len(int_time):
                    ax_iso = raw_ax.twinx()
                    t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], iso_aligned[int_mask])
                    ax_iso.plot(t_plot, s_plot, color='#5555ff', linewidth=0.5, alpha=0.8, label='Iso (415nm)')
                    ax_iso.set_ylabel('Iso', fontsize=8, color='#5555ff')
                    ax_iso.tick_params(axis='y', labelcolor='#5555ff')
                    ax_iso.set_facecolor('#252526')
                    for spine in ax_iso.spines.values():
                        spine.set_color('#3e3e42')
            else:
                # Fallback to raw signals if no intermediates
                if len(gcamp_time) > 0:
                    mask = (gcamp_time >= t_start) & (gcamp_time <= t_end)
                    t_plot, s_plot = self._subsample_for_preview(gcamp_time[mask], gcamp_signal[mask])
                    raw_ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP (470nm)')
                    raw_ax.set_ylabel('GCaMP', fontsize=8, color='#00cc00')
                    raw_ax.tick_params(axis='y', labelcolor='#00cc00')

                if len(iso_time) > 0:
                    ax_iso = raw_ax.twinx()
                    mask = (iso_time >= t_start) & (iso_time <= t_end)
                    t_plot, s_plot = self._subsample_for_preview(iso_time[mask], iso_signal[mask])
                    ax_iso.plot(t_plot, s_plot, color='#5555ff', linewidth=0.5, alpha=0.8, label='Iso (415nm)')
                    ax_iso.set_ylabel('Iso', fontsize=8, color='#5555ff')
                    ax_iso.tick_params(axis='y', labelcolor='#5555ff')
                    ax_iso.set_facecolor('#252526')
                    for spine in ax_iso.spines.values():
                        spine.set_color('#3e3e42')

            add_fit_range(raw_ax)
            photometry_axes.append(raw_ax)
            ax_idx += 1
            print(f"[Timing]   Plot raw aligned signals: {time.perf_counter() - t0:.3f}s")

            # Intermediate plots (before final ΔF/F)
            if show_intermediates and intermediates is not None and len(dff_time) > 0:
                t0_int = time.perf_counter()

                # Intermediate: Fitted iso vs GCaMP (same scale) - only for fitted method
                if method_name == 'fitted':
                    int_ax2 = axes[ax_idx]
                    gcamp_aligned = intermediates.get('gcamp_aligned')
                    fitted_iso = intermediates.get('fitted_iso')

                    r_sq = fit_params.get('r_squared', 0)
                    slope = fit_params.get('slope', 0)
                    # Add interpretation to legend label
                    quality = "low artifacts" if r_sq < 0.1 else ("moderate" if r_sq < 0.5 else "high artifacts")

                    if gcamp_aligned is not None and len(gcamp_aligned) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], gcamp_aligned[int_mask])
                        int_ax2.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP')

                    if fitted_iso is not None and len(fitted_iso) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], fitted_iso[int_mask])
                        int_ax2.plot(t_plot, s_plot, color='#ff6666', linewidth=0.8, alpha=0.9,
                                    label=f'Fitted Iso (slope={slope:.3f}, R²={r_sq:.3f}, {quality})')

                    int_ax2.set_ylabel('Fitted', fontsize=8)
                    int_ax2.legend(loc='upper right', fontsize=7, framealpha=0.7)
                    add_fit_range(int_ax2)
                    photometry_axes.append(int_ax2)
                    ax_idx += 1

                # Intermediate: Raw ΔF/F with detrend curve
                if detrend_method != 'none' and intermediates.get('dff_raw') is not None:
                    int_ax3 = axes[ax_idx]
                    dff_raw = intermediates.get('dff_raw')
                    detrend_curve = intermediates.get('detrend_curve')

                    if dff_raw is not None and len(dff_raw) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], dff_raw[int_mask])
                        int_ax3.plot(t_plot, s_plot, color='#888888', linewidth=0.5, alpha=0.7, label='Raw ΔF/F')

                    if detrend_curve is not None and len(detrend_curve) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], detrend_curve[int_mask])
                        detrend_name = fit_params.get('detrend_method', 'trend')
                        int_ax3.plot(t_plot, s_plot, color='#ff5555', linewidth=1.5, alpha=0.9, label=f'{detrend_name}')

                    int_ax3.set_ylabel('Raw ΔF/F (%)', fontsize=8)
                    int_ax3.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                    int_ax3.legend(loc='upper right', fontsize=7, framealpha=0.7)
                    add_fit_range(int_ax3)
                    photometry_axes.append(int_ax3)
                    ax_idx += 1

                print(f"[Timing]   Plot intermediate panels: {time.perf_counter() - t0_int:.3f}s")

            # Final ΔF/F plot (after intermediates)
            t0 = time.perf_counter()
            dff_ax = axes[ax_idx]
            if len(dff_time) > 0:
                mask = (dff_time >= t_start) & (dff_time <= t_end)
                t_plot, s_plot = self._subsample_for_preview(dff_time[mask], dff_signal[mask])
                dff_ax.plot(t_plot, s_plot, color='#ff9900', linewidth=0.5, alpha=0.8)
                dff_ax.set_ylabel('ΔF/F (%)', fontsize=8)
                dff_ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                add_fit_range(dff_ax)
                photometry_axes.append(dff_ax)

                # Force y-axis autoscale to fix flat line issue
                dff_ax.relim()
                dff_ax.autoscale_view(scaley=True)
            ax_idx += 1
            print(f"[Timing]   Plot final ΔF/F: {time.perf_counter() - t0:.3f}s")

            # Create span selectors on ALL photometry axes
            self._create_span_selectors(photometry_axes)

            # Plot AI channels
            t0 = time.perf_counter()
            if self._ai_data is not None and self._timestamps is not None:
                if self.chk_normalize_time.isChecked():
                    ai_t_min = np.min(self._timestamps) if len(self._timestamps) > 0 else 0
                    ai_time = (self._timestamps - ai_t_min) / 60000
                else:
                    ai_time = self._timestamps / 60000

                ai_t_start = 0 if self.chk_normalize_time.isChecked() else np.min(ai_time)
                ai_t_end = np.max(ai_time) if len(ai_time) > 0 else t_end

                for col, label in ai_plots:
                    ai_signal = self._ai_data[col].values
                    min_len = min(len(ai_time), len(ai_signal))
                    t_arr = ai_time[:min_len]
                    s_arr = ai_signal[:min_len]
                    mask = (t_arr >= ai_t_start) & (t_arr <= ai_t_end)

                    label_lower = label.lower()
                    if 'therm' in label_lower or 'stim' in label_lower or 'temp' in label_lower:
                        color = '#ff4444'
                    else:
                        color = '#cccccc'

                    axes[ax_idx].plot(t_arr[mask], s_arr[mask], color=color, linewidth=0.5, alpha=0.8)
                    axes[ax_idx].set_ylabel(label, fontsize=8)
                    ax_idx += 1
                print(f"[Timing]   Plot AI signals ({len(ai_plots)} panels): {time.perf_counter() - t0:.3f}s")

            # Set x-axis label on bottom plot
            axes[-1].set_xlabel('Time (minutes)', fontsize=9)

            t0 = time.perf_counter()
            self.preview_figure.tight_layout()
            print(f"[Timing]   tight_layout(): {time.perf_counter() - t0:.3f}s")

        except Exception as e:
            print(f"[Photometry] Error updating preview: {e}")
            import traceback
            traceback.print_exc()

        # Hide loading indicator and draw
        t0 = time.perf_counter()
        self.loading_label.setVisible(False)
        self.preview_canvas.draw()
        print(f"[Timing]   canvas.draw(): {time.perf_counter() - t0:.3f}s")
        print(f"[Timing]   === TOTAL PLOT TIME: {time.perf_counter() - plot_start:.2f}s ===")

    # =========================================================================
    # Save Action
    # =========================================================================

    def _on_save_data_file(self, load_after: bool = False):
        """Handle save button click - save photometry data to NPZ file.

        Args:
            load_after: If True, emit signal to load into main app after saving
        """
        # Validate that we have data
        if not self.file_paths['fp_data']:
            QMessageBox.warning(
                self,
                "Missing File",
                "Please select the photometry data file (FP_data)."
            )
            return

        if not self.file_paths['fp_data'].exists():
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected file does not exist:\n{self.file_paths['fp_data']}"
            )
            return

        if self._fp_data is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data before saving."
            )
            return

        # Build filename from source files
        fp_path = self.file_paths['fp_data']
        fp_name = fp_path.parent.name if fp_path.parent.name.lower().startswith('fp_data') else fp_path.stem

        if self.file_paths['ai_data']:
            ai_name = self.file_paths['ai_data'].stem.replace(' ', '_')
            filename = f"{fp_name}+{ai_name}_photometry.npz"
        else:
            filename = f"{fp_name}_photometry.npz"

        # Determine base folder (next to raw data)
        if fp_path.parent.name.lower().startswith('fp_data'):
            base_folder = fp_path.parent.parent
        else:
            base_folder = fp_path.parent

        # Build file path
        save_path = base_folder / filename

        # Check if file exists
        if save_path.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite?",
                f"File already exists:\n{filename}\n\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Save the data
        try:
            self._save_photometry_data(save_path)

            if load_after:
                # Store path for main app to load
                self._saved_path = save_path
                QMessageBox.information(
                    self,
                    "Saved",
                    f"Data saved to:\n{save_path}\n\nLoading into main app..."
                )
                self.accept()
            else:
                # Ask if user wants to switch to Tab 2 for processing
                reply = QMessageBox.question(
                    self,
                    "Data Saved",
                    f"Data saved to:\n{save_path}\n\nSwitch to Processing tab to configure ΔF/F?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # Load saved data and switch to Tab 2
                    self._load_photometry_npz(save_path)
                    self.tab_widget.setCurrentIndex(1)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save data:\n{str(e)}"
            )

    def _save_photometry_data(self, save_path: Path):
        """Save raw aligned photometry data to NPZ file.

        This saves the raw signals (iso, gcamp, AI channels) with aligned time vectors.
        NO ΔF/F is computed - that will be done when loading into main app based on
        session settings.
        """
        import time as time_module
        from datetime import datetime

        print(f"[Photometry] Saving data to {save_path}")
        t0 = time_module.perf_counter()

        # Get column mappings
        time_col = self.combo_time.currentData()
        led_col = self.combo_led_state.currentData()
        signal_col = self.combo_signal.currentData()

        # Process FP data
        data = self._fp_data.copy()
        if not np.issubdtype(data[led_col].dtype, np.number):
            data = data.iloc[1:].copy()
            for col in [time_col, led_col, signal_col]:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # Separate channels
        iso_mask = data[led_col] == 1
        gcamp_mask = data[led_col] == 2

        iso_time_raw = data.loc[iso_mask, time_col].values
        iso_signal = data.loc[iso_mask, signal_col].values
        gcamp_time_raw = data.loc[gcamp_mask, time_col].values
        gcamp_signal = data.loc[gcamp_mask, signal_col].values

        # Find global t_min for normalization (from all signals)
        all_times = [iso_time_raw, gcamp_time_raw]
        if self._timestamps is not None:
            all_times.append(self._timestamps)
        global_t_min = min(np.min(t) for t in all_times if len(t) > 0)

        # Normalize FP time to minutes, starting at 0
        fp_time = (iso_time_raw - global_t_min) / 60000  # iso and gcamp have same time points after interleaving
        iso = iso_signal
        gcamp = gcamp_signal

        # Estimate FP sample rate
        if len(fp_time) > 1:
            fp_dt = np.median(np.diff(fp_time)) * 60  # Convert back to seconds
            fp_sample_rate = 1.0 / fp_dt if fp_dt > 0 else 0
        else:
            fp_sample_rate = 0

        # Build save dict
        save_dict = {
            'source_fp_path': str(self.file_paths['fp_data']),
            'source_ai_path': str(self.file_paths['ai_data']) if self.file_paths['ai_data'] else '',
            'fp_time': fp_time,
            'iso': iso,
            'gcamp': gcamp,
        }

        # Add AI data if available
        if self._ai_data is not None and self._timestamps is not None:
            # Normalize AI time
            ai_time = (self._timestamps - global_t_min) / 60000

            # Estimate AI sample rate
            if len(ai_time) > 1:
                ai_dt = np.median(np.diff(ai_time)) * 60  # Convert back to seconds
                ai_sample_rate = 1.0 / ai_dt if ai_dt > 0 else 0
            else:
                ai_sample_rate = 0

            save_dict['ai_time'] = ai_time

            # Get enabled AI channels
            ai_channels = {}
            for i, col_info in self.ai_columns.items():
                if col_info['checkbox'].isChecked():
                    col_name = col_info['column']
                    label = col_info['label_edit'].text() or f"ai_{col_name}"
                    # Clean the label for use as dict key
                    label = label.replace(' ', '_')
                    ai_channels[label] = self._ai_data[col_name].values

            save_dict['ai_channels'] = ai_channels
        else:
            ai_sample_rate = 0

        # Add metadata
        save_dict['metadata'] = {
            'format_version': 1,
            'fp_sample_rate': fp_sample_rate,
            'ai_sample_rate': ai_sample_rate,
            'created': datetime.now().isoformat(),
            'fp_time_col': time_col,
            'fp_led_col': led_col,
            'fp_signal_col': signal_col,
        }

        # Save to NPZ
        np.savez(save_path, **save_dict)

        elapsed = time_module.perf_counter() - t0
        print(f"[Photometry] Data saved in {elapsed:.2f}s")
        print(f"[Photometry]   FP: {len(iso)} samples @ {fp_sample_rate:.1f} Hz")
        if 'ai_time' in save_dict:
            print(f"[Photometry]   AI: {len(save_dict['ai_time'])} samples @ {ai_sample_rate:.1f} Hz")
            print(f"[Photometry]   AI channels: {list(ai_channels.keys())}")

    # =========================================================================
    # Public Interface
    # =========================================================================

    def get_saved_path(self) -> Optional[Path]:
        """Get the path to the saved photometry data file (if Save & Load was used)."""
        return getattr(self, '_saved_path', None)

    def get_selected_files(self) -> Dict[str, Optional[Path]]:
        """Get the selected file paths."""
        return self.file_paths.copy()

    def get_column_mappings(self) -> Dict:
        """Get the column mappings."""
        return {
            'fp_data': {
                'time': self.combo_time.currentData(),
                'led_state': self.combo_led_state.currentData(),
                'signal': self.combo_signal.currentData()
            },
            'ai_data': {
                i: {
                    'column': info['column'],
                    'enabled': info['checkbox'].isChecked(),
                    'label': info['label_edit'].text()
                }
                for i, info in self.ai_columns.items()
            }
        }


# =============================================================================
# Convenience function for launching dialog
# =============================================================================

def show_photometry_import_dialog(parent=None, initial_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Show the photometry import dialog.

    Args:
        parent: Parent widget
        initial_path: Optional initial FP_data file path

    Returns:
        Dict with selected files and settings, or None if cancelled
    """
    dialog = PhotometryImportDialog(parent, initial_path)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return {
            'files': dialog.get_selected_files(),
            'columns': dialog.get_column_mappings()
        }
    return None
