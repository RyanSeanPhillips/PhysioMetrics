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
    QDoubleSpinBox, QStackedWidget, QToolButton, QMenu, QScrollArea,
    QGridLayout, QSpinBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt6.QtGui import QAction

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
        background-color: #3e3e42;
        color: #ffffff;
        padding: 5px 8px;
        border: 1px solid #555555;
        font-weight: bold;
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
        background-color: #555555;
        border-radius: 2px;
    }
    QSplitter::handle:horizontal {
        width: 6px;
        margin: 4px 2px;
        background-color: #555555;
    }
    QSplitter::handle:horizontal:hover {
        background-color: #007acc;
    }
    QSplitter::handle:horizontal:pressed {
        background-color: #0098ff;
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
    - Configure dF/F method (fitted vs simple)
    - Configure detrending (none, linear, exponential, biexponential)
    - Set fit range
    - Preview processed signals
    - Load into main application
    """

    # Signal emitted when data is ready to load into main app
    data_ready = pyqtSignal(dict)

    # Settings keys for recent paths
    SETTINGS_KEY_RECENT_FILES = "photometry/recent_files"
    SETTINGS_KEY_RECENT_FOLDERS = "photometry/recent_folders"
    SETTINGS_KEY_PINNED_PATHS = "photometry/pinned_paths"
    MAX_RECENT_ITEMS = 8

    def __init__(self, parent=None, initial_path: Optional[Path] = None,
                 photometry_npz_path: Optional[Path] = None,
                 initial_params: Optional[Dict] = None,
                 cached_photometry_data: Optional[Dict] = None):
        """
        Initialize the photometry import dialog.

        Args:
            parent: Parent widget
            initial_path: Path to raw FP_data CSV file (opens Tab 1)
            photometry_npz_path: Path to existing *_photometry.npz file (opens Tab 2)
            initial_params: Dict of processing parameters to restore (from gear icon edit)
                           Keys: dff_method, detrend_method, fit_range_start, fit_range_end,
                                 lowpass_enabled, lowpass_hz
            cached_photometry_data: Pre-loaded NPZ data dict (for fast reopening from gear icon).
                           If provided, skips loading from disk.
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
            'ai_data': None,
            'timestamps': None,
            'notes': None
        }

        # Store experiment assignments for multi-animal support
        self.experiment_assignments: Dict[str, str] = {}  # channel -> experiment_name
        self.num_experiments = 1

        # Store column mappings
        self.fp_columns = {
            'time': None,
            'led_state': None,
            'signal': None  # Legacy - kept for backward compatibility
        }
        self.fiber_columns: Dict[str, Dict] = {}  # fiber_col -> {'checkbox': QCheckBox, 'label_edit': QLineEdit, 'enabled': bool}
        self.ai_columns: Dict[int, Dict] = {}  # col_idx -> {'enabled': bool, 'label': str}

        # Store loaded data for preview
        self._fp_data = None
        self._ai_data = None
        self._timestamps = None

        # Photometry NPZ data (for Tab 2 - from saved file)
        self._photometry_data = None  # Dict from NPZ file
        self._photometry_npz_path = Path(photometry_npz_path) if photometry_npz_path else None

        # Store cached data for fast reopening (avoids reloading from disk)
        self._cached_photometry_data = cached_photometry_data

        # Store initial path for deferred loading
        self._initial_path = initial_path

        # Store initial params for restoring settings (from gear icon edit)
        self._initial_params = initial_params

        # Track which mode we're in
        self._start_on_tab2 = False

        # Initialize QSettings for recent paths
        self._settings = QSettings("PhysioMetrics", "PhotometryImport")

        # Determine which mode/tab to start in
        if cached_photometry_data is not None:
            # Cached data provided - instant opening on Tab 2
            self._start_on_tab2 = True
        elif photometry_npz_path and Path(photometry_npz_path).exists():
            # NPZ path provided - will load from disk on Tab 2
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

        # Switch to Tab 2 if using cached data or loading existing NPZ
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

    # =========================================================================
    # Style Helper Methods
    # =========================================================================

    def _get_compact_browse_button_style(self) -> str:
        """Get stylesheet for compact browse buttons."""
        return """
            QToolButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 4px;
            }
            QToolButton:hover {
                background-color: #4e4e52;
                border-color: #666666;
            }
            QToolButton::menu-button {
                border-left: 1px solid #555555;
                width: 14px;
            }
            QToolButton::menu-button:hover {
                background-color: #5e5e62;
            }
        """

    def _get_menu_style(self) -> str:
        """Get stylesheet for dropdown menus."""
        return """
            QMenu {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #555555;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QMenu::separator {
                height: 1px;
                background: #555555;
                margin: 4px 10px;
            }
        """

    def showEvent(self, event):
        """Override showEvent to defer data loading until dialog is visible."""
        super().showEvent(event)
        # Load data after dialog is shown (deferred to avoid blocking)
        if self._cached_photometry_data is not None:
            # Use cached data - instant open from gear icon
            QTimer.singleShot(10, self._use_cached_data)
        elif self._start_on_tab2 and self._photometry_npz_path:
            # Tab 2: Load existing NPZ file from disk
            QTimer.singleShot(50, lambda: self._load_photometry_npz(self._photometry_npz_path))
        elif self._initial_path and self._fp_data is None:
            # Tab 1: Load raw CSV files
            QTimer.singleShot(50, self._load_and_preview_data)

    def _use_cached_data(self):
        """Use pre-loaded cached data instead of loading from disk."""
        if self._cached_photometry_data is None:
            return

        # Convert cached photometry_raw format to _photometry_data format expected by Tab 2
        # Note: common_time from NPZ loader is in SECONDS, but Tab 2 expects MINUTES
        cached = self._cached_photometry_data
        common_time = cached.get('common_time', np.array([]))
        fp_time_minutes = common_time / 60.0 if len(common_time) > 0 else common_time
        self._photometry_data = {
            'fp_time': fp_time_minutes,
            'fibers': cached.get('fibers', {}),
            'ai_channels': cached.get('ai_channels', {}),
            'sample_rate': cached.get('sample_rate', 100.0),
        }

        # Update source label
        if self._photometry_npz_path:
            self.lbl_source_file.setText(f"Editing: {self._photometry_npz_path.name}")
        else:
            self.lbl_source_file.setText("Editing photometry settings")
        self.lbl_source_file.setStyleSheet("color: #00cc00; font-size: 10px;")

        # Apply initial params if provided
        if self._initial_params:
            self._apply_initial_params()

        # Update preview
        self._update_tab2_preview()

        print("[Photometry] Using cached data - dialog opened instantly")

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
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(8)

        # Create scroll area to wrap all content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #3e3e42;
                border-radius: 4px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #555555;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Container widget for scroll area
        scroll_content = QWidget()
        scroll_content_layout = QVBoxLayout(scroll_content)
        scroll_content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_content_layout.setSpacing(8)

        # Create VERTICAL splitter for all resizable config panels
        top_splitter = QSplitter(Qt.Orientation.Vertical)
        top_splitter.setHandleWidth(5)
        top_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3e3e42;
            }
            QSplitter::handle:hover {
                background-color: #0e639c;
            }
        """)

        # --- Section 1: Header + Source Files ---
        source_section = QWidget()
        source_section_layout = QVBoxLayout(source_section)
        source_section_layout.setContentsMargins(0, 0, 0, 0)
        source_section_layout.setSpacing(4)

        # Header
        header = QLabel("Step 1: Select Files & Map Columns")
        header.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; padding: 4px 0px;")
        source_section_layout.addWidget(header)

        # ===== SOURCE FILES: 2x2 Grid Layout =====
        source_files_group = QGroupBox("Source Files")
        source_files_layout = QGridLayout(source_files_group)
        source_files_layout.setSpacing(8)

        # Simple browse button style (no dropdown)
        browse_btn_style = """
            QPushButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 8px;
                min-width: 50px;
            }
            QPushButton:hover {
                background-color: #4e4e52;
                border-color: #666666;
            }
        """

        # ----- Photometry Data Section (Grid position 0,0) -----
        fp_frame = QFrame()
        fp_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        fp_frame.setStyleSheet("QFrame { background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; }")
        fp_frame_layout = QVBoxLayout(fp_frame)
        fp_frame_layout.setContentsMargins(8, 8, 8, 8)
        fp_frame_layout.setSpacing(4)

        fp_header = QLabel("Photometry Data (FP)")
        fp_header.setStyleSheet("font-weight: bold; color: #569cd6; border: none;")
        fp_frame_layout.addWidget(fp_header)

        # File selection row
        fp_file_row = QHBoxLayout()
        self.fp_data_edit = QLineEdit()
        self.fp_data_edit.setReadOnly(True)
        self.fp_data_edit.setPlaceholderText("Select FP_data CSV...")
        fp_file_row.addWidget(self.fp_data_edit, 1)

        # Simple browse button (no dropdown)
        self.btn_fp_browse = QPushButton("...")
        self.btn_fp_browse.setFixedWidth(32)
        self.btn_fp_browse.clicked.connect(lambda: self._browse_file('fp_data'))
        self.btn_fp_browse.setStyleSheet(browse_btn_style)
        fp_file_row.addWidget(self.btn_fp_browse)
        fp_frame_layout.addLayout(fp_file_row)

        # Column mapping row
        fp_col_row = QHBoxLayout()
        fp_col_row.addWidget(QLabel("Time:"))
        self.combo_time = QComboBox()
        self.combo_time.setFixedWidth(65)
        self.combo_time.currentIndexChanged.connect(self._on_column_changed)
        fp_col_row.addWidget(self.combo_time)
        fp_col_row.addWidget(QLabel("LED:"))
        self.combo_led_state = QComboBox()
        self.combo_led_state.setFixedWidth(65)
        self.combo_led_state.currentIndexChanged.connect(self._on_column_changed)
        fp_col_row.addWidget(self.combo_led_state)
        fp_col_row.addStretch()
        fp_frame_layout.addLayout(fp_col_row)

        # Fiber columns section (populated dynamically)
        fiber_row = QHBoxLayout()
        fiber_row.addWidget(QLabel("Fibers:"))
        self.fiber_columns_widget = QWidget()
        self.fiber_columns_layout = QHBoxLayout(self.fiber_columns_widget)
        self.fiber_columns_layout.setContentsMargins(0, 0, 0, 0)
        self.fiber_columns_layout.setSpacing(4)
        fiber_row.addWidget(self.fiber_columns_widget, 1)
        fp_frame_layout.addLayout(fiber_row)

        # Hidden combo for backward compatibility
        self.combo_signal = QComboBox()
        self.combo_signal.setVisible(False)

        # Compact preview table
        self.fp_preview_table = QTableWidget()
        self.fp_preview_table.setMinimumHeight(60)
        self.fp_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.fp_preview_table.horizontalHeader().setFixedHeight(24)
        self.fp_preview_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.fp_preview_table.verticalHeader().setVisible(False)
        fp_frame_layout.addWidget(self.fp_preview_table)

        source_files_layout.addWidget(fp_frame, 0, 0)  # Row 0, Col 0 (left column)

        # ----- Analog Inputs Section (Grid position 1,0 - below FP in left column) -----
        ai_frame = QFrame()
        ai_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        ai_frame.setStyleSheet("QFrame { background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; }")
        ai_frame_layout = QVBoxLayout(ai_frame)
        ai_frame_layout.setContentsMargins(8, 8, 8, 8)
        ai_frame_layout.setSpacing(4)

        ai_header = QLabel("Analog Inputs (AI)")
        ai_header.setStyleSheet("font-weight: bold; color: #4ec9b0; border: none;")
        ai_frame_layout.addWidget(ai_header)

        # File selection row
        ai_file_row = QHBoxLayout()
        self.ai_data_edit = QLineEdit()
        self.ai_data_edit.setReadOnly(True)
        self.ai_data_edit.setPlaceholderText("Optional: AI data CSV")
        ai_file_row.addWidget(self.ai_data_edit, 1)

        self.btn_ai_browse = QPushButton("...")
        self.btn_ai_browse.setFixedWidth(32)
        self.btn_ai_browse.clicked.connect(lambda: self._browse_file('ai_data'))
        self.btn_ai_browse.setStyleSheet(browse_btn_style)
        ai_file_row.addWidget(self.btn_ai_browse)
        ai_frame_layout.addLayout(ai_file_row)

        # Column checkboxes container (populated dynamically)
        self.ai_columns_widget = QWidget()
        self.ai_columns_layout = QHBoxLayout(self.ai_columns_widget)
        self.ai_columns_layout.setContentsMargins(0, 0, 0, 0)
        self.ai_columns_layout.setSpacing(4)
        ai_frame_layout.addWidget(self.ai_columns_widget)

        # Compact preview table
        self.ai_preview_table = QTableWidget()
        self.ai_preview_table.setMinimumHeight(60)
        self.ai_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.ai_preview_table.horizontalHeader().setFixedHeight(24)
        self.ai_preview_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.ai_preview_table.verticalHeader().setVisible(False)
        ai_frame_layout.addWidget(self.ai_preview_table)

        source_files_layout.addWidget(ai_frame, 1, 0)  # Row 1, Col 0 (left column)

        # ----- Timestamps Section (Grid position 0,1 - right column) -----
        ts_frame = QFrame()
        ts_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        ts_frame.setStyleSheet("QFrame { background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; }")
        ts_frame_layout = QVBoxLayout(ts_frame)
        ts_frame_layout.setContentsMargins(8, 8, 8, 8)
        ts_frame_layout.setSpacing(4)

        ts_header = QLabel("Timestamps")
        ts_header.setStyleSheet("font-weight: bold; color: #dcdcaa; border: none;")
        ts_frame_layout.addWidget(ts_header)

        # Timestamps file row
        ts_file_row = QHBoxLayout()
        self.timestamps_edit = QLineEdit()
        self.timestamps_edit.setReadOnly(True)
        self.timestamps_edit.setPlaceholderText("Auto-detected from FP folder")
        self.timestamps_edit.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: #888888; }")
        ts_file_row.addWidget(self.timestamps_edit, 1)

        self.btn_ts_browse = QPushButton("...")
        self.btn_ts_browse.setFixedWidth(32)
        self.btn_ts_browse.clicked.connect(lambda: self._browse_file('timestamps'))
        self.btn_ts_browse.setStyleSheet(browse_btn_style)
        ts_file_row.addWidget(self.btn_ts_browse)
        ts_frame_layout.addLayout(ts_file_row)

        # Timestamps preview info (shows count and range)
        self.ts_info_label = QLabel("No timestamps loaded")
        self.ts_info_label.setStyleSheet("color: #888888; font-size: 10px; border: none;")
        self.ts_info_label.setWordWrap(True)
        ts_frame_layout.addWidget(self.ts_info_label)

        ts_frame_layout.addStretch()
        source_files_layout.addWidget(ts_frame, 0, 1)  # Row 0, Col 1 (right column)

        # ----- Notes Section (Grid position 1,1 - right column) -----
        notes_frame = QFrame()
        notes_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        notes_frame.setStyleSheet("QFrame { background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; }")
        notes_frame_layout = QVBoxLayout(notes_frame)
        notes_frame_layout.setContentsMargins(8, 8, 8, 8)
        notes_frame_layout.setSpacing(4)

        notes_header_row = QHBoxLayout()
        notes_header = QLabel("Notes File")
        notes_header.setStyleSheet("font-weight: bold; color: #ce9178; border: none;")
        notes_header_row.addWidget(notes_header)
        notes_header_row.addStretch()

        # View popout button
        self.btn_notes_view = QPushButton("View")
        self.btn_notes_view.setFixedWidth(50)
        self.btn_notes_view.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:disabled { background-color: #3e3e42; color: #888888; }
        """)
        self.btn_notes_view.clicked.connect(self._show_notes_popout)
        self.btn_notes_view.setEnabled(False)
        notes_header_row.addWidget(self.btn_notes_view)
        notes_frame_layout.addLayout(notes_header_row)

        # Notes file row
        notes_file_row = QHBoxLayout()
        self.notes_edit = QLineEdit()
        self.notes_edit.setReadOnly(True)
        self.notes_edit.setPlaceholderText("Optional: .txt, .csv, .xlsx, .docx")
        notes_file_row.addWidget(self.notes_edit, 1)

        self.btn_notes_browse = QPushButton("...")
        self.btn_notes_browse.setFixedWidth(32)
        self.btn_notes_browse.clicked.connect(lambda: self._browse_file('notes'))
        self.btn_notes_browse.setStyleSheet(browse_btn_style)
        notes_file_row.addWidget(self.btn_notes_browse)
        notes_frame_layout.addLayout(notes_file_row)

        # Notes inline preview (first few lines)
        self.notes_preview_label = QLabel("No notes file selected")
        self.notes_preview_label.setStyleSheet("color: #888888; font-size: 10px; border: none; padding: 4px; background-color: #1e1e1e; border-radius: 2px;")
        self.notes_preview_label.setWordWrap(True)
        self.notes_preview_label.setMaximumHeight(60)
        notes_frame_layout.addWidget(self.notes_preview_label)

        notes_frame_layout.addStretch()
        source_files_layout.addWidget(notes_frame, 1, 1)  # Row 1, Col 1 (right column)

        # Set column stretch: left column (FP + AI) 2/3, right column (Timestamps + Notes) 1/3
        source_files_layout.setColumnStretch(0, 2)  # Left column: 2/3
        source_files_layout.setColumnStretch(1, 1)  # Right column: 1/3

        source_section_layout.addWidget(source_files_group)
        top_splitter.addWidget(source_section)

        # --- Section 2: Signal Preview (moved here for better visibility) ---
        preview_section = QWidget()
        preview_section_layout = QVBoxLayout(preview_section)
        preview_section_layout.setContentsMargins(0, 0, 0, 0)
        preview_section_layout.setSpacing(4)

        # Header row with loading label and toolbar
        preview_header_row = QHBoxLayout()
        plot_header = QLabel("Signal Preview")
        plot_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff; padding: 4px 0px;")
        preview_header_row.addWidget(plot_header)

        self.loading_label = QLabel("")
        self.loading_label.setStyleSheet("color: #ffcc00; font-style: italic;")
        self.loading_label.setVisible(False)
        preview_header_row.addWidget(self.loading_label)
        preview_header_row.addStretch()

        # Navigation toolbar
        self.preview_figure = Figure(figsize=(10, 4), dpi=100, facecolor='#1e1e1e')
        self.preview_canvas = FigureCanvas(self.preview_figure)
        self.nav_toolbar = NavigationToolbar(self.preview_canvas, preview_section)
        self.nav_toolbar.setStyleSheet("""
            QToolBar { background: transparent; border: none; spacing: 2px; }
            QToolButton { background: transparent; color: #888888; border: none; padding: 4px; border-radius: 3px; }
            QToolButton:hover { background-color: #3e3e42; color: #ffffff; }
            QToolButton:pressed { background-color: #094771; color: #ffffff; }
        """)
        preview_header_row.addWidget(self.nav_toolbar)
        preview_section_layout.addLayout(preview_header_row)

        # Tabbed preview widget
        self.preview_tab_widget = QTabWidget()
        self.preview_tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d30;
                color: #d4d4d4;
                padding: 4px 12px;
                border: 1px solid #3e3e42;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e;
                border-bottom: 1px solid #1e1e1e;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3e3e42;
            }
        """)

        # "All Channels" tab - main preview
        all_tab = QWidget()
        all_layout = QVBoxLayout(all_tab)
        all_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_canvas.setMinimumHeight(400)  # Increased for better signal visibility
        all_layout.addWidget(self.preview_canvas, 1)

        self.preview_tab_widget.addTab(all_tab, "All Channels")

        # Store references for experiment tabs (created dynamically)
        self.experiment_preview_tabs = {}
        self.experiment_figures = {}
        self.experiment_canvases = {}

        preview_section_layout.addWidget(self.preview_tab_widget, 1)

        # Hidden spinboxes for time range
        self.spin_time_start = QDoubleSpinBox()
        self.spin_time_start.setRange(0, 100000)
        self.spin_time_start.setDecimals(1)
        self.spin_time_start.setVisible(False)
        self.spin_time_end = QDoubleSpinBox()
        self.spin_time_end.setRange(0, 100000)
        self.spin_time_end.setDecimals(1)
        self.spin_time_end.setVisible(False)

        top_splitter.addWidget(preview_section)

        # --- Section 3: Channel Assignment ---
        channel_section = QWidget()
        channel_section_layout = QVBoxLayout(channel_section)
        channel_section_layout.setContentsMargins(0, 0, 0, 0)
        channel_section_layout.setSpacing(4)

        # ===== CHANNEL ASSIGNMENT Section =====
        channel_group = QGroupBox("Channel Assignment")
        channel_layout = QVBoxLayout(channel_group)
        channel_layout.setSpacing(4)

        # Number of experiments row
        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Number of experiments/animals:"))
        self.spin_num_experiments = QSpinBox()
        self.spin_num_experiments.setRange(1, 10)
        self.spin_num_experiments.setValue(1)
        self.spin_num_experiments.setFixedWidth(60)
        self.spin_num_experiments.valueChanged.connect(self._on_num_experiments_changed)
        exp_row.addWidget(self.spin_num_experiments)
        exp_row.addStretch()
        channel_layout.addLayout(exp_row)

        # Channel assignment table
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(4)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Label", "Assign To", "Type"])
        self.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.channel_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.channel_table.setMinimumHeight(120)  # Increased for better visibility
        self.channel_table.verticalHeader().setVisible(False)
        channel_layout.addWidget(self.channel_table)

        channel_section_layout.addWidget(channel_group)
        top_splitter.addWidget(channel_section)

        # --- Section 4: Output Files + Data Options ---
        output_section = QWidget()
        output_section_layout = QVBoxLayout(output_section)
        output_section_layout.setContentsMargins(0, 0, 0, 0)
        output_section_layout.setSpacing(4)

        # ===== OUTPUT FILES and DATA OPTIONS in a row =====
        options_row = QHBoxLayout()

        # Output Files section
        output_group = QGroupBox("Output Files")
        output_layout = QVBoxLayout(output_group)
        output_layout.setSpacing(4)

        # Base filename row
        base_row = QHBoxLayout()
        base_row.addWidget(QLabel("Base name:"))
        self.output_base_edit = QLineEdit()
        self.output_base_edit.setPlaceholderText("e.g., 2025-01-15_experiment")
        self.output_base_edit.textChanged.connect(self._update_output_file_list)
        base_row.addWidget(self.output_base_edit, 1)
        output_layout.addLayout(base_row)

        # Output files list
        self.output_files_label = QLabel("Output files will appear here...")
        self.output_files_label.setStyleSheet("color: #888888; font-size: 10px; padding: 4px;")
        self.output_files_label.setWordWrap(True)
        output_layout.addWidget(self.output_files_label)

        options_row.addWidget(output_group, 2)

        # Data Options section
        proc_group = QGroupBox("Data Options")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setSpacing(4)

        # Time normalization
        self.chk_normalize_time = QCheckBox("Normalize time (start at t=0)")
        self.chk_normalize_time.setChecked(True)
        self.chk_normalize_time.setToolTip("Align all signals to start at time 0")
        self.chk_normalize_time.stateChanged.connect(self._update_preview_plot)
        proc_layout.addWidget(self.chk_normalize_time)

        # Update Preview button
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
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #094771; }
        """)
        btn_update.clicked.connect(self._update_preview_plot)
        proc_layout.addWidget(btn_update)
        proc_layout.addStretch()

        # Placeholder attributes for backward compatibility
        self.combo_dff_method = None
        self.combo_detrend_method = None
        self.spin_detrend_start = None
        self.spin_detrend_end = None
        self.chk_select_range = None
        self.spin_exclude_start = None
        self.chk_lowpass = None
        self.spin_lowpass = None
        self.chk_show_intermediates = None

        options_row.addWidget(proc_group, 1)

        output_section_layout.addLayout(options_row)
        top_splitter.addWidget(output_section)

        # Set initial sizes for top splitter sections (Source Files, Signal Preview, Channel Assignment, Output Files)
        # Signal preview is 2.5x larger (300 -> 750), channel assignment increased for visibility
        top_splitter.setSizes([250, 750, 200, 100])

        # Set minimum height for the splitter so it doesn't collapse
        top_splitter.setMinimumHeight(800)

        scroll_content_layout.addWidget(top_splitter)
        scroll_area.setWidget(scroll_content)
        tab_layout.addWidget(scroll_area)

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
        """Create Tab 2: dF/F Processing and Preview."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        main_layout = QHBoxLayout()

        # Create splitter for resizable left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)  # Make handle wide enough to grab easily

        # =====================================================================
        # Left panel: Processing options
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 8, 8, 8)

        # Hidden label for tracking source (used internally)
        self.lbl_source_file = QLabel("")
        self.lbl_source_file.setVisible(False)

        # dF/F Method
        method_group = QGroupBox("dF/F Calculation")
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
        """Initialize Tab 2 settings with reasonable defaults.

        Note: Tab 1 no longer has processing controls - it's raw data assembly only.
        Tab 2 controls are independent and initialized with defaults in _create_tab2_processing().
        """
        # No syncing needed - Tab 2 has its own controls with defaults
        pass

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

            # Apply initial params if provided (from gear icon edit)
            if self._initial_params:
                self._apply_initial_params()

            # Update preview
            self._update_tab2_preview()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Load Error",
                f"Failed to load photometry data:\n{str(e)}"
            )

    def _apply_initial_params(self):
        """Apply initial processing parameters to Tab 2 widgets.

        Handles both naming conventions:
        - Tab 2 keys: dff_method, fit_range_start, fit_range_end
        - NPZ/Tab 1 keys: method, fit_start, fit_end
        """
        params = self._initial_params
        if not params:
            return

        # Block signals while setting values to avoid redundant preview updates
        self.combo_dff_method_tab2.blockSignals(True)
        self.combo_detrend_method_tab2.blockSignals(True)
        self.spin_fit_start_tab2.blockSignals(True)
        self.spin_fit_end_tab2.blockSignals(True)
        self.chk_lowpass_tab2.blockSignals(True)
        self.spin_lowpass_tab2.blockSignals(True)

        try:
            # dF/F method — check both 'dff_method' (Tab 2) and 'method' (NPZ/Tab 1)
            dff_method = params.get('dff_method', params.get('method'))
            if dff_method is not None:
                idx = self.combo_dff_method_tab2.findData(dff_method)
                if idx >= 0:
                    self.combo_dff_method_tab2.setCurrentIndex(idx)

            # Detrend method
            if 'detrend_method' in params:
                idx = self.combo_detrend_method_tab2.findData(params['detrend_method'])
                if idx >= 0:
                    self.combo_detrend_method_tab2.setCurrentIndex(idx)

            # Fit range — check both 'fit_range_start/end' (Tab 2) and 'fit_start/end' (NPZ/Tab 1)
            fit_start = params.get('fit_range_start', params.get('fit_start'))
            if fit_start is not None:
                self.spin_fit_start_tab2.setValue(float(fit_start))
            fit_end = params.get('fit_range_end', params.get('fit_end'))
            if fit_end is not None:
                self.spin_fit_end_tab2.setValue(float(fit_end))

            # Lowpass filter — check 'lowpass_enabled' (Tab 2) or infer from 'lowpass_hz' (NPZ)
            lowpass_hz = params.get('lowpass_hz')
            if 'lowpass_enabled' in params:
                self.chk_lowpass_tab2.setChecked(params['lowpass_enabled'])
            elif lowpass_hz is not None and lowpass_hz > 0:
                # NPZ format: infer enabled from lowpass_hz having a positive value
                self.chk_lowpass_tab2.setChecked(True)
            if lowpass_hz is not None and lowpass_hz > 0:
                self.spin_lowpass_tab2.setValue(lowpass_hz)

            print(f"[Photometry] Restored settings: {params}")

        finally:
            # Re-enable signals
            self.combo_dff_method_tab2.blockSignals(False)
            self.combo_detrend_method_tab2.blockSignals(False)
            self.spin_fit_start_tab2.blockSignals(False)
            self.spin_fit_end_tab2.blockSignals(False)
            self.chk_lowpass_tab2.blockSignals(False)
            self.spin_lowpass_tab2.blockSignals(False)

    def _on_processing_changed(self):
        """Handle changes to processing options - auto-update preview."""
        # Only update if we have data loaded
        if self._photometry_data is not None:
            self._update_tab2_preview()

    def _update_tab2_preview(self):
        """Update the preview plot on Tab 2 with processed data.

        Supports both single-fiber (legacy) and multi-fiber (v2) formats.
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

        # Extract time vector
        fp_time = self._photometry_data.get('fp_time', np.array([]))
        if len(fp_time) == 0:
            self.fig_tab2.clear()
            ax = self.fig_tab2.add_subplot(111)
            ax.set_facecolor('#252526')
            ax.text(0.5, 0.5, 'Invalid data format: missing time vector.',
                   ha='center', va='center', color='#ff6666', fontsize=12,
                   transform=ax.transAxes)
            self.canvas_tab2.draw()
            return

        # Check for multi-fiber format (v2)
        fibers_data = self._photometry_data.get('fibers')
        if fibers_data is not None:
            if hasattr(fibers_data, 'item'):
                fibers_data = fibers_data.item()

        # Build list of fibers to process
        if fibers_data and isinstance(fibers_data, dict) and len(fibers_data) > 0:
            # Multi-fiber format (v2)
            fiber_list = list(fibers_data.keys())
        else:
            # Legacy single-fiber format - create synthetic fiber entry
            iso = self._photometry_data.get('iso', np.array([]))
            gcamp = self._photometry_data.get('gcamp', np.array([]))
            if len(iso) == 0 or len(gcamp) == 0:
                self.fig_tab2.clear()
                ax = self.fig_tab2.add_subplot(111)
                ax.set_facecolor('#252526')
                ax.text(0.5, 0.5, 'Invalid data format: missing signals.',
                       ha='center', va='center', color='#ff6666', fontsize=12,
                       transform=ax.transAxes)
                self.canvas_tab2.draw()
                return
            fibers_data = {'Fiber': {'iso': iso, 'gcamp': gcamp, 'label': 'Fiber'}}
            fiber_list = ['Fiber']

        n_fibers = len(fiber_list)
        print(f"[Photometry] Tab2 preview: {n_fibers} fiber(s): {fiber_list}")

        # Compute dF/F for each fiber
        processed_fibers = {}
        for fiber_col in fiber_list:
            fiber_info = fibers_data[fiber_col]
            iso = fiber_info['iso']
            gcamp = fiber_info['gcamp']
            label = fiber_info.get('label', fiber_col)

            # Create time arrays that match each signal's length
            # (iso and gcamp may differ by 1-2 samples due to LED state separation)
            t_min = fp_time[0] if len(fp_time) > 0 else 0
            t_max = fp_time[-1] if len(fp_time) > 0 else 1
            iso_time = np.linspace(t_min, t_max, len(iso)) if len(iso) > 0 else np.array([])
            gcamp_time = np.linspace(t_min, t_max, len(gcamp)) if len(gcamp) > 0 else np.array([])

            try:
                dff_time, dff_signal, intermediates = self._compute_dff(
                    iso_time, iso, gcamp_time, gcamp,
                    method=dff_method,
                    detrend_method=detrend_method,
                    lowpass_hz=lowpass_hz,
                    exclude_start_min=0,
                    detrend_fit_start=fit_start,
                    detrend_fit_end=fit_end,
                    return_intermediates=True
                )
                processed_fibers[fiber_col] = {
                    'label': label,
                    'dff_time': dff_time,
                    'dff': dff_signal,
                    'intermediates': intermediates,
                    'iso': iso,
                    'gcamp': gcamp,
                }
            except Exception as e:
                print(f"[Photometry] Error computing dF/F for {fiber_col}: {e}")
                continue

        if not processed_fibers:
            self.fig_tab2.clear()
            ax = self.fig_tab2.add_subplot(111)
            ax.set_facecolor('#252526')
            ax.text(0.5, 0.5, 'Error computing dF/F for all fibers.',
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

        # Calculate number of plots per fiber
        # For each fiber: Raw aligned (1) + intermediates (0-2) + dF/F (1)
        n_intermediate_per_fiber = 0
        if show_intermediates:
            if dff_method == 'fitted':
                n_intermediate_per_fiber += 1
            if detrend_method != 'none':
                n_intermediate_per_fiber += 1

        # For multi-fiber: show simplified view (1 raw + 1 dF/F per fiber)
        # For single fiber: show detailed view with intermediates
        if n_fibers == 1:
            n_plots_per_fiber = 1 + n_intermediate_per_fiber + 1  # Raw + intermediates + dF/F
        else:
            # Multi-fiber: simplified view - just raw (dual-axis) + dF/F per fiber
            n_plots_per_fiber = 2  # Raw aligned + dF/F

        # Check for AI channels
        ai_time = self._photometry_data.get('ai_time')
        ai_channels = self._photometry_data.get('ai_channels')
        if ai_channels is not None:
            ai_channels = ai_channels.item() if hasattr(ai_channels, 'item') else ai_channels
            n_ai = len(ai_channels)
        else:
            n_ai = 0

        n_plots = n_fibers * n_plots_per_fiber + n_ai

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
        photometry_axes = []

        # Color palette for multiple fibers
        dff_colors = ['#ff9900', '#00ccff', '#ff66ff', '#66ff66']

        # Plot each fiber
        for fiber_idx, (fiber_col, fdata) in enumerate(processed_fibers.items()):
            label = fdata['label']
            intermediates = fdata['intermediates']
            dff_time = fdata['dff_time']
            dff_signal = fdata['dff']
            int_time = intermediates.get('time', dff_time) if intermediates else dff_time
            fit_params = intermediates.get('fit_params', {}) if intermediates else {}

            iso_aligned = intermediates.get('iso_aligned', fdata['iso']) if intermediates else fdata['iso']
            gcamp_aligned = intermediates.get('gcamp_aligned', fdata['gcamp']) if intermediates else fdata['gcamp']

            # Plot 1: Raw aligned signals (dual y-axis)
            ax = axes[ax_idx]
            t_plot, s_plot = self._subsample_for_preview(int_time, gcamp_aligned)
            ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP')
            ax.set_ylabel(f'GCaMP ({label})', fontsize=8, color='#00cc00')
            ax.tick_params(axis='y', labelcolor='#00cc00')

            ax_iso = ax.twinx()
            t_plot, s_plot = self._subsample_for_preview(int_time, iso_aligned)
            ax_iso.plot(t_plot, s_plot, color='#5555ff', linewidth=0.5, alpha=0.8, label='Iso')
            ax_iso.set_ylabel('Iso', fontsize=8, color='#5555ff')
            ax_iso.tick_params(axis='y', labelcolor='#5555ff')
            ax_iso.set_facecolor('#252526')
            for spine in ax_iso.spines.values():
                spine.set_color('#3e3e42')

            ax.set_title(f'Raw: {label} ({fiber_col})', fontsize=9, color='#cccccc', loc='left')
            add_fit_range(ax)
            photometry_axes.append(ax)
            ax_idx += 1

            # Intermediate plots (only for single fiber, or if intermediates enabled)
            if n_fibers == 1 and show_intermediates and intermediates:
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

                # Raw dF/F with detrend curve
                if detrend_method != 'none' and intermediates.get('dff_raw') is not None:
                    ax = axes[ax_idx]
                    dff_raw = intermediates['dff_raw']
                    detrend_curve = intermediates.get('detrend_curve')

                    t_plot, s_plot = self._subsample_for_preview(int_time, dff_raw)
                    ax.plot(t_plot, s_plot, color='#888888', linewidth=0.5, alpha=0.7, label='Raw dF/F')

                    if detrend_curve is not None and len(detrend_curve) == len(int_time):
                        detrend_name = fit_params.get('detrend_method', 'trend')
                        t_plot, s_plot = self._subsample_for_preview(int_time, detrend_curve)
                        ax.plot(t_plot, s_plot, color='#ff5555', linewidth=1.5, alpha=0.9, label=f'{detrend_name}')

                    ax.set_ylabel('Raw dF/F (%)', fontsize=8)
                    ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                    ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
                    add_fit_range(ax)
                    photometry_axes.append(ax)
                    ax_idx += 1

            # Final dF/F
            ax = axes[ax_idx]
            dff_color = dff_colors[fiber_idx % len(dff_colors)]
            t_plot, s_plot = self._subsample_for_preview(dff_time, dff_signal)
            ax.plot(t_plot, s_plot, color=dff_color, linewidth=0.5, alpha=0.8)
            ax.set_ylabel(f'dF/F ({label}) %', fontsize=8)
            ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
            ax.set_title(f'dF/F: {label}', fontsize=9, color='#cccccc', loc='left')
            add_fit_range(ax)
            photometry_axes.append(ax)
            ax.relim()
            ax.autoscale_view(scaley=True)
            ax_idx += 1

        # AI channels
        if ai_channels is not None and ai_time is not None:
            for ai_label, signal in ai_channels.items():
                if ax_idx < len(axes):
                    ax = axes[ax_idx]

                    # Color based on label (matching Tab 1)
                    label_lower = ai_label.lower()
                    if 'therm' in label_lower or 'stim' in label_lower or 'temp' in label_lower:
                        color = '#ff4444'
                    else:
                        color = '#cccccc'

                    t_plot, s_plot = self._subsample_for_preview(ai_time, signal)
                    ax.plot(t_plot, s_plot, color=color, linewidth=0.5, alpha=0.8)
                    ax.set_ylabel(ai_label, fontsize=8)
                    ax_idx += 1

        # Set x-label on bottom plot
        axes[-1].set_xlabel('Time (minutes)', fontsize=9)

        self.fig_tab2.tight_layout()

        # Create span selectors on photometry axes
        self._create_span_selectors_tab2(photometry_axes)

        self.canvas_tab2.draw()

    def _on_load_into_app(self):
        """Load processed photometry data into the main application.

        Supports both single-fiber (legacy) and multi-fiber (v2) data formats.
        """
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

        # Extract time vector
        fp_time = self._photometry_data.get('fp_time', np.array([]))
        if len(fp_time) == 0:
            QMessageBox.warning(self, "Invalid Data", "Missing time vector.")
            return

        # Check for multi-fiber format (v2)
        fibers_data = self._photometry_data.get('fibers')
        if fibers_data is not None:
            # Handle numpy object array from npz
            if hasattr(fibers_data, 'item'):
                fibers_data = fibers_data.item()

        # Build list of fibers to process
        if fibers_data and isinstance(fibers_data, dict) and len(fibers_data) > 0:
            # Multi-fiber format (v2)
            fiber_list = list(fibers_data.keys())
            print(f"[Photometry] Multi-fiber format detected: {fiber_list}")
        else:
            # Legacy single-fiber format - create synthetic fiber entry
            iso = self._photometry_data.get('iso', np.array([]))
            gcamp = self._photometry_data.get('gcamp', np.array([]))
            if len(iso) == 0 or len(gcamp) == 0:
                QMessageBox.warning(self, "Invalid Data", "Missing iso/gcamp signals.")
                return
            fibers_data = {
                'Fiber': {'iso': iso, 'gcamp': gcamp, 'label': 'Fiber'}
            }
            fiber_list = ['Fiber']
            print(f"[Photometry] Legacy single-fiber format")

        # Process each fiber
        lowpass_hz = processing_params['lowpass_hz'] if processing_params['lowpass_enabled'] else None
        processed_fibers = {}  # fiber_col -> {dff_time, dff, iso_aligned, gcamp_aligned}

        for fiber_col in fiber_list:
            fiber_info = fibers_data[fiber_col]
            iso = fiber_info['iso']
            gcamp = fiber_info['gcamp']
            label = fiber_info.get('label', fiber_col)

            print(f"[Photometry] Processing fiber {fiber_col} ({label})...")

            # Create time arrays that match each signal's length
            # (iso and gcamp may differ by 1-2 samples due to LED state separation)
            t_min = fp_time[0] if len(fp_time) > 0 else 0
            t_max = fp_time[-1] if len(fp_time) > 0 else 1
            iso_time = np.linspace(t_min, t_max, len(iso)) if len(iso) > 0 else np.array([])
            gcamp_time = np.linspace(t_min, t_max, len(gcamp)) if len(gcamp) > 0 else np.array([])

            try:
                dff_time, dff_signal, intermediates = self._compute_dff(
                    iso_time, iso, gcamp_time, gcamp,
                    method=processing_params['dff_method'],
                    detrend_method=processing_params['detrend_method'],
                    lowpass_hz=lowpass_hz,
                    exclude_start_min=0,
                    detrend_fit_start=processing_params['fit_range_start'],
                    detrend_fit_end=processing_params['fit_range_end'],
                    return_intermediates=True
                )

                processed_fibers[fiber_col] = {
                    'label': label,
                    'dff_time': dff_time,
                    'dff': dff_signal,
                    'iso_aligned': intermediates.get('iso_aligned', iso),
                    'gcamp_aligned': intermediates.get('gcamp_aligned', gcamp),
                }
                print(f"[Photometry]   {fiber_col}: {len(dff_signal)} samples")

            except Exception as e:
                QMessageBox.critical(
                    self, "Processing Error",
                    f"Failed to compute dF/F for fiber {fiber_col}:\n{str(e)}"
                )
                return

        if not processed_fibers:
            QMessageBox.warning(self, "Processing Error", "No fibers processed.")
            return

        # Use first fiber's time as reference
        first_fiber = processed_fibers[fiber_list[0]]
        time_seconds = first_fiber['dff_time'] * 60  # Convert minutes to seconds

        # Estimate sample rate
        if len(time_seconds) > 1:
            dt = (time_seconds[-1] - time_seconds[0]) / (len(time_seconds) - 1)
            sample_rate_hz = 1.0 / dt if dt > 0 else 20.0
        else:
            sample_rate_hz = 20.0

        # Get AI data if available
        ai_time = self._photometry_data.get('ai_time')
        ai_channels = self._photometry_data.get('ai_channels')
        if ai_channels is not None:
            ai_channels = ai_channels.item() if hasattr(ai_channels, 'item') else ai_channels

        # Determine master time base - use AI if available
        from scipy import interpolate

        sweeps = {}
        channel_names = []

        if ai_channels is not None and ai_time is not None:
            # Use AI time as master (higher sample rate preserves pleth signal)
            # Handle numpy array from npz - use the array directly, don't call .item()
            if hasattr(ai_time, 'item') and ai_time.ndim == 0:
                ai_time_arr = ai_time.item()  # 0-d array, extract scalar
            else:
                ai_time_arr = np.asarray(ai_time)  # Ensure it's a numpy array
            ai_time_seconds = ai_time_arr * 60  # Convert minutes to seconds
            master_time = ai_time_seconds

            # Calculate AI sample rate
            if len(ai_time_seconds) > 1:
                ai_dt = (ai_time_seconds[-1] - ai_time_seconds[0]) / (len(ai_time_seconds) - 1)
                ai_sample_rate = 1.0 / ai_dt if ai_dt > 0 else 1000.0
            else:
                ai_sample_rate = 1000.0

            print(f"[Photometry] Using AI time base: {ai_sample_rate:.1f} Hz ({len(ai_time_seconds)} samples)")

            # Upsample photometry channels for each fiber
            for fiber_col, fiber_proc in processed_fibers.items():
                label = fiber_proc['label']

                interp_iso = interpolate.interp1d(
                    time_seconds, fiber_proc['iso_aligned'],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                interp_gcamp = interpolate.interp1d(
                    time_seconds, fiber_proc['gcamp_aligned'],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )
                interp_dff = interpolate.interp1d(
                    time_seconds, fiber_proc['dff'],
                    kind='linear', bounds_error=False, fill_value='extrapolate'
                )

                # Create channel names with fiber label
                if len(processed_fibers) > 1:
                    iso_name = f'Iso ({label})'
                    gcamp_name = f'GCaMP ({label})'
                    dff_name = f'dF/F ({label})'
                else:
                    iso_name = 'Isosbestic (415nm)'
                    gcamp_name = 'GCaMP (470nm)'
                    dff_name = 'dF/F'

                sweeps[iso_name] = interp_iso(master_time).reshape(-1, 1)
                sweeps[gcamp_name] = interp_gcamp(master_time).reshape(-1, 1)
                sweeps[dff_name] = interp_dff(master_time).reshape(-1, 1)
                channel_names.extend([iso_name, gcamp_name, dff_name])

            # Add AI channels at native sample rate
            for label, signal in ai_channels.items():
                sweeps[label] = signal.reshape(-1, 1)
                channel_names.append(label)
                print(f"[Photometry] Added AI channel: {label}")

            time_seconds = master_time
            sample_rate_hz = ai_sample_rate

        else:
            # No AI channels - use photometry time base
            for fiber_col, fiber_proc in processed_fibers.items():
                label = fiber_proc['label']

                # Create channel names with fiber label
                if len(processed_fibers) > 1:
                    iso_name = f'Iso ({label})'
                    gcamp_name = f'GCaMP ({label})'
                    dff_name = f'dF/F ({label})'
                else:
                    iso_name = 'Isosbestic (415nm)'
                    gcamp_name = 'GCaMP (470nm)'
                    dff_name = 'dF/F'

                sweeps[iso_name] = fiber_proc['iso_aligned'].reshape(-1, 1)
                sweeps[gcamp_name] = fiber_proc['gcamp_aligned'].reshape(-1, 1)
                sweeps[dff_name] = fiber_proc['dff'].reshape(-1, 1)
                channel_names.extend([iso_name, gcamp_name, dff_name])

            print(f"[Photometry] Using FP time base: {sample_rate_hz:.1f} Hz ({len(time_seconds)} samples)")

        # Store raw data for recalculation (when user clicks gear icon)
        raw_photometry_data = {
            'fp_time': fp_time,
            'fibers': fibers_data,
            'ai_time': ai_time,
            'ai_channels': ai_channels,
            # Legacy fields for backward compatibility
            'iso': fibers_data[fiber_list[0]]['iso'],
            'gcamp': fibers_data[fiber_list[0]]['gcamp'],
        }

        # Determine primary dF/F channel name for gear icon
        if len(processed_fibers) > 1:
            first_label = processed_fibers[fiber_list[0]]['label']
            dff_channel_name = f'dF/F ({first_label})'
        else:
            dff_channel_name = 'dF/F'

        # Build complete result
        self._result_data = {
            'sweeps': sweeps,
            'channel_names': channel_names,
            't': time_seconds,
            'sr_hz': sample_rate_hz,
            'photometry_raw': raw_photometry_data,
            'photometry_params': processing_params,
            'photometry_npz_path': self._photometry_npz_path,
            'dff_channel_name': dff_channel_name,
            'fiber_count': len(processed_fibers),
        }

        # Store for compatibility
        self._processing_params = processing_params
        self._saved_path = self._photometry_npz_path

        print(f"[Photometry] Prepared data for main app:")
        print(f"  Fibers: {len(processed_fibers)}")
        print(f"  Channels: {channel_names}")
        print(f"  Samples: {len(time_seconds)}")
        print(f"  Duration: {time_seconds[-1]:.1f}s ({time_seconds[-1]/60:.1f} min)")
        print(f"  Sample rate: {sample_rate_hz:.1f} Hz")

        self.accept()

    def get_result_data(self) -> Optional[Dict]:
        """Get the processed data ready for loading into main app.

        Returns:
            Dict with keys:
            - 'sweeps': Dict[str, np.ndarray] - channel_name -> (n_samples, 1) array
            - 'channel_names': List[str] - ordered list of channel names
            - 't': np.ndarray - time vector in seconds
            - 'sr_hz': float - sample rate
            - 'photometry_raw': Dict - raw data for recalculation
            - 'photometry_params': Dict - current dF/F parameters
            - 'photometry_npz_path': Path - source file
            - 'dff_channel_name': str - name of the dF/F channel
        """
        return getattr(self, '_result_data', None)

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

    # =========================================================================
    # Recent Paths Management
    # =========================================================================

    def _get_recent_files(self) -> List[str]:
        """Get list of recent files from settings."""
        recent = self._settings.value(self.SETTINGS_KEY_RECENT_FILES, [])
        if isinstance(recent, str):
            recent = [recent] if recent else []
        # Filter out non-existent files
        return [f for f in recent if Path(f).exists()]

    def _get_recent_folders(self) -> List[str]:
        """Get list of recent folders from settings."""
        recent = self._settings.value(self.SETTINGS_KEY_RECENT_FOLDERS, [])
        if isinstance(recent, str):
            recent = [recent] if recent else []
        # Filter out non-existent folders
        return [f for f in recent if Path(f).exists()]

    def _get_pinned_paths(self) -> List[str]:
        """Get list of pinned paths from settings."""
        pinned = self._settings.value(self.SETTINGS_KEY_PINNED_PATHS, [])
        if isinstance(pinned, str):
            pinned = [pinned] if pinned else []
        return pinned

    def _add_recent_file(self, file_path: str):
        """Add a file to the recent files list."""
        recent = self._get_recent_files()
        # Remove if already exists (will re-add at top)
        if file_path in recent:
            recent.remove(file_path)
        # Add at top
        recent.insert(0, file_path)
        # Trim to max
        recent = recent[:self.MAX_RECENT_ITEMS]
        self._settings.setValue(self.SETTINGS_KEY_RECENT_FILES, recent)

    def _add_recent_folder(self, folder_path: str):
        """Add a folder to the recent folders list."""
        recent = self._get_recent_folders()
        # Remove if already exists (will re-add at top)
        if folder_path in recent:
            recent.remove(folder_path)
        # Add at top
        recent.insert(0, folder_path)
        # Trim to max
        recent = recent[:self.MAX_RECENT_ITEMS]
        self._settings.setValue(self.SETTINGS_KEY_RECENT_FOLDERS, recent)

    def _toggle_pinned_path(self, path: str):
        """Toggle a path as pinned/unpinned."""
        pinned = self._get_pinned_paths()
        if path in pinned:
            pinned.remove(path)
        else:
            pinned.append(path)
        self._settings.setValue(self.SETTINGS_KEY_PINNED_PATHS, pinned)

    def _is_path_pinned(self, path: str) -> bool:
        """Check if a path is pinned."""
        return path in self._get_pinned_paths()

    def _build_browse_menu(self, file_type: str, menu: QMenu):
        """Build the dropdown menu for browse button with recent/pinned paths."""
        menu.clear()

        recent_files = self._get_recent_files()
        recent_folders = self._get_recent_folders()
        pinned_paths = self._get_pinned_paths()

        # Pinned locations (if any)
        if pinned_paths:
            menu.addSection("Pinned Locations")
            for path in pinned_paths:
                p = Path(path)
                if p.exists():
                    if p.is_file():
                        # Pinned file - direct load
                        display_name = f"{p.name}"
                        parent_short = str(p.parent)[-40:] if len(str(p.parent)) > 40 else str(p.parent)
                        action = menu.addAction(f"📄 {display_name}")
                        action.setToolTip(str(p))
                        action.triggered.connect(lambda checked, fp=str(p), ft=file_type: self._load_recent_file(ft, fp))
                    else:
                        # Pinned folder - browse from here
                        display_name = p.name or str(p)
                        action = menu.addAction(f"📁 {display_name}")
                        action.setToolTip(str(p))
                        action.triggered.connect(lambda checked, fp=str(p), ft=file_type: self._browse_from_folder(ft, fp))

        # Recent files
        if recent_files:
            menu.addSection("Recent Files")
            for file_path in recent_files[:5]:  # Show top 5
                p = Path(file_path)
                display_name = p.name
                # Show parent folder hint
                parent_name = p.parent.name
                action = menu.addAction(f"📄 {display_name}")
                action.setToolTip(f"{file_path}\n\nClick to load directly\nRight-click to pin")
                action.triggered.connect(lambda checked, fp=file_path, ft=file_type: self._load_recent_file(ft, fp))

        # Recent folders
        if recent_folders:
            menu.addSection("Browse in Recent Folders")
            for folder_path in recent_folders[:5]:  # Show top 5
                p = Path(folder_path)
                display_name = p.name or str(p)
                # Shorten long paths
                if len(display_name) > 35:
                    display_name = "..." + display_name[-32:]
                action = menu.addAction(f"📁 {display_name}")
                action.setToolTip(f"{folder_path}\n\nClick to open browser here")
                action.triggered.connect(lambda checked, fp=folder_path, ft=file_type: self._browse_from_folder(ft, fp))

        # Separator and pin current option
        if self.file_paths.get(file_type):
            menu.addSeparator()
            current_path = str(self.file_paths[file_type])
            current_folder = str(self.file_paths[file_type].parent)

            if self._is_path_pinned(current_folder):
                action = menu.addAction("📌 Unpin current folder")
                action.triggered.connect(lambda: self._toggle_pinned_path(current_folder))
            else:
                action = menu.addAction("📌 Pin current folder")
                action.triggered.connect(lambda: self._toggle_pinned_path(current_folder))

        # If menu is empty, add a note
        if menu.isEmpty():
            action = menu.addAction("No recent paths")
            action.setEnabled(False)

    def _load_recent_file(self, file_type: str, file_path: str):
        """Load a file directly from the recent files menu."""
        path = Path(file_path)
        if not path.exists():
            QMessageBox.warning(self, "File Not Found", f"The file no longer exists:\n{file_path}")
            return

        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            self.file_paths[file_type] = path
            self._update_file_edits()

            # If FP data selected, auto-detect companion
            if file_type == 'fp_data':
                companions = photometry.find_companion_files(path)
                if companions.get('ai_data') and not self.file_paths['ai_data']:
                    self.file_paths['ai_data'] = companions['ai_data']
                    self._update_file_edits()

            # Reload data
            self._load_and_preview_data()

            # Update recent lists
            self._add_recent_file(str(path))
            self._add_recent_folder(str(path.parent))
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _browse_from_folder(self, file_type: str, folder_path: str):
        """Open the file browser starting at a specific folder."""
        folder = Path(folder_path)
        if not folder.exists():
            QMessageBox.warning(self, "Folder Not Found", f"The folder no longer exists:\n{folder_path}")
            return

        self._browse_file(file_type, start_dir=str(folder))

    def _browse_file(self, file_type: str, start_dir: str = None):
        """Open file browser for selecting a file.

        Args:
            file_type: 'fp_data', 'ai_data', 'timestamps', or 'notes'
            start_dir: Optional starting directory. If None, uses current file's parent
                       or last used folder.
        """
        # Show wait cursor
        self.setCursor(Qt.CursorShape.WaitCursor)

        try:
            # Determine starting directory (use provided start_dir if given)
            if start_dir is None:
                start_dir = ""
                if self.file_paths['fp_data']:
                    start_dir = str(self.file_paths['fp_data'].parent)
                    # If FP data is in subfolder, go up to parent
                    if self.file_paths['fp_data'].parent.name.lower().startswith('fp_data'):
                        start_dir = str(self.file_paths['fp_data'].parent.parent)
                elif self._get_recent_folders():
                    # Fall back to most recent folder
                    start_dir = self._get_recent_folders()[0]

            # File filter based on type
            if file_type == 'notes':
                filter_str = "Notes Files (*.txt *.csv *.xlsx *.xls *.docx);;All Files (*)"
            elif file_type == 'timestamps':
                filter_str = "CSV Files (*.csv);;All Files (*)"
            else:
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

                # If FP data selected, auto-detect companion files
                if file_type == 'fp_data':
                    companions = photometry.find_companion_files(Path(path))
                    if companions.get('ai_data') and not self.file_paths['ai_data']:
                        self.file_paths['ai_data'] = companions['ai_data']
                    if companions.get('timestamps') and not self.file_paths['timestamps']:
                        self.file_paths['timestamps'] = companions['timestamps']
                    self._update_file_edits()

                # Reload data for data files
                if file_type in ('fp_data', 'ai_data', 'timestamps'):
                    self._load_and_preview_data()

                # Update notes info
                if file_type == 'notes':
                    self._update_notes_info()

                # Update recent paths
                self._add_recent_file(path)
                self._add_recent_folder(str(Path(path).parent))
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

        # Timestamps
        if hasattr(self, 'timestamps_edit'):
            if self.file_paths.get('timestamps'):
                self.timestamps_edit.setText(str(self.file_paths['timestamps']))
                self.timestamps_edit.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: #d4d4d4; }")
            else:
                self.timestamps_edit.clear()
                self.timestamps_edit.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: #888888; }")

        # Notes
        if hasattr(self, 'notes_edit'):
            if self.file_paths.get('notes'):
                self.notes_edit.setText(str(self.file_paths['notes']))
                self.btn_notes_view.setEnabled(True)
            else:
                self.notes_edit.clear()
                self.btn_notes_view.setEnabled(False)

    def _update_notes_info(self):
        """Update the notes inline preview with first few lines."""
        if not hasattr(self, 'notes_preview_label'):
            return

        notes_path = self.file_paths.get('notes')
        if notes_path and notes_path.exists():
            try:
                suffix = notes_path.suffix.lower()
                preview_text = ""

                if suffix == '.txt':
                    content = notes_path.read_text(encoding='utf-8', errors='replace')
                    lines = content.strip().split('\n')[:3]
                    preview_text = '\n'.join(lines)
                    if len(content.strip().split('\n')) > 3:
                        preview_text += "\n..."

                elif suffix == '.csv':
                    import pandas as pd
                    df = pd.read_csv(notes_path, nrows=3, encoding='utf-8', on_bad_lines='skip')
                    preview_text = f"Columns: {', '.join(str(c) for c in df.columns[:5])}"
                    if len(df.columns) > 5:
                        preview_text += f" (+{len(df.columns)-5} more)"

                elif suffix in ['.xlsx', '.xls']:
                    import pandas as pd
                    xl = pd.ExcelFile(notes_path)
                    preview_text = f"Sheets: {', '.join(xl.sheet_names[:3])}"
                    if len(xl.sheet_names) > 3:
                        preview_text += f" (+{len(xl.sheet_names)-3} more)"

                elif suffix == '.docx':
                    preview_text = "Word document - click View to preview"

                if preview_text:
                    self.notes_preview_label.setText(preview_text)
                    self.notes_preview_label.setStyleSheet("color: #d4d4d4; font-size: 10px; border: none; padding: 4px; background-color: #1e1e1e; border-radius: 2px;")
                else:
                    self.notes_preview_label.setText("(empty file)")
                    self.notes_preview_label.setStyleSheet("color: #888888; font-size: 10px; border: none; padding: 4px; background-color: #1e1e1e; border-radius: 2px;")

            except Exception as e:
                self.notes_preview_label.setText(f"Error reading: {str(e)[:50]}")
                self.notes_preview_label.setStyleSheet("color: #f44747; font-size: 10px; border: none; padding: 4px; background-color: #1e1e1e; border-radius: 2px;")
        else:
            self.notes_preview_label.setText("No notes file selected")
            self.notes_preview_label.setStyleSheet("color: #888888; font-size: 10px; border: none; padding: 4px; background-color: #1e1e1e; border-radius: 2px;")

    def _update_timestamps_info(self):
        """Update the timestamps info label with file details."""
        if not hasattr(self, 'ts_info_label'):
            return

        ts_path = self.file_paths.get('timestamps')
        if ts_path and ts_path.exists():
            try:
                import pandas as pd
                # Read first and last few rows to show range
                df = pd.read_csv(ts_path, encoding='utf-8', on_bad_lines='skip')
                if len(df) > 0 and len(df.columns) > 0:
                    # Assume first column is timestamps
                    ts_col = df.iloc[:, 0]
                    n_samples = len(ts_col)
                    first_ts = ts_col.iloc[0]
                    last_ts = ts_col.iloc[-1]

                    # Format nicely
                    if isinstance(first_ts, (int, float)) and isinstance(last_ts, (int, float)):
                        duration = last_ts - first_ts
                        if duration > 60:
                            duration_str = f"{duration/60:.1f} min"
                        else:
                            duration_str = f"{duration:.1f} sec"
                        self.ts_info_label.setText(f"{n_samples:,} samples, {duration_str}\nRange: {first_ts:.2f} - {last_ts:.2f}")
                    else:
                        self.ts_info_label.setText(f"{n_samples:,} samples")

                    self.ts_info_label.setStyleSheet("color: #4ec9b0; font-size: 10px; border: none;")
                else:
                    self.ts_info_label.setText("Empty timestamps file")
                    self.ts_info_label.setStyleSheet("color: #888888; font-size: 10px; border: none;")
            except Exception as e:
                self.ts_info_label.setText(f"Error: {str(e)[:40]}")
                self.ts_info_label.setStyleSheet("color: #f44747; font-size: 10px; border: none;")
        else:
            self.ts_info_label.setText("No timestamps loaded")
            self.ts_info_label.setStyleSheet("color: #888888; font-size: 10px; border: none;")

    def _show_notes_popout(self):
        """Show notes file in a popout preview dialog."""
        notes_path = self.file_paths.get('notes')
        if not notes_path or not notes_path.exists():
            QMessageBox.warning(self, "No Notes File", "Please select a notes file first.")
            return

        # Create popout dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Notes: {notes_path.name}")
        dialog.setMinimumSize(600, 400)
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Create preview widget based on file type
        preview_widget = self._create_notes_preview_widget(notes_path)
        layout.addWidget(preview_widget, 1)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.close)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)

        dialog.exec()

    def _create_notes_preview_widget(self, file_path: Path) -> QWidget:
        """Create a preview widget for notes file."""
        from PyQt6.QtWidgets import QTextEdit

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        suffix = file_path.suffix.lower()

        try:
            if suffix in ['.xlsx', '.xls']:
                import pandas as pd
                sheets = pd.read_excel(file_path, sheet_name=None, nrows=500)
                if sheets:
                    sheet_tabs = QTabWidget()
                    for sheet_name, df in sheets.items():
                        table = QTableWidget()
                        table.setRowCount(len(df))
                        table.setColumnCount(len(df.columns))
                        table.setHorizontalHeaderLabels([str(c) for c in df.columns])
                        for i, row in df.iterrows():
                            for j, val in enumerate(row):
                                table.setItem(i, j, QTableWidgetItem(str(val) if pd.notna(val) else ""))
                        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
                        sheet_tabs.addTab(table, sheet_name)
                    layout.addWidget(sheet_tabs)
                else:
                    text_edit = QTextEdit()
                    text_edit.setPlainText("[Workbook is empty]")
                    text_edit.setReadOnly(True)
                    layout.addWidget(text_edit)

            elif suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(file_path, nrows=500, encoding='utf-8', on_bad_lines='skip')
                table = QTableWidget()
                table.setRowCount(len(df))
                table.setColumnCount(len(df.columns))
                table.setHorizontalHeaderLabels([str(c) for c in df.columns])
                for i, row in df.iterrows():
                    for j, val in enumerate(row):
                        table.setItem(i, j, QTableWidgetItem(str(val) if pd.notna(val) else ""))
                table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
                layout.addWidget(table)

            elif suffix == '.txt':
                content = file_path.read_text(encoding='utf-8', errors='replace')
                text_edit = QTextEdit()
                text_edit.setPlainText(content[:50000])  # Limit size
                text_edit.setReadOnly(True)
                text_edit.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #d4d4d4; font-family: monospace; }")
                layout.addWidget(text_edit)

            elif suffix == '.docx':
                try:
                    from docx import Document
                    doc = Document(str(file_path))
                    content_parts = [p.text for p in doc.paragraphs if p.text.strip()]
                    text_edit = QTextEdit()
                    text_edit.setPlainText('\n\n'.join(content_parts))
                    text_edit.setReadOnly(True)
                    text_edit.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #d4d4d4; }")
                    layout.addWidget(text_edit)
                except ImportError:
                    text_edit = QTextEdit()
                    text_edit.setPlainText("python-docx not installed. Install with: pip install python-docx")
                    text_edit.setReadOnly(True)
                    layout.addWidget(text_edit)
            else:
                text_edit = QTextEdit()
                text_edit.setPlainText(f"Unsupported file type: {suffix}")
                text_edit.setReadOnly(True)
                layout.addWidget(text_edit)

        except Exception as e:
            text_edit = QTextEdit()
            text_edit.setPlainText(f"Error reading file: {e}")
            text_edit.setReadOnly(True)
            layout.addWidget(text_edit)

        return container

    def _on_num_experiments_changed(self, value: int):
        """Handle change in number of experiments."""
        self.num_experiments = value
        self._update_channel_table()
        self._update_experiment_preview_tabs()
        self._update_output_file_list()

    def _update_channel_table(self):
        """Update the channel assignment table based on available channels."""
        if not hasattr(self, 'channel_table'):
            return

        # Collect all channels
        channels = []

        # Add FP fiber channels (check checkbox state)
        for fiber_col, info in self.fiber_columns.items():
            checkbox = info.get('checkbox')
            if checkbox and checkbox.isChecked():
                label_edit = info.get('label_edit')
                label_text = label_edit.text() if label_edit else fiber_col
                channels.append({
                    'name': fiber_col,
                    'label': label_text,
                    'source': 'FP',
                    'type': 'Photometry'
                })

        # Add AI channels (check checkbox state)
        for col_idx, info in self.ai_columns.items():
            checkbox = info.get('checkbox')
            if checkbox and checkbox.isChecked():
                label_edit = info.get('label_edit')
                label_text = label_edit.text() if label_edit else f'AI{col_idx}'
                channels.append({
                    'name': f"AI{col_idx}",
                    'label': label_text,
                    'source': 'AI',
                    'type': 'Analog'
                })

        # Update table
        self.channel_table.setRowCount(len(channels))

        for row, ch in enumerate(channels):
            # Channel name
            name_item = QTableWidgetItem(ch['name'])
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channel_table.setItem(row, 0, name_item)

            # Label (editable)
            label_item = QTableWidgetItem(ch['label'])
            self.channel_table.setItem(row, 1, label_item)

            # Assignment dropdown
            combo = QComboBox()
            combo.addItem("All", "all")
            for i in range(1, self.num_experiments + 1):
                combo.addItem(f"Exp {i}", f"exp{i}")
            combo.addItem("Shared", "shared")
            self.channel_table.setCellWidget(row, 2, combo)

            # Type
            type_item = QTableWidgetItem(ch['type'])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channel_table.setItem(row, 3, type_item)

    def _update_experiment_preview_tabs(self):
        """Update experiment preview tabs based on number of experiments."""
        if not hasattr(self, 'preview_tab_widget'):
            return

        # Remove existing experiment tabs (keep "All Channels" at index 0)
        while self.preview_tab_widget.count() > 1:
            widget = self.preview_tab_widget.widget(1)
            self.preview_tab_widget.removeTab(1)
            if widget:
                widget.deleteLater()

        # Clear stored references
        self.experiment_preview_tabs.clear()
        self.experiment_figures.clear()
        self.experiment_canvases.clear()

        # Add tabs for each experiment
        for i in range(1, self.num_experiments + 1):
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            tab_layout.setContentsMargins(0, 0, 0, 0)

            # Create figure and canvas for this experiment
            fig = Figure(figsize=(8, 6), dpi=100, facecolor='#1e1e1e')
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            tab_layout.addWidget(canvas, 1)

            self.preview_tab_widget.addTab(tab, f"Exp {i}")
            self.experiment_preview_tabs[i] = tab
            self.experiment_figures[i] = fig
            self.experiment_canvases[i] = canvas

    def _update_output_file_list(self):
        """Update the output files preview."""
        if not hasattr(self, 'output_files_label') or not hasattr(self, 'output_base_edit'):
            return

        base_name = self.output_base_edit.text().strip()
        if not base_name:
            self.output_files_label.setText("Enter a base name to see output files...")
            self.output_files_label.setStyleSheet("color: #888888; font-size: 10px; padding: 4px;")
            return

        # Generate output file list
        files = []
        if self.num_experiments == 1:
            files.append(f"{base_name}_photometry.npz")
        else:
            for i in range(1, self.num_experiments + 1):
                files.append(f"{base_name}_exp{i}_photometry.npz")

        # Add processing log
        files.append(f"{base_name}_processing_log.txt")

        file_list = "\n".join([f"• {f}" for f in files])
        self.output_files_label.setText(file_list)
        self.output_files_label.setStyleSheet("color: #4ec9b0; font-size: 10px; padding: 4px;")

    def _get_channel_assignments(self) -> Dict[str, Dict]:
        """Get channel assignments from the channel assignment table.

        Returns:
            Dict with structure:
            {
                'exp1': {
                    'fibers': ['G0', 'G1'],  # Fiber columns assigned to exp1
                    'ai': ['Pleth', 'Temp'],  # AI channel labels assigned to exp1
                },
                'exp2': {...},
                'shared': {...},  # Channels marked as shared (included in all experiments)
            }
        """
        if not hasattr(self, 'channel_table'):
            return {}

        assignments = {
            'shared': {'fibers': [], 'ai': []},
        }

        # Initialize experiment slots
        for i in range(1, self.num_experiments + 1):
            assignments[f'exp{i}'] = {'fibers': [], 'ai': []}

        # Read assignments from table
        for row in range(self.channel_table.rowCount()):
            name_item = self.channel_table.item(row, 0)
            label_item = self.channel_table.item(row, 1)
            combo = self.channel_table.cellWidget(row, 2)
            type_item = self.channel_table.item(row, 3)

            if not name_item or not combo or not type_item:
                continue

            channel_name = name_item.text()
            channel_label = label_item.text() if label_item else channel_name
            assignment = combo.currentData()  # 'all', 'exp1', 'exp2', 'shared'
            channel_type = type_item.text()  # 'Photometry' or 'Analog'

            # Determine which category (fibers or ai)
            category = 'fibers' if channel_type == 'Photometry' else 'ai'

            if assignment == 'all':
                # Add to all experiments
                for i in range(1, self.num_experiments + 1):
                    assignments[f'exp{i}'][category].append(channel_name)
            elif assignment == 'shared':
                assignments['shared'][category].append(channel_name)
            else:
                # Specific experiment
                if assignment in assignments:
                    assignments[assignment][category].append(channel_name)

        return assignments

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
            # Handle numpy arrays from NPZ files
            metadata = self._photometry_data.get('metadata', {})
            if hasattr(metadata, 'item'):
                metadata = metadata.item()  # Convert numpy array to dict
            if isinstance(metadata, dict) and metadata.get('source') == 'tab1_raw_data':
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
            print(f"[Photometry] AI path check: {self.file_paths['ai_data']}")
            if self.file_paths['ai_data'] and self.file_paths['ai_data'].exists():
                t0 = time.perf_counter()
                self._show_progress("Loading analog inputs (subsampled)...")
                try:
                    self._ai_data = photometry.load_ai_data_csv(self.file_paths['ai_data'], subsample=AI_SUBSAMPLE)
                    print(f"[Timing] Load AI data ({len(self._ai_data)} rows, {len(self._ai_data.columns)} columns, 1/{AI_SUBSAMPLE} subsampled): {time.perf_counter() - t0:.2f}s")
                    print(f"[Photometry] AI columns: {list(self._ai_data.columns)}")
                    t0 = time.perf_counter()
                    self._populate_ai_column_controls()
                    print(f"[Timing] Populate AI controls: {time.perf_counter() - t0:.2f}s")
                    # Update AI preview table
                    self._update_ai_preview_table()
                except Exception as e:
                    import traceback
                    print(f"[Photometry] Error loading AI data: {e}")
                    traceback.print_exc()
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
                        # Update timestamps display
                        self.timestamps_edit.setText(str(ts_path))
                        self.timestamps_edit.setStyleSheet("""
                            QLineEdit {
                                background-color: #1e1e1e;
                                color: #4ec9b0;
                                border: 1px solid #333333;
                                padding: 2px 4px;
                            }
                        """)
                    except Exception as e:
                        print(f"[Photometry] Error loading timestamps: {e}")
                        self._timestamps = None
                        self.timestamps_edit.setText(f"Error: {e}")
                        self.timestamps_edit.setStyleSheet("""
                            QLineEdit {
                                background-color: #1e1e1e;
                                color: #ff6666;
                                border: 1px solid #333333;
                                padding: 2px 4px;
                            }
                        """)
                else:
                    print(f"[Photometry] No timestamps file found")
                    self._timestamps = None
                    self.timestamps_edit.setText("Not found")
                    self.timestamps_edit.setStyleSheet("""
                        QLineEdit {
                            background-color: #1e1e1e;
                            color: #888888;
                            border: 1px solid #333333;
                            padding: 2px 4px;
                        }
                    """)

            # Stage 5: Update timestamps info
            self._update_timestamps_info()

            # Stage 6: Update channel assignment table
            self._update_channel_table()

            # Stage 7: Generate preview plot
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
            # Note: combo_signal is now hidden/legacy - fiber selection uses checkboxes

        self.combo_time.blockSignals(False)
        self.combo_led_state.blockSignals(False)

        # Populate fiber column checkboxes (new multi-fiber support)
        self._populate_fiber_column_controls()

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

    def _populate_fiber_column_controls(self):
        """Create checkbox + label controls for fiber signal columns (horizontal layout)."""
        # Clear existing controls
        while self.fiber_columns_layout.count():
            item = self.fiber_columns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.fiber_columns.clear()

        if self._fp_data is None:
            return

        # Detect fiber columns using the photometry module
        # Need to load with original headers to detect G0, G1, etc.
        # Also build mapping from original names to internal col# names
        fiber_cols = []
        fiber_col_mapping = {}  # original_name -> internal col# name

        try:
            fp_path = self.file_paths['fp_data']
            if fp_path and fp_path.exists():
                # Read with original headers
                df_with_headers = pd.read_csv(fp_path, nrows=10)
                fiber_cols = photometry.detect_fiber_columns(df_with_headers)

                # Build mapping from original column name to internal col# name
                original_columns = list(df_with_headers.columns)
                internal_columns = list(self._fp_data.columns)

                for orig_col in fiber_cols:
                    if orig_col in original_columns:
                        idx = original_columns.index(orig_col)
                        if idx < len(internal_columns):
                            fiber_col_mapping[orig_col] = internal_columns[idx]
        except Exception as e:
            print(f"[Photometry] Error detecting fiber columns: {e}")
            fiber_cols = []

        # If no fiber columns detected, fall back to col5 (legacy behavior)
        if not fiber_cols:
            cols = list(self._fp_data.columns)
            if len(cols) >= 5:
                fiber_cols = [cols[4]]
                fiber_col_mapping[cols[4]] = cols[4]

        print(f"[Photometry] Detected fiber columns: {fiber_cols}")
        print(f"[Photometry] Fiber column mapping: {fiber_col_mapping}")

        # Add each fiber as checkbox + label edit (horizontal)
        for i, col in enumerate(fiber_cols):
            checkbox = QCheckBox(f"{col}:")
            checkbox.setChecked(True)  # Enable all fibers by default
            self.fiber_columns_layout.addWidget(checkbox)

            label_edit = QLineEdit()
            label_edit.setPlaceholderText("Label")
            # Default label based on column name
            if col.upper().startswith('G'):
                label_edit.setText(f"Region {col[1:]}" if len(col) > 1 else "GCaMP")
            elif col.upper().startswith('R'):
                label_edit.setText(f"Red {col[1:]}" if len(col) > 1 else "Red")
            label_edit.setFixedWidth(70)
            self.fiber_columns_layout.addWidget(label_edit)

            # Store references (include internal column name for data access)
            internal_col = fiber_col_mapping.get(col, col)
            self.fiber_columns[col] = {
                'checkbox': checkbox,
                'label_edit': label_edit,
                'column': col,  # Original column name (for display)
                'internal_column': internal_col  # Internal col# name (for data access)
            }

            # Connect signals
            checkbox.stateChanged.connect(self._on_fiber_column_changed)
            label_edit.textChanged.connect(self._on_fiber_column_changed)

        # Add stretch at end
        self.fiber_columns_layout.addStretch()

        # Also update the legacy combo_signal for backward compatibility
        if fiber_cols:
            self.combo_signal.clear()
            for col in fiber_cols:
                self.combo_signal.addItem(col, col)

    def _on_fiber_column_changed(self):
        """Handle fiber column checkbox or label change."""
        self._update_channel_table()
        self._update_preview_plot()

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
        print(f"[Photometry] AI preview: {len(col_names)} columns, {len(rows)} rows")
        print(f"[Photometry] AI preview columns: {col_names}")
        if not rows:
            print(f"[Photometry] AI preview: no rows to display!")
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
        self._update_channel_table()
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
        Compute dF/F using specified method.

        Processing pipeline:
        1. Interpolate both channels to common time base
        2. (Optional) Exclude initial transient period
        3. (Optional) Low-pass filter both signals
        4. Compute dF/F using selected method
        5. (Optional) Detrend dF/F to remove drift

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
            'dff_raw': None,     # dF/F before detrending
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

        # Compute dF/F based on method
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
            print(f"[Photometry] dF/F fitting: slope={slope:.4f}, intercept={intercept:.4f}, R²={r_squared:.4f}")
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

            # dF/F = (GCaMP - fitted_iso) / fitted_iso
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

            # dF/F = (GCaMP_norm - Iso_norm) / Iso_norm * 100
            epsilon = np.abs(iso_norm).mean() * 1e-6
            dff = (gcamp_norm - iso_norm) / (iso_norm + epsilon) * 100
            print(f"[Photometry] Simple subtraction: iso_mean={iso_mean:.2f}, gcamp_mean={gcamp_mean:.2f}")

        # Store raw dF/F before detrending
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
                    print(f"[Photometry] No exponential decay detected (delta={a0:.3f}%), skipping")
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
                    print(f"[Photometry] No decay detected (delta={total_decay:.3f}%), skipping biexp")
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

    def _draw_raw_signals_preview(self, fiber_data: dict):
        """Draw simple raw signals preview for Tab 1 (Data Assembly) - no dF/F computation.

        Args:
            fiber_data: Dict mapping fiber_col -> {'iso_time', 'iso', 'gcamp_time', 'gcamp', 'label'}
        """
        import time
        from PyQt6.QtWidgets import QApplication

        t0 = time.perf_counter()
        self.loading_label.setText("Drawing raw signals preview...")
        QApplication.processEvents()

        # Count AI channels to include
        ai_plots = []
        if self._ai_data is not None:
            for i, col_info in self.ai_columns.items():
                if col_info['checkbox'].isChecked():
                    label = col_info['label_edit'].text() or f"AI {col_info['column']}"
                    ai_plots.append((col_info['column'], label))

        # Calculate number of panels: 1 dual-axis panel per fiber + AI channels
        n_fibers = len(fiber_data)
        n_plots = n_fibers + len(ai_plots)

        if n_plots == 0:
            self.loading_label.setVisible(False)
            return

        # Get time range across all fibers
        all_times = []
        for fiber_col, fdata in fiber_data.items():
            if len(fdata['iso_time']) > 0:
                all_times.append(fdata['iso_time'])
            if len(fdata['gcamp_time']) > 0:
                all_times.append(fdata['gcamp_time'])
        if all_times:
            all_times = np.concatenate(all_times)
            t_start = 0.0
            t_end = float(np.max(all_times))
        else:
            t_start = 0.0
            t_end = 60.0

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

        ax_idx = 0

        # Plot each fiber as dual-axis panel (Iso + GCaMP overlaid)
        for fiber_col, fdata in fiber_data.items():
            label = fdata.get('label', fiber_col)
            iso_time = fdata['iso_time']
            iso_signal = fdata['iso']
            gcamp_time = fdata['gcamp_time']
            gcamp_signal = fdata['gcamp']

            ax = axes[ax_idx]

            # GCaMP on primary axis (green)
            if len(gcamp_time) > 0:
                mask = (gcamp_time >= t_start) & (gcamp_time <= t_end)
                t_plot, s_plot = self._subsample_for_preview(gcamp_time[mask], gcamp_signal[mask])
                ax.plot(t_plot, s_plot, color='#00cc00', linewidth=0.5, alpha=0.8, label='GCaMP')
                ax.set_ylabel(f'GCaMP ({label})', fontsize=8, color='#00cc00')
                ax.tick_params(axis='y', labelcolor='#00cc00')

            # Iso on secondary axis (blue)
            ax_iso = ax.twinx()
            if len(iso_time) > 0:
                mask = (iso_time >= t_start) & (iso_time <= t_end)
                t_plot, s_plot = self._subsample_for_preview(iso_time[mask], iso_signal[mask])
                ax_iso.plot(t_plot, s_plot, color='#5555ff', linewidth=0.5, alpha=0.8, label='Iso')
                ax_iso.set_ylabel('Iso', fontsize=8, color='#5555ff')
                ax_iso.tick_params(axis='y', labelcolor='#5555ff')
                ax_iso.set_facecolor('#252526')
                for spine in ax_iso.spines.values():
                    spine.set_color('#3e3e42')

            # Title with fiber label
            ax.set_title(f'Raw: {label} ({fiber_col})', fontsize=9, color='#cccccc', loc='left')
            ax_idx += 1

        # Plot AI channels (using timestamps for time axis, same as Tab 2)
        if self._ai_data is not None and self._timestamps is not None and len(ai_plots) > 0:
            ai_colors = ['#ff9900', '#00cccc', '#ff66ff', '#ffff00', '#ff6666']

            # Use timestamps for AI time axis (convert to minutes, normalize if checkbox checked)
            if self.chk_normalize_time.isChecked():
                ai_t_min = np.min(self._timestamps) if len(self._timestamps) > 0 else 0
                ai_time = (self._timestamps - ai_t_min) / 60000
            else:
                ai_time = self._timestamps / 60000

            for idx, (col, label) in enumerate(ai_plots):
                color = ai_colors[idx % len(ai_colors)]
                ai_signal = self._ai_data[col].values

                # Handle length mismatch between timestamps and signal
                min_len = min(len(ai_time), len(ai_signal))
                t_arr = ai_time[:min_len]
                s_arr = ai_signal[:min_len]

                # Time mask for AI data
                ai_mask = (t_arr >= t_start) & (t_arr <= t_end)
                t_plot, s_plot = self._subsample_for_preview(t_arr[ai_mask], s_arr[ai_mask])

                axes[ax_idx].plot(t_plot, s_plot, color=color, linewidth=0.5, alpha=0.8)
                axes[ax_idx].set_ylabel(label, fontsize=8, color=color)
                axes[ax_idx].tick_params(axis='y', labelcolor=color)
                axes[ax_idx].set_title(label, fontsize=9, color='#cccccc', loc='left')
                ax_idx += 1

        # Set x-label on bottom axis
        axes[-1].set_xlabel('Time (minutes)', fontsize=8)

        # Adjust layout
        self.preview_figure.tight_layout()
        self.preview_canvas.draw()

        self.loading_label.setVisible(False)
        print(f"[Timing] Raw signals preview: {time.perf_counter() - t0:.3f}s")

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

        # Get enabled fiber columns - map original names to internal column names
        enabled_fibers = []  # List of (original_name, internal_name) tuples
        for fiber_col, fiber_info in self.fiber_columns.items():
            if fiber_info['checkbox'].isChecked():
                internal_col = fiber_info.get('internal_column', fiber_col)
                enabled_fibers.append((fiber_col, internal_col))

        # Fall back to legacy signal_col if no fiber columns available
        if not enabled_fibers:
            signal_col = self.combo_signal.currentData()
            if signal_col:
                enabled_fibers = [(signal_col, signal_col)]

        if not all([time_col, led_col]) or not enabled_fibers:
            self.loading_label.setVisible(False)
            self.preview_canvas.draw()
            return

        try:
            # Separate isosbestic (LED=1) and GCaMP (LED=2) for each fiber
            t0 = time.perf_counter()
            # Skip first row if it's header
            data = self._fp_data.copy()
            # Extract internal column names for data access
            internal_fiber_cols = [internal for _, internal in enabled_fibers]
            if not np.issubdtype(data[led_col].dtype, np.number):
                data = data.iloc[1:].copy()
                cols_to_convert = [time_col, led_col] + internal_fiber_cols
                for col in cols_to_convert:
                    if col in data.columns:
                        data[col] = pd.to_numeric(data[col], errors='coerce')

            iso_mask = data[led_col] == 1
            gcamp_mask = data[led_col] == 2

            # Get raw time values in ms (shared across all fibers)
            iso_time_raw = data.loc[iso_mask, time_col].values
            gcamp_time_raw = data.loc[gcamp_mask, time_col].values
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

            # Build fiber data dict for multi-fiber preview
            fiber_data = {}
            for original_col, internal_col in enabled_fibers:
                # Get label from UI if available
                if original_col in self.fiber_columns:
                    label = self.fiber_columns[original_col]['label_edit'].text() or original_col
                else:
                    label = original_col

                # Extract signal for this fiber using internal column name
                iso_signal = data.loc[iso_mask, internal_col].values
                gcamp_signal = data.loc[gcamp_mask, internal_col].values

                fiber_data[original_col] = {
                    'iso_time': iso_time,
                    'iso': iso_signal,
                    'gcamp_time': gcamp_time,
                    'gcamp': gcamp_signal,
                    'label': label
                }

            print(f"[Photometry] Preview: {len(fiber_data)} fiber(s): {list(fiber_data.keys())}")

            # Check if we're on Tab 0 (Data Assembly) - only show raw signals, no dF/F
            is_tab1_data_assembly = self.tab_widget.currentIndex() == 0

            if is_tab1_data_assembly:
                # Tab 1 (Data Assembly): Show ONLY raw signals - no dF/F computation
                self._draw_raw_signals_preview(fiber_data)
                return

            # Tab 2 (Processing): Compute dF/F and show intermediate steps
            # Note: Tab 2 typically uses _update_tab2_preview() instead
            # This is fallback code using first fiber from fiber_data
            t0 = time.perf_counter()
            self.loading_label.setText("Computing dF/F...")
            QApplication.processEvents()

            # Use first fiber for legacy Tab 2 code path
            first_fiber_key = list(fiber_data.keys())[0]
            first_fiber = fiber_data[first_fiber_key]
            iso_signal = first_fiber['iso']
            gcamp_signal = first_fiber['gcamp']

            # Get processing options from UI controls (use defaults if Tab 2 widgets not initialized)
            if hasattr(self, 'combo_dff_method') and self.combo_dff_method is not None:
                dff_method = self.combo_dff_method.currentData()
                detrend_method = self.combo_detrend_method.currentData()
                exclude_start = self.spin_exclude_start.value()
                lowpass_hz = self.spin_lowpass.value() if self.chk_lowpass.isChecked() else None
                detrend_fit_start = self.spin_detrend_start.value()
                detrend_fit_end = self.spin_detrend_end.value()
                show_intermediates = self.chk_show_intermediates.isChecked()
            else:
                # Default values for Tab 1 preview (Tab 2 widgets not yet created)
                dff_method = 'fitted'
                detrend_method = 'exponential'
                exclude_start = 0.0
                lowpass_hz = None
                detrend_fit_start = 5.0
                detrend_fit_end = 30.0
                show_intermediates = False

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
            print(f"[Timing]   dF/F computation: {time.perf_counter() - t0:.3f}s")

            # Count plots needed
            # Order: Iso, GCaMP, [intermediates if enabled], final dF/F, AI channels
            ai_plots = []
            if self._ai_data is not None:
                for i, col_info in self.ai_columns.items():
                    if col_info['checkbox'].isChecked():
                        label = col_info['label_edit'].text() or f"AI {col_info['column']}"
                        ai_plots.append((col_info['column'], label))

            # Calculate number of panels
            # Always show: Raw aligned signals (dual y-axis) + final dF/F + AI
            # Optional intermediates: Fitted iso vs GCaMP, Raw dF/F with detrend curve
            n_intermediate_plots = 0
            if show_intermediates and intermediates is not None:
                if dff_method == 'fitted':
                    n_intermediate_plots += 1  # Fitted iso vs GCaMP (same scale)
                if detrend_method != 'none':
                    n_intermediate_plots += 1  # Raw dF/F with detrend curve

            # Total: Raw aligned + intermediates + final dF/F + AI
            n_plots = 1 + n_intermediate_plots + 1 + len(ai_plots)

            # Update time range spinboxes (only if Tab 2 widgets exist)
            all_times = np.concatenate([iso_time, gcamp_time]) if len(iso_time) > 0 and len(gcamp_time) > 0 else iso_time
            max_time = float(np.max(all_times)) if len(all_times) > 0 else 60.0

            if hasattr(self, 'spin_time_start') and self.spin_time_start is not None:
                self.spin_time_start.setRange(0, max_time)
                self.spin_time_end.setRange(0, max_time)
                if self.spin_time_end.value() == 0:
                    self.spin_time_end.setValue(max_time)

            # Get time range and fit range for plotting (use defaults if Tab 2 widgets not initialized)
            if hasattr(self, 'spin_time_start') and self.spin_time_start is not None:
                t_start = self.spin_time_start.value()
                t_end = self.spin_time_end.value()
                fit_start = self.spin_detrend_start.value()
                fit_end = self.spin_detrend_end.value()
            else:
                # Default values for Tab 1 preview
                t_start = 0.0
                t_end = max_time
                fit_start = detrend_fit_start
                fit_end = detrend_fit_end

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

            # Intermediate plots (before final dF/F)
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

                # Intermediate: Raw dF/F with detrend curve
                if detrend_method != 'none' and intermediates.get('dff_raw') is not None:
                    int_ax3 = axes[ax_idx]
                    dff_raw = intermediates.get('dff_raw')
                    detrend_curve = intermediates.get('detrend_curve')

                    if dff_raw is not None and len(dff_raw) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], dff_raw[int_mask])
                        int_ax3.plot(t_plot, s_plot, color='#888888', linewidth=0.5, alpha=0.7, label='Raw dF/F')

                    if detrend_curve is not None and len(detrend_curve) == len(int_time):
                        t_plot, s_plot = self._subsample_for_preview(int_time[int_mask], detrend_curve[int_mask])
                        detrend_name = fit_params.get('detrend_method', 'trend')
                        int_ax3.plot(t_plot, s_plot, color='#ff5555', linewidth=1.5, alpha=0.9, label=f'{detrend_name}')

                    int_ax3.set_ylabel('Raw dF/F (%)', fontsize=8)
                    int_ax3.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                    int_ax3.legend(loc='upper right', fontsize=7, framealpha=0.7)
                    add_fit_range(int_ax3)
                    photometry_axes.append(int_ax3)
                    ax_idx += 1

                print(f"[Timing]   Plot intermediate panels: {time.perf_counter() - t0_int:.3f}s")

            # Final dF/F plot (after intermediates)
            t0 = time.perf_counter()
            dff_ax = axes[ax_idx]
            if len(dff_time) > 0:
                mask = (dff_time >= t_start) & (dff_time <= t_end)
                t_plot, s_plot = self._subsample_for_preview(dff_time[mask], dff_signal[mask])
                dff_ax.plot(t_plot, s_plot, color='#ff9900', linewidth=0.5, alpha=0.8)
                dff_ax.set_ylabel('dF/F (%)', fontsize=8)
                dff_ax.axhline(y=0, color='#666666', linewidth=0.5, linestyle='--')
                add_fit_range(dff_ax)
                photometry_axes.append(dff_ax)

                # Force y-axis autoscale to fix flat line issue
                dff_ax.relim()
                dff_ax.autoscale_view(scaley=True)
            ax_idx += 1
            print(f"[Timing]   Plot final dF/F: {time.perf_counter() - t0:.3f}s")

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
        """Handle save button click - save photometry data to NPZ file(s).

        For multi-experiment recordings, saves separate NPZ files for each experiment.
        Each file contains only channels assigned to that experiment plus shared channels.

        Args:
            load_after: If True, emit signal to load into main app after saving
        """
        from PyQt6.QtWidgets import QFileDialog
        from datetime import datetime

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

        # Get base name from UI or auto-generate
        base_name = self.output_base_edit.text().strip()
        if not base_name:
            # Auto-generate from source file
            fp_path = self.file_paths['fp_data']
            if fp_path.parent.name.lower().startswith('fp_data'):
                base_name = fp_path.parent.parent.name
            else:
                base_name = fp_path.stem
            base_name = base_name.replace(' ', '_')

        # Determine output folder (PhysioMetrics subfolder next to raw data)
        fp_path = self.file_paths['fp_data']
        if fp_path.parent.name.lower().startswith('fp_data'):
            experiment_folder = fp_path.parent.parent
        else:
            experiment_folder = fp_path.parent

        # Default to PhysioMetrics subfolder
        output_folder = experiment_folder / "PhysioMetrics"

        # Ask user to confirm or change output folder
        reply = QMessageBox.question(
            self,
            "Output Folder",
            f"Save to:\n{output_folder}\n\nUse this folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return
        elif reply == QMessageBox.StandardButton.No:
            # Let user choose different folder
            selected_folder = QFileDialog.getExistingDirectory(
                self,
                "Select Output Folder",
                str(experiment_folder),
                QFileDialog.Option.ShowDirsOnly
            )
            if not selected_folder:
                return  # User cancelled
            output_folder = Path(selected_folder)

        # Create the folder if it doesn't exist
        if not output_folder.exists():
            try:
                output_folder.mkdir(parents=True, exist_ok=True)
                print(f"[Photometry] Created output folder: {output_folder}")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Folder Error",
                    f"Could not create output folder:\n{output_folder}\n\nError: {e}"
                )
                return

        # Get channel assignments
        assignments = self._get_channel_assignments()

        # Build list of files to create
        files_to_create = []
        if self.num_experiments == 1:
            files_to_create.append({
                'exp_id': 'exp1',
                'filename': f"{base_name}_photometry.npz",
                'path': output_folder / f"{base_name}_photometry.npz"
            })
        else:
            for i in range(1, self.num_experiments + 1):
                files_to_create.append({
                    'exp_id': f'exp{i}',
                    'filename': f"{base_name}_exp{i}_photometry.npz",
                    'path': output_folder / f"{base_name}_exp{i}_photometry.npz"
                })

        # Processing log path
        log_path = output_folder / f"{base_name}_processing_log.txt"

        # Check for existing files
        existing_files = [f for f in files_to_create if f['path'].exists()]
        if log_path.exists():
            existing_files.append({'filename': log_path.name, 'path': log_path})

        if existing_files:
            file_list = "\n".join([f"  • {f['filename']}" for f in existing_files[:5]])
            if len(existing_files) > 5:
                file_list += f"\n  ... and {len(existing_files) - 5} more"
            reply = QMessageBox.question(
                self,
                "Overwrite Files?",
                f"The following files already exist:\n{file_list}\n\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Save the data
        try:
            saved_paths = []
            sibling_paths = [f['path'] for f in files_to_create]

            for file_info in files_to_create:
                exp_id = file_info['exp_id']
                save_path = file_info['path']

                # Get channels for this experiment (assigned + shared)
                exp_fibers = assignments.get(exp_id, {}).get('fibers', [])
                exp_ai = assignments.get(exp_id, {}).get('ai', [])
                shared_fibers = assignments.get('shared', {}).get('fibers', [])
                shared_ai = assignments.get('shared', {}).get('ai', [])

                # Combine assigned and shared
                all_fibers = list(set(exp_fibers + shared_fibers))
                all_ai = list(set(exp_ai + shared_ai))

                self._save_experiment_npz(
                    save_path=save_path,
                    exp_id=exp_id,
                    fiber_cols=all_fibers,
                    ai_labels=all_ai,
                    shared_fibers=shared_fibers,
                    shared_ai=shared_ai,
                    sibling_paths=[p for p in sibling_paths if p != save_path],
                    assignments=assignments
                )
                saved_paths.append(save_path)

            # Create processing log
            self._create_processing_log(
                log_path=log_path,
                base_name=base_name,
                output_folder=output_folder,
                saved_paths=saved_paths,
                assignments=assignments
            )

            # Show success and ask about next steps
            if len(saved_paths) == 1:
                msg = f"Data saved to:\n{saved_paths[0]}"
            else:
                msg = f"Saved {len(saved_paths)} experiment files to:\n{output_folder}"
                msg += f"\n\nProcessing log: {log_path.name}"

            if load_after:
                # For single experiment, load directly
                if len(saved_paths) == 1:
                    self._saved_path = saved_paths[0]
                    QMessageBox.information(self, "Saved", msg + "\n\nLoading into main app...")
                    self.accept()
                else:
                    # Multiple experiments - ask which to load
                    self._show_load_experiment_dialog(saved_paths, msg)
            else:
                # Ask if user wants to switch to Tab 2 for processing
                reply = QMessageBox.question(
                    self,
                    "Data Saved",
                    msg + "\n\nSwitch to Processing tab to configure dF/F?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                if reply == QMessageBox.StandardButton.Yes:
                    # Load first experiment and switch to Tab 2
                    self._load_photometry_npz(saved_paths[0])
                    self.tab_widget.setCurrentIndex(1)

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save data:\n{str(e)}"
            )

    def _show_load_experiment_dialog(self, saved_paths: List[Path], summary_msg: str):
        """Show dialog to select which experiment to load after multi-experiment save."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Load Experiment")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel(summary_msg))
        layout.addWidget(QLabel("\nWhich experiment would you like to load?"))

        combo = QComboBox()
        for i, path in enumerate(saved_paths, 1):
            combo.addItem(f"Experiment {i}: {path.name}", str(path))
        layout.addWidget(combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_path = Path(combo.currentData())
            self._saved_path = selected_path
            self.accept()

    def _save_experiment_npz(
        self,
        save_path: Path,
        exp_id: str,
        fiber_cols: List[str],
        ai_labels: List[str],
        shared_fibers: List[str],
        shared_ai: List[str],
        sibling_paths: List[Path],
        assignments: Dict
    ):
        """Save data for a single experiment to NPZ file.

        Args:
            save_path: Path to save the NPZ file
            exp_id: Experiment identifier (e.g., 'exp1')
            fiber_cols: List of fiber column names to include (e.g., ['G0', 'G1'])
            ai_labels: List of AI channel names to include (e.g., ['AI0', 'AI1'])
            shared_fibers: List of fiber columns marked as shared
            shared_ai: List of AI channels marked as shared
            sibling_paths: Paths to other experiment NPZ files from same recording
            assignments: Full channel assignments dict for metadata
        """
        import time as time_module
        from datetime import datetime

        print(f"[Photometry] Saving experiment {exp_id} to {save_path}")
        t0 = time_module.perf_counter()

        # Get column mappings
        time_col = self.combo_time.currentData()
        led_col = self.combo_led_state.currentData()

        # Get fiber labels from UI
        fiber_labels = {}
        for fiber_col, fiber_info in self.fiber_columns.items():
            if fiber_col in fiber_cols:
                label = fiber_info['label_edit'].text() or fiber_col
                fiber_labels[fiber_col] = label

        if not fiber_cols:
            print(f"[Photometry] Warning: No fiber columns assigned to {exp_id}")

        # Load FP data
        fp_path = self.file_paths['fp_data']
        data_with_headers = pd.read_csv(fp_path)

        # Map column names
        if time_col.startswith('col'):
            time_col_idx = int(time_col.replace('col', '')) - 1
            time_col_actual = data_with_headers.columns[time_col_idx]
        else:
            time_col_actual = time_col

        if led_col.startswith('col'):
            led_col_idx = int(led_col.replace('col', '')) - 1
            led_col_actual = data_with_headers.columns[led_col_idx]
        else:
            led_col_actual = led_col

        # Skip header row if first row contains non-numeric data
        first_led = data_with_headers[led_col_actual].iloc[0]
        if isinstance(first_led, str) or (isinstance(first_led, float) and np.isnan(first_led)):
            data_with_headers = data_with_headers.iloc[1:].copy()
            for col in data_with_headers.columns:
                data_with_headers[col] = pd.to_numeric(data_with_headers[col], errors='coerce')

        # Create LED masks
        iso_mask = data_with_headers[led_col_actual] == 1
        gcamp_mask = data_with_headers[led_col_actual] == 2

        # Get time arrays
        iso_time_raw = data_with_headers.loc[iso_mask, time_col_actual].values
        gcamp_time_raw = data_with_headers.loc[gcamp_mask, time_col_actual].values

        # Find global t_min for normalization
        all_times = [iso_time_raw, gcamp_time_raw]
        if self._timestamps is not None:
            all_times.append(self._timestamps)
        global_t_min = min(np.min(t) for t in all_times if len(t) > 0)

        # Normalize FP time to minutes, starting at 0
        fp_time = (iso_time_raw - global_t_min) / 60000

        # Estimate FP sample rate
        if len(fp_time) > 1:
            fp_dt = np.median(np.diff(fp_time)) * 60
            fp_sample_rate = 1.0 / fp_dt if fp_dt > 0 else 0
        else:
            fp_sample_rate = 0

        # Build per-fiber data (only for assigned fibers)
        fibers_data = {}
        for fiber_col in fiber_cols:
            if fiber_col in data_with_headers.columns:
                iso_signal = data_with_headers.loc[iso_mask, fiber_col].values
                gcamp_signal = data_with_headers.loc[gcamp_mask, fiber_col].values

                fibers_data[fiber_col] = {
                    'iso': iso_signal,
                    'gcamp': gcamp_signal,
                    'label': fiber_labels.get(fiber_col, fiber_col),
                    'is_shared': fiber_col in shared_fibers,
                }
                print(f"[Photometry]   Fiber {fiber_col}: {len(iso_signal)} iso, {len(gcamp_signal)} gcamp samples")

        # Build save dict
        save_dict = {
            'source_fp_path': str(self.file_paths['fp_data']),
            'source_ai_path': str(self.file_paths['ai_data']) if self.file_paths['ai_data'] else '',
            'source_timestamps_path': str(self.file_paths.get('timestamps', '')) if self.file_paths.get('timestamps') else '',
            'source_notes_path': str(self.file_paths.get('notes', '')) if self.file_paths.get('notes') else '',
            'fp_time': fp_time,
            'fibers': fibers_data,
        }

        # Legacy fields for backward compatibility (use first fiber if available)
        if fibers_data:
            first_fiber = list(fibers_data.keys())[0]
            save_dict['iso'] = fibers_data[first_fiber]['iso']
            save_dict['gcamp'] = fibers_data[first_fiber]['gcamp']

        # Add AI data if available
        ai_sample_rate = 0
        ai_channels = {}
        if self._ai_data is not None and self.file_paths['ai_data'] and self.file_paths['ai_data'].exists():
            # Reload AI data at full resolution
            ai_data_full = photometry.load_ai_data_csv(self.file_paths['ai_data'], subsample=1)

            # Load timestamps
            ts_path = photometry.find_timestamps_file(self.file_paths['fp_data'])
            if ts_path and ts_path.exists():
                timestamps_full = photometry.load_timestamps_csv(ts_path, subsample=1)
                ai_t_min = np.min(timestamps_full)
                actual_global_t_min = min(global_t_min, ai_t_min)
                ai_time = (timestamps_full - actual_global_t_min) / 60000

                if len(ai_time) > 1:
                    ai_dt = np.median(np.diff(ai_time)) * 60
                    ai_sample_rate = 1.0 / ai_dt if ai_dt > 0 else 0

                save_dict['ai_time'] = ai_time

                # Get assigned AI channels
                for i, col_info in self.ai_columns.items():
                    col_name = col_info['column']
                    label = col_info['label_edit'].text() or f"ai_{col_name}"
                    label_clean = label.replace(' ', '_')

                    # Check if this AI channel is assigned to this experiment
                    ai_key = f"AI{i}"
                    if ai_key in ai_labels:
                        ai_channels[label_clean] = {
                            'data': ai_data_full[col_name].values,
                            'is_shared': ai_key in shared_ai,
                        }
                        print(f"[Photometry]   AI {label_clean}: {len(ai_data_full)} samples")

                save_dict['ai_channels'] = {k: v['data'] for k, v in ai_channels.items()}

        # Build enhanced metadata (v3 format)
        save_dict['metadata'] = {
            'format_version': 3,  # v3 = multi-experiment support
            'experiment_id': exp_id,
            'fp_sample_rate': fp_sample_rate,
            'ai_sample_rate': ai_sample_rate,
            'created': datetime.now().isoformat(),
            'fp_time_col': time_col,
            'fp_led_col': led_col,
            'fiber_columns': fiber_cols,
            'fiber_labels': fiber_labels,
            'shared_fibers': shared_fibers,
            'shared_ai': shared_ai,
            'sibling_experiments': [str(p) for p in sibling_paths],
            'channel_assignments': assignments,
            'ai_channel_shared': {k: v['is_shared'] for k, v in ai_channels.items()},
        }

        # Save to NPZ
        np.savez(save_path, **save_dict)

        elapsed = time_module.perf_counter() - t0
        print(f"[Photometry] Experiment {exp_id} saved in {elapsed:.2f}s")

    def _create_processing_log(
        self,
        log_path: Path,
        base_name: str,
        output_folder: Path,
        saved_paths: List[Path],
        assignments: Dict
    ):
        """Create a human-readable processing log file.

        Args:
            log_path: Path to save the log file
            base_name: Base name used for output files
            output_folder: Output folder path
            saved_paths: List of saved NPZ file paths
            assignments: Channel assignments dict
        """
        from datetime import datetime

        lines = []
        lines.append("=" * 70)
        lines.append("PHOTOMETRY DATA PROCESSING LOG")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Base Name: {base_name}")
        lines.append(f"Output Folder: {output_folder}")
        lines.append("")

        # Source files
        lines.append("-" * 40)
        lines.append("SOURCE FILES")
        lines.append("-" * 40)
        lines.append(f"Photometry Data (FP): {self.file_paths['fp_data']}")
        lines.append(f"Analog Inputs (AI):   {self.file_paths.get('ai_data', 'Not specified')}")
        lines.append(f"Timestamps:           {self.file_paths.get('timestamps', 'Auto-detected')}")
        lines.append(f"Notes:                {self.file_paths.get('notes', 'Not specified')}")
        lines.append("")

        # Column mappings
        lines.append("-" * 40)
        lines.append("COLUMN MAPPINGS")
        lines.append("-" * 40)
        lines.append(f"Time Column:      {self.combo_time.currentData()}")
        lines.append(f"LED State Column: {self.combo_led_state.currentData()}")
        lines.append("")

        # Fiber columns
        lines.append("Fiber Columns:")
        for fiber_col, fiber_info in self.fiber_columns.items():
            if fiber_info.get('checkbox') and fiber_info['checkbox'].isChecked():
                label = fiber_info['label_edit'].text() or fiber_col
                lines.append(f"  {fiber_col}: {label}")
        lines.append("")

        # AI channels
        if self.ai_columns:
            lines.append("AI Channels:")
            for i, col_info in self.ai_columns.items():
                if col_info.get('checkbox') and col_info['checkbox'].isChecked():
                    label = col_info['label_edit'].text() or f"AI{i}"
                    lines.append(f"  AI{i} ({col_info['column']}): {label}")
            lines.append("")

        # Experiment assignments
        lines.append("-" * 40)
        lines.append("EXPERIMENT ASSIGNMENTS")
        lines.append("-" * 40)

        for exp_key in sorted(assignments.keys()):
            if exp_key == 'shared':
                lines.append("Shared Channels (included in all experiments):")
            else:
                lines.append(f"{exp_key.upper()}:")

            exp_data = assignments[exp_key]
            if exp_data.get('fibers'):
                lines.append(f"  Fibers: {', '.join(exp_data['fibers'])}")
            if exp_data.get('ai'):
                lines.append(f"  AI: {', '.join(exp_data['ai'])}")
            if not exp_data.get('fibers') and not exp_data.get('ai'):
                lines.append("  (no channels assigned)")
            lines.append("")

        # Output files
        lines.append("-" * 40)
        lines.append("OUTPUT FILES")
        lines.append("-" * 40)
        for path in saved_paths:
            lines.append(f"  {path.name}")
        lines.append(f"  {log_path.name}")
        lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append("END OF LOG")
        lines.append("=" * 70)

        # Write to file
        with open(log_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"[Photometry] Processing log saved to {log_path}")

    def _save_photometry_data(self, save_path: Path):
        """Save raw aligned photometry data to NPZ file.

        This saves the raw signals (iso, gcamp, AI channels) with aligned time vectors.
        NO dF/F is computed - that will be done when loading into main app based on
        session settings.

        Format v2: Multi-fiber support - saves 'fibers' dict with per-fiber data.
        """
        import time as time_module
        from datetime import datetime

        print(f"[Photometry] Saving data to {save_path}")
        t0 = time_module.perf_counter()

        # Get column mappings
        time_col = self.combo_time.currentData()
        led_col = self.combo_led_state.currentData()

        # Get enabled fiber columns
        enabled_fibers = []
        fiber_labels = {}
        for fiber_col, fiber_info in self.fiber_columns.items():
            if fiber_info['checkbox'].isChecked():
                enabled_fibers.append(fiber_col)
                label = fiber_info['label_edit'].text() or fiber_col
                fiber_labels[fiber_col] = label

        if not enabled_fibers:
            raise ValueError("No fiber columns selected")

        print(f"[Photometry] Saving {len(enabled_fibers)} fiber(s): {enabled_fibers}")

        # Need to reload FP data with original headers to access fiber columns by name
        fp_path = self.file_paths['fp_data']
        data_with_headers = pd.read_csv(fp_path)

        # Ensure numeric columns
        time_col_name = data_with_headers.columns[int(time_col.replace('col', '')) - 1] if time_col.startswith('col') else time_col
        led_col_name = data_with_headers.columns[int(led_col.replace('col', '')) - 1] if led_col.startswith('col') else led_col

        # Map col names to actual header names if needed
        if time_col.startswith('col'):
            time_col_idx = int(time_col.replace('col', '')) - 1
            time_col_actual = data_with_headers.columns[time_col_idx]
        else:
            time_col_actual = time_col

        if led_col.startswith('col'):
            led_col_idx = int(led_col.replace('col', '')) - 1
            led_col_actual = data_with_headers.columns[led_col_idx]
        else:
            led_col_actual = led_col

        # Skip header row if first row contains non-numeric data
        first_led = data_with_headers[led_col_actual].iloc[0]
        if isinstance(first_led, str) or (isinstance(first_led, float) and np.isnan(first_led)):
            data_with_headers = data_with_headers.iloc[1:].copy()
            for col in data_with_headers.columns:
                data_with_headers[col] = pd.to_numeric(data_with_headers[col], errors='coerce')

        # Create LED masks
        iso_mask = data_with_headers[led_col_actual] == 1
        gcamp_mask = data_with_headers[led_col_actual] == 2

        # Get time arrays
        iso_time_raw = data_with_headers.loc[iso_mask, time_col_actual].values
        gcamp_time_raw = data_with_headers.loc[gcamp_mask, time_col_actual].values

        # Find global t_min for normalization
        all_times = [iso_time_raw, gcamp_time_raw]
        if self._timestamps is not None:
            all_times.append(self._timestamps)
        global_t_min = min(np.min(t) for t in all_times if len(t) > 0)

        # Normalize FP time to minutes, starting at 0
        fp_time = (iso_time_raw - global_t_min) / 60000

        # Estimate FP sample rate
        if len(fp_time) > 1:
            fp_dt = np.median(np.diff(fp_time)) * 60  # Convert back to seconds
            fp_sample_rate = 1.0 / fp_dt if fp_dt > 0 else 0
        else:
            fp_sample_rate = 0

        # Build per-fiber data
        fibers_data = {}
        for fiber_col in enabled_fibers:
            iso_signal = data_with_headers.loc[iso_mask, fiber_col].values
            gcamp_signal = data_with_headers.loc[gcamp_mask, fiber_col].values

            fibers_data[fiber_col] = {
                'iso': iso_signal,
                'gcamp': gcamp_signal,
                'label': fiber_labels[fiber_col],
            }
            print(f"[Photometry]   Fiber {fiber_col} ({fiber_labels[fiber_col]}): {len(iso_signal)} iso, {len(gcamp_signal)} gcamp samples")

        # Build save dict with new format (v2)
        save_dict = {
            'source_fp_path': str(self.file_paths['fp_data']),
            'source_ai_path': str(self.file_paths['ai_data']) if self.file_paths['ai_data'] else '',
            'fp_time': fp_time,
            'fibers': fibers_data,
            # Legacy fields for backward compatibility (use first fiber)
            'iso': fibers_data[enabled_fibers[0]]['iso'],
            'gcamp': fibers_data[enabled_fibers[0]]['gcamp'],
        }

        # Add AI data if available - reload at FULL resolution (no subsampling)
        # The preview uses subsampled data for speed, but we need full resolution for analysis
        ai_sample_rate = 0
        print(f"[Photometry] AI data check: _ai_data={self._ai_data is not None}, "
              f"ai_path={self.file_paths['ai_data']}, "
              f"exists={self.file_paths['ai_data'].exists() if self.file_paths['ai_data'] else 'N/A'}")
        if self._ai_data is not None and self.file_paths['ai_data'] and self.file_paths['ai_data'].exists():
            print(f"[Photometry] Reloading AI data at FULL resolution for saving...")
            t_reload = time_module.perf_counter()

            # Reload AI data without subsampling
            ai_data_full = photometry.load_ai_data_csv(self.file_paths['ai_data'], subsample=1)
            print(f"[Timing] Reload AI data full ({len(ai_data_full)} rows): {time_module.perf_counter() - t_reload:.2f}s")

            # Reload timestamps without subsampling
            ts_path = photometry.find_timestamps_file(self.file_paths['fp_data'])
            if ts_path and ts_path.exists():
                t_reload = time_module.perf_counter()
                timestamps_full = photometry.load_timestamps_csv(ts_path, subsample=1)
                print(f"[Timing] Reload timestamps full ({len(timestamps_full)} rows): {time_module.perf_counter() - t_reload:.2f}s")

                # Recalculate global_t_min with full timestamps for consistency
                ai_t_min = np.min(timestamps_full)
                # Use the same global_t_min as FP data for alignment
                actual_global_t_min = min(global_t_min, ai_t_min)

                # Normalize AI time using consistent global minimum
                ai_time = (timestamps_full - actual_global_t_min) / 60000

                # Estimate AI sample rate
                if len(ai_time) > 1:
                    ai_dt = np.median(np.diff(ai_time)) * 60  # Convert back to seconds
                    ai_sample_rate = 1.0 / ai_dt if ai_dt > 0 else 0

                save_dict['ai_time'] = ai_time

                # Get enabled AI channels at full resolution
                ai_channels = {}
                for i, col_info in self.ai_columns.items():
                    if col_info['checkbox'].isChecked():
                        col_name = col_info['column']
                        label = col_info['label_edit'].text() or f"ai_{col_name}"
                        # Clean the label for use as dict key
                        label = label.replace(' ', '_')
                        ai_channels[label] = ai_data_full[col_name].values

                save_dict['ai_channels'] = ai_channels
                print(f"[Photometry] AI data saved at {ai_sample_rate:.1f} Hz (full resolution)")

        # Add metadata
        save_dict['metadata'] = {
            'format_version': 2,  # v2 = multi-fiber support
            'fp_sample_rate': fp_sample_rate,
            'ai_sample_rate': ai_sample_rate,
            'created': datetime.now().isoformat(),
            'fp_time_col': time_col,
            'fp_led_col': led_col,
            'fiber_columns': enabled_fibers,
            'fiber_labels': fiber_labels,
        }

        # Save to NPZ
        np.savez(save_path, **save_dict)

        elapsed = time_module.perf_counter() - t0
        print(f"[Photometry] Data saved in {elapsed:.2f}s")
        print(f"[Photometry]   FP: {len(fp_time)} time points @ {fp_sample_rate:.1f} Hz")
        print(f"[Photometry]   Fibers: {enabled_fibers}")
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
