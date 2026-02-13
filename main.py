from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem, QAbstractItemView, QTreeWidgetItem, QTableView
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QIcon
from consolidation import ConsolidationManager
from core.file_table_model import FileTableModel, ColumnDef, DEFAULT_COLUMNS
from core.file_table_delegates import ButtonDelegate, StatusDelegate, AutoCompleteDelegate

import re
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QDialogButtonBox, QPushButton, QHBoxLayout, QCheckBox, QMenu
)
from PyQt6.QtGui import QAction, QPainter
from PyQt6.QtPrintSupport import QPrinter

import csv, json



from pathlib import Path
from typing import List, Optional
import sys
import os

# Fix KMeans memory leak warning on Windows
os.environ['OMP_NUM_THREADS'] = '1'

# Set Windows AppUserModelID so app shows as "PhysioMetrics" in Task Manager
# and groups correctly in taskbar (instead of showing as "Python")
try:
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "RyanPhillips.PhysioMetrics.App.1"
    )
except Exception:
    pass  # Not on Windows or API not available

import numpy as np
import pandas as pd

from core.state import AppState
from core import abf_io, filters
from core.plotting import PlotHost
# PyQtGraph backend import (lazy - only loaded when needed)
PYQTGRAPH_AVAILABLE = False
try:
    from plotting.pyqtgraph_backend import PyQtGraphPlotHost, PYQTGRAPH_AVAILABLE as _PG_AVAIL
    PYQTGRAPH_AVAILABLE = _PG_AVAIL
except ImportError:
    PYQTGRAPH_AVAILABLE = False
from core import stim as stimdet   # stim detection

# Peak detection: Switch between standard and downsampled versions
from core import peaks as peakdet              # Standard (original)
# from core import peaks_downsampled as peakdet    # Downsampled (BROKEN - messes up peaks)

from core import metrics  # calculation of breath metrics
from core.navigation_manager import NavigationManager
from core import telemetry  # Anonymous usage tracking
from plotting import PlotManager
from export import ExportManager


# Import editing modes
from editing import EditingModes
# Import dialogs
from dialogs import SpectralAnalysisDialog, SaveMetaDialog, PeakNavigatorDialog
from dialogs.advanced_peak_editor_dialog import AdvancedPeakEditorDialog
from dialogs.photometry_import_dialog_v2 import PhotometryImportDialog
from dialogs.photometry.chooser_dialog import show_photometry_chooser
from core import photometry
from core.channel_manager import ChannelManagerWidget
from core.ai_notebook_manager import AINotebookManager
from core.code_notebook_manager import CodeNotebookManager
from core.project_builder_manager import ProjectBuilderManager
from core.scan_manager import ScanManager
from core.recovery_manager import RecoveryManager
from core.classifier_manager import ClassifierManager
from core.gmm_manager import GMMManager
from core.notes_preview_manager import NotesPreviewManager

# Import new event marker system
from viewmodels.event_marker_viewmodel import EventMarkerViewModel
from views.events.plot_integration import EventMarkerPlotIntegration

# Import version
from version_info import VERSION_STRING


ORG = "PhysioMetrics"
APP = "PhysioMetrics"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize managers
        self.consolidation_manager = ConsolidationManager(self)
        from core.project_manager import ProjectManager
        self.project_manager = ProjectManager()
        self._ai_notebook = None  # Initialized after UI load
        self._code_notebook = None  # Initialized after UI load
        self._project_builder = None  # Initialized after UI load

        ui_file = Path(__file__).parent / "ui" / "pleth_app_layout_05_horizontal.ui"
        uic.loadUi(ui_file, self)

        # icon_path = Path(__file__).parent / "assets" / "plethapp_thumbnail_light_02.ico"
        icon_path = Path(__file__).parent / "assets" / "plethapp_thumbnail_dark_round.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        # after uic.loadUi(ui_file, self)
        from PyQt6.QtWidgets import QWidget, QPushButton
        for w in self.findChildren(QWidget):
            if w.property("startHidden") is True:
                w.hide()

        self.setWindowTitle(f"PhysioMetrics v{VERSION_STRING}")

        # Enable dark title bar on Windows 11
        self._enable_dark_title_bar()

        # Tab order is now set in the UI file:
        # 0: Notes Files, 1: Project Files, 2: Consolidation, 3: Code Notebook

        # Style status bar to match dark theme (true black)
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #000000;
                color: #d4d4d4;
                border-top: 1px solid #3e3e42;
            }
            QStatusBar::item {
                border: none;
            }
        """)
        # Disable the resize grip (removes the dots on the right side)
        self.statusBar().setSizeGripEnabled(False)

        # Add message history tracking and dropdown
        self._status_message_history = []
        self._setup_status_history_dropdown()

        # Add filename display to status bar (right side, before history button)
        self._setup_filename_display()

        # Add editing instructions label to status bar (left side, supports rich text)
        self._setup_editing_instructions_label()

        self.settings = QSettings(ORG, APP)
        self.state = AppState()
        self.single_panel_mode = False  # flips True after stim channel selection

        # Notch filter parameters
        self.notch_filter_lower = None
        self.notch_filter_upper = None

        # Filter order
        self.filter_order = 4  # Default Butterworth filter order

        # Auto-GMM refresh (default OFF for better performance)
        self.auto_gmm_enabled = False
        # Track if eupnea/sniffing detection is out of date
        self.eupnea_sniffing_out_of_date = False

        # Z-score normalization
        self.use_zscore_normalization = True  # Default: enabled
        self.zscore_global_mean = None  # Global mean across all sweeps (cached)
        self.zscore_global_std = None   # Global std across all sweeps (cached)

        # GMM clustering cache (for fast dialog loading)
        self._cached_gmm_results = None

        # Outlier detection metrics (default set)
        self.outlier_metrics = ["if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"]

        # Cross-sweep outlier detection
        self.global_outlier_stats = None  # Dict[metric_key, (mean, std)] - computed across all sweeps
        self.metrics_by_sweep = {}  # Dict[sweep_idx, Dict[metric_key, metric_array]]
        self.onsets_by_sweep = {}  # Dict[sweep_idx, onsets_array]

        # Eupnea detection parameters
        self.eupnea_freq_threshold = 5.0  # Hz - frequency threshold for eupnea (used in frequency mode)
        self.eupnea_min_duration = 2.0  # seconds - minimum sustained duration for eupnea region
        self.eupnea_detection_mode = "gmm"  # "gmm" or "frequency" - default to GMM-based detection

        # --- Embed Plot Widget into MainPlot (QFrame in Designer) ---
        # Force pyqtgraph backend - matplotlib is disabled (doesn't support event markers)
        self._current_backend = 'pyqtgraph' if PYQTGRAPH_AVAILABLE else 'matplotlib'
        self.state.plotting_backend = self._current_backend

        # Create appropriate plot host
        self.plot_host = self._create_plot_host(self._current_backend)

        # Setup layout
        layout = self.MainPlot.layout()
        if layout is None:
            from PyQt6.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self.MainPlot)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_host)

        # --- Setup New Event Marker System ---
        self._setup_new_event_markers()

        # --- Setup Channel Manager Widget ---
        self._setup_channel_manager()

        # --- Setup Event Markers Widget ---
        self._setup_event_markers_widget()

        # --- Method Aliases (for backward compatibility with old code/plugins) ---
        self.refresh_plot = self.redraw_main_plot
        self._show_status = self._log_status_message

        saved_geom = self.settings.value("geometry")
        if saved_geom:
            self.restoreGeometry(saved_geom)

        # --- Wire browse ---
        # Setup browse button with split-button dropdown (replaces original button)
        # The new button's clicked signal is connected inside _setup_browse_dropdown
        self._setup_browse_dropdown()

        # Add Ctrl+O shortcut - triggers different buttons based on active tab
        from PyQt6.QtGui import QShortcut, QKeySequence
        ctrl_o_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        ctrl_o_shortcut.activated.connect(self.on_ctrl_o_pressed)

        # Add F1 shortcut for Help
        f1_shortcut = QShortcut(QKeySequence("F1"), self)
        f1_shortcut.activated.connect(self.on_help_clicked)

        # Add Z/X shortcuts for navigation with ApplicationShortcut context
        from PyQt6.QtCore import Qt
        z_shortcut = QShortcut(QKeySequence("Z"), self)
        z_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        z_shortcut.activated.connect(self._on_z_pressed)

        x_shortcut = QShortcut(QKeySequence("X"), self)
        x_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        x_shortcut.activated.connect(self._on_x_pressed)

        # Add Ctrl+R shortcut for hot reload (development) - works from any window
        ctrl_r_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        ctrl_r_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        ctrl_r_shortcut.activated.connect(self._on_hot_reload)

        # Add Ctrl+D shortcut to toggle adaptive downsampling (performance mode)
        ctrl_d_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        ctrl_d_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        ctrl_d_shortcut.activated.connect(lambda: self.plot_manager.toggle_auto_downsample())

        # --- Wire channel selection (immediate application) ---
        self.AnalyzeChanSelect.currentIndexChanged.connect(self.on_analyze_channel_changed)
        self.StimChanSelect.currentIndexChanged.connect(self.on_stim_channel_changed)
        self.EventsChanSelect.currentIndexChanged.connect(self.on_events_channel_changed)

        # --- Wire classifier selection ---
        # Replace "ML Model" with specific algorithms for all three dropdowns
        self.peak_detec_combo.clear()
        self.peak_detec_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])

        self.eup_sniff_combo.clear()
        self.eup_sniff_combo.addItems(["None (Clear)", "All Eupnea", "GMM", "XGBoost", "Random Forest", "MLP"])

        self.digh_combo.clear()
        self.digh_combo.addItems(["None (Clear)", "Manual", "XGBoost", "Random Forest", "MLP"])

        # === Initialize Classifier Manager (early - needed for auto-load) ===
        self._classifier_manager = ClassifierManager(self)

        # === Initialize GMM Manager ===
        self._gmm_manager = GMMManager(self)

        # === Initialize Notes Preview Manager ===
        self._notes_preview_manager = NotesPreviewManager(self)

        # Auto-load ML models from last used directory (if available)
        # This must happen BEFORE setting defaults and connecting signals
        self._classifier_manager.auto_load_ml_models_on_startup()

        # Update dropdown states based on loaded models (will disable ML options if models not loaded)
        self._classifier_manager.update_classifier_dropdowns()

        # NOW connect signals AFTER models are loaded and dropdowns are configured
        self.peak_detec_combo.currentTextChanged.connect(self._classifier_manager.on_classifier_changed)
        self.eup_sniff_combo.currentTextChanged.connect(self._classifier_manager.on_eupnea_sniff_classifier_changed)
        self.digh_combo.currentTextChanged.connect(self._classifier_manager.on_sigh_classifier_changed)

        # Set defaults - XGBoost for peak detection and sigh, GMM for eupnea/sniff
        # If models not loaded, _update_classifier_dropdowns() already fell back to Threshold/GMM/Manual
        self.peak_detec_combo.setCurrentText("XGBoost")
        self.eup_sniff_combo.setCurrentText("GMM")  # GMM is the default for eupnea/sniff classification
        self.digh_combo.setCurrentText("XGBoost")

        # --- Wire filter controls ---
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(150)       # ms
        self._redraw_timer.timeout.connect(self.redraw_main_plot)

        # filters: commit-on-finish, not per key
        # (update_and_redraw also calls _on_filter_changed for telemetry/Apply button)
        self.LowPassVal.editingFinished.connect(self.update_and_redraw)
        self.HighPassVal.editingFinished.connect(self.update_and_redraw)
        self.FilterOrderSpin.valueChanged.connect(self.update_and_redraw)

        # checkboxes toggled immediately, but we debounce the draw
        self.LowPass_checkBox.toggled.connect(self.update_and_redraw)
        self.HighPass_checkBox.toggled.connect(self.update_and_redraw)
        self.InvertSignal_checkBox.toggled.connect(self.update_and_redraw)

        # Spectral Analysis button
        self.SpectralAnalysisButton.clicked.connect(self.on_spectral_analysis_clicked)

        # Outlier Threshold button - REMOVED (moved to Analysis Options dialog)
        # self.OutlierThreshButton.clicked.connect(self.on_outlier_thresh_clicked)

        # Eupnea Threshold button

        # GMM Clustering button - REMOVED (moved to Analysis Options dialog)
        # self.GMMClusteringButton.clicked.connect(self.on_gmm_clustering_clicked)

        # Peak Navigator/Editor button
        self.editor_pushButton.clicked.connect(self.on_peak_navigator_clicked)

        # Auto-Update GMM checkbox
        # Auto-Update GMM checkbox moved to GMM dialog
        # Connect the manual update button
        self.UpdateEupneaSniffingButton.clicked.connect(self.on_update_eupnea_sniffing_clicked)

        # --- Initialize Navigation Manager ---
        self.navigation_manager = NavigationManager(self)
        self.WindowRangeValue.setText("10")  # Default window length

        # --- Initialize Plot Manager (BEFORE signal connections that may trigger plotting) ---
        self.plot_manager = PlotManager(self)

        # --- Initialize Export Manager ---
        self.export_manager = ExportManager(self)

        # --- Initialize Telemetry Heartbeat Timer ---
        # Send periodic user_engagement events to help GA4 recognize active users
        self.telemetry_heartbeat_timer = QTimer(self)
        self.telemetry_heartbeat_timer.timeout.connect(telemetry.log_user_engagement)
        self.telemetry_heartbeat_timer.start(45000)  # Send engagement event every 45 seconds

        # --- Peak-detect UI wiring ---
        # Prominence field and Apply button already exist in UI file
        # Connect the "More Options" button to open multi-tabbed analysis options dialog
        self.ThreshOptions.clicked.connect(self._open_analysis_options)

        # Detect button applies peaks with current parameters
        self.ApplyPeakFindPushButton.setEnabled(False)  # stays disabled until prominence detected
        self.ApplyPeakFindPushButton.clicked.connect(self._apply_peak_detection)

        # PeakPromValueSpinBox removed - prominence/threshold now set in Analysis Options dialog
        # Re-enable Apply button when user manually edits prominence (use spinbox)
        # self.PeakPromValueSpinBox.valueChanged.connect(lambda: self.ApplyPeakFindPushButton.setEnabled(True) if self.state.analyze_chan else None)

        # Connect spinbox to update threshold line on main plot
        # self.PeakPromValueSpinBox.valueChanged.connect(self._on_prominence_spinbox_changed)

        # Configure spinbox for fine-grained control (0.01 increments for second decimal place)
        # self.PeakPromValueSpinBox.setSingleStep(0.01)
        # self.PeakPromValueSpinBox.setDecimals(4)  # Show 4 decimal places
        # self.PeakPromValueSpinBox.setMinimum(0.0001)  # Minimum prominence value
        # self.PeakPromValueSpinBox.setMaximum(1000.0)  # Maximum prominence value

        # Store peak detection parameters (auto-populated when channel selected)
        self.peak_prominence = None
        self.peak_height_threshold = None  # Same as prominence by default
        self.peak_min_dist = 0.05  # Default minimum peak distance in seconds

        # Default values for eupnea and apnea thresholds
        self.ApneaThresh.setText("0.5")   # seconds - gaps longer than this are apnea

        # OutlierSD moved to Analysis Options dialog (Outlier Detection tab)
        # Create a simple object to store the value (mimics QLineEdit.text() interface)
        class OutlierSDHolder:
            def __init__(self, value="3.0"):
                self._value = value
            def text(self):
                return self._value
            def setText(self, value):
                self._value = str(value)

        self.OutlierSD = OutlierSDHolder("3.0")  # Default: 3.0 SD for outlier detection

        # Connect signals for apnea threshold changes to trigger redraw
        self.ApneaThresh.textChanged.connect(self._on_region_threshold_changed)
        # OutlierSD changes are handled in the Analysis Options dialog

        # --- y2 metric dropdown (choices only; plotting later) ---
        self.y2plot_dropdown.clear()
        self.state.y2_values_by_sweep.clear()
        self.plot_host.clear_y2()

        self.y2plot_dropdown.addItem("None", userData=None)
        for label, key in metrics.METRIC_SPECS:
            self.y2plot_dropdown.addItem(label, userData=key)

        # Make dropdown searchable by typing
        self.y2plot_dropdown.setEditable(True)
        self.y2plot_dropdown.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)

        # Add completer for case-insensitive searching
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QCompleter
        completer = QCompleter([self.y2plot_dropdown.itemText(i) for i in range(self.y2plot_dropdown.count())])
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)  # Match anywhere in the string
        self.y2plot_dropdown.setCompleter(completer)

        # ADD/DELETE Peak Mode: track selection in state
        self.state.y2_metric_key = None
        self.y2plot_dropdown.currentIndexChanged.connect(self.on_y2_metric_changed)

        # --- Display Control Widgets ---
        # Y-axis autoscale checkbox (percentile vs min/max)
        if hasattr(self, 'yautoscale_checkBox'):
            self.yautoscale_checkBox.setChecked(self.state.use_percentile_autoscale)
            self.yautoscale_checkBox.toggled.connect(self.on_yautoscale_toggled)

        # Y-axis padding spinbox (for percentile mode) — debounced to avoid
        # redundant redraws when holding arrow keys or scrolling quickly
        if hasattr(self, 'ypadding_SpinBox'):
            self.ypadding_SpinBox.setMinimum(0.0)
            self.ypadding_SpinBox.setMaximum(1.0)
            self.ypadding_SpinBox.setSingleStep(0.05)
            self.ypadding_SpinBox.setDecimals(2)
            self.ypadding_SpinBox.setValue(self.state.autoscale_padding)
            self._ypadding_debounce = QTimer()
            self._ypadding_debounce.setSingleShot(True)
            self._ypadding_debounce.setInterval(150)
            self._ypadding_debounce.timeout.connect(
                lambda: self.on_ypadding_changed(self.ypadding_SpinBox.value()))
            self.ypadding_SpinBox.valueChanged.connect(
                lambda _: self._ypadding_debounce.start())

        # Region display toggle checkboxes (line vs shade)
        if hasattr(self, 'eupnea_checkBox'):
            self.eupnea_checkBox.setChecked(self.state.eupnea_use_shade)
            self.eupnea_checkBox.toggled.connect(self.on_eupnea_display_toggled)

        if hasattr(self, 'sniffing_checkBox'):
            self.sniffing_checkBox.setChecked(self.state.sniffing_use_shade)
            self.sniffing_checkBox.toggled.connect(self.on_sniffing_display_toggled)

        if hasattr(self, 'Apnea_checkBox'):
            self.Apnea_checkBox.setChecked(self.state.apnea_use_shade)
            self.Apnea_checkBox.toggled.connect(self.on_apnea_display_toggled)

        if hasattr(self, 'Outliers_checkBox'):
            self.Outliers_checkBox.setChecked(self.state.outliers_use_shade)
            self.Outliers_checkBox.toggled.connect(self.on_outliers_display_toggled)

        # Dark mode toggle (for plot background)
        if hasattr(self, 'checkBox'):  # Generic name from UI - should be renamed to DarkModeCheckBox
            # Load saved dark mode preference (default to True = dark mode)
            dark_mode_enabled = self.settings.value("plot_dark_mode", True, type=bool)

            # Apply the theme to plot_host
            theme = "dark" if dark_mode_enabled else "light"
            self.plot_host.set_plot_theme(theme)

            # Set checkbox state to match loaded preference
            self.checkBox.setChecked(dark_mode_enabled)
            self.checkBox.toggled.connect(self.on_dark_mode_toggled)

        # PyQtGraph backend toggle (experimental high-performance plotting)
        self._setup_pyqtgraph_toggle()

        # Initialize editing modes manager
        self.editing_modes = EditingModes(self)

        # --- Mark Events button (event detection settings) ---
        self.MarkEventsButton.clicked.connect(self.on_mark_events_clicked)
        self.MarkEventsButton.setVisible(False)  # Hidden — unused in current UI

        # --- Sigh overlay artists ---
        self._sigh_artists = []         # matplotlib artists for sigh overlay

        # --- Wire omit button ---
        self.OmitSweepButton.setCheckable(True)
        self.OmitSweepButton.clicked.connect(self.on_omit_sweep_clicked)

        # --- Move Point mode ---
        self._is_dragging = False  # Track if currently dragging a point

        self.movePointButton.setCheckable(True)

        # Mark Sniff button

        #wire save analyzed data button
        self.SaveAnalyzedDataButton.clicked.connect(self.on_save_analyzed_clicked)

        # Wire save options button to configure export metrics
        self.saveoptions_pushButton.clicked.connect(self.on_save_options_clicked)

        # Wire view summary button to show PDF preview
        self.ViewSummary_pushButton.clicked.connect(self.on_view_summary_clicked)

        # Wire Help button (from UI file)
        self.helpbutton.clicked.connect(self.on_help_clicked)

        # Set pointer cursor for update notification label (defined in UI file)
        from PyQt6.QtGui import QCursor
        from PyQt6.QtCore import Qt
        self.update_notification_label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

        # Store update info for later use
        self.update_info = None

        # Start background update check
        self._check_for_updates_on_startup()



        # Defaults: 0.5–20 Hz band, all off initially
        self.HighPassVal.setText("0.5")
        self.LowPassVal.setText("20")
        self.HighPass_checkBox.setChecked(True)  # Default ON to remove baseline drift
        self.LowPass_checkBox.setChecked(True)
        self.InvertSignal_checkBox.setChecked(False)

        # Push defaults into state (no-op if no data yet)
        self.update_and_redraw()
        self._refresh_omit_button_label()

        # Connect matplotlib toolbar to turn off edit modes
        self.plot_host.set_toolbar_callback(self.editing_modes.turn_off_all_edit_modes)


        # --- Curation tab wiring ---
        self.FilePathButton.clicked.connect(self.on_curation_choose_dir_clicked)
        # Enable multiple selection for both list widgets
        self.FileList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.FilestoConsolidateList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Note: Move buttons and search filter are connected in NavigationManager
        self.FileListSearchBox.setPlaceholderText("Filter by keywords (e.g., 'gfp 2.5mW' or 'gfp, chr2')...")
        # Wire consolidate button
        self.ConsolidateSaveDataButton.clicked.connect(self.on_consolidate_save_data_clicked)

        # Wire new consolidation tab in Project Builder
        self.consolidationSourceList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.consolidationFilesList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.consolidationSaveButton.clicked.connect(self._on_consolidation_save_clicked)
        # Connect move buttons
        self.consolidationMoveAllRight.clicked.connect(self._consolidation_move_all_right)
        self.consolidationMoveSingleRight.clicked.connect(self._consolidation_move_selected_right)
        self.consolidationMoveSingleLeft.clicked.connect(self._consolidation_move_selected_left)
        self.consolidationMoveAllLeft.clicked.connect(self._consolidation_move_all_left)
        # Connect search filter
        self.consolidationSearchBox.textChanged.connect(self._filter_consolidation_source_list)
        # Connect browse button
        self.consolidationBrowseButton.clicked.connect(self._on_consolidation_browse_clicked)
        # Connect reset button (returns to project files)
        self.consolidationResetButton.clicked.connect(self._on_consolidation_reset_clicked)
        self.consolidationResetButton.hide()  # Hidden initially, shown when browsing custom folder
        # Track consolidation source mode (None = project files, path = custom folder)
        self._consolidation_custom_folder = None
        # Auto-refresh consolidation source when switching to the tab
        self.leftColumnTabs.currentChanged.connect(self._on_left_column_tab_changed)

        # ========================================
        # TESTING MODE: Auto-load file and set channels
        # ========================================
        # To enable: set environment variable PLETHAPP_TESTING=1
        # Windows CMD (two commands):
        #   set PLETHAPP_TESTING=1
        #   python run_debug.py
        # Windows PowerShell:
        #   $env:PLETHAPP_TESTING="1"; python run_debug.py
        # Linux/Mac:
        #   PLETHAPP_TESTING=1 python run_debug.py
        # Silently check environment variables (only print if set)
        # print(f"[DEBUG] PLETHAPP_TESTING environment variable = '{os.environ.get('PLETHAPP_TESTING')}'")
        # print(f"[DEBUG] PLETHAPP_PULSE_TEST environment variable = '{os.environ.get('PLETHAPP_PULSE_TEST')}'")
        if os.environ.get('PLETHAPP_TESTING') == '1':
            print("[TESTING MODE] Auto-loading test file...")
            # Check if pulse test mode
            if os.environ.get('PLETHAPP_PULSE_TEST') == '1':
                test_file = Path(r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\R2 R5 R1\25122001.abf")
                print("[TESTING MODE] Pulse test mode - loading 25122001.abf (25ms pulse experiment)")
            else:
                test_file = Path(r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\25121004.abf")
                print("[TESTING MODE] Standard test mode - loading 25121004.abf (30Hz stim)")
            print(f"[TESTING MODE] Checking if file exists: {test_file}")
            print(f"[TESTING MODE] File exists: {test_file.exists()}")
            if test_file.exists():
                print(f"[TESTING MODE] Scheduling auto-load in 100ms...")
                QTimer.singleShot(100, lambda: self._auto_load_test_file(test_file))
            else:
                print(f"[TESTING MODE] Warning: Test file not found: {test_file}")

        # === Initialize Project Builder Manager ===
        self._project_builder = ProjectBuilderManager(self)

        # === Initialize Scan Manager ===
        self._scan_manager = ScanManager(self)

        # === Initialize Recovery Manager ===
        self._recovery_manager = RecoveryManager(self)

        # NOTE: ClassifierManager initialized earlier (line ~236) because it's needed before signal connections

        # === Project Builder Connections ===
        self.browseDirectoryButton.clicked.connect(self._project_builder.on_project_browse_directory)
        self.scanFilesButton.clicked.connect(self._scan_manager.on_project_scan_files)
        # NOTE: experiment-based workflow buttons were removed from .ui file
        self.clearFilesButton.clicked.connect(self._project_builder.on_project_clear_files)
        self.newProjectButton.clicked.connect(self._project_builder.on_project_new)
        self.saveProjectButton.clicked.connect(self._project_builder.on_project_save)

        # Set up the new unified project name combo (replaces loadProjectCombo + projectNameEdit)
        self._setup_project_name_combo()

        # Notes Files tab connections
        if hasattr(self, 'searchNotesButton'):
            self.searchNotesButton.clicked.connect(self._project_builder.on_notes_search)
        if hasattr(self, 'browseNotesButton'):
            self.browseNotesButton.clicked.connect(self._project_builder.on_notes_browse)
        if hasattr(self, 'openNoteButton'):
            self.openNoteButton.clicked.connect(self._project_builder.on_notes_open)
        if hasattr(self, 'previewNoteButton'):
            self.previewNoteButton.clicked.connect(self._project_builder.on_notes_preview)
        if hasattr(self, 'linkNoteButton'):
            self.linkNoteButton.clicked.connect(self._project_builder.on_notes_link)
        if hasattr(self, 'notesFilterEdit'):
            self.notesFilterEdit.textChanged.connect(self._on_notes_filter_changed)

        # Initialize notes files model
        self._project_builder.init_notes_files_model()

        # Add extra notes action buttons (programmatically added)
        self._project_builder.add_notes_action_buttons()

        # Connect Project Builder buttons (defined in .ui file)
        self._connect_project_builder_buttons()

        # Project Builder state
        self._project_directory = None
        self._notes_directory = None  # Separate folder for notes files
        self._discovered_files_data = []  # Store file metadata from scan
        self._notes_files_data = []  # Store notes file metadata

        # === Master File List Setup ===
        # Hide the Project Organization section (right column) - we're using a flat master list instead
        self._setup_master_file_list()

        # Store file list data (each row = one analysis task: file + channel + animal)
        self._master_file_list = []  # List of task dicts
        self._active_master_list_row = None  # Track which row is being analyzed

        # optional: keep a handle to the chosen dir
        self._curation_dir = None

        # Set up right-click export context menu for main window
        self._setup_export_context_menu()

    def _setup_export_context_menu(self):
        """Set up right-click context menu for screenshot/PDF export."""
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_export_context_menu)

    def _show_export_context_menu(self, pos):
        """Show context menu with export options."""
        # Don't show this menu if right-click was on the plot area
        # (the plot has its own richer context menu via EventMarkerContextMenu)
        widget_at = self.childAt(pos)
        if widget_at and hasattr(self, 'plot_host'):
            if widget_at is self.plot_host or self.plot_host.isAncestorOf(widget_at):
                return
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d30;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #094771;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3e3e42;
                margin: 4px 10px;
            }
        """)

        # Export submenu
        export_menu = menu.addMenu("Export Window...")

        png_action = QAction("Export as PNG (High Resolution)", self)
        png_action.triggered.connect(lambda: self._export_window_as_image('png'))
        export_menu.addAction(png_action)

        pdf_action = QAction("Export as PDF", self)
        pdf_action.triggered.connect(self._export_window_as_pdf)
        export_menu.addAction(pdf_action)

        jpg_action = QAction("Export as JPEG", self)
        jpg_action.triggered.connect(lambda: self._export_window_as_image('jpg'))
        export_menu.addAction(jpg_action)

        export_menu.addSeparator()

        clipboard_action = QAction("Copy to Clipboard", self)
        clipboard_action.triggered.connect(self._copy_window_to_clipboard)
        export_menu.addAction(clipboard_action)

        # Performance Mode submenu
        if hasattr(self, 'plot_manager'):
            pm = self.plot_manager
            menu.addSeparator()
            perf_menu = menu.addMenu("Performance Mode")
            perf_menu.setStyleSheet(menu.styleSheet())

            override = pm._downsample_override

            action_auto = perf_menu.addAction("Auto (Recommended)")
            action_auto.setCheckable(True)
            action_auto.setChecked(override is None)

            action_fast = perf_menu.addAction("Fast (Downsample On)")
            action_fast.setCheckable(True)
            action_fast.setChecked(override is True)

            action_full = perf_menu.addAction("Full Resolution (Downsample Off)")
            action_full.setCheckable(True)
            action_full.setChecked(override is False)

            last_ms = pm._last_redraw_ms
            if last_ms > 0:
                perf_menu.addSeparator()
                info = perf_menu.addAction(f"Last redraw: {last_ms:.0f}ms")
                info.setEnabled(False)

            action_auto.triggered.connect(lambda: pm.toggle_auto_downsample(force=None))
            action_fast.triggered.connect(lambda: pm.toggle_auto_downsample(force=True))
            action_full.triggered.connect(lambda: pm.toggle_auto_downsample(force=False))

        menu.exec(self.mapToGlobal(pos))

    def _export_window_as_image(self, format: str):
        """Export main window as image."""
        # Use loaded filename if available
        if self.state.in_path:
            base_name = Path(self.state.in_path).stem
        else:
            base_name = "PhysioMetrics"
        default_name = f"{base_name}_screenshot.{format}"

        format_filters = {
            'png': "PNG Image (*.png)",
            'jpg': "JPEG Image (*.jpg)"
        }

        filepath, _ = QFileDialog.getSaveFileName(
            self, f"Export as {format.upper()}", default_name,
            format_filters.get(format, "All Files (*)")
        )

        if not filepath:
            return

        if not filepath.lower().endswith(f'.{format}'):
            filepath += f'.{format}'

        try:
            # Capture at 2x resolution
            pixmap = self.grab()
            scaled_size = pixmap.size() * 2
            pixmap = pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            quality = 95 if format == 'jpg' else -1
            if pixmap.save(filepath, quality=quality):
                QMessageBox.information(self, "Export Successful",
                    f"Screenshot exported to:\n{filepath}\n\n"
                    f"Resolution: {pixmap.width()} x {pixmap.height()} pixels")
                print(f"[export] Saved {format.upper()}: {filepath}")
            else:
                raise Exception("Failed to save image")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export:\n{str(e)}")

    def _export_window_as_pdf(self):
        """Export main window as PDF."""
        if self.state.in_path:
            base_name = Path(self.state.in_path).stem
        else:
            base_name = "PhysioMetrics"
        default_name = f"{base_name}_screenshot.pdf"

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export as PDF", default_name,
            "PDF Document (*.pdf)"
        )

        if not filepath:
            return

        if not filepath.lower().endswith('.pdf'):
            filepath += '.pdf'

        try:
            from PyQt6.QtGui import QPageLayout

            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(filepath)
            printer.setPageOrientation(QPageLayout.Orientation.Landscape)

            painter = QPainter()
            if not painter.begin(printer):
                raise Exception("Failed to initialize PDF painter")

            page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
            window_size = self.size()

            scale_x = page_rect.width() / window_size.width()
            scale_y = page_rect.height() / window_size.height()
            scale = min(scale_x, scale_y) * 0.95

            offset_x = (page_rect.width() - window_size.width() * scale) / 2
            offset_y = (page_rect.height() - window_size.height() * scale) / 2

            painter.translate(offset_x, offset_y)
            painter.scale(scale, scale)
            self.render(painter)
            painter.end()

            QMessageBox.information(self, "PDF Saved", f"PDF exported to:\n{filepath}")
            print(f"[export] Saved PDF: {filepath}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export PDF:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _copy_window_to_clipboard(self):
        """Copy window screenshot to clipboard."""
        try:
            pixmap = self.grab()
            scaled_size = pixmap.size() * 2
            pixmap = pixmap.scaled(
                scaled_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            clipboard = QApplication.clipboard()
            clipboard.setPixmap(pixmap)

            QMessageBox.information(self, "Copied to Clipboard",
                f"Screenshot copied!\n\n"
                f"Resolution: {pixmap.width()} x {pixmap.height()} pixels\n\n"
                "Paste with Ctrl+V into any application.")

        except Exception as e:
            QMessageBox.critical(self, "Copy Failed", f"Failed to copy:\n{str(e)}")

    def _auto_load_test_file(self, file_path: Path):
        """Helper function for testing mode - auto-loads file and sets channels."""
        print(f"[TESTING MODE] _auto_load_test_file() called with: {file_path}")
        print(f"[TESTING MODE] Calling load_file()...")
        self.load_file(file_path)
        print(f"[TESTING MODE] load_file() returned, starting polling timer...")

        # Poll until file is loaded, then set channels
        self._check_file_loaded_timer = QTimer()
        self._check_file_loaded_timer.timeout.connect(self._check_if_file_loaded)
        self._check_file_loaded_timer.start(100)  # Check every 100ms
        print(f"[TESTING MODE] Polling timer started")

    def _check_if_file_loaded(self):
        """Poll to see if file has finished loading."""
        # Check if state has been populated with data AND combos have been populated
        print(f"[TESTING MODE] Polling: t={self.state.t is not None}, sweeps={self.state.sweeps is not None}, "
              f"AnalyzeChanSelect.count={self.AnalyzeChanSelect.count()}, StimChanSelect.count={self.StimChanSelect.count()}")
        if (self.state.t is not None and
            self.state.sweeps is not None and
            len(self.state.sweeps) > 0 and
            self.AnalyzeChanSelect.count() > 0 and
            self.StimChanSelect.count() > 0):
            # File is loaded and combos are populated!
            print(f"[TESTING MODE] File loaded! Stopping timer and setting test channels...")
            self._check_file_loaded_timer.stop()
            self._set_test_channels()

    def _set_test_channels(self):
        """Helper function for testing mode - sets analysis and stim channels."""
        is_pulse_test = os.environ.get('PLETHAPP_PULSE_TEST') == '1'

        if is_pulse_test:
            print("[TESTING MODE] Setting analyze channel to 0 (first), stim channel to last...")
        else:
            print("[TESTING MODE] Setting analyze channel to 0, stim channel to 7...")

        print(f"[TESTING MODE] AnalyzeChanSelect has {self.AnalyzeChanSelect.count()} items")
        print(f"[TESTING MODE] StimChanSelect has {self.StimChanSelect.count()} items")

        # Set analyze channel to index 1 (channel 0, since index 0 is "All Channels")
        if self.AnalyzeChanSelect.count() > 1:
            self.AnalyzeChanSelect.setCurrentIndex(1)  # Channel 0 is at index 1
            print(f"[TESTING MODE] Set analyze channel to index 1: {self.AnalyzeChanSelect.currentText()}")
            # Manually trigger the channel change handler
            self.on_analyze_channel_changed(1)

        # Set stim channel (last channel for pulse test, channel 7 for standard test)
        stim_channel_set = False
        if is_pulse_test:
            # Select the last channel in the dropdown
            last_idx = self.StimChanSelect.count() - 1
            if last_idx >= 0:
                self.StimChanSelect.setCurrentIndex(last_idx)
                print(f"[TESTING MODE] Set stim channel to last (index {last_idx}): {self.StimChanSelect.currentText()}")
                stim_channel_set = True
                # Manually trigger the channel change handler
                self.on_stim_channel_changed(last_idx)
        else:
            # Set stim channel to 7 (need to find the index)
            for i in range(self.StimChanSelect.count()):
                item_text = self.StimChanSelect.itemText(i)
                # Check if this item contains "7" (handles "7", "IN 7", "Channel 7", etc.)
                # Extract the number from the item text
                if "7" in item_text.split():  # Split by whitespace and check if "7" is one of the words
                    print(f"[TESTING MODE] Found stim channel at index {i}: '{item_text}'")
                    self.StimChanSelect.setCurrentIndex(i)
                    stim_channel_set = True
                    # Manually trigger the channel change handler
                    self.on_stim_channel_changed(i)
                    break

        if not stim_channel_set:
            if is_pulse_test:
                print("[TESTING MODE] Warning: Could not find last stim channel")
            else:
                print("[TESTING MODE] Warning: Could not find stim channel '7'")

        # Wait a bit for channels to be processed, then click Apply Peak Detection
        QTimer.singleShot(200, self._click_apply_peak_detection)

    def _click_apply_peak_detection(self):
        """Helper for testing mode - clicks the Apply button for peak detection."""
        if self.ApplyPeakFindPushButton.isEnabled():
            print("[TESTING MODE] Clicking Apply Peak Detection button...")
            self.ApplyPeakFindPushButton.click()
            print("[TESTING MODE] Auto-load complete!")
        else:
            print("[TESTING MODE] Warning: Apply Peak Detection button is not enabled")
            print("[TESTING MODE] Auto-load complete (but peaks not detected)")

        # Log main screen view for telemetry
        telemetry.log_screen_view('Main Analysis Screen', screen_class='main')

    # ---------- File browse ----------
    def closeEvent(self, event):
        """Save window geometry on close."""
        self.settings.setValue("geometry", self.saveGeometry())

        # Log telemetry session end
        telemetry.log_session_end()

        super().closeEvent(event)

    def _setup_browse_dropdown(self):
        """Add dropdown menu to browse button for recent files and folders.

        The BrowseButton is a QToolButton with MenuButtonPopup mode (set in .ui file).
        Main area opens file dialog, arrow shows dropdown menu.
        """
        from PyQt6.QtWidgets import QMenu

        # Connect main button click to browse action
        self.BrowseButton.clicked.connect(self._on_browse_action)

        # Create menu for the dropdown arrow
        self._browse_menu = QMenu(self)
        self._browse_menu.setStyleSheet("""
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
        """)

        # Set the menu on the button (dropdown arrow will trigger it)
        self.BrowseButton.setMenu(self._browse_menu)

        # Build menu when about to show
        self._browse_menu.aboutToShow.connect(self._build_browse_menu)

    def _build_browse_menu(self):
        """Build the dropdown menu with recent files and folders."""
        from PyQt6.QtWidgets import QMenu
        from pathlib import Path

        self._browse_menu.clear()

        # Get recent files and folders
        recent_files = self._get_recent_files()
        recent_folders = self._get_recent_folders()
        pinned_paths = self._get_pinned_paths()

        # Pinned locations
        # Note: skip p.exists() checks here — they block on network paths (lab server).
        # Stale entries are handled gracefully when clicked (_load_recent_file / _browse_from_folder).
        if pinned_paths:
            self._browse_menu.addSection("📌 Pinned Locations")
            for path in pinned_paths:
                p = Path(path)
                # Show path context for pinned items too
                parts = p.parts
                if len(parts) >= 3:
                    display_name = f"{parts[-3]}/{parts[-2]}/{p.name}"
                elif len(parts) >= 2:
                    display_name = f"{parts[-2]}/{p.name}"
                else:
                    display_name = p.name
                if len(display_name) > 45:
                    display_name = "..." + display_name[-42:]

                if p.suffix:  # Has extension — likely a file
                    action = self._browse_menu.addAction(f"📄 {display_name}")
                    action.setToolTip(str(p))
                    action.triggered.connect(lambda checked, fp=str(p): self._load_recent_file(fp))
                else:
                    action = self._browse_menu.addAction(f"📁 {display_name}")
                    action.setToolTip(str(p))
                    action.triggered.connect(lambda checked, fp=str(p): self._browse_from_folder(fp))

        # Recent files
        if recent_files:
            self._browse_menu.addSection("Recent Files")
            for file_path in recent_files[:8]:
                p = Path(file_path)
                # Show 3 levels of path context for clarity (especially for FP_data files)
                parts = p.parts
                if len(parts) >= 4:
                    display_name = f"{parts[-4]}/.../{p.name}"
                elif len(parts) >= 3:
                    display_name = f"{parts[-3]}/{parts[-2]}/{p.name}"
                elif len(parts) >= 2:
                    display_name = f"{parts[-2]}/{p.name}"
                else:
                    display_name = p.name
                # Truncate if still too long
                if len(display_name) > 50:
                    display_name = "..." + display_name[-47:]
                action = self._browse_menu.addAction(f"📄 {display_name}")
                action.setToolTip(str(p))
                action.triggered.connect(lambda checked, fp=file_path: self._load_recent_file(fp))

        # Recent folders
        if recent_folders:
            self._browse_menu.addSection("Browse Recent Folders")
            for folder_path in recent_folders[:5]:
                p = Path(folder_path)
                display_name = p.name or str(p)
                if len(display_name) > 40:
                    display_name = "..." + display_name[-37:]
                action = self._browse_menu.addAction(f"📁 {display_name}")
                action.setToolTip(str(p))
                action.triggered.connect(lambda checked, fp=folder_path: self._browse_from_folder(fp))

        # Pin/unpin current folder
        if self.state.in_path:
            self._browse_menu.addSeparator()
            current_folder = str(Path(self.state.in_path).parent)
            if self._is_path_pinned(current_folder):
                action = self._browse_menu.addAction("📌 Unpin current folder")
                action.triggered.connect(lambda: self._toggle_pinned_path(current_folder))
            else:
                action = self._browse_menu.addAction("📌 Pin current folder")
                action.triggered.connect(lambda: self._toggle_pinned_path(current_folder))

        # If empty, show note
        if self._browse_menu.isEmpty():
            action = self._browse_menu.addAction("No recent files")
            action.setEnabled(False)

    def _get_recent_files(self):
        """Get list of recent files."""
        files = self.settings.value("recent_files", [])
        if isinstance(files, str):
            files = [files] if files else []
        return files or []

    def _get_recent_folders(self):
        """Get list of recent folders."""
        folders = self.settings.value("recent_folders", [])
        if isinstance(folders, str):
            folders = [folders] if folders else []
        return folders or []

    def _get_pinned_paths(self):
        """Get list of pinned paths."""
        paths = self.settings.value("pinned_paths", [])
        if isinstance(paths, str):
            paths = [paths] if paths else []
        return paths or []

    def _add_recent_file(self, file_path: str):
        """Add a file to the recent files list."""
        recent = self._get_recent_files()
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        self.settings.setValue("recent_files", recent[:15])

    def _add_recent_folder(self, folder_path: str):
        """Add a folder to the recent folders list."""
        recent = self._get_recent_folders()
        if folder_path in recent:
            recent.remove(folder_path)
        recent.insert(0, folder_path)
        self.settings.setValue("recent_folders", recent[:10])

    def _is_path_pinned(self, path: str) -> bool:
        """Check if a path is pinned."""
        return path in self._get_pinned_paths()

    def _toggle_pinned_path(self, path: str):
        """Toggle a path's pinned status."""
        pinned = self._get_pinned_paths()
        if path in pinned:
            pinned.remove(path)
        else:
            pinned.insert(0, path)
        self.settings.setValue("pinned_paths", pinned[:10])

    def _on_browse_action(self):
        """Open the file browser dialog (triggered from dropdown menu)."""
        from pathlib import Path
        from PyQt6.QtWidgets import QFileDialog

        last_dir = self.settings.value("last_dir", str(Path.home()))
        # Skip exists() check - it's slow on network drives and the dialog handles it gracefully
        if last_dir and not (last_dir.startswith('\\\\') or (len(last_dir) > 1 and last_dir[1] == ':')):
            # Only check exists() for non-network, non-drive paths
            if not Path(str(last_dir)).exists():
                last_dir = str(Path.home())

        # Show wait cursor while file dialog loads (can be slow on network drives)
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Opening file browser...")
        QApplication.processEvents()

        try:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select File(s)", last_dir,
                "All Supported (*.abf *.smrx *.edf *.pleth.npz *.csv);;Data Files (*.abf *.smrx *.edf);;PhysioMetrics Sessions (*.pleth.npz);;Photometry/CSV (*.csv);;ABF Files (*.abf);;SMRX Files (*.smrx);;EDF Files (*.edf);;All Files (*.*)"
            )
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().clearMessage()

        if not paths:
            return

        # Convert to Path objects
        file_paths = [Path(p) for p in paths]

        # Store the directory of the first file
        self.settings.setValue("last_dir", str(file_paths[0].parent))

        # Load the files using the common loading method
        self._load_files(file_paths)

    def _load_files(self, file_paths: list):
        """
        Load one or more files, handling different file types appropriately.

        This method encapsulates the file loading logic and is used by both
        on_browse_clicked and the recent files dropdown menu.

        Args:
            file_paths: List of Path objects to load
        """
        from pathlib import Path

        if not file_paths:
            return

        # Track recently used files and folders
        if file_paths:
            self._add_recent_file(str(file_paths[0]))
            self._add_recent_folder(str(file_paths[0].parent))

        # Update filename display in status bar
        if len(file_paths) == 1:
            self.filename_label.setText(f"File: {file_paths[0].name}")
        else:
            self.filename_label.setText(f"Files: {len(file_paths)} files ({file_paths[0].name}, ...)")

        # Check if any files are .pleth.npz (session files)
        npz_files = [f for f in file_paths if f.suffix == '.npz' or f.name.endswith('.pleth.npz')]

        if npz_files:
            # Can only load one NPZ session at a time
            if len(file_paths) > 1:
                self._show_warning("Cannot Mix File Types",
                    "Cannot load session files (.pleth.npz) together with data files.\n\n"
                    "Please select either:\n"
                    "• One or more data files (.abf, .smrx, .edf) for concatenation, OR\n"
                    "• One session file (.pleth.npz) to restore analysis"
                )
                return

            # Load the NPZ session
            self.load_npz_state(file_paths[0])

        # Check if this is a photometry file (CSV with FP_data pattern)
        elif len(file_paths) == 1 and photometry.detect_photometry_file(file_paths[0]):
            # Use the chooser dialog to check for existing NPZ and show options
            action, data = show_photometry_chooser(self, file_paths[0])

            if action == 'cancel':
                # User cancelled - do nothing
                pass
            elif action == 'create_new' or action == 'reprocess':
                # No NPZ exists OR user wants to reprocess - show import dialog with raw files
                self._show_photometry_import_dialog(file_paths[0])
            elif action == 'modify':
                # User wants to modify existing NPZ - open dialog with NPZ path
                self._show_photometry_import_dialog(file_paths[0], npz_path=data)
            elif action == 'load':
                # User selected a specific experiment to load directly
                npz_path = data['npz_path']
                exp_idx = data['experiment_index']
                self._load_photometry_npz_async(npz_path, exp_idx)

        else:
            # Load data files (ABF, SMRX, EDF)
            if len(file_paths) == 1:
                self.load_file(file_paths[0])
            else:
                self.load_multiple_files(file_paths)

    def _load_recent_file(self, file_path: str):
        """Load a file directly from the recent files menu."""
        from pathlib import Path
        path = Path(file_path)
        if not path.exists():
            self._show_warning("File Not Found", f"The file no longer exists:\n{file_path}")
            return

        # Use the existing file loading logic
        self._load_files([path])

    def _browse_from_folder(self, folder_path: str):
        """Open the file browser starting at a specific folder."""
        from pathlib import Path
        from PyQt6.QtWidgets import QFileDialog

        folder = Path(folder_path)
        if not folder.exists():
            self._show_warning("Folder Not Found", f"The folder no longer exists:\n{folder_path}")
            return

        self.setCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Opening file browser...")
        QApplication.processEvents()

        try:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select File(s)", str(folder),
                "All Supported (*.abf *.smrx *.edf *.pleth.npz *.csv);;Data Files (*.abf *.smrx *.edf);;PhysioMetrics Sessions (*.pleth.npz);;Photometry/CSV (*.csv);;ABF Files (*.abf);;SMRX Files (*.smrx);;EDF Files (*.edf);;All Files (*.*)"
            )
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().clearMessage()

        if not paths:
            return

        file_paths = [Path(p) for p in paths]
        self._load_files(file_paths)

    def _setup_status_history_dropdown(self):
        """Add a subtle dropdown button to the status bar for viewing message history."""
        from PyQt6.QtWidgets import QPushButton, QMenu
        from PyQt6.QtGui import QIcon
        from PyQt6.QtCore import QSize

        # Create a small button with just a "▼" symbol
        self.history_button = QPushButton("📋", self)
        self.history_button.setFixedSize(24, 20)
        self.history_button.setToolTip("View message history")
        self.history_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #d4d4d4;
                font-size: 14px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border-radius: 3px;
            }
        """)
        self.history_button.clicked.connect(self._show_message_history)

        # Add to status bar (right side)
        self.statusBar().addPermanentWidget(self.history_button)

    def _setup_filename_display(self):
        """Add a label to the status bar to display the current filename."""
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtCore import Qt

        # Create label for filename display
        self.filename_label = QLabel("No file loaded", self)
        self.filename_label.setStyleSheet("""
            QLabel {
                color: #d4d4d4;
                padding: 2px 8px;
                margin-right: 10px;
            }
        """)
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Add to status bar (right side, before history button)
        self.statusBar().addPermanentWidget(self.filename_label)

    def _setup_editing_instructions_label(self):
        """Add a label to the status bar for rich text editing instructions."""
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtCore import Qt

        # Create label for editing mode instructions (supports HTML)
        self.editing_instructions_label = QLabel("", self)
        self.editing_instructions_label.setStyleSheet("""
            QLabel {
                color: #d4d4d4;
                padding: 2px 8px;
                font-size: 11px;
            }
        """)
        self.editing_instructions_label.setTextFormat(Qt.TextFormat.RichText)
        self.editing_instructions_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        # Add to status bar (left side - use addWidget, not addPermanentWidget)
        self.statusBar().addWidget(self.editing_instructions_label, 1)  # stretch=1 to take available space

    def _setup_channel_manager(self):
        """Setup the Channel Manager using UI elements from the .ui file."""
        # Hide the old Channel Selection groupbox (replaced by Channel Manager)
        if hasattr(self, 'groupBox_2'):
            self.groupBox_2.hide()

        # Create the ChannelManagerWidget, passing it the UI elements to use
        self.channel_manager = ChannelManagerWidget(
            parent=self,
            summary_label=self.channelManagerSummaryLabel,
            expand_btn=self.channelManagerExpandBtn,
            preview_label=self.channelManagerPreviewLabel
        )

        # Connect signals
        self.channel_manager.apply_requested.connect(self._on_channel_manager_apply)
        self.channel_manager.settings_requested.connect(self._on_channel_settings_requested)

        # Add Mark Events button to Channel Manager groupbox header
        self._setup_mark_events_button()

    def _setup_mark_events_button(self):
        """Add Mark Events button to the Channel Manager header."""
        from PyQt6.QtWidgets import QPushButton
        from PyQt6.QtCore import Qt

        # Create Mark Events button with same styling as old one
        self.mark_events_btn = QPushButton("Mark Events")
        self.mark_events_btn.setFixedSize(70, 18)
        self.mark_events_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                background: transparent;
                text-decoration: underline;
                color: #4A9EFF;
                font-size: 8pt;
            }
            QPushButton:hover {
                color: #6BB6FF;
            }
        """)
        self.mark_events_btn.setToolTip("Open Event Detection dialog (requires Event channel)")
        self.mark_events_btn.clicked.connect(self.on_mark_events_clicked)

        # Hidden — unused in current UI, kept for potential future use
        self.mark_events_btn.setVisible(False)

    def _setup_new_event_markers(self):
        """Setup the new modular event marker system with right-click UX."""
        # Create the viewmodel (central state for markers)
        self._event_marker_viewmodel = EventMarkerViewModel(self)

        # Create the plot integration (connects viewmodel to plot_host)
        self._event_marker_integration = EventMarkerPlotIntegration(
            viewmodel=self._event_marker_viewmodel,
            plot_host=self.plot_host,
            parent=self,
        )

        # Set up callbacks for getting current state
        self._event_marker_integration.set_sweep_callback(lambda: self.state.sweep_idx)
        self._event_marker_integration.set_visible_range_callback(self._get_visible_x_range)
        self._event_marker_integration.set_channel_names_callback(self._get_channel_names_for_markers)
        self._event_marker_integration.set_signal_data_callback(self._get_signal_data_for_markers)

        # Connect signals
        self._event_marker_viewmodel.markers_changed.connect(self._on_new_markers_changed)
        self._event_marker_integration.generate_cta_requested.connect(self._on_generate_cta_requested)

        # Enable the integration (this hooks into the plot's context menu)
        self._event_marker_integration.enable()

        print("[EventMarkers] New event marker system initialized")

    def _get_visible_x_range(self):
        """Get the visible x-axis range from the plot."""
        try:
            main_plot = self.plot_host._get_main_plot()
            if main_plot is not None:
                view_range = main_plot.viewRange()
                return (view_range[0][0], view_range[0][1])
        except Exception:
            pass
        # Fallback to state window
        return (self.state.window_start_s, self.state.window_start_s + 10.0)

    def _get_signal_data_for_markers(self, channel_name: str, sweep_idx: int):
        """
        Get signal data for a channel at a specific sweep.

        Used by marker renderer to draw horizontal intersection lines
        when dragging marker edges.

        IMPORTANT: Returns PROCESSED data (with filters applied) to match
        what's displayed on the plot. Raw data would give incorrect Y values.

        Args:
            channel_name: Name of the channel to get data for
            sweep_idx: Sweep index

        Returns:
            Tuple of (time_array, signal_array) or (None, None) if not found
        """
        st = self.state
        if st.t is None:
            return None, None

        if channel_name in st.sweeps:
            # Use processed data to match what's displayed on the plot
            # (filters, mean subtraction, inversion, etc.)
            y_processed = self._get_processed_for(channel_name, sweep_idx)
            return st.t, y_processed
        return None, None

    def _get_channel_names_for_markers(self):
        """Get list of channel names for auto-detection menu."""
        if self.state.sweeps is None:
            return []
        return list(self.state.sweeps.keys())

    def _on_new_markers_changed(self):
        """Handle markers changed from new system."""
        # Mark state as modified
        self.state.modified = True
        # Update marker count display if we have the UI element
        self._update_marker_count_display()

    def _on_generate_cta_requested(self):
        """Handle request to generate Photometry CTA from context menu."""
        import numpy as np
        from dialogs.photometry_cta_dialog import PhotometryCTADialog
        from viewmodels.cta_viewmodel import CTAViewModel
        from PyQt6.QtWidgets import QMessageBox

        # Get all markers
        markers = list(self._event_marker_viewmodel.store.all())
        if not markers:
            QMessageBox.warning(self, "No Markers", "No event markers available for CTA generation.")
            return

        # Collect continuous signals suitable for CTA
        signals = {}
        metric_labels = {}

        # Get time array first
        time_array = self.state.t if hasattr(self.state, 't') and self.state.t is not None else None
        if time_array is None or len(time_array) == 0:
            QMessageBox.warning(self, "No Time Data", "Time array not available.")
            return

        # Ensure time_array is a 1D numpy array
        time_array = np.asarray(time_array).flatten()
        n_samples = len(time_array)

        # Get current sweep index
        sweep_idx = self.state.sweep_idx

        # Get dF/F channel name if set
        dff_channel_name = getattr(self.state, 'photometry_dff_channel', None)

        # Get all channel names
        channel_names = getattr(self.state, 'channel_names', [])

        if hasattr(self.state, 'sweeps') and self.state.sweeps:
            # Check if sweeps is keyed by channel names (photometry) or sweep indices (ABF)
            first_key = next(iter(self.state.sweeps.keys()), None)

            if isinstance(first_key, str):
                # Photometry structure: sweeps = {'G0-dF/F': data, 'AI-0': data, ...}
                for ch_name in channel_names:
                    if ch_name in self.state.sweeps:
                        ch_data = self.state.sweeps[ch_name]
                        if ch_data is not None:
                            ch_array = np.asarray(ch_data).flatten()
                            if len(ch_array) == n_samples:
                                signals[ch_name] = ch_array
                                if ch_name == dff_channel_name:
                                    metric_labels[ch_name] = f"{ch_name} (dF/F %)"
                                else:
                                    metric_labels[ch_name] = ch_name

            elif isinstance(first_key, int) and sweep_idx in self.state.sweeps:
                # ABF structure: sweeps = {0: {'y': data, ...}, 1: {...}, ...}
                sweep_data = self.state.sweeps[sweep_idx]

                for ch_name in channel_names:
                    if ch_name in sweep_data:
                        ch_data = sweep_data[ch_name]
                        if ch_data is not None:
                            ch_array = np.asarray(ch_data).flatten()
                            if len(ch_array) == n_samples:
                                signals[ch_name] = ch_array
                                if ch_name == dff_channel_name:
                                    metric_labels[ch_name] = f"{ch_name} (dF/F %)"
                                else:
                                    metric_labels[ch_name] = ch_name

                # Also check for 'y' key (legacy pleth format)
                if 'y' in sweep_data and 'y' not in signals:
                    y_data = sweep_data['y']
                    if y_data is not None:
                        y_array = np.asarray(y_data).flatten()
                        if len(y_array) == n_samples:
                            signals['pleth'] = y_array
                            metric_labels['pleth'] = 'Pleth Signal'

        # Add interpolated breath metrics (IF, Ti, Te, Amplitude, etc.)
        # These are per-breath values that need to be interpolated to a continuous signal
        breath_metrics_to_add = ['IF', 'Ti', 'Te', 'PIF', 'PEF', 'VT', 'MV', 'DVDT', 'EF50']
        breath_metric_labels = {
            'IF': 'Instantaneous Frequency (Hz)',
            'Ti': 'Inspiratory Time (s)',
            'Te': 'Expiratory Time (s)',
            'PIF': 'Peak Inspiratory Flow',
            'PEF': 'Peak Expiratory Flow',
            'VT': 'Tidal Volume',
            'MV': 'Minute Ventilation',
            'DVDT': 'dV/dt Max',
            'EF50': 'Expiratory Flow at 50%',
        }

        # Debug: Check for breath metrics
        print(f"[CTA Debug] Checking for breath metrics...")
        print(f"[CTA Debug]   has cached_traces_by_sweep: {hasattr(self, 'cached_traces_by_sweep')}")
        if hasattr(self, 'cached_traces_by_sweep'):
            print(f"[CTA Debug]   cached_traces_by_sweep keys: {list(self.cached_traces_by_sweep.keys()) if self.cached_traces_by_sweep else 'empty'}")
            print(f"[CTA Debug]   sweep_idx {sweep_idx} in cache: {sweep_idx in self.cached_traces_by_sweep}")

        if hasattr(self, 'cached_traces_by_sweep') and sweep_idx in self.cached_traces_by_sweep:
            cached_data = self.cached_traces_by_sweep[sweep_idx]
            print(f"[CTA Debug]   cached_data keys: {list(cached_data.keys())}")

            # Get breath timing info for interpolation
            breath_data = self.state.breath_by_sweep.get(sweep_idx, {})
            onsets = breath_data.get('onsets', [])
            print(f"[CTA Debug]   Number of breath onsets: {len(onsets)}")

            if len(onsets) > 1:
                # Get onset times (convert indices to times)
                onset_times = time_array[onsets] if max(onsets) < len(time_array) else None

                if onset_times is not None and len(onset_times) > 1:
                    for metric_key in breath_metrics_to_add:
                        if metric_key in cached_data:
                            metric_values = cached_data[metric_key]
                            if metric_values is not None and len(metric_values) > 0:
                                # Interpolate to continuous signal
                                # Each breath's metric value is held constant for that breath's duration
                                try:
                                    interp_signal = self._interpolate_breath_metric(
                                        time_array, onset_times, metric_values
                                    )
                                    if interp_signal is not None:
                                        signal_key = f"breath_{metric_key}"
                                        signals[signal_key] = interp_signal
                                        metric_labels[signal_key] = breath_metric_labels.get(metric_key, metric_key)
                                except Exception as e:
                                    print(f"[CTA] Warning: Failed to interpolate {metric_key}: {e}")

        if not signals:
            # Debug: print what we have in state
            print(f"[CTA Debug] sweep_idx: {sweep_idx}")
            print(f"[CTA Debug] state.sweeps keys: {list(self.state.sweeps.keys()) if self.state.sweeps else 'None'}")
            print(f"[CTA Debug] state.channel_names: {getattr(self.state, 'channel_names', 'Not set')}")
            if self.state.sweeps and sweep_idx in self.state.sweeps:
                print(f"[CTA Debug] sweep_data keys: {list(self.state.sweeps[sweep_idx].keys())}")
                for k, v in self.state.sweeps[sweep_idx].items():
                    if hasattr(v, '__len__'):
                        print(f"[CTA Debug]   {k}: len={len(v)}, type={type(v).__name__}")
                    else:
                        print(f"[CTA Debug]   {k}: {type(v).__name__}")
            print(f"[CTA Debug] n_samples (time_array): {n_samples}")

            # Try to offer channel selection as fallback
            if self.state.sweeps and sweep_idx in self.state.sweeps:
                sweep_data = self.state.sweeps[sweep_idx]
                available_channels = []
                for key, value in sweep_data.items():
                    if value is not None and hasattr(value, '__len__') and len(value) > 100:
                        available_channels.append((key, len(value)))

                if available_channels:
                    from PyQt6.QtWidgets import QInputDialog
                    items = [f"{name} ({length} samples)" for name, length in available_channels]
                    item, ok = QInputDialog.getItem(
                        self, "Select Signal",
                        f"No matching signals found (expected {n_samples} samples).\n\n"
                        "Available data arrays:",
                        items, 0, False
                    )
                    if ok and item:
                        # Extract channel name from selection
                        selected_name = available_channels[items.index(item)][0]
                        ch_data = sweep_data[selected_name]
                        signals[selected_name] = np.asarray(ch_data).flatten()
                        metric_labels[selected_name] = selected_name
                        # Also update time_array to match if needed
                        if len(signals[selected_name]) != n_samples:
                            # Create matching time array
                            time_array = np.linspace(0, len(signals[selected_name]) / self.state.sr_hz, len(signals[selected_name]))
                            n_samples = len(time_array)

            if not signals:
                QMessageBox.warning(
                    self, "No Signals",
                    "No continuous signal data available for CTA generation.\n\n"
                    "Please load data with photometry or analog channels."
                )
                return

        # Debug: Print summary of available signals
        print(f"[CTA Debug] Final signals available for CTA:")
        for key in signals.keys():
            label = metric_labels.get(key, key)
            print(f"[CTA Debug]   {key}: {label}")

        # Extract channel line colors from the actual plot
        channel_colors = {}
        if hasattr(self, 'plot_host') and hasattr(self.plot_host, '_subplots'):
            for plot in self.plot_host._subplots:
                if plot is None:
                    continue
                try:
                    label = plot.getAxis('left').labelText or ''
                    # Strip scale factor suffix
                    base_label = label.split('(')[0].strip() if '(' in label else label
                    for item in plot.listDataItems():
                        if hasattr(item, 'getData') and hasattr(item, 'opts'):
                            x, y = item.getData()
                            if x is not None and len(x) > 100:
                                pen = item.opts.get('pen')
                                if pen and hasattr(pen, 'color'):
                                    channel_colors[base_label] = pen.color().name()
                                break
                except Exception:
                    pass

        # Create and show the CTA dialog
        cta_viewmodel = CTAViewModel(self)
        dialog = PhotometryCTADialog(
            parent=self,
            viewmodel=cta_viewmodel,
            markers=markers,
            signals=signals,
            time_array=time_array,
            metric_labels=metric_labels,
            channel_colors=channel_colors,
        )
        dialog.exec()

    def _update_marker_count_display(self):
        """Update the marker count label in the UI."""
        if hasattr(self, 'eventCountLabel'):
            count = self._event_marker_viewmodel.marker_count
            sweep_count = self._event_marker_viewmodel.get_marker_count_for_sweep(self.state.sweep_idx)
            self.eventCountLabel.setText(f"{sweep_count} / {count}")

    def _interpolate_breath_metric(
        self,
        time_array: np.ndarray,
        onset_times: np.ndarray,
        metric_values: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Interpolate per-breath metric values to a continuous time series.

        Each breath's metric value is held constant from that breath's onset
        to the next breath's onset (step function / sample-and-hold).

        Args:
            time_array: Full time array for the recording
            onset_times: Array of breath onset times
            metric_values: Array of metric values (one per breath)

        Returns:
            Interpolated signal with same length as time_array, or None if error
        """
        import numpy as np

        # Validate inputs
        if len(onset_times) < 2 or len(metric_values) < 1:
            return None

        # Ensure arrays
        onset_times = np.asarray(onset_times).flatten()
        metric_values = np.asarray(metric_values).flatten()

        # Number of values should match number of inter-onset intervals
        # metric_values[i] corresponds to breath from onset_times[i] to onset_times[i+1]
        n_breaths = min(len(onset_times) - 1, len(metric_values))

        if n_breaths < 1:
            return None

        # Create output array initialized to NaN
        result = np.full(len(time_array), np.nan)

        # Fill in values for each breath interval
        for i in range(n_breaths):
            t_start = onset_times[i]
            t_end = onset_times[i + 1] if i + 1 < len(onset_times) else time_array[-1]

            # Find indices for this breath interval
            mask = (time_array >= t_start) & (time_array < t_end)
            result[mask] = metric_values[i]

        # Handle the last breath - extend to end of recording if needed
        if n_breaths > 0 and n_breaths < len(metric_values):
            last_onset = onset_times[n_breaths] if n_breaths < len(onset_times) else onset_times[-1]
            mask = time_array >= last_onset
            if np.any(mask) and n_breaths < len(metric_values):
                result[mask] = metric_values[n_breaths]

        return result

    def _refresh_event_markers(self, force: bool = False):
        """Refresh event marker display for the current sweep.

        Args:
            force: If True, refresh even if sweep hasn't changed (e.g. after adding markers).
        """
        # Skip if matplotlib backend (event markers only work with pyqtgraph)
        if self._current_backend == 'matplotlib':
            return

        if hasattr(self, '_event_marker_integration') and self._event_marker_integration.enabled:
            try:
                # Calculate time offset (same logic as in plot_manager.py)
                # When stimulus channel is active, plot time is normalized to first stim onset
                st = self.state
                s = st.sweep_idx
                t0 = 0.0
                if st.stim_chan and st.stim_spans_by_sweep:
                    spans = st.stim_spans_by_sweep.get(s, [])
                    if spans:
                        t0 = spans[0][0]

                # Skip if sweep + time offset haven't changed (avoids redundant refresh on zoom/pan)
                refresh_key = (s, t0)
                if not force and getattr(self, '_last_marker_refresh_key', None) == refresh_key:
                    return
                self._last_marker_refresh_key = refresh_key

                # Set the time offset on the integration before refreshing
                self._event_marker_integration.set_time_offset(t0)
                self._event_marker_integration.refresh()
                self._update_marker_count_display()
            except RuntimeError as e:
                # Plot host may have been deleted during backend switch
                if "deleted" in str(e):
                    print(f"[EventMarkers] Skipping refresh - plot host was deleted")
                else:
                    raise

    def _setup_event_markers_widget(self):
        """Setup the Event Markers QGroupBox widgets (defined in .ui file)."""
        from core.event_types import get_registry, get_all_event_types

        # Check if groupbox exists (widgets are defined in .ui file)
        if not hasattr(self, 'eventMarkersGroupBox'):
            print("Warning: eventMarkersGroupBox not found in UI")
            return

        # Hide the event markers groupbox (feature not fully released)
        self.eventMarkersGroupBox.hide()
        return

        # Populate event type combo
        self._populate_event_type_combo()

        # Apply styling to widgets from .ui file
        self._apply_event_markers_styling()

        # Connect signals
        self._connect_event_marker_signals()

    def _apply_event_markers_styling(self):
        """Apply dark theme styling to Event Markers widgets."""
        # Groupbox styling
        self.eventMarkersGroupBox.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 9pt;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)

        # Combo box styling
        combo_style = """
            QComboBox {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px 6px;
                font-size: 9pt;
            }
            QComboBox:hover { border-color: #007acc; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #d4d4d4; }
        """

        if hasattr(self, 'eventTypeCombo'):
            self.eventTypeCombo.setStyleSheet(combo_style)
        if hasattr(self, 'eventSourceChannelCombo'):
            self.eventSourceChannelCombo.setStyleSheet(combo_style)
        if hasattr(self, 'showTypesCombo'):
            self.showTypesCombo.setStyleSheet(combo_style)

        # Button styling
        button_style = """
            QPushButton {
                background-color: #3c3c3c;
                color: #d4d4d4;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 9pt;
            }
            QPushButton:hover { background-color: #4a4a4d; border-color: #007acc; }
        """

        if hasattr(self, 'newTypeBtn'):
            self.newTypeBtn.setStyleSheet(button_style + "QPushButton { font-weight: bold; }")
        if hasattr(self, 'detectSettingsBtn'):
            self.detectSettingsBtn.setStyleSheet(button_style + "QPushButton { font-size: 11pt; }")
        if hasattr(self, 'clearMarkersBtn'):
            self.clearMarkersBtn.setStyleSheet("""
                QPushButton {
                    background-color: #3c3c3c;
                    color: #d4d4d4;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 4px 8px;
                    font-size: 9pt;
                }
                QPushButton:hover { background-color: #4a4a4d; border-color: #c75050; color: #ff6b6b; }
            """)

        # Detect button (accent color)
        if hasattr(self, 'autoDetectBtn'):
            self.autoDetectBtn.setStyleSheet("""
                QPushButton {
                    background-color: #0e639c;
                    color: white;
                    border: none;
                    border-radius: 3px;
                    padding: 4px 12px;
                    font-size: 9pt;
                }
                QPushButton:hover { background-color: #1177bb; }
                QPushButton:pressed { background-color: #0d5a8c; }
            """)

        # Checkbox styling
        checkbox_style = """
            QCheckBox {
                color: #d4d4d4;
                font-size: 9pt;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #555555;
                border-radius: 2px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QCheckBox::indicator:hover { border-color: #007acc; }
        """

        for checkbox_name in ['showMarkersCheck', 'shadeRegionsCheck', 'showLabelsCheck', 'spanAllChannelsCheck']:
            if hasattr(self, checkbox_name):
                getattr(self, checkbox_name).setStyleSheet(checkbox_style)

        # Label styling
        label_style = "color: #d4d4d4; font-size: 9pt;"
        for label_name in ['eventTypeLabel', 'eventChannelLabel', 'showTypesLabel']:
            if hasattr(self, label_name):
                getattr(self, label_name).setStyleSheet(label_style)

        # Event count label (accent color)
        if hasattr(self, 'eventCountLabel'):
            self.eventCountLabel.setStyleSheet("color: #9cdcfe; font-size: 9pt;")

        # Tip label (muted)
        if hasattr(self, 'eventMarkersTipLabel'):
            self.eventMarkersTipLabel.setStyleSheet("color: #808080; font-size: 8pt; font-style: italic;")

    def _populate_event_type_combo(self):
        """Populate the event type dropdown with available types."""
        from core.event_types import get_all_event_types, get_event_color

        self.eventTypeCombo.clear()

        for event_type in get_all_event_types():
            # Add item with colored icon indicator
            self.eventTypeCombo.addItem(event_type.display_name, event_type.name)

        # Set to active type from state
        if hasattr(self, 'state') and self.state.active_event_type:
            idx = self.eventTypeCombo.findData(self.state.active_event_type)
            if idx >= 0:
                self.eventTypeCombo.setCurrentIndex(idx)

    def _connect_event_marker_signals(self):
        """Connect signals for event marker widgets."""
        # Event type selection changed
        self.eventTypeCombo.currentIndexChanged.connect(self._on_event_type_changed)

        # New type button
        self.newTypeBtn.clicked.connect(self._on_new_event_type_clicked)

        # Auto-detect button
        self.autoDetectBtn.clicked.connect(self._on_auto_detect_events_clicked)

        # Detection settings button
        self.detectSettingsBtn.clicked.connect(self._on_detect_settings_clicked)

        # Display options
        self.showMarkersCheck.toggled.connect(self._on_event_display_changed)
        self.shadeRegionsCheck.toggled.connect(self._on_event_display_changed)
        self.showLabelsCheck.toggled.connect(self._on_event_display_changed)
        self.spanAllChannelsCheck.toggled.connect(self._on_event_display_changed)

        # Filter combo
        self.showTypesCombo.currentIndexChanged.connect(self._on_event_display_changed)

        # Clear button
        self.clearMarkersBtn.clicked.connect(self._on_clear_markers_clicked)

    def _on_event_type_changed(self, index):
        """Handle event type selection change."""
        if index < 0:
            return
        type_name = self.eventTypeCombo.itemData(index)
        if type_name:
            self.state.active_event_type = type_name
            # Update usage count
            from core.event_types import get_registry
            get_registry().increment_usage(type_name)

    def _on_new_event_type_clicked(self):
        """Show dialog to create new event type."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox, QColorDialog, QPushButton
        from PyQt6.QtGui import QColor

        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Event Type")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g., startle_response")
        form.addRow("Name:", name_edit)

        display_edit = QLineEdit()
        display_edit.setPlaceholderText("e.g., Startle Response")
        form.addRow("Display Name:", display_edit)

        color_btn = QPushButton("#808080")
        color_btn.setStyleSheet("background-color: #808080; color: white;")
        selected_color = ["#808080"]

        def pick_color():
            color = QColorDialog.getColor(QColor(selected_color[0]), dialog)
            if color.isValid():
                selected_color[0] = color.name()
                color_btn.setText(color.name())
                color_btn.setStyleSheet(f"background-color: {color.name()}; color: white;")

        color_btn.clicked.connect(pick_color)
        form.addRow("Color:", color_btn)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_edit.text().strip().lower().replace(" ", "_")
            display_name = display_edit.text().strip() or name.replace("_", " ").title()
            color = selected_color[0]

            if name:
                from core.event_types import create_custom_type
                if create_custom_type(name, display_name, color):
                    self._populate_event_type_combo()
                    # Select the new type
                    idx = self.eventTypeCombo.findData(name)
                    if idx >= 0:
                        self.eventTypeCombo.setCurrentIndex(idx)
                    self._show_status(f"Created event type: {display_name}")
                else:
                    self._show_status(f"Event type '{name}' already exists", 3000)

    def _on_auto_detect_events_clicked(self):
        """Auto-detect events using the current type's detection method."""
        st = self.state

        # Get selected source channel from the Event Markers widget (not the Event channel type)
        if not hasattr(self, 'eventSourceChannelCombo'):
            return

        source_channel = self.eventSourceChannelCombo.currentText()
        if not source_channel or source_channel not in st.sweeps:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Channel Selected",
                "Please select a source channel for event detection."
            )
            return

        # Get current event type
        event_type = self.eventTypeCombo.currentData() or 'lick_bout'

        # Get type configuration for detection parameters
        from core.event_types import get_registry
        type_config = get_registry().get_type(event_type)

        if type_config is None:
            self._show_status(f"Unknown event type: {event_type}", 3000)
            return

        # For now, use threshold-based detection for all types
        # This will be expanded to support different detection methods
        detection_method = type_config.detection_method
        params = type_config.default_params

        # Run detection
        self._run_event_detection(source_channel, event_type, detection_method, params)

    def _run_event_detection(self, source_channel: str, event_type: str, method: str, params: dict):
        """Run event detection on the specified channel."""
        import numpy as np
        st = self.state

        # Get threshold parameters (with defaults)
        threshold = params.get('threshold', 0.5)
        min_duration_ms = params.get('min_duration_ms', 50)
        min_gap_s = params.get('min_gap_s', 1.0)

        min_duration_samples = int(min_duration_ms / 1000 * st.sr_hz) if st.sr_hz else 50

        total_events = 0
        n_sweeps = st.sweeps[source_channel].shape[1] if len(st.sweeps[source_channel].shape) > 1 else 1

        for sweep_idx in range(n_sweeps):
            # Get signal data
            if len(st.sweeps[source_channel].shape) > 1:
                signal = st.sweeps[source_channel][:, sweep_idx]
            else:
                signal = st.sweeps[source_channel]

            t = st.t

            # Find threshold crossings
            above = signal > threshold
            crossings = np.diff(above.astype(int))

            # Rising edges (onset) and falling edges (offset)
            onsets = np.where(crossings == 1)[0] + 1
            offsets = np.where(crossings == -1)[0] + 1

            # Pair onsets with offsets
            events = []
            for onset_idx in onsets:
                # Find next offset after this onset
                valid_offsets = offsets[offsets > onset_idx]
                if len(valid_offsets) > 0:
                    offset_idx = valid_offsets[0]

                    # Check minimum duration
                    if offset_idx - onset_idx >= min_duration_samples:
                        start_time = t[onset_idx]
                        end_time = t[offset_idx]
                        events.append((start_time, end_time))

            # Merge events that are closer than min_gap
            if events and min_gap_s > 0:
                merged = [events[0]]
                for start, end in events[1:]:
                    prev_start, prev_end = merged[-1]
                    if start - prev_end < min_gap_s:
                        # Merge with previous
                        merged[-1] = (prev_start, end)
                    else:
                        merged.append((start, end))
                events = merged

            # Add events to marker manager
            for start_time, end_time in events:
                st.event_markers.add_paired_marker(
                    sweep_idx=sweep_idx,
                    start_time=start_time,
                    end_time=end_time,
                    event_type=event_type,
                    source_channel=source_channel,
                    detection_method=method
                )
                total_events += 1

        # Update UI
        self._update_event_count_label()
        self.refresh_plot()

        # Increment usage count for this type
        from core.event_types import get_registry
        get_registry().increment_usage(event_type)

        self._show_status(f"Detected {total_events} {event_type} events across {n_sweeps} sweep(s)")

    def _on_detect_settings_clicked(self):
        """Show detection settings dialog for current event type."""
        # For now, open the existing event detection dialog
        # In the future, this will open a type-specific settings dialog
        # Check if we have a source channel selected
        if hasattr(self, 'eventSourceChannelCombo'):
            source_channel = self.eventSourceChannelCombo.currentText()
            # Temporarily set as event channel for the legacy dialog
            if source_channel and source_channel in self.state.sweeps:
                old_event_channel = self.state.event_channel
                self.state.event_channel = source_channel
                self.on_mark_events_clicked()
                # Note: event_channel will stay set, which is fine
                return

        # Fallback to legacy behavior
        self.on_mark_events_clicked()

    def _on_event_display_changed(self):
        """Handle changes to event display options."""
        # Refresh the plot to reflect display changes
        if hasattr(self, 'refresh_plot'):
            self.refresh_plot()

    def _on_clear_markers_clicked(self):
        """Show dialog to clear event markers."""
        from PyQt6.QtWidgets import QMessageBox

        # Get current type
        type_name = self.eventTypeCombo.currentData()
        type_display = self.eventTypeCombo.currentText()

        msg = QMessageBox(self)
        msg.setWindowTitle("Clear Event Markers")
        msg.setText("Which markers do you want to clear?")
        msg.setIcon(QMessageBox.Icon.Question)

        clear_type_btn = msg.addButton(f"Clear '{type_display}' (this sweep)", QMessageBox.ButtonRole.ActionRole)
        clear_type_all_btn = msg.addButton(f"Clear '{type_display}' (all sweeps)", QMessageBox.ButtonRole.ActionRole)
        clear_all_btn = msg.addButton("Clear ALL markers", QMessageBox.ButtonRole.DestructiveRole)
        cancel_btn = msg.addButton(QMessageBox.StandardButton.Cancel)

        msg.exec()

        clicked = msg.clickedButton()
        if clicked == cancel_btn:
            return

        sweep_idx = self.state.sweep_idx

        if clicked == clear_type_btn:
            self.state.event_markers.clear_type(type_name, sweep_idx)
            self._show_status(f"Cleared '{type_display}' markers from sweep {sweep_idx + 1}")
        elif clicked == clear_type_all_btn:
            self.state.event_markers.clear_type(type_name)
            self._show_status(f"Cleared all '{type_display}' markers")
        elif clicked == clear_all_btn:
            self.state.event_markers.clear()
            self._show_status("Cleared all event markers")

        self._update_event_count_label()
        self.refresh_plot()

    def _update_event_count_label(self):
        """Update the event count label."""
        if not hasattr(self, 'eventCountLabel'):
            return

        sweep_idx = self.state.sweep_idx
        sweep_count = self.state.event_markers.count_markers(sweep_idx)
        total_count = self.state.event_markers.count_markers()

        if total_count == 0:
            self.eventCountLabel.setText("Events: 0")
        elif sweep_count == total_count:
            self.eventCountLabel.setText(f"Events: {total_count}")
        else:
            self.eventCountLabel.setText(f"Events: {sweep_count} (sweep) / {total_count} (total)")

    def _update_event_source_channel_combo(self):
        """Update the source channel dropdown with available channels."""
        if not hasattr(self, 'eventSourceChannelCombo'):
            return

        self.eventSourceChannelCombo.clear()
        st = self.state

        # Add available channels
        if st.channel_names:
            for name in st.channel_names:
                self.eventSourceChannelCombo.addItem(name)

            # Select event channel if set
            if st.event_channel and st.event_channel in st.channel_names:
                idx = st.channel_names.index(st.event_channel)
                self.eventSourceChannelCombo.setCurrentIndex(idx)

    def _on_channel_manager_apply(self):
        """Called when channel manager requests to apply changes."""
        st = self.state

        # Get channel assignments from channel manager
        pleth_channel = self.channel_manager.get_pleth_channel()
        opto_stim_channel = self.channel_manager.get_opto_stim_channel()
        event_channel = self.channel_manager.get_event_channel()

        # Check if Pleth channel changed - if so, clear analysis data
        channel_changed = (pleth_channel != st.analyze_chan)

        if pleth_channel:
            if channel_changed:

                # Clear peaks and breath data (same as on_analyze_channel_changed)
                st.peaks_by_sweep.clear()
                st.sigh_by_sweep.clear()
                if hasattr(st, 'breath_by_sweep'):
                    st.breath_by_sweep.clear()

                # Clear omitted sweeps and regions
                st.omitted_sweeps.clear()
                st.omitted_ranges.clear()
                self._refresh_omit_button_label()

                # Clear sniffing regions
                if hasattr(st, 'sniff_regions_by_sweep'):
                    st.sniff_regions_by_sweep.clear()

                # Clear event markers (bout annotations)
                if hasattr(st, 'bout_annotations'):
                    st.bout_annotations.clear()

                # Clear z-score global statistics cache
                self.zscore_global_mean = None
                self.zscore_global_std = None

                # Clear Y2 plot data (same as on_analyze_channel_changed)
                st.y2_metric_key = None
                st.y2_values_by_sweep.clear()
                self.plot_host.clear_y2()
                # Reset Y2 dropdown to "None"
                self.y2plot_dropdown.blockSignals(True)
                self.y2plot_dropdown.setCurrentIndex(0)  # First item is "None"
                self.y2plot_dropdown.blockSignals(False)

            st.analyze_chan = pleth_channel

            # Clear processing cache so filters are re-applied
            st.proc_cache.clear()

            # Switch to single panel mode (required for proper navigation)
            if not self.single_panel_mode:
                self.single_panel_mode = True

            # Clear saved view to force fresh autoscale
            self.plot_host.clear_saved_view("single")
            self.plot_host.clear_saved_view("grid")

            # Auto-detect prominence threshold (required for peak detection)
            self._auto_detect_prominence_silent()

            # Enable peak detection button (always enable when channel changes or is set)
            self.ApplyPeakFindPushButton.setEnabled(True)
        else:
            # No Pleth channel - still clear saved view to show full data range
            self.plot_host.clear_saved_view("single")
            self.plot_host.clear_saved_view("grid")

        # Update Opto Stim channel (for blue stimulus overlays)
        stim_changed = (opto_stim_channel != st.stim_chan)
        if stim_changed:
            st.stim_chan = opto_stim_channel
            # Clear stimulus detection results when channel changes
            st.stim_onsets_by_sweep.clear()
            st.stim_offsets_by_sweep.clear()
            st.stim_spans_by_sweep.clear()
            st.stim_metrics_by_sweep.clear()
            # Recompute stimulus for current sweep if channel is set
            if st.stim_chan:
                self._compute_stim_for_current_sweep()

        # Update Event channel (for event marking)
        event_changed = (event_channel != st.event_channel)
        if event_changed:
            st.event_channel = event_channel
            # Clear bout annotations when event channel changes
            if hasattr(st, 'bout_annotations'):
                st.bout_annotations.clear()

        # Redraw the plot with new channel configuration
        self.redraw_main_plot()

    def _on_channel_settings_requested(self, channel_name: str):
        """Called when user clicks settings (gear) icon for a channel."""
        print(f"[ChannelManager] Settings requested for channel: '{channel_name}'")

        # Look up the channel config from channel manager
        configs = self.channel_manager.get_all_configs()
        if channel_name not in configs:
            print(f"[ChannelManager] Warning: No config found for channel '{channel_name}'")
            print(f"[ChannelManager]   Available channels: {list(configs.keys())}")
            return

        config = configs[channel_name]
        print(f"[ChannelManager] Config found: source={config.source}, callback={config.settings_callback is not None}")

        # Call the settings callback if one was registered
        if config.settings_callback is not None:
            print(f"[ChannelManager] Calling settings callback...")
            config.settings_callback()
        else:
            print(f"[ChannelManager] No settings callback for channel '{channel_name}'")

    def _populate_channel_manager(self, ch_names: list):
        """Populate the channel manager widget with available channels."""
        from core.channel_manager import ChannelConfig

        # Get the currently selected analyze channel (from old dropdown)
        analyze_chan = self.state.analyze_chan

        configs = []
        for i, name in enumerate(ch_names):
            # Auto-detect channel type from name
            name_lower = name.lower()
            if any(kw in name_lower for kw in ['pleth', 'resp', 'breath', 'flow', 'airway']):
                channel_type = "Pleth"
            elif any(kw in name_lower for kw in ['opto', 'stim', 'laser', 'led', 'light']):
                channel_type = "Opto Stim"
            else:
                channel_type = "Raw Signal"

            configs.append(ChannelConfig(
                name=name,
                visible=True,  # All channels visible by default as preview
                channel_type=channel_type,
                source="file",
                order=i
            ))

        self.channel_manager.set_channels(configs)

    def _show_editing_instructions(self, html: str):
        """Show rich text editing instructions in the status bar."""
        self.editing_instructions_label.setText(html)
        self._editing_instructions_html = html  # Track for restoration (separate from plain text messages)

    def _clear_editing_instructions(self):
        """Clear the editing instructions from status bar."""
        self.editing_instructions_label.setText("")
        self._editing_instructions_html = None

    def _restore_editing_instructions(self):
        """Restore the editing instructions if any are active."""
        if getattr(self, '_editing_instructions_html', None):
            self.editing_instructions_label.setText(self._editing_instructions_html)

    def _update_filename_display(self):
        """Update the filename display in status bar with file metadata."""
        st = self.state
        if not hasattr(st, 'file_info') or not st.file_info:
            self.filename_label.setText("No file loaded")
            return

        file_info = st.file_info[0]

        # Handle None path (e.g., photometry data without saved NPZ)
        if file_info.get('path') is None:
            # Check for display_name override
            if file_info.get('display_name'):
                filename = file_info['display_name']
            else:
                filename = "Photometry Data"
        else:
            filename = file_info['path'].name

        # Build metadata string for ABF files
        metadata_parts = []
        if file_info.get('protocol'):
            metadata_parts.append(file_info['protocol'])
        if file_info.get('n_channels'):
            metadata_parts.append(f"{file_info['n_channels']} ch")
        if file_info.get('file_type') == 'photometry':
            metadata_parts.append("Photometry")

        if len(st.file_info) == 1:
            if metadata_parts:
                self.filename_label.setText(f"File: {filename}  [{', '.join(metadata_parts)}]")
            else:
                self.filename_label.setText(f"File: {filename}")
        else:
            if metadata_parts:
                self.filename_label.setText(f"Files: {len(st.file_info)} files ({filename}, ...)  [{', '.join(metadata_parts)}]")
            else:
                self.filename_label.setText(f"Files: {len(st.file_info)} files ({filename}, ...)")

    def _enable_dark_title_bar(self, widget=None):
        """Enable dark title bar on Windows 10/11.

        Args:
            widget: QWidget to apply dark title bar to. If None, applies to self.
        """
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int

                # DWMWA_USE_IMMERSIVE_DARK_MODE
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20

                # Get window handle
                target = widget if widget else self
                hwnd = int(target.winId())

                # Set dark mode (1 = dark, 0 = light)
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    DWMWA_USE_IMMERSIVE_DARK_MODE,
                    byref(value),
                    sizeof(value)
                )
            except Exception as e:
                # Silently fail if not supported (Windows 10 older builds)
                pass

    def _show_message_history(self):
        """Show a menu with recent status bar messages."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
            }
            QMenu::item:selected {
                background-color: #3e3e42;
            }
        """)

        if not self._status_message_history:
            action = QAction("No messages yet", self)
            action.setEnabled(False)
            menu.addAction(action)
        else:
            # Show last 20 messages, most recent first
            for i, (timestamp, message) in enumerate(reversed(self._status_message_history[-20:])):
                action = QAction(f"{timestamp} - {message}", self)
                action.setEnabled(False)  # Not clickable, just for display
                menu.addAction(action)

        # Show menu below the button
        menu.exec(self.history_button.mapToGlobal(self.history_button.rect().bottomLeft()))

    def _show_error(self, title: str, message: str):
        """Show an error dialog with selectable text for easy copying."""
        msg = QMessageBox(QMessageBox.Icon.Critical, title, message,
                         QMessageBox.StandardButton.Ok, self)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                   Qt.TextInteractionFlag.TextSelectableByKeyboard)
        msg.exec()

    def _show_warning(self, title: str, message: str):
        """Show a warning dialog with selectable text for easy copying."""
        msg = QMessageBox(QMessageBox.Icon.Warning, title, message,
                         QMessageBox.StandardButton.Ok, self)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                   Qt.TextInteractionFlag.TextSelectableByKeyboard)
        msg.exec()

    def _show_info(self, title: str, message: str):
        """Show an info dialog with selectable text for easy copying."""
        msg = QMessageBox(QMessageBox.Icon.Information, title, message,
                         QMessageBox.StandardButton.Ok, self)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse |
                                   Qt.TextInteractionFlag.TextSelectableByKeyboard)
        msg.exec()

    def _log_status_message(self, message: str, timeout: int = 0):
        """Log a status message and show it on the status bar."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self._status_message_history.append((timestamp, message))

        # Keep only last 100 messages to avoid memory growth
        if len(self._status_message_history) > 100:
            self._status_message_history = self._status_message_history[-100:]

        # Show on status bar
        self.statusBar().showMessage(message, timeout)

    def keyPressEvent(self, event):
        """Handle keyboard events - delegate to editing modes first."""
        from PyQt6.QtCore import Qt

        # Try editing modes handler first
        if self.editing_modes.handle_key_press_event(event):
            event.accept()
            return

        # Fall back to default handling
        super().keyPressEvent(event)

    def on_ctrl_o_pressed(self):
        """Handle Ctrl+O shortcut - triggers different buttons based on active tab."""
        current_tab = self.Tabs.currentIndex()
        if current_tab == 0:  # Analysis tab
            self.on_browse_clicked()
        elif current_tab == 1:  # Curation tab
            self.on_curation_choose_dir_clicked()

    def _is_focus_in_text_widget(self) -> bool:
        """Check if focus is in a text input widget (to avoid shortcuts while typing)."""
        from PyQt6.QtWidgets import QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox
        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return False
        # Check if it's a text input type
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox)):
            return True
        # Check if it's an editable combobox
        if isinstance(focus_widget, QComboBox) and focus_widget.isEditable():
            return True
        return False

    def _on_z_pressed(self):
        """Handle Z key - navigate to previous sweep/window."""
        if self._is_focus_in_text_widget():
            return  # Don't navigate while typing
        # Analysis tab is at index 1 (Project Builder=0, Analysis=1, Curation=2)
        if self.Tabs.currentIndex() == 1:
            self.navigation_manager.on_unified_prev()

    def _on_x_pressed(self):
        """Handle X key - navigate to next sweep/window."""
        if self._is_focus_in_text_widget():
            return  # Don't navigate while typing
        # Analysis tab is at index 1 (Project Builder=0, Analysis=1, Curation=2)
        if self.Tabs.currentIndex() == 1:
            self.navigation_manager.on_unified_next()

    def _on_hot_reload(self):
        """Hot reload development modules (Ctrl+R).

        Reloads dialog and utility modules so code changes take effect
        the next time you open a dialog. Also refreshes the main plot
        so styling changes (colors, titles) appear immediately.

        Already-open dialogs keep old code - close and reopen them.
        """
        import importlib
        import sys

        # Modules to reload - order matters (dependencies first)
        modules_to_reload = [
            # Core utilities
            'core.photometry',
            'core.peaks',
            'core.metrics',
            'core.plot_themes',
            'core.plotting',

            # Photometry subsystem
            'dialogs.photometry.experiment_plotter',
            'dialogs.photometry.data_assembly_widget',
            'dialogs.photometry.chooser_dialog',
            'dialogs.photometry.processing_widget',
            'dialogs.photometry',

            # Dialogs
            'dialogs.peak_detection_dialog',
            'dialogs.advanced_peak_editor_dialog',
            'dialogs.gmm_clustering_dialog',
            'dialogs.prominence_threshold_dialog',
            'dialogs.photometry_import_dialog',
            'dialogs.photometry_import_dialog_v2',

            # Plotting
            'plotting.pyqtgraph_backend',
            'plotting.plot_manager',

            # Export
            'export.export_manager',

            # Views and editing
            'views.events.marker_renderer',
            'views.events.marker_editor',
            'views.events.context_menu',
            'views.events.plot_integration',
            'editing.editing_modes',
            'editing.event_marking_mode',

            # Event marker system
            'viewmodels.event_marker_viewmodel',
            'core.domain.events.models',
            'core.domain.events.store',
            'core.domain.events.registry',
            'core.services.event_marker_service',

            # CTA system
            'core.domain.cta.models',
            'core.services.cta_service',
            'viewmodels.cta_viewmodel',
            'dialogs.photometry_cta_dialog',
            'dialogs.export_mixin',

            # Detection framework
            'core.detection.base',
            'core.detection.threshold',
            'core.detection',
            'dialogs.marker_detection_dialog',
        ]

        reloaded = []
        failed = []

        for module_name in modules_to_reload:
            if module_name in sys.modules:
                try:
                    module = sys.modules[module_name]
                    importlib.reload(module)
                    reloaded.append(module_name)
                except Exception as e:
                    failed.append(f"{module_name}: {e}")

        # Auto-refresh the main plot to show styling changes immediately
        plot_refreshed = False
        if reloaded and self.state.in_path:
            try:
                self.redraw_main_plot()
                plot_refreshed = True
            except Exception as e:
                print(f"[Hot Reload] Plot refresh failed: {e}")

        # Show status in status bar
        if reloaded:
            msg = f"Hot reload: {len(reloaded)} modules"
            if plot_refreshed:
                msg += " + plot refreshed"
            if failed:
                msg += f" ({len(failed)} failed)"
            self.statusBar().showMessage(msg, 5000)
            print(f"[Hot Reload] Reloaded: {', '.join(reloaded)}")
            if failed:
                print(f"[Hot Reload] Failed: {', '.join(failed)}")
        else:
            self.statusBar().showMessage("Hot reload: No modules to reload", 3000)

    def on_browse_clicked(self):
        last_dir = self.settings.value("last_dir", str(Path.home()))
        # Skip exists() check - it's slow on network drives and the dialog handles it gracefully
        if last_dir and not (last_dir.startswith('\\\\') or (len(last_dir) > 1 and last_dir[1] == ':')):
            # Only check exists() for non-network, non-drive paths
            if not Path(str(last_dir)).exists():
                last_dir = str(Path.home())

        # Show wait cursor while file dialog loads (can be slow on network drives)
        self.setCursor(Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage("Opening file browser...")
        QApplication.processEvents()

        try:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Select File(s)", last_dir,
                "All Supported (*.abf *.smrx *.edf *.pleth.npz *.csv);;Data Files (*.abf *.smrx *.edf);;PhysioMetrics Sessions (*.pleth.npz);;Photometry/CSV (*.csv);;ABF Files (*.abf);;SMRX Files (*.smrx);;EDF Files (*.edf);;All Files (*.*)"
            )
        finally:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().clearMessage()

        if not paths:
            return

        # Convert to Path objects
        file_paths = [Path(p) for p in paths]

        # Store the directory of the first file
        self.settings.setValue("last_dir", str(file_paths[0].parent))

        # Load the files using the common loading method
        self._load_files(file_paths)

    def _show_photometry_import_dialog(self, initial_path: Path, npz_path: Path = None):
        """
        Show the photometry import wizard dialog.

        Args:
            initial_path: Path to the detected photometry file (FP_data*.csv)
            npz_path: If provided, open directly to Tab 2 with this NPZ file
        """
        dialog = PhotometryImportDialog(self, initial_path, photometry_npz_path=npz_path)
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            # Get the processed data
            result_data = dialog.get_result_data()

            if result_data is None:
                print("[Photometry] No result data from dialog")
                return

            # Load photometry data into the app
            self._load_photometry_data(result_data)

    def _load_photometry_npz_async(self, npz_path: Path, exp_idx: int):
        """Load a photometry experiment from NPZ in a background thread."""
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt
        from core.file_load_worker import FileLoadWorker

        progress = QProgressDialog(f"Loading photometry experiment...\n{npz_path.name}", None, 0, 0, self)
        progress.setWindowTitle("Loading Photometry Data")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.show()

        self._loading_progress = progress

        self._load_worker = FileLoadWorker(
            photometry.load_experiment_from_npz, npz_path, exp_idx
        )
        self._load_worker.finished.connect(self._on_photometry_npz_loaded)
        self._load_worker.error.connect(lambda msg: (
            progress.close(),
            self._show_error("Load Error", f"Failed to load photometry experiment:\n\n{msg.split(chr(10))[0]}")
        ))
        self._load_worker.start()

    def _on_photometry_npz_loaded(self, result_data):
        """Completion handler for photometry NPZ loading."""
        progress = self._loading_progress
        progress.close()

        if result_data:
            self._load_photometry_data(result_data)
        else:
            self._show_error("Load Error", "Failed to load experiment from NPZ file.")

    def _load_photometry_data(self, data: dict):
        """
        Load processed photometry data into the application state.

        Args:
            data: Dict from PhotometryImportDialog.get_result_data() containing:
                - sweeps, channel_names, t, sr_hz
                - photometry_raw, photometry_params, photometry_npz_path
                - dff_channel_name
        """
        st = self.state

        # Store the data in state
        st.sweeps = data['sweeps']
        st.channel_names = data['channel_names']
        st.t = data['t']
        st.sr_hz = data['sr_hz']

        # Store photometry-specific data for recalculation
        st.photometry_raw = data['photometry_raw']
        st.photometry_npz_path = data['photometry_npz_path']

        # Normalize photometry_params to use consistent keys across all paths.
        # NPZ loader uses: method, fit_start, fit_end
        # Tab 2 dialog uses: dff_method, fit_range_start, fit_range_end, lowpass_enabled
        # We store BOTH conventions so either consumer works correctly.
        raw_params = data['photometry_params']
        if raw_params:
            normalized = dict(raw_params)  # Copy
            # Ensure Tab 2 keys exist (from NPZ keys)
            if 'dff_method' not in normalized and 'method' in normalized:
                normalized['dff_method'] = normalized['method']
            if 'fit_range_start' not in normalized and 'fit_start' in normalized:
                normalized['fit_range_start'] = float(normalized['fit_start'])
            if 'fit_range_end' not in normalized and 'fit_end' in normalized:
                normalized['fit_range_end'] = float(normalized['fit_end'])
            # Ensure NPZ keys exist (from Tab 2 keys)
            if 'method' not in normalized and 'dff_method' in normalized:
                normalized['method'] = normalized['dff_method']
            if 'fit_start' not in normalized and 'fit_range_start' in normalized:
                normalized['fit_start'] = float(normalized['fit_range_start'])
            if 'fit_end' not in normalized and 'fit_range_end' in normalized:
                normalized['fit_end'] = float(normalized['fit_range_end'])
            # Ensure lowpass_enabled is consistent
            if 'lowpass_enabled' not in normalized:
                lp_hz = normalized.get('lowpass_hz')
                normalized['lowpass_enabled'] = lp_hz is not None and lp_hz > 0
            st.photometry_params = normalized
        else:
            st.photometry_params = raw_params
        st.photometry_dff_channel = data.get('dff_channel_name')

        # Store experiment metadata for reload/switch functionality
        st.photometry_experiment_index = data.get('experiment_index', 0)
        st.photometry_n_experiments = data.get('n_experiments', 1)
        st.photometry_animal_id = data.get('animal_id', '')

        # Set the file path (use NPZ path if available, otherwise construct from source)
        if data['photometry_npz_path']:
            st.in_path = data['photometry_npz_path']
        else:
            st.in_path = None

        # Initialize file_info for consistency
        st.file_info = [{
            'path': st.in_path,
            'sweep_start': 0,
            'sweep_end': 0,  # Single "sweep" for photometry
            'file_type': 'photometry',
        }]

        # Reset sweep index
        st.sweep_idx = 0
        self.navigation_manager.reset_window_state()

        # Clear any existing analysis data
        st.peaks_by_sweep.clear()
        st.breath_by_sweep.clear()
        st.sigh_by_sweep.clear()
        st.sniff_regions_by_sweep.clear()
        st.all_peaks_by_sweep.clear()
        st.all_breaths_by_sweep.clear()
        st.peak_metrics_by_sweep.clear()
        st.current_peak_metrics_by_sweep.clear()
        st.omitted_sweeps.clear()
        st.omitted_ranges.clear()
        st.proc_cache.clear()

        # Update filename display
        self._update_filename_display()

        # Set up channel manager with photometry channels
        self._setup_photometry_channel_manager(data)

        # Redraw
        self.plot_manager.redraw_main_plot()

        print(f"[Photometry] Loaded {len(st.channel_names)} channels into app")

    def _setup_photometry_channel_manager(self, data: dict):
        """
        Set up the channel manager for photometry data.

        Populates channel manager and combo boxes like ABF file loading,
        then marks ΔF/F as a computed channel with gear icon for reconfiguration.
        """
        from core.channel_manager import ChannelConfig

        st = self.state
        ch_names = st.channel_names
        dff_channel = data.get('dff_channel_name')
        channel_visibility = data.get('channel_visibility', {})  # Dict of ch_name -> bool

        # Populate the old combo boxes (for compatibility with rest of app)
        self.AnalyzeChanSelect.blockSignals(True)
        self.AnalyzeChanSelect.clear()
        self.AnalyzeChanSelect.addItem("All Channels")  # First option for grid view
        self.AnalyzeChanSelect.addItems(ch_names)
        self.AnalyzeChanSelect.setCurrentIndex(0)  # default = "All Channels" (grid mode)
        self.AnalyzeChanSelect.blockSignals(False)

        self.StimChanSelect.blockSignals(True)
        self.StimChanSelect.clear()
        self.StimChanSelect.addItem("None")
        self.StimChanSelect.addItems(ch_names)
        self.StimChanSelect.setCurrentIndex(0)  # select "None"
        self.StimChanSelect.blockSignals(False)

        self.EventsChanSelect.blockSignals(True)
        self.EventsChanSelect.clear()
        self.EventsChanSelect.addItem("None")
        self.EventsChanSelect.addItems(ch_names)
        self.EventsChanSelect.setCurrentIndex(0)  # select "None"
        self.EventsChanSelect.blockSignals(False)

        # Update event markers source channel combo
        self._update_event_source_channel_combo()

        # Build channel configs (similar to _populate_channel_manager but with photometry awareness)
        configs = []
        for i, ch_name in enumerate(ch_names):
            # Determine channel type with smart detection
            ch_lower = ch_name.lower()

            # Pleth/breathing channel detection (extended keywords)
            pleth_keywords = ['pleth', 'resp', 'breathing', 'breath', 'flow', 'airway', 'ventilation']
            # Opto stim channel detection
            opto_keywords = ['stim', 'opto', 'laser', 'led', 'light']
            # Thermal stim detection (treat as Opto for blue background)
            thermal_keywords = ['therm', 'temp', 'heat']

            if any(kw in ch_lower for kw in pleth_keywords):
                ch_type = 'Pleth'
            elif any(kw in ch_lower for kw in opto_keywords + thermal_keywords):
                ch_type = 'Opto Stim'
            else:
                ch_type = 'Raw Signal'

            # Determine if this is a computed channel (ΔF/F)
            is_computed = (dff_channel is not None and ch_name == dff_channel)

            # Create callback for gear icon on ΔF/F channel
            settings_callback = None
            if is_computed:
                settings_callback = self._open_photometry_settings

            # Determine visibility from data (default to True if not specified)
            is_visible = channel_visibility.get(ch_name, True)

            config = ChannelConfig(
                name=ch_name,
                visible=is_visible,
                channel_type=ch_type,
                source='computed' if is_computed else 'file',
                order=i,
                settings_callback=settings_callback,
            )
            configs.append(config)

        # Update channel manager (use self.channel_manager, not channel_manager_widget)
        if hasattr(self, 'channel_manager') and self.channel_manager is not None:
            self.channel_manager.set_channels(configs)

            # Auto-detect channels: prefer Pleth for analysis, then ΔF/F
            pleth_channels = [c.name for c in configs if c.channel_type == 'Pleth']
            if pleth_channels:
                st.analyze_chan = pleth_channels[0]
                # Update combo box
                idx = ch_names.index(st.analyze_chan) + 1  # +1 because "All Channels" is at index 0
                self.AnalyzeChanSelect.setCurrentIndex(idx)
            elif dff_channel:
                # Default to ΔF/F for analysis if no Pleth
                st.analyze_chan = dff_channel
                idx = ch_names.index(st.analyze_chan) + 1
                self.AnalyzeChanSelect.setCurrentIndex(idx)

            # Auto-detect Opto Stim channel
            opto_channels = [c.name for c in configs if c.channel_type == 'Opto Stim']
            if opto_channels:
                st.stim_chan = opto_channels[0]
                idx = ch_names.index(st.stim_chan) + 1  # +1 because "None" is at index 0
                self.StimChanSelect.setCurrentIndex(idx)
                # Trigger stim detection
                self.on_stim_channel_changed(idx)

            print(f"[Photometry] Channel manager populated with {len(configs)} channels")

    def _open_photometry_settings(self):
        """
        Open the photometry dialog to adjust ΔF/F calculation settings.

        Called when user clicks gear icon on ΔF/F channel.
        """
        st = self.state

        print("[Photometry] Gear icon clicked - opening settings dialog")

        if st.photometry_raw is None:
            self._show_error("No Photometry Data",
                           "No raw photometry data available for recalculation.\n\n"
                           "This can happen if the data wasn't loaded from a photometry file.")
            print("[Photometry] ERROR: photometry_raw is None")
            return

        # Build current experiment info for the dialog (for edit mode)
        current_exp_info = {
            'experiment_index': getattr(st, 'photometry_experiment_index', 0),
            'n_experiments': getattr(st, 'photometry_n_experiments', 1),
            'animal_id': getattr(st, 'photometry_animal_id', ''),
        }

        print(f"[Photometry] Opening dialog with cached data, exp_info={current_exp_info}")
        print(f"[Photometry]   npz_path={st.photometry_npz_path}")
        print(f"[Photometry]   raw data keys={list(st.photometry_raw.keys()) if st.photometry_raw else 'None'}")

        # Open dialog with cached data for instant opening
        # Pass current params to restore settings and current experiment info
        dialog = PhotometryImportDialog(
            self,
            initial_path=None,  # Don't need to reload files
            photometry_npz_path=st.photometry_npz_path,
            initial_params=st.photometry_params,  # Restore previous settings
            current_experiment=current_exp_info,  # Enable edit mode
            cached_photometry_data=st.photometry_raw,  # Use cached data - no disk loading!
        )

        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted:
            result_data = dialog.get_result_data()
            if result_data:
                # Update with new processing
                self._load_photometry_data(result_data)
                print("[Photometry] Updated dF/F with new settings")

                # Persist updated dF/F params back to NPZ so they survive app restart
                self._update_npz_dff_params()
        else:
            print("[Photometry] Dialog cancelled")

    def _update_npz_dff_params(self):
        """Write updated dF/F parameters back to the NPZ file.

        Called after the user adjusts processing settings via the gear icon.
        Updates the 'dff_params' field in the NPZ so changes persist on restart.
        """
        st = self.state
        npz_path = st.photometry_npz_path
        if npz_path is None or not Path(npz_path).exists():
            return

        params = st.photometry_params
        if not params:
            return

        try:
            # Build dff_params dict in NPZ format (uses NPZ key names)
            npz_params = {
                'method': params.get('method', params.get('dff_method', 'fitted')),
                'detrend_method': params.get('detrend_method', 'none'),
                'lowpass_hz': params.get('lowpass_hz'),
                'fit_start': params.get('fit_start', params.get('fit_range_start', 0)),
                'fit_end': params.get('fit_end', params.get('fit_range_end', 0)),
            }

            exp_idx = getattr(st, 'photometry_experiment_index', 0)

            # Load existing NPZ, update dff_params, re-save
            data = dict(np.load(npz_path, allow_pickle=True))

            # Parse existing dff_params or create new
            dff_params_all = {}
            if 'dff_params' in data:
                try:
                    dff_params_all = eval(str(data['dff_params'][0]))
                except Exception:
                    pass

            dff_params_all[exp_idx] = npz_params
            data['dff_params'] = np.array([str(dff_params_all)], dtype=object)

            np.savez(npz_path, **data)
            print(f"[Photometry] Updated dff_params in NPZ for experiment {exp_idx}")

        except Exception as e:
            print(f"[Photometry] Warning: Could not update NPZ dff_params: {e}")

    def load_file(self, path: Path):
        import time
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt
        from core.file_load_worker import FileLoadWorker

        # Determine file type for progress dialog title
        file_type = path.suffix.upper()[1:]  # .abf -> ABF, .smrx -> SMRX

        # Create progress dialog
        progress = QProgressDialog(f"Loading file...\n{path.name}", None, 0, 100, self)
        progress.setWindowTitle(f"Opening {file_type} File")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setCancelButton(None)  # No cancel button
        progress.setValue(0)
        progress.show()

        # Store context for completion handler
        self._loading_path = path
        self._loading_t_start = time.time()
        self._loading_progress = progress

        # Create worker with existing loader function
        self._load_worker = FileLoadWorker(abf_io.load_data_file, path)
        self._load_worker.progress.connect(lambda c, t, m: (
            progress.setValue(c),
            progress.setLabelText(f"{m}\n{path.name}")
        ))
        self._load_worker.finished.connect(self._on_single_file_loaded)
        self._load_worker.error.connect(lambda msg: (
            progress.close(),
            self._show_error("Load error", msg.split('\n\n')[0])
        ))
        self._load_worker.start()

    def _on_single_file_loaded(self, result):
        """Completion handler for single file loading (runs on main thread)."""
        import time

        path = self._loading_path
        t_start = self._loading_t_start
        progress = self._loading_progress
        progress.close()

        sr, sweeps_by_ch, ch_names, t, file_metadata = result

        st = self.state
        st.in_path = path
        # Set file_info for single file (for consistency with multi-file loading)
        n_sweeps = next(iter(sweeps_by_ch.values())).shape[1]
        st.file_info = [{
            'path': path,
            'sweep_start': 0,
            'sweep_end': n_sweeps - 1,
            **file_metadata  # Include protocol, n_channels, file_type, etc.
        }]
        st.sr_hz = sr
        st.sweeps = sweeps_by_ch
        st.channel_names = ch_names
        st.t = t
        st.sweep_idx = 0
        self.navigation_manager.reset_window_state()

        # Update filename display with metadata
        self._update_filename_display()

        # Log telemetry: file loaded with enhanced metrics
        file_ext = path.suffix.lower()[1:]  # .abf -> abf
        load_duration = time.time() - t_start
        file_size_mb = path.stat().st_size / (1024 * 1024)
        duration_minutes = (t[-1] - t[0]) / 60 if len(t) > 1 else 0

        telemetry.log_file_loaded(
            file_type=file_ext,
            num_sweeps=n_sweeps,
            num_breaths=None,  # Not detected yet
            file_size_mb=round(file_size_mb, 2),
            sampling_rate_hz=int(sr),
            duration_minutes=round(duration_minutes, 1),
            num_channels=len(ch_names)
        )

        # Log file loading timing
        telemetry.log_timing('file_load', load_duration,
                            file_size_mb=round(file_size_mb, 2),
                            num_sweeps=n_sweeps)

        # Reset peak results and trace cache
        if not hasattr(st, "peaks_by_sweep"):
            st.peaks_by_sweep = {}
            st.breath_by_sweep = {}
        else:
            st.peaks_by_sweep.clear()
            self.state.sigh_by_sweep.clear()
            st.breath_by_sweep.clear()
            self.state.omitted_sweeps.clear()
            self.state.omitted_ranges.clear()
            self._refresh_omit_button_label()

        # Clear cross-sweep outlier detection data
        self.metrics_by_sweep.clear()
        self.onsets_by_sweep.clear()
        self.global_outlier_stats = None

        # Clear export metric cache when loading new file
        self._export_metric_cache = {}

        # Clear z-score global statistics cache
        self.zscore_global_mean = None
        self.zscore_global_std = None

        # Reset adaptive downsampling for fresh file (let system re-evaluate)
        self.plot_manager._auto_downsample_active = False
        self.plot_manager._consecutive_slow_redraws = 0

        # Clear event markers (bout annotations)
        if hasattr(st, 'bout_annotations'):
            st.bout_annotations.clear()

        # Reset Apply button
        self.ApplyPeakFindPushButton.setEnabled(False)

        # Fill combos safely (no signal during population)
        self.AnalyzeChanSelect.blockSignals(True)
        self.AnalyzeChanSelect.clear()
        self.AnalyzeChanSelect.addItem("All Channels")  # First option for grid view
        self.AnalyzeChanSelect.addItems(ch_names)
        self.AnalyzeChanSelect.setCurrentIndex(0)  # default = "All Channels" (grid mode)
        self.AnalyzeChanSelect.blockSignals(False)

        self.StimChanSelect.blockSignals(True)
        self.StimChanSelect.clear()
        self.StimChanSelect.addItem("None")        # default
        self.StimChanSelect.addItems(ch_names)
        self.StimChanSelect.setCurrentIndex(0)     # select "None"
        self.StimChanSelect.blockSignals(False)

        # Populate Events Channel dropdown
        self.EventsChanSelect.blockSignals(True)
        self.EventsChanSelect.clear()
        self.EventsChanSelect.addItem("None")      # default - no event channel
        self.EventsChanSelect.addItems(ch_names)
        # Restore previous selection if it exists
        if st.event_channel and st.event_channel in ch_names:
            idx = ch_names.index(st.event_channel) + 1  # +1 because "None" is at index 0
            self.EventsChanSelect.setCurrentIndex(idx)
        else:
            self.EventsChanSelect.setCurrentIndex(0)  # select "None"
        self.EventsChanSelect.blockSignals(False)

        # Update event markers source channel combo
        self._update_event_source_channel_combo()

        # Populate Channel Manager widget
        self._populate_channel_manager(ch_names)

        #Clear peaks
        self.state.peaks_by_sweep.clear()
        self.state.sigh_by_sweep.clear()
        self.state.breath_by_sweep.clear()

        #Clear omitted sweeps and regions
        self.state.omitted_sweeps.clear()
        self.state.omitted_ranges.clear()
        self._refresh_omit_button_label()

        # Auto-detect stimulus and analysis channels
        # Note: auto_analysis is only returned if there's exactly one non-stim channel
        auto_stim, auto_analysis = abf_io.auto_select_channels(st.sweeps, ch_names)

        if auto_stim:
            # Found a stimulus channel - select it
            stim_idx = ch_names.index(auto_stim) + 1  # +1 because "None" is at index 0
            self.StimChanSelect.setCurrentIndex(stim_idx)
            # NOTE: Don't set st.stim_chan here - let on_stim_channel_changed do it
            # Otherwise the handler's "if new_stim != st.stim_chan" check fails and
            # stim detection is skipped!

            # Update channel manager to show this as Opto Stim
            self.channel_manager.set_channel_type(auto_stim, "Opto Stim")

            # Trigger stim detection (this sets st.stim_chan and computes stim spans)
            self.on_stim_channel_changed(stim_idx)

            if auto_analysis:
                # Only one analysis channel available - auto-select it
                analysis_idx = ch_names.index(auto_analysis) + 1  # +1 because "All Channels" is at index 0
                self.AnalyzeChanSelect.setCurrentIndex(analysis_idx)
                st.analyze_chan = auto_analysis
                self.single_panel_mode = True

                # Trigger analysis channel change
                self.on_analyze_channel_changed(analysis_idx)

                # Show completion message with auto-detection info
                t_elapsed = time.time() - t_start
                self._log_status_message(
                    f"File loaded ({t_elapsed:.1f}s) - Auto-detected: stim={auto_stim}, analysis={auto_analysis}",
                    4000
                )
            else:
                # Multiple analysis channels available - start in grid mode, let user choose
                st.analyze_chan = None
                self.single_panel_mode = False
                self.plot_host.clear_saved_view("grid")
                self.plot_all_channels()

                t_elapsed = time.time() - t_start
                self._log_status_message(
                    f"File loaded ({t_elapsed:.1f}s) - Auto-detected stim: {auto_stim}",
                    4000
                )
        else:
            # No stimulus channel detected - start in grid mode
            st.analyze_chan = None
            self.single_panel_mode = False

            st.stim_chan = None
            st.stim_onsets_by_sweep.clear()
            st.stim_offsets_by_sweep.clear()
            st.stim_spans_by_sweep.clear()
            st.stim_metrics_by_sweep.clear()

            self.plot_host.clear_saved_view("grid")
            self.plot_all_channels()

            # Show completion message
            t_elapsed = time.time() - t_start
            self._log_status_message(f"File loaded ({t_elapsed:.1f}s)", 3000)

        # Apply pending channel selections from Project Builder (if any)
        self._apply_pending_channel_selections(ch_names)

        # Refresh Analysis Options dialog tabs if open (file data now available)
        if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None and self._analysis_options_dialog.isVisible():
            # Note: Don't refresh Peak Detection tab here since no channel selected yet (in grid mode)
            # Peak Detection tab will update when user selects a channel
            pass

    def _apply_pending_channel_selections(self, ch_names):
        """
        Apply pending channel selections from Project Builder after file loads.

        When a file is opened from the Project Builder, we may have pre-selected
        analysis and stim channels that should be applied after loading.
        """
        st = self.state

        # Check for pending analysis channel
        pending_analysis = getattr(self, '_pending_analysis_channel', '')
        pending_stim = getattr(self, '_pending_stim_channels', [])

        if not pending_analysis and not pending_stim:
            return  # Nothing to apply

        print(f"[channel-selection] Applying pending selections: analysis={pending_analysis}, stim={pending_stim}")

        # Apply pending stim channel (first one if multiple)
        if pending_stim:
            # Take first stim channel from list
            stim_channel = pending_stim[0] if isinstance(pending_stim, list) else pending_stim

            # Find the channel in the dropdown
            # StimChanSelect has "None" at index 0, then channel names
            if stim_channel in ch_names:
                stim_idx = ch_names.index(stim_channel) + 1  # +1 for "None"
                self.StimChanSelect.setCurrentIndex(stim_idx)
                st.stim_chan = stim_channel
                self.on_stim_channel_changed(stim_idx)
                print(f"[channel-selection] Applied stim channel: {stim_channel}")

        # Apply pending analysis channel
        if pending_analysis:
            # Find the channel in the dropdown
            # AnalyzeChanSelect has "All Channels" at index 0, then channel names
            if pending_analysis in ch_names:
                analysis_idx = ch_names.index(pending_analysis) + 1  # +1 for "All Channels"
                self.AnalyzeChanSelect.setCurrentIndex(analysis_idx)
                st.analyze_chan = pending_analysis
                self.single_panel_mode = True
                self.on_analyze_channel_changed(analysis_idx)
                print(f"[channel-selection] Applied analysis channel: {pending_analysis}")

                self._log_status_message(
                    f"Loaded with pre-selected channels: analysis={pending_analysis}"
                    + (f", stim={pending_stim[0]}" if pending_stim else ""),
                    4000
                )

        # Clear pending selections
        self._pending_analysis_channel = ''
        self._pending_stim_channels = []

    def load_multiple_files(self, file_paths: List[Path]):
        """Load and concatenate multiple ABF files."""
        from PyQt6.QtWidgets import QProgressDialog, QMessageBox
        from PyQt6.QtCore import Qt
        from core.file_load_worker import FileLoadWorker

        # Validate files first
        valid, messages = abf_io.validate_files_for_concatenation(file_paths)

        if not valid:
            # Show error dialog
            self._show_error("File Validation Failed", "\n".join(messages))
            return
        elif messages:  # Warnings
            # Show warning dialog with option to proceed
            reply = QMessageBox.question(
                self,
                "File Validation Warnings",
                "The following warnings were detected:\n\n" + "\n".join(messages) + "\n\nDo you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Create progress dialog
        progress = QProgressDialog(f"Loading {len(file_paths)} files...", None, 0, 100, self)
        progress.setWindowTitle("Loading Multiple Files")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setCancelButton(None)  # No cancel button
        progress.setValue(0)
        progress.show()

        # Store context for completion handler
        self._loading_multi_paths = file_paths
        self._loading_progress = progress

        # Create worker with existing loader function
        self._load_worker = FileLoadWorker(abf_io.load_and_concatenate_abf_files, file_paths)
        self._load_worker.progress.connect(lambda c, t, m: (
            progress.setValue(c),
            progress.setLabelText(m)
        ))
        self._load_worker.finished.connect(self._on_multi_file_loaded)
        self._load_worker.error.connect(lambda msg: (
            progress.close(),
            self._show_error("Load error", msg.split('\n\n')[0])
        ))
        self._load_worker.start()

    def _on_multi_file_loaded(self, result):
        """Completion handler for multi-file loading (runs on main thread)."""
        file_paths = self._loading_multi_paths
        progress = self._loading_progress
        progress.close()

        sr, sweeps_by_ch, ch_names, t, file_info = result

        # Update state (similar to load_file, but with file_info)
        st = self.state
        st.in_path = file_paths[0]  # Store first file path for display
        st.file_info = file_info  # Store multi-file metadata
        st.sr_hz = sr
        st.sweeps = sweeps_by_ch
        st.channel_names = ch_names
        st.t = t
        st.sweep_idx = 0
        self.navigation_manager.reset_window_state()

        # Reset peak results and trace cache
        if not hasattr(st, "peaks_by_sweep"):
            st.peaks_by_sweep = {}
            st.breath_by_sweep = {}
        else:
            st.peaks_by_sweep.clear()
            self.state.sigh_by_sweep.clear()
            st.breath_by_sweep.clear()
            self.state.omitted_sweeps.clear()
            self.state.omitted_ranges.clear()
            self._refresh_omit_button_label()

        # Clear cross-sweep outlier detection data
        self.metrics_by_sweep.clear()
        self.onsets_by_sweep.clear()
        self.global_outlier_stats = None

        # Clear export metric cache when loading new file
        self._export_metric_cache = {}

        # Reset Apply button
        self.ApplyPeakFindPushButton.setEnabled(False)

        # Fill combos safely (no signal during population)
        self.AnalyzeChanSelect.blockSignals(True)
        self.AnalyzeChanSelect.clear()
        self.AnalyzeChanSelect.addItem("All Channels")  # First option for grid view
        self.AnalyzeChanSelect.addItems(ch_names)
        self.AnalyzeChanSelect.setCurrentIndex(0)  # default = "All Channels" (grid mode)
        self.AnalyzeChanSelect.blockSignals(False)

        self.StimChanSelect.blockSignals(True)
        self.StimChanSelect.clear()
        self.StimChanSelect.addItem("None")        # default
        self.StimChanSelect.addItems(ch_names)
        self.StimChanSelect.setCurrentIndex(0)     # select "None"
        self.StimChanSelect.blockSignals(False)

        # Populate Events Channel dropdown
        self.EventsChanSelect.blockSignals(True)
        self.EventsChanSelect.clear()
        self.EventsChanSelect.addItem("None")      # default - no event channel
        self.EventsChanSelect.addItems(ch_names)
        # Restore previous selection if it exists
        if st.event_channel and st.event_channel in ch_names:
            idx = ch_names.index(st.event_channel) + 1  # +1 because "None" is at index 0
            self.EventsChanSelect.setCurrentIndex(idx)
        else:
            self.EventsChanSelect.setCurrentIndex(0)  # select "None"
        self.EventsChanSelect.blockSignals(False)

        # Update event markers source channel combo
        self._update_event_source_channel_combo()

        # Populate Channel Manager widget
        self._populate_channel_manager(ch_names)

        #Clear peaks
        self.state.peaks_by_sweep.clear()
        self.state.sigh_by_sweep.clear()
        self.state.breath_by_sweep.clear()

        #Clear omitted sweeps and regions
        self.state.omitted_sweeps.clear()
        self.state.omitted_ranges.clear()
        self._refresh_omit_button_label()

        # Clear z-score global statistics cache
        self.zscore_global_mean = None
        self.zscore_global_std = None

        # Auto-detect stimulus and analysis channels
        # Note: auto_analysis is only returned if there's exactly one non-stim channel
        auto_stim, auto_analysis = abf_io.auto_select_channels(st.sweeps, ch_names)

        if auto_stim:
            # Found a stimulus channel - select it
            stim_idx = ch_names.index(auto_stim) + 1  # +1 because "None" is at index 0
            self.StimChanSelect.setCurrentIndex(stim_idx)
            # NOTE: Don't set st.stim_chan here - let on_stim_channel_changed do it
            # Otherwise the handler's "if new_stim != st.stim_chan" check fails and
            # stim detection is skipped!

            # Update channel manager to show this as Opto Stim
            self.channel_manager.set_channel_type(auto_stim, "Opto Stim")

            # Trigger stim detection (this sets st.stim_chan and computes stim spans)
            self.on_stim_channel_changed(stim_idx)

            if auto_analysis:
                # Only one analysis channel available - auto-select it
                analysis_idx = ch_names.index(auto_analysis) + 1  # +1 because "All Channels" is at index 0
                self.AnalyzeChanSelect.setCurrentIndex(analysis_idx)
                st.analyze_chan = auto_analysis
                self.single_panel_mode = True

                # Trigger analysis channel change
                self.on_analyze_channel_changed(analysis_idx)
            else:
                # Multiple analysis channels available - start in grid mode, let user choose
                st.analyze_chan = None
                self.single_panel_mode = False
                self.plot_host.clear_saved_view("grid")
                self.plot_all_channels()
        else:
            # No stimulus channel detected - start in grid mode
            st.analyze_chan = None
            self.single_panel_mode = False

            st.stim_chan = None
            st.stim_onsets_by_sweep.clear()
            st.stim_offsets_by_sweep.clear()
            st.stim_spans_by_sweep.clear()
            st.stim_metrics_by_sweep.clear()

            self.plot_host.clear_saved_view("grid")
            self.plot_all_channels()

        # Show success message with file info
        total_sweeps = next(iter(sweeps_by_ch.values())).shape[1]

        # Build file summary with padding information
        file_lines = []
        for i, info in enumerate(file_info):
            line = f"  {i+1}. {info['path'].name}: sweeps {info['sweep_start']}-{info['sweep_end']}"
            if info.get('padded', False):
                orig_dur = info['original_samples'] / sr
                padded_dur = info['padded_samples'] / sr
                line += f" (padded: {orig_dur:.2f}s → {padded_dur:.2f}s)"
            file_lines.append(line)

        file_summary = "\n".join(file_lines)

        # Check if any files were padded
        padded_count = sum(1 for info in file_info if info.get('padded', False))

        message = f"Loaded {len(file_paths)} files with {total_sweeps} total sweeps:\n\n{file_summary}"
        if padded_count > 0:
            message += f"\n\nNote: {padded_count} file(s) had different sweep lengths and were padded with NaN values."

        # Add auto-detection info to message
        if auto_stim:
            message += f"\n\nAuto-detected stimulus channel: {auto_stim}"
            if auto_analysis:
                message += f"\nAuto-selected analysis channel: {auto_analysis}"

        self._show_info("Files Loaded Successfully", message)

    # ---------- Session Save/Load ----------
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts globally."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QKeyEvent

        # Ctrl+S - Save Data (same as Save Data button)
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_S:
            self.on_save_analyzed_clicked()
            event.accept()
        else:
            # Pass event to parent for default handling
            super().keyPressEvent(event)

    def save_session_state(self):
        """Save current analysis state to .pleth.npz file (Ctrl+S)."""
        from core.npz_io import save_state_to_npz

        # Check if we have data loaded
        if not self.state.in_path:
            self._show_warning("No Data Loaded",
                "Please load a data file before saving session state."
            )
            return

        # Check if channel is selected
        if not self.state.analyze_chan:
            self._show_warning("No Channel Selected",
                "Please select a channel to analyze before saving.\n\n"
                "(Session state is saved per-channel, allowing independent analysis of multi-channel files)"
            )
            return

        # Default to analysis folder with simple naming (no metadata required for quick save)
        analysis_folder = self.state.in_path.parent / "Pleth_App_Analysis"
        analysis_folder.mkdir(exist_ok=True)

        # Use simple default name: {datafile}_{channel}_session.npz
        safe_channel = self.state.analyze_chan.replace(' ', '_').replace('/', '_').replace('\\', '_')
        default_filename = f"{self.state.in_path.stem}_{safe_channel}_session.npz"
        default_path = analysis_folder / default_filename

        # Let user modify path if desired
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Session State",
            str(default_path),
            "PhysioMetrics Session (*.pleth.npz);;All Files (*)"
        )

        if not save_path:
            return  # User cancelled

        save_path = Path(save_path)

        # Ask about including raw data (for portability vs file size)
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QCheckBox, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle("Save Options")
        layout = QVBoxLayout()

        label = QLabel(
            "Choose what to include in the session file:\n\n"
            "• Analysis results (peaks, edits, filters) - Always included\n"
            "• Raw signal data - Optional (makes file portable but larger)"
        )
        layout.addWidget(label)

        include_raw_checkbox = QCheckBox("Include raw signal data (for portability)")
        include_raw_checkbox.setToolTip(
            "If checked: File can be loaded without original .abf file (larger file ~65MB)\n"
            "If unchecked: File will reload from original .abf file (smaller file ~5-10MB)"
        )
        include_raw_checkbox.setChecked(False)  # Default: don't include (smaller files)
        layout.addWidget(include_raw_checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        dialog.setLayout(layout)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return  # User cancelled

        include_raw = include_raw_checkbox.isChecked()

        # Save to NPZ
        try:
            import time
            t_start = time.time()

            # Pass GMM cache to preserve user's cluster assignments
            gmm_cache = getattr(self, '_cached_gmm_results', None)

            # Collect app-level settings to preserve
            app_settings = {
                'filter_order': self.filter_order,
                'use_zscore_normalization': self.use_zscore_normalization,
                'notch_filter_lower': self.notch_filter_lower,
                'notch_filter_upper': self.notch_filter_upper,
                'apnea_threshold': self._parse_float(self.ApneaThresh) or 0.5,
                'active_eupnea_sniff_classifier': self.state.active_eupnea_sniff_classifier
            }

            # Get event markers for persistence
            event_markers_data = None
            if hasattr(self, '_event_marker_viewmodel') and self._event_marker_viewmodel:
                try:
                    event_markers_data = self._event_marker_viewmodel.save_to_npz()
                    marker_count = self._event_marker_viewmodel.marker_count
                    if marker_count > 0:
                        print(f"[npz-save] Saving {marker_count} event markers to session file")
                except Exception as e:
                    print(f"[npz-save] Warning: Failed to save event markers: {e}")

            save_state_to_npz(self.state, save_path, include_raw_data=include_raw, gmm_cache=gmm_cache, app_settings=app_settings, event_markers=event_markers_data)

            t_elapsed = time.time() - t_start
            file_size_mb = save_path.stat().st_size / (1024 * 1024)

            self._log_status_message(
                f"Session saved: {save_path.name} ({file_size_mb:.1f} MB, {t_elapsed:.1f}s)",
                timeout=5000
            )

            # Count eupnea and sniffing breaths for telemetry
            eupnea_count = 0
            sniff_count = 0
            for s in self.state.sweeps.keys():
                breath_data = self.state.breath_by_sweep.get(s, {})
                onsets = breath_data.get('onsets', [])
                sniff_regions = self.state.sniff_regions_by_sweep.get(s, [])

                for i in range(len(onsets) - 1):
                    # Check if breath midpoint falls in any sniffing region
                    t_start = self.state.t[onsets[i]]
                    t_end = self.state.t[onsets[i + 1]]
                    t_mid = (t_start + t_end) / 2.0

                    is_sniff = False
                    for (region_start, region_end) in sniff_regions:
                        if region_start <= t_mid <= region_end:
                            is_sniff = True
                            break

                    if is_sniff:
                        sniff_count += 1
                    else:
                        eupnea_count += 1

            # Log telemetry: file saved with per-file edit metrics (for ML evaluation)
            telemetry.log_file_saved(
                save_type='npz',
                eupnea_count=eupnea_count,
                sniff_count=sniff_count,
                file_size_mb=round(file_size_mb, 2),
                include_raw_data=include_raw,
                num_sweeps=len(self.state.sweeps)
            )

            # Update settings with last save location
            self.settings.setValue("last_npz_save_dir", str(save_path.parent))

        except Exception as e:
            self._show_error("Save Error",
                f"Failed to save session state:\n\n{str(e)}"
            )

    def load_npz_state(self, npz_path: Path, alternative_data_path: Path = None):
        """Load complete analysis state from .pleth.npz file."""
        from core.npz_io import load_state_from_npz, get_npz_metadata
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt
        from core.file_load_worker import FileLoadWorker
        import time

        t_start = time.time()

        # Get metadata for display (fast - reads only header)
        metadata = get_npz_metadata(npz_path)

        if 'error' in metadata:
            self._show_error("Load Error",
                f"Failed to read NPZ file:\n\n{metadata['error']}"
            )
            return

        # If no alternative path provided, check if stored path exists - if not, try project file list
        if alternative_data_path is None and hasattr(self, '_master_file_list') and self._master_file_list:
            # Get the original filename from metadata
            original_path_str = metadata.get('original_file', '')
            if original_path_str:
                stored_path = Path(original_path_str)

                # Only search project file list if the stored path doesn't exist
                if not stored_path.exists():
                    original_filename = stored_path.name
                    print(f"\n[Path Resolution] NPZ stored path not found: {stored_path}")
                    print(f"  Searching project file list for: {original_filename}")

                    # Search project's master file list for a matching filename
                    for task in self._master_file_list:
                        task_path = task.get('file_path', '')
                        if task_path and Path(task_path).name == original_filename:
                            candidate = Path(task_path)
                            if candidate.exists():
                                # Show detailed comparison of paths
                                print(f"  Found in project: {candidate}")

                                # Analyze the difference
                                stored_parts = stored_path.parts
                                project_parts = candidate.parts
                                if stored_parts != project_parts:
                                    # Find where they diverge
                                    for i, (s, p) in enumerate(zip(stored_parts, project_parts)):
                                        if s != p:
                                            print(f"  Path diverges at part {i}: '{s}' vs '{p}'")
                                            break
                                    if len(stored_parts) != len(project_parts):
                                        print(f"  Path depth differs: {len(stored_parts)} vs {len(project_parts)} parts")

                                alternative_data_path = candidate
                                break
                    else:
                        print(f"  Not found in project file list")

        # Show loading dialog
        progress = QProgressDialog(f"Loading session...\n{npz_path.name}", None, 0, 100, self)
        progress.setWindowTitle("Loading PhysioMetrics Session")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(10)
        progress.show()

        progress.setLabelText(f"Reading session file...\n{npz_path.name}")
        progress.setValue(30)

        # Store context for completion handler
        self._loading_npz_path = npz_path
        self._loading_npz_metadata = metadata
        self._loading_npz_t_start = t_start
        self._loading_progress = progress

        # Create worker — load_state_from_npz doesn't accept progress_callback
        self._load_worker = FileLoadWorker(
            load_state_from_npz, npz_path,
            reload_raw_data=True, alternative_data_path=alternative_data_path
        )
        self._load_worker.finished.connect(self._on_npz_loaded)
        self._load_worker.error_exc.connect(self._on_npz_load_error)
        self._load_worker.start()

    def _on_npz_load_error(self, exc):
        """Handle errors from NPZ loading worker."""
        from core.npz_io import OriginalFileNotFoundError
        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        progress = self._loading_progress
        npz_path = self._loading_npz_path
        progress.close()

        if isinstance(exc, OriginalFileNotFoundError):
            # Prompt user to locate the file
            reply = QMessageBox.question(
                self,
                "Original File Not Found",
                f"The original data file could not be found:\n\n"
                f"{exc.original_path}\n\n"
                f"Would you like to locate the file manually?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Get file extension from original path
                orig_ext = exc.original_path.suffix.lower()
                filter_map = {
                    '.abf': "ABF Files (*.abf)",
                    '.smrx': "Spike2 Files (*.smrx)",
                    '.edf': "EDF Files (*.edf)",
                }
                file_filter = filter_map.get(orig_ext, "All Data Files (*.abf *.smrx *.edf)")

                new_path, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Locate {exc.original_path.name}",
                    str(Path.home()),
                    file_filter
                )

                if new_path:
                    # Retry with the new path
                    self.load_npz_state(npz_path, alternative_data_path=Path(new_path))
                    return
        else:
            import traceback
            self._show_error("Load Error",
                f"Failed to load session state:\n\n{str(exc)}\n\n{traceback.format_exc()}"
            )

    def _on_npz_loaded(self, result):
        """Completion handler for NPZ session loading (runs on main thread)."""
        import time

        npz_path = self._loading_npz_path
        metadata = self._loading_npz_metadata
        t_start = self._loading_npz_t_start
        progress = self._loading_progress

        new_state, raw_data_loaded, gmm_cache, app_settings, event_markers = result

        try:
            if not raw_data_loaded:
                progress.close()
                self._show_error("Load Error",
                    "Could not load raw data from original file or NPZ.\n\n"
                    f"Original file: {new_state.in_path}\n\n"
                    "Please ensure the original data file is accessible."
                )
                return

            progress.setLabelText("Restoring analysis state...")
            progress.setValue(50)

            # Replace current state
            self.state = new_state
            st = self.state

            # Update manager references to new state
            self.plot_manager.state = new_state
            self.navigation_manager.state = new_state
            self.editing_modes.state = new_state
            self.export_manager.state = new_state

            # ===== RESTORE UI ELEMENTS =====

            # Update filename display in status bar (with metadata if available)
            self._update_filename_display()

            progress.setValue(60)

            # Restore channel combos
            self.AnalyzeChanSelect.blockSignals(True)
            self.AnalyzeChanSelect.clear()
            self.AnalyzeChanSelect.addItem("All Channels")
            self.AnalyzeChanSelect.addItems(st.channel_names)

            if st.analyze_chan and st.analyze_chan in st.channel_names:
                idx = st.channel_names.index(st.analyze_chan) + 1  # +1 for "All Channels"
                self.AnalyzeChanSelect.setCurrentIndex(idx)
                self.single_panel_mode = True
            else:
                self.AnalyzeChanSelect.setCurrentIndex(0)  # "All Channels"
                self.single_panel_mode = False

            self.AnalyzeChanSelect.blockSignals(False)

            self.StimChanSelect.blockSignals(True)
            self.StimChanSelect.clear()
            self.StimChanSelect.addItem("None")
            self.StimChanSelect.addItems(st.channel_names)

            if st.stim_chan and st.stim_chan in st.channel_names:
                idx = st.channel_names.index(st.stim_chan) + 1
                self.StimChanSelect.setCurrentIndex(idx)
            else:
                self.StimChanSelect.setCurrentIndex(0)

            self.StimChanSelect.blockSignals(False)

            self.EventsChanSelect.blockSignals(True)
            self.EventsChanSelect.clear()
            self.EventsChanSelect.addItem("None")
            self.EventsChanSelect.addItems(st.channel_names)

            if st.event_channel and st.event_channel in st.channel_names:
                idx = st.channel_names.index(st.event_channel) + 1
                self.EventsChanSelect.setCurrentIndex(idx)
            else:
                self.EventsChanSelect.setCurrentIndex(0)

            self.EventsChanSelect.blockSignals(False)

            # Update event markers source channel combo
            self._update_event_source_channel_combo()

            progress.setValue(70)

            # Restore filter settings
            self.LowPass_checkBox.setChecked(st.use_low)
            self.HighPass_checkBox.setChecked(st.use_high)
            self.InvertSignal_checkBox.setChecked(st.use_invert)
            # Note: use_mean_sub is stored but not restored (no UI checkbox)

            if st.low_hz:
                self.LowPassVal.setText(str(st.low_hz))
            if st.high_hz:
                self.HighPassVal.setText(str(st.high_hz))

            # Restore app-level settings (filter order, zscore, notch, apnea threshold)
            if app_settings is not None:
                self.filter_order = app_settings.get('filter_order', 4)
                self.use_zscore_normalization = app_settings.get('use_zscore_normalization', True)
                self.notch_filter_lower = app_settings.get('notch_filter_lower')
                self.notch_filter_upper = app_settings.get('notch_filter_upper')
                apnea_thresh = app_settings.get('apnea_threshold', 0.5)

                # Update UI elements
                if hasattr(self, 'FilterOrderSpin'):
                    self.FilterOrderSpin.setValue(self.filter_order)
                if hasattr(self, 'ApneaThresh'):
                    self.ApneaThresh.setText(str(apnea_thresh))

                # Restore eupnea/sniff classifier
                saved_classifier = app_settings.get('active_eupnea_sniff_classifier', 'gmm')
                self.state.active_eupnea_sniff_classifier = saved_classifier

                # Update dropdown to match saved classifier
                classifier_to_dropdown = {
                    'gmm': 'GMM',
                    'xgboost': 'XGBoost',
                    'rf': 'Random Forest',
                    'mlp': 'MLP',
                    'all_eupnea': 'All Eupnea',
                    'none': 'None (Clear)'
                }
                dropdown_text = classifier_to_dropdown.get(saved_classifier, 'GMM')
                self.eup_sniff_combo.blockSignals(True)
                self.eup_sniff_combo.setCurrentText(dropdown_text)
                self.eup_sniff_combo.blockSignals(False)

                print(f"[npz-load] Restored app settings: filter_order={self.filter_order}, "
                      f"zscore={self.use_zscore_normalization}, notch={self.notch_filter_lower}-{self.notch_filter_upper}, "
                      f"apnea_thresh={apnea_thresh}, eupnea_sniff_classifier={saved_classifier}")

            progress.setValue(80)

            # Clear caches (same as new file load)
            self.metrics_by_sweep.clear()
            self.onsets_by_sweep.clear()
            self.global_outlier_stats = None
            self._export_metric_cache = {}
            self.zscore_global_mean = None
            self.zscore_global_std = None

            # Update omit button label
            self._refresh_omit_button_label()

            # Disable peak apply button after session load to prevent accidental re-run
            # User already has peaks loaded from session, shouldn't re-detect
            self.ApplyPeakFindPushButton.setEnabled(False)
            self.ApplyPeakFindPushButton.setToolTip("Peak detection already complete (loaded from session). Modify parameters and click to re-detect.")

            progress.setValue(90)

            # Restore navigation and plot
            self.navigation_manager.reset_window_state()

            # Restore event markers from NPZ if present
            if event_markers and hasattr(self, '_event_marker_viewmodel') and self._event_marker_viewmodel:
                try:
                    count = self._event_marker_viewmodel.load_from_npz(event_markers)
                    print(f"[npz-load] Restored {count} event markers from session file")
                except Exception as e:
                    print(f"[npz-load] Warning: Failed to restore event markers: {e}")

            # Redraw plot (will use single_panel_mode to determine layout)
            self.redraw_main_plot()

            # Restore navigation position (after plotting)
            # Note: Window position is restored in redraw_main_plot via state.window_start_s

            # Restore GMM cache if it was saved (preserves user's cluster assignments)
            if gmm_cache is not None:
                print("[npz-load] Restoring GMM cache from session file...")
                self._cached_gmm_results = gmm_cache
            elif st.gmm_sniff_probabilities:
                # Fallback: rebuild GMM cache if probabilities exist but no cache was saved
                # (for backwards compatibility with old session files)
                # Always rebuild regardless of auto_gmm_enabled setting
                print("[npz-load] Rebuilding GMM cache from loaded probabilities (legacy fallback)...")
                self._run_automatic_gmm_clustering()

            progress.setValue(100)
            progress.close()

            # Show success message
            t_elapsed = time.time() - t_start
            file_size_mb = npz_path.stat().st_size / (1024 * 1024)

            self._log_status_message(
                f"Session loaded: {npz_path.name} ({file_size_mb:.1f} MB, {t_elapsed:.1f}s) - "
                f"Channel: {st.analyze_chan}, {metadata['n_peaks']} peaks",
                timeout=8000
            )

            # Update last directory
            self.settings.setValue("last_dir", str(npz_path.parent))

        except Exception as e:
            progress.close()
            import traceback
            self._show_error("Load Error",
                f"Failed to load session state:\n\n{str(e)}\n\n{traceback.format_exc()}"
            )

    def _proc_key(self, chan: str, sweep: int):
        st = self.state
        return (
            chan, sweep,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert,
            self.filter_order,
            self.notch_filter_lower, self.notch_filter_upper,
            self.use_zscore_normalization
        )

    def _compute_global_zscore_stats(self):
        """
        Compute global mean and std across all sweeps for z-score normalization.
        This ensures all sweeps are normalized relative to the same baseline.
        Uses a progress dialog with processEvents to keep UI responsive.
        """
        import numpy as np

        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return None, None

        Y = st.sweeps[st.analyze_chan]  # (n_samples, n_sweeps)
        n_sweeps = Y.shape[1]
        n_samples = Y.shape[0]
        use_notch = (self.notch_filter_lower is not None and self.notch_filter_upper is not None)

        # For large files, show progress and keep UI responsive
        show_progress = n_sweeps >= 20
        progress = None
        if show_progress:
            progress = QProgressDialog("Computing z-score statistics...", None, 0, n_sweeps, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(300)

        # Pre-allocate array instead of list + concatenate
        all_data = np.empty(n_samples * n_sweeps, dtype=np.float64)

        for sweep_idx in range(n_sweeps):
            y_raw = Y[:, sweep_idx]

            # Apply all filters EXCEPT z-score
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=self.filter_order
            )

            # Apply notch filter if configured
            if use_notch:
                y = self._apply_notch_filter(y, st.sr_hz, self.notch_filter_lower, self.notch_filter_upper)

            all_data[sweep_idx * n_samples:(sweep_idx + 1) * n_samples] = y

            if show_progress:
                progress.setValue(sweep_idx)
                if sweep_idx % 10 == 0:
                    QApplication.processEvents()

        if progress:
            progress.close()

        # Compute global statistics (excluding NaN values)
        valid_mask = ~np.isnan(all_data)
        if not np.any(valid_mask):
            return None, None

        global_mean = np.mean(all_data[valid_mask])
        global_std = np.std(all_data[valid_mask], ddof=1)

        print(f"[zscore] Computed global stats: mean={global_mean:.4f}, std={global_std:.4f}")
        return global_mean, global_std

    def plot_all_channels(self):
        """Delegate to PlotManager."""
        self.plot_manager.plot_all_channels()
        # Refresh event markers for current sweep (same as redraw_main_plot)
        self._refresh_event_markers()

    def on_analyze_channel_changed(self, idx: int):
        """Apply analyze channel selection immediately."""
        st = self.state
        if not st.channel_names:
            return

        # Check if "All Channels" was selected (idx 0)
        if idx == 0:
            # Switch to grid mode (multi-channel view)
            if self.single_panel_mode:
                self.single_panel_mode = False
                st.analyze_chan = None

                # Clear stimulus data but keep the channel selected in dropdown
                # so it will be recomputed when switching back to single channel
                st.stim_onsets_by_sweep.clear()
                st.stim_offsets_by_sweep.clear()
                st.stim_spans_by_sweep.clear()
                st.stim_metrics_by_sweep.clear()

                # Clear event markers (bout annotations)
                if hasattr(st, 'bout_annotations'):
                    st.bout_annotations.clear()

                st.proc_cache.clear()

                # Clear Y2 plot data
                st.y2_metric_key = None
                st.y2_values_by_sweep.clear()
                self.plot_host.clear_y2()
                # Reset Y2 dropdown to "None"
                self.y2plot_dropdown.blockSignals(True)
                self.y2plot_dropdown.setCurrentIndex(0)  # First item is "None"
                self.y2plot_dropdown.blockSignals(False)

                # Clear saved view to force fresh autoscale for grid mode
                self.plot_host.clear_saved_view("grid")
                self.plot_host.clear_saved_view("single")

                # Switch to grid plot
                self.plot_all_channels()
        elif 0 < idx <= len(st.channel_names):
            # Switch to single channel view
            new_chan = st.channel_names[idx - 1]  # -1 because idx 0 is "All Channels"
            if new_chan != st.analyze_chan or not self.single_panel_mode:
                # Check if session files exist for this channel in analysis folder
                if st.in_path and new_chan:
                    from core.npz_io import get_npz_metadata
                    from datetime import datetime

                    # Search analysis folder for session files matching this channel
                    analysis_folder = st.in_path.parent / "Pleth_App_Analysis"
                    session_files = []

                    if analysis_folder.exists():
                        safe_channel = new_chan.replace(' ', '_').replace('/', '_').replace('\\', '_')
                        data_stem = st.in_path.stem
                        pattern = f"*_{data_stem}_{safe_channel}_session.npz"
                        session_files = list(analysis_folder.glob(pattern))

                        # Sort by modification time (newest first)
                        session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

                    if session_files:
                        # Found session file(s) - use most recent
                        npz_path = session_files[0]

                        # Get metadata for display
                        metadata = get_npz_metadata(npz_path)

                        # Prompt user to load existing analysis
                        reply = QMessageBox.question(
                            self,
                            "Load Existing Analysis?",
                            f"Found saved analysis for channel '{new_chan}':\n\n"
                            f"{npz_path.name}\n"
                            f"Last modified: {metadata.get('modified_time', 'unknown')}\n"
                            f"Contains: {metadata.get('n_peaks', 0)} peaks\n"
                            f"GMM clustering: {'Yes' if metadata.get('has_gmm', False) else 'No'}\n\n"
                            f"Load this analysis?",
                            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                            QMessageBox.StandardButton.Yes  # Default to Yes
                        )

                        if reply == QMessageBox.StandardButton.Yes:
                            # User wants to load existing analysis
                            # Block signals to prevent recursive calls
                            self.AnalyzeChanSelect.blockSignals(True)
                            self.load_npz_state(npz_path)
                            self.AnalyzeChanSelect.blockSignals(False)

                            # Still run auto-detect to populate prominence field for this channel
                            self._auto_detect_prominence_silent()
                            return  # Don't continue with fresh channel switch

                # User declined or no NPZ exists - continue with fresh channel
                st.analyze_chan = new_chan

                # Log telemetry: channel selection
                telemetry.log_button_click('select_analyze_channel',
                                          channel_name=new_chan,
                                          channel_index=idx)

                st.proc_cache.clear()
                # Clear z-score global statistics cache
                self.zscore_global_mean = None
                self.zscore_global_std = None
                st.peaks_by_sweep.clear()
                st.sigh_by_sweep.clear()
                if hasattr(st, 'breath_by_sweep'):
                    st.breath_by_sweep.clear()

                # Clear omitted sweeps and regions
                st.omitted_sweeps.clear()
                st.omitted_ranges.clear()
                self._refresh_omit_button_label()

                # Clear sniffing regions
                if hasattr(st, 'sniff_regions_by_sweep'):
                    st.sniff_regions_by_sweep.clear()

                # Clear event markers (bout annotations)
                if hasattr(st, 'bout_annotations'):
                    st.bout_annotations.clear()

                # Immediately clear and refresh the Analysis Options dialog if open
                if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None:
                    try:
                        if self._analysis_options_dialog.isVisible():
                            current_tab = self._analysis_options_dialog.tabWidget.currentIndex()
                            print(f"[Channel Change] Clearing Analysis Options dialog (current tab: {current_tab})")

                            # Clear the cached dialogs so they'll recreate
                            if hasattr(self._analysis_options_dialog, 'prominence_dialog'):
                                self._analysis_options_dialog.prominence_dialog = None
                            if hasattr(self._analysis_options_dialog, 'gmm_dialog'):
                                self._analysis_options_dialog.gmm_dialog = None

                            # Immediately clear the current tab's content
                            if current_tab == 0:  # Peak Detection
                                container = self._analysis_options_dialog.peak_detection_container
                            elif current_tab == 1:  # Eup/Sniff Classification (GMM)
                                container = self._analysis_options_dialog.breath_classification_container
                            else:
                                container = None

                            if container:
                                layout = container.layout()
                                if layout:
                                    # Clear all widgets from the container
                                    while layout.count():
                                        child = layout.takeAt(0)
                                        if child.widget():
                                            child.widget().deleteLater()
                                    from PyQt6.QtWidgets import QApplication
                                    QApplication.processEvents()
                                    print(f"[Channel Change] Cleared tab {current_tab} content")

                                # Show "Loading..." placeholder
                                from PyQt6.QtWidgets import QLabel
                                from PyQt6.QtCore import Qt
                                label = QLabel("Loading new channel data...")
                                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                                label.setStyleSheet("padding: 20px; font-size: 14pt; color: #888;")
                                layout.addWidget(label)

                            # Mark for refresh after peak detection completes
                            self._channel_change_needs_dialog_refresh = True
                            self._dialog_tab_to_refresh = current_tab
                    except RuntimeError:
                        # Dialog was deleted
                        self._analysis_options_dialog = None

                # Clear Y2 plot data
                st.y2_metric_key = None
                st.y2_values_by_sweep.clear()
                self.plot_host.clear_y2()
                # Reset Y2 dropdown to "None"
                self.y2plot_dropdown.blockSignals(True)
                self.y2plot_dropdown.setCurrentIndex(0)  # First item is "None"
                self.y2plot_dropdown.blockSignals(False)

                # Reset navigation to first sweep
                st.sweep_idx = 0
                st.window_start_s = 0.0

                # Switch to single panel mode
                if not self.single_panel_mode:
                    self.single_panel_mode = True

                # If a stimulus channel is selected, recompute it for the current sweep
                if st.stim_chan is not None:
                    self._compute_stim_for_current_sweep()

                # Clear saved view to force fresh autoscale for single mode
                self.plot_host.clear_saved_view("single")
                self.plot_host.clear_saved_view("grid")

                # Auto-detect optimal prominence in background when channel selected
                self._auto_detect_prominence_silent()

                # Redraw plot
                self.redraw_main_plot()

                # Redraw threshold line after plot is redrawn (it gets cleared during redraw)
                if self.peak_height_threshold is not None:
                    self.plot_host.update_threshold_line(self.peak_height_threshold)

    def on_stim_channel_changed(self, idx: int):
        """Apply stimulus channel selection immediately."""
        st = self.state
        if not st.channel_names:
            return

        # idx 0 = "None", idx 1+ = channel names
        new_stim = None if idx == 0 else st.channel_names[idx - 1]
        if new_stim != st.stim_chan:
            st.stim_chan = new_stim

            # Clear stimulus detection results
            st.stim_onsets_by_sweep.clear()
            st.stim_offsets_by_sweep.clear()
            st.stim_spans_by_sweep.clear()
            st.stim_metrics_by_sweep.clear()

            # Compute stimulus for current sweep if a channel is selected
            if new_stim is not None:
                self._compute_stim_for_current_sweep()

            # Clear saved view to force fresh autoscale when stimulus changes
            self.plot_host.clear_saved_view("single")

            st.proc_cache.clear()
            self.redraw_main_plot()

    def on_events_channel_changed(self, idx: int):
        """Apply event channel selection immediately."""
        st = self.state
        if not st.channel_names:
            return

        # idx 0 = "None", idx 1+ = channel names
        new_event = None if idx == 0 else st.channel_names[idx - 1]
        if new_event != st.event_channel:
            st.event_channel = new_event
            self.redraw_main_plot()

    def _update_classifier_dropdowns(self):
        """Update dropdown options based on loaded models. Delegates to ClassifierManager."""
        self._classifier_manager.update_classifier_dropdowns()

    def _fallback_disabled_classifiers(self):
        """Check if current classifier selections are disabled. Delegates to ClassifierManager."""
        self._classifier_manager.fallback_disabled_classifiers()

    def _auto_rerun_peak_detection_if_needed(self):
        """
        Automatically re-run peak detection if:
        1. Models were just loaded
        2. Peaks have already been detected once (state.all_peaks_by_sweep is not empty)

        This updates predictions with newly loaded models without requiring manual re-detection.
        """
        # Only re-run if peaks have been detected
        if not self.state.all_peaks_by_sweep:
            print("[Auto Re-run] Skipping - no peaks detected yet")
            return

        # Only re-run if we have models loaded
        if not self.state.loaded_ml_models:
            print("[Auto Re-run] Skipping - no models loaded")
            return

        print("[Auto Re-run] Triggering peak detection to update predictions with newly loaded models")
        self.statusBar().showMessage("Updating predictions with loaded models...", 2000)

        # Trigger peak detection (this will use newly loaded models)
        self._apply_peak_detection()

    def _auto_load_ml_models_on_startup(self):
        """Silently load ML models on startup. Delegates to ClassifierManager."""
        self._classifier_manager.auto_load_ml_models_on_startup()

    def on_classifier_changed(self, text: str):
        """Handle classifier selection change. Delegates to ClassifierManager."""
        self._classifier_manager.on_classifier_changed(text)

    def _update_displayed_peaks_from_classifier(self):
        """Update peaks_by_sweep based on active classifier. Delegates to ClassifierManager."""
        self._classifier_manager.update_displayed_peaks_from_classifier()

    def on_eupnea_sniff_classifier_changed(self, text: str):
        """Handle eupnea/sniff classifier selection. Delegates to ClassifierManager."""
        self._classifier_manager.on_eupnea_sniff_classifier_changed(text)

    def _update_eupnea_sniff_from_classifier(self):
        """Copy eupnea/sniff predictions to breath_type_class. Delegates to ClassifierManager."""
        self._classifier_manager.update_eupnea_sniff_from_classifier()

    def _set_all_breaths_eupnea_sniff_class(self, class_value: int):
        """Set all breaths to a specific eupnea/sniff class. Delegates to ClassifierManager."""
        self._classifier_manager.set_all_breaths_eupnea_sniff_class(class_value)

    def _clear_all_eupnea_sniff_labels(self):
        """Clear all eupnea/sniff labels. Delegates to ClassifierManager."""
        self._classifier_manager.clear_all_eupnea_sniff_labels()

    def _clear_all_sigh_labels(self):
        """Clear all sigh labels. Delegates to ClassifierManager."""
        self._classifier_manager.clear_all_sigh_labels()

    def on_sigh_classifier_changed(self, text: str):
        """Handle sigh classifier selection. Delegates to ClassifierManager."""
        self._classifier_manager.on_sigh_classifier_changed(text)

    def _update_sigh_from_classifier(self):
        """Copy sigh predictions to sigh_class. Delegates to ClassifierManager."""
        self._classifier_manager.update_sigh_from_classifier()

    def on_mark_events_clicked(self):
        """Open Event Detection Settings dialog."""
        # Check if event channel is selected (from Channel Manager)
        st = self.state

        # Try to get event channel from Channel Manager first
        if hasattr(self, 'channel_manager'):
            event_chan = self.channel_manager.get_event_channel()
            if event_chan and event_chan != st.event_channel:
                # Sync state with Channel Manager
                st.event_channel = event_chan

        if st.event_channel is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Event Channel",
                "Please set a channel type to 'Event' in the Channel Manager first.\n\n"
                "1. Click the Channel Manager expand button (▼)\n"
                "2. Find your event channel (e.g., lick detector)\n"
                "3. Change its type dropdown to 'Event'\n"
                "4. Click Apply"
            )
            return

        # Check if dialog already exists and is visible
        if hasattr(self, '_event_detection_dialog') and self._event_detection_dialog.isVisible():
            # Bring existing dialog to front
            self._event_detection_dialog.raise_()
            self._event_detection_dialog.activateWindow()
            return

        # Create and show dialog (non-modal so plot can be interacted with)
        from dialogs.event_detection_dialog import EventDetectionDialog
        self._event_detection_dialog = EventDetectionDialog(parent=self, main_window=self)
        self._event_detection_dialog.show()  # Non-modal dialog


    def _compute_stim_for_current_sweep(self, thresh: float = 1.0):
        st = self.state
        if not st.stim_chan or st.stim_chan not in st.sweeps:
            return
        Y = st.sweeps[st.stim_chan]
        s = max(0, min(st.sweep_idx, Y.shape[1] - 1))
        y = Y[:, s]
        t = st.t

        on_idx, off_idx, spans_s, metrics = stimdet.detect_threshold_crossings(y, t, thresh=thresh)
        st.stim_onsets_by_sweep[s] = on_idx
        st.stim_offsets_by_sweep[s] = off_idx
        st.stim_spans_by_sweep[s] = spans_s
        st.stim_metrics_by_sweep[s] = metrics

        # Debug print
        if metrics:
            pw = metrics.get("pulse_width_s")
            dur = metrics.get("duration_s")
            hz = metrics.get("freq_hz")
            msg = f"[stim] sweep {s}: width={pw:.6f}s, duration={dur:.6f}s"
            if hz:
                msg += f", freq={hz:.3f}Hz"
            print(msg)

    def _detect_stims_all_sweeps(self, thresh: float = 1.0):
        """Detect stimulations on all sweeps (for export/preview)."""
        st = self.state
        if not st.stim_chan or st.stim_chan not in st.sweeps:
            return

        Y = st.sweeps[st.stim_chan]
        n_sweeps = Y.shape[1]
        t = st.t

        for s in range(n_sweeps):
            # Skip if already detected
            if s in st.stim_spans_by_sweep:
                continue

            y = Y[:, s]
            on_idx, off_idx, spans_s, metrics = stimdet.detect_threshold_crossings(y, t, thresh=thresh)
            st.stim_onsets_by_sweep[s] = on_idx
            st.stim_offsets_by_sweep[s] = off_idx
            st.stim_spans_by_sweep[s] = spans_s
            st.stim_metrics_by_sweep[s] = metrics

        print(f"[stim] Detected stims for all {n_sweeps} sweeps")

    # ---------- Filters & redraw ----------
    def update_and_redraw(self, *args):
        st = self.state

        # checkboxes
        st.use_low       = self.LowPass_checkBox.isChecked()
        st.use_high      = self.HighPass_checkBox.isChecked()
        # Mean subtraction is now controlled from Spectral Analysis dialog
        # st.use_mean_sub is set directly in the dialog handlers
        st.use_invert    = self.InvertSignal_checkBox.isChecked()

        # Filter order
        self.filter_order = self.FilterOrderSpin.value()


        # Peaks/breaths/classifications no longer valid if filters change
        if hasattr(self.state, "peaks_by_sweep"):
            self.state.peaks_by_sweep.clear()
        if hasattr(self.state, "breath_by_sweep"):
            self.state.breath_by_sweep.clear()
        if hasattr(self.state, "all_peaks_by_sweep"):
            self.state.all_peaks_by_sweep.clear()
        if hasattr(self.state, "sigh_by_sweep"):
            self.state.sigh_by_sweep.clear()
        if hasattr(self.state, "sniff_regions_by_sweep"):
            self.state.sniff_regions_by_sweep.clear()
        if hasattr(self.state, "y2_values_by_sweep"):
            self.state.y2_values_by_sweep.clear()
        if hasattr(self, 'plot_host') and self.plot_host:
            self.plot_host.clear_y2()

        # Re-enable detect button since peaks were just cleared
        if self.state.analyze_chan:
            self.ApplyPeakFindPushButton.setEnabled(True)

        # Clear z-score global statistics cache (filters changed)
        self.zscore_global_mean = None
        self.zscore_global_std = None



        def _val_if_enabled(line, checked: bool, cast=float, default=None):
            """Return a numeric value only if the box is checked and a value exists."""
            if not checked:
                return None
            txt = line.text().strip()
            if not txt:
                return None
            try:
                return cast(txt)
            except ValueError:
                return None

        # only take values if box is checked AND entry is valid
        st.low_hz  = _val_if_enabled(self.LowPassVal, st.use_low, float, None)
        st.high_hz = _val_if_enabled(self.HighPassVal, st.use_high, float, None)
        # Mean subtraction value is now controlled from Spectral Analysis dialog
        # st.mean_win_s is set directly in the dialog handlers

        # If the checkbox is checked but the box is empty/invalid, disable that filter automatically
        if st.use_low and st.low_hz is None:
            st.use_low = False
        if st.use_high and st.high_hz is None:
            st.use_high = False
        # Mean subtraction validation is handled in Spectral Analysis dialog

        # Invalidate processed cache
        st.proc_cache.clear()

        # Log telemetry and re-enable Apply button
        self._on_filter_changed()

        # Debounce redraw
        self._redraw_timer.start()

    def _current_trace(self):
        """Return (t, y_proc) for analyze channel & current sweep, using cached processing."""
        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return None, None

        Y = st.sweeps[st.analyze_chan]
        s = max(0, min(st.sweep_idx, Y.shape[1] - 1))
        key = self._proc_key(st.analyze_chan, s)

        # Fast path: reuse processed data if settings didn't change
        if key in st.proc_cache:
            return st.t, st.proc_cache[key]

        # Compute once, cache, and return
        y = Y[:, s]
        y2 = filters.apply_all_1d(
            y, st.sr_hz,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert,
            order=self.filter_order
        )

        # Apply notch filter if configured
        if self.notch_filter_lower is not None and self.notch_filter_upper is not None:
            y2 = self._apply_notch_filter(y2, st.sr_hz, self.notch_filter_lower, self.notch_filter_upper)

        # Apply z-score normalization if enabled (using global statistics)
        if self.use_zscore_normalization:
            # Compute global stats if not cached
            if self.zscore_global_mean is None or self.zscore_global_std is None:
                self.zscore_global_mean, self.zscore_global_std = self._compute_global_zscore_stats()
            y2 = filters.zscore_normalize(y2, self.zscore_global_mean, self.zscore_global_std)

        st.proc_cache[key] = y2
        return st.t, y2

    def redraw_main_plot(self):
        """Delegate to PlotManager."""
        self.plot_manager.redraw_main_plot()
        # Refresh event markers for current sweep
        self._refresh_event_markers()


    ##################################################
    ##Region threshold visualization                ##
    ##################################################
    def _on_region_threshold_changed(self, *_):
        """
        Called whenever eupnea or apnea threshold values change.
        Redraws the current sweep to update region overlays.
        """
        # Simply redraw current sweep, which will use the new threshold values
        self.redraw_main_plot()

    def _on_filter_changed(self, *_):
        """
        Called whenever filter settings change.
        Re-enables Apply button since peaks need to be recalculated with new filtering.
        """
        st = self.state

        # Log telemetry: filter settings changed
        telemetry.log_button_click('filter_changed',
                                   use_low=st.use_low,
                                   low_hz=st.low_hz if st.use_low else None,
                                   use_high=st.use_high,
                                   high_hz=st.high_hz if st.use_high else None,
                                   use_mean_sub=st.use_mean_sub,
                                   use_invert=st.use_invert)

        # Only re-enable if we have a threshold value and an analysis channel
        if self.peak_prominence is not None and self.state.analyze_chan:
            self.ApplyPeakFindPushButton.setEnabled(True)

    ##################################################
    ##Peak detection parameters                     ##
    ##################################################

    def _parse_float(self, line_edit):
        txt = line_edit.text().strip()
        if not txt:
            return None
        try:
            return float(txt)
        except ValueError:
            return None

    def _get_processed_for(self, chan: str, sweep_idx: int):
        """Return processed y for (channel, sweep_idx) using the same cache key logic.

        Filters are only applied to the primary Pleth channel (st.analyze_chan).
        All other channels return raw data — the current filter settings (HP/LP/notch)
        are designed for plethysmography signals and would distort other signal types.
        """
        st = self.state
        Y = st.sweeps[chan]
        s = max(0, min(sweep_idx, Y.shape[1]-1))

        # Only apply filters to the primary pleth channel
        if chan != st.analyze_chan:
            return Y[:, s].copy()

        key = (chan, s, st.use_low, st.low_hz, st.use_high, st.high_hz, st.use_mean_sub, st.mean_val, st.use_invert,
               self.notch_filter_lower, self.notch_filter_upper, self.use_zscore_normalization)
        if key in st.proc_cache:
            return st.proc_cache[key]
        y = Y[:, s]
        y2 = filters.apply_all_1d(
            y, st.sr_hz,
            st.use_low,  st.low_hz,
            st.use_high, st.high_hz,
            st.use_mean_sub, st.mean_val,
            st.use_invert
        )

        # Apply notch filter if configured
        if self.notch_filter_lower is not None and self.notch_filter_upper is not None:
            y2 = self._apply_notch_filter(y2, st.sr_hz, self.notch_filter_lower, self.notch_filter_upper)

        # Apply z-score normalization if enabled (using global statistics)
        if self.use_zscore_normalization:
            # Compute global stats if not cached
            if self.zscore_global_mean is None or self.zscore_global_std is None:
                self.zscore_global_mean, self.zscore_global_std = self._compute_global_zscore_stats()
            y2 = filters.zscore_normalize(y2, self.zscore_global_mean, self.zscore_global_std)

        st.proc_cache[key] = y2
        return y2

    def _apply_notch_filter(self, y, sr_hz, lower_freq, upper_freq):
        """Apply a notch (band-stop) filter to remove frequencies between lower_freq and upper_freq."""
        from scipy import signal
        import numpy as np

        print(f"[notch-filter] Applying notch filter: {lower_freq:.2f} - {upper_freq:.2f} Hz (sr={sr_hz} Hz)")

        # Design a butterworth band-stop filter
        nyquist = sr_hz / 2.0
        low = lower_freq / nyquist
        high = upper_freq / nyquist

        # Ensure frequencies are in valid range (0, 1)
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)

        if low >= high:
            print(f"[notch-filter] Invalid frequency range: {lower_freq}-{upper_freq} Hz")
            return y

        try:
            # Design 4th order Butterworth band-stop filter
            sos = signal.butter(4, [low, high], btype='bandstop', output='sos')
            # Apply filter (sos format is more numerically stable)
            y_filtered = signal.sosfiltfilt(sos, y)
            print(f"[notch-filter] Filter applied successfully. Signal range before: [{y.min():.3f}, {y.max():.3f}], after: [{y_filtered.min():.3f}, {y_filtered.max():.3f}]")
            return y_filtered
        except Exception as e:
            print(f"[notch-filter] Error applying filter: {e}")
            return y

    def _open_analysis_options(self, tab=None):
        """
        Open the multi-tabbed Analysis Options dialog.
        Consolidates Peak Detection, GMM Clustering, Outlier Detection, and ML Settings.
        Non-blocking so user can interact with main window while dialog is open.

        Args:
            tab (str, optional): Tab to open ('peak_detection', 'gmm', 'outliers', 'ml')
        """
        from dialogs.analysis_options_dialog import AnalysisOptionsDialog
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt

        st = self.state

        # Check if dialog already exists (reuse to preserve settings like last training data path)
        if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None:
            # Reuse existing dialog and bring to front
            self._analysis_options_dialog.show()
            self._analysis_options_dialog.raise_()
            self._analysis_options_dialog.activateWindow()
        else:
            # Show loading cursor while creating dialog (may take a moment for tab initialization)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QApplication.processEvents()  # Ensure cursor change is visible
            try:
                # Create and show the dialog (non-blocking)
                # Note: Individual tabs handle their own data requirements (e.g., ML Settings works without data)
                self._analysis_options_dialog = AnalysisOptionsDialog(st, parent=self)
                self._analysis_options_dialog.finished.connect(self._on_analysis_options_closed)
                self._analysis_options_dialog.show()
            finally:
                QApplication.restoreOverrideCursor()

        # Switch to requested tab if specified
        if tab and self._analysis_options_dialog is not None:
            self._analysis_options_dialog.set_active_tab(tab)

    def _on_analysis_options_closed(self):
        """Hide dialog when closed but keep it alive for fast re-opening."""
        # Don't destroy the dialog - keep it alive for instant re-opening
        # The dialog will be reused in _open_analysis_options()
        pass

    def _prebuild_analysis_options_dialog(self):
        """Pre-build the Analysis Options dialog in background for instant opening later."""
        # Only pre-build if dialog doesn't already exist
        if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None:
            return

        # Only pre-build if we have the required data
        if not hasattr(self, 'all_peak_heights') or self.all_peak_heights is None:
            return

        try:
            from dialogs.analysis_options_dialog import AnalysisOptionsDialog
            print("[Pre-build] Creating Analysis Options dialog in background...")

            # Create dialog but don't show it
            self._analysis_options_dialog = AnalysisOptionsDialog(self.state, parent=self)
            self._analysis_options_dialog.finished.connect(self._on_analysis_options_closed)

            print("[Pre-build] Dialog ready - will open instantly when requested")
        except Exception as e:
            print(f"[Pre-build] Error creating dialog: {e}")
            self._analysis_options_dialog = None

    def _open_prominence_histogram(self):
        """
        Open the multi-tab Analysis Options dialog to the Auto-Threshold tab.
        (Replaces the old single ProminenceThresholdDialog)
        """
        self._open_analysis_options(tab='peak_detection')
        telemetry.log_screen_view('Peak Detection Options Dialog', screen_class='config_dialog')

    def _calculate_local_minimum_threshold_silent(self, peak_heights):
        """
        Calculate valley threshold using exponential + Gaussian mixture model.
        Simplified version for silent auto-detection (no UI feedback).

        Args:
            peak_heights: Array of detected peak heights

        Returns:
            tuple: (valley_threshold, model_params_dict) or (None, None) if fitting fails
            model_params_dict contains: lambda_exp, mu1, sigma1, mu2, sigma2, w_exp, w_g1, w_g2
        """
        try:
            from scipy.optimize import curve_fit

            # Use 99th percentile to exclude outliers
            percentile_95 = np.percentile(peak_heights, 99)
            peaks_for_hist = peak_heights[peak_heights <= percentile_95]

            if len(peaks_for_hist) < 10:
                return (None, None)

            # Create histogram
            hist_range = (peaks_for_hist.min(), percentile_95)
            counts, bin_edges = np.histogram(peaks_for_hist, bins=200, range=hist_range)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]

            # Convert to density
            density = counts / (len(peaks_for_hist) * bin_width)

            # Try 2-Gaussian model first (for eupnea + sniffing)
            def exp_2gauss_model(x, lambda_exp, mu1, sigma1, mu2, sigma2, w_exp, w_g1):
                exp_comp = lambda_exp * np.exp(-lambda_exp * x)
                gauss1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
                gauss2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)
                w_g2 = max(0, 1 - w_exp - w_g1)
                return w_exp * exp_comp + w_g1 * gauss1 + w_g2 * gauss2

            try:
                p0_2g = [
                    1.0 / np.mean(bin_centers),  # lambda_exp
                    np.percentile(bin_centers, 40),  # mu1 (eupnea)
                    np.std(bin_centers) * 0.3,  # sigma1
                    np.percentile(bin_centers, 70),  # mu2 (sniffing)
                    np.std(bin_centers) * 0.3,  # sigma2
                    0.3,  # w_exp
                    0.4   # w_g1
                ]
                popt, _ = curve_fit(exp_2gauss_model, bin_centers, density, p0=p0_2g, maxfev=5000)
                fitted = exp_2gauss_model(bin_centers, *popt)

                # Check fit quality
                residuals = density - fitted
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((density - np.mean(density)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                if r_squared >= 0.7 and popt[5] >= 0.05 and popt[6] >= 0.05:
                    # Find valley between 0 and first Gaussian peak
                    search_end_idx = np.argmin(np.abs(bin_centers - popt[1]))
                    valley_idx = np.argmin(fitted[:search_end_idx])
                    threshold = float(bin_centers[valley_idx])

                    # Store model parameters for probability metrics
                    model_params = {
                        'lambda_exp': float(popt[0]),
                        'mu1': float(popt[1]),
                        'sigma1': float(popt[2]),
                        'mu2': float(popt[3]),
                        'sigma2': float(popt[4]),
                        'w_exp': float(popt[5]),
                        'w_g1': float(popt[6]),
                        'w_g2': float(max(0, 1 - popt[5] - popt[6]))
                    }
                    return (threshold, model_params)
            except:
                pass

            # Fallback: 1-Gaussian model
            def exp_gauss_model(x, lambda_exp, mu_gauss, sigma_gauss, w_exp):
                exp_component = lambda_exp * np.exp(-lambda_exp * x)
                gauss_component = (1 / (np.sqrt(2 * np.pi) * sigma_gauss)) * np.exp(-0.5 * ((x - mu_gauss) / sigma_gauss) ** 2)
                return w_exp * exp_component + (1 - w_exp) * gauss_component

            p0_1g = [
                1.0 / np.mean(bin_centers),
                np.median(bin_centers),
                np.std(bin_centers),
                0.3
            ]
            popt, _ = curve_fit(exp_gauss_model, bin_centers, density, p0=p0_1g, maxfev=5000)
            fitted = exp_gauss_model(bin_centers, *popt)

            # Check fit quality
            residuals = density - fitted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((density - np.mean(density)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared >= 0.7:
                # Find valley between 0 and Gaussian peak
                search_end_idx = np.argmin(np.abs(bin_centers - popt[1]))
                valley_idx = np.argmin(fitted[:search_end_idx])
                threshold = float(bin_centers[valley_idx])

                # Store model parameters for probability metrics (1-Gaussian fallback)
                model_params = {
                    'lambda_exp': float(popt[0]),
                    'mu1': float(popt[1]),
                    'sigma1': float(popt[2]),
                    'mu2': float(popt[1]),  # Same as mu1 for 1-Gaussian
                    'sigma2': float(popt[2]),  # Same as sigma1
                    'w_exp': float(popt[3]),
                    'w_g1': float(1 - popt[3]),
                    'w_g2': 0.0  # No second Gaussian in fallback
                }
                return (threshold, model_params)

            return (None, None)

        except Exception as e:
            print(f"[Valley Fit] Error: {e}")
            return (None, None)

    def _auto_detect_prominence_silent(self):
        """
        Auto-detect optimal prominence using Otsu's method in background (no dialog).
        Populates prominence field and enables Apply button.
        """
        import time
        from scipy.signal import find_peaks
        import numpy as np

        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        try:
            # Concatenate ALL sweeps for representative auto-threshold calculation
            print("[Auto-Detect] Calculating optimal prominence...")
            t_start = time.time()

            all_sweeps_data = []
            n_sweeps = st.sweeps[st.analyze_chan].shape[1]

            for sweep_idx in range(n_sweeps):
                if sweep_idx in st.omitted_sweeps:
                    continue
                y_sweep = self._get_processed_for(st.analyze_chan, sweep_idx)
                all_sweeps_data.append(y_sweep)

            if not all_sweeps_data:
                return

            y_data = np.concatenate(all_sweeps_data)

            # Detect all peaks with minimal prominence AND above baseline (height > 0)
            # height=0 filters out rebound peaks below baseline, giving cleaner 2-population model
            min_dist_samples = max(1, int(self.peak_min_dist * st.sr_hz))  # Ensure at least 1 for low sample rates
            peaks, props = find_peaks(y_data, height=0, prominence=0.001, distance=min_dist_samples)
            peak_heights = y_data[peaks]

            if len(peak_heights) < 10:
                print("[Auto-Detect] Not enough peaks found")
                return

            # Store peak heights for histogram reuse (so we don't recalculate during dragging)
            self.all_peak_heights = peak_heights

            # Otsu's method: auto-calculate optimal HEIGHT threshold
            heights_norm = ((peak_heights - peak_heights.min()) /
                        (peak_heights.max() - peak_heights.min()) * 255).astype(np.uint8)

            hist, bin_edges = np.histogram(heights_norm, bins=256, range=(0, 256))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2 + 1e-10))[::-1]

            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            optimal_bin = np.argmax(variance)
            optimal_thresh_norm = bin_centers[optimal_bin]

            # Convert back to original scale
            optimal_height = float((optimal_thresh_norm / 255.0 *
                            (peak_heights.max() - peak_heights.min()) +
                            peak_heights.min()))

            # Calculate local minimum threshold (valley between noise and signal)
            # This is more robust than Otsu for breath signals
            local_min_threshold, model_params = self._calculate_local_minimum_threshold_silent(peak_heights)

            # Choose threshold: prefer local minimum if available, fallback to Otsu
            if local_min_threshold is not None:
                chosen_threshold = local_min_threshold
                print(f"[Auto-Detect] Using valley threshold: {chosen_threshold:.4f} (Otsu: {optimal_height:.4f})")

                # Store model parameters for probability metrics
                import core.metrics as metrics
                metrics.set_threshold_model_params(model_params)
                print(f"[Auto-Detect] Stored model parameters for P(noise)/P(breath) calculation")
            else:
                chosen_threshold = optimal_height
                print(f"[Auto-Detect] Using Otsu threshold: {chosen_threshold:.4f} (no valley found)")
                # Clear model parameters if valley fit failed
                import core.metrics as metrics
                metrics.set_threshold_model_params(None)

            # Store and populate spinbox with auto-detected value
            # Use same value for both height and prominence thresholds
            self.peak_prominence = chosen_threshold
            # PeakPromValueSpinBox removed - threshold now shown in Analysis Options dialog
            # self.PeakPromValueSpinBox.setValue(chosen_threshold)

            # Store the height threshold value (will be used in peak detection)
            self.peak_height_threshold = chosen_threshold

            # Draw threshold line on plot
            self.plot_host.update_threshold_line(chosen_threshold)

            # Enable Apply button
            self.ApplyPeakFindPushButton.setEnabled(True)

            t_elapsed = time.time() - t_start
            # Status message already printed above with valley/Otsu choice
            self._log_status_message(f"Auto-detected threshold: {chosen_threshold:.4f}", 3000)

            # Pre-build the Analysis Options dialog in background (deferred)
            # This makes opening the dialog instant since it's already created
            QTimer.singleShot(500, self._prebuild_analysis_options_dialog)

            # Refresh Peak Detection tab if it's the one that needs updating after channel change
            if (hasattr(self, '_channel_change_needs_dialog_refresh') and
                self._channel_change_needs_dialog_refresh and
                getattr(self, '_dialog_tab_to_refresh', -1) == 0):  # Tab 0 = Peak Detection
                if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None:
                    try:
                        if self._analysis_options_dialog.isVisible():
                            print("[Auto-Detect] Refreshing Peak Detection tab after auto-detect complete")
                            self._analysis_options_dialog._refresh_peak_detection_tab()
                            self._channel_change_needs_dialog_refresh = False  # Mark as done
                    except RuntimeError:
                        self._analysis_options_dialog = None

        except Exception as e:
            print(f"[Auto-Detect] Error: {e}")
            import traceback
            traceback.print_exc()

    # PeakPromValueSpinBox removed - prominence/threshold now set in Analysis Options dialog
    # def _on_prominence_spinbox_changed(self):
    #     """Update threshold line on plot when spinbox value changes."""
    #     new_value = self.PeakPromValueSpinBox.value()
    #     if new_value > 0:
    #         self.peak_height_threshold = new_value
    #         self.plot_host.update_threshold_line(new_value)
    #
    #         # Synchronize with auto-threshold workflow: fit model for p_noise calculation
    #         self._fit_threshold_model_for_manual_slider()

    def _fit_threshold_model_for_manual_slider(self):
        """
        Fit the exponential + 2 Gaussians model for p_noise calculation.
        This is called when the manual threshold slider changes to synchronize
        with the auto-threshold workflow.
        """
        from scipy.signal import find_peaks
        import core.metrics as metrics

        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        try:
            # Check if we already have peak_heights from a previous auto-detect
            if not hasattr(self, 'all_peak_heights') or self.all_peak_heights is None:
                # Need to compute peak_heights by detecting peaks from current data
                all_sweeps_data = []
                n_sweeps = st.sweeps[st.analyze_chan].shape[1]

                for sweep_idx in range(n_sweeps):
                    if sweep_idx in st.omitted_sweeps:
                        continue
                    y_sweep = self._get_processed_for(st.analyze_chan, sweep_idx)
                    if y_sweep is not None:
                        all_sweeps_data.append(y_sweep)

                if not all_sweeps_data:
                    # No data available, clear model params
                    metrics.set_threshold_model_params(None)
                    return

                y_data = np.concatenate(all_sweeps_data)

                # Detect all peaks with minimal prominence AND above baseline (height > 0)
                min_dist_samples = max(1, int(self.peak_min_dist * st.sr_hz))  # Ensure at least 1 for low sample rates
                peaks, props = find_peaks(y_data, height=0, prominence=0.001, distance=min_dist_samples)
                peak_heights = y_data[peaks]

                if len(peak_heights) < 10:
                    # Not enough peaks, clear model params
                    metrics.set_threshold_model_params(None)
                    return

                # Store for reuse
                self.all_peak_heights = peak_heights

            # Fit the model using the peak heights
            local_min_threshold, model_params = self._calculate_local_minimum_threshold_silent(self.all_peak_heights)

            # Store model parameters for p_noise calculation
            if model_params is not None:
                metrics.set_threshold_model_params(model_params)
                print(f"[Manual Threshold] Fitted model for p_noise calculation")
            else:
                # Model fit failed, clear params
                metrics.set_threshold_model_params(None)
                print(f"[Manual Threshold] Model fit failed, p_noise will be unavailable")

        except Exception as e:
            print(f"[Manual Threshold] Error fitting model: {e}")
            import traceback
            traceback.print_exc()
            # Clear model params on error
            metrics.set_threshold_model_params(None)

    def _precompute_remaining_classifiers_async(self):
        """Pre-compute ML classifiers in background. Delegates to ClassifierManager."""
        self._classifier_manager.precompute_remaining_classifiers_async()

    def _detect_single_sweep_core(self, sweep_idx: int, y_proc: np.ndarray,
                                   thresh: float, min_dist_samples: int,
                                   direction: str = "up") -> dict:
        """
        Core peak detection logic for a single sweep (parallelizable).

        This method extracts the computationally expensive parts of peak detection
        that can be safely run in parallel. It does NOT modify shared state.

        Args:
            sweep_idx: Sweep index being processed
            y_proc: Pre-processed signal array
            thresh: Height threshold for labeling
            min_dist_samples: Minimum distance between peaks in samples
            direction: Peak direction ("up" or "down")

        Returns:
            Dict with detection results (to be merged into state later)
        """
        st = self.state

        # Step 1: Detect ALL peaks (no threshold filtering)
        all_peak_indices = peakdet.detect_peaks(
            y=y_proc, sr_hz=st.sr_hz,
            thresh=None,
            prominence=None,
            min_dist_samples=min_dist_samples,
            direction=direction,
            return_all=True
        )

        # Step 2: Compute breath features for ALL peaks
        all_breaths = peakdet.compute_breath_events(y_proc, all_peak_indices, sr_hz=st.sr_hz, exclude_sec=0.030)

        # Step 3: Label peaks using threshold
        all_peaks_data = peakdet.label_peaks_by_threshold(
            y=y_proc,
            peak_indices=all_peak_indices,
            thresh=thresh,
            direction=direction
        )
        all_peaks_data['labels_threshold_ro'] = all_peaks_data['labels'].copy()

        # Initialize ML prediction arrays as None (will be filled later if models loaded)
        all_peaks_data['labels_xgboost_ro'] = None
        all_peaks_data['labels_rf_ro'] = None
        all_peaks_data['labels_mlp_ro'] = None

        # Step 4: Compute p_noise if threshold model available
        try:
            import core.metrics as metrics_mod
            on = all_breaths.get('onsets', np.array([]))
            off = all_breaths.get('offsets', np.array([]))
            exm = all_breaths.get('expmins', np.array([]))
            exo = all_breaths.get('expoffs', np.array([]))
            p_noise_all = metrics_mod.compute_p_noise(st.t, y_proc, st.sr_hz, all_peak_indices, on, off, exm, exo)
            p_breath_all = 1.0 - p_noise_all if p_noise_all is not None else None
        except Exception:
            p_noise_all = None
            p_breath_all = None

        # Step 5: Compute peak candidate metrics
        peak_metrics = peakdet.compute_peak_candidate_metrics(
            y=y_proc,
            all_peak_indices=all_peak_indices,
            breath_events=all_breaths,
            sr_hz=st.sr_hz,
            p_noise=p_noise_all,
            p_breath=p_breath_all
        )

        # Step 6: Extract labeled peaks for display
        labeled_mask = all_peaks_data['labels'] == 1
        labeled_indices = all_peak_indices[labeled_mask]

        # Step 7: Compute breath events for labeled peaks
        labeled_breaths = peakdet.compute_breath_events(y_proc, labeled_indices, sr_hz=st.sr_hz, exclude_sec=0.030)

        # Step 8: Recalculate current metrics using labeled peaks as neighbors
        try:
            import core.metrics as metrics_mod
            p_noise_labeled = metrics_mod.compute_p_noise(
                st.t, y_proc, st.sr_hz, labeled_indices,
                labeled_breaths.get('onsets', np.array([])),
                labeled_breaths.get('offsets', np.array([])),
                labeled_breaths.get('expmins', np.array([])),
                labeled_breaths.get('expoffs', np.array([]))
            )
            p_breath_labeled = 1.0 - p_noise_labeled if p_noise_labeled is not None else None

            current_metrics = peakdet.compute_peak_candidate_metrics(
                y=y_proc,
                all_peak_indices=labeled_indices,
                breath_events=labeled_breaths,
                sr_hz=st.sr_hz,
                p_noise=p_noise_labeled,
                p_breath=p_breath_labeled
            )
        except Exception:
            current_metrics = peak_metrics

        return {
            'sweep_idx': sweep_idx,
            'all_peak_indices': all_peak_indices,
            'all_breaths': all_breaths,
            'all_peaks_data': all_peaks_data,
            'peak_metrics': peak_metrics,
            'current_metrics': current_metrics,
            'labeled_indices': labeled_indices,
            'labeled_breaths': labeled_breaths,
            'p_noise_all': p_noise_all,
            'p_breath_all': p_breath_all
        }

    def _apply_peak_detection(self):
        """
        Run peak detection on the ANALYZE channel for ALL sweeps,
        store indices per sweep, and redraw current sweep with peaks + breath markers.

        Supports parallel processing for large files (10+ sweeps) with automatic fallback
        to sequential processing if parallel execution fails.
        """
        import time
        t_start = time.time()

        st = self.state
        if not st.channel_names or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        self._log_status_message("Detecting peaks and breath features...")

        # Get prominence from stored value (set during auto-detect or in Analysis Options dialog)
        prom = getattr(self, 'peak_prominence', None)
        if prom is None or prom <= 0:
            self._show_warning("Invalid Prominence",
                             "No prominence threshold set. Please select an analysis channel first to auto-detect the threshold.")
            return

        # Use stored height threshold (set during auto-detect, same as prominence)
        thresh = getattr(self, 'peak_height_threshold', None)
        min_d = self.peak_min_dist
        direction = "up"  # Always detect peaks above threshold for breathing signals

        min_dist_samples = None
        if min_d is not None and min_d > 0:
            min_dist_samples = max(1, int(round(min_d * st.sr_hz)))

        # Detect on ALL sweeps for the analyze channel
        any_chan = next(iter(st.sweeps.values()))
        n_sweeps = any_chan.shape[1]
        st.peaks_by_sweep.clear()
        st.breath_by_sweep.clear()
        st.all_peaks_by_sweep.clear()  # ML training data: ALL peaks with labels
        st.all_breaths_by_sweep.clear()  # ML training data: breath events for ALL peaks
        st.peak_metrics_by_sweep.clear()  # ML training data: comprehensive peak metrics
        # st.sigh_by_sweep.clear()

        # === PARALLEL CORE DETECTION ===
        # For files with many sweeps, parallelize the core detection
        # ML predictions and state updates happen sequentially afterwards
        use_parallel = getattr(self, 'use_parallel_detection', True) and n_sweeps >= 10
        core_results = {}

        if use_parallel:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import os

                max_workers = min(os.cpu_count() or 4, 8)
                print(f"[peak-detection] Using parallel detection with {max_workers} workers for {n_sweeps} sweeps...")

                # Pre-compute all processed signals (uses cache, must be done in main thread)
                processed_signals = {}
                for s in range(n_sweeps):
                    processed_signals[s] = self._get_processed_for(st.analyze_chan, s)

                # Worker function for parallel execution
                def detect_sweep(sweep_idx):
                    y_proc = processed_signals[sweep_idx]
                    return self._detect_single_sweep_core(
                        sweep_idx, y_proc, thresh, min_dist_samples, direction
                    )

                # Execute in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(detect_sweep, s): s for s in range(n_sweeps)}
                    completed = 0
                    for future in as_completed(futures):
                        sweep_idx = futures[future]
                        result = future.result()
                        core_results[sweep_idx] = result
                        completed += 1
                        if completed % 10 == 0 or completed == n_sweeps:
                            print(f"[peak-detection] Completed {completed}/{n_sweeps} sweeps...")

                print(f"[peak-detection] Parallel core detection complete!")

            except Exception as e:
                print(f"[peak-detection] Parallel detection failed: {e}")
                print(f"[peak-detection] Falling back to sequential detection...")
                import traceback
                traceback.print_exc()
                core_results = {}  # Clear partial results, fall through to sequential

        # === SEQUENTIAL PROCESSING (fallback or for small files) ===
        for s in range(n_sweeps):
            # Use parallel results if available, otherwise compute sequentially
            if s in core_results:
                result = core_results[s]
                all_peak_indices = result['all_peak_indices']
                all_breaths = result['all_breaths']
                all_peaks_data = result['all_peaks_data']
                peak_metrics = result['peak_metrics']
                labeled_indices = result['labeled_indices']
                labeled_breaths = result['labeled_breaths']
                y_proc = self._get_processed_for(st.analyze_chan, s)

                # Store results in state
                st.all_breaths_by_sweep[s] = all_breaths
                st.all_peaks_by_sweep[s] = all_peaks_data
                st.peak_metrics_by_sweep[s] = peak_metrics
                st.current_peak_metrics_by_sweep[s] = result['current_metrics']

            else:
                # Sequential detection for this sweep (fallback or parallel not used)
                y_proc = self._get_processed_for(st.analyze_chan, s)

                # Step 1: Detect ALL peaks (no threshold filtering)
                # Note: User found that thresh=0 + min_distance works best (no prominence)
                all_peak_indices = peakdet.detect_peaks(
                    y=y_proc, sr_hz=st.sr_hz,
                    thresh=None,  # Don't filter by threshold yet
                    prominence=None,  # Don't use prominence for initial detection
                    min_dist_samples=min_dist_samples,
                    direction=direction,
                    return_all=True  # Return ALL detected peaks
                )

                # Step 2: Compute breath features for ALL peaks (including noise)
                # This is needed for ML training - noise peaks need features too
                all_breaths = peakdet.compute_breath_events(y_proc, all_peak_indices, sr_hz=st.sr_hz, exclude_sec=0.030)
                st.all_breaths_by_sweep[s] = all_breaths  # Store for ML metric computation

                # Step 3: Label peaks using auto-detected threshold
                all_peaks_data = peakdet.label_peaks_by_threshold(
                    y=y_proc,
                    peak_indices=all_peak_indices,
                    thresh=thresh,
                    direction=direction
                )
                # Store threshold predictions as read-only (for classifier switching)
                all_peaks_data['labels_threshold_ro'] = all_peaks_data['labels'].copy()

                # Step 3.1: Initialize ML prediction arrays as None (will be filled after metrics)
                if st.loaded_ml_models:
                    # We need peak_metrics first, so we'll come back to this after metrics computation
                    # For now, initialize ML label arrays as None
                    all_peaks_data['labels_xgboost_ro'] = None
                    all_peaks_data['labels_rf_ro'] = None
                    all_peaks_data['labels_mlp_ro'] = None

                # NOTE: 'labels' remains as the user-editable array (backward compatible)
                # It will be initialized from the active classifier after ML predictions run

                st.all_peaks_by_sweep[s] = all_peaks_data

                # Step 3.5: Compute comprehensive metrics for ML (merge detection, noise classification)
                # Get P(noise) and P(breath) if available from metrics module
                try:
                    import core.metrics as metrics_mod
                    # Extract breath events for p_noise computation
                    on = all_breaths.get('onsets', np.array([]))
                    off = all_breaths.get('offsets', np.array([]))
                    exm = all_breaths.get('expmins', np.array([]))
                    exo = all_breaths.get('expoffs', np.array([]))

                    # Check if threshold model is available
                    if s == 0:
                        print(f"[peak-detection] Threshold model params: {metrics_mod._threshold_model_params}")

                    p_noise_all = metrics_mod.compute_p_noise(st.t, y_proc, st.sr_hz, all_peak_indices, on, off, exm, exo)
                    p_breath_all = 1.0 - p_noise_all if p_noise_all is not None else None
                    if s == 0:
                        print(f"[peak-detection] Computed p_noise for {len(all_peak_indices)} peaks, sample values: {p_noise_all[:5] if p_noise_all is not None and len(p_noise_all) > 0 else 'None'}")
                except Exception as e:
                    print(f"[peak-detection] Warning: Could not compute P(noise): {e}")
                    import traceback
                    traceback.print_exc()
                    p_noise_all = None
                    p_breath_all = None

                peak_metrics = peakdet.compute_peak_candidate_metrics(
                    y=y_proc,
                    all_peak_indices=all_peak_indices,
                    breath_events=all_breaths,
                    sr_hz=st.sr_hz,
                    p_noise=p_noise_all,
                    p_breath=p_breath_all
                )
                st.peak_metrics_by_sweep[s] = peak_metrics  # Original metrics (never modified, for ML)
                st.current_peak_metrics_by_sweep[s] = peak_metrics  # Current metrics (updated after edits, for Y2 plotting)

            # === From here on, use results from either parallel or sequential detection ===
            # Get the data for this sweep (either from parallel results or just computed)
            all_peaks_data = st.all_peaks_by_sweep[s]
            all_breaths = st.all_breaths_by_sweep[s]
            peak_metrics = st.peak_metrics_by_sweep[s]
            all_peak_indices = all_peaks_data['indices']
            y_proc = self._get_processed_for(st.analyze_chan, s)

            # Step 3.7: Run ML predictions if models are loaded (now that we have peak_metrics)
            if st.loaded_ml_models and peak_metrics:
                import core.ml_prediction as ml_prediction

                # Determine which algorithms to run based on active classifiers
                # This avoids running all 9 models (3 algorithms × 3 model types) every time
                algorithms_to_run = set()

                # Add algorithm for breath detection (Model 1)
                if st.active_classifier in ['xgboost', 'rf', 'mlp']:
                    algorithms_to_run.add(st.active_classifier)

                # Add algorithm for eupnea/sniff (Model 3)
                if st.active_eupnea_sniff_classifier in ['xgboost', 'rf', 'mlp']:
                    algorithms_to_run.add(st.active_eupnea_sniff_classifier)

                # Add algorithm for sigh (Model 2)
                if st.active_sigh_classifier in ['xgboost', 'rf', 'mlp']:
                    algorithms_to_run.add(st.active_sigh_classifier)

                # If user wants to pre-compute all for instant switching, run all
                # TODO: Add checkbox in UI to toggle this behavior
                run_all_algorithms = False  # Set to True to restore old behavior

                if run_all_algorithms:
                    algorithms_to_run = {'xgboost', 'rf', 'mlp'}

                if s == 0:
                    print(f"[ML Prediction] Running algorithms: {sorted(algorithms_to_run)}")

                # Run predictions for selected algorithms only
                for algorithm in algorithms_to_run:
                    try:
                        # Enable debug output on first sweep to diagnose issues
                        predictions = ml_prediction.predict_with_cascade(
                            peak_metrics=peak_metrics,
                            models=st.loaded_ml_models,
                            algorithm=algorithm,
                            debug=(s == 0)  # Debug on first sweep only
                        )
                        # Store predictions as read-only (for classifier switching)
                        all_peaks_data[f'labels_{algorithm}_ro'] = predictions['final_labels']

                        # Store eupnea/sniff predictions (Model 3 output)
                        if 'eupnea_sniff_class' in predictions:
                            all_peaks_data[f'eupnea_sniff_{algorithm}_ro'] = predictions['eupnea_sniff_class']

                            if s == 0:
                                n_eupnea = np.sum(predictions['eupnea_sniff_class'] == 0)
                                n_sniff = np.sum(predictions['eupnea_sniff_class'] == 1)
                                n_unclass = np.sum(predictions['eupnea_sniff_class'] == -1)
                                print(f"[ML-{algorithm}] Eupnea: {n_eupnea}, Sniffing: {n_sniff}, Unclassified: {n_unclass}")
                        else:
                            all_peaks_data[f'eupnea_sniff_{algorithm}_ro'] = None

                        # Store sigh predictions (Model 2 output)
                        if 'sigh_class' in predictions:
                            all_peaks_data[f'sigh_{algorithm}_ro'] = predictions['sigh_class']

                            if s == 0:
                                n_normal = np.sum(predictions['sigh_class'] == 0)
                                n_sigh = np.sum(predictions['sigh_class'] == 1)
                                n_unclass_sigh = np.sum(predictions['sigh_class'] == -1)
                                print(f"[ML-{algorithm}] Normal: {n_normal}, Sigh: {n_sigh}, Unclassified: {n_unclass_sigh}")
                        else:
                            all_peaks_data[f'sigh_{algorithm}_ro'] = None

                        # Debug: Print prediction summary on first sweep
                        if s == 0:
                            n_breaths = np.sum(predictions['final_labels'] == 1)
                            n_noise = np.sum(predictions['final_labels'] == 0)
                            print(f"[ML-{algorithm}] Sweep {s}: {n_breaths} breaths, {n_noise} noise (total {len(predictions['final_labels'])} peaks)")
                    except KeyError:
                        # Models for this algorithm not loaded
                        if s == 0:
                            print(f"[ML-{algorithm}] Models not found, skipping")
                        all_peaks_data[f'labels_{algorithm}_ro'] = None
                    except Exception as e:
                        print(f"[ML-{algorithm}] Warning: Prediction failed: {e}")
                        all_peaks_data[f'labels_{algorithm}_ro'] = None

            # Debug: Show sample metrics for first sweep
            if s == 0 and len(peak_metrics) > 0:
                print(f"[peak-metrics] Computed {len(peak_metrics)} peak metrics for sweep {s}")
                # Show first potential merge candidate (small normalized gap + shallow trough)
                merge_candidates = [m for m in peak_metrics
                                   if m.get('gap_to_next_norm', 1.0) is not None
                                   and m.get('gap_to_next_norm') < 0.3
                                   and m.get('trough_ratio_next', 1.0) is not None
                                   and m.get('trough_ratio_next') < 0.15]
                if merge_candidates:
                    mc = merge_candidates[0]
                    print(f"[peak-metrics] Example merge candidate at peak {mc['peak_idx']}:")
                    print(f"  gap_to_next_norm={mc['gap_to_next_norm']:.2f}, trough_ratio_next={mc['trough_ratio_next']:.2f}")
                    print(f"  onset_above_zero={mc['onset_above_zero']}, prom_asymmetry={mc.get('prom_asymmetry', 'N/A')}")

            # Step 4: Initialize user-editable 'labels' array from active classifier
            # This is the array that gets displayed AND edited (backward compatible!)
            active_labels_key_ro = f'labels_{st.active_classifier}_ro'

            # Debug: Show what's available
            if s == 0:
                available_labels = [k for k in all_peaks_data.keys() if k.startswith('labels_') and k.endswith('_ro')]
                print(f"[peak-detection] Available label arrays: {available_labels}")
                for lbl_key in available_labels:
                    lbl_val = all_peaks_data.get(lbl_key)
                    if lbl_val is not None:
                        n_breaths = np.sum(lbl_val == 1)
                        print(f"[peak-detection]   {lbl_key}: {n_breaths} breaths out of {len(lbl_val)} peaks")
                    else:
                        print(f"[peak-detection]   {lbl_key}: None")

            if active_labels_key_ro in all_peaks_data and all_peaks_data[active_labels_key_ro] is not None:
                # Copy from selected classifier's read-only predictions
                all_peaks_data['labels'] = all_peaks_data[active_labels_key_ro].copy()
                if s == 0:
                    n_breaths = np.sum(all_peaks_data['labels'] == 1)
                    print(f"[peak-detection] Initialized 'labels' from {active_labels_key_ro} ({n_breaths} breaths)")
            else:
                # Fallback to threshold if active classifier not available
                # (labels already contains threshold predictions from step 3)
                if s == 0:
                    n_breaths = np.sum(all_peaks_data['labels'] == 1)
                    print(f"[peak-detection] FALLBACK: Using threshold labels ({n_breaths} breaths) - {active_labels_key_ro} was None or missing")

            # Step 4b: Initialize user-editable 'breath_type_class' array from active eupnea/sniff classifier
            if st.active_eupnea_sniff_classifier == 'gmm':
                # GMM will be computed on-demand later, initialize as None for now
                all_peaks_data['breath_type_class'] = None
                if s == 0:
                    print(f"[peak-detection] Initialized 'breath_type_class' as None (GMM will run on-demand)")
            else:
                # Copy from selected ML classifier's read-only predictions
                active_eup_sniff_key_ro = f'eupnea_sniff_{st.active_eupnea_sniff_classifier}_ro'
                if active_eup_sniff_key_ro in all_peaks_data and all_peaks_data[active_eup_sniff_key_ro] is not None:
                    all_peaks_data['breath_type_class'] = all_peaks_data[active_eup_sniff_key_ro].copy()
                    if s == 0:
                        print(f"[peak-detection] Initialized 'breath_type_class' from {active_eup_sniff_key_ro}")
                else:
                    # Fallback to None
                    all_peaks_data['breath_type_class'] = None
                    if s == 0:
                        print(f"[peak-detection] 'breath_type_class' initialized as None (classifier {st.active_eupnea_sniff_classifier} not available)")

            # Step 4c: Initialize user-editable 'sigh_class' array from active sigh classifier
            # First, create sigh_manual_ro from existing sigh_by_sweep (preserves manual annotations)
            n_peaks = len(all_peaks_data['indices'])
            sigh_manual = np.zeros(n_peaks, dtype=np.int8)
            if 'labels' in all_peaks_data:
                sigh_manual[all_peaks_data['labels'] == 0] = -1
            if s in st.sigh_by_sweep:
                for sigh_idx in st.sigh_by_sweep[s]:
                    peak_mask = all_peaks_data['indices'] == sigh_idx
                    if np.any(peak_mask):
                        sigh_manual[peak_mask] = 1
            all_peaks_data['sigh_manual_ro'] = sigh_manual.copy()

            if st.active_sigh_classifier == 'manual':
                # Manual mode - use the manual annotations
                all_peaks_data['sigh_class'] = sigh_manual.copy()
                if s == 0:
                    n_sighs = np.sum(all_peaks_data['sigh_class'] == 1)
                    print(f"[peak-detection] Initialized 'sigh_class' from manual annotations ({n_sighs} sighs)")
            else:
                # Copy from selected ML classifier's read-only predictions
                active_sigh_key_ro = f'sigh_{st.active_sigh_classifier}_ro'
                if active_sigh_key_ro in all_peaks_data and all_peaks_data[active_sigh_key_ro] is not None:
                    all_peaks_data['sigh_class'] = all_peaks_data[active_sigh_key_ro].copy()
                    if s == 0:
                        n_sighs = np.sum(all_peaks_data['sigh_class'] == 1)
                        print(f"[peak-detection] Initialized 'sigh_class' from {active_sigh_key_ro} ({n_sighs} sighs)")
                else:
                    # Fallback to zeros
                    n_peaks = len(all_peaks_data['indices'])
                    all_peaks_data['sigh_class'] = np.zeros(n_peaks, dtype=np.int8)
                    if 'labels' in all_peaks_data:
                        all_peaks_data['sigh_class'][all_peaks_data['labels'] == 0] = -1
                    if s == 0:
                        print(f"[peak-detection] 'sigh_class' initialized as zeros (classifier {st.active_sigh_classifier} not available)")

            # Extract labeled peaks for display
            labeled_mask = all_peaks_data['labels'] == 1
            labeled_indices = all_peaks_data['indices'][labeled_mask]
            st.peaks_by_sweep[s] = labeled_indices

            # Recompute breath events for only labeled peaks (for display)
            # This is simpler than trying to filter the all_breaths dict
            breaths = peakdet.compute_breath_events(y_proc, labeled_indices, sr_hz=st.sr_hz, exclude_sec=0.030)
            st.breath_by_sweep[s] = breaths

            # Recalculate current_peak_metrics_by_sweep using ONLY labeled peaks as neighbors
            # This ensures neighbor features (next_peak_*, prev_peak_*, etc.) are calculated
            # based on the filtered peak list, not the original all_peaks list
            try:
                import core.metrics as metrics_mod

                # Recompute p_noise for labeled peaks only
                p_noise_labeled = metrics_mod.compute_p_noise(st.t, y_proc, st.sr_hz, labeled_indices,
                                                               breaths.get('onsets', np.array([])),
                                                               breaths.get('offsets', np.array([])),
                                                               breaths.get('expmins', np.array([])),
                                                               breaths.get('expoffs', np.array([])))
                p_breath_labeled = 1.0 - p_noise_labeled if p_noise_labeled is not None else None

                # Recalculate metrics with labeled peaks only
                current_metrics = peakdet.compute_peak_candidate_metrics(
                    y=y_proc,
                    all_peak_indices=labeled_indices,  # Use filtered peaks as neighbors
                    breath_events=breaths,
                    sr_hz=st.sr_hz,
                    p_noise=p_noise_labeled,
                    p_breath=p_breath_labeled
                )
                st.current_peak_metrics_by_sweep[s] = current_metrics

                if s == 0:
                    print(f"[peak-metrics] Recalculated {len(current_metrics)} current metrics using {len(labeled_indices)} labeled peaks as neighbors")

            except Exception as e:
                print(f"[peak-metrics] Warning: Could not recalculate current metrics: {e}")
                import traceback
                traceback.print_exc()
                # Keep original metrics as fallback
                st.current_peak_metrics_by_sweep[s] = st.peak_metrics_by_sweep[s]

            # Debug: Show peak detection stats
            n_all = len(all_peak_indices)
            n_labeled = len(labeled_indices)
            n_noise = n_all - n_labeled
            if s == 0:  # Only print for first sweep to avoid spam
                print(f"[peak-detection] Sweep {s}: {n_all} total peaks ({n_labeled} breaths, {n_noise} noise)")

        # Summary statistics for ML training data
        total_all_peaks = sum(len(data['indices']) for data in st.all_peaks_by_sweep.values())
        total_labeled_breaths = sum(len(pks) for pks in st.peaks_by_sweep.values())
        total_noise_peaks = total_all_peaks - total_labeled_breaths
        print(f"[peak-detection] ML training data: {total_all_peaks} total peaks ({total_labeled_breaths} breaths, {total_noise_peaks} noise)")

        # Compute normalization statistics for relative metrics (Group B)
        print("[peak-detection] Computing normalization statistics for relative metrics...")
        self._compute_and_store_normalization_stats()

        # If a Y2 metric is selected, recompute it now that peaks/breaths changed
        if getattr(self.state, "y2_metric_key", None):
            self._compute_y2_all_sweeps()
            self.plot_host.clear_y2()

        # ALWAYS run GMM clustering in background (so users can toggle between classifiers)
        print("[peak-detection] Running automatic GMM clustering (for classifier toggling)...")
        self._run_automatic_gmm_clustering()
        self.eupnea_sniffing_out_of_date = False

        # Build eupnea/sniff regions based on active classifier (BEFORE first redraw)
        if st.active_eupnea_sniff_classifier == 'gmm':
            # GMM already ran above, regions already built
            print("[peak-detection] Using GMM classifier for display")
        else:
            # Use ML classifier predictions (already initialized in breath_type_class array)
            print(f"[peak-detection] Building eupnea/sniff regions from {st.active_eupnea_sniff_classifier} predictions...")
            import core.gmm_clustering as gmm_clustering
            try:
                gmm_clustering.build_eupnea_sniffing_regions(st, verbose=False)
                print(f"[peak-detection] Eupnea/sniff regions built from ML predictions")
            except Exception as e:
                print(f"[peak-detection] Warning: Could not build eupnea/sniff regions: {e}")

        # Sync sigh_by_sweep from sigh_class for display (BEFORE first redraw)
        print("[peak-detection] Syncing sigh markers from sigh_class array...")
        total_sighs = 0
        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]
            if 'sigh_class' in all_peaks and all_peaks['sigh_class'] is not None:
                sigh_mask = all_peaks['sigh_class'] == 1
                st.sigh_by_sweep[s] = all_peaks['indices'][sigh_mask]  # numpy array, not set!
                n_sighs = np.sum(sigh_mask)
                total_sighs += n_sighs
                if s == 0:
                    print(f"[peak-detection] Sweep {s}: {n_sighs} sighs synced to display")

        # Verify and sync: Ensure peaks_by_sweep matches active classifier
        # This fixes an issue where XGBoost results weren't showing on first detection
        active_key = f'labels_{st.active_classifier}_ro'
        first_sweep = next(iter(st.all_peaks_by_sweep.keys()), None)
        if first_sweep is not None:
            first_all_peaks = st.all_peaks_by_sweep[first_sweep]
            if active_key in first_all_peaks and first_all_peaks[active_key] is not None:
                # Active classifier predictions are available - make sure they're being used
                current_labels = first_all_peaks.get('labels')
                active_labels = first_all_peaks[active_key]
                if current_labels is not None and not np.array_equal(current_labels, active_labels):
                    print(f"[peak-detection] SYNC FIX: 'labels' doesn't match {active_key}, updating display...")
                    self._update_displayed_peaks_from_classifier()

        # SINGLE REDRAW: Show peaks, eupnea/sniff regions, and sigh markers all at once
        print(f"[peak-detection] Redrawing plot (peaks + eupnea/sniff + {total_sighs} sighs)...")
        self.redraw_main_plot()

        # Show completion message with elapsed time
        t_elapsed = time.time() - t_start

        # Log telemetry: peak detection with results
        total_peaks = sum(len(pks) for pks in st.peaks_by_sweep.values())
        total_breaths = sum(len(b.get('onsets', [])) for b in st.breath_by_sweep.values())

        telemetry.log_peak_detection(
            method='manual_threshold' if thresh else 'auto_threshold',
            num_peaks=total_peaks,
            threshold=thresh if thresh else prom,
            prominence=prom,
            min_distance_samples=min_dist_samples if min_dist_samples else 0,
            num_sweeps=n_sweeps
        )

        telemetry.log_timing('peak_detection', t_elapsed,
                            num_peaks=total_peaks,
                            num_breaths=total_breaths,
                            num_sweeps=n_sweeps)

        # Log warning if no peaks detected
        if total_peaks == 0:
            telemetry.log_warning('No peaks detected',
                                 threshold=thresh if thresh else prom,
                                 prominence=prom)

        self._log_status_message(f"Peak detection complete ({t_elapsed:.1f}s)", 3000)

        # Refresh GMM tab if it was marked for refresh after channel change
        # (Peak Detection tab was already refreshed after auto-detect)
        if hasattr(self, '_channel_change_needs_dialog_refresh') and self._channel_change_needs_dialog_refresh:
            self._channel_change_needs_dialog_refresh = False
            if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None:
                try:
                    if self._analysis_options_dialog.isVisible():
                        tab_index = getattr(self, '_dialog_tab_to_refresh', 0)
                        if tab_index == 1:  # Eup/Sniff Classification (GMM)
                            print("[Peak Detection] Refreshing GMM tab after peak detection complete")
                            self._analysis_options_dialog._refresh_breath_classification_tab()
                        # Tab 0 (Peak Detection) was already refreshed after auto-detect, skip it here
                except RuntimeError:
                    self._analysis_options_dialog = None

        # BACKGROUND PRE-COMPUTATION: Compute remaining classifiers asynchronously
        # This happens AFTER the UI is responsive, so user doesn't notice
        self._precompute_remaining_classifiers_async()

        # Disable Apply button after successful peak detection
        # Will be re-enabled if channel, filter, or file changes
        self.ApplyPeakFindPushButton.setEnabled(False)

        # Refresh Analysis Options dialog GMM tab if open (peaks are now detected)
        if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog is not None and self._analysis_options_dialog.isVisible():
            self._analysis_options_dialog._refresh_breath_classification_tab()

    def _compute_and_store_normalization_stats(self):
        """
        Compute global normalization statistics for relative metrics.

        This computes mean and std for key metrics across ALL detected peaks
        in all sweeps, enabling normalized (z-score) versions of metrics.
        """
        from scipy.signal import peak_prominences
        from core import metrics as core_metrics

        st = self.state
        if not st.peaks_by_sweep or not st.breath_by_sweep:
            return

        # Collect raw values across all sweeps
        all_amp_insp = []
        all_amp_exp = []
        all_peak_to_trough = []
        all_prominences = []
        all_ibi = []
        all_ti = []
        all_te = []

        for s in range(len(st.peaks_by_sweep)):
            if s not in st.peaks_by_sweep or s not in st.breath_by_sweep:
                continue

            pks = st.peaks_by_sweep[s]
            breaths = st.breath_by_sweep[s]

            if len(pks) == 0:
                continue

            y_proc = self._get_processed_for(st.analyze_chan, s)
            t = st.t  # Time vector

            onsets = breaths.get('onsets', np.array([]))
            offsets = breaths.get('offsets', np.array([]))
            expmins = breaths.get('expmins', np.array([]))

            # Compute prominences for all peaks
            if len(pks) > 0:
                proms = peak_prominences(y_proc, pks)[0]
                all_prominences.extend(proms)

            # Compute amp_insp for each breath cycle
            for i in range(len(onsets) - 1):
                onset_idx = int(onsets[i])
                next_onset_idx = int(onsets[i + 1])

                # Find peak in this cycle
                pk_mask = (pks >= onset_idx) & (pks < next_onset_idx)
                if not np.any(pk_mask):
                    continue
                pk_idx = int(pks[pk_mask][0])

                # Amp insp
                amp_insp = y_proc[pk_idx] - y_proc[onset_idx]
                all_amp_insp.append(amp_insp)

                # Amp exp (if expmin exists)
                em_mask = (expmins >= onset_idx) & (expmins < next_onset_idx)
                if np.any(em_mask):
                    em_idx = int(expmins[em_mask][0])
                    amp_exp = y_proc[next_onset_idx] - y_proc[em_idx]
                    all_amp_exp.append(amp_exp)

                    # Peak to trough
                    peak_to_trough = y_proc[pk_idx] - y_proc[em_idx]
                    all_peak_to_trough.append(peak_to_trough)

                # Ti
                if i < len(offsets):
                    offset_idx = int(offsets[i])
                    if onset_idx < offset_idx <= next_onset_idx:
                        ti = t[offset_idx] - t[onset_idx]
                        all_ti.append(ti)

                        # Te
                        te = t[next_onset_idx] - t[offset_idx]
                        all_te.append(te)

            # Compute IBI (inter-breath interval)
            for i in range(len(pks) - 1):
                ibi = t[pks[i + 1]] - t[pks[i]]
                all_ibi.append(ibi)

        # Compute global statistics
        stats = {}

        if len(all_amp_insp) > 0:
            stats['amp_insp'] = {'mean': float(np.nanmean(all_amp_insp)), 'std': float(np.nanstd(all_amp_insp))}

        if len(all_amp_exp) > 0:
            stats['amp_exp'] = {'mean': float(np.nanmean(all_amp_exp)), 'std': float(np.nanstd(all_amp_exp))}

        if len(all_peak_to_trough) > 0:
            stats['peak_to_trough'] = {'mean': float(np.nanmean(all_peak_to_trough)), 'std': float(np.nanstd(all_peak_to_trough))}

        if len(all_prominences) > 0:
            stats['prominence'] = {'mean': float(np.nanmean(all_prominences)), 'std': float(np.nanstd(all_prominences))}

        if len(all_ibi) > 0:
            stats['ibi'] = {'mean': float(np.nanmean(all_ibi)), 'std': float(np.nanstd(all_ibi))}

        if len(all_ti) > 0:
            stats['ti'] = {'mean': float(np.nanmean(all_ti)), 'std': float(np.nanstd(all_ti))}

        if len(all_te) > 0:
            stats['te'] = {'mean': float(np.nanmean(all_te)), 'std': float(np.nanstd(all_te))}

        # Store statistics in metrics module
        core_metrics.set_normalization_stats(stats)

        print(f"[normalization] Computed stats for {len(stats)} metric types")
        if 'prominence' in stats:
            print(f"[normalization]   prominence: mean={stats['prominence']['mean']:.4f}, std={stats['prominence']['std']:.4f}")

    def _compute_eupnea_from_gmm(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """Compute eupnea mask from GMM results. Delegates to GMMManager."""
        return self._gmm_manager.compute_eupnea_from_gmm(sweep_idx, signal_length)

    def _compute_eupnea_from_active_classifier(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """Compute eupnea mask from active classifier. Delegates to GMMManager."""
        return self._gmm_manager.compute_eupnea_from_active_classifier(sweep_idx, signal_length)

    def _run_automatic_gmm_clustering(self):
        """Run automatic GMM clustering. Delegates to GMMManager."""
        self._gmm_manager.run_automatic_gmm_clustering()

    def _collect_gmm_breath_features(self, feature_keys):
        """Collect per-breath features for GMM clustering. Delegates to GMMManager."""
        return self._gmm_manager.collect_gmm_breath_features(feature_keys)

    def _identify_gmm_sniffing_cluster(self, feature_matrix, cluster_labels, feature_keys, silhouette):
        """Identify sniffing cluster. Delegates to GMMManager."""
        return self._gmm_manager.identify_gmm_sniffing_cluster(feature_matrix, cluster_labels, feature_keys, silhouette)

    def _apply_gmm_sniffing_regions(self, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id):
        """Apply GMM sniffing regions. Delegates to GMMManager."""
        return self._gmm_manager.apply_gmm_sniffing_regions(breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id)

    def _store_gmm_probabilities_only(self, breath_cycles, cluster_probabilities, sniffing_cluster_id):
        """Store GMM probabilities. Delegates to GMMManager."""
        self._gmm_manager.store_gmm_probabilities_only(breath_cycles, cluster_probabilities, sniffing_cluster_id)

    ##################################################
    ##y2 plotting                                   ##
    ##################################################
    def _compute_y2_all_sweeps(self):
        """Compute active y2 metric for ALL sweeps on the analyze channel."""
        st = self.state
        key = getattr(st, "y2_metric_key", None)
        if not key:
            st.y2_values_by_sweep.clear()
            return
        if key not in metrics.METRICS:
            st.y2_values_by_sweep.clear()
            return
        if st.t is None or st.analyze_chan not in st.sweeps:
            st.y2_values_by_sweep.clear()
            return

        fn = metrics.METRICS[key]
        any_ch = next(iter(st.sweeps.values()))
        n_sweeps = any_ch.shape[1]
        st.y2_values_by_sweep = {}

        # Resolve dicts once before loop (avoids repeated getattr/hasattr per sweep)
        peaks_dict = getattr(st, "peaks_by_sweep", {})
        breath_dict = getattr(st, "breath_by_sweep", {})
        gmm_probs_dict = getattr(st, 'gmm_sniff_probabilities', {})
        cur_peak_metrics_dict = getattr(st, 'current_peak_metrics_by_sweep', {})
        orig_peak_metrics_dict = getattr(st, 'peak_metrics_by_sweep', {})

        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            # pull peaks/breaths if available
            pks = peaks_dict.get(s, None)
            breaths = breath_dict.get(s, {})
            on = breaths.get("onsets", None)
            off = breaths.get("offsets", None)
            exm = breaths.get("expmins", None)
            exo = breaths.get("expoffs", None)

            # Set GMM probabilities for this sweep (if available)
            gmm_probs = gmm_probs_dict.get(s, None)
            metrics.set_gmm_probabilities(gmm_probs)

            # Set peak candidate metrics for this sweep (if available)
            # Prefer current_peak_metrics_by_sweep (updated after edits) for Y2 plotting,
            # fallback to peak_metrics_by_sweep (original auto-detected) for ML training
            peak_metrics = cur_peak_metrics_dict.get(s, None)
            if peak_metrics is None:
                peak_metrics = orig_peak_metrics_dict.get(s, None)
            metrics.set_peak_metrics(peak_metrics)

            y2 = fn(st.t, y_proc, st.sr_hz, pks, on, off, exm, exo)
            st.y2_values_by_sweep[s] = y2

        # Clear GMM probabilities after computation
        metrics.set_gmm_probabilities(None)
        # Clear peak metrics after computation
        metrics.set_peak_metrics(None)

    def on_y2_metric_changed(self, idx: int):
        key = self.y2plot_dropdown.itemData(idx)
        self.state.y2_metric_key = key  # None or e.g. "if"

        # Recompute Y2 (needs peaks/breaths for most metrics; IF falls back to peaks)
        self._compute_y2_all_sweeps()

        # Force a redraw of current sweep
        # Also reset Y2 axis so it rescales to new data
        self.plot_host.clear_y2()
        self.redraw_main_plot()

    ##################################################
    ## Display Control Handlers ##
    ##################################################

    def on_yautoscale_toggled(self, checked: bool):
        """Toggle Y-axis autoscale between percentile and full range mode."""
        self.state.use_percentile_autoscale = checked
        mode = "percentile (99th)" if checked else "full range (min/max)"
        self._log_status_message(f"Y-axis autoscale: {mode}", 2000)
        self.redraw_main_plot()

    def on_ypadding_changed(self, value: float):
        """Adjust padding for percentile autoscale mode."""
        self.state.autoscale_padding = value
        self._log_status_message(f"Y-axis padding: {value*100:.0f}%", 2000)
        # Only redraw if in percentile mode
        if self.state.use_percentile_autoscale:
            self.redraw_main_plot()

    def on_eupnea_display_toggled(self, checked: bool):
        """Toggle eupnea region display between shade and line."""
        self.state.eupnea_use_shade = checked
        mode = "shading" if checked else "line"
        self._log_status_message(f"Eupnea display: {mode}", 2000)
        self.redraw_main_plot()

    def on_sniffing_display_toggled(self, checked: bool):
        """Toggle sniffing region display between shade and line."""
        self.state.sniffing_use_shade = checked
        mode = "shading" if checked else "line"
        self._log_status_message(f"Sniffing display: {mode}", 2000)
        self.redraw_main_plot()

    def on_apnea_display_toggled(self, checked: bool):
        """Toggle apnea region display between shade and line."""
        self.state.apnea_use_shade = checked
        mode = "shading" if checked else "line"
        self._log_status_message(f"Apnea display: {mode}", 2000)
        self.redraw_main_plot()

    def on_outliers_display_toggled(self, checked: bool):
        """Toggle outliers display between shade and line."""
        self.state.outliers_use_shade = checked
        mode = "shading" if checked else "line"
        self._log_status_message(f"Outliers display: {mode}", 2000)
        self.redraw_main_plot()

    def on_dark_mode_toggled(self, checked: bool):
        """Toggle plot dark mode (dark vs light background)."""
        theme = "dark" if checked else "light"
        self.plot_host.set_plot_theme(theme)

        # Save preference
        self.settings.setValue("plot_dark_mode", checked)

        mode_text = "Dark mode" if checked else "Light mode"
        self._log_status_message(f"Plot theme: {mode_text}", 2000)

    def _setup_pyqtgraph_toggle(self):
        """Create and setup the PyQtGraph backend toggle checkbox.

        NOTE: Matplotlib backend is currently disabled because it doesn't support
        the event marker system. PyQtGraph is now the only option. The checkbox
        is hidden but the code is preserved for potential future re-enabling.
        """
        from PyQt6.QtWidgets import QCheckBox
        from PyQt6.QtGui import QFont

        font = QFont()
        font.setPointSize(8)

        # Create the PyQtGraph checkbox (hidden - matplotlib disabled for now)
        self.pyqtgraph_checkbox = QCheckBox("PyQtGraph (Fast)")
        self.pyqtgraph_checkbox.setFont(font)
        self.pyqtgraph_checkbox.setToolTip(
            "Use PyQtGraph for faster plotting (recommended).\n"
            "Required for event markers. 10-50x faster for large datasets.\n"
            "Right-click on any panel to auto-scale Y-axis."
        )

        # Force pyqtgraph as the only backend (matplotlib doesn't support event markers)
        self.pyqtgraph_checkbox.setChecked(True)
        self.settings.setValue("use_pyqtgraph", True)

        # Update state
        self.state.plotting_backend = 'pyqtgraph'

        # Connect handler (kept for future use)
        self.pyqtgraph_checkbox.toggled.connect(self.on_pyqtgraph_toggled)

        # Hidden from users — pyqtgraph is the only supported backend
        self.pyqtgraph_checkbox.setVisible(False)

    def _create_plot_host(self, backend: str):
        """Create the appropriate plot host widget based on backend selection.

        Args:
            backend: 'matplotlib' or 'pyqtgraph'

        Returns:
            PlotHost or PyQtGraphPlotHost widget
        """
        if backend == 'pyqtgraph' and PYQTGRAPH_AVAILABLE:
            print(f"[PlotHost] Creating PyQtGraph backend (high-performance)")
            return PyQtGraphPlotHost(self.MainPlot)
        else:
            print(f"[PlotHost] Creating Matplotlib backend (full features)")
            return PlotHost(self.MainPlot)

    def _switch_plot_backend(self, new_backend: str):
        """Hot-swap the plotting backend.

        Args:
            new_backend: 'matplotlib' or 'pyqtgraph'
        """
        if new_backend == self._current_backend:
            return  # No change needed

        # Check availability
        if new_backend == 'pyqtgraph' and not PYQTGRAPH_AVAILABLE:
            print("[PlotHost] PyQtGraph not available, staying with matplotlib")
            return

        print(f"[PlotHost] Switching backend from {self._current_backend} to {new_backend}")

        # Get layout
        layout = self.MainPlot.layout()
        if layout is None:
            return

        # Properly clean up old plot_host to avoid dangling references
        old_plot_host = self.plot_host

        # Save current view ranges BEFORE destroying old plot_host
        saved_xlim = None
        saved_ylim = None
        try:
            if hasattr(old_plot_host, '_last_single'):
                saved_xlim = old_plot_host._last_single.get("xlim")
                saved_ylim = old_plot_host._last_single.get("ylim")
            # Also try to get current view directly from axes
            if saved_xlim is None and hasattr(old_plot_host, 'ax_main') and old_plot_host.ax_main is not None:
                if self._current_backend == 'matplotlib':
                    saved_xlim = old_plot_host.ax_main.get_xlim()
                    saved_ylim = old_plot_host.ax_main.get_ylim()
                elif self._current_backend == 'pyqtgraph':
                    view_range = old_plot_host.ax_main.viewRange()
                    saved_xlim = tuple(view_range[0])
                    saved_ylim = tuple(view_range[1])
            print(f"[PlotHost] Saving view ranges: X={saved_xlim}, Y={saved_ylim}")
        except Exception as e:
            print(f"[PlotHost] Could not save view ranges: {e}")

        # For matplotlib, disconnect toolbar before deletion to avoid QAction errors
        if self._current_backend == 'matplotlib':
            try:
                # Disconnect toolbar from canvas first
                if hasattr(old_plot_host, 'toolbar') and old_plot_host.toolbar:
                    old_plot_host.toolbar.setParent(None)
                    old_plot_host.toolbar.deleteLater()
                    old_plot_host.toolbar = None
                # Close the figure to release matplotlib resources
                if hasattr(old_plot_host, 'fig') and old_plot_host.fig:
                    import matplotlib.pyplot as plt
                    plt.close(old_plot_host.fig)
            except Exception as e:
                print(f"[PlotHost] Cleanup warning: {e}")

        # Remove from layout and hide
        layout.removeWidget(old_plot_host)
        old_plot_host.hide()
        old_plot_host.setParent(None)

        # Create new plot_host
        self._current_backend = new_backend
        self.state.plotting_backend = new_backend
        self.plot_host = self._create_plot_host(new_backend)
        layout.addWidget(self.plot_host)

        # Schedule old widget for deletion after event loop processes
        old_plot_host.deleteLater()

        # Re-initialize PlotManager (it reads plot_host from self)
        if hasattr(self, 'plot_manager'):
            from plotting.plot_manager import PlotManager
            self.plot_manager = PlotManager(self)

        # Handle event marker integration - only works with pyqtgraph
        if hasattr(self, '_event_marker_integration'):
            if new_backend == 'matplotlib':
                # Disable event markers for matplotlib (not supported)
                self._event_marker_integration.disable()
                print("[PlotHost] Event markers disabled (matplotlib backend)")
            else:
                # Re-create integration for new pyqtgraph plot_host
                self._event_marker_integration.disable()
                self._event_marker_integration = EventMarkerPlotIntegration(
                    viewmodel=self._event_marker_viewmodel,
                    plot_host=self.plot_host,
                    parent=self,
                )
                self._event_marker_integration.set_sweep_callback(lambda: self.state.sweep_idx)
                self._event_marker_integration.set_visible_range_callback(self._get_visible_x_range)
                self._event_marker_integration.set_channel_names_callback(self._get_channel_names_for_markers)
                self._event_marker_integration.set_signal_data_callback(self._get_signal_data_for_markers)
                self._event_marker_integration.enable()
                print("[PlotHost] Event markers re-enabled for new pyqtgraph backend")

        # Apply current theme
        dark_mode = self.settings.value("plot_dark_mode", True, type=bool)
        theme = "dark" if dark_mode else "light"
        self.plot_host.set_plot_theme(theme)

        # Transfer saved view ranges to new plot_host (instead of clearing)
        if saved_xlim is not None or saved_ylim is not None:
            if hasattr(self.plot_host, '_last_single'):
                self.plot_host._last_single["xlim"] = saved_xlim
                self.plot_host._last_single["ylim"] = saved_ylim
                # Enable view preservation so the saved ranges are used
                self.plot_host._preserve_x = True
                self.plot_host._preserve_y = True
                print(f"[PlotHost] Transferred view ranges to new backend")
        else:
            # No saved view - clear for fresh start
            self.plot_host.clear_saved_view("single")
            self.plot_host.clear_saved_view("grid")

        # Redraw if we have data
        if self.state.t is not None:
            self.redraw_main_plot()
            # After redraw, apply the saved view ranges directly if they weren't applied
            if saved_xlim is not None and hasattr(self.plot_host, 'ax_main') and self.plot_host.ax_main is not None:
                try:
                    if new_backend == 'pyqtgraph':
                        self.plot_host.ax_main.setXRange(saved_xlim[0], saved_xlim[1], padding=0)
                        if saved_ylim is not None:
                            self.plot_host.ax_main.setYRange(saved_ylim[0], saved_ylim[1], padding=0)
                    else:  # matplotlib
                        self.plot_host.ax_main.set_xlim(saved_xlim)
                        if saved_ylim is not None:
                            self.plot_host.ax_main.set_ylim(saved_ylim)
                        self.plot_host.canvas.draw_idle()
                    print(f"[PlotHost] Applied view ranges after redraw")
                except Exception as e:
                    print(f"[PlotHost] Could not apply view ranges: {e}")

        # Re-register editing mode callbacks on new plot_host if any mode is active
        if hasattr(self, 'editing_modes'):
            # If omit mode was active, re-enter it to register callbacks on new plot_host
            if getattr(self.editing_modes, '_omit_region_mode', False):
                print(f"[PlotHost] Re-registering omit mode callbacks for {new_backend}")
                # Exit and re-enter to re-register callbacks
                self.editing_modes._exit_omit_region_mode()
                self.editing_modes._enter_omit_region_mode(remove_mode=False)
            # Similarly for other editing modes if needed
            elif getattr(self.editing_modes, '_edit_mode', False):
                print(f"[PlotHost] Re-registering edit mode callbacks for {new_backend}")
                self.editing_modes._exit_edit_mode()
                self.editing_modes._enter_edit_mode()

    def on_pyqtgraph_toggled(self, checked: bool):
        """Toggle between matplotlib and PyQtGraph plotting backends."""
        from PyQt6.QtWidgets import QMessageBox

        backend = 'pyqtgraph' if checked else 'matplotlib'

        # Warn about matplotlib limitations
        if not checked:
            QMessageBox.information(
                self,
                "Matplotlib Backend",
                "Switching to matplotlib for figure export.\n\n"
                "Note: Event markers will not be visible in matplotlib mode.\n"
                "Switch back to PyQtGraph when done exporting."
            )

        # Check if pyqtgraph is available
        if checked and not PYQTGRAPH_AVAILABLE:
            QMessageBox.warning(
                self,
                "PyQtGraph Not Available",
                "PyQtGraph is not available. Install it with:\n\n"
                "pip install pyqtgraph\n\n"
                "Falling back to matplotlib."
            )
            self.pyqtgraph_checkbox.setChecked(False)
            return

        # Save preference
        self.settings.setValue("use_pyqtgraph", checked)

        # Hot-swap the backend
        self._switch_plot_backend(backend)

        # Log status
        backend_name = "PyQtGraph (fast)" if checked else "Matplotlib (full features)"
        self._log_status_message(f"Plotting backend: {backend_name}", 3000)

    ##################################################
    ## Turn Off All Edit Modes ##
    ##################################################
    # Note: These lines were orphaned code that was being executed as part of on_y2_metric_changed
    # They have been removed because they were clearing the callback after we restored it

    





    def on_update_eupnea_sniffing_clicked(self):
        """Handle Update Eupnea/Sniffing Detection button - rerun current classifier and apply."""
        import time
        from PyQt6.QtCore import QTimer

        st = self.state
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            self._log_status_message("No peaks detected yet - run peak detection first", 3000)
            return

        t_start = time.time()
        current_classifier = st.active_eupnea_sniff_classifier
        print(f"[update-eupnea] Manually updating eupnea/sniffing detection using: {current_classifier}")
        self._log_status_message(f"Updating eupnea/sniffing detection ({current_classifier})...")

        # Handle based on current classifier selection
        if current_classifier == 'all_eupnea':
            # Set all breaths to eupnea
            self._set_all_breaths_eupnea_sniff_class(0)
        elif current_classifier == 'none':
            # Clear all labels
            self._clear_all_eupnea_sniff_labels()
        elif current_classifier == 'gmm':
            # Run GMM clustering
            self._run_automatic_gmm_clustering()
        else:
            # ML model - re-run predictions and update
            self._update_eupnea_sniff_from_classifier()

        # Rebuild regions for display
        try:
            import core.gmm_clustering as gmm_clustering
            gmm_clustering.build_eupnea_sniffing_regions(self.state, verbose=False)
        except Exception as e:
            print(f"[update-eupnea] Warning: Could not rebuild regions: {e}")

        # Clear out-of-date flag
        self.eupnea_sniffing_out_of_date = False

        # LIGHTWEIGHT UPDATE: Just refresh eupnea/sniffing overlays without full redraw
        # This skips expensive outlier detection and metrics recomputation
        self._refresh_eupnea_overlays_only()

        # Show completion message with elapsed time
        t_elapsed = time.time() - t_start
        print(f"[update-eupnea] Eupnea/sniffing detection updated ({t_elapsed:.1f}s)")
        self._log_status_message(f"Eupnea/sniffing detection updated ({t_elapsed:.1f}s)", 2000)
        # Clear again after the success message disappears
        QTimer.singleShot(2100, lambda: self.statusBar().clearMessage())

    def _refresh_eupnea_overlays_only(self):
        """
        Lightweight update of eupnea/sniffing region overlays without full plot redraw.
        Used after GMM clustering to avoid expensive outlier detection recomputation.
        """
        st = self.state
        s = st.sweep_idx

        # Get current trace data
        t, y = self._current_trace()
        if t is None or y is None:
            return

        # Apply time normalization if stim channel exists
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t0 = 0.0
            t_plot = t

        # Get breath data
        br = st.breath_by_sweep.get(s, None)
        if not br:
            return

        # Compute ONLY eupnea mask from active classifier (fast, no metrics recomputation)
        eupnea_mask = self._compute_eupnea_from_active_classifier(s, len(y))

        # Get existing apnea threshold
        apnea_thresh = self._parse_float(self.ApneaThresh) or 0.5

        # Compute apnea mask (also fast, no metrics)
        pks = st.peaks_by_sweep.get(s, [])
        on_idx = br.get("onsets", [])
        off_idx = br.get("offsets", [])
        ex_idx = br.get("expmins", [])
        exoff_idx = br.get("expoffs", [])

        from core import metrics
        apnea_mask = metrics.detect_apneas(
            t, y, st.sr_hz, pks, on_idx, off_idx, ex_idx, exoff_idx,
            min_apnea_duration_sec=apnea_thresh
        )

        # Get eupnea and sniff regions for overlay display
        # IMPORTANT: Apply same time normalization as the trace (shift by t0 if stimulus active)
        sniff_regions_raw = st.sniff_regions_by_sweep.get(s, [])
        eupnea_regions_raw = st.eupnea_regions_by_sweep.get(s, [])

        # Normalize region times to match t_plot (shift by t0)
        sniff_regions = [(start - t0, end - t0) for start, end in sniff_regions_raw]
        eupnea_regions = [(start - t0, end - t0) for start, end in eupnea_regions_raw]

        # Update ONLY the region overlays (skips outlier masks entirely - huge speedup!)
        self.plot_host.update_region_overlays(t_plot, eupnea_mask, apnea_mask,
                                              outlier_mask=None, failure_mask=None,
                                              sniff_regions=sniff_regions,
                                              eupnea_regions=eupnea_regions,
                                              state=st)

        # Update sniffing region backgrounds (purple) - Note: This may be redundant with update_region_overlays
        # self.editing_modes.update_sniff_artists(t_plot, s)

        # Refresh canvas
        self.plot_host.canvas.draw_idle()
        print("[update-eupnea] Lightweight overlay refresh complete (skipped outlier detection)")

    def on_help_clicked(self):
        """Open the help dialog (F1) - non-modal."""
        from dialogs.help_dialog import HelpDialog

        # If help already open, bring to front instead of creating new window
        if hasattr(self, 'help_dialog') and self.help_dialog is not None:
            self.help_dialog.raise_()
            self.help_dialog.activateWindow()
            return

        # Create non-modal dialog
        self.help_dialog = HelpDialog(self, update_info=self.update_info)
        self.help_dialog.finished.connect(self._on_help_dialog_closed)
        self.help_dialog.show()  # Non-modal - doesn't block
        telemetry.log_screen_view('Help Dialog', screen_class='info_dialog')

    def _on_help_dialog_closed(self):
        """Clear help dialog reference when closed so it can be reopened."""
        self.help_dialog = None

    def _check_for_updates_on_startup(self):
        """Check for updates in background and update UI if available."""
        from PyQt6.QtCore import QThread, pyqtSignal

        class UpdateChecker(QThread):
            """Background thread for checking updates."""
            update_checked = pyqtSignal(object)  # Emits update_info or None

            def run(self):
                """Run update check in background."""
                from core import update_checker
                update_info = update_checker.check_for_updates()
                self.update_checked.emit(update_info)

        def on_update_checked(update_info):
            """Handle update check result."""
            # Store for help dialog (even if no updates, for beta status display)
            self.update_info = update_info

            from core import update_checker

            # Check if running beta (even without network)
            is_beta = update_checker.is_prerelease_version(update_checker.VERSION_STRING)

            if update_info:
                # Log status based on version type
                if is_beta:
                    latest_stable = update_info.get('latest_stable', {})
                    print(f"[Update Check] Running beta v{update_checker.VERSION_STRING}, latest stable: v{latest_stable.get('version', 'unknown')}")

                # Check if there's an update to show
                banner_info = update_checker.get_main_window_update_message(update_info)
                if banner_info:
                    text, url = banner_info
                    self.update_notification_label.setText(
                        f'<a href="{url}" style="color: #FFD700; text-decoration: underline;">{text}</a>'
                    )
                    self.update_notification_label.setVisible(True)
                    print(f"[Update Check] {text}")
                elif is_beta:
                    # No update, but running beta - show beta banner with report link
                    self._show_beta_banner()
                else:
                    print("[Update Check] You're up to date!")
            else:
                # Network error - still show beta banner if running beta
                if is_beta:
                    self._show_beta_banner()
                print("[Update Check] Could not check for updates")

        # Create and start background thread
        self.update_thread = UpdateChecker()
        self.update_thread.update_checked.connect(on_update_checked)
        self.update_thread.start()

    def _show_beta_banner(self):
        """Show beta version banner with report bugs link."""
        from version_info import VERSION_STRING
        issues_url = "https://github.com/RyanSeanPhillips/PhysioMetrics/issues"

        self.update_notification_label.setText(
            f'<span style="color: #FFD700;">Beta v{VERSION_STRING}</span> · '
            f'<a href="{issues_url}" style="color: #FFD700; text-decoration: underline;">Report Bug / Request Feature</a>'
        )
        self.update_notification_label.setVisible(True)
        print(f"[Update Check] Running beta v{VERSION_STRING} - showing beta banner")

    def on_spectral_analysis_clicked(self):
        """Open spectral analysis dialog and optionally apply notch filter."""
        st = self.state
        if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            self._show_warning("Spectral Analysis", "Please load data and select an analyze channel first.")
            return

        # Get current sweep data
        t, y = self._current_trace()
        if t is None or y is None:
            self._show_warning("Spectral Analysis", "No data available for current sweep.")
            return

        # Get stimulation spans for current sweep if available
        s = max(0, min(st.sweep_idx, self.navigation_manager._sweep_count() - 1))
        stim_spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []

        # Open dialog
        dlg = SpectralAnalysisDialog(
            parent=self, t=t, y=y, sr_hz=st.sr_hz, stim_spans=stim_spans,
            parent_window=self, use_zscore=self.use_zscore_normalization
        )
        telemetry.log_screen_view('Spectral Analysis Dialog', screen_class='analysis_dialog')
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Get filter parameters
            lower, upper = dlg.get_filter_params()
            print(f"[spectral-dialog] Dialog accepted. Filter params: lower={lower}, upper={upper}")

            if lower is not None and upper is not None:
                # Apply notch filter
                self.notch_filter_lower = lower
                self.notch_filter_upper = upper
                print(f"[notch-filter] Set notch filter: {lower:.2f} - {upper:.2f} Hz")

                # Clear processing cache to force recomputation with new filter
                st.proc_cache.clear()
                print(f"[notch-filter] Cleared processing cache")

                # Redraw to show filtered signal
                self.redraw_main_plot()
                print(f"[notch-filter] Main plot redrawn")

            else:
                print("[notch-filter] No filter applied (lower or upper is None)")
        else:
            print("[spectral-dialog] Dialog was not accepted (user cancelled or closed)")

    def on_outlier_thresh_clicked(self):
        """
        Open the multi-tab Analysis Options dialog to the Outlier Detection tab.
        (Replaces the old single OutlierMetricsDialog)
        """
        self._open_analysis_options(tab='outliers')
        telemetry.log_screen_view('Outlier Detection Options Dialog', screen_class='config_dialog')

    def on_gmm_clustering_clicked(self):
        """
        Open the multi-tab Analysis Options dialog to the Eup/Sniff Classification tab.
        (Replaces the old single GMMClusteringDialog)
        """
        self._open_analysis_options(tab='gmm')
        telemetry.log_screen_view('GMM Clustering Dialog', screen_class='analysis_dialog')

    def on_peak_navigator_clicked(self):
        """Open Advanced Peak Editor dialog for edge case review and curation."""
        # Check if dialog is already open
        if hasattr(self, '_advanced_peak_editor_dlg') and self._advanced_peak_editor_dlg is not None:
            # Bring existing dialog to front
            self._advanced_peak_editor_dlg.raise_()
            self._advanced_peak_editor_dlg.activateWindow()
            return

        # Create and show non-blocking dialog
        dlg = AdvancedPeakEditorDialog(main_window=self, parent=self)
        self._advanced_peak_editor_dlg = dlg  # Store reference to prevent garbage collection

        telemetry.log_screen_view('Advanced Peak Editor Dialog', screen_class='curation_dialog')
        telemetry.log_feature_used('advanced_peak_editor')

        # Connect cleanup and refresh on close
        def on_dialog_closed():
            self._advanced_peak_editor_dlg = None
            # Refresh plot after dialog closes (in case user made edits)
            if hasattr(self.state, 'all_peaks_by_sweep') and self.state.all_peaks_by_sweep:
                self.redraw_main_plot()

        dlg.finished.connect(on_dialog_closed)
        dlg.show()  # Non-blocking - can interact with main window

    def _refresh_omit_button_label(self):
        """Update Omit button text based on omit mode state.

        Simple states:
        - "Omit" when not in omit mode
        - "Omit (ON)" when in omit mode
        """
        if getattr(self.editing_modes, "_omit_region_mode", False):
            self.OmitSweepButton.setText("Omit (ON)")
            self.OmitSweepButton.setToolTip("Click to exit omit mode. Ctrl+Shift+click to toggle full sweep.")
        else:
            self.OmitSweepButton.setText("Omit")
            self.OmitSweepButton.setToolTip("Click to enter omit mode for marking regions to exclude.")

    def on_omit_sweep_clicked(self):
        """Handle Omit Sweep button - simple toggle to enter/exit omit region mode."""
        st = self.state
        if self.navigation_manager._sweep_count() == 0:
            return

        # Simple toggle: button checked state determines mode
        if self.OmitSweepButton.isChecked():
            # Entering omit region mode
            self.editing_modes._enter_omit_region_mode(remove_mode=False)
        else:
            # Exiting omit region mode
            self.editing_modes._exit_omit_region_mode()

    def _dim_axes_for_omitted(self, ax, label=True):
        """Delegate to PlotManager."""
        self.plot_manager.dim_axes_for_omitted(ax, label)


    ##################################################
    ##Save Data to File                             ##
    ##################################################
    def _load_save_dialog_history(self) -> dict:
        """Load autocomplete history for the Save Data dialog from QSettings."""
        return self.export_manager._load_save_dialog_history()

    def _update_save_dialog_history(self, vals: dict):
        """Update autocomplete history with new values from the Save Data dialog."""
        return self.export_manager._update_save_dialog_history(vals)

    def _sanitize_token(self, s: str) -> str:
        """Delegate to ExportManager."""
        return self.export_manager._sanitize_token(s)

    def _suggest_stim_string(self) -> str:
        """
        Build a stim name like '20Hz10s15ms' from detected stim metrics
        or '15msPulse' / '5sPulse' for single pulses.
        Rounding:
        - freq_hz -> nearest Hz
        - duration_s -> nearest second
        - pulse_width_s -> nearest millisecond (or nearest second if >1s)
        """
        return self.export_manager._suggest_stim_string()

    def on_save_analyzed_clicked(self):
        """Save analyzed data to disk after prompting for location/name."""
        return self.export_manager.on_save_analyzed_clicked()

    def on_view_summary_clicked(self):
        """Display interactive preview of the PDF summary without saving."""
        return self.export_manager.on_view_summary_clicked()

    def on_save_options_clicked(self):
        """Open dialog to configure which metrics to export."""
        from dialogs.export_options_dialog import ExportOptionsDialog

        # Get current export options from state (or None for defaults)
        current_options = getattr(self.state, 'export_metric_options', None)

        # Create and show dialog
        dialog = ExportOptionsDialog(self, current_options)

        # Connect signal to save changes
        def on_options_changed(new_options):
            self.state.export_metric_options = new_options
            self._log_status_message(f"Export options updated ({sum(new_options.values())} metrics enabled)", 2000)

        dialog.options_changed.connect(on_options_changed)
        dialog.exec()

    def _metric_keys_in_order(self):
        """Return metric keys in the UI order (from metrics.METRIC_SPECS)."""
        return self.export_manager._metric_keys_in_order()

    def _compute_metric_trace(self, key, t, y, sr_hz, peaks, breaths):
        """
        Call the metric function, passing expoffs if it exists.
        Falls back to legacy signature when needed.
        """
        return self.export_manager._compute_metric_trace(key, t, y, sr_hz, peaks, breaths)

    def _get_stim_masks(self, s: int):
        """
        Build (baseline_mask, stim_mask, post_mask) boolean arrays over st.t for sweep s.
        Uses union of all stim spans for 'stim'.
        """
        return self.export_manager._get_stim_masks(s)

    def _nanmean_sem(self, X, axis=0):
        """
        Robust mean/SEM that avoids NumPy RuntimeWarnings when there are
        0 or 1 finite values along the chosen axis.
        """
        return self.export_manager._nanmean_sem(X, axis)

    def _export_all_analyzed_data(self, preview_only=False, progress_dialog=None):
        """
        Exports (or previews) analyzed data.

        If preview_only=True: Shows interactive PDF preview dialog without saving files.
        If preview_only=False: Prompts for location/name and exports files.

        Exports:
        1) <base>_bundle.npz
            - Downsampled processed trace (kept sweeps only)
            - Downsampled y2 metric traces (all keys)
            - Peaks/breaths/sighs per kept sweep
            - Stim spans per kept sweep
            - Meta

        2) <base>_means_by_time.csv
            - t (relative to global stim start if present)
            - For each metric: optional per-sweep traces, mean, sem
            - Then the same block normalized by per-sweep baseline window (_norm)
            - Then the same block normalized by pooled eupneic baseline (_norm_eupnea)

        3) <base>_breaths.csv
            - Wide layout:
                RAW blocks:  ALL | BASELINE | STIM | POST
                NORM blocks: ALL | BASELINE | STIM | POST (per-sweep time-based)
                NORM_EUPNEA blocks: ALL | BASELINE | STIM | POST (pooled eupneic)
            - Includes `is_sigh` column (1 if any sigh peak in that breath interval)

        4) <base>_events.csv
            - Event intervals: stimulus on/off, apnea episodes, eupnea regions
            - Columns: sweep, event_type, start_time, end_time, duration
            - Times are relative to global stim start if present

        5) <base>_summary.pdf (or preview dialog if preview_only=True)
        """
        return self.export_manager._export_all_analyzed_data(preview_only, progress_dialog)

    def _mean_sem_1d(self, arr: np.ndarray):
        """Finite-only mean and SEM (ddof=1) for a 1D array. Returns (mean, sem).
        If no finite values -> (nan, nan). If only 1 finite value -> (mean, nan)."""
        return self.export_manager._mean_sem_1d(arr)

    def _save_metrics_summary_pdf(self, pdf_path, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, meta, stim_zero, stim_dur):
        """Delegate to ExportManager."""
        return self.export_manager._save_metrics_summary_pdf(pdf_path, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, meta, stim_zero, stim_dur)

    def _show_summary_preview_dialog(self, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, stim_zero, stim_dur):
        """Display interactive preview dialog with the three summary figures."""
        return self.export_manager._show_summary_preview_dialog(t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, stim_zero, stim_dur)

    def _sigh_sample_indices(self, s: int, pks: np.ndarray | None) -> set[int]:
        """
        Return a set of SAMPLE indices (into st.t / y) for sigh-marked peaks on sweep s,
        regardless of how they were originally stored.

        Accepts any of these storage patterns per sweep:
        • sample indices (ints 0..N-1)
        • indices INTO the peaks list (ints 0..len(pks)-1), which we map via pks[idx]
        • times in seconds (floats), which we map to nearest sample via searchsorted
        • numpy array / list / set in any of the above forms
        """
        return self.export_manager._sigh_sample_indices(s, pks)

    def on_curation_choose_dir_clicked(self):
        # Get last used directory from settings, default to home
        last_dir = self.settings.value("curation_last_dir", str(Path.home()))
        
        # Ensure the path exists, otherwise fall back to home
        if not Path(last_dir).exists():
            last_dir = str(Path.home())
        
        base = QFileDialog.getExistingDirectory(
            self, "Choose a folder to scan", last_dir
        )
        
        if not base:
            return
        
        # Save the selected directory for next time
        self.settings.setValue("curation_last_dir", base)
        
        groups = self._scan_csv_groups(Path(base))
        self._populate_file_list_from_groups(groups)

    def _scan_csv_groups(self, base_dir: Path):
        """
        Walk base_dir recursively and group CSVs by common root:
        root + '_breaths.csv'
        root + '_timeseries.csv'
        root + '_events.csv'
        Returns a list of dicts: {key, root, dir, breaths, means, events}
        """
        groups = {}
        for dirpath, _, filenames in os.walk(str(base_dir)):
            for fn in filenames:
                lower = fn.lower()
                if not lower.endswith(".csv"):
                    continue

                kind = None
                if lower.endswith("_breaths.csv"):
                    root = fn[:-len("_breaths.csv")]
                    kind = "breaths"
                elif lower.endswith("_timeseries.csv"):
                    root = fn[:-len("_timeseries.csv")]
                    kind = "means"
                elif lower.endswith("_means_by_time.csv"):  # Legacy support
                    root = fn[:-len("_means_by_time.csv")]
                    kind = "means"
                elif lower.endswith("_events.csv"):
                    root = fn[:-len("_events.csv")]
                    kind = "events"

                if kind is None:
                    continue

                dir_p = Path(dirpath)
                key = str((dir_p / root).resolve()).lower()  # unique per dir+root (case-insensitive on Win)
                entry = groups.get(key)
                if entry is None:
                    entry = {"key": key, "root": root, "dir": dir_p, "breaths": None, "means": None, "events": None}
                    groups[key] = entry
                entry[kind] = str(dir_p / fn)

        # Return as a stable, sorted list
        return sorted(groups.values(), key=lambda e: (str(e["dir"]).lower(), e["root"].lower()))


    def _populate_file_list_from_groups(self, groups: list[dict]):
        """
        Fill left list (FileList) with one item per root. Display only name,
        store both full paths in UserRole for later consolidation.
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QListWidgetItem

        self.FileList.clear()
        # Do not clear the right list automatically so users don't lose selections:
        # self.FilestoConsolidateList.clear()

        for g in groups:
            root = g["root"]
            has_b = bool(g["breaths"])
            has_m = bool(g["means"])
            has_e = bool(g["events"])

            # Build suffix showing what files are present
            parts = []
            if has_b:
                parts.append("breaths")
            if has_m:
                parts.append("timeseries")
            if has_e:
                parts.append("events")

            if parts:
                suffix = f"[{' + '.join(parts)}]"
            else:
                # Shouldn't happen; skip if nothing is present
                continue

            item = QListWidgetItem(f"{root}  {suffix}")
            tt_lines = [f"Root: {root}", f"Dir:  {g['dir']}"]
            if g["breaths"]:
                tt_lines.append(f"breaths:    {g['breaths']}")
            if g["means"]:
                tt_lines.append(f"timeseries: {g['means']}")
            if g["events"]:
                tt_lines.append(f"events:     {g['events']}")
            item.setToolTip("\n".join(tt_lines))

            # Store full metadata for later use
            item.setData(Qt.ItemDataRole.UserRole, g)  # {'key', 'root', 'dir', 'breaths', 'means', 'events'}

            self.FileList.addItem(item)

        # Optional: sort visually
        self.FileList.sortItems()


    def _curation_scan_and_fill(self, root: Path):
        """Scan for matching CSVs and fill FileList with filenames (store full paths in item data)."""
        from PyQt6.QtWidgets import QListWidgetItem
        from PyQt6.QtCore import Qt

        # Clear existing items
        self.FileList.clear()

        # Patterns to include (recursive)
        patterns = ["*_breaths.csv", "*_timeseries.csv", "*_means_by_time.csv", "*_events.csv"]

        files = []
        try:
            for pat in patterns:
                files.extend(root.rglob(pat))
        except Exception as e:
            self._show_error("Scan error", f"Failed to scan folder:\n{root}\n\n{e}")
            return

        # Deduplicate & sort (by name, then path for stability)
        uniq = {}
        for p in files:
            try:
                # Only include files (ignore dirs, weird links)
                if p.is_file():
                    # keep all—even if names clash—because display is name-only,
                    # but we keep full path in item data and tooltip
                    uniq[str(p)] = p
            except Exception:
                pass

        files_sorted = sorted(uniq.values(), key=lambda x: (x.name.lower(), str(x).lower()))

        if not files_sorted:
            try:
                self._log_status_message("No matching CSV files found in the selected folder.", 4000)
            except Exception:
                pass
            return

        for p in files_sorted:
            item = QListWidgetItem(p.name)
            item.setToolTip(str(p))  # show full path on hover
            item.setData(Qt.ItemDataRole.UserRole, str(p))  # keep full path for later use
            self.FileList.addItem(item)

        # Optional: sort in the widget (already sorted, but harmless)
        self.FileList.sortItems()

    def _list_has_path(self, lw, full_path: str) -> bool:
        """Return True if any item in lw has UserRole == full_path."""
        for i in range(lw.count()):
            it = lw.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == full_path:
                return True
        return False


    # ========================================
    # Consolidation Tab (Project Builder) Methods
    # ========================================

    def _populate_consolidation_source_list(self):
        """
        Populate the consolidation source list by scanning for analyzed CSV files.
        Uses the same logic as the Curation tab - looks for _breaths.csv, _timeseries.csv, _events.csv
        """
        self.consolidationSourceList.clear()

        # Determine which folder to scan
        if self._consolidation_custom_folder:
            scan_folder = self._consolidation_custom_folder
            self.consolidationSourceLabel.setText("Custom Folder")
            folder_str = str(scan_folder)
            if len(folder_str) > 30:
                folder_str = "..." + folder_str[-27:]
            self.consolidationFolderLabel.setText(folder_str)
            self.consolidationFolderLabel.setToolTip(str(scan_folder))
        elif self._project_directory:
            scan_folder = Path(self._project_directory)
            self.consolidationSourceLabel.setText("Project Files")
            self.consolidationFolderLabel.setText("")
        else:
            # No folder to scan
            self.consolidationSourceLabel.setText("No Folder")
            self.consolidationFolderLabel.setText("Use Browse or load a project")
            return

        # Scan for CSV groups (same as Curation tab)
        groups = self._scan_csv_groups(scan_folder)

        if not groups:
            self._log_status_message("No analyzed files found (looking for *_breaths.csv, *_timeseries.csv, *_events.csv)", 4000)
            return

        # Populate the list
        for g in groups:
            root = g["root"]
            has_b = bool(g["breaths"])
            has_m = bool(g["means"])
            has_e = bool(g["events"])

            # Build suffix showing what files are present
            parts = []
            if has_b:
                parts.append("breaths")
            if has_m:
                parts.append("timeseries")
            if has_e:
                parts.append("events")

            if not parts:
                continue

            suffix = f"[{' + '.join(parts)}]"
            display_name = f"{root}  {suffix}"

            # Build tooltip
            tt_lines = [f"Root: {root}", f"Dir:  {g['dir']}"]
            if g["breaths"]:
                tt_lines.append(f"breaths:    {g['breaths']}")
            if g["means"]:
                tt_lines.append(f"timeseries: {g['means']}")
            if g["events"]:
                tt_lines.append(f"events:     {g['events']}")

            item = QListWidgetItem(display_name)
            item.setToolTip("\n".join(tt_lines))

            # Store the same metadata format as Curation tab for compatibility
            item.setData(Qt.ItemDataRole.UserRole, g)  # {'key', 'root', 'dir', 'breaths', 'means', 'events'}

            self.consolidationSourceList.addItem(item)

        self._log_status_message(f"Found {len(groups)} analyzed file(s) ready for consolidation", 3000)

    def _on_consolidation_browse_clicked(self):
        """Handle browse button click - select a custom folder for consolidation."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with Analyzed Files",
            str(self._project_directory) if self._project_directory else ""
        )

        if folder:
            self._consolidation_custom_folder = Path(folder)
            self.consolidationResetButton.show()  # Show reset button when using custom folder
            self._populate_consolidation_source_list()

    def _on_consolidation_reset_clicked(self):
        """Handle reset button click - return to project files."""
        self._consolidation_custom_folder = None
        self.consolidationResetButton.hide()  # Hide reset button when back to project files
        self._populate_consolidation_source_list()

    def _on_left_column_tab_changed(self, index: int):
        """Handle tab changes in the left column (Project Builder sub-tabs).
        Refreshes the Consolidation source list when switching to that tab.
        """
        # Get the widget at this index to check if it's the consolidation tab
        widget = self.leftColumnTabs.widget(index)
        if widget and widget.objectName() == 'consolidationContainer':
            # Only refresh from project files if not using a custom folder
            if not self._consolidation_custom_folder:
                self._populate_consolidation_source_list()

    def _filter_consolidation_source_list(self, text: str):
        """Filter the consolidation source list based on search text.
        Searches the root name, directory path, and display text.
        """
        search_terms = text.lower().strip().split()

        for i in range(self.consolidationSourceList.count()):
            item = self.consolidationSourceList.item(i)
            if not item:
                continue

            # Get metadata for comprehensive search
            metadata = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(metadata, dict):
                # Build searchable text from CSV group metadata
                searchable = " ".join([
                    str(metadata.get('root', '')),
                    str(metadata.get('dir', '')),
                    item.text(),
                ]).lower()
            else:
                # Fallback to display text
                searchable = item.text().lower()

            # All search terms must match (AND logic)
            matches = all(term in searchable for term in search_terms) if search_terms else True
            item.setHidden(not matches)

    def _consolidation_move_all_right(self):
        """Move all visible items from source to consolidation list."""
        moved = 0
        skipped = 0
        for i in range(self.consolidationSourceList.count()):
            item = self.consolidationSourceList.item(i)
            if item and not item.isHidden():
                # Check if already in destination
                if not self._item_in_list(self.consolidationFilesList, item.data(Qt.ItemDataRole.UserRole)):
                    new_item = QListWidgetItem(item.text())
                    new_item.setData(Qt.ItemDataRole.UserRole, item.data(Qt.ItemDataRole.UserRole))
                    self.consolidationFilesList.addItem(new_item)
                    moved += 1
                else:
                    skipped += 1
        if moved or skipped:
            self.statusbar.showMessage(f"Added {moved} file(s) to consolidation. Skipped {skipped} duplicate(s).", 3000)

    def _consolidation_move_selected_right(self):
        """Move selected items from source to consolidation list."""
        moved = 0
        skipped = 0
        for item in self.consolidationSourceList.selectedItems():
            if not self._item_in_list(self.consolidationFilesList, item.data(Qt.ItemDataRole.UserRole)):
                new_item = QListWidgetItem(item.text())
                new_item.setData(Qt.ItemDataRole.UserRole, item.data(Qt.ItemDataRole.UserRole))
                self.consolidationFilesList.addItem(new_item)
                moved += 1
            else:
                skipped += 1
        if moved or skipped:
            self.statusbar.showMessage(f"Added {moved} file(s) to consolidation. Skipped {skipped} duplicate(s).", 3000)

    def _consolidation_move_selected_left(self):
        """Remove selected items from consolidation list."""
        items = self.consolidationFilesList.selectedItems()
        for item in items:
            row = self.consolidationFilesList.row(item)
            self.consolidationFilesList.takeItem(row)
        if items:
            self.statusbar.showMessage(f"Removed {len(items)} file(s) from consolidation.", 3000)

    def _consolidation_move_all_left(self):
        """Remove all items from consolidation list."""
        count = self.consolidationFilesList.count()
        self.consolidationFilesList.clear()
        if count:
            self.statusbar.showMessage(f"Removed all {count} file(s) from consolidation.", 3000)

    def _item_in_list(self, list_widget, user_data) -> bool:
        """Check if an item with given user data exists in the list.
        Uses the 'key' field from CSV group metadata for comparison.
        """
        # Get the key for comparison (unique identifier for CSV group)
        if isinstance(user_data, dict):
            check_key = user_data.get('key', '')
        else:
            check_key = str(user_data) if user_data else ''

        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if not item:
                continue
            item_data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(item_data, dict):
                item_key = item_data.get('key', '')
            else:
                item_key = str(item_data) if item_data else ''

            if item_key and item_key == check_key:
                return True
        return False

    def _on_consolidation_save_clicked(self):
        """Handle consolidation save from Project Builder tab.
        Uses the same data format as the Curation tab for compatibility.
        """
        from PyQt6.QtCore import Qt

        # Get all items from the consolidation list
        items = []
        for i in range(self.consolidationFilesList.count()):
            item = self.consolidationFilesList.item(i)
            if item:
                items.append(item)

        if not items:
            QMessageBox.warning(self, "No Files", "No files selected for consolidation.")
            return

        # Extract file paths in the same format as Curation tab
        means_files = []
        breaths_files = []
        events_files = []

        for item in items:
            meta = item.data(Qt.ItemDataRole.UserRole) or {}
            root = meta.get("root", item.text())
            if meta.get("means"):
                means_files.append((root, Path(meta["means"])))
            if meta.get("breaths"):
                breaths_files.append((root, Path(meta["breaths"])))
            if meta.get("events"):
                events_files.append((root, Path(meta["events"])))

        if not means_files and not breaths_files and not events_files:
            QMessageBox.warning(self, "No CSV Files", "No CSV files found in the selected items.")
            return

        # Use the consolidation manager directly with the file lists
        self.consolidation_manager.consolidate_csv_files(means_files, breaths_files, events_files)

    def _propose_consolidated_filename(self, files: list) -> tuple[str, list[str]]:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._propose_consolidated_filename(files)

    def on_consolidate_save_data_clicked(self):
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager.on_consolidate_save_data_clicked()

    def _consolidate_breaths_histograms(self, files: list[tuple[str, Path]]) -> dict:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_breaths_histograms(files)

    def _consolidate_events(self, files: list[tuple[str, Path]]) -> pd.DataFrame:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_events(files)

    def _consolidate_stimulus(self, files: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_stimulus(files)

    def _try_load_npz_v2(self, npz_path: Path) -> dict | None:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._try_load_npz_v2(npz_path)

    def _extract_timeseries_from_npz(self, npz_data: dict, metric: str, variant: str = 'raw') -> tuple[np.ndarray, np.ndarray]:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._extract_timeseries_from_npz(npz_data, metric, variant)

    def _consolidate_from_npz_v2(self, npz_data_by_root: dict, files: list[tuple[str, Path]], metrics: list[str]) -> dict:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_from_npz_v2(npz_data_by_root, files, metrics)

    def _consolidate_means_files(self, files: list[tuple[str, Path]]) -> dict:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_means_files(files)

    def _consolidate_breaths_sighs(self, files: list[tuple[str, Path]]) -> pd.DataFrame:
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._consolidate_breaths_sighs(files)

    def _save_consolidated_to_excel(self, consolidated: dict, save_path: Path):
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._save_consolidated_to_excel(consolidated, save_path)

    def _add_events_charts(self, ws, header_row):
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._add_events_charts(ws, header_row)

    def _add_sighs_chart(self, ws, header_row):
        """Delegate to ConsolidationManager."""
        return self.consolidation_manager._add_sighs_chart(ws, header_row)

    # ========== PROJECT BUILDER METHODS ==========

    def _setup_master_file_list(self):
        """Set up the master file list using Model/View architecture."""
        from PyQt6.QtWidgets import QHeaderView, QVBoxLayout, QWidget
        from PyQt6.QtCore import Qt

        # NOTE: Project Organization section (right column) was removed from .ui file
        # The old experiment-based workflow widgets are no longer present

        # === Set up QTableView with Model/View architecture ===
        # The QTableView is already in the .ui file - just set up the model and delegates
        table = self.discoveredFilesTable

        # Create and set the model
        self._file_table_model = FileTableModel(self)
        table.setModel(self._file_table_model)

        # Configure table view properties (some may be set in .ui, but ensure they're correct)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(False)  # We handle sorting ourselves
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)

        # Enable drag-drop column reordering
        header = table.horizontalHeader()
        header.setSectionsMovable(True)
        header.setDragEnabled(True)
        header.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

        # Set up delegates for special columns
        self._setup_table_delegates(table)

        # Set column widths from column definitions
        for i, col_def in enumerate(self._file_table_model.get_visible_columns()):
            table.setColumnWidth(i, col_def.width)
            if col_def.fixed:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)

        # Connect signals
        table.doubleClicked.connect(self._on_table_double_clicked)
        table.clicked.connect(self._on_table_clicked)  # Single click for Notes column
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_master_list_context_menu)

        # Connect model signals
        self._file_table_model.cell_edited.connect(self._on_model_cell_edited)

        # Connect header click for sorting
        header.sectionClicked.connect(self._on_header_sort_clicked)

        # Connect header context menu for column visibility
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._on_header_context_menu)

        # Install event filter for resize events
        table.viewport().installEventFilter(self)
        self._table_resize_timer = None

        print("[project-builder] Master file list configured with Model/View architecture")

    def _setup_table_delegates(self, table: QTableView):
        """Set up item delegates for special column types."""
        model = self._file_table_model

        # Button delegate for actions column
        actions_col = model.get_column_index('actions')
        if actions_col >= 0:
            self._button_delegate = ButtonDelegate(table)
            self._button_delegate.analyze_clicked.connect(self._on_analyze_row)
            self._button_delegate.delete_clicked.connect(self._on_delete_row)
            self._button_delegate.info_clicked.connect(self._on_info_row)
            table.setItemDelegateForColumn(actions_col, self._button_delegate)

        # Status delegate
        status_col = model.get_column_index('status')
        if status_col >= 0:
            self._status_delegate = StatusDelegate(table)
            table.setItemDelegateForColumn(status_col, self._status_delegate)

        # Autocomplete delegates for editable columns with history
        for col_def in model.get_visible_columns():
            if col_def.editable and col_def.history_key:
                col_idx = model.get_column_index(col_def.key)
                if col_idx >= 0:
                    delegate = AutoCompleteDelegate(table, history_key=col_def.history_key)
                    table.setItemDelegateForColumn(col_idx, delegate)

    def _on_table_double_clicked(self, index):
        """Handle double-click on table - open file for analysis."""
        if not index.isValid():
            return
        row = index.row()
        col_key = self._file_table_model.get_column_key(index.column())

        # If double-clicking on an editable column, let the delegate handle it
        col_def = self._file_table_model.get_column_def(col_key) if col_key else None
        if col_def and col_def.editable:
            return  # Let editing happen

        # Otherwise, analyze the file
        self._analyze_file_at_row(row)

    def _on_table_clicked(self, index):
        """Handle single click on table - check for Notes column click."""
        if not index.isValid():
            return

        col_key = self._file_table_model.get_column_key(index.column())
        if col_key == 'linked_notes':
            row = index.row()
            row_data = self._file_table_model.get_row_data(row)
            if row_data:
                file_name = row_data.get('file_name', '')
                if file_name:
                    self._show_linked_notes_for_file(file_name)

    def _show_linked_notes_for_file(self, file_name: str):
        """Show notes for file. Delegates to NotesPreviewManager."""
        self._notes_preview_manager.show_linked_notes_for_file(file_name)

    def _show_notes_preview_dialog(self, files: list, title: str, info_text: str = None, highlight_stem: str = None):
        """Show notes preview dialog. Delegates to NotesPreviewManager."""
        self._notes_preview_manager.show_notes_preview_dialog(files, title, info_text, highlight_stem)

    def _create_note_preview_widget(self, note_info: dict, abf_stem: str, search_term: str = None):
        """Create note preview widget. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.create_note_preview_widget(note_info, abf_stem, search_term)

    def _create_highlighted_table(self, df, abf_stem: str, search_term: str = None):
        """Create highlighted table. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.create_highlighted_table(df, abf_stem, search_term)

    def _create_highlighted_text(self, content: str, abf_stem: str, search_term: str = None):
        """Create highlighted text. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.create_highlighted_text(content, abf_stem, search_term)

    def _df_contains_stem(self, df, abf_stem: str) -> bool:
        """Check if DataFrame contains stem. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.df_contains_stem(df, abf_stem)

    def _get_notes_for_abf(self, abf_stem: str) -> list:
        """Get notes for ABF file. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.get_notes_for_abf(abf_stem)

    def _get_fuzzy_notes_for_abf(self, abf_stem: str, search_range: int = 5) -> tuple:
        """Get fuzzy matched notes. Delegates to NotesPreviewManager."""
        return self._notes_preview_manager.get_fuzzy_notes_for_abf(abf_stem, search_range)

    def _update_linked_notes_column(self):
        """Update linked notes column. Delegates to NotesPreviewManager."""
        self._notes_preview_manager.update_linked_notes_column()

    def _on_model_cell_edited(self, row: int, column_key: str, value):
        """Handle cell edits from the model."""
        # Update master file list
        if row < len(self._master_file_list):
            self._master_file_list[row][column_key] = value

            # Update autocomplete history for certain columns
            if column_key == 'experiment' and value:
                self._update_experiment_history(str(value))
            elif column_key == 'strain' and value:
                self._update_autocomplete_history('strain_history', str(value))

            # Autosave project
            self._project_builder.project_autosave()

    def _on_analyze_row(self, row: int):
        """Handle analyze button click."""
        self._analyze_file_at_row(row)

    def _on_delete_row(self, row: int):
        """Handle delete button click - remove row from table."""
        if row < len(self._master_file_list):
            del self._master_file_list[row]
            self._file_table_model.remove_row(row)
            self._project_builder.project_autosave()

    def _on_info_row(self, row: int):
        """Handle info button click - show file details."""
        if row < len(self._master_file_list):
            task = self._master_file_list[row]
            file_path = task.get('file_path', 'Unknown')
            protocol = task.get('protocol', 'Unknown')
            channels = task.get('channel_count', 0)
            sweeps = task.get('sweep_count', 0)

            info_text = f"""File: {file_path}
Protocol: {protocol}
Channels: {channels}
Sweeps: {sweeps}"""

            QMessageBox.information(self, "File Information", info_text)

    def _on_header_context_menu(self, pos):
        """Show context menu for column visibility and custom columns."""
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtCore import QPoint

        header = self.discoveredFilesTable.horizontalHeader()
        menu = QMenu(self)

        # Add show/hide options for each column
        all_cols = self._file_table_model.get_all_column_defs()
        for col_def in all_cols:
            if col_def.key == 'actions':  # Don't allow hiding actions column
                continue
            action = menu.addAction(col_def.header or col_def.key)
            action.setCheckable(True)
            col_idx = self._file_table_model.get_column_index(col_def.key)
            action.setChecked(col_idx >= 0)  # Visible if index >= 0
            action.setData(col_def.key)
            action.triggered.connect(lambda checked, key=col_def.key: self._toggle_column_visibility(key, checked))

        menu.addSeparator()

        # Custom columns submenu
        custom_menu = menu.addMenu("Custom Columns")
        add_custom_action = custom_menu.addAction("+ Add Custom Column...")
        add_custom_action.triggered.connect(self._add_custom_column_dialog)

        # List existing custom columns with remove option
        custom_cols = self._file_table_model.get_custom_columns()
        if custom_cols:
            custom_menu.addSeparator()
            for col_def in custom_cols:
                remove_action = custom_menu.addAction(f"Remove '{col_def.header}'")
                remove_action.triggered.connect(
                    lambda checked, key=col_def.key: self._remove_custom_column(key)
                )

        menu.addSeparator()

        # Reset to default
        reset_action = menu.addAction("Reset Column Order")
        reset_action.triggered.connect(self._reset_column_order)

        # Show menu at cursor position
        menu.exec(header.mapToGlobal(pos))

    def _toggle_column_visibility(self, key: str, visible: bool):
        """Toggle visibility of a column."""
        if visible:
            self._file_table_model.show_column(key)
        else:
            self._file_table_model.hide_column(key)

    def _add_custom_column_dialog(self):
        """Show dialog to add a custom column."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self, "Add Custom Column",
            "Enter column name (e.g., 'Notes', 'Coordinates', 'Site'):"
        )

        if ok and name.strip():
            name = name.strip()
            key = self._file_table_model.add_custom_column(name)
            if key:
                self._log_status_message(f"✓ Added custom column: {name}", 2000)
                # Refresh table to show new column
                self._rebuild_table_from_master_list()
                self._auto_fit_table_columns()
                self._project_builder.project_autosave()
            else:
                self._show_warning("Column Exists", f"A column named '{name}' already exists.")

    def _remove_custom_column(self, key: str):
        """Remove a custom column."""
        from PyQt6.QtWidgets import QMessageBox

        # Confirm removal
        result = QMessageBox.question(
            self, "Remove Custom Column",
            f"Remove this custom column?\n\nThis will delete all data in this column.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if result == QMessageBox.StandardButton.Yes:
            if self._file_table_model.remove_custom_column(key):
                # Also remove from master file list
                for task in self._master_file_list:
                    task.pop(key, None)
                self._log_status_message(f"✓ Removed custom column", 2000)
                self._rebuild_table_from_master_list()
                self._auto_fit_table_columns()
                self._project_builder.project_autosave()

    def _reset_column_order(self):
        """Reset column order to default."""
        self._file_table_model.reset_column_order()
        # Re-setup delegates after reset
        self._setup_table_delegates(self.discoveredFilesTable)

    def _analyze_file_at_row(self, row: int):
        """Load and analyze file at given row."""
        if row >= len(self._master_file_list):
            return

        task = self._master_file_list[row]
        file_path = task.get('file_path')
        if not file_path:
            return

        from pathlib import Path
        if not Path(file_path).exists():
            self._show_warning("File Not Found", f"File no longer exists:\n{file_path}")
            return

        print(f"[master-list] Opening file for analysis: {file_path}")

        # Add to recent files so it appears in Browse dropdown
        self._add_recent_file(str(file_path))
        self._add_recent_folder(str(Path(file_path).parent))
        self.settings.sync()  # Force save to disk
        print(f"[master-list] Added to recent files: {file_path}")

        # Track which row is being analyzed
        self._active_master_list_row = row

        # Store pending channel selections from Project Builder
        # These will be used after file loads to override auto-detection
        self._pending_analysis_channel = task.get('channel', '')
        self._pending_stim_channels = task.get('stim_channels', [])

        # Log what we're planning to select
        if self._pending_analysis_channel or self._pending_stim_channels:
            print(f"[master-list] Pre-selecting: analysis={self._pending_analysis_channel}, stim={self._pending_stim_channels}")

        # Load the file
        self.load_file(Path(file_path))

        # Switch to Analysis tab (tab widget is called 'Tabs' in the UI file)
        # Tab 0 = Project Builder, Tab 1 = Analysis
        if hasattr(self, 'Tabs'):
            self.Tabs.setCurrentIndex(1)  # Analysis tab is index 1
            print("[master-list] Switched to Analysis tab")

    def _update_autocomplete_history(self, history_key: str, value: str):
        """Add a value to autocomplete history."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("PhysioMetrics", "BreathAnalysis")
        history = settings.value(f"autocomplete/{history_key}", [])
        if not isinstance(history, list):
            history = []
        if value in history:
            history.remove(value)
        history.insert(0, value)
        history = history[:50]
        settings.setValue(f"autocomplete/{history_key}", history)

    # ========== COMPATIBILITY LAYER ==========
    # These methods provide backward compatibility with old QTableWidget patterns

    def _get_table_cell_value(self, row: int, column_key: str):
        """Get cell value from model by column key (replaces table.item(row, col).text())."""
        return self._file_table_model.get_cell_value(row, column_key)

    def _set_table_cell_value(self, row: int, column_key: str, value):
        """Set cell value in model by column key (replaces table.setItem(row, col, item))."""
        self._file_table_model.set_cell_value(row, column_key, value)
        # Also update master file list
        if row < len(self._master_file_list):
            self._master_file_list[row][column_key] = value

    def _get_table_row_count(self) -> int:
        """Get number of rows in table."""
        return self._file_table_model.rowCount()

    def _clear_table(self):
        """Clear all data from table."""
        self._file_table_model.clear()

    def _format_exports_summary(self, task: dict) -> str:
        """
        Format a compact summary of exports for display in the Exports column.

        Example output: "1 PDF, 3 CSV, 1 NPZ"
        """
        exports = task.get('exports', {})
        if not exports:
            return ''

        parts = []

        # Count CSVs
        csv_count = sum([
            1 if exports.get('timeseries_csv') else 0,
            1 if exports.get('breaths_csv') else 0,
            1 if exports.get('events_csv') else 0,
        ])

        if exports.get('pdf'):
            parts.append('1 PDF')
        if csv_count > 0:
            parts.append(f'{csv_count} CSV')
        if exports.get('npz'):
            parts.append('1 NPZ')
        if exports.get('ml_training'):
            parts.append('ML')
        if exports.get('session_state'):
            parts.append('Session')

        return ', '.join(parts) if parts else ''

    def _format_exports_tooltip(self, task: dict) -> str:
        """
        Format a detailed tooltip for the Exports column.

        Shows what was exported, when, where, and app version.
        """
        exports = task.get('exports', {})
        export_path = task.get('export_path', '')
        export_date = task.get('export_date', '')
        export_version = task.get('export_version', '')

        if not exports and not export_path:
            return 'No exports yet'

        lines = []

        # Header with date
        if export_date:
            # Format ISO date nicely
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(export_date)
                date_str = dt.strftime('%Y-%m-%d %H:%M')
            except:
                date_str = export_date
            lines.append(f"Exported: {date_str}")

        # App version
        if export_version:
            lines.append(f"App version: {export_version}")

        # Export path
        if export_path:
            lines.append(f"Location: {export_path}")

        lines.append('')  # Blank line

        # List of exports
        lines.append("Files saved:")
        if exports.get('pdf'):
            lines.append("  ✓ Summary PDF")
        if exports.get('timeseries_csv'):
            lines.append("  ✓ Timeseries CSV")
        if exports.get('breaths_csv'):
            lines.append("  ✓ Breaths CSV")
        if exports.get('events_csv'):
            lines.append("  ✓ Events CSV")
        if exports.get('npz'):
            lines.append("  ✓ NPZ Bundle")
        if exports.get('session_state'):
            lines.append("  ✓ Session State")
        if exports.get('ml_training'):
            lines.append("  ✓ ML Training Data")

        return '\n'.join(lines)

    def _create_exports_table_item(self, task: dict):
        """Create a QTableWidgetItem for the Exports column with summary and tooltip."""
        from PyQt6.QtWidgets import QTableWidgetItem

        summary = self._format_exports_summary(task)
        tooltip = self._format_exports_tooltip(task)

        item = QTableWidgetItem(summary)
        item.setToolTip(tooltip)

        return item

    def _on_header_sort_clicked(self, column):
        """
        Custom sorting that keeps parent rows with their sub-rows.
        Sorts groups by the parent's value in the clicked column.
        """
        if not self._master_file_list:
            return

        # Track sort direction (toggle on repeated clicks)
        if not hasattr(self, '_sort_column'):
            self._sort_column = -1
            self._sort_ascending = True

        if self._sort_column == column:
            self._sort_ascending = not self._sort_ascending
        else:
            self._sort_column = column
            self._sort_ascending = True

        # Group tasks by parent file path
        groups = {}  # file_path -> {'parent': task, 'sub_rows': [tasks]}
        parent_order = []  # Track order of parents for sorting

        for task in self._master_file_list:
            file_path = str(task.get('file_path', ''))
            is_sub_row = task.get('is_sub_row', False)

            if file_path not in groups:
                groups[file_path] = {'parent': None, 'sub_rows': []}
                parent_order.append(file_path)

            if is_sub_row:
                groups[file_path]['sub_rows'].append(task)
            else:
                groups[file_path]['parent'] = task

        # Get sort key for a group (uses parent's column value)
        def get_sort_key(file_path):
            group = groups[file_path]
            parent = group['parent']
            if not parent:
                # No parent, use first sub-row
                if group['sub_rows']:
                    parent = group['sub_rows'][0]
                else:
                    return ''

            # Get value from the appropriate column
            # Map column index to task field
            column_to_field = {
                0: 'file_name',
                1: 'protocol',
                2: 'channel_count',
                3: 'sweep_count',
                4: 'keywords_display',
                5: 'channel',
                6: 'stim_channel',
                7: 'events_channel',
                8: 'strain',
                9: 'stim_type',
                10: 'power',
                11: 'sex',
                12: 'animal_id',
                13: 'status',
                15: 'export_path',
            }
            field = column_to_field.get(column, 'file_name')
            value = parent.get(field, '')
            # Convert to string for consistent sorting
            return str(value).lower() if value else ''

        # Sort parent_order by the sort key
        parent_order.sort(key=get_sort_key, reverse=not self._sort_ascending)

        # Rebuild _master_file_list with sorted groups
        new_master_list = []
        for file_path in parent_order:
            group = groups[file_path]
            if group['parent']:
                new_master_list.append(group['parent'])
            new_master_list.extend(group['sub_rows'])

        self._master_file_list = new_master_list

        # Rebuild the table
        self._rebuild_table_from_master_list()

        # Show sort indicator
        direction = "↑" if self._sort_ascending else "↓"
        # Get column name from model's headerData
        col_name = self._file_table_model.headerData(column, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
        if not col_name:
            col_name = f"Col {column}"
        self._log_status_message(f"Sorted by {col_name} {direction}", 2000)

    # =========================================================================
    # Table Management - Delegating Wrappers
    # These methods delegate to ProjectBuilderManager for implementation
    # =========================================================================

    def _rebuild_table_from_master_list(self):
        """Rebuild table from master list. Delegates to ProjectBuilderManager."""
        self._project_builder.rebuild_table_from_master_list()

    def _get_experiment_history(self) -> list:
        """Get experiment history. Delegates to ProjectBuilderManager."""
        return self._project_builder.get_experiment_history()

    def _update_experiment_history(self, experiment_name: str):
        """Update experiment history. Delegates to ProjectBuilderManager."""
        self._project_builder.update_experiment_history(experiment_name)

    def mark_active_analysis_complete(self, channel_used: str = None, stim_channel_used: str = None,
                                       events_channel_used: str = None, export_info: dict = None):
        """Mark analysis complete. Delegates to ProjectBuilderManager."""
        self._project_builder.mark_active_analysis_complete(
            channel_used, stim_channel_used, events_channel_used, export_info)

    def _create_sub_row_from_analysis(self, source_row: int, channel_used: str,
                                       stim_channel_used: str, events_channel_used: str,
                                       export_info: dict):
        """Create sub-row from analysis. Delegates to ProjectBuilderManager."""
        self._project_builder.create_sub_row_from_analysis(
            source_row, channel_used, stim_channel_used, events_channel_used, export_info)

    def _update_task_with_export_info(self, task: dict, info: dict, row: int):
        """Update task with export info. Delegates to ProjectBuilderManager."""
        self._project_builder.update_task_with_export_info(task, info, row)

    def _create_sub_row_from_saved_data(self, source_task: dict, source_row: int,
                                         channel: str, info: dict):
        """Create sub-row from saved data. Delegates to ProjectBuilderManager."""
        self._project_builder.create_sub_row_from_saved_data(source_task, source_row, channel, info)

    def eventFilter(self, obj, event):
        """Handle events for installed event filters (e.g., table resize, project combo clicks)."""
        from PyQt6.QtCore import QEvent, QTimer

        # Check if this is a resize event for the table viewport
        if (hasattr(self, 'discoveredFilesTable') and
            obj == self.discoveredFilesTable.viewport() and
            event.type() == QEvent.Type.Resize):

            # Debounce resize events to avoid excessive recalculation
            if self._table_resize_timer is not None:
                self._table_resize_timer.stop()

            self._table_resize_timer = QTimer()
            self._table_resize_timer.setSingleShot(True)
            self._table_resize_timer.timeout.connect(self._auto_fit_table_columns)
            self._table_resize_timer.start(100)  # 100ms delay

        # Handle project name combo click behavior: single-click = dropdown, double-click = edit
        if hasattr(self, 'projectNameCombo'):
            combo = self.projectNameCombo
            line_edit = combo.lineEdit()
            if obj == line_edit:
                if event.type() == QEvent.Type.MouseButtonPress:
                    # Single click - show dropdown popup (if not already in edit mode)
                    if not getattr(self, '_project_combo_edit_mode', False):
                        combo.showPopup()
                        return True  # Consume the event
                elif event.type() == QEvent.Type.MouseButtonDblClick:
                    # Double click - enable editing mode
                    self._project_combo_edit_mode = True
                    line_edit.setReadOnly(False)
                    line_edit.selectAll()
                    return True  # Consume the event
                elif event.type() == QEvent.Type.FocusOut:
                    # Focus lost - exit edit mode
                    self._project_combo_edit_mode = False
                    line_edit.setReadOnly(True)

        return super().eventFilter(obj, event)

    def _auto_fit_table_columns(self):
        """Auto-fit table columns. Delegates to ProjectBuilderManager."""
        self._project_builder.auto_fit_table_columns()

    def _apply_row_styling(self, row: int, is_sub_row: bool = False):
        """Apply row styling. Delegates to ProjectBuilderManager."""
        self._project_builder.apply_row_styling(row, is_sub_row)

    def _apply_all_row_styling(self):
        """Apply all row styling. Delegates to ProjectBuilderManager."""
        self._project_builder.apply_all_row_styling()

    def _clean_parent_rows_with_subrows(self):
        """Clean parent rows. Delegates to ProjectBuilderManager."""
        self._project_builder.clean_parent_rows_with_subrows()

    def _on_analyze_row(self, row):
        """Analyze row. Delegates to ProjectBuilderManager."""
        self._project_builder.on_analyze_row(row)

    def _on_add_row_for_file(self, source_row, force_override=False):
        """Add row for file. Delegates to ProjectBuilderManager."""
        self._project_builder.on_add_row_for_file(source_row, force_override)

    def _on_remove_sub_row(self, row):
        """Remove sub-row. Delegates to ProjectBuilderManager."""
        self._project_builder.on_remove_sub_row(row)

    def _on_master_list_context_menu(self, position):
        """Context menu for table. Delegates to ProjectBuilderManager."""
        self._project_builder.on_master_list_context_menu(position)

    def _export_table_to_csv(self, selected_rows=None):
        """Export table to CSV. Delegates to ProjectBuilderManager."""
        self._project_builder.export_table_to_csv(selected_rows)

    def _bulk_set_column(self, rows, column, field_name, dialog_title, preset_value=None):
        """Bulk set column. Delegates to ProjectBuilderManager."""
        self._project_builder.bulk_set_column(rows, column, field_name, dialog_title, preset_value)

    def _resolve_conflicts_use_npz(self, rows):
        """Resolve conflicts using NPZ. Delegates to ProjectBuilderManager."""
        self._project_builder.resolve_conflicts_use_npz(rows)

    def _resolve_conflicts_keep_table(self, rows):
        """Resolve conflicts keeping table. Delegates to ProjectBuilderManager."""
        self._project_builder.resolve_conflicts_keep_table(rows)

    def _clear_scan_warnings(self, rows):
        """Clear scan warnings. Delegates to ProjectBuilderManager."""
        self._project_builder.clear_scan_warnings(rows)

    def _show_conflict_details(self, rows):
        """Show conflict details. Delegates to ProjectBuilderManager."""
        self._project_builder.show_conflict_details(rows)

    def _update_row_status_icon(self, row, task):
        """Update row status icon. Delegates to ProjectBuilderManager."""
        self._project_builder.update_row_status_icon(row, task)

    def _combine_selected_files(self, rows):
        """Combine selected files. Delegates to ProjectBuilderManager."""
        self._project_builder.combine_selected_files(rows)


    # =========================================================================
    # Notes Files Tab Methods
    # =========================================================================

    def _on_notes_filter_changed(self):
        """Filter notes table rows based on search text."""
        if not hasattr(self, 'notesFilterEdit') or not hasattr(self, 'notesFilesTable'):
            return

        filter_text = self.notesFilterEdit.text().strip().lower()
        table = self.notesFilesTable
        total_count = self._notes_model.rowCount()
        visible_count = 0

        # If no filter, show all rows
        if not filter_text:
            for row in range(total_count):
                table.setRowHidden(row, False)
            visible_count = total_count
        else:
            # Filter rows - search across name, type, and location columns
            for row in range(total_count):
                match = False
                for col in range(self._notes_model.columnCount()):
                    item = self._notes_model.item(row, col)
                    if item and filter_text in item.text().lower():
                        match = True
                        break
                table.setRowHidden(row, not match)
                if match:
                    visible_count += 1

        # Update filter count label
        if hasattr(self, 'notesFilterCountLabel'):
            if filter_text:
                self.notesFilterCountLabel.setText(f"{visible_count} of {total_count}")
            else:
                self.notesFilterCountLabel.setText(f"{total_count} files")

    def _setup_project_name_combo(self):
        """Set up the unified project name combo (shows name, loads recent, editable for rename)."""
        if not hasattr(self, 'projectNameCombo'):
            print("[project-builder] WARNING: projectNameCombo not found in UI")
            return

        combo = self.projectNameCombo
        self._project_combo_updating = False  # Prevent recursive updates
        self._project_combo_edit_mode = False  # Track if user double-clicked to edit

        # Add a visible dropdown arrow button after the combo
        # (Qt's default arrow doesn't show well on dark themes)
        from PyQt6.QtWidgets import QPushButton
        parent_widget = combo.parentWidget()
        if parent_widget and parent_widget.layout():
            layout = parent_widget.layout()
            # Find combo's index in layout
            combo_idx = -1
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item and item.widget() == combo:
                    combo_idx = i
                    break

            if combo_idx >= 0:
                # Add dropdown arrow button right after combo
                dropdown_btn = QPushButton("▼")
                dropdown_btn.setFixedSize(20, 28)
                dropdown_btn.setToolTip("Select project (click) or double-click name to edit")
                dropdown_btn.setStyleSheet("""
                    QPushButton {
                        background-color: transparent;
                        color: #aaaaaa;
                        border: none;
                        font-size: 9px;
                        padding: 0;
                        margin-left: -8px;
                    }
                    QPushButton:hover {
                        color: #ffffff;
                    }
                """)
                dropdown_btn.setFlat(True)
                dropdown_btn.clicked.connect(lambda: combo.showPopup())
                layout.insertWidget(combo_idx + 1, dropdown_btn)

        # Install event filter on lineEdit for single-click dropdown, double-click edit behavior
        line_edit = combo.lineEdit()
        if line_edit:
            line_edit.installEventFilter(self)
            line_edit.setReadOnly(True)  # Start read-only, double-click enables editing

        # Populate with "None" + recent projects
        self._populate_project_name_combo()

        # Connect selection change (for loading projects)
        combo.currentIndexChanged.connect(self._on_project_combo_selected)

        # Connect text editing (for renaming)
        combo.lineEdit().editingFinished.connect(self._on_project_name_edited)

        print("[project-builder] Set up projectNameCombo with None + recent projects")

    def _populate_project_name_combo(self, current_name: str = None):
        """Populate the project name combo with None + recent projects."""
        if not hasattr(self, 'projectNameCombo'):
            return

        combo = self.projectNameCombo
        self._project_combo_updating = True
        combo.blockSignals(True)
        combo.clear()

        # Add "None" option (index 0)
        combo.addItem("No Project", None)

        # Add recent projects
        recent_projects = self.project_manager.get_recent_projects()
        for project_info in recent_projects:
            combo.addItem(project_info['name'], project_info['path'])

        # Set current selection
        if current_name:
            # Find and select the current project
            idx = combo.findText(current_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                # Project name not in list - set as editable text
                combo.setCurrentText(current_name)
        else:
            combo.setCurrentIndex(0)  # "No Project"

        combo.blockSignals(False)
        self._project_combo_updating = False

    def _on_project_combo_selected(self, index: int):
        """Handle selection from project name combo dropdown."""
        if self._project_combo_updating:
            return

        combo = self.projectNameCombo

        if index == 0:
            # "None" selected - clear current project
            self._clear_current_project()
            return

        # Get project path from item data
        project_path = combo.itemData(index)
        if not project_path:
            return

        project_path = Path(project_path)
        if not project_path.exists():
            self._log_status_message(f"Project file not found: {project_path.name}", 3000)
            self._populate_project_name_combo()  # Refresh list
            return

        # IMPORTANT: Cancel any pending autosave before loading new project
        self._project_builder.cancel_pending_autosave()

        try:
            project_data = self.project_manager.load_project(project_path)
            self._populate_ui_from_project(project_data)
            self._log_status_message(f"Loaded project: {project_data['project_name']}", 2000)
        except Exception as e:
            self._show_error("Load Failed", f"Failed to load project:\n{e}")
            self._populate_project_name_combo()  # Reset

    def _on_project_name_edited(self):
        """Handle project name editing (rename)."""
        if self._project_combo_updating:
            return

        # Reset edit mode state
        self._project_combo_edit_mode = False
        combo = self.projectNameCombo
        if combo.lineEdit():
            combo.lineEdit().setReadOnly(True)

        new_name = combo.currentText().strip()

        if not new_name or new_name == "No Project":
            # Empty or invalid name - revert
            self._populate_project_name_combo(getattr(self, '_current_project_name', None))
            return

        # Get current project name
        old_name = getattr(self, '_current_project_name', None)

        if new_name == old_name:
            return  # No change

        if old_name and self._project_directory:
            # Rename existing project - need to rename/delete the old file
            old_filename = self._sanitize_project_filename(old_name)
            new_filename = self._sanitize_project_filename(new_name)

            old_path = Path(self._project_directory) / f"{old_filename}.physiometrics"
            new_path = Path(self._project_directory) / f"{new_filename}.physiometrics"

            # Update the project name first
            self._current_project_name = new_name

            # For rename, do immediate save (not debounced) to ensure new file exists
            # before we delete the old one
            self._project_builder._do_autosave()

            # Delete old file if it exists and is different from new file
            if old_path.exists() and old_path != new_path:
                try:
                    # Remove old project from recent projects list
                    self.project_manager.remove_recent_project(old_path)

                    old_path.unlink()
                    # Also delete backup if exists
                    old_backup = old_path.with_suffix('.physiometrics.bak')
                    if old_backup.exists():
                        old_backup.unlink()
                    print(f"[project-rename] Deleted old project file: {old_path.name}")
                except Exception as e:
                    print(f"[project-rename] Warning: Could not delete old file: {e}")

            # Update recent projects list
            self._populate_project_name_combo(new_name)
            self._log_status_message(f"Project renamed to: {new_name}", 2000)
        else:
            # Setting name for new unsaved project
            self._current_project_name = new_name
            self._log_status_message(f"Project name set: {new_name}", 2000)

    def _sanitize_project_filename(self, name: str) -> str:
        """Convert project name to valid filename (mirrors project_manager logic)."""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return name.strip()

    def _clear_current_project(self):
        """Clear the current project (when None is selected)."""
        self._current_project_name = None
        self._project_directory = None
        self._master_file_list = []
        self._discovered_files_data = []
        self._notes_files_data = []

        # Clear UI
        if hasattr(self, 'directoryPathEdit'):
            self.directoryPathEdit.clear()
        if hasattr(self, '_file_table_model'):
            self._file_table_model.clear_all()
        if hasattr(self, '_notes_model'):
            self._notes_model.removeRows(0, self._notes_model.rowCount())
        if hasattr(self, 'summaryLabel'):
            self.summaryLabel.setText("Summary: No project loaded")

        self._log_status_message("Project closed", 2000)

    def _connect_project_builder_buttons(self):
        """Connect signals for Project Builder buttons defined in .ui file."""
        # Connect Scan Saved Data button
        if hasattr(self, 'scanSavedDataButton'):
            self.scanSavedDataButton.clicked.connect(self._scan_manager.on_project_scan_saved_data)

        # Connect Resolve Conflicts button
        if hasattr(self, 'resolveConflictsButton'):
            self.resolveConflictsButton.clicked.connect(self.on_resolve_all_conflicts)

        # Connect Show Full Content checkbox
        if hasattr(self, 'tableFullContentCheckbox'):
            self.tableFullContentCheckbox.stateChanged.connect(self._on_table_column_mode_changed)

        # Initialize AI Notebook Manager and connect chatbot panel widgets
        self._ai_notebook = AINotebookManager(self)
        if hasattr(self, 'chatSendButton'):
            self.chatSendButton.clicked.connect(self._ai_notebook.on_chat_send)
        if hasattr(self, 'chatInputEdit'):
            self.chatInputEdit.returnPressed.connect(self._ai_notebook.on_chat_send)
        if hasattr(self, 'chatSettingsButton'):
            self.chatSettingsButton.clicked.connect(self._ai_notebook.open_ai_settings)
        if hasattr(self, 'chatClearButton'):
            self.chatClearButton.clicked.connect(self._ai_notebook.clear_chat_history)
        if hasattr(self, 'chatStopButton'):
            self.chatStopButton.clicked.connect(self._ai_notebook.on_chat_stop)

        # Initialize model selection combo
        if hasattr(self, 'modelSelectCombo'):
            self._ai_notebook.init_model_selector()

        # Initialize Code Notebook Manager and connect widgets
        self._code_notebook = CodeNotebookManager(self)
        if hasattr(self, 'runCodeButton'):
            self.runCodeButton.clicked.connect(self._code_notebook.on_run_code)
        if hasattr(self, 'clearOutputButton'):
            self.clearOutputButton.clicked.connect(self._code_notebook.on_clear_code_output)

        # Add Pop Out and Save Figure buttons to notebook header
        self._code_notebook.add_notebook_extra_buttons()

        # Connect table filter widgets
        if hasattr(self, 'tableFilterEdit'):
            self.tableFilterEdit.textChanged.connect(self._on_table_filter_changed)
        if hasattr(self, 'filterColumnCombo'):
            self.filterColumnCombo.currentIndexChanged.connect(self._on_table_filter_changed)

        # Set up main horizontal splitter (left column vs AI assistant, 75%/25%)
        if hasattr(self, 'mainContentSplitter'):
            self.mainContentSplitter.setSizes([750, 250])

    def _on_table_column_mode_changed(self, state):
        """Handle toggle of the 'Show Full Content' checkbox."""
        self._auto_fit_table_columns()

    def _on_table_filter_changed(self, *args):
        """Filter table rows based on search text and selected column."""
        if not hasattr(self, 'tableFilterEdit') or not hasattr(self, 'filterColumnCombo'):
            return

        filter_text = self.tableFilterEdit.text().strip().lower()
        column_filter = self.filterColumnCombo.currentText()

        # Map column filter to data keys
        column_map = {
            'All Columns': None,  # Search all
            'File Name': 'file_name',
            'Protocol': 'protocol',
            'Animal ID': 'animal_id',
            'Strain': 'strain',
            'Keywords': 'keywords_display',
        }
        target_column = column_map.get(column_filter)

        # Get table view (it's called discoveredFilesTable in the UI)
        if not hasattr(self, 'discoveredFilesTable'):
            return

        table = self.discoveredFilesTable
        visible_count = 0
        total_count = len(self._master_file_list) if self._master_file_list else 0

        # If no filter, show all rows
        if not filter_text:
            for row in range(self._file_table_model.rowCount()):
                table.setRowHidden(row, False)
            if hasattr(self, 'filterCountLabel'):
                self.filterCountLabel.setText("")
            return

        # Filter rows
        for row in range(self._file_table_model.rowCount()):
            if row >= len(self._master_file_list):
                continue

            task = self._master_file_list[row]
            match = False

            if target_column:
                # Search specific column
                value = str(task.get(target_column, '')).lower()
                match = filter_text in value
            else:
                # Search all relevant columns - include file path for better searching
                searchable_columns = ['file_name', 'protocol', 'animal_id', 'strain',
                                     'keywords_display', 'power', 'sex', 'stim_type',
                                     'experiment', 'file_path']
                for col in searchable_columns:
                    value = str(task.get(col, '')).lower()
                    if filter_text in value:
                        match = True
                        break

            table.setRowHidden(row, not match)
            if match:
                visible_count += 1

        # Update count label
        if hasattr(self, 'filterCountLabel'):
            if visible_count == total_count:
                self.filterCountLabel.setText("")
            else:
                self.filterCountLabel.setText(f"({visible_count} of {total_count})")

    def on_resolve_all_conflicts(self):
        """Show dialog to resolve all conflicts at once."""
        # Find all rows with warnings
        rows_with_warnings = []
        rows_with_conflicts = []
        for i, task in enumerate(self._master_file_list):
            warnings = task.get('scan_warnings', {})
            if warnings:
                rows_with_warnings.append(i)
                if warnings.get('conflicts'):
                    rows_with_conflicts.append(i)

        if not rows_with_warnings:
            self._show_info("No Warnings", "No scan warnings or conflicts to resolve.")
            return

        # Show the conflict details dialog with all warning rows
        self._show_conflict_details(rows_with_warnings)

        # Update button state after resolution
        self._update_resolve_conflicts_button()

    def _update_resolve_conflicts_button(self):
        """Update the Resolve Conflicts button text and enabled state."""
        if not hasattr(self, 'resolveConflictsButton'):
            return

        warnings_count = sum(1 for task in self._master_file_list if task.get('scan_warnings'))

        self.resolveConflictsButton.setEnabled(warnings_count > 0)
        if warnings_count > 0:
            self.resolveConflictsButton.setText(f"⚠ Resolve Conflicts ({warnings_count})")
        else:
            self.resolveConflictsButton.setText("⚠ Resolve Conflicts")

    # NOTE: Old experiment methods removed (on_project_add_experiment, on_project_remove_experiment,
    # on_project_export_experiment). These buttons are hidden and we now use the master file list.

    def on_edit_project_name(self):
        """Edit the current project name (legacy - combo is now directly editable)."""
        from PyQt6.QtWidgets import QInputDialog

        current_name = getattr(self, '_current_project_name', '') or "Untitled Project"

        new_name, ok = QInputDialog.getText(
            self, "Edit Project Name", "Enter project name:",
            text=current_name
        )

        if ok and new_name.strip():
            self._current_project_name = new_name.strip()
            if hasattr(self, 'projectNameCombo'):
                self.projectNameCombo.setCurrentText(new_name.strip())
            self._log_status_message(f"Project renamed to: {new_name.strip()}", 2000)

    def on_project_load(self, index):
        """Load a project from recent projects list (legacy - now handled by _on_project_combo_selected)."""
        # This method is kept for backward compatibility but is no longer connected
        # Project loading is now handled by _on_project_combo_selected via projectNameCombo
        pass

    def _populate_load_project_combo(self):
        """Populate the project name combo (legacy wrapper)."""
        # Use the new unified method
        current_name = getattr(self, '_current_project_name', None)
        self._populate_project_name_combo(current_name)

    def _populate_ui_from_project(self, project_data):
        """Populate UI with loaded project data."""
        # Set project name and directory
        project_name = project_data['project_name']
        self._current_project_name = project_name
        if hasattr(self, 'projectNameCombo'):
            self.projectNameCombo.setCurrentText(project_name)
        self._project_directory = str(project_data['data_directory'])
        self.directoryPathEdit.setText(self._project_directory)

        # Populate discovered files data
        self._discovered_files_data = project_data['files']

        # Count protocols and check if keywords need extraction
        protocols = set()
        need_keywords = any('keywords_display' not in f for f in self._discovered_files_data)

        if need_keywords:
            total_files = len(self._discovered_files_data)
            # Show progress bar
            self.projectProgressBar.setVisible(True)
            self.projectProgressBar.setValue(0)
            self.projectProgressBar.setFormat(f"Extracting keywords: 0/{total_files} (0%)")

            self._log_status_message(f"Extracting path keywords for {total_files} files...", 0)
            QApplication.processEvents()  # Update status message

            # Extract keywords for files that don't have them
            from core.fast_abf_reader import extract_path_keywords
            base_dir = Path(self._project_directory)

            for idx, file_data in enumerate(self._discovered_files_data):
                if 'keywords_display' not in file_data:
                    # Extract keywords
                    file_path = file_data.get('file_path')
                    if file_path and not Path(file_path).is_absolute():
                        file_path = base_dir / file_path

                    if file_path:
                        path_info = extract_path_keywords(Path(file_path), base_dir)

                        # Format keywords for display - show relative path as the primary keyword
                        keywords_display = []

                        # Add relative path (the main keyword showing file location)
                        if path_info.get('relative_path'):
                            keywords_display.append(path_info['relative_path'])
                        elif path_info['subdirs']:
                            # Fallback to subdirs if relative_path not available
                            keywords_display.append('/'.join(path_info['subdirs']))

                        # Also add power levels and animal IDs if detected
                        if path_info['power_levels']:
                            keywords_display.extend(path_info['power_levels'])
                        if path_info['animal_ids']:
                            keywords_display.extend([f"ID:{id}" for id in path_info['animal_ids']])

                        file_data['path_keywords'] = path_info
                        file_data['keywords_display'] = ', '.join(keywords_display) if keywords_display else ''

                # Update progress every 50 files
                if idx % 50 == 0 or idx == total_files - 1:
                    progress_pct = int((idx + 1) / total_files * 100)
                    self.projectProgressBar.setValue(progress_pct)
                    self.projectProgressBar.setFormat(f"Extracting keywords: {idx + 1}/{total_files} ({progress_pct}%)")
                    QApplication.processEvents()

        # Show progress for data preparation
        total_files = len(self._discovered_files_data)
        self.projectProgressBar.setVisible(True)
        self.projectProgressBar.setValue(0)
        self.projectProgressBar.setFormat(f"Loading files: 0/{total_files} (0%)")
        QApplication.processEvents()

        # Ensure all files have required fields
        for i, file_data in enumerate(self._discovered_files_data):
            # Ensure file has all needed fields
            if 'channel' not in file_data:
                file_data['channel'] = ''
            if 'stim_channel' not in file_data:
                file_data['stim_channel'] = ''
            if 'events_channel' not in file_data:
                file_data['events_channel'] = ''
            if 'stim_channels' not in file_data:
                file_data['stim_channels'] = []
            if 'stim_frequency' not in file_data:
                file_data['stim_frequency'] = ''
            if 'strain' not in file_data:
                file_data['strain'] = ''
            if 'stim_type' not in file_data:
                file_data['stim_type'] = ''
            if 'power' not in file_data:
                # Try to auto-fill from keywords
                path_kw = file_data.get('path_keywords', {})
                file_data['power'] = path_kw.get('power_levels', [''])[0] if path_kw.get('power_levels') else ''
            if 'sex' not in file_data:
                file_data['sex'] = ''
            if 'animal_id' not in file_data:
                # Try to auto-fill from keywords
                path_kw = file_data.get('path_keywords', {})
                file_data['animal_id'] = path_kw.get('animal_ids', [''])[0] if path_kw.get('animal_ids') else ''
            if 'status' not in file_data:
                file_data['status'] = 'pending'
            # Export tracking fields
            if 'export_path' not in file_data:
                file_data['export_path'] = ''
            if 'export_date' not in file_data:
                file_data['export_date'] = ''
            if 'export_version' not in file_data:
                file_data['export_version'] = ''
            if 'exports' not in file_data:
                file_data['exports'] = {
                    'npz': False,
                    'timeseries_csv': False,
                    'breaths_csv': False,
                    'events_csv': False,
                    'pdf': False,
                    'session_state': False,
                    'ml_training': False,
                }

            if file_data.get('protocol'):
                protocols.add(file_data['protocol'])

            # Update progress every 50 files
            if i % 50 == 0 or i == total_files - 1:
                progress_pct = int((i + 1) / total_files * 100)
                self.projectProgressBar.setValue(progress_pct)
                self.projectProgressBar.setFormat(f"Loading files: {i + 1}/{total_files} ({progress_pct}%)")
                QApplication.processEvents()

        # Sync master file list with discovered files
        self._master_file_list = [dict(f) for f in self._discovered_files_data]

        # Rebuild the table using the model
        self._rebuild_table_from_master_list()

        # Auto-fit columns
        self._auto_fit_table_columns()

        # Hide progress bar
        self.projectProgressBar.setVisible(False)
        self.projectProgressBar.setValue(0)

        # Update summary
        summary_text = f"Summary: {len(self._discovered_files_data)} ABF files | {len(protocols)} protocols"
        self.summaryLabel.setText(summary_text)

        # Restore notes data if available
        if project_data.get('notes_directory'):
            self._notes_directory = project_data['notes_directory']
            if hasattr(self, 'notesFolderEdit'):
                self.notesFolderEdit.setText(self._notes_directory)
        if project_data.get('notes_files'):
            self._notes_files_data = project_data['notes_files']
            # Populate notes table
            self._populate_notes_table_from_data()

        # Populate consolidation source list with analyzed files from project
        self._populate_consolidation_source_list()

        # Note: Old experiment structure is ignored - we now use the flat master file list
        # If old project had experiments, their task data could be migrated here if needed

    def _populate_notes_table_from_data(self):
        """Populate notes table. Delegates to NotesPreviewManager."""
        self._notes_preview_manager.populate_notes_table_from_data()

    # NOTE: _load_experiments_from_project method removed - no longer using experiment tree

    def _ask_locate_project(self, project_name, expected_path):
        """
        Ask user if they want to locate a missing project file.

        Returns:
            "locate" if user wants to locate, "cancel" otherwise
        """
        from PyQt6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Project Not Found")
        msg.setText(f"Cannot find project '{project_name}'")
        msg.setInformativeText(f"Expected location:\n{expected_path}\n\nWould you like to locate the project file?")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        msg.setDefaultButton(QMessageBox.StandardButton.Yes)

        result = msg.exec()
        if result == QMessageBox.StandardButton.Yes:
            return "locate"
        return "cancel"

if __name__ == "__main__":
    from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QLabel, QWidget
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtCore import Qt, QTimer
    from version_info import VERSION_STRING

    app = QApplication(sys.argv)

    # Check for first launch and show welcome dialog
    from core import config as app_config
    if app_config.is_first_launch():
        from dialogs import FirstLaunchDialog
        first_launch_dialog = FirstLaunchDialog()
        if first_launch_dialog.exec():
            # User clicked Continue - save preferences
            telemetry_enabled, crash_reports_enabled = first_launch_dialog.get_preferences()
            cfg = app_config.load_config()
            cfg['telemetry_enabled'] = telemetry_enabled
            cfg['crash_reports_enabled'] = crash_reports_enabled
            cfg['first_launch'] = False
            app_config.save_config(cfg)
        else:
            # User closed dialog - use defaults and continue
            app_config.mark_first_launch_complete()

    # Initialize telemetry (after first-launch dialog)
    telemetry.init_telemetry()

    # Initialize error reporter (writes session lock, registers cleanup)
    from core import error_reporting
    error_reporter = error_reporting.init_error_reporter()

    # Check if previous session crashed (before installing new hook)
    previous_crash = error_reporting.was_previous_session_crashed()
    pending_report = error_reporting.get_appropriate_crash_report() if previous_crash else None

    # Install global exception handler for crash tracking
    def exception_hook(exctype, value, tb):
        """Catch unhandled exceptions, save report, and optionally show dialog."""
        import traceback

        # Log crash to GA4 telemetry
        telemetry.log_crash(
            error_message=f"{exctype.__name__}: {str(value)[:100]}",
            traceback_depth=len(traceback.extract_tb(tb))
        )

        # Save crash report locally
        if app_config.is_crash_reports_enabled():
            try:
                session_data = telemetry.get_session_data()
                crash_path = error_reporting.save_crash_report(
                    exctype, value, tb, session_data
                )

                # Try to show crash report dialog
                try:
                    app_instance = QApplication.instance()
                    if app_instance:
                        from dialogs.crash_report_dialog import CrashReportDialog
                        report = error_reporting.ErrorReporter.get_instance().load_crash_report(crash_path)
                        if report:
                            dialog = CrashReportDialog(report, on_startup=False)
                            dialog.exec()
                except Exception as dialog_error:
                    print(f"[Crash Report] Could not show dialog: {dialog_error}")

            except Exception as save_error:
                print(f"[Crash Report] Failed to save: {save_error}")

        # Call default handler to print traceback
        sys.__excepthook__(exctype, value, tb)

    sys.excepthook = exception_hook

    # Create splash screen
    # Try to load icon (with fallback path handling)
    splash_paths = [
        Path(__file__).parent / "images" / "plethapp_splash_dark-01.png",
        Path(__file__).parent / "images" / "plethapp_splash.png",
        Path(__file__).parent / "images" / "plethapp_thumbnail_dark_round.ico",
        Path(__file__).parent / "assets" / "plethapp_thumbnail_dark_round.ico",
    ]

    splash_pix = None
    for splash_path in splash_paths:
        if splash_path.exists():
            splash_pix = QPixmap(str(splash_path))
            break

    if splash_pix is None or splash_pix.isNull():
        # Fallback: create simple splash with text
        splash_pix = QPixmap(250, 190)
        splash_pix.fill(Qt.GlobalColor.darkGray)

    # Scale to reasonable size for display (increased ~25% from original 150 to 190)
    splash_pix = splash_pix.scaled(190, 190, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)

    # Add loading message
    splash.showMessage(
        f"Loading PhysioMetrics v{VERSION_STRING}...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white
    )
    splash.show()
    app.processEvents()

    # Create main window (this is where the loading time happens)
    w = MainWindow()

    # Close splash and show main window
    splash.finish(w)
    w.show()

    # Check for previous crash and show dialog (after window is visible)
    if pending_report and app_config.is_crash_reports_enabled():
        from dialogs.crash_report_dialog import show_crash_report_dialog

        def show_previous_crash_dialog():
            show_crash_report_dialog(pending_report, on_startup=True, parent=w)

        # Delay to let the main window fully initialize
        QTimer.singleShot(500, show_previous_crash_dialog)

    sys.exit(app.exec())