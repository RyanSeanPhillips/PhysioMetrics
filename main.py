from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem, QAbstractItemView, QTableWidgetItem, QTreeWidgetItem
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QIcon
from consolidation import ConsolidationManager

import re
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox, QLabel,
    QDialogButtonBox, QPushButton, QHBoxLayout, QCheckBox
)

import csv, json



from pathlib import Path
from typing import List
import sys
import os

# Fix KMeans memory leak warning on Windows
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd

from core.state import AppState
from core import abf_io, filters
from core.plotting import PlotHost
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

        ui_file = Path(__file__).parent / "ui" / "pleth_app_layout_04_horizontal.ui"
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

        # Style status bar to match dark theme
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
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

        # --- Embed Matplotlib into MainPlot (QFrame in Designer) ---
        self.plot_host = PlotHost(self.MainPlot)
        layout = self.MainPlot.layout()
        if layout is None:
            from PyQt6.QtWidgets import QVBoxLayout
            layout = QVBoxLayout(self.MainPlot)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_host)

        saved_geom = self.settings.value("geometry")
        if saved_geom:
            self.restoreGeometry(saved_geom)

        # --- Wire browse ---
        self.BrowseButton.clicked.connect(self.on_browse_clicked)

        # Add Ctrl+O shortcut - triggers different buttons based on active tab
        from PyQt6.QtGui import QShortcut, QKeySequence
        ctrl_o_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        ctrl_o_shortcut.activated.connect(self.on_ctrl_o_pressed)

        # Add F1 shortcut for Help
        f1_shortcut = QShortcut(QKeySequence("F1"), self)
        f1_shortcut.activated.connect(self.on_help_clicked)

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

        # Auto-load ML models from last used directory (if available)
        # This must happen BEFORE setting defaults and connecting signals
        self._auto_load_ml_models_on_startup()

        # Update dropdown states based on loaded models (will disable ML options if models not loaded)
        self._update_classifier_dropdowns()

        # NOW connect signals AFTER models are loaded and dropdowns are configured
        self.peak_detec_combo.currentTextChanged.connect(self.on_classifier_changed)
        self.eup_sniff_combo.currentTextChanged.connect(self.on_eupnea_sniff_classifier_changed)
        self.digh_combo.currentTextChanged.connect(self.on_sigh_classifier_changed)

        # Set defaults to XGBoost (matches state.py default) - this will now trigger signals
        # If models not loaded, _update_classifier_dropdowns() already fell back to Threshold/GMM/Manual
        self.peak_detec_combo.setCurrentText("XGBoost")
        self.eup_sniff_combo.setCurrentText("XGBoost")
        self.digh_combo.setCurrentText("XGBoost")

        # --- Wire filter controls ---
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(150)       # ms
        self._redraw_timer.timeout.connect(self.redraw_main_plot)

        # filters: commit-on-finish, not per key
        self.LowPassVal.editingFinished.connect(self.update_and_redraw)
        self.HighPassVal.editingFinished.connect(self.update_and_redraw)
        self.FilterOrderSpin.valueChanged.connect(self.update_and_redraw)

        # checkboxes toggled immediately, but we debounce the draw
        self.LowPass_checkBox.toggled.connect(self.update_and_redraw)
        self.HighPass_checkBox.toggled.connect(self.update_and_redraw)
        self.InvertSignal_checkBox.toggled.connect(self.update_and_redraw)

        # Re-enable Apply button when filters change (peaks need to be recalculated)
        self.LowPassVal.editingFinished.connect(self._on_filter_changed)
        self.HighPassVal.editingFinished.connect(self._on_filter_changed)
        self.FilterOrderSpin.valueChanged.connect(self._on_filter_changed)
        self.LowPass_checkBox.toggled.connect(self._on_filter_changed)
        self.HighPass_checkBox.toggled.connect(self._on_filter_changed)
        self.InvertSignal_checkBox.toggled.connect(self._on_filter_changed)

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

        # Y-axis padding spinbox (for percentile mode)
        if hasattr(self, 'ypadding_SpinBox'):
            self.ypadding_SpinBox.setMinimum(0.0)
            self.ypadding_SpinBox.setMaximum(1.0)
            self.ypadding_SpinBox.setSingleStep(0.05)
            self.ypadding_SpinBox.setDecimals(2)
            self.ypadding_SpinBox.setValue(self.state.autoscale_padding)
            self.ypadding_SpinBox.valueChanged.connect(self.on_ypadding_changed)

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

        # Initialize editing modes manager
        self.editing_modes = EditingModes(self)

        # --- Mark Events button (event detection settings) ---
        self.MarkEventsButton.clicked.connect(self.on_mark_events_clicked)

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

        # === Project Builder Connections ===
        self.browseDirectoryButton.clicked.connect(self.on_project_browse_directory)
        self.scanFilesButton.clicked.connect(self.on_project_scan_files)
        # NOTE: addToProjectButton, addExperimentButton, removeExperimentButton, exportExperimentButton
        # are hidden - no longer using experiment-based workflow (see _setup_master_file_list)
        self.clearFilesButton.clicked.connect(self.on_project_clear_files)
        self.newProjectButton.clicked.connect(self.on_project_new)
        self.saveProjectButton.clicked.connect(self.on_project_save)
        self.loadProjectCombo.currentIndexChanged.connect(self.on_project_load)

        # Add "Scan Saved Data" button programmatically
        self._add_scan_saved_data_button()

        # Project Builder state
        self._project_directory = None
        self._discovered_files_data = []  # Store file metadata from scan

        # Hide Browse Directory button (New Project handles this now)
        self.browseDirectoryButton.hide()

        # Make project name read-only and add edit button
        self.projectNameEdit.setReadOnly(True)
        self.projectNameEdit.setStyleSheet("""
            QLineEdit {
                background-color: #2b2b2b;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        # Add edit button next to project name
        from PyQt6.QtWidgets import QPushButton
        self.editProjectNameButton = QPushButton("✏")
        self.editProjectNameButton.setMaximumWidth(30)
        self.editProjectNameButton.setMaximumHeight(30)
        self.editProjectNameButton.setToolTip("Edit project name")
        self.editProjectNameButton.setStyleSheet("""
            QPushButton {
                background-color: #3e3e42;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 2px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        self.editProjectNameButton.clicked.connect(self.on_edit_project_name)

        # Find the correct layout containing projectNameEdit (may be nested)
        def find_widget_in_layout(layout, widget):
            """Recursively find widget in layout and return (layout, index)."""
            if not layout:
                return None, -1

            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item:
                    # Check if this item is the widget
                    if item.widget() == widget:
                        return layout, i
                    # Check if this item is a nested layout
                    if item.layout():
                        result_layout, result_index = find_widget_in_layout(item.layout(), widget)
                        if result_layout:
                            return result_layout, result_index
            return None, -1

        # Search for the layout containing projectNameEdit
        parent_widget = self.projectNameEdit.parent()
        parent_layout = parent_widget.layout()

        target_layout, widget_index = find_widget_in_layout(parent_layout, self.projectNameEdit)

        if target_layout and widget_index >= 0:
            # Insert edit button right after the name edit
            target_layout.insertWidget(widget_index + 1, self.editProjectNameButton)
            print(f"[project-builder] OK Inserted edit button next to project name")
        else:
            # Fallback: try adding to parent layout
            if parent_layout:
                parent_layout.addWidget(self.editProjectNameButton)
                print(f"[project-builder] WARNING Added edit button to parent layout (fallback)")
            else:
                print(f"[project-builder] ERROR: Could not find layout for edit button")

        # Populate recent projects dropdown
        self._populate_load_project_combo()

        # === Master File List Setup ===
        # Hide the Project Organization section (right column) - we're using a flat master list instead
        self._setup_master_file_list()

        # Store file list data (each row = one analysis task: file + channel + animal)
        self._master_file_list = []  # List of task dicts
        self._active_master_list_row = None  # Track which row is being analyzed

        # optional: keep a handle to the chosen dir
        self._curation_dir = None

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

    def _update_filename_display(self):
        """Update the filename display in status bar with file metadata."""
        st = self.state
        if not hasattr(st, 'file_info') or not st.file_info:
            self.filename_label.setText("No file loaded")
            return

        file_info = st.file_info[0]
        filename = file_info['path'].name

        # Build metadata string for ABF files
        metadata_parts = []
        if file_info.get('protocol'):
            metadata_parts.append(file_info['protocol'])
        if file_info.get('n_channels'):
            metadata_parts.append(f"{file_info['n_channels']} ch")

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

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int

                # DWMWA_USE_IMMERSIVE_DARK_MODE
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20

                # Get window handle
                hwnd = int(self.winId())

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

    def on_browse_clicked(self):
        last_dir = self.settings.value("last_dir", str(Path.home()))
        if not Path(str(last_dir)).exists():
            last_dir = str(Path.home())

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select File(s)", last_dir,
            "All Supported (*.abf *.smrx *.edf *.pleth.npz);;Data Files (*.abf *.smrx *.edf);;PhysioMetrics Sessions (*.pleth.npz);;ABF Files (*.abf);;SMRX Files (*.smrx);;EDF Files (*.edf);;All Files (*.*)"
        )
        if not paths:
            return

        # Convert to Path objects
        file_paths = [Path(p) for p in paths]

        # Store the directory of the first file
        self.settings.setValue("last_dir", str(file_paths[0].parent))

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
        else:
            # Load data files (ABF, SMRX, EDF)
            if len(file_paths) == 1:
                self.load_file(file_paths[0])
            else:
                self.load_multiple_files(file_paths)

    def load_file(self, path: Path):
        import time
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        t_start = time.time()

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
        QApplication.processEvents()

        def update_progress(current, total, message):
            """Callback to update progress dialog."""
            progress.setValue(current)
            progress.setLabelText(f"{message}\n{path.name}")
            QApplication.processEvents()

        try:
            # Load data file (supports .abf, .smrx, .edf)
            sr, sweeps_by_ch, ch_names, t, file_metadata = abf_io.load_data_file(path, progress_callback=update_progress)
        except Exception as e:
            progress.close()
            self._show_error("Load error", str(e))
            return
        finally:
            progress.close()


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
        # This cache stores computed metric traces during export for reuse in PDF generation
        self._export_metric_cache = {}

        # Clear z-score global statistics cache
        self.zscore_global_mean = None
        self.zscore_global_std = None

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
            st.stim_chan = auto_stim

            # Trigger stim detection
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
        from PyQt6.QtWidgets import QProgressDialog, QApplication, QMessageBox
        from PyQt6.QtCore import Qt

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
        QApplication.processEvents()

        def update_progress(current, total, message):
            """Callback to update progress dialog."""
            progress.setValue(current)
            progress.setLabelText(message)
            QApplication.processEvents()

        try:
            # Load and concatenate files
            sr, sweeps_by_ch, ch_names, t, file_info = abf_io.load_and_concatenate_abf_files(
                file_paths, progress_callback=update_progress
            )
        except Exception as e:
            progress.close()
            self._show_error("Load error", str(e))
            return
        finally:
            progress.close()

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
        # This cache stores computed metric traces during export for reuse in PDF generation
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
            st.stim_chan = auto_stim

            # Trigger stim detection
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
                'apnea_threshold': self._parse_float(self.ApneaThresh) or 0.5
            }

            save_state_to_npz(self.state, save_path, include_raw_data=include_raw, gmm_cache=gmm_cache, app_settings=app_settings)

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

    def load_npz_state(self, npz_path: Path):
        """Load complete analysis state from .pleth.npz file."""
        from core.npz_io import load_state_from_npz, get_npz_metadata
        import time

        t_start = time.time()

        # Get metadata for display
        metadata = get_npz_metadata(npz_path)

        if 'error' in metadata:
            self._show_error("Load Error",
                f"Failed to read NPZ file:\n\n{metadata['error']}"
            )
            return

        # Show loading dialog
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        progress = QProgressDialog(f"Loading session...\n{npz_path.name}", None, 0, 100, self)
        progress.setWindowTitle("Loading PhysioMetrics Session")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setValue(10)
        progress.show()
        QApplication.processEvents()

        try:
            # Load state from NPZ
            progress.setLabelText(f"Reading session file...\n{npz_path.name}")
            progress.setValue(30)
            QApplication.processEvents()

            new_state, raw_data_loaded, gmm_cache, app_settings = load_state_from_npz(npz_path, reload_raw_data=True)

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
            QApplication.processEvents()

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
            QApplication.processEvents()

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

            progress.setValue(70)
            QApplication.processEvents()

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

                print(f"[npz-load] Restored app settings: filter_order={self.filter_order}, "
                      f"zscore={self.use_zscore_normalization}, notch={self.notch_filter_lower}-{self.notch_filter_upper}, "
                      f"apnea_thresh={apnea_thresh}")

            progress.setValue(80)
            QApplication.processEvents()

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
            QApplication.processEvents()

            # Restore navigation and plot
            self.navigation_manager.reset_window_state()

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
        """
        import numpy as np

        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return None, None

        Y = st.sweeps[st.analyze_chan]  # (n_samples, n_sweeps)

        # Collect all processed data across sweeps (apply filters but not z-score)
        all_data = []
        for sweep_idx in range(Y.shape[1]):
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
            if self.notch_filter_lower is not None and self.notch_filter_upper is not None:
                y = self._apply_notch_filter(y, st.sr_hz, self.notch_filter_lower, self.notch_filter_upper)

            all_data.append(y)

        # Concatenate all sweeps
        concatenated = np.concatenate(all_data)

        # Compute global statistics (excluding NaN values)
        valid_mask = ~np.isnan(concatenated)
        if not np.any(valid_mask):
            return None, None

        global_mean = np.mean(concatenated[valid_mask])
        global_std = np.std(concatenated[valid_mask], ddof=1)

        print(f"[zscore] Computed global stats: mean={global_mean:.4f}, std={global_std:.4f}")
        return global_mean, global_std

    def plot_all_channels(self):
        """Delegate to PlotManager."""
        self.plot_manager.plot_all_channels()

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
        """Update dropdown options based on loaded models."""
        models = self.state.loaded_ml_models or {}

        # Check which models are available (use prefix matching for accuracy suffix)
        has_model1_xgboost = any(k.startswith('model1_xgboost') for k in models.keys())
        has_model1_rf = any(k.startswith('model1_rf') for k in models.keys())
        has_model1_mlp = any(k.startswith('model1_mlp') for k in models.keys())

        has_model3_xgboost = any(k.startswith('model3_xgboost') for k in models.keys())
        has_model3_rf = any(k.startswith('model3_rf') for k in models.keys())
        has_model3_mlp = any(k.startswith('model3_mlp') for k in models.keys())

        has_model2_xgboost = any(k.startswith('model2_xgboost') for k in models.keys())
        has_model2_rf = any(k.startswith('model2_rf') for k in models.keys())
        has_model2_mlp = any(k.startswith('model2_mlp') for k in models.keys())

        # Update Model 1 (Breath Detection) dropdown
        # Index 0=Threshold (always enabled), 1=XGBoost, 2=RF, 3=MLP
        model = self.peak_detec_combo.model()
        model.item(0).setEnabled(True)  # Threshold always available
        model.item(1).setEnabled(has_model1_xgboost)
        model.item(2).setEnabled(has_model1_rf)
        model.item(3).setEnabled(has_model1_mlp)

        # Update Model 3 (Eupnea/Sniff) dropdown
        # Index 0=GMM (always enabled), 1=XGBoost, 2=RF, 3=MLP
        model = self.eup_sniff_combo.model()
        model.item(0).setEnabled(True)  # GMM always available
        model.item(1).setEnabled(has_model3_xgboost)
        model.item(2).setEnabled(has_model3_rf)
        model.item(3).setEnabled(has_model3_mlp)

        # Update Model 2 (Sigh) dropdown
        # Index 0=Manual (always enabled), 1=XGBoost, 2=RF, 3=MLP
        model = self.digh_combo.model()
        model.item(0).setEnabled(True)  # Manual always available
        model.item(1).setEnabled(has_model2_xgboost)
        model.item(2).setEnabled(has_model2_rf)
        model.item(3).setEnabled(has_model2_mlp)

        # If current selection is disabled, fall back to default
        self._fallback_disabled_classifiers()

    def _fallback_disabled_classifiers(self):
        """Check if current classifier selections are disabled and fall back to defaults."""
        # Model 1 (Breath Detection)
        current_text = self.peak_detec_combo.currentText()
        current_index = self.peak_detec_combo.currentIndex()
        if current_index >= 0:
            model = self.peak_detec_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 1 classifier '{current_text}' not available, falling back to Threshold")
                self.peak_detec_combo.setCurrentText("Threshold")
                self.state.active_classifier = "threshold"

        # Model 3 (Eupnea/Sniff)
        current_text = self.eup_sniff_combo.currentText()
        current_index = self.eup_sniff_combo.currentIndex()
        if current_index >= 0:
            model = self.eup_sniff_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 3 classifier '{current_text}' not available, falling back to GMM")
                self.eup_sniff_combo.setCurrentText("GMM")
                self.state.active_eupnea_sniff_classifier = "gmm"

        # Model 2 (Sigh)
        current_text = self.digh_combo.currentText()
        current_index = self.digh_combo.currentIndex()
        if current_index >= 0:
            model = self.digh_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 2 classifier '{current_text}' not available, falling back to Manual")
                self.digh_combo.setCurrentText("Manual")
                self.state.active_sigh_classifier = "manual"

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
        """Silently load ML models from last used directory on startup."""
        from pathlib import Path
        import core.ml_prediction as ml_prediction

        # Get last used models directory from settings
        last_models_dir = self.settings.value("ml_models_path", None)

        # Only auto-load if we have a saved path
        if not last_models_dir:
            print("[Auto-load] No saved models directory")
            return

        models_path = Path(last_models_dir)

        # Check if directory still exists
        if not models_path.exists() or not models_path.is_dir():
            print(f"[Auto-load] Saved models directory no longer exists: {models_path}")
            return

        try:
            # Look for model files
            model_files = list(models_path.glob("model*.pkl"))

            if not model_files:
                print(f"[Auto-load] No model files found in: {models_path}")
                return

            # Load all models
            loaded_models = {}
            for model_file in model_files:
                try:
                    model, metadata = ml_prediction.load_model(model_file)
                    model_key = model_file.stem
                    loaded_models[model_key] = {
                        'model': model,
                        'metadata': metadata,
                        'path': str(model_file)
                    }
                except Exception as e:
                    print(f"[Auto-load] Warning: Failed to load {model_file.name}: {e}")

            if loaded_models:
                # Store in state
                self.state.loaded_ml_models = loaded_models

                # Update dropdown states
                self._update_classifier_dropdowns()

                print(f"[Auto-load] Successfully loaded {len(loaded_models)} models from {models_path}")
                self.statusBar().showMessage(f"Auto-loaded {len(loaded_models)} ML models", 3000)
            else:
                print(f"[Auto-load] Failed to load any models from {models_path}")

        except Exception as e:
            print(f"[Auto-load] Error loading models: {e}")
            import traceback
            traceback.print_exc()

    def on_classifier_changed(self, text: str):
        """Handle classifier selection change from main window dropdown."""
        # Map display text to internal classifier name
        classifier_map = {
            "Threshold": "threshold",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp"
        }

        new_classifier = classifier_map.get(text, "threshold")

        # Update state (only print if actually changing)
        if self.state.active_classifier != new_classifier:
            self.state.active_classifier = new_classifier
            print(f"[Classifier] Switched to: {new_classifier}")

            # Check if ML models are loaded for this classifier
            if new_classifier != 'threshold':
                if not self.state.loaded_ml_models:
                    print(f"[Classifier] ERROR: No ML models loaded at all!")
                    self.statusBar().showMessage(f"WARNING: No ML models loaded. Load models from ML Training tab first.", 5000)
                    # Revert to threshold
                    self.peak_detec_combo.blockSignals(True)
                    self.peak_detec_combo.setCurrentText("Threshold")
                    self.peak_detec_combo.blockSignals(False)
                    self.state.active_classifier = "threshold"
                    print(f"[Classifier] Reverted to threshold (no models)")
                    return
                else:
                    # Find the model key (it may have accuracy suffix like "model1_xgboost_100%")
                    model_key_prefix = f'model1_{new_classifier}'
                    matching_keys = [k for k in self.state.loaded_ml_models.keys() if k.startswith(model_key_prefix)]

                    if not matching_keys:
                        print(f"[Classifier] ERROR: No model matching {model_key_prefix} found in loaded models!")
                        self.statusBar().showMessage(f"WARNING: {text} models not loaded. Load models from ML Training tab first.", 5000)
                        # Revert to threshold
                        self.peak_detec_combo.blockSignals(True)
                        self.peak_detec_combo.setCurrentText("Threshold")
                        self.peak_detec_combo.blockSignals(False)
                        self.state.active_classifier = "threshold"
                        print(f"[Classifier] Reverted to threshold (model not found)")
                        return
                    else:
                        print(f"[Classifier] Model {matching_keys[0]} found! Proceeding...")

            # Re-run peak detection to apply new classifier
            # This will recompute peaks_by_sweep using the new classifier's labels
            if self.state.analyze_chan and self.state.peaks_by_sweep:
                self.statusBar().showMessage(f"Switching to {text} classifier...", 2000)
                # Don't need to re-run full detection, just update which peaks are displayed
                self._update_displayed_peaks_from_classifier()
                # Guard: Only redraw if plot_manager exists (avoid error during initialization)
                if hasattr(self, 'plot_manager'):
                    self.redraw_main_plot()

    def _update_displayed_peaks_from_classifier(self):
        """Update peaks_by_sweep and breath_by_sweep based on active classifier.

        This copies the selected classifier's read-only predictions to the user-editable
        'labels' array, then updates the display.
        """
        st = self.state

        # For each sweep, copy active classifier's predictions to 'labels' (user-editable)
        for s in st.all_peaks_by_sweep.keys():
            all_peaks_data = st.all_peaks_by_sweep[s]

            # Get read-only predictions from active classifier
            active_labels_key_ro = f'labels_{st.active_classifier}_ro'
            if active_labels_key_ro in all_peaks_data and all_peaks_data[active_labels_key_ro] is not None:
                # Copy to user-editable array (this overwrites any manual edits!)
                all_peaks_data['labels'] = all_peaks_data[active_labels_key_ro].copy()
                # Reset label_source to 'auto' since these are fresh auto-predictions
                all_peaks_data['label_source'] = np.array(['auto'] * len(all_peaks_data['labels']))
                print(f"[Classifier Update] Sweep {s}: Copied {active_labels_key_ro} to 'labels', found {np.sum(all_peaks_data['labels'] == 1)} breaths")
            else:
                # Fallback to threshold if active classifier not available
                if 'labels_threshold_ro' in all_peaks_data and all_peaks_data['labels_threshold_ro'] is not None:
                    all_peaks_data['labels'] = all_peaks_data['labels_threshold_ro'].copy()
                    all_peaks_data['label_source'] = np.array(['auto'] * len(all_peaks_data['labels']))
                    print(f"[Classifier Update] Sweep {s}: Falling back to threshold, found {np.sum(all_peaks_data['labels'] == 1)} breaths")
                else:
                    print(f"[Classifier Update] Sweep {s}: WARNING - No predictions available!")
                    continue

            # Extract labeled peaks from user-editable 'labels' array
            labeled_mask = all_peaks_data['labels'] == 1
            labeled_indices = all_peaks_data['indices'][labeled_mask]
            st.peaks_by_sweep[s] = labeled_indices
            print(f"[Classifier Update] Sweep {s}: Updated peaks_by_sweep with {len(labeled_indices)} peaks")

            # Recompute breath events for labeled peaks
            y_proc = self._get_processed_for(st.analyze_chan, s)
            import core.peaks as peakdet
            breaths = peakdet.compute_breath_events(y_proc, labeled_indices, sr_hz=st.sr_hz, exclude_sec=0.030)
            st.breath_by_sweep[s] = breaths

    def on_eupnea_sniff_classifier_changed(self, text: str):
        """Handle eupnea/sniff classifier selection change."""
        classifier_map = {
            "GMM": "gmm",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp",
            "All Eupnea": "all_eupnea",
            "None (Clear)": "none"
        }
        # Reverse mapping for reverting
        classifier_reverse = {v: k for k, v in classifier_map.items()}

        new_classifier = classifier_map.get(text, "gmm")

        old_classifier = self.state.active_eupnea_sniff_classifier
        self.state.active_eupnea_sniff_classifier = new_classifier

        # Only print if actually switching
        if old_classifier != new_classifier:
            print(f"[Eupnea/Sniff Classifier] Switched to: {new_classifier}")

        # Handle different classifier types
        if new_classifier == 'all_eupnea':
            # Label all breaths as eupnea (class 0)
            self._set_all_breaths_eupnea_sniff_class(0)
            print(f"[Eupnea/Sniff Classifier] Set all breaths to eupnea (for anesthesia experiments)")
        elif new_classifier == 'none':
            # Clear all labels - don't save any eupnea/sniff classification
            self._clear_all_eupnea_sniff_labels()
            print(f"[Eupnea/Sniff Classifier] Cleared all eupnea/sniff labels")
        elif new_classifier == 'gmm':
            # Check if GMM has been run (gmm_class_ro should exist)
            first_sweep_peaks = self.state.all_peaks_by_sweep.get(0, {})
            if 'gmm_class_ro' not in first_sweep_peaks or first_sweep_peaks['gmm_class_ro'] is None:
                print(f"[Eupnea/Sniff Classifier] GMM clustering has not been run yet - running now with default settings...")
                self.statusBar().showMessage(f"Running GMM clustering with default settings...", 2000)

                # Run GMM clustering with default settings
                self._run_automatic_gmm_clustering()
                print(f"[Eupnea/Sniff Classifier] GMM clustering complete")

            # Update gmm_class from GMM predictions
            self._update_eupnea_sniff_from_classifier()
        else:
            # ML model selected - check if loaded
            model_key_prefix = f'model3_{new_classifier}'
            matching_keys = [k for k in self.state.loaded_ml_models.keys() if k.startswith(model_key_prefix)]

            if not matching_keys:
                print(f"[Eupnea/Sniff Classifier] ERROR: Model {model_key_prefix} not found!")
                self.statusBar().showMessage(f"WARNING: {text} model not loaded. Load Model 3 from ML Training tab.", 5000)
                # Revert to GMM
                self.eup_sniff_combo.blockSignals(True)
                self.eup_sniff_combo.setCurrentText("GMM")
                self.eup_sniff_combo.blockSignals(False)
                self.state.active_eupnea_sniff_classifier = "gmm"
                return

            # Update gmm_class from ML model predictions
            self._update_eupnea_sniff_from_classifier()

        # Rebuild sniff/eupnea regions for display
        try:
            import core.gmm_clustering as gmm_clustering
            gmm_clustering.build_eupnea_sniffing_regions(self.state, verbose=False)
        except Exception as e:
            print(f"[Eupnea/Sniff Classifier] Warning: Could not rebuild regions: {e}")
            import traceback
            traceback.print_exc()

        # Redraw plot
        # Guard: Only redraw if plot_manager exists (avoid error during initialization)
        if hasattr(self, 'plot_manager'):
            self.redraw_main_plot()

    def _update_eupnea_sniff_from_classifier(self):
        """Copy selected classifier's predictions to gmm_class array."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            # Get read-only predictions from active classifier
            if st.active_eupnea_sniff_classifier == 'gmm':
                source_key = 'gmm_class_ro'
            else:
                source_key = f'eupnea_sniff_{st.active_eupnea_sniff_classifier}_ro'

            # Debug: Check what keys are available
            if s == 0:
                available_keys = [k for k in all_peaks.keys() if 'eupnea' in k or 'gmm' in k]
                print(f"[Eupnea/Sniff Update] Sweep {s}: Available keys: {available_keys}")
                print(f"[Eupnea/Sniff Update] Sweep {s}: Looking for: {source_key}")

            if source_key in all_peaks and all_peaks[source_key] is not None:
                # Copy to user-editable array
                old_gmm_class = all_peaks['gmm_class'].copy() if 'gmm_class' in all_peaks else None
                all_peaks['gmm_class'] = all_peaks[source_key].copy()
                all_peaks['eupnea_sniff_source'] = np.array([st.active_eupnea_sniff_classifier] * len(all_peaks['indices']))

                # Debug: Show what changed
                if s == 0:
                    n_eupnea = np.sum(all_peaks['gmm_class'] == 0)
                    n_sniff = np.sum(all_peaks['gmm_class'] == 1)
                    n_unclass = np.sum(all_peaks['gmm_class'] == -1)
                    print(f"[Eupnea/Sniff Update] Sweep {s}: Copied {source_key} to gmm_class")
                    print(f"[Eupnea/Sniff Update] Sweep {s}: Eupnea: {n_eupnea}, Sniffing: {n_sniff}, Unclassified: {n_unclass}")
                    if old_gmm_class is not None:
                        n_changed = np.sum(old_gmm_class != all_peaks['gmm_class'])
                        print(f"[Eupnea/Sniff Update] Sweep {s}: Changed {n_changed} classifications")
            else:
                print(f"[Eupnea/Sniff Update] Sweep {s}: WARNING - No predictions available for {st.active_eupnea_sniff_classifier}!")
                if s == 0:
                    if source_key in all_peaks:
                        print(f"[Eupnea/Sniff Update] Sweep {s}: Key exists but is None")
                    else:
                        print(f"[Eupnea/Sniff Update] Sweep {s}: Key does not exist in all_peaks")

    def _set_all_breaths_eupnea_sniff_class(self, class_value: int):
        """
        Set all breaths to a specific eupnea/sniff class.

        Args:
            class_value: 0 = eupnea, 1 = sniffing, -1 = unclassified/other
        """
        import numpy as np
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if 'indices' in all_peaks and all_peaks['indices'] is not None:
                n_breaths = len(all_peaks['indices'])
                # Set all to the specified class
                all_peaks['gmm_class'] = np.full(n_breaths, class_value, dtype=int)
                all_peaks['eupnea_sniff_source'] = np.array([st.active_eupnea_sniff_classifier] * n_breaths)

                if s == 0:
                    print(f"[Eupnea/Sniff] Sweep {s}: Set {n_breaths} breaths to class {class_value}")

    def _clear_all_eupnea_sniff_labels(self):
        """
        Clear all eupnea/sniff labels - used when user selects 'None (Clear)'.

        This removes the gmm_class array entirely so labels won't be saved.
        Also clears the sniff and eupnea regions from the display.
        """
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            # Remove gmm_class array - labels won't be saved
            if 'gmm_class' in all_peaks:
                del all_peaks['gmm_class']
            if 'eupnea_sniff_source' in all_peaks:
                del all_peaks['eupnea_sniff_source']

            if s == 0:
                print(f"[Eupnea/Sniff] Sweep {s}: Cleared all labels")

        # Clear the region dictionaries for display
        st.sniff_regions_by_sweep.clear()
        st.eupnea_regions_by_sweep.clear()

    def _clear_all_sigh_labels(self):
        """
        Clear all sigh labels - used when user selects 'None (Clear)'.

        This removes the sigh_class array entirely so labels won't be saved.
        Also clears sigh_by_sweep for display.
        """
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            # Remove sigh_class array - labels won't be saved
            if 'sigh_class' in all_peaks:
                del all_peaks['sigh_class']
            if 'sigh_source' in all_peaks:
                del all_peaks['sigh_source']

            if s == 0:
                print(f"[Sigh] Sweep {s}: Cleared all labels")

        # Clear sigh_by_sweep for display
        st.sigh_by_sweep.clear()

    def on_sigh_classifier_changed(self, text: str):
        """Handle sigh classifier selection change."""
        classifier_map = {
            "Manual": "manual",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp",
            "None (Clear)": "none"
        }
        # Reverse mapping for reverting
        classifier_reverse = {v: k for k, v in classifier_map.items()}

        new_classifier = classifier_map.get(text, "manual")

        old_classifier = self.state.active_sigh_classifier
        self.state.active_sigh_classifier = new_classifier

        # Only print if actually switching
        if old_classifier != new_classifier:
            print(f"[Sigh Classifier] Switched to: {new_classifier}")

        # Handle different classifier types
        if new_classifier == 'none':
            # Clear all sigh labels
            self._clear_all_sigh_labels()
            print(f"[Sigh Classifier] Cleared all sigh labels")
        elif new_classifier == 'manual':
            # Update from manual annotations
            self._update_sigh_from_classifier()
        else:
            # ML model - check if loaded
            model_key_prefix = f'model2_{new_classifier}'
            matching_keys = [k for k in self.state.loaded_ml_models.keys() if k.startswith(model_key_prefix)]

            if not matching_keys:
                print(f"[Sigh Classifier] ERROR: Model {model_key_prefix} not found!")
                self.statusBar().showMessage(f"WARNING: {text} model not loaded. Load Model 2 from ML Training tab.", 5000)
                # Revert to Manual
                self.digh_combo.blockSignals(True)
                self.digh_combo.setCurrentText("Manual")
                self.digh_combo.blockSignals(False)
                self.state.active_sigh_classifier = "manual"
                return

            # Update from ML model predictions
            self._update_sigh_from_classifier()

        # Redraw plot
        # Guard: Only redraw if plot_manager exists (avoid error during initialization)
        if hasattr(self, 'plot_manager'):
            self.redraw_main_plot()

    def _update_sigh_from_classifier(self):
        """Copy selected classifier's predictions to sigh_class array and update sigh_by_sweep."""
        import numpy as np
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            # Get read-only predictions from active classifier
            if st.active_sigh_classifier == 'manual':
                # For manual mode, restore from sigh_manual_ro (preserves only manual annotations)
                # This clears ML predictions and only shows manually-added sighs
                if 'sigh_manual_ro' in all_peaks and all_peaks['sigh_manual_ro'] is not None:
                    all_peaks['sigh_class'] = all_peaks['sigh_manual_ro'].copy()
                    all_peaks['sigh_source'] = np.array(['manual'] * len(all_peaks['indices']))
                    if s == 0:
                        n_sighs = np.sum(all_peaks['sigh_class'] == 1)
                        print(f"[Sigh Update] Sweep {s}: Restored manual annotations from sigh_manual_ro ({n_sighs} sighs)")
                else:
                    # No manual annotations saved - clear all sighs (start fresh)
                    n_breaths = len(all_peaks['indices']) if 'indices' in all_peaks else 0
                    if n_breaths > 0:
                        all_peaks['sigh_class'] = np.zeros(n_breaths, dtype=np.int8)
                        # Mark deleted peaks as unclassified
                        if 'labels' in all_peaks:
                            all_peaks['sigh_class'][all_peaks['labels'] == 0] = -1
                        all_peaks['sigh_source'] = np.array(['manual'] * n_breaths)
                    if s == 0:
                        print(f"[Sigh Update] Sweep {s}: Cleared all sighs (manual mode, no saved annotations)")

                # Also clear sigh_by_sweep for this sweep
                if s in st.sigh_by_sweep:
                    # Only keep sighs that are in sigh_manual_ro
                    if 'sigh_manual_ro' in all_peaks and all_peaks['sigh_manual_ro'] is not None:
                        manual_sigh_mask = all_peaks['sigh_manual_ro'] == 1
                        st.sigh_by_sweep[s] = all_peaks['indices'][manual_sigh_mask].tolist()
                    else:
                        st.sigh_by_sweep[s] = []
                continue
            else:
                source_key = f'sigh_{st.active_sigh_classifier}_ro'

            # Debug: Check what keys are available
            if s == 0 and st.active_sigh_classifier != 'manual':
                available_keys = [k for k in all_peaks.keys() if 'sigh' in k]
                print(f"[Sigh Update] Sweep {s}: Available keys: {available_keys}")
                print(f"[Sigh Update] Sweep {s}: Looking for: {source_key}")

            if st.active_sigh_classifier != 'manual' and source_key in all_peaks and all_peaks[source_key] is not None:
                # Copy to user-editable array
                old_sigh_class = all_peaks['sigh_class'].copy() if 'sigh_class' in all_peaks else None
                all_peaks['sigh_class'] = all_peaks[source_key].copy()
                all_peaks['sigh_source'] = np.array([st.active_sigh_classifier] * len(all_peaks['indices']))

                # Debug: Show what changed
                if s == 0:
                    n_normal = np.sum(all_peaks['sigh_class'] == 0)
                    n_sigh = np.sum(all_peaks['sigh_class'] == 1)
                    n_unclass = np.sum(all_peaks['sigh_class'] == -1)
                    print(f"[Sigh Update] Sweep {s}: Copied {source_key} to sigh_class")
                    print(f"[Sigh Update] Sweep {s}: Normal: {n_normal}, Sigh: {n_sigh}, Unclassified: {n_unclass}")
                    if old_sigh_class is not None:
                        n_changed = np.sum(old_sigh_class != all_peaks['sigh_class'])
                        print(f"[Sigh Update] Sweep {s}: Changed {n_changed} classifications")
            elif st.active_sigh_classifier != 'manual':
                print(f"[Sigh Update] Sweep {s}: WARNING - No predictions available for {st.active_sigh_classifier}!")

            # Update sigh_by_sweep from sigh_class for display (backward compatibility)
            if 'sigh_class' in all_peaks:
                sigh_mask = all_peaks['sigh_class'] == 1
                st.sigh_by_sweep[s] = all_peaks['indices'][sigh_mask]

    def on_mark_events_clicked(self):
        """Open Event Detection Settings dialog."""
        # Check if event channel is selected
        st = self.state
        if st.event_channel is None:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Event Channel",
                "Please select an event channel from the 'Events Chan Select' dropdown first."
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


        # Peaks/breaths no longer valid if filters change
        if hasattr(self.state, "peaks_by_sweep"):
            self.state.peaks_by_sweep.clear()
        if hasattr(self.state, "breath_by_sweep"):
            self.state.breath_by_sweep.clear()

        # Peaks/breaths/y2 no longer valid if filters change
        if hasattr(self.state, "peaks_by_sweep"):
            self.state.peaks_by_sweep.clear()
        if hasattr(self.state, "breath_by_sweep"):
            self.state.breath_by_sweep.clear()
            self.state.y2_values_by_sweep.clear()
            self.plot_host.clear_y2()

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
        """Return processed y for (channel, sweep_idx) using the same cache key logic."""
        st = self.state
        Y = st.sweeps[chan]
        s = max(0, min(sweep_idx, Y.shape[1]-1))
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
        if tab:
            self._analysis_options_dialog.set_active_tab(tab)

    def _on_analysis_options_closed(self):
        """Clear analysis options dialog reference when closed so it can be reopened."""
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
            min_dist_samples = int(self.peak_min_dist * st.sr_hz)
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
                min_dist_samples = int(self.peak_min_dist * st.sr_hz)
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
        """
        Pre-compute remaining ML classifiers in the background after UI is responsive.
        This enables instant classifier switching without slowing down initial display.
        """
        from PyQt6.QtCore import QTimer
        import core.ml_prediction as ml_prediction
        import time

        st = self.state

        # Determine which algorithms still need to be computed
        already_computed = set()
        need_to_compute = set()

        # Check what was already run
        if st.active_classifier in ['xgboost', 'rf', 'mlp']:
            already_computed.add(st.active_classifier)
        if st.active_eupnea_sniff_classifier in ['xgboost', 'rf', 'mlp']:
            already_computed.add(st.active_eupnea_sniff_classifier)
        if st.active_sigh_classifier in ['xgboost', 'rf', 'mlp']:
            already_computed.add(st.active_sigh_classifier)

        # Determine what still needs to be computed
        all_algorithms = {'xgboost', 'rf', 'mlp'}
        need_to_compute = all_algorithms - already_computed

        if not need_to_compute:
            print("[Background] No additional classifiers to pre-compute")
            return

        print(f"[Background] Will pre-compute {sorted(need_to_compute)} classifiers in background...")

        def compute_remaining():
            """Background worker function."""
            t_start = time.time()
            total_computed = 0

            try:
                for s in st.all_peaks_by_sweep.keys():
                    all_peaks = st.all_peaks_by_sweep[s]

                    # Get peak metrics for this sweep
                    if s not in st.peak_metrics_by_sweep:
                        continue
                    peak_metrics = st.peak_metrics_by_sweep[s]

                    # Run predictions for remaining algorithms
                    for algorithm in sorted(need_to_compute):
                        try:
                            predictions = ml_prediction.predict_with_cascade(
                                peak_metrics=peak_metrics,
                                models=st.loaded_ml_models,
                                algorithm=algorithm
                            )

                            # Store predictions as read-only (for classifier switching)
                            all_peaks[f'labels_{algorithm}_ro'] = predictions['final_labels']

                            # Store eupnea/sniff predictions (Model 3 output)
                            if 'eupnea_sniff_class' in predictions:
                                all_peaks[f'eupnea_sniff_{algorithm}_ro'] = predictions['eupnea_sniff_class']

                            # Store sigh predictions (Model 2 output)
                            if 'sigh_class' in predictions:
                                all_peaks[f'sigh_{algorithm}_ro'] = predictions['sigh_class']

                            total_computed += 1

                        except KeyError:
                            if s == 0:
                                print(f"[Background] Model {algorithm} not found, skipping")
                        except Exception as e:
                            print(f"[Background] Warning: {algorithm} prediction failed: {e}")

                t_elapsed = time.time() - t_start
                print(f"[Background] Pre-computed {len(need_to_compute)} classifiers in {t_elapsed:.1f}s")
                print(f"[Background] Classifier switching is now instant!")

            except Exception as e:
                print(f"[Background] Error during pre-computation: {e}")
                import traceback
                traceback.print_exc()

        # Schedule background computation with 100ms delay (let UI settle first)
        QTimer.singleShot(100, compute_remaining)

    def _apply_peak_detection(self):
        """
        Run peak detection on the ANALYZE channel for ALL sweeps,
        store indices per sweep, and redraw current sweep with peaks + breath markers.
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


        for s in range(n_sweeps):
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
                        predictions = ml_prediction.predict_with_cascade(
                            peak_metrics=peak_metrics,
                            models=st.loaded_ml_models,
                            algorithm=algorithm
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
            if active_labels_key_ro in all_peaks_data and all_peaks_data[active_labels_key_ro] is not None:
                # Copy from selected classifier's read-only predictions
                all_peaks_data['labels'] = all_peaks_data[active_labels_key_ro].copy()
                if s == 0:
                    print(f"[peak-detection] Initialized 'labels' from {active_labels_key_ro}")
            else:
                # Fallback to threshold if active classifier not available
                # (labels already contains threshold predictions from step 3)
                if s == 0:
                    print(f"[peak-detection] Using threshold labels (active classifier {st.active_classifier} not available)")

            # Step 4b: Initialize user-editable 'gmm_class' array from active eupnea/sniff classifier
            if st.active_eupnea_sniff_classifier == 'gmm':
                # GMM will be computed on-demand later, initialize as None for now
                all_peaks_data['gmm_class'] = None
                if s == 0:
                    print(f"[peak-detection] Initialized 'gmm_class' as None (GMM will run on-demand)")
            else:
                # Copy from selected ML classifier's read-only predictions
                active_eup_sniff_key_ro = f'eupnea_sniff_{st.active_eupnea_sniff_classifier}_ro'
                if active_eup_sniff_key_ro in all_peaks_data and all_peaks_data[active_eup_sniff_key_ro] is not None:
                    all_peaks_data['gmm_class'] = all_peaks_data[active_eup_sniff_key_ro].copy()
                    if s == 0:
                        print(f"[peak-detection] Initialized 'gmm_class' from {active_eup_sniff_key_ro}")
                else:
                    # Fallback to None
                    all_peaks_data['gmm_class'] = None
                    if s == 0:
                        print(f"[peak-detection] 'gmm_class' initialized as None (classifier {st.active_eupnea_sniff_classifier} not available)")

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
            # Use ML classifier predictions (already initialized in gmm_class array)
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
        if hasattr(self, '_analysis_options_dialog') and self._analysis_options_dialog.isVisible():
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
        """
        Compute eupnea mask from GMM clustering results.

        Eupnea = breaths that are NOT sniffing (based on GMM classification).
        Groups consecutive eupnic breaths into continuous regions.

        Args:
            sweep_idx: Index of the current sweep
            signal_length: Length of the signal array

        Returns:
            Boolean array (as float 0/1) marking eupneic regions
        """
        import numpy as np

        eupnea_mask = np.zeros(signal_length, dtype=bool)

        # Check if GMM probabilities are available for this sweep
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            return eupnea_mask.astype(float)

        if sweep_idx not in self.state.gmm_sniff_probabilities:
            return eupnea_mask.astype(float)

        # Get breath data for this sweep
        breath_data = self.state.breath_by_sweep.get(sweep_idx)
        if breath_data is None:
            return eupnea_mask.astype(float)

        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))

        if len(onsets) == 0:
            return eupnea_mask.astype(float)

        t = self.state.t
        gmm_probs = self.state.gmm_sniff_probabilities[sweep_idx]

        # Identify eupnic breaths and group consecutive ones
        eupnic_groups = []
        current_group_start = None
        current_group_end = None
        last_eupnic_idx = None

        # Get peaks for this sweep (needed to map breath → peak)
        peaks = self.state.peaks_by_sweep.get(sweep_idx)
        if peaks is None or len(peaks) != len(onsets):
            return eupnea_mask.astype(float)  # Peaks and onsets must be aligned

        for breath_idx in range(len(onsets)):
            # Get peak sample index for this breath
            # Peaks and breaths are 1:1 aligned by array index
            peak_sample_idx = int(peaks[breath_idx])

            # Look up GMM probability using peak sample index
            if peak_sample_idx not in gmm_probs:
                # Close current group if exists
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = None
                    current_group_end = None
                    last_eupnic_idx = None
                continue

            sniff_prob = gmm_probs[peak_sample_idx]

            # Eupnea if sniffing probability < 0.5 (i.e., more likely eupnea)
            if sniff_prob < 0.5:
                # Get time range for this breath
                start_idx = int(onsets[breath_idx])

                # Get offset time
                if breath_idx < len(offsets):
                    end_idx = int(offsets[breath_idx])
                else:
                    # Fallback: use next onset or end of trace
                    if breath_idx + 1 < len(onsets):
                        end_idx = int(onsets[breath_idx + 1])
                    else:
                        end_idx = signal_length

                # Check if this is consecutive with the last eupnic breath
                if last_eupnic_idx is None or breath_idx != last_eupnic_idx + 1:
                    # Not consecutive - save current group and start new one
                    if current_group_start is not None:
                        eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = start_idx
                    current_group_end = end_idx
                else:
                    # Consecutive breath - extend the current group
                    current_group_end = end_idx

                last_eupnic_idx = breath_idx
            else:
                # Non-eupnic breath - close current group if exists
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = None
                    current_group_end = None
                    last_eupnic_idx = None

        # Save final group if exists
        if current_group_start is not None:
            eupnic_groups.append((current_group_start, current_group_end))

        # Mark all continuous eupnic regions
        for start_idx, end_idx in eupnic_groups:
            eupnea_mask[start_idx:end_idx] = True

        return eupnea_mask.astype(float)

    def _run_automatic_gmm_clustering(self):
        """
        Automatically run GMM clustering after peak detection to identify sniffing breaths.
        Uses streamlined default features (if, ti, amp_insp, max_dinsp) and 2 clusters.
        Silently marks identified sniffing breaths with purple background.
        """
        import time
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        import numpy as np

        t_start = time.time()
        st = self.state

        # Check if we have breath data
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            print("[auto-gmm] No breath data available, skipping automatic GMM clustering")
            return

        # Streamlined default features for eupnea/sniffing separation
        feature_keys = ["if", "ti", "amp_insp", "max_dinsp"]
        n_clusters = 2

        print(f"\n[auto-gmm] Running automatic GMM clustering with {n_clusters} clusters...")
        print(f"[auto-gmm] Features: {', '.join(feature_keys)}")
        self._log_status_message("Running GMM clustering...")

        try:
            # Collect breath features from all analyzed sweeps
            feature_matrix, breath_cycles = self._collect_gmm_breath_features(feature_keys)

            if len(feature_matrix) < n_clusters:
                print(f"[auto-gmm] Not enough breaths ({len(feature_matrix)}) for {n_clusters} clusters, skipping")
                return

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Fit GMM
            gmm_model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            cluster_labels = gmm_model.fit_predict(feature_matrix_scaled)

            # Get probability estimates for each breath
            cluster_probabilities = gmm_model.predict_proba(feature_matrix_scaled)

            # Check clustering quality
            silhouette = silhouette_score(feature_matrix_scaled, cluster_labels) if n_clusters > 1 else -1
            print(f"[auto-gmm] Silhouette score: {silhouette:.3f}")

            # Identify sniffing cluster
            sniffing_cluster_id = self._identify_gmm_sniffing_cluster(
                feature_matrix, cluster_labels, feature_keys, silhouette
            )

            if sniffing_cluster_id is None:
                print("[auto-gmm] Could not identify sniffing cluster, skipping")
                return

            # Apply GMM sniffing regions to plot (stores probabilities AND creates regions)
            self._apply_gmm_sniffing_regions(
                breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id
            )

            n_sniffing_breaths = np.sum(cluster_labels == sniffing_cluster_id)
            print(f"[auto-gmm] Identified {n_sniffing_breaths} sniffing breaths and applied to plot")

            # Cache results for fast dialog loading
            self._cached_gmm_results = {
                'cluster_labels': cluster_labels,
                'cluster_probabilities': cluster_probabilities,
                'feature_matrix': feature_matrix,
                'breath_cycles': breath_cycles,
                'sniffing_cluster_id': sniffing_cluster_id,
                'feature_keys': feature_keys
            }
            print("[auto-gmm] Cached GMM results for fast dialog loading")

            # Show completion message with elapsed time
            t_elapsed = time.time() - t_start

            # Log telemetry: GMM clustering success
            eupnea_count = len(cluster_labels) - n_sniffing_breaths
            telemetry.log_feature_used('gmm_clustering')
            telemetry.log_timing('gmm_clustering', t_elapsed,
                                num_breaths=len(cluster_labels),
                                num_clusters=n_clusters,
                                silhouette_score=round(silhouette, 3))

            telemetry.log_breath_statistics(
                num_breaths=len(cluster_labels),
                sniff_count=int(n_sniffing_breaths),
                eupnea_count=int(eupnea_count),
                silhouette_score=round(silhouette, 3)
            )

            self._log_status_message(f"GMM clustering complete ({t_elapsed:.1f}s)", 2000)

        except Exception as e:
            print(f"[auto-gmm] Error during automatic GMM clustering: {e}")
            t_elapsed = time.time() - t_start

            # Log telemetry: GMM clustering failure
            telemetry.log_crash(f"GMM clustering failed: {type(e).__name__}",
                               operation='gmm_clustering',
                               num_breaths=len(feature_matrix) if 'feature_matrix' in locals() else 0)

            self._log_status_message(f"GMM clustering failed ({t_elapsed:.1f}s)", 3000)
            import traceback
            traceback.print_exc()

    def _collect_gmm_breath_features(self, feature_keys):
        """Collect per-breath features for GMM clustering."""
        import numpy as np
        from core import metrics, filters

        feature_matrix = []
        breath_cycles = []
        st = self.state

        for sweep_idx in sorted(st.breath_by_sweep.keys()):
            breath_data = st.breath_by_sweep[sweep_idx]

            if sweep_idx not in st.peaks_by_sweep:
                continue

            peaks = st.peaks_by_sweep[sweep_idx]
            t = st.t
            y_raw = st.sweeps[st.analyze_chan][:, sweep_idx]

            # Apply filters
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=self.filter_order
            )

            # Apply notch filter if configured
            if self.notch_filter_lower is not None and self.notch_filter_upper is not None:
                y = self._apply_notch_filter(y, st.sr_hz,
                                              self.notch_filter_lower,
                                              self.notch_filter_upper)

            # Apply z-score normalization if enabled (using global statistics)
            if self.use_zscore_normalization:
                # Compute global stats if not cached
                if self.zscore_global_mean is None or self.zscore_global_std is None:
                    self.zscore_global_mean, self.zscore_global_std = self._compute_global_zscore_stats()
                y = filters.zscore_normalize(y, self.zscore_global_mean, self.zscore_global_std)

            # Get breath events
            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            expmins = breath_data.get('expmins', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            if len(onsets) == 0:
                continue

            # Compute metrics
            metrics_dict = {}
            for feature_key in feature_keys:
                if feature_key in metrics.METRICS:
                    metric_arr = metrics.METRICS[feature_key](
                        t, y, st.sr_hz, peaks, onsets, offsets, expmins, expoffs
                    )
                    metrics_dict[feature_key] = metric_arr

            # Extract per-breath values
            n_breaths = len(onsets)
            for breath_idx in range(n_breaths):
                start = int(onsets[breath_idx])
                breath_features = []
                valid_breath = True

                for feature_key in feature_keys:
                    if feature_key not in metrics_dict:
                        valid_breath = False
                        break

                    metric_arr = metrics_dict[feature_key]
                    if start < len(metric_arr):
                        val = metric_arr[start]
                        if np.isnan(val) or not np.isfinite(val):
                            valid_breath = False
                            break
                        breath_features.append(val)
                    else:
                        valid_breath = False
                        break

                if valid_breath and len(breath_features) == len(feature_keys):
                    feature_matrix.append(breath_features)
                    breath_cycles.append((sweep_idx, breath_idx))

        return np.array(feature_matrix), breath_cycles

    def _identify_gmm_sniffing_cluster(self, feature_matrix, cluster_labels, feature_keys, silhouette):
        """Identify which cluster represents sniffing based on IF and Ti."""
        import numpy as np

        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # Get indices of IF and Ti features
        if_idx = feature_keys.index('if') if 'if' in feature_keys else None
        ti_idx = feature_keys.index('ti') if 'ti' in feature_keys else None

        if if_idx is None and ti_idx is None:
            print("[auto-gmm] Cannot identify sniffing without 'if' or 'ti' features")
            return None

        # Compute mean IF and Ti for each cluster
        cluster_stats = {}
        for cluster_id in unique_labels:
            mask = cluster_labels == cluster_id
            stats = {}
            if if_idx is not None:
                stats['mean_if'] = np.mean(feature_matrix[mask, if_idx])
            if ti_idx is not None:
                stats['mean_ti'] = np.mean(feature_matrix[mask, ti_idx])
            cluster_stats[cluster_id] = stats

        # Identify sniffing: highest IF and/or lowest Ti
        cluster_scores = {}
        for cluster_id in unique_labels:
            score = 0
            if if_idx is not None:
                if_vals = [cluster_stats[c]['mean_if'] for c in unique_labels]
                if_rank = sorted(if_vals).index(cluster_stats[cluster_id]['mean_if'])
                score += if_rank / (n_clusters - 1) if n_clusters > 1 else 0
            if ti_idx is not None:
                ti_vals = [cluster_stats[c]['mean_ti'] for c in unique_labels]
                ti_rank = sorted(ti_vals, reverse=True).index(cluster_stats[cluster_id]['mean_ti'])
                score += ti_rank / (n_clusters - 1) if n_clusters > 1 else 0
            cluster_scores[cluster_id] = score

        sniffing_cluster_id = max(cluster_scores, key=cluster_scores.get)

        # Log cluster statistics
        for cluster_id in unique_labels:
            stats_str = ", ".join([f"{k}={v:.3f}" for k, v in cluster_stats[cluster_id].items()])
            marker = " (SNIFFING)" if cluster_id == sniffing_cluster_id else ""
            print(f"[auto-gmm]   Cluster {cluster_id}: {stats_str}{marker}")

        # Validate quality (warn but don't block)
        sniff_stats = cluster_stats[sniffing_cluster_id]
        if silhouette < 0.25:
            print(f"[auto-gmm] WARNING: Low cluster separation (silhouette={silhouette:.3f})")
            print(f"[auto-gmm]   Breathing patterns may be very similar (e.g., anesthetized mouse)")
        if if_idx is not None and sniff_stats['mean_if'] < 5.0:
            print(f"[auto-gmm] WARNING: 'Sniffing' cluster has low IF ({sniff_stats['mean_if']:.2f} Hz)")
            print(f"[auto-gmm]   May be normal variation, not true sniffing (typical sniffing: 5-8 Hz)")

        return sniffing_cluster_id

    def _apply_gmm_sniffing_regions(self, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id):
        """Apply GMM cluster results using shared functions from core.gmm_clustering.

        Stores classifications in all_peaks_by_sweep and builds both eupnea and sniffing regions.
        Stores probabilities for each breath (for backward compatibility).

        Args:
            breath_cycles: List of (sweep_idx, breath_idx) tuples
            cluster_labels: Hard cluster assignments
            cluster_probabilities: Probability matrix (n_breaths, n_clusters)
            sniffing_cluster_id: Which cluster is sniffing
        """
        import numpy as np
        from core import gmm_clustering

        # Store probabilities by (sweep_idx, breath_idx) for backward compatibility
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            self.state.gmm_sniff_probabilities = {}
        self.state.gmm_sniff_probabilities.clear()

        for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
            # Get probability of this breath being sniffing
            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]

            # Store probability
            if sweep_idx not in self.state.gmm_sniff_probabilities:
                self.state.gmm_sniff_probabilities[sweep_idx] = {}
            self.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Check if GMM is the active classifier
        is_gmm_active = self.state.active_eupnea_sniff_classifier == 'gmm'

        # Store classifications in all_peaks_by_sweep
        # - Always store to gmm_class_ro (for switching classifiers later)
        # - Only update gmm_class (editable) if GMM is the active classifier
        n_classified = gmm_clustering.store_gmm_classifications_in_peaks(
            self.state, breath_cycles, cluster_labels, sniffing_cluster_id,
            cluster_probabilities, confidence_threshold=0.5,
            update_editable=is_gmm_active
        )

        # Only build regions if GMM is the active classifier
        # (otherwise, the active classifier's predictions are already in gmm_class)
        if is_gmm_active:
            # Build BOTH eupnea AND sniffing regions from stored classifications
            results = gmm_clustering.build_eupnea_sniffing_regions(
                self.state, verbose=False, log_prefix="[auto-gmm]"
            )
        else:
            # Just return dummy results - regions will be built from ML predictions
            results = {'n_sniffing': 0, 'n_eupnea': 0, 'total_sniff_regions': 0, 'total_eupnea_regions': 0}
            print(f"[auto-gmm] GMM results cached but not applied (active classifier: {self.state.active_eupnea_sniff_classifier})")

        # Calculate probability statistics
        all_sniff_probs = []
        for sweep_idx in self.state.gmm_sniff_probabilities:
            for breath_idx in self.state.gmm_sniff_probabilities[sweep_idx]:
                prob = self.state.gmm_sniff_probabilities[sweep_idx][breath_idx]
                all_sniff_probs.append(prob)

        if all_sniff_probs:
            all_sniff_probs = np.array(all_sniff_probs)
            sniff_probs_of_sniff_breaths = all_sniff_probs[all_sniff_probs >= 0.5]  # Breaths classified as sniffing
            if len(sniff_probs_of_sniff_breaths) > 0:
                mean_conf = np.mean(sniff_probs_of_sniff_breaths)
                min_conf = np.min(sniff_probs_of_sniff_breaths)
                uncertain_count = np.sum((sniff_probs_of_sniff_breaths >= 0.5) & (sniff_probs_of_sniff_breaths < 0.7))
                print(f"[auto-gmm]   Sniffing probability: mean={mean_conf:.3f}, min={min_conf:.3f}")
                if uncertain_count > 0:
                    print(f"[auto-gmm]   WARNING: {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")

        # Report results
        print(f"[auto-gmm]   Created {results['total_sniff_regions']} sniffing region(s) across sweeps")
        print(f"[auto-gmm]   Created {results['total_eupnea_regions']} eupnea region(s) across sweeps")

        return results['n_sniffing']

    def _store_gmm_probabilities_only(self, breath_cycles, cluster_probabilities, sniffing_cluster_id):
        """Store GMM sniffing probabilities without applying regions to plot.

        This is used by automatic GMM clustering to store results without
        immediately marking sniffing regions. User must manually enable
        "Apply Sniffing Detection" in GMM dialog to see markings.

        Args:
            breath_cycles: List of (sweep_idx, breath_idx) tuples
            cluster_probabilities: Probability matrix (n_breaths, n_clusters)
            sniffing_cluster_id: Which cluster is sniffing
        """
        import numpy as np

        # Store probabilities by (sweep_idx, breath_idx)
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            self.state.gmm_sniff_probabilities = {}
        self.state.gmm_sniff_probabilities.clear()

        for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
            if sweep_idx not in self.state.gmm_sniff_probabilities:
                self.state.gmm_sniff_probabilities[sweep_idx] = {}

            # Get probability of this breath being sniffing
            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]
            self.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Calculate probability statistics
        all_sniff_probs = []
        for sweep_idx in self.state.gmm_sniff_probabilities:
            for breath_idx in self.state.gmm_sniff_probabilities[sweep_idx]:
                prob = self.state.gmm_sniff_probabilities[sweep_idx][breath_idx]
                all_sniff_probs.append(prob)

        if all_sniff_probs:
            all_sniff_probs = np.array(all_sniff_probs)
            sniff_probs_of_sniff_breaths = all_sniff_probs[all_sniff_probs >= 0.5]  # Breaths classified as sniffing
            if len(sniff_probs_of_sniff_breaths) > 0:
                mean_conf = np.mean(sniff_probs_of_sniff_breaths)
                min_conf = np.min(sniff_probs_of_sniff_breaths)
                uncertain_count = np.sum((sniff_probs_of_sniff_breaths >= 0.5) & (sniff_probs_of_sniff_breaths < 0.7))
                print(f"[auto-gmm]   Sniffing probability: mean={mean_conf:.3f}, min={min_conf:.3f}")
                if uncertain_count > 0:
                    print(f"[auto-gmm]   WARNING: {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")

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

        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            # pull peaks/breaths if available
            pks = getattr(st, "peaks_by_sweep", {}).get(s, None)
            # breaths = getattr(st, "breath_by_sweep", {}).get(s, {}) if hasattr(st, "breath_by_sweep") else {}
            breaths = getattr(st, "breath_by_sweep", {}).get(s, {})
            on = breaths.get("onsets", None)
            off = breaths.get("offsets", None)
            exm = breaths.get("expmins", None)
            exo = breaths.get("expoffs", None)

            # Set GMM probabilities for this sweep (if available)
            gmm_probs = None
            if hasattr(st, 'gmm_sniff_probabilities') and s in st.gmm_sniff_probabilities:
                gmm_probs = st.gmm_sniff_probabilities[s]
            metrics.set_gmm_probabilities(gmm_probs)

            # Set peak candidate metrics for this sweep (if available)
            # Prefer current_peak_metrics_by_sweep (updated after edits) for Y2 plotting,
            # fallback to peak_metrics_by_sweep (original auto-detected) for ML training
            peak_metrics = None
            if hasattr(st, 'current_peak_metrics_by_sweep') and s in st.current_peak_metrics_by_sweep:
                peak_metrics = st.current_peak_metrics_by_sweep[s]
            elif hasattr(st, 'peak_metrics_by_sweep') and s in st.peak_metrics_by_sweep:
                peak_metrics = st.peak_metrics_by_sweep[s]
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

        # Compute ONLY eupnea mask from GMM (fast, no metrics recomputation)
        eupnea_mask = self._compute_eupnea_from_gmm(s, len(y))

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
            if update_info:
                # Store for help dialog
                self.update_info = update_info

                # Update main window label
                from core import update_checker
                text, url = update_checker.get_main_window_update_message(update_info)
                self.update_notification_label.setText(f'<a href="{url}" style="color: #FFD700; text-decoration: underline;">{text}</a>')
                self.update_notification_label.setVisible(True)
                print(f"[Update Check] New version available: {update_info.get('version')}")
            else:
                # No update available - keep label hidden
                print("[Update Check] You're up to date!")

        # Create and start background thread
        self.update_thread = UpdateChecker()
        self.update_thread.update_checked.connect(on_update_checked)
        self.update_thread.start()

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
        # Always open dialog - it will handle the "no peaks" case internally
        dlg = AdvancedPeakEditorDialog(main_window=self, parent=self)
        telemetry.log_screen_view('Advanced Peak Editor Dialog', screen_class='curation_dialog')
        dlg.exec()  # Modal dialog
        telemetry.log_feature_used('advanced_peak_editor')

        # Refresh plot after dialog closes (in case user made edits)
        if hasattr(self.state, 'all_peaks_by_sweep') and self.state.all_peaks_by_sweep:
            self.redraw_main_plot()

    def _refresh_omit_button_label(self):
        """Update Omit button text based on whether current sweep is omitted."""
        s = max(0, min(self.state.sweep_idx, self.navigation_manager._sweep_count() - 1))
        if s in self.state.omitted_sweeps:
            self.OmitSweepButton.setText("Un-omit")
            self.OmitSweepButton.setToolTip("This sweep will be excluded from saving and stats.")
        else:
            self.OmitSweepButton.setText("Omit")
            self.OmitSweepButton.setToolTip("Mark this sweep to be excluded from saving and stats.")

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
        """Set up the master file list by hiding Project Organization and reconfiguring table."""
        from PyQt6.QtWidgets import QHeaderView, QComboBox, QPushButton, QHBoxLayout, QWidget
        from PyQt6.QtCore import Qt

        # === Hide the Project Organization section (right column) ===
        # Hide all widgets in the right column
        if hasattr(self, 'projectOrganizationLabel'):
            self.projectOrganizationLabel.hide()
        if hasattr(self, 'experimentsTreeWidget'):
            self.experimentsTreeWidget.hide()
        if hasattr(self, 'experimentDetailsGroup'):
            self.experimentDetailsGroup.hide()
        if hasattr(self, 'removeExperimentButton'):
            self.removeExperimentButton.hide()
        if hasattr(self, 'exportExperimentButton'):
            self.exportExperimentButton.hide()
        if hasattr(self, 'addExperimentButton'):
            self.addExperimentButton.hide()
        if hasattr(self, 'experimentsInProjectLabel'):
            self.experimentsInProjectLabel.hide()

        # Hide the vertical line separator between columns
        if hasattr(self, 'line_vertical'):
            self.line_vertical.hide()

        # Hide the "Add to Project" button since we're not using experiments
        if hasattr(self, 'addToProjectButton'):
            self.addToProjectButton.hide()

        # Make the left column stretch to fill available space
        # Find the two-column horizontal layout and adjust stretch factors
        from PyQt6.QtWidgets import QHBoxLayout
        two_col_layout = self.findChild(QHBoxLayout, 'projectTwoColumnLayout')
        if two_col_layout:
            # Set stretch factors: left column gets all space, right column gets none
            two_col_layout.setStretch(0, 1)  # leftColumnLayout
            two_col_layout.setStretch(1, 0)  # line_vertical (hidden)
            two_col_layout.setStretch(2, 0)  # rightColumnLayout (hidden)

        # === Reconfigure the table columns ===
        table = self.discoveredFilesTable

        # Column structure:
        # 0: File Name
        # 1: Protocol
        # 2: Avail Ch (available channels count)
        # 3: Sweeps (number of sweeps)
        # 4: Channel (selected for analysis)
        # 5: Stim Ch (detected stimulus channel)
        # 6: Keywords
        # 7: Strain (editable)
        # 8: Stim Type (auto-detected from stim frequency)
        # 9: Power (editable, auto-fill)
        # 10: Sex (editable)
        # 11: Animal ID (editable)
        # 12: Status
        # 13: Actions (+/- buttons)

        new_columns = [
            "File Name",      # 0
            "Protocol",       # 1
            "Avail Ch",       # 2 (available channels)
            "Sweeps",         # 3 (number of sweeps)
            "Keywords",       # 4 (path keywords - parent level info)
            "Experiment",     # 5 (user-defined experiment grouping) - NEW
            "Channel",        # 6 (analyzed channel - filled on save)
            "Stim Ch",        # 7 (stim channel - filled on save)
            "Events Ch",      # 8 (events channel - filled on save)
            "Strain",         # 9
            "Stim Type",      # 10 (auto-detected or manual)
            "Power",          # 11
            "Sex",            # 12
            "Animal ID",      # 13
            "Status",         # 14
            "",               # 15 (actions - Analyze/+ buttons)
            "Exports"         # 16 (export summary with tooltip)
        ]

        table.setColumnCount(len(new_columns))
        table.setHorizontalHeaderLabels(new_columns)

        # Disable default sorting - we'll implement custom grouped sorting
        table.setSortingEnabled(False)

        # Set column properties
        header = table.horizontalHeader()

        # Connect header click for custom grouped sorting
        header.sectionClicked.connect(self._on_header_sort_clicked)

        # 0: File name "File Name"
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(0, 70)
        # 1: Protocol "Protocol"
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(1, 58)
        # 2: Available channels "Avail Ch"
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(2, 52)
        # 3: Sweeps "Sweeps"
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(3, 50)
        # 4: Keywords "Keywords"
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(4, 62)
        # 5: Experiment "Experiment" - NEW user-defined grouping
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(5, 70)
        # 6: Channel "Channel"
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(6, 55)
        # 7: Stim Ch "Stim Ch"
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(7, 52)
        # 8: Events Ch "Events Ch"
        header.setSectionResizeMode(8, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(8, 62)
        # 9: Strain "Strain"
        header.setSectionResizeMode(9, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(9, 45)
        # 10: Stim Type "Stim Type"
        header.setSectionResizeMode(10, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(10, 62)
        # 11: Power "Power"
        header.setSectionResizeMode(11, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(11, 45)
        # 12: Sex "Sex"
        header.setSectionResizeMode(12, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(12, 30)
        # 13: Animal ID "Animal ID"
        header.setSectionResizeMode(13, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(13, 62)
        # 14: Status "Status"
        header.setSectionResizeMode(14, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(14, 45)
        # 15: Actions "" (no header) - needs room for Analyze/+/- buttons
        header.setSectionResizeMode(15, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(15, 85)
        # 16: Exports "Exports"
        header.setSectionResizeMode(16, QHeaderView.ResizeMode.Interactive)
        table.setColumnWidth(16, 55)

        # Connect cell changed signal for editable columns
        table.cellChanged.connect(self._on_master_list_cell_changed)

        # Connect double-click to open file for analysis
        table.cellDoubleClicked.connect(self._on_master_list_double_click)

        # Enable context menu for bulk editing
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        table.customContextMenuRequested.connect(self._on_master_list_context_menu)

        # Install event filter to handle resize events for auto-fit columns
        table.viewport().installEventFilter(self)
        self._table_resize_timer = None  # For debouncing resize events

        print("[project-builder] Master file list configured with new columns")

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
        col_name = self.discoveredFilesTable.horizontalHeaderItem(column).text() if self.discoveredFilesTable.horizontalHeaderItem(column) else f"Col {column}"
        self._log_status_message(f"Sorted by {col_name} {direction}", 2000)

    def _rebuild_table_from_master_list(self):
        """Rebuild the table from the current _master_file_list order."""
        from PyQt6.QtWidgets import QTableWidgetItem

        table = self.discoveredFilesTable
        table.blockSignals(True)
        table.setRowCount(0)

        for i, task in enumerate(self._master_file_list):
            table.insertRow(i)
            is_sub_row = task.get('is_sub_row', False)

            # Column 0: File Name or channel for sub-rows
            if is_sub_row:
                # Show channel with filename in parentheses for clarity
                file_name = task.get('file_name', '')
                channel = task.get('channel', '')
                item = QTableWidgetItem(f"  ↳ {channel}")
                item.setToolTip(f"File: {file_name}\nChannel: {channel}")
                table.setItem(i, 0, item)
            else:
                table.setItem(i, 0, QTableWidgetItem(task.get('file_name', '')))

            # Columns 1-4: Show parent info on sub-rows too (helps with sorting/filtering)
            table.setItem(i, 1, QTableWidgetItem(task.get('protocol', '')))
            if is_sub_row:
                table.setItem(i, 2, QTableWidgetItem(''))  # Avail Ch not relevant for sub-row
                table.setItem(i, 3, QTableWidgetItem(''))  # Sweeps not relevant for sub-row
            else:
                table.setItem(i, 2, QTableWidgetItem(str(task.get('channel_count', ''))))
                table.setItem(i, 3, QTableWidgetItem(str(task.get('sweep_count', ''))))
            table.setItem(i, 4, QTableWidgetItem(task.get('keywords_display', '')))

            # Column 5: Experiment (user-defined grouping)
            table.setItem(i, 5, QTableWidgetItem(task.get('experiment', '')))

            # Columns 6-8: Channel, Stim Ch, Events Ch
            table.setItem(i, 6, QTableWidgetItem(task.get('channel', '')))
            table.setItem(i, 7, QTableWidgetItem(task.get('stim_channel', '')))
            table.setItem(i, 8, QTableWidgetItem(task.get('events_channel', '')))

            # Columns 9-13: Metadata
            table.setItem(i, 9, QTableWidgetItem(task.get('strain', '')))
            table.setItem(i, 10, QTableWidgetItem(task.get('stim_type', '')))
            table.setItem(i, 11, QTableWidgetItem(task.get('power', '')))
            table.setItem(i, 12, QTableWidgetItem(task.get('sex', '')))
            table.setItem(i, 13, QTableWidgetItem(task.get('animal_id', '')))

            # Column 14: Status (with warnings if applicable)
            status = task.get('status', 'pending')
            warnings = task.get('scan_warnings', {})
            has_conflicts = bool(warnings.get('conflicts'))
            has_older_files = warnings.get('older_npz_count', 0) > 0

            if is_sub_row:
                if status == 'completed':
                    if has_conflicts:
                        status_icon = '●⚠'
                    elif has_older_files:
                        status_icon = '●📁'
                    else:
                        status_icon = '●'
                elif status == 'in_progress':
                    status_icon = '◐'
                else:
                    status_icon = '○'

                status_item = QTableWidgetItem(status_icon)

                # Add tooltip for warnings
                if has_conflicts or has_older_files:
                    tooltip_parts = []
                    if has_conflicts:
                        tooltip_parts.append("⚠ Data conflicts (table vs NPZ):")
                        for c in warnings['conflicts']:
                            tooltip_parts.append(f"  • {c}")
                    if has_older_files:
                        count = warnings['older_npz_count']
                        tooltip_parts.append(f"📁 {count} older NPZ file(s) found:")
                        for older in warnings.get('older_npz_files', [])[:3]:
                            tooltip_parts.append(f"  • {Path(older['file']).name} ({older['date']})")
                        if count > 3:
                            tooltip_parts.append(f"  • ... and {count - 3} more")
                    status_item.setToolTip('\n'.join(tooltip_parts))

                table.setItem(i, 14, status_item)
            else:
                # Parent row - count completed sub-rows
                status_icon = ''  # Will be set by _apply_all_row_styling
                table.setItem(i, 14, QTableWidgetItem(status_icon))

            # Column 15: Action buttons
            if is_sub_row:
                self._add_sub_row_action_buttons(i)
            else:
                self._add_row_action_button(i)

            # Column 16: Exports
            table.setItem(i, 16, self._create_exports_table_item(task))

        table.blockSignals(False)

        # Apply styling
        self._apply_all_row_styling()

    def _on_master_list_cell_changed(self, row, column):
        """Handle cell edits in the master file list."""
        # Editable columns: 5 (Experiment), 9 (Strain), 10 (Stim Type), 11 (Power), 12 (Sex), 13 (Animal ID)
        editable_columns = {5: 'experiment', 9: 'strain', 10: 'stim_type', 11: 'power', 12: 'sex', 13: 'animal_id'}

        if column not in editable_columns:
            return

        if row >= len(self._master_file_list):
            return

        field = editable_columns[column]
        item = self.discoveredFilesTable.item(row, column)
        if item:
            value = item.text().strip()
            self._master_file_list[row][field] = value
            print(f"[master-list] Updated row {row} {field} = '{value}'")

            # Track experiment names in history for autocomplete
            if field == 'experiment' and value:
                self._update_experiment_history(value)

    def _get_experiment_history(self) -> list:
        """Get list of previously used experiment names from QSettings."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("PhysioMetrics", "BreathAnalysis")
        history = settings.value("project_builder/experiment_history", [])
        return history if isinstance(history, list) else []

    def _update_experiment_history(self, experiment_name: str):
        """Add an experiment name to history (max 50 entries, most recent first)."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("PhysioMetrics", "BreathAnalysis")

        history = self._get_experiment_history()

        # Remove if already exists (to move to front)
        if experiment_name in history:
            history.remove(experiment_name)

        # Add to front
        history.insert(0, experiment_name)

        # Keep max 50
        history = history[:50]

        settings.setValue("project_builder/experiment_history", history)

    def _on_master_list_double_click(self, row, column):
        """Handle double-click on master list row - open file for analysis."""
        if row >= len(self._master_file_list):
            return

        task = self._master_file_list[row]
        file_path = task.get('file_path')

        if file_path and Path(file_path).exists():
            print(f"[master-list] Opening file for analysis: {file_path}")
            # Track which row is being analyzed
            self._active_master_list_row = row

            # Store pending channel selections from Project Builder
            # These will be used after file loads to override auto-detection
            self._pending_analysis_channel = task.get('channel', '')
            self._pending_stim_channels = task.get('stim_channels', [])

            # Log what we're planning to select
            if self._pending_analysis_channel or self._pending_stim_channels:
                print(f"[master-list] Pre-selecting: analysis={self._pending_analysis_channel}, stim={self._pending_stim_channels}")

            self.load_file(Path(file_path))
            # Switch to Analysis tab
            if hasattr(self, 'mainTabWidget'):
                self.mainTabWidget.setCurrentIndex(0)  # Analysis tab

    def mark_active_analysis_complete(self, channel_used: str = None, stim_channel_used: str = None,
                                       events_channel_used: str = None, export_info: dict = None):
        """
        Mark the currently active master list row as completed.
        Called after successful export/save of analysis data.

        Args:
            channel_used: The channel that was analyzed (e.g., "AD0")
            stim_channel_used: The stim channel used (e.g., "AD1")
            events_channel_used: The events channel used (e.g., "AD2")
            export_info: Dict with export metadata and file flags:
                - export_path: Where files were saved
                - export_date: When exported
                - export_version: App version used
                - exports: Dict of boolean flags for each export type
                - strain, stim_type, power, sex, animal_id: Metadata from dialog
        """
        # Get current file path from state
        current_file_path = str(self.state.in_path) if self.state.in_path else None

        # Find matching row(s) by file path if active row not set
        row = self._active_master_list_row

        if row is None and current_file_path:
            # Try to find a row matching the current file
            # Priority: 1) Same file + same channel, 2) Same file + completed (different channel), 3) Same file + pending
            from pathlib import Path
            current_normalized = str(Path(current_file_path).resolve())

            matching_rows = []  # [(row_idx, task), ...]
            for i, task in enumerate(self._master_file_list):
                task_path = task.get('file_path', '')
                if task_path:
                    task_normalized = str(Path(task_path).resolve())
                    if task_normalized == current_normalized:
                        matching_rows.append((i, task))

            if matching_rows:
                # Priority 1: Find row with same channel (to update existing)
                for i, task in matching_rows:
                    if task.get('channel', '') == channel_used:
                        row = i
                        print(f"[master-list] Found matching row {i} with same channel {channel_used}")
                        break

                # Priority 2: Find completed row with different channel (to create sub-row)
                if row is None:
                    for i, task in matching_rows:
                        if task.get('status', 'pending') == 'completed':
                            row = i
                            print(f"[master-list] Found completed row {i} (channel: {task.get('channel', '')})")
                            break

                # Priority 3: Find any pending row (first time analysis)
                if row is None:
                    for i, task in matching_rows:
                        if task.get('status', 'pending') == 'pending':
                            row = i
                            print(f"[master-list] Found pending row {i}")
                            break

                # Fallback: Use first matching row
                if row is None and matching_rows:
                    row = matching_rows[0][0]
                    print(f"[master-list] Using first matching row {row}")

        if row is None:
            print(f"[master-list] No matching row found for file: {current_file_path}")
            return

        if row >= len(self._master_file_list):
            return

        task = self._master_file_list[row]

        # Check if this is a parent row or sub-row
        is_sub_row = task.get('is_sub_row', False)
        existing_channel = task.get('channel', '')
        existing_status = task.get('status', 'pending')

        # Debug logging to understand the flow
        print(f"[master-list] mark_active_analysis_complete called:")
        print(f"  - Row: {row}")
        print(f"  - File: {task.get('file_name', 'unknown')}")
        print(f"  - Is sub-row: {is_sub_row}")
        print(f"  - Existing channel in row: '{existing_channel}'")
        print(f"  - Channel just analyzed: '{channel_used}'")
        print(f"  - Existing status: '{existing_status}'")

        # For parent rows, ALWAYS create sub-rows instead of updating the parent
        # This keeps parent as a "header" row showing file-level info only
        if not is_sub_row:
            print(f"[master-list] ✓ Parent row detected - creating sub-row for {channel_used}")

            # If parent had existing completed analysis, move it to sub-row first
            if existing_channel and existing_status == 'completed':
                # Check if there's already a sub-row for the existing channel
                has_existing_subrow = False
                for t in self._master_file_list:
                    if (t.get('is_sub_row') and
                        str(t.get('file_path')) == str(task.get('file_path')) and
                        t.get('channel') == existing_channel):
                        has_existing_subrow = True
                        break

                if not has_existing_subrow:
                    # Move existing channel data to a sub-row first
                    existing_export_info = {
                        'export_path': task.get('export_path', ''),
                        'export_date': task.get('export_date', ''),
                        'export_version': task.get('export_version', ''),
                        'exports': task.get('exports', {}),
                        'strain': task.get('strain', ''),
                        'stim_type': task.get('stim_type', ''),
                        'power': task.get('power', ''),
                        'sex': task.get('sex', ''),
                        'animal_id': task.get('animal_id', ''),
                    }
                    self._create_sub_row_from_analysis(
                        row, existing_channel,
                        task.get('stim_channel', ''),
                        task.get('events_channel', ''),
                        existing_export_info
                    )
                    print(f"[master-list]   - Moved existing {existing_channel} to sub-row")

            # Create sub-row for the new analysis
            self._create_sub_row_from_analysis(
                row, channel_used, stim_channel_used, events_channel_used, export_info
            )
            print(f"[master-list]   - Created sub-row for {channel_used}")

            # Apply styling to update colors and clean parent
            self._apply_all_row_styling()

            self._active_master_list_row = None
            return

        # For sub-rows, update the existing sub-row
        task['status'] = 'completed'
        if channel_used:
            task['channel'] = channel_used
        if stim_channel_used:
            task['stim_channel'] = stim_channel_used
        if events_channel_used:
            task['events_channel'] = events_channel_used

        # Update export tracking info
        if export_info:
            task['export_path'] = export_info.get('export_path', '')
            task['export_date'] = export_info.get('export_date', '')
            task['export_version'] = export_info.get('export_version', '')
            task['exports'] = export_info.get('exports', task.get('exports', {}))

            # Update metadata from save dialog
            if export_info.get('strain'):
                task['strain'] = export_info['strain']
            if export_info.get('stim_type'):
                task['stim_type'] = export_info['stim_type']
            if export_info.get('power'):
                task['power'] = export_info['power']
            if export_info.get('sex'):
                task['sex'] = export_info['sex']
            if export_info.get('animal_id'):
                task['animal_id'] = export_info['animal_id']

        # Update the table display
        table = self.discoveredFilesTable
        if row < table.rowCount():
            from PyQt6.QtWidgets import QTableWidgetItem

            # Status column (14) - show completed icon
            status_item = table.item(row, 14)
            if status_item:
                status_item.setText('●')  # Filled circle for completed
            else:
                table.setItem(row, 14, QTableWidgetItem('●'))

            # Channel column (6)
            if channel_used:
                channel_item = table.item(row, 6)
                if channel_item:
                    channel_item.setText(channel_used)
                else:
                    table.setItem(row, 6, QTableWidgetItem(channel_used))

            # Stim Ch column (7)
            if stim_channel_used:
                stim_item = table.item(row, 7)
                if stim_item:
                    stim_item.setText(stim_channel_used)
                else:
                    table.setItem(row, 7, QTableWidgetItem(stim_channel_used))

            # Events Ch column (8)
            if events_channel_used:
                events_item = table.item(row, 8)
                if events_item:
                    events_item.setText(events_channel_used)
                else:
                    table.setItem(row, 8, QTableWidgetItem(events_channel_used))

            # Metadata columns (update if provided and currently empty)
            if export_info:
                # Strain column (9)
                if export_info.get('strain'):
                    strain_item = table.item(row, 9)
                    if strain_item:
                        strain_item.setText(export_info['strain'])
                    else:
                        table.setItem(row, 9, QTableWidgetItem(export_info['strain']))

                # Stim Type column (10)
                if export_info.get('stim_type'):
                    stim_type_item = table.item(row, 10)
                    if stim_type_item:
                        stim_type_item.setText(export_info['stim_type'])
                    else:
                        table.setItem(row, 10, QTableWidgetItem(export_info['stim_type']))

                # Power column (11)
                if export_info.get('power'):
                    power_item = table.item(row, 11)
                    if power_item:
                        power_item.setText(export_info['power'])
                    else:
                        table.setItem(row, 11, QTableWidgetItem(export_info['power']))

                # Sex column (12)
                if export_info.get('sex'):
                    sex_item = table.item(row, 12)
                    if sex_item:
                        sex_item.setText(export_info['sex'])
                    else:
                        table.setItem(row, 12, QTableWidgetItem(export_info['sex']))

                # Animal ID column (13)
                if export_info.get('animal_id'):
                    animal_item = table.item(row, 13)
                    if animal_item:
                        animal_item.setText(export_info['animal_id'])
                    else:
                        table.setItem(row, 13, QTableWidgetItem(export_info['animal_id']))

            # Exports column (16) - update with summary and tooltip
            table.setItem(row, 16, self._create_exports_table_item(task))

        print(f"[master-list] Marked row {row} as completed:")
        print(f"  - Channel: {channel_used}")
        print(f"  - Stim Ch: {stim_channel_used}")
        print(f"  - Events Ch: {events_channel_used}")
        if export_info:
            print(f"  - Export path: {export_info.get('export_path', 'N/A')}")
            exports = export_info.get('exports', {})
            saved = [k for k, v in exports.items() if v]
            print(f"  - Saved: {', '.join(saved) if saved else 'none'}")

        # Clear active row - analysis is done
        self._active_master_list_row = None

    def _create_sub_row_from_analysis(self, source_row: int, channel_used: str,
                                       stim_channel_used: str, events_channel_used: str,
                                       export_info: dict):
        """
        Create a new sub-row when user analyzes a different channel from an existing analyzed row.

        This prevents overwriting previous analysis when the user analyzes the same file
        but with a different channel.

        Args:
            source_row: Row index of the source task
            channel_used: The channel that was analyzed
            stim_channel_used: The stim channel used
            events_channel_used: The events channel used
            export_info: Export metadata dict from save dialog
        """
        from PyQt6.QtWidgets import QTableWidgetItem

        if source_row >= len(self._master_file_list):
            return

        source_task = self._master_file_list[source_row]
        table = self.discoveredFilesTable
        file_path = source_task.get('file_path')

        # Create new task based on source but with new channel and analysis info
        new_task = {
            'file_path': file_path,
            'file_name': source_task.get('file_name', ''),
            'protocol': source_task.get('protocol', ''),
            'channel_count': source_task.get('channel_count', 0),
            'sweep_count': source_task.get('sweep_count', 0),
            'channel_names': source_task.get('channel_names', []),
            'stim_channels': source_task.get('stim_channels', []),
            'path_keywords': source_task.get('path_keywords', {}),
            'keywords_display': source_task.get('keywords_display', ''),
            # Analysis results
            'channel': channel_used,
            'stim_channel': stim_channel_used or '',
            'events_channel': events_channel_used or '',
            # Copy or update metadata
            'strain': export_info.get('strain', '') if export_info else source_task.get('strain', ''),
            'stim_type': export_info.get('stim_type', '') if export_info else source_task.get('stim_type', ''),
            'power': export_info.get('power', '') if export_info else source_task.get('power', ''),
            'sex': export_info.get('sex', '') if export_info else source_task.get('sex', ''),
            'animal_id': export_info.get('animal_id', '') if export_info else '',
            'status': 'completed',
            'is_sub_row': True,
            'parent_file': file_path,
            # Export tracking
            'export_path': export_info.get('export_path', '') if export_info else '',
            'export_date': export_info.get('export_date', '') if export_info else '',
            'export_version': export_info.get('export_version', '') if export_info else '',
            'exports': export_info.get('exports', {}) if export_info else {
                'npz': False,
                'timeseries_csv': False,
                'breaths_csv': False,
                'events_csv': False,
                'pdf': False,
                'session_state': False,
                'ml_training': False,
            }
        }

        # Insert after the source row
        insert_row = source_row + 1
        self._master_file_list.insert(insert_row, new_task)

        # Block signals while inserting
        table.blockSignals(True)

        # Insert table row with 16-column structure
        table.insertRow(insert_row)
        table.setItem(insert_row, 0, QTableWidgetItem(f"  ↳ {channel_used}"))  # Show channel, not filename
        table.setItem(insert_row, 1, QTableWidgetItem(''))  # Leave protocol empty on sub-rows
        table.setItem(insert_row, 2, QTableWidgetItem(''))  # Leave channel count empty on sub-rows
        table.setItem(insert_row, 3, QTableWidgetItem(''))  # Leave sweep count empty on sub-rows
        table.setItem(insert_row, 4, QTableWidgetItem(''))  # Leave keywords empty on sub-rows
        table.setItem(insert_row, 5, QTableWidgetItem(channel_used))  # Channel
        table.setItem(insert_row, 6, QTableWidgetItem(stim_channel_used or ''))  # Stim Ch
        table.setItem(insert_row, 7, QTableWidgetItem(events_channel_used or ''))  # Events Ch
        table.setItem(insert_row, 8, QTableWidgetItem(new_task['strain']))
        table.setItem(insert_row, 9, QTableWidgetItem(new_task['stim_type']))
        table.setItem(insert_row, 10, QTableWidgetItem(new_task['power']))
        table.setItem(insert_row, 11, QTableWidgetItem(new_task['sex']))
        table.setItem(insert_row, 12, QTableWidgetItem(new_task['animal_id']))
        table.setItem(insert_row, 13, QTableWidgetItem('●'))  # Status - completed
        # Add action buttons
        self._add_sub_row_action_buttons(insert_row)
        # Column 15: Exports
        table.setItem(insert_row, 15, self._create_exports_table_item(new_task))

        table.blockSignals(False)

        # Apply sub-row styling
        self._apply_row_styling(insert_row, is_sub_row=True)

        print(f"[master-list] Created new sub-row at {insert_row} for channel {channel_used}")
        print(f"  - This preserves the previous analysis of {source_task.get('channel', 'unknown')} in row {source_row}")

    def _update_task_with_export_info(self, task: dict, info: dict, row: int, table):
        """
        Update a task dict and its table row with export info from saved data scan.

        Args:
            task: The task dict from _master_file_list
            info: Export info dict from NPZ metadata
            row: Table row index
            table: The QTableWidget
        """
        from PyQt6.QtWidgets import QTableWidgetItem

        # Track conflicts and multiple files
        conflicts = []
        older_files = info.get('older_npz_files', [])

        # Check for conflicts between existing task data and NPZ data
        metadata_fields = [
            ('strain', 'strain', 'Strain'),
            ('stim_type', 'stim_type', 'Stim Type'),
            ('power', 'power', 'Power'),
            ('sex', 'sex', 'Sex'),
            ('animal_id', 'animal_id', 'Animal ID'),
        ]
        for task_key, info_key, display_name in metadata_fields:
            task_val = task.get(task_key, '').strip()
            info_val = (info.get(info_key, '') or '').strip()
            if task_val and info_val and task_val != info_val:
                conflicts.append(f"{display_name}: table='{task_val}' vs NPZ='{info_val}'")

        # Store conflict info in task
        if conflicts or older_files:
            task['scan_warnings'] = {
                'conflicts': conflicts,
                'older_npz_count': len(older_files),
                'older_npz_files': older_files,
            }
            if conflicts:
                print(f"[scan-saved] Conflicts found for row {row}: {conflicts}")
            if older_files:
                print(f"[scan-saved] {len(older_files)} older NPZ file(s) found for row {row}")

        # Update task with export info
        task['export_path'] = info.get('export_path', '')
        task['export_date'] = info.get('export_date', '')
        task['status'] = 'completed'
        task['exports'] = {
            'npz': info.get('npz', False),
            'timeseries_csv': info.get('timeseries_csv', False),
            'breaths_csv': info.get('breaths_csv', False),
            'events_csv': info.get('events_csv', False),
            'pdf': info.get('pdf', False),
            'session_state': info.get('session_state', False),
            'ml_training': info.get('ml_training', False),
        }

        # Update metadata from NPZ if available and not already set (don't overwrite)
        if info.get('strain') and not task.get('strain'):
            task['strain'] = info['strain']
        if info.get('stim_type') and not task.get('stim_type'):
            task['stim_type'] = info['stim_type']
        if info.get('power') and not task.get('power'):
            task['power'] = info['power']
        if info.get('sex') and not task.get('sex'):
            task['sex'] = info['sex']
        if info.get('animal_id') and not task.get('animal_id'):
            task['animal_id'] = info['animal_id']
        if info.get('stim_channel') and not task.get('stim_channel'):
            task['stim_channel'] = info['stim_channel']
        if info.get('events_channel') and not task.get('events_channel'):
            task['events_channel'] = info['events_channel']

        # Update table row
        if row < table.rowCount():
            # Status column - show warning if conflicts or multiple NPZ files
            warnings = task.get('scan_warnings', {})
            has_conflicts = bool(warnings.get('conflicts'))
            has_older_files = warnings.get('older_npz_count', 0) > 0

            if has_conflicts or has_older_files:
                # Show warning indicator
                status_text = '●⚠' if has_conflicts else '●📁'
                status_item = QTableWidgetItem(status_text)
                # Build tooltip
                tooltip_parts = []
                if has_conflicts:
                    tooltip_parts.append("⚠ Data conflicts (table vs NPZ):")
                    for c in warnings['conflicts']:
                        tooltip_parts.append(f"  • {c}")
                if has_older_files:
                    count = warnings['older_npz_count']
                    tooltip_parts.append(f"📁 {count} older NPZ file(s) found:")
                    for older in warnings.get('older_npz_files', [])[:3]:  # Show first 3
                        tooltip_parts.append(f"  • {Path(older['file']).name} ({older['date']})")
                    if count > 3:
                        tooltip_parts.append(f"  • ... and {count - 3} more")
                status_item.setToolTip('\n'.join(tooltip_parts))
                table.setItem(row, 13, status_item)
            else:
                table.setItem(row, 13, QTableWidgetItem('●'))  # Status completed, no warnings

            table.setItem(row, 15, self._create_exports_table_item(task))

            # Update channel columns from NPZ if available
            if task.get('stim_channel'):
                table.setItem(row, 6, QTableWidgetItem(task['stim_channel']))
            if task.get('events_channel'):
                table.setItem(row, 7, QTableWidgetItem(task['events_channel']))

            # Update metadata columns from NPZ if available
            if task.get('strain'):
                table.setItem(row, 8, QTableWidgetItem(task['strain']))
            if task.get('stim_type'):
                table.setItem(row, 9, QTableWidgetItem(task['stim_type']))
            if task.get('power'):
                table.setItem(row, 10, QTableWidgetItem(task['power']))
            if task.get('sex'):
                table.setItem(row, 11, QTableWidgetItem(task['sex']))
            if task.get('animal_id'):
                table.setItem(row, 12, QTableWidgetItem(task['animal_id']))

    def _create_sub_row_from_saved_data(self, source_task: dict, source_row: int,
                                         channel: str, info: dict):
        """
        Create a new sub-row from scanned saved data for an additional channel.

        Args:
            source_task: The source task dict to base the new row on
            source_row: Row index of the source task
            channel: The channel for this saved data
            info: Export info dict from NPZ metadata
        """
        from PyQt6.QtWidgets import QTableWidgetItem

        table = self.discoveredFilesTable
        file_path = source_task.get('file_path')

        # Track any older/duplicate NPZ files found
        older_files = info.get('older_npz_files', [])
        scan_warnings = {}
        if older_files:
            scan_warnings = {
                'conflicts': [],
                'older_npz_count': len(older_files),
                'older_npz_files': older_files,
            }

        # Create new task based on source but with new channel and analysis info
        new_task = {
            'file_path': file_path,
            'file_name': source_task.get('file_name', ''),
            'protocol': source_task.get('protocol', ''),
            'channel_count': source_task.get('channel_count', 0),
            'sweep_count': source_task.get('sweep_count', 0),
            'channel_names': source_task.get('channel_names', []),
            'stim_channels': source_task.get('stim_channels', []),
            'path_keywords': source_task.get('path_keywords', {}),
            'keywords_display': source_task.get('keywords_display', ''),
            # Analysis results
            'channel': channel,
            'stim_channel': info.get('stim_channel', ''),
            'events_channel': info.get('events_channel', ''),
            # Metadata from NPZ
            'strain': info.get('strain', '') or source_task.get('strain', ''),
            'stim_type': info.get('stim_type', '') or source_task.get('stim_type', ''),
            'power': info.get('power', '') or source_task.get('power', ''),
            'sex': info.get('sex', '') or source_task.get('sex', ''),
            'animal_id': info.get('animal_id', ''),
            'status': 'completed',
            'is_sub_row': True,
            'parent_file': file_path,
            # Export tracking
            'export_path': info.get('export_path', ''),
            'export_date': info.get('export_date', ''),
            'exports': {
                'npz': info.get('npz', False),
                'timeseries_csv': info.get('timeseries_csv', False),
                'breaths_csv': info.get('breaths_csv', False),
                'events_csv': info.get('events_csv', False),
                'pdf': info.get('pdf', False),
                'session_state': info.get('session_state', False),
                'ml_training': info.get('ml_training', False),
            }
        }
        if scan_warnings:
            new_task['scan_warnings'] = scan_warnings

        # Find where to insert - after all existing rows for this file
        insert_row = source_row + 1
        while insert_row < len(self._master_file_list):
            next_task = self._master_file_list[insert_row]
            if str(next_task.get('file_path')) != str(file_path):
                break
            insert_row += 1

        self._master_file_list.insert(insert_row, new_task)

        # Block signals while inserting
        table.blockSignals(True)

        # Insert table row with 16-column structure
        table.insertRow(insert_row)
        table.setItem(insert_row, 0, QTableWidgetItem(f"  ↳ {channel}"))  # Show channel, not filename
        table.setItem(insert_row, 1, QTableWidgetItem(''))  # Leave protocol empty on sub-rows
        table.setItem(insert_row, 2, QTableWidgetItem(''))  # Leave channel count empty on sub-rows
        table.setItem(insert_row, 3, QTableWidgetItem(''))  # Leave sweep count empty on sub-rows
        table.setItem(insert_row, 4, QTableWidgetItem(''))  # Leave keywords empty on sub-rows
        table.setItem(insert_row, 5, QTableWidgetItem(channel))  # Channel
        table.setItem(insert_row, 6, QTableWidgetItem(new_task['stim_channel']))  # Stim Ch
        table.setItem(insert_row, 7, QTableWidgetItem(new_task['events_channel']))  # Events Ch
        table.setItem(insert_row, 8, QTableWidgetItem(new_task['strain']))
        table.setItem(insert_row, 9, QTableWidgetItem(new_task['stim_type']))
        table.setItem(insert_row, 10, QTableWidgetItem(new_task['power']))
        table.setItem(insert_row, 11, QTableWidgetItem(new_task['sex']))
        table.setItem(insert_row, 12, QTableWidgetItem(new_task['animal_id']))

        # Status column - show warning if multiple NPZ files found
        if scan_warnings and scan_warnings.get('older_npz_count', 0) > 0:
            status_item = QTableWidgetItem('●📁')
            count = scan_warnings['older_npz_count']
            tooltip_parts = [f"📁 {count} older NPZ file(s) found:"]
            for older in scan_warnings.get('older_npz_files', [])[:3]:
                tooltip_parts.append(f"  • {Path(older['file']).name} ({older['date']})")
            if count > 3:
                tooltip_parts.append(f"  • ... and {count - 3} more")
            status_item.setToolTip('\n'.join(tooltip_parts))
            table.setItem(insert_row, 13, status_item)
        else:
            table.setItem(insert_row, 13, QTableWidgetItem('●'))  # Status - completed

        # Add action buttons
        self._add_sub_row_action_buttons(insert_row)
        # Column 15: Exports
        table.setItem(insert_row, 15, self._create_exports_table_item(new_task))

        table.blockSignals(False)

        # Apply sub-row styling
        self._apply_row_styling(insert_row, is_sub_row=True)

        print(f"[scan-saved] Created sub-row at {insert_row} for channel {channel}")

    def eventFilter(self, obj, event):
        """Handle events for installed event filters (e.g., table resize)."""
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

        return super().eventFilter(obj, event)

    def _auto_fit_table_columns(self):
        """
        Auto-fit column widths intelligently:
        - Minimum width = header text width
        - Expand to fit content if needed
        - Constrain total width to visible area (unless full-content mode enabled)
        - Expandable columns (File Name, Protocol, Keywords, Exports) get extra space
        """
        table = self.discoveredFilesTable
        header = table.horizontalHeader()

        # Check if full content mode is enabled
        full_content_mode = getattr(self, 'tableFullContentCheckbox', None)
        full_content_mode = full_content_mode.isChecked() if full_content_mode else False

        # Get header text widths (minimum for each column)
        font_metrics = table.fontMetrics()
        header_widths = {}
        for col in range(table.columnCount()):
            header_item = table.horizontalHeaderItem(col)
            if header_item:
                text = header_item.text()
                # Add some padding for header
                header_widths[col] = font_metrics.horizontalAdvance(text) + 16
            else:
                header_widths[col] = 30  # Default for empty headers

        # Get content widths by temporarily resizing to contents
        table.resizeColumnsToContents()
        content_widths = {}
        for col in range(table.columnCount()):
            content_widths[col] = table.columnWidth(col)

        # Columns that can expand to fill space (in priority order)
        expandable_cols = [0, 5, 4, 1, 16]  # File Name, Experiment, Keywords, Protocol, Exports

        # Fixed columns with specific widths (buttons need room)
        fixed_col_widths = {15: 85}  # Actions button column

        if full_content_mode:
            # Full content mode: show all content, enable horizontal scrolling
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            for col in range(table.columnCount()):
                if col in fixed_col_widths:
                    # Fixed columns always use their specified width
                    width = fixed_col_widths[col]
                else:
                    # Use max of header width and content width
                    width = max(header_widths.get(col, 30), content_widths.get(col, 30))
                table.setColumnWidth(col, width)
        else:
            # Fit-to-view mode: constrain to visible width
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

            # Get available width (viewport width minus scrollbar and margins)
            available_width = table.viewport().width()
            if available_width < 100:
                # Fallback if viewport not ready
                available_width = table.width() - 20

            # Calculate base widths (max of header and content, but reasonable)
            base_widths = {}
            for col in range(table.columnCount()):
                header_w = header_widths.get(col, 30)
                content_w = content_widths.get(col, 30)

                if col in fixed_col_widths:
                    # Fixed columns: use specified width
                    base_widths[col] = fixed_col_widths[col]
                elif col in expandable_cols:
                    # Expandable: use header as minimum, allow content to expand
                    base_widths[col] = max(header_w, min(content_w, header_w + 50))
                else:
                    # Regular columns: use header width unless content is slightly larger
                    base_widths[col] = max(header_w, min(content_w, header_w + 20))

            # Calculate total base width
            total_base = sum(base_widths.values())

            if total_base <= available_width:
                # We have extra space - distribute to expandable columns
                extra_space = available_width - total_base

                # Calculate how much each expandable column wants
                expandable_wants = {}
                for col in expandable_cols:
                    content_w = content_widths.get(col, 30)
                    current_w = base_widths.get(col, 30)
                    want = max(0, content_w - current_w)
                    expandable_wants[col] = want

                total_want = sum(expandable_wants.values())

                if total_want > 0 and extra_space > 0:
                    # Distribute extra space proportionally to what columns want
                    for col in expandable_cols:
                        if expandable_wants[col] > 0:
                            share = min(expandable_wants[col],
                                       extra_space * expandable_wants[col] / total_want)
                            base_widths[col] += int(share)

                # Apply widths
                for col in range(table.columnCount()):
                    table.setColumnWidth(col, base_widths[col])
            else:
                # Need to shrink - reduce expandable columns first
                overflow = total_base - available_width

                # Try to take from expandable columns proportionally
                expandable_excess = {}
                for col in expandable_cols:
                    header_w = header_widths.get(col, 30)
                    current_w = base_widths.get(col, 30)
                    excess = max(0, current_w - header_w)
                    expandable_excess[col] = excess

                total_excess = sum(expandable_excess.values())

                if total_excess >= overflow:
                    # Can shrink expandable columns enough
                    for col in expandable_cols:
                        if expandable_excess[col] > 0:
                            reduction = int(overflow * expandable_excess[col] / total_excess)
                            base_widths[col] -= reduction
                else:
                    # Need to shrink all columns proportionally
                    scale = available_width / total_base
                    for col in range(table.columnCount()):
                        header_w = header_widths.get(col, 30)
                        new_w = max(header_w, int(base_widths[col] * scale))
                        base_widths[col] = new_w

                # Apply widths
                for col in range(table.columnCount()):
                    table.setColumnWidth(col, base_widths[col])

    def _apply_row_styling(self, row: int, is_sub_row: bool = False):
        """Apply visual styling to a table row based on whether it's a parent or sub-row."""
        from PyQt6.QtGui import QColor, QBrush, QFont
        from PyQt6.QtCore import Qt

        table = self.discoveredFilesTable
        if row >= table.rowCount():
            return

        if is_sub_row:
            # Sub-rows: lighter background, normal weight text
            bg_color = QColor(50, 55, 60)  # Slightly lighter than parent
            text_color = QColor(200, 200, 200)  # Slightly muted text
            font_bold = False
        else:
            # Parent rows: darker background, BOLD text
            bg_color = QColor(35, 40, 45)  # Default dark theme background
            text_color = QColor(255, 255, 255)  # White text
            font_bold = True

        # Apply to all cells in the row
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item:
                item.setBackground(QBrush(bg_color))
                item.setForeground(QBrush(text_color))
                # Set font weight
                font = item.font()
                font.setBold(font_bold)
                item.setFont(font)

    def _apply_all_row_styling(self):
        """Apply styling to all rows based on whether they are parent or sub-rows."""
        table = self.discoveredFilesTable
        for row in range(table.rowCount()):
            if row < len(self._master_file_list):
                is_sub = self._master_file_list[row].get('is_sub_row', False)
                self._apply_row_styling(row, is_sub_row=is_sub)

        # Also clean parent rows that have sub-rows
        self._clean_parent_rows_with_subrows()

    def _clean_parent_rows_with_subrows(self):
        """Clear channel-specific columns from parent rows that have sub-rows."""
        from PyQt6.QtWidgets import QTableWidgetItem

        table = self.discoveredFilesTable

        # Find parent rows that have sub-rows
        parent_to_subrows = {}  # parent_file_path -> [sub_row_indices]

        for row, task in enumerate(self._master_file_list):
            if task.get('is_sub_row'):
                parent_path = str(task.get('file_path', ''))
                if parent_path not in parent_to_subrows:
                    parent_to_subrows[parent_path] = []
                parent_to_subrows[parent_path].append(row)

        # Find parent row for each file that has sub-rows
        for row, task in enumerate(self._master_file_list):
            if task.get('is_sub_row'):
                continue

            file_path = str(task.get('file_path', ''))
            if file_path in parent_to_subrows:
                sub_rows = parent_to_subrows[file_path]
                num_analyzed = len(sub_rows)

                # Clear channel-specific columns from parent row
                # Columns: 5-Channel, 6-Stim Ch, 7-Events Ch, 12-Animal ID
                # Note: Column 4 (Keywords) is parent-level info, don't clear
                if row < table.rowCount():
                    table.setItem(row, 5, QTableWidgetItem(''))   # Channel
                    table.setItem(row, 6, QTableWidgetItem(''))   # Stim Ch
                    table.setItem(row, 7, QTableWidgetItem(''))   # Events Ch
                    table.setItem(row, 12, QTableWidgetItem(''))  # Animal ID

                    # Show aggregate status in Status column (13)
                    # Count completed sub-rows
                    completed = sum(1 for sr in sub_rows
                                    if self._master_file_list[sr].get('status') == 'completed')
                    status_text = f"{completed} ✓" if completed > 0 else ''
                    table.setItem(row, 13, QTableWidgetItem(status_text))

                    # Clear Exports column (15) for parent - exports are per-channel
                    table.setItem(row, 15, QTableWidgetItem(''))

                # Also clear these fields in the task dict
                task['channel'] = ''
                task['stim_channel'] = ''
                task['events_channel'] = ''
                task['animal_id'] = ''
                task['status'] = ''  # Parent doesn't have a single status
                task['exports'] = {}

    def _add_row_action_button(self, row):
        """Add Analyze and '+' buttons to the actions column."""
        from PyQt6.QtWidgets import QPushButton, QWidget, QHBoxLayout

        # Create container widget
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Add 'Analyze' button (opens file in Analysis tab)
        analyze_btn = QPushButton("▶")
        analyze_btn.setMaximumWidth(22)
        analyze_btn.setMaximumHeight(22)
        analyze_btn.setToolTip("Open this file in the Analysis tab")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        analyze_btn.clicked.connect(lambda checked, r=row: self._on_analyze_row(r))
        layout.addWidget(analyze_btn)

        # Add '+' button (add another row for different channel/animal)
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(22)
        add_btn.setMaximumHeight(22)
        add_btn.setToolTip("Add another row for this file (different channel/animal)")
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        add_btn.clicked.connect(lambda checked, r=row: self._on_add_row_for_file(r))
        layout.addWidget(add_btn)

        self.discoveredFilesTable.setCellWidget(row, 15, container)

    def _on_analyze_row(self, row):
        """Open the file from the specified row in the Analysis tab."""
        if row >= len(self._master_file_list):
            return

        task = self._master_file_list[row]
        file_path = task.get('file_path')

        if not file_path or not Path(file_path).exists():
            self._show_warning("File Not Found", f"Cannot find file:\n{file_path}")
            return

        print(f"[master-list] Analyzing file from row {row}: {file_path}")

        # Track which row is being analyzed
        self._active_master_list_row = row

        # Store pending channel selections if specified
        self._pending_analysis_channel = task.get('channel', '')
        self._pending_stim_channels = task.get('stim_channels', [])

        # Load the file
        self.load_file(Path(file_path))

        # Switch to Analysis tab
        if hasattr(self, 'mainTabWidget'):
            self.mainTabWidget.setCurrentIndex(0)

        # Update status to "in progress"
        task['status'] = 'in_progress'
        table = self.discoveredFilesTable
        if row < table.rowCount():
            status_item = table.item(row, 14)
            if status_item:
                status_item.setText('◐')  # Half-filled circle for in-progress
            else:
                from PyQt6.QtWidgets import QTableWidgetItem
                table.setItem(row, 14, QTableWidgetItem('◐'))

    def _on_add_row_for_file(self, source_row, force_override=False):
        """Add a new row for the same file (for multi-channel/multi-animal analysis)."""
        if source_row >= len(self._master_file_list):
            return

        source_task = self._master_file_list[source_row]
        table = self.discoveredFilesTable
        file_path = source_task.get('file_path')

        # Find the parent task (may be source_task or a different row)
        parent_task = source_task
        if source_task.get('is_sub_row'):
            # Find the actual parent row
            for task in self._master_file_list:
                if not task.get('is_sub_row') and str(task.get('file_path')) == str(file_path):
                    parent_task = task
                    break

        # Count existing SUB-ROWS for this file (parent row doesn't count as an analysis)
        existing_subrows = []
        for i, task in enumerate(self._master_file_list):
            if task.get('is_sub_row') and str(task.get('file_path')) == str(file_path):
                existing_subrows.append(i)

        # Calculate available analysis channels from PARENT task (has file metadata)
        total_channels = parent_task.get('channel_count', 0)

        # Get stim channels from file metadata
        stim_channels_from_metadata = parent_task.get('stim_channels', [])
        if not isinstance(stim_channels_from_metadata, list):
            stim_channels_from_metadata = [stim_channels_from_metadata] if stim_channels_from_metadata else []

        # Also check existing sub-rows for stim/events channels they use
        stim_events_from_subrows = set()
        for i in existing_subrows:
            task = self._master_file_list[i]
            stim_ch = task.get('stim_channel', '')
            events_ch = task.get('events_channel', '')
            if stim_ch:
                stim_events_from_subrows.add(stim_ch)
            if events_ch:
                stim_events_from_subrows.add(events_ch)

        # Combine all non-analysis channels
        all_stim_channels = set(stim_channels_from_metadata) | stim_events_from_subrows
        num_stim_channels = len(all_stim_channels)
        available_analysis_channels = total_channels - num_stim_channels

        # Check if we can add more rows (compare sub-row count to available channels)
        if len(existing_subrows) >= available_analysis_channels and not force_override:
            from PyQt6.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setWindowTitle("Channel Limit Reached")
            msg.setText(
                f"This file has {total_channels} channels.\n"
                f"Detected {num_stim_channels} stim/events channel(s): {', '.join(sorted(all_stim_channels)) if all_stim_channels else 'none'}\n"
                f"Maximum {available_analysis_channels} analysis rows allowed.\n"
                f"Already have {len(existing_subrows)} analyzed channel(s)."
            )
            msg.setInformativeText("Do you want to add a row anyway?")
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg.setDefaultButton(QMessageBox.StandardButton.No)
            if msg.exec() != QMessageBox.StandardButton.Yes:
                return
            # User clicked Yes - force override
            force_override = True

        # Find the next available channel (use parent_task for channel metadata)
        channel_names = parent_task.get('channel_names', [])
        if not channel_names:
            # Generate default channel names
            channel_names = [f"AD{i}" for i in range(total_channels)]

        # Filter out stim/events channels to get analysis channels
        analysis_channels = [ch for ch in channel_names if ch not in all_stim_channels]

        # Find which channels are already used (from sub-rows only)
        used_channels = set()
        for i in existing_subrows:
            ch = self._master_file_list[i].get('channel', '')
            if ch:
                used_channels.add(ch)

        # Get next available channel
        next_channel = ''
        for ch in analysis_channels:
            if ch not in used_channels:
                next_channel = ch
                break

        # If no analysis channels available but user forced override, pick any unused channel
        if not next_channel and force_override:
            for ch in channel_names:
                if ch not in used_channels:
                    next_channel = ch
                    break

        # Create a new task based on parent task (for file metadata) and source task (for user metadata)
        new_task = {
            'file_path': file_path,
            'file_name': parent_task.get('file_name', ''),
            'protocol': parent_task.get('protocol', ''),
            'channel_count': total_channels,
            'sweep_count': parent_task.get('sweep_count', 0),
            'channel_names': channel_names,
            'stim_channels': stim_channels,
            'path_keywords': parent_task.get('path_keywords', {}),
            'keywords_display': parent_task.get('keywords_display', ''),
            # Auto-populated channel
            'channel': next_channel,
            'stim_channel': '',  # User can set this
            'events_channel': '',  # Filled on save
            # Copy metadata from source (may have user edits)
            'strain': source_task.get('strain', ''),
            'stim_type': source_task.get('stim_type', ''),
            'power': source_task.get('power', ''),
            'sex': source_task.get('sex', ''),
            'animal_id': '',  # Different animal - leave blank
            'status': 'pending',
            'is_sub_row': True,
            'parent_file': file_path,
            # Export tracking (filled on save)
            'export_path': '',
            'export_date': '',
            'export_version': '',
            'exports': {
                'npz': False,
                'timeseries_csv': False,
                'breaths_csv': False,
                'events_csv': False,
                'pdf': False,
                'session_state': False,
                'ml_training': False,
            }
        }

        # Insert after the source row
        insert_row = source_row + 1
        self._master_file_list.insert(insert_row, new_task)

        # Block signals while inserting
        table.blockSignals(True)

        # Insert table row with sub-row format (channel in col 0, not filename)
        table.insertRow(insert_row)
        table.setItem(insert_row, 0, QTableWidgetItem(f"  ↳ {next_channel}"))  # Show channel, not filename
        table.setItem(insert_row, 1, QTableWidgetItem(''))  # Leave protocol empty on sub-rows
        table.setItem(insert_row, 2, QTableWidgetItem(''))  # Leave channel count empty on sub-rows
        table.setItem(insert_row, 3, QTableWidgetItem(''))  # Leave sweep count empty on sub-rows
        table.setItem(insert_row, 4, QTableWidgetItem(''))  # Leave keywords empty on sub-rows
        table.setItem(insert_row, 5, QTableWidgetItem(next_channel))  # Auto-populated channel
        table.setItem(insert_row, 6, QTableWidgetItem(''))  # Stim Ch - filled on save
        table.setItem(insert_row, 7, QTableWidgetItem(''))  # Events Ch - filled on save
        table.setItem(insert_row, 8, QTableWidgetItem(new_task['strain']))
        table.setItem(insert_row, 9, QTableWidgetItem(new_task['stim_type']))
        table.setItem(insert_row, 10, QTableWidgetItem(new_task['power']))
        table.setItem(insert_row, 11, QTableWidgetItem(new_task['sex']))
        table.setItem(insert_row, 12, QTableWidgetItem(''))  # Animal ID - different
        table.setItem(insert_row, 13, QTableWidgetItem('○'))  # Status

        # Add action buttons (with remove option for sub-rows)
        self._add_sub_row_action_buttons(insert_row)
        # Column 15: Exports (empty initially)
        table.setItem(insert_row, 15, self._create_exports_table_item(new_task))

        table.blockSignals(False)

        remaining = available_analysis_channels - len(existing_rows) - 1
        print(f"[master-list] Added sub-row at {insert_row} with channel {next_channel} ({remaining} more available)")

    def _add_sub_row_action_buttons(self, row):
        """Add action buttons for sub-rows (Analyze, + and - buttons)."""
        from PyQt6.QtWidgets import QPushButton, QWidget, QHBoxLayout

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        # Add 'Analyze' button
        analyze_btn = QPushButton("▶")
        analyze_btn.setMaximumWidth(20)
        analyze_btn.setMaximumHeight(20)
        analyze_btn.setToolTip("Open this file in the Analysis tab")
        analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        analyze_btn.clicked.connect(lambda checked, r=row: self._on_analyze_row(r))
        layout.addWidget(analyze_btn)

        # Add '-' button (remove sub-row)
        remove_btn = QPushButton("-")
        remove_btn.setMaximumWidth(20)
        remove_btn.setMaximumHeight(20)
        remove_btn.setToolTip("Remove this row")
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        remove_btn.clicked.connect(lambda checked, r=row: self._on_remove_sub_row(r))
        layout.addWidget(remove_btn)

        self.discoveredFilesTable.setCellWidget(row, 15, container)

    def _on_remove_sub_row(self, row):
        """Remove a sub-row from the master list."""
        if row >= len(self._master_file_list):
            return

        task = self._master_file_list[row]

        # Only allow removing sub-rows, not primary rows
        if not task.get('is_sub_row', False):
            self._show_warning("Cannot Remove",
                "Cannot remove the primary row for a file.\n"
                "Use 'Omit' to hide files you don't want to analyze.")
            return

        # Confirm deletion if this row has been analyzed
        if task.get('status') == 'completed':
            from PyQt6.QtWidgets import QMessageBox

            # Get export info for confirmation message
            export_summary = self._format_exports_summary(task)
            export_path = task.get('export_path', '')
            channel = task.get('channel', 'unknown channel')

            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("Delete Analyzed Row?")
            msg.setText(f"This row ({channel}) has been analyzed.")

            details = []
            if export_summary:
                details.append(f"Exports: {export_summary}")
            if export_path:
                details.append(f"Saved to: {export_path}")
            if details:
                msg.setInformativeText("\n".join(details) + "\n\nNote: The exported files will NOT be deleted.")

            msg.setDetailedText(
                "This will remove the row from the Project Builder list only.\n\n"
                "The exported data files on disk remain unchanged. "
                "You can add a new row and re-analyze if needed."
            )
            msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            msg.setDefaultButton(QMessageBox.StandardButton.Cancel)

            if msg.exec() != QMessageBox.StandardButton.Yes:
                return  # User cancelled

        # Remove from data and table
        table = self.discoveredFilesTable
        table.blockSignals(True)

        del self._master_file_list[row]
        table.removeRow(row)

        table.blockSignals(False)

        print(f"[master-list] Removed sub-row at {row}")

    def _on_master_list_context_menu(self, position):
        """Show context menu for bulk editing selected rows."""
        from PyQt6.QtWidgets import QMenu, QInputDialog
        from PyQt6.QtGui import QAction

        table = self.discoveredFilesTable
        selected_rows = set(index.row() for index in table.selectedIndexes())

        if not selected_rows:
            return

        menu = QMenu(self)

        # Bulk edit options
        edit_menu = menu.addMenu(f"Set for {len(selected_rows)} selected rows")

        # Experiment (column 5) - with history suggestions
        experiment_menu = edit_menu.addMenu("Set Experiment")
        exp_history = self._get_experiment_history()
        if exp_history:
            for exp_name in exp_history[:10]:  # Show top 10
                exp_action = QAction(exp_name, self)
                exp_action.triggered.connect(lambda checked, e=exp_name: self._bulk_set_column(selected_rows, 5, 'experiment', "Set Experiment", e))
                experiment_menu.addAction(exp_action)
            experiment_menu.addSeparator()
        custom_exp_action = QAction("Custom...", self)
        custom_exp_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 5, 'experiment', "Set Experiment"))
        experiment_menu.addAction(custom_exp_action)

        # Strain (column 9)
        strain_action = QAction("Set Strain...", self)
        strain_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 9, 'strain', "Set Strain"))
        edit_menu.addAction(strain_action)

        # Stim Type (column 10)
        stim_action = QAction("Set Stim Type...", self)
        stim_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 10, 'stim_type', "Set Stim Type"))
        edit_menu.addAction(stim_action)

        # Power (column 11)
        power_action = QAction("Set Power...", self)
        power_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 11, 'power', "Set Power"))
        edit_menu.addAction(power_action)

        # Sex (column 12)
        sex_action = QAction("Set Sex...", self)
        sex_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 12, 'sex', "Set Sex"))
        edit_menu.addAction(sex_action)

        # Animal ID (column 13)
        animal_action = QAction("Set Animal ID...", self)
        animal_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 13, 'animal_id', "Set Animal ID"))
        edit_menu.addAction(animal_action)

        # Channel (column 6)
        channel_action = QAction("Set Channel...", self)
        channel_action.triggered.connect(lambda: self._bulk_set_column(selected_rows, 6, 'channel', "Set Channel"))
        edit_menu.addAction(channel_action)

        menu.addSeparator()

        # Combine files option (if multiple files selected)
        if len(selected_rows) > 1:
            combine_action = QAction(f"Combine {len(selected_rows)} files for analysis...", self)
            combine_action.triggered.connect(lambda: self._combine_selected_files(selected_rows))
            menu.addAction(combine_action)

        # Check if any selected rows have scan warnings (conflicts or multiple NPZ files)
        rows_with_warnings = []
        rows_with_conflicts = []
        for row in selected_rows:
            if row < len(self._master_file_list):
                task = self._master_file_list[row]
                warnings = task.get('scan_warnings', {})
                if warnings:
                    rows_with_warnings.append(row)
                    if warnings.get('conflicts'):
                        rows_with_conflicts.append(row)

        if rows_with_warnings:
            menu.addSeparator()
            conflict_menu = menu.addMenu(f"⚠ Resolve Warnings ({len(rows_with_warnings)} rows)")

            if rows_with_conflicts:
                # Use NPZ values - overwrite table with NPZ data
                use_npz_action = QAction("Use NPZ values (overwrite table)", self)
                use_npz_action.triggered.connect(lambda: self._resolve_conflicts_use_npz(rows_with_conflicts))
                conflict_menu.addAction(use_npz_action)

                # Keep table values - dismiss the warning
                keep_table_action = QAction("Keep table values (dismiss warning)", self)
                keep_table_action.triggered.connect(lambda: self._resolve_conflicts_keep_table(rows_with_warnings))
                conflict_menu.addAction(keep_table_action)

                conflict_menu.addSeparator()

            # View details
            view_details_action = QAction("View warning details...", self)
            view_details_action.triggered.connect(lambda: self._show_conflict_details(rows_with_warnings))
            conflict_menu.addAction(view_details_action)

            # Clear all warnings
            clear_action = QAction("Clear all warnings", self)
            clear_action.triggered.connect(lambda: self._clear_scan_warnings(rows_with_warnings))
            conflict_menu.addAction(clear_action)

        menu.exec(table.viewport().mapToGlobal(position))

    def _bulk_set_column(self, rows, column, field_name, dialog_title, preset_value=None):
        """Set a column value for multiple rows at once.

        Args:
            rows: Set of row indices to update
            column: Column index in the table
            field_name: Field name in the task dict
            dialog_title: Title for the input dialog
            preset_value: If provided, use this value without showing dialog
        """
        from PyQt6.QtWidgets import QInputDialog, QTableWidgetItem

        if preset_value is not None:
            value = preset_value
            ok = True
        else:
            # Get existing values to suggest
            existing_values = set()
            for row in rows:
                if row < len(self._master_file_list):
                    val = self._master_file_list[row].get(field_name, '')
                    if val:
                        existing_values.add(val)

            # Default to first existing value or empty
            default_value = list(existing_values)[0] if existing_values else ''

            value, ok = QInputDialog.getText(
                self, dialog_title,
                f"Enter value for {len(rows)} rows:",
                text=default_value
            )

        if ok:
            table = self.discoveredFilesTable
            table.blockSignals(True)

            for row in rows:
                if row < len(self._master_file_list):
                    self._master_file_list[row][field_name] = value
                if row < table.rowCount():
                    item = table.item(row, column)
                    if item:
                        item.setText(value)
                    else:
                        table.setItem(row, column, QTableWidgetItem(value))

            table.blockSignals(False)

            # Update experiment history if this is the experiment field
            if field_name == 'experiment' and value:
                self._update_experiment_history(value)

            print(f"[master-list] Bulk set {field_name} = '{value}' for {len(rows)} rows")

    def _resolve_conflicts_use_npz(self, rows):
        """Resolve conflicts by overwriting table values with NPZ values."""
        from PyQt6.QtWidgets import QTableWidgetItem

        table = self.discoveredFilesTable
        table.blockSignals(True)

        resolved_count = 0
        for row in rows:
            if row >= len(self._master_file_list):
                continue

            task = self._master_file_list[row]
            warnings = task.get('scan_warnings', {})
            conflicts = warnings.get('conflicts', [])

            if not conflicts:
                continue

            # Parse conflicts and apply NPZ values
            # Conflicts are in format: "Field: table='X' vs NPZ='Y'"
            for conflict in conflicts:
                try:
                    # Extract field name and NPZ value
                    field_part = conflict.split(':')[0].strip()
                    npz_part = conflict.split("NPZ='")[1].split("'")[0]

                    # Map display name to field and column
                    field_map = {
                        'Strain': ('strain', 8),
                        'Stim Type': ('stim_type', 9),
                        'Power': ('power', 10),
                        'Sex': ('sex', 11),
                        'Animal ID': ('animal_id', 12),
                    }

                    if field_part in field_map:
                        field_name, column = field_map[field_part]
                        task[field_name] = npz_part
                        if row < table.rowCount():
                            table.setItem(row, column, QTableWidgetItem(npz_part))
                        print(f"[conflict-resolve] Row {row}: Set {field_name} = '{npz_part}' from NPZ")
                except Exception as e:
                    print(f"[conflict-resolve] Error parsing conflict '{conflict}': {e}")

            # Clear the warnings after resolving
            task.pop('scan_warnings', None)
            resolved_count += 1

            # Update status icon
            self._update_row_status_icon(row, task)

        table.blockSignals(False)
        self._log_status_message(f"✓ Resolved conflicts for {resolved_count} rows (used NPZ values)", 3000)
        self._update_resolve_conflicts_button()

    def _resolve_conflicts_keep_table(self, rows):
        """Resolve conflicts by keeping table values and dismissing warnings."""
        resolved_count = 0
        for row in rows:
            if row >= len(self._master_file_list):
                continue

            task = self._master_file_list[row]
            if task.get('scan_warnings'):
                task.pop('scan_warnings', None)
                resolved_count += 1
                self._update_row_status_icon(row, task)

        self._log_status_message(f"✓ Dismissed warnings for {resolved_count} rows (kept table values)", 3000)
        self._update_resolve_conflicts_button()

    def _clear_scan_warnings(self, rows):
        """Clear all scan warnings for selected rows."""
        cleared_count = 0
        for row in rows:
            if row >= len(self._master_file_list):
                continue

            task = self._master_file_list[row]
            if task.get('scan_warnings'):
                task.pop('scan_warnings', None)
                cleared_count += 1
                self._update_row_status_icon(row, task)

        self._log_status_message(f"✓ Cleared warnings for {cleared_count} rows", 3000)
        self._update_resolve_conflicts_button()

    def _show_conflict_details(self, rows):
        """Show a dialog with detailed conflict information."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Scan Warning Details")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        # Build details text
        details = []
        for row in rows:
            if row >= len(self._master_file_list):
                continue

            task = self._master_file_list[row]
            warnings = task.get('scan_warnings', {})
            if not warnings:
                continue

            file_name = task.get('file_name', 'Unknown')
            channel = task.get('channel', '')
            details.append(f"═══ Row {row}: {file_name} ({channel}) ═══")

            conflicts = warnings.get('conflicts', [])
            if conflicts:
                details.append("⚠ Data Conflicts:")
                for c in conflicts:
                    details.append(f"   • {c}")

            older_files = warnings.get('older_npz_files', [])
            if older_files:
                details.append(f"📁 {len(older_files)} Older NPZ File(s):")
                for older in older_files:
                    details.append(f"   • {Path(older['file']).name}")
                    details.append(f"     Date: {older['date']}")
                    details.append(f"     Path: {older['file']}")

            details.append("")

        text_edit.setText('\n'.join(details) if details else "No warning details found.")
        layout.addWidget(text_edit)

        # Buttons
        btn_layout = QHBoxLayout()

        btn_use_npz = QPushButton("Use NPZ Values")
        btn_use_npz.clicked.connect(lambda: (self._resolve_conflicts_use_npz(rows), dialog.accept()))
        btn_layout.addWidget(btn_use_npz)

        btn_keep_table = QPushButton("Keep Table Values")
        btn_keep_table.clicked.connect(lambda: (self._resolve_conflicts_keep_table(rows), dialog.accept()))
        btn_layout.addWidget(btn_keep_table)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)
        dialog.exec()

    def _update_row_status_icon(self, row, task):
        """Update the status icon for a row based on its current state."""
        from PyQt6.QtWidgets import QTableWidgetItem

        table = self.discoveredFilesTable
        if row >= table.rowCount():
            return

        is_sub_row = task.get('is_sub_row', False)
        status = task.get('status', 'pending')
        warnings = task.get('scan_warnings', {})
        has_conflicts = bool(warnings.get('conflicts'))
        has_older_files = warnings.get('older_npz_count', 0) > 0

        if is_sub_row:
            if status == 'completed':
                if has_conflicts:
                    status_icon = '●⚠'
                elif has_older_files:
                    status_icon = '●📁'
                else:
                    status_icon = '●'
            elif status == 'in_progress':
                status_icon = '◐'
            else:
                status_icon = '○'

            status_item = QTableWidgetItem(status_icon)

            # Add tooltip for warnings
            if has_conflicts or has_older_files:
                tooltip_parts = []
                if has_conflicts:
                    tooltip_parts.append("⚠ Data conflicts (table vs NPZ):")
                    for c in warnings['conflicts']:
                        tooltip_parts.append(f"  • {c}")
                if has_older_files:
                    count = warnings['older_npz_count']
                    tooltip_parts.append(f"📁 {count} older NPZ file(s) found")
                status_item.setToolTip('\n'.join(tooltip_parts))

            table.setItem(row, 13, status_item)

    def _combine_selected_files(self, rows):
        """Combine selected files for multi-file analysis."""
        from PyQt6.QtWidgets import QMessageBox

        # Get file paths for selected rows
        file_paths = []
        for row in sorted(rows):
            if row < len(self._master_file_list):
                task = self._master_file_list[row]
                file_path = task.get('file_path')
                if file_path and Path(file_path).exists():
                    file_paths.append(Path(file_path))

        if len(file_paths) < 2:
            self._show_warning("Cannot Combine", "Need at least 2 valid files to combine.")
            return

        # Confirm with user
        file_names = [p.name for p in file_paths]
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Combine Files")
        msg.setText(f"Combine {len(file_paths)} files for analysis?")
        msg.setInformativeText("Files will be concatenated in order:\n• " + "\n• ".join(file_names[:5]))
        if len(file_names) > 5:
            msg.setInformativeText(msg.informativeText() + f"\n... and {len(file_names) - 5} more")
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if msg.exec() == QMessageBox.StandardButton.Yes:
            # Load files using multi-file loader
            print(f"[master-list] Combining {len(file_paths)} files for analysis")
            self.load_multiple_files(file_paths)
            # Switch to Analysis tab
            if hasattr(self, 'mainTabWidget'):
                self.mainTabWidget.setCurrentIndex(0)

    def on_project_browse_directory(self):
        """Browse for directory containing recordings."""
        from PyQt6.QtWidgets import QFileDialog

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Recordings",
            "",
            QFileDialog.Option.ShowDirsOnly
        )

        if directory:
            self._project_directory = directory
            self.directoryPathEdit.setText(directory)
            self._log_status_message(f"Selected directory: {directory}", 2000)
            print(f"[project-builder] Selected directory: {directory}")

    def on_project_scan_files(self):
        """Scan directory for new files - additive mode, preserves existing data."""
        if not self._project_directory:
            self._show_warning("No Directory", "Please select a directory first using 'Browse Directory'.")
            return

        from core import project_builder
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import QThread, pyqtSignal

        # Prevent multiple concurrent scans
        if hasattr(self, '_metadata_thread') and self._metadata_thread and self._metadata_thread.isRunning():
            self._show_warning("Scan In Progress", "A scan is already running. Please wait for it to complete.")
            return

        progress = None
        try:
            # Disable scan button during operation
            self.scanFilesButton.setEnabled(False)

            recursive = self.recursiveCheckBox.isChecked()

            # PHASE 1: Quick file discovery
            progress = QProgressDialog("Finding files...", "Cancel", 0, 0, self)
            progress.setWindowTitle("Scanning Directory")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()
            QApplication.processEvents()

            files = project_builder.discover_files(self._project_directory, recursive=recursive)
            abf_files = files['abf_files']
            excel_files = files['excel_files']

            if progress.wasCanceled():
                progress.close()
                self.scanFilesButton.setEnabled(True)
                self._log_status_message("Scan cancelled", 2000)
                return

            progress.close()

            # PHASE 2: Build set of existing file paths to avoid duplicates
            # Use string paths for comparison (normalized)
            existing_paths = set()
            for task in self._master_file_list:
                fp = task.get('file_path')
                if fp:
                    # Normalize path for comparison
                    existing_paths.add(str(Path(fp).resolve()))

            # Find new files only
            new_abf_files = []
            for abf_path in abf_files:
                normalized = str(abf_path.resolve())
                if normalized not in existing_paths:
                    new_abf_files.append(abf_path)

            if not new_abf_files:
                self._log_status_message(f"No new files found ({len(abf_files)} files already in list)", 3000)
                self.scanFilesButton.setEnabled(True)
                return

            # Extract path keywords for each new file
            from core.fast_abf_reader import extract_path_keywords

            table = self.discoveredFilesTable
            start_row = table.rowCount()  # Insert after existing rows
            new_files_data = []

            for i, abf_path in enumerate(new_abf_files):
                file_size_mb = abf_path.stat().st_size / (1024 * 1024)

                # Extract keywords from path
                path_info = extract_path_keywords(abf_path, Path(self._project_directory))

                # Format keywords for display
                keywords_display = []
                if path_info['power_levels']:
                    keywords_display.extend(path_info['power_levels'])
                if path_info['animal_ids']:
                    keywords_display.extend([f"ID:{id}" for id in path_info['animal_ids']])
                if path_info['keywords']:
                    keywords_display.extend(path_info['keywords'])

                # Auto-fill power and animal ID from path keywords
                power_auto = path_info['power_levels'][0] if path_info['power_levels'] else ''
                animal_id_auto = path_info['animal_ids'][0] if path_info['animal_ids'] else ''

                file_info = {
                    'file_path': abf_path,
                    'file_name': abf_path.name,
                    'protocol': '...',
                    'channel_count': 0,
                    'sweep_count': 0,
                    'stim_channels': [],  # Detected stimulus channels (e.g., ['AD1', 'AD2'])
                    'stim_frequency': '',  # Detected stim frequency (e.g., '30Hz')
                    'path_keywords': path_info,
                    'keywords_display': ', '.join(keywords_display) if keywords_display else '',
                    # Metadata fields (filled on save)
                    'channel': '',  # Analyzed channel - filled on save
                    'stim_channel': '',  # Stim channel used - filled on save
                    'events_channel': '',  # Events channel used - filled on save
                    'strain': '',
                    'stim_type': '',  # Auto-filled from stim_frequency
                    'power': power_auto,
                    'sex': '',
                    'animal_id': animal_id_auto,
                    'status': 'pending',
                    # Export tracking (filled on save)
                    'export_path': '',
                    'export_date': '',
                    'export_version': '',
                    'exports': {
                        'npz': False,
                        'timeseries_csv': False,
                        'breaths_csv': False,
                        'events_csv': False,
                        'pdf': False,
                        'session_state': False,
                        'ml_training': False,
                    }
                }
                new_files_data.append(file_info)
                self._master_file_list.append(file_info)

                # Insert new table row
                row_idx = start_row + i
                table.insertRow(row_idx)
                table.setItem(row_idx, 0, QTableWidgetItem(file_info['file_name']))  # File Name
                table.setItem(row_idx, 1, QTableWidgetItem('Loading...'))  # Protocol
                table.setItem(row_idx, 2, QTableWidgetItem(''))  # Avail Ch (loaded later)
                table.setItem(row_idx, 3, QTableWidgetItem(''))  # Sweeps (loaded later)
                table.setItem(row_idx, 4, QTableWidgetItem(file_info['keywords_display']))  # Keywords
                table.setItem(row_idx, 5, QTableWidgetItem(''))  # Channel (filled on save)
                table.setItem(row_idx, 6, QTableWidgetItem(''))  # Stim Ch (filled on save)
                table.setItem(row_idx, 7, QTableWidgetItem(''))  # Events Ch (filled on save)
                table.setItem(row_idx, 8, QTableWidgetItem(''))  # Strain
                table.setItem(row_idx, 9, QTableWidgetItem(''))  # Stim Type (auto-detected or manual)
                table.setItem(row_idx, 10, QTableWidgetItem(power_auto))  # Power (auto-filled)
                table.setItem(row_idx, 11, QTableWidgetItem(''))  # Sex
                table.setItem(row_idx, 12, QTableWidgetItem(animal_id_auto))  # Animal ID (auto-filled)
                table.setItem(row_idx, 13, QTableWidgetItem('○'))  # Status (pending)
                # Column 14: Actions buttons
                self._add_row_action_button(row_idx)
                # Column 15: Exports (empty initially)
                table.setItem(row_idx, 15, self._create_exports_table_item(file_info))

            table.resizeColumnsToContents()

            # Update summary - show total files in list
            total_files = len(self._master_file_list)
            summary_text = f"Summary: {total_files} ABF files total | Added {len(new_abf_files)} new files | Loading protocols..."
            self.summaryLabel.setText(summary_text)
            self._log_status_message(f"Added {len(new_abf_files)} new files, loading metadata...", 3000)

            # Store reference to new files data for background metadata update
            self._discovered_files_data = new_files_data

            # PHASE 3: Load metadata in background thread (only for new files)
            self._start_background_metadata_loading(new_abf_files, start_row_offset=start_row)

        except Exception as e:
            if progress:
                progress.close()
            self.scanFilesButton.setEnabled(True)  # Re-enable on error
            self._show_error("Scan Error", f"Failed to scan directory:\n{e}")
            print(f"[project-builder] Error: {e}")
            import traceback
            traceback.print_exc()

    def _start_background_metadata_loading(self, abf_files, start_row_offset=0):
        """Start background thread to load metadata using parallel processing.

        Args:
            abf_files: List of ABF file paths to load
            start_row_offset: Row offset for updating table (for additive scans)
        """
        from PyQt6.QtCore import QThread, pyqtSignal

        # Store the row offset for use in the update callback
        self._metadata_row_offset = start_row_offset

        class MetadataThread(QThread):
            # Batch updates: send list of results instead of individual items
            batch_progress = pyqtSignal(list, int)  # [(index, metadata), ...], total
            finished = pyqtSignal(set)  # protocols

            def __init__(self, files):
                super().__init__()
                self.files = files
                self.should_stop = False

            def run(self):
                from core.fast_abf_reader import read_abf_metadata_parallel
                protocols = set()
                batch = []
                batch_size = 25  # Update UI every 25 files to reduce signal traffic

                def callback(index, total, metadata):
                    if self.should_stop:
                        return

                    if metadata:
                        protocols.add(metadata['protocol'])

                    # Collect results in batches
                    batch.append((index, metadata))

                    # Emit batch when it reaches batch_size or at the end
                    if len(batch) >= batch_size or index == total - 1:
                        self.batch_progress.emit(batch[:], total)  # Send copy of batch
                        batch.clear()

                try:
                    # Use parallel processing with 4 workers
                    read_abf_metadata_parallel(self.files, progress_callback=callback, max_workers=4)
                    self.finished.emit(protocols)
                except Exception as e:
                    print(f"[project-builder] Error during parallel loading: {e}")
                    import traceback
                    traceback.print_exc()
                    self.finished.emit(protocols)  # Still emit finish signal

        # Show progress bar
        self.projectProgressBar.setVisible(True)
        self.projectProgressBar.setValue(0)
        self.projectProgressBar.setFormat(f"Loading metadata: 0/{len(abf_files)} (0%)")

        self._metadata_thread = MetadataThread(abf_files)
        self._metadata_thread.batch_progress.connect(self._update_file_metadata_batch)
        self._metadata_thread.finished.connect(self._metadata_finished)
        self._metadata_thread.start()
        print(f"[project-builder] Started background loading for {len(abf_files)} files (offset={start_row_offset})")

    def _update_file_metadata_batch(self, batch, total):
        """Update table cells with a batch of loaded metadata (called from main thread via signal)."""
        table = self.discoveredFilesTable

        # Get the row offset (for additive scans)
        row_offset = getattr(self, '_metadata_row_offset', 0)

        # Process all items in the batch
        for index, metadata in batch:
            if metadata:
                # Update internal data structures
                if index < len(self._discovered_files_data):
                    self._discovered_files_data[index]['protocol'] = metadata['protocol']
                    self._discovered_files_data[index]['channel_count'] = metadata.get('channel_count', 0)
                    self._discovered_files_data[index]['sweep_count'] = metadata.get('sweep_count', 0)
                    self._discovered_files_data[index]['stim_channels'] = metadata.get('stim_channels', [])
                    self._discovered_files_data[index]['stim_frequency'] = metadata.get('stim_frequency', '')
                    # Auto-fill stim_type from detected frequency
                    if metadata.get('stim_frequency') and not self._discovered_files_data[index].get('stim_type'):
                        self._discovered_files_data[index]['stim_type'] = metadata.get('stim_frequency')

                # Master list index includes the offset
                master_idx = row_offset + index
                if master_idx < len(self._master_file_list):
                    self._master_file_list[master_idx]['protocol'] = metadata['protocol']
                    self._master_file_list[master_idx]['channel_count'] = metadata.get('channel_count', 0)
                    self._master_file_list[master_idx]['sweep_count'] = metadata.get('sweep_count', 0)
                    self._master_file_list[master_idx]['stim_channels'] = metadata.get('stim_channels', [])
                    self._master_file_list[master_idx]['stim_frequency'] = metadata.get('stim_frequency', '')
                    # Auto-fill stim_type from detected frequency
                    if metadata.get('stim_frequency') and not self._master_file_list[master_idx].get('stim_type'):
                        self._master_file_list[master_idx]['stim_type'] = metadata.get('stim_frequency')

                # Update table cells - use row_offset for correct table row
                # Col 1: Protocol, Col 2: Avail Ch, Col 3: Sweeps, Col 7: Stim Ch, Col 10: Stim Type
                table_row = row_offset + index
                if table_row < table.rowCount():
                    table.setItem(table_row, 1, QTableWidgetItem(metadata['protocol']))
                    table.setItem(table_row, 2, QTableWidgetItem(str(metadata.get('channel_count', ''))))
                    table.setItem(table_row, 3, QTableWidgetItem(str(metadata.get('sweep_count', ''))))
                    # Stim channels (column 7) - only update if row hasn't been analyzed yet
                    current_stim = table.item(table_row, 7)
                    if not current_stim or not current_stim.text():
                        stim_ch_list = metadata.get('stim_channels', [])
                        stim_ch_display = ', '.join(stim_ch_list) if stim_ch_list else ''
                        table.setItem(table_row, 7, QTableWidgetItem(stim_ch_display))
                    # Stim type auto-detection (column 10) - only if not already set
                    stim_freq = metadata.get('stim_frequency', '')
                    if stim_freq:
                        current_stim_type = table.item(table_row, 10)
                        if not current_stim_type or not current_stim_type.text():
                            table.setItem(table_row, 10, QTableWidgetItem(stim_freq))

        # Update progress bar and status (use last item in batch for progress)
        if batch:
            last_index = batch[-1][0]
            progress_pct = int((last_index + 1) / total * 100)
            self.projectProgressBar.setValue(progress_pct)
            self.projectProgressBar.setFormat(f"Loading metadata: {last_index + 1}/{total} ({progress_pct}%)")
            self._log_status_message(f"Loading metadata... {last_index + 1}/{total}", 0)

    def _metadata_finished(self, protocols):
        """Called when background loading completes."""
        # Auto-fit columns with padding
        self._auto_fit_table_columns()

        # Apply row styling (parent vs sub-row colors)
        self._apply_all_row_styling()

        # Hide progress bar
        self.projectProgressBar.setVisible(False)
        self.projectProgressBar.setValue(0)

        # Count total files in master list
        total_files = len(self._master_file_list)
        new_files = len(self._discovered_files_data)

        summary_text = f"Summary: {total_files} ABF files | {len(protocols)} protocols"
        self.summaryLabel.setText(summary_text)
        self._log_status_message(f"✓ Loaded metadata for {new_files} files ({total_files} total)", 3000)
        print(f"[project-builder] Complete! {len(protocols)} protocols: {sorted(protocols)}")

        # Re-enable scan button
        self.scanFilesButton.setEnabled(True)

        # Reset row offset
        self._metadata_row_offset = 0

    def _populate_discovered_files_table(self, abf_files):
        """Populate the discovered files table with ABF metadata."""
        table = self.discoveredFilesTable

        # Clear existing rows
        table.setRowCount(0)

        # Populate rows
        for i, file_info in enumerate(abf_files):
            table.insertRow(i)

            # File Name
            table.setItem(i, 0, QTableWidgetItem(file_info['file_name']))

            # Protocol
            table.setItem(i, 1, QTableWidgetItem(file_info['protocol']))

            # Duration (convert to minutes)
            duration_min = file_info['duration_sec'] / 60.0
            table.setItem(i, 2, QTableWidgetItem(f"{duration_min:.1f} min"))

            # Channels (count)
            channel_count = len(file_info['channels'])
            table.setItem(i, 3, QTableWidgetItem(str(channel_count)))

            # Size (MB)
            table.setItem(i, 4, QTableWidgetItem(f"{file_info['file_size_mb']:.2f}"))

        # Resize columns to content
        table.resizeColumnsToContents()

        print(f"[project-builder] Populated table with {len(abf_files)} files")

    # NOTE: Old experiment-based methods removed (on_project_add_files, _add_files_to_experiment)
    # We now use the master file list approach where all files are in a flat table with metadata columns

    def on_project_clear_files(self):
        """Clear the discovered files table and master file list."""
        self.discoveredFilesTable.setRowCount(0)
        self._discovered_files_data = []
        self._master_file_list = []
        self.summaryLabel.setText("Summary: No files scanned")
        # Hide progress bar
        self.projectProgressBar.setVisible(False)
        self.projectProgressBar.setValue(0)
        self._log_status_message("Cleared discovered files", 1500)

    def _add_scan_saved_data_button(self):
        """Add the 'Scan Saved Data', 'Resolve Conflicts' buttons and column mode checkbox to the Project Builder toolbar."""
        from PyQt6.QtWidgets import QPushButton, QCheckBox

        # Create the Scan Saved Data button
        self.scanSavedDataButton = QPushButton("📁 Scan Saved Data")
        self.scanSavedDataButton.setToolTip(
            "Scan for existing exported data files and auto-populate the table.\n"
            "Looks for Pleth_App_analysis folders and matches saved files to ABF names."
        )
        self.scanSavedDataButton.setStyleSheet("""
            QPushButton {
                background-color: #4a7c4c;
                color: white;
                border: 1px solid #3d6e3f;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a8c5c;
            }
            QPushButton:pressed {
                background-color: #3a6c3c;
            }
        """)
        self.scanSavedDataButton.clicked.connect(self.on_project_scan_saved_data)

        # Create the Resolve Conflicts button
        self.resolveConflictsButton = QPushButton("⚠ Resolve Conflicts")
        self.resolveConflictsButton.setToolTip(
            "View and resolve all data conflicts and warnings.\n"
            "Shows rows where table values differ from saved NPZ files."
        )
        self.resolveConflictsButton.setStyleSheet("""
            QPushButton {
                background-color: #7c6a4a;
                color: white;
                border: 1px solid #6e5d3f;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8c7a5a;
            }
            QPushButton:pressed {
                background-color: #6c5a3a;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        self.resolveConflictsButton.clicked.connect(self.on_resolve_all_conflicts)
        self.resolveConflictsButton.setEnabled(False)  # Disabled until conflicts exist

        # Create the "Show Full Content" checkbox for column width mode
        self.tableFullContentCheckbox = QCheckBox("Show Full Content")
        self.tableFullContentCheckbox.setToolTip(
            "When checked: columns expand to show all content (enables horizontal scrolling)\n"
            "When unchecked: columns fit to visible area, clipping long text"
        )
        self.tableFullContentCheckbox.setChecked(False)
        self.tableFullContentCheckbox.stateChanged.connect(self._on_table_column_mode_changed)

        # Create the AI Settings button
        self.aiSettingsButton = QPushButton("🤖 AI Assistant")
        self.aiSettingsButton.setToolTip(
            "Configure AI integration (Claude, GPT, Gemini)\n"
            "Use AI to help organize files, extract metadata from notes,\n"
            "and get analysis suggestions."
        )
        self.aiSettingsButton.setStyleSheet("""
            QPushButton {
                background-color: #5a4a7c;
                color: white;
                border: 1px solid #4a3d6e;
                border-radius: 4px;
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a5a8c;
            }
            QPushButton:pressed {
                background-color: #4a3a6c;
            }
        """)
        self.aiSettingsButton.clicked.connect(self._open_ai_settings)

        # Find the layout containing scanFilesButton and add next to it
        # The scan button is in the projectFilesFrame area
        if hasattr(self, 'scanFilesButton'):
            parent = self.scanFilesButton.parent()
            if parent and parent.layout():
                layout = parent.layout()
                # Find the position of scanFilesButton and insert after it
                for i in range(layout.count()):
                    widget = layout.itemAt(i).widget()
                    if widget == self.scanFilesButton:
                        layout.insertWidget(i + 1, self.scanSavedDataButton)
                        layout.insertWidget(i + 2, self.resolveConflictsButton)
                        layout.insertWidget(i + 3, self.tableFullContentCheckbox)
                        layout.insertWidget(i + 4, self.aiSettingsButton)
                        break
                else:
                    # Fallback: just add it to the layout
                    layout.addWidget(self.scanSavedDataButton)
                    layout.addWidget(self.resolveConflictsButton)
                    layout.addWidget(self.tableFullContentCheckbox)
                    layout.addWidget(self.aiSettingsButton)

    def _on_table_column_mode_changed(self, state):
        """Handle toggle of the 'Show Full Content' checkbox."""
        self._auto_fit_table_columns()

    def _open_ai_settings(self):
        """Open the AI Settings dialog for configuring AI integration."""
        try:
            from dialogs.ai_settings_dialog import AISettingsDialog

            # Gather file metadata from the table to pass to the dialog
            files_metadata = []
            table = self.discoveredFilesTable
            for row in range(table.rowCount()):
                metadata = {}
                # Get file name (column 0)
                item = table.item(row, 0)
                metadata['file_name'] = item.text() if item else ''

                # Get protocol (column 1)
                item = table.item(row, 1)
                metadata['protocol'] = item.text() if item else ''

                # Get keywords (column 4)
                item = table.item(row, 4)
                metadata['keywords_display'] = item.text() if item else ''

                # Get experiment (column 5)
                item = table.item(row, 5)
                metadata['experiment'] = item.text() if item else ''

                # Get file path from master list if available
                if row < len(self._master_file_list):
                    file_data = self._master_file_list[row]
                    if isinstance(file_data, dict):
                        metadata['file_path'] = str(file_data.get('file_path', ''))
                    else:
                        metadata['file_path'] = str(file_data)

                files_metadata.append(metadata)

            dialog = AISettingsDialog(self, files_metadata=files_metadata)
            dialog.exec()

        except ImportError as e:
            self._show_warning("AI Module Not Found",
                             f"Could not load AI settings dialog:\n{e}\n\n"
                             "Make sure the dialogs/ai_settings_dialog.py file exists.")
        except Exception as e:
            self._show_error("Error", f"Failed to open AI settings:\n{e}")

    def on_project_scan_saved_data(self):
        """Scan for existing saved data files and auto-populate the table."""
        if not self._master_file_list:
            self._show_warning("No Files", "Please scan for ABF files first.")
            return

        from PyQt6.QtWidgets import QProgressDialog
        import re

        # Show progress dialog
        progress = QProgressDialog("Scanning for saved data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Scanning Saved Data")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        # Look for Pleth_App_analysis folders
        base_dir = Path(self._project_directory) if self._project_directory else Path.cwd()
        analysis_folders = list(base_dir.glob("**/Pleth_App_analysis"))

        if not analysis_folders:
            progress.close()
            self._show_info("No Saved Data", "No 'Pleth_App_analysis' folders found.\n\nAnalyzed data is saved to this folder.")
            return

        # Build a mapping of ABF names to their saved data
        saved_data_map = {}  # abf_stem -> {channel -> export_info}

        progress.setLabelText(f"Scanning {len(analysis_folders)} analysis folders...")
        progress.setValue(10)
        QApplication.processEvents()

        # Collect ALL NPZ files first, then sort by modification time (newest first)
        # This ensures newer files take priority over older ones in subfolders
        all_npz_files = []
        for folder in analysis_folders:
            for npz_file in folder.glob("*_bundle.npz"):
                try:
                    mtime = npz_file.stat().st_mtime
                except:
                    mtime = 0
                all_npz_files.append((npz_file, folder, mtime))

        # Sort by modification time, newest first
        all_npz_files.sort(key=lambda x: x[2], reverse=True)
        print(f"[scan-saved] Found {len(all_npz_files)} NPZ bundle files, processing newest first")

        for npz_file, folder, mtime in all_npz_files:
            # Parse the filename to extract ABF name and channel
            stem = npz_file.stem.replace('_bundle', '')

            # Try to read metadata from NPZ file for reliable source file info
            npz_metadata = None
            source_file_from_npz = None
            channel_from_npz = None

            try:
                import numpy as np
                import json
                with np.load(npz_file, allow_pickle=True) as data:
                    # Bundle NPZ stores meta as JSON string in 'meta_json'
                    if 'meta_json' in data:
                        meta_str = str(data['meta_json'])
                        npz_metadata = json.loads(meta_str)
                        # Get source file and channel from metadata
                        if isinstance(npz_metadata, dict):
                            source_file_from_npz = npz_metadata.get('abf_path', '')
                            channel_from_npz = npz_metadata.get('analyze_channel', '')
                    # ML training NPZ stores source_file directly
                    elif 'source_file' in data:
                        source_file_from_npz = str(data['source_file'])
                        npz_metadata = {'source_file': source_file_from_npz}
            except Exception as e:
                print(f"[scan-saved] Could not read metadata from {npz_file.name}: {e}")

            # Try to match to an ABF file using multiple methods
            for task in self._master_file_list:
                abf_path = Path(task.get('file_path', ''))
                abf_name = abf_path.stem
                if not abf_name:
                    continue

                # Method 1: Exact match from NPZ metadata (most reliable)
                matched = False
                if source_file_from_npz:
                    source_stem = Path(source_file_from_npz).stem
                    if source_stem == abf_name:
                        matched = True
                        print(f"[scan-saved] Matched {npz_file.name} to {abf_name} via NPZ metadata")

                # Method 2: ABF name appears anywhere in the filename
                if not matched and abf_name in stem:
                    matched = True
                    print(f"[scan-saved] Matched {npz_file.name} to {abf_name} via substring")

                if matched:
                    # Found a match - extract more info
                    key = str(task.get('file_path'))

                    # Determine channel - prefer NPZ metadata, fall back to filename regex
                    channel = channel_from_npz or ''
                    if not channel:
                        channel_match = re.search(r'_(AD\d+)_', stem) or re.search(r'_(AD\d+)$', stem)
                        channel = channel_match.group(1) if channel_match else ''

                    if key not in saved_data_map:
                        saved_data_map[key] = {}

                    # Skip if we already have data for this file+channel (since we're processing newest first)
                    if channel in saved_data_map[key]:
                        # Track this as an older/duplicate NPZ file
                        if 'older_npz_files' not in saved_data_map[key][channel]:
                            saved_data_map[key][channel]['older_npz_files'] = []
                        saved_data_map[key][channel]['older_npz_files'].append({
                            'file': str(npz_file),
                            'date': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M'),
                        })
                        print(f"[scan-saved] Skipping older {npz_file.name} - already have newer data for {abf_name} {channel}")
                        break

                    export_info = {
                        'export_path': str(folder),
                        'export_date': '',
                        'npz': True,
                        'timeseries_csv': (folder / f"{stem}_means_by_time.csv").exists(),
                        'breaths_csv': (folder / f"{stem}_breaths.csv").exists(),
                        'events_csv': (folder / f"{stem}_events.csv").exists(),
                        'pdf': (folder / f"{stem}_summary.pdf").exists(),
                        'session_state': (folder / f"{stem}_session.npz").exists(),
                        'ml_training': False,  # Would need to check ML folder
                        'npz_metadata': npz_metadata,  # Store for later use
                    }

                    # Use the mtime we already collected during sorting
                    from datetime import datetime
                    export_info['export_date'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')

                    # Extract additional metadata from ui_meta if available
                    if npz_metadata:
                        ui_meta = npz_metadata.get('ui_meta', {})
                        if isinstance(ui_meta, dict):
                            export_info['strain'] = ui_meta.get('strain', '')
                            export_info['stim_type'] = ui_meta.get('stim', '')  # Dialog saves as 'stim'
                            export_info['power'] = ui_meta.get('power', '')
                            export_info['sex'] = ui_meta.get('sex', '')
                            export_info['animal_id'] = ui_meta.get('animal', '')  # Dialog saves as 'animal', not 'animal_id'
                            # Also get stim and events channels from NPZ metadata (not ui_meta)
                            export_info['stim_channel'] = npz_metadata.get('stim_chan', '')
                            export_info['events_channel'] = npz_metadata.get('event_channel', '')

                    saved_data_map[key][channel] = export_info
                    print(f"[scan-saved] Using {npz_file.name} (newest) for {abf_name} {channel}")
                    break

        progress.setValue(50)
        QApplication.processEvents()

        # Debug: Print what was matched
        print(f"[scan-saved] saved_data_map summary:")
        for file_key, channels in saved_data_map.items():
            file_name = Path(file_key).name
            channel_list = list(channels.keys())
            print(f"  {file_name}: {channel_list}")

        if not saved_data_map:
            progress.close()
            self._show_info("No Matches", "No saved data files matched the current ABF files.")
            return

        # Update the master list and table with found saved data
        # We need to handle multiple channels per file by creating sub-rows
        updated_count = 0
        new_rows_created = 0
        table = self.discoveredFilesTable
        table.blockSignals(True)

        # Track which file+channel combinations already exist in the master list
        existing_file_channels = set()
        for task in self._master_file_list:
            fp = str(task.get('file_path', ''))
            ch = task.get('channel', '')
            if fp and ch:
                existing_file_channels.add((fp, ch))

        # First pass: update existing sub-rows that have matching channels
        # Parent rows will have sub-rows created in second pass
        processed_files = set()
        for row, task in enumerate(self._master_file_list):
            file_key = str(task.get('file_path'))
            if file_key not in saved_data_map:
                continue

            existing_channel = task.get('channel', '')
            is_sub_row = task.get('is_sub_row', False)
            export_data = saved_data_map[file_key]

            # Only update if this is a sub-row with a matching channel
            if is_sub_row and existing_channel and existing_channel in export_data:
                info = export_data[existing_channel]
                self._update_task_with_export_info(task, info, row, table)
                updated_count += 1
                processed_files.add(file_key)
                print(f"[scan-saved] Updated sub-row {row}: {task.get('file_name')} channel {existing_channel}")

            # Parent rows: Don't update them directly - all channels will get sub-rows in second pass
            elif not is_sub_row:
                processed_files.add(file_key)  # Mark as processed so we know to create sub-rows

            # Update progress
            if row % 10 == 0:
                progress.setValue(50 + int(25 * row / len(self._master_file_list)))
                QApplication.processEvents()

        # Second pass: create new sub-rows for additional channels not yet in the list
        # Collect info about what to add (file_key, channel, info) - we'll find row indices at insertion time
        rows_to_add = []  # [(file_key, channel, info), ...]

        for file_key, export_data in saved_data_map.items():
            # Check which channels need new rows for this file
            for channel, info in export_data.items():
                if (file_key, channel) not in existing_file_channels:
                    rows_to_add.append((file_key, channel, info))
                    existing_file_channels.add((file_key, channel))  # Mark as will be added

        # Add new sub-rows for additional channels
        # Find the correct row index at insertion time (not pre-computed) to handle index shifts
        print(f"[scan-saved] Second pass: {len(rows_to_add)} sub-rows to add")
        for file_key, channel, info in rows_to_add:
            # Find the PARENT task (not sub-row) for this file to get full metadata
            parent_task = None
            parent_row = None
            for row, task in enumerate(self._master_file_list):
                if str(task.get('file_path')) == file_key and not task.get('is_sub_row', False):
                    parent_task = task
                    parent_row = row
                    break

            # If no parent found, use any matching task (fallback)
            if not parent_task:
                for row, task in enumerate(self._master_file_list):
                    if str(task.get('file_path')) == file_key:
                        parent_task = task
                        parent_row = row
                        break

            if parent_task:
                print(f"[scan-saved] Adding sub-row for {Path(file_key).name} channel {channel} at row {parent_row}")
                self._create_sub_row_from_saved_data(parent_task, parent_row, channel, info)
                new_rows_created += 1
                updated_count += 1
            else:
                print(f"[scan-saved] WARNING: Could not find source task for {file_key}")

        progress.setValue(90)
        QApplication.processEvents()

        table.blockSignals(False)

        # Auto-fit columns and apply row styling
        self._auto_fit_table_columns()
        self._apply_all_row_styling()

        progress.close()

        # Count warnings
        warnings_count = 0
        conflicts_count = 0
        older_files_count = 0
        for task in self._master_file_list:
            warnings = task.get('scan_warnings', {})
            if warnings:
                warnings_count += 1
                if warnings.get('conflicts'):
                    conflicts_count += 1
                if warnings.get('older_npz_count', 0) > 0:
                    older_files_count += 1

        msg = f"✓ Found saved data for {updated_count} analyses"
        if new_rows_created > 0:
            msg += f" ({new_rows_created} new rows created)"
        if warnings_count > 0:
            warning_details = []
            if conflicts_count > 0:
                warning_details.append(f"{conflicts_count} conflicts")
            if older_files_count > 0:
                warning_details.append(f"{older_files_count} with multiple NPZ files")
            msg += f" ⚠ {', '.join(warning_details)}"
        self._log_status_message(msg, 5000 if warnings_count > 0 else 3000)
        print(f"[project-builder] Found saved data for {updated_count} analyses across {len(analysis_folders)} folders")
        if warnings_count > 0:
            print(f"[project-builder] Warnings: {conflicts_count} conflicts, {older_files_count} with older NPZ files")

        # Enable/disable the Resolve Conflicts button based on warnings
        if hasattr(self, 'resolveConflictsButton'):
            self.resolveConflictsButton.setEnabled(warnings_count > 0)
            if warnings_count > 0:
                self.resolveConflictsButton.setText(f"⚠ Resolve Conflicts ({warnings_count})")
            else:
                self.resolveConflictsButton.setText("⚠ Resolve Conflicts")

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
        """Edit the current project name."""
        from PyQt6.QtWidgets import QInputDialog

        current_name = self.projectNameEdit.text().strip()
        if not current_name:
            current_name = "Untitled Project"

        new_name, ok = QInputDialog.getText(
            self, "Edit Project Name", "Enter project name:",
            text=current_name
        )

        if ok and new_name.strip():
            self.projectNameEdit.setText(new_name.strip())
            self._log_status_message(f"Project renamed to: {new_name.strip()}", 2000)

    def on_project_new(self):
        """Create a new project - prompt for name and directory."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("New Project")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # Project name
        layout.addWidget(QLabel("Project Name:"))
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("Enter project name...")
        layout.addWidget(name_edit)

        layout.addSpacing(10)

        # Directory selection
        layout.addWidget(QLabel("Data Directory:"))
        dir_layout = QHBoxLayout()
        dir_edit = QLineEdit()
        dir_edit.setPlaceholderText("Select directory containing data files...")
        dir_edit.setReadOnly(True)
        dir_layout.addWidget(dir_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.setMaximumWidth(100)
        def browse_dir():
            directory = QFileDialog.getExistingDirectory(dialog, "Select Data Directory")
            if directory:
                dir_edit.setText(directory)
                # Auto-fill project name if empty
                if not name_edit.text().strip():
                    name_edit.setText(Path(directory).name)
        browse_btn.clicked.connect(browse_dir)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        layout.addSpacing(20)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        create_btn = QPushButton("Create Project")
        create_btn.setDefault(True)
        create_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(create_btn)

        layout.addLayout(button_layout)

        # Show dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            project_name = name_edit.text().strip()
            directory = dir_edit.text().strip()

            if not project_name:
                self._show_warning("Missing Information", "Please enter a project name.")
                return

            if not directory:
                self._show_warning("Missing Information", "Please select a data directory.")
                return

            # Clear everything
            self.discoveredFilesTable.setRowCount(0)
            self._discovered_files_data = []
            self._master_file_list = []

            # Set new project info
            self._project_directory = directory
            self.directoryPathEdit.setText(directory)
            self.projectNameEdit.setText(project_name)
            self.summaryLabel.setText("Summary: No files scanned")

            self._log_status_message(f"✓ New project created: {project_name}", 2000)

    def on_project_save(self):
        """Save current project to data directory."""
        if not self._project_directory:
            self._show_warning("No Directory", "Please select a directory first.")
            return

        if not self._discovered_files_data:
            self._show_warning("No Files", "Please scan for files first.")
            return

        # Get project name
        project_name = self.projectNameEdit.text().strip()
        if not project_name:
            # Prompt for project name
            from PyQt6.QtWidgets import QInputDialog
            project_name, ok = QInputDialog.getText(
                self, "Save Project", "Enter project name:",
                text=Path(self._project_directory).name
            )
            if not ok or not project_name.strip():
                return
            project_name = project_name.strip()
            self.projectNameEdit.setText(project_name)

        try:
            # Use master file list as the source of truth
            # (it contains all metadata edits the user made)
            files_to_save = self._master_file_list if self._master_file_list else self._discovered_files_data
            print(f"[project-save] Saving {len(files_to_save)} files from master list")

            # Save project (no experiments - using flat file list now)
            project_path = self.project_manager.save_project(
                project_name,
                Path(self._project_directory),
                files_to_save,
                []  # No experiments in new workflow
            )

            self._show_info("Project Saved", f"Project saved to:\n{project_path}")
            self._log_status_message(f"✓ Project saved: {project_name}", 3000)

            # Update recent projects dropdown
            self._populate_load_project_combo()

        except Exception as e:
            self._show_error("Save Failed", f"Failed to save project:\n{e}")
            print(f"[project] Error saving: {e}")
            import traceback
            traceback.print_exc()

    def on_project_load(self, index):
        """Load a project from recent projects list."""
        if index <= 0:  # Skip the "Load Project..." placeholder
            return

        recent_projects = self.project_manager.get_recent_projects()
        if index - 1 >= len(recent_projects):
            return

        project_info = recent_projects[index - 1]
        project_path = Path(project_info['path'])

        try:
            # Try to load project
            project_data = self.project_manager.load_project(project_path)

        except FileNotFoundError:
            # Project file was moved or deleted
            response = self._ask_locate_project(project_info['name'], project_path)
            if response == "locate":
                # Ask user to locate the file
                from PyQt6.QtWidgets import QFileDialog
                new_path, _ = QFileDialog.getOpenFileName(
                    self, "Locate Project File",
                    str(project_path.parent),
                    "PhysioMetrics Project (*.physiometrics)"
                )
                if not new_path:
                    # Reset combo to placeholder
                    self.loadProjectCombo.setCurrentIndex(0)
                    return

                new_path = Path(new_path)
                try:
                    # Load from new location
                    project_data = self.project_manager.load_project(new_path)
                    # Update path in recent projects
                    self.project_manager.update_recent_project_path(project_path, new_path)
                    self._populate_load_project_combo()
                except Exception as e:
                    self._show_error("Load Failed", f"Failed to load project:\n{e}")
                    self.loadProjectCombo.setCurrentIndex(0)
                    return
            else:
                # User cancelled
                self.loadProjectCombo.setCurrentIndex(0)
                return

        except Exception as e:
            self._show_error("Load Failed", f"Failed to load project:\n{e}")
            self.loadProjectCombo.setCurrentIndex(0)
            print(f"[project] Error loading: {e}")
            import traceback
            traceback.print_exc()
            return

        # Populate UI with loaded data
        self._populate_ui_from_project(project_data)
        self._log_status_message(f"✓ Loaded project: {project_data['project_name']}", 3000)

        # Reset combo to placeholder after loading
        self.loadProjectCombo.setCurrentIndex(0)

    def _populate_load_project_combo(self):
        """Populate the Load Project dropdown with recent projects."""
        combo = self.loadProjectCombo
        combo.blockSignals(True)  # Prevent triggering load during population
        combo.clear()

        # Add placeholder
        combo.addItem("Load Project...")

        # Add recent projects
        recent_projects = self.project_manager.get_recent_projects()
        for project_info in recent_projects:
            combo.addItem(f"{project_info['name']}")

        combo.blockSignals(False)

    def _populate_ui_from_project(self, project_data):
        """Populate UI with loaded project data."""
        # Set project name and directory
        self.projectNameEdit.setText(project_data['project_name'])
        self._project_directory = str(project_data['data_directory'])
        self.directoryPathEdit.setText(self._project_directory)

        # Populate discovered files table
        self._discovered_files_data = project_data['files']
        table = self.discoveredFilesTable

        # Disable updates for faster population
        table.setUpdatesEnabled(False)
        table.setRowCount(0)

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

                        # Format keywords for display
                        keywords_display = []
                        if path_info['power_levels']:
                            keywords_display.extend(path_info['power_levels'])
                        if path_info['animal_ids']:
                            keywords_display.extend([f"ID:{id}" for id in path_info['animal_ids']])
                        if path_info['keywords']:
                            keywords_display.extend(path_info['keywords'])

                        file_data['path_keywords'] = path_info
                        file_data['keywords_display'] = ', '.join(keywords_display) if keywords_display else ''

                # Update progress every 50 files
                if idx % 50 == 0 or idx == total_files - 1:
                    progress_pct = int((idx + 1) / total_files * 100)
                    self.projectProgressBar.setValue(progress_pct)
                    self.projectProgressBar.setFormat(f"Extracting keywords: {idx + 1}/{total_files} ({progress_pct}%)")
                    QApplication.processEvents()

        # Show progress for table population
        total_files = len(self._discovered_files_data)
        self.projectProgressBar.setVisible(True)
        self.projectProgressBar.setValue(0)
        self.projectProgressBar.setFormat(f"Loading files: 0/{total_files} (0%)")
        QApplication.processEvents()

        # Populate table with 15-column structure
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

            table.insertRow(i)
            table.setItem(i, 0, QTableWidgetItem(file_data.get('file_name', '')))  # File Name
            table.setItem(i, 1, QTableWidgetItem(file_data.get('protocol', '')))  # Protocol
            table.setItem(i, 2, QTableWidgetItem(str(file_data.get('channel_count', ''))))  # Avail Ch
            table.setItem(i, 3, QTableWidgetItem(str(file_data.get('sweep_count', ''))))  # Sweeps
            table.setItem(i, 4, QTableWidgetItem(file_data.get('keywords_display', '')))  # Keywords
            table.setItem(i, 5, QTableWidgetItem(file_data.get('channel', '')))  # Channel
            table.setItem(i, 6, QTableWidgetItem(file_data.get('stim_channel', '')))  # Stim Ch
            table.setItem(i, 7, QTableWidgetItem(file_data.get('events_channel', '')))  # Events Ch
            table.setItem(i, 8, QTableWidgetItem(file_data.get('strain', '')))  # Strain
            table.setItem(i, 9, QTableWidgetItem(file_data.get('stim_type', '')))  # Stim Type
            table.setItem(i, 10, QTableWidgetItem(file_data.get('power', '')))  # Power
            table.setItem(i, 11, QTableWidgetItem(file_data.get('sex', '')))  # Sex
            table.setItem(i, 12, QTableWidgetItem(file_data.get('animal_id', '')))  # Animal ID
            # Status icon based on status field
            status = file_data.get('status', 'pending')
            status_icon = '●' if status == 'completed' else ('◐' if status == 'in_progress' else '○')
            table.setItem(i, 13, QTableWidgetItem(status_icon))  # Status
            # Add action buttons (column 14)
            self._add_row_action_button(i)
            # Column 15: Exports
            table.setItem(i, 15, self._create_exports_table_item(file_data))

            if file_data.get('protocol'):
                protocols.add(file_data['protocol'])

            # Update progress every 50 files
            if i % 50 == 0 or i == total_files - 1:
                progress_pct = int((i + 1) / total_files * 100)
                self.projectProgressBar.setValue(progress_pct)
                self.projectProgressBar.setFormat(f"Loading files: {i + 1}/{total_files} ({progress_pct}%)")
                QApplication.processEvents()

        # Re-enable updates and resize
        table.setUpdatesEnabled(True)
        table.resizeColumnsToContents()

        # Hide progress bar
        self.projectProgressBar.setVisible(False)
        self.projectProgressBar.setValue(0)

        # Update summary
        summary_text = f"Summary: {len(self._discovered_files_data)} ABF files | {len(protocols)} protocols"
        self.summaryLabel.setText(summary_text)

        # Sync master file list with discovered files
        self._master_file_list = self._discovered_files_data.copy()

        # Note: Old experiment structure is ignored - we now use the flat master file list
        # If old project had experiments, their task data could be migrated here if needed

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

    app = QApplication(sys.argv)

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
        splash_pix = QPixmap(200, 150)
        splash_pix.fill(Qt.GlobalColor.darkGray)

    # Scale to smaller size for faster display
    splash_pix = splash_pix.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    splash = QSplashScreen(splash_pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)

    # Add loading message
    splash.showMessage(
        "Loading PhysioMetrics...",
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

    sys.exit(app.exec())