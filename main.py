from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem, QAbstractItemView
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
from core import peaks as peakdet   # peak detection
from core import metrics  # calculation of breath metrics
from core.navigation_manager import NavigationManager
from plotting import PlotManager
from export import ExportManager


# Import editing modes
from editing import EditingModes
# Import dialogs
from dialogs import GMMClusteringDialog, SpectralAnalysisDialog, OutlierMetricsDialog, SaveMetaDialog

# Import version
from version_info import VERSION_STRING


ORG = "PlethApp"
APP = "PlethApp"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize managers
        self.consolidation_manager = ConsolidationManager(self)
        ui_file = Path(__file__).parent / "ui" / "pleth_app_layout_02_horizontal.ui"
        uic.loadUi(ui_file, self)

        # icon_path = Path(__file__).parent / "assets" / "plethapp_thumbnail_light_02.ico"
        icon_path = Path(__file__).parent / "assets" / "plethapp_thumbnail_dark_round.ico"
        self.setWindowIcon(QIcon(str(icon_path)))
        # after uic.loadUi(ui_file, self)
        from PyQt6.QtWidgets import QWidget
        for w in self.findChildren(QWidget):
            if w.property("startHidden") is True:
                w.hide()

        self.setWindowTitle(f"PlethAnalysis v{VERSION_STRING}")

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

        # --- Wire channel selection (immediate application) ---
        self.AnalyzeChanSelect.currentIndexChanged.connect(self.on_analyze_channel_changed)
        self.StimChanSelect.currentIndexChanged.connect(self.on_stim_channel_changed)
        self.EventsChanSelect.currentIndexChanged.connect(self.on_events_channel_changed)


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

        # Spectral Analysis button
        self.SpectralAnalysisButton.clicked.connect(self.on_spectral_analysis_clicked)

        # Outlier Threshold button
        self.OutlierThreshButton.clicked.connect(self.on_outlier_thresh_clicked)

        # Eupnea Threshold button

        # GMM Clustering button
        self.GMMClusteringButton.clicked.connect(self.on_gmm_clustering_clicked)

        # Auto-Update GMM checkbox
        # Auto-Update GMM checkbox moved to GMM dialog
        # Connect the manual update button
        self.UpdateEupneaSniffingButton.clicked.connect(self.on_update_eupnea_sniffing_clicked)

        # --- Initialize Navigation Manager ---
        self.navigation_manager = NavigationManager(self)
        self.WindowRangeValue.setText("20")  # Default window length

        # --- Initialize Plot Manager (BEFORE signal connections that may trigger plotting) ---
        self.plot_manager = PlotManager(self)

        # --- Initialize Export Manager ---
        self.export_manager = ExportManager(self)

        # --- Peak-detect UI wiring ---
        # Apply button opens dialog to auto-detect optimal prominence
        self.ApplyPeakFindPushButton.setText("Auto-Detect && Apply Peaks")
        self.ApplyPeakFindPushButton.setEnabled(False)  # stays disabled until data loaded
        self.ApplyPeakFindPushButton.clicked.connect(self._auto_detect_and_apply_peaks)

        # Store peak detection parameters (auto-populated from dialog)
        self.peak_prominence = None
        self.peak_min_dist = 0.05  # Default minimum peak distance in seconds
        self.peak_quality_score = None
        self.peak_is_bimodal = None

        # Default values for eupnea and apnea thresholds
        self.ApneaThresh.setText("0.5")   # seconds - gaps longer than this are apnea
        self.OutlierSD.setText("3.0")     # SD - standard deviations for outlier detection

        # Connect signals for apnea/outlier threshold changes to trigger redraw
        self.ApneaThresh.textChanged.connect(self._on_region_threshold_changed)
        self.OutlierSD.textChanged.connect(self._on_region_threshold_changed)

        # --- y2 metric dropdown (choices only; plotting later) ---
        self.y2plot_dropdown.clear()
        self.state.y2_values_by_sweep.clear()
        self.plot_host.clear_y2()

        self.y2plot_dropdown.addItem("None", userData=None)
        for label, key in metrics.METRIC_SPECS:
            self.y2plot_dropdown.addItem(label, userData=key)

        # ADD/DELETE Peak Mode: track selection in state
        self.state.y2_metric_key = None
        self.y2plot_dropdown.currentIndexChanged.connect(self.on_y2_metric_changed)


        # Initialize editing modes manager
        self.editing_modes = EditingModes(self)

        # --- Mark Events button (event detection settings) ---
        self.MarkEventsButton.clicked.connect(self.on_mark_events_clicked)

        # --- Sigh overlay artists ---
        self._sigh_artists = []         # matplotlib artists for sigh overlay

        # --- Wire omit button ---
        self.OmitSweepButton.clicked.connect(self.on_omit_sweep_clicked)


        # Button in your UI: objectName 'addSighButton'
        self.addSighButton.setCheckable(True)

        # --- Move Point mode ---
        self._is_dragging = False  # Track if currently dragging a point

        self.movePointButton.setCheckable(True)

        # Mark Sniff button

        #wire save analyzed data button
        self.SaveAnalyzedDataButton.clicked.connect(self.on_save_analyzed_clicked)

        # Wire view summary button to show PDF preview
        self.ViewSummary_pushButton.clicked.connect(self.on_view_summary_clicked)



        # Defaults: 0.5–20 Hz band, all off initially
        self.HighPassVal.setText("0.5")
        self.LowPassVal.setText("20")
        self.HighPass_checkBox.setChecked(False)
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
        print(f"[DEBUG] PLETHAPP_TESTING environment variable = '{os.environ.get('PLETHAPP_TESTING')}'")
        if os.environ.get('PLETHAPP_TESTING') == '1':
            print("[TESTING MODE] Auto-loading test file...")
            test_file = Path(r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\25121004.abf")
            print(f"[TESTING MODE] Checking if file exists: {test_file}")
            print(f"[TESTING MODE] File exists: {test_file.exists()}")
            if test_file.exists():
                print(f"[TESTING MODE] Scheduling auto-load in 100ms...")
                QTimer.singleShot(100, lambda: self._auto_load_test_file(test_file))
            else:
                print(f"[TESTING MODE] Warning: Test file not found: {test_file}")

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
        print("[TESTING MODE] Setting analyze channel to 0, stim channel to 7...")
        print(f"[TESTING MODE] AnalyzeChanSelect has {self.AnalyzeChanSelect.count()} items")
        print(f"[TESTING MODE] StimChanSelect has {self.StimChanSelect.count()} items")

        # Set analyze channel to index 1 (channel 0, since index 0 is "All Channels")
        if self.AnalyzeChanSelect.count() > 1:
            self.AnalyzeChanSelect.setCurrentIndex(1)  # Channel 0 is at index 1
            print(f"[TESTING MODE] Set analyze channel to index 1: {self.AnalyzeChanSelect.currentText()}")
            # Manually trigger the channel change handler
            self.on_analyze_channel_changed(1)

        # Set stim channel to 7 (need to find the index)
        stim_channel_set = False
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

    # ---------- File browse ----------
    def closeEvent(self, event):
        """Save window geometry on close."""
        self.settings.setValue("geometry", self.saveGeometry())
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
            "All Supported (*.abf *.smrx *.edf *.pleth.npz);;Data Files (*.abf *.smrx *.edf);;PlethApp Sessions (*.pleth.npz);;ABF Files (*.abf);;SMRX Files (*.smrx);;EDF Files (*.edf);;All Files (*.*)"
        )
        if not paths:
            return

        # Convert to Path objects
        file_paths = [Path(p) for p in paths]

        # Store the directory of the first file
        self.settings.setValue("last_dir", str(file_paths[0].parent))

        # Update UI with file info
        if len(file_paths) == 1:
            self.BrowseFilePath.setText(str(file_paths[0]))
        else:
            self.BrowseFilePath.setText(f"{len(file_paths)} files selected: {file_paths[0].name}, ...")

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
            # Load data file (supports .abf and .smrx)
            sr, sweeps_by_ch, ch_names, t = abf_io.load_data_file(path, progress_callback=update_progress)
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
            'sweep_end': n_sweeps - 1
        }]
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




        # Reset Apply button and its enable logic
        self.ApplyPeakFindPushButton.setEnabled(False)
        self._maybe_enable_peak_apply()


        


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

        #Clear omitted sweeps
        self.state.omitted_sweeps.clear()
        self._refresh_omit_button_label()





        # Start in grid mode (All Channels view)
        st.analyze_chan = None  # None = grid mode showing all channels
        self.single_panel_mode = False  # Start in grid mode

        # No stim selected by default
        st.stim_chan = None
        st.stim_onsets_by_sweep.clear()
        st.stim_offsets_by_sweep.clear()
        st.stim_spans_by_sweep.clear()
        st.stim_metrics_by_sweep.clear()

        # Start in multi-panel (all channels) view
        self.single_panel_mode = False
        self.plot_host.clear_saved_view("grid")  # fresh autoscale for grid
        self.plot_all_channels()

        # Show completion message with elapsed time
        t_elapsed = time.time() - t_start
        self._log_status_message(f"✓ File loaded ({t_elapsed:.1f}s)", 3000)

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
            self._refresh_omit_button_label()

        # Clear cross-sweep outlier detection data
        self.metrics_by_sweep.clear()
        self.onsets_by_sweep.clear()
        self.global_outlier_stats = None

        # Clear export metric cache when loading new file
        # This cache stores computed metric traces during export for reuse in PDF generation
        self._export_metric_cache = {}

        # Reset Apply button and its enable logic
        self.ApplyPeakFindPushButton.setEnabled(False)
        self._maybe_enable_peak_apply()

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

        #Clear omitted sweeps
        self.state.omitted_sweeps.clear()
        self._refresh_omit_button_label()

        # Clear z-score global statistics cache
        self.zscore_global_mean = None
        self.zscore_global_std = None

        # Start in grid mode (All Channels view)
        st.analyze_chan = None  # None = grid mode showing all channels
        self.single_panel_mode = False  # Start in grid mode

        # No stim selected by default
        st.stim_chan = None
        st.stim_onsets_by_sweep.clear()
        st.stim_offsets_by_sweep.clear()
        st.stim_spans_by_sweep.clear()
        st.stim_metrics_by_sweep.clear()

        # Start in multi-panel (all channels) view
        self.single_panel_mode = False
        self.plot_host.clear_saved_view("grid")  # fresh autoscale for grid
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
            "PlethApp Session (*.pleth.npz);;All Files (*)"
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
                f"✓ Session saved: {save_path.name} ({file_size_mb:.1f} MB, {t_elapsed:.1f}s)",
                timeout=5000
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
        progress.setWindowTitle("Loading PlethApp Session")
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

            # Update file path display
            if len(st.file_info) == 1:
                self.BrowseFilePath.setText(str(st.file_info[0]['path']))
            else:
                self.BrowseFilePath.setText(f"{len(st.file_info)} files: {st.file_info[0]['path'].name}, ...")

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
                f"✓ Session loaded: {npz_path.name} ({file_size_mb:.1f} MB, {t_elapsed:.1f}s) - "
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
                            return  # Don't continue with fresh channel switch

                # User declined or no NPZ exists - continue with fresh channel
                st.analyze_chan = new_chan
                st.proc_cache.clear()
                # Clear z-score global statistics cache
                self.zscore_global_mean = None
                self.zscore_global_std = None
                st.peaks_by_sweep.clear()
                st.sigh_by_sweep.clear()
                if hasattr(st, 'breath_by_sweep'):
                    st.breath_by_sweep.clear()

                # Clear sniffing regions
                if hasattr(st, 'sniff_regions_by_sweep'):
                    st.sniff_regions_by_sweep.clear()

                # Clear event markers (bout annotations)
                if hasattr(st, 'bout_annotations'):
                    st.bout_annotations.clear()

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

                # Enable auto-detect button now that we have data
                self.ApplyPeakFindPushButton.setEnabled(True)
                self.redraw_main_plot()

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
        self._maybe_enable_peak_apply()  # re-enable Apply Peak if threshold is valid

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

    def _auto_detect_and_apply_peaks(self):
        """
        Open auto-detection dialog to determine optimal prominence,
        then immediately apply peak detection to all sweeps.
        """
        from dialogs.prominence_threshold_dialog import ProminenceThresholdDialog
        import time

        st = self.state
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            self._show_warning("No Data", "Load and select a channel first.")
            return

        # Concatenate ALL sweeps for representative auto-threshold calculation
        print("[Auto-Detect] Concatenating all sweeps for auto-threshold analysis...")
        all_sweeps_data = []
        n_sweeps = st.sweeps[st.analyze_chan].shape[1]

        for sweep_idx in range(n_sweeps):
            # Skip omitted sweeps
            if sweep_idx in st.omitted_sweeps:
                continue

            y_sweep = self._get_processed_for(st.analyze_chan, sweep_idx)
            all_sweeps_data.append(y_sweep)

        if not all_sweeps_data:
            self._show_warning("No Data", "All sweeps are omitted.")
            return

        # Concatenate all sweeps
        y_data = np.concatenate(all_sweeps_data)
        print(f"[Auto-Detect] Concatenated {len(all_sweeps_data)} sweeps, total length: {len(y_data)} samples")

        # Open dialog with current parameters
        dialog = ProminenceThresholdDialog(
            parent=self,
            y_data=y_data,
            sr_hz=st.sr_hz,
            current_prom=self.peak_prominence,
            current_min_dist=self.peak_min_dist
        )

        if dialog.exec() == dialog.DialogCode.Accepted:
            # Store auto-detected values
            vals = dialog.get_values()
            self.peak_prominence = vals['prominence']
            self.peak_min_dist = vals['min_dist']
            self.peak_quality_score = vals['quality_score']
            self.peak_is_bimodal = vals['is_bimodal']

            print(f"[Auto-Detect] prominence={self.peak_prominence:.4f}, min_dist={self.peak_min_dist:.4f}")
            print(f"[Auto-Detect] Quality: {self.peak_quality_score:.1f}/10, Bimodal: {self.peak_is_bimodal}")

            # Now apply peak detection with auto-detected parameters
            self._apply_peak_detection()

    def _apply_peak_detection(self):
        """
        Run peak detection on the ANALYZE channel for ALL sweeps using auto-detected parameters,
        store indices per sweep, and redraw current sweep with peaks + breath markers.
        """
        import time
        t_start = time.time()

        st = self.state
        if not st.channel_names or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        self._log_status_message("Detecting peaks and breath features...")

        # Use auto-detected parameters
        prom = self.peak_prominence
        thresh = None  # Height threshold not used (prominence-based detection only)
        min_d = self.peak_min_dist
        direction = "up"  # Always detect peaks above threshold for breathing signals

        if prom is None:
            self._show_warning("No Prominence", "Run auto-detection first.")
            return

        min_dist_samples = None
        if min_d is not None and min_d > 0:
            min_dist_samples = max(1, int(round(min_d * st.sr_hz)))

        # Detect on ALL sweeps for the analyze channel
        any_chan = next(iter(st.sweeps.values()))
        n_sweeps = any_chan.shape[1]
        st.peaks_by_sweep.clear()
        st.breath_by_sweep.clear()
        # st.sigh_by_sweep.clear()


        for s in range(n_sweeps):
            y_proc = self._get_processed_for(st.analyze_chan, s)
            pks, breaths = peakdet.detect_peaks_and_breaths(
                y=y_proc, sr_hz=st.sr_hz,
                thresh=thresh,
                prominence=prom,
                min_dist_samples=min_dist_samples,
                direction=direction,
            )
            st.peaks_by_sweep[s] = pks
            st.breath_by_sweep[s] = breaths  # dict with 'onsets', 'offsets', 'expmins'

        # If a Y2 metric is selected, recompute it now that peaks/breaths changed
        if getattr(self.state, "y2_metric_key", None):
            self._compute_y2_all_sweeps()
            self.plot_host.clear_y2()

        # First redraw: Show detected peaks/breaths immediately
        print("[peak-detection] Redrawing plot with detected peaks...")
        self.redraw_main_plot()

        # Force Qt to process events so the plot updates before GMM starts
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        # Automatically run GMM clustering to identify and mark sniffing breaths
        print("[peak-detection] Running automatic GMM clustering...")
        self._run_automatic_gmm_clustering()

        # Clear out-of-date flag (GMM just ran)
        self.eupnea_sniffing_out_of_date = False

        # LIGHTWEIGHT UPDATE: Just refresh eupnea/sniffing overlays without full redraw
        # This skips expensive outlier detection (which already ran in first redraw)
        print("[peak-detection] Adding GMM-detected eupnea/sniffing overlays...")
        self._refresh_eupnea_overlays_only()

        # Show completion message with elapsed time
        t_elapsed = time.time() - t_start
        self._log_status_message(f"✓ Peak detection complete ({t_elapsed:.1f}s)", 3000)

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

        for breath_idx in range(len(onsets)):
            if breath_idx not in gmm_probs:
                # Close current group if exists
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = None
                    current_group_end = None
                    last_eupnic_idx = None
                continue

            sniff_prob = gmm_probs[breath_idx]

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
            print(f"[auto-gmm] ✓ Identified {n_sniffing_breaths} sniffing breaths and applied to plot")

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
            self._log_status_message(f"✓ GMM clustering complete ({t_elapsed:.1f}s)", 2000)

        except Exception as e:
            print(f"[auto-gmm] Error during automatic GMM clustering: {e}")
            t_elapsed = time.time() - t_start
            self._log_status_message(f"✗ GMM clustering failed ({t_elapsed:.1f}s)", 3000)
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
            print(f"[auto-gmm] ⚠️ Warning: Low cluster separation (silhouette={silhouette:.3f})")
            print(f"[auto-gmm]   Breathing patterns may be very similar (e.g., anesthetized mouse)")
        if if_idx is not None and sniff_stats['mean_if'] < 5.0:
            print(f"[auto-gmm] ⚠️ Warning: 'Sniffing' cluster has low IF ({sniff_stats['mean_if']:.2f} Hz)")
            print(f"[auto-gmm]   May be normal variation, not true sniffing (typical sniffing: 5-8 Hz)")

        return sniffing_cluster_id

    def _apply_gmm_sniffing_regions(self, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id):
        """Apply GMM sniffing cluster results by marking regions on the plot.

        Groups consecutive sniffing breaths into continuous regions.
        Stores probabilities for each breath.

        Args:
            breath_cycles: List of (sweep_idx, breath_idx) tuples
            cluster_labels: Hard cluster assignments
            cluster_probabilities: Probability matrix (n_breaths, n_clusters)
            sniffing_cluster_id: Which cluster is sniffing
        """
        import numpy as np

        # Store probabilities by (sweep_idx, breath_idx)
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            self.state.gmm_sniff_probabilities = {}
        self.state.gmm_sniff_probabilities.clear()

        # Group breath cycles by sweep
        breaths_by_sweep = {}
        for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
            if sweep_idx not in breaths_by_sweep:
                breaths_by_sweep[sweep_idx] = []

            # Get probability of this breath being sniffing
            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]

            # Store probability
            if sweep_idx not in self.state.gmm_sniff_probabilities:
                self.state.gmm_sniff_probabilities[sweep_idx] = {}
            self.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

            breaths_by_sweep[sweep_idx].append((breath_idx, cluster_labels[i], sniff_prob))

        sniffing_regions_by_sweep = {}
        n_sniffing = 0

        for sweep_idx, breath_list in breaths_by_sweep.items():
            breath_data = self.state.breath_by_sweep.get(sweep_idx)
            if breath_data is None:
                continue

            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            t = self.state.t

            # Sort breaths by index
            breath_list = sorted(breath_list, key=lambda x: x[0])

            # Group consecutive sniffing breaths into continuous regions
            regions = []
            current_group_start = None
            current_group_end = None
            last_sniff_idx = None

            for breath_idx, cluster_id, sniff_prob in breath_list:
                if cluster_id == sniffing_cluster_id:
                    n_sniffing += 1

                    if breath_idx >= len(onsets):
                        continue

                    # Get time range for this breath
                    start_time = t[int(onsets[breath_idx])]

                    # Get offset time
                    if breath_idx < len(offsets):
                        end_idx = int(offsets[breath_idx])
                    else:
                        # Fallback: use next onset or end of trace
                        if breath_idx + 1 < len(onsets):
                            end_idx = int(onsets[breath_idx + 1])
                        else:
                            end_idx = len(t) - 1
                    end_time = t[end_idx]

                    # Check if this is consecutive with the last sniffing breath (no eupnea breath in between)
                    if last_sniff_idx is None or breath_idx != last_sniff_idx + 1:
                        # Not consecutive - save current group and start new one
                        if current_group_start is not None:
                            regions.append((current_group_start, current_group_end))
                        current_group_start = start_time
                        current_group_end = end_time
                    else:
                        # Consecutive breath - extend the current group
                        current_group_end = end_time

                    last_sniff_idx = breath_idx
                else:
                    # Non-sniffing breath - close current group if exists
                    if current_group_start is not None:
                        regions.append((current_group_start, current_group_end))
                        current_group_start = None
                        current_group_end = None
                        last_sniff_idx = None

            # Save final group if exists
            if current_group_start is not None:
                regions.append((current_group_start, current_group_end))

            if regions:
                sniffing_regions_by_sweep[sweep_idx] = regions

        # Replace existing sniffing regions with GMM-detected ones
        total_merged = 0
        for sweep_idx, regions in sniffing_regions_by_sweep.items():
            self.state.sniff_regions_by_sweep[sweep_idx] = regions
            total_merged += len(regions)

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
                    print(f"[auto-gmm]   ⚠️ {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")

        print(f"[auto-gmm]   Created {total_merged} continuous sniffing region(s) across {len(sniffing_regions_by_sweep)} sweep(s)")

        return n_sniffing

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
                    print(f"[auto-gmm]   ⚠️ {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")

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

            y2 = fn(st.t, y_proc, st.sr_hz, pks, on, off, exm, exo)
            st.y2_values_by_sweep[s] = y2

        # Clear GMM probabilities after computation
        metrics.set_gmm_probabilities(None)

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
    ## Turn Off All Edit Modes ##
    ##################################################
    # Note: These lines were orphaned code that was being executed as part of on_y2_metric_changed
    # They have been removed because they were clearing the callback after we restored it

    





    def on_update_eupnea_sniffing_clicked(self):
        """Handle Update Eupnea/Sniffing Detection button - manually rerun GMM and apply."""
        import time
        from PyQt6.QtCore import QTimer

        st = self.state
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            self._log_status_message("No peaks detected yet - run peak detection first", 3000)
            return

        t_start = time.time()
        print("[update-eupnea] Manually updating eupnea/sniffing detection...")
        self._log_status_message("Updating eupnea/sniffing detection...")

        # Run GMM clustering and apply sniffing regions
        self._run_automatic_gmm_clustering()

        # Clear out-of-date flag
        self.eupnea_sniffing_out_of_date = False

        # LIGHTWEIGHT UPDATE: Just refresh eupnea/sniffing overlays without full redraw
        # This skips expensive outlier detection and metrics recomputation
        self._refresh_eupnea_overlays_only()

        # Show completion message with elapsed time
        t_elapsed = time.time() - t_start
        print(f"[update-eupnea] ✓ Eupnea/sniffing detection updated ({t_elapsed:.1f}s)")
        self._log_status_message(f"✓ Eupnea/sniffing detection updated ({t_elapsed:.1f}s)", 2000)
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

        # Update ONLY the region overlays (skips outlier masks entirely - huge speedup!)
        self.plot_host.update_region_overlays(t_plot, eupnea_mask, apnea_mask,
                                              outlier_mask=None, failure_mask=None)

        # Update sniffing region backgrounds (purple)
        self.editing_modes.update_sniff_artists(t_plot, s)

        # Refresh canvas
        self.plot_host.canvas.draw_idle()
        print("[update-eupnea] Lightweight overlay refresh complete (skipped outlier detection)")

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
        """Open dialog to select which metrics to use for outlier detection."""
        # Get all available metrics from core.metrics
        from core.metrics import METRICS

        # Filter to only numeric metrics (exclude region detection functions)
        numeric_metrics = {k: v for k, v in METRICS.items()
                          if k not in ["eupnic", "apnea", "regularity"]}

        # Create and show dialog
        dlg = OutlierMetricsDialog(parent=self,
                                    available_metrics=list(numeric_metrics.keys()),
                                    selected_metrics=self.outlier_metrics)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # Update selected metrics
            self.outlier_metrics = dlg.get_selected_metrics()
            print(f"[outlier-metrics] Updated outlier detection metrics: {self.outlier_metrics}")

            # Redraw to apply new outlier detection
            self.redraw_main_plot()

    def on_gmm_clustering_clicked(self):
        """Open GMM clustering dialog to automatically classify breaths."""
        # Check if we have breath data
        st = self.state
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Breath Data",
                "Please detect peaks first using 'Apply Peak Find' button.\n\n"
                "GMM clustering requires breath metrics to classify breathing patterns."
            )
            return

        # Create and show GMM dialog
        dlg = GMMClusteringDialog(parent=self, main_window=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # User applied clustering results
            print("[gmm-clustering] Results applied to main plot")
            self.redraw_main_plot()

    def _refresh_omit_button_label(self):
        """Update Omit button text based on whether current sweep is omitted."""
        s = max(0, min(self.state.sweep_idx, self.navigation_manager._sweep_count() - 1))
        if s in self.state.omitted_sweeps:
            self.OmitSweepButton.setText("Un-omit Sweep")
            self.OmitSweepButton.setToolTip("This sweep will be excluded from saving and stats.")
        else:
            self.OmitSweepButton.setText("Omit Sweep")
            self.OmitSweepButton.setToolTip("Mark this sweep to be excluded from saving and stats.")

    def on_omit_sweep_clicked(self):
        """Toggle omission for the current sweep and refresh plot/label."""
        st = self.state
        if self.navigation_manager._sweep_count() == 0:
            return
        s = max(0, min(st.sweep_idx, self.navigation_manager._sweep_count() - 1))
        if s in st.omitted_sweeps:
            st.omitted_sweeps.remove(s)
            try: self._log_status_message(f"Sweep {s+1}: included", 3000)
            except Exception: pass
        else:
            st.omitted_sweeps.add(s)
            try: self._log_status_message(f"Sweep {s+1}: omitted", 3000)
            except Exception: pass

        self._refresh_omit_button_label()
        self.redraw_main_plot()

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
        "Loading PlethApp...",
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