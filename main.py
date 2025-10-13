from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem, QAbstractItemView
from PyQt6.QtCore import QSettings, QTimer, Qt
from PyQt6.QtGui import QIcon

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

        self.settings = QSettings(ORG, APP)
        self.state = AppState()
        self.single_panel_mode = False  # flips True after stim channel selection

        # Notch filter parameters
        self.notch_filter_lower = None
        self.notch_filter_upper = None

        # Filter order
        self.filter_order = 4  # Default Butterworth filter order

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
        self.AutoUpdateGMMCheckbox.toggled.connect(self.on_auto_update_gmm_toggled)

        # --- Initialize Navigation Manager ---
        self.navigation_manager = NavigationManager(self)
        self.WindowRangeValue.setText("20")  # Default window length

        # --- Initialize Plot Manager (BEFORE signal connections that may trigger plotting) ---
        self.plot_manager = PlotManager(self)

        # --- Initialize Export Manager ---
        self.export_manager = ExportManager(self)

        # --- Peak-detect UI wiring ---
        self.ApplyPeakFindPushButton.setEnabled(False)  # stays disabled until threshold typed
        self.ApplyPeakFindPushButton.clicked.connect(self.on_apply_peak_find_clicked)
        # --- live threshold line on the main plot ---
        self._threshold_value = None
        self._thresh_line_artists = []
        self.ThreshVal.textChanged.connect(self._on_thresh_text_changed)


        self.ThreshVal.textChanged.connect(self._maybe_enable_peak_apply)
        self.PeakPromValue.textChanged.connect(self._maybe_enable_peak_apply)
        self.MinPeakDistValue.textChanged.connect(self._maybe_enable_peak_apply)

        # Default values for peak detection
        self.ThreshVal.setText("0.4")         # threshold
        self.PeakPromValue.setText("0.4")     # prominence
        self.MinPeakDistValue.setText("0.05") # min peak distance (seconds)

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
        # Example: set PLETHAPP_TESTING=1 && python main.py
        if os.environ.get('PLETHAPP_TESTING') == '1':
            print("[TESTING MODE] Auto-loading test file...")
            test_file = Path(r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\25121004.abf")
            if test_file.exists():
                QTimer.singleShot(100, lambda: self._auto_load_test_file(test_file))
            else:
                print(f"[TESTING MODE] Warning: Test file not found: {test_file}")

        # optional: keep a handle to the chosen dir
        self._curation_dir = None

    def _auto_load_test_file(self, file_path: Path):
        """Helper function for testing mode - auto-loads file and sets channels."""
        print(f"[TESTING MODE] Loading {file_path}...")
        self.load_file(file_path)

        # Poll until file is loaded, then set channels
        self._check_file_loaded_timer = QTimer()
        self._check_file_loaded_timer.timeout.connect(self._check_if_file_loaded)
        self._check_file_loaded_timer.start(100)  # Check every 100ms

    def _check_if_file_loaded(self):
        """Poll to see if file has finished loading."""
        # Check if state has been populated with data AND combos have been populated
        if (self.state.t is not None and
            self.state.sweeps is not None and
            len(self.state.sweeps) > 0 and
            self.AnalyzeChanSelect.count() > 0 and
            self.StimChanSelect.count() > 0):
            # File is loaded and combos are populated!
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
            self, "Select File(s)", last_dir, "Data Files (*.abf *.smrx);;ABF Files (*.abf);;SMRX Files (*.smrx);;All Files (*.*)"
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

        # Load single file or concatenate multiple files
        if len(file_paths) == 1:
            self.load_file(file_paths[0])
        else:
            self.load_multiple_files(file_paths)

    def load_file(self, path: Path):
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

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
            QMessageBox.critical(self, "Load error", str(e))
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

    def load_multiple_files(self, file_paths: List[Path]):
        """Load and concatenate multiple ABF files."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication, QMessageBox
        from PyQt6.QtCore import Qt

        # Validate files first
        valid, messages = abf_io.validate_files_for_concatenation(file_paths)

        if not valid:
            # Show error dialog
            QMessageBox.critical(self, "File Validation Failed", "\n".join(messages))
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
            QMessageBox.critical(self, "Load error", str(e))
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

        QMessageBox.information(
            self,
            "Files Loaded Successfully",
            message
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

                self.ApplyPeakFindPushButton.setEnabled(False)
                self._maybe_enable_peak_apply()
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


    def _maybe_enable_peak_apply(self):
        """
        Enable ApplyPeakFindPushButton if there's a numeric threshold AND we have data.
        We keep it simple: whenever user edits any peak param, allow pressing Apply again.
        """
        st = self.state
        has_data = bool(st.channel_names)
        txt = self.ThreshVal.text().strip()
        try:
            float(txt)
            ok_thresh = True
        except Exception:
            ok_thresh = False

        self.ApplyPeakFindPushButton.setEnabled(has_data and ok_thresh)

    ##################################################
    ##threshold plotting                            ##
    ##################################################
    def _on_thresh_text_changed(self, *_):
        """
        Called whenever the ThreshVal text changes.
        Parses the float (or clears if invalid) and refreshes the dashed line(s).
        """
        self._threshold_value = self._parse_float(self.ThreshVal)  # None if invalid/empty
        self._refresh_threshold_lines()

    def _on_region_threshold_changed(self, *_):
        """
        Called whenever eupnea or apnea threshold values change.
        Redraws the current sweep to update region overlays.
        """
        # Simply redraw current sweep, which will use the new threshold values
        self.redraw_main_plot()

    def _refresh_threshold_lines(self):
        """Delegate to PlotManager."""
        self.plot_manager.refresh_threshold_lines()



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

    def _on_peak_param_changed(self, *args):
        """Enable Apply if threshold is a valid number and we have data."""
        th = self._parse_float(self.ThreshVal)
        has_data = (self.state.t is not None) and (self.state.analyze_chan in (self.state.sweeps or {}))
        self.ApplyPeakFindPushButton.setEnabled(th is not None and has_data)



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


    def on_apply_peak_find_clicked(self):
        """
        Run peak detection on the ANALYZE channel for ALL sweeps,
        store indices per sweep, and redraw current sweep with peaks + breath markers.
        """
        st = self.state
        if not st.channel_names or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            self.ApplyPeakFindPushButton.setEnabled(False)
            return

        # Parse UI parameters
        def _num(line):
            txt = line.text().strip()
            return float(txt) if txt else None

        thresh = _num(self.ThreshVal)                 # required (button enabled only if valid)
        prom   = _num(self.PeakPromValue)             # optional
        min_d  = _num(self.MinPeakDistValue)          # seconds
        direction = "up"  # Always detect peaks above threshold for breathing signals

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

        # Disable until parameters change again
        self.ApplyPeakFindPushButton.setEnabled(False)

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

        # Second redraw: Show GMM-detected sniffing/eupnea regions
        print("[peak-detection] Redrawing plot with GMM results...")
        self.redraw_main_plot()

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
        Uses default features (if, ti, te, amp_insp, amp_exp, max_dinsp, max_dexp) and 2 clusters.
        Silently marks identified sniffing breaths with purple background.
        """
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        import numpy as np

        st = self.state

        # Check if we have breath data
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            print("[auto-gmm] No breath data available, skipping automatic GMM clustering")
            return

        # Default features for eupnea/sniffing separation (including derivative peaks)
        feature_keys = ["if", "ti", "te", "amp_insp", "amp_exp", "max_dinsp", "max_dexp"]
        n_clusters = 2

        print(f"\n[auto-gmm] Running automatic GMM clustering with {n_clusters} clusters...")
        print(f"[auto-gmm] Features: {', '.join(feature_keys)}")

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

            # Apply sniffing regions to plot with probabilities
            n_sniffing_breaths = self._apply_gmm_sniffing_regions(
                breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id
            )

            print(f"[auto-gmm] ✓ Marked {n_sniffing_breaths} sniffing breaths across sweeps")

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

        except Exception as e:
            print(f"[auto-gmm] Error during automatic GMM clustering: {e}")
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

        # Clear click callback and reset cursor
        self.plot_host.clear_click_callback()
        self.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    





    def on_auto_update_gmm_toggled(self, checked: bool):
        """Handle Auto-Update GMM checkbox toggle - run GMM when enabled."""
        if checked:
            # User just enabled auto-update - run GMM clustering if we have peaks
            st = self.state
            if st.peaks_by_sweep and len(st.peaks_by_sweep) > 0:
                print("[auto-update-gmm] Auto-update enabled - recalculating sniffing/eupnea regions...")
                self._run_automatic_gmm_clustering()
                self.redraw_main_plot()

    def on_spectral_analysis_clicked(self):
        """Open spectral analysis dialog and optionally apply notch filter."""
        st = self.state
        if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            QMessageBox.warning(self, "Spectral Analysis", "Please load data and select an analyze channel first.")
            return

        # Get current sweep data
        t, y = self._current_trace()
        if t is None or y is None:
            QMessageBox.warning(self, "Spectral Analysis", "No data available for current sweep.")
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
            try: self.statusbar.showMessage(f"Sweep {s+1}: included", 3000)
            except Exception: pass
        else:
            st.omitted_sweeps.add(s)
            try: self.statusbar.showMessage(f"Sweep {s+1}: omitted", 3000)
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
            QMessageBox.critical(self, "Scan error", f"Failed to scan folder:\n{root}\n\n{e}")
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
                self.statusbar.showMessage("No matching CSV files found in the selected folder.", 4000)
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
        """Generate a descriptive filename based on the files being consolidated.

        Returns:
            tuple: (proposed_filename, list of warnings)
        """
        warnings = []

        if not files:
            return "consolidated_data.xlsx", warnings

        # Parse filenames to extract common metadata
        # Expected format: Strain_Virus_Location_Sex_Animal_Stim_Power_ABF_Channel_*.csv
        # Also support older format without Location
        parsed_files = []
        for _, filepath in files:
            stem = filepath.stem  # filename without extension
            parts = stem.split('_')

            # Try to intelligently parse - check if we have location field
            # Location field is typically anatomical abbreviations like preBotC, RTN, etc.
            # Sex is typically M, F, or Unknown (short)
            # We'll use heuristics: if part[2] is 1-2 chars, it's probably sex (old format)

            if len(parts) >= 7 and len(parts[3]) <= 2:
                # New format with location: Strain_Virus_Location_Sex_Animal_Stim_Power_...
                parsed_files.append({
                    'strain': parts[0] if len(parts) > 0 else '',
                    'virus': parts[1] if len(parts) > 1 else '',
                    'location': parts[2] if len(parts) > 2 else '',
                    'sex': parts[3] if len(parts) > 3 else '',
                    'animal': parts[4] if len(parts) > 4 else '',
                    'stim': parts[5] if len(parts) > 5 else '',
                    'power': parts[6] if len(parts) > 6 else '',
                })
            elif len(parts) >= 6 and len(parts[2]) <= 2:
                # Old format without location: Strain_Virus_Sex_Animal_Stim_Power_...
                parsed_files.append({
                    'strain': parts[0] if len(parts) > 0 else '',
                    'virus': parts[1] if len(parts) > 1 else '',
                    'location': '',
                    'sex': parts[2] if len(parts) > 2 else '',
                    'animal': parts[3] if len(parts) > 3 else '',
                    'stim': parts[4] if len(parts) > 4 else '',
                    'power': parts[5] if len(parts) > 5 else '',
                })

        if not parsed_files:
            return "consolidated_data.xlsx", warnings

        # Find common values across all files and check for variations
        common = {}
        for key in ['strain', 'virus', 'location', 'sex', 'stim', 'power']:
            values = set(f[key] for f in parsed_files if f.get(key))
            if len(values) == 1:
                common[key] = values.pop()
            elif len(values) > 1:
                # Warn about different stimulation parameters
                if key == 'stim':
                    warnings.append(f"Multiple stimulation types detected: {', '.join(sorted(values))}")
                elif key == 'power':
                    warnings.append(f"Multiple laser powers detected: {', '.join(sorted(values))}")

        # Build descriptive filename from common fields
        parts = []
        if common.get('strain'):
            parts.append(common['strain'])
        if common.get('virus'):
            parts.append(common['virus'])
        if common.get('location'):
            parts.append(common['location'])
        if common.get('sex'):
            parts.append(common['sex'])

        # If multiple animals, indicate that
        animals = set(f['animal'] for f in parsed_files if f.get('animal'))
        if len(animals) == 1:
            parts.append(animals.pop())
        elif len(animals) > 1:
            parts.append(f"N{len(animals)}")  # Capital N

        if common.get('stim'):
            parts.append(common['stim'])
        if common.get('power'):
            parts.append(common['power'])

        # Add "consolidated" suffix
        parts.append("consolidated")

        if parts:
            return "_".join(parts) + ".xlsx", warnings
        else:
            return "consolidated_data.xlsx", warnings

    def on_consolidate_save_data_clicked(self):
        """Consolidate data from selected files into a single Excel file."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QProgressDialog
        import pandas as pd
        from pathlib import Path

        # Get all selected files from right list
        items = []
        for i in range(self.FilestoConsolidateList.count()):
            item = self.FilestoConsolidateList.item(i)
            if item:
                items.append(item)

        if not items:
            QMessageBox.warning(self, "Consolidate", "No files selected to consolidate.")
            return

        # Separate by file type
        means_files = []
        breaths_files = []
        events_files = []

        for item in items:
            meta = item.data(Qt.ItemDataRole.UserRole) or {}
            if meta.get("means"):
                means_files.append((meta["root"], Path(meta["means"])))
            if meta.get("breaths"):
                breaths_files.append((meta["root"], Path(meta["breaths"])))
            if meta.get("events"):
                events_files.append((meta["root"], Path(meta["events"])))

        if not means_files and not breaths_files and not events_files:
            QMessageBox.warning(self, "Consolidate", "No CSV files selected.")
            return

        # Choose save location first with intelligent default name
        files_for_naming = means_files or breaths_files
        proposed_filename, warnings = self._propose_consolidated_filename(files_for_naming)

        # Show warnings if any
        if warnings:
            warning_msg = "Warning about files being consolidated:\n\n" + "\n".join(f"• {w}" for w in warnings)
            warning_msg += "\n\nDo you want to continue?"
            reply = QMessageBox.question(
                self, "Consolidation Warning",
                warning_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        if means_files:
            default_name = str(means_files[0][1].parent / proposed_filename)
        elif breaths_files:
            default_name = str(breaths_files[0][1].parent / proposed_filename)
        else:
            default_name = proposed_filename

        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save consolidated data as...",
            default_name,
            "Excel Files (*.xlsx)"
        )

        if not save_path:
            return

        # Create progress dialog
        n_total_files = len(means_files) + len(breaths_files)
        progress = QProgressDialog("Consolidating data...", "Cancel", 0, 100, self)
        progress.setWindowTitle("PlethAnalysis")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # Process files
        try:
            consolidated_data = {}

            if means_files:
                progress.setLabelText(f"Processing time series data ({len(means_files)} files)...")
                progress.setValue(10)
                QApplication.processEvents()
                if progress.wasCanceled():
                    return

                consolidated_data.update(self._consolidate_means_files(means_files))
                progress.setValue(40)
                QApplication.processEvents()

            if breaths_files:
                if progress.wasCanceled():
                    return

                progress.setLabelText(f"Processing breath histograms ({len(breaths_files)} files)...")
                progress.setValue(50)
                QApplication.processEvents()

                histogram_data = self._consolidate_breaths_histograms(breaths_files)
                consolidated_data.update(histogram_data)

                progress.setValue(70)
                QApplication.processEvents()
                if progress.wasCanceled():
                    return

                # Extract sigh data
                progress.setLabelText("Extracting sigh data...")
                sighs_df = self._consolidate_breaths_sighs(breaths_files)
                consolidated_data['sighs'] = {
                    'time_series': sighs_df,
                    'raw_summary': {},
                    'norm_summary': {},
                    'windows': []
                }
                progress.setValue(80)
                QApplication.processEvents()

            if events_files:
                if progress.wasCanceled():
                    return

                progress.setLabelText(f"Processing events data ({len(events_files)} files)...")
                progress.setValue(82)
                QApplication.processEvents()

                events_df = self._consolidate_events(events_files)
                print(f"Events DataFrame shape: {events_df.shape}")
                print(f"Events columns: {events_df.columns.tolist()}")
                if len(events_df) > 0:
                    print(f"First few event types: {events_df['event_type'].unique()[:5]}")
                consolidated_data['events'] = {
                    'time_series': events_df,
                    'raw_summary': {},
                    'norm_summary': {},
                    'eupnea_summary': {},
                    'windows': []
                }

                # Process stimulus events separately
                progress.setLabelText(f"Processing stimulus data ({len(events_files)} files)...")
                progress.setValue(83)
                QApplication.processEvents()

                stimulus_df, stim_warnings = self._consolidate_stimulus(events_files)
                print(f"Stimulus DataFrame shape: {stimulus_df.shape}")
                if len(stimulus_df) > 0:
                    print(f"Stimulus columns: {stimulus_df.columns.tolist()}")
                    print(f"Stimulus rows: {len(stimulus_df)}")
                consolidated_data['stimulus'] = {
                    'time_series': stimulus_df,
                    'raw_summary': {},
                    'norm_summary': {},
                    'eupnea_summary': {},
                    'windows': []
                }

                # Add stimulus warnings to consolidated warnings
                if stim_warnings:
                    if '_warnings' not in consolidated_data:
                        consolidated_data['_warnings'] = []
                    consolidated_data['_warnings'].extend(stim_warnings)

            if progress.wasCanceled():
                return

            progress.setLabelText("Saving Excel file and generating charts...")
            progress.setValue(85)
            QApplication.processEvents()

            self._save_consolidated_to_excel(consolidated_data, Path(save_path))
            progress.setValue(100)

            n_files = len(means_files) + len(breaths_files)

            # Check for warnings from consolidation
            if '_warnings' in consolidated_data:
                warnings_text = consolidated_data['_warnings']
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Warning)
                msg_box.setWindowTitle("Consolidation Completed with Warnings")
                msg_box.setText(f"Consolidated {n_files} files successfully.\nSaved to: {save_path}\n\nHowever, some files required special handling:")
                msg_box.setDetailedText(warnings_text)
                msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg_box.exec()
            else:
                QMessageBox.information(
                    self, "Success",
                    f"Consolidated {n_files} files.\nSaved to: {save_path}"
                )

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Consolidation Error", f"Failed to consolidate:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            progress.close()

   

    def _consolidate_breaths_histograms(self, files: list[tuple[str, Path]]) -> dict:
        """
        Process breaths CSV files and create density histograms for each metric.
        Returns dict: {metric_name: DataFrame with histogram bins and densities per file}
        """
        import pandas as pd
        import numpy as np
        
        # Metrics to extract from breaths data
        metrics = ['if', 'amp_insp', 'amp_exp', 'area_insp', 'area_exp', 'ti', 'te', 'vent_proxy']
        
        # Regions in the breaths CSV
        regions = {
            'all': '',
            'baseline': '_baseline',
            'stim': '_stim', 
            'post': '_post'
        }
        
        consolidated = {}
        
        # Helper to calculate mean and SEM
        def calc_mean_sem(data_array):
            mean = np.nanmean(data_array, axis=1)
            n = np.sum(np.isfinite(data_array), axis=1)
            std = np.nanstd(data_array, axis=1, ddof=1)
            sem = np.where(n >= 2, std / np.sqrt(n), np.nan)
            return mean, sem
        
        # Process each metric (combine all regions into one sheet)
        for metric in metrics:
            combined_df = None
            
            # Process RAW data
            for region_name, suffix in regions.items():
                col_name = f"{metric}{suffix}" if suffix else metric
                
                # Collect data for this region
                all_data = []
                file_roots = []
                
                for root, path in files:
                    try:
                        df = pd.read_csv(path, low_memory=False)

                        if col_name in df.columns:
                            values = df[col_name].dropna().values
                            if len(values) > 0:
                                all_data.append(values)
                                file_roots.append(root)
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
                
                if not all_data:
                    continue
                
                # Common bin edges for this region
                all_combined = np.concatenate(all_data)
                n_bins = 30
                bin_edges = np.histogram_bin_edges(all_combined, bins=n_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                # Calculate density histogram for each file
                region_data = {f'bin_center_{region_name}': bin_centers}
                
                density_arrays = []
                for root, values in zip(file_roots, all_data):
                    counts, _ = np.histogram(values, bins=bin_edges)
                    # Convert to density (normalize so integral = 1)
                    bin_widths = np.diff(bin_edges)
                    density = counts / (counts.sum() * bin_widths)
                    region_data[f'{root}_{region_name}'] = density
                    density_arrays.append(density)
                
                # Calculate mean and SEM for this region
                if density_arrays:
                    density_matrix = np.column_stack(density_arrays)
                    mean, sem = calc_mean_sem(density_matrix)
                    region_data[f'mean_{region_name}'] = mean
                    region_data[f'sem_{region_name}'] = sem
                
                # Create DataFrame for this region
                region_df = pd.DataFrame(region_data)
                
                # Combine with other regions
                if combined_df is None:
                    combined_df = region_df
                else:
                    # Merge horizontally
                    combined_df = pd.concat([combined_df, region_df], axis=1)
                
                # Add blank column separator AFTER each region
                combined_df[''] = ''
            
            # Process NORMALIZED data
            for region_name, suffix in regions.items():
                col_name = f"{metric}{suffix}_norm" if suffix else f"{metric}_norm"
                
                # Collect normalized data for this region
                all_data_norm = []
                file_roots_norm = []
                
                for root, path in files:
                    try:
                        df = pd.read_csv(path, low_memory=False)

                        if col_name in df.columns:
                            values = df[col_name].dropna().values
                            if len(values) > 0:
                                all_data_norm.append(values)
                                file_roots_norm.append(root)
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
                
                if not all_data_norm:
                    continue
                
                # Common bin edges for normalized region
                all_combined_norm = np.concatenate(all_data_norm)
                n_bins = 30
                bin_edges_norm = np.histogram_bin_edges(all_combined_norm, bins=n_bins)
                bin_centers_norm = (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2
                
                # Calculate density histogram for each file
                region_data_norm = {f'bin_center_{region_name}_norm': bin_centers_norm}
                
                density_arrays_norm = []
                for root, values in zip(file_roots_norm, all_data_norm):
                    counts, _ = np.histogram(values, bins=bin_edges_norm)
                    bin_widths = np.diff(bin_edges_norm)
                    density = counts / (counts.sum() * bin_widths)
                    region_data_norm[f'{root}_{region_name}_norm'] = density
                    density_arrays_norm.append(density)
                
                # Calculate mean and SEM for normalized region
                if density_arrays_norm:
                    density_matrix_norm = np.column_stack(density_arrays_norm)
                    mean_norm, sem_norm = calc_mean_sem(density_matrix_norm)
                    region_data_norm[f'mean_{region_name}_norm'] = mean_norm
                    region_data_norm[f'sem_{region_name}_norm'] = sem_norm
                
                # Create DataFrame for this normalized region
                region_df_norm = pd.DataFrame(region_data_norm)
                
                # Merge horizontally
                combined_df = pd.concat([combined_df, region_df_norm], axis=1)
                
                # Add blank column separator AFTER each normalized region
                combined_df[''] = ''

            # Process EUPNEA-NORMALIZED data
            for region_name, suffix in regions.items():
                col_name = f"{metric}{suffix}_norm_eupnea" if suffix else f"{metric}_norm_eupnea"

                # Collect eupnea-normalized data for this region
                all_data_eupnea = []
                file_roots_eupnea = []

                for root, path in files:
                    try:
                        df = pd.read_csv(path, low_memory=False)

                        if col_name in df.columns:
                            values = df[col_name].dropna().values
                            if len(values) > 0:
                                all_data_eupnea.append(values)
                                file_roots_eupnea.append(root)
                    except Exception as e:
                        print(f"Error reading {path}: {e}")

                if not all_data_eupnea:
                    continue

                # Common bin edges for eupnea-normalized region
                all_combined_eupnea = np.concatenate(all_data_eupnea)
                n_bins = 30
                bin_edges_eupnea = np.histogram_bin_edges(all_combined_eupnea, bins=n_bins)
                bin_centers_eupnea = (bin_edges_eupnea[:-1] + bin_edges_eupnea[1:]) / 2

                # Calculate density histogram for each file
                region_data_eupnea = {f'bin_center_{region_name}_eupnea': bin_centers_eupnea}

                density_arrays_eupnea = []
                for root, values in zip(file_roots_eupnea, all_data_eupnea):
                    counts, _ = np.histogram(values, bins=bin_edges_eupnea)
                    bin_widths = np.diff(bin_edges_eupnea)
                    density = counts / (counts.sum() * bin_widths)
                    region_data_eupnea[f'{root}_{region_name}_eupnea'] = density
                    density_arrays_eupnea.append(density)

                # Calculate mean and SEM for eupnea-normalized region
                if density_arrays_eupnea:
                    density_matrix_eupnea = np.column_stack(density_arrays_eupnea)
                    mean_eupnea, sem_eupnea = calc_mean_sem(density_matrix_eupnea)
                    region_data_eupnea[f'mean_{region_name}_eupnea'] = mean_eupnea
                    region_data_eupnea[f'sem_{region_name}_eupnea'] = sem_eupnea

                # Create DataFrame for this eupnea-normalized region
                region_df_eupnea = pd.DataFrame(region_data_eupnea)

                # Merge horizontally
                combined_df = pd.concat([combined_df, region_df_eupnea], axis=1)

                # Add blank column separator AFTER each eupnea-normalized region
                combined_df[''] = ''

            if combined_df is not None:
                consolidated[f'{metric}_histogram'] = {
                    'time_series': combined_df,
                    'raw_summary': {},
                    'norm_summary': {},
                    'eupnea_summary': {},
                    'windows': []
                }

        return consolidated


    def _consolidate_events(self, files: list[tuple[str, Path]]) -> pd.DataFrame:
        """
        Consolidate events CSV files from multiple experiments (excluding stimulus events).
        Adds experiment, experiment_number, and global_sweep_number columns.
        """
        import pandas as pd
        import numpy as np

        all_events = []

        for exp_num, (root, path) in enumerate(files, start=1):
            try:
                df = pd.read_csv(path, low_memory=False)

                if len(df) > 0:
                    # Filter out stimulus events
                    if 'event_type' in df.columns:
                        df = df[~df['event_type'].str.contains('stimulus', case=False, na=False)]

                    if len(df) > 0:  # Check if we still have data after filtering
                        # Add experiment identifier columns
                        df.insert(0, 'experiment', root)
                        df.insert(1, 'experiment_number', exp_num)

                        # Calculate global sweep number
                        # Each experiment has sweeps numbered 1, 2, 3, etc.
                        # We need to add an offset based on previous experiments
                        if 'sweep' in df.columns:
                            # Find the maximum sweep number in this experiment
                            max_sweep = df['sweep'].max()
                            # Calculate offset (total sweeps from previous experiments)
                            sweep_offset = sum([pd.read_csv(files[i][1], low_memory=False)['sweep'].max()
                                              for i in range(exp_num - 1)
                                              if 'sweep' in pd.read_csv(files[i][1], low_memory=False).columns])
                            df.insert(2, 'global_sweep_number', df['sweep'] + sweep_offset)

                        all_events.append(df)

            except Exception as e:
                print(f"Error reading events file {path}: {e}")

        if all_events:
            return pd.concat(all_events, ignore_index=True)
        else:
            return pd.DataFrame()

    def _consolidate_stimulus(self, files: list[tuple[str, Path]]) -> tuple[pd.DataFrame, list[str]]:
        """
        Extract stimulus events and validate consistency across all experiments/sweeps.
        Returns: (stimulus_df with one instance, list of warnings)
        """
        import pandas as pd
        import numpy as np

        all_stimulus = []
        warnings = []
        reference_stim = None

        for exp_num, (root, path) in enumerate(files, start=1):
            try:
                df = pd.read_csv(path, low_memory=False)

                if 'event_type' in df.columns:
                    # Extract only stimulus events
                    stim_events = df[df['event_type'].str.contains('stimulus', case=False, na=False)].copy()

                    if len(stim_events) > 0:
                        # Determine which column names are used (could be start_time/end_time or t_start/t_end)
                        start_col = 'start_time' if 'start_time' in stim_events.columns else 't_start'
                        end_col = 'end_time' if 'end_time' in stim_events.columns else 't_end'

                        # Get unique stimulus events (should be same across all sweeps)
                        unique_stim = stim_events.drop_duplicates(subset=['event_type', start_col, end_col])

                        if reference_stim is None:
                            # First experiment sets the reference
                            reference_stim = unique_stim[[start_col, end_col]].values
                            # Store with experiment identifier
                            result_df = unique_stim.copy()
                            result_df.insert(0, 'experiment', root)
                            all_stimulus.append(result_df)
                        else:
                            # Validate against reference
                            current_stim = unique_stim[[start_col, end_col]].values

                            # Check if stimulus events match
                            if not (len(reference_stim) == len(current_stim) and
                                    np.allclose(reference_stim.astype(float),
                                               current_stim.astype(float),
                                               rtol=1e-5)):
                                warnings.append(f"WARNING: Stimulus timing differs in experiment '{root}'")
                                print(f"  Stimulus mismatch in {root}")

            except Exception as e:
                print(f"Error reading stimulus from {path}: {e}")
                import traceback
                traceback.print_exc()

        if all_stimulus:
            # Return only the first instance (they should all be identical)
            return all_stimulus[0], warnings
        else:
            return pd.DataFrame(), warnings

    

    # -------------------- NPZ-based fast consolidation helpers --------------------

    def _try_load_npz_v2(self, npz_path: Path) -> dict | None:
        """
        Try to load NPZ version 2 bundle with full timeseries data.
        Returns dict with data if successful, None otherwise.
        """
        # Silently skip if file doesn't exist (expected for old format)
        if not npz_path.exists():
            return None

        try:
            data = np.load(npz_path, allow_pickle=True)

            # Check if this is version 2
            version = data.get('npz_version', None)
            if version is None or int(version) < 2:
                # Silently skip v1 NPZ files (will use CSV fallback)
                return None

            # Verify required keys exist
            required_keys = ['timeseries_t', 'timeseries_keys']
            if not all(k in data for k in required_keys):
                print(f"[NPZ] Warning: {npz_path.name} missing required keys, using CSV fallback")
                return None

            return dict(data)
        except Exception as e:
            # Only warn on actual load errors (not missing files)
            print(f"[NPZ] Warning: Failed to load {npz_path.name}: {e}")
            return None

    def _extract_timeseries_from_npz(self, npz_data: dict, metric: str, variant: str = 'raw') -> tuple[np.ndarray, np.ndarray]:
        """
        Extract timeseries data for a specific metric from NPZ bundle.

        Args:
            npz_data: Loaded NPZ data dictionary
            metric: Metric name (e.g., 'if', 'amp_insp')
            variant: One of 'raw', 'norm', 'eupnea'

        Returns:
            (t, Y) where t is time vector and Y is (M, S) metric matrix
        """
        t = npz_data['timeseries_t']

        if variant == 'raw':
            key = f'ts_raw_{metric}'
        elif variant == 'norm':
            key = f'ts_norm_{metric}'
        elif variant == 'eupnea':
            key = f'ts_eupnea_{metric}'
        else:
            raise ValueError(f"Unknown variant: {variant}")

        Y = npz_data[key]
        return t, Y

    def _consolidate_from_npz_v2(self, npz_data_by_root: dict, files: list[tuple[str, Path]], metrics: list[str]) -> dict:
        """
        Fast consolidation using NPZ v2 bundles with pre-computed timeseries data.
        Much faster than CSV because:
        1. No CSV parsing overhead
        2. No interpolation needed (all files share same time base from stimulus alignment)
        3. Direct numpy array operations
        """
        import pandas as pd
        import numpy as np

        # Helper function to calculate mean and SEM
        def calc_mean_sem(data_array):
            """Calculate mean and SEM from array of values."""
            n = np.sum(np.isfinite(data_array), axis=1)
            mean = np.full(data_array.shape[0], np.nan)
            sem = np.full(data_array.shape[0], np.nan)
            valid_rows = n > 0
            if valid_rows.any():
                mean[valid_rows] = np.nanmean(data_array[valid_rows, :], axis=1)
                sem_rows = n >= 2
                if sem_rows.any():
                    std = np.nanstd(data_array[sem_rows, :], axis=1, ddof=1)
                    sem[sem_rows] = std / np.sqrt(n[sem_rows])
            return mean, sem

        # Helper function to calculate window mean
        def window_mean(t, y, t_start, t_end):
            """Calculate mean of y values where t is between t_start and t_end."""
            mask = (t >= t_start) & (t < t_end)
            if mask.sum() == 0:
                return np.nan
            return np.nanmean(y[mask])

        # Define time windows for summary stats
        windows = [
            ('Baseline (-10 to 0s)', -10.0, 0.0),
            ('Baseline (-5 to 0s)', -5.0, 0.0),
            ('Stim (0-15s)', 0.0, 15.0),
            ('Stim (0-5s)', 0.0, 5.0),
            ('Stim (5-10s)', 5.0, 10.0),
            ('Stim (10-15s)', 10.0, 15.0),
            ('Post (15-25s)', 15.0, 25.0),
            ('Post (15-20s)', 15.0, 20.0),
            ('Post (20-25s)', 20.0, 25.0),
            ('Post (25-30s)', 25.0, 30.0),
        ]

        # Determine common time base by scanning all files
        # This allows experiments with different durations to be consolidated
        all_t_mins = []
        all_t_maxs = []
        all_steps = []

        for root, _ in files:
            t_file = npz_data_by_root[root]['timeseries_t']
            all_t_mins.append(t_file.min())
            all_t_maxs.append(t_file.max())
            if len(t_file) > 1:
                all_steps.append(np.median(np.diff(t_file)))

        # Create common time grid spanning the union of all time ranges
        t_common_min = min(all_t_mins)
        t_common_max = max(all_t_maxs)
        t_common_step = np.median(all_steps) if all_steps else 0.1

        # Generate uniform time grid
        t_common = np.arange(t_common_min, t_common_max + t_common_step/2, t_common_step)

        print(f"[NPZ] Common time grid: {t_common_min:.2f}s to {t_common_max:.2f}s, step={t_common_step:.4f}s ({len(t_common)} points)")

        consolidated = {}

        # Get first file root for checking metric existence
        first_root = files[0][0]

        # Process each metric
        for metric in metrics:
            # Check if metric exists in first NPZ
            test_key = f'ts_raw_{metric}'
            if test_key not in npz_data_by_root[first_root]:
                continue

            # Build all columns as dict first to avoid fragmentation
            data_dict = {'t': t_common}

            # Process each variant: raw, norm, eupnea
            # Note: eupnea suffix is '_eupnea' not '_norm_eupnea' to match CSV consolidation
            for variant, suffix in [('raw', ''), ('norm', '_norm'), ('eupnea', '_eupnea')]:
                # Collect data from all files
                all_means = []
                file_means = {}  # Store for window calculations

                for root, _ in files:
                    t_file, Y = self._extract_timeseries_from_npz(npz_data_by_root[root], metric, variant)

                    # Compute mean across sweeps (axis 1)
                    y_mean_file = np.nanmean(Y, axis=1)

                    # Interpolate to common time grid if needed
                    if len(t_file) != len(t_common) or not np.allclose(t_file, t_common, rtol=0.01):
                        from scipy.interpolate import interp1d
                        # Use linear interpolation, extrapolate with NaN outside range
                        interp_func = interp1d(t_file, y_mean_file, kind='linear',
                                              bounds_error=False, fill_value=np.nan)
                        y_mean = interp_func(t_common)
                    else:
                        y_mean = y_mean_file

                    all_means.append(y_mean)
                    file_means[root] = (t_file, y_mean_file)  # Store original for window calculations

                    # Add individual file column
                    data_dict[f'{root}{suffix}'] = y_mean

                # Stack all means into matrix (M, num_files)
                all_means_matrix = np.column_stack(all_means)

                # Compute mean and SEM across files
                mean, sem = calc_mean_sem(all_means_matrix)
                data_dict[f'mean{suffix}'] = mean
                data_dict[f'sem{suffix}'] = sem

                # Add separator column
                data_dict[f' {suffix}'] = ''

            # Create DataFrame once with all columns
            result_df = pd.DataFrame(data_dict)

            # Build summary data dicts (for Excel summary section)
            # These use the original (non-interpolated) data for window calculations
            # We need to collect these from each variant's file_means
            raw_summary = {}
            norm_summary = {}
            eupnea_summary = {}

            # Re-extract data for each variant to populate summary dicts
            for root, _ in files:
                # Raw data
                t_file, Y = self._extract_timeseries_from_npz(npz_data_by_root[root], metric, 'raw')
                y_mean_file = np.nanmean(Y, axis=1)
                raw_summary[root] = (t_file, y_mean_file)

                # Normalized data
                t_file, Y = self._extract_timeseries_from_npz(npz_data_by_root[root], metric, 'norm')
                y_mean_file = np.nanmean(Y, axis=1)
                norm_summary[root] = (t_file, y_mean_file)

                # Eupnea-normalized data
                t_file, Y = self._extract_timeseries_from_npz(npz_data_by_root[root], metric, 'eupnea')
                y_mean_file = np.nanmean(Y, axis=1)
                eupnea_summary[root] = (t_file, y_mean_file)

            # Package result in same format as CSV consolidation
            consolidated[metric] = {
                'time_series': result_df,
                'raw_summary': raw_summary,
                'norm_summary': norm_summary,
                'eupnea_summary': eupnea_summary,
                'windows': windows
            }

        print(f"[NPZ] ✓ Fast consolidation complete")
        return consolidated

    def _consolidate_means_files(self, files: list[tuple[str, Path]]) -> dict:
        """
        Consolidate means_by_time CSV files.
        Interpolates all data to a common time base and adds summary statistics.
        Returns dict: {metric_name: DataFrame with time series + summary stats}
        """
        import pandas as pd
        import numpy as np
        from scipy.interpolate import interp1d

        metrics = [
            'if', 'amp_insp', 'amp_exp', 'area_insp', 'area_exp',
            'ti', 'te', 'vent_proxy'
        ]

        # Try fast NPZ-based consolidation first (5-10× faster)
        npz_data_by_root = {}
        npz_success_count = 0
        for root, csv_path in files:
            # Look for NPZ bundle next to the CSV (handle both naming patterns)
            if csv_path.name.endswith('_timeseries.csv'):
                npz_name = csv_path.name.replace('_timeseries.csv', '_bundle.npz')
            elif csv_path.name.endswith('_means_by_time.csv'):
                npz_name = csv_path.name.replace('_means_by_time.csv', '_bundle.npz')
            else:
                continue  # Unknown CSV pattern

            npz_path = csv_path.parent / npz_name
            npz_data = self._try_load_npz_v2(npz_path)
            if npz_data is not None:
                npz_data_by_root[root] = npz_data
                npz_success_count += 1

        # If all files have NPZ v2, use fast path (with interpolation support)
        if npz_success_count == len(files) and npz_success_count > 0:
            print(f"[NPZ Fast Path] Loading {npz_success_count} files from NPZ bundles (v2)...")
            return self._consolidate_from_npz_v2(npz_data_by_root, files, metrics)
        elif npz_success_count > 0:
            print(f"[NPZ] Only {npz_success_count}/{len(files)} files have NPZ v2, falling back to CSV...")
        else:
            print(f"[CSV] No NPZ v2 bundles found, using CSV files...")

        # Determine common time base by scanning all files
        # This allows experiments with different durations to be consolidated
        print("Scanning all files to determine common time range...")
        all_t_mins = []
        all_t_maxs = []
        all_steps = []

        for root, path in files:
            df_temp = pd.read_csv(path, low_memory=False)
            t_temp = df_temp['t'].values
            all_t_mins.append(t_temp.min())
            all_t_maxs.append(t_temp.max())
            if len(t_temp) > 1:
                all_steps.append(np.median(np.diff(t_temp)))

        # Create common time grid spanning the union of all time ranges
        t_common_min = min(all_t_mins)
        t_common_max = max(all_t_maxs)
        t_common_step = np.median(all_steps) if all_steps else 0.1

        # Generate uniform time grid
        t_common = np.arange(t_common_min, t_common_max + t_common_step/2, t_common_step)

        print(f"Common time grid: {t_common_min:.2f}s to {t_common_max:.2f}s, step={t_common_step:.4f}s ({len(t_common)} points)")

        # Load first file for column checking
        first_root, first_path = files[0]
        df_first = pd.read_csv(first_path, low_memory=False)

        # Track files with potential issues
        warning_messages = []
        files_needing_interpolation = []
        files_with_poor_overlap = []
        files_with_different_sampling = []

        consolidated = {}
        
        # Helper function to calculate mean and SEM
        def calc_mean_sem(data_array):
            """Calculate mean and SEM from array of values."""
            # Count valid values per row
            n = np.sum(np.isfinite(data_array), axis=1)

            # Initialize outputs with NaN
            mean = np.full(data_array.shape[0], np.nan)
            sem = np.full(data_array.shape[0], np.nan)

            # Only calculate for rows with at least one valid value
            valid_rows = n > 0
            if valid_rows.any():
                mean[valid_rows] = np.nanmean(data_array[valid_rows, :], axis=1)

                # Only calculate SEM for rows with at least 2 values
                sem_rows = n >= 2
                if sem_rows.any():
                    std = np.nanstd(data_array[sem_rows, :], axis=1, ddof=1)
                    sem[sem_rows] = std / np.sqrt(n[sem_rows])

            return mean, sem
        
        # Helper function to calculate window mean
        def window_mean(t, y, t_start, t_end):
            """Calculate mean of y values where t is between t_start and t_end."""
            mask = (t >= t_start) & (t < t_end)
            if mask.sum() == 0:
                return np.nan
            return np.nanmean(y[mask])
        
        # Define time windows for summary stats
        windows = [
            ('Baseline (-10 to 0s)', -10.0, 0.0),
            ('Baseline (-5 to 0s)', -5.0, 0.0),
            ('Stim (0-15s)', 0.0, 15.0),
            ('Stim (0-5s)', 0.0, 5.0),
            ('Stim (5-10s)', 5.0, 10.0),
            ('Stim (10-15s)', 10.0, 15.0),
            ('Post (15-25s)', 15.0, 25.0),
            ('Post (15-20s)', 15.0, 20.0),
            ('Post (20-25s)', 20.0, 25.0),
            ('Post (25-30s)', 25.0, 30.0),
        ]
        
        # Process each metric (combining raw, time-normalized, and eupnea-normalized in same sheet)
        for metric in metrics:
            metric_mean_col = f"{metric}_mean"
            metric_norm_col = f"{metric}_norm_mean"
            metric_eupnea_col = f"{metric}_norm_eupnea_mean"

            # Check if metric exists
            if metric_mean_col not in df_first.columns:
                continue

            result_df = pd.DataFrame({'t': t_common})

            # Store data for summary calculations
            raw_data_dict = {}
            norm_data_dict = {}
            eupnea_data_dict = {}
            
            # Collect raw data from all files
            raw_data_cols = []
            for root, path in files:
                df = pd.read_csv(path, low_memory=False)

                if metric_mean_col not in df.columns:
                    print(f"Warning: {metric_mean_col} not found in {root}")
                    continue

                t_file = df['t'].values
                y_file = df[metric_mean_col].values

                # Check time range and sampling
                t_file_min, t_file_max = t_file.min(), t_file.max()
                t_file_step = np.median(np.diff(t_file)) if len(t_file) > 1 else np.nan

                # Calculate overlap percentage
                overlap_start = max(t_common_min, t_file_min)
                overlap_end = min(t_common_max, t_file_max)
                overlap_range = overlap_end - overlap_start
                common_range = t_common_max - t_common_min
                overlap_pct = 100 * overlap_range / common_range if common_range > 0 else 0

                # Check for different sampling rates
                if not np.isnan(t_file_step) and not np.isnan(t_common_step):
                    sampling_diff_pct = 100 * abs(t_file_step - t_common_step) / t_common_step
                    if sampling_diff_pct > 10 and root != first_root:
                        files_with_different_sampling.append(
                            f"{root}: {t_file_step:.4f}s vs reference {t_common_step:.4f}s ({sampling_diff_pct:.1f}% difference)"
                        )

                # Check for poor overlap
                if overlap_pct < 80 and root != first_root:
                    files_with_poor_overlap.append(
                        f"{root}: {overlap_pct:.1f}% overlap (range: {t_file_min:.1f} to {t_file_max:.1f}s)"
                    )

                # Always interpolate to common time grid (different files may have different durations)
                mask = np.isfinite(y_file)
                if mask.sum() >= 2:
                    try:
                        # Check if exact match first (optimization)
                        if len(t_file) == len(t_common) and np.allclose(t_file, t_common, rtol=1e-5, atol=1e-8):
                            result_df[root] = y_file
                            raw_data_dict[root] = (t_common, y_file)
                        else:
                            # Interpolate to common grid
                            if root not in files_needing_interpolation:
                                files_needing_interpolation.append(root)
                            print(f"Interpolating {root} to common time base for {metric}")
                            f_interp = interp1d(
                                t_file[mask], y_file[mask],
                                kind='linear',
                                bounds_error=False,
                                fill_value=np.nan
                            )
                            y_interp = f_interp(t_common)
                            result_df[root] = y_interp
                            raw_data_dict[root] = (t_common, y_interp)

                            # Count how many points were extrapolated (NaN after interpolation)
                            n_extrapolated = np.sum(np.isnan(y_interp) & ~np.isnan(t_common))
                            if n_extrapolated > 0:
                                extrap_pct = 100 * n_extrapolated / len(t_common)
                                if extrap_pct > 5:
                                    print(f"  Warning: {extrap_pct:.1f}% of points extrapolated (outside data range)")
                    except Exception as e:
                        print(f"Error interpolating {root} for {metric}: {e}")
                        result_df[root] = np.nan
                else:
                    print(f"  Warning: Insufficient data points for interpolation in {root}")
                    result_df[root] = np.nan

                raw_data_cols.append(root)
            
            # Calculate raw mean and SEM
            if raw_data_cols:
                raw_data = result_df[raw_data_cols].values
                result_df['mean'], result_df['sem'] = calc_mean_sem(raw_data)
            
            # Collect normalized data from all files
            if metric_norm_col in df_first.columns:
                norm_data_cols = []
                for root, path in files:
                    df = pd.read_csv(path, low_memory=False)
                    
                    if metric_norm_col not in df.columns:
                        continue
                    
                    t_file = df['t'].values
                    y_file = df[metric_norm_col].values
                    
                    norm_col_name = f"{root}_norm"

                    # Always interpolate to common time grid
                    mask = np.isfinite(y_file)
                    if mask.sum() >= 2:
                        try:
                            # Check if exact match first (optimization)
                            if len(t_file) == len(t_common) and np.allclose(t_file, t_common, rtol=1e-5, atol=1e-8):
                                result_df[norm_col_name] = y_file
                                norm_data_dict[root] = (t_common, y_file)
                            else:
                                f_interp = interp1d(
                                    t_file[mask], y_file[mask],
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=np.nan
                                )
                                y_interp = f_interp(t_common)
                                result_df[norm_col_name] = y_interp
                                norm_data_dict[root] = (t_common, y_interp)
                        except:
                            result_df[norm_col_name] = np.nan
                    else:
                        result_df[norm_col_name] = np.nan
                    
                    norm_data_cols.append(norm_col_name)
                
                # Calculate normalized mean and SEM
                if norm_data_cols:
                    norm_data = result_df[norm_data_cols].values
                    result_df['mean_norm'], result_df['sem_norm'] = calc_mean_sem(norm_data)

            # Collect eupnea-normalized data from all files
            if metric_eupnea_col in df_first.columns:
                eupnea_data_cols = []
                for root, path in files:
                    df = pd.read_csv(path, low_memory=False)

                    if metric_eupnea_col not in df.columns:
                        continue

                    t_file = df['t'].values
                    y_file = df[metric_eupnea_col].values

                    eupnea_col_name = f"{root}_eupnea"

                    # Always interpolate to common time grid
                    mask = np.isfinite(y_file)
                    if mask.sum() >= 2:
                        try:
                            # Check if exact match first (optimization)
                            if len(t_file) == len(t_common) and np.allclose(t_file, t_common, rtol=1e-5, atol=1e-8):
                                result_df[eupnea_col_name] = y_file
                                eupnea_data_dict[root] = (t_common, y_file)
                            else:
                                f_interp = interp1d(
                                    t_file[mask], y_file[mask],
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=np.nan
                                )
                                y_interp = f_interp(t_common)
                                result_df[eupnea_col_name] = y_interp
                                eupnea_data_dict[root] = (t_common, y_interp)
                        except:
                            result_df[eupnea_col_name] = np.nan
                    else:
                        result_df[eupnea_col_name] = np.nan

                    eupnea_data_cols.append(eupnea_col_name)

                # Calculate eupnea-normalized mean and SEM
                if eupnea_data_cols:
                    eupnea_data = result_df[eupnea_data_cols].values
                    result_df['mean_eupnea'], result_df['sem_eupnea'] = calc_mean_sem(eupnea_data)

            # Insert blank columns between data blocks
            # First blank: before time-normalized data
            norm_start_idx = None
            for i, col in enumerate(result_df.columns):
                if '_norm' in str(col) and '_eupnea' not in str(col):
                    norm_start_idx = i
                    break
            if norm_start_idx is not None:
                result_df.insert(norm_start_idx, '', '')

            # Second blank: before eupnea-normalized data
            eupnea_start_idx = None
            for i, col in enumerate(result_df.columns):
                if '_eupnea' in str(col):
                    eupnea_start_idx = i
                    break
            if eupnea_start_idx is not None:
                result_df.insert(eupnea_start_idx, ' ', '')

            # Build summary statistics (as rows below the time series)
            # This will be saved as a separate section in Excel
            consolidated[metric] = {
                'time_series': result_df,
                'raw_summary': raw_data_dict,
                'norm_summary': norm_data_dict,
                'eupnea_summary': eupnea_data_dict,
                'windows': windows
            }

        # Build warning summary
        if files_needing_interpolation or files_with_poor_overlap or files_with_different_sampling:
            warning_parts = []

            if files_needing_interpolation:
                warning_parts.append("FILES REQUIRING INTERPOLATION:")
                warning_parts.append(f"Reference file (no interpolation): {first_root}")
                warning_parts.append(f"Time range: {t_common_min:.2f} to {t_common_max:.2f}s")
                warning_parts.append(f"Sample interval: {t_common_step:.4f}s\n")
                for f in files_needing_interpolation:
                    warning_parts.append(f"  • {f}")
                warning_parts.append("")

            if files_with_different_sampling:
                warning_parts.append("FILES WITH DIFFERENT SAMPLING RATES:")
                for msg in files_with_different_sampling:
                    warning_parts.append(f"  • {msg}")
                warning_parts.append("")

            if files_with_poor_overlap:
                warning_parts.append("FILES WITH POOR TIME OVERLAP (<80%):")
                for msg in files_with_poor_overlap:
                    warning_parts.append(f"  • {msg}")
                warning_parts.append("")

            # Store warning message for display after processing
            consolidated['_warnings'] = '\n'.join(warning_parts)

        return consolidated


    def _consolidate_breaths_sighs(self, files: list[tuple[str, Path]]) -> pd.DataFrame:
        """
        Extract all breaths marked as sighs (is_sigh == 1) from breaths CSV files.
        Adds experiment_number and global_sweep_number columns.
        Returns DataFrame with sigh breaths from all files.
        """
        import pandas as pd
        import numpy as np

        # Columns to extract for raw data
        raw_cols = ['sweep', 'breath', 't', 'region', 'is_sigh',
                    'if', 'amp_insp', 'amp_exp', 'area_insp', 'area_exp',
                    'ti', 'te', 'vent_proxy']

        # Columns to extract for normalized data
        norm_cols = ['sweep', 'breath', 't', 'region', 'is_sigh',
                    'if_norm', 'amp_insp_norm', 'amp_exp_norm', 'area_insp_norm', 'area_exp_norm',
                    'ti_norm', 'te_norm', 'vent_proxy_norm']

        all_sighs_raw = []
        all_sighs_norm = []

        for exp_num, (root, path) in enumerate(files, start=1):
            try:
                df = pd.read_csv(path, low_memory=False)

                # Filter for sighs (is_sigh == 1)
                if 'is_sigh' in df.columns:
                    sigh_mask = df['is_sigh'] == 1

                    # Extract raw sigh data
                    available_raw_cols = [col for col in raw_cols if col in df.columns]
                    if available_raw_cols and sigh_mask.sum() > 0:
                        sigh_df_raw = df.loc[sigh_mask, available_raw_cols].copy()
                        sigh_df_raw.insert(0, 'file', root)
                        sigh_df_raw.insert(1, 'experiment_number', exp_num)

                        # Calculate global sweep number
                        if 'sweep' in sigh_df_raw.columns:
                            # Calculate offset from previous experiments
                            sweep_offset = sum([pd.read_csv(files[i][1], low_memory=False)['sweep'].max()
                                              for i in range(exp_num - 1)
                                              if 'sweep' in pd.read_csv(files[i][1], low_memory=False).columns])
                            sigh_df_raw.insert(2, 'global_sweep_number', sigh_df_raw['sweep'] + sweep_offset)

                        all_sighs_raw.append(sigh_df_raw)

                    # Extract normalized sigh data
                    available_norm_cols = [col for col in norm_cols if col in df.columns]
                    if available_norm_cols and sigh_mask.sum() > 0:
                        sigh_df_norm = df.loc[sigh_mask, available_norm_cols].copy()
                        sigh_df_norm.insert(0, 'file', root)
                        all_sighs_norm.append(sigh_df_norm)

            except Exception as e:
                print(f"Error reading sighs from {path}: {e}")

        # Combine all sigh data
        if all_sighs_raw:
            combined_raw = pd.concat(all_sighs_raw, ignore_index=True)
        else:
            combined_raw = pd.DataFrame(columns=['file', 'experiment_number', 'global_sweep_number'] + raw_cols)

        if all_sighs_norm:
            combined_norm = pd.concat(all_sighs_norm, ignore_index=True)
        else:
            combined_norm = pd.DataFrame(columns=['file'] + norm_cols)

        # Combine raw and normalized with blank column separator
        combined_raw[''] = ''
        combined_sighs = pd.concat([combined_raw, combined_norm], axis=1)

        return combined_sighs

   

    def _save_consolidated_to_excel(self, consolidated: dict, save_path: Path):
        """Save consolidated dataframes to a single Excel file with multiple sheets."""
        import pandas as pd
        import numpy as np
        from openpyxl import load_workbook
        from openpyxl.styles import Font
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.chart import ScatterChart, Reference, Series
        from openpyxl.chart.marker import Marker  # Fixed typo here
        
        # Helper to calculate window means
        def window_mean(t, y, t_start, t_end):
            mask = (t >= t_start) & (t < t_end)
            if mask.sum() == 0:
                return np.nan
            return np.nanmean(y[mask])
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            # Define sheet order: time series metrics first, then sighs, then events/stimulus, then histograms
            time_series_metrics = []
            sighs_sheets = []
            event_sheets = []
            histogram_sheets = []

            for metric_name in consolidated.keys():
                if metric_name == '_warnings':
                    continue
                elif metric_name == 'sighs':
                    sighs_sheets.append(metric_name)
                elif metric_name in ['events', 'stimulus']:
                    event_sheets.append(metric_name)
                elif '_histogram' in metric_name:
                    histogram_sheets.append(metric_name)
                else:
                    time_series_metrics.append(metric_name)

            # Process sheets in desired order: time series -> sighs -> events/stimulus -> histograms
            ordered_sheets = time_series_metrics + sighs_sheets + event_sheets + histogram_sheets

            for metric_name in ordered_sheets:
                data_dict = consolidated[metric_name]
                time_series_df = data_dict['time_series']
                raw_summary = data_dict.get('raw_summary', {})
                norm_summary = data_dict.get('norm_summary', {})
                eupnea_summary = data_dict.get('eupnea_summary', {})
                windows = data_dict.get('windows', [])
                
                sheet_name = metric_name[:31]
                
                # Write time series data starting at A1
                time_series_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0, startcol=0)
                
                # Only add summary if it exists (means files have it, histogram files don't)
                if raw_summary:
                    # Calculate starting column for summary (after time series + 2 blank columns)
                    summary_start_col = len(time_series_df.columns) + 2
                    
                    # Get the worksheet to write summary data
                    worksheet = writer.sheets[sheet_name]
                    
                    # Build raw summary DataFrame
                    raw_summary_rows = []
                    for root in raw_summary.keys():
                        t, y = raw_summary[root]
                        row = {'File': root}
                        for window_name, t_start, t_end in windows:
                            row[window_name] = window_mean(t, y, t_start, t_end)
                        raw_summary_rows.append(row)
                    
                    if raw_summary_rows:
                        raw_summary_df = pd.DataFrame(raw_summary_rows)
                        
                        # Write raw summary starting at top right
                        for r_idx, row in enumerate(dataframe_to_rows(raw_summary_df, index=False, header=True)):
                            for c_idx, value in enumerate(row):
                                worksheet.cell(row=r_idx + 1, column=summary_start_col + c_idx, value=value)
                    
                    # Build normalized summary DataFrame  
                    if norm_summary:
                        # Start normalized summary after raw summary + 2 blank columns
                        norm_start_col = summary_start_col + len(raw_summary_df.columns) + 2
                        
                        norm_summary_rows = []
                        for root in norm_summary.keys():
                            t, y = norm_summary[root]
                            row = {'File': root}
                            for window_name, t_start, t_end in windows:
                                row[f"{window_name}_norm"] = window_mean(t, y, t_start, t_end)
                            norm_summary_rows.append(row)
                        
                        if norm_summary_rows:
                            norm_summary_df = pd.DataFrame(norm_summary_rows)

                            # Write normalized summary to the right of raw summary
                            for r_idx, row in enumerate(dataframe_to_rows(norm_summary_df, index=False, header=True)):
                                for c_idx, value in enumerate(row):
                                    worksheet.cell(row=r_idx + 1, column=norm_start_col + c_idx, value=value)

                    # Build eupnea-normalized summary DataFrame
                    if eupnea_summary:
                        # Start eupnea summary after normalized summary + 2 blank columns
                        eupnea_start_col = norm_start_col + len(norm_summary_df.columns) + 2 if norm_summary else summary_start_col

                        eupnea_summary_rows = []
                        for root in eupnea_summary.keys():
                            t, y = eupnea_summary[root]
                            row = {'File': root}
                            for window_name, t_start, t_end in windows:
                                row[f"{window_name}_eupnea"] = window_mean(t, y, t_start, t_end)
                            eupnea_summary_rows.append(row)

                        if eupnea_summary_rows:
                            eupnea_summary_df = pd.DataFrame(eupnea_summary_rows)

                            # Write eupnea summary to the right of normalized summary
                            for r_idx, row in enumerate(dataframe_to_rows(eupnea_summary_df, index=False, header=True)):
                                for c_idx, value in enumerate(row):
                                    worksheet.cell(row=r_idx + 1, column=eupnea_start_col + c_idx, value=value)

                print(f"Saved sheet: {sheet_name}")
        
        # Apply bold formatting and add charts
        wb = load_workbook(save_path)
        bold_font = Font(bold=True)
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            print(f"Processing sheet: '{sheet_name}'")

            # Bold header row only (much faster than bolding entire columns)
            header_row = ws[1]

            for cell in header_row:
                cell.font = bold_font
            
            # Add charts based on sheet type
            if sheet_name == 'sighs':
                print(f"Matched sighs sheet condition for '{sheet_name}'")
                # Create sigh timeline chart
                self._add_sighs_chart(ws, header_row)

            elif sheet_name == 'events':
                print(f"Matched events sheet condition for '{sheet_name}'")
                # Create eupnea and apnea timeline charts
                self._add_events_charts(ws, header_row)

            # Stimulus sheet - no charts needed (just tabular data)
            elif sheet_name == 'stimulus':
                pass  # No charts for stimulus

            # Add charts for histogram sheets
            elif '_histogram' in sheet_name:
                regions = ['all', 'baseline', 'stim', 'post']
                
                # Chart 1: Raw means overlay (reverted to original style)
                chart1 = ScatterChart()
                chart1.title = f"{sheet_name} - Raw Mean Histograms"
                chart1.style = 2
                chart1.x_axis.title = "Bin Center"
                chart1.y_axis.title = "Density"

                # Enable axes display
                chart1.x_axis.delete = False
                chart1.y_axis.delete = False

                # Enable axis tick marks and labels (major only, no minor)
                chart1.x_axis.tickLblPos = "nextTo"  # Changed from "low" to fix label positioning
                chart1.y_axis.tickLblPos = "nextTo"
                chart1.x_axis.majorTickMark = "out"
                chart1.y_axis.majorTickMark = "out"
                chart1.x_axis.minorTickMark = "none"
                chart1.y_axis.minorTickMark = "none"

                # Disable gridlines
                chart1.x_axis.majorGridlines = None
                chart1.y_axis.majorGridlines = None

                # Set y-axis to start at 0
                chart1.y_axis.scaling.min = 0
                
                for region in regions:
                    bin_col = None
                    mean_col = None
                    
                    for idx, cell in enumerate(header_row, start=1):
                        if cell.value == f'bin_center_{region}':
                            bin_col = idx
                        elif cell.value == f'mean_{region}':
                            mean_col = idx
                    
                    if bin_col and mean_col:
                        xvalues = Reference(ws, min_col=bin_col, min_row=2, max_row=ws.max_row)
                        yvalues = Reference(ws, min_col=mean_col, min_row=2, max_row=ws.max_row)
                        
                        series = Series(yvalues, xvalues, title=region)
                        chart1.series.append(series)
                
                chart1.width = 10
                chart1.height = 6
                ws.add_chart(chart1, f"A{ws.max_row + 3}")

                # Chart 2: Time-normalized means overlay
                chart2 = ScatterChart()
                chart2.title = f"{sheet_name} - Time-Normalized Mean Histograms"
                chart2.style = 2
                chart2.x_axis.title = "Bin Center (time-normalized)"
                chart2.y_axis.title = "Density"

                # Enable axes display
                chart2.x_axis.delete = False
                chart2.y_axis.delete = False

                # Enable axis tick marks and labels (major only, no minor)
                chart2.x_axis.tickLblPos = "nextTo"
                chart2.y_axis.tickLblPos = "nextTo"
                chart2.x_axis.majorTickMark = "out"
                chart2.y_axis.majorTickMark = "out"
                chart2.x_axis.minorTickMark = "none"
                chart2.y_axis.minorTickMark = "none"

                # Disable gridlines
                chart2.x_axis.majorGridlines = None
                chart2.y_axis.majorGridlines = None

                # Set y-axis to start at 0
                chart2.y_axis.scaling.min = 0

                for region in regions:
                    bin_col_norm = None
                    mean_col_norm = None

                    for idx, cell in enumerate(header_row, start=1):
                        if cell.value == f'bin_center_{region}_norm':
                            bin_col_norm = idx
                        elif cell.value == f'mean_{region}_norm':
                            mean_col_norm = idx

                    if bin_col_norm and mean_col_norm:
                        xvalues = Reference(ws, min_col=bin_col_norm, min_row=2, max_row=ws.max_row)
                        yvalues = Reference(ws, min_col=mean_col_norm, min_row=2, max_row=ws.max_row)

                        series = Series(yvalues, xvalues, title=f"{region}_norm")
                        chart2.series.append(series)

                chart2.width = 10
                chart2.height = 6
                ws.add_chart(chart2, f"K{ws.max_row + 3}")

                # Chart 3: Eupnea-normalized means overlay
                chart3 = ScatterChart()
                chart3.title = f"{sheet_name} - Eupnea-Normalized Mean Histograms"
                chart3.style = 2
                chart3.x_axis.title = "Bin Center (eupnea-normalized)"
                chart3.y_axis.title = "Density"

                # Enable axes display
                chart3.x_axis.delete = False
                chart3.y_axis.delete = False

                # Enable axis tick marks and labels (major only, no minor)
                chart3.x_axis.tickLblPos = "nextTo"
                chart3.y_axis.tickLblPos = "nextTo"
                chart3.x_axis.majorTickMark = "out"
                chart3.y_axis.majorTickMark = "out"
                chart3.x_axis.minorTickMark = "none"
                chart3.y_axis.minorTickMark = "none"

                # Disable gridlines
                chart3.x_axis.majorGridlines = None
                chart3.y_axis.majorGridlines = None

                # Set y-axis to start at 0
                chart3.y_axis.scaling.min = 0

                for region in regions:
                    bin_col_eupnea = None
                    mean_col_eupnea = None

                    for idx, cell in enumerate(header_row, start=1):
                        if cell.value == f'bin_center_{region}_eupnea':
                            bin_col_eupnea = idx
                        elif cell.value == f'mean_{region}_eupnea':
                            mean_col_eupnea = idx

                    if bin_col_eupnea and mean_col_eupnea:
                        xvalues = Reference(ws, min_col=bin_col_eupnea, min_row=2, max_row=ws.max_row)
                        yvalues = Reference(ws, min_col=mean_col_eupnea, min_row=2, max_row=ws.max_row)

                        series = Series(yvalues, xvalues, title=f"{region}_eupnea")
                        chart3.series.append(series)

                chart3.width = 10
                chart3.height = 6
                ws.add_chart(chart3, f"U{ws.max_row + 3}")  # Position to the right of chart2

            # Add charts for time series sheets (not histograms)
            elif '_histogram' not in sheet_name:
                t_col = None
                mean_col = None
                sem_col = None
                mean_norm_col = None
                sem_norm_col = None
                mean_eupnea_col = None
                sem_eupnea_col = None

                # Find t, mean, sem, mean_norm, sem_norm, mean_eupnea, and sem_eupnea columns
                for idx, cell in enumerate(header_row, start=1):
                    if cell.value == 't':
                        t_col = idx
                    elif cell.value == 'mean':
                        mean_col = idx
                    elif cell.value == 'sem':
                        sem_col = idx
                    elif cell.value == 'mean_norm':
                        mean_norm_col = idx
                    elif cell.value == 'sem_norm':
                        sem_norm_col = idx
                    elif cell.value == 'mean_eupnea':
                        mean_eupnea_col = idx
                    elif cell.value == 'sem_eupnea':
                        sem_eupnea_col = idx
                
                # Position charts near top of sheet (row 5)
                chart_row = 5

                # Find individual file columns for plotting
                # Raw data columns: between 'mean' and first '_norm' column
                raw_file_cols = []
                norm_file_cols = []
                eupnea_file_cols = []

                for idx, cell in enumerate(header_row, start=1):
                    col_name = str(cell.value or '')
                    # Skip time, mean, sem, and blank columns
                    if col_name in ['t', 'mean', 'sem', 'mean_norm', 'sem_norm', 'mean_eupnea', 'sem_eupnea', '', 'None']:
                        continue
                    # Skip columns with numeric or empty names
                    if not col_name or col_name.isspace():
                        continue
                    if '_eupnea' in col_name:
                        eupnea_file_cols.append(idx)
                    elif '_norm' in col_name:
                        norm_file_cols.append(idx)
                    else:
                        # Must be a raw data file column
                        if t_col and idx > t_col:  # Make sure it's after the time column
                            raw_file_cols.append(idx)

                # Chart 1: Raw mean vs time
                if t_col and mean_col:
                    chart1 = ScatterChart()
                    chart1.title = f"{sheet_name} - Mean vs Time (Raw)"
                    chart1.style = 13

                    chart1.x_axis.title = "Time (s)"
                    chart1.y_axis.title = sheet_name

                    # Enable axes display
                    chart1.x_axis.delete = False
                    chart1.y_axis.delete = False

                    # Enable axis tick marks and labels (major only, no minor)
                    chart1.x_axis.tickLblPos = "nextTo"
                    chart1.y_axis.tickLblPos = "nextTo"
                    chart1.x_axis.majorTickMark = "out"
                    chart1.y_axis.majorTickMark = "out"
                    chart1.x_axis.minorTickMark = "none"
                    chart1.y_axis.minorTickMark = "none"

                    # Position y-axis on the left side
                    chart1.y_axis.crosses = "min"

                    # Disable gridlines
                    chart1.x_axis.majorGridlines = None
                    chart1.y_axis.majorGridlines = None

                    # Hide legend
                    chart1.legend = None

                    xvalues = Reference(ws, min_col=t_col, min_row=2, max_row=ws.max_row)

                    # Add individual file traces in light gray
                    for file_col in raw_file_cols:
                        yvalues_file = Reference(ws, min_col=file_col, min_row=2, max_row=ws.max_row)
                        series_file = Series(yvalues_file, xvalues, title="")
                        series_file.marker = Marker('none')
                        series_file.smooth = True
                        series_file.graphicalProperties.line.solidFill = "D3D3D3"  # Light gray
                        series_file.graphicalProperties.line.width = 8000  # Thinner than mean
                        chart1.series.append(series_file)

                    # Add mean line on top
                    yvalues = Reference(ws, min_col=mean_col, min_row=2, max_row=ws.max_row)
                    series = Series(yvalues, xvalues, title="Mean")
                    series.marker = Marker('none')
                    series.smooth = True
                    series.graphicalProperties.line.solidFill = "4472C4"  # Solid blue line
                    series.graphicalProperties.line.width = 12700  # 1pt line width
                    chart1.series.append(series)

                    chart1.width = 10
                    chart1.height = 6
                    ws.add_chart(chart1, f"A{chart_row}")
                
                # Chart 2: Time-normalized mean vs time
                if t_col and mean_norm_col:
                    chart2 = ScatterChart()
                    chart2.title = f"{sheet_name} - Mean vs Time (Time-Normalized)"
                    chart2.style = 13

                    chart2.x_axis.title = "Time (s)"
                    chart2.y_axis.title = f"{sheet_name} (time-normalized)"

                    # Enable axes display
                    chart2.x_axis.delete = False
                    chart2.y_axis.delete = False

                    # Enable axis tick marks and labels (major only, no minor)
                    chart2.x_axis.tickLblPos = "nextTo"
                    chart2.y_axis.tickLblPos = "nextTo"
                    chart2.x_axis.majorTickMark = "out"
                    chart2.y_axis.majorTickMark = "out"
                    chart2.x_axis.minorTickMark = "none"
                    chart2.y_axis.minorTickMark = "none"

                    # Position y-axis on the left side
                    chart2.y_axis.crosses = "min"

                    # Disable gridlines
                    chart2.x_axis.majorGridlines = None
                    chart2.y_axis.majorGridlines = None

                    # Hide legend
                    chart2.legend = None

                    xvalues = Reference(ws, min_col=t_col, min_row=2, max_row=ws.max_row)

                    # Add individual file traces in light gray
                    for file_col in norm_file_cols:
                        yvalues_file = Reference(ws, min_col=file_col, min_row=2, max_row=ws.max_row)
                        series_file = Series(yvalues_file, xvalues, title="")
                        series_file.marker = Marker('none')
                        series_file.smooth = True
                        series_file.graphicalProperties.line.solidFill = "D3D3D3"  # Light gray
                        series_file.graphicalProperties.line.width = 8000  # Thinner than mean
                        chart2.series.append(series_file)

                    # Add mean line on top
                    yvalues = Reference(ws, min_col=mean_norm_col, min_row=2, max_row=ws.max_row)
                    series = Series(yvalues, xvalues, title="Mean")
                    series.marker = Marker('none')
                    series.smooth = True
                    series.graphicalProperties.line.solidFill = "ED7D31"  # Solid orange line
                    series.graphicalProperties.line.width = 12700  # 1pt line width
                    chart2.series.append(series)

                    chart2.width = 10
                    chart2.height = 6
                    ws.add_chart(chart2, f"K{chart_row}")

                # Chart 3: Eupnea-normalized mean vs time
                if t_col and mean_eupnea_col:
                    chart3 = ScatterChart()
                    chart3.title = f"{sheet_name} - Mean vs Time (Eupnea-Normalized)"
                    chart3.style = 13

                    chart3.x_axis.title = "Time (s)"
                    chart3.y_axis.title = f"{sheet_name} (eupnea-normalized)"

                    # Enable axes display
                    chart3.x_axis.delete = False
                    chart3.y_axis.delete = False

                    # Enable axis tick marks and labels (major only, no minor)
                    chart3.x_axis.tickLblPos = "nextTo"
                    chart3.y_axis.tickLblPos = "nextTo"
                    chart3.x_axis.majorTickMark = "out"
                    chart3.y_axis.majorTickMark = "out"
                    chart3.x_axis.minorTickMark = "none"
                    chart3.y_axis.minorTickMark = "none"

                    # Position y-axis on the left side
                    chart3.y_axis.crosses = "min"

                    # Disable gridlines
                    chart3.x_axis.majorGridlines = None
                    chart3.y_axis.majorGridlines = None

                    # Hide legend
                    chart3.legend = None

                    xvalues = Reference(ws, min_col=t_col, min_row=2, max_row=ws.max_row)

                    # Add individual file traces in light gray
                    for file_col in eupnea_file_cols:
                        yvalues_file = Reference(ws, min_col=file_col, min_row=2, max_row=ws.max_row)
                        series_file = Series(yvalues_file, xvalues, title="")
                        series_file.marker = Marker('none')
                        series_file.smooth = True
                        series_file.graphicalProperties.line.solidFill = "D3D3D3"  # Light gray
                        series_file.graphicalProperties.line.width = 8000  # Thinner than mean
                        chart3.series.append(series_file)

                    # Add mean line on top
                    yvalues = Reference(ws, min_col=mean_eupnea_col, min_row=2, max_row=ws.max_row)
                    series = Series(yvalues, xvalues, title="Mean")
                    series.marker = Marker('none')
                    series.smooth = True
                    series.graphicalProperties.line.solidFill = "70AD47"  # Solid green line
                    series.graphicalProperties.line.width = 12700  # 1pt line width
                    chart3.series.append(series)

                    chart3.width = 10
                    chart3.height = 6
                    ws.add_chart(chart3, f"U{chart_row}")  # Position to the right of chart2

        wb.save(save_path)
        print(f"Applied bold formatting and charts. Consolidated Excel file saved: {save_path}")


    def _add_events_charts(self, ws, header_row):
        """Add eupnea, apnea, and sniffing timeline charts to the events sheet."""
        from openpyxl.chart import ScatterChart, Reference, Series
        from openpyxl.chart.marker import Marker
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from openpyxl.drawing.image import Image as XLImage
        from PIL import Image as PILImage
        import io
        import numpy as np

        print(f"_add_events_charts called for sheet with {ws.max_row} rows")

        # Find required columns (handle both naming conventions)
        exp_num_col = None
        sweep_col = None
        global_sweep_col = None
        event_type_col = None
        t_start_col = None
        t_end_col = None

        for idx, cell in enumerate(header_row, start=1):
            if cell.value == 'experiment_number':
                exp_num_col = idx
            elif cell.value == 'sweep':
                sweep_col = idx
            elif cell.value == 'global_sweep_number':
                global_sweep_col = idx
            elif cell.value == 'event_type':
                event_type_col = idx
            elif cell.value in ['t_start', 'start_time']:
                t_start_col = idx
            elif cell.value in ['t_end', 'end_time']:
                t_end_col = idx

        if not all([exp_num_col, global_sweep_col, event_type_col, t_start_col, t_end_col]):
            print(f"Events sheet missing required columns for charts. Found columns: {[cell.value for cell in header_row]}")
            return

        # Read data from sheet
        events = []
        for row_idx in range(2, ws.max_row + 1):
            exp_num = ws.cell(row=row_idx, column=exp_num_col).value
            global_sweep = ws.cell(row=row_idx, column=global_sweep_col).value
            event_type = ws.cell(row=row_idx, column=event_type_col).value
            t_start = ws.cell(row=row_idx, column=t_start_col).value
            t_end = ws.cell(row=row_idx, column=t_end_col).value

            if all([exp_num is not None, global_sweep is not None,
                    event_type is not None, t_start is not None, t_end is not None]):
                events.append({
                    'exp_num': int(exp_num),
                    'global_sweep': int(global_sweep),
                    'event_type': str(event_type).lower(),
                    't_start': float(t_start),
                    't_end': float(t_end)
                })

        if not events:
            print("No events found for chart generation")
            return

        # Separate eupnea, apnea, and sniffing events
        eupnea_events = [e for e in events if 'eupnea' in e['event_type']]
        apnea_events = [e for e in events if 'apnea' in e['event_type']]
        sniffing_events = [e for e in events if 'sniff' in e['event_type']]

        if not eupnea_events and not apnea_events and not sniffing_events:
            print("No eupnea, apnea, or sniffing events found for charts")
            return

        # Create colormap for experiments
        n_experiments = max([e['exp_num'] for e in events])
        colors = plt.cm.tab10(np.linspace(0, 1, n_experiments))

        # Position charts at the top (starting at row 1)
        # We'll place them to the right of the data columns
        chart_start_row = 1

        # Create eupnea chart
        if eupnea_events:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for event in eupnea_events:
                color = colors[event['exp_num'] - 1]
                ax1.plot([event['t_start'], event['t_end']],
                        [event['global_sweep'], event['global_sweep']],
                        color=color, linewidth=2, solid_capstyle='butt')

            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Global Sweep Number')
            ax1.set_title('Eupnea Periods Across Experiments')
            ax1.grid(True, alpha=0.3)

            # Save figure to bytes
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=100, bbox_inches='tight')
            buf1.seek(0)
            plt.close(fig1)

            # Insert image into Excel
            img1 = XLImage(buf1)
            img1.width = 600
            img1.height = 360
            ws.add_image(img1, f"A{chart_start_row}")
            print(f"Added eupnea chart at A{chart_start_row}")

        # Create apnea chart
        if apnea_events:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for event in apnea_events:
                color = colors[event['exp_num'] - 1]
                ax2.plot([event['t_start'], event['t_end']],
                        [event['global_sweep'], event['global_sweep']],
                        color=color, linewidth=2, solid_capstyle='butt')

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Global Sweep Number')
            ax2.set_title('Apnea Periods Across Experiments')
            ax2.grid(True, alpha=0.3)

            # Save figure to bytes
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=100, bbox_inches='tight')
            buf2.seek(0)
            plt.close(fig2)

            # Insert image into Excel
            img2 = XLImage(buf2)
            img2.width = 600
            img2.height = 360
            # Position to the right of first chart (column K)
            ws.add_image(img2, f"K{chart_start_row}")
            print(f"Added apnea chart at K{chart_start_row}")

        # Create sniffing chart
        if sniffing_events:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for event in sniffing_events:
                color = colors[event['exp_num'] - 1]
                ax3.plot([event['t_start'], event['t_end']],
                        [event['global_sweep'], event['global_sweep']],
                        color=color, linewidth=2, solid_capstyle='butt')

            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Global Sweep Number')
            ax3.set_title('Sniffing Bouts Across Experiments')
            ax3.grid(True, alpha=0.3)

            # Save figure to bytes
            buf3 = io.BytesIO()
            fig3.savefig(buf3, format='png', dpi=100, bbox_inches='tight')
            buf3.seek(0)
            plt.close(fig3)

            # Insert image into Excel
            img3 = XLImage(buf3)
            img3.width = 600
            img3.height = 360
            # Position to the right of apnea chart (column U)
            ws.add_image(img3, f"U{chart_start_row}")
            print(f"Added sniffing chart at U{chart_start_row}")

    def _add_sighs_chart(self, ws, header_row):
        """Add sigh timeline scatter plot to the sighs sheet."""
        import matplotlib.pyplot as plt
        from openpyxl.drawing.image import Image as XLImage
        import io
        import numpy as np

        print(f"_add_sighs_chart called for sheet with {ws.max_row} rows")

        # Find required columns
        exp_num_col = None
        global_sweep_col = None
        t_col = None

        for idx, cell in enumerate(header_row, start=1):
            if cell.value == 'experiment_number':
                exp_num_col = idx
            elif cell.value == 'global_sweep_number':
                global_sweep_col = idx
            elif cell.value == 't':
                t_col = idx

        if not all([exp_num_col, global_sweep_col, t_col]):
            print(f"Sighs sheet missing required columns for chart. Found columns: {[cell.value for cell in header_row]}")
            return

        # Read data from sheet
        sighs = []
        for row_idx in range(2, ws.max_row + 1):
            exp_num = ws.cell(row=row_idx, column=exp_num_col).value
            global_sweep = ws.cell(row=row_idx, column=global_sweep_col).value
            t = ws.cell(row=row_idx, column=t_col).value

            if all([exp_num is not None, global_sweep is not None, t is not None]):
                sighs.append({
                    'exp_num': int(exp_num),
                    'global_sweep': int(global_sweep),
                    't': float(t)
                })

        if not sighs:
            print("No sigh data found for chart generation")
            return

        # Create colormap for experiments
        n_experiments = max([s['exp_num'] for s in sighs])
        colors = plt.cm.tab10(np.linspace(0, 1, n_experiments))

        # Position chart at the top (starting at row 1)
        chart_start_row = 1

        # Create sigh scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each sigh as a yellow asterisk, colored by experiment
        for sigh in sighs:
            color = colors[sigh['exp_num'] - 1]
            ax.scatter(sigh['t'], sigh['global_sweep'],
                      marker='*', s=200, color='gold', edgecolors=color, linewidths=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Global Sweep Number')
        ax.set_title('Sigh Events Across Experiments')
        ax.grid(True, alpha=0.3)

        # Save figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        # Insert image into Excel
        img = XLImage(buf)
        img.width = 600
        img.height = 360
        ws.add_image(img, f"A{chart_start_row}")
        print(f"Added sigh chart at A{chart_start_row}")

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