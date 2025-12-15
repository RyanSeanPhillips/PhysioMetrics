"""
ExportManager - Handles all data export operations.

Extracted from main.py for better maintainability and easier customization
for different experiment types.
"""

import re
import csv
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QDialog, QProgressDialog, QApplication
from PyQt6.QtCore import Qt, Qt as QtCore_Qt
from core import metrics, telemetry
from dialogs import SaveMetaDialog

# Enable line profiling when running with kernprof -l
# Otherwise, @profile decorator is a no-op (zero overhead)
try:
    profile  # Check if already defined by kernprof
except NameError:
    def profile(func):
        """No-op decorator when not profiling."""
        return func


def is_sample_in_omitted_region(sweep_idx: int, sample_idx: int, omitted_ranges: dict) -> bool:
    """Check if a sample index falls within an omitted region for a given sweep.

    Args:
        sweep_idx: Sweep number
        sample_idx: Sample index within the sweep
        omitted_ranges: Dictionary mapping sweep index to list of (start, end) sample ranges

    Returns:
        True if the sample is within an omitted region, False otherwise
    """
    if sweep_idx not in omitted_ranges:
        return False

    for (start_idx, end_idx) in omitted_ranges[sweep_idx]:
        if start_idx <= sample_idx <= end_idx:
            return True
    return False


def create_omitted_mask(sweep_idx: int, trace_length: int, omitted_ranges: dict, omitted_sweeps: set) -> np.ndarray:
    """Create a boolean mask for omitted regions in a trace.

    Args:
        sweep_idx: Sweep number
        trace_length: Length of the trace in samples
        omitted_ranges: Dictionary mapping sweep index to list of (start, end) sample ranges
        omitted_sweeps: Set of sweep indices that are fully omitted

    Returns:
        Boolean array where True = keep (not omitted), False = omit
    """
    # Start with all samples kept
    mask = np.ones(trace_length, dtype=bool)

    # If full sweep is omitted, mask everything
    if sweep_idx in omitted_sweeps:
        mask[:] = False
        return mask

    # Otherwise, mask partial regions
    if sweep_idx in omitted_ranges:
        for (start_idx, end_idx) in omitted_ranges[sweep_idx]:
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, trace_length - 1))
            end_idx = max(0, min(end_idx, trace_length - 1))
            mask[start_idx:end_idx+1] = False

    return mask


class ExportManager:
    """Manages all data export operations for the main window."""

    # =========================================================================
    # METRIC CATEGORIES FOR EXPORT
    # =========================================================================

    # TIMESERIES CSV: Core metrics for time-aligned plotting and consolidation
    # These are the most useful metrics for physiological analysis
    # Must include all metrics required by consolidation_manager.py
    _TIMESERIES_METRICS = [
        "if",           # Instantaneous frequency (Hz) - primary breathing rate
        "amp_insp",     # Inspiratory amplitude - breath depth
        "amp_exp",      # Expiratory amplitude - breath depth
        "area_insp",    # Inspiratory area - volume proxy
        "area_exp",     # Expiratory area - volume proxy
        "ti",           # Inspiratory time - required by consolidation
        "te",           # Expiratory time - required by consolidation
        "ti_te_ratio",  # Ti/Te ratio (duty cycle)
        "ibi",          # Inter-breath interval
        "vent_proxy",   # Ventilation proxy - required by consolidation
        "regularity",   # Breathing regularity (RMSSD)
        "sniff_conf",   # Sniffing confidence (GMM)
        "eupnea_conf",  # Eupnea confidence (GMM)
    ]

    # ML-ONLY METRICS: Neighbor comparison features for merge detection and classification
    # These are only useful for ML training, not for physiological analysis
    _ML_ONLY_METRICS = {
        # Neighbor comparison features (for merge detection)
        "gap_to_next_norm", "trough_ratio_next", "onset_height_ratio",
        "prom_asymmetry", "prom_asymmetry_signed", "amplitude_normalized",
        "total_prominence", "trough_asymmetry_signed",
        # Next/prev peak comparisons
        "next_peak_amplitude", "prev_peak_amplitude",
        "amplitude_ratio_to_next", "amplitude_ratio_to_prev",
        "amplitude_diff_to_next_signed", "amplitude_diff_to_prev_signed",
        "next_peak_prominence", "prev_peak_prominence",
        "prominence_ratio_to_next", "prominence_ratio_to_prev",
        "next_peak_onset_height_ratio", "prev_peak_onset_height_ratio",
        "next_peak_ti", "prev_peak_ti", "ti_ratio_to_next", "ti_ratio_to_prev",
        "next_peak_te", "prev_peak_te", "te_ratio_to_next", "te_ratio_to_prev",
        # Probability metrics (for ML classification)
        "p_noise", "p_breath", "p_edge", "p_edge_all_peaks",
        # Shape metrics (for sigh detection ML)
        "n_inflections", "rise_variability", "n_shoulder_peaks",
        "shoulder_prominence", "rise_autocorr", "peak_sharpness",
        "trough_sharpness", "skewness", "kurtosis",
        # Derivatives (rarely needed for analysis)
        "d1", "d2",
        # Region detection (binary, not useful as continuous metrics)
        "eupnic", "apnea",
    }

    # LEGACY: Old exclusion list - kept for reference during transition
    # TODO: Remove after confirming new lists work correctly
    _EXCLUDE_FOR_CSV = {
        "d1", "d2", "eupnic", "apnea",
        # Phase 2.2: Sigh detection features (for future ML use)
        "n_inflections", "rise_variability", "n_shoulder_peaks",
        "shoulder_prominence", "rise_autocorr", "peak_sharpness",
        "trough_sharpness", "skewness", "kurtosis",
        # Phase 2.3 Group A: Shape & ratio metrics (for future ML use)
        "peak_to_trough", "amp_ratio", "ti_te_ratio",
        "area_ratio", "total_area", "ibi",
        # Phase 2.3 Group B: Normalized metrics (for future ML use)
        "amp_insp_norm", "amp_exp_norm", "peak_to_trough_norm",
        "prominence_norm", "ibi_norm", "ti_norm", "te_norm",
    }

    def __init__(self, main_window):
        """
        Initialize the ExportManager.

        Args:
            main_window: Reference to MainWindow instance
        """
        self.window = main_window

    def _is_breath_sniffing(self, sweep_idx, breath_idx, onsets):
        """
        Check if a breath is marked as sniffing based on GMM results.
        Simple approach: Check if breath midpoint falls in any sniffing region.

        Args:
            sweep_idx: Sweep index
            breath_idx: Breath index (0-based)
            onsets: Array of onset indices

        Returns:
            True if breath is in a sniffing region, False otherwise
        """
        st = self.window.state
        sniff_regions = st.sniff_regions_by_sweep.get(sweep_idx, [])
        if not sniff_regions or breath_idx >= len(onsets) - 1:
            return False

        # Get breath time range (onset to next onset)
        t_start = st.t[onsets[breath_idx]]
        t_end = st.t[onsets[breath_idx + 1]]
        t_mid = (t_start + t_end) / 2.0

        # Check if midpoint falls in any sniffing region
        for (region_start, region_end) in sniff_regions:
            if region_start <= t_mid <= region_end:
                return True
        return False

    def _show_message_box(self, icon, title, text, parent=None):
        """
        Show a message box with selectable text for easy copying.

        Args:
            icon: QMessageBox.Icon (Information, Warning, Critical, Question)
            title: Dialog title
            text: Message text
            parent: Parent widget (defaults to self.window)

        Returns:
            QMessageBox result
        """
        if parent is None:
            parent = self.window

        msg_box = QMessageBox(parent)
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        return msg_box.exec()

    def _get_export_strategy(self, experiment_type: str):
        """
        Get the appropriate export strategy based on experiment type.

        Args:
            experiment_type: One of "30hz_stim", "hargreaves", "licking"

        Returns:
            ExportStrategy instance for the specified experiment type
        """
        from export.strategies import Stim30HzStrategy, HargreavesStrategy, LickingStrategy

        if experiment_type == "30hz_stim":
            return Stim30HzStrategy(self.window)
        elif experiment_type == "hargreaves":
            return HargreavesStrategy(self.window)
        elif experiment_type == "licking":
            return LickingStrategy(self.window)
        else:
            # Default to 30Hz
            print(f"[export] Unknown experiment type '{experiment_type}', using default (30Hz)")
            return Stim30HzStrategy(self.window)

    def _is_pulse_experiment(self, kept_sweeps: list) -> bool:
        """
        Detect if this is a brief pulse perturbation experiment.

        Returns True if:
        - Stim channel exists
        - Each sweep has exactly 1 pulse
        - Pulse width < 1.0s

        Args:
            kept_sweeps: List of sweep indices to check

        Returns:
            True if this is a pulse experiment (single brief pulse per sweep)
        """
        st = self.window.state

        # Debug logging
        print(f"[Pulse Detection] Checking {len(kept_sweeps)} sweeps...")
        print(f"[Pulse Detection] stim_chan: {st.stim_chan}")

        if not st.stim_chan:
            print("[Pulse Detection] No stim channel - NOT a pulse experiment")
            return False

        pulse_count_by_sweep = []
        for s in kept_sweeps[:5]:  # Check first 5 sweeps for debugging
            spans = st.stim_spans_by_sweep.get(s, [])
            metrics = st.stim_metrics_by_sweep.get(s, {})
            pulse_width = metrics.get("pulse_width_s", None)
            print(f"[Pulse Detection] Sweep {s}: {len(spans)} pulses, pulse_width={pulse_width}")

        for s in kept_sweeps:
            spans = st.stim_spans_by_sweep.get(s, [])
            if len(spans) != 1:
                print(f"[Pulse Detection] Sweep {s} has {len(spans)} pulses (need exactly 1) - NOT a pulse experiment")
                return False  # Need exactly 1 pulse per sweep

            # Check pulse width
            metrics = st.stim_metrics_by_sweep.get(s, {})
            pulse_width = metrics.get("pulse_width_s", None)
            if pulse_width is None:
                print(f"[Pulse Detection] Sweep {s} has no pulse_width_s metric - NOT a pulse experiment")
                return False
            if pulse_width >= 1.0:
                print(f"[Pulse Detection] Sweep {s} has pulse_width={pulse_width}s (need <1.0s) - NOT a pulse experiment")
                return False

        print(f"[Pulse Detection] ✓ All {len(kept_sweeps)} sweeps have exactly 1 pulse <1.0s - IS a pulse experiment")
        return True

    def _load_save_dialog_history(self) -> dict:
        """Load autocomplete history for the Save Data dialog from QSettings."""
        history = {
            'strain': [],
            'virus': [],
            'location': [],
            'stim': [],
            'power': [],
            'animal': []
        }

        for key in history.keys():
            saved_list = self.window.settings.value(f"save_history/{key}", [])
            if isinstance(saved_list, str):
                # Single value, convert to list
                saved_list = [saved_list] if saved_list else []
            history[key] = saved_list if saved_list else []

        return history

    def _get_current_file_info(self, abf_stem: str) -> dict:
        """Get file info from state and Project Builder for the current file.

        Args:
            abf_stem: The filename stem (without extension) of the current file

        Returns:
            Dict with keys: protocol, keywords, and any other relevant info
        """
        file_info = {
            'protocol': '',
            'keywords': ''
        }

        st = self.window.state

        # Get protocol from state.file_info (populated during file load)
        if hasattr(st, 'file_info') and st.file_info:
            # file_info is a list of dicts for multi-file loading
            for info in st.file_info:
                if 'protocol' in info:
                    file_info['protocol'] = info.get('protocol', '')
                    break  # Use first file's protocol

        # Try to get keywords from Project Builder table model
        if hasattr(self.window, 'project_builder') and self.window.project_builder:
            pb = self.window.project_builder
            # Check if file table model exists and has the file
            if hasattr(pb, 'file_table_model') and pb.file_table_model:
                model = pb.file_table_model
                # Search for this file in the model
                for row in range(model.rowCount()):
                    row_data = model._files[row] if row < len(model._files) else {}
                    file_name = row_data.get('file_name', '')
                    # Match by filename (without extension)
                    if file_name and Path(file_name).stem == abf_stem:
                        file_info['keywords'] = row_data.get('keywords', '')
                        # Also get protocol from table if not already set
                        if not file_info['protocol']:
                            file_info['protocol'] = row_data.get('protocol', '')
                        break

        return file_info

    def _update_save_dialog_history(self, vals: dict):
        """Update autocomplete history with new values from the Save Data dialog."""
        max_history = 20  # Keep last 20 unique values for each field

        for key in ['strain', 'virus', 'location', 'stim', 'power', 'animal']:
            value = vals.get(key, '').strip()
            if not value:
                continue

            # Get current history
            current = self.window.settings.value(f"save_history/{key}", [])
            if isinstance(current, str):
                current = [current] if current else []
            elif not isinstance(current, list):
                current = []

            # Remove value if already exists (we'll re-add at front)
            if value in current:
                current.remove(value)

            # Add to front
            current.insert(0, value)

            # Trim to max_history
            current = current[:max_history]

            # Save back
            self.window.settings.setValue(f"save_history/{key}", current)

    def _load_last_save_values(self) -> dict:
        """Load the last used values for the Save Data dialog to auto-populate fields."""
        return {
            'strain': self.window.settings.value("last_save/strain", ""),
            'virus': self.window.settings.value("last_save/virus", ""),
            'location': self.window.settings.value("last_save/location", ""),
            'stim': self.window.settings.value("last_save/stim", ""),
            'power': self.window.settings.value("last_save/power", ""),
            'animal': self.window.settings.value("last_save/animal", ""),
            'sex': self.window.settings.value("last_save/sex", "")
        }

    def _save_last_save_values(self, vals: dict):
        """Save the last used values from the Save Data dialog for auto-population."""
        for key in ['strain', 'virus', 'location', 'stim', 'power', 'animal', 'sex']:
            value = vals.get(key, '').strip()
            self.window.settings.setValue(f"last_save/{key}", value)

    def _sanitize_token(self, s: str) -> str:
        if not s:
            return ""
        s = s.strip()
        s = s.replace(" ", "_")
        # allow alnum, underscore, hyphen, dot
        s = re.sub(r"[^A-Za-z0-9._-]+", "", s)
        # squeeze repeats of underscores/hyphens
        s = re.sub(r"_+", "_", s)
        s = re.sub(r"-+", "-", s)
        return s



    def _suggest_stim_string(self) -> str:
        """
        Build a stim name like '20Hz10s15ms' from detected stim metrics
        or '15msPulse' / '5sPulse' for single pulses.
        Rounding:
        - freq_hz -> nearest Hz
        - duration_s -> nearest second
        - pulse_width_s -> nearest millisecond (or nearest second if >1s)
        """
        st = self.window.state
        if not getattr(st, "stim_chan", None):
            return ""

        # Make sure current sweep has metrics if possible
        try:
            self.window._compute_stim_for_current_sweep()
        except Exception:
            pass

        # Prefer current sweep, else first one that has metrics
        m = st.stim_metrics_by_sweep.get(st.sweep_idx) if hasattr(st, "stim_metrics_by_sweep") else None
        if not m:
            for _s, _m in getattr(st, "stim_metrics_by_sweep", {}).items():
                if _m:
                    m = _m
                    break
        if not m:
            return ""

        freq_hz   = m.get("freq_hz", None)
        dur_s     = m.get("duration_s", None)
        pw_s      = m.get("pulse_width_s", None)

        def _round_int(x):  # safe int rounding
            if x is None:
                return None
            try:
                return int(round(float(x)))
            except Exception:
                return None

        # TRAIN: if freq present and > 0, use Freq + Duration + PW
        if freq_hz is not None and np.isfinite(freq_hz) and freq_hz > 0:
            F = _round_int(freq_hz)              # Hz
            D = _round_int(dur_s)                # s
            MS = _round_int((pw_s or 0) * 1000)  # ms

            parts = []
            if F is not None and F > 0:  parts.append(f"{F}Hz")
            if D is not None and D > 0:  parts.append(f"{D}s")
            if MS is not None and MS > 0: parts.append(f"{MS}ms")
            return "".join(parts) if parts else ""

        # SINGLE PULSE (no usable freq)
        # Prefer pulse width; if missing, fall back to duration
        width_s = pw_s if (pw_s is not None and np.isfinite(pw_s)) else dur_s
        if width_s is None:
            return ""

        if width_s < 1.0:
            ms = _round_int(width_s * 1000)
            if ms is None or ms <= 0:
                ms = 1
            return f"{ms}msPulse"
        else:
            secs = _round_int(width_s)
            if secs is None or secs <= 0:
                secs = 1
            return f"{secs}sPulse"




    def on_save_analyzed_clicked(self):
        """Save analyzed data to disk after prompting for location/name."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        t_start = time.time()
        self.window._log_status_message("Preparing data export...")

        # If eupnea/sniffing detection is out of date, auto-update first
        if getattr(self.window, 'eupnea_sniffing_out_of_date', False):
            print("[Save Data] Eupnea/sniffing detection out of date - auto-updating...")
            self.window._log_status_message("Updating eupnea/sniffing detection...")
            QApplication.processEvents()
            self.window._run_automatic_gmm_clustering()
            self.window.eupnea_sniffing_out_of_date = False
            self.window.redraw_main_plot()
            # Clear the persistent warning
            self.window.statusBar().clearMessage()
            print("[Save Data] Eupnea/sniffing detection updated")
            self.window._log_status_message("Preparing data export...")

        # Create progress dialog
        progress = QProgressDialog("Preparing data export...", None, 0, 100, self.window)
        progress.setWindowTitle("PhysioMetrics")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        QApplication.processEvents()

        try:
            self._export_all_analyzed_data(preview_only=False, progress_dialog=progress)
            # Show completion message with elapsed time
            t_elapsed = time.time() - t_start
            self.window._log_status_message(f"✓ Data export complete ({t_elapsed:.1f}s)", 3000)

            # Count eupnea and sniffing breaths for telemetry
            st = self.window.state
            eupnea_count = 0
            sniff_count = 0
            for s in st.sweeps.keys():
                breath_data = st.breath_by_sweep.get(s, {})
                onsets = breath_data.get('onsets', [])

                for i in range(len(onsets) - 1):
                    if self._is_breath_sniffing(s, i, onsets):
                        sniff_count += 1
                    else:
                        eupnea_count += 1

            # Log telemetry: CSV/NPZ export with per-file edit metrics (for ML evaluation)
            telemetry.log_file_saved(
                save_type='csv_bundle',
                eupnea_count=eupnea_count,
                sniff_count=sniff_count,
                num_sweeps=len(self.window.state.sweeps)
            )

            # Update Project Builder table by re-scanning saved data
            # This is more reliable than direct table manipulation - the scan finds
            # what's actually on disk and updates/creates rows accordingly
            if hasattr(self.window, 'on_project_scan_saved_data') and hasattr(self.window, '_master_file_list'):
                if self.window._master_file_list:  # Only scan if project is open
                    # Get the save directory for a focused scan
                    save_dir = getattr(self.window, '_save_dir', None)
                    if save_dir:
                        from pathlib import Path
                        save_dir = Path(save_dir)
                        print(f"[Export] Triggering focused rescan of: {save_dir}")
                        # Use QTimer to defer the scan slightly so the UI can update first
                        from PyQt6.QtCore import QTimer
                        QTimer.singleShot(100, lambda: self.window.on_project_scan_saved_data(
                            scan_folder=save_dir, silent=True
                        ))
                    else:
                        print("[Export] No save directory found, skipping rescan")
        except Exception as e:
            t_elapsed = time.time() - t_start
            self.window._log_status_message(f"✗ Data export failed ({t_elapsed:.1f}s)", 3000)
            raise
        finally:
            progress.close()


    def on_view_summary_clicked(self):
        """Display interactive preview of the PDF summary without saving."""
        from PyQt6.QtWidgets import QProgressDialog, QApplication
        from PyQt6.QtCore import Qt

        t_start = time.time()
        self._preview_start_time = t_start  # Store for use in preview dialog methods
        self.window._log_status_message("Generating summary...")

        # Auto-detect stims on all sweeps if stim channel exists
        st = self.window.state
        if st.stim_chan:
            print("[View Summary] Detecting stims on all sweeps...")
            self.window._log_status_message("Detecting stimulations...")
            QApplication.processEvents()
            self.window._detect_stims_all_sweeps()
            print("[View Summary] Stim detection complete")
            self.window._log_status_message("Generating summary...")

        # If eupnea/sniffing detection is out of date, auto-update first
        if getattr(self.window, 'eupnea_sniffing_out_of_date', False):
            print("[View Summary] Eupnea/sniffing detection out of date - auto-updating...")
            self.window._log_status_message("Updating eupnea/sniffing detection...")
            QApplication.processEvents()
            self.window._run_automatic_gmm_clustering()
            self.window.eupnea_sniffing_out_of_date = False
            self.window.redraw_main_plot()
            # Clear the persistent warning
            self.window.statusBar().clearMessage()
            print("[View Summary] Eupnea/sniffing detection updated")
            self.window._log_status_message("Generating summary...")

        # Create progress dialog
        progress = QProgressDialog("Generating summary preview...", None, 0, 100, self.window)
        progress.setWindowTitle("PhysioMetrics")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        QApplication.processEvents()

        try:
            self._export_all_analyzed_data(preview_only=True, progress_dialog=progress)
            # Timing message is shown when dialog appears, not when user closes it
        except Exception as e:
            t_elapsed = time.time() - t_start
            self.window._log_status_message(f"✗ Summary failed ({t_elapsed:.1f}s)", 3000)
            raise
        finally:
            progress.close()


    # metrics we won't include in CSV exports and PDFs
    # Note: These metrics are still computed and available for ML export,
    # but hidden from user-facing CSV/PDF outputs for clarity and performance
    _EXCLUDE_FOR_CSV = {
        "d1", "d2", "eupnic", "apnea",
        # Phase 2.2: Sigh detection features (for future ML use)
        "n_inflections", "rise_variability", "n_shoulder_peaks",
        "shoulder_prominence", "rise_autocorr", "peak_sharpness",
        "trough_sharpness", "skewness", "kurtosis",
        # Phase 2.3 Group A: Shape & ratio metrics (for future ML use)
        "peak_to_trough", "amp_ratio", "ti_te_ratio",
        "area_ratio", "total_area", "ibi",
        # Phase 2.3 Group B: Normalized metrics (for future ML use)
        "amp_insp_norm", "amp_exp_norm", "peak_to_trough_norm",
        "prominence_norm", "ibi_norm", "ti_norm", "te_norm",
    }

    # Core metrics for View Summary preview (fast, focused display)
    # Only these metrics are computed and displayed in preview mode
    _PREVIEW_CORE_METRICS = [
        "if",          # Instantaneous frequency
        "amp_insp",    # Inspiratory amplitude
        "amp_exp",     # Expiratory amplitude
        "area_insp",   # Inspiratory area
        "area_exp",    # Expiratory area
        "sniff_conf",  # Sniffing confidence (GMM) - interesting for analysis
    ]


    def _metric_keys_in_order(self):
        """Return metric keys in the UI order (from metrics.METRIC_SPECS)."""
        return [key for (_, key) in metrics.METRIC_SPECS]


    def _compute_metric_trace(self, key, t, y, sr_hz, peaks, breaths, sweep=None):
        """
        Call the metric function, passing expoffs if it exists.
        Falls back to legacy signature when needed.

        Args:
            sweep: Optional sweep index for setting GMM probabilities (needed for sniff_conf/eupnea_conf)
        """
        st = self.window.state
        fn = metrics.METRICS[key]
        on  = breaths.get("onsets")   if breaths else None
        off = breaths.get("offsets")  if breaths else None
        exm = breaths.get("expmins")  if breaths else None
        exo = breaths.get("expoffs")  if breaths else None

        # Set GMM probabilities if computing sniff_conf or eupnea_conf
        gmm_probs = None
        if sweep is not None and hasattr(st, 'gmm_sniff_probabilities') and sweep in st.gmm_sniff_probabilities:
            gmm_probs = st.gmm_sniff_probabilities[sweep]
            metrics.set_gmm_probabilities(gmm_probs)

        try:
            result = fn(t, y, sr_hz, peaks, on, off, exm, exo)  # new signature
        except TypeError:
            result = fn(t, y, sr_hz, peaks, on, off, exm)       # legacy signature
        finally:
            # Clear GMM probabilities after computation
            if gmm_probs is not None:
                metrics.set_gmm_probabilities(None)

        return result

    def _compute_all_metrics_for_sweep(self, sweep_idx: int, keys_to_compute: list,
                                        ds_idx: np.ndarray, N: int) -> dict:
        """
        Compute all metrics for a single sweep (parallelizable helper).

        Args:
            sweep_idx: Sweep index
            keys_to_compute: List of metric keys to compute
            ds_idx: Downsampling indices
            N: Total trace length

        Returns:
            Dict with:
                - 'sweep_idx': The sweep index
                - 'traces': Dict mapping metric key to downsampled trace
                - 'y_proc_ds': Downsampled processed signal
                - 'peaks', 'onsets', 'offsets', 'expmins', 'expoffs': Breath events
                - 'sighs': Sigh indices
                - 'omit_mask': Omitted region mask (or None)
        """
        st = self.window.state

        # Get processed signal
        y_proc = self.window._get_processed_for(st.analyze_chan, sweep_idx)

        # Get peaks and breath events
        pks = np.asarray(st.peaks_by_sweep.get(sweep_idx, np.array([], dtype=int)), dtype=int)
        br = st.breath_by_sweep.get(sweep_idx, None)
        if br is None and pks.size:
            import core.peaks as peakdet
            br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
        if br is None:
            br = {
                "onsets": np.array([], dtype=int),
                "offsets": np.array([], dtype=int),
                "expmins": np.array([], dtype=int),
                "expoffs": np.array([], dtype=int),
            }

        # Get sighs
        sighs = np.asarray(st.sigh_by_sweep.get(sweep_idx, np.array([], dtype=int)), dtype=int)
        sighs = sighs[(sighs >= 0) & (sighs < len(y_proc))]

        # Build omitted mask
        has_omitted = sweep_idx in st.omitted_sweeps or sweep_idx in st.omitted_ranges
        if has_omitted:
            omit_mask = create_omitted_mask(sweep_idx, N, st.omitted_ranges, st.omitted_sweeps)
        else:
            omit_mask = None

        # Compute all metrics
        traces = {}
        for k in keys_to_compute:
            # Special handling for probability metrics - use ALL peaks (including noise)
            if k in ('p_noise', 'p_breath'):
                all_pks_data = st.all_peaks_by_sweep.get(sweep_idx, None)
                all_br = st.all_breaths_by_sweep.get(sweep_idx, None)
                if all_pks_data is not None and all_br is not None:
                    all_pks = all_pks_data['indices']
                    y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, all_pks, all_br, sweep=sweep_idx)
                else:
                    y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=sweep_idx)
            else:
                y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=sweep_idx)

            if y2 is not None and len(y2) == N:
                # Apply omitted mask
                if omit_mask is not None:
                    y2 = y2.copy()  # Don't modify original
                    y2[~omit_mask] = np.nan
                traces[k] = y2[ds_idx]
            else:
                traces[k] = None

        return {
            'sweep_idx': sweep_idx,
            'traces': traces,
            'y_proc_ds': y_proc[ds_idx],
            'peaks': pks,
            'onsets': np.asarray(br.get("onsets", []), dtype=int),
            'offsets': np.asarray(br.get("offsets", []), dtype=int),
            'expmins': np.asarray(br.get("expmins", []), dtype=int),
            'expoffs': np.asarray(br.get("expoffs", []), dtype=int),
            'sighs': sighs,
            'omit_mask': omit_mask
        }


    def _get_stim_masks(self, s: int):
        """
        Build (baseline_mask, stim_mask, post_mask) boolean arrays over st.t for sweep s.
        Uses union of all stim spans for 'stim'.
        """
        st = self.window.state
        t = st.t
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if not spans:
            # no stim: whole trace is baseline; stim/post empty
            B = np.ones_like(t, dtype=bool)
            Z = np.zeros_like(t, dtype=bool)
            return B, Z, Z

        starts = np.array([a for (a, _) in spans], dtype=float)
        ends   = np.array([b for (_, b) in spans], dtype=float)
        t0 = np.min(starts)
        t1 = np.max(ends)

        stim_mask = np.zeros_like(t, dtype=bool)
        for (a, b) in spans:
            stim_mask |= (t >= a) & (t <= b)

        baseline_mask = t < t0
        post_mask     = t > t1
        return baseline_mask, stim_mask, post_mask

    #     """Return (nanmean, nansem) along axis; SEM uses ddof=1 where n>=2 else NaN."""
    #     with np.errstate(invalid="ignore"):
    #         # std with ddof=1; guard n<2


    def _nanmean_sem(self, X, axis=0):
        """
        Robust mean/SEM that avoids NumPy RuntimeWarnings when there are
        0 or 1 finite values along the chosen axis.
        """
        import numpy as np

        A = np.asarray(X, dtype=float)
        if A.size == 0:
            return np.nan, np.nan

        # Move the target axis to 0 so we can index rows easily
        A0 = np.moveaxis(A, axis, 0)      # shape: (N, ...rest...)
        # Collapse the rest to a single trailing dimension for simplicity
        tail = int(np.prod(A0.shape[1:])) or 1
        A2 = A0.reshape(A0.shape[0], tail)  # (N, T)

        finite = np.isfinite(A2)
        n = finite.sum(axis=0)              # (T,)

        mean = np.full((tail,), np.nan, dtype=float)
        sem  = np.full((tail,), np.nan, dtype=float)

        # rows/columns (along axis) with at least one finite value
        msk_mean = n > 0
        if np.any(msk_mean):
            mean[msk_mean] = np.nanmean(A2[:, msk_mean], axis=0)

        # rows/columns with at least two finite values (only here compute SEM)
        msk_sem = n >= 2
        if np.any(msk_sem):
            std = np.nanstd(A2[:, msk_sem], axis=0, ddof=1)
            sem[msk_sem] = std / np.sqrt(n[msk_sem])

        # reshape back to the original tail and move axis back
        mean = mean.reshape(A0.shape[1:])
        sem  = sem.reshape(A0.shape[1:])
        # move axis back to original position (only if result has enough dimensions)
        # After reduction along one axis, result has (A.ndim - 1) dimensions
        # We can only moveaxis if result is multi-dimensional
        if mean.ndim > 1:
            mean = np.moveaxis(mean, 0, axis)
            sem  = np.moveaxis(sem, 0, axis)

        return mean, sem








    def _get_ml_training_folder(self):
        """
        Get or create centralized ML training data folder.

        Priority order:
        1. Check for user-configured custom path in QSettings
        2. Auto-detect lab default path (Z:/Drivers and Software/PhysioMetrics/ML Training Data)
        3. Prompt user to select folder on first use

        Returns:
            Path to ML training folder, or None if user cancels
        """
        from pathlib import Path

        # 1. Check for saved ML training folder location
        saved_ml_folder = self.window.settings.value("ml_training_folder", None)
        if saved_ml_folder:
            path = Path(saved_ml_folder)
            if path.exists():
                return path
            # Path no longer exists - clear it and fall through to detection
            self.window.settings.remove("ml_training_folder")
            print(f"[ML training] Saved folder no longer exists, re-detecting: {saved_ml_folder}")

        # 2. Try lab default path (check multiple potential drive letters)
        lab_subpath = "Drivers and Software/PhysioMetrics/ML Training Data"
        for drive in ['Z:', 'Y:', 'X:', 'W:', 'V:', 'U:']:
            candidate = Path(f"{drive}/{lab_subpath}")
            parent_path = candidate.parent  # PhysioMetrics folder
            if parent_path.exists():
                try:
                    candidate.mkdir(exist_ok=True)
                    self.window.settings.setValue("ml_training_folder", str(candidate))
                    print(f"[ML training] Auto-detected lab drive: {candidate}")
                    return candidate
                except Exception as e:
                    print(f"[ML training] Could not create folder on {drive}: {e}")
                    continue

        # 3. Fallback: Prompt user to select folder
        return self._prompt_for_ml_folder()

    def _prompt_for_ml_folder(self):
        """Prompt user to select ML training folder."""
        from pathlib import Path
        from PyQt6.QtWidgets import QMessageBox

        # Show info message first
        QMessageBox.information(
            self.window,
            "ML Training Folder",
            "Please select a folder to store ML training data.\n\n"
            "This folder will be used for all future exports.\n"
            "You can change it later using the gear icon in the save dialog\n"
            "or in the ML Settings tab of the Analysis Options dialog."
        )

        # Let user choose directory
        default_root = self.window.settings.value("save_root", str(self.window.state.in_path.parent))
        folder = QFileDialog.getExistingDirectory(
            self.window,
            "Select ML Training Data Folder",
            str(default_root)
        )

        if folder:
            path = Path(folder)
            self.window.settings.setValue("ml_training_folder", str(path))
            print(f"[ML training] User selected folder: {path}")
            return path

        # User cancelled - use temp fallback in home directory
        fallback = Path.home() / "PhysioMetrics_ML_Training"
        try:
            fallback.mkdir(exist_ok=True)
            print(f"[ML training] Using fallback folder: {fallback}")
            return fallback
        except Exception as e:
            print(f"[ML training] Could not create fallback folder: {e}")
            return None

    @profile
    def _export_ml_training_data(self, npz_path, source_name, include_waveforms=False, metadata=None):
        """
        Export ML training data as compact .npz file.

        Saves a single .npz file containing:
        - Peak metrics arrays (original detection)
        - Labels (still_exists, was_modified)
        - Merge decisions
        - Source file metadata
        - User metadata (quality score, labeler info)
        - [Optional] Waveform cutouts around each peak

        Args:
            npz_path: Path for output .npz file
            source_name: Source filename (for metadata)
            include_waveforms: If True, include raw waveform segments for neural net training
            metadata: Dict with 'system_username', 'computer_name', 'user_name', 'quality_score'
        """
        st = self.window.state

        print(f"[ML training] Exporting training data to {npz_path.name}...")
        if include_waveforms:
            print(f"[ML training] Including waveform cutouts (this will increase file size)")

        # ========================================================================
        # SIMPLIFIED: Just read labels directly from all_peaks_by_sweep!
        # ========================================================================
        peak_data = []

        # Check if all_peaks_by_sweep exists (needed for label-based export)
        if not hasattr(st, 'all_peaks_by_sweep') or not st.all_peaks_by_sweep:
            error_msg = (
                "ML export requires peak detection data.\n\n"
                "Please run peak detection first:\n"
                "1. Click 'Detect Peaks' button\n"
                "2. Then try exporting ML training data again\n\n"
                "Note: If you loaded an old session file, you'll need to re-run peak detection."
            )
            raise ValueError(error_msg)

        # Check if data is in new format (dict with labels) vs old format (just arrays)
        first_sweep = next(iter(st.all_peaks_by_sweep.keys()))
        sample_data = st.all_peaks_by_sweep[first_sweep]

        if isinstance(sample_data, np.ndarray):
            # Old format detected - data is just arrays, not dicts with labels
            error_msg = (
                "ML export requires updated peak detection data.\n\n"
                "This file was processed with an older version.\n"
                "Please re-run peak detection:\n"
                "1. Click 'Detect Peaks' button\n"
                "2. Then try exporting ML training data again\n\n"
                "Your manual edits will be preserved if you saved them in a session file."
            )
            raise ValueError(error_msg)

        # Build merge lookup for labeling peaks involved in merges
        # This creates labels for ML merge detection training
        merge_lookup = {}  # {(sweep_idx, peak_idx): {'was_merged_away': bool, 'merged_with_next_peak': peak_idx}}

        if hasattr(st, 'user_merge_decisions'):
            for sweep_idx, decisions in st.user_merge_decisions.items():
                for decision in decisions:
                    peak1_idx = decision.get('peak1_idx')
                    peak2_idx = decision.get('peak2_idx')
                    removed_idx = decision.get('removed_idx')
                    kept_idx = decision.get('kept_idx')

                    # Mark the removed peak as merged away
                    if removed_idx is not None:
                        merge_lookup[(sweep_idx, removed_idx)] = {
                            'was_merged_away': True,
                            'merged_with_next_peak': None
                        }

                    # Mark the first peak as having been merged with next
                    # (The one that initiated the merge, even if it was removed)
                    if peak1_idx is not None and peak2_idx is not None:
                        if (sweep_idx, peak1_idx) not in merge_lookup:
                            merge_lookup[(sweep_idx, peak1_idx)] = {
                                'was_merged_away': False,
                                'merged_with_next_peak': peak2_idx
                            }
                        else:
                            # Already marked as merged away, just add the next peak info
                            merge_lookup[(sweep_idx, peak1_idx)]['merged_with_next_peak'] = peak2_idx

        for sweep_idx in sorted(st.all_peaks_by_sweep.keys()):
            all_peaks = st.all_peaks_by_sweep.get(sweep_idx)
            if all_peaks is None or 'indices' not in all_peaks:
                continue

            metrics = st.peak_metrics_by_sweep.get(sweep_idx, [])
            sigh_indices = set(st.sigh_by_sweep.get(sweep_idx, []))

            # Get breath type classification if available (stored in all_peaks_by_sweep)
            # Note: breath_type_class contains the active classifier's predictions (GMM, XGBoost, RF, or MLP)
            breath_type_class_array = all_peaks.get('breath_type_class', None)

            # Get sigh classification if available
            sigh_class_array = all_peaks.get('sigh_class', None)

            # Iterate through ALL peaks (label=0 and label=1)
            for i, (peak_idx, label, label_source) in enumerate(zip(
                all_peaks['indices'],
                all_peaks['labels'],
                all_peaks['label_source']
            )):
                # Get metrics for this peak (aligned by index)
                metric_dict = metrics[i] if i < len(metrics) else {}

                # Check if this peak is labeled as a sigh
                # Use sigh_class_array if available, otherwise fall back to sigh_by_sweep
                # NaN = no label (for ML training to ignore)
                if sigh_class_array is not None and i < len(sigh_class_array):
                    sigh_val = sigh_class_array[i]
                    is_sigh = 1 if sigh_val == 1 else (0 if sigh_val == 0 else float('nan'))
                elif peak_idx in sigh_indices:
                    is_sigh = 1
                else:
                    # No sigh classification available - use NaN to indicate "no label"
                    is_sigh = float('nan') if sigh_class_array is None else 0

                # Determine eupnea/sniffing classification from breath_type_class
                # NaN = no label (for ML training to ignore)
                is_eupnea = float('nan')
                is_sniffing = float('nan')

                # Only classify if this is a labeled breath (label=1) and breath_type_class exists
                if label == 1 and breath_type_class_array is not None and i < len(breath_type_class_array):
                    breath_type_class = breath_type_class_array[i]
                    if breath_type_class == 0:
                        is_eupnea = 1
                        is_sniffing = 0
                    elif breath_type_class == 1:
                        is_eupnea = 0
                        is_sniffing = 1
                    elif breath_type_class == -1:
                        # Unclassified - use NaN
                        is_eupnea = float('nan')
                        is_sniffing = float('nan')

                # Get export options (which metrics to include)
                export_options = getattr(st, 'export_metric_options', None)

                # Build record with core fields (always exported)
                record = {
                    'sweep_idx': sweep_idx,
                    'peak_idx': peak_idx,
                    'is_breath': label,  # 0 or 1 (THE KEY LABEL!)
                    'label_source': label_source,  # 'auto' or 'user'
                }

                # Add classification labels (respecting export options)
                if export_options is None or export_options.get('is_sigh', True):
                    record['is_sigh'] = is_sigh
                if export_options is None or export_options.get('is_eupnea', True):
                    record['is_eupnea'] = is_eupnea
                if export_options is None or export_options.get('is_sniffing', True):
                    record['is_sniffing'] = is_sniffing

                # Add merge labels (for ML merge detection training)
                merge_info = merge_lookup.get((sweep_idx, peak_idx), None)
                if merge_info:
                    record['was_merged_away'] = 1 if merge_info['was_merged_away'] else 0
                    record['merge_with_next'] = 1 if merge_info['merged_with_next_peak'] is not None else 0
                else:
                    record['was_merged_away'] = 0
                    record['merge_with_next'] = 0

                # Add all metrics (filtered by export options if available)
                for key, value in metric_dict.items():
                    if key not in ['sweep_idx', 'peak_idx'] and not isinstance(value, dict):
                        # Check if this metric should be exported
                        if export_options is None or export_options.get(key, True):
                            record[key] = value

                peak_data.append(record)

        # ========================================================================
        # DATASET 2: Export BREATHS ONLY with recalculated features
        # (for Model 2 - breath type classification)
        # ========================================================================
        breaths_only_data = []

        # Check if we have recalculated metrics available
        has_recalc_metrics = hasattr(st, 'current_peak_metrics_by_sweep') and st.current_peak_metrics_by_sweep

        for sweep_idx in sorted(st.peaks_by_sweep.keys()):
            # Get only the "real" breath peaks (already filtered)
            breath_peaks = st.peaks_by_sweep.get(sweep_idx, [])
            if len(breath_peaks) == 0:
                continue

            # Try to get recalculated metrics (computed after filtering)
            if has_recalc_metrics:
                recalc_metrics = st.current_peak_metrics_by_sweep.get(sweep_idx, [])
            else:
                # Fallback: use original metrics (same as all_peaks)
                recalc_metrics = st.peak_metrics_by_sweep.get(sweep_idx, [])

            # Explicitly recompute p_noise and p_breath for filtered breaths
            # (these may be missing from recalc_metrics if session was saved before recalc feature)
            p_noise_array = None
            p_breath_array = None
            try:
                import core.metrics as metrics_mod

                # Get processed signal for this sweep
                y_proc = self.window._get_processed_for(st.analyze_chan, sweep_idx)

                # Get breath events for filtered peaks
                breath_events = st.breath_by_sweep.get(sweep_idx, {})

                # Compute p_noise for filtered breaths
                p_noise_array = metrics_mod.compute_p_noise(
                    st.t, y_proc, st.sr_hz, breath_peaks,
                    breath_events.get('onsets', np.array([])),
                    breath_events.get('offsets', np.array([])),
                    breath_events.get('expmins', np.array([])),
                    breath_events.get('expoffs', np.array([]))
                )

                # Compute p_breath (complement of p_noise)
                if p_noise_array is not None:
                    p_breath_array = 1.0 - p_noise_array

            except Exception as e:
                print(f"[ML export] Warning: Could not compute p_noise for sweep {sweep_idx}: {e}")
                # Will use None values (will be NaN in export)

            sigh_indices = set(st.sigh_by_sweep.get(sweep_idx, []))

            # Get breath type and sigh classifications
            all_peaks = st.all_peaks_by_sweep.get(sweep_idx)
            breath_type_class_array = all_peaks.get('breath_type_class', None) if all_peaks else None
            sigh_class_array = all_peaks.get('sigh_class', None) if all_peaks else None

            # Iterate through ONLY the breath peaks
            for breath_idx, peak_idx in enumerate(breath_peaks):
                # Find this peak in all_peaks to get breath type class
                if all_peaks is not None and 'indices' in all_peaks:
                    all_peak_indices = all_peaks['indices']
                    pos_in_all = np.where(all_peak_indices == peak_idx)[0]
                    if len(pos_in_all) > 0:
                        i_in_all = pos_in_all[0]
                        label = all_peaks['labels'][i_in_all]
                        label_source = all_peaks['label_source'][i_in_all]

                        # Get breath type class - NaN if not available
                        is_eupnea = float('nan')
                        is_sniffing = float('nan')
                        if breath_type_class_array is not None and i_in_all < len(breath_type_class_array):
                            breath_type_class = breath_type_class_array[i_in_all]
                            if breath_type_class == 0:
                                is_eupnea = 1
                                is_sniffing = 0
                            elif breath_type_class == 1:
                                is_eupnea = 0
                                is_sniffing = 1
                            # breath_type_class == -1 means unclassified, leave as NaN

                        # Get sigh class - NaN if not available
                        if sigh_class_array is not None and i_in_all < len(sigh_class_array):
                            sigh_val = sigh_class_array[i_in_all]
                            is_sigh = 1 if sigh_val == 1 else (0 if sigh_val == 0 else float('nan'))
                        elif peak_idx in sigh_indices:
                            is_sigh = 1
                        else:
                            is_sigh = float('nan') if sigh_class_array is None else 0
                    else:
                        # Shouldn't happen, but handle gracefully
                        label = 1
                        label_source = 'unknown'
                        is_eupnea = float('nan')
                        is_sniffing = float('nan')
                        is_sigh = float('nan')
                else:
                    label = 1
                    label_source = 'unknown'
                    is_eupnea = float('nan')
                    is_sniffing = float('nan')
                    is_sigh = float('nan')

                # Get recalculated metrics for this breath
                # NOTE: recalc_metrics is indexed by breath_idx (position in filtered list)
                # NOT by i_in_all (position in all_peaks)
                metric_dict = recalc_metrics[breath_idx] if breath_idx < len(recalc_metrics) else {}

                # Build record
                record = {
                    'sweep_idx': sweep_idx,
                    'peak_idx': peak_idx,
                    'is_breath': label,  # Should always be 1 for this dataset
                    'label_source': label_source,
                }

                # Add classification labels
                if export_options is None or export_options.get('is_sigh', True):
                    record['is_sigh'] = is_sigh
                if export_options is None or export_options.get('is_eupnea', True):
                    record['is_eupnea'] = is_eupnea
                if export_options is None or export_options.get('is_sniffing', True):
                    record['is_sniffing'] = is_sniffing

                # Merge labels (always 0 for filtered breaths - they survived filtering)
                record['was_merged_away'] = 0  # If it's here, it wasn't merged away
                record['merge_with_next'] = 0  # Merges already resolved

                # Add p_noise and p_breath (explicitly computed for filtered breaths)
                # These MUST be included for ML training - sample from arrays at peak position
                if p_noise_array is not None and peak_idx < len(p_noise_array):
                    record['p_noise'] = float(p_noise_array[peak_idx])
                else:
                    record['p_noise'] = None

                if p_breath_array is not None and peak_idx < len(p_breath_array):
                    record['p_breath'] = float(p_breath_array[peak_idx])
                else:
                    record['p_breath'] = None

                # Add recalculated metrics
                for key, value in metric_dict.items():
                    if key not in ['sweep_idx', 'peak_idx'] and not isinstance(value, dict):
                        if export_options is None or export_options.get(key, True):
                            record[key] = value

                breaths_only_data.append(record)

        print(f"[ML training] Dataset 1 (ALL_PEAKS): {len(peak_data)} peaks")
        print(f"[ML training] Dataset 2 (BREATHS_ONLY): {len(breaths_only_data)} breaths (recalculated features)")

        # Optionally extract waveform cutouts for neural net training
        waveform_cutouts = None
        waveform_window_samples = None
        if include_waveforms and peak_data:
            print(f"[ML training] Extracting waveform cutouts...")

            # Define standard window: ±1 second around peak
            window_seconds = 1.0
            window_samples = int(window_seconds * st.sr_hz)
            waveform_window_samples = 2 * window_samples  # Total length

            # OPTIMIZATION: Cache processed traces per sweep (avoid reprocessing for every peak)
            # This reduces calls to _get_processed_for from N_peaks to N_sweeps
            trace_cache = {}

            cutouts = []
            for record in peak_data:
                sweep_idx = record['sweep_idx']
                peak_idx = record['peak_idx']

                # Get processed trace from cache or compute once per sweep
                if sweep_idx not in trace_cache:
                    trace_cache[sweep_idx] = self.window._get_processed_for(st.analyze_chan, sweep_idx)
                y_proc = trace_cache[sweep_idx]

                # Extract window around peak
                start_idx = max(0, peak_idx - window_samples)
                end_idx = min(len(y_proc), peak_idx + window_samples)

                segment = y_proc[start_idx:end_idx]

                # Pad to standard length if needed
                if len(segment) < waveform_window_samples:
                    # Calculate how much we actually extracted vs how much we wanted
                    actual_before = peak_idx - start_idx  # samples before peak
                    actual_after = end_idx - peak_idx      # samples after peak

                    # Calculate padding needed to reach desired window size
                    pad_before = max(0, window_samples - actual_before)  # pad at start if peak near beginning
                    pad_after = max(0, window_samples - actual_after)    # pad at end if peak near end

                    segment = np.pad(segment, (pad_before, pad_after), mode='edge')

                # Truncate if too long (shouldn't happen, but safety check)
                if len(segment) > waveform_window_samples:
                    segment = segment[:waveform_window_samples]

                cutouts.append(segment)

            # Clear cache to free memory
            del trace_cache

            waveform_cutouts = np.array(cutouts, dtype=np.float32)  # [n_peaks, window_samples]
            print(f"[ML training] ✓ Extracted {len(cutouts)} waveform cutouts ({waveform_cutouts.shape})")

        # Collect merge decisions
        merge_data = []
        if hasattr(st, 'user_merge_decisions'):
            for sweep_idx, decisions in st.user_merge_decisions.items():
                for decision in decisions:
                    merge_data.append({
                        'sweep_idx': sweep_idx,
                        'peak1_idx': decision.get('peak1_idx'),
                        'peak2_idx': decision.get('peak2_idx'),
                        'kept_idx': decision.get('kept_idx'),
                        'removed_idx': decision.get('removed_idx'),
                        'timestamp': decision.get('timestamp'),
                    })

        # Convert to numpy arrays for compact storage
        import pandas as pd

        # Dataset 1: ALL_PEAKS (for Model 1 - breath vs noise classification)
        if peak_data:
            df_peaks = pd.DataFrame(peak_data)
            # Get column names and data
            peak_columns = df_peaks.columns.tolist()
            peak_arrays = {f"all_peaks_{col}": df_peaks[col].values for col in peak_columns}
        else:
            peak_columns = []
            peak_arrays = {}

        # Dataset 2: BREATHS_ONLY (for Model 2 - breath type classification with recalculated features)
        if breaths_only_data:
            df_breaths = pd.DataFrame(breaths_only_data)
            breaths_columns = df_breaths.columns.tolist()
            breaths_arrays = {f"breaths_{col}": df_breaths[col].values for col in breaths_columns}
        else:
            breaths_columns = []
            breaths_arrays = {}

        # Merge decisions
        if merge_data:
            df_merges = pd.DataFrame(merge_data)
            merge_columns = df_merges.columns.tolist()
            merge_arrays = {f"merge_{col}": df_merges[col].values for col in merge_columns}
        else:
            merge_columns = []
            merge_arrays = {}

        # Get app version
        try:
            import version_info
            app_version = version_info.VERSION_STRING
        except:
            app_version = 'unknown'

        # Save to .npz with metadata
        save_dict = {
            # File metadata
            'source_file': source_name,
            'app_version': app_version,
            'timestamp': metadata.get('timestamp', '') if metadata else '',
            'n_all_peaks': len(peak_data),
            'n_breaths_only': len(breaths_only_data),
            'n_merges': len(merge_data),
            'all_peaks_columns': np.array(peak_columns, dtype=object),
            'breaths_columns': np.array(breaths_columns, dtype=object),
            'merge_columns': np.array(merge_columns, dtype=object),
            'has_waveforms': include_waveforms,
            'has_dual_datasets': True,  # Flag indicating dual-dataset structure
            'sampling_rate_hz': float(st.sr_hz) if st.sr_hz else 0.0,
            # User metadata (for ML dataset filtering and tracking)
            'system_username': metadata.get('system_username', 'unknown') if metadata else 'unknown',
            'computer_name': metadata.get('computer_name', 'unknown') if metadata else 'unknown',
            'user_name': metadata.get('user_name', '') if metadata else '',
            'animal_state': metadata.get('animal_state', '') if metadata else '',
            'anesthetic_type': metadata.get('anesthetic_type', '') if metadata else '',
            'drug': metadata.get('drug', '') if metadata else '',
            'drug_concentration': metadata.get('drug_concentration', '') if metadata else '',
            'gas_condition': metadata.get('gas_condition', '') if metadata else '',
            'notes': metadata.get('notes', '') if metadata else '',
            'quality_score': int(metadata.get('quality_score', 5)) if metadata else 5,
            # Dataset 1: ALL_PEAKS (for Model 1 - breath vs noise classification)
            **peak_arrays,
            # Dataset 2: BREATHS_ONLY (for Model 2 - breath type with recalculated features)
            **breaths_arrays,
            # Merge decisions
            **merge_arrays,
        }

        # Add waveform data if included
        if waveform_cutouts is not None:
            save_dict['waveform_cutouts'] = waveform_cutouts
            save_dict['waveform_window_samples'] = waveform_window_samples
            save_dict['waveform_window_seconds'] = 2.0  # ±1 second

        np.savez_compressed(npz_path, **save_dict)

        # Count different breath types for summary (from ALL_PEAKS dataset)
        n_total = len(peak_data)
        n_breaths = sum(1 for r in peak_data if r['is_breath'] == 1)
        n_rejected = n_total - n_breaths
        n_sighs = sum(1 for r in peak_data if r['is_sigh'] == 1)
        n_eupnea = sum(1 for r in peak_data if r['is_eupnea'] == 1)
        n_sniffing = sum(1 for r in peak_data if r['is_sniffing'] == 1)
        n_merged_away = sum(1 for r in peak_data if r.get('was_merged_away', 0) == 1)
        n_merge_with_next = sum(1 for r in peak_data if r.get('merge_with_next', 0) == 1)

        # Count from BREATHS_ONLY dataset (for validation)
        n_breaths_only = len(breaths_only_data)
        n_breaths_sighs = sum(1 for r in breaths_only_data if r['is_sigh'] == 1)
        n_breaths_eupnea = sum(1 for r in breaths_only_data if r['is_eupnea'] == 1)
        n_breaths_sniffing = sum(1 for r in breaths_only_data if r['is_sniffing'] == 1)

        print(f"[ML training] ✓ Dataset 1 (ALL_PEAKS): {n_total} peaks ({n_breaths} breaths, {n_rejected} rejected)")
        print(f"[ML training]   → {n_sighs} sighs, {n_eupnea} eupnea, {n_sniffing} sniffing")
        print(f"[ML training]   → Merge labels: {n_merged_away} peaks merged away, {n_merge_with_next} peaks merged with next")
        print(f"[ML training] ✓ Dataset 2 (BREATHS_ONLY): {n_breaths_only} breaths (recalculated features)")
        print(f"[ML training]   → {n_breaths_sighs} sighs, {n_breaths_eupnea} eupnea, {n_breaths_sniffing} sniffing")
        if metadata:
            print(f"[ML training]   → Timestamp: {metadata.get('timestamp', 'unknown')}")
            print(f"[ML training]   → System: {metadata.get('system_username', 'unknown')}@{metadata.get('computer_name', 'unknown')}")
            user_label = metadata.get('user_name', '')
            if user_label:
                print(f"[ML training]   → User label: {user_label}")
            animal_state = metadata.get('animal_state', '')
            if animal_state:
                print(f"[ML training]   → Animal state: {animal_state}")
            anesthetic = metadata.get('anesthetic_type', '')
            if anesthetic:
                print(f"[ML training]   → Anesthetic: {anesthetic}")
            drug = metadata.get('drug', '')
            if drug:
                dose = metadata.get('drug_concentration', '')
                dose_str = f" ({dose})" if dose else ""
                print(f"[ML training]   → Drug: {drug}{dose_str}")
            gas = metadata.get('gas_condition', '')
            if gas:
                print(f"[ML training]   → Gas: {gas}")
            notes = metadata.get('notes', '')
            if notes:
                print(f"[ML training]   → Notes: {notes[:50]}{'...' if len(notes) > 50 else ''}")
            print(f"[ML training]   → Quality score: {metadata.get('quality_score', 5)}/10")
        print(f"[ML training]   → App version: {app_version}")
        if include_waveforms:
            print(f"[ML training] ✓ Waveforms: {waveform_cutouts.shape} @ {st.sr_hz:.1f} Hz")
        print(f"[ML training] ✓ File size: {npz_path.stat().st_size / 1024:.1f} KB")

        # Return counts for success message
        return {
            'n_total': n_total,
            'n_breaths': n_breaths,
            'n_rejected': n_rejected,
            'n_sighs': n_sighs,
            'n_eupnea': n_eupnea,
            'n_sniffing': n_sniffing,
            'n_breaths_only': n_breaths_only,
            'n_merges': len(merge_data),
            'n_merged_away': n_merged_away,
            'n_merge_with_next': n_merge_with_next
        }

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
        import numpy as np, csv, json, time
        from PyQt6.QtWidgets import QApplication

        st = self.window.state
        if not getattr(st, "in_path", None):
            self._show_message_box(QMessageBox.Icon.Information, "View Summary" if preview_only else "Save analyzed data", "Open an ABF first.")
            return
        if st.t is None or not st.analyze_chan or st.analyze_chan not in st.sweeps:
            self._show_message_box(QMessageBox.Icon.Warning, "View Summary" if preview_only else "Save analyzed data", "No analyzed data available.")
            return

        # -------------------- Preview caching (skip computation if unchanged) --------------------
        if preview_only:
            # Generate fingerprint of current state
            def _compute_preview_fingerprint():
                """Compute a fingerprint of data that affects preview output."""
                import hashlib
                parts = []
                # File identity
                parts.append(str(st.in_path))
                parts.append(st.analyze_chan or "")
                # Sweep/omission state
                parts.append(str(sorted(getattr(st, "omitted_sweeps", set()))))
                # Peak count per sweep (changes indicate edits)
                for s in range(st.sweeps[st.analyze_chan].shape[1] if st.analyze_chan in st.sweeps else 0):
                    pks = st.peaks_by_sweep.get(s, [])
                    parts.append(f"{s}:{len(pks)}")
                # GMM state (sniff regions affect sniff_conf metric)
                sniff_count = sum(len(st.sniff_regions_by_sweep.get(s, [])) for s in st.sniff_regions_by_sweep)
                parts.append(f"sniff:{sniff_count}")
                # Stim spans (affect time alignment)
                if st.stim_chan:
                    stim_count = sum(len(st.stim_spans_by_sweep.get(s, [])) for s in st.stim_spans_by_sweep)
                    parts.append(f"stim:{stim_count}")
                return hashlib.md5("|".join(parts).encode()).hexdigest()

            current_fingerprint = _compute_preview_fingerprint()

            # Check if we have valid cached preview data
            if (hasattr(self, '_preview_cache') and
                self._preview_cache.get('fingerprint') == current_fingerprint):
                print("[Preview] Using cached preview data (no changes detected)")
                cached = self._preview_cache

                # For pulse experiments, show pulse preview first
                if cached.get('is_pulse_exp') and cached.get('pulse_figures'):
                    print("[Preview] Showing cached pulse preview...")
                    fig_cta, fig_3d, fig_3d_stim, fig_prob = cached['pulse_figures']
                    self._show_pulse_4page_preview(fig_cta, fig_3d, fig_3d_stim, fig_prob)

                # Show standard summary preview
                self._show_summary_preview_dialog(
                    t_ds_csv=cached['t_ds_csv'],
                    y2_ds_by_key=cached['y2_ds_by_key'],
                    keys_for_csv=cached['keys_for_csv'],
                    label_by_key=cached['label_by_key'],
                    stim_zero=cached['stim_zero'],
                    stim_dur=cached['stim_dur'],
                )
                return

        # -------------------- Prompt for save location (skip if preview_only) --------------------
        if not preview_only:
            # --- Build an auto stim string from current sweep metrics, if available ---
            def _auto_stim_from_metrics() -> str:
                s = max(0, min(getattr(st, "sweep_idx", 0), self.window.navigation_manager._sweep_count()-1))
                m = st.stim_metrics_by_sweep.get(s, {}) if getattr(st, "stim_metrics_by_sweep", None) else {}
                if not m:
                    return ""
                def _ri(x):
                    try: return int(round(float(x)))
                    except Exception: return None

                f = _ri(m.get("freq_hz"))
                d = _ri(m.get("duration_s"))
                pw_s = m.get("pulse_width_s")
                n_pulses = m.get("n_pulses", None)

                if f is not None and d is not None and pw_s is not None:
                    pw_str = f"{_ri(pw_s)}s" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}ms"
                    return f"{f}Hz{d}s{pw_str}"

                if pw_s is not None and (n_pulses == 1 or f is None):
                    return f"{_ri(pw_s)}sPulse" if pw_s >= 1.0 else f"{_ri(pw_s * 1000)}msPulse"
                return ""

            abf_stem = st.in_path.stem
            chan = st.analyze_chan or ""
            auto_stim = _auto_stim_from_metrics()

            # --- Load autocomplete history and last values---
            history = self._load_save_dialog_history()
            last_values = self._load_last_save_values()

            # --- Build file info from state and project builder ---
            file_info = self._get_current_file_info(abf_stem)

            # --- Name builder dialog (with auto stim suggestion and file info) ---
            # Dialog is non-modal so user can click on Project Builder tab
            dlg = SaveMetaDialog(
                abf_name=abf_stem,
                channel=chan,
                parent=self.window,
                auto_stim=auto_stim,
                history=history,
                last_values=last_values,
                main_window=self.window,
                file_info=file_info
            )
            # Make dialog non-modal but stay on top
            dlg.setWindowModality(QtCore_Qt.WindowModality.NonModal)
            dlg.setWindowFlag(QtCore_Qt.WindowType.WindowStaysOnTopHint, True)

            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            dialog_vals = dlg.values()  # Renamed to avoid collision

            # --- Update history with new values and save last used values ---
            self._update_save_dialog_history(dialog_vals)
            self._save_last_save_values(dialog_vals)
            suggested = self._sanitize_token(dialog_vals["preview"]) or "analysis"
            want_picker = bool(dialog_vals.get("choose_dir", False))

            target_exact = "Pleth_App_analysis"          # <- required folder name (lowercase 'a')
            target_lower = target_exact.lower()

            def _nearest_analysis_ancestor(p: Path) -> Path | None:
                # Return the closest ancestor (including self) named Pleth_App_analysis (case-insensitive)
                for cand in [p] + list(p.parents):
                    if cand.name.lower() == target_lower:
                        return cand
                return None

            if want_picker:
                # User chooses a folder; we keep your previous smart logic here.
                default_root = Path(self.window.settings.value("save_root", str(st.in_path.parent)))
                chosen = QFileDialog.getExistingDirectory(
                    self.window,  # Use window (QWidget) as parent, not self (ExportManager)
                    "Choose a folder (files may go into an existing 'Pleth_App_analysis' here)",
                    str(default_root)
                )
                if not chosen:
                    return
                chosen_path = Path(chosen)

                # 1) If chosen folder is inside an ancestor named Pleth_App_analysis → save THERE (the ancestor)
                anc = _nearest_analysis_ancestor(chosen_path)
                if anc is not None:
                    final_dir = anc
                else:
                    # 2) If the chosen folder already contains 'Pleth_App_analysis' or 'Pleth_App_Analysis' subfolder → use it
                    sub_exact   = chosen_path / "Pleth_App_analysis"
                    sub_variant = chosen_path / "Pleth_App_Analysis"
                    if sub_exact.is_dir():
                        final_dir = sub_exact
                    elif sub_variant.is_dir():
                        final_dir = sub_variant
                    else:
                        # 3) Otherwise, create Pleth_App_analysis directly under the chosen folder
                        final_dir = chosen_path / target_exact
                        try:
                            final_dir.mkdir(parents=True, exist_ok=True)
                        except Exception as e:
                            self._show_message_box(QMessageBox.Icon.Critical, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                            return

                # Remember the last *picker* root only when the picker is used
                self.window.settings.setValue("save_root", str(chosen_path))

            else:
                # UNCHECKED: Always use the CURRENT ABF DIRECTORY (not a remembered one)
                base_root = st.in_path.parent if getattr(st, "in_path", None) else Path.cwd()
                final_dir = base_root / target_exact
                try:
                    final_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    self._show_message_box(QMessageBox.Icon.Critical, "Save analyzed data", f"Could not create folder:\n{final_dir}\n\n{e}")
                    return
                # IMPORTANT: Do NOT overwrite 'save_root' here — we don't want to "remember" anything for the unchecked case.

            # Set base name + meta, then export
            self.window._save_dir = final_dir
            self.window._save_base = suggested
            self.window._save_meta = dialog_vals

            # Get export strategy based on experiment type
            experiment_type = dialog_vals.get("experiment_type", "30hz_stim")
            export_strategy = self._get_export_strategy(experiment_type)
            print(f"[export] Using strategy: {export_strategy.get_strategy_name()}")

            # Extract file export flags from dialog
            save_npz = dialog_vals.get("save_npz", True)  # Always True
            save_timeseries_csv = dialog_vals.get("save_timeseries_csv", True)
            save_breaths_csv = dialog_vals.get("save_breaths_csv", True)
            save_events_csv = dialog_vals.get("save_events_csv", True)
            save_pdf = dialog_vals.get("save_pdf", True)
            save_session = dialog_vals.get("save_session", True)  # Session state (.pleth.npz)
            save_ml_training = dialog_vals.get("save_ml_training", False)  # ML training data (3 CSVs)

            # Check for duplicate files before saving (only check files that will be saved)
            existing_files = []
            expected_suffixes = []
            if save_npz: expected_suffixes.append("_bundle.npz")
            if save_timeseries_csv: expected_suffixes.append("_means_by_time.csv")
            if save_breaths_csv: expected_suffixes.append("_breaths.csv")
            if save_events_csv: expected_suffixes.append("_events.csv")
            if save_pdf: expected_suffixes.append("_summary.pdf")
            if save_session: expected_suffixes.append("_session.npz")
            # Note: ML training data goes to separate centralized folder, not checked here
            for suffix in expected_suffixes:
                filepath = final_dir / (suggested + suffix)
                if filepath.exists():
                    existing_files.append(filepath.name)

            if existing_files:
                # Show warning dialog
                file_list = "\n  • ".join(existing_files)
                reply = QMessageBox.question(
                    self.window,
                    "Overwrite Existing Files?",
                    f"The following files already exist in:\n{final_dir}\n\n  • {file_list}\n\n"
                    f"Do you want to overwrite them?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No  # Default to No for safety
                )
                if reply == QMessageBox.StandardButton.No:
                    return  # User chose not to overwrite

            base_path = self.window._save_dir / self.window._save_base
            print(f"[save] base path set: {base_path}")
            try:
                self.window.statusbar.showMessage(f"Saving to: {base_path}", 4000)
            except Exception:
                pass

        # -------------------- knobs --------------------
        DS_TARGET_HZ    = 50.0
        CSV_FLUSH_EVERY = 2000
        INCLUDE_TRACES  = bool(getattr(self, "_csv_include_traces", True))
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12

        # -------------------- basics --------------------
        if progress_dialog:
            progress_dialog.setLabelText("Preparing data...")
            progress_dialog.setValue(5)
            QApplication.processEvents()

        any_ch    = next(iter(st.sweeps.values()))
        n_sweeps  = int(any_ch.shape[1])
        N         = int(len(st.t))

        kept_sweeps = [s for s in range(n_sweeps) if s not in getattr(st, "omitted_sweeps", set())]
        S = len(kept_sweeps)
        if S == 0:
            self._show_message_box(QMessageBox.Icon.Warning, "Save analyzed data", "All sweeps are omitted. Nothing to save.")
            return

        # Downsample index used for NPZ + time CSV
        ds_step = max(1, int(round(float(st.sr_hz) / DS_TARGET_HZ))) if st.sr_hz else 1
        ds_idx  = np.arange(0, N, ds_step, dtype=int)
        M       = int(len(ds_idx))

        # Global stim zero and duration (union across KEPT sweeps)
        global_s0, global_s1 = None, None
        if st.stim_chan:
            for s in kept_sweeps:
                spans = st.stim_spans_by_sweep.get(s, [])
                if spans:
                    starts = [a for (a, _) in spans]
                    ends   = [b for (_, b) in spans]
                    m0 = float(min(starts)); m1 = float(max(ends))
                    global_s0 = m0 if global_s0 is None else min(global_s0, m0)
                    global_s1 = m1 if global_s1 is None else max(global_s1, m1)
        have_global_stim = (global_s0 is not None and global_s1 is not None)
        global_dur = (global_s1 - global_s0) if have_global_stim else None

        # Time for NPZ (raw) and for CSV (normalized to global_s0 if present)
        t_ds_raw = st.t[ds_idx]
        csv_t0   = (global_s0 if have_global_stim else 0.0)
        t_ds_csv = (st.t - csv_t0)[ds_idx]

        # -------------------- containers --------------------
        all_keys     = self._metric_keys_in_order()
        # Use focused metrics for both preview AND PDF (same panels, much faster)
        # Full metric set is still available in the timeseries CSV export
        keys_to_compute = self._PREVIEW_CORE_METRICS
        Y_proc_ds    = np.full((M, S), np.nan, dtype=float)
        y2_ds_by_key = {k: np.full((M, S), np.nan, dtype=float) for k in keys_to_compute}

        # Eupnea masks cached and stored for interval plotting (NOT used for normalization)
        # Normalization uses GMM sniff regions directly (much faster)
        self._eupnea_masks_cache = {}  # Instance variable for caching across export sections
        eupnea_masks_by_sweep = {}     # Local dict for PDF interval overlays

        peaks_by_sweep, on_by_sweep, off_by_sweep, exm_by_sweep, exo_by_sweep = [], [], [], [], []
        sigh_by_sweep = []

        # Cache omitted masks per sweep (computed once, reused for all metrics)
        omitted_masks_cache = {}

        # -------------------- fill per kept sweep (with optional parallel processing) --------------------
        # Use parallel processing for large exports (10+ sweeps) with automatic fallback
        use_parallel_export = getattr(self.window, 'use_parallel_export', True) and S >= 10
        parallel_results = {}

        if use_parallel_export:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                import os

                max_workers = min(os.cpu_count() or 4, 8)
                print(f"[Export] Using parallel metric computation with {max_workers} workers for {S} sweeps...")

                # Worker function for parallel execution
                def compute_sweep_metrics(sweep_idx):
                    return self._compute_all_metrics_for_sweep(
                        sweep_idx, keys_to_compute, ds_idx, N
                    )

                # Execute in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(compute_sweep_metrics, s): s for s in kept_sweeps}
                    completed = 0
                    for future in as_completed(futures):
                        sweep_idx = futures[future]
                        result = future.result()
                        parallel_results[sweep_idx] = result
                        completed += 1
                        if completed % 10 == 0 or completed == S:
                            print(f"[Export] Computed metrics for {completed}/{S} sweeps...")

                print(f"[Export] Parallel metric computation complete!")

            except Exception as e:
                print(f"[Export] Parallel export failed: {e}")
                print(f"[Export] Falling back to sequential export...")
                import traceback
                traceback.print_exc()
                parallel_results = {}  # Clear partial results

        # Process results (from parallel or sequential)
        for col, s in enumerate(kept_sweeps):
            if s in parallel_results:
                # Use pre-computed parallel results
                result = parallel_results[s]
                Y_proc_ds[:, col] = result['y_proc_ds']
                peaks_by_sweep.append(result['peaks'])
                on_by_sweep.append(result['onsets'])
                off_by_sweep.append(result['offsets'])
                exm_by_sweep.append(result['expmins'])
                exo_by_sweep.append(result['expoffs'])
                sigh_by_sweep.append(result['sighs'])
                omitted_masks_cache[s] = result['omit_mask']

                # Fill metric traces
                for k in keys_to_compute:
                    trace = result['traces'].get(k)
                    if trace is not None:
                        y2_ds_by_key[k][:, col] = trace
            else:
                # Sequential computation for this sweep
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                Y_proc_ds[:, col] = y_proc[ds_idx]

                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br  = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    # backfill breaths for this sweep so exports are consistent
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    st.breath_by_sweep[s] = br
                if br is None:
                    br = {
                        "onsets":  np.array([], dtype=int),
                        "offsets": np.array([], dtype=int),
                        "expmins": np.array([], dtype=int),
                        "expoffs": np.array([], dtype=int),
                    }

                peaks_by_sweep.append(pks)
                on_by_sweep.append(np.asarray(br.get("onsets",  []), dtype=int))
                off_by_sweep.append(np.asarray(br.get("offsets", []), dtype=int))
                exm_by_sweep.append(np.asarray(br.get("expmins", []), dtype=int))
                exo_by_sweep.append(np.asarray(br.get("expoffs", []), dtype=int))

                # sighs for this sweep (peak indices)
                sighs = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                # Keep only valid in-range indices
                sighs = sighs[(sighs >= 0) & (sighs < len(y_proc))]
                sigh_by_sweep.append(sighs)

                # Build omitted mask once per sweep (only if needed)
                has_omitted = s in st.omitted_sweeps or s in st.omitted_ranges
                if has_omitted:
                    omitted_masks_cache[s] = create_omitted_mask(s, N, st.omitted_ranges, st.omitted_sweeps)
                else:
                    omitted_masks_cache[s] = None  # No masking needed

                for k in keys_to_compute:
                    # Special handling for probability metrics - use ALL peaks (including noise)
                    if k in ('p_noise', 'p_breath'):
                        all_pks_data = st.all_peaks_by_sweep.get(s, None)
                        all_br = st.all_breaths_by_sweep.get(s, None)
                        if all_pks_data is not None and all_br is not None:
                            all_pks = all_pks_data['indices']
                            y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, all_pks, all_br, sweep=s)
                        else:
                            # Fallback if all_peaks not available
                            y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=s)
                    else:
                        y2 = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=s)

                    if y2 is not None and len(y2) == N:
                        # Apply cached omitted mask if it exists
                        omit_mask = omitted_masks_cache[s]
                        if omit_mask is not None:
                            y2[~omit_mask] = np.nan

                        y2_ds_by_key[k][:, col] = y2[ds_idx]

        if progress_dialog:
            progress_dialog.setLabelText("Computing metrics...")
            progress_dialog.setValue(15)
            QApplication.processEvents()

        # -------------------- Build cached traces (needed for event CTA, not basic preview) --------------------
        # For event-aligned CTA, we need full metric traces. Skip this for basic preview mode.
        keys_for_cta = [k for k in all_keys if k not in self._EXCLUDE_FOR_CSV]
        cached_traces_by_sweep = {}

        # Only build cached traces if not in preview mode (for full export/CTA)
        if not preview_only:
            def _build_cached_traces_if_needed():
                """Build cached metric traces for CTA preview/export if not already cached."""
                cached = {}

                for s in kept_sweeps:
                    y_proc = self.window._get_processed_for(st.analyze_chan, s)
                    pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                    br = st.breath_by_sweep.get(s, None)
                    if br is None and pks.size:
                        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                        st.breath_by_sweep[s] = br
                    if br is None:
                        continue

                    traces_for_sweep = {}
                    for k in keys_for_cta:
                        if k in metrics.METRICS:
                            # Special handling for probability metrics - use ALL peaks (including noise)
                            if k in ('p_noise', 'p_breath'):
                                all_pks_data = st.all_peaks_by_sweep.get(s, None)
                                all_br = st.all_breaths_by_sweep.get(s, None)
                                if all_pks_data is not None and all_br is not None:
                                    all_pks = all_pks_data['indices']
                                    trace = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, all_pks, all_br, sweep=s)
                                else:
                                    trace = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=s)
                            else:
                                trace = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=s)

                            # Apply cached omitted mask if it exists (reuse from main loop)
                            if trace is not None:
                                omit_mask = omitted_masks_cache.get(s)
                                if omit_mask is not None:
                                    trace[~omit_mask] = np.nan
                            traces_for_sweep[k] = trace
                    cached[s] = traces_for_sweep

                return cached

            # Build cached traces for CTA (only for full export, not preview)
            cached_traces_by_sweep = _build_cached_traces_if_needed()

        # -------------------- Save files (skip if preview_only) --------------------
        if not preview_only:
            # -------------------- (1) NPZ bundle (downsampled) --------------------
            base     = self.window._save_dir / self.window._save_base
            npz_path = base.with_name(base.name + "_bundle.npz")
    
            # Pack stim spans in KEPT order (align with columns)
            stim_obj = np.empty(S, dtype=object)
            for col, s in enumerate(kept_sweeps):
                spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
                stim_obj[col] = np.array(spans, dtype=float).reshape(-1, 2) if spans else np.empty((0, 2), dtype=float)
    
            peaks_obj = np.array(peaks_by_sweep, dtype=object)
            on_obj    = np.array(on_by_sweep,  dtype=object)
            off_obj   = np.array(off_by_sweep, dtype=object)
            exm_obj   = np.array(exm_by_sweep, dtype=object)
            exo_obj   = np.array(exo_by_sweep, dtype=object)
            sigh_obj  = np.array(sigh_by_sweep, dtype=object)

            # Pack sniffing regions (aligned with kept_sweeps)
            sniff_obj = np.empty(S, dtype=object)
            for col, s in enumerate(kept_sweeps):
                regions = st.sniff_regions_by_sweep.get(s, [])
                # Convert to numpy array of shape (N_regions, 2) for start/end times
                sniff_obj[col] = np.array(regions, dtype=float).reshape(-1, 2) if regions else np.empty((0, 2), dtype=float)

            # Pack bout annotations (event channel markings) - aligned with kept_sweeps
            bout_obj = np.empty(S, dtype=object)
            for col, s in enumerate(kept_sweeps):
                bouts = st.bout_annotations.get(s, [])
                # Convert list of dicts to structured format for NPZ
                if bouts:
                    bout_obj[col] = np.array([
                        (b['start_time'], b['end_time'], b['id'])
                        for b in bouts
                    ], dtype=[('start_time', float), ('end_time', float), ('id', int)])
                else:
                    bout_obj[col] = np.array([], dtype=[('start_time', float), ('end_time', float), ('id', int)])

            # Only save metrics that were computed (keys_to_compute, not all_keys)
            y2_kwargs_ds = {f"y2_{k}_ds": y2_ds_by_key[k] for k in y2_ds_by_key}
    
            meta = {
                "analyze_channel": st.analyze_chan,
                "sr_hz": float(st.sr_hz),
                "n_sweeps_total": int(n_sweeps),
                "n_sweeps_kept": int(S),
                "kept_sweeps": [int(s) for s in kept_sweeps],                  # original indices
                "omitted_sweeps": sorted(int(x) for x in getattr(st, "omitted_sweeps", set())),
                "abf_path": str(getattr(st, "in_path", "")),
                "ui_meta": getattr(self.window, "_save_meta", {}),
                "excluded_for_csv": sorted(list(self._EXCLUDE_FOR_CSV)),
                "ds_target_hz": float(DS_TARGET_HZ),
                "ds_step": int(ds_step),
                "csv_time_zero": float(csv_t0),
                "csv_includes_traces": bool(INCLUDE_TRACES),
                "norm_window_s": float(NORM_BASELINE_WINDOW_S),
                # Filter settings (for future NPZ reopening)
                "use_low": bool(st.use_low),
                "use_high": bool(st.use_high),
                "use_mean_sub": bool(st.use_mean_sub),
                "use_invert": bool(st.use_invert),
                "low_hz": float(st.low_hz) if st.low_hz else None,
                "high_hz": float(st.high_hz) if st.high_hz else None,
                "mean_val": float(st.mean_val),
                "filter_order": int(self.window.filter_order),
                # Notch filter (if active)
                "notch_filter_lower": float(self.window.notch_filter_lower) if self.window.notch_filter_lower else None,
                "notch_filter_upper": float(self.window.notch_filter_upper) if self.window.notch_filter_upper else None,
                # Channel info (for reopening)
                "channel_names": list(st.channel_names) if st.channel_names else [],
                "stim_chan": str(st.stim_chan) if st.stim_chan else None,
                "event_channel": str(st.event_channel) if st.event_channel else None,
                # Navigation state
                "window_dur_s": float(st.window_dur_s),
            }
    
            # Save enhanced NPZ bundle with timeseries data (for fast consolidation)
            # This will be updated after timeseries normalization is computed
            _npz_timeseries_data = {
                't_ds': t_ds_raw,
                'Y_proc_ds': Y_proc_ds,
                'peaks_by_sweep': peaks_obj,
                'onsets_by_sweep': on_obj,
                'offsets_by_sweep': off_obj,
                'expmins_by_sweep': exm_obj,
                'expoffs_by_sweep': exo_obj,
                'sigh_idx_by_sweep': sigh_obj,
                'sniff_regions_by_sweep': sniff_obj,
                'bout_annotations_by_sweep': bout_obj,
                'stim_spans_by_sweep': stim_obj,
                'meta_json': json.dumps(meta),
                **y2_kwargs_ds,
            }
    
            # -------------------- helpers for normalization (time CSV) --------------------
            def _per_sweep_baseline_for_time(A_ds: np.ndarray) -> np.ndarray:
                """
                A_ds: (M,S) downsampled metric matrix.
                Returns b[S]: mean over last NORM_BASELINE_WINDOW_S before 0; fallback to first W after 0.
                """
                b = np.full((A_ds.shape[1],), np.nan, dtype=float)
                mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
                mask_post = (t_ds_csv >=  0.0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)
                for sidx in range(A_ds.shape[1]):
                    col = A_ds[:, sidx]
                    vals = col[mask_pre]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = col[mask_post]
                        vals = vals[np.isfinite(vals)]
                    if vals.size:
                        b[sidx] = float(np.mean(vals))
                return b
    
            def _normalize_matrix_by_baseline(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
                out = np.full_like(A_ds, np.nan)
                for sidx in range(A_ds.shape[1]):
                    bs = b[sidx]
                    if np.isfinite(bs) and abs(bs) > EPS_BASE:
                        out[:, sidx] = A_ds[:, sidx] / bs
                return out
            
        
    
    
            # -------------------- (2) Per-time CSV (raw + normalized appended) --------------------
            csv_time_path = base.with_name(base.name + "_timeseries.csv")
            # Use curated list of core metrics for timeseries (13 metrics)
            # These are the most useful for analysis and include all metrics needed by consolidation
            keys_for_csv = [k for k in self._TIMESERIES_METRICS if k in y2_ds_by_key]
    
            if progress_dialog:
                progress_dialog.setLabelText("Computing time-based normalization...")
                progress_dialog.setValue(25)
                QApplication.processEvents()

            # Build normalized stacks per metric (TIME-BASED, per-sweep)
            y2_ds_by_key_norm = {}
            baseline_by_key   = {}
            for k in keys_for_csv:
                b = _per_sweep_baseline_for_time(y2_ds_by_key[k])
                baseline_by_key[k] = b
                y2_ds_by_key_norm[k] = _normalize_matrix_by_baseline(y2_ds_by_key[k], b)

            if progress_dialog:
                progress_dialog.setLabelText("Computing eupnea-based normalization...")
                progress_dialog.setValue(35)
                QApplication.processEvents()

            # Build EUPNEA-BASED normalization (pooled across all sweeps)
            # Use efficient approach: extract from pre-computed matrices
            y2_ds_by_key_norm_eupnea = {}
            eupnea_baseline_by_key = {}

            # Pre-populate eupnea masks cache for all kept sweeps
            # Use active classifier (GMM, XGBoost, RF, MLP) for eupnea detection
            active_classifier = getattr(st, 'active_eupnea_sniff_classifier', 'gmm')
            print(f"[CSV-time] Pre-computing eupnea masks for {len(kept_sweeps)} sweeps using {active_classifier} classifier...")
            for s in kept_sweeps:
                if s not in self._eupnea_masks_cache:
                    y_proc = self.window._get_processed_for(st.analyze_chan, s)
                    # Use active classifier for eupnea detection (works with GMM, XGBoost, RF, MLP)
                    eupnea_mask = self.window._compute_eupnea_from_active_classifier(s, len(y_proc))
                    # Fallback to traditional method if no classifier data available
                    if eupnea_mask.sum() == 0:
                        pks = st.peaks_by_sweep.get(s, np.array([]))
                        br = st.breath_by_sweep.get(s, {})
                        eupnea_mask = metrics.detect_eupnic_regions(
                            st.t, y_proc, st.sr_hz, pks,
                            br.get('onsets', np.array([])),
                            br.get('offsets', np.array([])),
                            br.get('expmins', np.array([])),
                            br.get('expoffs', np.array([])),
                            freq_threshold_hz=5.0, min_duration_sec=2.0
                        )
                    self._eupnea_masks_cache[s] = eupnea_mask

            # Compute baselines by extracting from y2_ds_by_key matrices (using pre-computed cache)
            # OPTIMIZATION: Build index mapping once instead of calling argmin repeatedly
            print(f"[CSV-time] Computing eupnea baselines for {len(keys_for_csv)} metrics...")

            # Pre-compute mapping from downsampled indices to original indices
            # MEMORY-EFFICIENT: Use searchsorted instead of creating huge distance matrix
            t0 = float(global_s0) if have_global_stim else 0.0
            t_targets = t_ds_csv + t0  # Shape: (len(t_ds_csv),)

            # Use searchsorted for O(log n) lookups without memory overhead
            # This finds the insertion point for each target, then we check neighbors
            insert_idx = np.searchsorted(st.t, t_targets)
            insert_idx = np.clip(insert_idx, 1, len(st.t) - 1)  # Ensure valid range for neighbor check

            # Check if left or right neighbor is closer
            left_dist = np.abs(st.t[insert_idx - 1] - t_targets)
            right_dist = np.abs(st.t[insert_idx] - t_targets)
            ds_to_orig_idx = np.where(left_dist < right_dist, insert_idx - 1, insert_idx)

            # Identify baseline indices (t < 0)
            baseline_ds_mask = t_ds_csv < 0
            poststim_ds_mask = (t_ds_csv >= 0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)

            for k in keys_for_csv:
                Y_raw = y2_ds_by_key.get(k, None)
                if Y_raw is None or Y_raw.size == 0:
                    eupnea_baseline_by_key[k] = np.nan
                    y2_ds_by_key_norm_eupnea[k] = np.full_like(Y_raw, np.nan) if Y_raw is not None else None
                    continue

                pooled_vals = []

                # Collect eupneic baseline values from downsampled data
                for col_idx, s in enumerate(kept_sweeps):
                    eupnea_mask = self._eupnea_masks_cache.get(s, None)
                    if eupnea_mask is None:
                        continue

                    # Check baseline time points
                    for ds_idx in np.where(baseline_ds_mask)[0]:
                        i_orig = ds_to_orig_idx[ds_idx]

                        # Check if this point is eupneic
                        if i_orig < len(eupnea_mask) and eupnea_mask[i_orig] > 0:
                            val = Y_raw[ds_idx, col_idx]
                            if np.isfinite(val):
                                pooled_vals.append(val)

                # Fallback: include post-stim eupneic periods if insufficient
                if len(pooled_vals) < 10:
                    for col_idx, s in enumerate(kept_sweeps):
                        eupnea_mask = self._eupnea_masks_cache.get(s, None)
                        if eupnea_mask is None:
                            continue

                        for ds_idx in np.where(poststim_ds_mask)[0]:
                            i_orig = ds_to_orig_idx[ds_idx]

                            if i_orig < len(eupnea_mask) and eupnea_mask[i_orig] > 0:
                                val = Y_raw[ds_idx, col_idx]
                                if np.isfinite(val):
                                    pooled_vals.append(val)

                # Compute baseline and normalize
                eupnea_b = float(np.mean(pooled_vals)) if len(pooled_vals) > 0 else np.nan
                eupnea_baseline_by_key[k] = eupnea_b

                if np.isfinite(eupnea_b) and abs(eupnea_b) > EPS_BASE:
                    y2_ds_by_key_norm_eupnea[k] = Y_raw / eupnea_b
                else:
                    y2_ds_by_key_norm_eupnea[k] = np.full_like(Y_raw, np.nan)

            # headers: raw first, then time-based norm, then eupnea-based norm
            header = ["t"]
            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_s{j+1}" for j in range(S)]
                header += [f"{k}_mean", f"{k}_sem"]

            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_norm_s{j+1}" for j in range(S)]
                header += [f"{k}_norm_mean", f"{k}_norm_sem"]

            for k in keys_for_csv:
                if INCLUDE_TRACES:
                    header += [f"{k}_norm_eupnea_s{j+1}" for j in range(S)]
                header += [f"{k}_norm_eupnea_mean", f"{k}_norm_eupnea_sem"]
    
            if progress_dialog:
                progress_dialog.setLabelText("Writing time-series CSV...")
                progress_dialog.setValue(50)
                QApplication.processEvents()

            t_start = time.time()
            self.window.setCursor(Qt.CursorShape.WaitCursor)
            try:
                # Build DataFrame from existing numpy arrays (2-3× faster than row-by-row)
                import pandas as pd

                # Build columns in exact order to match header
                data = {}
                data['t'] = t_ds_csv

                # RAW block
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_s{j+1}'] = y2_ds_by_key[k][:, j]
                    # Compute mean and SEM across sweeps
                    means, sems = self._nanmean_sem(y2_ds_by_key[k], axis=1)
                    data[f'{k}_mean'] = means
                    data[f'{k}_sem'] = sems

                # NORMALIZED block (time-based, per-sweep)
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_norm_s{j+1}'] = y2_ds_by_key_norm[k][:, j]
                    means_n, sems_n = self._nanmean_sem(y2_ds_by_key_norm[k], axis=1)
                    data[f'{k}_norm_mean'] = means_n
                    data[f'{k}_norm_sem'] = sems_n

                # NORMALIZED block (eupnea-based, pooled)
                for k in keys_for_csv:
                    if INCLUDE_TRACES:
                        for j in range(S):
                            data[f'{k}_norm_eupnea_s{j+1}'] = y2_ds_by_key_norm_eupnea[k][:, j]
                    means_e, sems_e = self._nanmean_sem(y2_ds_by_key_norm_eupnea[k], axis=1)
                    data[f'{k}_norm_eupnea_mean'] = means_e
                    data[f'{k}_norm_eupnea_sem'] = sems_e

                # Create DataFrame and optionally write to CSV
                df = pd.DataFrame(data, columns=header)
                if save_timeseries_csv:
                    df.to_csv(csv_time_path, index=False, float_format='%.9g', na_rep='')

            finally:
                self.window.unsetCursor()

            t_elapsed = time.time() - t_start
            if save_timeseries_csv:
                print(f"[CSV] ✓ Time-series data written in {t_elapsed:.2f}s")
            else:
                print(f"[CSV] ⊘ Time-series CSV skipped (computed in {t_elapsed:.2f}s)")

            # Save enhanced NPZ version 3 with timeseries data (for fast consolidation)
            # Version 3: Added bout_annotations_by_sweep and event_channel
            _npz_timeseries_data['npz_version'] = 3
            _npz_timeseries_data['timeseries_t'] = t_ds_csv
            _npz_timeseries_data['timeseries_keys'] = list(keys_for_csv)
            # Save raw, norm, and eupnea-norm metric matrices
            for k in keys_for_csv:
                _npz_timeseries_data[f'ts_raw_{k}'] = y2_ds_by_key[k]
                _npz_timeseries_data[f'ts_norm_{k}'] = y2_ds_by_key_norm[k]
                _npz_timeseries_data[f'ts_eupnea_{k}'] = y2_ds_by_key_norm_eupnea[k]

            np.savez_compressed(npz_path, **_npz_timeseries_data)

            # Count total sniffing regions across all sweeps
            total_sniff_regions = sum(len(st.sniff_regions_by_sweep.get(s, [])) for s in kept_sweeps)
            print(f"[NPZ] ✓ Enhanced bundle saved (v2) with timeseries data")
            print(f"      - {total_sniff_regions} sniffing region(s) saved across {S} sweep(s)")

            if progress_dialog:
                progress_dialog.setLabelText("Writing breath-by-breath CSV...")
                progress_dialog.setValue(60)
                QApplication.processEvents()

            # -------------------- (3) Per-breath CSV (WIDE; with breath classifications) --------------------
            breaths_path = base.with_name(base.name + "_breaths.csv")

            BREATH_COLS = [
                "sweep", "breath", "t", "region", "is_sigh", "is_sniffing", "is_eupnea", "is_apnea",
                "if", "amp_insp", "amp_exp", "area_insp", "area_exp",
                "ti", "te", "vent_proxy",
            ]
            def _headers_for_block(suffix: str | None) -> list[str]:
                if not suffix: return BREATH_COLS[:]
                return [f"{c}_{suffix}" for c in BREATH_COLS]
    
            def _headers_for_block_norm(suffix: str | None) -> list[str]:
                base_cols = _headers_for_block(suffix)
                return [h + "_norm" for h in base_cols]

            def _headers_for_block_norm_eupnea(suffix: str | None) -> list[str]:
                base_cols = _headers_for_block(suffix)
                return [h + "_norm_eupnea" for h in base_cols]

            rows_all, rows_bl, rows_st, rows_po = [], [], [], []
            rows_all_N, rows_bl_N, rows_st_N, rows_po_N = [], [], [], []
            rows_all_E, rows_bl_E, rows_st_E, rows_po_E = [], [], [], []

            need_keys = ["if", "amp_insp", "amp_exp", "area_insp", "area_exp", "ti", "te", "vent_proxy"]

            # Compute EUPNEA-BASED baselines (pooled across all sweeps)
            # Uses GMM sniff regions directly - no need to compute eupnea masks
            eupnea_b_by_k = {}

            # Get thresholds from UI (for breath classification)
            eupnea_thresh = self.window.eupnea_freq_threshold  # Hz
            apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5    # seconds

            t_start = time.time()
            print(f"[CSV] Pre-computing eupnea masks for {len(kept_sweeps)} sweeps...")
            for s in kept_sweeps:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    st.breath_by_sweep[s] = br
                if br is None:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                # No need to compute masks - we'll check breaths directly using GMM results

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Breath events processed in {t_elapsed:.2f}s")

            # Now compute baselines by collecting breath values from eupneic periods
            # IMPORTANT: Cache traces to reuse in main export loop AND PDF generation
            t_start = time.time()
            print(f"[CSV] Computing eupnea baselines and caching traces for {len(need_keys)} metrics...")
            eupnea_baseline_breaths = {k: [] for k in need_keys}

            # Use pre-computed cached traces (already built before save/preview split)
            # This cache is reused in PDF generation to avoid recomputing expensive metrics.
            if not hasattr(self.window, '_export_metric_cache'):
                self.window._export_metric_cache = {}

            for s in kept_sweeps:
                # Keep UI responsive during long computation
                if progress_dialog:
                    QApplication.processEvents()

                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None or pks.size < 2:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)

                # Get pre-computed traces from cache
                traces_for_sweep = cached_traces_by_sweep.get(s, {})

                # Store globally for PDF reuse
                self.window._export_metric_cache[s] = traces_for_sweep

                # Collect baseline eupneic breath values
                for i, mid_idx in enumerate(mids):
                    if t_rel_all[i] >= 0:
                        continue  # Only baseline (t < 0)

                    # Check if this breath is eupneic (NOT sniffing based on GMM results)
                    if not self._is_breath_sniffing(s, i, on):
                        for k in need_keys:
                            trace = traces_for_sweep.get(k, None)
                            if trace is not None and len(trace) == len(st.t):
                                val = trace[mid_idx]
                                if np.isfinite(val):
                                    eupnea_baseline_breaths[k].append(val)

            # Compute baseline means with fallback if insufficient baseline data
            for k in need_keys:
                if len(eupnea_baseline_breaths[k]) >= 10:
                    eupnea_b_by_k[k] = float(np.mean(eupnea_baseline_breaths[k]))
                elif len(eupnea_baseline_breaths[k]) > 0:
                    eupnea_b_by_k[k] = float(np.mean(eupnea_baseline_breaths[k]))
                else:
                    # Fallback: include post-stim eupneic breaths using CACHED traces
                    fallback_vals = []
                    for s in kept_sweeps:
                        traces_cached = cached_traces_by_sweep.get(s, None)
                        if traces_cached is None:
                            continue

                        br = st.breath_by_sweep.get(s, None)
                        if br is None:
                            continue

                        on = np.asarray(br.get("onsets", []), dtype=int)
                        if on.size < 2:
                            continue

                        mids = (on[:-1] + on[1:]) // 2
                        t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)

                        trace = traces_cached.get(k, None)
                        if trace is None or len(trace) != len(st.t):
                            continue

                        for i, mid_idx in enumerate(mids):
                            if 0 <= t_rel_all[i] <= NORM_BASELINE_WINDOW_S:
                                # SIMPLIFIED: Use GMM directly instead of eupnea_mask
                                if not self._is_breath_sniffing(s, i, on):
                                    val = trace[mid_idx]
                                    if np.isfinite(val):
                                        fallback_vals.append(val)

                    eupnea_b_by_k[k] = float(np.mean(fallback_vals)) if len(fallback_vals) > 0 else np.nan

            t_elapsed = time.time() - t_start
            print(f"[CSV] ✓ Baselines and traces computed in {t_elapsed:.2f}s")

            t_start = time.time()
            print(f"[CSV] Writing breath-by-breath data...")
            for s in kept_sweeps:
                # Use cached traces if available, otherwise compute
                traces = cached_traces_by_sweep.get(s, None)
                if traces is None:
                    # Compute if not cached (shouldn't happen, but safety fallback)
                    y_proc = self.window._get_processed_for(st.analyze_chan, s)
                    pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                    br = st.breath_by_sweep.get(s, None)
                    if br is None and pks.size:
                        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                        st.breath_by_sweep[s] = br
                    if br is None:
                        br = {"onsets": np.array([], dtype=int)}

                    traces = {}
                    for k in need_keys:
                        if k in metrics.METRICS:
                            traces[k] = self._compute_metric_trace(k, st.t, y_proc, st.sr_hz, pks, br, sweep=s)
                        else:
                            traces[k] = None
                else:
                    # Use cached data
                    br = st.breath_by_sweep.get(s, None)
                    if br is None:
                        br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
    
                # Per-sweep breath-based baselines (use breath midpoints)
                t_rel_all = (st.t[mids] - (global_s0 if have_global_stim else 0.0)).astype(float)
                mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
                mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)
    
                b_by_k = {}
                for k in need_keys:
                    arr = traces.get(k, None)
                    if arr is None or len(arr) != N:
                        b_by_k[k] = np.nan
                        continue
                    vals = arr[mids[mask_pre_b]]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = arr[mids[mask_post_b]]
                        vals = vals[np.isfinite(vals)]
                    b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan
    
                # Breath classification flags: sigh, sniffing, eupnea, apnea
                # - is_sigh: Any sigh peak within breath interval [on[j], on[j+1])
                # - is_sniffing: Breath midpoint falls within marked sniffing region
                # - is_eupnea: Breath midpoint is part of eupneic (normal) breathing pattern
                # - is_apnea: Breath is preceded by long gap (recovery breath after apnea)

                # Sigh: Any sigh peak within breath interval [on[j], on[j+1])
                sigh_idx = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                sigh_idx = sigh_idx[(sigh_idx >= 0) & (sigh_idx < len(y_proc))]
                is_sigh_per_breath = np.zeros(on.size - 1, dtype=int)
                if sigh_idx.size:
                    for j in range(on.size - 1):
                        a = int(on[j]); b = int(on[j+1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_per_breath[j] = 1

                # Sniffing: Breath midpoint falls within any sniffing region
                sniff_regions = st.sniff_regions_by_sweep.get(s, [])
                is_sniffing_per_breath = np.zeros(on.size - 1, dtype=int)
                if sniff_regions:
                    for j, mid_idx in enumerate(mids):
                        t_mid = st.t[int(mid_idx)]
                        for sn_start, sn_end in sniff_regions:
                            if sn_start <= t_mid <= sn_end:
                                is_sniffing_per_breath[j] = 1
                                break

                # Eupnea: SIMPLIFIED - eupneic = NOT sniffing (already computed above)
                # No need for eupnea_mask - just use inverse of sniffing flag
                is_eupnea_per_breath = (1 - is_sniffing_per_breath).astype(int)

                # Apnea: Breath preceded by long inter-breath interval (recovery breath after apnea)
                is_apnea_per_breath = np.zeros(on.size - 1, dtype=int)
                for j in range(len(mids)):
                    if j > 0:  # Need a previous onset to check inter-breath interval
                        ibi = st.t[int(on[j])] - st.t[int(on[j-1])]  # Time since last breath
                        if ibi > apnea_thresh:
                            # This breath (starting at on[j]) comes after a long gap
                            is_apnea_per_breath[j] = 1

                for i, idx in enumerate(mids, start=1):
                    # Skip breaths in omitted regions
                    if is_sample_in_omitted_region(s, int(idx), st.omitted_ranges):
                        continue

                    t_rel = float(st.t[int(idx)] - (global_s0 if have_global_stim else 0.0))
                    breath_idx = i - 1  # 0-indexed for accessing arrays
                    sigh_flag = str(int(is_sigh_per_breath[breath_idx]))
                    sniff_flag = str(int(is_sniffing_per_breath[breath_idx]))
                    eupnea_flag = str(int(is_eupnea_per_breath[breath_idx]))
                    apnea_flag = str(int(is_apnea_per_breath[breath_idx]))
    
                    # ----- RAW: ALL
                    row_all = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        row_all.append(f"{v:.9g}" if np.isfinite(v) else "")
                    rows_all.append(row_all)

                    # ----- NORM: ALL (time-based, per-sweep)
                    row_allN = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        row_allN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                    rows_all_N.append(row_allN)

                    # ----- NORM_EUPNEA: ALL (eupnea-based, pooled)
                    row_allE = [str(s + 1), str(i), f"{t_rel:.9g}", "all", sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                    for k in need_keys:
                        v = np.nan
                        arr = traces.get(k, None)
                        if arr is not None and len(arr) == N:
                            v = arr[int(idx)]
                        eb = eupnea_b_by_k.get(k, np.nan)
                        ve = (v / eb) if (np.isfinite(v) and np.isfinite(eb) and abs(eb) > EPS_BASE) else np.nan
                        row_allE.append(f"{ve:.9g}" if np.isfinite(ve) else "")
                    rows_all_E.append(row_allE)

                    if have_global_stim:
                        # Determine region based on experiment type
                        if experiment_type == "licking":
                            # Check if breath occurred during a licking bout
                            t_abs = st.t[int(idx)]
                            bout_list = st.bout_annotations.get(s, [])
                            in_bout = any(b['start_time'] <= t_abs <= b['end_time'] for b in bout_list)
                            region = "licking" if in_bout else "not_licking"
                        elif experiment_type == "hargreaves":
                            # Check if breath occurred during a heat trial
                            t_abs = st.t[int(idx)]
                            bout_list = st.bout_annotations.get(s, [])
                            # Find which bout (if any) this breath belongs to
                            in_bout_during = False
                            in_bout_post = False
                            for b in bout_list:
                                if b['start_time'] <= t_abs <= b['end_time']:
                                    in_bout_during = True
                                    break
                                elif b['end_time'] < t_abs <= b['end_time'] + 10.0:  # Post-withdrawal window (10s)
                                    in_bout_post = True
                            if in_bout_during:
                                region = "during_heat"
                            elif in_bout_post:
                                region = "post_withdrawal"
                            else:
                                region = "baseline"
                        else:
                            # Default: 30Hz stim behavior (baseline/stim/post)
                            region = "Baseline" if t_rel < 0 else ("Stim" if t_rel <= global_dur else "Post")

                        # RAW regional row
                        row_reg = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            row_reg.append(f"{v:.9g}" if np.isfinite(v) else "")
                        if region == "Baseline": rows_bl.append(row_reg)
                        elif region == "Stim":  rows_st.append(row_reg)
                        else:                   rows_po.append(row_reg)

                        # NORM regional row (time-based, per-sweep)
                        row_regN = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            b = b_by_k.get(k, np.nan)
                            vn = (v / b) if (np.isfinite(v) and np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                            row_regN.append(f"{vn:.9g}" if np.isfinite(vn) else "")
                        if region == "Baseline": rows_bl_N.append(row_regN)
                        elif region == "Stim":  rows_st_N.append(row_regN)
                        else:                   rows_po_N.append(row_regN)

                        # NORM_EUPNEA regional row (eupnea-based, pooled)
                        row_regE = [str(s + 1), str(i), f"{t_rel:.9g}", region, sigh_flag, sniff_flag, eupnea_flag, apnea_flag]
                        for k in need_keys:
                            v = np.nan
                            arr = traces.get(k, None)
                            if arr is not None and len(arr) == N:
                                v = arr[int(idx)]
                            eb = eupnea_b_by_k.get(k, np.nan)
                            ve = (v / eb) if (np.isfinite(v) and np.isfinite(eb) and abs(eb) > EPS_BASE) else np.nan
                            row_regE.append(f"{ve:.9g}" if np.isfinite(ve) else "")
                        if region == "Baseline": rows_bl_E.append(row_regE)
                        elif region == "Stim":  rows_st_E.append(row_regE)
                        else:                   rows_po_E.append(row_regE)
    
            def _pad_row(row, want_len):
                if row is None: return [""] * want_len
                if len(row) < want_len: return row + [""] * (want_len - len(row))
                return row
    
            headers_all = _headers_for_block(None)
            headers_bl  = _headers_for_block("baseline")
            headers_st  = _headers_for_block("stim")
            headers_po  = _headers_for_block("post")

            headers_allN = _headers_for_block_norm(None)
            headers_blN  = _headers_for_block_norm("baseline")
            headers_stN  = _headers_for_block_norm("stim")
            headers_poN  = _headers_for_block_norm("post")

            headers_allE = _headers_for_block_norm_eupnea(None)
            headers_blE  = _headers_for_block_norm_eupnea("baseline")
            headers_stE  = _headers_for_block_norm_eupnea("stim")
            headers_poE  = _headers_for_block_norm_eupnea("post")

            have_stim_blocks = have_global_stim and (len(rows_bl) + len(rows_st) + len(rows_po) > 0)

            # Build DataFrames from row lists (1.5-2× faster than row-by-row)
            import pandas as pd

            if not have_stim_blocks:
                # RAW + NORM + NORM_EUPNEA (ALL only)
                df_all = pd.DataFrame(rows_all, columns=headers_all) if rows_all else pd.DataFrame(columns=headers_all)
                df_all_N = pd.DataFrame(rows_all_N, columns=headers_allN) if rows_all_N else pd.DataFrame(columns=headers_allN)
                df_all_E = pd.DataFrame(rows_all_E, columns=headers_allE) if rows_all_E else pd.DataFrame(columns=headers_allE)

                # Add separator columns and concatenate
                sep1 = pd.DataFrame({'': [''] * len(df_all)}) if len(df_all) > 0 else pd.DataFrame({'': []})
                sep2 = pd.DataFrame({'': [''] * len(df_all)}) if len(df_all) > 0 else pd.DataFrame({'': []})

                df_combined = pd.concat([df_all, sep1, df_all_N, sep2, df_all_E], axis=1)
            else:
                # RAW blocks, then NORM blocks, then NORM_EUPNEA blocks
                df_all = pd.DataFrame(rows_all, columns=headers_all) if rows_all else pd.DataFrame(columns=headers_all)
                df_bl = pd.DataFrame(rows_bl, columns=headers_bl) if rows_bl else pd.DataFrame(columns=headers_bl)
                df_st = pd.DataFrame(rows_st, columns=headers_st) if rows_st else pd.DataFrame(columns=headers_st)
                df_po = pd.DataFrame(rows_po, columns=headers_po) if rows_po else pd.DataFrame(columns=headers_po)

                df_all_N = pd.DataFrame(rows_all_N, columns=headers_allN) if rows_all_N else pd.DataFrame(columns=headers_allN)
                df_bl_N = pd.DataFrame(rows_bl_N, columns=headers_blN) if rows_bl_N else pd.DataFrame(columns=headers_blN)
                df_st_N = pd.DataFrame(rows_st_N, columns=headers_stN) if rows_st_N else pd.DataFrame(columns=headers_stN)
                df_po_N = pd.DataFrame(rows_po_N, columns=headers_poN) if rows_po_N else pd.DataFrame(columns=headers_poN)

                df_all_E = pd.DataFrame(rows_all_E, columns=headers_allE) if rows_all_E else pd.DataFrame(columns=headers_allE)
                df_bl_E = pd.DataFrame(rows_bl_E, columns=headers_blE) if rows_bl_E else pd.DataFrame(columns=headers_blE)
                df_st_E = pd.DataFrame(rows_st_E, columns=headers_stE) if rows_st_E else pd.DataFrame(columns=headers_stE)
                df_po_E = pd.DataFrame(rows_po_E, columns=headers_poE) if rows_po_E else pd.DataFrame(columns=headers_poE)

                # Create separator columns (use max length for consistency)
                max_len = max(
                    len(df_all), len(df_bl), len(df_st), len(df_po),
                    len(df_all_N), len(df_bl_N), len(df_st_N), len(df_po_N),
                    len(df_all_E), len(df_bl_E), len(df_st_E), len(df_po_E)
                ) if any([len(df_all), len(df_bl), len(df_st), len(df_po)]) else 0

                sep = pd.DataFrame({'': [''] * max_len}) if max_len > 0 else pd.DataFrame({'': []})

                # Concatenate with separators
                df_combined = pd.concat([
                    df_all, sep.copy(), df_bl, sep.copy(), df_st, sep.copy(), df_po, sep.copy(),
                    df_all_N, sep.copy(), df_bl_N, sep.copy(), df_st_N, sep.copy(), df_po_N, sep.copy(),
                    df_all_E, sep.copy(), df_bl_E, sep.copy(), df_st_E, sep.copy(), df_po_E
                ], axis=1)

            # Optionally write to CSV
            if save_breaths_csv:
                df_combined.to_csv(breaths_path, index=False, na_rep='')

            t_elapsed = time.time() - t_start
            if save_breaths_csv:
                print(f"[CSV] ✓ Breath data written in {t_elapsed:.2f}s")
            else:
                print(f"[CSV] ⊘ Breaths CSV skipped (computed in {t_elapsed:.2f}s)")

            if progress_dialog:
                progress_dialog.setLabelText("Writing events CSV...")
                progress_dialog.setValue(70)
                QApplication.processEvents()

            # -------------------- (4) Events CSV (stimulus, apnea, eupnea intervals) --------------------
            t_start = time.time()
            events_path = base.with_name(base.name + "_events.csv")

            events_rows = []

            # Get thresholds from UI
            eupnea_thresh = self.window.eupnea_freq_threshold  # Hz
            apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5    # seconds

            for s in kept_sweeps:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks    = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br     = st.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                if br is None:
                    br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                # Stimulus events
                stim_on_idx = st.stim_onsets_by_sweep.get(s, None)
                stim_off_idx = st.stim_offsets_by_sweep.get(s, None)

                if stim_on_idx is not None and len(stim_on_idx) > 0:
                    for i, on_idx in enumerate(stim_on_idx):
                        start_time = float(st.t[int(on_idx)])
                        if stim_off_idx is not None and i < len(stim_off_idx):
                            end_time = float(st.t[int(stim_off_idx[i])])
                        else:
                            end_time = np.nan

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            if np.isfinite(end_time):
                                end_time -= global_s0

                        duration = end_time - start_time if np.isfinite(end_time) else np.nan

                        events_rows.append([
                            str(s + 1),
                            "stimulus",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}" if np.isfinite(end_time) else "",
                            f"{duration:.9g}" if np.isfinite(duration) else ""
                        ])

                # Apnea and eupnea events (only if we have breath data)
                if on.size >= 2:
                    # Detect apnea regions
                    apnea_mask = metrics.detect_apneas(
                        st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                        min_apnea_duration_sec=apnea_thresh
                    )

                    # Retrieve eupnea mask from cache (computed earlier)
                    eupnea_mask = self._eupnea_masks_cache.get(s, None)
                    if eupnea_mask is None:
                        # Fallback: compute if not cached - use active classifier
                        eupnea_mask = self.window._compute_eupnea_from_active_classifier(s, len(y_proc))
                        # Fallback to traditional method if no classifier data available
                        if eupnea_mask.sum() == 0:
                            eupnea_mask = metrics.detect_eupnic_regions(
                                st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                                freq_threshold_hz=eupnea_thresh
                            )
                        self._eupnea_masks_cache[s] = eupnea_mask

                    # Convert masks to interval lists
                    def mask_to_intervals(mask):
                        """
                        Convert boolean mask to list of (start_idx, end_idx) intervals.
                        VECTORIZED: uses np.diff to find transitions (10-20× faster than Python loop).
                        """
                        if len(mask) == 0:
                            return []

                        # Pad mask with zeros at both ends to detect edge transitions
                        # Find transitions: 0→1 (region starts) and 1→0 (region ends)
                        padded = np.concatenate(([0], mask.astype(int), [0]))
                        diff = np.diff(padded)
                        starts = np.where(diff == 1)[0]      # indices where region begins
                        ends = np.where(diff == -1)[0] - 1    # indices where region ends (inclusive)

                        return list(zip(starts.tolist(), ends.tolist()))

                    # Add apnea intervals
                    apnea_intervals = mask_to_intervals(apnea_mask > 0)
                    for start_idx, end_idx in apnea_intervals:
                        start_time = float(st.t[int(start_idx)])
                        end_time = float(st.t[int(end_idx)])

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            end_time -= global_s0

                        duration = end_time - start_time

                        events_rows.append([
                            str(s + 1),
                            "apnea",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}",
                            f"{duration:.9g}"
                        ])

                    # Add eupnea intervals
                    eupnea_intervals = mask_to_intervals(eupnea_mask > 0)
                    for start_idx, end_idx in eupnea_intervals:
                        start_time = float(st.t[int(start_idx)])
                        end_time = float(st.t[int(end_idx)])

                        # Convert to relative time if global stim available
                        if have_global_stim:
                            start_time -= global_s0
                            end_time -= global_s0

                        duration = end_time - start_time

                        events_rows.append([
                            str(s + 1),
                            "eupnea",
                            f"{start_time:.9g}",
                            f"{end_time:.9g}",
                            f"{duration:.9g}"
                        ])

                # Add sniffing bout intervals (GMM-based)
                sniff_regions = st.sniff_regions_by_sweep.get(s, [])
                for start_time, end_time in sniff_regions:
                    # start_time and end_time are already in seconds

                    # Convert to relative time if global stim available
                    if have_global_stim:
                        start_time_rel = start_time - global_s0
                        end_time_rel = end_time - global_s0
                    else:
                        start_time_rel = start_time
                        end_time_rel = end_time

                    duration = end_time - start_time

                    events_rows.append([
                        str(s + 1),
                        "sniffing",
                        f"{start_time_rel:.9g}",
                        f"{end_time_rel:.9g}",
                        f"{duration:.9g}"
                    ])

                # Add eupnea regions (GMM-based) if available
                eupnea_regions = getattr(st, 'eupnea_regions_by_sweep', {}).get(s, [])
                for start_time, end_time in eupnea_regions:
                    # start_time and end_time are already in seconds

                    # Convert to relative time if global stim available
                    if have_global_stim:
                        start_time_rel = start_time - global_s0
                        end_time_rel = end_time - global_s0
                    else:
                        start_time_rel = start_time
                        end_time_rel = end_time

                    duration = end_time - start_time

                    events_rows.append([
                        str(s + 1),
                        "eupnea_gmm",  # Different label to distinguish from frequency-based eupnea
                        f"{start_time_rel:.9g}",
                        f"{end_time_rel:.9g}",
                        f"{duration:.9g}"
                    ])

                # Add bout annotations (user-marked event channel markers)
                bout_list = st.bout_annotations.get(s, [])
                for bout in bout_list:
                    start_time = bout['start_time']
                    end_time = bout['end_time']

                    # Convert to relative time if global stim available
                    if have_global_stim:
                        start_time_rel = start_time - global_s0
                        end_time_rel = end_time - global_s0
                    else:
                        start_time_rel = start_time
                        end_time_rel = end_time

                    duration = end_time - start_time

                    events_rows.append([
                        str(s + 1),
                        "bout",
                        f"{start_time_rel:.9g}",
                        f"{end_time_rel:.9g}",
                        f"{duration:.9g}"
                    ])

            # Optionally write events CSV using pandas
            import pandas as pd
            df_events = pd.DataFrame(
                events_rows,
                columns=["sweep", "event_type", "start_time", "end_time", "duration"]
            ) if events_rows else pd.DataFrame(columns=["sweep", "event_type", "start_time", "end_time", "duration"])
            if save_events_csv:
                df_events.to_csv(events_path, index=False, na_rep='')

            # Count event types
            if events_rows:
                event_counts = {}
                for row in events_rows:
                    event_type = row[1]
                    event_counts[event_type] = event_counts.get(event_type, 0) + 1
                event_summary = ", ".join([f"{count} {etype}" for etype, count in sorted(event_counts.items())])
            else:
                event_summary = "no events"

            t_elapsed = time.time() - t_start
            if save_events_csv:
                print(f"[CSV] ✓ Events data written in {t_elapsed:.2f}s ({event_summary})")
            else:
                print(f"[CSV] ⊘ Events CSV skipped (computed in {t_elapsed:.2f}s, {event_summary})")

        # -------------------- (4) Summary PDF or Preview --------------------
        t_start = time.time()
        if progress_dialog:
            progress_dialog.setLabelText("Generating summary figures..." if preview_only else "Generating PDF...")
            progress_dialog.setValue(80)
            QApplication.processEvents()

        # Use same core metrics for both preview AND PDF (matching y2_ds_by_key computed at line 1798)
        # Full metric set is still available in the timeseries CSV export
        keys_for_timeplots = self._PREVIEW_CORE_METRICS
        label_by_key = {key: label for (label, key) in metrics.METRIC_SPECS if key in keys_for_timeplots}

        if preview_only:
            # Show interactive preview dialog instead of saving
            # Check experiment type to determine which preview to show

            print("=" * 60)
            print(f"PREVIEW MODE - showing {len(keys_for_timeplots)} core metrics")
            print("=" * 60)

            # First check for pulse experiments (Phase 1 testing)
            is_pulse_exp = self._is_pulse_experiment(kept_sweeps)
            print(f"is_pulse_exp = {is_pulse_exp}")
            print("=" * 60)

            # Check if event channel data exists
            has_event_data = (st.event_channel and st.bout_annotations and
                             any(st.bout_annotations.get(s, []) for s in kept_sweeps))

            try:
                if is_pulse_exp:
                    # PHASE 2+: Pulse experiment - show CTA + 3D (offset-aligned) + 3D (stim-aligned) + probability
                    print("[Preview] Pulse experiment detected - showing 4-page preview")

                    # Generate pulse figures with detailed error tracking
                    try:
                        print("[Preview] Generating CTA figure...")
                        fig_cta = self._generate_pulse_cta_test_figure(kept_sweeps)
                        print("[Preview] CTA figure generated successfully")
                    except Exception as e:
                        print(f"[Preview] ERROR in CTA generation: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                    try:
                        print("[Preview] Generating 3D phase-sorted figure (offset-aligned)...")
                        fig_3d = self._generate_pulse_3d_test_figure(kept_sweeps)
                        print("[Preview] 3D offset-aligned figure generated successfully")
                    except Exception as e:
                        print(f"[Preview] ERROR in 3D generation: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                    try:
                        print("[Preview] Generating 3D phase-sorted figure (stim-aligned)...")
                        fig_3d_stim = self._generate_pulse_3d_stim_aligned_test_figure(kept_sweeps)
                        print("[Preview] 3D stim-aligned figure generated successfully")
                    except Exception as e:
                        print(f"[Preview] ERROR in 3D stim-aligned generation: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                    try:
                        print("[Preview] Generating probability curve...")
                        fig_prob = self._generate_pulse_probability_test_figure(kept_sweeps)
                        print("[Preview] Probability curve generated successfully")
                    except Exception as e:
                        print(f"[Preview] ERROR in probability curve generation: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                    # Cache the computed preview data for pulse experiments
                    self._preview_cache = {
                        'fingerprint': current_fingerprint,
                        'is_pulse_exp': True,
                        'pulse_figures': (fig_cta, fig_3d, fig_3d_stim, fig_prob),
                        't_ds_csv': t_ds_csv,
                        'y2_ds_by_key': y2_ds_by_key,
                        'keys_for_csv': keys_for_timeplots,
                        'label_by_key': label_by_key,
                        'stim_zero': global_s0 if have_global_stim else None,
                        'stim_dur': global_dur if have_global_stim else None,
                    }
                    print(f"[Preview] Cached pulse preview data (fingerprint: {current_fingerprint[:8]}...)")

                    # Show 4-page pulse preview
                    print("[Preview] Showing 4-page pulse preview dialog...")
                    self._show_pulse_4page_preview(fig_cta, fig_3d, fig_3d_stim, fig_prob)

                    # Also show standard metrics preview for pulse experiments
                    print("[Preview] Also showing standard metrics preview...")
                    self._show_summary_preview_dialog(
                        t_ds_csv=t_ds_csv,
                        y2_ds_by_key=y2_ds_by_key,
                        keys_for_csv=keys_for_timeplots,
                        label_by_key=label_by_key,
                        stim_zero=(global_s0 if have_global_stim else None),
                        stim_dur=(global_dur if have_global_stim else None),
                    )

                elif has_event_data:
                    # Show event-aligned CTA preview
                    self._show_event_cta_preview_dialog(
                        kept_sweeps=kept_sweeps,
                        cached_traces_by_sweep=cached_traces_by_sweep,
                        keys_for_csv=keys_for_timeplots,
                        label_by_key=label_by_key,
                    )
                else:
                    # Cache the computed preview data for repeated views
                    self._preview_cache = {
                        'fingerprint': current_fingerprint,
                        'is_pulse_exp': False,
                        'pulse_figures': None,
                        't_ds_csv': t_ds_csv,
                        'y2_ds_by_key': y2_ds_by_key,
                        'keys_for_csv': keys_for_timeplots,
                        'label_by_key': label_by_key,
                        'stim_zero': global_s0 if have_global_stim else None,
                        'stim_dur': global_dur if have_global_stim else None,
                    }
                    print(f"[Preview] Cached preview data (fingerprint: {current_fingerprint[:8]}...)")

                    # Show standard summary preview
                    self._show_summary_preview_dialog(
                        t_ds_csv=t_ds_csv,
                        y2_ds_by_key=y2_ds_by_key,
                        keys_for_csv=keys_for_timeplots,
                        label_by_key=label_by_key,
                        stim_zero=(global_s0 if have_global_stim else None),
                        stim_dur=(global_dur if have_global_stim else None),
                    )
            except Exception as e:
                self._show_message_box(QMessageBox.Icon.Critical, "View Summary", f"Error generating preview:\n{e}")
                import traceback
                traceback.print_exc()
        else:
            # -------------------- PDF Generation --------------------
            pdf_path = None
            event_cta_pdf_path = None
            pulse_pdf_path = None

            if save_pdf:
                # Check experiment type
                is_pulse_exp = self._is_pulse_experiment(kept_sweeps)
                has_event_data = (st.event_channel and st.bout_annotations and
                                 any(st.bout_annotations.get(s, []) for s in kept_sweeps))

                # Generate pulse-specific PDF if pulse experiment
                if is_pulse_exp:
                    print("[PDF] Pulse experiment detected, generating pulse analysis PDF...")
                    pulse_pdf_path = base.with_name(base.name + "_pulse.pdf")
                    try:
                        self._save_pulse_analysis_pdf(pulse_pdf_path, kept_sweeps)
                    except Exception as e:
                        print(f"[save][pulse-pdf] skipped: {e}")
                        import traceback
                        traceback.print_exc()
                        pulse_pdf_path = None

                # Generate event-aligned CTA PDF if event data present
                if has_event_data:
                    # Event-aligned CTA PDF (skip standard PDF for event experiments)
                    print("[PDF] Event channel detected, generating CTA PDF...")
                    event_cta_pdf_path = base.with_name(base.name + "_event_cta.pdf")
                    try:
                        self._save_event_aligned_cta_pdf(
                            out_path=event_cta_pdf_path,
                            kept_sweeps=kept_sweeps,
                            cached_traces_by_sweep=cached_traces_by_sweep,
                            keys_for_csv=keys_for_timeplots,
                            label_by_key=label_by_key,
                        )
                    except Exception as e:
                        print(f"[save][event-cta-pdf] skipped: {e}")
                        import traceback
                        traceback.print_exc()
                        event_cta_pdf_path = None

                # Generate standard summary PDF for pulse experiments or normal experiments
                # (skip only for event experiments)
                if not has_event_data or is_pulse_exp:
                    print("[PDF] Generating standard summary PDF...")
                    pdf_path = base.with_name(base.name + "_summary.pdf")
                    try:
                        self._save_metrics_summary_pdf(
                            out_path=pdf_path,
                            t_ds_csv=t_ds_csv,
                            y2_ds_by_key=y2_ds_by_key,
                            keys_for_csv=keys_for_timeplots,
                            label_by_key=label_by_key,
                            stim_zero=(global_s0 if have_global_stim else None),
                            stim_dur=(global_dur if have_global_stim else None),
                        )
                    except Exception as e:
                        print(f"[save][summary-pdf] skipped: {e}")

            t_elapsed = time.time() - t_start
            if save_pdf:
                print(f"[PDF] ✓ PDF saved in {t_elapsed:.2f}s")
            else:
                print(f"[PDF] ⊘ PDF generation skipped ({t_elapsed:.2f}s saved)")

            # -------------------- done --------------------
            if progress_dialog:
                progress_dialog.setValue(100)
                QApplication.processEvents()

            # Save session state if requested
            session_path = None
            if save_session:
                from core.npz_io import save_state_to_npz
                try:
                    # Session file goes in analysis folder with rich metadata naming
                    session_path = base.with_name(base.name + "_session.npz")
                    # Get GMM cache from main window to preserve user's cluster assignments
                    gmm_cache = getattr(self.window, '_cached_gmm_results', None)

                    # Collect app-level settings from main window
                    app_settings = {
                        'filter_order': self.window.filter_order,
                        'use_zscore_normalization': self.window.use_zscore_normalization,
                        'notch_filter_lower': self.window.notch_filter_lower,
                        'notch_filter_upper': self.window.notch_filter_upper,
                        'apnea_threshold': self.window._parse_float(self.window.ApneaThresh) or 0.5,
                        'active_eupnea_sniff_classifier': self.window.state.active_eupnea_sniff_classifier
                    }

                    save_state_to_npz(st, session_path, include_raw_data=False, gmm_cache=gmm_cache, app_settings=app_settings)
                    print(f"[session] ✓ Session state saved: {session_path.name}")
                except Exception as e:
                    print(f"[session] ✗ Session save failed: {e}")
                    session_path = None

            # Export ML training data if requested
            ml_training_path = None
            ml_counts = None
            if save_ml_training:
                try:
                    # Build metadata from main dialog values (no second popup needed!)
                    import os
                    import socket
                    from datetime import datetime

                    metadata = {
                        'system_username': os.getenv('USERNAME') or os.getenv('USER') or 'unknown',
                        'computer_name': socket.gethostname(),
                        'timestamp': datetime.now().isoformat(),
                        'animal_state': dialog_vals.get('ml_animal_state', 'awake'),
                        'gas': dialog_vals.get('ml_gas', 'room_air'),
                        'quality_score': dialog_vals.get('ml_quality_score', 7),
                    }

                    # Get centralized ML training folder (user can choose where)
                    ml_folder = self._get_ml_training_folder()
                    if ml_folder:
                        # Create filename with timestamp to avoid collisions
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        ml_filename = f"{suggested}_{timestamp}_ml_training.npz"
                        ml_training_path = ml_folder / ml_filename

                        # Check if waveforms should be included
                        include_waveforms = dialog_vals.get("ml_include_waveforms", False)

                        ml_counts = self._export_ml_training_data(
                            ml_training_path, suggested,
                            include_waveforms=include_waveforms,
                            metadata=metadata
                        )
                        print(f"[ML training] ✓ ML training data exported to {ml_training_path}")
                    else:
                        print(f"[ML training] ⚠ ML training export cancelled by user")
                except Exception as e:
                    print(f"[ML training] ✗ ML training export failed: {e}")
                    import traceback
                    tb_str = traceback.format_exc()
                    print(tb_str)
                    ml_training_path = None
                    ml_counts = None

                    # Show user-facing warning dialog with selectable text and full traceback
                    error_dialog = QMessageBox(self.window)
                    error_dialog.setIcon(QMessageBox.Icon.Warning)
                    error_dialog.setWindowTitle("ML Training Export Failed")
                    error_dialog.setText("ML training data could not be exported.")
                    error_dialog.setDetailedText(tb_str)  # Full traceback in expandable section
                    error_dialog.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    error_dialog.exec()

            # Build success message (only include files that were actually saved)
            file_list = [npz_path.name]  # NPZ always saved
            if save_timeseries_csv:
                file_list.append(csv_time_path.name)
            if save_breaths_csv:
                file_list.append(breaths_path.name)
            if save_events_csv:
                file_list.append(events_path.name)
            if save_pdf and pulse_pdf_path:
                file_list.append(pulse_pdf_path.name)
            if save_pdf and pdf_path:
                file_list.append(pdf_path.name)
            if save_pdf and event_cta_pdf_path:
                file_list.append(event_cta_pdf_path.name)
            if save_session and session_path:
                file_list.append(f"{session_path.name} (session)")
            if save_ml_training and ml_training_path and ml_counts:
                # Build detailed ML summary with counts
                ml_summary = f"{ml_training_path.name} (ML training → {ml_training_path.parent.name}/)\n"
                ml_summary += f"  {ml_counts['n_breaths']} breaths: "
                ml_summary += f"{ml_counts['n_eupnea']} eupnea, {ml_counts['n_sniffing']} sniffing, "
                ml_summary += f"{ml_counts['n_sighs']} sighs, {ml_counts['n_rejected']} rejected\n"
                ml_summary += f"  {ml_counts['n_merges']} merges: "
                ml_summary += f"{ml_counts['n_merge_with_next']} merged with next, {ml_counts['n_merged_away']} removed"
                file_list.append(ml_summary)
            elif save_ml_training and ml_training_path:
                # Fallback if counts not available
                file_list.append(f"{ml_training_path.name} (ML training → {ml_training_path.parent.name}/)")
            msg = "Saved:\n" + "\n".join(f"- {name}" for name in file_list)
            print("[save]", msg)
            try:
                self.window.statusbar.showMessage(msg, 6000)
            except Exception:
                pass

            # Show success dialog
            self._show_message_box(
                QMessageBox.Icon.Information,
                "Save Successful",
                f"Files saved successfully to:\n{self.window._save_dir}\n\n{msg}"
            )



    def _mean_sem_1d(self, arr: np.ndarray):
        """Finite-only mean and SEM (ddof=1) for a 1D array. Returns (mean, sem).
        If no finite values -> (nan, nan). If only 1 finite value -> (mean, nan)."""
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        n = int(finite.sum())
        if n == 0:
            return (np.nan, np.nan)
        vals = arr[finite]
        m = float(np.mean(vals))
        if n >= 2:
            s = float(np.std(vals, ddof=1))
            sem = s / np.sqrt(n)
        else:
            sem = np.nan
        return (m, sem)

    def _plot_pulse_cta_overlay(self, ax_all, ax_eupnea, ax_sniff, kept_sweeps: list, global_s0: float = None):
        """
        Plot CTA overlay: 3-panel plot with all sweeps, eupnea-only, and sniffing-only.

        Time window: ±1s around stim onset

        Args:
            ax_all: Matplotlib axis for all traces
            ax_eupnea: Matplotlib axis for eupnea traces only
            ax_sniff: Matplotlib axis for sniffing traces only
            kept_sweeps: List of sweep indices to include
            global_s0: Global stimulus onset time (seconds), or None
        """
        import numpy as np

        st = self.window.state

        # Get per-sweep data
        if st.analyze_chan not in st.sweeps:
            for ax in [ax_all, ax_eupnea, ax_sniff]:
                ax.text(0.5, 0.5, 'No analysis channel data available',
                       ha='center', va='center', transform=ax.transAxes)
            return

        sweep_data = st.sweeps[st.analyze_chan]  # Shape: (n_samples, n_sweeps)
        sr_hz = st.sr_hz if st.sr_hz else 1000.0  # Sample rate

        # Downsample for efficiency (target ~200 Hz for visualization)
        DS_TARGET_HZ = 200.0
        ds_step = max(1, int(round(sr_hz / DS_TARGET_HZ)))

        # Collect traces separated by breath type
        aligned_traces_all = []
        aligned_traces_eupnea = []
        aligned_traces_sniff = []

        # Collect breath events for overlay (all traces)
        breath_event_data_all = {
            'onsets': [],          # Inspiratory onsets
            'offsets': [],         # Inspiratory offsets (end of inspiration)
            'peaks': [],           # Inspiratory peaks
            'expoffs': [],         # Expiratory offsets
            'expeaks': []          # Expiratory peaks (minima)
        }

        # Collect latency-to-peak data (time from stim to first inspiratory peak)
        latencies_to_peak_all = []
        latencies_to_peak_eupnea = []
        latencies_to_peak_sniff = []

        print(f"[CTA] Processing {len(kept_sweeps)} sweeps, sr_hz={sr_hz}, ds_step={ds_step}")

        for s in kept_sweeps:
            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                print(f"[CTA] Sweep {s}: No stim spans")
                continue

            stim_onset = spans[0][0]  # First (and only) pulse onset (in seconds)

            # Get this sweep's data
            if s >= sweep_data.shape[1]:
                print(f"[CTA] Sweep {s}: Index out of range")
                continue
            y_sweep = sweep_data[:, s]

            # Create time vector for this sweep (0 to duration)
            n_samples = len(y_sweep)
            t_sweep = np.arange(n_samples) / sr_hz

            print(f"[CTA] Sweep {s}: stim_onset={stim_onset:.2f}s, sweep_dur={t_sweep[-1]:.2f}s")

            # Extract window around stim: [onset - 1s, onset + 1s]
            window_start = max(0, stim_onset - 1.0)  # Don't go negative
            window_end = min(t_sweep[-1], stim_onset + 1.0)  # Don't exceed sweep length

            # Find indices
            idx_start = np.searchsorted(t_sweep, window_start)
            idx_end = np.searchsorted(t_sweep, window_end)

            # Bounds check
            if idx_start >= len(t_sweep) or idx_end > len(t_sweep) or idx_start >= idx_end:
                print(f"[CTA] Sweep {s}: Bad indices start={idx_start}, end={idx_end}, len={len(t_sweep)}")
                continue

            # Extract and downsample segment
            t_segment = t_sweep[idx_start:idx_end:ds_step] - stim_onset  # Centered at 0
            y_segment = y_sweep[idx_start:idx_end:ds_step]

            print(f"[CTA] Sweep {s}: Extracted {len(t_segment)} samples")

            # Determine if breath before stim is eupneic or sniffing
            breath_events = st.breath_by_sweep.get(s)
            is_sniffing_before_stim = False
            if breath_events is not None:
                onsets = breath_events.get("onsets", [])
                if len(onsets) > 0:
                    onset_times = t_sweep[onsets]
                    onset_before_mask = onset_times < stim_onset
                    if np.any(onset_before_mask):
                        # Find the last breath onset before stim
                        breath_idx_before = int(np.sum(onset_before_mask)) - 1
                        is_sniffing_before_stim = self._is_breath_sniffing(s, breath_idx_before, onsets)

            if len(t_segment) > 1:
                aligned_traces_all.append((t_segment, y_segment, is_sniffing_before_stim))
                if is_sniffing_before_stim:
                    aligned_traces_sniff.append((t_segment, y_segment, is_sniffing_before_stim))
                else:
                    aligned_traces_eupnea.append((t_segment, y_segment, is_sniffing_before_stim))

            # Collect breath events and latencies from breath AFTER stimulus
            if breath_events is not None:
                # Get event indices
                onsets = breath_events.get("onsets", [])
                offsets = breath_events.get("offsets", [])  # Inspiratory offsets
                expoffs = breath_events.get("expoffs", [])  # Expiratory offsets
                peaks = st.peaks_by_sweep.get(s, np.array([]))  # Inspiratory peaks
                expeaks = breath_events.get("expeaks", [])  # Expiratory peaks (minima)

                # Find inspiratory onset AFTER stim (start of breath after stim)
                if len(onsets) > 0:
                    onset_times = t_sweep[onsets]
                    onset_after_mask = onset_times > stim_onset
                    if np.any(onset_after_mask):
                        t_onset = onset_times[onset_after_mask][0]  # First onset after stim
                        if -1.0 <= (t_onset - stim_onset) <= 1.0:  # Within window
                            y_onset = y_sweep[onsets[np.where(onset_after_mask)[0][0]]]
                            breath_event_data_all['onsets'].append((t_onset - stim_onset, y_onset))

                # Find inspiratory offset AFTER stim
                if len(offsets) > 0:
                    offset_times = t_sweep[offsets]
                    offset_after_mask = offset_times > stim_onset
                    if np.any(offset_after_mask):
                        t_offset = offset_times[offset_after_mask][0]  # First offset after stim
                        if -1.0 <= (t_offset - stim_onset) <= 1.0:
                            y_offset = y_sweep[offsets[np.where(offset_after_mask)[0][0]]]
                            breath_event_data_all['offsets'].append((t_offset - stim_onset, y_offset))

                # Find inspiratory peak AFTER stim
                if len(peaks) > 0:
                    peak_times = t_sweep[peaks]
                    peak_after_mask = peak_times > stim_onset
                    if np.any(peak_after_mask):
                        t_peak = peak_times[peak_after_mask][0]  # First peak after stim
                        latency = t_peak - stim_onset  # Latency from stim to peak
                        latencies_to_peak_all.append(latency)
                        if is_sniffing_before_stim:
                            latencies_to_peak_sniff.append(latency)
                        else:
                            latencies_to_peak_eupnea.append(latency)
                        if -1.0 <= latency <= 1.0:
                            y_peak = y_sweep[peaks[np.where(peak_after_mask)[0][0]]]
                            breath_event_data_all['peaks'].append((latency, y_peak))

                # Find expiratory offset AFTER stim
                if len(expoffs) > 0:
                    expoff_times = t_sweep[expoffs]
                    expoff_after_mask = expoff_times > stim_onset
                    if np.any(expoff_after_mask):
                        t_expoff = expoff_times[expoff_after_mask][0]  # First expoff after stim
                        if -1.0 <= (t_expoff - stim_onset) <= 1.0:
                            y_expoff = y_sweep[expoffs[np.where(expoff_after_mask)[0][0]]]
                            breath_event_data_all['expoffs'].append((t_expoff - stim_onset, y_expoff))

                # Find expiratory peak AFTER stim
                if len(expeaks) > 0:
                    expeak_times = t_sweep[expeaks]
                    expeak_after_mask = expeak_times > stim_onset
                    if np.any(expeak_after_mask):
                        t_expeak = expeak_times[expeak_after_mask][0]  # First expeak after stim
                        if -1.0 <= (t_expeak - stim_onset) <= 1.0:
                            y_expeak = y_sweep[expeaks[np.where(expeak_after_mask)[0][0]]]
                            breath_event_data_all['expeaks'].append((t_expeak - stim_onset, y_expeak))

        print(f"[CTA] Successfully aligned {len(aligned_traces_all)} traces (Eupnea: {len(aligned_traces_eupnea)}, Sniff: {len(aligned_traces_sniff)})")

        # Helper function to plot a single panel
        def plot_cta_panel(ax, traces, latencies, breath_events, title, mean_color='red', show_breath_events=True):
            """Plot a single CTA panel with traces, mean, and optional breath events."""
            if not traces:
                ax.text(0.5, 0.5, 'No data available',
                       ha='center', va='center', transform=ax.transAxes)
                return

            # Overlay all traces (semi-transparent gray)
            for t_seg, y_seg, is_sniffing in traces:
                ax.plot(t_seg, y_seg, color='gray', alpha=0.3, linewidth=0.5)

            # Compute and plot mean
            # Interpolate to common time base
            t_common = np.linspace(-1.0, 1.0, 1000)
            y_interp_all = []
            for t_seg, y_seg, is_sniffing in traces:
                # Only interpolate if we have valid data
                if len(t_seg) > 1:
                    y_interp = np.interp(t_common, t_seg, y_seg, left=np.nan, right=np.nan)
                    y_interp_all.append(y_interp)

            if y_interp_all:
                y_mean = np.nanmean(y_interp_all, axis=0)
                y_std = np.nanstd(y_interp_all, axis=0)
                y_sem = y_std / np.sqrt(len(y_interp_all))

                ax.plot(t_common, y_mean, color=mean_color, linewidth=2,
                       label=f'Mean (n={len(traces)})')
                ax.fill_between(t_common, y_mean - y_sem, y_mean + y_sem,
                               color=mean_color, alpha=0.2, label='±1 SEM')

            # Blue shaded stim region (25ms pulse)
            ax.axvspan(0, 0.025, color='blue', alpha=0.15, zorder=1)

            # Plot breath event overlays with transparency (only for "All" panel)
            if show_breath_events:
                if len(breath_events['onsets']) > 0:
                    t_onsets, y_onsets = zip(*breath_events['onsets'])
                    ax.scatter(t_onsets, y_onsets, color='green', marker='o', s=20, alpha=0.4, label='Insp Onset', zorder=5)

                if len(breath_events['offsets']) > 0:
                    t_offsets, y_offsets = zip(*breath_events['offsets'])
                    ax.scatter(t_offsets, y_offsets, color='purple', marker='s', s=20, alpha=0.4, label='Insp Offset', zorder=5)

                if len(breath_events['peaks']) > 0:
                    t_peaks, y_peaks = zip(*breath_events['peaks'])
                    ax.scatter(t_peaks, y_peaks, color='red', marker='^', s=30, alpha=0.4, label='Insp Peak', zorder=5)

                if len(breath_events['expoffs']) > 0:
                    t_expoffs, y_expoffs = zip(*breath_events['expoffs'])
                    ax.scatter(t_expoffs, y_expoffs, color='orange', marker='s', s=20, alpha=0.4, label='Exp Offset', zorder=5)

                if len(breath_events['expeaks']) > 0:
                    t_expeaks, y_expeaks = zip(*breath_events['expeaks'])
                    ax.scatter(t_expeaks, y_expeaks, color='cyan', marker='v', s=30, alpha=0.4, label='Exp Peak', zorder=5)

            # Calculate and display latency-to-peak statistics
            if len(latencies) > 0:
                latency_mean = float(np.mean(latencies))
                if len(latencies) >= 2:
                    latency_std = float(np.std(latencies, ddof=1))
                    latency_sem = latency_std / np.sqrt(len(latencies))
                    latency_text = f"Latency: {latency_mean*1000:.1f}ms ± {latency_sem*1000:.1f}ms SEM (±{latency_std*1000:.1f}ms STD, n={len(latencies)})"
                else:
                    latency_text = f"Latency: {latency_mean*1000:.1f}ms (n=1)"

                # Add text box with latency info
                ax.text(0.02, 0.98, latency_text,
                       transform=ax.transAxes,
                       fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                print(f"[CTA {title}] {latency_text}")

            ax.set_xlabel('Time relative to stim onset (s)', fontsize=9)
            ax.set_ylabel('Airflow (mV)', fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=6)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.0, 1.0)

        # Plot all three panels
        plot_cta_panel(ax_all, aligned_traces_all, latencies_to_peak_all, breath_event_data_all,
                      'CTA - All Traces', mean_color='red', show_breath_events=True)
        plot_cta_panel(ax_eupnea, aligned_traces_eupnea, latencies_to_peak_eupnea, {},
                      'CTA - Eupnea Only', mean_color='green', show_breath_events=False)
        plot_cta_panel(ax_sniff, aligned_traces_sniff, latencies_to_peak_sniff, {},
                      'CTA - Sniffing Only', mean_color='purple', show_breath_events=False)

    def _plot_pulse_2d_sorted_traces(self, ax, kept_sweeps: list):
        """
        Plot 2D phase-sorted aligned traces (simplified version for debugging).

        X-axis: Time (relative to inspiratory onset alignment point)
        Y-axis: Amplitude
        Color: Phase (blue = early, red = late)

        Args:
            ax: Matplotlib 2D axis to plot on
            kept_sweeps: List of sweep indices to include
        """
        import numpy as np
        from matplotlib import cm

        st = self.window.state

        # Get per-sweep data
        if st.analyze_chan not in st.sweeps:
            ax.text(0.5, 0.5, 'No analysis channel data available',
                     ha='center', va='center', transform=ax.transAxes)
            return

        sweep_data = st.sweeps[st.analyze_chan]
        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        # Downsample for efficiency
        DS_TARGET_HZ = 200.0
        ds_step = max(1, int(round(sr_hz / DS_TARGET_HZ)))

        # Collect traces with phase information
        traces_with_phase = []

        print(f"[2D Phase] Processing {len(kept_sweeps)} sweeps for phase-sorted traces...")

        for s in kept_sweeps:
            breath_events = st.breath_by_sweep.get(s)
            if breath_events is None or len(breath_events) == 0:
                print(f"[2D Phase] Sweep {s}: No breath events")
                continue

            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                print(f"[2D Phase] Sweep {s}: No stim spans")
                continue

            stim_onset = spans[0][0]  # Pulse onset time (seconds)

            # Get breath events
            onsets = breath_events.get("onsets", [])
            offsets = breath_events.get("offsets", [])  # Inspiratory offsets (end of inspiration)
            peaks = st.peaks_by_sweep.get(s, np.array([]))  # Inspiratory peaks

            if len(onsets) == 0 or len(offsets) == 0 or len(peaks) == 0:
                print(f"[2D Phase] Sweep {s}: No onsets/offsets/peaks")
                continue

            # Get this sweep's data
            if s >= sweep_data.shape[1]:
                print(f"[2D Phase] Sweep {s}: Sweep index out of bounds")
                continue
            y_sweep = sweep_data[:, s]
            t_sweep = np.arange(len(y_sweep)) / sr_hz

            # Find inspiratory offset BEFORE stim (alignment point)
            offset_times = t_sweep[offsets]
            offset_before_mask = offset_times < stim_onset
            offset_before_stim = offset_times[offset_before_mask]

            if len(offset_before_stim) == 0:
                print(f"[2D Phase] Sweep {s}: No inspiratory offset before stim")
                continue

            t_align = float(offset_before_stim[-1])  # Last inspiratory offset before stim
            offset_idx_align = int(offsets[np.sum(offset_before_mask) - 1])

            # Find inspiratory peak AFTER stim
            peak_times = t_sweep[peaks]
            peak_after_mask = peak_times > stim_onset
            peak_after_stim = peak_times[peak_after_mask]

            if len(peak_after_stim) == 0:
                print(f"[2D Phase] Sweep {s}: No inspiratory peak after stim")
                continue

            peak_indices_after = np.where(peak_after_mask)[0]
            peak_idx_end = int(peaks[peak_indices_after[0]])  # First peak after stim

            # Calculate phase: time from alignment point to stim
            phase = float(stim_onset - t_align)  # Time in seconds

            # Extract segment from inspiratory offset to peak after stim
            idx_start = int(offset_idx_align)
            idx_end = int(peak_idx_end)

            if idx_start >= idx_end or idx_end > len(y_sweep):
                print(f"[2D Phase] Sweep {s}: Invalid index range")
                continue

            t_segment = t_sweep[idx_start:idx_end:ds_step] - t_align  # Relative to alignment
            y_segment = y_sweep[idx_start:idx_end:ds_step]
            stim_t_aligned = stim_onset - t_align

            if len(t_segment) > 1:
                traces_with_phase.append({
                    'sweep': s,
                    'phase': phase,
                    't': t_segment,
                    'y': y_segment,
                    'stim_t': stim_t_aligned,
                })
                print(f"[2D Phase] Sweep {s}: OK (phase={phase:.3f}s, {len(t_segment)} points)")

        print(f"[2D Phase] Collected {len(traces_with_phase)} traces with phase information")

        if not traces_with_phase:
            ax.text(0.5, 0.5, 'No data available for phase-sorted traces',
                     ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by phase
        traces_with_phase.sort(key=lambda x: x['phase'])

        # Normalize phases to [0, 1] for coloring
        phases = np.array([tr['phase'] for tr in traces_with_phase])
        phase_min, phase_max = phases.min(), phases.max()
        if phase_max > phase_min:
            phase_norm = (phases - phase_min) / (phase_max - phase_min)
        else:
            phase_norm = np.zeros_like(phases)

        # Colormap for phase gradient
        cmap = cm.get_cmap('coolwarm')  # Blue (0) → Red (1)

        # Calculate vertical spacing for traces
        # Find min/max amplitude across all traces to determine spacing
        all_amplitudes = np.concatenate([trace['y'] for trace in traces_with_phase])
        amp_range = np.max(all_amplitudes) - np.min(all_amplitudes)
        vertical_spacing = amp_range * 1.2  # 120% of amplitude range for spacing

        # Plot each trace with vertical offset based on sort order
        for i, trace in enumerate(traces_with_phase):
            color = cmap(phase_norm[i])
            y_offset = i * vertical_spacing
            ax.plot(trace['t'], trace['y'] + y_offset,
                   color=color, linewidth=1.0, alpha=0.7)

            # Mark stimulus time with a small vertical tick
            stim_t = trace['stim_t']
            y_at_stim = np.interp(stim_t, trace['t'], trace['y']) + y_offset
            ax.plot(stim_t, y_at_stim, 'r|', markersize=8, markeredgewidth=1.5, alpha=0.6)

        ax.set_xlabel('Time (s) from insp offset', fontsize=9)
        ax.set_ylabel('Sweep # (sorted by stim phase, bottom=early, top=late)', fontsize=9)
        ax.set_title(f'Phase-Sorted Aligned Traces (n={len(traces_with_phase)})\nBlue=Early Phase, Red=Late Phase',
                    fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')  # Only show horizontal grid lines

        # Add colorbar for phase
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=phase_min, vmax=phase_max))
        sm.set_array([])

        # Note: colorbar for 2D axes
        try:
            import matplotlib.pyplot as plt
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            cbar.set_label('Stim Phase (s from insp offset)', rotation=270, labelpad=15, fontsize=8)
        except:
            pass  # Skip colorbar if it fails

    def _plot_pulse_3d_sorted_traces(self, ax, kept_sweeps: list):
        """
        Plot 3D phase-sorted aligned traces.

        X-axis: Time (relative to inspiratory onset alignment point)
        Y-axis: Sweep number (sorted by stim phase)
        Z-axis: Amplitude

        Args:
            ax: Matplotlib 3D axis to plot on (must have projection='3d')
            kept_sweeps: List of sweep indices to include
        """
        import numpy as np
        from matplotlib import cm

        st = self.window.state

        # Get per-sweep data
        if st.analyze_chan not in st.sweeps:
            ax.text2D(0.5, 0.5, 'No analysis channel data available',
                     ha='center', va='center', transform=ax.transAxes)
            return

        sweep_data = st.sweeps[st.analyze_chan]
        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        # Downsample for efficiency
        DS_TARGET_HZ = 200.0
        ds_step = max(1, int(round(sr_hz / DS_TARGET_HZ)))

        # Collect traces with phase information (same as 2D version)
        traces_with_phase = []

        print(f"[3D] Processing {len(kept_sweeps)} sweeps for phase-sorted traces...")

        for s in kept_sweeps:
            breath_events = st.breath_by_sweep.get(s)
            if breath_events is None or len(breath_events) == 0:
                print(f"[3D] Sweep {s}: No breath events")
                continue

            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                print(f"[3D] Sweep {s}: No stim spans")
                continue

            stim_onset = spans[0][0]  # Pulse onset time (seconds)

            # Get breath events
            onsets = breath_events.get("onsets", [])
            offsets = breath_events.get("offsets", [])  # Inspiratory offsets (end of inspiration)
            peaks = st.peaks_by_sweep.get(s, np.array([]))  # Inspiratory peaks
            expoffs = breath_events.get("expoffs", [])  # Expiratory offsets

            if len(onsets) == 0 or len(offsets) == 0 or len(peaks) == 0 or len(expoffs) == 0:
                print(f"[3D] Sweep {s}: No onsets/offsets/peaks/expoffs")
                continue

            # Get this sweep's data
            if s >= sweep_data.shape[1]:
                print(f"[3D] Sweep {s}: Sweep index out of bounds")
                continue
            y_sweep = sweep_data[:, s]
            t_sweep = np.arange(len(y_sweep)) / sr_hz

            # Find inspiratory offset BEFORE stim (alignment point)
            offset_times = t_sweep[offsets]
            offset_before_mask = offset_times < stim_onset
            offset_before_stim = offset_times[offset_before_mask]

            if len(offset_before_stim) == 0:
                print(f"[3D] Sweep {s}: No inspiratory offset before stim")
                continue

            t_align = float(offset_before_stim[-1])  # Last inspiratory offset before stim
            offset_idx_align = int(offsets[np.sum(offset_before_mask) - 1])

            # Find first inspiratory peak AFTER stim
            peak_times = t_sweep[peaks]
            peak_after_mask = peak_times > stim_onset
            peak_after_stim = peak_times[peak_after_mask]

            if len(peak_after_stim) == 0:
                print(f"[3D] Sweep {s}: No inspiratory peak after stim")
                continue

            first_peak_after_stim_time = peak_after_stim[0]

            # Find first expiratory offset AFTER the first inspiratory peak after stim
            expoff_times = t_sweep[expoffs]
            expoff_after_peak_mask = expoff_times > first_peak_after_stim_time
            expoff_after_peak = expoff_times[expoff_after_peak_mask]

            if len(expoff_after_peak) == 0:
                print(f"[3D] Sweep {s}: No expiratory offset after peak")
                continue

            expoff_indices_after = np.where(expoff_after_peak_mask)[0]
            expoff_idx_end = int(expoffs[expoff_indices_after[0]])  # First expiratory offset after peak

            # Calculate phase: time from alignment point to stim
            phase = float(stim_onset - t_align)  # Time in seconds

            # Extract segment from inspiratory offset to expiratory offset after peak
            idx_start = int(offset_idx_align)
            idx_end = int(expoff_idx_end)

            if idx_start >= idx_end or idx_end > len(y_sweep):
                print(f"[3D] Sweep {s}: Invalid index range")
                continue

            t_segment = t_sweep[idx_start:idx_end:ds_step] - t_align  # Relative to alignment
            y_segment = y_sweep[idx_start:idx_end:ds_step]
            stim_t_aligned = stim_onset - t_align

            # Determine if breath before stim is eupneic or sniffing
            onset_times = t_sweep[onsets]
            onset_before_mask = onset_times < stim_onset
            is_sniffing_before = False
            if np.any(onset_before_mask):
                breath_idx_before = int(np.sum(onset_before_mask)) - 1
                is_sniffing_before = self._is_breath_sniffing(s, breath_idx_before, onsets)

            if len(t_segment) > 1:
                traces_with_phase.append({
                    'sweep': s,
                    'phase': phase,
                    't': t_segment,
                    'y': y_segment,
                    'stim_t': stim_t_aligned,
                    'is_sniffing': is_sniffing_before,
                })
                breath_type = "sniff" if is_sniffing_before else "eupnea"
                print(f"[3D] Sweep {s}: OK (phase={phase:.3f}s, {breath_type}, {len(t_segment)} points)")

        print(f"[3D] Collected {len(traces_with_phase)} traces with phase information")

        if not traces_with_phase:
            ax.text2D(0.5, 0.5, 'No data available for phase-sorted traces',
                     ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by breath type FIRST (eupnea=0, sniffing=1), then by phase
        traces_with_phase.sort(key=lambda x: (x['is_sniffing'], x['phase']))

        # Count eupnea vs sniffing for dividing line
        n_eupnea = sum(1 for tr in traces_with_phase if not tr['is_sniffing'])
        n_sniff = len(traces_with_phase) - n_eupnea
        print(f"[3D] Sorted: {n_eupnea} eupnea, {n_sniff} sniffing")

        # Calculate global z-range across all traces for uniform stim marker height
        all_y_values = np.concatenate([trace['y'] for trace in traces_with_phase])
        global_z_min = all_y_values.min()
        global_z_max = all_y_values.max()

        # Plot each trace in 3D (color by breath type: green=eupnea, purple=sniffing)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for i, trace in enumerate(traces_with_phase):
            # Color by breath type (matching main plotting window)
            trace_color = 'purple' if trace['is_sniffing'] else 'green'

            # X = time, Y = sweep index (sorted), Z = amplitude
            ax.plot(trace['t'],
                   np.ones_like(trace['t']) * i,  # Y = sweep index
                   trace['y'],  # Z = amplitude
                   color=trace_color, linewidth=1.0, alpha=0.8)

            # Mark stim time with transparent blue fill (25ms width, no edge lines)
            stim_t = trace['stim_t']
            stim_width = 0.025  # 25ms in seconds
            t_stim_start = stim_t
            t_stim_end = stim_t + stim_width

            # Create transparent blue curtain spanning full amplitude range
            verts = [
                [(t_stim_start, i, global_z_min),
                 (t_stim_end, i, global_z_min),
                 (t_stim_end, i, global_z_max),
                 (t_stim_start, i, global_z_max)]
            ]
            poly = Poly3DCollection(verts, alpha=0.15, facecolor='blue', edgecolor='none')
            ax.add_collection3d(poly)

            # Add color-coded line at t=0, amplitude=0 for breath type
            # Green for eupnea, purple for sniffing
            breath_color = 'purple' if trace['is_sniffing'] else 'green'
            ax.plot([0, 0], [i, i], [0, 0], color=breath_color, marker='o', markersize=3, alpha=0.7)

        ax.set_xlabel('Time (s) from insp offset', fontsize=9)
        ax.set_ylabel('Sweep # (sorted by breath type, then phase)', fontsize=9)
        ax.set_zlabel('Amplitude (mV)', fontsize=9)
        ax.set_title(f'Phase-Sorted Aligned Traces (n={len(traces_with_phase)})\nGreen=Eupnea, Purple=Sniffing',
                    fontsize=11, fontweight='bold')

    def _plot_pulse_3d_stim_aligned(self, ax, kept_sweeps: list):
        """
        Plot 3D phase-sorted aligned traces - ALIGNED TO STIMULUS instead of inspiratory offset.

        Same trace cutout (insp offset to next peak) but centered at t=0 on stimulus.

        Args:
            ax: Matplotlib 3D axis to plot on (must have projection='3d')
            kept_sweeps: List of sweep indices to include
        """
        import numpy as np
        from matplotlib import cm

        st = self.window.state

        # Get per-sweep data
        if st.analyze_chan not in st.sweeps:
            ax.text2D(0.5, 0.5, 'No analysis channel data available',
                     ha='center', va='center', transform=ax.transAxes)
            return

        sweep_data = st.sweeps[st.analyze_chan]
        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        # Downsample for efficiency
        DS_TARGET_HZ = 200.0
        ds_step = max(1, int(round(sr_hz / DS_TARGET_HZ)))

        # Collect traces (same as before but will re-align to stim)
        traces_with_phase = []

        print(f"[3D Stim-Aligned] Processing {len(kept_sweeps)} sweeps...")

        for s in kept_sweeps:
            breath_events = st.breath_by_sweep.get(s)
            if breath_events is None or len(breath_events) == 0:
                continue

            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                continue

            stim_onset = spans[0][0]

            onsets = breath_events.get("onsets", [])
            offsets = breath_events.get("offsets", [])
            peaks = st.peaks_by_sweep.get(s, np.array([]))
            expoffs = breath_events.get("expoffs", [])

            if len(onsets) == 0 or len(offsets) == 0 or len(peaks) == 0 or len(expoffs) == 0:
                continue

            if s >= sweep_data.shape[1]:
                continue
            y_sweep = sweep_data[:, s]
            t_sweep = np.arange(len(y_sweep)) / sr_hz

            # Find inspiratory offset BEFORE stim
            offset_times = t_sweep[offsets]
            offset_before_mask = offset_times < stim_onset
            offset_before_stim = offset_times[offset_before_mask]

            if len(offset_before_stim) == 0:
                continue

            t_offset = float(offset_before_stim[-1])
            offset_idx_align = int(offsets[np.sum(offset_before_mask) - 1])

            # Find first inspiratory peak AFTER stim
            peak_times = t_sweep[peaks]
            peak_after_mask = peak_times > stim_onset
            peak_after_stim = peak_times[peak_after_mask]

            if len(peak_after_stim) == 0:
                continue

            first_peak_after_stim_time = peak_after_stim[0]

            # Find first expiratory offset AFTER the first inspiratory peak after stim
            expoff_times = t_sweep[expoffs]
            expoff_after_peak_mask = expoff_times > first_peak_after_stim_time
            expoff_after_peak = expoff_times[expoff_after_peak_mask]

            if len(expoff_after_peak) == 0:
                continue

            expoff_indices_after = np.where(expoff_after_peak_mask)[0]
            expoff_idx_end = int(expoffs[expoff_indices_after[0]])

            # Calculate phase
            phase = float(stim_onset - t_offset)

            # Extract segment from inspiratory offset to expiratory offset after peak
            idx_start = int(offset_idx_align)
            idx_end = int(expoff_idx_end)

            if idx_start >= idx_end or idx_end > len(y_sweep):
                continue

            # Align to STIMULUS (t=0 at stim) instead of inspiratory offset
            t_segment = t_sweep[idx_start:idx_end:ds_step] - stim_onset  # Relative to STIM
            y_segment = y_sweep[idx_start:idx_end:ds_step]
            offset_t_aligned = t_offset - stim_onset  # Where offset is relative to stim

            # Determine if breath before stim is eupneic or sniffing
            onset_times = t_sweep[onsets]
            onset_before_mask = onset_times < stim_onset
            is_sniffing_before = False
            if np.any(onset_before_mask):
                breath_idx_before = int(np.sum(onset_before_mask)) - 1
                is_sniffing_before = self._is_breath_sniffing(s, breath_idx_before, onsets)

            if len(t_segment) > 1:
                traces_with_phase.append({
                    'sweep': s,
                    'phase': phase,
                    't': t_segment,
                    'y': y_segment,
                    'offset_t': offset_t_aligned,  # For reference
                    'is_sniffing': is_sniffing_before,
                })

        print(f"[3D Stim-Aligned] Collected {len(traces_with_phase)} traces")

        if not traces_with_phase:
            ax.text2D(0.5, 0.5, 'No data available',
                     ha='center', va='center', transform=ax.transAxes)
            return

        # Sort by breath type FIRST (eupnea=0, sniffing=1), then by phase
        traces_with_phase.sort(key=lambda x: (x['is_sniffing'], x['phase']))

        # Count eupnea vs sniffing
        n_eupnea = sum(1 for tr in traces_with_phase if not tr['is_sniffing'])
        n_sniff = len(traces_with_phase) - n_eupnea
        print(f"[3D Stim-Aligned] Sorted: {n_eupnea} eupnea, {n_sniff} sniffing")

        # Calculate global z-range across all traces for uniform stim marker height
        all_y_values = np.concatenate([trace['y'] for trace in traces_with_phase])
        global_z_min = all_y_values.min()
        global_z_max = all_y_values.max()

        # Plot each trace (color by breath type: green=eupnea, purple=sniffing)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for i, trace in enumerate(traces_with_phase):
            # Color by breath type (matching main plotting window)
            trace_color = 'purple' if trace['is_sniffing'] else 'green'

            ax.plot(trace['t'],
                   np.ones_like(trace['t']) * i,
                   trace['y'],
                   color=trace_color, linewidth=1.0, alpha=0.8)

            # Mark stim time at t=0 with transparent blue fill (25ms width, no edge lines)
            stim_width = 0.025
            t_stim_start = 0.0
            t_stim_end = stim_width

            # Create transparent blue curtain spanning full amplitude range
            verts = [
                [(t_stim_start, i, global_z_min),
                 (t_stim_end, i, global_z_min),
                 (t_stim_end, i, global_z_max),
                 (t_stim_start, i, global_z_max)]
            ]
            poly = Poly3DCollection(verts, alpha=0.15, facecolor='blue', edgecolor='none')
            ax.add_collection3d(poly)

            # Add color-coded line at t=0, amplitude=0 for breath type
            # Green for eupnea, purple for sniffing
            breath_color = 'purple' if trace['is_sniffing'] else 'green'
            ax.plot([0, 0], [i, i], [0, 0], color=breath_color, marker='o', markersize=3, alpha=0.7)

        ax.set_xlabel('Time (s) from stimulus', fontsize=9)
        ax.set_ylabel('Sweep # (sorted by breath type, then phase)', fontsize=9)
        ax.set_zlabel('Amplitude (mV)', fontsize=9)
        ax.set_title(f'Phase-Sorted Traces - Stimulus Aligned (n={len(traces_with_phase)})\nGreen=Eupnea, Purple=Sniffing',
                    fontsize=11, fontweight='bold')

    def _plot_pulse_probability_curve(self, ax_all, ax_eupnea, ax_sniff, kept_sweeps: list):
        """
        Plot probability of next breath occurring as a function of time after previous breath.
        3-panel layout: All breaths, Eupnea only, Sniffing only.

        Uses all breaths EXCEPT those before and after stimulation to calculate baseline
        probability curve.

        Args:
            ax_all: Matplotlib axis for all breaths
            ax_eupnea: Matplotlib axis for eupnea breaths only
            ax_sniff: Matplotlib axis for sniffing breaths only
            kept_sweeps: List of sweep indices to include
        """
        import numpy as np

        st = self.window.state

        # Collect inter-breath intervals separated by breath type (excluding stim-affected breaths)
        intervals_all = []
        intervals_eupnea = []
        intervals_sniff = []

        print(f"[Probability] Processing {len(kept_sweeps)} sweeps for inter-breath intervals...")

        for s in kept_sweeps:
            breath_events = st.breath_by_sweep.get(s)
            if breath_events is None or len(breath_events) == 0:
                continue

            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                continue

            stim_onset = spans[0][0]  # Pulse onset time (seconds)

            # Get inspiratory offsets
            offsets = breath_events.get("offsets", [])
            if len(offsets) < 3:  # Need at least 3 offsets to exclude stim-affected ones
                continue

            # Get sweep data for time conversion
            if s >= st.sweeps[st.analyze_chan].shape[1]:
                continue
            y_sweep = st.sweeps[st.analyze_chan][:, s]
            sr_hz = st.sr_hz if st.sr_hz else 1000.0
            t_sweep = np.arange(len(y_sweep)) / sr_hz

            offset_times = t_sweep[offsets]

            # Find offset before and after stim
            offset_before_mask = offset_times < stim_onset
            offset_after_mask = offset_times > stim_onset

            if not offset_before_mask.any() or not offset_after_mask.any():
                continue

            offset_before_idx = np.where(offset_before_mask)[0][-1]  # Last before stim
            offset_after_idx = np.where(offset_after_mask)[0][0]  # First after stim

            # Get onsets for breath classification
            onsets = breath_events.get("onsets", [])
            if len(onsets) == 0:
                continue

            # Calculate intervals for all non-stim-affected breaths
            for i in range(len(offset_times) - 1):
                # Skip the interval involving the breath before or after stim
                if i == offset_before_idx or i == offset_after_idx:
                    continue
                if i + 1 == offset_before_idx or i + 1 == offset_after_idx:
                    continue

                interval = offset_times[i + 1] - offset_times[i]
                if 0.1 < interval < 10.0:  # Sanity check (100ms to 10s)
                    intervals_all.append(interval)

                    # Classify by breath type (based on the breath at index i)
                    is_sniffing = self._is_breath_sniffing(s, i, onsets)
                    if is_sniffing:
                        intervals_sniff.append(interval)
                    else:
                        intervals_eupnea.append(interval)

        print(f"[Probability] Collected {len(intervals_all)} inter-breath intervals (Eupnea: {len(intervals_eupnea)}, Sniff: {len(intervals_sniff)})")

        # Collect stimulated breath timing data separated by breath type
        stim_breath_data_all = []  # (stim_time_from_prev_offset, breath_time_from_prev_offset, is_sniffing)
        stim_breath_data_eupnea = []  # (stim_time, breath_time, is_sniffing=False)
        stim_breath_data_sniff = []  # (stim_time, breath_time, is_sniffing=True)

        for s in kept_sweeps:
            breath_events = st.breath_by_sweep.get(s)
            if breath_events is None or len(breath_events) == 0:
                continue

            spans = st.stim_spans_by_sweep.get(s, [])
            if not spans:
                continue

            stim_onset = spans[0][0]

            # Get inspiratory offsets
            offsets = breath_events.get("offsets", [])
            if len(offsets) < 2:
                continue

            # Get sweep data
            if s >= st.sweeps[st.analyze_chan].shape[1]:
                continue
            y_sweep = st.sweeps[st.analyze_chan][:, s]
            sr_hz = st.sr_hz if st.sr_hz else 1000.0
            t_sweep = np.arange(len(y_sweep)) / sr_hz

            offset_times = t_sweep[offsets]

            # Find offset before stim
            offset_before_mask = offset_times < stim_onset
            if not offset_before_mask.any():
                continue

            offset_before_idx = np.where(offset_before_mask)[0][-1]
            t_prev_offset = offset_times[offset_before_idx]

            # Find offset after stim
            offset_after_mask = offset_times > stim_onset
            if not offset_after_mask.any():
                continue

            offset_after_idx = np.where(offset_after_mask)[0][0]
            t_next_offset = offset_times[offset_after_idx]

            # Calculate timing relative to previous offset
            stim_time = stim_onset - t_prev_offset
            breath_time = t_next_offset - t_prev_offset

            if 0 < stim_time < 10.0 and 0 < breath_time < 10.0:
                # Classify by breath type (based on breath before stim)
                onsets = breath_events.get("onsets", [])
                is_sniffing = False
                if len(onsets) > 0:
                    onset_times_sweep = t_sweep[onsets]
                    onset_before_mask_sweep = onset_times_sweep < stim_onset
                    if np.any(onset_before_mask_sweep):
                        breath_idx_before = int(np.sum(onset_before_mask_sweep)) - 1
                        is_sniffing = self._is_breath_sniffing(s, breath_idx_before, onsets)

                stim_breath_data_all.append((stim_time, breath_time, is_sniffing))

                if is_sniffing:
                    stim_breath_data_sniff.append((stim_time, breath_time, is_sniffing))
                else:
                    stim_breath_data_eupnea.append((stim_time, breath_time, is_sniffing))

        print(f"[Probability] Collected {len(stim_breath_data_all)} stimulated breath timings (Eupnea: {len(stim_breath_data_eupnea)}, Sniff: {len(stim_breath_data_sniff)})")

        # Determine shared bin edges and limits
        all_intervals = intervals_all + intervals_eupnea + intervals_sniff
        if len(all_intervals) < 10:
            for ax in [ax_all, ax_eupnea, ax_sniff]:
                ax.text(0.5, 0.5, 'Insufficient data',
                       ha='center', va='center', transform=ax.transAxes)
            return

        all_intervals_arr = np.array(all_intervals)
        shared_bin_edges = np.linspace(0, np.percentile(all_intervals_arr, 99), 50)
        bin_width = shared_bin_edges[1] - shared_bin_edges[0]
        bin_centers = (shared_bin_edges[:-1] + shared_bin_edges[1:]) / 2

        # Calculate histograms (counts, not density)
        counts_all, _ = np.histogram(intervals_all, bins=shared_bin_edges)
        counts_eupnea, _ = np.histogram(intervals_eupnea, bins=shared_bin_edges)
        counts_sniff, _ = np.histogram(intervals_sniff, bins=shared_bin_edges)

        # Sample counts for labels
        n_all = len(intervals_all)
        n_eupnea = len(intervals_eupnea)
        n_sniff = len(intervals_sniff)

        # Find max count for shared y-axis
        max_count = max(counts_all.max(),
                       counts_eupnea.max() if len(counts_eupnea) > 0 else 0,
                       counts_sniff.max() if len(counts_sniff) > 0 else 0)

        # Helper function to plot overlaid panel (panel 1)
        def plot_all_panel(ax):
            """Plot panel 1 with all three histograms overlaid."""
            # Plot all breaths as black line (no fill)
            ax.plot(bin_centers, counts_all, 'k-', linewidth=2, label=f'All (n={n_all})', zorder=3)

            # Plot eupnea as green filled curve
            ax.fill_between(bin_centers, 0, counts_eupnea, color='green', alpha=0.3, label=f'Eupnea (n={n_eupnea})', zorder=2)
            ax.plot(bin_centers, counts_eupnea, color='green', linewidth=1, alpha=0.7, zorder=2)

            # Plot sniffing as purple filled curve
            ax.fill_between(bin_centers, 0, counts_sniff, color='purple', alpha=0.3, label=f'Sniffing (n={n_sniff})', zorder=1)
            ax.plot(bin_centers, counts_sniff, color='purple', linewidth=1, alpha=0.7, zorder=1)

            # Cumulative probabilities on secondary axis
            ax2 = ax.twinx()
            cumul_all = np.cumsum(counts_all) / counts_all.sum() if counts_all.sum() > 0 else np.zeros_like(counts_all)
            cumul_eupnea = np.cumsum(counts_eupnea) / counts_eupnea.sum() if counts_eupnea.sum() > 0 else np.zeros_like(counts_eupnea)
            cumul_sniff = np.cumsum(counts_sniff) / counts_sniff.sum() if counts_sniff.sum() > 0 else np.zeros_like(counts_sniff)

            ax2.plot(bin_centers, cumul_all, 'k--', linewidth=1.5, label='Cumul All', alpha=0.7)
            ax2.plot(bin_centers, cumul_eupnea, color='green', linestyle='--', linewidth=1.5, label='Cumul Eupnea', alpha=0.7)
            ax2.plot(bin_centers, cumul_sniff, color='purple', linestyle='--', linewidth=1.5, label='Cumul Sniffing', alpha=0.7)

            ax2.set_ylabel('Cumulative Probability', fontsize=8)
            ax2.set_ylim(0, 1.0)
            ax2.tick_params(labelsize=7)
            ax2.grid(False)

            # Create histogram for post-stim breath intervals (separated by breath type)
            post_stim_times_eupnea = [bt for st, bt, is_sniff in stim_breath_data_all if not is_sniff]
            post_stim_times_sniff = [bt for st, bt, is_sniff in stim_breath_data_all if is_sniff]

            if post_stim_times_eupnea:
                counts_post_eupnea, _ = np.histogram(post_stim_times_eupnea, bins=shared_bin_edges)
                ax.fill_between(bin_centers, 0, counts_post_eupnea, color='green',
                               alpha=0.15, edgecolor='green', linewidth=1, linestyle=':', label='Post-Stim Eupnea')

            if post_stim_times_sniff:
                counts_post_sniff, _ = np.histogram(post_stim_times_sniff, bins=shared_bin_edges)
                ax.fill_between(bin_centers, 0, counts_post_sniff, color='purple',
                               alpha=0.15, edgecolor='purple', linewidth=1, linestyle=':', label='Post-Stim Sniff')

            # Plot stim markers color-coded and sorted by breath type
            if stim_breath_data_all:
                # Sort by breath type first (eupnea=0, sniff=1), then by stim_time
                stim_sorted = sorted(stim_breath_data_all, key=lambda x: (x[2], x[0]))

                # Calculate spacing to fit within plot
                n_markers = len(stim_sorted)
                available_height = max_count * 0.4  # Use 40% of max for markers
                y_spacing = available_height / max(n_markers, 1)
                y_base = max_count * 1.05

                for i, (stim_time, breath_time, is_sniffing) in enumerate(stim_sorted):
                    y_pos = y_base + (i * y_spacing)
                    breath_color = 'purple' if is_sniffing else 'green'

                    ax.plot(stim_time, y_pos, 'o', color='blue', markersize=3, alpha=0.7, zorder=10)
                    ax.plot(breath_time, y_pos, 'o', color=breath_color, markersize=3, alpha=0.7, zorder=10)
                    ax.plot([stim_time, breath_time], [y_pos, y_pos], '-', color=breath_color,
                           linewidth=0.6, alpha=0.5, zorder=9)

            ax.set_xlabel('Time since previous insp offset (s)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_title('Baseline Breath Probability - All Breaths', fontsize=9, fontweight='bold')
            ax.legend(loc='upper left', fontsize=6)
            ax.set_ylim(0, max_count * 1.6)  # Increased to fit markers
            ax.set_xlim(shared_bin_edges[0], shared_bin_edges[-1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=7)

        # Helper function for single-type panels (panels 2 and 3)
        def plot_single_panel(ax, intervals, stim_data, title, color):
            """Plot a single breath type panel."""
            if len(intervals) < 10:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
                return

            # Calculate counts
            counts, _ = np.histogram(intervals, bins=shared_bin_edges)
            n_intervals = len(intervals)

            # Plot as filled transparent curve (matching panel 1 style)
            ax.fill_between(bin_centers, 0, counts, color=color, alpha=0.3, label=f'{title.split("-")[1].strip()} (n={n_intervals})')
            ax.plot(bin_centers, counts, color=color, linewidth=1.5, alpha=0.7)

            # Cumulative probability
            ax2 = ax.twinx()
            cumul = np.cumsum(counts) / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
            ax2.plot(bin_centers, cumul, color=color, linestyle='--', linewidth=2)
            ax2.set_ylabel('Cumulative Probability', fontsize=8, color=color)
            ax2.tick_params(axis='y', labelcolor=color, labelsize=7)
            ax2.set_ylim(0, 1.0)
            ax2.grid(False)

            # Create histogram for post-stim breath intervals
            if stim_data:
                post_stim_times = [bt for st, bt, is_sniff in stim_data]
                counts_post, _ = np.histogram(post_stim_times, bins=shared_bin_edges)
                n_post = len(post_stim_times)
                ax.fill_between(bin_centers, 0, counts_post, color=color,
                               alpha=0.15, edgecolor=color, linewidth=1, linestyle=':', label=f'Post-Stim (n={n_post})')

            # Plot stim markers sorted by stim time
            if stim_data:
                stim_sorted = sorted(stim_data, key=lambda x: x[0])

                # Calculate spacing to fit within plot
                n_markers = len(stim_sorted)
                available_height = max_count * 0.4
                y_spacing = available_height / max(n_markers, 1)
                y_base = max_count * 1.05

                for i, (stim_time, breath_time, is_sniffing) in enumerate(stim_sorted):
                    y_pos = y_base + (i * y_spacing)
                    ax.plot(stim_time, y_pos, 'o', color='blue', markersize=3, alpha=0.7, zorder=10)
                    ax.plot(breath_time, y_pos, 'o', color=color, markersize=3, alpha=0.7, zorder=10)
                    ax.plot([stim_time, breath_time], [y_pos, y_pos], '-', color=color,
                           linewidth=0.6, alpha=0.5, zorder=9)

            # Stats
            intervals_arr = np.array(intervals)
            mean_int = float(np.mean(intervals_arr))
            std_int = float(np.std(intervals_arr))
            stats_text = (f'n={n_intervals}\n'
                         f'Mean: {mean_int:.2f} s\n'
                         f'SD: {std_int:.2f} s\n'
                         f'Stim: n={len(stim_data)}')
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=7, va='top', ha='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            ax.set_xlabel('Time since previous insp offset (s)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.set_ylim(0, max_count * 1.5)
            ax.set_xlim(shared_bin_edges[0], shared_bin_edges[-1])
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(labelsize=7)
            if ax.get_legend_handles_labels()[0]:  # Only show legend if there are items
                ax.legend(loc='upper left', fontsize=6)

        # Plot all three panels
        plot_all_panel(ax_all)
        plot_single_panel(ax_eupnea, intervals_eupnea, stim_breath_data_eupnea,
                         'Baseline Breath Probability - Eupnea Only', 'green')
        plot_single_panel(ax_sniff, intervals_sniff, stim_breath_data_sniff,
                         'Baseline Breath Probability - Sniffing Only', 'purple')

    def _generate_pulse_cta_test_figure(self, kept_sweeps: list):
        """
        Generate a standalone test figure with 3-panel CTA overlay plot.

        This is a temporary helper for Phase 1 testing before full integration.

        Args:
            kept_sweeps: List of sweep indices to include

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, (ax_all, ax_eupnea, ax_sniff) = plt.subplots(1, 3, figsize=(18, 5))
        self._plot_pulse_cta_overlay(ax_all, ax_eupnea, ax_sniff, kept_sweeps, global_s0=None)
        fig.tight_layout()

        return fig

    def _generate_pulse_2d_test_figure(self, kept_sweeps: list):
        """
        Generate a standalone test figure with just the 2D phase-sorted traces.

        Args:
            kept_sweeps: List of sweep indices to include

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        self._plot_pulse_2d_sorted_traces(ax, kept_sweeps)
        fig.tight_layout()

        return fig

    def _generate_pulse_3d_test_figure(self, kept_sweeps: list):
        """
        Generate a standalone test figure with the 3D phase-sorted traces.

        Args:
            kept_sweeps: List of sweep indices to include

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._plot_pulse_3d_sorted_traces(ax, kept_sweeps)
        fig.tight_layout()

        return fig

    def _generate_pulse_3d_stim_aligned_test_figure(self, kept_sweeps: list):
        """
        Generate a standalone test figure with the 3D phase-sorted traces (stimulus-aligned).

        Args:
            kept_sweeps: List of sweep indices to include

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        self._plot_pulse_3d_stim_aligned(ax, kept_sweeps)
        fig.tight_layout()

        return fig

    def _generate_pulse_probability_test_figure(self, kept_sweeps: list):
        """
        Generate a standalone test figure with 3-panel baseline breath probability curves.

        Args:
            kept_sweeps: List of sweep indices to include

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        fig, (ax_all, ax_eupnea, ax_sniff) = plt.subplots(1, 3, figsize=(18, 6))
        self._plot_pulse_probability_curve(ax_all, ax_eupnea, ax_sniff, kept_sweeps)
        fig.tight_layout()

        return fig

    def _save_pulse_analysis_pdf(self, out_path, kept_sweeps: list):
        """
        Generate and save pulse analysis PDF with 4 pages:
        - Page 1: CTA overlay (3-panel)
        - Page 2: 3D phase-sorted traces (offset-aligned)
        - Page 3: 3D phase-sorted traces (stim-aligned)
        - Page 4: Probability curves (3-panel)

        Args:
            out_path: Path to save PDF
            kept_sweeps: List of sweep indices to include
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        print(f"[Pulse PDF] Generating pulse analysis PDF for {len(kept_sweeps)} sweeps...")

        # Generate all 4 figures
        try:
            print("[Pulse PDF] Generating CTA figure...")
            fig_cta = self._generate_pulse_cta_test_figure(kept_sweeps)

            print("[Pulse PDF] Generating 3D offset-aligned figure...")
            fig_3d_offset = self._generate_pulse_3d_test_figure(kept_sweeps)

            print("[Pulse PDF] Generating 3D stim-aligned figure...")
            fig_3d_stim = self._generate_pulse_3d_stim_aligned_test_figure(kept_sweeps)

            print("[Pulse PDF] Generating probability curves...")
            fig_prob = self._generate_pulse_probability_test_figure(kept_sweeps)

            # Save to PDF
            with PdfPages(out_path) as pdf:
                pdf.savefig(fig_cta, dpi=150)
                pdf.savefig(fig_3d_offset, dpi=150)
                pdf.savefig(fig_3d_stim, dpi=150)
                pdf.savefig(fig_prob, dpi=150)

            # Close figures
            plt.close(fig_cta)
            plt.close(fig_3d_offset)
            plt.close(fig_3d_stim)
            plt.close(fig_prob)

            print(f"[Pulse PDF] ✓ Pulse analysis PDF saved to {out_path.name}")

        except Exception as e:
            print(f"[Pulse PDF] ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _save_metrics_summary_pdf(
        self,
        out_path,
        t_ds_csv: np.ndarray,
        y2_ds_by_key: dict,
        keys_for_csv: list[str],
        label_by_key: dict[str, str],
        stim_zero: float | None,
        stim_dur: float | None,
        return_figures: bool = False,
         ):
        """
        Build a three-page PDF (or return figures for preview).

        If return_figures=True: Returns (fig1, fig2, fig3) tuple instead of saving.
        If return_figures=False: Saves PDF to out_path.

        • Page 1: rows = metrics, cols = [all sweeps | mean±SEM | histograms] using RAW data
        • Page 2: same layout, using NORMALIZED data (time-based, per-sweep baseline)
        • Page 3: same layout, using NORMALIZED data (eupnea-based, pooled baseline)
        • NEW: overlay orange star markers at times where sighs occurred (first two columns)
            and at x = metric value of sigh breaths (histogram column).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        st = self.window.state
        n_sweeps_total = next(iter(y2_ds_by_key.values())).shape[1] if y2_ds_by_key else 0
        M = len(t_ds_csv)
        have_stim = (stim_zero is not None and stim_dur is not None)

        # --- normalization knobs (match exporter) ---
        NORM_BASELINE_WINDOW_S = float(getattr(self, "_norm_window_s", 10.0))
        EPS_BASE = 1e-12

        # ---------- Helpers ----------
        def _per_sweep_baseline_time(A_ds: np.ndarray) -> np.ndarray:
            b = np.full((A_ds.shape[1],), np.nan, dtype=float)
            mask_pre  = (t_ds_csv >= -NORM_BASELINE_WINDOW_S) & (t_ds_csv < 0.0)
            mask_post = (t_ds_csv >=  0.0) & (t_ds_csv <= NORM_BASELINE_WINDOW_S)
            for s in range(A_ds.shape[1]):
                col = A_ds[:, s]
                vals = col[mask_pre]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    vals = col[mask_post]
                    vals = vals[np.isfinite(vals)]
                if vals.size:
                    b[s] = float(np.mean(vals))
            return b

        def _normalize_matrix(A_ds: np.ndarray, b: np.ndarray) -> np.ndarray:
            out = np.full_like(A_ds, np.nan)
            for s in range(A_ds.shape[1]):
                bs = b[s]
                if np.isfinite(bs) and abs(bs) > EPS_BASE:
                    out[:, s] = A_ds[:, s] / bs
            return out

        def _plot_sigh_time_stars(ax, sigh_times_rel):
            """Overlay stars at sigh times near the top of a time-series axis without changing limits."""
            if not sigh_times_rel:
                return
            ylim = ax.get_ylim()
            y_star = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.plot(
                sigh_times_rel,
                [y_star] * len(sigh_times_rel),
                linestyle="none",
                marker="*",
                markersize=9,
                color="#ff9f1a",
                markeredgecolor="#a35400",
                alpha=0.95,
                zorder=6,
            )
            ax.set_ylim(*ylim)

        def _plot_hist_stars(ax, star_x_vals):
            """Overlay stars along the top of histogram axes at given x positions (metric values on sigh breaths)."""
            if not star_x_vals:
                return
            ylim = ax.get_ylim()
            y_star = ylim[1] - 0.06 * (ylim[1] - ylim[0])
            ax.scatter(
                np.asarray(star_x_vals, dtype=float),
                np.full(len(star_x_vals), y_star, dtype=float),
                marker="*",
                s=70,
                linewidths=0.9,
                edgecolors="#a35400",
                facecolors="#ff9f1a",
                zorder=6,
                clip_on=False,
            )
            ax.set_ylim(*ylim)

        # Build histogram pools AND collect (a) sigh times (for time-series stars)
        # and (b) metric values at sigh breaths (for histogram stars).
        def _build_hist_vals_raw_and_norm():
            hist_raw = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            hist_nrm = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_raw_by_key  = {k: [] for k in keys_for_csv}   # x-positions for stars on histogram (raw)
            sigh_vals_norm_by_key = {k: [] for k in keys_for_csv}   # x-positions for stars on histogram (norm)
            sigh_times_rel = []  # for time-series stars (seconds, relative to stim_zero if provided)

            kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                    if s not in getattr(st, "omitted_sweeps", set())]
            t0 = float(stim_zero) if stim_zero is not None else 0.0

            for s in kept:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks    = np.asarray(self.window.state.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br     = self.window.state.breath_by_sweep.get(s, None)
                if br is None and pks.size:
                    try:
                        br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                    except TypeError:
                        br = peakdet.compute_breath_events(y_proc, pks)
                    self.window.state.breath_by_sweep[s] = br
                if br is None:
                    br = {"onsets": np.array([], dtype=int)}

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - t0).astype(float)

                # Whether each breath interval [on[j], on[j+1]) contains a sigh peak
                sigh_idx = np.asarray(self.window.state.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                is_sigh_breath = np.zeros(len(mids), dtype=bool)
                if sigh_idx.size:
                    for j in range(len(mids)):
                        a = int(on[j]); b = int(on[j + 1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_breath[j] = True
                            # time for time-series star (use breath midpoint)
                            sigh_times_rel.append(float(st.t[int(mids[j])] - t0))

                # OPTIMIZATION: Use downsampled data instead of computing full-resolution traces
                # This avoids allocating huge arrays (can be 300M+ elements for long recordings)
                col_idx = kept.index(s)  # Column in y2_ds_by_key for this sweep

                # Per-sweep baselines for normalization (breath-based) - use downsampled data
                mask_pre_b  = (t_rel_all >= -NORM_BASELINE_WINDOW_S) & (t_rel_all < 0.0)
                mask_post_b = (t_rel_all >=  0.0) & (t_rel_all <= NORM_BASELINE_WINDOW_S)
                b_by_k = {}
                for k in keys_for_csv:
                    Y_ds = y2_ds_by_key.get(k, None)
                    if Y_ds is None or Y_ds.size == 0:
                        b_by_k[k] = np.nan
                        continue
                    # Extract baseline values from downsampled timeseries
                    baseline_t_rel = t_rel_all[mask_pre_b]
                    if baseline_t_rel.size == 0:
                        baseline_t_rel = t_rel_all[mask_post_b]
                    if baseline_t_rel.size > 0:
                        # Find closest downsampled indices for baseline breaths
                        ds_indices = np.array([np.argmin(np.abs(t_ds_csv - t)) for t in baseline_t_rel])
                        ds_indices = np.clip(ds_indices, 0, Y_ds.shape[0] - 1)
                        vals = Y_ds[ds_indices, col_idx]
                        vals = vals[np.isfinite(vals)]
                        b_by_k[k] = float(np.mean(vals)) if vals.size else np.nan
                    else:
                        b_by_k[k] = np.nan

                # Fill histogram pools & record sigh metric values - use downsampled data
                for j, (idx_mid, t_rel) in enumerate(zip(mids, t_rel_all)):
                    # Find closest downsampled index for this breath
                    ds_idx_j = np.argmin(np.abs(t_ds_csv - t_rel))
                    ds_idx_j = min(ds_idx_j, M - 1)  # Clip to valid range

                    for k in keys_for_csv:
                        Y_ds = y2_ds_by_key.get(k, None)
                        if Y_ds is None or Y_ds.size == 0:
                            continue
                        v = float(Y_ds[ds_idx_j, col_idx])
                        if not np.isfinite(v):
                            continue

                        # RAW pools
                        hist_raw[k]["all"].append(v)
                        if have_stim:
                            if t_rel < 0:
                                hist_raw[k]["baseline"].append(v)
                            elif 0.0 <= t_rel <= stim_dur:
                                hist_raw[k]["stim"].append(v)
                            else:
                                hist_raw[k]["post"].append(v)

                        # NORM pools
                        b = b_by_k.get(k, np.nan)
                        vn = (v / b) if (np.isfinite(b) and abs(b) > EPS_BASE) else np.nan
                        if np.isfinite(vn):
                            hist_nrm[k]["all"].append(vn)
                            if have_stim:
                                if t_rel < 0:
                                    hist_nrm[k]["baseline"].append(vn)
                                elif 0.0 <= t_rel <= stim_dur:
                                    hist_nrm[k]["stim"].append(vn)
                                else:
                                    hist_nrm[k]["post"].append(vn)

                        # SIGH stars (hist column): collect x = metric values on sigh breaths
                        if is_sigh_breath[j]:
                            sigh_vals_raw_by_key[k].append(v)
                            if np.isfinite(vn):
                                sigh_vals_norm_by_key[k].append(vn)

            # Dedup very-close times for cleaner star rows in time plots
            if len(st.t) > 1:
                dt = float(np.median(np.diff(st.t)))
            else:
                dt = (1.0 / float(st.sr_hz)) if getattr(st, "sr_hz", None) else 0.01
            eps = max(1e-9, 2.0 * dt)
            sigh_times_rel = sorted(sigh_times_rel)
            sigh_times_clean = []
            for tt in sigh_times_rel:
                if not sigh_times_clean or abs(tt - sigh_times_clean[-1]) > eps:
                    sigh_times_clean.append(tt)

            return hist_raw, hist_nrm, sigh_vals_raw_by_key, sigh_vals_norm_by_key, sigh_times_clean

        def _rowwise_mean_sem(Y2D: np.ndarray):
                    import numpy as np
                    # Y2D shape: (M, S) = (time, sweeps)
                    if Y2D is None or Y2D.size == 0:
                        M = 0
                        return np.array([]), np.array([])
                    M = Y2D.shape[0]
                    n = np.sum(np.isfinite(Y2D), axis=1)          # (M,)
                    mean = np.full(M, np.nan, float)
                    sem  = np.full(M, np.nan, float)

                    msk_mean = n > 0
                    if np.any(msk_mean):
                        mean[msk_mean] = np.nanmean(Y2D[msk_mean, :], axis=1)

                    msk_sem = n >= 2
                    if np.any(msk_sem):
                        std = np.nanstd(Y2D[msk_sem, :], axis=1, ddof=1)
                        sem[msk_sem] = std / np.sqrt(n[msk_sem])
                    return mean, sem

        def _plot_grid(fig, axes, Y_by_key, hist_vals, sigh_hist_vals_by_key, sigh_times_rel, title_suffix):
            """Render one page (grid) given series & histogram data dicts + star overlays."""
            nrows = max(1, len(keys_for_csv))
            for r, k in enumerate(keys_for_csv):
                label = label_by_key.get(k, k)
                is_bottom_row = (r == nrows - 1)
                is_top_row = (r == 0)

                # --- col 1: all sweeps overlaid ---
                ax1 = axes[r, 0]
                Y = Y_by_key.get(k, None)
                if Y is not None and Y.shape[0] == M:
                    for s in range(Y.shape[1]):
                        ax1.plot(t_ds_csv, Y[:, s], lw=0.8, alpha=0.45)
                if have_stim:
                    ax1.axvspan(0.0, stim_dur, color="#2E5090", alpha=0.25)
                    ax1.axvline(0.0, color="#2E5090", lw=1.0, alpha=0.8)
                    ax1.axvline(stim_dur, color="#2E5090", lw=1.0, alpha=0.8, ls="--")
                _plot_sigh_time_stars(ax1, sigh_times_rel)
                # Use less padding on top row to make room for figure title
                title_pad = 3 if is_top_row else 8
                ax1.set_title(f"{label} — all sweeps{title_suffix}", fontsize=9, pad=title_pad)
                ax1.set_ylabel(label, fontsize=8)  # Add y-label with metric name
                ax1.set_xlabel("Time (s)", fontsize=8)

                # --- col 2: mean ± SEM ---
                ax2 = axes[r, 1]
                if Y is not None and Y.shape[0] == M:
                    mean, sem = _rowwise_mean_sem(Y)
                    ax2.plot(t_ds_csv, mean, lw=1.8)
                    ax2.fill_between(t_ds_csv, mean - sem, mean + sem, alpha=0.25, linewidth=0)

                if have_stim:
                    ax2.axvspan(0.0, stim_dur, color="#2E5090", alpha=0.25)
                    ax2.axvline(0.0, color="#2E5090", lw=1.0, alpha=0.8)
                    ax2.axvline(stim_dur, color="#2E5090", lw=1.0, alpha=0.8, ls="--")
                _plot_sigh_time_stars(ax2, sigh_times_rel)
                ax2.set_title(f"{label} — mean ± SEM{title_suffix}", fontsize=9, pad=title_pad)
                ax2.set_ylabel(label, fontsize=8)  # Add y-label with metric name
                ax2.set_xlabel("Time (s)", fontsize=8)

                # --- col 3: line histograms (density) + stars at sigh metric values ---
                ax3 = axes[r, 2]
                groups = []
                for nm in ("all", "baseline", "stim", "post"):
                    vals = np.asarray(hist_vals[k][nm], dtype=float)
                    if vals.size:
                        groups.append(vals)

                if len(groups):
                    combined = np.concatenate(groups)
                    # Remove non-finite values before calculating bin edges
                    combined_finite = combined[np.isfinite(combined)]

                    # Only proceed if we have sufficient data for meaningful histogram
                    if combined_finite.size >= 2:
                        try:
                            # Check if there's sufficient variation in the data
                            if np.std(combined_finite) < 1e-10:
                                # All values are essentially identical - use fixed bins around that value
                                mean_val = np.mean(combined_finite)
                                edges = np.linspace(mean_val - 0.1, mean_val + 0.1, 10)
                            else:
                                edges = np.histogram_bin_edges(combined_finite, bins="auto")

                            centers = 0.5 * (edges[:-1] + edges[1:])

                            # Get total count for scaling sub-histograms
                            vals_all = np.asarray(hist_vals[k]["all"], dtype=float)
                            vals_all_finite = vals_all[np.isfinite(vals_all)]
                            n_total = len(vals_all_finite)

                            def _plot_histogram(vals, lbl, color, fill=False, line_style='-', scale_to_total=False, linewidth=1.5):
                                """
                                Plot histogram as line or filled curve.

                                Args:
                                    scale_to_total: If True, scale density by (n_vals / n_total) so that
                                                   the sub-histograms are proportional to the "All" histogram
                                    linewidth: Line width for the outline
                                """
                                vals = np.asarray(vals, dtype=float)
                                vals_finite = vals[np.isfinite(vals)]
                                if vals_finite.size == 0:
                                    return
                                try:
                                    # Add count to label
                                    label_with_count = f"{lbl} (n={len(vals_finite)})"

                                    # Calculate density
                                    dens, _ = np.histogram(vals_finite, bins=edges, density=True)

                                    # Scale density if requested (for baseline/stim/post relative to "All")
                                    if scale_to_total and n_total > 0:
                                        scale_factor = len(vals_finite) / n_total
                                        dens = dens * scale_factor

                                    if fill:
                                        # Filled transparent curve with solid outline
                                        ax3.fill_between(centers, 0, dens, color=color, alpha=0.3, label=label_with_count)
                                        ax3.plot(centers, dens, color=color, linewidth=linewidth, alpha=0.7, linestyle=line_style)
                                    else:
                                        # Line only (no fill)
                                        ax3.plot(centers, dens, color=color, linewidth=1.8, linestyle=line_style, label=label_with_count)
                                except (ValueError, RuntimeWarning) as e:
                                    # Skip plotting this line if histogram fails
                                    print(f"[Export] Warning: Could not plot histogram for {lbl}: {e}")
                                    pass

                            # Plot "All" as black line only (no fill) - standard density
                            _plot_histogram(hist_vals[k]["all"], "All", color='black', fill=False, scale_to_total=False)

                            # Plot baseline, post, then stim as filled histograms (stim last = on top)
                            # Stim gets thicker line to stand out
                            if have_stim:
                                _plot_histogram(hist_vals[k]["baseline"], "Baseline", color='gray', fill=True, scale_to_total=True, linewidth=1.5)
                                _plot_histogram(hist_vals[k]["post"], "Post", color='orange', fill=True, scale_to_total=True, linewidth=1.5)
                                _plot_histogram(hist_vals[k]["stim"], "Stim", color='blue', fill=True, scale_to_total=True, linewidth=2.0)
                        except Exception as e:
                            # If histogram generation fails entirely, show error message on plot
                            ax3.text(0.5, 0.5, f'Histogram error:\n{str(e)}',
                                   ha='center', va='center', transform=ax3.transAxes, fontsize=8)
                    else:
                        # Insufficient data for histogram
                        ax3.text(0.5, 0.5, 'Insufficient data\nfor histogram',
                               ha='center', va='center', transform=ax3.transAxes, fontsize=8)

                ax3.set_title(f"{label} — distribution (density){title_suffix}", fontsize=9, pad=title_pad)
                ax3.set_ylabel("Density", fontsize=8)
                ax3.set_xlabel(label, fontsize=8)  # Add metric name as x-label
                # Only show legend on top row to save space
                if len(ax3.lines) and is_top_row:
                    ax3.legend(loc="best", fontsize=7)

                # NEW: stars for histogram at sigh metric values (use "all" sigh values)
                _plot_hist_stars(ax3, sigh_hist_vals_by_key.get(k, []))

                # Reduce tick label font size for all axes
                ax1.tick_params(labelsize=7)
                ax2.tick_params(labelsize=7)
                ax3.tick_params(labelsize=7)

            fig.tight_layout()

        # ---------- Prepare both RAW and NORMALIZED datasets ----------
        y2_ds_by_key_norm = {}
        for k in keys_for_csv:
            Y = y2_ds_by_key.get(k, None)
            if Y is None or not Y.size:
                y2_ds_by_key_norm[k] = None
                continue
            b = _per_sweep_baseline_time(Y)
            y2_ds_by_key_norm[k] = _normalize_matrix(Y, b)

        # Prepare EUPNEA-BASED normalized datasets
        y2_ds_by_key_norm_eupnea = {}
        eupnea_baselines_by_key = {}  # Store computed baselines for histogram building
        eupnea_thresh = self.window.eupnea_freq_threshold  # Hz

        # OPTIMIZATION: Compute eupnea masks once per sweep, reuse for all metrics
        eupnea_masks_by_sweep = {}
        kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                if s not in getattr(st, "omitted_sweeps", set())]

        print(f"[PDF] Computing eupnea masks for {len(kept)} sweeps...")
        for s in kept:
            y_proc = self.window._get_processed_for(st.analyze_chan, s)
            pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
            br = st.breath_by_sweep.get(s, None)
            if br is None and pks.size:
                try:
                    br = peakdet.compute_breath_events(y_proc, pks, st.sr_hz)
                except TypeError:
                    br = peakdet.compute_breath_events(y_proc, pks)

            if br is not None:
                on = np.asarray(br.get("onsets", []), dtype=int)
                off = np.asarray(br.get("offsets", []), dtype=int)
                expmins = np.asarray(br.get("expmins", []), dtype=int)
                expoffs = np.asarray(br.get("expoffs", []), dtype=int)

                if on.size >= 2:
                    # Retrieve eupnea mask from cache (computed earlier)
                    eupnea_mask = self._eupnea_masks_cache.get(s, None)
                    if eupnea_mask is None:
                        # Fallback: compute if not cached - use active classifier
                        eupnea_mask = self.window._compute_eupnea_from_active_classifier(s, len(y_proc))
                        # Fallback to traditional method if no classifier data available
                        if eupnea_mask.sum() == 0:
                            eupnea_mask = metrics.detect_eupnic_regions(
                                st.t, y_proc, st.sr_hz, pks, on, off, expmins, expoffs,
                                freq_threshold_hz=eupnea_thresh,
                                min_duration_sec=self.window.eupnea_min_duration
                            )
                        self._eupnea_masks_cache[s] = eupnea_mask

                    eupnea_masks_by_sweep[s] = eupnea_mask

        print(f"[PDF] Computed {len(eupnea_masks_by_sweep)} eupnea masks (for intervals only)")

        print(f"[PDF] Building histograms (raw and time-normalized)...")
        # Pools + sigh overlays
        (hist_vals_raw,
        hist_vals_norm,
        sigh_vals_raw_by_key,
        sigh_vals_norm_by_key,
        sigh_times_rel) = _build_hist_vals_raw_and_norm()
        print(f"[PDF] Built raw and time-normalized histograms")

        # Build EUPNEA-normalized histograms using simplified GMM-direct approach
        print(f"[PDF] Building eupnea-normalized histograms (using GMM directly)...")

        def _build_eupnea_normalized_hists():
            """
            Simplified GMM-direct approach:
            1. Collect all breath metric values from eupneic baseline periods
            2. Eupneic = NOT sniffing (based on GMM sniff_regions_by_sweep)
            3. Compute mean baseline per metric
            4. Normalize all breath values by that baseline

            This is MUCH faster than computing full eupnea masks because we just
            check each breath directly against sniff regions.
            """
            # Step 1: Collect eupneic baseline breath values
            eupnea_baseline_breaths = {k: [] for k in keys_for_csv}
            all_breath_values_raw = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_raw = {k: [] for k in keys_for_csv}

            kept = [s for s in range(next(iter(st.sweeps.values())).shape[1])
                    if s not in getattr(st, "omitted_sweeps", set())]
            t0 = float(stim_zero) if stim_zero is not None else 0.0

            for s in kept:
                y_proc = self.window._get_processed_for(st.analyze_chan, s)
                pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                br = st.breath_by_sweep.get(s, None)
                if br is None or pks.size < 2:
                    continue

                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2
                t_rel_all = (st.t[mids] - t0).astype(float)

                # Check which breaths are sighs
                sigh_idx = np.asarray(st.sigh_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
                is_sigh_breath = np.zeros(len(mids), dtype=bool)
                if sigh_idx.size:
                    for j in range(len(mids)):
                        a = int(on[j]); b = int(on[j + 1])
                        if np.any((sigh_idx >= a) & (sigh_idx < b)):
                            is_sigh_breath[j] = True

                # For each breath, check if it's eupneic (NOT sniffing) using GMM directly
                for i, mid_idx in enumerate(mids):
                    t_rel = t_rel_all[i]
                    # SIMPLIFIED: Check if breath is eupneic by checking if NOT in sniffing region
                    is_eupneic = not self._is_breath_sniffing(s, i, on)
                    is_baseline = t_rel < 0

                    # Determine region
                    if have_stim:
                        if t_rel < 0:
                            region = "baseline"
                        elif t_rel <= stim_dur:
                            region = "stim"
                        else:
                            region = "post"
                    else:
                        region = "all"

                    # Extract metric values from raw histograms (already computed by _build_hist_vals_raw_and_norm)
                    # We'll use the same raw values that were already extracted
                    # But we need to recompute - let's just use the data from hist_vals_raw
                    # Actually, we can't easily map back. Let's sample from y2_ds_by_key at this time point

                    for k in keys_for_csv:
                        Y_raw = y2_ds_by_key.get(k, None)
                        if Y_raw is None or Y_raw.size == 0:
                            continue

                        # Find closest downsampled time point
                        ds_idx = np.argmin(np.abs(t_ds_csv - t_rel))
                        if ds_idx >= Y_raw.shape[0]:
                            continue

                        col_idx = kept.index(s)
                        val = Y_raw[ds_idx, col_idx]

                        if not np.isfinite(val):
                            continue

                        # Collect for baseline calculation if eupneic and baseline
                        if is_eupneic and is_baseline:
                            eupnea_baseline_breaths[k].append(val)

                        # Collect all breath values (will normalize later)
                        all_breath_values_raw[k]["all"].append(val)
                        if have_stim:
                            all_breath_values_raw[k][region].append(val)

                        # Collect sigh values
                        if is_sigh_breath[i]:
                            sigh_vals_raw[k].append(val)

            # Step 2: Compute eupneic baseline means
            eupnea_baselines = {}
            for k in keys_for_csv:
                if len(eupnea_baseline_breaths[k]) >= 10:
                    eupnea_baselines[k] = np.mean(eupnea_baseline_breaths[k])
                elif len(eupnea_baseline_breaths[k]) > 0:
                    # Use what we have if < 10
                    eupnea_baselines[k] = np.mean(eupnea_baseline_breaths[k])
                else:
                    eupnea_baselines[k] = np.nan

            # Step 3: Normalize all values
            hist_norm_eupnea = {k: {"all": [], "baseline": [], "stim": [], "post": []} for k in keys_for_csv}
            sigh_vals_norm_eupnea = {k: [] for k in keys_for_csv}

            for k in keys_for_csv:
                baseline = eupnea_baselines.get(k, np.nan)
                if not np.isfinite(baseline) or abs(baseline) < EPS_BASE:
                    continue

                for region in ["all", "baseline", "stim", "post"]:
                    hist_norm_eupnea[k][region] = [v / baseline for v in all_breath_values_raw[k][region]]

                sigh_vals_norm_eupnea[k] = [v / baseline for v in sigh_vals_raw[k]]

            # Also store baselines and normalized time series for plotting
            y2_norm_eupnea = {}
            for k in keys_for_csv:
                Y_raw = y2_ds_by_key.get(k, None)
                baseline = eupnea_baselines.get(k, np.nan)
                if Y_raw is not None and np.isfinite(baseline) and abs(baseline) > EPS_BASE:
                    y2_norm_eupnea[k] = Y_raw / baseline
                else:
                    y2_norm_eupnea[k] = None

            return hist_norm_eupnea, sigh_vals_norm_eupnea, eupnea_baselines, y2_norm_eupnea

        hist_vals_norm_eupnea, sigh_vals_norm_eupnea_by_key, eupnea_baselines_by_key, y2_ds_by_key_norm_eupnea = _build_eupnea_normalized_hists()
        print(f"[PDF] Built eupnea-normalized histograms")

        # ---------- Create three-page PDF ----------
        nrows = max(1, len(keys_for_csv))
        fig_w = 13
        fig_h = max(4.0, 2.6 * nrows)  # Reduced from 5.25 to 2.6 per row (half of 5.25)

        # Page 1 — RAW
        fig1, axes1 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig1, axes1, y2_ds_by_key, hist_vals_raw, sigh_vals_raw_by_key, sigh_times_rel, title_suffix="")
        fig1.suptitle("Summary — raw", fontsize=11)
        fig1.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Page 2 — NORMALIZED (time-based)
        fig2, axes2 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig2, axes2, y2_ds_by_key_norm, hist_vals_norm, sigh_vals_norm_by_key, sigh_times_rel, title_suffix=" (norm)")
        fig2.suptitle("Summary — normalized (time-based)", fontsize=11)
        fig2.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Page 3 — NORMALIZED (eupnea-based)
        fig3, axes3 = plt.subplots(nrows, 3, figsize=(fig_w, fig_h), squeeze=False)
        _plot_grid(fig3, axes3, y2_ds_by_key_norm_eupnea, hist_vals_norm_eupnea, sigh_vals_norm_eupnea_by_key, sigh_times_rel, title_suffix=" (norm eupnea)")
        fig3.suptitle("Summary — normalized (eupnea-based)", fontsize=11)
        fig3.tight_layout(rect=[0, 0, 1, 0.99])  # Leave 1% at top for suptitle

        # Either return figures or save to PDF
        if return_figures:
            return fig1, fig2, fig3
        else:
            with PdfPages(out_path) as pdf:
                pdf.savefig(fig1, dpi=150)
                pdf.savefig(fig2, dpi=150)
                pdf.savefig(fig3, dpi=150)
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

    def _save_event_aligned_cta_pdf(self, out_path, kept_sweeps, cached_traces_by_sweep, keys_for_csv, label_by_key):
        """
        Generate PDF with cycle-triggered averages (CTAs) aligned to event onsets/offsets.

        PDF Layout (3 columns per row, one row per metric):
        - Column 1: CTA around event onsets (-2s to +2s)
        - Column 2: CTA around event offsets (-2s to +2s)
        - Column 3: Histograms comparing during events vs outside events

        Args:
            out_path: Path to save PDF
            kept_sweeps: List of sweep indices to include
            cached_traces_by_sweep: Dict of {sweep: {metric: trace_array}}
            keys_for_csv: List of metric keys to plot
            label_by_key: Dict of {metric: display_label}
        """
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        st = self.window.state

        # Check if event channel exists and has annotations
        if not st.event_channel or not st.bout_annotations:
            print("[CTA PDF] No event channel or annotations found, skipping event-aligned PDF")
            return

        print(f"[CTA PDF] Generating event-aligned CTA PDF for {len(kept_sweeps)} sweeps...")

        # Generate figure using shared helper
        fig = self._generate_event_cta_figure(kept_sweeps, cached_traces_by_sweep, keys_for_csv, label_by_key)

        # Save to PDF
        with PdfPages(out_path) as pdf:
            pdf.savefig(fig, dpi=150)
        plt.close(fig)

        print(f"[CTA PDF] ✓ Event-aligned CTA PDF saved to {out_path.name}")

    def _show_event_cta_preview_dialog(self, kept_sweeps, cached_traces_by_sweep, keys_for_csv, label_by_key):
        """Display interactive preview dialog with event-aligned CTA figure."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QScrollArea, QSizePolicy
        from PyQt6.QtCore import Qt, QObject, QEvent
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        st = self.window.state

        # Generate the CTA figure using similar logic to _save_event_aligned_cta_pdf
        # (but return the figure instead of saving)
        print("[CTA Preview] Generating event-aligned CTA figure...")

        fig = self._generate_event_cta_figure(kept_sweeps, cached_traces_by_sweep, keys_for_csv, label_by_key)

        # -------------------- Display in dialog --------------------
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Event-Aligned CTA Preview")
        dialog.resize(1300, 900)

        main_layout = QVBoxLayout(dialog)

        # Close button at top
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        main_layout.addWidget(close_btn)

        # Create scroll area for the canvas with mouse wheel support (match default preview)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow canvas to resize to fit width
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvas for displaying matplotlib figure
        canvas = FigureCanvas(fig)

        # Set size policy to allow width scaling but keep height fixed (match default preview)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        canvas.setMinimumWidth(800)  # Minimum width only

        # Install event filter to forward wheel events to scroll area (match default preview)
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    # Forward wheel event to scroll area's viewport
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True  # Event handled
                return False  # Let other events pass through

        wheel_filter = WheelEventFilter(scroll_area)

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas.installEventFilter(filter_obj)

        scroll_area.setWidget(canvas)
        main_layout.addWidget(scroll_area)

        # Show timing message before dialog appears
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Summary generated ({t_elapsed:.1f}s)", 5000)

        # Show dialog modally
        dialog.exec()

        # Clean up
        plt.close(fig)

    def _generate_event_cta_figure(self, kept_sweeps, cached_traces_by_sweep, keys_for_csv, label_by_key):
        """
        Generate the event-aligned CTA figure (shared by save and preview).
        Returns the matplotlib figure object.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import stats

        st = self.window.state

        # Parameters
        WINDOW_BEFORE = 2.0  # seconds before event
        WINDOW_AFTER = 2.0   # seconds after event

        # Collect all event-aligned windows
        onset_windows = {k: [] for k in keys_for_csv}
        offset_windows = {k: [] for k in keys_for_csv}
        during_event_vals = {k: [] for k in keys_for_csv}
        outside_event_vals = {k: [] for k in keys_for_csv}

        for s in kept_sweeps:
            traces = cached_traces_by_sweep.get(s, {})
            bout_list = st.bout_annotations.get(s, [])

            if not bout_list or not traces:
                continue

            for k in keys_for_csv:
                trace = traces.get(k, None)
                if trace is None or len(trace) != len(st.t):
                    continue

                br = st.breath_by_sweep.get(s, None)
                if br is None:
                    continue
                on = np.asarray(br.get("onsets", []), dtype=int)
                if on.size < 2:
                    continue

                mids = (on[:-1] + on[1:]) // 2

                for bout in bout_list:
                    onset_time = bout['start_time']
                    offset_time = bout['end_time']

                    # CTA around ONSET
                    mask_onset = (st.t >= onset_time - WINDOW_BEFORE) & (st.t <= onset_time + WINDOW_AFTER)
                    if np.any(mask_onset):
                        t_window = st.t[mask_onset] - onset_time
                        vals_window = trace[mask_onset]
                        onset_windows[k].append((t_window, vals_window))

                    # CTA around OFFSET
                    mask_offset = (st.t >= offset_time - WINDOW_BEFORE) & (st.t <= offset_time + WINDOW_AFTER)
                    if np.any(mask_offset):
                        t_window = st.t[mask_offset] - offset_time
                        vals_window = trace[mask_offset]
                        offset_windows[k].append((t_window, vals_window))

                    # Histogram data
                    for i, mid_idx in enumerate(mids):
                        t_mid = st.t[int(mid_idx)]
                        val = trace[int(mid_idx)]

                        if not np.isfinite(val):
                            continue

                        if onset_time <= t_mid <= offset_time:
                            during_event_vals[k].append(val)
                        else:
                            outside_event_vals[k].append(val)

        # Create figure (reduced height by 1/3 from 2.5x standard)
        n_metrics = len(keys_for_csv)
        fig_w = 13
        fig_h = max(6.67, 4.33 * n_metrics)  # 4.33 inches per row (~1.67x standard PDF)
        fig = plt.figure(figsize=(fig_w, fig_h))

        for idx, k in enumerate(keys_for_csv):
            label = label_by_key.get(k, k)

            # Column 1: Onset CTA
            ax1 = plt.subplot(n_metrics, 3, idx*3 + 1)
            if onset_windows[k]:
                for t_win, vals_win in onset_windows[k]:
                    ax1.plot(t_win, vals_win, 'b-', alpha=0.15, linewidth=0.5)

                t_common = np.linspace(-WINDOW_BEFORE, WINDOW_AFTER, 200)
                interp_traces = []
                for t_win, vals_win in onset_windows[k]:
                    if len(t_win) > 1:
                        interp_vals = np.interp(t_common, t_win, vals_win, left=np.nan, right=np.nan)
                        interp_traces.append(interp_vals)

                if interp_traces:
                    trace_matrix = np.array(interp_traces)
                    mean_trace = np.nanmean(trace_matrix, axis=0)
                    sem_trace = stats.sem(trace_matrix, axis=0, nan_policy='omit')

                    ax1.plot(t_common, mean_trace, 'b-', linewidth=2, label=f'Mean (n={len(interp_traces)})')
                    ax1.fill_between(t_common, mean_trace - sem_trace, mean_trace + sem_trace,
                                     alpha=0.3, color='blue', label='SEM')

                ax1.axvline(0, color='red', linestyle='--', linewidth=1, label='Event Onset')
                ax1.set_xlabel('Time from Event Onset (s)')
                ax1.set_ylabel(label)
                ax1.set_title(f'{label} - Event Onset CTA')
                ax1.legend(fontsize=8, loc='best')
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title(f'{label} - Event Onset CTA')

            # Column 2: Offset CTA
            ax2 = plt.subplot(n_metrics, 3, idx*3 + 2)
            if offset_windows[k]:
                for t_win, vals_win in offset_windows[k]:
                    ax2.plot(t_win, vals_win, 'g-', alpha=0.15, linewidth=0.5)

                t_common = np.linspace(-WINDOW_BEFORE, WINDOW_AFTER, 200)
                interp_traces = []
                for t_win, vals_win in offset_windows[k]:
                    if len(t_win) > 1:
                        interp_vals = np.interp(t_common, t_win, vals_win, left=np.nan, right=np.nan)
                        interp_traces.append(interp_vals)

                if interp_traces:
                    trace_matrix = np.array(interp_traces)
                    mean_trace = np.nanmean(trace_matrix, axis=0)
                    sem_trace = stats.sem(trace_matrix, axis=0, nan_policy='omit')

                    ax2.plot(t_common, mean_trace, 'g-', linewidth=2, label=f'Mean (n={len(interp_traces)})')
                    ax2.fill_between(t_common, mean_trace - sem_trace, mean_trace + sem_trace,
                                     alpha=0.3, color='green', label='SEM')

                ax2.axvline(0, color='red', linestyle='--', linewidth=1, label='Event Offset')
                ax2.set_xlabel('Time from Event Offset (s)')
                ax2.set_ylabel(label)
                ax2.set_title(f'{label} - Event Offset CTA')
                ax2.legend(fontsize=8, loc='best')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f'{label} - Event Offset CTA')

            # Match y-axis limits between onset and offset plots
            if onset_windows[k] and offset_windows[k]:
                ylim1 = ax1.get_ylim()
                ylim2 = ax2.get_ylim()
                combined_min = min(ylim1[0], ylim2[0])
                combined_max = max(ylim1[1], ylim2[1])
                ax1.set_ylim(combined_min, combined_max)
                ax2.set_ylim(combined_min, combined_max)

            # Column 3: Histograms
            ax3 = plt.subplot(n_metrics, 3, idx*3 + 3)
            during = np.array(during_event_vals[k])
            outside = np.array(outside_event_vals[k])

            if during.size > 0 and outside.size > 0:
                all_vals = np.concatenate([during, outside])
                all_vals_finite = all_vals[np.isfinite(all_vals)]

                # Check if we have sufficient data for histogram
                if all_vals_finite.size >= 2 and np.std(all_vals_finite) > 1e-10:
                    try:
                        bins = np.histogram_bin_edges(all_vals_finite, bins=100)

                        ax3.hist(outside, bins=bins, alpha=0.6, color='blue', label=f'Outside Events (n={len(outside)})', density=True)
                        ax3.hist(during, bins=bins, alpha=0.6, color='orange', label=f'During Events (n={len(during)})', density=True)
                    except (ValueError, RuntimeWarning) as e:
                        ax3.text(0.5, 0.5, f'Histogram error:\n{str(e)}',
                               ha='center', va='center', transform=ax3.transAxes, fontsize=8)
                else:
                    ax3.text(0.5, 0.5, 'Insufficient variation\nin data for histogram',
                           ha='center', va='center', transform=ax3.transAxes, fontsize=8)

                if len(during) > 1 and len(outside) > 1:
                    t_stat, p_val = stats.ttest_ind(during, outside, nan_policy='omit')
                    mean_during = np.nanmean(during)
                    mean_outside = np.nanmean(outside)
                    std_during = np.nanstd(during)
                    std_outside = np.nanstd(outside)

                    # Format p-value: use scientific notation for very small values
                    p_str = f'{p_val:.4f}' if p_val >= 0.001 else f'{p_val:.2e}'

                    stats_text = (f'During: {mean_during:.2f} ± {std_during:.2f}\n'
                                  f'Outside: {mean_outside:.2f} ± {std_outside:.2f}\n'
                                  f'p = {p_str}')
                    ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                            fontsize=8, verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax3.set_xlabel(label)
                ax3.set_ylabel('Density')
                ax3.set_title(f'{label} - During vs Outside Events')
                ax3.legend(fontsize=8, loc='upper left')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title(f'{label} - During vs Outside Events')

        plt.tight_layout()
        return fig

    def _show_pulse_4page_preview(self, fig_cta, fig_3d_offset, fig_3d_stim, fig_prob):
        """Display 4-page preview dialog: CTA + 3D (offset-aligned) + 3D (stim-aligned) + Probability."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QWidget
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
        from PyQt6.QtCore import Qt, QEvent, QObject
        from PyQt6.QtWidgets import QSizePolicy, QStackedWidget
        import time

        dialog = QDialog(self.window)
        dialog.setWindowTitle("Pulse Analysis Preview")
        dialog.resize(1300, 900)

        main_layout = QVBoxLayout(dialog)

        # Page selector controls
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 4 (CTA Overlay)")
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Create canvases and toolbars for each figure
        def create_canvas_with_toolbar(fig):
            """Create a widget containing a canvas and navigation toolbar."""
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)

            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar2QT(canvas, container)

            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)

            return container, canvas

        # Create page widgets with toolbars
        container_cta, canvas_cta = create_canvas_with_toolbar(fig_cta)
        container_3d_offset, canvas_3d_offset = create_canvas_with_toolbar(fig_3d_offset)
        container_3d_stim, canvas_3d_stim = create_canvas_with_toolbar(fig_3d_stim)
        container_prob, canvas_prob = create_canvas_with_toolbar(fig_prob)

        # Stacked widget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(container_cta)         # index 0
        canvas_stack.addWidget(container_3d_offset)   # index 1
        canvas_stack.addWidget(container_3d_stim)     # index 2
        canvas_stack.addWidget(container_prob)        # index 3
        canvas_stack.setCurrentIndex(0)

        # Event filter for wheel scrolling
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True
                return False

        wheel_filter = WheelEventFilter(scroll_area)

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas_cta.installEventFilter(filter_obj)
        canvas_3d_offset.installEventFilter(filter_obj)
        canvas_3d_stim.installEventFilter(filter_obj)
        canvas_prob.installEventFilter(filter_obj)

        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Navigation
        current_page = [1]

        def update_page():
            labels = [
                "Page 1 of 4 (CTA Overlay)",
                "Page 2 of 4 (3D Phase-Sorted - Offset Aligned)",
                "Page 3 of 4 (3D Phase-Sorted - Stimulus Aligned)",
                "Page 4 of 4 (Baseline Breath Probability)"
            ]
            idx = current_page[0] - 1
            canvas_stack.setCurrentIndex(idx)
            page_label.setText(labels[idx])
            prev_btn.setEnabled(current_page[0] > 1)
            next_btn.setEnabled(current_page[0] < 4)
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(4, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()

        # Show timing message
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Pulse analysis preview generated ({t_elapsed:.1f}s)", 5000)

        # Clean up figures when dialog closes
        def cleanup():
            plt.close(fig_cta)
            plt.close(fig_3d_offset)
            plt.close(fig_3d_stim)
            plt.close(fig_prob)

        dialog.finished.connect(cleanup)

        # Show dialog non-modally so standard preview can be generated in parallel
        dialog.show()

    def _show_pulse_3page_preview(self, fig_cta, fig_3d, fig_prob):
        """Display 3-page preview dialog: CTA + 3D phase-sorted + Probability curve."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtCore import Qt, QEvent, QObject
        from PyQt6.QtWidgets import QSizePolicy, QStackedWidget
        import time

        dialog = QDialog(self.window)
        dialog.setWindowTitle("Pulse Analysis Preview")
        dialog.resize(1300, 900)

        main_layout = QVBoxLayout(dialog)

        # Page selector controls
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 3 (CTA Overlay)")
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvases
        canvas_cta = FigureCanvas(fig_cta)
        canvas_3d = FigureCanvas(fig_3d)
        canvas_prob = FigureCanvas(fig_prob)

        for canvas in [canvas_cta, canvas_3d, canvas_prob]:
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)

        # Stacked widget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(canvas_cta)   # index 0
        canvas_stack.addWidget(canvas_3d)    # index 1
        canvas_stack.addWidget(canvas_prob)  # index 2
        canvas_stack.setCurrentIndex(0)

        # Event filter for wheel scrolling
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True
                return False

        wheel_filter = WheelEventFilter(scroll_area)

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas_cta.installEventFilter(filter_obj)
        canvas_3d.installEventFilter(filter_obj)
        canvas_prob.installEventFilter(filter_obj)

        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Navigation
        current_page = [1]

        def update_page():
            labels = [
                "Page 1 of 3 (CTA Overlay)",
                "Page 2 of 3 (3D Phase-Sorted Traces)",
                "Page 3 of 3 (Baseline Breath Probability)"
            ]
            idx = current_page[0] - 1
            canvas_stack.setCurrentIndex(idx)
            page_label.setText(labels[idx])
            prev_btn.setEnabled(current_page[0] > 1)
            next_btn.setEnabled(current_page[0] < 3)
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(3, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()

        # Show timing message
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Pulse analysis preview generated ({t_elapsed:.1f}s)", 5000)

        # Show dialog modally
        dialog.exec()

        # Clean up
        plt.close(fig_cta)
        plt.close(fig_3d)
        plt.close(fig_prob)

    def _show_pulse_2page_preview(self, fig_cta, fig_3d):
        """Display 2-page preview dialog: CTA + 3D phase-sorted traces."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtCore import Qt, QEvent, QObject
        from PyQt6.QtWidgets import QSizePolicy, QStackedWidget
        import time

        dialog = QDialog(self.window)
        dialog.setWindowTitle("Pulse Analysis Preview")
        dialog.resize(1300, 900)

        main_layout = QVBoxLayout(dialog)

        # Page selector controls
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 2 (CTA Overlay)")  # Will be updated by update_page()
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvases
        canvas_cta = FigureCanvas(fig_cta)
        canvas_3d = FigureCanvas(fig_3d)

        for canvas in [canvas_cta, canvas_3d]:
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)

        # Stacked widget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(canvas_cta)  # index 0
        canvas_stack.addWidget(canvas_3d)   # index 1
        canvas_stack.setCurrentIndex(0)

        # Event filter for wheel scrolling
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True
                return False

        wheel_filter = WheelEventFilter(scroll_area)

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas_cta.installEventFilter(filter_obj)
        canvas_3d.installEventFilter(filter_obj)

        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Navigation
        current_page = [1]

        def update_page():
            labels = [
                "Page 1 of 2 (CTA Overlay)",
                "Page 2 of 2 (3D Phase-Sorted Traces)"
            ]
            idx = current_page[0] - 1
            canvas_stack.setCurrentIndex(idx)
            page_label.setText(labels[idx])
            prev_btn.setEnabled(current_page[0] > 1)
            next_btn.setEnabled(current_page[0] < 2)
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(2, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()

        # Show timing message
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Pulse analysis generated ({t_elapsed:.1f}s)", 5000)

        dialog.exec()

        # Cleanup
        plt.close(fig_cta)
        plt.close(fig_3d)

    def _show_summary_preview_dialog_with_pulse(self, fig_pulse, fig1, fig2, fig3):
        """Display interactive preview dialog with 4 pages: pulse CTA + 3 standard pages."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtCore import Qt, QEvent, QObject
        from PyQt6.QtWidgets import QSizePolicy, QStackedWidget
        import time

        # -------------------- Display in dialog --------------------
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Summary Preview - Pulse Experiment")
        dialog.resize(1300, 900)

        main_layout = QVBoxLayout(dialog)

        # Page selector controls at top
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 4 (Pulse CTA)")
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Create scroll area for the canvas
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvas for each figure
        canvas_pulse = FigureCanvas(fig_pulse)
        canvas1 = FigureCanvas(fig1)
        canvas2 = FigureCanvas(fig2)
        canvas3 = FigureCanvas(fig3)

        # Set size policies
        for canvas in [canvas_pulse, canvas1, canvas2, canvas3]:
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)

        # Create stacked widget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(canvas_pulse)  # index 0
        canvas_stack.addWidget(canvas1)       # index 1
        canvas_stack.addWidget(canvas2)       # index 2
        canvas_stack.addWidget(canvas3)       # index 3
        canvas_stack.setCurrentIndex(0)

        # Install event filter for wheel scrolling
        class WheelEventFilter:
            def __init__(self, scroll_area):
                self.scroll_area = scroll_area

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Wheel:
                    scrollbar = self.scroll_area.verticalScrollBar()
                    scrollbar.setValue(scrollbar.value() - event.angleDelta().y())
                    return True
                return False

        wheel_filter = WheelEventFilter(scroll_area)

        class EventFilterObject(QObject):
            def __init__(self, filter_func):
                super().__init__()
                self.filter_func = filter_func

            def eventFilter(self, obj, event):
                return self.filter_func(obj, event)

        filter_obj = EventFilterObject(wheel_filter.eventFilter)
        canvas_pulse.installEventFilter(filter_obj)
        canvas1.installEventFilter(filter_obj)
        canvas2.installEventFilter(filter_obj)
        canvas3.installEventFilter(filter_obj)

        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Page navigation
        current_page = [1]

        def update_page():
            labels = [
                "Page 1 of 4 (Pulse CTA)",
                "Page 2 of 4 (Raw Metrics)",
                "Page 3 of 4 (Normalized - Time-based)",
                "Page 4 of 4 (Normalized - Eupnea-based)"
            ]
            idx = current_page[0] - 1
            canvas_stack.setCurrentIndex(idx)
            page_label.setText(labels[idx])
            prev_btn.setEnabled(current_page[0] > 1)
            next_btn.setEnabled(current_page[0] < 4)
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(4, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()

        # Show timing message
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Summary generated ({t_elapsed:.1f}s)", 5000)

        dialog.exec()

        # Cleanup
        plt.close(fig_pulse)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    def _show_simple_figure_preview(self, fig, title="Preview"):
        """
        Display a simple preview dialog with a single matplotlib figure.

        Args:
            fig: Matplotlib figure to display
            title: Window title
        """
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QScrollArea
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from PyQt6.QtCore import Qt
        import time

        dialog = QDialog(self.window)
        dialog.setWindowTitle(title)
        dialog.resize(1200, 700)

        main_layout = QVBoxLayout(dialog)

        # Scroll area for the canvas
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Canvas for displaying matplotlib figure
        canvas = FigureCanvas(fig)
        from PyQt6.QtWidgets import QSizePolicy
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        canvas.setMinimumWidth(800)

        scroll_area.setWidget(canvas)
        main_layout.addWidget(scroll_area)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        main_layout.addWidget(close_btn)

        # Show timing message
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Preview generated ({t_elapsed:.1f}s)", 5000)

        dialog.exec()

        # Cleanup
        plt.close(fig)


    def _show_summary_preview_dialog(self, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, stim_zero, stim_dur):
        """Display interactive preview dialog with the three summary figures."""
        import matplotlib.pyplot as plt
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
        from PyQt6.QtCore import QObject, QEvent, Qt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

        # Generate figures using the same method that saves PDFs, but get figures back instead
        fig1, fig2, fig3 = self._save_metrics_summary_pdf(
            out_path=None,  # Not used when return_figures=True
            t_ds_csv=t_ds_csv,
            y2_ds_by_key=y2_ds_by_key,
            keys_for_csv=keys_for_csv,
            label_by_key=label_by_key,
            stim_zero=stim_zero,
            stim_dur=stim_dur,
            return_figures=True,
        )

        # -------------------- Display in dialog --------------------
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Summary Preview")
        dialog.resize(1300, 900)  # Slightly larger window

        main_layout = QVBoxLayout(dialog)

        # Page selector controls at top
        control_layout = QHBoxLayout()
        page_label = QLabel("Page 1 of 3 (Raw)")
        prev_btn = QPushButton("← Previous")
        next_btn = QPushButton("Next →")
        close_btn = QPushButton("Close")

        control_layout.addWidget(prev_btn)
        control_layout.addWidget(page_label)
        control_layout.addWidget(next_btn)
        control_layout.addStretch()
        control_layout.addWidget(close_btn)

        main_layout.addLayout(control_layout)

        # Create scroll area for the canvas with mouse wheel support
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # Allow canvas to resize to fit width
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Enable mouse wheel scrolling by setting focus policy
        scroll_area.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        # Canvas for displaying matplotlib figures
        canvas1 = FigureCanvas(fig1)
        canvas2 = FigureCanvas(fig2)
        canvas3 = FigureCanvas(fig3)

        # Disconnect matplotlib's scroll event handler to prevent zoom conflicts
        # This allows our event filter to handle scrolling instead
        for canvas in [canvas1, canvas2, canvas3]:
            # Disconnect scroll_event callbacks if they exist
            try:
                if hasattr(canvas, 'callbacks') and hasattr(canvas.callbacks, 'callbacks'):
                    if 'scroll_event' in canvas.callbacks.callbacks:
                        # Get list of callback IDs for scroll events
                        scroll_cids = list(canvas.callbacks.callbacks.get('scroll_event', {}).keys())
                        for cid in scroll_cids:
                            canvas.mpl_disconnect(cid)
            except Exception as e:
                print(f"[Preview] Could not disconnect scroll events: {e}")

        # Set size policies to allow width scaling but keep height fixed
        from PyQt6.QtWidgets import QSizePolicy
        for canvas in [canvas1, canvas2, canvas3]:
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            canvas.setMinimumWidth(800)  # Minimum width

        # Create a container widget to hold all three canvases (so they don't get deleted)
        from PyQt6.QtWidgets import QWidget, QStackedWidget
        canvas_stack = QStackedWidget()
        canvas_stack.addWidget(canvas1)  # index 0
        canvas_stack.addWidget(canvas2)  # index 1
        canvas_stack.addWidget(canvas3)  # index 2
        canvas_stack.setCurrentIndex(0)  # Start with page 1

        # Add eventFilter method to dialog to forward wheel events to scroll area
        # This matches the pattern used in GMM dialog which works correctly
        def eventFilter_for_dialog(obj, event):
            if event.type() == QEvent.Type.Wheel:
                # Check if event is from any of the canvases
                if obj in [canvas1, canvas2, canvas3]:
                    # Forward wheel event to scroll area
                    scroll_area.verticalScrollBar().setValue(
                        scroll_area.verticalScrollBar().value() - event.angleDelta().y() // 2
                    )
                    return True  # Event handled
            return False  # Let other events pass through

        # Install dialog as event filter on all canvases
        dialog.eventFilter = eventFilter_for_dialog
        canvas1.installEventFilter(dialog)
        canvas2.installEventFilter(dialog)
        canvas3.installEventFilter(dialog)

        # Put the stacked widget in the scroll area
        scroll_area.setWidget(canvas_stack)
        main_layout.addWidget(scroll_area)

        # Page navigation
        current_page = [1]  # Use list to allow modification in nested function

        def update_page():
            if current_page[0] == 1:
                canvas_stack.setCurrentIndex(0)  # Show canvas1
                page_label.setText("Page 1 of 3 (Raw)")
                prev_btn.setEnabled(False)
                next_btn.setEnabled(True)
            elif current_page[0] == 2:
                canvas_stack.setCurrentIndex(1)  # Show canvas2
                page_label.setText("Page 2 of 3 (Normalized - Time-based)")
                prev_btn.setEnabled(True)
                next_btn.setEnabled(True)
            else:
                canvas_stack.setCurrentIndex(2)  # Show canvas3
                page_label.setText("Page 3 of 3 (Normalized - Eupnea-based)")
                prev_btn.setEnabled(True)
                next_btn.setEnabled(False)
            # Reset scroll position to top when changing pages
            scroll_area.verticalScrollBar().setValue(0)

        def go_prev():
            current_page[0] = max(1, current_page[0] - 1)
            update_page()

        def go_next():
            current_page[0] = min(3, current_page[0] + 1)
            update_page()

        prev_btn.clicked.connect(go_prev)
        next_btn.clicked.connect(go_next)
        close_btn.clicked.connect(dialog.close)

        update_page()

        # Show timing message before dialog appears
        if hasattr(self, '_preview_start_time'):
            t_elapsed = time.time() - self._preview_start_time
            self.window._log_status_message(f"✓ Summary generated ({t_elapsed:.1f}s)", 5000)

        # Clean up figures when dialog closes
        def cleanup():
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

        dialog.finished.connect(cleanup)

        # Show dialog non-modally so both pulse and standard previews can be viewed side-by-side
        dialog.show()



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
        st = self.window.state
        N = len(st.t)

        # Prefer the name you've been using; try some alternates if needed.
        candidates = None
        for name in ("sighs_by_sweep", "sigh_indices_by_sweep", "sigh_peaks_by_sweep", "sigh_mask_by_sweep"):
            if hasattr(st, name):
                candidates = getattr(st, name).get(s, None)
                if candidates is not None:
                    break

        if candidates is None:
            return set()

        arr = np.asarray(list(candidates))
        out: set[int] = set()

        # If integer-like
        if arr.dtype.kind in "iu":
            arr = arr.astype(int)
            if pks is not None and arr.size and arr.max(initial=-1) < len(pks):
                # Looks like indexes INTO peaks -> map to sample indexes
                for idx in arr:
                    if 0 <= idx < len(pks):
                        i = int(pks[idx])
                        if 0 <= i < N:
                            out.add(i)
            else:
                # Already sample indexes
                for i in arr:
                    if 0 <= i < N:
                        out.add(int(i))
            return out

        # If float-like -> assume times (seconds)
        if arr.dtype.kind in "f":
            t = st.t
            for val in arr:
                try:
                    i = int(np.clip(np.searchsorted(t, float(val)), 0, N - 1))
                    out.add(i)
                except Exception:
                    pass
            return out

        # Fallback: try to coerce each element
        for v in arr:
            try:
                i = int(v)
                if 0 <= i < N:
                    out.add(i)
            except Exception:
                try:
                    i = int(np.clip(np.searchsorted(st.t, float(v)), 0, N - 1))
                    out.add(i)
                except Exception:
                    pass
        return out



    ##################################################
    ##Curration                                     ##
    ##################################################
    #     if not base:
    #         return

