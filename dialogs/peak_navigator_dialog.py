"""
Peak Navigator Dialog for PlethApp.

This dialog provides an interface for navigating through different categories of peaks
for efficient manual curation and ML training data labeling.
"""

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QComboBox, QDoubleSpinBox, QSpinBox
)
from PyQt6.QtCore import Qt

from dialogs.export_mixin import ExportMixin


class PeakNavigatorDialog(ExportMixin, QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.state = main_window.state if main_window else None

        self.setWindowTitle("Peak Navigator & Curation")
        self.resize(450, 500)

        # Apply VS Code Dark+ theme to match main window
        self.setStyleSheet("""
            /* VS Code Dark+ Theme */
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
            }
            QGroupBox {
                color: #d4d4d4;
                border: 1px solid white;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                color: #4A9EFF;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid white;
                border-radius: 2px;
                background-color: #252526;
            }
            QCheckBox::indicator:checked {
                background-color: #4A9EFF;
                border: 1px solid white;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8f;
            }
            QPushButton:disabled {
                background-color: #3e3e3e;
                color: #6e6e6e;
            }
            QComboBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid white;
                padding: 4px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #cccccc;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #252526;
                color: #cccccc;
                selection-background-color: #094771;
                border: 1px solid white;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid white;
                padding: 3px;
                border-radius: 3px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                background-color: #0e639c;
                border: none;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #1177bb;
            }
        """)

        # Navigation state
        self.current_category = "merge_candidates"
        self.current_index = 0
        self.candidate_peaks = []  # List of (sweep_idx, peak_sample_idx) tuples

        self._setup_ui()
        self._enable_dark_title_bar()
        self.setup_export_menu()

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

    def _setup_ui(self):
        """Set up the dialog UI."""
        layout = QVBoxLayout()

        # Category selection
        category_group = QGroupBox("Peak Category")
        category_layout = QVBoxLayout()

        self.category_combo = QComboBox()
        self.category_combo.addItem("Merge Candidates (shoulder peaks)", "merge_candidates")
        self.category_combo.addItem("Noise Peaks (high → low amplitude)", "noise_peaks")
        self.category_combo.addItem("Breath Peaks (low → high amplitude)", "breath_peaks")
        self.category_combo.addItem("Sigh Candidates (high amplitude)", "sigh_candidates")
        self.category_combo.currentIndexChanged.connect(self._on_category_changed)

        category_layout.addWidget(QLabel("Select category to navigate:"))
        category_layout.addWidget(self.category_combo)
        category_group.setLayout(category_layout)
        layout.addWidget(category_group)

        # Filters group
        filters_group = QGroupBox("Filters")
        filters_layout = QVBoxLayout()

        # Onset height ratio filter (for merge candidates)
        self.onset_layout = QHBoxLayout()
        self.onset_label = QLabel("onset_height_ratio >")
        self.onset_layout.addWidget(self.onset_label)
        self.onset_threshold = QDoubleSpinBox()
        self.onset_threshold.setRange(0.0, 1.0)
        self.onset_threshold.setSingleStep(0.01)
        self.onset_threshold.setValue(0.05)  # Lower default - catches more candidates
        self.onset_threshold.setDecimals(3)
        self.onset_threshold.valueChanged.connect(self._on_filters_changed)
        self.onset_layout.addWidget(self.onset_threshold)
        self.onset_layout.addStretch()
        filters_layout.addLayout(self.onset_layout)

        # Gap to next normalized filter (for merge candidates)
        self.gap_layout = QHBoxLayout()
        self.gap_label = QLabel("gap_to_next_norm <")
        self.gap_layout.addWidget(self.gap_label)
        self.gap_threshold = QDoubleSpinBox()
        self.gap_threshold.setRange(0.0, 2.0)
        self.gap_threshold.setSingleStep(0.05)
        self.gap_threshold.setValue(0.5)  # Wider default - shoulder peaks can have moderate gaps
        self.gap_threshold.setDecimals(2)
        self.gap_threshold.valueChanged.connect(self._on_filters_changed)
        self.gap_layout.addWidget(self.gap_threshold)
        self.gap_layout.addStretch()
        filters_layout.addLayout(self.gap_layout)

        # P(noise) filter (for noise peaks)
        self.pnoise_layout = QHBoxLayout()
        self.pnoise_label = QLabel("p_noise >")
        self.pnoise_layout.addWidget(self.pnoise_label)
        self.pnoise_threshold = QDoubleSpinBox()
        self.pnoise_threshold.setRange(0.0, 1.0)
        self.pnoise_threshold.setSingleStep(0.05)
        self.pnoise_threshold.setValue(0.50)
        self.pnoise_threshold.setDecimals(2)
        self.pnoise_threshold.valueChanged.connect(self._on_filters_changed)
        self.pnoise_layout.addWidget(self.pnoise_threshold)
        self.pnoise_layout.addStretch()
        filters_layout.addLayout(self.pnoise_layout)

        # Current sweep only checkbox
        self.current_sweep_only = QCheckBox("Current sweep only")
        self.current_sweep_only.setChecked(True)
        self.current_sweep_only.stateChanged.connect(self._on_filters_changed)
        filters_layout.addWidget(self.current_sweep_only)

        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)

        # View settings group
        view_group = QGroupBox("View Settings")
        view_layout = QHBoxLayout()
        view_layout.addWidget(QLabel("Window duration (s):"))
        self.window_duration = QDoubleSpinBox()
        self.window_duration.setRange(0.5, 60.0)
        self.window_duration.setSingleStep(0.5)
        self.window_duration.setValue(10.0)
        self.window_duration.setDecimals(1)
        self.window_duration.valueChanged.connect(self._on_window_duration_changed)
        view_layout.addWidget(self.window_duration)
        view_layout.addStretch()
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Statistics
        self.stats_label = QLabel("Found: 0 candidates")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet("font-size: 12pt; font-weight: bold; color: #4A9EFF;")
        layout.addWidget(self.stats_label)

        # Navigation controls
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("◄ Prev")
        self.prev_button.clicked.connect(self._navigate_prev)
        self.prev_button.setEnabled(False)
        nav_layout.addWidget(self.prev_button)

        self.position_label = QLabel("Peak - of -")
        self.position_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.position_label.setStyleSheet("font-size: 11pt;")
        nav_layout.addWidget(self.position_label)

        self.next_button = QPushButton("Next ►")
        self.next_button.clicked.connect(self._navigate_next)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)

        layout.addLayout(nav_layout)

        # Action buttons
        action_layout = QHBoxLayout()

        self.refresh_button = QPushButton("Refresh Candidates")
        self.refresh_button.clicked.connect(self._refresh_candidates)
        action_layout.addWidget(self.refresh_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        action_layout.addWidget(self.close_button)

        layout.addLayout(action_layout)

        layout.addStretch()
        self.setLayout(layout)

        # Initialize filter visibility
        self._update_filter_visibility()

        # Initialize candidates
        self._refresh_candidates()

    def _on_category_changed(self, index):
        """Handle category dropdown change."""
        self.current_category = self.category_combo.itemData(index)
        self._update_filter_visibility()
        self._refresh_candidates()

    def _on_filters_changed(self):
        """Handle filter value changes."""
        self._refresh_candidates()

    def _on_window_duration_changed(self):
        """Handle window duration change."""
        # Just update - will be used next time we jump to a peak
        pass

    def _update_filter_visibility(self):
        """Show/hide filters based on current category."""
        # Hide all filters first
        self.onset_label.setVisible(False)
        self.onset_threshold.setVisible(False)
        self.gap_label.setVisible(False)
        self.gap_threshold.setVisible(False)
        self.pnoise_label.setVisible(False)
        self.pnoise_threshold.setVisible(False)

        # Show relevant filters based on category
        if self.current_category == "merge_candidates":
            self.onset_label.setVisible(True)
            self.onset_threshold.setVisible(True)
            self.gap_label.setVisible(True)
            self.gap_threshold.setVisible(True)
        elif self.current_category == "noise_peaks":
            self.pnoise_label.setVisible(True)
            self.pnoise_threshold.setVisible(True)
        # breath_peaks and sigh_candidates don't need metric filters

    def _refresh_candidates(self):
        """Refresh the list of candidate peaks based on current category and filters."""
        if not self.state:
            self.candidate_peaks = []
            self._update_ui()
            return

        self.candidate_peaks = []

        # Determine which sweeps to search
        if self.current_sweep_only.isChecked():
            sweeps_to_search = [self.state.sweep_idx]
        else:
            sweeps_to_search = sorted(self.state.peaks_by_sweep.keys())

        # Build candidate list based on category
        if self.current_category == "merge_candidates":
            self._find_merge_candidates(sweeps_to_search)
        elif self.current_category == "noise_peaks":
            self._find_noise_peaks(sweeps_to_search)
        elif self.current_category == "breath_peaks":
            self._find_breath_peaks(sweeps_to_search)
        elif self.current_category == "sigh_candidates":
            self._find_sigh_candidates(sweeps_to_search)

        # Reset to first candidate
        self.current_index = 0
        self._update_ui()

        # Jump to first candidate if available
        if len(self.candidate_peaks) > 0:
            self._jump_to_current_candidate()

        print(f"[peak-navigator] Found {len(self.candidate_peaks)} candidates for category '{self.current_category}'")

    def _find_merge_candidates(self, sweeps):
        """Find peaks that are likely merge candidates (shoulder peaks)."""
        onset_thresh = self.onset_threshold.value()
        gap_thresh = self.gap_threshold.value()

        print(f"[peak-navigator] Looking for merge candidates with onset>{onset_thresh}, gap<{gap_thresh}")

        for s in sweeps:
            # Get current peak metrics (if available)
            metrics_dict = None
            if hasattr(self.state, 'current_peak_metrics_by_sweep') and s in self.state.current_peak_metrics_by_sweep:
                metrics_dict = self.state.current_peak_metrics_by_sweep[s]
                print(f"[peak-navigator] Sweep {s}: Found {len(metrics_dict)} metrics in current_peak_metrics_by_sweep")
            elif hasattr(self.state, 'peak_metrics_by_sweep') and s in self.state.peak_metrics_by_sweep:
                metrics_dict = self.state.peak_metrics_by_sweep[s]
                print(f"[peak-navigator] Sweep {s}: Found {len(metrics_dict)} metrics in peak_metrics_by_sweep")
            else:
                print(f"[peak-navigator] Sweep {s}: No metrics found")

            if not metrics_dict:
                continue

            # Filter for merge candidates
            count_before = len(self.candidate_peaks)

            # Debug: Check first few metrics to see what's available
            if s == 0:
                sample_metric = metrics_dict[0] if len(metrics_dict) > 0 else {}
                print(f"[peak-navigator] Sample metric ALL keys: {list(sample_metric.keys())}")
                print(f"[peak-navigator] Sample onset_height_ratio: {sample_metric.get('onset_height_ratio')}")
                print(f"[peak-navigator] Sample gap_to_next_norm: {sample_metric.get('gap_to_next_norm')}")
                print(f"[peak-navigator] Sample amplitude_normalized: {sample_metric.get('amplitude_normalized')}")
                print(f"[peak-navigator] Sample amplitude_absolute: {sample_metric.get('amplitude_absolute')}")

            # Track how many pass each filter
            count_has_both = 0
            count_passes_onset = 0
            count_passes_gap = 0

            for m in metrics_dict:
                onset_ratio = m.get('onset_height_ratio')
                gap_norm = m.get('gap_to_next_norm')

                if onset_ratio is not None and gap_norm is not None:
                    count_has_both += 1
                    if onset_ratio > onset_thresh:
                        count_passes_onset += 1
                    if gap_norm < gap_thresh:
                        count_passes_gap += 1

                    if onset_ratio > onset_thresh and gap_norm < gap_thresh:
                        peak_idx = m.get('peak_idx')
                        if peak_idx is not None:
                            self.candidate_peaks.append((s, peak_idx))

            if s == 0:
                print(f"[peak-navigator] Sweep {s}: {count_has_both} peaks with both metrics")
                print(f"[peak-navigator] Sweep {s}: {count_passes_onset} pass onset>{onset_thresh}, {count_passes_gap} pass gap<{gap_thresh}")

            count_added = len(self.candidate_peaks) - count_before
            print(f"[peak-navigator] Sweep {s}: Added {count_added} merge candidates")

        # Sort by onset_height_ratio (descending) - worst first
        self._sort_by_metric('onset_height_ratio', reverse=True, sweeps=sweeps)

    def _find_noise_peaks(self, sweeps):
        """Find peaks classified as noise, sorted by p_noise (highest first)."""
        pnoise_thresh = self.pnoise_threshold.value()

        print(f"[peak-navigator] Looking for noise peaks with p_noise>{pnoise_thresh}")

        for s in sweeps:
            # Get current peak metrics
            metrics_dict = None
            if hasattr(self.state, 'current_peak_metrics_by_sweep') and s in self.state.current_peak_metrics_by_sweep:
                metrics_dict = self.state.current_peak_metrics_by_sweep[s]
            elif hasattr(self.state, 'peak_metrics_by_sweep') and s in self.state.peak_metrics_by_sweep:
                metrics_dict = self.state.peak_metrics_by_sweep[s]

            if not metrics_dict:
                print(f"[peak-navigator] Sweep {s}: No metrics found")
                continue

            # Filter for noise peaks using p_noise
            count_before = len(self.candidate_peaks)

            for m in metrics_dict:
                p_noise = m.get('p_noise')
                if p_noise is not None and not np.isnan(p_noise) and p_noise > pnoise_thresh:
                    peak_idx = m.get('peak_idx')
                    if peak_idx is not None:
                        self.candidate_peaks.append((s, peak_idx))

            count_added = len(self.candidate_peaks) - count_before
            print(f"[peak-navigator] Sweep {s}: Added {count_added} noise peaks")

        # Sort by p_noise (descending) - highest noise probability first
        self._sort_by_metric('p_noise', reverse=True, sweeps=sweeps)

    def _find_breath_peaks(self, sweeps):
        """Find peaks classified as breaths, sorted by amplitude (low to high)."""
        for s in sweeps:
            # Get current peak metrics
            metrics_dict = None
            if hasattr(self.state, 'current_peak_metrics_by_sweep') and s in self.state.current_peak_metrics_by_sweep:
                metrics_dict = self.state.current_peak_metrics_by_sweep[s]
            elif hasattr(self.state, 'peak_metrics_by_sweep') and s in self.state.peak_metrics_by_sweep:
                metrics_dict = self.state.peak_metrics_by_sweep[s]

            if not metrics_dict:
                continue

            # Get all breath peaks (p_breath > 0.5 or just all peaks with low p_noise)
            for m in metrics_dict:
                p_breath = m.get('p_breath')
                p_noise = m.get('p_noise')

                # Consider it a breath if p_breath > 0.5 OR p_noise < 0.5
                is_breath = False
                if p_breath is not None and p_breath > 0.5:
                    is_breath = True
                elif p_noise is not None and p_noise < 0.5:
                    is_breath = True

                if is_breath:
                    peak_idx = m.get('peak_idx')
                    if peak_idx is not None:
                        self.candidate_peaks.append((s, peak_idx))

        # Sort by amplitude (ascending) - smallest first
        self._sort_by_metric('amplitude_absolute', reverse=False, sweeps=sweeps)

    def _find_sigh_candidates(self, sweeps):
        """Find peaks that might be sighs (high amplitude)."""
        print(f"[peak-navigator] Looking for sigh candidates (top 10% amplitude)")

        for s in sweeps:
            # Get current peak metrics
            metrics_dict = None
            if hasattr(self.state, 'current_peak_metrics_by_sweep') and s in self.state.current_peak_metrics_by_sweep:
                metrics_dict = self.state.current_peak_metrics_by_sweep[s]
            elif hasattr(self.state, 'peak_metrics_by_sweep') and s in self.state.peak_metrics_by_sweep:
                metrics_dict = self.state.peak_metrics_by_sweep[s]

            if not metrics_dict:
                print(f"[peak-navigator] Sweep {s}: No metrics found")
                continue

            # Collect all peaks with their amplitudes
            peak_amps = [(m.get('peak_idx'), m.get('amplitude_absolute')) for m in metrics_dict
                        if m.get('peak_idx') is not None and m.get('amplitude_absolute') is not None]

            if len(peak_amps) == 0:
                print(f"[peak-navigator] Sweep {s}: No peaks with amplitude_absolute")
                continue

            # Calculate 90th percentile threshold for this sweep
            amps = [a for _, a in peak_amps]
            amp_90th = np.percentile(amps, 90)

            count_before = len(self.candidate_peaks)
            # Add peaks above 90th percentile
            for peak_idx, amp in peak_amps:
                if amp >= amp_90th:
                    self.candidate_peaks.append((s, peak_idx))

            count_added = len(self.candidate_peaks) - count_before
            print(f"[peak-navigator] Sweep {s}: Added {count_added} sigh candidates (threshold={amp_90th:.2f})")

        # Sort by amplitude (descending) - largest first
        self._sort_by_metric('amplitude_absolute', reverse=True, sweeps=sweeps)

    def _sort_by_metric(self, metric_key, reverse=False, sweeps=None):
        """Sort candidate_peaks by a metric value."""
        # Build a list of (sweep, peak_idx, metric_value) tuples
        items = []
        for s, peak_idx in self.candidate_peaks:
            metrics_dict = None
            if hasattr(self.state, 'current_peak_metrics_by_sweep') and s in self.state.current_peak_metrics_by_sweep:
                metrics_dict = self.state.current_peak_metrics_by_sweep[s]
            elif hasattr(self.state, 'peak_metrics_by_sweep') and s in self.state.peak_metrics_by_sweep:
                metrics_dict = self.state.peak_metrics_by_sweep[s]

            if metrics_dict:
                m = next((m for m in metrics_dict if m.get('peak_idx') == peak_idx), None)
                if m and metric_key in m:
                    items.append((s, peak_idx, m[metric_key]))

        # Sort by metric value
        items.sort(key=lambda x: x[2], reverse=reverse)

        # Update candidate_peaks
        self.candidate_peaks = [(s, peak_idx) for s, peak_idx, _ in items]

    def _update_ui(self):
        """Update UI elements based on current state."""
        n_candidates = len(self.candidate_peaks)

        # Update statistics
        if self.current_sweep_only.isChecked():
            self.stats_label.setText(f"Found: {n_candidates} candidates in sweep {self.state.sweep_idx}")
        else:
            self.stats_label.setText(f"Found: {n_candidates} candidates across all sweeps")

        # Update position label and buttons
        if n_candidates > 0:
            self.position_label.setText(f"Peak {self.current_index + 1} of {n_candidates}")
            self.prev_button.setEnabled(self.current_index > 0)
            self.next_button.setEnabled(self.current_index < n_candidates - 1)
        else:
            self.position_label.setText("Peak - of -")
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)

    def _navigate_prev(self):
        """Navigate to previous candidate."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_ui()
            self._jump_to_current_candidate()

    def _navigate_next(self):
        """Navigate to next candidate."""
        if self.current_index < len(self.candidate_peaks) - 1:
            self.current_index += 1
            self._update_ui()
            self._jump_to_current_candidate()

    def _jump_to_current_candidate(self):
        """Jump main window view to the current candidate peak."""
        if self.current_index >= len(self.candidate_peaks):
            return

        sweep_idx, peak_sample_idx = self.candidate_peaks[self.current_index]

        if not self.main_window:
            return

        # Switch to the correct sweep if needed
        if self.state.sweep_idx != sweep_idx:
            self.state.sweep_idx = sweep_idx
            # Refresh omit button label (like NavigationManager does)
            if hasattr(self.main_window, '_refresh_omit_button_label'):
                self.main_window._refresh_omit_button_label()
            # Redraw to show new sweep
            self.main_window.redraw_main_plot()

        # Use the dialog's window duration setting
        window_dur = self.window_duration.value()

        # Center the window on this peak
        peak_time = peak_sample_idx / self.state.sr_hz

        # Center window on peak
        new_window_start = max(0.0, peak_time - window_dur / 2.0)

        # Ensure we don't go past the end
        t_max = self.state.t[-1] if self.state.t is not None else 0
        if new_window_start + window_dur > t_max:
            new_window_start = max(0.0, t_max - window_dur)

        # Set the window using NavigationManager's _set_window() method
        if hasattr(self.main_window, 'navigation_manager'):
            self.main_window.navigation_manager._set_window(new_window_start, window_dur)

        print(f"[peak-navigator] Jumped to peak at {peak_time:.2f}s in sweep {sweep_idx} (window: {window_dur}s)")
