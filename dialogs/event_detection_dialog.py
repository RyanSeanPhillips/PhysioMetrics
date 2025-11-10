"""
Event Detection Settings Dialog for PhysioMetrics.

This dialog provides tools for detecting and marking events in the event channel:
- Automatic threshold-based detection
- Manual region marking (click-and-drag like Mark Sniff)
- Event merging based on minimum gap time
- Region deletion (Shift+click)
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QGroupBox, QCheckBox, QMessageBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt
import numpy as np
from scipy.ndimage import gaussian_filter1d


class EventDetectionDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Event Detection Settings")

        # Main layout
        main_layout = QVBoxLayout(self)

        # Title
        title = QLabel("Event Detection & Marking")
        title.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Info label
        info = QLabel(f"Event Channel: {main_window.state.event_channel}")
        info.setStyleSheet("color: #B0B0B0; margin-bottom: 15px;")
        main_layout.addWidget(info)

        # === Automatic Detection Section ===
        auto_group = QGroupBox("Automatic Event Detection")
        auto_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        auto_layout = QVBoxLayout(auto_group)

        # Detection mode radio buttons
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detection Mode:"))

        self.detection_mode_group = QButtonGroup(self)

        self.default_radio = QRadioButton("Default")
        self.default_radio.setChecked(True)
        self.default_radio.setToolTip("Default threshold detection settings")
        self.default_radio.toggled.connect(self.on_default_mode_selected)
        self.detection_mode_group.addButton(self.default_radio)
        mode_layout.addWidget(self.default_radio)

        self.licking_radio = QRadioButton("Licking")
        self.licking_radio.setToolTip("Licking detection - same as Default")
        self.licking_radio.toggled.connect(self.on_licking_mode_selected)
        self.detection_mode_group.addButton(self.licking_radio)
        mode_layout.addWidget(self.licking_radio)

        self.hargreaves_radio = QRadioButton("Hargreaves")
        self.hargreaves_radio.setToolTip("Thermal sensitivity detection - finds onset where signal leaves noise (< 15s)")
        self.hargreaves_radio.toggled.connect(self.on_hargreaves_mode_selected)
        self.detection_mode_group.addButton(self.hargreaves_radio)
        mode_layout.addWidget(self.hargreaves_radio)

        self.last_used_radio = QRadioButton("Last Used")
        self.last_used_radio.setToolTip("Restore previously used settings")
        self.last_used_radio.toggled.connect(self.on_last_used_mode_selected)
        self.detection_mode_group.addButton(self.last_used_radio)
        mode_layout.addWidget(self.last_used_radio)

        mode_layout.addStretch()
        auto_layout.addLayout(mode_layout)

        # Create two-column layout for parameters
        params_layout = QHBoxLayout()

        # Left column
        left_column = QVBoxLayout()
        params_layout.addLayout(left_column)

        # Right column
        right_column = QVBoxLayout()
        params_layout.addLayout(right_column)

        # LEFT COLUMN - Threshold and Duration
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-1000, 1000)
        self.threshold_spin.setValue(0.5)
        self.threshold_spin.setSingleStep(0.1)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setToolTip("Events detected when signal crosses this threshold")
        self.threshold_spin.valueChanged.connect(self.on_threshold_changed)
        thresh_layout.addWidget(self.threshold_spin)
        thresh_layout.addStretch()
        left_column.addLayout(thresh_layout)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Min Duration (s):"))
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.001, 100)
        self.min_duration_spin.setValue(0.050)
        self.min_duration_spin.setSingleStep(0.01)
        self.min_duration_spin.setDecimals(3)
        self.min_duration_spin.setToolTip("Minimum time signal must stay above threshold")
        dur_layout.addWidget(self.min_duration_spin)
        dur_layout.addStretch()
        left_column.addLayout(dur_layout)

        # RIGHT COLUMN - Gap and Checkboxes
        gap_layout = QHBoxLayout()
        gap_layout.addWidget(QLabel("Min Gap (s):"))
        self.min_gap_spin = QDoubleSpinBox()
        self.min_gap_spin.setRange(0, 100)
        self.min_gap_spin.setValue(1.0)
        self.min_gap_spin.setSingleStep(0.05)
        self.min_gap_spin.setDecimals(3)
        self.min_gap_spin.setToolTip("Events closer than this will be merged into one")
        gap_layout.addWidget(self.min_gap_spin)
        gap_layout.addStretch()
        right_column.addLayout(gap_layout)

        self.auto_merge_check = QCheckBox("Auto-merge events")
        self.auto_merge_check.setChecked(True)
        self.auto_merge_check.setToolTip("Automatically merge close events after detection")
        right_column.addWidget(self.auto_merge_check)

        auto_layout.addLayout(params_layout)

        # Checkboxes row (full width)
        checkboxes_layout = QHBoxLayout()

        self.shade_events_check = QCheckBox("Shade regions")
        self.shade_events_check.setChecked(False)
        self.shade_events_check.setToolTip("Show cyan shaded regions for detected events")
        self.shade_events_check.stateChanged.connect(self.on_shading_changed)
        checkboxes_layout.addWidget(self.shade_events_check)

        self.snap_markers_check = QCheckBox("Snap markers")
        self.snap_markers_check.setChecked(False)
        self.snap_markers_check.setToolTip(
            "Automatically adjust event boundaries:\n"
            "• Onset: Snap to nearest minimum before detected point\n"
            "• Offset: Snap to nearest threshold crossing after detected point"
        )
        checkboxes_layout.addWidget(self.snap_markers_check)

        self.show_labels_check = QCheckBox("Show labels")
        self.show_labels_check.setChecked(False)  # Default to OFF (Hargreaves mode overrides this)
        self.show_labels_check.setToolTip("Show time and duration labels on event markers")
        self.show_labels_check.stateChanged.connect(self.on_labels_changed)
        checkboxes_layout.addWidget(self.show_labels_check)

        checkboxes_layout.addStretch()
        auto_layout.addLayout(checkboxes_layout)

        # Detect button
        detect_btn = QPushButton("Detect Events")
        detect_btn.clicked.connect(self.on_detect_clicked)
        detect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        auto_layout.addWidget(detect_btn)

        main_layout.addWidget(auto_group)

        # === Manual Marking Instructions ===
        instructions_group = QGroupBox("Manual Event Marking (Active)")
        instructions_group.setStyleSheet("QGroupBox { font-weight: bold; color: #1abc9c; }")
        instructions_layout = QVBoxLayout(instructions_group)

        instructions = QLabel(
            "Manual marking is automatically enabled while this dialog is open:\n\n"
            "• Click and drag to create new event regions\n"
            "• Click near region edges (green/red lines) to drag and adjust them\n"
            "• Dragging edges to overlap will merge regions\n"
            "• Shift+Click inside a region to delete it"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #A0A0A0; padding: 5px;")
        instructions_layout.addWidget(instructions)

        main_layout.addWidget(instructions_group)

        # === Management Section ===
        mgmt_group = QGroupBox("Event Management")
        mgmt_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        mgmt_layout = QHBoxLayout(mgmt_group)

        # Clear button
        clear_btn = QPushButton("Clear All Events")
        clear_btn.clicked.connect(self.on_clear_clicked)
        clear_btn.setStyleSheet("background-color: #e74c3c; color: white; padding: 8px;")
        mgmt_layout.addWidget(clear_btn)

        # Event count label
        self.event_count_label = QLabel("Events: 0")
        self.event_count_label.setStyleSheet("font-weight: normal; color: #A0A0A0;")
        mgmt_layout.addWidget(self.event_count_label)
        mgmt_layout.addStretch()

        main_layout.addWidget(mgmt_group)

        # Update event count
        self.update_event_count()

        # Auto-enable manual marking mode
        self._enable_marking_mode()

        # Dialog buttons
        main_layout.addStretch()
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setDefault(True)
        button_layout.addWidget(close_btn)

        main_layout.addLayout(button_layout)

        # Resize dialog to minimum size that fits all content
        self.adjustSize()

    def on_threshold_changed(self, value):
        """Redraw plot when threshold changes to update threshold line."""
        self.main_window.redraw_main_plot()

    def on_shading_changed(self, state):
        """Redraw plot when shading checkbox changes."""
        self.main_window.redraw_main_plot()

    def on_labels_changed(self, state):
        """Redraw plot when labels checkbox changes."""
        self.main_window.redraw_main_plot()

    def on_default_mode_selected(self, checked):
        """Reset parameters to default detection settings."""
        if checked:
            self.threshold_spin.setValue(0.5)
            self.min_duration_spin.setValue(0.050)
            self.min_gap_spin.setValue(1.0)
            self.show_labels_check.setChecked(False)  # Labels OFF for Default mode
            print("[event-detection] Switched to Default mode")

    def on_licking_mode_selected(self, checked):
        """Reset parameters to licking detection defaults (same as default)."""
        if checked:
            self.threshold_spin.setValue(0.5)
            self.min_duration_spin.setValue(0.050)
            self.min_gap_spin.setValue(1.0)
            self.show_labels_check.setChecked(False)  # Labels OFF for Licking mode
            print("[event-detection] Switched to Licking mode")

    def on_hargreaves_mode_selected(self, checked):
        """Keep current parameters for Hargreaves mode."""
        if checked:
            self.show_labels_check.setChecked(True)  # Labels ON for Hargreaves mode
            print("[event-detection] Switched to Hargreaves mode")

    def on_last_used_mode_selected(self, checked):
        """Restore last used settings."""
        if checked:
            self._restore_last_used_settings()
            print("[event-detection] Restored last used settings")

    def _save_current_settings(self):
        """Save current settings to main window state for later recall."""
        self.main_window._event_detection_last_settings = {
            'threshold': self.threshold_spin.value(),
            'min_duration': self.min_duration_spin.value(),
            'min_gap': self.min_gap_spin.value(),
            'auto_merge': self.auto_merge_check.isChecked(),
            'shade': self.shade_events_check.isChecked(),
            'snap': self.snap_markers_check.isChecked(),
            'labels': self.show_labels_check.isChecked()
        }

    def _restore_last_used_settings(self):
        """Restore previously saved settings."""
        if not hasattr(self.main_window, '_event_detection_last_settings'):
            # No saved settings - use defaults
            self.on_default_mode_selected(True)
            return

        settings = self.main_window._event_detection_last_settings
        self.threshold_spin.setValue(settings.get('threshold', 0.5))
        self.min_duration_spin.setValue(settings.get('min_duration', 0.050))
        self.min_gap_spin.setValue(settings.get('min_gap', 1.0))
        self.auto_merge_check.setChecked(settings.get('auto_merge', True))
        self.shade_events_check.setChecked(settings.get('shade', False))
        self.snap_markers_check.setChecked(settings.get('snap', False))
        self.show_labels_check.setChecked(settings.get('labels', True))

    def _enable_marking_mode(self):
        """Enable manual event marking mode automatically."""
        # Turn off all other editing modes
        self.main_window.editing_modes.turn_off_all_edit_modes()

        # Enable event marking mode
        from editing.event_marking_mode import enable_event_marking
        enable_event_marking(self.main_window, self)

        print("[event-marking] Manual marking mode auto-enabled (dialog opened)")

    def _disable_marking_mode(self):
        """Disable manual event marking mode."""
        from editing.event_marking_mode import disable_event_marking
        disable_event_marking(self.main_window)

        print("[event-marking] Manual marking mode disabled (dialog closed)")

    def on_detect_clicked(self):
        """Detect events automatically based on threshold across ALL sweeps."""
        threshold = self.threshold_spin.value()
        min_duration = self.min_duration_spin.value()
        st = self.main_window.state

        if st.event_channel not in st.sweeps:
            QMessageBox.warning(self, "Error", "Event channel data not found")
            return

        # Get event channel data (all sweeps)
        event_data_all = st.sweeps[st.event_channel]  # Shape: (n_samples, n_sweeps)
        t = st.t
        n_sweeps = event_data_all.shape[1]

        # Check if Hargreaves mode is selected
        hargreaves_mode = self.hargreaves_radio.isChecked()

        # Clear ALL existing events
        st.bout_annotations.clear()

        total_events_detected = 0

        # Process each sweep
        for swp in range(n_sweeps):
            event_data = event_data_all[:, swp]

            # Detect threshold crossings
            above_thresh = event_data > threshold
            crossings = np.diff(above_thresh.astype(int))

            # Find start and end indices
            starts = np.where(crossings == 1)[0] + 1  # Rising edge
            ends = np.where(crossings == -1)[0] + 1   # Falling edge

            # Handle edge cases
            if above_thresh[0]:
                starts = np.concatenate([[0], starts])
            if above_thresh[-1]:
                ends = np.concatenate([ends, [len(above_thresh)]])

            # Apply Hargreaves minimum-finding if enabled
            if hargreaves_mode and len(starts) > 0:
                starts = self._find_hargreaves_onsets(event_data, t, starts, threshold)

            # Filter by minimum duration
            events = []
            for start_idx, end_idx in zip(starts, ends):
                duration = t[end_idx] - t[start_idx]
                if duration >= min_duration:
                    # Additional Hargreaves constraint: onset must be before 15s
                    if hargreaves_mode and t[start_idx] >= 15.0:
                        continue  # Skip events starting at or after 15s

                    events.append((float(t[start_idx]), float(t[end_idx])))

            # Store events for this sweep
            if events:
                st.bout_annotations[swp] = []
                for start_time, end_time in events:
                    bout = {
                        'start_time': start_time,
                        'end_time': end_time,
                        'id': len(st.bout_annotations[swp]) + 1
                    }
                    st.bout_annotations[swp].append(bout)

                total_events_detected += len(events)

                # Auto-merge if enabled
                if self.auto_merge_check.isChecked():
                    self.merge_events(swp)

        print(f"[event-detection] Detected {total_events_detected} events across {n_sweeps} sweeps")

        # Update display
        self.main_window.redraw_main_plot()
        self.update_event_count()

    def _find_hargreaves_onsets(self, signal, t, detected_starts, threshold):
        """
        Find the first point where signal leaves noise level before threshold crossing.

        For Hargreaves thermal sensitivity testing, the true onset is the point
        where the temperature first starts rising above the baseline noise level.

        Args:
            signal: Event channel data (1D array)
            t: Time array
            detected_starts: Indices of threshold crossings (rising edges)
            threshold: Detection threshold

        Returns:
            Adjusted start indices (where signal first leaves noise level)
        """
        adjusted_starts = []
        max_lookback_sec = 1.0  # Search 1 second before threshold crossing
        sr_hz = self.main_window.state.sr_hz
        max_lookback_idx = int(max_lookback_sec * sr_hz)

        for start_idx in detected_starts:
            # Define search window (backward from threshold crossing)
            search_start = max(0, start_idx - max_lookback_idx)
            search_end = start_idx

            if search_end <= search_start:
                adjusted_starts.append(start_idx)  # Fallback to original
                continue

            # Get signal in search window
            signal_window = signal[search_start:search_end]

            # Estimate baseline noise level (use early portion of window as baseline)
            baseline_samples = min(int(0.2 * len(signal_window)), len(signal_window) // 2)
            if baseline_samples > 10:
                baseline_region = signal_window[:baseline_samples]
                baseline_mean = np.mean(baseline_region)
                baseline_std = np.std(baseline_region)
                noise_threshold = baseline_mean + 3 * baseline_std  # 3 sigma above baseline
            else:
                # Fallback: use median and MAD of entire window
                baseline_mean = np.median(signal_window)
                mad = np.median(np.abs(signal_window - baseline_mean))
                noise_threshold = baseline_mean + 3 * mad * 1.4826  # Convert MAD to std equivalent

            # Search backward from threshold crossing to find where signal first leaves noise
            # Start from the threshold crossing and work backward
            onset_idx_abs = None
            for i in range(len(signal_window) - 1, -1, -1):
                idx_abs = search_start + i

                # Check if signal is significantly above noise level
                if signal[idx_abs] > noise_threshold:
                    # Signal is elevated - keep going backward to find the START
                    onset_idx_abs = idx_abs
                else:
                    # Signal is at noise level - we've found the transition point
                    if onset_idx_abs is not None:
                        # We just transitioned from elevated to baseline
                        # The onset is at onset_idx_abs (first elevated point)
                        break

            if onset_idx_abs is not None and onset_idx_abs < start_idx:
                # Verify the signal at this point is below threshold
                if signal[onset_idx_abs] < threshold:
                    adjusted_starts.append(onset_idx_abs)
                    print(f"[hargreaves] Adjusted onset from {t[start_idx]:.3f}s to {t[onset_idx_abs]:.3f}s (signal left noise at {signal[onset_idx_abs]:.3f}, noise threshold: {noise_threshold:.3f})")
                else:
                    adjusted_starts.append(start_idx)  # Fallback
            else:
                # Fallback: use the point where signal value is lowest in the window
                min_idx_rel = np.argmin(signal_window)
                min_idx_abs = search_start + min_idx_rel

                if signal[min_idx_abs] < threshold:
                    adjusted_starts.append(min_idx_abs)
                    print(f"[hargreaves] Adjusted onset from {t[start_idx]:.3f}s to {t[min_idx_abs]:.3f}s (lowest point fallback)")
                else:
                    adjusted_starts.append(start_idx)  # Fallback

        return np.array(adjusted_starts)

    def merge_events(self, sweep_idx):
        """Merge events that are closer than min_gap."""
        st = self.main_window.state
        min_gap = self.min_gap_spin.value()

        if sweep_idx not in st.bout_annotations or not st.bout_annotations[sweep_idx]:
            return

        # Sort events by start time
        events = sorted(st.bout_annotations[sweep_idx], key=lambda x: x['start_time'])

        # Merge overlapping or close events
        merged = []
        current = events[0].copy()

        for event in events[1:]:
            gap = event['start_time'] - current['end_time']
            if gap <= min_gap:
                # Merge by extending current event
                current['end_time'] = max(current['end_time'], event['end_time'])
                print(f"[event-merge] Merged events: gap was {gap:.3f}s")
            else:
                # No merge - save current and start new
                merged.append(current)
                current = event.copy()

        # Add the last event
        merged.append(current)

        # Reassign IDs
        for i, event in enumerate(merged):
            event['id'] = i + 1

        # Update state
        st.bout_annotations[sweep_idx] = merged
        print(f"[event-merge] Result: {len(merged)} events after merging")

    def on_clear_clicked(self):
        """Clear all events from current sweep."""
        reply = QMessageBox.question(
            self,
            "Clear Events",
            "Clear all events from the current sweep?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            st = self.main_window.state
            swp = st.sweep_idx

            if swp in st.bout_annotations:
                del st.bout_annotations[swp]
                print(f"[event-clear] Cleared all events from sweep {swp}")

            self.main_window.redraw_main_plot()
            self.update_event_count()

    def update_event_count(self):
        """Update the event count label to show current sweep and total."""
        st = self.main_window.state
        swp = st.sweep_idx

        # Count for current sweep
        current_count = 0
        if swp in st.bout_annotations:
            current_count = len(st.bout_annotations[swp])

        # Total across all sweeps
        total_count = sum(len(events) for events in st.bout_annotations.values())

        self.event_count_label.setText(f"Events: {current_count} (sweep {swp+1}) | Total: {total_count}")

    def closeEvent(self, event):
        """Clean up when dialog is closed."""
        # Save current settings for "Last Used" mode
        self._save_current_settings()

        # Disable marking mode
        self._disable_marking_mode()

        event.accept()
