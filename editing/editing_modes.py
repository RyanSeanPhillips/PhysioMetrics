"""
EditingModes class for PhysioMetrics.

Handles all interactive editing modes:
- Add/Delete Peaks
- Add/Delete Sighs
- Move Points (peaks, onsets, offsets, expmins, expoffs)
- Mark Sniffing Regions
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


class EditingModes:
    """Manages all editing mode states and operations for the main window."""

    def __init__(self, parent_window):
        """
        Initialize editing modes manager.

        Args:
            parent_window: Reference to MainWindow instance
        """
        self.window = parent_window

        # Mode flags
        self._add_peaks_mode = False
        self._delete_peaks_mode = False
        self._add_sigh_mode = False
        self._move_point_mode = False
        self._mark_sniff_mode = False

        # Move Point mode state
        self._selected_point = None  # {'type': 'peak'|'onset'|'offset'|'expmin'|'expoff', 'index': int, 'sweep': int, 'original_index': int}
        self._move_point_artist = None  # Visual marker for selected point
        self._key_press_cid = None  # Connection ID for matplotlib key events
        self._motion_cid = None  # Connection ID for mouse motion events
        self._release_cid = None  # Connection ID for mouse release events

        # Mark Sniff mode state
        self._sniff_start_x = None  # X-coordinate where drag started
        self._sniff_drag_artist = None  # Visual indicator while dragging
        self._sniff_artists = []  # Matplotlib artists for sniff overlays
        self._sniff_edge_mode = None  # 'start' or 'end' if dragging an edge, None if creating new region
        self._sniff_region_index = None  # Index of region being edited

        # Omit Region mode state
        self._omit_region_mode = False
        self._omit_region_remove_mode = False  # True for removing regions, False for adding
        self._omit_region_start_x = None  # X-coordinate where drag started
        self._omit_region_drag_artist = None  # Visual indicator while dragging
        self._omit_edge_mode = None  # 'start' or 'end' if dragging an edge, None if creating new region
        self._omit_region_index = None  # Index of region being edited
        self._key_cid = None  # Connection ID for matplotlib key press events (omit mode)

        # Peak editing window size
        self._peak_edit_half_win_s = 0.08  # ±80ms window for peak operations

        # Connect button signals
        self._connect_buttons()

    def _connect_buttons(self):
        """Connect all editing mode button signals."""
        # Set buttons as checkable
        self.window.addPeaksButton.setCheckable(True)
        self.window.deletePeaksButton.setCheckable(True)
        self.window.addSighButton.setCheckable(True)
        self.window.movePointButton.setCheckable(True)
        self.window.markSniffButton.setCheckable(True)

        # Connect signals
        self.window.addPeaksButton.toggled.connect(self.on_add_peaks_toggled)
        self.window.deletePeaksButton.toggled.connect(self.on_delete_peaks_toggled)
        self.window.addSighButton.toggled.connect(self.on_add_sigh_toggled)
        self.window.movePointButton.toggled.connect(self.on_move_point_toggled)
        self.window.markSniffButton.toggled.connect(self.on_mark_sniff_toggled)

    def handle_key_press_event(self, event) -> bool:
        """
        Handle keyboard events for editing modes.

        Args:
            event: QKeyEvent from MainWindow

        Returns:
            True if event was handled, False otherwise
        """
        # Note: Omit region mode uses matplotlib canvas key events (spacebar)
        # not Qt key events, so no handler needed here for snap functionality

        # Move point mode handlers
        if self._move_point_mode and self._selected_point:
            if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
                # Move point left or right
                direction = -1 if event.key() == Qt.Key.Key_Left else 1
                # Check if Shift is held for snap-to-zero-crossing
                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self._move_selected_point(direction, snap_to_zero=shift_held)
                return True
            elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                # Save the moved point
                self._save_moved_point()
                return True
            elif event.key() == Qt.Key.Key_Escape:
                # Cancel move
                self._cancel_move_point()
                return True

        return False

    # ========== Canvas Event Handlers (for matplotlib) ==========

    def _on_canvas_key_press(self, event):
        """Handle matplotlib canvas key events for move point mode."""
        if not self._move_point_mode or not self._selected_point:
            return

        if event.key in ('left', 'right'):
            # Move point left or right
            direction = -1 if event.key == 'left' else 1
            self._move_selected_point(direction)
        elif event.key in ('enter', 'return'):
            # Save the moved point
            self._save_moved_point()
        elif event.key == 'escape':
            # Cancel move
            self._cancel_move_point()

    def _on_canvas_motion(self, event):
        """Handle mouse motion for click-and-drag point movement."""
        if not self._move_point_mode:
            return

        # Toolbar already disabled when entering move mode

        if not self._selected_point:
            return

        # Only move if mouse button is pressed (dragging)
        if event.button != 1 or event.xdata is None or event.inaxes is None:
            return

        st = self.window.state
        s = self._selected_point['sweep']
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        # Get time basis
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # Find closest sample to mouse position
        new_idx = int(np.clip(np.searchsorted(t_plot, float(event.xdata)), 0, len(t_plot) - 1))

        # Constrain movement between adjacent peaks
        new_idx = self._constrain_to_peak_boundaries(new_idx, s)

        # Check if Shift key is held for snap-to-zero-crossing
        shift_held = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

        if shift_held:
            dy = np.gradient(y)
            new_idx = self._find_nearest_zero_crossing(y, dy, new_idx, search_radius=200)
            new_idx = self._constrain_to_peak_boundaries(new_idx, s)  # Re-constrain after snap

        # Update the point to new position
        self._update_point_position(new_idx, t_plot, y, s)

    def _on_canvas_release(self, event):
        """Handle mouse release - auto-save the moved point."""
        if not self._move_point_mode or not self._selected_point:
            return

        if event.button == 1:  # Left click release
            # Auto-save and recompute metrics
            self._save_moved_point(recompute_metrics=True)

    # ========== Turn Off All Edit Modes ==========

    def turn_off_all_edit_modes(self):
        """Turn off all edit modes (add/delete peaks, add sigh, move point, mark sniff)."""
        # Turn off Add Peaks mode
        if getattr(self, "_add_peaks_mode", False):
            self._add_peaks_mode = False
            self.window.addPeaksButton.blockSignals(True)
            self.window.addPeaksButton.setChecked(False)
            self.window.addPeaksButton.blockSignals(False)
            self.window.addPeaksButton.setText("Add Peaks")

        # Turn off Delete Peaks mode
        if getattr(self, "_delete_peaks_mode", False):
            self._delete_peaks_mode = False
            self.window.deletePeaksButton.blockSignals(True)
            self.window.deletePeaksButton.setChecked(False)
            self.window.deletePeaksButton.blockSignals(False)
            self.window.deletePeaksButton.setText("Delete Peaks")

        # Turn off Add Sigh mode
        if getattr(self, "_add_sigh_mode", False):
            self._add_sigh_mode = False
            self.window.addSighButton.blockSignals(True)
            self.window.addSighButton.setChecked(False)
            self.window.addSighButton.blockSignals(False)
            self.window.addSighButton.setText("ADD/DEL Sigh")

        # Turn off Move Point mode
        if getattr(self, "_move_point_mode", False):
            self._move_point_mode = False
            self.window.movePointButton.blockSignals(True)
            self.window.movePointButton.setChecked(False)
            self.window.movePointButton.blockSignals(False)
            self.window.movePointButton.setText("Move Point")
            # Clean up any selected point visualization
            if self._move_point_artist:
                self._move_point_artist.remove()
                self._move_point_artist = None
            self._selected_point = None

        # Turn off Mark Sniff mode
        if getattr(self, "_mark_sniff_mode", False):
            self._mark_sniff_mode = False
            self.window.markSniffButton.blockSignals(True)
            self.window.markSniffButton.setChecked(False)
            self.window.markSniffButton.blockSignals(False)
            self.window.markSniffButton.setText("Mark Sniff")
            # Disconnect matplotlib events if connected
            if self._motion_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._motion_cid)
                self._motion_cid = None
            if self._release_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._release_cid)
                self._release_cid = None
            # Clear drag state
            if self._sniff_drag_artist:
                self._sniff_drag_artist.remove()
                self._sniff_drag_artist = None
            self._sniff_start_x = None
            self._sniff_edge_mode = None
            self._sniff_region_index = None

        # Clear click callback and reset cursor
        self.window.plot_host.clear_click_callback()
        self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    # Alias for backward compatibility
    _turn_off_all_edit_modes = turn_off_all_edit_modes

    # ========== ADD PEAKS MODE ==========

    def on_add_peaks_toggled(self, checked: bool):
        """Enter/exit Add Peaks mode, mutually exclusive with other modes."""
        self._add_peaks_mode = checked

        if checked:
            # Turn off matplotlib toolbar modes (zoom, pan)
            self.window.plot_host.turn_off_toolbar_modes()

            # turn OFF delete mode
            if getattr(self, "_delete_peaks_mode", False):
                self._delete_peaks_mode = False
                self.window.deletePeaksButton.blockSignals(True)
                self.window.deletePeaksButton.setChecked(False)
                self.window.deletePeaksButton.blockSignals(False)
                self.window.deletePeaksButton.setText("Delete Peaks")

            # turn OFF sigh mode + reset its label
            if getattr(self, "_add_sigh_mode", False):
                self._add_sigh_mode = False
                self.window.addSighButton.blockSignals(True)
                self.window.addSighButton.setChecked(False)
                self.window.addSighButton.blockSignals(False)
                self.window.addSighButton.setText("ADD/DEL Sigh")

            # turn OFF move point mode
            if getattr(self, "_move_point_mode", False):
                self._move_point_mode = False
                self.window.movePointButton.blockSignals(True)
                self.window.movePointButton.setChecked(False)
                self.window.movePointButton.blockSignals(False)
                self.window.movePointButton.setText("Move Point")

            # turn OFF mark sniff mode
            if getattr(self, "_mark_sniff_mode", False):
                self._mark_sniff_mode = False
                self.window.markSniffButton.blockSignals(True)
                self.window.markSniffButton.setChecked(False)
                self.window.markSniffButton.blockSignals(False)
                self.window.markSniffButton.setText("Mark Sniff")

            self.window.addPeaksButton.setText("Add Peaks (ON) [Shift=Del, Ctrl=Sigh]")
            self.window.plot_host.set_click_callback(self._on_plot_click_add_peak)
            self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.window.addPeaksButton.setText("Add Peaks")
            if not getattr(self, "_delete_peaks_mode", False) and not getattr(self, "_add_sigh_mode", False):
                self.window.plot_host.clear_click_callback()
                self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _compute_single_breath_events(self, y, peak_idx, prev_peak_idx, next_peak_idx, sr_hz):
        """
        Compute breath events for a single peak.

        Args:
            y: Signal array
            peak_idx: Index of the peak to compute events for
            prev_peak_idx: Index of previous peak (or 0 if first)
            next_peak_idx: Index of next peak (or len(y)-1 if last)
            sr_hz: Sample rate

        Returns:
            dict with keys: onset, offset, expmin, expoff (single values, not arrays)
        """
        from core import peaks as peakdet

        # Use the same logic as compute_breath_events but for a single peak
        # Call compute_breath_events with a 3-peak array (prev, current, next)
        # and extract the middle result

        # Build a minimal peaks array with context
        temp_peaks = []
        indices_map = {}  # Map temp array index to which peak it is

        if prev_peak_idx is not None and prev_peak_idx != peak_idx:
            temp_peaks.append(prev_peak_idx)
            indices_map[len(temp_peaks)-1] = 'prev'

        temp_peaks.append(peak_idx)
        target_idx = len(temp_peaks) - 1  # Index of our peak in temp array
        indices_map[target_idx] = 'target'

        if next_peak_idx is not None and next_peak_idx != peak_idx:
            temp_peaks.append(next_peak_idx)
            indices_map[len(temp_peaks)-1] = 'next'

        temp_peaks = np.array(temp_peaks, dtype=int)

        # Compute breath events for this minimal set
        breaths = peakdet.compute_breath_events(y, temp_peaks, sr_hz)

        # Extract the events for the target peak
        result = {}
        for key in ['onsets', 'offsets', 'expmins', 'expoffs']:
            if key in breaths and target_idx < len(breaths[key]):
                result[key] = int(breaths[key][target_idx])
            else:
                result[key] = None

        return result

    def _on_plot_click_add_peak(self, xdata, ydata, event, _force_mode=None):
        """Handle plot click to add a peak."""
        # Only when "Add Peaks (ON)" or force mode is 'add'
        if _force_mode != 'add' and not getattr(self, "_add_peaks_mode", False):
            return
        if event.inaxes is None or xdata is None:
            return

        # Check for modifier keys (but not if we're already in force mode to prevent recursion)
        if _force_mode is None:
            modifiers = QApplication.keyboardModifiers()
            shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            ctrl_held = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

            # Shift toggles to delete mode
            if shift_held:
                self._on_plot_click_delete_peak(xdata, ydata, event, _force_mode='delete')
                return

            # Ctrl switches to add sigh mode
            if ctrl_held:
                self._on_plot_click_add_sigh(xdata, ydata, event, _force_mode='sigh')
                return

        st = self.window.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        # Current sweep + processed trace (what user sees)
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        # Use the same time basis as the plot (normalized if stim)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # ±080 ms search window around click
        half_win_s = 0.08
        half_win_n = max(1, int(round(half_win_s * st.sr_hz)))
        i_center = int(np.clip(np.searchsorted(t_plot, float(xdata)), 0, len(t_plot) - 1))
        i0 = max(0, i_center - half_win_n)
        i1 = min(len(y) - 1, i_center + half_win_n)
        if i1 <= i0:
            return

        # Find local maximum (breathing signals always use upward peaks)
        seg = y[i0:i1 + 1]
        loc = int(np.argmax(seg))
        i_peak = i0 + loc

        # ---- NEW: enforce minimum separation from existing peaks ----
        # Use UI "MinPeakDistValue" if valid; fallback to 0.05 s
        try:
            sep_s = float(self.window.MinPeakDistValue.text().strip())
            if not (sep_s > 0):
                raise ValueError
        except Exception:
            sep_s = 0.05
        sep_n = max(1, int(round(sep_s * st.sr_hz)))

        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        if pks.size and np.any(np.abs(pks - i_peak) <= sep_n):
            print(f"[add-peak] Rejected: candidate within {sep_s:.3f}s of an existing peak.")
            self.window._log_status_message(f"✗ Peak too close (< {sep_s:.2f}s)", 2000)
            # Restore out-of-date warning after temporary message expires
            if getattr(self.window, 'eupnea_sniffing_out_of_date', False):
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2100, lambda: self.window._log_status_message("⚠️ Eupnea/sniffing detection out of date"))
            return

        # Insert, sort, and store
        pks_new = np.sort(np.append(pks, i_peak))
        new_idx = np.where(pks_new == i_peak)[0][0]  # Find index of newly added peak
        st.peaks_by_sweep[s] = pks_new

        # Surgically compute breath events for just this new peak
        prev_peak = pks_new[new_idx - 1] if new_idx > 0 else None
        next_peak = pks_new[new_idx + 1] if new_idx < len(pks_new) - 1 else None

        new_events = self._compute_single_breath_events(
            y, i_peak, prev_peak, next_peak, st.sr_hz
        )

        # Insert the new breath events at the correct position
        breaths = st.breath_by_sweep.get(s, {})
        if not breaths:
            # Initialize empty breath dict if it doesn't exist
            breaths = {'onsets': np.array([], dtype=int),
                      'offsets': np.array([], dtype=int),
                      'expmins': np.array([], dtype=int),
                      'expoffs': np.array([], dtype=int)}

        for key in ['onsets', 'offsets', 'expmins', 'expoffs']:
            arr = np.asarray(breaths.get(key, []), dtype=int)
            if new_events.get(key) is not None:
                # Insert at the correct position
                breaths[key] = np.insert(arr, new_idx, new_events[key])
            else:
                # Insert placeholder if computation failed
                breaths[key] = np.insert(arr, new_idx, 0)

        st.breath_by_sweep[s] = breaths
        print(f"[add-peak] Surgically added breath events at index {new_idx}")

        # Log telemetry: manual edit
        from core import telemetry
        telemetry.log_edit('add_peak',
                          num_peaks_after=len(pks_new),
                          sweep_index=s)

        # Show success message
        self.window._log_status_message("✓ Peak added", 1500)

        # Recompute Y2 metric if selected
        if getattr(st, "y2_metric_key", None):
            self.window._compute_y2_all_sweeps()

        # Refresh plot FIRST to show updated peaks
        self.window.redraw_main_plot()

        # Re-run GMM clustering to update sniffing regions (if auto-update enabled)
        if getattr(self.window, 'auto_gmm_enabled', False):
            self.window._run_automatic_gmm_clustering()
            # LIGHTWEIGHT UPDATE: Just refresh eupnea/sniffing overlays (no full redraw)
            self.window._refresh_eupnea_overlays_only()
            # Clear out-of-date flag and status bar
            self.window.eupnea_sniffing_out_of_date = False
            self.window.statusBar().clearMessage()
        else:
            # Mark eupnea/sniffing detection as out of date
            self.window.eupnea_sniffing_out_of_date = True
            # Show persistent warning (no timeout - stays until Update button is clicked)
            self.window._log_status_message("⚠️ Eupnea/sniffing detection out of date")

    # ========== DELETE PEAKS MODE ==========

    def on_delete_peaks_toggled(self, checked: bool):
        """Enter/exit Delete Peaks mode, mutually exclusive with other modes."""
        self._delete_peaks_mode = checked

        if checked:
            # Turn off matplotlib toolbar modes (zoom, pan)
            self.window.plot_host.turn_off_toolbar_modes()

            # turn OFF add mode
            if getattr(self, "_add_peaks_mode", False):
                self._add_peaks_mode = False
                self.window.addPeaksButton.blockSignals(True)
                self.window.addPeaksButton.setChecked(False)
                self.window.addPeaksButton.blockSignals(False)
                self.window.addPeaksButton.setText("Add Peaks")

            # turn OFF sigh mode + reset its label
            if getattr(self, "_add_sigh_mode", False):
                self._add_sigh_mode = False
                self.window.addSighButton.blockSignals(True)
                self.window.addSighButton.setChecked(False)
                self.window.addSighButton.blockSignals(False)
                self.window.addSighButton.setText("ADD/DEL Sigh")

            # turn OFF move point mode
            if getattr(self, "_move_point_mode", False):
                self._move_point_mode = False
                self.window.movePointButton.blockSignals(True)
                self.window.movePointButton.setChecked(False)
                self.window.movePointButton.blockSignals(False)
                self.window.movePointButton.setText("Move Point")

            # turn OFF mark sniff mode
            if getattr(self, "_mark_sniff_mode", False):
                self._mark_sniff_mode = False
                self.window.markSniffButton.blockSignals(True)
                self.window.markSniffButton.setChecked(False)
                self.window.markSniffButton.blockSignals(False)
                self.window.markSniffButton.setText("Mark Sniff")

            self.window.deletePeaksButton.setText("Delete Peaks (ON) [Shift=Add, Ctrl=Sigh]")
            self.window.plot_host.set_click_callback(self._on_plot_click_delete_peak)
            self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.window.deletePeaksButton.setText("Delete Peaks")
            if not getattr(self, "_add_peaks_mode", False) and not getattr(self, "_add_sigh_mode", False):
                self.window.plot_host.clear_click_callback()
                self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_plot_click_delete_peak(self, xdata, ydata, event, _force_mode=None):
        """Handle plot click to delete a peak."""
        # Only when "Delete Peaks (ON)" or force mode is 'delete'
        if _force_mode != 'delete' and (not getattr(self, "_delete_peaks_mode", False) or getattr(event, "button", 1) != 1):
            return
        if event.inaxes is None or xdata is None:
            return

        # Check for modifier keys (but not if we're already in force mode to prevent recursion)
        if _force_mode is None:
            modifiers = QApplication.keyboardModifiers()
            shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
            ctrl_held = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

            # Shift toggles to add mode
            if shift_held:
                self._on_plot_click_add_peak(xdata, ydata, event, _force_mode='add')
                return

            # Ctrl switches to add sigh mode
            if ctrl_held:
                self._on_plot_click_add_sigh(xdata, ydata, event, _force_mode='sigh')
                return

        st = self.window.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        # Current sweep & processed trace
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        # Match the plotted time basis (normalized if stim spans exist)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # Find the index corresponding to the click position
        i_click = int(np.clip(np.searchsorted(t_plot, float(xdata)), 0, len(t_plot) - 1))

        # Existing peaks for this sweep
        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        if pks.size == 0:
            print("[delete-peak] No peaks to delete in this sweep.")
            return

        # Find the closest peak to the click position
        distances = np.abs(pks - i_click)
        closest_idx = np.argmin(distances)
        closest_peak = pks[closest_idx]

        # Optional: Only delete if within reasonable distance (e.g., ±80ms window)
        half_win_s = float(getattr(self, "_peak_edit_half_win_s", 0.08))
        half_win_n = max(1, int(round(half_win_s * st.sr_hz)))

        if distances[closest_idx] > half_win_n:
            print(f"[delete-peak] Closest peak is too far ({distances[closest_idx]} samples > {half_win_n} samples).")
            self.window._log_status_message(f"✗ Click too far from peak (> {half_win_s:.2f}s)", 2000)
            # Restore out-of-date warning after temporary message expires
            if getattr(self.window, 'eupnea_sniffing_out_of_date', False):
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(2100, lambda: self.window._log_status_message("⚠️ Eupnea/sniffing detection out of date"))
            return

        # Delete only the closest peak
        pks_new = np.delete(pks, closest_idx)
        print(f"[delete-peak] Deleted peak at index {closest_peak} (distance: {distances[closest_idx]} samples)")

        # Log telemetry: manual edit
        from core import telemetry
        telemetry.log_edit('delete_peak',
                          num_peaks_after=len(pks_new),
                          sweep_index=s)

        self.window._log_status_message("✓ Peak deleted", 1500)
        st.peaks_by_sweep[s] = pks_new

        # Surgically delete corresponding breath events (no recomputation needed)
        breaths = st.breath_by_sweep.get(s, {})
        if breaths:
            # Delete entries at closest_idx from all breath event arrays
            for key in ['onsets', 'offsets', 'expmins', 'expoffs']:
                if key in breaths:
                    arr = np.asarray(breaths[key], dtype=int)
                    if closest_idx < len(arr):
                        breaths[key] = np.delete(arr, closest_idx)
            st.breath_by_sweep[s] = breaths
            print(f"[delete-peak] Surgically removed breath events at index {closest_idx}")

        # If a Y2 metric is selected, recompute
        if getattr(st, "y2_metric_key", None):
            self.window._compute_y2_all_sweeps()

        # Refresh plot FIRST to show updated peaks
        self.window.redraw_main_plot()

        # Re-run GMM clustering to update sniffing regions (if auto-update enabled)
        if getattr(self.window, 'auto_gmm_enabled', False):
            self.window._run_automatic_gmm_clustering()
            # LIGHTWEIGHT UPDATE: Just refresh eupnea/sniffing overlays (no full redraw)
            self.window._refresh_eupnea_overlays_only()
            # Clear out-of-date flag and status bar
            self.window.eupnea_sniffing_out_of_date = False
            self.window.statusBar().clearMessage()
        else:
            # Mark eupnea/sniffing detection as out of date
            self.window.eupnea_sniffing_out_of_date = True
            # Show persistent warning (no timeout - stays until Update button is clicked)
            self.window._log_status_message("⚠️ Eupnea/sniffing detection out of date")

    # ========== ADD SIGH MODE ==========

    def on_add_sigh_toggled(self, checked: bool):
        """Enter/exit Add Sigh mode, mutually exclusive with other modes."""
        self._add_sigh_mode = checked

        if checked:
            # Turn off matplotlib toolbar modes (zoom, pan)
            self.window.plot_host.turn_off_toolbar_modes()

            # turn OFF other modes (without triggering their slots)
            if getattr(self, "_add_peaks_mode", False):
                self._add_peaks_mode = False
                self.window.addPeaksButton.blockSignals(True)
                self.window.addPeaksButton.setChecked(False)
                self.window.addPeaksButton.blockSignals(False)
                self.window.addPeaksButton.setText("Add Peaks")

            if getattr(self, "_delete_peaks_mode", False):
                self._delete_peaks_mode = False
                self.window.deletePeaksButton.blockSignals(True)
                self.window.deletePeaksButton.setChecked(False)
                self.window.deletePeaksButton.blockSignals(False)
                self.window.deletePeaksButton.setText("Delete Peaks")

            # turn OFF move point mode
            if getattr(self, "_move_point_mode", False):
                self._move_point_mode = False
                self.window.movePointButton.blockSignals(True)
                self.window.movePointButton.setChecked(False)
                self.window.movePointButton.blockSignals(False)
                self.window.movePointButton.setText("Move Point")

            # turn OFF mark sniff mode
            if getattr(self, "_mark_sniff_mode", False):
                self._mark_sniff_mode = False
                self.window.markSniffButton.blockSignals(True)
                self.window.markSniffButton.setChecked(False)
                self.window.markSniffButton.blockSignals(False)
                self.window.markSniffButton.setText("Mark Sniff")

            self.window.addSighButton.setText("Add Sigh (ON)")
            self.window.plot_host.set_click_callback(self._on_plot_click_add_sigh)
            self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.window.addSighButton.setText("Add Sigh")
            # only clear if no other edit mode is active
            if not getattr(self, "_add_peaks_mode", False) and not getattr(self, "_delete_peaks_mode", False):
                self.window.plot_host.clear_click_callback()
                self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_plot_click_add_sigh(self, xdata, ydata, event, _force_mode=None):
        """Handle plot click to add or remove a sigh marker."""
        # Only in sigh mode or force mode is 'sigh'
        if _force_mode != 'sigh' and (not getattr(self, "_add_sigh_mode", False) or event.inaxes is None or xdata is None):
            return
        if event.inaxes is None or xdata is None:
            return

        st = self.window.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        # Current sweep + processed trace (what you're seeing)
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        # Use same time basis as the plot (normalized if stim)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # We ONLY snap to existing detected peaks
        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        if pks.size == 0:
            print("[sigh] No peaks detected in this sweep—cannot place a sigh.")
            return

        # Find the nearest PEAK index to the click
        i_click = int(np.clip(np.searchsorted(t_plot, float(xdata)), 0, len(t_plot) - 1))
        i_nearest_peak = int(pks[np.argmin(np.abs(pks - i_click))])

        # Toggle: add if absent, remove if present
        current = set(map(int, st.sigh_by_sweep.get(s, [])))
        if i_nearest_peak in current:
            current.remove(i_nearest_peak)
            print(f"[sigh] Removed sigh at peak index {i_nearest_peak} (t={t_plot[i_nearest_peak]:.3f}s)")
        else:
            current.add(i_nearest_peak)
            print(f"[sigh] Added sigh at peak index {i_nearest_peak} (t={t_plot[i_nearest_peak]:.3f}s)")

        st.sigh_by_sweep[s] = np.array(sorted(current), dtype=int)

        # Redraw to see star(s)
        self.window.redraw_main_plot()

    # ========== MOVE POINT MODE ==========

    def on_move_point_toggled(self, checked: bool):
        """Enter/exit Move Point mode; mutually exclusive with other edit modes."""
        self._move_point_mode = checked

        if checked:
            # Turn off matplotlib toolbar modes (zoom, pan)
            self.window.plot_host.turn_off_toolbar_modes()

            # Turn OFF other edit modes
            if getattr(self, "_add_peaks_mode", False):
                self._add_peaks_mode = False
                self.window.addPeaksButton.blockSignals(True)
                self.window.addPeaksButton.setChecked(False)
                self.window.addPeaksButton.blockSignals(False)
                self.window.addPeaksButton.setText("Add Peaks")

            if getattr(self, "_delete_peaks_mode", False):
                self._delete_peaks_mode = False
                self.window.deletePeaksButton.blockSignals(True)
                self.window.deletePeaksButton.setChecked(False)
                self.window.deletePeaksButton.blockSignals(False)
                self.window.deletePeaksButton.setText("Delete Peaks")

            if getattr(self, "_add_sigh_mode", False):
                self._add_sigh_mode = False
                self.window.addSighButton.blockSignals(True)
                self.window.addSighButton.setChecked(False)
                self.window.addSighButton.blockSignals(False)
                self.window.addSighButton.setText("ADD/DEL Sigh")

            # turn OFF mark sniff mode
            if getattr(self, "_mark_sniff_mode", False):
                self._mark_sniff_mode = False
                self.window.markSniffButton.blockSignals(True)
                self.window.markSniffButton.setChecked(False)
                self.window.markSniffButton.blockSignals(False)
                self.window.markSniffButton.setText("Mark Sniff")

            self.window.movePointButton.setText("Move Point (ON) [Shift=Snap]")
            self.window.plot_host.set_click_callback(self._on_plot_click_move_point)
            self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)

            # Connect matplotlib events
            self._key_press_cid = self.window.plot_host.canvas.mpl_connect('key_press_event', self._on_canvas_key_press)
            self._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', self._on_canvas_motion)
            self._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', self._on_canvas_release)

            # Disable matplotlib's built-in toolbar - turn off any active modes
            # The toolbar has a mode attribute we can check
            if hasattr(self.window.plot_host.toolbar, 'mode') and self.window.plot_host.toolbar.mode != '':
                # There's an active mode - turn it off by calling the same method again (toggle)
                if self.window.plot_host.toolbar.mode == 'pan/zoom':
                    self.window.plot_host.toolbar.pan()
                elif self.window.plot_host.toolbar.mode == 'zoom rect':
                    self.window.plot_host.toolbar.zoom()

            self.window.plot_host.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.window.plot_host.canvas.setFocus()
        else:
            self.window.movePointButton.setText("Move Point")

            # Disconnect matplotlib events
            if self._key_press_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._key_press_cid)
                self._key_press_cid = None
            if self._motion_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._motion_cid)
                self._motion_cid = None
            if self._release_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._release_cid)
                self._release_cid = None

            # Re-enable toolbar (user can re-select zoom/pan if they want)
            self.window.plot_host.canvas.toolbar.setEnabled(True)

            # Clear selected point
            self._selected_point = None

            # Only clear if no other edit mode is active
            if not getattr(self, "_add_peaks_mode", False) and not getattr(self, "_delete_peaks_mode", False) and not getattr(self, "_add_sigh_mode", False):
                self.window.plot_host.clear_click_callback()
                self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_plot_click_move_point(self, xdata, ydata, event):
        """Select a point (peak/onset/offset/exp) to move."""
        if not getattr(self, "_move_point_mode", False):
            return
        if event.inaxes is None or xdata is None:
            return

        # No need to check toolbar here - it's disabled when entering move mode

        st = self.window.state
        if st.t is None or st.analyze_chan not in st.sweeps:
            return

        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        # Get time basis (normalized if stim)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        # Find closest point among all types (within a reasonable distance)
        pks = np.asarray(st.peaks_by_sweep.get(s, np.array([], dtype=int)), dtype=int)
        breaths = st.breath_by_sweep.get(s, {})

        # Debug: print what's in the breaths dict
        print(f"[move-point-debug] breath_by_sweep keys for sweep {s}: {list(breaths.keys()) if breaths else 'None'}")

        onsets = np.asarray(breaths.get('onsets', np.array([], dtype=int)), dtype=int)
        offsets = np.asarray(breaths.get('offsets', np.array([], dtype=int)), dtype=int)
        expmins = np.asarray(breaths.get('expmins', np.array([], dtype=int)), dtype=int)
        expoffs = np.asarray(breaths.get('expoffs', np.array([], dtype=int)), dtype=int)

        # Debug: print what's available
        print(f"[move-point-debug] Available points - peaks: {pks.size}, onsets: {onsets.size}, offsets: {offsets.size}, expmins: {expmins.size}, expoffs: {expoffs.size}")

        # Find closest point (only consider points within 0.5 seconds)
        max_distance = 0.5  # seconds
        candidates = []

        if pks.size > 0:
            dists = np.abs(t_plot[pks] - xdata)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                candidates.append(('peak', pks[min_idx], dists[min_idx]))

        if onsets.size > 0:
            dists = np.abs(t_plot[onsets] - xdata)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                candidates.append(('onset', onsets[min_idx], dists[min_idx]))

        if offsets.size > 0:
            dists = np.abs(t_plot[offsets] - xdata)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                candidates.append(('offset', offsets[min_idx], dists[min_idx]))

        if expmins.size > 0:
            dists = np.abs(t_plot[expmins] - xdata)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                candidates.append(('expmin', expmins[min_idx], dists[min_idx]))

        if expoffs.size > 0:
            dists = np.abs(t_plot[expoffs] - xdata)
            min_idx = np.argmin(dists)
            if dists[min_idx] < max_distance:
                candidates.append(('expoff', expoffs[min_idx], dists[min_idx]))

        if not candidates:
            print("[move-point] No points within 0.5s of click location - click closer to a point")
            return

        # Debug: print all candidates
        print(f"[move-point-debug] Candidates within range: {[(c[0], f'{c[2]:.3f}s') for c in candidates]}")

        # Select closest
        point_type, idx, dist = min(candidates, key=lambda x: x[2])

        # Store selection (keep original_index for finding it later)
        self._selected_point = {'type': point_type, 'index': idx, 'sweep': s, 'original_index': idx}

        # No visual feedback marker needed - the existing point markers are sufficient
        print(f"[move-point] Selected {point_type} at index {idx} - use arrow keys to move (Shift=snap)")

    def _find_nearest_zero_crossing(self, y, dy, current_idx, search_radius=200):
        """
        Find nearest zero crossing in signal y or derivative dy.

        Args:
            y: Raw signal array
            dy: First derivative of signal
            current_idx: Current index position
            search_radius: Search window in samples (default 200)

        Returns:
            Index of nearest zero crossing, or current_idx if none found
        """
        # Define search window
        start = max(0, current_idx - search_radius)
        end = min(len(y), current_idx + search_radius)

        # Find zero crossings in y (sign changes)
        y_segment = y[start:end]
        y_crossings = np.where(np.diff(np.sign(y_segment)))[0] + start

        # Find zero crossings in dy (sign changes)
        dy_segment = dy[start:end]
        dy_crossings = np.where(np.diff(np.sign(dy_segment)))[0] + start

        # Combine all candidates
        all_crossings = np.concatenate([y_crossings, dy_crossings]) if len(y_crossings) or len(dy_crossings) else np.array([])

        if len(all_crossings) == 0:
            return current_idx

        # Find closest to current position
        distances = np.abs(all_crossings - current_idx)
        closest_idx = all_crossings[np.argmin(distances)]

        return int(closest_idx)

    def _constrain_to_peak_boundaries(self, new_idx, s):
        """Constrain point movement to respect breath event structure."""
        if not self._selected_point:
            return new_idx

        st = self.window.state
        point_type = self._selected_point['type']
        original_idx = self._selected_point['original_index']

        # Peaks can move freely within trace bounds
        if point_type == 'peak':
            return new_idx

        # Get breath events for this sweep
        breaths = st.breath_by_sweep.get(s, {})
        if not breaths:
            return new_idx

        # Get all breath event arrays
        onsets = np.asarray(breaths.get('onsets', []), dtype=int)
        offsets = np.asarray(breaths.get('offsets', []), dtype=int)
        expmins = np.asarray(breaths.get('expmins', []), dtype=int)
        expoffs = np.asarray(breaths.get('expoffs', []), dtype=int)
        pks = np.asarray(st.peaks_by_sweep.get(s, []), dtype=int)

        # Find which breath index this point belongs to
        # Use original_idx to find the breath cycle
        if point_type == 'onset' and len(onsets):
            breath_idx = np.argmin(np.abs(onsets - original_idx))
        elif point_type == 'offset' and len(offsets):
            breath_idx = np.argmin(np.abs(offsets - original_idx))
        elif point_type == 'expmin' and len(expmins):
            breath_idx = np.argmin(np.abs(expmins - original_idx))
        elif point_type == 'expoff' and len(expoffs):
            breath_idx = np.argmin(np.abs(expoffs - original_idx))
        else:
            # Fallback to peak-based constraints
            if pks.size < 2:
                return new_idx
            peak_before = pks[pks <= original_idx]
            peak_after = pks[pks > original_idx]
            min_bound = peak_before[-1] if len(peak_before) > 0 else 0
            max_bound = peak_after[0] if len(peak_after) > 0 else len(st.sweeps[st.analyze_chan][:, s])
            return int(np.clip(new_idx, min_bound, max_bound))

        # Define tighter constraints based on breath event structure
        # Structure: onset[i] -> peak[i] -> offset[i] -> expmin[i] -> expoff[i] -> onset[i+1]

        min_bound = 0
        max_bound = len(st.sweeps[st.analyze_chan][:, s])

        if point_type == 'onset':
            # Onset must be before its peak and after previous expoff
            if breath_idx < len(pks):
                max_bound = pks[breath_idx]
            if breath_idx > 0 and breath_idx - 1 < len(expoffs):
                min_bound = expoffs[breath_idx - 1]

        elif point_type == 'offset':
            # Offset must be after its peak and before its expmin
            if breath_idx < len(pks):
                min_bound = pks[breath_idx]
            if breath_idx < len(expmins):
                max_bound = expmins[breath_idx]

        elif point_type == 'expmin':
            # Expmin must be after offset and before expoff
            if breath_idx < len(offsets):
                min_bound = offsets[breath_idx]
            if breath_idx < len(expoffs):
                max_bound = expoffs[breath_idx]

        elif point_type == 'expoff':
            # Expoff must be after expmin and before next onset
            if breath_idx < len(expmins):
                min_bound = expmins[breath_idx]
            if breath_idx + 1 < len(onsets):
                max_bound = onsets[breath_idx + 1]

        return int(np.clip(new_idx, min_bound, max_bound))

    def _move_selected_point(self, direction, snap_to_zero=False):
        """Move the selected point left or right by 1 sample (for arrow keys).

        Args:
            direction: -1 for left, 1 for right
            snap_to_zero: If True, snap to nearest zero crossing in y or dy/dt
        """
        if not self._selected_point:
            return

        old_idx = self._selected_point['index']
        new_idx = old_idx + direction

        # Get trace for bounds checking
        t, y = self.window._current_trace()
        if t is None or y is None:
            return

        if new_idx < 0 or new_idx >= len(t):
            print("[move-point] Cannot move beyond trace bounds")
            return

        # Constrain to peak boundaries
        st = self.window.state
        s = self._selected_point['sweep']
        new_idx = self._constrain_to_peak_boundaries(new_idx, s)

        # Apply zero-crossing snap if Shift is held
        if snap_to_zero:
            dy = np.gradient(y)
            new_idx = self._find_nearest_zero_crossing(y, dy, new_idx, search_radius=200)
            new_idx = self._constrain_to_peak_boundaries(new_idx, s)  # Re-constrain after snap
            print(f"[move-point] Snapped to zero crossing at index {new_idx}")

        # Get time basis
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
        else:
            t_plot = t

        self._update_point_position(new_idx, t_plot, y, s)

    def _update_point_position(self, new_idx, t_plot, y, s):
        """Update point to new index position (shared by arrow keys and drag)."""
        if not self._selected_point:
            return

        st = self.window.state
        point_type = self._selected_point['type']
        old_idx = self._selected_point['index']

        # Update index
        self._selected_point['index'] = new_idx

        # Update the actual data array
        if point_type == 'peak':
            pks = st.peaks_by_sweep.get(s, np.array([], dtype=int))
            if pks.size > 0:
                old_idx_estimate = self._selected_point.get('original_index', old_idx)
                distances = np.abs(pks - old_idx_estimate)
                replace_idx = np.argmin(distances)
                pks[replace_idx] = new_idx
        elif point_type in ('onset', 'offset', 'expmin', 'expoff'):
            breaths = st.breath_by_sweep.get(s, {})
            key_map = {'onset': 'onsets', 'offset': 'offsets', 'expmin': 'expmins', 'expoff': 'expoffs'}
            key = key_map[point_type]
            if key in breaths:
                arr = breaths[key]
                if arr.size > 0:
                    old_idx_estimate = self._selected_point.get('original_index', old_idx)
                    distances = np.abs(arr - old_idx_estimate)
                    replace_idx = np.argmin(distances)
                    arr[replace_idx] = new_idx

        # Update scatter plot markers
        breaths = st.breath_by_sweep.get(s, {})
        on_idx = breaths.get('onsets', np.array([], dtype=int))
        off_idx = breaths.get('offsets', np.array([], dtype=int))
        ex_idx = breaths.get('expmins', np.array([], dtype=int))
        exoff_idx = breaths.get('expoffs', np.array([], dtype=int))

        t_on = t_plot[on_idx] if len(on_idx) else None
        y_on = y[on_idx] if len(on_idx) else None
        t_off = t_plot[off_idx] if len(off_idx) else None
        y_off = y[off_idx] if len(off_idx) else None
        t_exp = t_plot[ex_idx] if len(ex_idx) else None
        y_exp = y[ex_idx] if len(ex_idx) else None
        t_exof = t_plot[exoff_idx] if len(exoff_idx) else None
        y_exof = y[exoff_idx] if len(exoff_idx) else None

        self.window.plot_host.update_breath_markers(
            t_on=t_on, y_on=y_on,
            t_off=t_off, y_off=y_off,
            t_exp=t_exp, y_exp=y_exp,
            t_exoff=t_exof, y_exoff=y_exof,
            size=36
        )

        # Update peaks scatter plot
        pks = st.peaks_by_sweep.get(s, np.array([], dtype=int))
        if len(pks) > 0:
            self.window.plot_host.update_peaks(t_plot[pks], y[pks], size=50)

        # Just update the canvas
        self.window.plot_host.canvas.draw_idle()

    def _save_moved_point(self, recompute_metrics=False):
        """Save the moved point position and clear selection."""
        if not self._selected_point:
            return

        # Point has already been updated during movement, just need to clear selection
        point_type = self._selected_point['type']
        new_idx = self._selected_point['index']

        print(f"[move-point] Saved {point_type} at index {new_idx}")

        # Clear selection
        self._selected_point = None

        # Recompute metrics if requested (after drag release)
        if recompute_metrics:
            # Trigger eupnea/outlier region recalculation by calling redraw
            self.window.redraw_main_plot()
            # Toolbar stays disabled during move mode
        else:
            # Just redraw
            self.window.plot_host.canvas.draw_idle()

    def _cancel_move_point(self):
        """Cancel the point move operation and restore original position."""
        if not self._selected_point:
            return

        # Restore original position
        st = self.window.state
        s = self._selected_point['sweep']
        point_type = self._selected_point['type']
        original_idx = self._selected_point.get('original_index')
        current_idx = self._selected_point['index']

        if original_idx != current_idx:
            # Restore the point to its original position
            if point_type == 'peak':
                pks = st.peaks_by_sweep.get(s, np.array([], dtype=int))
                if pks.size > 0:
                    distances = np.abs(pks - current_idx)
                    replace_idx = np.argmin(distances)
                    pks[replace_idx] = original_idx
            elif point_type in ('onset', 'offset', 'expmin', 'expoff'):
                breaths = st.breath_by_sweep.get(s, {})
                key_map = {'onset': 'onsets', 'offset': 'offsets', 'expmin': 'expmins', 'expoff': 'expoffs'}
                key = key_map[point_type]
                if key in breaths:
                    arr = breaths[key]
                    if arr.size > 0:
                        distances = np.abs(arr - current_idx)
                        replace_idx = np.argmin(distances)
                        arr[replace_idx] = original_idx

        # Clear selection
        self._selected_point = None

        # Redraw to show restored position
        self.window.plot_host.canvas.draw_idle()
        print("[move-point] Move cancelled, position restored")

    # ========== MARK SNIFF MODE ==========

    def on_mark_sniff_toggled(self, checked: bool):
        """Enter/exit Mark Sniff mode."""
        self._mark_sniff_mode = checked

        if checked:
            # Turn off matplotlib toolbar modes (zoom, pan)
            self.window.plot_host.turn_off_toolbar_modes()

            # Turn OFF other edit modes
            if getattr(self, "_add_peaks_mode", False):
                self._add_peaks_mode = False
                self.window.addPeaksButton.blockSignals(True)
                self.window.addPeaksButton.setChecked(False)
                self.window.addPeaksButton.blockSignals(False)
                self.window.addPeaksButton.setText("Add Peaks")

            if getattr(self, "_delete_peaks_mode", False):
                self._delete_peaks_mode = False
                self.window.deletePeaksButton.blockSignals(True)
                self.window.deletePeaksButton.setChecked(False)
                self.window.deletePeaksButton.blockSignals(False)
                self.window.deletePeaksButton.setText("Delete Peaks")

            if getattr(self, "_add_sigh_mode", False):
                self._add_sigh_mode = False
                self.window.addSighButton.blockSignals(True)
                self.window.addSighButton.setChecked(False)
                self.window.addSighButton.blockSignals(False)
                self.window.addSighButton.setText("ADD/DEL Sigh")

            if getattr(self, "_move_point_mode", False):
                self._move_point_mode = False
                self.window.movePointButton.blockSignals(True)
                self.window.movePointButton.setChecked(False)
                self.window.movePointButton.blockSignals(False)
                self.window.movePointButton.setText("Move Point")

            self.window.markSniffButton.setText("Mark Sniff (ON) [Shift=Delete]")
            self.window.plot_host.set_click_callback(self._on_plot_click_mark_sniff)
            self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)

            # Connect matplotlib events for drag functionality
            self._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', self._on_sniff_drag)
            self._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', self._on_sniff_release)
        else:
            self.window.markSniffButton.setText("Mark Sniff")

            # Disconnect matplotlib events
            if self._motion_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._motion_cid)
                self._motion_cid = None
            if self._release_cid is not None:
                self.window.plot_host.canvas.mpl_disconnect(self._release_cid)
                self._release_cid = None

            # Clear drag artist
            if self._sniff_drag_artist:
                try:
                    self._sniff_drag_artist.remove()
                except:
                    pass
                self._sniff_drag_artist = None
                self.window.plot_host.canvas.draw_idle()

            # Only clear if no other edit mode is active
            if not getattr(self, "_add_peaks_mode", False) and not getattr(self, "_delete_peaks_mode", False) and not getattr(self, "_add_sigh_mode", False) and not getattr(self, "_move_point_mode", False):
                self.window.plot_host.clear_click_callback()
                self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_plot_click_mark_sniff(self, xdata, ydata, event):
        """Start marking a sniffing region (click-and-drag) or grab an edge to adjust.
        Shift+click on a region to delete it."""
        if not getattr(self, "_mark_sniff_mode", False):
            return
        if event.inaxes is None or xdata is None:
            return

        # Check if Shift key is held
        shift_held = (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

        # Check if click is near an existing region edge
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        regions = self.window.state.sniff_regions_by_sweep.get(s, [])

        # Convert to plot time for comparison
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
        else:
            t0 = 0.0

        # SHIFT+CLICK: Delete region
        if shift_held and regions:
            for i, (start_time, end_time) in enumerate(regions):
                plot_start = start_time - t0
                plot_end = end_time - t0

                # Check if click is INSIDE this region
                if plot_start <= xdata <= plot_end:
                    # Delete this region
                    del self.window.state.sniff_regions_by_sweep[s][i]
                    print(f"[mark-sniff] Deleted region {i}: {start_time:.3f} - {end_time:.3f} s")

                    # Log telemetry: manual edit
                    from core import telemetry
                    telemetry.log_edit('delete_sniff_region',
                                      num_regions_after=len(self.window.state.sniff_regions_by_sweep.get(s, [])),
                                      sweep_index=s)

                    self.window.redraw_main_plot()
                    return

        # Edge detection threshold (in plot time units)
        edge_threshold = 0.3  # seconds

        # Check each region for edge proximity (for adjusting edges)
        for i, (start_time, end_time) in enumerate(regions):
            plot_start = start_time - t0
            plot_end = end_time - t0

            # Check if near start edge
            if abs(xdata - plot_start) < edge_threshold:
                self._sniff_edge_mode = 'start'
                self._sniff_region_index = i
                self._sniff_start_x = plot_end  # The other edge stays fixed
                print(f"[mark-sniff] Grabbed START edge of region {i}")
                return

            # Check if near end edge
            if abs(xdata - plot_end) < edge_threshold:
                self._sniff_edge_mode = 'end'
                self._sniff_region_index = i
                self._sniff_start_x = plot_start  # The other edge stays fixed
                print(f"[mark-sniff] Grabbed END edge of region {i}")
                return

        # Not near any edge - start creating new region
        self._sniff_edge_mode = None
        self._sniff_region_index = None
        self._sniff_start_x = xdata
        print(f"[mark-sniff] Started new region at x={xdata:.3f}")

    def _on_sniff_drag(self, event):
        """Update visual indicator while dragging to mark sniffing region."""
        if not getattr(self, "_mark_sniff_mode", False):
            return
        if self._sniff_start_x is None or event.inaxes is None or event.xdata is None:
            return

        # Get plot axes
        ax = self.window.plot_host.ax_main
        if ax is None:
            return

        # Remove previous drag indicator
        if self._sniff_drag_artist:
            try:
                self._sniff_drag_artist.remove()
            except:
                pass

        # Draw semi-transparent purple rectangle
        x_start = self._sniff_start_x
        x_end = event.xdata
        x_left = min(x_start, x_end)
        x_right = max(x_start, x_end)

        self._sniff_drag_artist = ax.axvspan(x_left, x_right, alpha=0.3, color='purple', zorder=10)
        self.window.plot_host.canvas.draw_idle()

    def _on_sniff_release(self, event):
        """Finalize the sniffing region when mouse is released."""
        if not getattr(self, "_mark_sniff_mode", False):
            return
        if self._sniff_start_x is None or event.inaxes is None or event.xdata is None:
            return

        x_start = self._sniff_start_x
        x_end = event.xdata
        x_left = min(x_start, x_end)
        x_right = max(x_start, x_end)

        # Minimum width check (avoid accidental clicks)
        if abs(x_right - x_left) < 0.1:  # Less than 0.1 seconds
            print(f"[mark-sniff] Region too small, ignoring")
            self._sniff_start_x = None
            self._sniff_edge_mode = None
            self._sniff_region_index = None
            if self._sniff_drag_artist:
                try:
                    self._sniff_drag_artist.remove()
                except:
                    pass
                self._sniff_drag_artist = None
                self.window.plot_host.canvas.draw_idle()
            return

        # Convert from normalized time back to actual time if needed
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []

        if st.stim_chan and spans:
            t0 = spans[0][0]
            # x_left and x_right are in normalized time, convert to actual time
            actual_start = x_left + t0
            actual_end = x_right + t0
        else:
            actual_start = x_left
            actual_end = x_right

        # Snap to nearest breath events
        actual_start, actual_end = self._snap_sniff_to_breath_events(s, actual_start, actual_end)

        # Initialize regions list if needed
        if s not in self.window.state.sniff_regions_by_sweep:
            self.window.state.sniff_regions_by_sweep[s] = []

        # Handle edge dragging vs new region creation
        if self._sniff_edge_mode is not None and self._sniff_region_index is not None:
            # Editing existing region - update it
            old_start, old_end = self.window.state.sniff_regions_by_sweep[s][self._sniff_region_index]
            if self._sniff_edge_mode == 'start':
                # Update start edge, keep end fixed
                self.window.state.sniff_regions_by_sweep[s][self._sniff_region_index] = (actual_start, old_end)
                print(f"[mark-sniff] Updated START edge of region {self._sniff_region_index}: {actual_start:.3f} - {old_end:.3f} s")
            else:  # 'end'
                # Update end edge, keep start fixed
                self.window.state.sniff_regions_by_sweep[s][self._sniff_region_index] = (old_start, actual_end)
                print(f"[mark-sniff] Updated END edge of region {self._sniff_region_index}: {old_start:.3f} - {actual_end:.3f} s")
        else:
            # Creating new region - add it
            self.window.state.sniff_regions_by_sweep[s].append((actual_start, actual_end))
            print(f"[mark-sniff] Added new sniff region: {actual_start:.3f} - {actual_end:.3f} s (sweep {s})")

        # Merge overlapping regions
        self._merge_sniff_regions(s)

        # Clear drag state
        self._sniff_start_x = None
        self._sniff_edge_mode = None
        self._sniff_region_index = None
        if self._sniff_drag_artist:
            try:
                self._sniff_drag_artist.remove()
            except:
                pass
            self._sniff_drag_artist = None

        # Redraw to show permanent overlay
        self.window.redraw_main_plot()

    def _snap_sniff_to_breath_events(self, sweep_idx: int, start_time: float, end_time: float):
        """Snap sniff region edges to nearest breath events.

        Left edge (start) snaps to nearest inspiratory onset.
        Right edge (end) snaps to nearest expiratory offset.
        """
        # Get current trace to convert indices to times
        t, y = self.window._current_trace()
        if t is None:
            print("[mark-sniff] No trace available for snapping")
            return start_time, end_time

        # Get breath events for this sweep
        breaths = self.window.state.breath_by_sweep.get(sweep_idx, {})
        onsets = np.asarray(breaths.get('onsets', []), dtype=int)
        expoffs = np.asarray(breaths.get('expoffs', []), dtype=int)

        snapped_start = start_time
        snapped_end = end_time

        # Snap start to nearest onset
        if onsets.size > 0:
            onset_times = t[onsets]
            distances = np.abs(onset_times - start_time)
            nearest_idx = np.argmin(distances)

            # Only snap if within reasonable distance (e.g., 1 second)
            if distances[nearest_idx] < 1.0:
                snapped_start = onset_times[nearest_idx]
                print(f"[mark-sniff] Snapped START to onset at {snapped_start:.3f}s (was {start_time:.3f}s)")

        # Snap end to nearest expiratory offset
        if expoffs.size > 0:
            expoff_times = t[expoffs]
            distances = np.abs(expoff_times - end_time)
            nearest_idx = np.argmin(distances)

            # Only snap if within reasonable distance (e.g., 1 second)
            if distances[nearest_idx] < 1.0:
                snapped_end = expoff_times[nearest_idx]
                print(f"[mark-sniff] Snapped END to expiratory offset at {snapped_end:.3f}s (was {end_time:.3f}s)")

        return snapped_start, snapped_end

    def _merge_sniff_regions(self, sweep_idx: int):
        """Merge overlapping or directly adjacent sniffing regions for a given sweep.

        Note: This is primarily for manually marked regions. GMM-based regions
        are already merged by consecutive breath index before being added.
        """
        regions = self.window.state.sniff_regions_by_sweep.get(sweep_idx, [])
        if len(regions) <= 1:
            return

        # Sort regions by start time
        regions = sorted(regions, key=lambda x: x[0])

        # Merge overlapping or directly adjacent regions
        merged = []
        current_start, current_end = regions[0]

        for start, end in regions[1:]:
            if start <= current_end:  # Overlapping or directly adjacent
                # Merge by extending current region
                current_end = max(current_end, end)
                print(f"[mark-sniff] Merged overlapping regions into: {current_start:.3f} - {current_end:.3f} s")
            else:
                # No overlap - save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last region
        merged.append((current_start, current_end))

        # Update state
        self.window.state.sniff_regions_by_sweep[sweep_idx] = merged
        print(f"[mark-sniff] After merging: {len(merged)} region(s) on sweep {sweep_idx}")

    def update_sniff_artists(self, t_plot, sweep_idx: int):
        """(Re)draw purple overlays for marked sniffing regions (current sweep only)."""
        # Clear existing artists
        for art in self._sniff_artists:
            try:
                art.remove()
            except:
                pass
        self._sniff_artists = []

        # Get sniff regions for this sweep
        s = int(sweep_idx)
        regions = self.window.state.sniff_regions_by_sweep.get(s, [])
        if not regions:
            return

        # Get time offset for normalization (if stim channel)
        st = self.window.state
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
        else:
            t0 = 0.0

        # Draw each region as a semi-transparent purple rectangle
        ax = self.window.plot_host.ax_main
        if ax is None:
            return

        for (start_time, end_time) in regions:
            # Convert to plot time (normalized if stim)
            plot_start = start_time - t0
            plot_end = end_time - t0

            # Draw overlay
            artist = ax.axvspan(plot_start, plot_end, alpha=0.25, color='purple', zorder=5, label='Sniffing')
            self._sniff_artists.append(artist)

        self.window.plot_host.canvas.draw_idle()

    # ========== Omit Region Mode ==========

    def toggle_omit_region_mode(self, remove_mode=False):
        """Toggle omit region selection mode on/off."""
        if self._omit_region_mode:
            # Turn off
            self._exit_omit_region_mode()
        else:
            # Turn on
            self._enter_omit_region_mode(remove_mode)

    def _enter_omit_region_mode(self, remove_mode=False):
        """Enter omit region selection mode."""
        self._omit_region_mode = True
        self._omit_region_remove_mode = remove_mode

        # Turn off matplotlib toolbar modes
        self.window.plot_host.turn_off_toolbar_modes()

        # Turn OFF other edit modes
        if getattr(self, "_add_peaks_mode", False):
            self._add_peaks_mode = False
            self.window.addPeaksButton.blockSignals(True)
            self.window.addPeaksButton.setChecked(False)
            self.window.addPeaksButton.blockSignals(False)
            self.window.addPeaksButton.setText("Add Peaks")

        if getattr(self, "_delete_peaks_mode", False):
            self._delete_peaks_mode = False
            self.window.deletePeaksButton.blockSignals(True)
            self.window.deletePeaksButton.setChecked(False)
            self.window.deletePeaksButton.blockSignals(False)
            self.window.deletePeaksButton.setText("Delete Peaks")

        if getattr(self, "_add_sigh_mode", False):
            self._add_sigh_mode = False
            self.window.addSighButton.blockSignals(True)
            self.window.addSighButton.setChecked(False)
            self.window.addSighButton.blockSignals(False)
            self.window.addSighButton.setText("ADD/DEL Sigh")

        if getattr(self, "_move_point_mode", False):
            self._move_point_mode = False
            self.window.movePointButton.blockSignals(True)
            self.window.movePointButton.setChecked(False)
            self.window.movePointButton.blockSignals(False)
            self.window.movePointButton.setText("Move Point")

        if getattr(self, "_mark_sniff_mode", False):
            self._mark_sniff_mode = False
            self.window.markSniffButton.blockSignals(True)
            self.window.markSniffButton.setChecked(False)
            self.window.markSniffButton.blockSignals(False)
            self.window.markSniffButton.setText("Mark Sniff")

        # Update omit button appearance
        self.window.OmitSweepButton.setChecked(True)
        self.window.OmitSweepButton.setText("Omit (ON)")
        msg = "OMIT REGION MODE: Click-drag to create/adjust | Ctrl+click to delete | Ctrl+Shift+click full sweep | Press 'R' to snap to breaths"
        try:
            self.window._log_status_message(msg, 0)  # 0 = persistent message
            self.window._persistent_status_message = msg  # Store for refresh
        except Exception: pass

        # Set up event handlers
        self.window.plot_host.set_click_callback(self._on_plot_click_omit_region)
        self.window.plot_host.setCursor(Qt.CursorShape.CrossCursor)

        # Give focus to the canvas so it can receive key events
        self.window.plot_host.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.window.plot_host.canvas.setFocus()

        # Connect matplotlib events for drag functionality and keyboard
        self._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', self._on_omit_region_drag)
        self._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', self._on_omit_region_release)
        self._key_cid = self.window.plot_host.canvas.mpl_connect('key_press_event', self._on_omit_region_key_press)

    def _exit_omit_region_mode(self):
        """Exit omit region selection mode."""
        self._omit_region_mode = False
        self._omit_region_remove_mode = False

        # Update button appearance
        self.window.OmitSweepButton.setChecked(False)
        self.window._refresh_omit_button_label()

        # Disconnect matplotlib events
        if self._motion_cid is not None:
            self.window.plot_host.canvas.mpl_disconnect(self._motion_cid)
            self._motion_cid = None
        if self._release_cid is not None:
            self.window.plot_host.canvas.mpl_disconnect(self._release_cid)
            self._release_cid = None
        if self._key_cid is not None:
            self.window.plot_host.canvas.mpl_disconnect(self._key_cid)
            self._key_cid = None

        # Clear drag artist
        if self._omit_region_drag_artist:
            try:
                self._omit_region_drag_artist.remove()
            except:
                pass
            self._omit_region_drag_artist = None
            self.window.plot_host.canvas.draw_idle()

        # Clear click callback and restore cursor
        self.window.plot_host.clear_click_callback()
        self.window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

        # Clear persistent status message
        try:
            self.window._persistent_status_message = None
            self.window._log_status_message("Exited omit region mode", 3000)
        except Exception: pass

    def _on_plot_click_omit_region(self, xdata, ydata, event):
        """Start marking an omitted region (click-and-drag) or grab an edge to adjust.
        Ctrl+click on a region to delete it.
        Ctrl+Shift+click to toggle full sweep omission."""
        if not getattr(self, "_omit_region_mode", False):
            return
        if event.inaxes is None or xdata is None:
            return

        # Check modifier keys
        modifiers = QApplication.keyboardModifiers()
        shift_held = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        ctrl_held = bool(modifiers & Qt.KeyboardModifier.ControlModifier)

        # Get current sweep
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))

        # Check if full sweep is already omitted (unless doing Ctrl+Shift to toggle)
        if s in st.omitted_sweeps and not (ctrl_held and shift_held):
            print(f"[omit-region] Full sweep {s} already omitted - ignoring region operations")
            try: self.window._log_status_message("Full sweep already omitted (use Ctrl+Shift+click to un-omit)", 2000)
            except Exception: pass
            return

        # Ctrl+Shift: Toggle full sweep (DON'T exit mode, keep button ON)
        if ctrl_held and shift_held:
            print(f"[omit-region] Ctrl+Shift detected in click handler - toggling full sweep")
            # Get current sweep
            st = self.window.state
            s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))

            # Toggle sweep omission directly (keep mode active)
            if s in st.omitted_sweeps:
                st.omitted_sweeps.remove(s)
                try: self.window._log_status_message(f"Sweep {s+1}: included", 2000)
                except Exception: pass
            else:
                st.omitted_sweeps.add(s)
                # Clear smaller omitted regions when full sweep is omitted (they're redundant)
                if s in st.omitted_ranges:
                    print(f"[omit-region] Clearing {len(st.omitted_ranges[s])} smaller regions (full sweep now omitted)")
                    del st.omitted_ranges[s]
                try: self.window._log_status_message(f"Sweep {s+1}: omitted (cleared region markers)", 2000)
                except Exception: pass

            # Refresh and restore persistent message
            self.window._refresh_omit_button_label()
            self.window.redraw_main_plot()
            if hasattr(self.window, '_persistent_status_message') and self.window._persistent_status_message:
                try: self.window._log_status_message(self.window._persistent_status_message, 0)
                except Exception: pass
            return

        # Get current sweep
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))

        # Get omitted regions for this sweep
        regions = st.omitted_ranges.get(s, [])

        # Convert to plot time for comparison
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
        else:
            t0 = 0.0

        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        # CTRL+CLICK (no shift): Delete region
        if ctrl_held and not shift_held and regions:
            for i, (i_start, i_end) in enumerate(regions):
                t_start = i_start / sr_hz - t0
                t_end = i_end / sr_hz - t0

                # Check if click is INSIDE this region
                if t_start <= xdata <= t_end:
                    # Delete this region
                    del st.omitted_ranges[s][i]
                    if not st.omitted_ranges[s]:
                        del st.omitted_ranges[s]
                    print(f"[omit-region] Deleted region {i}: {t_start:.3f} - {t_end:.3f} s")

                    self.window.redraw_main_plot()
                    # Restore persistent message
                    if hasattr(self.window, '_persistent_status_message') and self.window._persistent_status_message:
                        try: self.window._log_status_message(self.window._persistent_status_message, 0)
                        except Exception: pass
                    return

        # Edge detection threshold (in plot time units)
        edge_threshold = 0.3  # seconds

        # Check each region for edge proximity (for adjusting edges)
        for i, (i_start, i_end) in enumerate(regions):
            t_start = i_start / sr_hz - t0
            t_end = i_end / sr_hz - t0

            # Check if near start edge
            if abs(xdata - t_start) < edge_threshold:
                self._omit_edge_mode = 'start'
                self._omit_region_index = i
                self._omit_region_start_x = t_end  # The other edge stays fixed
                print(f"[omit-region] Grabbed START edge of region {i}")
                return

            # Check if near end edge
            if abs(xdata - t_end) < edge_threshold:
                self._omit_edge_mode = 'end'
                self._omit_region_index = i
                self._omit_region_start_x = t_start  # The other edge stays fixed
                print(f"[omit-region] Grabbed END edge of region {i}")
                return

        # Not near any edge - start creating new region (or removing if in remove mode)
        self._omit_edge_mode = None
        self._omit_region_index = None
        self._omit_region_start_x = xdata
        print(f"[omit-region] Started new region at x={xdata:.3f}")

    def _on_omit_region_key_press(self, event):
        """Handle matplotlib canvas key press events for omit region mode."""
        if not getattr(self, "_omit_region_mode", False):
            return

        print(f"[omit-region] Canvas key press: '{event.key}'")

        # 'r' key to snap regions to breath onsets (R for "Region snap")
        if event.key in ('r', 'R'):
            print(f"[omit-region] R key detected on canvas, calling snap function")
            self._snap_all_omit_regions_to_breaths()

    def _on_omit_region_drag(self, event):
        """Update visual indicator while dragging to mark omitted region."""
        if not getattr(self, "_omit_region_mode", False):
            return
        if self._omit_region_start_x is None or event.inaxes is None or event.xdata is None:
            return

        # Get plot axes
        ax = self.window.plot_host.ax_main
        if ax is None:
            return

        # Remove previous drag indicator
        if self._omit_region_drag_artist:
            try:
                self._omit_region_drag_artist.remove()
            except:
                pass

        # Draw semi-transparent rectangle
        x_start = self._omit_region_start_x
        x_end = event.xdata
        x_left = min(x_start, x_end)
        x_right = max(x_start, x_end)

        # Choose color based on mode (gray for add, red for remove)
        color = 'red' if self._omit_region_remove_mode else 'gray'
        self._omit_region_drag_artist = ax.axvspan(x_left, x_right, alpha=0.3, color=color, zorder=10)
        self.window.plot_host.canvas.draw_idle()

    def _on_omit_region_release(self, event):
        """Finalize the omitted region when mouse is released."""
        if not getattr(self, "_omit_region_mode", False):
            return
        if self._omit_region_start_x is None or event.inaxes is None or event.xdata is None:
            return

        x_start = self._omit_region_start_x
        x_end = event.xdata
        x_left = min(x_start, x_end)
        x_right = max(x_start, x_end)

        # Minimum width check (avoid accidental clicks)
        if abs(x_right - x_left) < 0.05:  # Less than 50ms
            print(f"[omit-region] Region too small, ignoring")
            self._omit_region_start_x = None
            self._omit_edge_mode = None
            self._omit_region_index = None
            if self._omit_region_drag_artist:
                try:
                    self._omit_region_drag_artist.remove()
                except:
                    pass
                self._omit_region_drag_artist = None
                self.window.plot_host.canvas.draw_idle()
            return

        # Get current sweep and convert plot time to sample indices
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))
        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        # Adjust for stim normalization
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_left_absolute = x_left + t0
            t_right_absolute = x_right + t0
        else:
            t_left_absolute = x_left
            t_right_absolute = x_right

        # Note: Snapping disabled for now - user may want to mark artifacts before breath detection
        # Can add a "Snap Regions to Breaths" button later if needed
        # t_left_absolute, t_right_absolute = self._snap_omit_to_breath_events(s, t_left_absolute, t_right_absolute)

        # Convert to sample indices
        i_left = int(t_left_absolute * sr_hz)
        i_right = int(t_right_absolute * sr_hz)

        # Additional check: Ensure minimum width of at least 10 samples
        if abs(i_right - i_left) < 10:
            print(f"[omit-region] Region too small in samples ({abs(i_right - i_left)} samples), ignoring")
            self._reset_omit_region_state()
            if self._omit_region_drag_artist:
                try:
                    self._omit_region_drag_artist.remove()
                except:
                    pass
                self._omit_region_drag_artist = None
                self.window.plot_host.canvas.draw_idle()
            return

        # Handle edge adjustment vs new region
        if self._omit_edge_mode and self._omit_region_index is not None:
            # Adjusting an existing region's edge
            if s not in st.omitted_ranges or self._omit_region_index >= len(st.omitted_ranges[s]):
                # Region no longer exists, bail out
                self._reset_omit_region_state()
                return

            old_i_start, old_i_end = st.omitted_ranges[s][self._omit_region_index]

            if self._omit_edge_mode == 'start':
                # Moving start edge
                new_i_start = min(i_left, i_right)
                st.omitted_ranges[s][self._omit_region_index] = (new_i_start, old_i_end)
                print(f"[omit-region] Adjusted START edge of region {self._omit_region_index}")
            else:  # 'end'
                # Moving end edge
                new_i_end = max(i_left, i_right)
                st.omitted_ranges[s][self._omit_region_index] = (old_i_start, new_i_end)
                print(f"[omit-region] Adjusted END edge of region {self._omit_region_index}")

        elif self._omit_region_remove_mode:
            # Remove mode: Find and remove overlapping ranges
            if s in st.omitted_ranges:
                new_ranges = []
                for (i0, i1) in st.omitted_ranges[s]:
                    # Keep ranges that don't overlap with selected region
                    if i1 < i_left or i0 > i_right:
                        new_ranges.append((i0, i1))
                    else:
                        # Trim overlapping ranges
                        if i0 < i_left:
                            new_ranges.append((i0, min(i1, i_left)))
                        if i1 > i_right:
                            new_ranges.append((max(i0, i_right), i1))

                if new_ranges:
                    st.omitted_ranges[s] = new_ranges
                else:
                    del st.omitted_ranges[s]
        else:
            # Add mode: Append new range
            if s not in st.omitted_ranges:
                st.omitted_ranges[s] = []
            st.omitted_ranges[s].append((i_left, i_right))

            print(f"[omit-region] Added region to sweep {s}: samples {i_left}-{i_right} (time {x_left:.2f}-{x_right:.2f}s)")
            print(f"[omit-region] Total regions for sweep {s}: {len(st.omitted_ranges[s])}")
            print(f"[omit-region] All regions: {st.omitted_ranges.get(s, [])}")

        # Merge overlapping regions
        self._merge_omit_regions(s)

        # Reset state and redraw
        self._reset_omit_region_state()
        print(f"[omit-region] Calling redraw_main_plot()...")
        self.window.redraw_main_plot()

        # Restore persistent status message after redraw
        if hasattr(self.window, '_persistent_status_message') and self.window._persistent_status_message:
            try: self.window._log_status_message(self.window._persistent_status_message, 0)
            except Exception: pass

    def _reset_omit_region_state(self):
        """Reset omit region drag state."""
        self._omit_region_start_x = None
        self._omit_edge_mode = None
        self._omit_region_index = None
        if self._omit_region_drag_artist:
            try:
                self._omit_region_drag_artist.remove()
            except:
                pass
            self._omit_region_drag_artist = None
            self.window.plot_host.canvas.draw_idle()

    def _snap_all_omit_regions_to_breaths(self):
        """Snap all omitted regions on current sweep to breath onsets (onset to onset)."""
        st = self.window.state
        s = max(0, min(st.sweep_idx, self.window.navigation_manager._sweep_count() - 1))

        if s not in st.omitted_ranges:
            try: self.window._log_status_message("No omitted regions to snap on this sweep", 2000)
            except Exception: pass
            return

        # Get breath events
        breaths = st.breath_by_sweep.get(s, {})
        onsets = np.asarray(breaths.get('onsets', []), dtype=int)

        if onsets.size == 0:
            try: self.window._log_status_message("No breath onsets detected - run peak detection first", 3000)
            except Exception: pass
            return

        # Get current trace for time conversion
        t, y = self.window._current_trace()
        if t is None:
            return

        sr_hz = st.sr_hz if st.sr_hz else 1000.0
        regions = st.omitted_ranges[s]
        snapped_regions = []

        for (i_start, i_end) in regions:
            # Convert to time
            t_start = i_start / sr_hz
            t_end = i_end / sr_hz

            # Snap using existing function
            t_start_snapped, t_end_snapped = self._snap_omit_to_breath_events(s, t_start, t_end)

            # Convert back to sample indices
            i_start_snapped = int(t_start_snapped * sr_hz)
            i_end_snapped = int(t_end_snapped * sr_hz)

            snapped_regions.append((i_start_snapped, i_end_snapped))

        # Update state
        st.omitted_ranges[s] = snapped_regions

        # Merge any overlapping regions after snapping
        self._merge_omit_regions(s)

        # Redraw and restore message
        self.window.redraw_main_plot()
        if hasattr(self.window, '_persistent_status_message') and self.window._persistent_status_message:
            try: self.window._log_status_message(self.window._persistent_status_message, 0)
            except Exception: pass

        print(f"[omit-region] Snapped {len(regions)} region(s) to breath onsets")

    def _snap_omit_to_breath_events(self, sweep_idx: int, start_time: float, end_time: float):
        """Snap omit region edges to nearest breath events.

        Both edges snap to inspiratory onsets (onset to onset).
        This captures complete breaths within the region.
        """
        # Get current trace to convert indices to times
        t, y = self.window._current_trace()
        if t is None:
            print("[omit-region] No trace available for snapping")
            return start_time, end_time

        # Get breath events for this sweep
        breaths = self.window.state.breath_by_sweep.get(sweep_idx, {})
        onsets = np.asarray(breaths.get('onsets', []), dtype=int)

        if onsets.size == 0:
            print("[omit-region] No breath onsets available for snapping")
            return start_time, end_time

        onset_times = t[onsets]
        snapped_start = start_time
        snapped_end = end_time

        # Snap start to nearest onset (prefer earlier onset if tie)
        distances_start = np.abs(onset_times - start_time)
        nearest_start_idx = np.argmin(distances_start)

        # Only snap if within reasonable distance (e.g., 1 second)
        if distances_start[nearest_start_idx] < 1.0:
            snapped_start = onset_times[nearest_start_idx]
            print(f"[omit-region] Snapped START to onset at {snapped_start:.3f}s (was {start_time:.3f}s)")

        # Snap end to nearest onset (prefer later onset if tie)
        # Find onsets that are after the start
        later_onsets = onset_times[onset_times > snapped_start]
        if later_onsets.size > 0:
            distances_end = np.abs(later_onsets - end_time)
            nearest_end_idx = np.argmin(distances_end)

            # Only snap if within reasonable distance (e.g., 1 second)
            if distances_end[nearest_end_idx] < 1.0:
                snapped_end = later_onsets[nearest_end_idx]
                print(f"[omit-region] Snapped END to onset at {snapped_end:.3f}s (was {end_time:.3f}s)")

        return snapped_start, snapped_end

    def _merge_omit_regions(self, sweep_idx: int):
        """Merge overlapping or directly adjacent omitted regions for a given sweep."""
        st = self.window.state
        if sweep_idx not in st.omitted_ranges:
            return

        regions = st.omitted_ranges[sweep_idx]
        if len(regions) <= 1:
            return

        # Convert sample indices to times for merging
        sr_hz = st.sr_hz if st.sr_hz else 1000.0
        time_regions = [(i_start / sr_hz, i_end / sr_hz) for (i_start, i_end) in regions]

        # Sort regions by start time
        time_regions = sorted(time_regions, key=lambda x: x[0])

        # Merge overlapping or directly adjacent regions
        merged = []
        current_start, current_end = time_regions[0]

        for start, end in time_regions[1:]:
            if start <= current_end:  # Overlapping or directly adjacent
                # Merge by extending current region
                current_end = max(current_end, end)
                print(f"[omit-region] Merged overlapping regions into: {current_start:.3f} - {current_end:.3f}s")
            else:
                # No overlap - save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last region
        merged.append((current_start, current_end))

        # Convert back to sample indices
        merged_samples = [(int(t_start * sr_hz), int(t_end * sr_hz)) for (t_start, t_end) in merged]

        # Update state
        st.omitted_ranges[sweep_idx] = merged_samples
        print(f"[omit-region] After merging: {len(merged_samples)} region(s) on sweep {sweep_idx}")
