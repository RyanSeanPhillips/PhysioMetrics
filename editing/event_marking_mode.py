"""
Event Marking Mode for PhysioMetrics.

Handles manual event region marking with drag-and-drop functionality:
- Click and drag to create new event regions
- Click near edges to drag and adjust them
- Shift+Click to delete regions
- Automatic merging when edges overlap
- Optional snapping to signal features (minima, threshold crossings)
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from scipy.signal import find_peaks


# Module-level state
_event_marking_active = False
_drag_start_x = None
_drag_artists = []  # List of artists (for both subplots)
_edge_mode = None  # 'start' or 'end' if dragging an edge
_region_index = None  # Index of region being edited
_motion_cid = None
_release_cid = None
_dialog_ref = None  # Reference to the dialog for updating event count


def get_dragging_boundary():
    """
    Get information about the boundary currently being dragged.

    Returns:
        tuple: (region_index, edge_mode, sweep_idx) if dragging, else (None, None, None)
    """
    if _event_marking_active and _drag_start_x is not None and _edge_mode is not None and _region_index is not None:
        # Need to get sweep from somewhere - we'll pass main_window when needed
        return (_region_index, _edge_mode, None)  # Sweep will be determined in plotting
    return (None, None, None)


def enable_event_marking(main_window, dialog):
    """Enable event marking mode."""
    global _event_marking_active, _dialog_ref
    _event_marking_active = True
    _dialog_ref = dialog

    # Create closure that captures main_window
    def click_handler(xdata, ydata, event):
        _on_plot_click(xdata, ydata, event, main_window)

    # Set up click callback
    main_window.plot_host.set_click_callback(click_handler)
    main_window.plot_host.setCursor(Qt.CursorShape.CrossCursor)

    # Connect matplotlib motion and release events
    global _motion_cid, _release_cid
    _motion_cid = main_window.plot_host.canvas.mpl_connect('motion_notify_event', lambda e: _on_drag_motion(e, main_window))
    _release_cid = main_window.plot_host.canvas.mpl_connect('button_release_event', lambda e: _on_drag_release(e, main_window))

    print("[event-marking] Mode enabled")


def disable_event_marking(main_window):
    """Disable event marking mode."""
    global _event_marking_active, _drag_start_x, _drag_artists, _edge_mode, _region_index, _dialog_ref
    _event_marking_active = False
    _drag_start_x = None
    _edge_mode = None
    _region_index = None
    _dialog_ref = None

    # Clear drag artists
    if _drag_artists:
        for artist in _drag_artists:
            try:
                artist.remove()
            except:
                pass
        _drag_artists = []
        main_window.plot_host.canvas.draw_idle()

    # Disconnect matplotlib events
    global _motion_cid, _release_cid
    if _motion_cid is not None:
        main_window.plot_host.canvas.mpl_disconnect(_motion_cid)
        _motion_cid = None
    if _release_cid is not None:
        main_window.plot_host.canvas.mpl_disconnect(_release_cid)
        _release_cid = None

    # Clear click callback
    main_window.plot_host.clear_click_callback()
    main_window.plot_host.setCursor(Qt.CursorShape.ArrowCursor)

    print("[event-marking] Mode disabled")


def _on_plot_click(xdata, ydata, event, main_window=None):
    """Handle plot click for event marking."""
    if not _event_marking_active or event.inaxes is None or xdata is None:
        return

    # Get main_window from parent widget chain if not provided
    if main_window is None:
        widget = event.canvas
        while widget is not None and not hasattr(widget, 'state'):
            widget = widget.parent()
        main_window = widget
        if main_window is None:
            return

    # Check if Shift key is held for deletion
    shift_held = (QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

    st = main_window.state
    swp = st.sweep_idx

    # Get time normalization offset
    spans = st.stim_spans_by_sweep.get(swp, []) if st.stim_chan else []
    t0 = spans[0][0] if (st.stim_chan and spans) else 0.0

    # Check existing events
    events = st.bout_annotations.get(swp, [])

    # SHIFT+CLICK: Delete event
    if shift_held and events:
        for i, event_data in enumerate(events):
            plot_start = event_data['start_time'] - t0
            plot_end = event_data['end_time'] - t0

            # Check if click is INSIDE this event
            if plot_start <= xdata <= plot_end:
                del st.bout_annotations[swp][i]
                print(f"[event-marking] Deleted event {event_data['id']}: {event_data['start_time']:.3f} - {event_data['end_time']:.3f} s")
                main_window.redraw_main_plot()
                if _dialog_ref:
                    _dialog_ref.update_event_count()
                return

    # Edge detection threshold (in plot time units)
    edge_threshold = 0.3  # seconds

    # Check each event for edge proximity
    global _edge_mode, _region_index, _drag_start_x
    for i, event_data in enumerate(events):
        plot_start = event_data['start_time'] - t0
        plot_end = event_data['end_time'] - t0

        # Check if near start edge
        if abs(xdata - plot_start) < edge_threshold:
            _edge_mode = 'start'
            _region_index = i
            _drag_start_x = plot_end  # The other edge stays fixed
            print(f"[event-marking] Grabbed START edge of event {event_data['id']}")
            # Trigger redraw to hide the boundary being dragged
            main_window.redraw_main_plot()
            return

        # Check if near end edge
        if abs(xdata - plot_end) < edge_threshold:
            _edge_mode = 'end'
            _region_index = i
            _drag_start_x = plot_start  # The other edge stays fixed
            print(f"[event-marking] Grabbed END edge of event {event_data['id']}")
            # Trigger redraw to hide the boundary being dragged
            main_window.redraw_main_plot()
            return

    # Not near any edge - start creating new event
    _edge_mode = None
    _region_index = None
    _drag_start_x = xdata
    print(f"[event-marking] Started new event at x={xdata:.3f}")


def _on_drag_motion(event, main_window):
    """Update visual indicator while dragging - shows vertical cursor line and horizontal reference."""
    global _drag_artists, _drag_start_x, _edge_mode

    if not _event_marking_active or _drag_start_x is None or event.inaxes is None or event.xdata is None:
        return

    # Get plot axes
    ax_main = main_window.plot_host.ax_main
    if ax_main is None:
        return

    # Remove previous drag indicators
    for artist in _drag_artists:
        try:
            artist.remove()
        except:
            pass
    _drag_artists = []

    # Determine color based on which edge is being dragged
    if _edge_mode == 'start':
        line_color = 'green'  # Start boundary
    elif _edge_mode == 'end':
        line_color = 'red'    # End boundary
    else:
        line_color = 'cyan'   # New event

    # Draw vertical line on pleth subplot
    artist_main = ax_main.axvline(event.xdata, color=line_color, linestyle='-', linewidth=2, alpha=0.8, zorder=10)
    _drag_artists.append(artist_main)

    # Also draw on event subplot if it exists
    ax_event = getattr(main_window.plot_host, 'ax_event', None)
    if ax_event is not None:
        artist_event = ax_event.axvline(event.xdata, color=line_color, linestyle='-', linewidth=2, alpha=0.8, zorder=10)
        _drag_artists.append(artist_event)

        # Add horizontal reference line at cursor's intersection with event signal
        # This helps user align marker with signal features
        if (_edge_mode == 'start' or _edge_mode == 'end'):
            # Get the event signal value at the cursor's x-position
            st = main_window.state
            swp = st.sweep_idx

            # Get time normalization offset
            spans = st.stim_spans_by_sweep.get(swp, []) if st.stim_chan else []
            t0 = spans[0][0] if (st.stim_chan and spans) else 0.0

            # Convert cursor x-position (plot coordinates) to actual time
            actual_time = event.xdata + t0

            # Find nearest index in time array and get event signal value
            try:
                idx = np.argmin(np.abs(st.t - actual_time))

                # Get event signal value at this index
                if st.event_channel and st.event_channel in st.sweeps:
                    event_signal = st.sweeps[st.event_channel][:, swp]
                    y_value = event_signal[idx]

                    # Draw thin dotted black horizontal line at event signal value
                    artist_horiz = ax_event.axhline(y_value, color='black', linestyle=':', linewidth=1.0, alpha=0.7, zorder=9)
                    _drag_artists.append(artist_horiz)
            except:
                pass  # Silently skip if there's any issue

    main_window.plot_host.canvas.draw_idle()


def _on_drag_release(event, main_window):
    """Finalize the event region when mouse is released."""
    global _drag_start_x, _drag_artists, _edge_mode, _region_index

    if not _event_marking_active or _drag_start_x is None or event.inaxes is None or event.xdata is None:
        return

    x_start = _drag_start_x
    x_end = event.xdata
    x_left = min(x_start, x_end)
    x_right = max(x_start, x_end)

    # Minimum width check (avoid accidental clicks)
    if abs(x_right - x_left) < 0.05:  # Less than 50ms
        print(f"[event-marking] Event too small, ignoring")
        _drag_start_x = None
        _edge_mode = None
        _region_index = None
        for artist in _drag_artists:
            try:
                artist.remove()
            except:
                pass
        _drag_artists = []
        main_window.plot_host.canvas.draw_idle()
        return

    # Convert from normalized time back to actual time if needed
    st = main_window.state
    swp = st.sweep_idx
    spans = st.stim_spans_by_sweep.get(swp, []) if st.stim_chan else []

    if st.stim_chan and spans:
        t0 = spans[0][0]
        actual_start = x_left + t0
        actual_end = x_right + t0
    else:
        actual_start = x_left
        actual_end = x_right

    # Check if snapping is enabled
    snap_enabled = False
    if _dialog_ref is not None:
        try:
            snap_enabled = _dialog_ref.snap_markers_check.isChecked()
        except:
            pass

    # Apply snapping if enabled
    if snap_enabled:
        actual_start = _snap_to_signal_features(actual_start, is_onset=True, main_window=main_window)
        actual_end = _snap_to_signal_features(actual_end, is_onset=False, main_window=main_window)

    # Initialize events list if needed
    if swp not in st.bout_annotations:
        st.bout_annotations[swp] = []

    # Handle edge dragging vs new event creation
    if _edge_mode is not None and _region_index is not None:
        # Editing existing event - update it
        old_event = st.bout_annotations[swp][_region_index]
        old_start = old_event['start_time']
        old_end = old_event['end_time']

        if _edge_mode == 'start':
            # Update start edge, keep end fixed
            st.bout_annotations[swp][_region_index]['start_time'] = actual_start
            print(f"[event-marking] Updated START edge of event {old_event['id']}: {actual_start:.3f} - {old_end:.3f} s")
        else:  # 'end'
            # Update end edge, keep start fixed
            st.bout_annotations[swp][_region_index]['end_time'] = actual_end
            print(f"[event-marking] Updated END edge of event {old_event['id']}: {old_start:.3f} - {actual_end:.3f} s")

        # Check for overlap with other events and merge if needed
        _check_and_merge_overlaps(main_window, swp)
    else:
        # Creating new event - add it
        new_event = {
            'start_time': actual_start,
            'end_time': actual_end,
            'id': len(st.bout_annotations[swp]) + 1
        }
        st.bout_annotations[swp].append(new_event)
        print(f"[event-marking] Added new event: {actual_start:.3f} - {actual_end:.3f} s (sweep {swp})")

        # Check for overlap with existing events and merge if needed
        _check_and_merge_overlaps(main_window, swp)

    # Clear drag state
    _drag_start_x = None
    _edge_mode = None
    _region_index = None
    for artist in _drag_artists:
        try:
            artist.remove()
        except:
            pass
    _drag_artists = []

    # Redraw to show permanent overlay
    main_window.redraw_main_plot()
    if _dialog_ref:
        _dialog_ref.update_event_count()


def _snap_to_signal_features(time_point, is_onset, main_window):
    """
    Snap event boundary to nearest signal feature.

    Args:
        time_point: Detected boundary time (in actual time, not normalized)
        is_onset: True if snapping onset (finds minimum before), False if offset (finds threshold crossing after)
        main_window: Reference to main window for accessing state and event data

    Returns:
        Snapped time point
    """
    st = main_window.state
    swp = st.sweep_idx

    # Get event channel data and time array
    if not st.event_channel or st.event_channel not in st.sweeps:
        return time_point

    event_data = st.sweeps[st.event_channel][:, swp]
    t = st.t

    # Find index corresponding to detected time point
    try:
        idx = np.argmin(np.abs(t - time_point))
    except:
        return time_point

    # Get threshold from dialog if available
    threshold = 0.5  # Default
    if hasattr(main_window, '_event_detection_dialog') and main_window._event_detection_dialog is not None:
        try:
            threshold = main_window._event_detection_dialog.threshold_spin.value()
        except:
            pass

    # Define search window (2 seconds or 20% of trace length, whichever is smaller)
    max_window_sec = 2.0
    max_window_idx = int(max_window_sec * st.sr_hz)
    max_window_idx = min(max_window_idx, int(0.2 * len(t)))

    if is_onset:
        # ONSET: Find nearest local minimum BEFORE the detected point
        search_start = max(0, idx - max_window_idx)
        search_end = min(idx + int(0.1 * max_window_idx), len(event_data))  # Small forward window

        if search_end <= search_start:
            return time_point

        search_region = event_data[search_start:search_end]

        # Find local minima
        minima_idx, _ = find_peaks(-search_region, distance=int(0.1 * st.sr_hz))

        if len(minima_idx) > 0:
            # Find closest minimum to our detected point
            absolute_minima_idx = minima_idx + search_start
            distances = np.abs(absolute_minima_idx - idx)
            closest_min_idx = absolute_minima_idx[np.argmin(distances)]

            # Only use if it's before the detected point
            if closest_min_idx <= idx:
                snapped_time = t[closest_min_idx]
                print(f"[event-snap] Onset snapped from {time_point:.3f} to {snapped_time:.3f}s (minimum)")
                return snapped_time

        # Fallback: Just find the minimum value in the search window
        min_idx = search_start + np.argmin(search_region)
        if min_idx < idx:
            snapped_time = t[min_idx]
            print(f"[event-snap] Onset snapped from {time_point:.3f} to {snapped_time:.3f}s (lowest point)")
            return snapped_time

    else:
        # OFFSET: Find nearest threshold crossing AFTER the detected point
        search_start = max(0, idx - int(0.1 * max_window_idx))  # Small backward window
        search_end = min(idx + max_window_idx, len(event_data))

        if search_end <= search_start:
            return time_point

        search_region = event_data[search_start:search_end]

        # Find downward threshold crossings (signal drops below threshold)
        above_thresh = search_region > threshold
        crossings = np.diff(above_thresh.astype(int))
        crossing_indices = np.where(crossings == -1)[0] + 1  # Falling edge

        if len(crossing_indices) > 0:
            # Find closest crossing after the detected point
            absolute_crossing_idx = crossing_indices + search_start
            forward_crossings = absolute_crossing_idx[absolute_crossing_idx >= idx]

            if len(forward_crossings) > 0:
                closest_crossing_idx = forward_crossings[0]  # First crossing after
                snapped_time = t[closest_crossing_idx]
                print(f"[event-snap] Offset snapped from {time_point:.3f} to {snapped_time:.3f}s (threshold crossing)")
                return snapped_time

    # No suitable feature found - return original
    return time_point


def _check_and_merge_overlaps(main_window, sweep_idx):
    """Check for overlapping events and merge them automatically."""
    st = main_window.state

    if sweep_idx not in st.bout_annotations or len(st.bout_annotations[sweep_idx]) <= 1:
        return

    # Sort events by start time
    events = sorted(st.bout_annotations[sweep_idx], key=lambda x: x['start_time'])

    # Merge overlapping events
    merged = []
    current = events[0].copy()

    for event in events[1:]:
        # Check if events overlap
        if event['start_time'] <= current['end_time']:
            # Overlap detected - merge by extending current event
            current['end_time'] = max(current['end_time'], event['end_time'])
            print(f"[event-marking] Auto-merged overlapping events")
        else:
            # No overlap - save current and start new
            merged.append(current)
            current = event.copy()

    # Add the last event
    merged.append(current)

    # Reassign IDs
    for i, event in enumerate(merged):
        event['id'] = i + 1

    # Update state
    st.bout_annotations[sweep_idx] = merged
    print(f"[event-marking] Result: {len(merged)} events after overlap check")
