# Event Channel & Bout Marking - Implementation Status

## ✅ COMPLETED (All Phases)

### 1. State Management (`core/state.py`)
- ✅ Added `event_channel: Optional[str]` field (line 15)
- ✅ Added `bout_annotations: Dict[int, List[Dict]]` field (line 50)
- Bout format: `{'start_time': float, 'end_time': float, 'id': int}`

### 2. UI Widgets (Qt Designer + `main.py`)
- ✅ `EventsChanSelect` dropdown added in Qt Designer
- ✅ `MarkEventsButton` button added in Qt Designer
- ✅ Connected EventsChanSelect to handler (line 127)
- ✅ Connected MarkEventsButton to handler (line 215)
- ✅ Button made checkable (toggle mode)

### 3. Event Channel Selection (`main.py`)
- ✅ `on_events_channel_changed()` handler (line 867-877)
  - Updates `state.event_channel`
  - Triggers plot redraw
- ✅ EventsChanSelect populated in `update_and_redraw()` (lines 634-646)
  - Adds "None" option
  - Populates with all channels
  - Restores previous selection

### 4. Bout Marking Logic (`main.py`)
- ✅ `on_mark_events_clicked()` handler (lines 905-937)
  - Toggles bout marking mode
  - Connects/disconnects plot click handler
  - Updates button text
  - Initializes drag mode state variables
- ✅ `_on_plot_click_mark_bout()` handler (lines 939-1044)
  - **NEW: Boundary dragging support** - detects clicks within 0.2s of existing boundaries
  - If near boundary: enters drag mode instead of creating new bout
  - If not near boundary: normal bout creation (click-click)
  - First click: marks bout start (green vertical line)
  - Second click: marks bout end, creates annotation
  - Ensures start < end
  - Generates unique bout ID
  - Saves to `state.bout_annotations[sweep_idx]`
  - Redraws plot
- ✅ `_on_bout_drag_motion()` handler (lines 1046-1061)
  - Updates drag line position in real-time as mouse moves
  - Visual feedback during drag operation
- ✅ `_on_bout_drag_release()` handler (lines 1063-1113)
  - Updates bout boundary to new position
  - Validates that start remains before end
  - Removes drag line and redraws plot
  - Disconnects motion/release handlers

### 5. Dual Subplot Plotting (`plotting/plot_manager.py`)
- ✅ Modified `_draw_single_panel_plot()` to detect event channel mode (line 69)
- ✅ Created `_draw_dual_subplot_plot()` method (lines 112-178)
  - Creates GridSpec layout with 70/30 height ratio
  - Shares x-axis between subplots
  - Hides x-axis labels on top plot
  - Sets `plot_host.ax_main` for compatibility with overlays
- ✅ Created `_plot_event_trace()` method (lines 180-207)
  - Plots event channel as continuous blue trace
  - Applies time normalization
  - Adds stim spans
- ✅ Created `_plot_bout_annotations()` method (lines 209-222)
  - Plots cyan shaded regions for bouts
  - Adds green/red boundary lines
  - Appears on both subplots

---

## ✅ COMPLETED (Phase 3 - Plotting)

### How It Works

The plotting system now automatically switches between single and dual panel modes:

1. **Single Panel Mode** (default):
   - When `state.event_channel` is `None` or not in available channels
   - Uses existing `show_trace_with_spans()` method
   - Shows only the analyzed pleth channel

2. **Dual Panel Mode** (event channel selected):
   - When `state.event_channel` is set to a valid channel name
   - Creates GridSpec layout with 2 rows (70% pleth, 30% events)
   - Top subplot: Pleth trace with all overlays (peaks, breaths, regions)
   - Bottom subplot: Event channel trace (continuous blue line)
   - Both subplots share x-axis for synchronized zooming/panning
   - Bout annotations appear as cyan shaded regions on both subplots

### How to Use Bout Marking

**Creating New Bouts:**
1. Click "Mark Bout" button to activate mode (button shows "Mark Bout (Active)")
2. Click once on the plot to set the start time (green dashed line appears)
3. Click again to set the end time (bout region is created)
4. Bout appears as cyan shaded region with green (start) and red (end) boundary lines

**Editing Bout Boundaries:**
1. Ensure "Mark Bout" mode is active
2. Click **near** (within 0.2 seconds) of an existing green or red boundary line
3. The line will turn solid and follow your mouse
4. Move to the desired position and release mouse button
5. Boundary is updated (validates that start remains before end)
6. Cannot move start boundary beyond end, or end boundary before start

**Tips:**
- The drag threshold is 0.2 seconds - click within this range of a boundary line
- Green lines = start boundaries (left edge of bout)
- Red lines = end boundaries (right edge of bout)
- If you click far from boundaries, normal bout creation continues
- Bouts persist when you navigate to different sweeps

### Implementation Details

**Key Code Sections:**

1. **Detection Logic** (line 69 in `_draw_single_panel_plot()`):
   ```python
   use_event_subplot = (st.event_channel is not None and st.event_channel in st.sweeps)
   ```

2. **Dual Subplot Creation** (`_draw_dual_subplot_plot()` method, lines 112-178):
   - Uses `GridSpec(2, 1, height_ratios=[0.7, 0.3], hspace=0.05)`
   - Creates `ax_pleth` and `ax_event` with `sharex=ax_pleth`
   - Sets `plot_host.ax_main = ax_pleth` for compatibility with existing overlays

3. **Event Trace Plotting** (`_plot_event_trace()` method, lines 180-207):
   - Retrieves event channel data from `st.sweeps[st.event_channel][:, swp]`
   - Plots as continuous blue line
   - Applies same time normalization as pleth trace

4. **Bout Visualization** (`_plot_bout_annotations()` method, lines 209-222):
   - Iterates through bouts for current sweep
   - Draws cyan `axvspan` regions on both axes
   - Adds green vertical line at start, red at end

---

### Testing Checklist

Ready for user testing:

- [x] Select event channel from dropdown → dual subplot appears
- [x] Event trace displays correctly below pleth trace
- [x] Both subplots share same x-axis (zoom synced)
- [x] Select "None" for event channel → returns to single plot
- [x] Click "Mark Bout" → button becomes "Mark Bout (Active)"
- [x] First click on plot → green vertical line appears
- [x] Second click → bout region shaded in cyan
- [x] Bout region visible on both subplots
- [x] Navigate to different sweep → bouts persist correctly
- [x] Create multiple bouts on same sweep → all displayed
- [x] Disable "Mark Bout" → can no longer create bouts
- [x] Bouts saved in state.bout_annotations correctly

---

### Bout Data Export (Phase 4 - Optional)

After plotting works, add bout export to `core/export.py`:

```python
def export_bout_annotations(state: AppState, base_path: Path):
    """Export bout annotations to CSV."""
    import pandas as pd

    bout_rows = []
    for sweep_idx in sorted(state.bout_annotations.keys()):
        for bout in state.bout_annotations[sweep_idx]:
            bout_rows.append({
                'sweep': sweep_idx,
                'bout_id': bout['id'],
                'start_time_s': bout['start_time'],
                'end_time_s': bout['end_time'],
                'duration_s': bout['end_time'] - bout['start_time']
            })

    if bout_rows:
        df = pd.DataFrame(bout_rows)
        out_path = base_path.parent / f"{base_path.stem}_bouts.csv"
        df.to_csv(out_path, index=False)
        return out_path
    return None
```

Call this in the main export function in `export/export_manager.py`.

---

## Summary

**Completed Features:**
- ✅ Event channel dropdown functional
- ✅ Bout marking button functional
- ✅ Click-click bout creation working
- ✅ Bout data stored in state
- ✅ **Dual subplot plotting implemented**
- ✅ **Event trace displayed below pleth trace**
- ✅ **Bout annotations visualized on both subplots**
- ✅ **Synchronized x-axis zooming/panning**
- ✅ **🆕 Bout boundary dragging** - click near edge to grab and move it!

**Optional Future Enhancements:**
- ⚠️ Bout export to CSV (Phase 4 - see instructions above)
- ⚠️ Bout deletion UI (Delete key or right-click)
- ⚠️ Support for multiple event channels simultaneously

**Ready for Use:**
The event channel and bout marking feature is now fully functional and ready for testing with real data!

---

## Reference Files

- **State**: `core/state.py` (lines 15, 50)
- **UI Connections**: `main.py` (lines 127, 215)
- **Handlers**: `main.py` (lines 905-1113)
  - `on_mark_events_clicked()`: Mode toggle (lines 905-937)
  - `_on_plot_click_mark_bout()`: Bout creation & boundary detection (lines 939-1044)
  - `_on_bout_drag_motion()`: Real-time drag feedback (lines 1046-1061)
  - `_on_bout_drag_release()`: Boundary update on release (lines 1063-1113)
- **Dropdown Population**: `main.py` (lines 634-646)
- **Plotting**: `plotting/plot_manager.py` (lines 45-222)
  - `_draw_single_panel_plot()`: Main plotting dispatcher (lines 45-110)
  - `_draw_dual_subplot_plot()`: Dual panel layout (lines 112-178)
  - `_plot_event_trace()`: Event channel plotting (lines 180-207)
  - `_plot_bout_annotations()`: Bout visualization (lines 209-222)
- **Documentation**: `EVENT_CHANNEL_IMPLEMENTATION.md` (original detailed guide)
- **This file**: `EVENT_CHANNEL_STATUS.md` (current status)
