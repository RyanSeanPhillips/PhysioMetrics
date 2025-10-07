# Immediate Channel Selection Changes

## Summary

Modified the application to apply channel selections immediately without requiring an Apply button. Added "All Channels" option to switch back to grid view, and stimulus is now drawn as soon as a stimulus channel is selected.

## Changes Made

### 1. Removed Apply Channel Button Logic

**Files Modified**: `main.py`

- Removed `ApplyChanPushButton` click handler connection
- Removed pending selection tracking (`_pending_analyze_idx`, `_pending_stim_idx`)
- Deleted `on_apply_channels_clicked()` function
- Deleted `_update_apply_button_enabled()` function

### 2. Immediate Channel Selection

**Function**: `on_analyze_channel_changed(idx)`

**New Behavior**:
- **idx = 0**: "All Channels" selected → switches to grid mode (multi-channel view)
  - Sets `st.analyze_chan = None`
  - Sets `single_panel_mode = False`
  - Clears stimulus channel
  - Redraws plot immediately

- **idx > 0**: Specific channel selected → switches to single channel view
  - Sets `st.analyze_chan = channel_names[idx - 1]` (offset by 1 for "All Channels")
  - Sets `single_panel_mode = True`
  - Clears all peak detection results
  - Redraws plot immediately

### 3. Immediate Stimulus Selection

**Function**: `on_stim_channel_changed(idx)`

**New Behavior**:
- **idx = 0**: "None" selected → clears stimulus
- **idx > 0**: Specific channel selected → applies stimulus immediately
  - Clears previous stimulus detection results
  - Calls `_compute_stim_for_current_sweep()` automatically
  - Redraws plot with stimulus overlay

### 4. Added "All Channels" Option

**Function**: `load_file(path)`

**Changes**:
- Analyze dropdown now starts with "All Channels" as first option
- Channel list follows after "All Channels"
- Default selection is "All Channels" (grid mode)
- Initial state: `st.analyze_chan = None`, `single_panel_mode = False`

**Before**:
```
Analyze: [Channel 1] [Channel 2] [Channel 3]
```

**After**:
```
Analyze: [All Channels] [Channel 1] [Channel 2] [Channel 3]
```

### 5. Removed Mean Subtraction Connections

**Files Modified**: `main.py`

Removed the following connections (controls are hidden in horizontal UI):
- `self.MeanSubractVal.editingFinished.connect(self.update_and_redraw)`
- `self.MeanSubtract_checkBox.toggled.connect(self.update_and_redraw)`

**Note**: Mean Subtraction controls are not present in the horizontal UI layout. If you want to add them to the Spectral Analysis window, that can be done separately.

## User Workflow Changes

### Before:
1. Select channel from dropdown
2. Click "Apply" button
3. View changes

### After:
1. Select channel from dropdown
2. Changes apply immediately (no Apply button needed)
3. Select "All Channels" to return to grid view

### Stimulus:
- Stimulus now appears automatically when a stimulus channel is selected
- No Apply button required

## UI Compatibility

These changes work with the **horizontal layout** (`pleth_app_layout_02_horizontal.ui`) where:
- The Apply Channel button has been removed by the user
- Mean Subtraction controls are not visible in the compact groupboxes

## Testing Recommendations

1. Load an ABF or SMRX file
2. Verify "All Channels" appears as first option in Analyze dropdown
3. Verify grid mode shows all channels at startup
4. Select a specific channel → should switch to single channel view immediately
5. Select "All Channels" → should switch back to grid view
6. Select a stimulus channel → should appear immediately on the plot
7. Navigate between sweeps → stimulus should persist

## Known Issues / Edge Cases

- If you select "All Channels" while in single panel mode, it will switch back to grid mode (expected behavior)
- Stimulus channel selection only affects the display if an analyze channel is also selected
- Peak detection results are cleared when switching analyze channels (to avoid stale data)

## Future Enhancements

- Add Mean Subtraction controls to Spectral Analysis window as requested
- Consider adding a keyboard shortcut to toggle between grid and single channel views
- Add visual indication when stimulus is being displayed
