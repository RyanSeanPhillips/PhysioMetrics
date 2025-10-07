# Alignment Fixes for Horizontal Layout

## Issues to Fix in Qt Designer

### 1. Peak Detection GroupBox
**Current layout:**
- Row 1: Threshold, Min Peak Dist
- Row 2: Prominence (alone)

**Should be:**
- Row 1: Threshold, Prominence, Min Peak Dist
- Row 2: [Apply button centered]

**Fix in Qt Designer:**
1. Open the Peak Detection groupbox
2. Delete the separate `horizontalLayout_9`
3. Add Prominence controls to `horizontalLayout_8` between Threshold and Min Peak Dist
4. Move Apply Peak Find button inside the groupbox at row 2

### 2. Filtering GroupBox - Duplicate Filter Order
**Problem:** Has two separate "Filter Order" controls (one for Low Pass, one for High Pass)

**Should be:** Single shared filter order that applies to both

**Fix in Qt Designer:**
1. Delete `FilterOrderSpin_2` and `FilterOrderLabel_2` from row 2
2. Keep only `FilterOrderSpin` and `FilterOrderLabel` in row 1
3. Layout should be:
   ```
   Row 1: Low Pass [val] | Order: [spin] | Mean Sub [val] | [Spectral Analysis]
   Row 2: High Pass [val] | Invert Signal
   ```

### 3. Channel Selection GroupBox - Add Apply Button
**Current:** Just dropdowns, Apply button is outside groupbox

**Should be:** Apply button inside groupbox at bottom

**Fix in Qt Designer:**
1. Move `ApplyChanPushButton` inside groupBox_2
2. Add it as third row in verticalLayout_8
3. Center it horizontally

### 4. Rename Unnamed Widgets
Replace all `<widget class="QWidget" name="">` with proper names:
- `peakDetectionControls`
- `filterRow1`
- `filterRow2`
- `channelControls`

### 5. Consistent Spacing
All groupboxes should have:
- 10px left/right padding
- 20px top margin (for title)
- 10px spacing between rows
- Same height (adjust to tallest groupbox)

## Recommended Groupbox Heights
- Channel Selection: 120px (3 rows: 2 dropdowns + Apply)
- Filtering: 110px (2 rows + Spectral button)
- Peak Detection: 110px (2 rows + Apply)

All groupboxes should align horizontally at y=30.
