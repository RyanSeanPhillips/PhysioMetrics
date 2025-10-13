# Phase 5b: Consolidation System Extraction Plan

## Overview
Extract all multi-file consolidation functionality from main.py into a dedicated ConsolidationManager class. This is similar in scope to Phase 5a (single-file export).

## Methods to Extract (13 total, estimated ~2,200 lines)

### UI Entry Point (1 method, ~202 lines)
1. **`on_consolidate_save_data_clicked()`** (line 2243, ~202 lines)
   - Main consolidation button handler
   - File type separation (means/breaths/events)
   - Consolidation orchestration
   - Excel export

### Filename Proposal (1 method, ~94 lines)
2. **`_propose_consolidated_filename()`** (line 2149, ~94 lines)
   - Intelligent filename generation
   - Cross-file validation (same date, same stim, etc.)
   - Warning generation

### Core Consolidation Methods (5 methods, ~1,056 lines)
3. **`_consolidate_breaths_histograms()`** (line 2445, ~218 lines)
   - Consolidate breath-by-breath histogram data
   - Groups by metric across files

4. **`_consolidate_events()`** (line 2663, ~46 lines)
   - Consolidate eupnea/apnea/sniffing events
   - Adds experiment numbers

5. **`_consolidate_stimulus()`** (line 2709, ~62 lines)
   - Consolidate stimulus timing data
   - Adds experiment numbers

6. **`_consolidate_from_npz_v2()`** (line 2827, ~165 lines)
   - Load and consolidate from NPZ v2 format
   - Extract time-series metrics from NPZ bundles

7. **`_consolidate_means_files()`** (line 2992, ~369 lines)
   - Consolidate means_by_time CSV files
   - Most complex consolidation method

8. **`_consolidate_breaths_sighs()`** (line 3361, ~76 lines)
   - Consolidate sigh detection data
   - Cross-file sigh aggregation

### NPZ Helper Methods (2 methods, ~56 lines)
9. **`_try_load_npz_v2()`** (line 2771, ~30 lines)
   - Attempt to load NPZ v2 format
   - Error handling for NPZ loading

10. **`_extract_timeseries_from_npz()`** (line 2801, ~26 lines)
    - Extract specific metric from NPZ data
    - Handles raw/normalized variants

### Excel Export Methods (3 methods, ~811 lines)
11. **`_save_consolidated_to_excel()`** (line 3437, ~520 lines)
    - Main Excel export orchestrator
    - Creates all sheets (means, histograms, events, sighs, stimulus)
    - Calls chart methods

12. **`_add_events_charts()`** (line 3957, ~163 lines)
    - Add timeline charts to events sheet
    - Matplotlib → Excel image embedding

13. **`_add_sighs_chart()`** (line 4120, ~128 lines)
    - Add timeline charts to sighs sheet
    - Similar to events charts

**TOTAL ESTIMATED: ~2,219 lines**

---

## Implementation Strategy

### Option 1: Full Extraction (Recommended)
Extract all consolidation methods to `consolidation/consolidation_manager.py`:

**Pros**:
- Complete separation of consolidation logic
- Easier to customize for different experiment types
- Consistent with Phase 5a approach
- All consolidation code in one place

**Cons**:
- Large extraction (~2,200 lines)
- High risk of missing references
- Time-consuming (~4-5 hours)

**Estimated time**: 4-5 hours

### Option 2: Partial Extraction (Lower Risk)
Extract only core methods, keep Excel export in main.py:

**Pros**:
- Lower risk
- Faster (~2-3 hours)

**Cons**:
- Less clean separation
- Excel code still in main.py

**Estimated time**: 2-3 hours

---

## Recommended Approach: **Option 1 (Full Extraction)**

### Rationale:
1. User wants consolidation logic customizable for different experiments
2. Consistent with Phase 5a approach (full extraction worked well)
3. We have good tools now (check scripts, automated extraction)
4. We just made a backup commit, so safe to proceed

### What to Extract:

1. Create `consolidation/__init__.py`
2. Create `consolidation/consolidation_manager.py` with:
   - All 13 methods
   - ConsolidationManager class
   - `__init__(self, main_window)` pattern

3. Update `main.py`:
   - Add import: `from consolidation import ConsolidationManager`
   - Initialize: `self.consolidation_manager = ConsolidationManager(self)`
   - Replace all 13 methods with delegation calls

### Expected Transformations:

```python
# In consolidation_manager.py:
class ConsolidationManager:
    def __init__(self, main_window):
        self.window = main_window

    def on_consolidate_save_data_clicked(self):
        # All references: self.X → self.window.X
        # Except internal method calls within ConsolidationManager
        ...

# In main.py:
def on_consolidate_save_data_clicked(self):
    """Delegate to ConsolidationManager."""
    return self.consolidation_manager.on_consolidate_save_data_clicked()
```

---

## Potential Issues & Solutions

### Issue 1: Missing self.window References
**Solution**: Use the `check_export_references.py` pattern
- Create `check_consolidation_references.py`
- Systematically verify all transformations

### Issue 2: Qt Dialog Parents
**Solution**: All dialogs must use `self.window` as parent, not `self`
- QMessageBox
- QFileDialog
- QProgressDialog

### Issue 3: Method Calls Between Consolidation Methods
**Solution**: Keep internal calls as `self.method()`, only external calls need `self.window.method()`

Example:
```python
# Internal call (both in ConsolidationManager):
def on_consolidate_save_data_clicked(self):
    filename = self._propose_consolidated_filename(files)  # Keep as self
    data = self._consolidate_means_files(files)           # Keep as self

# External call (calling MainWindow method):
def _consolidate_means_files(self, files):
    y = self.window._get_processed_for(chan, sweep)  # Need self.window
```

---

## Testing Strategy

### Before Extraction:
1. ✅ Backup created (commit 7a04642)
2. Test consolidation manually - ensure it works before extraction

### During Extraction:
1. Create automated extraction script (like Phase 5a)
2. Use replacement script for delegation
3. Create verification script

### After Extraction:
1. Verify compilation
2. Test consolidation with multiple files
3. Verify Excel output
4. Check charts are embedded correctly

---

## Files to Create

1. `consolidation/__init__.py` (9 lines)
2. `consolidation/consolidation_manager.py` (~2,250 lines)
3. `extract_consolidation_methods.py` (automation script)
4. `replace_consolidation_methods.py` (automation script)
5. `check_consolidation_references.py` (verification script)

---

## Files to Modify

1. **main.py**:
   - Add import
   - Initialize consolidation_manager
   - Replace 13 methods with delegation

---

## Expected Results

### Lines Removed from main.py:
- **Before Phase 5b**: 4,245 lines
- **After Phase 5b**: ~2,000-2,100 lines (estimated)
- **Removed**: ~2,200 lines (52% reduction in this phase)

### Overall Progress (Phases 1-5b):
- **Original main.py**: 15,833 lines
- **After Phase 5b**: ~2,000 lines
- **Total reduction**: ~87% overall!

---

## Decision Point

**Proceed with Phase 5b Option 1 (Full Extraction)?**

**Recommended**: Yes
- Consistent with Phase 5a success
- Good tools and experience from Phase 5a
- Safe backup point
- User wants customizable consolidation logic

**Alternative**: Address pending bugs first, do Phase 5b later

---

## Next Steps (If Approved)

1. Test consolidation manually to ensure it works
2. Create extraction automation scripts
3. Extract all 13 methods to ConsolidationManager
4. Replace methods in main.py with delegation
5. Create verification script
6. Run verification and fix issues
7. Test consolidation functionality
8. Commit Phase 5b

Estimated completion time: 4-5 hours
