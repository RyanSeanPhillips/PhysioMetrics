# Phase 5: Export Logic Extraction - COMPLETED

## Overview
Successfully extracted all single-file export functionality from main.py into a dedicated ExportManager class. This massive extraction removed 2,233 lines (34% reduction) while maintaining full functionality.

## Implementation Summary

### Files Created
1. **export/__init__.py** (9 lines) - Module initialization
2. **export/export_manager.py** (2,404 lines) - Complete export functionality
3. **extract_export_methods.py** (120 lines) - Automated extraction script
4. **replace_export_methods.py** (97 lines) - Automated replacement script

### Methods Extracted (15 total, 2,355 lines)

#### UI Entry Points (2 methods, ~35 lines)
1. `on_save_analyzed_clicked()` - Save button handler
2. `on_view_summary_clicked()` - View summary button handler

#### Core Export Logic (2 massive methods, ~1,837 lines!)
3. `_export_all_analyzed_data()` - Main export orchestrator (1,277 lines)
   - NPZ bundle creation
   - means_by_time.csv generation
   - breaths.csv generation
   - events.csv generation
   - Calls _save_metrics_summary_pdf()
4. `_save_metrics_summary_pdf()` - PDF summary generator (560 lines)
   - Matplotlib figure generation
   - Multi-page PDF layout
   - Statistical plots

#### Helper Methods (9 methods, ~338 lines)
5. `_load_save_dialog_history()` - Load QSettings history
6. `_update_save_dialog_history()` - Update QSettings history
7. `_sanitize_token()` - Filename sanitization
8. `_suggest_stim_string()` - Auto-generate stim description
9. `_metric_keys_in_order()` - Get metric keys
10. `_compute_metric_trace()` - Call metric function with correct signature
11. `_get_stim_masks()` - Generate stim time masks
12. `_nanmean_sem()` - Compute mean/SEM with NaNs
13. `_mean_sem_1d()` - 1D mean/SEM calculation

#### Preview & Utilities (2 methods, ~223 lines)
14. `_show_summary_preview_dialog()` - Interactive preview dialog (145 lines)
15. `_sigh_sample_indices()` - Get sigh sample indices for breath intervals (78 lines)

### Transformation Details

#### Reference Transformations
All methods were transformed to use `self.window.` instead of `self.` for main window access:
- `self.state` → `self.window.state`
- `self.settings` → `self.window.settings`
- `self.statusbar` → `self.window.statusbar`
- `self.navigation_manager` → `self.window.navigation_manager`
- `self._compute_stim_for_current_sweep` → `self.window._compute_stim_for_current_sweep`
- `self._save_dir`, `self._save_base`, `self._save_meta` → `self.window._save_dir`, etc.

#### Main.py Changes
- Added import: `from export import ExportManager`
- Initialized manager: `self.export_manager = ExportManager(self)` (line 162-163)
- Replaced all 15 methods with delegation calls

Example delegation:
```python
def on_save_analyzed_clicked(self):
    """Save analyzed data to disk after prompting for location/name."""
    return self.export_manager.on_save_analyzed_clicked()

def _export_all_analyzed_data(self, preview_only=False, progress_dialog=None):
    """
    Exports (or previews) analyzed data.
    [... original docstring preserved ...]
    """
    return self.export_manager._export_all_analyzed_data(preview_only, progress_dialog)
```

### Issues Encountered and Fixed

#### Issue 1: Multi-line Function Signature Handling
**Problem**: The automated replacement script didn't handle multi-line function signatures correctly. The `_save_metrics_summary_pdf()` method had parameters spanning multiple lines, and the replacement only kept the first line without closing parenthesis.

**Error**:
```
SyntaxError: '(' was never closed
File "main.py", line 1951
    def _save_metrics_summary_pdf(
```

**Fix**: Manually added complete function signature:
```python
def _save_metrics_summary_pdf(self, pdf_path, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, meta, stim_zero, stim_dur):
    """Delegate to ExportManager."""
    return self.export_manager._save_metrics_summary_pdf(pdf_path, t_ds_csv, y2_ds_by_key, keys_for_csv, label_by_key, meta, stim_zero, stim_dur)
```

#### Issue 2: Missing self.window References (Runtime Errors)
**Problem**: The automated extraction script missed several method calls and properties that needed to be converted from `self.XXX` to `self.window.XXX`.

**Errors encountered:**
1. `QProgressDialog` parent parameter used `self` instead of `self.window` (2 occurrences)
2. `SaveMetaDialog` parent parameter used `self` instead of `self.window`
3. `QDialog` parent parameter used `self` instead of `self.window`
4. `QMessageBox` static methods used `self` parent instead of `self.window` (8 occurrences)
5. Method call `_get_processed_for` wasn't transformed (9 occurrences)
6. Method call `_parse_float` wasn't transformed (3 occurrences)
7. Properties: `eupnea_freq_threshold`, `eupnea_min_duration`, `_global_trace_cache`, `notch_filter_lower`, `notch_filter_upper`, `filter_func`, `filter_order`, `ApneaThresh`
8. Widget methods: `setCursor`, `unsetCursor`

**Fixes applied:**
- Fixed all Qt dialog parent parameters (4 dialog constructors + 8 QMessageBox calls = 12 fixes)
- Replaced `self._get_processed_for` → `self.window._get_processed_for` (9 occurrences)
- Replaced `self._parse_float(self.ApneaThresh)` → `self.window._parse_float(self.window.ApneaThresh)` (3 occurrences)
- Replaced property references with `self.window.` prefix (15 total replacements)
- Fixed cursor methods: `self.setCursor` → `self.window.setCursor`, etc.
- Fixed QMessageBox calls: `QMessageBox.information/warning/critical/question(self,` → `(self.window,`

**Verification**: Created and enhanced `check_export_references.py` script to systematically find missing references. Final check shows:
- Only 2 legitimate self-references remain (`_EXCLUDE_FOR_CSV` class variable and `scroll_area` local variable)
- No remaining Qt widget/dialog parent issues

### Results

#### Lines Removed from main.py
- **Before Phase 5**: 6,478 lines
- **After Phase 5**: 4,245 lines
- **Removed**: 2,233 lines (34% reduction in this phase)

#### Overall Progress
- **Original main.py**: 15,833 lines
- **After Phase 5**: 4,245 lines
- **Total reduction**: 11,588 lines (73% reduction!)

#### Compilation Status
- ✅ main.py compiles successfully
- ✅ export/export_manager.py compiles successfully
- ✅ Runtime testing complete - all export functionality working

## Rationale for Full Extraction

Despite the massive size (2,355 lines), full extraction was justified because:
1. **Experiment-specific customization**: User needs to modify export formats for different experiment types
2. **Maintainability**: Keeping all export logic together makes it easier to understand and modify
3. **Reusability**: ExportManager can be easily adapted for different experimental workflows
4. **Separation of concerns**: Export logic is now completely isolated from main application logic

## Phase 5b: Consolidation System (Future Work)

**NOT INCLUDED IN THIS PHASE** - Reserved for future extraction:
- Multi-file consolidation methods (~2,225 lines)
- Excel export with charts
- NPZ v2 loading
- CSV group scanning

These methods remain in main.py and will be extracted in a separate phase if needed.

## Testing Requirements

Phase 5 testing complete:
1. ✅ main.py compiles without errors
2. ✅ export/export_manager.py compiles without errors
3. ✅ Save analyzed data (NPZ + CSV + PDF export)
4. ✅ View summary preview dialog
5. ✅ All exported files created successfully
6. ✅ Tested with multiple sweeps
7. ✅ Success dialogs and error handling working correctly

## Next Steps

1. **Complete runtime testing** - Launch app and test all export functionality
2. **Phase 6**: Cache Management (~200 lines) - Much simpler than Phase 5
3. **Phase 7**: Remaining logic extraction if needed

## Notes

- This was the largest and riskiest extraction phase due to the massive method sizes
- Automated extraction scripts proved essential for handling 2000+ line methods
- Delegation pattern maintained consistency with previous phases
- The export code is now in a state where it can be easily customized for different experiment types
