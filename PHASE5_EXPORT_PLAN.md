# Phase 5: Export Logic Extraction Plan

## Overview
Phase 5 involves extracting export functionality from main.py. However, the export code is MASSIVE (~2500+ lines) and splits into two distinct subsystems:

### Phase 5a: Single-File Export (THIS PHASE)
Basic export for single analyzed files:
- NPZ bundle export
- CSV exports (means_by_time, breaths, events)
- PDF summary generation
- Preview functionality

**Estimated size:** ~1300 lines

### Phase 5b: Consolidation System (FUTURE PHASE)
Multi-file consolidation and Excel export:
- Consolidation of multiple analyzed files
- Excel export with charts
- NPZ v2 loading
- CSV group scanning

**Estimated size:** ~1200 lines (to be extracted separately later)

---

## Phase 5a: Methods to Extract

### UI Entry Points (2 methods, ~35 lines)
1. `on_save_analyzed_clicked()` (line 1983, ~17 lines) - Save button handler
2. `on_view_summary_clicked()` (line 2001, ~17 lines) - View summary button handler

### Core Export Logic (1 massive method, ~1277 lines!)
3. `_export_all_analyzed_data()` (line 2125, ~1277 lines) - Main export orchestrator
   - NPZ bundle creation
   - means_by_time.csv generation
   - breaths.csv generation
   - events.csv generation
   - Calls _save_metrics_summary_pdf()

### PDF Generation (1 method, ~560 lines)
4. `_save_metrics_summary_pdf()` (line 3421, ~560 lines) - PDF summary generator
   - Matplotlib figure generation
   - Multi-page PDF layout
   - Statistical plots

### Helper Methods (9 methods, ~200 lines)
5. `_load_save_dialog_history()` (line 1849, ~20 lines) - Load QSettings history
6. `_update_save_dialog_history()` (line 1869, ~29 lines) - Update QSettings history
7. `_sanitize_token()` (line 1898, ~13 lines) - Filename sanitization
8. `_suggest_stim_string()` (line 1911, ~72 lines) - Auto-generate stim description
9. `_metric_keys_in_order()` (line 2023, ~4 lines) - Get metric keys
10. `_compute_metric_trace()` (line 2027, ~15 lines) - Call metric function with correct signature
11. `_get_stim_masks()` (line 2042, ~31 lines) - Generate stim time masks
12. `_nanmean_sem()` (line 2073, ~52 lines) - Compute mean/SEM with NaNs
13. `_mean_sem_1d()` (line 3402, ~19 lines) - 1D mean/SEM calculation

### Preview Dialog (1 method, ~145 lines)
14. `_show_summary_preview_dialog()` (line 3981, ~145 lines) - Interactive preview dialog

### Sigh Detection Helper (1 method, ~78 lines)
15. `_sigh_sample_indices()` (line 4126, ~78 lines) - Get sigh sample indices for breath intervals

**TOTAL ESTIMATED: ~2,355 lines**

---

## Phase 5b: Methods to SKIP (Consolidation - Future Phase)

### Consolidation Entry Point (1 method)
- `on_consolidate_save_data_clicked()` (line 4473, ~202 lines)

### Curation UI Methods (4 methods)
- `on_curation_choose_dir_clicked()` (line 4204, ~21 lines)
- `_scan_csv_groups()` (line 4225, ~44 lines)
- `_populate_file_list_from_groups()` (line 4269, ~52 lines)
- `_list_has_path()` (line 4370, ~9 lines)
- `_propose_consolidated_filename()` (line 4379, ~94 lines)

### Consolidation Core Methods (8 methods)
- `_consolidate_breaths_histograms()` (line 4675, ~218 lines)
- `_consolidate_events()` (line 4893, ~46 lines)
- `_consolidate_stimulus()` (line 4939, ~62 lines)
- `_try_load_npz_v2()` (line 5001, ~30 lines)
- `_extract_timeseries_from_npz()` (line 5031, ~26 lines)
- `_consolidate_from_npz_v2()` (line 5057, ~165 lines)
- `_consolidate_means_files()` (line 5222, ~369 lines)
- `_consolidate_breaths_sighs()` (line 5591, ~76 lines)

### Excel Export Methods (3 methods)
- `_save_consolidated_to_excel()` (line 5667, ~520 lines)
- `_add_events_charts()` (line 6187, ~163 lines)
- `_add_sighs_chart()` (line 6350, ~128 lines)

**CONSOLIDATION TOTAL: ~2,225 lines** (to be extracted in a future phase)

---

## Implementation Strategy for Phase 5a

### Option 1: Full Extraction (Risky, High Effort)
Extract all single-file export logic to `export/export_manager.py`:
- **Pros**: Clean separation, matches modularization goals
- **Cons**: 2,355 lines to move, high risk of breaking, very time-consuming
- **Estimated time**: 6-8 hours

### Option 2: Partial Extraction (Moderate Risk, Medium Effort)
Extract only helper methods and keep core methods as thin wrappers:
- Move helpers to ExportManager
- Keep `_export_all_analyzed_data()` and `_save_metrics_summary_pdf()` in main.py
- Create ExportManager with utility methods
- **Pros**: Lower risk, faster
- **Cons**: Less clean separation
- **Estimated time**: 2-3 hours

### Option 3: Minimal Extraction (Low Risk, Low Effort)
Only extract small helper methods:
- Move 9 helper methods to ExportManager
- Keep main export methods in main.py
- Add delegation calls
- **Pros**: Very safe, quick
- **Cons**: Minimal benefit, large methods still in main.py
- **Estimated time**: 1-2 hours

---

## Recommended Approach: **Option 2 (Partial Extraction)**

### Rationale:
1. `_export_all_analyzed_data()` is 1,277 lines - extremely risky to move
2. `_save_metrics_summary_pdf()` is 560 lines - also risky
3. Moving helper methods provides benefit with low risk
4. We can always do full extraction later if needed

### What to Extract:
1. Create `export/export_manager.py` with:
   - All 9 helper methods
   - `_show_summary_preview_dialog()`
   - `_sigh_sample_indices()`

2. Keep in main.py (as thin wrappers or unchanged):
   - `on_save_analyzed_clicked()` - UI handler (delegates to manager)
   - `on_view_summary_clicked()` - UI handler (delegates to manager)
   - `_export_all_analyzed_data()` - **Keep unchanged** (too risky to move)
   - `_save_metrics_summary_pdf()` - **Keep unchanged** (too risky to move)

3. Update references in main.py to call:
   - `self.export_manager.load_save_dialog_history()`
   - `self.export_manager.sanitize_token()`
   - etc.

### Expected Reduction:
- **Lines removed from main.py**: ~400-500 lines (helper methods)
- **Risk level**: LOW (not touching the massive export methods)
- **Time estimate**: 2-3 hours

---

## Alternative: Skip Phase 5 Entirely

Given:
- Already achieved 59% reduction in main.py
- Export code is working correctly
- High risk/effort ratio for full extraction
- Other pending bugs and improvements

**Recommendation**: Consider skipping Phase 5 and moving to Phase 6 (Cache Management - simpler, ~200 lines) OR focusing on bug fixes.

---

## Decision Point

**Which approach should we take?**

1. **Option 2** (Partial Extraction) - Extract helpers, keep main export methods
2. **Option 1** (Full Extraction) - Move everything (high risk/effort)
3. **Skip Phase 5** - Move to Phase 6 or bug fixes

**Waiting for user decision...**
