# Unreleased Changes

## Session: 2026-03-31 — Photometry crash fix + AI channel display

### Bug Fixes
- **Fix crash loading corrupted/recovered photometry files**: `np.linspace` with negative `n_points` when AI and timestamp row counts don't match. Added early validation of time range and `n_points <= 0` guard before `linspace` calls (`dialogs/photometry/data_assembly_widget.py`)
- **Fix AI channels not appearing in photometry channel table**: `_populate_ai_column_controls()` required a removed `ai_columns_container` widget, silently skipping `self.ai_columns` population. AI channels now populate regardless of container widget existence (`dialogs/photometry/data_assembly_widget.py`)

## Session: 2026-02-20 — MVVM Refactoring Phase A+B

### Architecture — MVVM Service Extraction
- **AnalysisConfig / FilterConfig / PeakDetectionConfig**: Pure dataclasses capturing all analysis parameters (`core/domain/analysis/models.py`)
- **AnalysisService**: Headless signal processing + peak detection (`core/services/analysis_service.py`)
  - `get_processed_signal()` — pure filter chain, replaces `MainWindow._get_processed_for` body
  - `compute_normalization_stats()` — pure z-score stats, replaces `_compute_global_zscore_stats` body
  - `detect_single_sweep()` — pure peak detection, replaces `_detect_single_sweep_core` body
  - `auto_detect_threshold()` — pure Otsu + valley fitting
  - `analyze_file()` — full headless analysis pipeline (load → detect → metrics → CSV)
  - `analyze_folder()` — batch processing over a folder
- **ClassifierService**: Pure ML classifier operations (`core/services/classifier_service.py`)
  - Model loading, prediction (all 3 tiers), GMM clustering
  - Label management (apply, clear, set all)
- **ClassifierViewModel**: Qt bridge with signals (`viewmodels/classifier_viewmodel.py`)
- **StimService**: Pure stim detection across sweeps (`core/services/stim_service.py`)
- **notch_filter_1d**: Pure notch filter added to `core/filters.py`
- **AppState**: Added `filter_config` and `peak_config` fields
- **MainWindow delegates**: `_get_processed_for`, `_detect_single_sweep_core`, `_compute_global_zscore_stats`, `_apply_notch_filter`, `_detect_stims_all_sweeps` now delegate to services

### New: Batch Analysis
- **batch_mcp.py**: MCP server + CLI for headless batch analysis (`tools/batch_mcp.py`)
  - `batch_analyze_file` — single file analysis
  - `batch_analyze_folder` — folder batch processing
  - CLI mode: `python tools/batch_mcp.py <path> [--pattern *.abf]`
- **.mcp.json**: Added `batch` MCP server entry

### Tests
- **test_batch_analysis.py**: 10 pytest tests covering config models, signal processing, auto-threshold, and full pipeline integration
