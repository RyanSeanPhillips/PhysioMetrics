# Unreleased Changes

## Session: 2026-04-07 — Minimap Navigation Rework + UI Polish

### Minimap Navigation Bar
- **Always expanded**: Minimap stays at full 36px height permanently (no more collapse/hover)
- **Green viewport**: Changed from blue to green to distinguish from blue laser stim markers
- **Dark overlays**: Areas outside viewport are dimmed for clear boundary visibility
- **Edge handles**: Arrow markers (`|>` / `<|`) with resize cursor and hover glow (thicker + brighter on hover)
- **Double-click to center**: Double-click anywhere on minimap to jump viewport to that position
- **Duration tooltip**: Shows window size (e.g., "12.4s") while dragging viewport edges
- **Full-zoom visibility**: 3% X-axis padding ensures edges are visible/grabbable at full zoom
- **Location**: `plotting/navigation_bar.py`, `plotting/pyqtgraph_backend.py`

### Tab Bar Corner Widget
- **Always-visible controls**: Moved beta version label, "Report Bug / Request Feature" link, Claude Code button, and Help link from Analysis tab to tab bar top-right corner
- **Visible on all tabs**: Controls persist regardless of which tab is selected
- **Location**: `main.py:_setup_tab_corner_widget()`

### Bug Fix: Navigation Mode Button
- **Fix**: `_on_nav_window_changed` crashed with `set_xlim` AttributeError — now checks for `_subplots` first instead of relying on `state.plotting_backend` string
- **Location**: `main.py:_on_nav_window_changed()`

---

## Session: 2026-04-07 — Telemetry Fix + CTA Improvements

### Critical Fix: App Startup Hang (1.5GB Memory)
- **Root cause**: `telemetry.log` grew to 1.14GB due to feedback loop — failed GA4 sends were cached locally, then re-sent on startup (spawning millions of threads), failing again, re-caching, growing forever
- **Fix**: Rewrote telemetry as fire-and-forget (no local caching). If GA4 send fails, event is silently dropped
- **Self-healing**: Startup health check deletes any existing `telemetry.log` on launch — fixes affected installs automatically
- **GA4 accuracy**: Removed engagement event spam (was sending every 45s, now just session_start/session_end + 45s heartbeat ping). Should fix over-counted active users
- **Location**: `core/telemetry.py` (full rewrite), `main.py` (health check + timer)

### CTA Dialog Improvements
- **Always available**: "Generate CTA..." now always shows in right-click menu (was hidden when no event markers placed). Only requires a file to be loaded
- **Non-blocking**: CTA dialog no longer blocks the main window — can interact with both simultaneously
- **Renamed**: "Generate Photometry CTA..." → "Generate CTA..." (works for any signal, not just photometry)
- **Z-score spinboxes**: Fixed keyboard input not working in baseline Start/End fields
- **Tab close button**: X button now always visible on tabs (was only visible on hover)

### Right-Click Context Menu
- **Color-coded sections**: Menu organized into Markers (green), Analysis (blue), View (gray) with colored circle icons
- **Section headers**: Bold dividers between groups for easier scanning

### HDF5 Save Disabled for Beta
- Parallel HDF5 save disabled — was writing unused `.hdf5` files alongside every `.pmx` save
- Code preserved in `core/hdf5_io.py` for next release (save + load complete, load path not yet connected)

### Database Recovery
- **Stale WAL cleanup**: Removes orphaned WAL/SHM files on startup if no other PhysioMetrics process running
- **Startup backup**: Daily integrity-checked DB backup with row count tracking
- **Recovery chain**: WAL cleanup → backup restore (picks backup with most rows) → fresh start

## Session: 2026-04-06 — Automated Testing + Bug Fixes

### New: Per-beat HR Export
- **`_hr.csv` export**: New CSV with per-beat R-peak data (sweep, beat, time, rr_interval_ms, hr_bpm, sample_index, quality_label)
- **HR/RR metrics in CTA dialog**: Heart Rate (BPM) and RR Interval (ms) now available as CTA metrics
- **Location**: `export/export_manager.py` (new section after events CSV), `main.py` (breath_metric_keys dict)

### Bug Fixes
- **Off-screen window recovery**: App now validates restored window geometry is on a visible screen; resets to center if off-screen (fixes hang on disconnected monitor)
- **CTA stale data on file switch**: CTA workspace and viewmodel now cleared when loading a new file (was keeping stale CTAs from previous file)
- **EKG state clearing on ABF load**: `reset_analysis_state()` now clears ekg_chan, ecg_config, ecg_results_by_sweep (was only cleared on photometry load path)
- **CTA "Export All CSVs" button**: Now passes proper metadata from each tab's `_build_export_metadata()` + adds traceback logging on failure
- **CTA overlay mode colors**: Overlay mode now derives condition colors from user's per-metric color selection (lightness variants) instead of hardcoded palette

### Testing Infrastructure (68+ tests)
- pytest-qt GUI testing with session-scoped MainWindow + DialogWatcher
- Visual regression with screenshot baselines
- Benchmark suite with CPU/memory profiling via psutil
- CTA system tests: model round-trip, service calculation, save/reload, state clearing
- Export verification: all 4 CSV files validated (timeseries, breaths, events, hr)
- Verification plot generated purely from CSV exports

## Session: 2026-04-03 — EKG Channel Type + Heart Rate Analysis

### New Feature: EKG Channel Type
- **EKG channel type** in Channel Manager with auto-detection from channel names (ecg, ekg, heart, cardiac)
- **R-peak detection**: Pan-Tompkins algorithm with species presets (mouse/rat/human), adjustable sensitivity
- **Polarity control**: Auto / Upright / Inverted — user selects how R-peaks are identified
- **EKG panel visualization**: Raw trace (faint) + bandpass-filtered signal (blue) + R-peak markers (red triangles) + Pan-Tompkins threshold overlay (green/orange)
- **EKG Settings Dialog** (gear icon): Live preview with 3 panels — filtered signal + peaks, integrated signal + threshold, cycle-triggered average (CTA) waveform. All sliders update detection in real time.
- **Click-to-edit R-peaks**: ADD/DEL mode works on EKG panel — click to add missing peaks, click existing peaks to delete
- **Y-axis labels**: "Pleth (channel)" and "EKG (channel)" instead of raw names

### New Feature: Heart Rate Y2 Metrics
- **Heart rate (BPM)** — instantaneous HR plotted on Pleth panel right axis (orange)
- **RR interval (ms)** — inter-beat interval trace
- **RSA amplitude (BPM)** — respiratory sinus arrhythmia per breath cycle (pink), shows cardiac-respiratory coupling strength
- **Y2 autoscaling** — tightly fits actual data range instead of 0-max
- **Y2 scroll lock** — mouse wheel only zooms X axis, not Y2

### Session Save/Export
- **Session persistence**: R-peaks, labels, config, polarity saved in .pmx files, full round-trip
- **CSV export**: HR and RR interval columns added to timeseries export
- **EKG state cleared** on new file load (no stale data from previous file)

### Architecture
- `core/domain/ecg/models.py` — ECGConfig, ECGResult, HRVResult dataclasses
- `core/services/ecg_service.py` — pure Python detection + HRV + RSA service
- `dialogs/ekg_settings_dialog.py` — settings dialog with embedded matplotlib preview
- Zero new dependencies (scipy only)
- Full plan: `_internal/docs/PLANNING/EKG_HEART_RATE_INTEGRATION.md`

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
