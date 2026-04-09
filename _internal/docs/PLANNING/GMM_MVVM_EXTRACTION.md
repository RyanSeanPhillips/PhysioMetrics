# GMM MVVM Extraction Plan

> **Status**: Complete (all 4 commits done)
> **Date started**: 2026-04-07
> **Full plan**: `.claude/plans/snoopy-hatching-crayon.md`

## Overview

Extract `core/gmm_manager.py` (570 lines, 14 `self.mw` refs) into MVVM architecture:
- `core/services/gmm_service.py` — pure Python, no Qt dependencies
- `viewmodels/gmm_viewmodel.py` — QObject + signals, wraps GMMService

This is the first manager extraction (Step 4A of the MVVM refactoring plan).

## Three GMM Files

| File | Role | Action |
|------|------|--------|
| `core/gmm_clustering.py` (251 lines) | Pure utility: region building + peak storage | **Keep as-is** |
| `core/gmm_manager.py` (570 lines) | Orchestrator with `self.mw` | **Extract** into GMMService + GMMViewModel |
| `core/services/classifier_service.py` | Has pure `run_gmm()` algorithm | **Reuse** from GMMService |

## Commit History

### Commit 1: Write GMM Tests (behavioral baseline)
- **File**: `tests/test_gmm_clustering.py` (16 tests)
- **Unit tests (1-7)**: FakeMW mock + synthetic data
  - Feature collection shape and NaN handling
  - Sniffing cluster identification by IF/Ti
  - Region application stores in state
  - End-to-end pipeline (features -> GMM -> regions -> cache)
  - Eupnea mask computation from breath_type_class
  - Empty state graceful handling
- **Integration tests (8-15)**: Real MainWindow + ABF files
  - GMM runs after peak detection (probabilities populated)
  - Eupnea/sniffing regions created
  - Cache populated with expected keys
  - Plot renders with shading
  - Classifier switching rebuilds regions
  - GMM dialog opens with cached data
  - NPZ roundtrip preserves GMM data
  - File switch clears GMM state
- **Multi-channel test**: GMM across 3 pleth channels on awake recording

### Commit 2: Create GMMService (done)
- **File**: `core/services/gmm_service.py` (~380 lines)
- Pure Python, no Qt dependencies. All methods take `state` + `FilterConfig` as params.
- `GMMResult` dataclass with `to_cache_dict()`/`from_cache_dict()` for legacy compat
- Key functions: `run_automatic_clustering()`, `collect_breath_features()`, `identify_sniffing_cluster()`, `apply_sniffing_regions()`, `compute_eupnea_from_gmm()`, `compute_eupnea_from_active_classifier()`
- 5 service tests (S1-S5) verify identical output to GMMManager
- `self.mw` replacements: state param, FilterConfig, zscore_stats_fn callback, direct filters.notch_filter_1d()

### Commit 3: Create GMMViewModel + wire main.py (done)
- **File**: `viewmodels/gmm_viewmodel.py` (~140 lines)
- QObject with signals: `clustering_started`, `clustering_completed`, `clustering_failed`, `status_message`
- Provider pattern: `set_state_provider()`, `set_filter_config_provider()`, `set_zscore_stats_provider()`
- Cache property on MainWindow: `_cached_gmm_results` → delegates to `_gmm_vm.cached_results`
- All 7 delegate methods updated: `_gmm_manager.xxx()` → `_gmm_vm.xxx()`
- Added `_get_filter_config()` helper on MainWindow
- Zero regressions: 21 GMM tests + 28 existing tests all pass

### Commit 4: Remove GMMManager from main.py (done)
- Removed `from core.gmm_manager import GMMManager` from main.py
- `core/gmm_manager.py` kept on disk (tests use it as baseline reference)
- No production code imports GMMManager anymore
- 49 tests passing (21 GMM + 28 existing)

## Callers (from codeindex impact analysis)

`run_automatic_gmm_clustering` transitive callers:
- `main.py:_apply_peak_detection` (after peak detection completes)
- `main.py:_restore_session_view` (on NPZ reload)
- `main.py:on_update_eupnea_sniffing_clicked` (manual refresh button)
- `classifier_manager.py:on_eupnea_sniff_classifier_changed` (dropdown switch)
- `export_manager.py:on_save_analyzed_clicked` (pre-export refresh)
- `export_manager.py:on_view_summary_clicked` (pre-summary refresh)

`build_eupnea_sniffing_regions` direct callers:
- `gmm_manager.py:apply_gmm_sniffing_regions`
- `classifier_manager.py:on_eupnea_sniff_classifier_changed`
- `gmm_clustering_dialog.py:_apply_sniffing_to_plot`
- `main.py:_apply_peak_detection`
- `main.py:on_update_eupnea_sniffing_clicked`

## Debugging Notes

If GMM clustering breaks after the extraction:
1. Check `tests/test_gmm_clustering.py` — run all 16 tests
2. The unit tests (1-7) test GMMManager/GMMService in isolation
3. Integration tests (8-15) test the full pipeline through MainWindow
4. Key state fields: `state.gmm_sniff_probabilities`, `state.sniff_regions_by_sweep`, `state.eupnea_regions_by_sweep`, `state.all_peaks_by_sweep[sweep]['breath_type_class']`
5. Cache: `main_window._cached_gmm_results` (dict with cluster_labels, probabilities, feature_matrix, breath_cycles, sniffing_cluster_id, feature_keys)
