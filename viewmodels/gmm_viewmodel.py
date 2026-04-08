"""
GMMViewModel — Qt bridge for GMM clustering operations.

Wraps GMMService with pyqtSignals so the UI can react to clustering
changes without the service knowing about Qt. Uses provider callbacks
for state, filter config, and z-score stats to avoid direct MainWindow refs.

See _internal/docs/PLANNING/GMM_MVVM_EXTRACTION.md for context.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.services import gmm_service as svc
from core.services.gmm_service import GMMResult


class GMMViewModel(QObject):
    """ViewModel for GMM clustering.

    Signals:
        clustering_started: Emitted when clustering begins.
        clustering_completed: Emitted with GMMResult when clustering succeeds.
        clustering_failed: Emitted with error message when clustering fails.
        status_message: Emitted with (message, duration_ms) for status bar.
    """

    clustering_started = pyqtSignal()
    clustering_completed = pyqtSignal(object)   # GMMResult
    clustering_failed = pyqtSignal(str)
    status_message = pyqtSignal(str, int)       # (message, duration_ms)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Provider callbacks (set by main.py wiring)
        self._state_provider: Optional[Callable] = None
        self._filter_config_provider: Optional[Callable] = None
        self._zscore_stats_provider: Optional[Callable] = None

        # Cache (replaces MainWindow._cached_gmm_results)
        self._cached_results: Optional[Dict[str, Any]] = None

    # ── Provider setup ──────────────────────────────────────────────

    def set_state_provider(self, fn: Callable):
        """Set callback that returns the current AppState."""
        self._state_provider = fn

    def set_filter_config_provider(self, fn: Callable):
        """Set callback that returns a FilterConfig."""
        self._filter_config_provider = fn

    def set_zscore_stats_provider(self, fn: Callable):
        """Set callback that returns (global_mean, global_std)."""
        self._zscore_stats_provider = fn

    # ── Cache property ──────────────────────────────────────────────

    @property
    def cached_results(self) -> Optional[Dict[str, Any]]:
        """Legacy cache dict for backward compat with dialog/export."""
        return self._cached_results

    def set_cached_results(self, value: Optional[Dict[str, Any]]):
        """Set cache from external source (e.g., NPZ reload)."""
        self._cached_results = value

    # ── Clustering ──────────────────────────────────────────────────

    def run_automatic_gmm_clustering(self):
        """Run automatic GMM clustering. Main entry point.

        Emits clustering_started, then clustering_completed or clustering_failed.
        Also emits status_message for the status bar.
        """
        state = self._state_provider() if self._state_provider else None
        if state is None:
            self.clustering_failed.emit("No state available")
            return

        filter_config = self._filter_config_provider() if self._filter_config_provider else None
        zscore_fn = self._zscore_stats_provider

        self.clustering_started.emit()
        self.status_message.emit("Running GMM clustering...", 0)

        result = svc.run_automatic_clustering(state, filter_config, zscore_fn)

        if result is not None:
            self._cached_results = result.to_cache_dict()
            self.clustering_completed.emit(result)
            elapsed_msg = f"GMM clustering complete"
            self.status_message.emit(elapsed_msg, 2000)
        else:
            self.clustering_failed.emit("GMM clustering returned no results")
            self.status_message.emit("GMM clustering failed", 3000)

    # ── Feature collection (for dialog) ─────────────────────────────

    def collect_gmm_breath_features(self, feature_keys):
        """Collect features for the GMM dialog. Returns (feature_matrix, breath_cycles)."""
        state = self._state_provider() if self._state_provider else None
        if state is None:
            return np.empty((0, len(feature_keys))), []
        filter_config = self._filter_config_provider() if self._filter_config_provider else None
        return svc.collect_breath_features(state, feature_keys, filter_config, self._zscore_stats_provider)

    def identify_gmm_sniffing_cluster(self, feature_matrix, cluster_labels, feature_keys, silhouette):
        """Identify sniffing cluster. Delegates to service."""
        return svc.identify_sniffing_cluster(feature_matrix, cluster_labels, feature_keys, silhouette)

    def apply_gmm_sniffing_regions(self, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id):
        """Apply GMM regions. Delegates to service."""
        state = self._state_provider() if self._state_provider else None
        if state is None:
            return 0
        return svc.apply_sniffing_regions(state, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id)

    def store_gmm_probabilities_only(self, breath_cycles, cluster_probabilities, sniffing_cluster_id):
        """Store probabilities without regions. Delegates to service."""
        state = self._state_provider() if self._state_provider else None
        if state is None:
            return
        svc.store_probabilities_only(state, breath_cycles, cluster_probabilities, sniffing_cluster_id)

    # ── Eupnea mask computation ─────────────────────────────────────

    def compute_eupnea_from_gmm(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """Compute eupnea mask from GMM results."""
        state = self._state_provider() if self._state_provider else None
        if state is None:
            return np.zeros(signal_length, dtype=float)
        return svc.compute_eupnea_from_gmm(state, sweep_idx, signal_length)

    def compute_eupnea_from_active_classifier(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """Compute eupnea mask from active classifier."""
        state = self._state_provider() if self._state_provider else None
        if state is None:
            return np.zeros(signal_length, dtype=float)
        return svc.compute_eupnea_from_active_classifier(state, sweep_idx, signal_length)
