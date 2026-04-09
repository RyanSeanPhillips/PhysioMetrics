"""
GMMService — pure Python GMM clustering operations.

No Qt dependencies. Extracted from GMMManager (core/gmm_manager.py) as part of
the MVVM refactoring. See _internal/docs/PLANNING/GMM_MVVM_EXTRACTION.md.

All methods are stateless: they take state/config as parameters and return
results. The caller (GMMViewModel or GMMManager) is responsible for storing
results and emitting signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core import filters, gmm_clustering, metrics


# ── Result dataclass ────────────────────────────────────────────────


@dataclass
class GMMResult:
    """Immutable result from a GMM clustering run."""
    cluster_labels: np.ndarray
    cluster_probabilities: np.ndarray
    feature_matrix: np.ndarray
    breath_cycles: List[Tuple[int, int]]
    sniffing_cluster_id: Optional[int]
    feature_keys: List[str]
    silhouette_score: float = -1.0
    n_classified: int = 0

    def to_cache_dict(self) -> Dict[str, Any]:
        """Convert to the legacy cache dict format used by MainWindow."""
        return {
            'cluster_labels': self.cluster_labels,
            'cluster_probabilities': self.cluster_probabilities,
            'feature_matrix': self.feature_matrix,
            'breath_cycles': self.breath_cycles,
            'sniffing_cluster_id': self.sniffing_cluster_id,
            'feature_keys': self.feature_keys,
        }

    @classmethod
    def from_cache_dict(cls, d: Dict[str, Any]) -> GMMResult:
        """Reconstruct from legacy cache dict."""
        return cls(
            cluster_labels=d['cluster_labels'],
            cluster_probabilities=d['cluster_probabilities'],
            feature_matrix=d['feature_matrix'],
            breath_cycles=d['breath_cycles'],
            sniffing_cluster_id=d['sniffing_cluster_id'],
            feature_keys=d['feature_keys'],
        )


# ── FilterConfig import (avoid circular) ────────────────────────────

def _import_filter_config():
    from core.domain.analysis.models import FilterConfig
    return FilterConfig


# ── Feature Collection ──────────────────────────────────────────────


def collect_breath_features(
    state,
    feature_keys: List[str],
    filter_config=None,
    zscore_stats_fn: Optional[Callable[[], Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Collect per-breath features for GMM clustering.

    Args:
        state: AppState with peaks_by_sweep, breath_by_sweep, sweeps, etc.
        feature_keys: Which metrics to compute (e.g. ["if", "ti", "amp_insp", "max_dinsp"])
        filter_config: FilterConfig with filter/notch/zscore params. If None, uses state defaults.
        zscore_stats_fn: Callable returning (global_mean, global_std) for z-score. Called lazily.

    Returns:
        (feature_matrix, breath_cycles) where feature_matrix is (n_breaths, n_features)
        and breath_cycles is list of (sweep_idx, breath_idx) tuples.
    """
    feature_matrix = []
    breath_cycles = []

    # Extract filter params from FilterConfig or state defaults
    if filter_config is not None:
        fc = filter_config
    else:
        fc = None

    for sweep_idx in sorted(state.breath_by_sweep.keys()):
        breath_data = state.breath_by_sweep[sweep_idx]

        if sweep_idx not in state.peaks_by_sweep:
            continue

        peaks = state.peaks_by_sweep[sweep_idx]
        y_raw = state.sweeps[state.analyze_chan][:, sweep_idx]

        # Apply filters
        if fc is not None:
            y = filters.apply_all_1d(
                y_raw, state.sr_hz,
                fc.use_low, fc.low_hz,
                fc.use_high, fc.high_hz,
                fc.use_mean_sub, fc.mean_val,
                fc.use_invert,
                order=fc.filter_order,
            )
            # Notch filter
            if fc.notch_lower is not None and fc.notch_upper is not None:
                y = filters.notch_filter_1d(y, state.sr_hz, fc.notch_lower, fc.notch_upper)
            # Z-score normalization
            if fc.use_zscore:
                g_mean = fc.zscore_global_mean
                g_std = fc.zscore_global_std
                if (g_mean is None or g_std is None) and zscore_stats_fn is not None:
                    g_mean, g_std = zscore_stats_fn()
                if g_mean is not None and g_std is not None:
                    y = filters.zscore_normalize(y, g_mean, g_std)
        else:
            # Minimal filtering from state only (legacy path)
            y = filters.apply_all_1d(
                y_raw, state.sr_hz,
                state.use_low, state.low_hz,
                state.use_high, state.high_hz,
                state.use_mean_sub, state.mean_val,
                state.use_invert,
                order=4,
            )

        # Get breath events
        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))
        expmins = breath_data.get('expmins', np.array([]))
        expoffs = breath_data.get('expoffs', np.array([]))

        if len(onsets) == 0:
            continue

        # Compute metrics
        t = state.t
        metrics_dict = {}
        for fk in feature_keys:
            if fk in metrics.METRICS:
                metric_arr = metrics.METRICS[fk](
                    t, y, state.sr_hz, peaks, onsets, offsets, expmins, expoffs
                )
                metrics_dict[fk] = metric_arr

        # Extract per-breath values
        n_breaths = len(onsets)
        for breath_idx in range(n_breaths):
            start = int(onsets[breath_idx])
            breath_features = []
            valid_breath = True

            for fk in feature_keys:
                if fk not in metrics_dict:
                    valid_breath = False
                    break

                metric_arr = metrics_dict[fk]
                if start < len(metric_arr):
                    val = metric_arr[start]
                    if np.isnan(val) or not np.isfinite(val):
                        valid_breath = False
                        break
                    breath_features.append(val)
                else:
                    valid_breath = False
                    break

            if valid_breath and len(breath_features) == len(feature_keys):
                feature_matrix.append(breath_features)
                breath_cycles.append((sweep_idx, breath_idx))

    return np.array(feature_matrix) if feature_matrix else np.empty((0, len(feature_keys))), breath_cycles


# ── Sniffing Cluster Identification ────────────────────────────────


def identify_sniffing_cluster(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    feature_keys: List[str],
    silhouette: float,
) -> Optional[int]:
    """Identify which cluster represents sniffing based on IF and Ti.

    Sniffing = highest instantaneous frequency (IF) and/or lowest inspiratory time (Ti).

    Args:
        feature_matrix: (n_breaths, n_features) array
        cluster_labels: Cluster assignment per breath
        feature_keys: Feature names (must include 'if' or 'ti')
        silhouette: Clustering quality score (for warning only)

    Returns:
        Cluster ID of the sniffing cluster, or None if can't determine.
    """
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)

    if_idx = feature_keys.index('if') if 'if' in feature_keys else None
    ti_idx = feature_keys.index('ti') if 'ti' in feature_keys else None

    if if_idx is None and ti_idx is None:
        print("[gmm-service] Cannot identify sniffing without 'if' or 'ti' features")
        return None

    # Compute mean IF and Ti for each cluster
    cluster_stats = {}
    for cluster_id in unique_labels:
        mask = cluster_labels == cluster_id
        stats = {}
        if if_idx is not None:
            stats['mean_if'] = np.mean(feature_matrix[mask, if_idx])
        if ti_idx is not None:
            stats['mean_ti'] = np.mean(feature_matrix[mask, ti_idx])
        cluster_stats[cluster_id] = stats

    # Score: highest IF rank + lowest Ti rank → sniffing
    cluster_scores = {}
    for cluster_id in unique_labels:
        score = 0
        if if_idx is not None:
            if_vals = [cluster_stats[c]['mean_if'] for c in unique_labels]
            if_rank = sorted(if_vals).index(cluster_stats[cluster_id]['mean_if'])
            score += if_rank / (n_clusters - 1) if n_clusters > 1 else 0
        if ti_idx is not None:
            ti_vals = [cluster_stats[c]['mean_ti'] for c in unique_labels]
            ti_rank = sorted(ti_vals, reverse=True).index(cluster_stats[cluster_id]['mean_ti'])
            score += ti_rank / (n_clusters - 1) if n_clusters > 1 else 0
        cluster_scores[cluster_id] = score

    sniffing_cluster_id = max(cluster_scores, key=cluster_scores.get)

    # Log cluster statistics
    for cluster_id in unique_labels:
        stats_str = ", ".join([f"{k}={v:.3f}" for k, v in cluster_stats[cluster_id].items()])
        marker = " (SNIFFING)" if cluster_id == sniffing_cluster_id else ""
        print(f"[gmm-service]   Cluster {cluster_id}: {stats_str}{marker}")

    # Quality warnings
    sniff_stats = cluster_stats[sniffing_cluster_id]
    if silhouette < 0.25:
        print(f"[gmm-service] WARNING: Low cluster separation (silhouette={silhouette:.3f})")
        print(f"[gmm-service]   Breathing patterns may be very similar (e.g., anesthetized mouse)")
    if if_idx is not None and sniff_stats['mean_if'] < 5.0:
        print(f"[gmm-service] WARNING: 'Sniffing' cluster has low IF ({sniff_stats['mean_if']:.2f} Hz)")
        print(f"[gmm-service]   May be normal variation, not true sniffing (typical sniffing: 5-8 Hz)")

    return sniffing_cluster_id


# ── Region Application ─────────────────────────────────────────────


def apply_sniffing_regions(
    state,
    breath_cycles: List[Tuple[int, int]],
    cluster_labels: np.ndarray,
    cluster_probabilities: np.ndarray,
    sniffing_cluster_id: int,
) -> int:
    """Apply GMM cluster results: store classifications and build regions.

    Stores classifications in all_peaks_by_sweep and builds both eupnea
    and sniffing regions if GMM is the active classifier.

    Args:
        state: AppState (modified in-place)
        breath_cycles: List of (sweep_idx, breath_idx) tuples
        cluster_labels: Cluster assignment per breath
        cluster_probabilities: (n_breaths, n_clusters) probability matrix
        sniffing_cluster_id: Which cluster is sniffing

    Returns:
        Number of sniffing breaths identified.
    """
    # Store probabilities by (sweep_idx, breath_idx)
    if not hasattr(state, 'gmm_sniff_probabilities'):
        state.gmm_sniff_probabilities = {}
    state.gmm_sniff_probabilities.clear()

    for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
        sniff_prob = cluster_probabilities[i, sniffing_cluster_id]
        if sweep_idx not in state.gmm_sniff_probabilities:
            state.gmm_sniff_probabilities[sweep_idx] = {}
        state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

    # Check if GMM is the active classifier
    is_gmm_active = state.active_eupnea_sniff_classifier == 'gmm'

    # Store classifications in all_peaks_by_sweep
    n_classified = gmm_clustering.store_gmm_classifications_in_peaks(
        state, breath_cycles, cluster_labels, sniffing_cluster_id,
        cluster_probabilities, confidence_threshold=0.5,
        update_editable=is_gmm_active,
    )

    # Only build regions if GMM is the active classifier
    if is_gmm_active:
        results = gmm_clustering.build_eupnea_sniffing_regions(
            state, verbose=False, log_prefix="[gmm-service]"
        )
    else:
        results = {'n_sniffing': 0, 'n_eupnea': 0, 'total_sniff_regions': 0, 'total_eupnea_regions': 0}
        print(f"[gmm-service] GMM results cached but not applied (active classifier: {state.active_eupnea_sniff_classifier})")

    # Probability statistics
    _log_probability_stats(state)

    # Report results
    n_sniffing = np.sum(cluster_labels == sniffing_cluster_id)
    print(f"[gmm-service]   Created {results['total_sniff_regions']} sniffing region(s) across sweeps")
    print(f"[gmm-service]   Created {results['total_eupnea_regions']} eupnea region(s) across sweeps")

    return int(n_sniffing)


def store_probabilities_only(
    state,
    breath_cycles: List[Tuple[int, int]],
    cluster_probabilities: np.ndarray,
    sniffing_cluster_id: int,
):
    """Store GMM sniffing probabilities without applying regions to plot."""
    if not hasattr(state, 'gmm_sniff_probabilities'):
        state.gmm_sniff_probabilities = {}
    state.gmm_sniff_probabilities.clear()

    for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
        if sweep_idx not in state.gmm_sniff_probabilities:
            state.gmm_sniff_probabilities[sweep_idx] = {}
        sniff_prob = cluster_probabilities[i, sniffing_cluster_id]
        state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

    _log_probability_stats(state)


def _log_probability_stats(state):
    """Log summary statistics of sniffing probabilities."""
    all_sniff_probs = []
    for sweep_probs in state.gmm_sniff_probabilities.values():
        if isinstance(sweep_probs, dict):
            all_sniff_probs.extend(sweep_probs.values())
        else:
            all_sniff_probs.extend(sweep_probs)

    if all_sniff_probs:
        all_sniff_probs = np.array(all_sniff_probs)
        sniff_probs_of_sniff_breaths = all_sniff_probs[all_sniff_probs >= 0.5]
        if len(sniff_probs_of_sniff_breaths) > 0:
            mean_conf = np.mean(sniff_probs_of_sniff_breaths)
            min_conf = np.min(sniff_probs_of_sniff_breaths)
            uncertain_count = np.sum((sniff_probs_of_sniff_breaths >= 0.5) & (sniff_probs_of_sniff_breaths < 0.7))
            print(f"[gmm-service]   Sniffing probability: mean={mean_conf:.3f}, min={min_conf:.3f}")
            if uncertain_count > 0:
                print(f"[gmm-service]   WARNING: {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")


# ── Eupnea Mask Computation ────────────────────────────────────────


def compute_eupnea_from_gmm(state, sweep_idx: int, signal_length: int) -> np.ndarray:
    """Compute eupnea mask from GMM clustering results.

    Eupnea = breaths where sniffing probability < 0.5.
    Groups consecutive eupnic breaths into continuous regions.

    Returns:
        Float array (0/1) of shape (signal_length,) marking eupneic regions.
    """
    eupnea_mask = np.zeros(signal_length, dtype=bool)

    if not hasattr(state, 'gmm_sniff_probabilities'):
        return eupnea_mask.astype(float)

    if sweep_idx not in state.gmm_sniff_probabilities:
        return eupnea_mask.astype(float)

    breath_data = state.breath_by_sweep.get(sweep_idx)
    if breath_data is None:
        return eupnea_mask.astype(float)

    onsets = breath_data.get('onsets', np.array([]))
    offsets = breath_data.get('offsets', np.array([]))
    if len(onsets) == 0:
        return eupnea_mask.astype(float)

    peaks = state.peaks_by_sweep.get(sweep_idx)
    if peaks is None or len(peaks) != len(onsets):
        return eupnea_mask.astype(float)

    gmm_probs = state.gmm_sniff_probabilities[sweep_idx]

    eupnic_groups = []
    current_group_start = None
    current_group_end = None
    last_eupnic_idx = None

    for breath_idx in range(len(onsets)):
        peak_sample_idx = int(peaks[breath_idx])

        if peak_sample_idx not in gmm_probs:
            if current_group_start is not None:
                eupnic_groups.append((current_group_start, current_group_end))
                current_group_start = None
                current_group_end = None
                last_eupnic_idx = None
            continue

        sniff_prob = gmm_probs[peak_sample_idx]

        if sniff_prob < 0.5:
            start_idx = int(onsets[breath_idx])
            if breath_idx < len(offsets):
                end_idx = int(offsets[breath_idx])
            elif breath_idx + 1 < len(onsets):
                end_idx = int(onsets[breath_idx + 1])
            else:
                end_idx = signal_length

            if last_eupnic_idx is None or breath_idx != last_eupnic_idx + 1:
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                current_group_start = start_idx
                current_group_end = end_idx
            else:
                current_group_end = end_idx

            last_eupnic_idx = breath_idx
        else:
            if current_group_start is not None:
                eupnic_groups.append((current_group_start, current_group_end))
                current_group_start = None
                current_group_end = None
                last_eupnic_idx = None

    if current_group_start is not None:
        eupnic_groups.append((current_group_start, current_group_end))

    for start_idx, end_idx in eupnic_groups:
        eupnea_mask[start_idx:end_idx] = True

    return eupnea_mask.astype(float)


def compute_eupnea_from_active_classifier(state, sweep_idx: int, signal_length: int) -> np.ndarray:
    """Compute eupnea mask using the active eupnea/sniff classifier.

    Works with any classifier (GMM, XGBoost, RF, MLP) by reading from
    the breath_type_class array in all_peaks_by_sweep.

    Returns:
        Float array (0/1) of shape (signal_length,) marking eupneic regions.
    """
    eupnea_mask = np.zeros(signal_length, dtype=bool)

    all_peaks = state.all_peaks_by_sweep.get(sweep_idx)
    breath_data = state.breath_by_sweep.get(sweep_idx)

    if all_peaks is None or breath_data is None:
        return eupnea_mask.astype(float)

    breath_type_class = all_peaks.get('breath_type_class')
    if breath_type_class is None:
        return compute_eupnea_from_gmm(state, sweep_idx, signal_length)

    onsets = breath_data.get('onsets', np.array([]))
    offsets = breath_data.get('offsets', np.array([]))

    if len(onsets) == 0 or len(breath_type_class) != len(onsets):
        return eupnea_mask.astype(float)

    eupnic_groups = []
    current_group_start = None
    current_group_end = None
    last_eupnic_idx = None

    for breath_idx in range(len(onsets)):
        is_eupnic = (breath_type_class[breath_idx] == 0)

        if is_eupnic:
            start_idx = int(onsets[breath_idx])
            if breath_idx < len(offsets):
                end_idx = int(offsets[breath_idx])
            elif breath_idx + 1 < len(onsets):
                end_idx = int(onsets[breath_idx + 1])
            else:
                end_idx = signal_length

            if last_eupnic_idx is None or breath_idx != last_eupnic_idx + 1:
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                current_group_start = start_idx
                current_group_end = end_idx
            else:
                current_group_end = end_idx

            last_eupnic_idx = breath_idx
        else:
            if current_group_start is not None:
                eupnic_groups.append((current_group_start, current_group_end))
                current_group_start = None
                current_group_end = None
                last_eupnic_idx = None

    if current_group_start is not None:
        eupnic_groups.append((current_group_start, current_group_end))

    for start_idx, end_idx in eupnic_groups:
        eupnea_mask[start_idx:end_idx] = True

    return eupnea_mask.astype(float)


# ── Full Pipeline ──────────────────────────────────────────────────


def run_automatic_clustering(
    state,
    filter_config=None,
    zscore_stats_fn: Optional[Callable[[], Tuple[float, float]]] = None,
) -> Optional[GMMResult]:
    """Run automatic GMM clustering to identify sniffing breaths.

    This is the main entry point — collects features, fits GMM, identifies
    sniffing cluster, applies regions, and returns the result.

    Args:
        state: AppState with peaks, breaths, sweeps
        filter_config: FilterConfig for signal processing. If None, uses state defaults.
        zscore_stats_fn: Callable returning (global_mean, global_std)

    Returns:
        GMMResult with all clustering data, or None if clustering failed.
    """
    import time as _time
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    t_start = _time.time()

    # Check if we have breath data
    if not state.peaks_by_sweep or len(state.peaks_by_sweep) == 0:
        print("[gmm-service] No breath data available, skipping automatic GMM clustering")
        return None

    feature_keys = ["if", "ti", "amp_insp", "max_dinsp"]
    n_clusters = 2

    print(f"\n[gmm-service] Running automatic GMM clustering with {n_clusters} clusters...")
    print(f"[gmm-service] Features: {', '.join(feature_keys)}")

    try:
        # Collect features
        feature_matrix, breath_cycles = collect_breath_features(
            state, feature_keys, filter_config, zscore_stats_fn,
        )

        if len(feature_matrix) < n_clusters:
            print(f"[gmm-service] Not enough breaths ({len(feature_matrix)}) for {n_clusters} clusters, skipping")
            return None

        # Standardize and fit GMM
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        gmm_model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        cluster_labels = gmm_model.fit_predict(feature_matrix_scaled)
        cluster_probabilities = gmm_model.predict_proba(feature_matrix_scaled)

        silhouette = silhouette_score(feature_matrix_scaled, cluster_labels) if n_clusters > 1 else -1
        print(f"[gmm-service] Silhouette score: {silhouette:.3f}")

        # Identify sniffing cluster
        sniffing_cluster_id = identify_sniffing_cluster(
            feature_matrix, cluster_labels, feature_keys, silhouette,
        )

        if sniffing_cluster_id is None:
            print("[gmm-service] Could not identify sniffing cluster, skipping")
            return None

        # Apply regions
        n_sniffing = apply_sniffing_regions(
            state, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id,
        )

        print(f"[gmm-service] Identified {n_sniffing} sniffing breaths and applied to plot")

        t_elapsed = _time.time() - t_start

        # Log telemetry
        from core import telemetry
        eupnea_count = len(cluster_labels) - n_sniffing
        telemetry.log_feature_used('gmm_clustering')
        telemetry.log_timing('gmm_clustering', t_elapsed,
                             num_breaths=len(cluster_labels),
                             num_clusters=n_clusters,
                             silhouette_score=round(silhouette, 3))
        telemetry.log_breath_statistics(
            num_breaths=len(cluster_labels),
            sniff_count=int(n_sniffing),
            eupnea_count=int(eupnea_count),
            silhouette_score=round(silhouette, 3),
        )

        return GMMResult(
            cluster_labels=cluster_labels,
            cluster_probabilities=cluster_probabilities,
            feature_matrix=feature_matrix,
            breath_cycles=breath_cycles,
            sniffing_cluster_id=sniffing_cluster_id,
            feature_keys=feature_keys,
            silhouette_score=silhouette,
            n_classified=n_sniffing,
        )

    except Exception as e:
        t_elapsed = _time.time() - t_start
        print(f"[gmm-service] Error during automatic GMM clustering: {e}")

        from core import telemetry
        telemetry.log_crash(f"GMM clustering failed: {type(e).__name__}",
                            operation='gmm_clustering',
                            num_breaths=len(feature_matrix) if 'feature_matrix' in locals() else 0)

        import traceback
        traceback.print_exc()
        return None
