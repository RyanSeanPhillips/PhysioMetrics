"""
GMMManager - Handles GMM clustering for eupnea/sniffing classification.

Extracted from main.py to improve code maintainability.
Manages automatic GMM clustering, feature collection, and sniffing region detection.
"""

import numpy as np
from core import metrics, filters, gmm_clustering, telemetry


class GMMManager:
    """Manages GMM clustering operations for breath classification."""

    def __init__(self, main_window):
        """
        Args:
            main_window: Reference to MainWindow for state and UI access
        """
        self.mw = main_window

    @property
    def state(self):
        return self.mw.state

    # =========================================================================
    # Eupnea Mask Computation
    # =========================================================================

    def compute_eupnea_from_gmm(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """
        Compute eupnea mask from GMM clustering results.

        Eupnea = breaths that are NOT sniffing (based on GMM classification).
        Groups consecutive eupnic breaths into continuous regions.

        Args:
            sweep_idx: Index of the current sweep
            signal_length: Length of the signal array

        Returns:
            Boolean array (as float 0/1) marking eupneic regions
        """
        eupnea_mask = np.zeros(signal_length, dtype=bool)

        # Check if GMM probabilities are available for this sweep
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            return eupnea_mask.astype(float)

        if sweep_idx not in self.state.gmm_sniff_probabilities:
            return eupnea_mask.astype(float)

        # Get breath data for this sweep
        breath_data = self.state.breath_by_sweep.get(sweep_idx)
        if breath_data is None:
            return eupnea_mask.astype(float)

        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))

        if len(onsets) == 0:
            return eupnea_mask.astype(float)

        t = self.state.t
        gmm_probs = self.state.gmm_sniff_probabilities[sweep_idx]

        # Identify eupnic breaths and group consecutive ones
        eupnic_groups = []
        current_group_start = None
        current_group_end = None
        last_eupnic_idx = None

        # Get peaks for this sweep (needed to map breath → peak)
        peaks = self.state.peaks_by_sweep.get(sweep_idx)
        if peaks is None or len(peaks) != len(onsets):
            return eupnea_mask.astype(float)  # Peaks and onsets must be aligned

        for breath_idx in range(len(onsets)):
            # Get peak sample index for this breath
            peak_sample_idx = int(peaks[breath_idx])

            # Look up GMM probability using peak sample index
            if peak_sample_idx not in gmm_probs:
                # Close current group if exists
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = None
                    current_group_end = None
                    last_eupnic_idx = None
                continue

            sniff_prob = gmm_probs[peak_sample_idx]

            # Eupnea if sniffing probability < 0.5 (i.e., more likely eupnea)
            if sniff_prob < 0.5:
                # Get time range for this breath
                start_idx = int(onsets[breath_idx])

                # Get offset time
                if breath_idx < len(offsets):
                    end_idx = int(offsets[breath_idx])
                else:
                    # Fallback: use next onset or end of trace
                    if breath_idx + 1 < len(onsets):
                        end_idx = int(onsets[breath_idx + 1])
                    else:
                        end_idx = signal_length

                # Check if this is consecutive with the last eupnic breath
                if last_eupnic_idx is None or breath_idx != last_eupnic_idx + 1:
                    # Not consecutive - save current group and start new one
                    if current_group_start is not None:
                        eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = start_idx
                    current_group_end = end_idx
                else:
                    # Consecutive breath - extend the current group
                    current_group_end = end_idx

                last_eupnic_idx = breath_idx
            else:
                # Non-eupnic breath - close current group if exists
                if current_group_start is not None:
                    eupnic_groups.append((current_group_start, current_group_end))
                    current_group_start = None
                    current_group_end = None
                    last_eupnic_idx = None

        # Save final group if exists
        if current_group_start is not None:
            eupnic_groups.append((current_group_start, current_group_end))

        # Mark all continuous eupnic regions
        for start_idx, end_idx in eupnic_groups:
            eupnea_mask[start_idx:end_idx] = True

        return eupnea_mask.astype(float)

    def compute_eupnea_from_active_classifier(self, sweep_idx: int, signal_length: int) -> np.ndarray:
        """
        Compute eupnea mask using the active eupnea/sniff classifier.

        This is a generalized method that works with any classifier (GMM, XGBoost, RF, MLP)
        by reading from the breath_type_class array in all_peaks_by_sweep.

        Args:
            sweep_idx: Index of the current sweep
            signal_length: Length of the signal array

        Returns:
            Boolean array (as float 0/1) marking eupneic regions
        """
        eupnea_mask = np.zeros(signal_length, dtype=bool)

        # Get breath data and peak data for this sweep
        all_peaks = self.state.all_peaks_by_sweep.get(sweep_idx)
        breath_data = self.state.breath_by_sweep.get(sweep_idx)

        if all_peaks is None or breath_data is None:
            return eupnea_mask.astype(float)

        # Get breath_type_class array (contains active classifier's predictions)
        breath_type_class = all_peaks.get('breath_type_class')
        if breath_type_class is None:
            # Fall back to old GMM probability method
            return self.compute_eupnea_from_gmm(sweep_idx, signal_length)

        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))

        if len(onsets) == 0 or len(breath_type_class) != len(onsets):
            return eupnea_mask.astype(float)

        # Group consecutive eupnic breaths into continuous regions
        eupnic_groups = []
        current_group_start = None
        current_group_end = None
        last_eupnic_idx = None

        for breath_idx in range(len(onsets)):
            # Eupnea = breath_type_class == 0
            is_eupnic = (breath_type_class[breath_idx] == 0)

            if is_eupnic:
                # Get time range for this breath
                start_idx = int(onsets[breath_idx])

                # Get offset time
                if breath_idx < len(offsets):
                    end_idx = int(offsets[breath_idx])
                else:
                    if breath_idx + 1 < len(onsets):
                        end_idx = int(onsets[breath_idx + 1])
                    else:
                        end_idx = signal_length

                # Check if this is consecutive with the last eupnic breath
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

        # Save final group if exists
        if current_group_start is not None:
            eupnic_groups.append((current_group_start, current_group_end))

        # Mark all continuous eupnic regions
        for start_idx, end_idx in eupnic_groups:
            eupnea_mask[start_idx:end_idx] = True

        return eupnea_mask.astype(float)

    # =========================================================================
    # Automatic GMM Clustering
    # =========================================================================

    def run_automatic_gmm_clustering(self):
        """
        Automatically run GMM clustering after peak detection to identify sniffing breaths.
        Uses streamlined default features (if, ti, amp_insp, max_dinsp) and 2 clusters.
        """
        import time
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score

        t_start = time.time()
        st = self.state

        # Check if we have breath data
        if not st.peaks_by_sweep or len(st.peaks_by_sweep) == 0:
            print("[auto-gmm] No breath data available, skipping automatic GMM clustering")
            return

        # Streamlined default features for eupnea/sniffing separation
        feature_keys = ["if", "ti", "amp_insp", "max_dinsp"]
        n_clusters = 2

        print(f"\n[auto-gmm] Running automatic GMM clustering with {n_clusters} clusters...")
        print(f"[auto-gmm] Features: {', '.join(feature_keys)}")
        self.mw._log_status_message("Running GMM clustering...")

        try:
            # Collect breath features from all analyzed sweeps
            feature_matrix, breath_cycles = self.collect_gmm_breath_features(feature_keys)

            if len(feature_matrix) < n_clusters:
                print(f"[auto-gmm] Not enough breaths ({len(feature_matrix)}) for {n_clusters} clusters, skipping")
                return

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Fit GMM
            gmm_model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            cluster_labels = gmm_model.fit_predict(feature_matrix_scaled)

            # Get probability estimates for each breath
            cluster_probabilities = gmm_model.predict_proba(feature_matrix_scaled)

            # Check clustering quality
            silhouette = silhouette_score(feature_matrix_scaled, cluster_labels) if n_clusters > 1 else -1
            print(f"[auto-gmm] Silhouette score: {silhouette:.3f}")

            # Identify sniffing cluster
            sniffing_cluster_id = self.identify_gmm_sniffing_cluster(
                feature_matrix, cluster_labels, feature_keys, silhouette
            )

            if sniffing_cluster_id is None:
                print("[auto-gmm] Could not identify sniffing cluster, skipping")
                return

            # Apply GMM sniffing regions to plot
            self.apply_gmm_sniffing_regions(
                breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id
            )

            n_sniffing_breaths = np.sum(cluster_labels == sniffing_cluster_id)
            print(f"[auto-gmm] Identified {n_sniffing_breaths} sniffing breaths and applied to plot")

            # Cache results for fast dialog loading
            self.mw._cached_gmm_results = {
                'cluster_labels': cluster_labels,
                'cluster_probabilities': cluster_probabilities,
                'feature_matrix': feature_matrix,
                'breath_cycles': breath_cycles,
                'sniffing_cluster_id': sniffing_cluster_id,
                'feature_keys': feature_keys
            }
            print("[auto-gmm] Cached GMM results for fast dialog loading")

            # Show completion message with elapsed time
            t_elapsed = time.time() - t_start

            # Log telemetry
            eupnea_count = len(cluster_labels) - n_sniffing_breaths
            telemetry.log_feature_used('gmm_clustering')
            telemetry.log_timing('gmm_clustering', t_elapsed,
                                num_breaths=len(cluster_labels),
                                num_clusters=n_clusters,
                                silhouette_score=round(silhouette, 3))

            telemetry.log_breath_statistics(
                num_breaths=len(cluster_labels),
                sniff_count=int(n_sniffing_breaths),
                eupnea_count=int(eupnea_count),
                silhouette_score=round(silhouette, 3)
            )

            self.mw._log_status_message(f"GMM clustering complete ({t_elapsed:.1f}s)", 2000)

        except Exception as e:
            print(f"[auto-gmm] Error during automatic GMM clustering: {e}")
            t_elapsed = time.time() - t_start

            # Log telemetry: GMM clustering failure
            telemetry.log_crash(f"GMM clustering failed: {type(e).__name__}",
                               operation='gmm_clustering',
                               num_breaths=len(feature_matrix) if 'feature_matrix' in locals() else 0)

            self.mw._log_status_message(f"GMM clustering failed ({t_elapsed:.1f}s)", 3000)
            import traceback
            traceback.print_exc()

    def collect_gmm_breath_features(self, feature_keys):
        """Collect per-breath features for GMM clustering."""
        feature_matrix = []
        breath_cycles = []
        st = self.state

        for sweep_idx in sorted(st.breath_by_sweep.keys()):
            breath_data = st.breath_by_sweep[sweep_idx]

            if sweep_idx not in st.peaks_by_sweep:
                continue

            peaks = st.peaks_by_sweep[sweep_idx]
            t = st.t
            y_raw = st.sweeps[st.analyze_chan][:, sweep_idx]

            # Apply filters
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=self.mw.filter_order
            )

            # Apply notch filter if configured
            if self.mw.notch_filter_lower is not None and self.mw.notch_filter_upper is not None:
                y = self.mw._apply_notch_filter(y, st.sr_hz,
                                              self.mw.notch_filter_lower,
                                              self.mw.notch_filter_upper)

            # Apply z-score normalization if enabled
            if self.mw.use_zscore_normalization:
                if self.mw.zscore_global_mean is None or self.mw.zscore_global_std is None:
                    self.mw.zscore_global_mean, self.mw.zscore_global_std = self.mw._compute_global_zscore_stats()
                y = filters.zscore_normalize(y, self.mw.zscore_global_mean, self.mw.zscore_global_std)

            # Get breath events
            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            expmins = breath_data.get('expmins', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            if len(onsets) == 0:
                continue

            # Compute metrics
            metrics_dict = {}
            for feature_key in feature_keys:
                if feature_key in metrics.METRICS:
                    metric_arr = metrics.METRICS[feature_key](
                        t, y, st.sr_hz, peaks, onsets, offsets, expmins, expoffs
                    )
                    metrics_dict[feature_key] = metric_arr

            # Extract per-breath values
            n_breaths = len(onsets)
            for breath_idx in range(n_breaths):
                start = int(onsets[breath_idx])
                breath_features = []
                valid_breath = True

                for feature_key in feature_keys:
                    if feature_key not in metrics_dict:
                        valid_breath = False
                        break

                    metric_arr = metrics_dict[feature_key]
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

        return np.array(feature_matrix), breath_cycles

    def identify_gmm_sniffing_cluster(self, feature_matrix, cluster_labels, feature_keys, silhouette):
        """Identify which cluster represents sniffing based on IF and Ti."""
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels)

        # Get indices of IF and Ti features
        if_idx = feature_keys.index('if') if 'if' in feature_keys else None
        ti_idx = feature_keys.index('ti') if 'ti' in feature_keys else None

        if if_idx is None and ti_idx is None:
            print("[auto-gmm] Cannot identify sniffing without 'if' or 'ti' features")
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

        # Identify sniffing: highest IF and/or lowest Ti
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
            print(f"[auto-gmm]   Cluster {cluster_id}: {stats_str}{marker}")

        # Validate quality (warn but don't block)
        sniff_stats = cluster_stats[sniffing_cluster_id]
        if silhouette < 0.25:
            print(f"[auto-gmm] WARNING: Low cluster separation (silhouette={silhouette:.3f})")
            print(f"[auto-gmm]   Breathing patterns may be very similar (e.g., anesthetized mouse)")
        if if_idx is not None and sniff_stats['mean_if'] < 5.0:
            print(f"[auto-gmm] WARNING: 'Sniffing' cluster has low IF ({sniff_stats['mean_if']:.2f} Hz)")
            print(f"[auto-gmm]   May be normal variation, not true sniffing (typical sniffing: 5-8 Hz)")

        return sniffing_cluster_id

    def apply_gmm_sniffing_regions(self, breath_cycles, cluster_labels, cluster_probabilities, sniffing_cluster_id):
        """Apply GMM cluster results using shared functions from core.gmm_clustering.

        Stores classifications in all_peaks_by_sweep and builds both eupnea and sniffing regions.
        """
        # Store probabilities by (sweep_idx, breath_idx) for backward compatibility
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            self.state.gmm_sniff_probabilities = {}
        self.state.gmm_sniff_probabilities.clear()

        for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]

            if sweep_idx not in self.state.gmm_sniff_probabilities:
                self.state.gmm_sniff_probabilities[sweep_idx] = {}
            self.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Check if GMM is the active classifier
        is_gmm_active = self.state.active_eupnea_sniff_classifier == 'gmm'

        # Store classifications in all_peaks_by_sweep
        n_classified = gmm_clustering.store_gmm_classifications_in_peaks(
            self.state, breath_cycles, cluster_labels, sniffing_cluster_id,
            cluster_probabilities, confidence_threshold=0.5,
            update_editable=is_gmm_active
        )

        # Only build regions if GMM is the active classifier
        if is_gmm_active:
            results = gmm_clustering.build_eupnea_sniffing_regions(
                self.state, verbose=False, log_prefix="[auto-gmm]"
            )
        else:
            results = {'n_sniffing': 0, 'n_eupnea': 0, 'total_sniff_regions': 0, 'total_eupnea_regions': 0}
            print(f"[auto-gmm] GMM results cached but not applied (active classifier: {self.state.active_eupnea_sniff_classifier})")

        # Calculate probability statistics
        all_sniff_probs = []
        for sweep_idx in self.state.gmm_sniff_probabilities:
            for breath_idx in self.state.gmm_sniff_probabilities[sweep_idx]:
                prob = self.state.gmm_sniff_probabilities[sweep_idx][breath_idx]
                all_sniff_probs.append(prob)

        if all_sniff_probs:
            all_sniff_probs = np.array(all_sniff_probs)
            sniff_probs_of_sniff_breaths = all_sniff_probs[all_sniff_probs >= 0.5]
            if len(sniff_probs_of_sniff_breaths) > 0:
                mean_conf = np.mean(sniff_probs_of_sniff_breaths)
                min_conf = np.min(sniff_probs_of_sniff_breaths)
                uncertain_count = np.sum((sniff_probs_of_sniff_breaths >= 0.5) & (sniff_probs_of_sniff_breaths < 0.7))
                print(f"[auto-gmm]   Sniffing probability: mean={mean_conf:.3f}, min={min_conf:.3f}")
                if uncertain_count > 0:
                    print(f"[auto-gmm]   WARNING: {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")

        # Report results
        print(f"[auto-gmm]   Created {results['total_sniff_regions']} sniffing region(s) across sweeps")
        print(f"[auto-gmm]   Created {results['total_eupnea_regions']} eupnea region(s) across sweeps")

        return results['n_sniffing']

    def store_gmm_probabilities_only(self, breath_cycles, cluster_probabilities, sniffing_cluster_id):
        """Store GMM sniffing probabilities without applying regions to plot."""
        # Store probabilities by (sweep_idx, breath_idx)
        if not hasattr(self.state, 'gmm_sniff_probabilities'):
            self.state.gmm_sniff_probabilities = {}
        self.state.gmm_sniff_probabilities.clear()

        for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
            if sweep_idx not in self.state.gmm_sniff_probabilities:
                self.state.gmm_sniff_probabilities[sweep_idx] = {}

            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]
            self.state.gmm_sniff_probabilities[sweep_idx][breath_idx] = sniff_prob

        # Calculate probability statistics
        all_sniff_probs = []
        for sweep_idx in self.state.gmm_sniff_probabilities:
            for breath_idx in self.state.gmm_sniff_probabilities[sweep_idx]:
                prob = self.state.gmm_sniff_probabilities[sweep_idx][breath_idx]
                all_sniff_probs.append(prob)

        if all_sniff_probs:
            all_sniff_probs = np.array(all_sniff_probs)
            sniff_probs_of_sniff_breaths = all_sniff_probs[all_sniff_probs >= 0.5]
            if len(sniff_probs_of_sniff_breaths) > 0:
                mean_conf = np.mean(sniff_probs_of_sniff_breaths)
                min_conf = np.min(sniff_probs_of_sniff_breaths)
                uncertain_count = np.sum((sniff_probs_of_sniff_breaths >= 0.5) & (sniff_probs_of_sniff_breaths < 0.7))
                print(f"[auto-gmm]   Sniffing probability: mean={mean_conf:.3f}, min={min_conf:.3f}")
                if uncertain_count > 0:
                    print(f"[auto-gmm]   WARNING: {uncertain_count} breaths have uncertain classification (50-70% sniffing probability)")
