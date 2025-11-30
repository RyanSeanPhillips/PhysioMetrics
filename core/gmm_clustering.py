"""
Shared GMM clustering utilities for breath classification.

This module provides functions for building eupnea and sniffing regions
from GMM classification results stored in all_peaks_by_sweep.
"""

import numpy as np


def build_eupnea_sniffing_regions(state, verbose=True, log_prefix="[gmm]"):
    """
    Build eupnea and sniffing regions from GMM classifications in all_peaks_by_sweep.

    Reads the 'gmm_class' field from state.all_peaks_by_sweep where:
    - 0 = eupnea
    - 1 = sniffing
    - -1 = unclassified

    Creates time-based regions by grouping consecutive breaths of the same type.

    Args:
        state: Application state object containing:
            - all_peaks_by_sweep: Dict with gmm_class field
            - breath_by_sweep: Dict with onsets/offsets
            - peaks_by_sweep: Dict with peak indices
            - t: Time array
            - sniff_regions_by_sweep: Will be populated
            - eupnea_regions_by_sweep: Will be populated
        verbose: If True, print region creation messages
        log_prefix: Prefix for log messages

    Returns:
        dict with keys:
            - n_sniffing: Number of sniffing breaths
            - n_eupnea: Number of eupnea breaths
            - total_sniff_regions: Number of sniffing regions created
            - total_eupnea_regions: Number of eupnea regions created
    """
    # Initialize region storage
    if not hasattr(state, 'sniff_regions_by_sweep'):
        state.sniff_regions_by_sweep = {}
    state.sniff_regions_by_sweep.clear()

    if not hasattr(state, 'eupnea_regions_by_sweep'):
        state.eupnea_regions_by_sweep = {}
    state.eupnea_regions_by_sweep.clear()

    # Counters
    total_sniff_regions = 0
    total_eupnea_regions = 0
    n_confident_sniffs = 0
    n_confident_eupnea = 0

    # Iterate through all sweeps
    for sweep_idx in sorted(state.all_peaks_by_sweep.keys()):
        all_peaks = state.all_peaks_by_sweep[sweep_idx]

        # Skip if no GMM classification
        if 'gmm_class' not in all_peaks:
            continue

        # Get breath data for time mapping
        breath_data = state.breath_by_sweep.get(sweep_idx)
        if breath_data is None:
            continue

        onsets = breath_data.get('onsets', np.array([]))
        offsets = breath_data.get('offsets', np.array([]))
        t = state.t

        if len(onsets) == 0:
            continue

        # Build regions by iterating through labeled breaths (label=1)
        sniffing_runs = []
        eupnea_runs = []
        current_sniff_start = None
        current_sniff_end = None
        current_eupnea_start = None
        current_eupnea_end = None

        # Iterate through all peaks (but only process label=1 breaths with gmm classification)
        for i, (peak_idx, label, gmm_class) in enumerate(zip(
            all_peaks['indices'],
            all_peaks['labels'],
            all_peaks['gmm_class']
        )):
            # Skip non-breath peaks (label=0) or unclassified peaks (gmm_class=-1)
            # IMPORTANT: Don't close runs when skipping - let consecutive breaths stay continuous
            # Only close when switching between eupnea ↔ sniffing
            if label == 0 or gmm_class == -1:
                continue

            # Find this peak's breath index to get onset/offset
            peaks_array = state.peaks_by_sweep.get(sweep_idx, np.array([]))
            breath_mask = peaks_array == peak_idx
            if not breath_mask.any():
                continue
            breath_idx = np.where(breath_mask)[0][0]

            if breath_idx >= len(onsets):
                continue

            # Get time range for this breath
            start_idx = int(onsets[breath_idx])
            if breath_idx < len(offsets):
                end_idx = int(offsets[breath_idx])
            else:
                if breath_idx + 1 < len(onsets):
                    end_idx = int(onsets[breath_idx + 1])
                else:
                    end_idx = len(t) - 1

            # Classify based on gmm_class label
            if gmm_class == 1:  # Sniffing
                n_confident_sniffs += 1

                if current_sniff_start is None:
                    current_sniff_start = start_idx
                    current_sniff_end = end_idx
                else:
                    current_sniff_end = end_idx

                # Close eupnea run
                if current_eupnea_start is not None:
                    eupnea_runs.append((current_eupnea_start, current_eupnea_end))
                    current_eupnea_start = None
                    current_eupnea_end = None

            elif gmm_class == 0:  # Eupnea
                n_confident_eupnea += 1

                if current_eupnea_start is None:
                    current_eupnea_start = start_idx
                    current_eupnea_end = end_idx
                else:
                    current_eupnea_end = end_idx

                # Close sniffing run
                if current_sniff_start is not None:
                    sniffing_runs.append((current_sniff_start, current_sniff_end))
                    current_sniff_start = None
                    current_sniff_end = None

        # Add final runs
        if current_sniff_start is not None:
            sniffing_runs.append((current_sniff_start, current_sniff_end))
        if current_eupnea_start is not None:
            eupnea_runs.append((current_eupnea_start, current_eupnea_end))

        # Convert runs to time ranges and store SNIFFING regions
        if sniffing_runs:
            if sweep_idx not in state.sniff_regions_by_sweep:
                state.sniff_regions_by_sweep[sweep_idx] = []

            for (start_idx, end_idx) in sniffing_runs:
                start_time = t[start_idx]
                end_time = t[end_idx]
                state.sniff_regions_by_sweep[sweep_idx].append((start_time, end_time))
                if verbose:
                    print(f"{log_prefix} Created sniffing region: {start_time:.3f} - {end_time:.3f} s")

            total_sniff_regions += len(sniffing_runs)

        # Convert runs to time ranges and store EUPNEA regions
        if eupnea_runs:
            if sweep_idx not in state.eupnea_regions_by_sweep:
                state.eupnea_regions_by_sweep[sweep_idx] = []

            for (start_idx, end_idx) in eupnea_runs:
                start_time = t[start_idx]
                end_time = t[end_idx]
                state.eupnea_regions_by_sweep[sweep_idx].append((start_time, end_time))
                if verbose:
                    print(f"{log_prefix} Created eupnea region: {start_time:.3f} - {end_time:.3f} s")

            total_eupnea_regions += len(eupnea_runs)

    return {
        'n_sniffing': n_confident_sniffs,
        'n_eupnea': n_confident_eupnea,
        'total_sniff_regions': total_sniff_regions,
        'total_eupnea_regions': total_eupnea_regions
    }


def store_gmm_classifications_in_peaks(state, breath_cycles, cluster_labels, sniffing_cluster_id,
                                       cluster_probabilities=None, confidence_threshold=0.5,
                                       update_editable=True):
    """
    Store GMM classification results in all_peaks_by_sweep['gmm_class'] field.

    This allows the classification to persist through manual peak edits, since it's
    keyed by peak sample index (which is stable).

    Args:
        state: Application state object
        breath_cycles: List of (sweep_idx, breath_idx) tuples
        cluster_labels: Array of cluster IDs for each breath
        sniffing_cluster_id: Which cluster is sniffing
        cluster_probabilities: Optional probability matrix (n_breaths, n_clusters)
        confidence_threshold: Minimum probability to classify as sniffing (default 0.5)
        update_editable: If True, also update gmm_class (editable). If False, only update
                        gmm_class_ro (read-only reference for classifier switching).

    Returns:
        Number of breaths classified
    """
    n_classified = 0

    for i, (sweep_idx, breath_idx) in enumerate(breath_cycles):
        # Get peak sample index for this breath
        peaks = state.peaks_by_sweep.get(sweep_idx)
        if peaks is None or breath_idx >= len(peaks):
            continue
        peak_sample_idx = int(peaks[breath_idx])

        # Get all_peaks for this sweep
        all_peaks = state.all_peaks_by_sweep.get(sweep_idx)
        if all_peaks is None:
            continue

        # Initialize gmm_class and gmm_class_ro arrays if they don't exist
        if 'gmm_class' not in all_peaks:
            all_peaks['gmm_class'] = np.full(len(all_peaks['indices']), -1, dtype=np.int8)
        if 'gmm_class_ro' not in all_peaks:
            all_peaks['gmm_class_ro'] = np.full(len(all_peaks['indices']), -1, dtype=np.int8)

        # Find this peak in all_peaks_by_sweep
        peak_mask = all_peaks['indices'] == peak_sample_idx
        if not peak_mask.any():
            continue
        peak_pos = np.where(peak_mask)[0][0]

        # Determine classification
        if cluster_probabilities is not None:
            # Use probability threshold
            sniff_prob = cluster_probabilities[i, sniffing_cluster_id]
            gmm_class = 1 if sniff_prob >= confidence_threshold else 0
        else:
            # Use hard cluster assignment
            gmm_class = 1 if cluster_labels[i] == sniffing_cluster_id else 0

        # Store in read-only array (always), and editable array (if update_editable=True)
        all_peaks['gmm_class_ro'][peak_pos] = gmm_class  # Read-only reference for classifier switching
        if update_editable:
            all_peaks['gmm_class'][peak_pos] = gmm_class
        n_classified += 1

    return n_classified
