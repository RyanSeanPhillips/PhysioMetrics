"""
CTA (Condition-Triggered Average) Service.

Business logic for calculating CTAs from event markers and signals.
This service is Qt-free and can be used from CLI or UI.
"""

from typing import List, Dict, Optional, Tuple, Callable
import numpy as np
from scipy import stats

from core.domain.cta import CTAConfig, CTATrace, CTAResult, CTACollection
from core.domain.events import EventMarker, MarkerType


class CTAService:
    """
    Service for CTA calculation and persistence.

    This class contains all the business logic for:
    - Extracting event-aligned windows from signals
    - Interpolating to common time base
    - Computing mean and SEM
    - Serializing to/from NPZ format
    """

    def __init__(self):
        """Initialize the CTA service."""
        pass

    def calculate_cta(
        self,
        time_array: np.ndarray,
        signal: np.ndarray,
        event_times: List[float],
        event_ids: Optional[List[str]] = None,
        baseline_ref_times: Optional[List[float]] = None,
        paired_event_times: Optional[List[float]] = None,
        sweep_idx: int = 0,
        config: Optional[CTAConfig] = None,
        metric_key: str = 'signal',
        metric_label: str = 'Signal',
        category: str = '',
        label: str = '',
        alignment: str = 'onset',
    ) -> CTAResult:
        """
        Calculate CTA for a single metric aligned to event times.

        Args:
            time_array: Time values for the signal (seconds)
            signal: Signal values (same length as time_array)
            event_times: List of event times to align to (seconds)
            event_ids: Optional list of event IDs (same length as event_times)
            baseline_ref_times: Optional list of times for z-score baseline calculation.
                               If None, uses event_times. For withdrawal CTAs, pass onset
                               times here so baseline is calculated relative to onset.
            sweep_idx: Sweep index for the traces
            config: CTA configuration (uses defaults if None)
            metric_key: Key for the metric
            metric_label: Human-readable label for the metric
            category: Marker category
            label: Marker label
            alignment: 'onset' or 'withdrawal'

        Returns:
            CTAResult with traces, mean, and SEM
        """
        if config is None:
            config = CTAConfig()

        if event_ids is None:
            event_ids = [f"event_{i}" for i in range(len(event_times))]

        # If no baseline reference times provided, use event times
        if baseline_ref_times is None:
            baseline_ref_times = event_times

        traces = []
        window_before = float(config.window_before)
        window_after = float(config.window_after)
        n_points = int(config.n_points)  # Ensure integer for np.linspace

        # Auto-scale n_points for large windows to maintain ~16.7 Hz resolution
        # Default: 1000 points over 60s = 16.7 pts/s
        total_window = window_before + window_after
        min_points = int(total_window * 16.7)
        if min_points > n_points:
            n_points = min_points
        zscore_baseline = config.zscore_baseline
        baseline_start = float(config.baseline_start)
        baseline_end = float(config.baseline_end)

        # Extract window around each event
        for i, event_time in enumerate(event_times):
            # Find time indices for window
            mask = (time_array >= event_time - window_before) & (time_array <= event_time + window_after)

            if not np.any(mask):
                continue

            t_window = time_array[mask] - event_time  # Relative to event
            vals_window = signal[mask].copy()  # Copy to avoid modifying original

            # Skip if too few points
            if len(t_window) < 10:
                continue

            # Apply z-score normalization to baseline period if enabled
            if zscore_baseline:
                # Get the reference time for baseline (onset time for withdrawal CTAs)
                baseline_ref_time = baseline_ref_times[i] if i < len(baseline_ref_times) else event_time

                # Calculate baseline from reference time (e.g., onset), not alignment time
                # Baseline period is baseline_ref_time + baseline_start to baseline_ref_time + baseline_end
                baseline_abs_start = baseline_ref_time + baseline_start
                baseline_abs_end = baseline_ref_time + baseline_end

                # Find baseline values in signal (using absolute times)
                baseline_mask_abs = (time_array >= baseline_abs_start) & (time_array <= baseline_abs_end)
                if np.sum(baseline_mask_abs) > 1:
                    baseline_vals = signal[baseline_mask_abs]
                    baseline_mean = np.nanmean(baseline_vals)
                    baseline_std = np.nanstd(baseline_vals)
                    if baseline_std > 1e-10:  # Avoid division by zero
                        vals_window = (vals_window - baseline_mean) / baseline_std
                    else:
                        # If std is ~0, just subtract mean (flat baseline)
                        vals_window = vals_window - baseline_mean

            # Compute offset to paired event (onset→withdrawal or withdrawal→onset)
            paired_offset = None
            if paired_event_times is not None and i < len(paired_event_times) and paired_event_times[i] is not None:
                paired_offset = float(paired_event_times[i] - event_time)

            trace = CTATrace(
                event_id=event_ids[i] if i < len(event_ids) else f"event_{i}",
                sweep_idx=sweep_idx,
                event_time=float(event_time),
                time=t_window,
                values=vals_window,
                paired_event_offset=paired_offset,
            )
            traces.append(trace)

        # Create common time base
        t_common = np.linspace(-window_before, window_after, n_points)

        # Interpolate all traces to common time base
        interp_traces = []
        for trace in traces:
            if len(trace.time) > 1:
                interp_vals = np.interp(
                    t_common, trace.time, trace.values,
                    left=np.nan, right=np.nan
                )
                interp_traces.append(interp_vals)

        # Compute mean and SEM
        mean = None
        sem = None
        if interp_traces:
            trace_matrix = np.array(interp_traces)
            mean = np.nanmean(trace_matrix, axis=0)
            sem = stats.sem(trace_matrix, axis=0, nan_policy='omit')

        return CTAResult(
            metric_key=metric_key,
            metric_label=metric_label,
            alignment=alignment,
            category=category,
            label=label,
            config=config,
            traces=traces,
            time_common=t_common,
            mean=mean,
            sem=sem,
            n_events=len(traces),
        )

    def calculate_for_markers(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
        config: Optional[CTAConfig] = None,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> CTACollection:
        """
        Calculate CTAs for all markers and metrics.

        Args:
            markers: List of EventMarkers to align to
            signals: Dictionary of signals keyed by metric name
            time_array: Time values for signals
            metric_labels: Optional human-readable labels for metrics
            config: CTA configuration
            progress_callback: Optional callback for progress updates (0-100)

        Returns:
            CTACollection containing all results
        """
        if config is None:
            config = CTAConfig()

        if metric_labels is None:
            metric_labels = {}

        collection = CTACollection(config=config)

        # Group markers by category:label
        marker_groups: Dict[Tuple[str, str], List[EventMarker]] = {}
        for marker in markers:
            key = (marker.category, marker.label)
            if key not in marker_groups:
                marker_groups[key] = []
            marker_groups[key].append(marker)

        # Total work items for progress
        total_items = len(marker_groups) * len(signals) * 2  # *2 for onset + withdrawal
        current_item = 0

        for (category, label), group_markers in marker_groups.items():
            # Extract event times
            onset_times = [m.start_time for m in group_markers]
            onset_ids = [m.id for m in group_markers]

            # For onset CTA: paired event = withdrawal time (for paired marker lines)
            onset_paired_times = []
            for m in group_markers:
                if m.is_paired and m.end_time is not None:
                    onset_paired_times.append(m.end_time)
                else:
                    onset_paired_times.append(None)

            # Withdrawal times (only for paired markers)
            # Keep track of corresponding onset times for z-score baseline
            withdrawal_times = []
            withdrawal_ids = []
            withdrawal_onset_times = []  # Corresponding onset times for baseline
            if config.include_withdrawal:
                for m in group_markers:
                    if m.is_paired and m.end_time is not None:
                        withdrawal_times.append(m.end_time)
                        withdrawal_ids.append(m.id)
                        withdrawal_onset_times.append(m.start_time)  # Use onset for baseline

            # Calculate CTA for each metric
            for metric_key, signal in signals.items():
                metric_label = metric_labels.get(metric_key, metric_key)

                # Onset CTA
                if onset_times:
                    result = self.calculate_cta(
                        time_array=time_array,
                        signal=signal,
                        event_times=onset_times,
                        event_ids=onset_ids,
                        paired_event_times=onset_paired_times,
                        config=config,
                        metric_key=metric_key,
                        metric_label=metric_label,
                        category=category,
                        label=label,
                        alignment='onset',
                    )
                    collection.add_result(result)

                current_item += 1
                if progress_callback:
                    progress_callback(int(100 * current_item / total_items))

                # Withdrawal CTA
                # Use onset times as baseline reference so z-score is calculated
                # relative to the pre-stimulus baseline, not pre-withdrawal
                if withdrawal_times and config.include_withdrawal:
                    result = self.calculate_cta(
                        time_array=time_array,
                        signal=signal,
                        event_times=withdrawal_times,
                        event_ids=withdrawal_ids,
                        baseline_ref_times=withdrawal_onset_times,  # Z-score to onset baseline
                        paired_event_times=withdrawal_onset_times,  # Show onset lines on withdrawal CTA
                        config=config,
                        metric_key=metric_key,
                        metric_label=metric_label,
                        category=category,
                        label=label,
                        alignment='withdrawal',
                    )
                    collection.add_result(result)

                current_item += 1
                if progress_callback:
                    progress_callback(int(100 * current_item / total_items))

        return collection

    def compute_histogram_stats(
        self,
        signal: np.ndarray,
        time_array: np.ndarray,
        markers: List[EventMarker],
    ) -> Dict[str, any]:
        """
        Compute statistics comparing values during vs outside events.

        Args:
            signal: Signal values
            time_array: Time values
            markers: Paired markers defining event regions

        Returns:
            Dictionary with 'during' and 'outside' values, and statistics
        """
        during_vals = []
        outside_vals = []

        # Create mask for "during event" times
        during_mask = np.zeros(len(time_array), dtype=bool)
        for marker in markers:
            if marker.is_paired and marker.end_time is not None:
                mask = (time_array >= marker.start_time) & (time_array <= marker.end_time)
                during_mask |= mask

        during_vals = signal[during_mask]
        outside_vals = signal[~during_mask]

        # Compute statistics
        result = {
            'during_vals': during_vals,
            'outside_vals': outside_vals,
            'during_mean': np.nanmean(during_vals) if len(during_vals) > 0 else np.nan,
            'during_std': np.nanstd(during_vals) if len(during_vals) > 0 else np.nan,
            'outside_mean': np.nanmean(outside_vals) if len(outside_vals) > 0 else np.nan,
            'outside_std': np.nanstd(outside_vals) if len(outside_vals) > 0 else np.nan,
            'n_during': len(during_vals),
            'n_outside': len(outside_vals),
        }

        # T-test if we have enough data
        if len(during_vals) > 1 and len(outside_vals) > 1:
            try:
                t_stat, p_val = stats.ttest_ind(
                    during_vals, outside_vals, nan_policy='omit'
                )
                result['t_stat'] = t_stat
                result['p_val'] = p_val
            except Exception:
                result['t_stat'] = np.nan
                result['p_val'] = np.nan
        else:
            result['t_stat'] = np.nan
            result['p_val'] = np.nan

        return result

    def to_npz_dict(self, collection: CTACollection) -> Dict[str, any]:
        """
        Serialize a CTACollection to NPZ format.

        Args:
            collection: The collection to serialize

        Returns:
            Dictionary suitable for np.savez_compressed
        """
        return collection.to_npz_dict()

    def from_npz_dict(self, data: Dict[str, any]) -> Optional[CTACollection]:
        """
        Deserialize a CTACollection from NPZ data.

        Args:
            data: Dictionary from np.load

        Returns:
            CTACollection or None if no CTA data present
        """
        return CTACollection.from_npz_dict(data)

    def export_to_csv(
        self,
        collection: CTACollection,
        filepath: str,
        include_individual_traces: bool = True,
    ) -> None:
        """
        Export CTA data to CSV format.

        Args:
            collection: The CTA collection to export
            filepath: Path to save CSV
            include_individual_traces: Whether to include individual event traces
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            if include_individual_traces:
                writer.writerow([
                    'marker_type', 'alignment', 'metric', 'event_idx',
                    'event_id', 'sweep_idx', 'time', 'value'
                ])
            else:
                writer.writerow([
                    'marker_type', 'alignment', 'metric', 'time', 'mean', 'sem'
                ])

            for key, result in collection.results.items():
                marker_type = f"{result.category}:{result.label}"

                if include_individual_traces:
                    # Individual traces
                    for i, trace in enumerate(result.traces):
                        for t, v in zip(trace.time, trace.values):
                            writer.writerow([
                                marker_type, result.alignment, result.metric_key,
                                i, trace.event_id, trace.sweep_idx, t, v
                            ])
                else:
                    # Summary only
                    if result.time_common is not None and result.mean is not None:
                        for t, m, s in zip(result.time_common, result.mean,
                                          result.sem if result.sem is not None else [np.nan] * len(result.mean)):
                            writer.writerow([
                                marker_type, result.alignment, result.metric_key,
                                t, m, s
                            ])

    def export_to_csv_wide(
        self,
        collection: CTACollection,
        filepath: str,
    ) -> None:
        """
        Export all CTA data into a single wide-format CSV.

        Columns: time, metric1_event1, metric1_event2, ..., metric1_mean, metric1_sem,
                        metric2_event1, metric2_event2, ..., metric2_mean, metric2_sem, ...

        All metrics share the same time column (common time base).

        Args:
            collection: The CTA collection to export
            filepath: Path to save CSV
        """
        import csv
        from scipy import interpolate

        # Collect all results that have data, grouped by alignment
        results_with_data = []
        for key, result in collection.results.items():
            if result.time_common is not None and len(result.traces) > 0:
                results_with_data.append(result)

        if not results_with_data:
            return

        # Use the first result's time base as the common time
        t_common = results_with_data[0].time_common
        n_points = len(t_common)

        # Build columns for each result
        all_columns = []  # list of (header, values_array)
        all_columns.append(('time', t_common))

        blank = np.full(n_points, np.nan)
        for ri, result in enumerate(results_with_data):
            safe_metric = result.metric_key.replace('/', '_')
            prefix = f"{safe_metric}_{result.alignment}"

            # Interpolate each trace onto the common time base
            for i, trace in enumerate(result.traces):
                if len(trace.time) < 2:
                    continue
                f_interp = interpolate.interp1d(
                    trace.time, trace.values,
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                all_columns.append((f"{prefix}_event{i + 1}", f_interp(t_common)))

            # Add mean and sem
            if result.mean is not None:
                all_columns.append((f"{prefix}_mean", result.mean))
            if result.sem is not None:
                all_columns.append((f"{prefix}_sem", result.sem))

            # Add 2 blank separator columns between metrics
            if ri < len(results_with_data) - 1:
                all_columns.append(('', blank))
                all_columns.append(('', blank))

        # Write single CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([col[0] for col in all_columns])
            for j in range(n_points):
                writer.writerow([col[1][j] for col in all_columns])

    def export_conditions_to_csv_wide(
        self,
        condition_collections: Dict[str, CTACollection],
        filepath: str,
    ) -> None:
        """
        Export per-condition CTA data into a single wide-format CSV.

        Each column header is prefixed with the condition name so all
        conditions appear side-by-side.

        Args:
            condition_collections: Dict of condition_name -> CTACollection
            filepath: Path to save CSV
        """
        import csv
        from scipy import interpolate

        # Collect all results with data across all conditions
        all_columns = []
        t_common = None

        result_count = 0
        for cond_name, collection in sorted(condition_collections.items()):
            safe_cond = cond_name.replace('/', '_').replace(' ', '_')

            for key, result in collection.results.items():
                if result.time_common is None or len(result.traces) == 0:
                    continue

                # Use the first result's time base as the common time
                if t_common is None:
                    t_common = result.time_common
                    all_columns.append(('time', t_common))

                n_points = len(t_common)
                safe_metric = result.metric_key.replace('/', '_')
                prefix = f"{safe_cond}_{safe_metric}_{result.alignment}"

                # Add 2 blank separator columns between metric groups
                if result_count > 0:
                    blank = np.full(n_points, np.nan)
                    all_columns.append(('', blank))
                    all_columns.append(('', blank))

                for i, trace in enumerate(result.traces):
                    if len(trace.time) < 2:
                        continue
                    f_interp = interpolate.interp1d(
                        trace.time, trace.values,
                        kind='linear', bounds_error=False, fill_value=np.nan
                    )
                    all_columns.append((f"{prefix}_event{i + 1}", f_interp(t_common)))

                if result.mean is not None:
                    all_columns.append((f"{prefix}_mean", result.mean))
                if result.sem is not None:
                    all_columns.append((f"{prefix}_sem", result.sem))

                result_count += 1

        if not all_columns or t_common is None:
            return

        n_points = len(t_common)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([col[0] for col in all_columns])
            for j in range(n_points):
                writer.writerow([col[1][j] for col in all_columns])
