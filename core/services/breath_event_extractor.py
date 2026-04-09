"""
Breath Event Extractor Service.

Converts already-detected respiratory peak/breath arrays into ephemeral
EventMarker objects suitable for CTA triggering. No Qt dependencies.
"""

from typing import List, Dict, Optional
import numpy as np
import uuid

from core.domain.events import EventMarker, MarkerType


class BreathEventExtractor:
    """
    Extracts breath events from peak/breath arrays and returns EventMarkers.

    These markers are ephemeral — they are passed directly to the CTA
    calculation and are NOT stored in the marker store.
    """

    # Trigger point options — maps to breath timing arrays
    TRIGGER_POINTS = {
        'inspiratory_onset': 'Inspiratory onset',
        'inspiratory_peak': 'Inspiratory peak',
        'inspiratory_offset': 'Inspiratory offset (exp onset)',
        'expiratory_trough': 'Expiratory trough',
        'expiratory_offset': 'Expiratory offset',
        'sniffing_bout_onset': 'Sniffing bout onset',
        'sniffing_bout_offset': 'Sniffing bout offset',
        'eupnea_bout_onset': 'Eupnea bout onset',
        'eupnea_bout_offset': 'Eupnea bout offset',
    }

    # Triggers that imply a breath type and disable the type filter
    BOUT_TRIGGERS = {
        'sniffing_bout_onset', 'sniffing_bout_offset',
        'eupnea_bout_onset', 'eupnea_bout_offset',
    }

    # Breath type filter options
    BREATH_TYPES = {
        'all': 'All breaths',
        'eupnea': 'Eupnea only',
        'sniffing': 'Sniffing only',
        'sigh': 'Sighs only',
    }

    def extract(
        self,
        trigger_point: str,
        breath_type_filter: str,
        max_events: int,
        min_bout_length: int,
        auto_assign_conditions: bool,
        all_peaks: Dict,
        all_breaths: Dict,
        sigh_indices: np.ndarray,
        time_array: np.ndarray,
        sr_hz: float,
        sweep_idx: int = 0,
        existing_markers: Optional[List[EventMarker]] = None,
        time_window_start: float = 0.0,
        time_window_end: float = 0.0,
    ) -> List[EventMarker]:
        """
        Extract breath events as ephemeral EventMarkers.

        Args:
            trigger_point: One of TRIGGER_POINTS keys
            breath_type_filter: One of BREATH_TYPES keys
            max_events: Maximum events to return (0 = unlimited)
            min_bout_length: Minimum consecutive breaths of same type
            auto_assign_conditions: Assign conditions from overlapping markers
            all_peaks: Dict with 'indices', 'labels', 'breath_type_class' arrays
            all_breaths: Dict with 'onsets', 'offsets', 'expmins', 'expoffs' arrays
            sigh_indices: Array of peak indices marked as sighs
            time_array: Full time array for index→time conversion
            sr_hz: Sample rate in Hz
            sweep_idx: Sweep index
            existing_markers: Markers for condition auto-assignment

        Returns:
            List of ephemeral EventMarker objects
        """
        if not all_peaks or not all_breaths:
            return []

        peak_indices = all_peaks.get('indices', np.array([]))
        labels = all_peaks.get('labels', np.array([]))
        breath_type_class = all_peaks.get('breath_type_class', np.array([]))

        if len(peak_indices) == 0:
            return []

        # Step 1: Get breath mask (labels == 1, not noise)
        breath_mask = labels == 1 if len(labels) == len(peak_indices) else np.ones(len(peak_indices), dtype=bool)

        # Step 2: Apply breath type filter (unless trigger is a bout on/offset)
        if trigger_point not in self.BOUT_TRIGGERS:
            if breath_type_filter == 'eupnea' and len(breath_type_class) == len(peak_indices):
                breath_mask = breath_mask & (breath_type_class == 0)
            elif breath_type_filter == 'sniffing' and len(breath_type_class) == len(peak_indices):
                breath_mask = breath_mask & (breath_type_class == 1)
            elif breath_type_filter == 'sigh':
                # Filter to only sigh peaks
                if sigh_indices is not None and len(sigh_indices) > 0:
                    sigh_set = set(sigh_indices.tolist())
                    sigh_mask = np.array([peak_indices[i] in sigh_set for i in range(len(peak_indices))])
                    breath_mask = breath_mask & sigh_mask
                else:
                    return []  # No sighs detected

        # Step 2b: Apply time window filter
        if time_window_start > 0 or time_window_end > 0:
            peak_times = np.array([
                time_array[int(idx)] if 0 <= int(idx) < len(time_array) else 0
                for idx in peak_indices
            ])
            if time_window_start > 0:
                breath_mask = breath_mask & (peak_times >= time_window_start)
            if time_window_end > 0:
                breath_mask = breath_mask & (peak_times <= time_window_end)

        # Step 3: Apply min bout length filter
        if min_bout_length > 1 and breath_type_filter in ('eupnea', 'sniffing') and trigger_point not in self.BOUT_TRIGGERS:
            breath_mask = self._filter_by_bout_length(breath_mask, breath_type_class,
                                                       peak_indices, min_bout_length,
                                                       breath_type_filter)

        # Step 4: Get trigger sample indices
        trigger_indices = self._get_trigger_indices(
            trigger_point, breath_mask, peak_indices, all_breaths,
            breath_type_class, min_bout_length
        )

        if len(trigger_indices) == 0:
            return []

        # Step 5: Convert to times
        event_times = []
        for idx in trigger_indices:
            if 0 <= idx < len(time_array):
                event_times.append(time_array[int(idx)])

        if not event_times:
            return []

        # Step 6: Apply max events limit
        if max_events > 0 and len(event_times) > max_events:
            event_times = event_times[:max_events]

        # Step 7: Create EventMarker objects
        label_suffix = ''
        if trigger_point in self.BOUT_TRIGGERS:
            label_suffix = ''  # Bout triggers already encode the type
        elif breath_type_filter != 'all':
            label_suffix = f'_{breath_type_filter}'

        category = 'breath_event'
        label = f'{trigger_point}{label_suffix}'

        markers = []
        for t in event_times:
            condition = None
            if auto_assign_conditions and existing_markers:
                condition = self._find_condition_at_time(t, existing_markers)

            marker = EventMarker(
                id=str(uuid.uuid4()),
                start_time=t,
                end_time=None,
                sweep_idx=sweep_idx,
                category=category,
                label=label,
                marker_type=MarkerType.SINGLE,
                source_channel='breath',
                detection_method='breath_extractor',
                condition=condition,
                visible=True,
            )
            markers.append(marker)

        return markers

    def _get_trigger_indices(
        self,
        trigger_point: str,
        breath_mask: np.ndarray,
        peak_indices: np.ndarray,
        all_breaths: Dict,
        breath_type_class: np.ndarray,
        min_bout_length: int,
    ) -> np.ndarray:
        """Get sample indices for the selected trigger point, filtered by breath mask."""
        valid_peak_positions = np.where(breath_mask)[0]

        if trigger_point == 'inspiratory_peak':
            return peak_indices[valid_peak_positions]

        elif trigger_point == 'inspiratory_onset':
            return self._get_breath_array_at_positions(all_breaths, 'onsets', peak_indices, valid_peak_positions)

        elif trigger_point == 'inspiratory_offset':
            # Inspiratory offset = expiratory onset = expmins position
            # Actually, for pleth the peak IS the insp→exp transition
            # offsets array = peak positions (same as indices)
            return self._get_breath_array_at_positions(all_breaths, 'offsets', peak_indices, valid_peak_positions)

        elif trigger_point == 'expiratory_trough':
            return self._get_breath_array_at_positions(all_breaths, 'expmins', peak_indices, valid_peak_positions)

        elif trigger_point == 'expiratory_offset':
            return self._get_breath_array_at_positions(all_breaths, 'expoffs', peak_indices, valid_peak_positions)

        elif trigger_point in ('sniffing_bout_onset', 'sniffing_bout_offset',
                               'eupnea_bout_onset', 'eupnea_bout_offset'):
            return self._get_bout_transitions(
                trigger_point, breath_mask, peak_indices, all_breaths,
                breath_type_class, min_bout_length
            )

        return np.array([])

    def _get_breath_array_at_positions(
        self, all_breaths: Dict, key: str, peak_indices: np.ndarray,
        valid_positions: np.ndarray,
    ) -> np.ndarray:
        """Get values from a breath array at valid peak positions."""
        arr = all_breaths.get(key, np.array([]))
        if len(arr) == 0:
            return np.array([])
        # Clip positions to array length
        valid = valid_positions[valid_positions < len(arr)]
        if len(valid) == 0:
            return np.array([])
        return arr[valid]

    def _get_bout_transitions(
        self,
        trigger_point: str,
        breath_mask: np.ndarray,
        peak_indices: np.ndarray,
        all_breaths: Dict,
        breath_type_class: np.ndarray,
        min_bout_length: int,
    ) -> np.ndarray:
        """Get bout onset/offset indices for sniffing or eupnea transitions."""
        if len(breath_type_class) != len(peak_indices):
            return np.array([])

        # Determine target type and whether we want onset or offset
        if 'sniffing' in trigger_point:
            target_class = 1  # sniffing
        else:
            target_class = 0  # eupnea

        is_onset = 'onset' in trigger_point

        # Only consider valid breaths (labels == 1)
        labels = breath_mask.copy()
        is_target = (breath_type_class == target_class) & labels

        # Find bout boundaries
        onsets = all_breaths.get('onsets', np.array([]))
        result = []

        bout_start = None
        bout_positions = []

        for i in range(len(is_target)):
            if is_target[i]:
                if bout_start is None:
                    bout_start = i
                    bout_positions = [i]
                else:
                    bout_positions.append(i)
            else:
                if bout_start is not None and len(bout_positions) >= max(1, min_bout_length):
                    if is_onset:
                        pos = bout_positions[0]
                    else:
                        pos = bout_positions[-1]
                    # Use inspiratory onset time for the trigger
                    if pos < len(onsets):
                        result.append(onsets[pos])
                    elif pos < len(peak_indices):
                        result.append(peak_indices[pos])
                bout_start = None
                bout_positions = []

        # Handle final bout
        if bout_start is not None and len(bout_positions) >= max(1, min_bout_length):
            if is_onset:
                pos = bout_positions[0]
            else:
                pos = bout_positions[-1]
            if pos < len(onsets):
                result.append(onsets[pos])
            elif pos < len(peak_indices):
                result.append(peak_indices[pos])

        return np.array(result, dtype=int) if result else np.array([], dtype=int)

    def _filter_by_bout_length(
        self,
        breath_mask: np.ndarray,
        breath_type_class: np.ndarray,
        peak_indices: np.ndarray,
        min_bout_length: int,
        breath_type: str,
    ) -> np.ndarray:
        """Filter to only include breaths from bouts of at least min_bout_length."""
        target_class = 0 if breath_type == 'eupnea' else 1
        filtered = np.zeros_like(breath_mask)

        is_target = (breath_type_class == target_class) & breath_mask
        bout_start = None
        bout_positions = []

        for i in range(len(is_target)):
            if is_target[i]:
                if bout_start is None:
                    bout_start = i
                    bout_positions = [i]
                else:
                    bout_positions.append(i)
            else:
                if bout_start is not None and len(bout_positions) >= min_bout_length:
                    for pos in bout_positions:
                        filtered[pos] = True
                bout_start = None
                bout_positions = []

        if bout_start is not None and len(bout_positions) >= min_bout_length:
            for pos in bout_positions:
                filtered[pos] = True

        return filtered

    def _find_condition_at_time(
        self,
        time: float,
        markers: List[EventMarker],
    ) -> Optional[str]:
        """Find condition from paired markers that overlap the given time."""
        for m in markers:
            if not m.is_paired or m.end_time is None or not m.condition:
                continue
            if m.start_time <= time <= m.end_time:
                return m.condition
        return None
