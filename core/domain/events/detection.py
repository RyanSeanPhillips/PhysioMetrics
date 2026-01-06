"""
Event detection algorithms.

This module contains pure Python/NumPy detection algorithms that
identify events in signal data. No UI dependencies.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .models import EventMarker, MarkerType


@dataclass
class DetectionResult:
    """Result of an event detection run."""

    markers: List[EventMarker]
    params_used: Dict[str, Any]
    stats: Dict[str, Any]  # e.g., {'num_found': 10, 'mean_duration': 0.5}


@dataclass
class ThresholdParams:
    """Parameters for threshold-based detection."""

    threshold: float = 0.5
    direction: str = "rising"  # 'rising', 'falling', 'both'
    min_duration_ms: float = 50.0  # Minimum event duration
    min_gap_ms: float = 100.0  # Minimum gap between events
    marker_type: MarkerType = MarkerType.PAIRED
    debounce_ms: float = 10.0  # Debounce for noisy signals


@dataclass
class TTLParams:
    """Parameters for TTL/digital signal detection."""

    high_threshold: float = 2.5  # Voltage above which signal is HIGH
    low_threshold: float = 0.5  # Voltage below which signal is LOW
    min_pulse_ms: float = 1.0  # Minimum pulse width to detect
    marker_type: MarkerType = MarkerType.SINGLE  # SINGLE for edges, PAIRED for pulses
    detect: str = "rising"  # 'rising', 'falling', 'both', 'pulse'


@dataclass
class PeakParams:
    """Parameters for peak-based detection."""

    prominence: float = 0.1
    min_distance_ms: float = 100.0
    width_ms: Optional[Tuple[float, float]] = None  # (min, max) width
    marker_type: MarkerType = MarkerType.SINGLE


def detect_threshold_crossings(
    signal: np.ndarray,
    sample_rate: float,
    params: ThresholdParams,
    sweep_idx: int = 0,
    category: str = "custom",
    label: str = "threshold_event",
    source_channel: Optional[str] = None,
    group_id: Optional[str] = None,
) -> DetectionResult:
    """
    Detect events based on threshold crossings.

    For PAIRED markers: detects regions where signal is above/below threshold.
    For SINGLE markers: detects threshold crossing points.

    Args:
        signal: 1D signal array
        sample_rate: Samples per second
        params: Detection parameters
        sweep_idx: Sweep index for created markers
        category: Category for created markers
        label: Label for created markers
        source_channel: Channel name for metadata
        group_id: Group ID for linking markers

    Returns:
        DetectionResult with detected markers
    """
    if len(signal) == 0:
        return DetectionResult(markers=[], params_used=_params_to_dict(params), stats={'num_found': 0})

    # Convert ms to samples
    min_duration_samples = int(params.min_duration_ms * sample_rate / 1000)
    min_gap_samples = int(params.min_gap_ms * sample_rate / 1000)
    debounce_samples = int(params.debounce_ms * sample_rate / 1000)

    # Find threshold crossings
    if params.direction == "rising":
        above = signal >= params.threshold
    elif params.direction == "falling":
        above = signal <= params.threshold
    else:  # both
        above = np.abs(signal - params.threshold) < (np.max(signal) - np.min(signal)) * 0.1

    # Apply debounce (morphological closing to remove noise)
    if debounce_samples > 1:
        kernel = np.ones(debounce_samples)
        # Close gaps smaller than debounce
        above_float = above.astype(float)
        above = np.convolve(above_float, kernel, mode='same') > 0.5

    # Find rising and falling edges
    edges = np.diff(above.astype(int))
    rising_edges = np.where(edges == 1)[0] + 1
    falling_edges = np.where(edges == -1)[0] + 1

    markers = []

    if params.marker_type == MarkerType.PAIRED:
        # Pair rising and falling edges to create regions
        events = _pair_edges(rising_edges, falling_edges, len(signal))

        # Filter by duration and gap
        filtered_events = []
        last_end = -min_gap_samples

        for start, end in events:
            duration = end - start
            gap = start - last_end

            if duration >= min_duration_samples and gap >= min_gap_samples:
                filtered_events.append((start, end))
                last_end = end

        # Create markers
        for start, end in filtered_events:
            start_time = start / sample_rate
            end_time = end / sample_rate
            markers.append(EventMarker(
                sweep_idx=sweep_idx,
                marker_type=MarkerType.PAIRED,
                start_time=start_time,
                end_time=end_time,
                category=category,
                label=label,
                source_channel=source_channel,
                detection_method="threshold",
                detection_params=_params_to_dict(params),
                group_id=group_id,
            ))

    else:  # SINGLE markers at crossings
        if params.direction == "rising":
            crossings = rising_edges
        elif params.direction == "falling":
            crossings = falling_edges
        else:
            crossings = np.sort(np.concatenate([rising_edges, falling_edges]))

        # Filter by minimum gap
        filtered_crossings = []
        last_crossing = -min_gap_samples

        for crossing in crossings:
            if crossing - last_crossing >= min_gap_samples:
                filtered_crossings.append(crossing)
                last_crossing = crossing

        # Create markers
        for crossing in filtered_crossings:
            time = crossing / sample_rate
            markers.append(EventMarker(
                sweep_idx=sweep_idx,
                marker_type=MarkerType.SINGLE,
                start_time=time,
                category=category,
                label=label,
                source_channel=source_channel,
                detection_method="threshold",
                detection_params=_params_to_dict(params),
                group_id=group_id,
            ))

    stats = {
        'num_found': len(markers),
        'mean_duration': np.mean([m.duration for m in markers]) if markers and params.marker_type == MarkerType.PAIRED else 0,
    }

    return DetectionResult(
        markers=markers,
        params_used=_params_to_dict(params),
        stats=stats,
    )


def detect_ttl_events(
    signal: np.ndarray,
    sample_rate: float,
    params: TTLParams,
    sweep_idx: int = 0,
    category: str = "stimulus",
    label: str = "ttl_trigger",
    source_channel: Optional[str] = None,
    group_id: Optional[str] = None,
) -> DetectionResult:
    """
    Detect TTL/digital events (pulses or edges).

    Args:
        signal: 1D signal array (typically 0-5V TTL)
        sample_rate: Samples per second
        params: Detection parameters
        sweep_idx: Sweep index for created markers
        category: Category for created markers
        label: Label for created markers
        source_channel: Channel name for metadata
        group_id: Group ID for linking markers

    Returns:
        DetectionResult with detected markers
    """
    if len(signal) == 0:
        return DetectionResult(markers=[], params_used=_params_to_dict(params), stats={'num_found': 0})

    min_pulse_samples = int(params.min_pulse_ms * sample_rate / 1000)

    # Digitize signal with hysteresis
    high = signal >= params.high_threshold
    low = signal <= params.low_threshold

    # Apply hysteresis to create clean digital signal
    digital = np.zeros(len(signal), dtype=bool)
    state = False
    for i in range(len(signal)):
        if high[i]:
            state = True
        elif low[i]:
            state = False
        digital[i] = state

    # Find edges
    edges = np.diff(digital.astype(int))
    rising_edges = np.where(edges == 1)[0] + 1
    falling_edges = np.where(edges == -1)[0] + 1

    markers = []

    if params.detect == "pulse" or params.marker_type == MarkerType.PAIRED:
        # Detect complete pulses
        events = _pair_edges(rising_edges, falling_edges, len(signal))

        for start, end in events:
            if end - start >= min_pulse_samples:
                start_time = start / sample_rate
                end_time = end / sample_rate
                markers.append(EventMarker(
                    sweep_idx=sweep_idx,
                    marker_type=MarkerType.PAIRED,
                    start_time=start_time,
                    end_time=end_time,
                    category=category,
                    label=label,
                    source_channel=source_channel,
                    detection_method="ttl",
                    detection_params=_params_to_dict(params),
                    group_id=group_id,
                ))

    else:  # Single edge markers
        if params.detect == "rising":
            crossings = rising_edges
        elif params.detect == "falling":
            crossings = falling_edges
        else:  # both
            crossings = np.sort(np.concatenate([rising_edges, falling_edges]))

        for crossing in crossings:
            time = crossing / sample_rate
            markers.append(EventMarker(
                sweep_idx=sweep_idx,
                marker_type=MarkerType.SINGLE,
                start_time=time,
                category=category,
                label=label,
                source_channel=source_channel,
                detection_method="ttl",
                detection_params=_params_to_dict(params),
                group_id=group_id,
            ))

    stats = {
        'num_found': len(markers),
        'num_rising': len(rising_edges),
        'num_falling': len(falling_edges),
    }

    return DetectionResult(
        markers=markers,
        params_used=_params_to_dict(params),
        stats=stats,
    )


def detect_peaks(
    signal: np.ndarray,
    sample_rate: float,
    params: PeakParams,
    sweep_idx: int = 0,
    category: str = "custom",
    label: str = "peak",
    source_channel: Optional[str] = None,
    group_id: Optional[str] = None,
) -> DetectionResult:
    """
    Detect peaks in signal using scipy.signal.find_peaks.

    Args:
        signal: 1D signal array
        sample_rate: Samples per second
        params: Detection parameters
        sweep_idx: Sweep index for created markers
        category: Category for created markers
        label: Label for created markers
        source_channel: Channel name for metadata
        group_id: Group ID for linking markers

    Returns:
        DetectionResult with detected markers
    """
    from scipy.signal import find_peaks

    if len(signal) == 0:
        return DetectionResult(markers=[], params_used=_params_to_dict(params), stats={'num_found': 0})

    # Convert ms to samples
    min_distance = int(params.min_distance_ms * sample_rate / 1000)

    # Build kwargs for find_peaks
    kwargs = {
        'prominence': params.prominence,
        'distance': max(1, min_distance),
    }

    if params.width_ms:
        min_width = int(params.width_ms[0] * sample_rate / 1000)
        max_width = int(params.width_ms[1] * sample_rate / 1000)
        kwargs['width'] = (min_width, max_width)

    # Find peaks
    peaks, properties = find_peaks(signal, **kwargs)

    markers = []

    if params.marker_type == MarkerType.PAIRED and 'widths' in properties:
        # Create paired markers spanning peak width
        left_ips = properties.get('left_ips', peaks)
        right_ips = properties.get('right_ips', peaks)

        for i, peak in enumerate(peaks):
            start_sample = int(left_ips[i]) if i < len(left_ips) else peak
            end_sample = int(right_ips[i]) if i < len(right_ips) else peak

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            markers.append(EventMarker(
                sweep_idx=sweep_idx,
                marker_type=MarkerType.PAIRED,
                start_time=start_time,
                end_time=end_time,
                category=category,
                label=label,
                source_channel=source_channel,
                detection_method="peak",
                detection_params=_params_to_dict(params),
                group_id=group_id,
            ))
    else:
        # Single markers at peak locations
        for peak in peaks:
            time = peak / sample_rate
            markers.append(EventMarker(
                sweep_idx=sweep_idx,
                marker_type=MarkerType.SINGLE,
                start_time=time,
                category=category,
                label=label,
                source_channel=source_channel,
                detection_method="peak",
                detection_params=_params_to_dict(params),
                group_id=group_id,
            ))

    stats = {
        'num_found': len(markers),
        'mean_prominence': float(np.mean(properties.get('prominences', [0]))) if 'prominences' in properties else 0,
    }

    return DetectionResult(
        markers=markers,
        params_used=_params_to_dict(params),
        stats=stats,
    )


def find_nearest_peak(
    signal: np.ndarray,
    sample_rate: float,
    time: float,
    search_radius_ms: float = 50.0,
    direction: str = "both",  # 'up', 'down', 'both'
) -> Optional[float]:
    """
    Find the nearest peak/valley to a time point (for snapping).

    Args:
        signal: 1D signal array
        sample_rate: Samples per second
        time: Time point to search around
        search_radius_ms: Search radius in milliseconds
        direction: 'up' for peaks, 'down' for valleys, 'both' for either

    Returns:
        Time of nearest peak/valley, or None if not found
    """
    from scipy.signal import find_peaks

    center_sample = int(time * sample_rate)
    radius_samples = int(search_radius_ms * sample_rate / 1000)

    start = max(0, center_sample - radius_samples)
    end = min(len(signal), center_sample + radius_samples)

    if end <= start:
        return None

    segment = signal[start:end]
    candidates = []

    if direction in ('up', 'both'):
        peaks, _ = find_peaks(segment, prominence=0.01)
        candidates.extend(peaks + start)

    if direction in ('down', 'both'):
        valleys, _ = find_peaks(-segment, prominence=0.01)
        candidates.extend(valleys + start)

    if not candidates:
        return None

    # Find nearest to center
    nearest = min(candidates, key=lambda x: abs(x - center_sample))
    return nearest / sample_rate


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _pair_edges(
    rising_edges: np.ndarray,
    falling_edges: np.ndarray,
    signal_length: int
) -> List[Tuple[int, int]]:
    """
    Pair rising and falling edges to create event regions.

    Handles cases where signal starts high or ends high.

    Returns:
        List of (start, end) sample pairs
    """
    events = []

    # Handle case where signal starts high
    if len(falling_edges) > 0 and (len(rising_edges) == 0 or falling_edges[0] < rising_edges[0]):
        events.append((0, falling_edges[0]))
        falling_edges = falling_edges[1:]

    # Pair remaining edges
    for i, rising in enumerate(rising_edges):
        # Find next falling edge after this rising edge
        falling_after = falling_edges[falling_edges > rising]
        if len(falling_after) > 0:
            events.append((rising, falling_after[0]))
            # Remove used falling edge
            falling_edges = falling_edges[falling_edges != falling_after[0]]
        else:
            # Signal ends high
            events.append((rising, signal_length - 1))
            break

    return events


def _params_to_dict(params) -> Dict[str, Any]:
    """Convert params dataclass to dict, handling enums."""
    result = {}
    for field_name in params.__dataclass_fields__:
        value = getattr(params, field_name)
        if hasattr(value, 'value'):  # Enum
            result[field_name] = value.value
        else:
            result[field_name] = value
    return result
