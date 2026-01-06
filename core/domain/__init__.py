# Domain layer - business logic with no UI dependencies
from .events import (
    EventMarker,
    MarkerType,
    EventCategory,
    BUILTIN_CATEGORIES,
    CategoryRegistry,
    get_category_registry,
    MarkerStore,
    DetectionResult,
    ThresholdParams,
    TTLParams,
    PeakParams,
    detect_threshold_crossings,
    detect_ttl_events,
    detect_peaks,
    find_nearest_peak,
)

__all__ = [
    # Events
    'EventMarker',
    'MarkerType',
    'EventCategory',
    'BUILTIN_CATEGORIES',
    'CategoryRegistry',
    'get_category_registry',
    'MarkerStore',
    'DetectionResult',
    'ThresholdParams',
    'TTLParams',
    'PeakParams',
    'detect_threshold_crossings',
    'detect_ttl_events',
    'detect_peaks',
    'find_nearest_peak',
]
