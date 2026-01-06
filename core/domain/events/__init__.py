# Event marker domain models
from .models import EventMarker, MarkerType
from .categories import EventCategory, BUILTIN_CATEGORIES, CategoryRegistry, get_category_registry
from .store import MarkerStore
from .detection import (
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
    # Models
    'EventMarker',
    'MarkerType',
    # Categories
    'EventCategory',
    'BUILTIN_CATEGORIES',
    'CategoryRegistry',
    'get_category_registry',
    # Store
    'MarkerStore',
    # Detection
    'DetectionResult',
    'ThresholdParams',
    'TTLParams',
    'PeakParams',
    'detect_threshold_crossings',
    'detect_ttl_events',
    'detect_peaks',
    'find_nearest_peak',
]
