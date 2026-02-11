"""
Event detection framework for PhysioMetrics.

This module provides a pluggable detection system for automatically
identifying events in signal data.
"""

from .base import EventDetector, DetectorRegistry, DetectionResult
from .threshold import ThresholdCrossingDetector
from .hargreaves import HargreavesDetector

__all__ = [
    'EventDetector',
    'DetectorRegistry',
    'DetectionResult',
    'ThresholdCrossingDetector',
    'HargreavesDetector',
]
