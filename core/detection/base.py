"""
Base classes for event detection.

Provides abstract base detector class and common utilities for
implementing specific detection algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Type
from enum import Enum
import numpy as np


class ParamType(Enum):
    """Parameter types for UI generation."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    CHOICE = "choice"  # Dropdown selection


@dataclass
class ParamSpec:
    """Specification for a detector parameter."""
    name: str
    label: str
    param_type: ParamType
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Tuple[str, Any]]] = None  # [(display, value), ...]
    tooltip: Optional[str] = None
    unit: Optional[str] = None  # e.g., "s", "V", "Hz"


@dataclass
class DetectionResult:
    """Result of event detection."""
    events: List[Tuple[float, float]]  # List of (start_time, end_time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional info

    @property
    def count(self) -> int:
        return len(self.events)

    def filter_by_duration(self, min_duration: float) -> 'DetectionResult':
        """Filter events by minimum duration."""
        filtered = [
            (start, end) for start, end in self.events
            if (end - start) >= min_duration
        ]
        return DetectionResult(
            events=filtered,
            metadata={**self.metadata, 'filtered_by_duration': min_duration}
        )

    def filter_by_gap(self, min_gap: float) -> 'DetectionResult':
        """Merge events that are closer than min_gap."""
        if not self.events:
            return DetectionResult(events=[], metadata=self.metadata)

        # Sort by start time
        sorted_events = sorted(self.events, key=lambda x: x[0])
        merged = [sorted_events[0]]

        for start, end in sorted_events[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_gap:
                # Merge: extend previous event
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))

        return DetectionResult(
            events=merged,
            metadata={**self.metadata, 'merged_by_gap': min_gap}
        )


class EventDetector(ABC):
    """
    Abstract base class for event detectors.

    Subclasses implement specific detection algorithms (threshold crossing,
    Hargreaves thermal, etc.) while this base class provides:
    - Common parameter handling
    - Post-detection filtering (min duration, min gap)
    - UI parameter specification
    """

    # Override in subclasses
    name: str = "Base Detector"
    description: str = "Abstract base detector"

    def __init__(self):
        self._params: Dict[str, Any] = {}
        # Initialize with defaults
        for spec in self.get_param_specs():
            self._params[spec.name] = spec.default

    @abstractmethod
    def get_param_specs(self) -> List[ParamSpec]:
        """
        Return parameter specifications for this detector.

        Used by the UI to generate appropriate input widgets.
        """
        pass

    @abstractmethod
    def _detect_raw(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float
    ) -> List[Tuple[float, float]]:
        """
        Perform raw detection without filtering.

        Args:
            signal: 1D signal array
            time: 1D time array (same length as signal)
            sample_rate: Sample rate in Hz

        Returns:
            List of (start_time, end_time) tuples
        """
        pass

    def detect(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        sample_rate: float,
        min_duration: float = 0.0,
        min_gap: float = 0.0,
    ) -> DetectionResult:
        """
        Detect events with optional filtering.

        Args:
            signal: 1D signal array
            time: 1D time array
            sample_rate: Sample rate in Hz
            min_duration: Minimum event duration in seconds
            min_gap: Minimum gap between events (closer events are merged)

        Returns:
            DetectionResult with detected events
        """
        # Perform raw detection
        raw_events = self._detect_raw(signal, time, sample_rate)

        result = DetectionResult(
            events=raw_events,
            metadata={
                'detector': self.name,
                'params': self._params.copy(),
                'raw_count': len(raw_events),
            }
        )

        # Apply filters
        if min_duration > 0:
            result = result.filter_by_duration(min_duration)

        if min_gap > 0:
            result = result.filter_by_gap(min_gap)

        return result

    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value."""
        self._params[name] = value

    def get_param(self, name: str) -> Any:
        """Get a parameter value."""
        return self._params.get(name)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Set multiple parameters at once."""
        self._params.update(params)

    def get_params(self) -> Dict[str, Any]:
        """Get all current parameters."""
        return self._params.copy()

    def reset_params(self) -> None:
        """Reset all parameters to defaults."""
        for spec in self.get_param_specs():
            self._params[spec.name] = spec.default


class DetectorRegistry:
    """
    Registry for available event detectors.

    Allows registration and lookup of detector classes by name.
    """

    _detectors: Dict[str, Type[EventDetector]] = {}

    @classmethod
    def register(cls, detector_class: Type[EventDetector]) -> Type[EventDetector]:
        """
        Register a detector class.

        Can be used as a decorator:
            @DetectorRegistry.register
            class MyDetector(EventDetector):
                ...
        """
        cls._detectors[detector_class.name] = detector_class
        return detector_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[EventDetector]]:
        """Get a detector class by name."""
        return cls._detectors.get(name)

    @classmethod
    def create(cls, name: str) -> Optional[EventDetector]:
        """Create an instance of a detector by name."""
        detector_class = cls.get(name)
        if detector_class:
            return detector_class()
        return None

    @classmethod
    def all(cls) -> List[Type[EventDetector]]:
        """Get all registered detector classes."""
        return list(cls._detectors.values())

    @classmethod
    def names(cls) -> List[str]:
        """Get names of all registered detectors."""
        return list(cls._detectors.keys())
