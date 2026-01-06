"""
Event marker domain models.

This module contains the core data structures for event markers,
designed to be independent of any UI framework (Qt-free).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import uuid
import json


class MarkerType(Enum):
    """Type of event marker."""
    SINGLE = "single"    # Point in time (vertical line)
    PAIRED = "paired"    # Region with start and end


@dataclass
class EventMarker:
    """
    A single event marker representing either a point in time or a region.

    Attributes:
        id: Unique identifier for this marker
        sweep_idx: Which sweep this marker belongs to (-1 for cross-sweep)
        marker_type: SINGLE (point) or PAIRED (region)
        start_time: Start time in seconds (always present)
        end_time: End time in seconds (only for PAIRED markers)
        category: Category key ('respiratory', 'behavior', 'stimulus', etc.)
        label: Specific label within category ('lick_bout', 'inspiratory_onset', etc.)
        source_channel: Channel name used for detection (if auto-detected)
        detection_method: How marker was created ('manual', 'threshold', 'ttl', 'peak')
        detection_params: Parameters used for auto-detection
        color_override: Custom color (None = use category color)
        notes: User notes for this marker
        group_id: ID linking markers from same auto-detection run
    """

    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sweep_idx: int = 0

    # Timing
    marker_type: MarkerType = MarkerType.SINGLE
    start_time: float = 0.0
    end_time: Optional[float] = None

    # Classification
    category: str = "custom"
    label: str = "marker"

    # Detection metadata
    source_channel: Optional[str] = None
    detection_method: str = "manual"
    detection_params: Dict[str, Any] = field(default_factory=dict)

    # Display
    color_override: Optional[str] = None
    notes: Optional[str] = None

    # Grouping
    group_id: Optional[str] = None

    @property
    def duration(self) -> float:
        """Duration of marker (0 for single markers)."""
        if self.marker_type == MarkerType.PAIRED and self.end_time is not None:
            return self.end_time - self.start_time
        return 0.0

    @property
    def is_paired(self) -> bool:
        """Check if this is a paired (region) marker."""
        return self.marker_type == MarkerType.PAIRED

    @property
    def is_single(self) -> bool:
        """Check if this is a single (point) marker."""
        return self.marker_type == MarkerType.SINGLE

    @property
    def center_time(self) -> float:
        """Get center time of marker (same as start_time for single markers)."""
        if self.is_paired and self.end_time is not None:
            return (self.start_time + self.end_time) / 2
        return self.start_time

    def contains_time(self, t: float, tolerance: float = 0.0) -> bool:
        """
        Check if a time point falls within this marker.

        For paired markers: checks if t is between start and end (with tolerance)
        For single markers: checks if t is near start_time (within tolerance)

        Args:
            t: Time point to check (seconds)
            tolerance: Tolerance for near-match (seconds)

        Returns:
            True if time is within/near this marker
        """
        if self.is_paired and self.end_time is not None:
            return self.start_time - tolerance <= t <= self.end_time + tolerance
        return abs(t - self.start_time) <= tolerance

    def overlaps(self, other: 'EventMarker') -> bool:
        """Check if this marker overlaps with another marker."""
        if self.sweep_idx != other.sweep_idx and self.sweep_idx != -1 and other.sweep_idx != -1:
            return False

        self_end = self.end_time if self.end_time is not None else self.start_time
        other_end = other.end_time if other.end_time is not None else other.start_time

        return not (self_end < other.start_time or self.start_time > other_end)

    def move(self, delta: float) -> None:
        """Move marker by delta seconds."""
        self.start_time += delta
        if self.end_time is not None:
            self.end_time += delta

    def set_times(self, start: float, end: Optional[float] = None) -> None:
        """Set marker times, ensuring start <= end for paired markers."""
        if end is not None and end < start:
            start, end = end, start
        self.start_time = start
        self.end_time = end

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'sweep_idx': self.sweep_idx,
            'marker_type': self.marker_type.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'category': self.category,
            'label': self.label,
            'source_channel': self.source_channel,
            'detection_method': self.detection_method,
            'detection_params': self.detection_params,
            'color_override': self.color_override,
            'notes': self.notes,
            'group_id': self.group_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMarker':
        """Create EventMarker from dictionary."""
        marker_type = data.get('marker_type', 'single')
        if isinstance(marker_type, str):
            marker_type = MarkerType(marker_type)

        return cls(
            id=data.get('id', str(uuid.uuid4())[:8]),
            sweep_idx=data.get('sweep_idx', 0),
            marker_type=marker_type,
            start_time=data.get('start_time', 0.0),
            end_time=data.get('end_time'),
            category=data.get('category', 'custom'),
            label=data.get('label', 'marker'),
            source_channel=data.get('source_channel'),
            detection_method=data.get('detection_method', 'manual'),
            detection_params=data.get('detection_params', {}),
            color_override=data.get('color_override'),
            notes=data.get('notes'),
            group_id=data.get('group_id'),
        )

    def copy(self) -> 'EventMarker':
        """Create a copy of this marker with a new ID."""
        new_marker = EventMarker.from_dict(self.to_dict())
        new_marker.id = str(uuid.uuid4())[:8]
        return new_marker

    def __repr__(self) -> str:
        if self.is_paired:
            return f"EventMarker({self.id}, {self.category}/{self.label}, {self.start_time:.3f}-{self.end_time:.3f}s)"
        return f"EventMarker({self.id}, {self.category}/{self.label}, {self.start_time:.3f}s)"
