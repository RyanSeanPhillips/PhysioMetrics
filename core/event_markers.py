"""
Event marker data structures and management for PhysioMetrics.

This module provides the EventMarker dataclass and helper functions
for managing event markers across sweeps.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import json


# ============================================================================
# Event Marker Data Structure
# ============================================================================

@dataclass
class EventMarker:
    """
    Represents a single event marker in the recording.

    Event markers can be either:
    - Single: A point in time (end_time is None)
    - Paired: A duration with start and end times
    """

    id: int                             # Unique ID within sweep
    start_time: float                   # Absolute time in seconds
    end_time: Optional[float]           # None for single markers, time for paired
    event_type: str                     # Type key (e.g., 'lick_bout', 'hargreaves')
    source_channel: str                 # Channel used for detection
    detection_method: str               # 'manual', 'threshold', 'hargreaves_thermal', etc.
    color_override: Optional[str] = None  # Override type color, or None
    notes: Optional[str] = None         # User annotations

    @property
    def is_paired(self) -> bool:
        """Check if this is a paired (duration) marker."""
        return self.end_time is not None

    @property
    def duration(self) -> float:
        """Get duration in seconds (0 for single markers)."""
        if self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def contains_time(self, t: float) -> bool:
        """Check if a time point is within this marker's range."""
        if self.end_time is None:
            # Single marker - check if within small tolerance
            return abs(t - self.start_time) < 0.1
        return self.start_time <= t <= self.end_time

    def overlaps(self, other: 'EventMarker') -> bool:
        """Check if this marker overlaps with another."""
        if self.end_time is None or other.end_time is None:
            return False  # Single markers don't overlap
        return not (self.end_time < other.start_time or self.start_time > other.end_time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'event_type': self.event_type,
            'source_channel': self.source_channel,
            'detection_method': self.detection_method,
            'color_override': self.color_override,
            'notes': self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventMarker':
        """Create from dictionary."""
        return cls(
            id=data.get('id', 0),
            start_time=data.get('start_time', 0.0),
            end_time=data.get('end_time'),
            event_type=data.get('event_type', 'custom'),
            source_channel=data.get('source_channel', ''),
            detection_method=data.get('detection_method', 'manual'),
            color_override=data.get('color_override'),
            notes=data.get('notes'),
        )

    @classmethod
    def from_legacy_bout(cls, bout_dict: Dict, source_channel: str = '') -> 'EventMarker':
        """
        Create from legacy bout_annotations format.

        Legacy format: {'start_time': float, 'end_time': float, 'id': int}
        """
        return cls(
            id=bout_dict.get('id', 0),
            start_time=bout_dict.get('start_time', 0.0),
            end_time=bout_dict.get('end_time'),
            event_type='lick_bout',  # Default legacy type
            source_channel=source_channel,
            detection_method='legacy_import',
            color_override=None,
            notes=None,
        )


# ============================================================================
# Event Marker Collection Management
# ============================================================================

class EventMarkerManager:
    """
    Manages event markers across all sweeps.

    Provides methods for adding, removing, querying, and persisting markers.
    """

    def __init__(self):
        # Storage: sweep_idx -> list of EventMarker
        self._markers: Dict[int, List[EventMarker]] = {}
        self._next_id: Dict[int, int] = {}  # Next ID per sweep

    def clear(self):
        """Clear all markers."""
        self._markers.clear()
        self._next_id.clear()

    def clear_sweep(self, sweep_idx: int):
        """Clear all markers for a specific sweep."""
        if sweep_idx in self._markers:
            self._markers[sweep_idx] = []
            self._next_id[sweep_idx] = 1

    def clear_type(self, event_type: str, sweep_idx: Optional[int] = None):
        """
        Clear all markers of a specific type.

        Args:
            event_type: Type to clear
            sweep_idx: If specified, only clear from this sweep
        """
        if sweep_idx is not None:
            if sweep_idx in self._markers:
                self._markers[sweep_idx] = [
                    m for m in self._markers[sweep_idx]
                    if m.event_type != event_type
                ]
        else:
            for sweep in self._markers:
                self._markers[sweep] = [
                    m for m in self._markers[sweep]
                    if m.event_type != event_type
                ]

    def _get_next_id(self, sweep_idx: int) -> int:
        """Get next available ID for a sweep."""
        if sweep_idx not in self._next_id:
            self._next_id[sweep_idx] = 1
        next_id = self._next_id[sweep_idx]
        self._next_id[sweep_idx] = next_id + 1
        return next_id

    def add_marker(
        self,
        sweep_idx: int,
        start_time: float,
        end_time: Optional[float],
        event_type: str,
        source_channel: str,
        detection_method: str = 'manual',
        color_override: Optional[str] = None,
        notes: Optional[str] = None
    ) -> EventMarker:
        """
        Add a new event marker.

        Args:
            sweep_idx: Sweep index
            start_time: Start time in seconds
            end_time: End time (None for single marker)
            event_type: Event type key
            source_channel: Source channel name
            detection_method: How this marker was created
            color_override: Optional color override
            notes: Optional notes

        Returns:
            The created EventMarker
        """
        if sweep_idx not in self._markers:
            self._markers[sweep_idx] = []

        marker = EventMarker(
            id=self._get_next_id(sweep_idx),
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            source_channel=source_channel,
            detection_method=detection_method,
            color_override=color_override,
            notes=notes,
        )

        self._markers[sweep_idx].append(marker)
        self._sort_markers(sweep_idx)

        return marker

    def add_paired_marker(
        self,
        sweep_idx: int,
        start_time: float,
        end_time: float,
        event_type: str,
        source_channel: str,
        detection_method: str = 'manual'
    ) -> EventMarker:
        """Convenience method for adding paired markers."""
        return self.add_marker(
            sweep_idx=sweep_idx,
            start_time=start_time,
            end_time=end_time,
            event_type=event_type,
            source_channel=source_channel,
            detection_method=detection_method
        )

    def add_single_marker(
        self,
        sweep_idx: int,
        time: float,
        event_type: str,
        source_channel: str,
        detection_method: str = 'manual'
    ) -> EventMarker:
        """Convenience method for adding single markers."""
        return self.add_marker(
            sweep_idx=sweep_idx,
            start_time=time,
            end_time=None,
            event_type=event_type,
            source_channel=source_channel,
            detection_method=detection_method
        )

    def remove_marker(self, sweep_idx: int, marker_id: int) -> bool:
        """
        Remove a marker by ID.

        Args:
            sweep_idx: Sweep index
            marker_id: Marker ID to remove

        Returns:
            True if removed, False if not found
        """
        if sweep_idx not in self._markers:
            return False

        original_len = len(self._markers[sweep_idx])
        self._markers[sweep_idx] = [
            m for m in self._markers[sweep_idx] if m.id != marker_id
        ]
        return len(self._markers[sweep_idx]) < original_len

    def get_markers(self, sweep_idx: int) -> List[EventMarker]:
        """Get all markers for a sweep, sorted by start time."""
        return self._markers.get(sweep_idx, [])

    def get_markers_by_type(self, sweep_idx: int, event_type: str) -> List[EventMarker]:
        """Get markers of a specific type for a sweep."""
        return [m for m in self.get_markers(sweep_idx) if m.event_type == event_type]

    def get_marker_at_time(self, sweep_idx: int, time: float) -> Optional[EventMarker]:
        """Find a marker that contains the given time."""
        for marker in self.get_markers(sweep_idx):
            if marker.contains_time(time):
                return marker
        return None

    def get_marker_by_id(self, sweep_idx: int, marker_id: int) -> Optional[EventMarker]:
        """Get a marker by its ID."""
        for marker in self.get_markers(sweep_idx):
            if marker.id == marker_id:
                return marker
        return None

    def get_nearest_marker_edge(
        self,
        sweep_idx: int,
        time: float,
        tolerance: float = 0.3
    ) -> Optional[Tuple[EventMarker, str]]:
        """
        Find nearest marker edge within tolerance.

        Args:
            sweep_idx: Sweep index
            time: Time to search near
            tolerance: Maximum distance in seconds

        Returns:
            Tuple of (marker, edge_type) where edge_type is 'start' or 'end',
            or None if no edge is within tolerance
        """
        best_marker = None
        best_edge = None
        best_dist = tolerance

        for marker in self.get_markers(sweep_idx):
            # Check start edge
            dist_start = abs(time - marker.start_time)
            if dist_start < best_dist:
                best_dist = dist_start
                best_marker = marker
                best_edge = 'start'

            # Check end edge (for paired markers)
            if marker.end_time is not None:
                dist_end = abs(time - marker.end_time)
                if dist_end < best_dist:
                    best_dist = dist_end
                    best_marker = marker
                    best_edge = 'end'

        if best_marker is not None:
            return (best_marker, best_edge)
        return None

    def update_marker_time(
        self,
        sweep_idx: int,
        marker_id: int,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> bool:
        """
        Update marker times.

        Args:
            sweep_idx: Sweep index
            marker_id: Marker ID
            start_time: New start time (None to keep current)
            end_time: New end time (None to keep current)

        Returns:
            True if updated, False if not found
        """
        marker = self.get_marker_by_id(sweep_idx, marker_id)
        if marker is None:
            return False

        if start_time is not None:
            marker.start_time = start_time
        if end_time is not None:
            marker.end_time = end_time

        # Ensure start <= end for paired markers
        if marker.end_time is not None and marker.start_time > marker.end_time:
            marker.start_time, marker.end_time = marker.end_time, marker.start_time

        self._sort_markers(sweep_idx)
        return True

    def update_marker_type(self, sweep_idx: int, marker_id: int, event_type: str) -> bool:
        """Change the type of a marker."""
        marker = self.get_marker_by_id(sweep_idx, marker_id)
        if marker is None:
            return False
        marker.event_type = event_type
        return True

    def update_marker_notes(self, sweep_idx: int, marker_id: int, notes: str) -> bool:
        """Update marker notes."""
        marker = self.get_marker_by_id(sweep_idx, marker_id)
        if marker is None:
            return False
        marker.notes = notes
        return True

    def _sort_markers(self, sweep_idx: int):
        """Sort markers by start time."""
        if sweep_idx in self._markers:
            self._markers[sweep_idx].sort(key=lambda m: m.start_time)

    def merge_overlapping(self, sweep_idx: int, event_type: Optional[str] = None):
        """
        Merge overlapping markers of the same type.

        Args:
            sweep_idx: Sweep index
            event_type: If specified, only merge this type
        """
        if sweep_idx not in self._markers:
            return

        markers = self._markers[sweep_idx]
        if not markers:
            return

        # Group by type
        by_type: Dict[str, List[EventMarker]] = {}
        for m in markers:
            if event_type is not None and m.event_type != event_type:
                continue
            if m.event_type not in by_type:
                by_type[m.event_type] = []
            by_type[m.event_type].append(m)

        merged_ids = set()

        for type_name, type_markers in by_type.items():
            # Sort by start time
            type_markers.sort(key=lambda m: m.start_time)

            i = 0
            while i < len(type_markers) - 1:
                current = type_markers[i]
                next_m = type_markers[i + 1]

                # Only merge paired markers
                if current.end_time is None or next_m.end_time is None:
                    i += 1
                    continue

                # Check for overlap
                if current.end_time >= next_m.start_time:
                    # Merge: extend current to include next
                    current.end_time = max(current.end_time, next_m.end_time)
                    merged_ids.add(next_m.id)
                    type_markers.pop(i + 1)
                else:
                    i += 1

        # Remove merged markers
        self._markers[sweep_idx] = [
            m for m in markers if m.id not in merged_ids
        ]

    def count_markers(self, sweep_idx: Optional[int] = None, event_type: Optional[str] = None) -> int:
        """
        Count markers.

        Args:
            sweep_idx: If specified, count only this sweep
            event_type: If specified, count only this type

        Returns:
            Number of markers matching criteria
        """
        count = 0
        sweeps = [sweep_idx] if sweep_idx is not None else self._markers.keys()

        for s in sweeps:
            markers = self._markers.get(s, [])
            if event_type is not None:
                markers = [m for m in markers if m.event_type == event_type]
            count += len(markers)

        return count

    def count_by_type(self, sweep_idx: Optional[int] = None) -> Dict[str, int]:
        """Get marker counts grouped by type."""
        counts: Dict[str, int] = {}
        sweeps = [sweep_idx] if sweep_idx is not None else self._markers.keys()

        for s in sweeps:
            for marker in self._markers.get(s, []):
                counts[marker.event_type] = counts.get(marker.event_type, 0) + 1

        return counts

    # ========================================================================
    # Serialization
    # ========================================================================

    def to_dict(self) -> Dict[int, List[Dict]]:
        """Convert all markers to dictionary for serialization."""
        return {
            sweep_idx: [m.to_dict() for m in markers]
            for sweep_idx, markers in self._markers.items()
        }

    def from_dict(self, data: Dict[int, List[Dict]]):
        """Load markers from dictionary."""
        self.clear()
        for sweep_idx, marker_dicts in data.items():
            sweep_idx = int(sweep_idx)  # Handle string keys from JSON
            self._markers[sweep_idx] = [
                EventMarker.from_dict(d) for d in marker_dicts
            ]
            # Update next_id based on loaded markers
            if self._markers[sweep_idx]:
                max_id = max(m.id for m in self._markers[sweep_idx])
                self._next_id[sweep_idx] = max_id + 1

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def from_json(self, json_str: str):
        """Load from JSON string."""
        data = json.loads(json_str)
        self.from_dict(data)

    def import_legacy_bouts(self, bout_annotations: Dict[int, List[Dict]], source_channel: str = ''):
        """
        Import from legacy bout_annotations format.

        Args:
            bout_annotations: Dict of sweep_idx -> list of bout dicts
            source_channel: Channel name to assign to imported markers
        """
        for sweep_idx, bouts in bout_annotations.items():
            sweep_idx = int(sweep_idx)
            if sweep_idx not in self._markers:
                self._markers[sweep_idx] = []

            for bout in bouts:
                marker = EventMarker.from_legacy_bout(bout, source_channel)
                marker.id = self._get_next_id(sweep_idx)
                self._markers[sweep_idx].append(marker)

            self._sort_markers(sweep_idx)

    def export_to_csv_rows(self, sweep_idx: Optional[int] = None) -> List[Dict]:
        """
        Export markers as CSV-compatible rows.

        Returns:
            List of dicts suitable for CSV export
        """
        rows = []
        sweeps = [sweep_idx] if sweep_idx is not None else sorted(self._markers.keys())

        for s in sweeps:
            for marker in self._markers.get(s, []):
                rows.append({
                    'sweep': s,
                    'event_id': marker.id,
                    'event_type': marker.event_type,
                    'start_time_s': marker.start_time,
                    'end_time_s': marker.end_time if marker.end_time else '',
                    'duration_s': marker.duration if marker.is_paired else '',
                    'source_channel': marker.source_channel,
                    'detection_method': marker.detection_method,
                    'notes': marker.notes or '',
                })

        return rows

    def to_numpy_dtype(self):
        """Get numpy dtype for structured array export."""
        return np.dtype([
            ('id', 'i4'),
            ('start_time', 'f8'),
            ('end_time', 'f8'),
            ('event_type', 'U32'),
            ('source_channel', 'U64'),
            ('detection_method', 'U32'),
            ('notes', 'U256'),
        ])

    def to_numpy_array(self, sweep_idx: int) -> np.ndarray:
        """Convert markers for a sweep to numpy structured array."""
        markers = self.get_markers(sweep_idx)
        dtype = self.to_numpy_dtype()

        arr = np.zeros(len(markers), dtype=dtype)
        for i, m in enumerate(markers):
            arr[i] = (
                m.id,
                m.start_time,
                m.end_time if m.end_time is not None else np.nan,
                m.event_type,
                m.source_channel,
                m.detection_method,
                m.notes or '',
            )
        return arr


# ============================================================================
# Module-level convenience functions
# ============================================================================

def create_marker_manager() -> EventMarkerManager:
    """Create a new EventMarkerManager instance."""
    return EventMarkerManager()
