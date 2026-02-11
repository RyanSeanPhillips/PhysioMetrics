"""
Event marker storage.

This module provides in-memory storage for event markers with
efficient querying and modification operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any, Iterator
from collections import defaultdict
import json

from .models import EventMarker, MarkerType


class MarkerStore:
    """
    In-memory storage for event markers.

    Provides efficient indexing by sweep, category, and group for fast queries.
    Supports undo/redo through command pattern integration.

    Attributes:
        markers: Dict mapping marker ID to EventMarker
        _by_sweep: Index of marker IDs by sweep
        _by_category: Index of marker IDs by category
        _by_group: Index of marker IDs by group
        _modified: Whether store has unsaved changes
    """

    def __init__(self):
        self._markers: Dict[str, EventMarker] = {}
        self._by_sweep: Dict[int, Set[str]] = defaultdict(set)
        self._by_category: Dict[str, Set[str]] = defaultdict(set)
        self._by_group: Dict[str, Set[str]] = defaultdict(set)
        self._modified: bool = False
        self._change_callbacks: List[Callable[[], None]] = []

    @property
    def modified(self) -> bool:
        """Whether the store has unsaved changes."""
        return self._modified

    @modified.setter
    def modified(self, value: bool) -> None:
        self._modified = value

    def on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback to be called when markers change."""
        self._change_callbacks.append(callback)

    def remove_change_callback(self, callback: Callable[[], None]) -> None:
        """Remove a change callback."""
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)

    def _notify_change(self) -> None:
        """Notify all listeners of a change."""
        self._modified = True
        for callback in self._change_callbacks:
            try:
                callback()
            except Exception:
                pass  # Don't let callback errors break the store

    # -------------------------------------------------------------------------
    # Basic CRUD operations
    # -------------------------------------------------------------------------

    def add(self, marker: EventMarker) -> str:
        """
        Add a marker to the store.

        Args:
            marker: Marker to add

        Returns:
            Marker ID
        """
        self._markers[marker.id] = marker
        self._index_marker(marker)
        self._notify_change()
        return marker.id

    def add_many(self, markers: List[EventMarker]) -> List[str]:
        """
        Add multiple markers at once.

        Args:
            markers: List of markers to add

        Returns:
            List of marker IDs
        """
        ids = []
        for marker in markers:
            self._markers[marker.id] = marker
            self._index_marker(marker)
            ids.append(marker.id)
        if markers:
            self._notify_change()
        return ids

    def get(self, marker_id: str) -> Optional[EventMarker]:
        """Get a marker by ID."""
        return self._markers.get(marker_id)

    def update(self, marker: EventMarker, old_category: Optional[str] = None, old_sweep_idx: Optional[int] = None) -> bool:
        """
        Update an existing marker.

        Args:
            marker: Marker with updated values (must have existing ID)
            old_category: Previous category for proper index cleanup (optional)
            old_sweep_idx: Previous sweep index for proper index cleanup (optional)

        Returns:
            True if updated, False if marker not found
        """
        if marker.id not in self._markers:
            return False

        # If old values provided, manually clean up indices
        # This handles the case where the marker object was modified in-place before update() was called
        if old_category is not None and old_category != marker.category:
            self._by_category[old_category].discard(marker.id)
        if old_sweep_idx is not None and old_sweep_idx != marker.sweep_idx:
            self._by_sweep[old_sweep_idx].discard(marker.id)

        # Update the marker in storage
        self._markers[marker.id] = marker

        # Re-index with new values (this is idempotent - adding to a set twice is fine)
        self._index_marker(marker)
        self._notify_change()
        return True

    def remove(self, marker_id: str) -> Optional[EventMarker]:
        """
        Remove a marker by ID.

        Args:
            marker_id: ID of marker to remove

        Returns:
            Removed marker, or None if not found
        """
        marker = self._markers.pop(marker_id, None)
        if marker:
            self._unindex_marker(marker)
            self._notify_change()
        return marker

    def remove_many(self, marker_ids: List[str]) -> List[EventMarker]:
        """
        Remove multiple markers at once.

        Args:
            marker_ids: List of marker IDs to remove

        Returns:
            List of removed markers
        """
        removed = []
        for marker_id in marker_ids:
            marker = self._markers.pop(marker_id, None)
            if marker:
                self._unindex_marker(marker)
                removed.append(marker)
        if removed:
            self._notify_change()
        return removed

    def clear(self) -> int:
        """
        Remove all markers.

        Returns:
            Number of markers removed
        """
        count = len(self._markers)
        self._markers.clear()
        self._by_sweep.clear()
        self._by_category.clear()
        self._by_group.clear()
        if count > 0:
            self._notify_change()
        return count

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    def _index_marker(self, marker: EventMarker) -> None:
        """Add marker to indices."""
        self._by_sweep[marker.sweep_idx].add(marker.id)
        self._by_category[marker.category].add(marker.id)
        if marker.group_id:
            self._by_group[marker.group_id].add(marker.id)

    def _unindex_marker(self, marker: EventMarker) -> None:
        """Remove marker from indices."""
        self._by_sweep[marker.sweep_idx].discard(marker.id)
        self._by_category[marker.category].discard(marker.id)
        if marker.group_id:
            self._by_group[marker.group_id].discard(marker.id)

    def reindex(self) -> None:
        """Rebuild all indices (useful after bulk modifications)."""
        self._by_sweep.clear()
        self._by_category.clear()
        self._by_group.clear()
        for marker in self._markers.values():
            self._index_marker(marker)

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._markers)

    def __iter__(self) -> Iterator[EventMarker]:
        return iter(self._markers.values())

    def __contains__(self, marker_id: str) -> bool:
        return marker_id in self._markers

    def all(self) -> List[EventMarker]:
        """Get all markers."""
        return list(self._markers.values())

    def get_by_sweep(self, sweep_idx: int) -> List[EventMarker]:
        """Get all markers for a specific sweep."""
        return [
            self._markers[mid]
            for mid in self._by_sweep.get(sweep_idx, set())
            if mid in self._markers
        ]

    def get_by_category(self, category: str) -> List[EventMarker]:
        """Get all markers of a specific category."""
        return [
            self._markers[mid]
            for mid in self._by_category.get(category, set())
            if mid in self._markers
        ]

    def get_by_label(self, category: str, label: str) -> List[EventMarker]:
        """Get all markers with a specific category and label."""
        return [
            m for m in self.get_by_category(category)
            if m.label == label
        ]

    def get_by_group(self, group_id: str) -> List[EventMarker]:
        """Get all markers in a specific group."""
        return [
            self._markers[mid]
            for mid in self._by_group.get(group_id, set())
            if mid in self._markers
        ]

    def get_by_type(self, marker_type: MarkerType) -> List[EventMarker]:
        """Get all markers of a specific type (single or paired)."""
        return [m for m in self._markers.values() if m.marker_type == marker_type]

    def get_in_time_range(
        self,
        start_time: float,
        end_time: float,
        sweep_idx: Optional[int] = None
    ) -> List[EventMarker]:
        """
        Get markers that overlap with a time range.

        Args:
            start_time: Start of range (seconds)
            end_time: End of range (seconds)
            sweep_idx: Optional sweep filter

        Returns:
            List of overlapping markers
        """
        if sweep_idx is not None:
            markers = self.get_by_sweep(sweep_idx)
        else:
            markers = self.all()

        result = []
        for m in markers:
            m_end = m.end_time if m.end_time is not None else m.start_time
            if not (m_end < start_time or m.start_time > end_time):
                result.append(m)
        return result

    def get_at_time(
        self,
        time: float,
        tolerance: float = 0.01,
        sweep_idx: Optional[int] = None
    ) -> List[EventMarker]:
        """
        Get markers at or near a specific time.

        Args:
            time: Time point (seconds)
            tolerance: Tolerance for single markers (seconds)
            sweep_idx: Optional sweep filter

        Returns:
            List of markers at this time
        """
        if sweep_idx is not None:
            markers = self.get_by_sweep(sweep_idx)
        else:
            markers = self.all()

        return [m for m in markers if m.contains_time(time, tolerance)]

    def get_nearest(
        self,
        time: float,
        sweep_idx: Optional[int] = None,
        max_distance: Optional[float] = None
    ) -> Optional[EventMarker]:
        """
        Get the marker nearest to a time point.

        Args:
            time: Time point (seconds)
            sweep_idx: Optional sweep filter
            max_distance: Maximum distance to consider (seconds)

        Returns:
            Nearest marker, or None if none found
        """
        if sweep_idx is not None:
            markers = self.get_by_sweep(sweep_idx)
        else:
            markers = self.all()

        if not markers:
            return None

        def distance(m: EventMarker) -> float:
            if m.is_paired and m.end_time is not None:
                if m.start_time <= time <= m.end_time:
                    return 0.0
                return min(abs(time - m.start_time), abs(time - m.end_time))
            return abs(time - m.start_time)

        nearest = min(markers, key=distance)
        d = distance(nearest)

        if max_distance is not None and d > max_distance:
            return None

        return nearest

    def count_by_sweep(self) -> Dict[int, int]:
        """Get count of markers per sweep."""
        return {
            sweep: len(ids)
            for sweep, ids in self._by_sweep.items()
        }

    def count_by_category(self) -> Dict[str, int]:
        """Get count of markers per category."""
        return {
            cat: len(ids)
            for cat, ids in self._by_category.items()
        }

    def get_sweeps_with_markers(self) -> List[int]:
        """Get list of sweep indices that have markers."""
        return sorted([
            sweep for sweep, ids in self._by_sweep.items()
            if ids
        ])

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert store to dictionary for serialization."""
        return {
            'markers': [m.to_dict() for m in self._markers.values()]
        }

    def to_json(self) -> str:
        """Convert store to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarkerStore':
        """Create store from dictionary."""
        store = cls()
        for marker_data in data.get('markers', []):
            marker = EventMarker.from_dict(marker_data)
            store._markers[marker.id] = marker
            store._index_marker(marker)
        store._modified = False
        return store

    @classmethod
    def from_json(cls, json_str: str) -> 'MarkerStore':
        """Create store from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def merge(self, other: 'MarkerStore', resolve_conflicts: str = 'keep_both') -> int:
        """
        Merge another store into this one.

        Args:
            other: Store to merge from
            resolve_conflicts: How to handle ID conflicts:
                - 'keep_both': Assign new ID to incoming marker
                - 'keep_existing': Skip incoming marker
                - 'keep_incoming': Replace existing marker

        Returns:
            Number of markers added
        """
        added = 0
        for marker in other.all():
            if marker.id in self._markers:
                if resolve_conflicts == 'keep_existing':
                    continue
                elif resolve_conflicts == 'keep_incoming':
                    self.update(marker)
                    added += 1
                else:  # keep_both
                    new_marker = marker.copy()  # Gets new ID
                    self.add(new_marker)
                    added += 1
            else:
                self.add(marker)
                added += 1
        return added

    # -------------------------------------------------------------------------
    # Bulk operations
    # -------------------------------------------------------------------------

    def delete_by_category(self, category: str) -> int:
        """
        Delete all markers of a category.

        Returns:
            Number of markers deleted
        """
        ids = list(self._by_category.get(category, set()))
        removed = self.remove_many(ids)
        return len(removed)

    def delete_by_label(self, category: str, label: str) -> int:
        """
        Delete all markers with a specific category and label.

        Returns:
            Number of markers deleted
        """
        to_delete = [m.id for m in self.get_by_label(category, label)]
        removed = self.remove_many(to_delete)
        return len(removed)

    def delete_by_sweep(self, sweep_idx: int) -> int:
        """
        Delete all markers in a sweep.

        Returns:
            Number of markers deleted
        """
        ids = list(self._by_sweep.get(sweep_idx, set()))
        removed = self.remove_many(ids)
        return len(removed)

    def delete_by_group(self, group_id: str) -> int:
        """
        Delete all markers in a group.

        Returns:
            Number of markers deleted
        """
        ids = list(self._by_group.get(group_id, set()))
        removed = self.remove_many(ids)
        return len(removed)

    def move_all_by_offset(self, offset: float, sweep_idx: Optional[int] = None) -> int:
        """
        Move all markers by a time offset.

        Args:
            offset: Time offset in seconds
            sweep_idx: Optional sweep filter

        Returns:
            Number of markers moved
        """
        if sweep_idx is not None:
            markers = self.get_by_sweep(sweep_idx)
        else:
            markers = self.all()

        for marker in markers:
            marker.move(offset)

        if markers:
            self._notify_change()
        return len(markers)
