"""
Event marker service.

This module provides the service layer for event markers, orchestrating
domain operations and providing higher-level methods for the UI.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import uuid
import json
import numpy as np

from ..domain.events import (
    EventMarker,
    MarkerType,
    EventCategory,
    MarkerStore,
    CategoryRegistry,
    get_category_registry,
    ThresholdParams,
    TTLParams,
    PeakParams,
    DetectionResult,
    detect_threshold_crossings,
    detect_ttl_events,
    detect_peaks,
    find_nearest_peak,
)


@dataclass
class UndoableAction:
    """Represents an undoable action."""
    action_type: str  # 'add', 'remove', 'update', 'bulk_add', 'bulk_remove'
    markers: List[EventMarker]  # Markers involved (before state for update/remove)
    new_markers: Optional[List[EventMarker]] = None  # After state for update


class EventMarkerService:
    """
    Service for managing event markers.

    Provides high-level operations for creating, modifying, and querying
    event markers. Supports undo/redo and integrates with the category registry.
    """

    def __init__(self, store: Optional[MarkerStore] = None):
        """
        Initialize the service.

        Args:
            store: Optional existing MarkerStore. Creates new if not provided.
        """
        self._store = store if store is not None else MarkerStore()
        self._registry = get_category_registry()
        self._undo_stack: List[UndoableAction] = []
        self._redo_stack: List[UndoableAction] = []
        self._max_undo = 50
        self._selected_type: Tuple[str, str] = ('stimulus', 'stim_on')  # (category, label)

    @property
    def store(self) -> MarkerStore:
        """Get the marker store."""
        return self._store

    @property
    def registry(self) -> CategoryRegistry:
        """Get the category registry."""
        return self._registry

    @property
    def selected_type(self) -> Tuple[str, str]:
        """Get the currently selected marker type (category, label)."""
        return self._selected_type

    @selected_type.setter
    def selected_type(self, value: Tuple[str, str]) -> None:
        """Set the currently selected marker type."""
        self._selected_type = value
        # Increment usage count for the category
        self._registry.increment_usage(value[0])

    # -------------------------------------------------------------------------
    # Marker creation
    # -------------------------------------------------------------------------

    def add_single_marker(
        self,
        time: float,
        sweep_idx: int = 0,
        category: Optional[str] = None,
        label: Optional[str] = None,
        source_channel: Optional[str] = None,
        snap_to_peak: bool = False,
        signal: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
    ) -> EventMarker:
        """
        Add a single (point) marker.

        Args:
            time: Time position in seconds
            sweep_idx: Sweep index
            category: Category (uses selected_type if None)
            label: Label (uses selected_type if None)
            source_channel: Channel name for metadata
            snap_to_peak: Whether to snap to nearest peak
            signal: Signal data for snapping (required if snap_to_peak)
            sample_rate: Sample rate for snapping (required if snap_to_peak)

        Returns:
            Created marker
        """
        if category is None:
            category = self._selected_type[0]
        if label is None:
            label = self._selected_type[1]

        # Snap to peak if requested
        if snap_to_peak and signal is not None and sample_rate is not None:
            snapped = find_nearest_peak(signal, sample_rate, time, search_radius_ms=50.0)
            if snapped is not None:
                time = snapped

        marker = EventMarker(
            sweep_idx=sweep_idx,
            marker_type=MarkerType.SINGLE,
            start_time=time,
            category=category,
            label=label,
            source_channel=source_channel,
            detection_method='manual',
        )

        self._store.add(marker)
        self._push_undo(UndoableAction('add', [marker]))
        return marker

    def add_paired_marker(
        self,
        start_time: float,
        sweep_idx: int = 0,
        category: Optional[str] = None,
        label: Optional[str] = None,
        source_channel: Optional[str] = None,
        end_time: Optional[float] = None,
        end_offset_percent: float = 10.0,
        visible_range: Optional[Tuple[float, float]] = None,
        detection_method: str = 'manual',
    ) -> EventMarker:
        """
        Add a paired (region) marker.

        The end time can be explicitly specified, or automatically calculated
        as a percentage of the visible range past the start time.

        Args:
            start_time: Start time position in seconds
            sweep_idx: Sweep index
            category: Category (uses selected_type if None)
            label: Label (uses selected_type if None)
            source_channel: Channel name for metadata
            end_time: Explicit end time (if None, calculated from offset)
            end_offset_percent: Percentage of visible range for end offset
            visible_range: (start, end) of visible x-range for calculating offset
            detection_method: How the marker was created ('manual', 'threshold', etc.)

        Returns:
            Created marker
        """
        if category is None:
            category = self._selected_type[0]
        if label is None:
            label = self._selected_type[1]

        # Use explicit end_time if provided, otherwise calculate
        if end_time is None:
            if visible_range is not None:
                range_width = visible_range[1] - visible_range[0]
                offset = range_width * (end_offset_percent / 100.0)
            else:
                offset = 0.5  # Default 0.5s if no range provided

            # Ensure minimum offset
            offset = max(offset, 0.1)
            end_time = start_time + offset

        marker = EventMarker(
            sweep_idx=sweep_idx,
            marker_type=MarkerType.PAIRED,
            start_time=start_time,
            end_time=end_time,
            category=category,
            label=label,
            source_channel=source_channel,
            detection_method=detection_method,
        )

        self._store.add(marker)
        self._push_undo(UndoableAction('add', [marker]))
        return marker

    # -------------------------------------------------------------------------
    # Marker modification
    # -------------------------------------------------------------------------

    def move_marker(self, marker_id: str, new_start: float, new_end: Optional[float] = None) -> bool:
        """
        Move a marker to new time positions.

        Args:
            marker_id: ID of marker to move
            new_start: New start time
            new_end: New end time (for paired markers)

        Returns:
            True if moved, False if marker not found
        """
        marker = self._store.get(marker_id)
        if marker is None:
            return False

        old_marker = marker.copy()
        marker.set_times(new_start, new_end)
        self._store.update(marker)
        self._push_undo(UndoableAction('update', [old_marker], [marker.copy()]))
        return True

    def update_marker(
        self,
        marker_id: str,
        category: Optional[str] = None,
        label: Optional[str] = None,
        condition: Optional[str] = None,
        color_override: Optional[str] = None,
        line_width: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Update marker properties.

        Args:
            marker_id: ID of marker to update
            category: New category (or None to keep current)
            label: New label (or None to keep current)
            condition: Experimental condition (or None to keep current)
            color_override: New color override (or None to keep current)
            line_width: Custom line width (or None to keep current, 0 to reset to default)
            notes: New notes (or None to keep current)

        Returns:
            True if updated, False if marker not found
        """
        marker = self._store.get(marker_id)
        if marker is None:
            return False

        # Save old values BEFORE modifying (for undo and index cleanup)
        old_marker = marker.copy()
        old_category = marker.category
        old_sweep_idx = marker.sweep_idx

        if category is not None:
            marker.category = category
        if label is not None:
            marker.label = label
        if condition is not None:
            marker.condition = condition if condition else None
        if color_override is not None:
            marker.color_override = color_override if color_override else None
        if line_width is not None:
            marker.line_width = line_width if line_width > 0 else None
        if notes is not None:
            marker.notes = notes if notes else None

        # Pass old values to store for proper index cleanup
        self._store.update(marker, old_category=old_category, old_sweep_idx=old_sweep_idx)
        self._push_undo(UndoableAction('update', [old_marker], [marker.copy()]))
        return True

    def convert_marker_type(self, marker_id: str, new_type: MarkerType) -> bool:
        """
        Convert a marker between single and paired types.

        Args:
            marker_id: ID of marker to convert
            new_type: Target marker type

        Returns:
            True if converted, False if marker not found or already that type
        """
        marker = self._store.get(marker_id)
        if marker is None or marker.marker_type == new_type:
            return False

        old_marker = marker.copy()

        marker.marker_type = new_type
        if new_type == MarkerType.SINGLE:
            marker.end_time = None
        elif new_type == MarkerType.PAIRED and marker.end_time is None:
            marker.end_time = marker.start_time + 0.5

        self._store.update(marker)
        self._push_undo(UndoableAction('update', [old_marker], [marker.copy()]))
        return True

    def delete_marker(self, marker_id: str) -> Optional[EventMarker]:
        """
        Delete a marker.

        Args:
            marker_id: ID of marker to delete

        Returns:
            Deleted marker, or None if not found
        """
        marker = self._store.remove(marker_id)
        if marker:
            self._push_undo(UndoableAction('remove', [marker]))
        return marker

    def delete_markers(self, marker_ids: List[str]) -> List[EventMarker]:
        """
        Delete multiple markers.

        Args:
            marker_ids: IDs of markers to delete

        Returns:
            List of deleted markers
        """
        removed = self._store.remove_many(marker_ids)
        if removed:
            self._push_undo(UndoableAction('bulk_remove', removed))
        return removed

    def delete_all_of_type(self, category: str, label: str) -> int:
        """
        Delete all markers of a specific type.

        Args:
            category: Category to delete
            label: Label to delete

        Returns:
            Number of markers deleted
        """
        markers = self._store.get_by_label(category, label)
        if markers:
            ids = [m.id for m in markers]
            removed = self._store.remove_many(ids)
            self._push_undo(UndoableAction('bulk_remove', removed))
            return len(removed)
        return 0

    def delete_all_for_sweep(self, sweep_idx: int) -> int:
        """
        Delete all markers for a specific sweep.

        Args:
            sweep_idx: Sweep index to delete markers for

        Returns:
            Number of markers deleted
        """
        markers = self._store.get_by_sweep(sweep_idx)
        if markers:
            ids = [m.id for m in markers]
            removed = self._store.remove_many(ids)
            self._push_undo(UndoableAction('bulk_remove', removed))
            return len(removed)
        return 0

    def delete_category_all_sweeps(self, category: str) -> int:
        """
        Delete all markers of a category across all sweeps.

        Args:
            category: Category to delete

        Returns:
            Number of markers deleted
        """
        markers = self._store.get_by_category(category)
        if markers:
            ids = [m.id for m in markers]
            removed = self._store.remove_many(ids)
            self._push_undo(UndoableAction('bulk_remove', removed))
            return len(removed)
        return 0

    def delete_category_for_sweep(self, category: str, sweep_idx: int) -> int:
        """
        Delete all markers of a category in a specific sweep.

        Args:
            category: Category to delete
            sweep_idx: Sweep index to delete from

        Returns:
            Number of markers deleted
        """
        # Get markers that match both category AND sweep
        sweep_markers = self._store.get_by_sweep(sweep_idx)
        matching = [m for m in sweep_markers if m.category == category]
        if matching:
            ids = [m.id for m in matching]
            removed = self._store.remove_many(ids)
            self._push_undo(UndoableAction('bulk_remove', removed))
            return len(removed)
        return 0

    def clear_all(self) -> int:
        """
        Clear all markers.

        Returns:
            Number of markers removed
        """
        markers = self._store.all()
        if markers:
            self._push_undo(UndoableAction('bulk_remove', [m.copy() for m in markers]))
        return self._store.clear()

    # -------------------------------------------------------------------------
    # Auto-detection
    # -------------------------------------------------------------------------

    def auto_detect_threshold(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: ThresholdParams,
        sweep_idx: int = 0,
        category: Optional[str] = None,
        label: Optional[str] = None,
        source_channel: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run threshold-based auto-detection.

        Args:
            signal: Signal data
            sample_rate: Sample rate in Hz
            params: Detection parameters
            sweep_idx: Sweep index for markers
            category: Category (uses selected_type if None)
            label: Label (uses selected_type if None)
            source_channel: Channel name

        Returns:
            DetectionResult with markers and stats
        """
        if category is None:
            category = self._selected_type[0]
        if label is None:
            label = self._selected_type[1]

        group_id = str(uuid.uuid4())[:8]

        result = detect_threshold_crossings(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            category=category,
            label=label,
            source_channel=source_channel,
            group_id=group_id,
        )

        if result.markers:
            self._store.add_many(result.markers)
            self._push_undo(UndoableAction('bulk_add', [m.copy() for m in result.markers]))

        return result

    def auto_detect_ttl(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: TTLParams,
        sweep_idx: int = 0,
        category: Optional[str] = None,
        label: Optional[str] = None,
        source_channel: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run TTL-based auto-detection.

        Args:
            signal: Signal data
            sample_rate: Sample rate in Hz
            params: Detection parameters
            sweep_idx: Sweep index for markers
            category: Category (uses selected_type if None)
            label: Label (uses selected_type if None)
            source_channel: Channel name

        Returns:
            DetectionResult with markers and stats
        """
        if category is None:
            category = self._selected_type[0]
        if label is None:
            label = self._selected_type[1]

        group_id = str(uuid.uuid4())[:8]

        result = detect_ttl_events(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            category=category,
            label=label,
            source_channel=source_channel,
            group_id=group_id,
        )

        if result.markers:
            self._store.add_many(result.markers)
            self._push_undo(UndoableAction('bulk_add', [m.copy() for m in result.markers]))

        return result

    def auto_detect_peaks(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: PeakParams,
        sweep_idx: int = 0,
        category: Optional[str] = None,
        label: Optional[str] = None,
        source_channel: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run peak-based auto-detection.

        Args:
            signal: Signal data
            sample_rate: Sample rate in Hz
            params: Detection parameters
            sweep_idx: Sweep index for markers
            category: Category (uses selected_type if None)
            label: Label (uses selected_type if None)
            source_channel: Channel name

        Returns:
            DetectionResult with markers and stats
        """
        if category is None:
            category = self._selected_type[0]
        if label is None:
            label = self._selected_type[1]

        group_id = str(uuid.uuid4())[:8]

        result = detect_peaks(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            category=category,
            label=label,
            source_channel=source_channel,
            group_id=group_id,
        )

        if result.markers:
            self._store.add_many(result.markers)
            self._push_undo(UndoableAction('bulk_add', [m.copy() for m in result.markers]))

        return result

    # -------------------------------------------------------------------------
    # Undo/Redo
    # -------------------------------------------------------------------------

    def undo(self) -> bool:
        """
        Undo the last action.

        Returns:
            True if undo was performed, False if nothing to undo
        """
        if not self._undo_stack:
            return False

        action = self._undo_stack.pop()

        if action.action_type == 'add':
            for marker in action.markers:
                self._store.remove(marker.id)
        elif action.action_type == 'remove':
            for marker in action.markers:
                self._store.add(marker)
        elif action.action_type == 'update':
            for old_marker in action.markers:
                self._store.update(old_marker)
        elif action.action_type == 'bulk_add':
            for marker in action.markers:
                self._store.remove(marker.id)
        elif action.action_type == 'bulk_remove':
            for marker in action.markers:
                self._store.add(marker)

        self._redo_stack.append(action)
        return True

    def redo(self) -> bool:
        """
        Redo the last undone action.

        Returns:
            True if redo was performed, False if nothing to redo
        """
        if not self._redo_stack:
            return False

        action = self._redo_stack.pop()

        if action.action_type == 'add':
            for marker in action.markers:
                self._store.add(marker)
        elif action.action_type == 'remove':
            for marker in action.markers:
                self._store.remove(marker.id)
        elif action.action_type == 'update':
            if action.new_markers:
                for new_marker in action.new_markers:
                    self._store.update(new_marker)
        elif action.action_type == 'bulk_add':
            for marker in action.markers:
                self._store.add(marker)
        elif action.action_type == 'bulk_remove':
            for marker in action.markers:
                self._store.remove(marker.id)

        self._undo_stack.append(action)
        return True

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0

    def clear_undo_history(self) -> None:
        """Clear undo/redo history."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    def _push_undo(self, action: UndoableAction) -> None:
        """Push action to undo stack, clearing redo stack."""
        self._undo_stack.append(action)
        self._redo_stack.clear()
        # Limit stack size
        while len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def to_npz_dict(self) -> Dict[str, Any]:
        """
        Get data for NPZ storage (version 2 format).

        Returns:
            Dict ready to be saved in NPZ file
        """
        data = {
            'markers': [m.to_dict() for m in self._store.all()],
            'custom_categories': self._registry.get_custom_only_dict(),
            'settings': {
                'last_selected_type': list(self._selected_type),
            }
        }

        return {
            'event_markers_version': np.array([2]),
            'event_markers_json': json.dumps(data),
        }

    def load_from_npz(self, npz_data: Dict[str, Any]) -> int:
        """
        Load markers from NPZ data.

        Handles both v2 format and legacy format (auto-migration).

        Args:
            npz_data: Dict from NPZ file

        Returns:
            Number of markers loaded
        """
        # Check for v2 format
        if 'event_markers_version' in npz_data:
            version = int(npz_data['event_markers_version'][0])
            if version >= 2:
                return self._load_v2_format(npz_data)

        # Check for legacy format
        if 'bout_sweep_indices' in npz_data:
            return self._load_legacy_format(npz_data)

        return 0

    def _load_v2_format(self, npz_data: Dict[str, Any]) -> int:
        """Load from v2 format."""
        json_str = str(npz_data['event_markers_json'])
        data = json.loads(json_str)

        # Load markers
        self._store.clear()
        for marker_data in data.get('markers', []):
            marker = EventMarker.from_dict(marker_data)
            self._store.add(marker)

        # Load custom categories
        if 'custom_categories' in data:
            self._registry.load_customizations(data['custom_categories'])

        # Load settings
        if 'settings' in data:
            settings = data['settings']
            if 'last_selected_type' in settings:
                self._selected_type = tuple(settings['last_selected_type'])

        self._store.modified = False
        return len(self._store)

    def _load_legacy_format(self, npz_data: Dict[str, Any]) -> int:
        """
        Load and migrate from legacy bout_annotations format.

        Legacy format stored per-sweep JSON arrays with 'start_time', 'end_time', 'id'.
        """
        self._store.clear()

        sweep_indices = npz_data['bout_sweep_indices']
        for sweep_idx in sweep_indices:
            key = f'bout_sweep_{sweep_idx}_json'
            if key in npz_data:
                bouts_json = str(npz_data[key])
                bouts = json.loads(bouts_json)
                for bout in bouts:
                    marker = EventMarker(
                        sweep_idx=int(sweep_idx),
                        marker_type=MarkerType.PAIRED,
                        start_time=bout.get('start_time', 0.0),
                        end_time=bout.get('end_time', bout.get('start_time', 0.0) + 0.5),
                        category='behavior',  # Default for legacy lick bouts
                        label='lick_bout',
                        detection_method='manual',
                    )
                    self._store.add(marker)

        self._store.modified = False
        return len(self._store)

    # -------------------------------------------------------------------------
    # Queries (delegated to store with additional convenience)
    # -------------------------------------------------------------------------

    def get_markers_for_sweep(self, sweep_idx: int) -> List[EventMarker]:
        """Get all markers for a sweep."""
        return self._store.get_by_sweep(sweep_idx)

    def get_markers_in_view(
        self,
        start_time: float,
        end_time: float,
        sweep_idx: Optional[int] = None
    ) -> List[EventMarker]:
        """Get markers visible in a time range."""
        return self._store.get_in_time_range(start_time, end_time, sweep_idx)

    def get_marker_at_position(
        self,
        time: float,
        sweep_idx: int,
        tolerance: float = 0.02
    ) -> Optional[EventMarker]:
        """Get marker at or near a position (for click detection)."""
        markers = self._store.get_at_time(time, tolerance, sweep_idx)
        if markers:
            # Return the one closest to the click
            return min(markers, key=lambda m: abs(m.start_time - time))
        return None

    def get_color_for_marker(self, marker: EventMarker) -> str:
        """Get the display color for a marker."""
        if marker.color_override:
            return marker.color_override
        category = self._registry.get_or_default(marker.category)
        return category.color

    def get_marker_count(self) -> int:
        """Get total marker count."""
        return len(self._store)

    def get_marker_count_by_sweep(self, sweep_idx: int) -> int:
        """Get marker count for a specific sweep."""
        return len(self._store.get_by_sweep(sweep_idx))
