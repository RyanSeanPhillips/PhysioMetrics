"""
Event marker view model.

This module provides Qt integration for the event marker system,
exposing signals for UI updates and commands for user actions.
"""

from typing import List, Optional, Tuple, Set
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np

from core.domain.events import (
    EventMarker,
    MarkerType,
    EventCategory,
    MarkerStore,
    ThresholdParams,
    TTLParams,
    PeakParams,
    get_category_registry,
)
from core.services import EventMarkerService


class EventMarkerViewModel(QObject):
    """
    View model for event markers.

    Provides Qt signals for UI binding and commands for user actions.
    Acts as the bridge between the UI and the domain/service layers.

    Signals:
        markers_changed: Emitted when any markers change (add/remove/modify)
        selection_changed: Emitted when marker selection changes
        type_changed: Emitted when selected marker type changes
        undo_state_changed: Emitted when undo/redo availability changes
    """

    # Signals
    markers_changed = pyqtSignal()
    selection_changed = pyqtSignal(list)  # List of selected marker IDs
    type_changed = pyqtSignal(str, str)  # category, label
    undo_state_changed = pyqtSignal(bool, bool)  # can_undo, can_redo

    def __init__(self, parent: Optional[QObject] = None):
        """
        Initialize the view model.

        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)

        self._service = EventMarkerService()
        self._selected_ids: Set[str] = set()
        self._show_markers = True

        # Connect store changes to our signal
        self._service.store.on_change(self._on_store_changed)

    @property
    def service(self) -> EventMarkerService:
        """Get the underlying service."""
        return self._service

    @property
    def store(self) -> MarkerStore:
        """Get the marker store."""
        return self._service.store

    @property
    def show_markers(self) -> bool:
        """Whether to show markers in the plot."""
        return self._show_markers

    @show_markers.setter
    def show_markers(self, value: bool) -> None:
        """Set marker visibility."""
        if self._show_markers != value:
            self._show_markers = value
            self.markers_changed.emit()

    # -------------------------------------------------------------------------
    # Type selection (sticky dropdown)
    # -------------------------------------------------------------------------

    @property
    def selected_type(self) -> Tuple[str, str]:
        """Get the currently selected marker type (category, label)."""
        return self._service.selected_type

    def set_selected_type(self, category: str, label: str) -> None:
        """
        Set the selected marker type.

        Args:
            category: Category name
            label: Label name
        """
        self._service.selected_type = (category, label)
        self.type_changed.emit(category, label)

    def get_type_display_name(self) -> str:
        """Get display name for current type (e.g., 'Lick Bout')."""
        category, label = self._service.selected_type
        cat = self._service.registry.get_or_default(category)
        return cat.get_display_label(label)

    def get_available_types(self) -> List[Tuple[str, str, str]]:
        """
        Get all available types for the dropdown.

        Returns:
            List of (category_name, label, display_name) tuples
        """
        result = []
        registry = self._service.registry
        for cat in registry.get_all():
            for label in cat.labels:
                display = cat.get_display_label(label)
                result.append((cat.name, label, f"{cat.display_name}: {display}"))
        return result

    def get_categories(self) -> List[EventCategory]:
        """Get all available categories."""
        return self._service.registry.get_all()

    # -------------------------------------------------------------------------
    # Marker creation commands
    # -------------------------------------------------------------------------

    def add_single_marker(
        self,
        time: float,
        sweep_idx: int = 0,
        snap_to_peak: bool = False,
        signal: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
    ) -> EventMarker:
        """
        Add a single marker at the current type.

        Args:
            time: Time position in seconds
            sweep_idx: Sweep index
            snap_to_peak: Whether to snap to nearest peak
            signal: Signal data for snapping
            sample_rate: Sample rate for snapping

        Returns:
            Created marker
        """
        marker = self._service.add_single_marker(
            time=time,
            sweep_idx=sweep_idx,
            snap_to_peak=snap_to_peak,
            signal=signal,
            sample_rate=sample_rate,
        )
        self._emit_undo_state()
        return marker

    def add_paired_marker(
        self,
        start_time: float,
        sweep_idx: int = 0,
        visible_range: Optional[Tuple[float, float]] = None,
    ) -> EventMarker:
        """
        Add a paired marker at the current type.

        Args:
            start_time: Start time position in seconds
            sweep_idx: Sweep index
            visible_range: Visible x-range for calculating end offset

        Returns:
            Created marker
        """
        marker = self._service.add_paired_marker(
            start_time=start_time,
            sweep_idx=sweep_idx,
            visible_range=visible_range,
        )
        self._emit_undo_state()
        return marker

    # -------------------------------------------------------------------------
    # Marker modification commands
    # -------------------------------------------------------------------------

    def move_marker(self, marker_id: str, new_start: float, new_end: Optional[float] = None) -> bool:
        """Move a marker to new time positions."""
        result = self._service.move_marker(marker_id, new_start, new_end)
        if result:
            self._emit_undo_state()
        return result

    def update_marker(
        self,
        marker_id: str,
        category: Optional[str] = None,
        label: Optional[str] = None,
        color_override: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Update marker properties."""
        result = self._service.update_marker(
            marker_id=marker_id,
            category=category,
            label=label,
            color_override=color_override,
            notes=notes,
        )
        if result:
            self._emit_undo_state()
        return result

    def convert_marker_type(self, marker_id: str, new_type: MarkerType) -> bool:
        """Convert a marker between single and paired types."""
        result = self._service.convert_marker_type(marker_id, new_type)
        if result:
            self._emit_undo_state()
        return result

    def delete_selected(self) -> int:
        """
        Delete all selected markers.

        Returns:
            Number of markers deleted
        """
        if not self._selected_ids:
            return 0

        removed = self._service.delete_markers(list(self._selected_ids))
        self._selected_ids.clear()
        self.selection_changed.emit([])
        self._emit_undo_state()
        return len(removed)

    def delete_marker(self, marker_id: str) -> bool:
        """Delete a specific marker."""
        marker = self._service.delete_marker(marker_id)
        if marker:
            self._selected_ids.discard(marker_id)
            self.selection_changed.emit(list(self._selected_ids))
            self._emit_undo_state()
            return True
        return False

    def delete_all_of_current_type(self) -> int:
        """Delete all markers of the currently selected type."""
        category, label = self._service.selected_type
        count = self._service.delete_all_of_type(category, label)
        if count > 0:
            self._selected_ids.clear()
            self.selection_changed.emit([])
            self._emit_undo_state()
        return count

    def clear_all(self) -> int:
        """Clear all markers."""
        count = self._service.clear_all()
        if count > 0:
            self._selected_ids.clear()
            self.selection_changed.emit([])
            self._emit_undo_state()
        return count

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    @property
    def selected_ids(self) -> Set[str]:
        """Get IDs of selected markers."""
        return self._selected_ids.copy()

    def select_marker(self, marker_id: str, add_to_selection: bool = False) -> None:
        """
        Select a marker.

        Args:
            marker_id: ID of marker to select
            add_to_selection: If True, add to selection; if False, replace selection
        """
        if not add_to_selection:
            self._selected_ids.clear()
        self._selected_ids.add(marker_id)
        self.selection_changed.emit(list(self._selected_ids))

    def deselect_marker(self, marker_id: str) -> None:
        """Deselect a marker."""
        self._selected_ids.discard(marker_id)
        self.selection_changed.emit(list(self._selected_ids))

    def select_all_in_sweep(self, sweep_idx: int) -> None:
        """Select all markers in a sweep."""
        markers = self._service.get_markers_for_sweep(sweep_idx)
        self._selected_ids = {m.id for m in markers}
        self.selection_changed.emit(list(self._selected_ids))

    def deselect_all(self) -> None:
        """Deselect all markers."""
        self._selected_ids.clear()
        self.selection_changed.emit([])

    def is_selected(self, marker_id: str) -> bool:
        """Check if a marker is selected."""
        return marker_id in self._selected_ids

    def get_selected_markers(self) -> List[EventMarker]:
        """Get the selected marker objects."""
        return [
            self._service.store.get(mid)
            for mid in self._selected_ids
            if self._service.store.get(mid) is not None
        ]

    # -------------------------------------------------------------------------
    # Undo/Redo
    # -------------------------------------------------------------------------

    def undo(self) -> bool:
        """Undo the last action."""
        result = self._service.undo()
        self._emit_undo_state()
        return result

    def redo(self) -> bool:
        """Redo the last undone action."""
        result = self._service.redo()
        self._emit_undo_state()
        return result

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._service.can_undo()

    @property
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._service.can_redo()

    def _emit_undo_state(self) -> None:
        """Emit the current undo/redo state."""
        self.undo_state_changed.emit(self.can_undo, self.can_redo)

    # -------------------------------------------------------------------------
    # Auto-detection
    # -------------------------------------------------------------------------

    def auto_detect_threshold(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: ThresholdParams,
        sweep_idx: int = 0,
        source_channel: Optional[str] = None,
    ) -> int:
        """
        Run threshold detection.

        Returns:
            Number of markers detected
        """
        result = self._service.auto_detect_threshold(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            source_channel=source_channel,
        )
        self._emit_undo_state()
        return result.stats['num_found']

    def auto_detect_ttl(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: TTLParams,
        sweep_idx: int = 0,
        source_channel: Optional[str] = None,
    ) -> int:
        """
        Run TTL detection.

        Returns:
            Number of markers detected
        """
        result = self._service.auto_detect_ttl(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            source_channel=source_channel,
        )
        self._emit_undo_state()
        return result.stats['num_found']

    def auto_detect_peaks(
        self,
        signal: np.ndarray,
        sample_rate: float,
        params: PeakParams,
        sweep_idx: int = 0,
        source_channel: Optional[str] = None,
    ) -> int:
        """
        Run peak detection.

        Returns:
            Number of markers detected
        """
        result = self._service.auto_detect_peaks(
            signal=signal,
            sample_rate=sample_rate,
            params=params,
            sweep_idx=sweep_idx,
            source_channel=source_channel,
        )
        self._emit_undo_state()
        return result.stats['num_found']

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    def get_markers_for_sweep(self, sweep_idx: int) -> List[EventMarker]:
        """Get all markers for a sweep."""
        return self._service.get_markers_for_sweep(sweep_idx)

    def get_markers_in_view(
        self,
        start_time: float,
        end_time: float,
        sweep_idx: Optional[int] = None
    ) -> List[EventMarker]:
        """Get markers visible in a time range."""
        return self._service.get_markers_in_view(start_time, end_time, sweep_idx)

    def get_marker_at_position(
        self,
        time: float,
        sweep_idx: int,
        tolerance: float = 0.02
    ) -> Optional[EventMarker]:
        """Get marker at or near a position."""
        return self._service.get_marker_at_position(time, sweep_idx, tolerance)

    def get_color_for_marker(self, marker: EventMarker) -> str:
        """Get the display color for a marker."""
        return self._service.get_color_for_marker(marker)

    @property
    def marker_count(self) -> int:
        """Get total marker count."""
        return self._service.get_marker_count()

    def get_marker_count_for_sweep(self, sweep_idx: int) -> int:
        """Get marker count for a specific sweep."""
        return self._service.get_marker_count_by_sweep(sweep_idx)

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_to_npz(self) -> dict:
        """Get data for NPZ storage."""
        return self._service.to_npz_dict()

    def load_from_npz(self, npz_data: dict) -> int:
        """
        Load markers from NPZ data.

        Returns:
            Number of markers loaded
        """
        count = self._service.load_from_npz(npz_data)
        self._selected_ids.clear()
        self.selection_changed.emit([])
        self._service.clear_undo_history()
        self._emit_undo_state()
        return count

    def set_store(self, store: MarkerStore) -> None:
        """
        Replace the marker store (e.g., when loading a new file).

        Args:
            store: New marker store
        """
        # Disconnect old store
        try:
            self._service.store.remove_change_callback(self._on_store_changed)
        except (ValueError, AttributeError):
            pass

        # Create new service with new store
        self._service = EventMarkerService(store)
        self._service.store.on_change(self._on_store_changed)

        self._selected_ids.clear()
        self.selection_changed.emit([])
        self.markers_changed.emit()
        self._emit_undo_state()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _on_store_changed(self) -> None:
        """Handle store change notification."""
        self.markers_changed.emit()
