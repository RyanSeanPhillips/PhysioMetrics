"""
Event marker editor.

This module handles interactive editing of markers (dragging, resizing)
and keyboard shortcuts for quick marker creation.
"""

from typing import Optional, Tuple, Callable
from PyQt6.QtCore import QObject, Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QWidget
import pyqtgraph as pg

from viewmodels.event_marker_viewmodel import EventMarkerViewModel
from core.domain.events import MarkerType


class MarkerEditor(QObject):
    """
    Handles interactive editing of event markers.

    Provides:
    - Click-and-drag to move markers
    - Keyboard shortcuts (S+click, P+click) for quick creation
    - Handle detection for paired marker boundary adjustment
    """

    # Signals
    marker_moved = pyqtSignal(str, float, float)  # marker_id, new_start, new_end
    marker_created = pyqtSignal(str)              # marker_id
    edit_started = pyqtSignal(str)                # marker_id
    edit_finished = pyqtSignal(str)               # marker_id

    # Editing states
    IDLE = 0
    DRAGGING_MARKER = 1
    DRAGGING_START_HANDLE = 2
    DRAGGING_END_HANDLE = 3

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        plot_widget: pg.PlotWidget,
        get_sweep_idx: Callable[[], int],
        get_visible_range: Callable[[], Tuple[float, float]],
        plot_host=None,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize the editor.

        Args:
            viewmodel: The event marker view model
            plot_widget: The PyQtGraph PlotWidget for mouse events
            get_sweep_idx: Callback to get current sweep index
            get_visible_range: Callback to get visible x-range
            plot_host: Reference to PyQtGraphPlotHost for coordinate conversion
            parent: Parent QObject
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._plot_widget = plot_widget
        self._get_sweep_idx = get_sweep_idx
        self._get_visible_range = get_visible_range
        self._plot_host = plot_host

        # State
        self._state = self.IDLE
        self._active_marker_id: Optional[str] = None
        self._drag_start_pos: Optional[float] = None
        self._original_start: Optional[float] = None
        self._original_end: Optional[float] = None

        # Keyboard modifiers
        self._s_pressed = False
        self._p_pressed = False

        # Handle tolerance (in data coordinates)
        self._handle_tolerance = 0.05  # 50ms - wider tolerance for easier clicking

        # Install event filter on the graphics layout widget
        self._plot_widget.installEventFilter(self)

        # Also try viewport for mouse events
        if hasattr(self._plot_widget, 'viewport'):
            viewport = self._plot_widget.viewport()
            if viewport:
                viewport.installEventFilter(self)

    def eventFilter(self, obj: QObject, event) -> bool:
        """Filter events for keyboard shortcuts and mouse handling."""
        from PyQt6.QtCore import QEvent

        # Handle key events (on the plot widget)
        if event.type() == QEvent.Type.KeyPress:
            return self._handle_key_press(event)
        elif event.type() == QEvent.Type.KeyRelease:
            return self._handle_key_release(event)

        # Handle mouse events on the graphics scene
        elif event.type() == QEvent.Type.GraphicsSceneMousePress:
            if event.button() == Qt.MouseButton.LeftButton:
                return self._handle_scene_mouse_press(event)
        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
            return self._handle_scene_mouse_move(event)
        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease:
            if event.button() == Qt.MouseButton.LeftButton:
                return self._handle_scene_mouse_release(event)

        return False

    def _get_data_pos_from_scene(self, scene_pos) -> Optional[QPointF]:
        """Convert scene position to data coordinates."""
        plot = None

        # Try to get main plot from plot_host first
        if self._plot_host and hasattr(self._plot_host, '_get_main_plot'):
            plot = self._plot_host._get_main_plot()
        elif hasattr(self._plot_widget, '_get_main_plot'):
            plot = self._plot_widget._get_main_plot()
        elif hasattr(self._plot_widget, 'plotItem'):
            plot = self._plot_widget.plotItem
        else:
            # Try to find a PlotItem in the layout
            try:
                for i in range(self._plot_widget.ci.layout.count()):
                    item = self._plot_widget.ci.layout.itemAt(i)
                    if hasattr(item, 'vb'):
                        plot = item
                        break
            except:
                pass

        if plot is None or not hasattr(plot, 'vb'):
            return None

        # Check if click is within plot bounds
        if not plot.sceneBoundingRect().contains(scene_pos):
            return None

        # Convert to data coordinates
        return plot.vb.mapSceneToView(scene_pos)

    def _handle_scene_mouse_press(self, event) -> bool:
        """Handle mouse press on the graphics scene."""
        data_pos = self._get_data_pos_from_scene(event.scenePos())
        if data_pos is None:
            return False

        # Check if we're clicking on a marker BEFORE calling handle_mouse_press
        x = data_pos.x()
        marker = self._viewmodel.get_marker_at_position(
            x, self._get_sweep_idx(), self._handle_tolerance
        )

        if marker:
            # Select the marker but let PyQtGraph handle the actual drag
            # (InfiniteLine items with movable=True handle their own mouse events)
            self._viewmodel.select_marker(marker.id)
            return False  # Let event propagate to movable InfiniteLine

        return False  # Let PyQtGraph handle clicks on empty space

    def _handle_scene_mouse_move(self, event) -> bool:
        """Handle mouse move on the graphics scene."""
        if self._state == self.IDLE:
            return False
        data_pos = self._get_data_pos_from_scene(event.scenePos())
        if data_pos is None:
            return False
        return self.handle_mouse_move(data_pos)

    def _handle_scene_mouse_release(self, event) -> bool:
        """Handle mouse release on the graphics scene."""
        if self._state == self.IDLE:
            return False
        data_pos = self._get_data_pos_from_scene(event.scenePos())
        if data_pos is None:
            return False
        return self.handle_mouse_release(data_pos, event.button())

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        """Handle key press events."""
        if event.key() == Qt.Key.Key_S:
            self._s_pressed = True
            return False  # Don't consume, let click handler use it
        elif event.key() == Qt.Key.Key_P:
            self._p_pressed = True
            return False
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete selected markers
            if self._viewmodel.selected_ids:
                self._viewmodel.delete_selected()
                return True
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel drag or deselect
            if self._state != self.IDLE:
                self._cancel_drag()
                return True
            else:
                self._viewmodel.deselect_all()
                return True
        elif event.key() == Qt.Key.Key_Z and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._viewmodel.redo()
            else:
                self._viewmodel.undo()
            return True
        elif event.key() == Qt.Key.Key_A and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Select all in current sweep
            self._viewmodel.select_all_in_sweep(self._get_sweep_idx())
            return True

        return False

    def _handle_key_release(self, event: QKeyEvent) -> bool:
        """Handle key release events."""
        if event.key() == Qt.Key.Key_S:
            self._s_pressed = False
        elif event.key() == Qt.Key.Key_P:
            self._p_pressed = False
        return False

    def handle_mouse_press(self, pos: QPointF, button: Qt.MouseButton, modifiers: Qt.KeyboardModifier) -> bool:
        """
        Handle mouse press event on the plot.

        Args:
            pos: Position in data coordinates
            button: Mouse button
            modifiers: Keyboard modifiers

        Returns:
            True if event was handled
        """
        if button != Qt.MouseButton.LeftButton:
            return False

        x = pos.x()

        # Check for keyboard shortcut quick-add
        if self._s_pressed:
            self._quick_add_single(x)
            return True
        if self._p_pressed:
            self._quick_add_paired(x)
            return True

        # Check if clicking on an existing marker
        marker = self._viewmodel.get_marker_at_position(
            x, self._get_sweep_idx(), self._handle_tolerance
        )

        if marker:
            # Start dragging
            return self._start_drag(marker, x, modifiers)
        else:
            # Deselect if clicking empty space (without Ctrl)
            if not (modifiers & Qt.KeyboardModifier.ControlModifier):
                self._viewmodel.deselect_all()

        return False

    def handle_mouse_move(self, pos: QPointF) -> bool:
        """
        Handle mouse move event during drag.

        Args:
            pos: Position in data coordinates

        Returns:
            True if event was handled
        """
        if self._state == self.IDLE:
            return False

        x = pos.x()

        if self._state == self.DRAGGING_MARKER:
            self._update_drag_marker(x)
        elif self._state == self.DRAGGING_START_HANDLE:
            self._update_drag_start_handle(x)
        elif self._state == self.DRAGGING_END_HANDLE:
            self._update_drag_end_handle(x)

        return True

    def handle_mouse_release(self, pos: QPointF, button: Qt.MouseButton) -> bool:
        """
        Handle mouse release event.

        Args:
            pos: Position in data coordinates
            button: Mouse button

        Returns:
            True if event was handled
        """
        if button != Qt.MouseButton.LeftButton:
            return False

        if self._state == self.IDLE:
            return False

        x = pos.x()
        self._finish_drag(x)
        return True

    def _quick_add_single(self, x: float) -> None:
        """Quick-add a single marker at position."""
        marker = self._viewmodel.add_single_marker(
            time=x,
            sweep_idx=self._get_sweep_idx(),
        )
        self._viewmodel.select_marker(marker.id)
        self.marker_created.emit(marker.id)

    def _quick_add_paired(self, x: float) -> None:
        """Quick-add a paired marker at position."""
        marker = self._viewmodel.add_paired_marker(
            start_time=x,
            sweep_idx=self._get_sweep_idx(),
            visible_range=self._get_visible_range(),
        )
        self._viewmodel.select_marker(marker.id)
        self.marker_created.emit(marker.id)

    def _start_drag(self, marker, x: float, modifiers: Qt.KeyboardModifier) -> bool:
        """Start dragging a marker."""
        self._active_marker_id = marker.id
        self._drag_start_pos = x
        self._original_start = marker.start_time
        self._original_end = marker.end_time

        # Select the marker
        add_to_selection = modifiers & Qt.KeyboardModifier.ControlModifier
        self._viewmodel.select_marker(marker.id, add_to_selection)

        # Determine if clicking on a handle or the whole marker
        if marker.is_paired and marker.end_time is not None:
            # Check if near start or end handle
            if abs(x - marker.start_time) <= self._handle_tolerance:
                self._state = self.DRAGGING_START_HANDLE
            elif abs(x - marker.end_time) <= self._handle_tolerance:
                self._state = self.DRAGGING_END_HANDLE
            else:
                self._state = self.DRAGGING_MARKER
        else:
            self._state = self.DRAGGING_MARKER

        self.edit_started.emit(marker.id)
        return True

    def _update_drag_marker(self, x: float) -> None:
        """Update marker position during drag."""
        if not self._active_marker_id or self._drag_start_pos is None:
            return

        delta = x - self._drag_start_pos
        new_start = self._original_start + delta
        new_end = None
        if self._original_end is not None:
            new_end = self._original_end + delta

        # Update visual (preview)
        self.marker_moved.emit(self._active_marker_id, new_start, new_end or 0)

    def _update_drag_start_handle(self, x: float) -> None:
        """Update start handle position during drag."""
        if not self._active_marker_id:
            return

        # Clamp to not exceed end
        new_start = x
        if self._original_end is not None and new_start > self._original_end:
            new_start = self._original_end - 0.01

        self.marker_moved.emit(self._active_marker_id, new_start, self._original_end or 0)

    def _update_drag_end_handle(self, x: float) -> None:
        """Update end handle position during drag."""
        if not self._active_marker_id:
            return

        # Clamp to not go before start
        new_end = x
        if self._original_start is not None and new_end < self._original_start:
            new_end = self._original_start + 0.01

        self.marker_moved.emit(self._active_marker_id, self._original_start or 0, new_end)

    def _finish_drag(self, x: float) -> None:
        """Finish dragging and commit changes."""
        if not self._active_marker_id:
            self._reset_state()
            return

        # Calculate final position
        if self._state == self.DRAGGING_MARKER:
            delta = x - (self._drag_start_pos or 0)
            new_start = (self._original_start or 0) + delta
            new_end = None
            if self._original_end is not None:
                new_end = self._original_end + delta
        elif self._state == self.DRAGGING_START_HANDLE:
            new_start = x
            new_end = self._original_end
            # Ensure start < end
            if new_end is not None and new_start > new_end:
                new_start, new_end = new_end, new_start
        elif self._state == self.DRAGGING_END_HANDLE:
            new_start = self._original_start or 0
            new_end = x
            # Ensure start < end
            if new_end < new_start:
                new_start, new_end = new_end, new_start
        else:
            self._reset_state()
            return

        # Commit the change
        self._viewmodel.move_marker(self._active_marker_id, new_start, new_end)
        self.edit_finished.emit(self._active_marker_id)

        self._reset_state()

    def _cancel_drag(self) -> None:
        """Cancel the current drag operation."""
        if self._active_marker_id and self._original_start is not None:
            # Restore original position (visually)
            self.marker_moved.emit(
                self._active_marker_id,
                self._original_start,
                self._original_end or 0
            )

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset editor state."""
        self._state = self.IDLE
        self._active_marker_id = None
        self._drag_start_pos = None
        self._original_start = None
        self._original_end = None

    @property
    def is_editing(self) -> bool:
        """Check if currently editing a marker."""
        return self._state != self.IDLE

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self._plot_widget.removeEventFilter(self)
            scene = self._plot_widget.scene()
            if scene:
                scene.removeEventFilter(self)
        except:
            pass
        self._reset_state()
