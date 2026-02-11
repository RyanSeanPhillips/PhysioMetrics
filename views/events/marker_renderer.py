"""
Event marker renderer.

This module renders event markers on PyQtGraph plots as vertical lines
(single markers) or shaded regions (paired markers).
"""

from typing import Optional, List, Dict, Tuple, Callable
import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication

from viewmodels.event_marker_viewmodel import EventMarkerViewModel
from core.domain.events import EventMarker, MarkerType


class SyncedInfiniteLine(pg.InfiniteLine):
    """InfiniteLine that emits signals on hover for cross-panel sync."""

    hoverChanged = pyqtSignal(bool)  # Emits True on hover enter, False on leave

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._synced_lines: List['SyncedInfiniteLine'] = []
        self._is_synced_hover = False  # Track if hover is from sync vs actual

    def set_synced_lines(self, lines: List['SyncedInfiniteLine']):
        """Set the list of lines to sync hover state with."""
        self._synced_lines = [ln for ln in lines if ln is not self]

    def hoverEvent(self, ev):
        """Override to emit hover signal and sync across panels."""
        was_hovering = self.mouseHovering
        super().hoverEvent(ev)

        # Only emit/sync if this is a real hover change (not synced)
        if self.mouseHovering != was_hovering and not self._is_synced_hover:
            self.hoverChanged.emit(self.mouseHovering)
            # Sync hover state to other lines
            for line in self._synced_lines:
                line._set_synced_hover(self.mouseHovering)

    def _set_synced_hover(self, hovering: bool):
        """Set hover state from sync (without triggering further sync)."""
        if self.mouseHovering != hovering:
            self._is_synced_hover = True
            self.mouseHovering = hovering
            self.currentPen = self.hoverPen if hovering else self.pen
            self.update()
            self._is_synced_hover = False


class SmartLinearRegionItem(pg.LinearRegionItem):
    """
    LinearRegionItem with improved edge grabbing and Ctrl modifier support.

    Behavior:
    - Normal drag in fill area: moves both edges together (default behavior)
    - Ctrl+drag in fill area: moves only the nearest edge
    - Drag on edge lines: always moves just that edge

    This gives users precise control when they need it while preserving
    the convenient "move both" default behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctrl_drag_mode = False  # True when Ctrl is held during drag
        self._drag_edge_index = None  # Which edge we're dragging (0 or 1)
        self._drag_start_pos = None   # Initial mouse position
        self._drag_start_region = None  # Initial region bounds

    def mouseDragEvent(self, ev):
        """Handle mouse drag with Ctrl modifier for single-edge mode."""
        # Check if Ctrl is held
        ctrl_held = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if ev.isStart():
            # Starting a drag - determine mode
            self._drag_start_pos = ev.buttonDownPos()
            self._drag_start_region = list(self.getRegion())

            if ctrl_held:
                # Ctrl+drag: move only nearest edge
                self._ctrl_drag_mode = True
                # Determine which edge is closer
                click_x = self._drag_start_pos.x()
                dist_to_start = abs(click_x - self._drag_start_region[0])
                dist_to_end = abs(click_x - self._drag_start_region[1])
                self._drag_edge_index = 0 if dist_to_start < dist_to_end else 1
            else:
                # Normal drag: move both edges
                self._ctrl_drag_mode = False
                self._drag_edge_index = None

        if self._ctrl_drag_mode and self._drag_edge_index is not None:
            # Single-edge drag mode
            if ev.isFinish():
                self._ctrl_drag_mode = False
                self._drag_edge_index = None
                self.sigRegionChangeFinished.emit(self)
            else:
                # Calculate new position for just the dragged edge
                delta = ev.pos().x() - self._drag_start_pos.x()
                new_region = list(self._drag_start_region)
                new_region[self._drag_edge_index] += delta

                # Ensure edges don't cross (maintain order)
                if new_region[0] > new_region[1]:
                    if self._drag_edge_index == 0:
                        new_region[0] = new_region[1]
                    else:
                        new_region[1] = new_region[0]

                self.setRegion(new_region)
            ev.accept()
        else:
            # Normal drag behavior - delegate to parent
            super().mouseDragEvent(ev)


class MarkerRenderer:
    """
    Renders event markers on PyQtGraph plots.

    Manages the visual elements (InfiniteLines, LinearRegionItems) for
    displaying markers, and updates them when markers change.
    Supports rendering on multiple panels simultaneously.
    """

    # Signals for marker interaction
    marker_clicked = None  # Will be set by plot_integration
    marker_drag_started = None
    marker_drag_finished = None

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        plot_item: pg.PlotItem,
        get_all_plots: Optional[Callable] = None,
        on_marker_clicked: Optional[Callable] = None,
        on_marker_dragged: Optional[Callable] = None,
        get_signal_data: Optional[Callable] = None,
        get_sweep_idx: Optional[Callable] = None,
    ):
        """
        Initialize the renderer.

        Args:
            viewmodel: The event marker view model
            plot_item: The main PyQtGraph PlotItem (for reference)
            get_all_plots: Optional callback to get all plot panels for multi-panel rendering
            on_marker_clicked: Callback when marker is clicked: fn(marker_id)
            on_marker_dragged: Callback when marker is dragged: fn(marker_id, new_pos)
            get_signal_data: Callback to get signal data: fn(channel_name, sweep_idx) -> (t, y)
            get_sweep_idx: Callback to get current sweep index: fn() -> int
        """
        self._viewmodel = viewmodel
        self._plot_item = plot_item
        self._get_all_plots = get_all_plots
        self._on_marker_clicked = on_marker_clicked
        self._on_marker_dragged = on_marker_dragged
        self._get_signal_data = get_signal_data
        self._get_sweep_idx = get_sweep_idx

        # Track rendered elements by marker ID (each contains list of items for multi-panel)
        self._single_lines: Dict[str, List[pg.InfiniteLine]] = {}
        self._paired_regions: Dict[str, List[pg.LinearRegionItem]] = {}
        self._selection_highlights: Dict[str, List[pg.InfiniteLine]] = {}

        # Track marker labels (text items showing position/duration info)
        # Only shown on top plot panel to avoid clutter
        self._marker_labels: Dict[str, pg.TextItem] = {}
        self._marker_numbers: Dict[str, int] = {}  # Marker ID -> sequential number
        self._next_marker_number = 1

        # Time offset for display (when stimulus channel normalizes time to stim onset)
        # Markers are stored in absolute time, display time = absolute - offset
        self._time_offset: float = 0.0

        # Horizontal intersection lines (shown during drag)
        # Maps marker_id -> (plot, h_line, circle_marker) for the currently dragged marker
        self._intersection_items: Dict[str, Tuple] = {}

        # Derivative overlay (shown during drag when Shift is held or toggle is on)
        self._derivative_overlay: Optional[pg.PlotDataItem] = None
        self._derivative_plot: Optional[pg.PlotItem] = None
        self._show_derivative_on_drag: bool = False  # Persistent toggle from menu
        self._derivative_smooth_window: int = 51  # Smoothing window for derivative

        # Connect to viewmodel signals
        self._viewmodel.markers_changed.connect(self._on_markers_changed)
        self._viewmodel.selection_changed.connect(self._on_selection_changed)

    def _get_plots(self) -> List[pg.PlotItem]:
        """Get all plots to render on."""
        if self._get_all_plots:
            plots = self._get_all_plots()
            if plots:
                return plots
        return [self._plot_item] if self._plot_item else []

    def _get_channel_for_plot(self, plot: pg.PlotItem) -> Optional[str]:
        """Get the channel name from a plot's Y-axis label.

        Note: Returns the base channel name, stripping any scale factor suffix.
        For scale factor handling, use _get_channel_and_scale_for_plot().
        """
        try:
            left_axis = plot.getAxis('left')
            if left_axis and hasattr(left_axis, 'labelText'):
                label = left_axis.labelText
                # Strip scale factor suffix like "(X0.001)" or "(x0.001)"
                if label and '(' in label:
                    label = label.split('(')[0].strip()
                return label
        except Exception:
            pass
        return None

    def _get_channel_and_scale_for_plot(self, plot: pg.PlotItem) -> Tuple[Optional[str], float]:
        """Get channel name and scale factor from a plot's Y-axis label.

        Returns:
            Tuple of (channel_name, scale_factor). Scale factor defaults to 1.0.
            Example: "AI-0 (X0.001)" returns ("AI-0", 0.001)
        """
        import re
        try:
            left_axis = plot.getAxis('left')
            if left_axis and hasattr(left_axis, 'labelText'):
                label = left_axis.labelText
                if not label:
                    return None, 1.0

                # Look for scale factor pattern like "(X0.001)" or "(x0.001)"
                # This handles explicit scale annotations in channel names
                scale_match = re.search(r'\(X?([\d.]+)\)', label, re.IGNORECASE)
                if scale_match:
                    scale = float(scale_match.group(1))
                    channel_name = label.split('(')[0].strip()
                    return channel_name, scale
                else:
                    return label, 1.0
        except Exception:
            pass
        return None, 1.0

    def _get_plotted_data_from_plot(self, plot: pg.PlotItem) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the main trace data directly from a plot's data items.

        This extracts the actual plotted data (including any downsampling/transforms),
        ensuring the intersection Y value matches what's visually displayed.

        Args:
            plot: The PlotItem to extract data from

        Returns:
            Tuple of (x_data, y_data) or (None, None) if no data found
        """
        try:
            # Get all data items on the plot
            data_items = plot.listDataItems()
            if not data_items:
                return None, None

            # Find the main trace - it's typically a PlotDataItem (line plot)
            # Skip ScatterPlotItem (markers) and other items
            for item in data_items:
                # Check if it's a line plot (PlotDataItem or PlotCurveItem)
                if hasattr(item, 'getData'):
                    x_data, y_data = item.getData()
                    # Skip items with no data or scatter plots (which have small data arrays)
                    if x_data is not None and y_data is not None:
                        # Main trace typically has many points (>100)
                        # Scatter plots (peaks, markers) have fewer points
                        if len(x_data) > 100:
                            return x_data, y_data

            # Fallback: return first item with data if no large trace found
            for item in data_items:
                if hasattr(item, 'getData'):
                    x_data, y_data = item.getData()
                    if x_data is not None and y_data is not None and len(x_data) > 0:
                        return x_data, y_data

        except Exception:
            pass
        return None, None

    def _draw_intersection_line(self, marker_id: str, plot: pg.PlotItem, x_display: float, color: QColor) -> None:
        """
        Draw a horizontal line at the Y value where the marker intersects the signal.

        Gets Y value directly from the plotted data items to ensure accuracy.

        Args:
            marker_id: ID of the marker being dragged
            plot: The plot to draw on
            x_display: X position in display coordinates
            color: Color for the intersection line
        """
        # Clear any existing intersection items for this marker
        self._clear_intersection_lines(marker_id)

        # Get data directly from the plot's data items (the actual displayed data)
        t_data, y_data = self._get_plotted_data_from_plot(plot)
        if t_data is None or y_data is None or len(t_data) == 0:
            return

        # Interpolate to find Y value at marker position
        # Note: x_display is already in display coordinates, same as t_data
        try:
            y_value = np.interp(x_display, t_data, y_data)
            if np.isnan(y_value):
                return
        except Exception:
            return

        # Create horizontal line
        h_pen = QPen(color)
        h_pen.setWidth(0)  # Cosmetic pen
        h_pen.setStyle(Qt.PenStyle.DashLine)

        h_line = pg.InfiniteLine(
            pos=y_value,
            angle=0,  # Horizontal
            pen=h_pen,
            movable=False,
        )
        h_line.setZValue(1050)  # Above markers but below labels
        plot.addItem(h_line, ignoreBounds=True)

        # Create small circle at intersection point
        circle = pg.ScatterPlotItem(
            [x_display], [y_value],
            size=8,
            pen=pg.mkPen(color, width=1),
            brush=pg.mkBrush(color.lighter(150)),
            symbol='o',
        )
        circle.setZValue(1060)
        plot.addItem(circle, ignoreBounds=True)

        # Store references for cleanup
        self._intersection_items[marker_id] = (plot, h_line, circle)

    def _clear_intersection_lines(self, marker_id: Optional[str] = None) -> None:
        """
        Clear intersection lines.

        Args:
            marker_id: If provided, only clear lines for this marker.
                      If None, clear all intersection lines.
        """
        def safe_remove_item(plot, item):
            """Safely remove an item, checking if it's still in a valid scene."""
            if item is None:
                return
            try:
                # Check if item is still in a scene before removing
                scene = item.scene()
                if scene is not None:
                    plot.removeItem(item)
            except (RuntimeError, AttributeError):
                # Item may have been deleted or plot is no longer valid
                pass

        if marker_id is not None:
            if marker_id in self._intersection_items:
                stored = self._intersection_items[marker_id]
                plot = stored[0]
                items = stored[1] if isinstance(stored[1], list) else [stored[1], stored[2] if len(stored) > 2 else None]
                for item in items:
                    safe_remove_item(plot, item)
                del self._intersection_items[marker_id]
        else:
            # Clear all
            for mid, stored in list(self._intersection_items.items()):
                plot = stored[0]
                items = stored[1] if isinstance(stored[1], list) else [stored[1], stored[2] if len(stored) > 2 else None]
                for item in items:
                    safe_remove_item(plot, item)
            self._intersection_items.clear()

    def set_show_derivative_on_drag(self, enabled: bool) -> None:
        """Set whether to show derivative overlay during drag (menu toggle)."""
        self._show_derivative_on_drag = enabled

    def get_show_derivative_on_drag(self) -> bool:
        """Get current state of derivative overlay toggle."""
        return self._show_derivative_on_drag

    def _should_show_derivative(self) -> bool:
        """Check if derivative should be shown (Shift held or toggle enabled)."""
        if self._show_derivative_on_drag:
            return True
        # Check if Shift is held
        modifiers = QApplication.keyboardModifiers()
        return bool(modifiers & Qt.KeyboardModifier.ShiftModifier)

    def _draw_derivative_overlay(self, plot: pg.PlotItem) -> None:
        """
        Draw derivative overlay on the given plot, scaled to fit Y range.

        The derivative is calculated from the plotted data and scaled to fit
        within the current Y axis range for visual comparison.
        """
        # Clear any existing overlay
        self._clear_derivative_overlay()

        # Get data from plot
        t_data, y_data = self._get_plotted_data_from_plot(plot)
        if t_data is None or y_data is None or len(t_data) < 100:
            return

        # Calculate smoothed derivative
        try:
            from scipy.signal import savgol_filter

            window = self._derivative_smooth_window
            if window % 2 == 0:
                window += 1
            if window >= len(y_data):
                window = len(y_data) - 1 if len(y_data) % 2 == 0 else len(y_data) - 2
            if window < 5:
                return

            dt = t_data[1] - t_data[0] if len(t_data) > 1 else 1.0
            derivative = savgol_filter(y_data, window, polyorder=3, deriv=1, delta=dt)
        except Exception:
            # Fallback to simple gradient
            dt = t_data[1] - t_data[0] if len(t_data) > 1 else 1.0
            derivative = np.gradient(y_data, dt)

        # Scale derivative to fit current Y axis range
        view_range = plot.vb.viewRange()
        y_min, y_max = view_range[1]
        y_range = y_max - y_min

        # Use middle 60% of Y range for derivative to avoid clutter at edges
        deriv_y_min = y_min + y_range * 0.2
        deriv_y_max = y_max - y_range * 0.2
        deriv_range = deriv_y_max - deriv_y_min

        # Normalize derivative to fit this range
        deriv_min = np.nanmin(derivative)
        deriv_max = np.nanmax(derivative)
        deriv_data_range = deriv_max - deriv_min

        if deriv_data_range > 0:
            scaled_derivative = deriv_y_min + (derivative - deriv_min) / deriv_data_range * deriv_range
        else:
            scaled_derivative = np.full_like(derivative, (deriv_y_min + deriv_y_max) / 2)

        # Create overlay
        pen = pg.mkPen(color='#ff7043', width=1, style=Qt.PenStyle.SolidLine)
        self._derivative_overlay = pg.PlotDataItem(
            t_data, scaled_derivative,
            pen=pen,
            antialias=True,
        )
        self._derivative_overlay.setZValue(950)  # Below markers but above signal
        self._derivative_overlay.setOpacity(0.7)

        plot.addItem(self._derivative_overlay, ignoreBounds=True)
        self._derivative_plot = plot

    def _clear_derivative_overlay(self) -> None:
        """Clear the derivative overlay from the plot."""
        if self._derivative_overlay is not None and self._derivative_plot is not None:
            try:
                if self._derivative_overlay.scene() is not None:
                    self._derivative_plot.removeItem(self._derivative_overlay)
            except (RuntimeError, AttributeError):
                pass
        self._derivative_overlay = None
        self._derivative_plot = None

    def _find_plot_for_line(self, line: pg.InfiniteLine) -> Optional[pg.PlotItem]:
        """Find which plot a line belongs to by checking parent ViewBox."""
        try:
            # Get the line's parent ViewBox
            parent = line.parentItem()
            if parent is not None:
                for plot in self._get_plots():
                    if plot.vb is parent:
                        return plot

            # Fallback: check scene containment
            line_pos = line.scenePos()
            for plot in self._get_plots():
                if plot.sceneBoundingRect().contains(line_pos):
                    return plot
        except Exception:
            pass
        return None

    def _get_plot_under_mouse(self) -> Optional[pg.PlotItem]:
        """Find which plot the mouse cursor is currently over.

        This is more reliable than finding the plot for a line/region during drag,
        especially when the mouse moves between panels.
        """
        from PyQt6.QtGui import QCursor

        try:
            cursor_pos = QCursor.pos()
            plots = self._get_plots()

            for plot in plots:
                scene = plot.scene()
                if scene is None:
                    continue
                views = scene.views()
                if not views:
                    continue
                view = views[0]

                # Convert global cursor position to scene coordinates
                view_pos = view.mapFromGlobal(cursor_pos)
                scene_pos = view.mapToScene(view_pos)

                if plot.sceneBoundingRect().contains(scene_pos):
                    return plot
        except Exception:
            pass

        return None

    def _find_plot_for_region(self, region: pg.LinearRegionItem) -> Optional[pg.PlotItem]:
        """Find which plot a region belongs to by checking parent ViewBox."""
        try:
            plots = self._get_plots()
            if not plots:
                return None

            # Method 1: Check parent ViewBox identity
            parent = region.parentItem()
            if parent is not None:
                for plot in plots:
                    if plot.vb is parent:
                        return plot

            # Method 2: Check via region's internal edge lines
            # LinearRegionItem has internal lines that might have clearer parent info
            if hasattr(region, 'lines') and region.lines:
                for edge_line in region.lines:
                    line_parent = edge_line.parentItem()
                    if line_parent is not None:
                        for plot in plots:
                            if plot.vb is line_parent:
                                return plot

            # Method 3: Use scene position of one of the region's edge lines
            # This is more reliable than using the region's bounding rect center
            if hasattr(region, 'lines') and region.lines:
                edge_line = region.lines[0]
                line_scene_pos = edge_line.scenePos()
                for plot in plots:
                    if plot.sceneBoundingRect().contains(line_scene_pos):
                        return plot

            # Method 4: Use the center of the region's scene bounding rect
            region_rect = region.sceneBoundingRect()
            if region_rect.isValid():
                center = region_rect.center()
                for plot in plots:
                    plot_rect = plot.sceneBoundingRect()
                    if plot_rect.contains(center):
                        return plot

            # Method 5: Check if any part of region overlaps with any plot
            region_rect = region.sceneBoundingRect()
            if region_rect.isValid():
                for plot in plots:
                    plot_rect = plot.sceneBoundingRect()
                    if plot_rect.intersects(region_rect):
                        return plot

        except Exception:
            pass
        return None

    def _draw_intersection_lines_for_region(self, marker_id: str, plot: pg.PlotItem,
                                            start_x: float, end_x: float, color: QColor) -> None:
        """
        Draw horizontal lines at both edges of a paired marker region.

        Gets Y values directly from the plotted data items to ensure accuracy.

        Args:
            marker_id: ID of the marker being dragged
            plot: The plot to draw on
            start_x: Start X position in display coordinates
            end_x: End X position in display coordinates
            color: Color for the intersection lines
        """
        # Clear any existing intersection items for this marker
        self._clear_intersection_lines(marker_id)

        # Get data directly from the plot's data items (the actual displayed data)
        t_data, y_data = self._get_plotted_data_from_plot(plot)
        if t_data is None or y_data is None or len(t_data) == 0:
            return

        items_to_store = []

        # Draw intersection line for start edge
        # Note: start_x is already in display coordinates, same as t_data
        try:
            y_start = np.interp(start_x, t_data, y_data)
            if not np.isnan(y_start):
                # Horizontal line for start
                h_pen_start = QPen(color)
                h_pen_start.setWidth(0)
                h_pen_start.setStyle(Qt.PenStyle.DashLine)
                h_line_start = pg.InfiniteLine(pos=y_start, angle=0, pen=h_pen_start, movable=False)
                h_line_start.setZValue(1050)
                plot.addItem(h_line_start, ignoreBounds=True)

                # Circle at start intersection
                circle_start = pg.ScatterPlotItem(
                    [start_x], [y_start], size=8,
                    pen=pg.mkPen(color, width=1), brush=pg.mkBrush(color.lighter(150)), symbol='o'
                )
                circle_start.setZValue(1060)
                plot.addItem(circle_start, ignoreBounds=True)
                items_to_store.extend([h_line_start, circle_start])
        except Exception:
            pass

        # Draw intersection line for end edge
        try:
            y_end = np.interp(end_x, t_data, y_data)
            if not np.isnan(y_end):
                # Horizontal line for end
                h_pen_end = QPen(color)
                h_pen_end.setWidth(0)
                h_pen_end.setStyle(Qt.PenStyle.DashLine)
                h_line_end = pg.InfiniteLine(pos=y_end, angle=0, pen=h_pen_end, movable=False)
                h_line_end.setZValue(1050)
                plot.addItem(h_line_end, ignoreBounds=True)

                # Circle at end intersection
                circle_end = pg.ScatterPlotItem(
                    [end_x], [y_end], size=8,
                    pen=pg.mkPen(color, width=1), brush=pg.mkBrush(color.lighter(150)), symbol='o'
                )
                circle_end.setZValue(1060)
                plot.addItem(circle_end, ignoreBounds=True)
                items_to_store.extend([h_line_end, circle_end])
        except Exception:
            pass

        # Store all items for cleanup (store as tuple with plot reference)
        if items_to_store:
            self._intersection_items[marker_id] = (plot, items_to_store)

    def render_markers(self, sweep_idx: int, time_offset: float = 0.0) -> None:
        """
        Render all markers for a specific sweep.

        Args:
            sweep_idx: Sweep index to render markers for
            time_offset: Time offset in seconds to subtract from marker times for display.
                        When stimulus channel is active, plots are normalized so t=0 is
                        at first stim onset. Markers are stored in absolute time, so we
                        subtract this offset when displaying.
        """
        self.clear()
        self._time_offset = time_offset

        if not self._viewmodel.show_markers:
            return

        # Verify plot is ready
        if self._plot_item is None:
            print(f"[MarkerRenderer] ERROR: plot_item is None!")
            return

        if not hasattr(self._plot_item, 'vb') or self._plot_item.vb is None:
            print(f"[MarkerRenderer] ERROR: plot_item has no ViewBox!")
            return

        # Reset marker numbering for fresh render
        self._marker_numbers.clear()
        self._next_marker_number = 1

        markers = self._viewmodel.get_markers_for_sweep(sweep_idx)

        # Sort markers by start time for consistent numbering
        sorted_markers = sorted(markers, key=lambda m: m.start_time)

        for marker in sorted_markers:
            self._render_marker(marker)

        # Update selection highlights
        self._update_selection_highlights()

    def clear(self) -> None:
        """Remove all rendered markers from all plots."""
        plots = self._get_plots()

        def safe_remove(plot, item):
            """Only remove item if it's still in a valid scene."""
            try:
                if item is not None and item.scene() is not None:
                    plot.removeItem(item)
            except (RuntimeError, AttributeError):
                pass

        for lines in self._single_lines.values():
            for line in lines:
                for plot in plots:
                    safe_remove(plot, line)
        self._single_lines.clear()

        for regions in self._paired_regions.values():
            for region in regions:
                for plot in plots:
                    safe_remove(plot, region)
        self._paired_regions.clear()

        for highlights in self._selection_highlights.values():
            for highlight in highlights:
                for plot in plots:
                    safe_remove(plot, highlight)
        self._selection_highlights.clear()

        # Remove marker labels (added to ViewBox, not PlotItem)
        for label in self._marker_labels.values():
            for plot in plots:
                try:
                    if label is not None and label.scene() is not None:
                        plot.vb.removeItem(label)
                except (RuntimeError, AttributeError):
                    pass
                safe_remove(plot, label)
        self._marker_labels.clear()

        # NOTE: Do NOT clear intersection lines here - they should persist during drag
        # They are cleared explicitly when drag ends in _handle_region_moved/_handle_line_moved

    def _render_marker(self, marker: EventMarker) -> None:
        """Render a single marker."""
        color = self._viewmodel.get_color_for_marker(marker)
        qcolor = QColor(color)

        if marker.marker_type == MarkerType.SINGLE:
            self._render_single_marker(marker, qcolor)
        else:
            self._render_paired_marker(marker, qcolor)

    def _render_single_marker(self, marker: EventMarker, color: QColor) -> None:
        """Render a single (point) marker as a vertical line on all panels."""
        plots = self._get_plots()
        lines = []

        # Assign marker number
        marker_num = self._next_marker_number
        self._marker_numbers[marker.id] = marker_num
        self._next_marker_number += 1

        # Calculate display time (subtract offset from absolute time)
        display_time = marker.start_time - self._time_offset

        # Determine line width: custom > default (1)
        base_width = marker.line_width if marker.line_width is not None else 1

        for plot in plots:
            pen = QPen(color)
            pen.setWidth(base_width)
            pen.setStyle(Qt.PenStyle.SolidLine)

            # Create hover pen - brighter color, noticeably thicker
            hover_color = color.lighter(150)
            hover_pen = QPen(hover_color)
            hover_pen.setWidth(max(base_width + 2, 3))  # At least 3px on hover

            line = SyncedInfiniteLine(
                pos=display_time,
                angle=90,
                pen=pen,
                hoverPen=hover_pen,
                movable=True,  # All lines movable for any-panel dragging
                label=None,
            )
            line.setZValue(1000)  # High z-value to ensure visibility
            line.marker_id = marker.id

            # Connect signals for interaction on all lines
            # sigPositionChanged fires during drag for real-time sync
            line.sigPositionChanged.connect(
                lambda ln=line, mid=marker.id: self._sync_line_position(mid, ln)
            )
            # sigPositionChangeFinished fires at end to persist the change
            if self._on_marker_dragged:
                line.sigPositionChangeFinished.connect(
                    lambda ln=line, mid=marker.id: self._handle_line_moved(mid, ln)
                )

            try:
                plot.addItem(line)
                lines.append(line)
            except Exception as e:
                print(f"[MarkerRenderer] ERROR adding line: {e}")

        # Link all lines for synced hover
        for line in lines:
            line.set_synced_lines(lines)

        self._single_lines[marker.id] = lines

        # Add label on top plot only (to avoid clutter)
        if plots:
            top_plot = plots[0]
            # Use display time for the label text (what user sees on plot)
            label_text = self._format_single_label(marker_num, display_time)
            label = pg.TextItem(
                text=label_text,
                color=color,
                anchor=(0.5, 0),  # Anchor at top-center of text
            )
            label.setFont(pg.QtGui.QFont('Arial', 8))
            label.setZValue(1100)  # Above lines
            label.marker_id = marker.id

            # Add to ViewBox directly with ignoreBounds so it doesn't affect auto-range
            top_plot.vb.addItem(label, ignoreBounds=True)

            # Position at display time X, near top of current view (but won't expand range)
            view_range = top_plot.vb.viewRange()
            top_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.02
            label.setPos(display_time, top_y)

            self._marker_labels[marker.id] = label

    def _format_single_label(self, marker_num: int, time: float) -> str:
        """Format label text for a single marker."""
        return f"#{marker_num}\n{time:.3f}s"

    def _format_paired_label(self, marker_num: int, start: float, end: float) -> str:
        """Format label text for a paired marker."""
        duration = end - start
        return f"#{marker_num}\n{start:.3f}s\n\u0394{duration:.3f}s"

    def _sync_line_position(self, marker_id: str, line: pg.InfiniteLine) -> None:
        """Sync line position across all panels during drag (real-time)."""
        new_pos = line.value()
        if marker_id in self._single_lines:
            for other_line in self._single_lines[marker_id]:
                if other_line is not line:
                    other_line.blockSignals(True)
                    other_line.setValue(new_pos)
                    other_line.blockSignals(False)

        # Update label text and X position (keep Y unchanged to avoid range issues)
        if marker_id in self._marker_labels:
            label = self._marker_labels[marker_id]
            marker_num = self._marker_numbers.get(marker_id, 0)
            label.setText(self._format_single_label(marker_num, new_pos))
            # Only update X position, keep Y where it was
            current_pos = label.pos()
            label.setPos(new_pos, current_pos.y())

        # Draw horizontal intersection line on the panel being dragged
        plot = self._find_plot_for_line(line)
        if plot:
            # Get marker color for intersection line
            marker = self._viewmodel.store.get(marker_id)
            if marker:
                color = QColor(self._viewmodel.get_color_for_marker(marker))
                self._draw_intersection_line(marker_id, plot, new_pos, color)

        # Draw derivative overlay if Shift held or toggle enabled
        # Use mouse position to determine which panel (more reliable during cross-panel drag)
        if self._should_show_derivative():
            deriv_plot = self._get_plot_under_mouse()
            if deriv_plot:
                self._draw_derivative_overlay(deriv_plot)
        else:
            self._clear_derivative_overlay()

    def _handle_line_moved(self, marker_id: str, line: pg.InfiniteLine) -> None:
        """Handle when a marker line drag finishes - persist the change."""
        new_pos_display = line.value()
        # Final sync (in case any were missed)
        self._sync_line_position(marker_id, line)
        # Clear intersection lines and derivative overlay
        self._clear_intersection_lines(marker_id)
        self._clear_derivative_overlay()
        # Convert display position back to absolute time for storage
        new_pos_absolute = new_pos_display + self._time_offset
        # Notify callback to persist the change
        if self._on_marker_dragged:
            self._on_marker_dragged(marker_id, new_pos_absolute, None)

    def _render_paired_marker(self, marker: EventMarker, color: QColor) -> None:
        """Render a paired (region) marker as a shaded region on all panels."""
        plots = self._get_plots()
        regions = []

        # Assign marker number
        marker_num = self._next_marker_number
        self._marker_numbers[marker.id] = marker_num
        self._next_marker_number += 1

        end_time = marker.end_time or marker.start_time + 0.1

        # Calculate display times (subtract offset from absolute times)
        display_start = marker.start_time - self._time_offset
        display_end = end_time - self._time_offset

        # Determine edge line width: custom > default (2)
        edge_width = marker.line_width if marker.line_width is not None else 2

        for plot in plots:
            # Create brush with alpha for fill
            fill_color = QColor(color)
            fill_color.setAlpha(30)  # Light fill so edges are visible
            brush = QBrush(fill_color)

            # Edge pen with configurable width
            pen = QPen(color)
            pen.setWidth(edge_width)

            # Create hover brush (slightly brighter fill)
            hover_color = QColor(color.lighter(130))
            hover_color.setAlpha(50)
            hover_brush = QBrush(hover_color)

            region = SmartLinearRegionItem(
                values=[display_start, display_end],
                orientation='vertical',
                brush=brush,
                pen=pen,
                movable=True,  # All regions movable for any-panel dragging
                hoverBrush=hover_brush,
                swapMode='none',  # Prevent edges from crossing
            )
            region.setZValue(900)  # High z-value for visibility
            region.marker_id = marker.id

            # Customize the internal edge lines' hover pen to match single markers
            # and increase the grab tolerance (hoverable width around the line)
            for edge_line in region.lines:
                # Wider + brighter hover pen for clear visual feedback
                hover_color = color.lighter(150)
                wide_hover_pen = QPen(hover_color)
                wide_hover_pen.setWidth(max(edge_width + 2, 4))  # Noticeably wider on hover
                edge_line.setHoverPen(wide_hover_pen)
                # Edge pen with configurable width
                edge_pen = QPen(color)
                edge_pen.setWidth(edge_width)
                edge_line.setPen(edge_pen)
                # Increase the hover/grab tolerance from default ~5px to 20px
                # This makes it much easier to grab the edges without being pixel-perfect
                if hasattr(edge_line, 'span'):
                    edge_line.span = (0, 1)  # Full span
                # Set a wider hover bounds - this is the key to easier grabbing
                edge_line.setBounds([None, None])
                # PyQtGraph InfiniteLine uses _maxMarkerWidth for hover detection
                edge_line._maxMarkerWidth = 20  # Increase from default ~5
                # Set hover cursor to indicate draggable
                edge_line.setCursor(Qt.CursorShape.SizeHorCursor)

            # Connect signals for interaction on all regions
            # sigRegionChanged fires during drag for real-time sync
            region.sigRegionChanged.connect(
                lambda rgn=region, mid=marker.id: self._sync_region_position(mid, rgn)
            )
            # sigRegionChangeFinished fires at end to persist the change
            if self._on_marker_dragged:
                region.sigRegionChangeFinished.connect(
                    lambda rgn=region, mid=marker.id: self._handle_region_moved(mid, rgn)
                )

            try:
                plot.addItem(region)
                regions.append(region)
            except Exception as e:
                print(f"[MarkerRenderer] ERROR adding region: {e}")

        # Set up hover sync for region edge lines across panels
        self._setup_region_hover_sync(regions)

        self._paired_regions[marker.id] = regions

        # Add label on top plot only (to avoid clutter)
        if plots:
            top_plot = plots[0]
            # Use display times for the label text (what user sees on plot)
            label_text = self._format_paired_label(marker_num, display_start, display_end)
            label = pg.TextItem(
                text=label_text,
                color=color,
                anchor=(0.5, 0),  # Anchor at top-center of text
            )
            label.setFont(pg.QtGui.QFont('Arial', 8))
            label.setZValue(1100)  # Above regions
            label.marker_id = marker.id

            # Add to ViewBox directly with ignoreBounds so it doesn't affect auto-range
            top_plot.vb.addItem(label, ignoreBounds=True)

            # Position at center of region (in display coordinates), near top of current view
            view_range = top_plot.vb.viewRange()
            top_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.02
            center_x = (display_start + display_end) / 2
            label.setPos(center_x, top_y)

            self._marker_labels[marker.id] = label

    def _setup_region_hover_sync(self, regions: List[pg.LinearRegionItem]) -> None:
        """Set up hover synchronization for regions and edge lines across panels."""
        # Collect all edge lines from all regions
        all_edge_lines = []
        for region in regions:
            all_edge_lines.extend(region.lines)

        # Install hover event override on each edge line
        for i, region in enumerate(regions):
            for j, edge_line in enumerate(region.lines):
                # Store reference to other lines for syncing
                edge_line._other_edge_lines = [
                    other_region.lines[j] for other_region in regions if other_region is not region
                ]
                # Override hoverEvent
                original_hover = edge_line.hoverEvent
                def make_synced_hover(line, orig_hover):
                    def synced_hover(ev):
                        was_hovering = line.mouseHovering
                        orig_hover(ev)
                        if line.mouseHovering != was_hovering:
                            for other in line._other_edge_lines:
                                other.mouseHovering = line.mouseHovering
                                other.currentPen = other.hoverPen if line.mouseHovering else other.pen
                                other.update()
                    return synced_hover
                edge_line.hoverEvent = make_synced_hover(edge_line, original_hover)

        # Install hover event override on each region for fill area hover sync
        for region in regions:
            # Store reference to other regions
            region._synced_regions = [r for r in regions if r is not region]
            region._is_synced_hover = False  # Track if hover is from sync vs actual
            region._normal_brush = region.brush  # Store original brush for restoration

            # Override hoverEvent on the region
            original_region_hover = region.hoverEvent

            def make_region_synced_hover(rgn, orig_hover):
                def synced_region_hover(ev):
                    # Check hover state before calling original
                    was_hovering = getattr(rgn, '_region_hovering', False)

                    # Call original hover handler
                    orig_hover(ev)

                    # Determine new hover state by checking if event is enter or exit
                    is_hovering = False
                    if ev.isEnter():
                        is_hovering = True
                    elif ev.isExit():
                        is_hovering = False
                    else:
                        # For move events, maintain current state
                        is_hovering = was_hovering

                    rgn._region_hovering = is_hovering

                    # Only sync if this is a real hover change (not synced)
                    if is_hovering != was_hovering and not rgn._is_synced_hover:
                        for other in rgn._synced_regions:
                            other._is_synced_hover = True
                            other._region_hovering = is_hovering
                            # Set brush directly using stored references
                            if is_hovering:
                                other.setBrush(other.hoverBrush)
                            else:
                                other.setBrush(other._normal_brush)
                            other.update()
                            other._is_synced_hover = False

                return synced_region_hover

            region.hoverEvent = make_region_synced_hover(region, original_region_hover)

    def _sync_region_position(self, marker_id: str, region: pg.LinearRegionItem) -> None:
        """Sync region position across all panels during drag (real-time)."""
        new_start, new_end = region.getRegion()
        if marker_id in self._paired_regions:
            for other_region in self._paired_regions[marker_id]:
                if other_region is not region:
                    other_region.blockSignals(True)
                    other_region.setRegion([new_start, new_end])
                    other_region.blockSignals(False)

        # Update label text and X position (keep Y unchanged to avoid range issues)
        if marker_id in self._marker_labels:
            label = self._marker_labels[marker_id]
            marker_num = self._marker_numbers.get(marker_id, 0)
            label.setText(self._format_paired_label(marker_num, new_start, new_end))
            # Position at center of region, keep Y where it was
            current_pos = label.pos()
            center_x = (new_start + new_end) / 2
            label.setPos(center_x, current_pos.y())

        # Draw horizontal intersection lines on the panel being dragged
        plot = self._find_plot_for_region(region)
        if plot:
            # Get marker color for intersection lines
            marker = self._viewmodel.store.get(marker_id)
            if marker:
                color = QColor(self._viewmodel.get_color_for_marker(marker))
                self._draw_intersection_lines_for_region(marker_id, plot, new_start, new_end, color)

        # Draw derivative overlay if Shift held or toggle enabled
        # Use mouse position to determine which panel (more reliable during cross-panel drag)
        if self._should_show_derivative():
            deriv_plot = self._get_plot_under_mouse()
            if deriv_plot:
                self._draw_derivative_overlay(deriv_plot)
        else:
            self._clear_derivative_overlay()

    def _handle_region_moved(self, marker_id: str, region: pg.LinearRegionItem) -> None:
        """Handle when a region marker drag finishes - persist the change."""
        new_start_display, new_end_display = region.getRegion()
        # Final sync (in case any were missed)
        self._sync_region_position(marker_id, region)
        # Clear intersection lines and derivative overlay
        self._clear_intersection_lines(marker_id)
        self._clear_derivative_overlay()
        # Convert display positions back to absolute time for storage
        new_start_absolute = new_start_display + self._time_offset
        new_end_absolute = new_end_display + self._time_offset
        # Notify callback to persist the change
        if self._on_marker_dragged:
            self._on_marker_dragged(marker_id, new_start_absolute, new_end_absolute)

    def _update_selection_highlights(self) -> None:
        """Update selection highlights for selected markers on all panels."""
        plots = self._get_plots()

        # Remove old highlights
        for highlights in self._selection_highlights.values():
            for highlight in highlights:
                for plot in plots:
                    try:
                        plot.removeItem(highlight)
                    except:
                        pass
        self._selection_highlights.clear()

        # Add highlights for selected markers
        selected_ids = self._viewmodel.selected_ids
        for marker_id in selected_ids:
            marker = self._viewmodel.store.get(marker_id)
            if marker is None:
                continue

            if marker.marker_type == MarkerType.SINGLE:
                # Calculate display time
                display_time = marker.start_time - self._time_offset
                highlights = []
                for plot in plots:
                    highlight_color = QColor('#FFFFFF')
                    highlight_pen = QPen(highlight_color)
                    highlight_pen.setWidth(4)
                    highlight_pen.setStyle(Qt.PenStyle.DashLine)

                    line = pg.InfiniteLine(
                        pos=display_time,
                        angle=90,
                        pen=highlight_pen,
                        movable=False,
                    )
                    line.setZValue(1500)  # Above everything
                    plot.addItem(line)
                    highlights.append(line)
                self._selection_highlights[marker_id] = highlights
            else:
                # For paired markers, highlight the boundaries
                for time in [marker.start_time, marker.end_time]:
                    if time is not None:
                        # Calculate display time
                        display_time = time - self._time_offset
                        highlights = []
                        for plot in plots:
                            highlight_color = QColor('#FFFFFF')
                            highlight_pen = QPen(highlight_color)
                            highlight_pen.setWidth(4)
                            highlight_pen.setStyle(Qt.PenStyle.DashLine)

                            line = pg.InfiniteLine(
                                pos=display_time,
                                angle=90,
                                pen=highlight_pen,
                                movable=False,
                            )
                            line.setZValue(1500)
                            plot.addItem(line)
                            highlights.append(line)
                        key = f"{marker_id}_{time}"
                        self._selection_highlights[key] = highlights

    def _on_markers_changed(self) -> None:
        """Handle markers changed signal."""
        # Re-render for current sweep
        # The parent should call render_markers with the correct sweep
        pass

    def _on_selection_changed(self, selected_ids: List[str]) -> None:
        """Handle selection changed signal."""
        self._update_selection_highlights()

    def get_marker_at_position(
        self,
        x: float,
        tolerance: float = 0.05
    ) -> Optional[str]:
        """
        Find marker ID at a given x position.

        Args:
            x: X coordinate (time)
            tolerance: Tolerance for click detection

        Returns:
            Marker ID if found, None otherwise
        """
        # Check single markers first (they're smaller targets)
        for marker_id, lines in self._single_lines.items():
            if lines:  # Use first line for position check
                line_x = lines[0].value()
                if abs(x - line_x) <= tolerance:
                    return marker_id

        # Check paired markers (regions)
        for marker_id, regions in self._paired_regions.items():
            if regions:  # Use first region for bounds check
                bounds = regions[0].getRegion()
                # Check if click is near edges (for selection/dragging)
                if abs(x - bounds[0]) <= tolerance or abs(x - bounds[1]) <= tolerance:
                    return marker_id
                # Check if click is inside region
                if bounds[0] <= x <= bounds[1]:
                    return marker_id

        return None

    def update_marker_position(
        self,
        marker_id: str,
        new_start: float,
        new_end: Optional[float] = None
    ) -> None:
        """
        Update the visual position of a marker (for dragging preview).

        Args:
            marker_id: ID of marker to update
            new_start: New start position
            new_end: New end position (for paired markers)
        """
        if marker_id in self._single_lines:
            for line in self._single_lines[marker_id]:
                line.setValue(new_start)
        elif marker_id in self._paired_regions:
            end = new_end if new_end is not None else new_start + 0.1
            for region in self._paired_regions[marker_id]:
                region.setRegion([new_start, end])

        # Update selection highlight if this marker is selected
        if marker_id in self._viewmodel.selected_ids:
            self._update_selection_highlights()

    def set_marker_movable(self, marker_id: str, movable: bool) -> None:
        """
        Set whether a marker can be moved (for editing mode).

        Args:
            marker_id: ID of marker
            movable: Whether marker should be movable
        """
        if marker_id in self._single_lines:
            for line in self._single_lines[marker_id]:
                line.setMovable(movable)
        elif marker_id in self._paired_regions:
            for region in self._paired_regions[marker_id]:
                region.setMovable(movable)

    def get_all_marker_items(self) -> List[Tuple[str, object]]:
        """
        Get all rendered marker items (returns first item from each marker).

        Returns:
            List of (marker_id, graphics_item) tuples
        """
        items = []
        for marker_id, lines in self._single_lines.items():
            if lines:
                items.append((marker_id, lines[0]))
        for marker_id, regions in self._paired_regions.items():
            if regions:
                items.append((marker_id, regions[0]))
        return items

    def cleanup(self) -> None:
        """Clean up resources and disconnect signals."""
        self.clear()
        try:
            self._viewmodel.markers_changed.disconnect(self._on_markers_changed)
            self._viewmodel.selection_changed.disconnect(self._on_selection_changed)
        except (TypeError, RuntimeError):
            pass  # Already disconnected
