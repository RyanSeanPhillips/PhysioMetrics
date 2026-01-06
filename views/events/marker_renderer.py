"""
Event marker renderer.

This module renders event markers on PyQtGraph plots as vertical lines
(single markers) or shaded regions (paired markers).
"""

from typing import Optional, List, Dict, Tuple, Callable
import pyqtgraph as pg
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtCore import Qt, pyqtSignal

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
    ):
        """
        Initialize the renderer.

        Args:
            viewmodel: The event marker view model
            plot_item: The main PyQtGraph PlotItem (for reference)
            get_all_plots: Optional callback to get all plot panels for multi-panel rendering
            on_marker_clicked: Callback when marker is clicked: fn(marker_id)
            on_marker_dragged: Callback when marker is dragged: fn(marker_id, new_pos)
        """
        self._viewmodel = viewmodel
        self._plot_item = plot_item
        self._get_all_plots = get_all_plots
        self._on_marker_clicked = on_marker_clicked
        self._on_marker_dragged = on_marker_dragged

        # Track rendered elements by marker ID (each contains list of items for multi-panel)
        self._single_lines: Dict[str, List[pg.InfiniteLine]] = {}
        self._paired_regions: Dict[str, List[pg.LinearRegionItem]] = {}
        self._selection_highlights: Dict[str, List[pg.InfiniteLine]] = {}

        # Track marker labels (text items showing position/duration info)
        # Only shown on top plot panel to avoid clutter
        self._marker_labels: Dict[str, pg.TextItem] = {}
        self._marker_numbers: Dict[str, int] = {}  # Marker ID -> sequential number
        self._next_marker_number = 1

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

    def render_markers(self, sweep_idx: int) -> None:
        """
        Render all markers for a specific sweep.

        Args:
            sweep_idx: Sweep index to render markers for
        """
        self.clear()

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

        for lines in self._single_lines.values():
            for line in lines:
                for plot in plots:
                    try:
                        plot.removeItem(line)
                    except:
                        pass
        self._single_lines.clear()

        for regions in self._paired_regions.values():
            for region in regions:
                for plot in plots:
                    try:
                        plot.removeItem(region)
                    except:
                        pass
        self._paired_regions.clear()

        for highlights in self._selection_highlights.values():
            for highlight in highlights:
                for plot in plots:
                    try:
                        plot.removeItem(highlight)
                    except:
                        pass
        self._selection_highlights.clear()

        # Remove marker labels (added to ViewBox, not PlotItem)
        for label in self._marker_labels.values():
            for plot in plots:
                try:
                    plot.vb.removeItem(label)
                except:
                    pass
                try:
                    plot.removeItem(label)
                except:
                    pass
        self._marker_labels.clear()

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

        for plot in plots:
            # Use cosmetic pen (width 0) for 1-pixel line regardless of zoom
            pen = QPen(color)
            pen.setWidth(0)  # Cosmetic pen - always 1 pixel
            pen.setStyle(Qt.PenStyle.SolidLine)

            # Create hover pen - bright red, same thickness as normal
            hover_pen = QPen(QColor('#FF4444'))
            hover_pen.setWidth(0)  # Cosmetic pen - same as normal

            line = SyncedInfiniteLine(
                pos=marker.start_time,
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
            label_text = self._format_single_label(marker_num, marker.start_time)
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

            # Position at marker X, near top of current view (but won't expand range)
            view_range = top_plot.vb.viewRange()
            top_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.02
            label.setPos(marker.start_time, top_y)

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

    def _handle_line_moved(self, marker_id: str, line: pg.InfiniteLine) -> None:
        """Handle when a marker line drag finishes - persist the change."""
        new_pos = line.value()
        # Final sync (in case any were missed)
        self._sync_line_position(marker_id, line)
        # Notify callback to persist the change
        if self._on_marker_dragged:
            self._on_marker_dragged(marker_id, new_pos, None)

    def _render_paired_marker(self, marker: EventMarker, color: QColor) -> None:
        """Render a paired (region) marker as a shaded region on all panels."""
        plots = self._get_plots()
        regions = []

        # Assign marker number
        marker_num = self._next_marker_number
        self._marker_numbers[marker.id] = marker_num
        self._next_marker_number += 1

        end_time = marker.end_time or marker.start_time + 0.1

        # Create red hover pen for edge lines - same thickness as normal
        hover_pen = QPen(QColor('#FF4444'))
        hover_pen.setWidth(0)  # Cosmetic pen - same as normal

        for plot in plots:
            # Create brush with alpha for fill
            fill_color = QColor(color)
            fill_color.setAlpha(30)  # Light fill so edges are visible
            brush = QBrush(fill_color)

            # Use cosmetic pen for thin edge lines
            pen = QPen(color)
            pen.setWidth(0)  # Cosmetic pen - always 1 pixel

            # Create hover brush (slightly brighter fill)
            hover_color = QColor(color.lighter(130))
            hover_color.setAlpha(50)
            hover_brush = QBrush(hover_color)

            region = pg.LinearRegionItem(
                values=[marker.start_time, end_time],
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
            for edge_line in region.lines:
                edge_line.setHoverPen(hover_pen)

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
            label_text = self._format_paired_label(marker_num, marker.start_time, end_time)
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

            # Position at center of region, near top of current view
            view_range = top_plot.vb.viewRange()
            top_y = view_range[1][1] - (view_range[1][1] - view_range[1][0]) * 0.02
            center_x = (marker.start_time + end_time) / 2
            label.setPos(center_x, top_y)

            self._marker_labels[marker.id] = label

    def _setup_region_hover_sync(self, regions: List[pg.LinearRegionItem]) -> None:
        """Set up hover synchronization for region edge lines across panels."""
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

    def _handle_region_moved(self, marker_id: str, region: pg.LinearRegionItem) -> None:
        """Handle when a region marker drag finishes - persist the change."""
        new_start, new_end = region.getRegion()
        # Final sync (in case any were missed)
        self._sync_region_position(marker_id, region)
        # Notify callback to persist the change
        if self._on_marker_dragged:
            self._on_marker_dragged(marker_id, new_start, new_end)

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
                highlights = []
                for plot in plots:
                    highlight_color = QColor('#FFFFFF')
                    highlight_pen = QPen(highlight_color)
                    highlight_pen.setWidth(4)
                    highlight_pen.setStyle(Qt.PenStyle.DashLine)

                    line = pg.InfiniteLine(
                        pos=marker.start_time,
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
                        highlights = []
                        for plot in plots:
                            highlight_color = QColor('#FFFFFF')
                            highlight_pen = QPen(highlight_color)
                            highlight_pen.setWidth(4)
                            highlight_pen.setStyle(Qt.PenStyle.DashLine)

                            line = pg.InfiniteLine(
                                pos=time,
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
        tolerance: float = 0.02
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
