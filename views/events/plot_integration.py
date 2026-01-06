"""
Event marker plot integration.

This module integrates the event marker system with PyQtGraphPlotHost,
providing the glue between the marker viewmodel, renderer, editor,
and the existing plot backend.
"""

from typing import Optional, Callable, List, Tuple
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QCursor, QColor, QPen
import pyqtgraph as pg

from viewmodels.event_marker_viewmodel import EventMarkerViewModel
from .context_menu import EventMarkerContextMenu, MarkerContextMenu
from .marker_renderer import MarkerRenderer
from .marker_editor import MarkerEditor


class EventMarkerPlotIntegration(QObject):
    """
    Integrates the event marker system with a PyQtGraphPlotHost.

    This class:
    - Manages the MarkerRenderer for visual display
    - Manages the MarkerEditor for interactive editing
    - Provides context menu integration
    - Handles keyboard shortcuts (S+click, P+click)

    Usage:
        viewmodel = EventMarkerViewModel()
        integration = EventMarkerPlotIntegration(viewmodel, plot_host)
        integration.set_sweep_callback(lambda: current_sweep_idx)
        integration.enable()
    """

    # Signals
    marker_added = pyqtSignal(str)    # marker_id
    marker_deleted = pyqtSignal(str)  # marker_id
    markers_changed = pyqtSignal()

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        plot_host,  # PyQtGraphPlotHost
        parent: Optional[QObject] = None,
    ):
        """
        Initialize the integration.

        Args:
            viewmodel: The event marker view model
            plot_host: The PyQtGraphPlotHost to integrate with
            parent: Parent QObject
        """
        super().__init__(parent)

        self._viewmodel = viewmodel
        self._plot_host = plot_host
        self._enabled = False

        # Callbacks for getting current state
        self._get_sweep_idx: Callable[[], int] = lambda: 0
        self._get_visible_range: Callable[[], Tuple[float, float]] = lambda: (0.0, 10.0)
        self._get_channel_names: Callable[[], List[str]] = lambda: []
        self._get_signal_data: Optional[Callable[[str, int], Tuple[float, Optional]]] = None

        # Components (created on enable)
        self._renderer: Optional[MarkerRenderer] = None
        self._editor: Optional[MarkerEditor] = None

        # Keyboard state for shortcuts
        self._s_held = False
        self._p_held = False

        # Preview cursor lines (shown when S or P is held)
        self._preview_lines: List = []  # Start position lines
        self._preview_end_lines: List = []  # End position lines (paired mode only)
        self._preview_visible = False
        self._is_paired_preview = False  # Track if showing paired preview

        # Original context menu reference
        self._original_context_menu_fn = None

        # Connect viewmodel signals
        self._viewmodel.markers_changed.connect(self._on_markers_changed)

    def set_sweep_callback(self, fn: Callable[[], int]) -> None:
        """Set callback to get current sweep index."""
        self._get_sweep_idx = fn

    def set_visible_range_callback(self, fn: Callable[[], Tuple[float, float]]) -> None:
        """Set callback to get visible x-range."""
        self._get_visible_range = fn

    def set_channel_names_callback(self, fn: Callable[[], List[str]]) -> None:
        """Set callback to get available channel names."""
        self._get_channel_names = fn

    def set_signal_data_callback(self, fn: Callable) -> None:
        """
        Set callback to get signal data for a channel.

        The callback should return (sample_rate, signal_array) or (sample_rate, None).
        """
        self._get_signal_data = fn

    def enable(self) -> None:
        """Enable the event marker integration."""
        if self._enabled:
            return

        # Intercept context menu (works even before data is loaded)
        self._original_context_menu_fn = self._plot_host._show_context_menu
        self._plot_host._show_context_menu = self._show_enhanced_context_menu

        # Install keyboard handler at application level to capture S/P keys regardless of focus
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        # Also install on graphics layout for mouse events
        self._plot_host.graphics_layout.installEventFilter(self)

        self._enabled = True

        # Try to create renderer/editor if plot is ready
        self._ensure_renderer_editor()

    def _ensure_renderer_editor(self) -> bool:
        """Create renderer and editor if plot is ready. Returns True if ready."""
        # Get the main plot widget
        main_plot = self._plot_host._get_main_plot()
        if main_plot is None:
            return False

        # Check if we need to recreate renderer (plot reference may have changed)
        if self._renderer is not None:
            old_plot = self._renderer._plot_item
            if old_plot is not main_plot:
                self._renderer.cleanup()
                self._renderer = None
            else:
                # Same plot, renderer is still valid
                return True

        # Create renderer with callback to get all subplots for multi-panel rendering
        def get_all_plots():
            if hasattr(self._plot_host, 'get_subplots'):
                return self._plot_host.get_subplots()
            elif hasattr(self._plot_host, '_subplots'):
                return self._plot_host._subplots
            return [main_plot]

        self._renderer = MarkerRenderer(
            self._viewmodel,
            main_plot,
            get_all_plots=get_all_plots,
            on_marker_dragged=self._on_marker_drag_finished,
        )

        # Create editor if needed
        if self._editor is None:
            self._editor = MarkerEditor(
                viewmodel=self._viewmodel,
                plot_widget=self._plot_host.graphics_layout,
                get_sweep_idx=self._get_sweep_idx,
                get_visible_range=self._get_visible_range,
                plot_host=self._plot_host,
            )

            # Connect editor signals
            self._editor.marker_moved.connect(self._on_marker_moved)
            self._editor.marker_created.connect(self._on_marker_created)

        return True

    def disable(self) -> None:
        """Disable the event marker integration."""
        if not self._enabled:
            return

        # Hide and cleanup preview lines
        self._hide_preview()

        # Restore original context menu
        if self._original_context_menu_fn:
            self._plot_host._show_context_menu = self._original_context_menu_fn
            self._original_context_menu_fn = None

        # Remove event filters
        try:
            self._plot_host.graphics_layout.removeEventFilter(self)
        except:
            pass

        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self)
        except:
            pass

        # Cleanup renderer
        if self._renderer:
            self._renderer.cleanup()
            self._renderer = None

        # Cleanup editor
        if self._editor:
            self._editor.cleanup()
            self._editor = None

        self._enabled = False

    def refresh(self) -> None:
        """Refresh the marker display for the current sweep."""
        if not self._enabled:
            return

        # Ensure renderer exists (may not if called before first data load)
        if not self._ensure_renderer_editor():
            return

        self._renderer.render_markers(self._get_sweep_idx())

    def eventFilter(self, obj: QObject, event) -> bool:
        """Filter events for keyboard shortcuts and preview cursor."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QApplication

        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            # Check for S/P + click shortcuts
            if event.button() == Qt.MouseButton.LeftButton:
                s_pressed = self._s_held
                p_pressed = self._p_held

                if s_pressed or p_pressed:
                    # Hide preview and create actual marker
                    self._hide_preview()
                    return self._handle_shortcut_click(event, s_pressed, p_pressed)

        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
            # Update preview position when S or P is held
            if self._s_held or self._p_held:
                self._update_preview_position(event.scenePos())

        elif event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_S and not self._s_held:
                self._s_held = True
                try:
                    self._show_preview(is_paired=False)
                except Exception as e:
                    print(f"[EventMarkerPlotIntegration] Preview error: {e}")
                    self._s_held = False  # Reset on error
            elif key == Qt.Key.Key_P and not self._p_held:
                self._p_held = True
                try:
                    self._show_preview(is_paired=True)
                except Exception as e:
                    print(f"[EventMarkerPlotIntegration] Preview error: {e}")
                    self._p_held = False  # Reset on error

        elif event.type() == QEvent.Type.KeyRelease:
            key = event.key()
            if key == Qt.Key.Key_S:
                self._s_held = False
                self._hide_preview()
            elif key == Qt.Key.Key_P:
                self._p_held = False
                self._hide_preview()

        return False

    def _show_preview(self, is_paired: bool = False) -> None:
        """Show preview cursor lines on all panels."""
        if self._preview_visible:
            return

        # Get all plots
        plots = []
        if hasattr(self._plot_host, '_subplots'):
            plots = [p for p in self._plot_host._subplots if p is not None]
        if not plots:
            main_plot = self._plot_host._get_main_plot()
            if main_plot:
                plots = [main_plot]

        if not plots:
            return

        # Get current mouse position in scene coordinates
        # graphics_layout IS a QGraphicsView, so use it directly
        view = self._plot_host.graphics_layout
        scene = view.scene() if view else None
        cursor_pos = QCursor.pos()

        initial_x = None
        if view and scene:
            # Convert global cursor pos to scene pos
            view_pos = view.mapFromGlobal(cursor_pos)
            scene_pos = view.mapToScene(view_pos)

            # Find which plot contains the cursor and get X coordinate
            for plot in plots:
                if plot.sceneBoundingRect().contains(scene_pos):
                    mouse_point = plot.vb.mapSceneToView(scene_pos)
                    initial_x = mouse_point.x()
                    break

        # Create preview pen - dashed, semi-transparent
        color = QColor('#00FF00') if not is_paired else QColor('#00FFFF')
        color.setAlpha(180)
        pen = QPen(color)
        pen.setWidth(0)
        pen.setStyle(Qt.PenStyle.DashLine)

        # Create preview lines on all panels
        self._preview_lines = []
        self._preview_end_lines = []
        self._is_paired_preview = is_paired

        # Calculate offset for paired marker end position
        end_offset = 0.5  # Default
        if is_paired:
            visible_range = self._get_visible_range()
            if visible_range:
                range_width = visible_range[1] - visible_range[0]
                end_offset = range_width * 0.10  # 10% of visible range
                end_offset = max(end_offset, 0.1)  # Minimum 0.1s

        initial_end_x = (initial_x + end_offset) if initial_x is not None else end_offset

        for plot in plots:
            # Start position line
            line = pg.InfiniteLine(
                pos=initial_x if initial_x is not None else 0,
                angle=90,
                pen=pen,
                movable=False,
            )
            line.setZValue(2000)  # Above markers
            plot.addItem(line)
            self._preview_lines.append((plot, line))

            # End position line (only for paired markers)
            if is_paired:
                end_line = pg.InfiniteLine(
                    pos=initial_end_x,
                    angle=90,
                    pen=pen,
                    movable=False,
                )
                end_line.setZValue(2000)
                plot.addItem(end_line)
                self._preview_end_lines.append((plot, end_line))

        self._preview_visible = True

    def _hide_preview(self) -> None:
        """Hide and remove preview cursor lines."""
        if not self._preview_visible:
            return

        for plot, line in self._preview_lines:
            try:
                plot.removeItem(line)
            except:
                pass

        for plot, line in self._preview_end_lines:
            try:
                plot.removeItem(line)
            except:
                pass

        self._preview_lines = []
        self._preview_end_lines = []
        self._preview_visible = False
        self._is_paired_preview = False

    def _update_preview_position(self, scene_pos) -> None:
        """Update preview cursor position based on mouse."""
        if not self._preview_visible or not self._preview_lines:
            return

        # Find the plot under the mouse to get correct X coordinate
        x_pos = None
        for plot, line in self._preview_lines:
            if plot.sceneBoundingRect().contains(scene_pos):
                mouse_point = plot.vb.mapSceneToView(scene_pos)
                x_pos = mouse_point.x()
                break

        if x_pos is None:
            return

        # Update all preview lines to this X position
        for plot, line in self._preview_lines:
            line.setValue(x_pos)

        # Update end position lines for paired markers
        if self._is_paired_preview and self._preview_end_lines:
            # Calculate offset for end position
            visible_range = self._get_visible_range()
            if visible_range:
                range_width = visible_range[1] - visible_range[0]
                end_offset = range_width * 0.10  # 10% of visible range
                end_offset = max(end_offset, 0.1)  # Minimum 0.1s
            else:
                end_offset = 0.5

            end_x = x_pos + end_offset
            for plot, line in self._preview_end_lines:
                line.setValue(end_x)

    def _handle_shortcut_click(self, event, s_pressed: bool = False, p_pressed: bool = False) -> bool:
        """Handle S+click or P+click shortcut."""
        pos = event.scenePos()

        # Find which plot was clicked (check all subplots, not just main)
        clicked_plot = None
        if hasattr(self._plot_host, '_subplots'):
            for plot in self._plot_host._subplots:
                if plot and plot.sceneBoundingRect().contains(pos):
                    clicked_plot = plot
                    break

        if clicked_plot is None:
            # Fallback to main plot
            main_plot = self._plot_host._get_main_plot()
            if main_plot and main_plot.sceneBoundingRect().contains(pos):
                clicked_plot = main_plot

        if clicked_plot is None:
            return False

        # Convert to data coordinates using the clicked plot
        mouse_point = clicked_plot.vb.mapSceneToView(pos)
        x = mouse_point.x()

        if s_pressed or self._s_held:
            # Add single marker
            marker = self._viewmodel.add_single_marker(
                time=x,
                sweep_idx=self._get_sweep_idx(),
            )
            self.marker_added.emit(marker.id)
            self.refresh()
            return True

        if p_pressed or self._p_held:
            # Add paired marker
            marker = self._viewmodel.add_paired_marker(
                start_time=x,
                sweep_idx=self._get_sweep_idx(),
                visible_range=self._get_visible_range(),
            )
            self.marker_added.emit(marker.id)
            self.refresh()
            return True

        return False

    def _show_enhanced_context_menu(self, event, plot):
        """Show enhanced context menu with event marker options."""
        # Use the plot that was actually clicked, not _get_main_plot()
        # This ensures correct coordinate conversion
        if plot is None:
            plot = self._plot_host._get_main_plot()
        if plot is None:
            return

        pos = event.scenePos()

        if not plot.sceneBoundingRect().contains(pos):
            return

        # Convert to data coordinates using the clicked plot's ViewBox
        mouse_point = plot.vb.mapSceneToView(pos)
        click_time = mouse_point.x()

        # Check if clicking on an existing marker
        marker = self._viewmodel.get_marker_at_position(
            click_time,
            self._get_sweep_idx(),
            tolerance=0.02,
        )

        if marker:
            # Show marker-specific context menu
            self._show_marker_context_menu(marker.id)
        else:
            # Show general context menu with marker options
            self._show_add_marker_context_menu(click_time, plot)

    def _show_add_marker_context_menu(self, click_time: float, plot):
        """Show context menu for adding markers."""
        # Build existing actions list
        existing_actions = []

        # Create a temporary menu to get existing actions
        temp_menu = QMenu()

        # Auto Scale Y
        action_auto_y = temp_menu.addAction("Auto Scale Y")
        action_auto_y.triggered.connect(lambda: self._plot_host._auto_scale_y_for_plot(plot))
        existing_actions.append(action_auto_y)

        # Auto Scale All Y (if multiple panels)
        if len(self._plot_host._subplots) > 1:
            action_auto_all = temp_menu.addAction("Auto Scale All Y")
            action_auto_all.triggered.connect(self._plot_host._auto_scale_y_all_panels)
            existing_actions.append(action_auto_all)

        # Reset View
        action_reset = temp_menu.addAction("Reset View")
        action_reset.triggered.connect(lambda: plot.autoRange())
        existing_actions.append(action_reset)

        # Export (PyQtGraph's export dialog)
        action_export = temp_menu.addAction("Export...")
        action_export.triggered.connect(lambda: self._show_export_dialog(plot))
        existing_actions.append(action_export)

        # Create event marker context menu
        menu = EventMarkerContextMenu(
            viewmodel=self._viewmodel,
            click_time=click_time,
            existing_actions=existing_actions,
            channel_names=self._get_channel_names(),
        )

        # Connect signals
        menu.add_single_requested.connect(self._on_add_single_requested)
        menu.add_paired_requested.connect(self._on_add_paired_requested)
        menu.auto_detect_requested.connect(self._on_auto_detect_requested)
        menu.settings_requested.connect(self._on_settings_requested)

        # Show menu
        menu.exec(QCursor.pos())

    def _show_marker_context_menu(self, marker_id: str):
        """Show context menu for an existing marker."""
        menu = MarkerContextMenu(
            viewmodel=self._viewmodel,
            marker_id=marker_id,
        )

        # Connect signals
        menu.edit_requested.connect(self._on_edit_marker_requested)
        menu.category_changed.connect(self._on_category_changed)
        menu.label_changed.connect(self._on_label_changed)
        menu.color_requested.connect(self._on_color_requested)
        menu.note_requested.connect(self._on_note_requested)
        menu.convert_requested.connect(self._on_convert_requested)
        menu.delete_requested.connect(self._on_delete_requested)
        menu.delete_all_type_requested.connect(self._on_delete_all_type_requested)

        # Show menu
        menu.exec(QCursor.pos())

    # -------------------------------------------------------------------------
    # Menu action handlers
    # -------------------------------------------------------------------------

    def _on_add_single_requested(self, time: float) -> None:
        """Handle add single marker request."""
        marker = self._viewmodel.add_single_marker(
            time=time,
            sweep_idx=self._get_sweep_idx(),
        )
        self.marker_added.emit(marker.id)
        self.refresh()

    def _on_add_paired_requested(self, start_time: float) -> None:
        """Handle add paired marker request."""
        marker = self._viewmodel.add_paired_marker(
            start_time=start_time,
            sweep_idx=self._get_sweep_idx(),
            visible_range=self._get_visible_range(),
        )
        self.marker_added.emit(marker.id)
        self.refresh()

    def _on_auto_detect_requested(self, channel_name: str) -> None:
        """Handle auto-detect request."""
        # TODO: Show detection settings dialog
        pass

    def _on_settings_requested(self) -> None:
        """Handle settings request."""
        # TODO: Show settings dialog
        pass

    def _show_export_dialog(self, plot) -> None:
        """Show export dialog for saving plot as image."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox, QInputDialog

        try:
            # Ask what to export
            options = ["All Channels", "This Channel Only"]
            choice, ok = QInputDialog.getItem(
                None,
                "Export Scope",
                "What would you like to export?",
                options,
                0,  # Default to "All Channels"
                False  # Not editable
            )

            if not ok:
                return  # User cancelled

            export_all = (choice == "All Channels")

            # Get file path from user
            file_path, selected_filter = QFileDialog.getSaveFileName(
                None,
                "Export Plot",
                "",
                "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg);;CSV Data (*.csv)"
            )

            if not file_path:
                return  # User cancelled

            # Determine what to export
            if export_all:
                # Export the entire graphics layout (all channels)
                export_item = self._plot_host.graphics_layout.scene()
            else:
                # Export just this plot
                export_item = plot

            # Determine format and export
            if file_path.endswith('.pdf') or 'PDF' in selected_filter:
                if not file_path.endswith('.pdf'):
                    file_path += '.pdf'
                self._export_to_pdf(export_item, file_path, export_all)

            elif file_path.endswith('.svg') or 'SVG' in selected_filter:
                from pyqtgraph.exporters import SVGExporter
                if export_all:
                    exporter = SVGExporter(self._plot_host.graphics_layout.scene())
                else:
                    exporter = SVGExporter(plot)
                if not file_path.endswith('.svg'):
                    file_path += '.svg'
                exporter.export(file_path)

            elif file_path.endswith('.csv') or 'CSV' in selected_filter:
                from pyqtgraph.exporters import CSVExporter
                if export_all:
                    # Export each plot's data to separate sections
                    self._export_all_csv(file_path)
                else:
                    exporter = CSVExporter(plot)
                    if not file_path.endswith('.csv'):
                        file_path += '.csv'
                    exporter.export(file_path)

            else:  # PNG
                from pyqtgraph.exporters import ImageExporter
                if export_all:
                    exporter = ImageExporter(self._plot_host.graphics_layout.scene())
                else:
                    exporter = ImageExporter(plot)
                if not file_path.endswith('.png'):
                    file_path += '.png'
                exporter.export(file_path)

            print(f"[EventMarkerPlotIntegration] Exported to: {file_path}")

        except Exception as e:
            print(f"[EventMarkerPlotIntegration] Export error: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(None, "Export Error", f"Failed to export: {e}")

    def _export_to_pdf(self, item, file_path: str, export_all: bool) -> None:
        """Export to PDF using Qt's printing system."""
        from PyQt6.QtGui import QPainter, QPageSize, QPageLayout
        from PyQt6.QtCore import QMarginsF, QSizeF, QRectF
        from PyQt6.QtPrintSupport import QPrinter

        # Use ScreenResolution to match scene coordinates (avoids DPI scaling issues)
        printer = QPrinter(QPrinter.PrinterMode.ScreenResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(file_path)

        if export_all:
            scene = self._plot_host.graphics_layout.scene()
            source_rect = scene.sceneRect()
        else:
            scene = item.scene()
            source_rect = item.sceneBoundingRect()

        # Calculate page size in millimeters (Qt uses mm for page sizes)
        # Assume 96 DPI screen, convert pixels to mm (1 inch = 25.4mm, 96 pixels = 1 inch)
        width_mm = source_rect.width() * 25.4 / 96.0
        height_mm = source_rect.height() * 25.4 / 96.0

        # Ensure minimum size and reasonable aspect ratio
        width_mm = max(width_mm, 50)
        height_mm = max(height_mm, 50)

        page_size = QPageSize(QSizeF(width_mm, height_mm), QPageSize.Unit.Millimeter)
        page_layout = QPageLayout(page_size, QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0))
        printer.setPageLayout(page_layout)

        painter = QPainter(printer)
        if painter.isActive():
            # Set a font that Illustrator handles well (Arial/Helvetica family)
            from PyQt6.QtGui import QFont
            painter.setFont(QFont("Arial", 10))

            # Get the target rect on the printer (full page)
            target_rect = QRectF(printer.pageRect(QPrinter.Unit.DevicePixel))

            # Render scene to fit the target rect
            scene.render(painter, target_rect, source_rect)
            painter.end()

    def _export_all_csv(self, file_path: str) -> None:
        """Export all channels' data to a single CSV file."""
        import csv

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            for i, plot in enumerate(self._plot_host._subplots):
                # Get plot label
                try:
                    label = plot.getAxis('left').labelText or f"Channel {i+1}"
                except:
                    label = f"Channel {i+1}"

                # Write header for this channel
                writer.writerow([f"# {label}"])
                writer.writerow(["Time", "Value"])

                # Get data from plot items
                for item in plot.listDataItems():
                    if hasattr(item, 'getData'):
                        x_data, y_data = item.getData()
                        if x_data is not None and y_data is not None:
                            for x, y in zip(x_data, y_data):
                                writer.writerow([x, y])

                writer.writerow([])  # Blank line between channels

    def _on_edit_marker_requested(self, marker_id: str) -> None:
        """Handle edit marker request."""
        # TODO: Show edit dialog
        pass

    def _on_category_changed(self, marker_id: str, new_category: str) -> None:
        """Handle category change."""
        self._viewmodel.update_marker(marker_id, category=new_category)

    def _on_label_changed(self, marker_id: str, new_label: str) -> None:
        """Handle label change."""
        self._viewmodel.update_marker(marker_id, label=new_label)

    def _on_color_requested(self, marker_id: str) -> None:
        """Handle color picker request."""
        # TODO: Show color dialog
        pass

    def _on_note_requested(self, marker_id: str) -> None:
        """Handle add note request."""
        # TODO: Show note dialog
        pass

    def _on_convert_requested(self, marker_id: str, new_type: str) -> None:
        """Handle convert type request."""
        from core.domain.events import MarkerType
        marker_type = MarkerType.SINGLE if new_type == 'single' else MarkerType.PAIRED
        self._viewmodel.convert_marker_type(marker_id, marker_type)

    def _on_delete_requested(self, marker_id: str) -> None:
        """Handle delete marker request."""
        self._viewmodel.delete_marker(marker_id)
        self.marker_deleted.emit(marker_id)

    def _on_delete_all_type_requested(self, category: str, label: str) -> None:
        """Handle delete all of type request."""
        self._viewmodel.service.delete_all_of_type(category, label)
        self.markers_changed.emit()

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------

    def _on_markers_changed(self) -> None:
        """Handle viewmodel markers changed signal."""
        self.refresh()
        self.markers_changed.emit()

    def _on_marker_moved(self, marker_id: str, new_start: float, new_end: float) -> None:
        """Handle marker moved from editor (preview during drag)."""
        if self._renderer:
            self._renderer.update_marker_position(marker_id, new_start, new_end if new_end > 0 else None)

    def _on_marker_drag_finished(self, marker_id: str, new_start: float, new_end) -> None:
        """Handle marker drag finished from renderer (commit the move)."""
        self._viewmodel.move_marker(marker_id, new_start, new_end)
        self.markers_changed.emit()

    def _on_marker_created(self, marker_id: str) -> None:
        """Handle marker created from editor."""
        self.marker_added.emit(marker_id)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def viewmodel(self) -> EventMarkerViewModel:
        """Get the viewmodel."""
        return self._viewmodel

    @property
    def enabled(self) -> bool:
        """Check if integration is enabled."""
        return self._enabled
