"""
Event marker plot integration.

This module integrates the event marker system with PyQtGraphPlotHost,
providing the glue between the marker viewmodel, renderer, editor,
and the existing plot backend.
"""

from typing import Optional, Callable, List, Tuple
from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtWidgets import QMenu, QColorDialog
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
    generate_cta_requested = pyqtSignal()  # Request to open CTA dialog

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

        # Time offset for stimulus-normalized display
        # When a stimulus channel exists, plots are shifted so t=0 is at stim onset
        # Markers are stored in absolute time, so we need this offset to display correctly
        self._time_offset: float = 0.0

        # Components (created on enable)
        self._renderer: Optional[MarkerRenderer] = None
        self._editor: Optional[MarkerEditor] = None

        # Keyboard state for shortcuts
        self._s_held = False
        self._d_held = False  # D for double/paired markers

        # Preview cursor lines (shown when S or D is held)
        self._preview_lines: List = []  # Start position lines
        self._preview_end_lines: List = []  # End position lines (paired mode only)
        self._preview_visible = False
        self._is_paired_preview = False  # Track if showing paired preview

        # Original context menu reference
        self._original_context_menu_fn = None

        # Auto-detection preview items and threshold line
        self._preview_items: List = []
        self._threshold_line = None
        self._threshold_plot = None

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

    def set_time_offset(self, offset: float) -> None:
        """
        Set the time offset for display.

        When a stimulus channel is present, the plot time is normalized so t=0
        is at the first stimulus onset. This offset is subtracted from marker
        times when displaying, and added back when creating/editing markers.

        Args:
            offset: Time offset in seconds (typically first stim onset time)
        """
        self._time_offset = offset

    def enable(self) -> None:
        """Enable the event marker integration."""
        if self._enabled:
            return

        # Intercept context menu (works even before data is loaded)
        # Only works with pyqtgraph backend which has _show_context_menu
        if hasattr(self._plot_host, '_show_context_menu'):
            self._original_context_menu_fn = self._plot_host._show_context_menu
            self._plot_host._show_context_menu = self._show_enhanced_context_menu
        else:
            # Matplotlib backend - context menu interception not supported
            self._original_context_menu_fn = None
            print("[EventMarkers] Event markers not available with matplotlib backend")
            # Show user warning
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                None,
                "Event Markers Unavailable",
                "Event markers require PyQtGraph, which could not be loaded.\n\n"
                "This may indicate a missing dependency. Please reinstall\n"
                "the application or contact support if this persists."
            )

        # Install keyboard handler at application level to capture S/P keys regardless of focus
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.installEventFilter(self)

        # Also install on graphics layout for mouse events (pyqtgraph only)
        if hasattr(self._plot_host, 'graphics_layout'):
            self._plot_host.graphics_layout.installEventFilter(self)

        self._enabled = True

        # Try to create renderer/editor if plot is ready
        self._ensure_renderer_editor()

    def _ensure_renderer_editor(self) -> bool:
        """Create renderer and editor if plot is ready. Returns True if ready."""
        # Check if backend supports required methods (pyqtgraph only)
        if not hasattr(self._plot_host, '_get_main_plot') or not hasattr(self._plot_host, 'graphics_layout'):
            # Matplotlib backend - event markers not supported
            return False

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
            get_signal_data=self._get_signal_data,
            get_sweep_idx=self._get_sweep_idx,
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
        except Exception:
            pass

        try:
            from PyQt6.QtWidgets import QApplication
            app = QApplication.instance()
            if app:
                app.removeEventFilter(self)
        except Exception:
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

        self._renderer.render_markers(self._get_sweep_idx(), time_offset=self._time_offset)

    def on_sweep_changed(self, new_sweep_idx: int) -> None:
        """
        Called when the sweep changes. Re-renders markers for the new sweep.

        Args:
            new_sweep_idx: The new sweep index
        """
        if not self._enabled:
            return

        # Ensure renderer exists
        if not self._ensure_renderer_editor():
            return

        # Re-render markers for the new sweep
        self._renderer.render_markers(new_sweep_idx, time_offset=self._time_offset)

    def eventFilter(self, obj: QObject, event) -> bool:
        """Filter events for keyboard shortcuts and preview cursor."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtWidgets import QApplication

        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            # Check for S/D + click shortcuts
            if event.button() == Qt.MouseButton.LeftButton:
                s_pressed = self._s_held
                d_pressed = self._d_held

                if s_pressed or d_pressed:
                    # Hide preview and create actual marker
                    self._hide_preview()
                    return self._handle_shortcut_click(event, s_pressed, d_pressed)

        elif event.type() == QEvent.Type.GraphicsSceneMouseMove:
            # Update preview position when S or D is held
            if self._s_held or self._d_held:
                self._update_preview_position(event.scenePos())

        elif event.type() == QEvent.Type.KeyPress:
            key = event.key()
            modifiers = event.modifiers()

            # Ctrl+Z for undo
            if key == Qt.Key.Key_Z and modifiers & Qt.KeyboardModifier.ControlModifier:
                if self._viewmodel.can_undo:
                    self._viewmodel.undo()
                    self.refresh()
                    return True  # Consume the event

            # Ctrl+Y for redo
            elif key == Qt.Key.Key_Y and modifiers & Qt.KeyboardModifier.ControlModifier:
                if self._viewmodel.can_redo:
                    self._viewmodel.redo()
                    self.refresh()
                    return True  # Consume the event

            # S key for single marker preview
            elif key == Qt.Key.Key_S and not self._s_held:
                self._s_held = True
                try:
                    self._show_preview(is_paired=False)
                except Exception as e:
                    print(f"[EventMarkerPlotIntegration] Preview error: {e}")
                    self._s_held = False  # Reset on error

            # D key for paired/double marker preview
            elif key == Qt.Key.Key_D and not self._d_held:
                self._d_held = True
                try:
                    self._show_preview(is_paired=True)
                except Exception as e:
                    print(f"[EventMarkerPlotIntegration] Preview error: {e}")
                    self._d_held = False  # Reset on error

        elif event.type() == QEvent.Type.KeyRelease:
            key = event.key()
            if key == Qt.Key.Key_S:
                self._s_held = False
                self._hide_preview()
            elif key == Qt.Key.Key_D:
                self._d_held = False
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
            except Exception:
                pass

        for plot, line in self._preview_end_lines:
            try:
                plot.removeItem(line)
            except Exception:
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

    def _handle_shortcut_click(self, event, s_pressed: bool = False, d_pressed: bool = False) -> bool:
        """Handle S+click or D+click shortcut."""
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

        # Convert from display time back to absolute time (add offset)
        # Display shows time normalized to stim onset (t_display = t_absolute - offset)
        # So to get absolute time: t_absolute = t_display + offset
        x_absolute = x + self._time_offset

        if s_pressed or self._s_held:
            # Add single marker
            marker = self._viewmodel.add_single_marker(
                time=x_absolute,
                sweep_idx=self._get_sweep_idx(),
            )
            self.marker_added.emit(marker.id)
            self.refresh()
            return True

        if d_pressed or self._d_held:
            # Add paired/double marker
            marker = self._viewmodel.add_paired_marker(
                start_time=x_absolute,
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
        click_time_display = mouse_point.x()

        # Convert from display time to absolute time for marker lookup
        click_time_absolute = click_time_display + self._time_offset

        # Calculate dynamic tolerance based on visible range
        # Use 3% of visible range, but with min/max bounds for easier clicking
        visible_range = self._get_visible_range()
        if visible_range:
            range_width = visible_range[1] - visible_range[0]
            tolerance = max(0.05, min(0.5, range_width * 0.03))  # 3% of range, bounded 0.05-0.5s
        else:
            tolerance = 0.1  # Generous default

        # Check if clicking on an existing marker (use absolute time for lookup)
        marker = self._viewmodel.get_marker_at_position(
            click_time_absolute,
            self._get_sweep_idx(),
            tolerance=tolerance,
        )

        if marker:
            # Show marker-specific context menu
            self._show_marker_context_menu(marker.id)
        else:
            # Show general context menu with marker options
            # Pass absolute time so markers are created at correct positions
            self._show_add_marker_context_menu(click_time_absolute, plot)

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

        # Determine which channel was clicked
        clicked_channel = self._get_channel_for_plot(plot)

        # Create event marker context menu
        menu = EventMarkerContextMenu(
            viewmodel=self._viewmodel,
            click_time=click_time,
            existing_actions=existing_actions,
            channel_names=self._get_channel_names(),
            show_derivative_on_drag=self._renderer.get_show_derivative_on_drag() if self._renderer else False,
            clicked_channel=clicked_channel,
        )

        # Connect signals
        menu.add_single_requested.connect(self._on_add_single_requested)
        menu.add_paired_requested.connect(self._on_add_paired_requested)
        menu.auto_detect_requested.connect(self._on_auto_detect_requested)
        menu.settings_requested.connect(self._on_settings_requested)
        menu.delete_all_requested.connect(self._on_delete_all_requested)
        menu.delete_all_type_requested.connect(self._on_delete_all_type_requested)
        menu.delete_all_sweep_requested.connect(self._on_delete_all_sweep_requested)
        menu.delete_category_sweep_requested.connect(self._on_delete_category_sweep_requested)
        menu.delete_category_all_requested.connect(self._on_delete_category_all_requested)
        menu.derivative_toggle_changed.connect(self._on_derivative_toggle_changed)
        menu.generate_cta_requested.connect(self._on_generate_cta_requested)

        # Add Performance Mode submenu
        mw = self._plot_host._find_main_window() if hasattr(self._plot_host, '_find_main_window') else None
        pm = getattr(mw, 'plot_manager', None) if mw else None
        if pm:
            menu.addSeparator()
            perf_menu = menu.addMenu("Performance Mode")
            override = pm._downsample_override
            action_auto = perf_menu.addAction("Auto (Recommended)")
            action_auto.setCheckable(True)
            action_auto.setChecked(override is None)
            action_fast = perf_menu.addAction("Fast (Downsample On)")
            action_fast.setCheckable(True)
            action_fast.setChecked(override is True)
            action_full = perf_menu.addAction("Full Resolution (Downsample Off)")
            action_full.setCheckable(True)
            action_full.setChecked(override is False)
            last_ms = pm._last_redraw_ms
            if last_ms > 0:
                perf_menu.addSeparator()
                info = perf_menu.addAction(f"Last redraw: {last_ms:.0f}ms")
                info.setEnabled(False)
            action_auto.triggered.connect(lambda: pm.toggle_auto_downsample(force=None))
            action_fast.triggered.connect(lambda: pm.toggle_auto_downsample(force=True))
            action_full.triggered.connect(lambda: pm.toggle_auto_downsample(force=False))

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
        menu.line_width_changed.connect(self._on_line_width_changed)
        menu.grab_width_changed.connect(self._on_grab_width_changed)
        menu.note_requested.connect(self._on_note_requested)
        menu.convert_requested.connect(self._on_convert_requested)
        menu.delete_requested.connect(self._on_delete_requested)
        menu.delete_all_requested.connect(self._on_delete_all_requested)
        menu.delete_all_type_requested.connect(self._on_delete_all_type_requested)
        menu.delete_all_sweep_requested.connect(self._on_delete_all_sweep_requested)
        menu.delete_category_sweep_requested.connect(self._on_delete_category_sweep_requested)
        menu.delete_category_all_requested.connect(self._on_delete_category_all_requested)
        menu.edit_in_detector_requested.connect(self._on_edit_in_detector)

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
        from dialogs.marker_detection_dialog import MarkerDetectionDialog

        # Get channel names
        channel_names = []
        if self._get_channel_names:
            channel_names = self._get_channel_names()

        if not channel_names:
            return

        # Get full-resolution signal data for detection (not downsampled plot data)
        def get_signal_from_plot(channel: str):
            # Use raw data callback — plot items may be downsampled for display
            if self._get_signal_data:
                t_data, y_data = self._get_signal_data(channel, self._get_sweep_idx())
                if t_data is not None and y_data is not None and len(t_data) > 1:
                    sample_rate = 1.0 / (t_data[1] - t_data[0])
                    return t_data, y_data, sample_rate

            # Fallback to plot items if raw callback not available
            plot = self._find_plot_for_channel(channel)
            if plot is not None:
                for item in plot.listDataItems():
                    if hasattr(item, 'getData'):
                        x_data, y_data = item.getData()
                        if x_data is not None and y_data is not None and len(x_data) > 100:
                            sample_rate = 1.0 / (x_data[1] - x_data[0]) if len(x_data) > 1 else 1000.0
                            return x_data + self._time_offset, y_data, sample_rate
            return None

        # Create and show detection dialog
        dialog = MarkerDetectionDialog(
            viewmodel=self._viewmodel,
            channel_names=channel_names,
            get_signal_data=get_signal_from_plot,
            sweep_idx=self._get_sweep_idx(),
            parent=None,
            initial_channel=channel_name,
            time_offset=self._time_offset,  # Pass time offset for display alignment
        )

        # Connect preview signal to show temporary markers
        dialog.preview_requested.connect(self._on_preview_events)
        dialog.detection_complete.connect(self._on_detection_complete)
        dialog.threshold_changed.connect(self._on_threshold_changed)
        dialog.threshold_cleared.connect(self._clear_threshold_line)

        dialog.exec()

        # Clear any preview markers and threshold line
        self._clear_preview_markers()
        self._clear_threshold_line()

    def _on_edit_in_detector(self, marker_id: str) -> None:
        """Handle 'Edit Category in Detector' request from marker context menu."""
        marker = self._viewmodel.store.get(marker_id)
        if marker is None:
            return

        category = marker.category
        sweep_idx = self._get_sweep_idx()

        # Gather all markers of the same category in the current sweep
        all_markers = self._viewmodel.get_markers_for_sweep(sweep_idx)
        initial_events = []
        for m in all_markers:
            if m.category == category:
                if m.is_paired and m.end_time is not None:
                    initial_events.append((m.start_time, m.end_time))
                else:
                    # Single marker — create a small region around it
                    initial_events.append((m.start_time, m.start_time + 0.5))
        initial_events.sort(key=lambda e: e[0])

        # Determine channel — try to find from the marker's stored data or use first channel
        channel_names = self._get_channel_names() if self._get_channel_names else []
        initial_channel = channel_names[0] if channel_names else None

        if not channel_names:
            return

        # Get full-resolution signal data (same as _on_auto_detect_requested)
        def get_signal_from_plot(channel: str):
            if self._get_signal_data:
                t_data, y_data = self._get_signal_data(channel, self._get_sweep_idx())
                if t_data is not None and y_data is not None and len(t_data) > 1:
                    sample_rate = 1.0 / (t_data[1] - t_data[0])
                    return t_data, y_data, sample_rate
            plot = self._find_plot_for_channel(channel)
            if plot is not None:
                for item in plot.listDataItems():
                    if hasattr(item, 'getData'):
                        x_data, y_data = item.getData()
                        if x_data is not None and y_data is not None and len(x_data) > 100:
                            sample_rate = 1.0 / (x_data[1] - x_data[0]) if len(x_data) > 1 else 1000.0
                            return x_data + self._time_offset, y_data, sample_rate
            return None

        from dialogs.marker_detection_dialog import MarkerDetectionDialog

        dialog = MarkerDetectionDialog(
            viewmodel=self._viewmodel,
            channel_names=channel_names,
            get_signal_data=get_signal_from_plot,
            sweep_idx=sweep_idx,
            parent=None,
            initial_channel=initial_channel,
            time_offset=self._time_offset,
            initial_events=initial_events,
        )

        dialog.preview_requested.connect(self._on_preview_events)
        dialog.detection_complete.connect(self._on_detection_complete)
        dialog.threshold_changed.connect(self._on_threshold_changed)
        dialog.threshold_cleared.connect(self._clear_threshold_line)

        dialog.exec()

        self._clear_preview_markers()
        self._clear_threshold_line()

    def _on_preview_events(self, events: list) -> None:
        """Show preview of detected events (temporary markers)."""
        # Clear any existing preview markers
        self._clear_preview_markers()

        if not events:
            return

        # Get all plots
        plots = []
        if hasattr(self._plot_host, 'get_subplots'):
            plots = self._plot_host.get_subplots()
        elif hasattr(self._plot_host, '_subplots'):
            plots = self._plot_host._subplots
        else:
            main_plot = self._plot_host._get_main_plot()
            if main_plot:
                plots = [main_plot]

        # Draw preview regions on all plots
        for start, end in events:
            # Convert to display time
            display_start = start - self._time_offset
            display_end = end - self._time_offset

            for plot in plots:
                region = pg.LinearRegionItem(
                    values=[display_start, display_end],
                    brush=pg.mkBrush(255, 200, 0, 40),  # Yellow with low alpha
                    pen=pg.mkPen(255, 200, 0, 150),
                    movable=False,
                )
                region.setZValue(800)  # Below actual markers
                plot.addItem(region)
                self._preview_items.append((plot, region))

    def _clear_preview_markers(self) -> None:
        """Clear any preview markers from the plot."""
        for plot, item in self._preview_items:
            try:
                plot.removeItem(item)
            except Exception:
                pass

        self._preview_items.clear()

    def _on_detection_complete(self, count: int) -> None:
        """Handle detection completion."""
        self._clear_preview_markers()
        self.refresh()

    def _on_threshold_changed(self, threshold: float, channel_name: str) -> None:
        """Show threshold line on the specified channel's plot."""
        # Clear existing threshold line
        self._clear_threshold_line()

        # Find the plot for this channel
        plot = self._find_plot_for_channel(channel_name)
        if plot is None:
            return

        # Create horizontal threshold line (orange dashed)
        pen = pg.mkPen(color=(255, 165, 0), width=2, style=Qt.PenStyle.DashLine)
        line = pg.InfiniteLine(
            pos=threshold,
            angle=0,  # Horizontal
            pen=pen,
            movable=False,
            label=f'Threshold: {threshold:.3f}',
            labelOpts={'position': 0.05, 'color': (255, 165, 0), 'fill': (0, 0, 0, 100)}
        )
        line.setZValue(1000)  # Above most items
        plot.addItem(line)

        self._threshold_line = line
        self._threshold_plot = plot

    def _get_channel_for_plot(self, plot) -> Optional[str]:
        """Get the channel name displayed on a given plot panel.

        The Y-axis label is set to config.name (the channel name) by
        plot_manager._draw_channels_pyqtgraph, so reading it back is the
        most reliable approach — it works regardless of which channels
        the channel manager has made visible.
        """
        if plot is None:
            return None

        # Read the Y-axis label — this IS the channel name set by plot_manager
        try:
            left_axis = plot.getAxis('left')
            if left_axis and hasattr(left_axis, 'labelText'):
                label = left_axis.labelText
                if label:
                    # Strip pyqtgraph auto-appended scale factor like " (X0.001)"
                    if '(' in label:
                        label = label.split('(')[0].strip()
                    if label:
                        return label
        except Exception:
            pass

        return None

    def _find_plot_for_channel(self, channel_name: str):
        """Find the plot panel that displays a specific channel."""
        # Get all plots
        plots = []
        if hasattr(self._plot_host, 'get_subplots'):
            plots = self._plot_host.get_subplots()
        elif hasattr(self._plot_host, '_subplots'):
            plots = self._plot_host._subplots
        else:
            main_plot = self._plot_host._get_main_plot()
            if main_plot:
                plots = [main_plot]

        # Find plot by Y-axis label
        for plot in plots:
            try:
                left_axis = plot.getAxis('left')
                if left_axis and hasattr(left_axis, 'labelText'):
                    label = left_axis.labelText
                    if not label:
                        continue
                    # Strip pyqtgraph auto-appended scale factor like " (X0.001)"
                    clean_label = label.split('(')[0].strip() if '(' in label else label
                    if clean_label == channel_name or channel_name in label:
                        return plot
            except Exception:
                pass

        # Fallback: return first plot if only one exists
        if len(plots) == 1:
            return plots[0]

        return None

    def _clear_threshold_line(self) -> None:
        """Clear the threshold line from the plot."""
        if self._threshold_line is not None:
            try:
                if self._threshold_plot:
                    self._threshold_plot.removeItem(self._threshold_line)
            except Exception:
                pass
            self._threshold_line = None
            self._threshold_plot = None

    def _on_settings_requested(self) -> None:
        """Handle settings request - show marker display settings dialog."""
        from dialogs.marker_settings_dialog import MarkerSettingsDialog

        if not self._renderer:
            return

        dialog = MarkerSettingsDialog(
            parent=None,
            single_width=self._renderer._default_single_width,
            paired_width=self._renderer._default_paired_width,
            fill_alpha=self._renderer._default_fill_alpha,
        )
        if dialog.exec() == MarkerSettingsDialog.DialogCode.Accepted:
            self._renderer.set_default_widths(
                single_width=dialog.single_width,
                paired_width=dialog.paired_width,
                fill_alpha=dialog.fill_alpha,
            )
            # Reset per-marker line_width overrides if requested
            if dialog.apply_to_all:
                for marker in self._viewmodel.store.all():
                    if marker.line_width is not None:
                        self._viewmodel.update_marker(marker.id, line_width=-1)  # -1 = reset to None
            self.refresh()

    def _on_derivative_toggle_changed(self, enabled: bool) -> None:
        """Handle derivative overlay toggle change from context menu."""
        if self._renderer:
            self._renderer.set_show_derivative_on_drag(enabled)

    def _on_generate_cta_requested(self) -> None:
        """Handle request to generate photometry CTA."""
        # Emit signal for main window to handle
        self.generate_cta_requested.emit()

    def _show_export_dialog(self, plot) -> None:
        """Show export dialog for saving plot as image.

        NOTE: This export functionality doesn't really belong in EventMarkerPlotIntegration.
        It's here because this class intercepts the context menu. Consider refactoring to
        move export methods (_show_export_dialog, _export_to_pdf, _export_all_csv) to the
        plot host or a dedicated PlotExporter class.
        """
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
        """Export all channels' main trace data to a single CSV file.

        Only exports the primary signal trace per panel, skipping overlay items
        like scatter plots (peaks), event marker lines, and threshold indicators.
        """
        import csv

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)

            for i, plot in enumerate(self._plot_host._subplots):
                # Get plot label
                try:
                    label = plot.getAxis('left').labelText or f"Channel {i+1}"
                except Exception:
                    label = f"Channel {i+1}"

                # Write header for this channel
                writer.writerow([f"# {label}"])
                writer.writerow(["Time", "Value"])

                # Find the main trace only — skip scatter plots, markers, overlays.
                # The main trace is the PlotDataItem with the most points (>100).
                best_x, best_y = None, None
                best_len = 0
                for item in plot.listDataItems():
                    if hasattr(item, 'getData'):
                        x_data, y_data = item.getData()
                        if x_data is not None and y_data is not None and len(x_data) > best_len:
                            best_x, best_y = x_data, y_data
                            best_len = len(x_data)

                if best_x is not None and best_len > 100:
                    for x, y in zip(best_x, best_y):
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
        marker = self._viewmodel.store.get(marker_id)
        if marker is None:
            return

        # Get current color as starting color
        current_color = QColor(self._viewmodel.get_color_for_marker(marker))

        color = QColorDialog.getColor(
            current_color,
            None,
            f"Set Marker Color — {marker.category}/{marker.label}",
        )
        if color.isValid():
            self._viewmodel.update_marker(marker_id, color_override=color.name())
            self.refresh()

    def _on_line_width_changed(self, marker_id: str, width: int) -> None:
        """Handle line thickness change from context menu."""
        self._viewmodel.update_marker(marker_id, line_width=width)
        self.refresh()

    def _on_grab_width_changed(self, marker_id: str, grab_width: int) -> None:
        """Handle edge grab width change for paired markers."""
        if marker_id in self._renderer._paired_regions:
            for region in self._renderer._paired_regions[marker_id]:
                for edge_line in region.lines:
                    edge_line._maxMarkerSize = grab_width

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

    def _on_delete_all_requested(self) -> None:
        """Handle delete all markers request."""
        count = self._viewmodel.clear_all()
        if count > 0:
            self.refresh()
            self.markers_changed.emit()

    def _on_delete_all_type_requested(self, category: str, label: str) -> None:
        """Handle delete all of type request."""
        self._viewmodel.service.delete_all_of_type(category, label)
        self.refresh()
        self.markers_changed.emit()

    def _on_delete_all_sweep_requested(self) -> None:
        """Handle delete all in current sweep request."""
        sweep_idx = self._get_sweep_idx()
        count = self._viewmodel.delete_all_for_sweep(sweep_idx)
        if count > 0:
            self.refresh()
            self.markers_changed.emit()

    def _on_delete_category_sweep_requested(self, category: str) -> None:
        """Handle delete category in current sweep request."""
        sweep_idx = self._get_sweep_idx()
        count = self._viewmodel.delete_category_for_sweep(category, sweep_idx)
        if count > 0:
            self.refresh()
            self.markers_changed.emit()

    def _on_delete_category_all_requested(self, category: str) -> None:
        """Handle delete category in all sweeps request."""
        count = self._viewmodel.delete_category_all_sweeps(category)
        if count > 0:
            self.refresh()
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
