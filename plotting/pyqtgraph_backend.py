"""
PyQtGraph-based plotting backend for high-performance data visualization.

This module provides a drop-in replacement for the matplotlib-based PlotHost class,
offering significantly better performance for large datasets (10-50x faster).

Usage:
    from plotting.pyqtgraph_backend import PyQtGraphPlotHost
    plot_host = PyQtGraphPlotHost(parent=widget)

Features:
- GPU-accelerated rendering via OpenGL
- Automatic downsampling for smooth pan/zoom with millions of points
- Same API as matplotlib PlotHost for easy integration
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

try:
    import pyqtgraph as pg
    from pyqtgraph import PlotWidget, GraphicsLayoutWidget
    from core.plot_themes import DARK_THEME, LIGHT_THEME
    PYQTGRAPH_AVAILABLE = True

    # Monkey-patch PyQtGraph to suppress 'autoRangeEnabled' AttributeError
    # This error occurs when PlotDataItems try to access their view during reparenting
    # (e.g., when GraphicsLayoutWidget.clear() is called and items reference stale views)
    # The error doesn't affect functionality, just causes console noise
    # Guard against re-patching on hot reload (which would cause infinite recursion)
    if hasattr(pg, 'PlotDataItem') and not hasattr(pg.PlotDataItem, '_viewRangeChanged_patched'):
        _original_viewRangeChanged = pg.PlotDataItem.viewRangeChanged

        def _safe_viewRangeChanged(self):
            """Wrapped viewRangeChanged that handles stale view references gracefully."""
            try:
                _original_viewRangeChanged(self)
            except AttributeError as e:
                if 'autoRangeEnabled' in str(e):
                    # Silently ignore - this happens during reparenting and doesn't affect functionality
                    pass
                else:
                    raise  # Re-raise other AttributeErrors

        pg.PlotDataItem.viewRangeChanged = _safe_viewRangeChanged
        pg.PlotDataItem._viewRangeChanged_patched = True  # Mark as patched to prevent re-patching

except ImportError:
    PYQTGRAPH_AVAILABLE = False
    print("[PyQtGraph Backend] pyqtgraph not installed. Install with: pip install pyqtgraph")


# Default colors matching matplotlib PlotHost
PLETH_COLOR = (255, 255, 255)  # White for dark mode
PEAK_COLOR = (255, 0, 0)       # Red
ONSET_COLOR = (46, 204, 113)   # Green
OFFSET_COLOR = (243, 156, 18)  # Orange
EXPMIN_COLOR = (31, 120, 180)  # Blue
EXPOFF_COLOR = (155, 89, 182)  # Purple


def safe_clear_graphics_layout(layout):
    """Safely clear PyQtGraph GraphicsLayoutWidget, handling inconsistent item tracking.

    PyQtGraph's GraphicsLayoutWidget.clear() can raise KeyError or ValueError when
    items have been removed or reparented in ways that leave the internal tracking
    inconsistent. This wrapper handles those cases gracefully.

    Args:
        layout: GraphicsLayoutWidget to clear
    """
    try:
        layout.clear()
    except (KeyError, ValueError, RuntimeError) as e:
        # Fallback: manually remove items
        try:
            for item in list(layout.items.keys()) if hasattr(layout, 'items') else []:
                try:
                    layout.removeItem(item)
                except (KeyError, ValueError, RuntimeError):
                    pass
        except (AttributeError, RuntimeError):
            pass  # Layout may have been deleted


class _MatplotlibCompatEvent:
    """Wrapper to provide matplotlib-compatible event attributes for PyQtGraph events.

    This allows editing modes designed for matplotlib to work with PyQtGraph.
    """

    # PyQt6 Qt.MouseButton values to matplotlib button mapping
    # Qt: LeftButton=1, RightButton=2, MiddleButton=4
    # Matplotlib: left=1, middle=2, right=3
    _BUTTON_MAP = {
        1: 1,   # Left -> 1
        2: 3,   # Right -> 3 (matplotlib uses 3 for right-click)
        4: 2,   # Middle -> 2
    }

    def __init__(self, pyqt_event, plot_widget, xdata, ydata):
        self._pyqt_event = pyqt_event
        self.inaxes = plot_widget  # The plot widget acts as the "axes"
        self.xdata = xdata
        self.ydata = ydata
        # Provide button info (matplotlib uses 1=left, 2=middle, 3=right)
        qt_button = pyqt_event.button()
        if hasattr(qt_button, 'value'):
            # PyQt6 enum
            qt_val = qt_button.value
        else:
            qt_val = int(qt_button) if qt_button else 1
        # Map Qt button to matplotlib convention
        self.button = self._BUTTON_MAP.get(qt_val, 1)

        # Store keyboard modifiers from event (more reliable than QApplication.keyboardModifiers())
        # This captures the modifier state at the time of the event, not when handler runs
        try:
            qt_modifiers = pyqt_event.modifiers()
            self.shift_held = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)
            self.ctrl_held = bool(qt_modifiers & Qt.KeyboardModifier.ControlModifier)
            self.alt_held = bool(qt_modifiers & Qt.KeyboardModifier.AltModifier)
        except:
            # Fallback to QApplication if event doesn't have modifiers
            from PyQt6.QtWidgets import QApplication
            qt_modifiers = QApplication.keyboardModifiers()
            self.shift_held = bool(qt_modifiers & Qt.KeyboardModifier.ShiftModifier)
            self.ctrl_held = bool(qt_modifiers & Qt.KeyboardModifier.ControlModifier)
            self.alt_held = bool(qt_modifiers & Qt.KeyboardModifier.AltModifier)

    def __getattr__(self, name):
        """Forward unknown attributes to the underlying PyQtGraph event."""
        return getattr(self._pyqt_event, name)


class PyQtGraphPlotHost(QWidget):
    """
    PyQtGraph-based plot host with same API as matplotlib PlotHost.

    This provides a high-performance alternative for large datasets while
    maintaining API compatibility with the existing PlotHost class.
    """

    # Signal emitted when user clicks on plot
    plot_clicked = pyqtSignal(float, float, object)

    def __init__(self, parent=None):
        super().__init__(parent)

        if not PYQTGRAPH_AVAILABLE:
            raise ImportError("pyqtgraph is required for PyQtGraphPlotHost")

        # Configure pyqtgraph for quality and performance
        pg.setConfigOptions(
            antialias=True,   # Smooth lines (prevents stair-step aliasing on vertical strokes)
            useOpenGL=True,   # GPU acceleration (minimizes performance impact of antialiasing)
            enableExperimental=True,
        )

        # Main graphics layout (can hold multiple plots)
        self.graphics_layout = GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#000000')  # True black background

        # Main plot widget
        self.plot_widget = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setLabel('left', 'Signal')

        # Enable auto-range initially
        self.plot_widget.enableAutoRange()

        # Configure mouse behavior: X-axis only, shift required for pan
        self._configure_plot_mouse(self.plot_widget)

        # Disable PyQtGraph's built-in ViewBox context menu
        self.plot_widget.vb.setMenuEnabled(False)
        self.plot_widget.setMenuEnabled(False)

        # Store references to plot items
        self._main_trace = None
        self._peak_scatter = None
        self._onset_scatter = None
        self._offset_scatter = None
        self._expmin_scatter = None
        self._expoff_scatter = None
        self._sigh_scatter = None
        self._threshold_line = None
        self._y2_line = None
        self._span_items = []
        self._region_overlays = []

        # For compatibility with matplotlib PlotHost API
        self.ax_main = self.plot_widget  # Primary plot
        self.ax_event = None  # Event subplot (will be added dynamically)
        self.ax_y2 = None     # Y2 axis (ViewBox)
        self.fig = self       # Self-reference for compatibility
        self.canvas = self    # Self-reference for compatibility
        self.toolbar = None   # No toolbar in pyqtgraph version (built-in)

        # View preservation
        self._preserve_x = True
        self._preserve_y = False
        self._last_single = {"xlim": None, "ylim": None}
        self._last_grid = {"xlim": None, "ylims": []}

        # Auto-range control - disable after first draw to prevent jumpiness
        self._autorange_enabled = True

        # Click callback
        self._external_click_cb = None

        # Theme state
        self._current_theme = 'dark'

        # Edit mode state (disables context menu when active)
        self._edit_mode_active = False

        # Flag to force auto-range on next draw (set by clear_saved_view)
        self._force_autorange = False

        # Connect mouse click
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

        # Disable default context menu (we handle right-click for editing)
        self.plot_widget.setMenuEnabled(False)

        # Drag support for editing modes
        self._drag_callback = None  # Callback for drag events: fn(event_type, xdata, ydata, event)
        self._drag_start_pos = None  # Scene position where drag started
        self._dragging = False  # Are we currently dragging?
        self._drag_visual = None  # Visual indicator during drag (e.g., selection rectangle)

        # Keyboard event callback
        self._key_callback = None  # Callback for key events: fn(key, modifiers)

        # Connect scene mouse events for drag support
        self.graphics_layout.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        # Make widget focusable for keyboard events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.graphics_layout)

        # Store subplots for multi-panel mode
        self._subplots = [self.plot_widget]

    # ------- Theme Support -------
    def _get_theme_colors(self):
        """Get marker colors based on current theme."""
        t = DARK_THEME if self._current_theme == 'dark' else LIGHT_THEME
        return {
            'onset': t['onset_color'],
            'offset': t['offset_color'],
            'expmin': t['expmin_color'],
            'expoff': t['expoff_color'],
            'peak': t['peak_color'],
            'trace': t['trace_color'],
            'text': t['text_color'],
            'background': t['figure_facecolor'],
        }

    def set_plot_theme(self, theme_name):
        """Apply dark or light theme."""
        self._current_theme = theme_name

        # Get colors from theme
        theme = DARK_THEME if theme_name == 'dark' else LIGHT_THEME
        bg_color = theme['figure_facecolor']
        text_color = theme['text_color']
        trace_color = theme['trace_color']

        self.graphics_layout.setBackground(bg_color)

        # Update all subplot styles
        for plot in self._subplots:
            plot.getAxis('bottom').setTextPen(text_color)
            plot.getAxis('left').setTextPen(text_color)
            plot.getAxis('bottom').setPen(text_color)
            plot.getAxis('left').setPen(text_color)

        # Update trace color if exists
        if self._main_trace is not None:
            self._main_trace.setPen(pg.mkPen(trace_color, width=1))

        # Update Y2 line color if exists
        if self._y2_line is not None:
            self._y2_line.setPen(pg.mkPen(trace_color, width=1))

        # Update all traces in all subplots (for multi-panel mode)
        for plot in self._subplots:
            for item in plot.listDataItems():
                # Update PlotDataItems (traces) but skip threshold lines
                if item is not self._threshold_line:
                    current_pen = item.opts.get('pen')
                    if current_pen is not None:
                        # Preserve line width, just update color
                        width = 1
                        if hasattr(current_pen, 'width'):
                            width = current_pen.width()
                        elif isinstance(current_pen, dict) and 'width' in current_pen:
                            width = current_pen['width']
                        item.setPen(pg.mkPen(trace_color, width=width))

        # Update marker colors for theme
        self._refresh_marker_colors()

    def _refresh_marker_colors(self):
        """Refresh marker colors based on current theme."""
        colors = self._get_theme_colors()

        # Update breath markers (onset, offset, expmin, expoff)
        marker_mapping = [
            ('_onset_scatter', colors['onset']),
            ('_offset_scatter', colors['offset']),
            ('_expmin_scatter', colors['expmin']),
            ('_expoff_scatter', colors['expoff']),
            ('_peak_scatter', colors['peak']),
        ]

        for attr_name, color in marker_mapping:
            scatter = getattr(self, attr_name, None)
            if scatter is not None:
                try:
                    qcolor = QColor(color)
                    scatter.setBrush(pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue()))
                except Exception:
                    pass  # Scatter may be invalid

    # ------- View Preservation -------
    def set_preserve(self, x: bool = True, y: bool = False):
        """Set view preservation mode."""
        self._preserve_x = bool(x)
        self._preserve_y = bool(y)

    def set_preserve_y(self, preserve: bool):
        """Set Y-axis preservation mode (for Auto Y Scale toggle)."""
        self._preserve_y = bool(preserve)
        if not preserve:
            # Clear stored Y limits when disabling preservation
            self._last_single["ylim"] = None
            self._last_grid["ylims"] = []

    def clear_saved_view(self, mode: str = None):
        """Clear saved view state and force auto-range on next draw."""
        if mode is None or mode == "single":
            self._last_single = {"xlim": None, "ylim": None}
        if mode is None or mode == "grid":
            self._last_grid = {"xlim": None, "ylims": []}
        # Force auto-range on next draw to show full data
        self._force_autorange = True

    def set_xlim(self, x0: float, x1: float):
        """Set x-axis limits."""
        main_plot = self._get_main_plot()
        if main_plot is not None:
            main_plot.setXRange(x0, x1, padding=0)
        self._last_single["xlim"] = (x0, x1)

    def disable_autorange(self):
        """Disable auto-range to prevent jumpiness during edits."""
        self._autorange_enabled = False
        for plot in self._subplots:
            plot.disableAutoRange()

    def enable_autorange(self):
        """Re-enable auto-range."""
        self._autorange_enabled = True
        for plot in self._subplots:
            plot.enableAutoRange()

    # ------- Main Plotting API -------
    def show_trace_with_spans(self, t, y, spans_s, title: str = "",
                              max_points: int = None, ylabel: str = "Signal", state=None):
        """
        Display main trace with optional stimulus spans.

        This is the primary plotting method, matching PlotHost API.
        """
        # Use _get_main_plot() to get valid plot reference
        main_plot = self._get_main_plot()
        if main_plot is None:
            # Fallback to plot_widget if available
            main_plot = self.plot_widget
            if main_plot is None:
                print("[PyQtGraph] Warning: No valid plot widget for show_trace_with_spans")
                return

        # Save previous view
        prev_xlim = self._last_single["xlim"] if self._preserve_x else None
        prev_ylim = self._last_single["ylim"] if self._preserve_y else None

        # Clear existing items
        self._clear_all_items()

        # Get trace color based on theme
        trace_color = '#d4d4d4' if self._current_theme == 'dark' else '#000000'

        # Downsample if needed (pyqtgraph handles this well, but we can help)
        t_plot, y_plot = self._downsample(t, y, max_points)

        # Plot main trace
        self._main_trace = main_plot.plot(
            t_plot, y_plot,
            pen=pg.mkPen(trace_color, width=1),
            name='trace'
        )

        # Add stimulus spans (blue regions)
        # Use a subtle border pen to ensure thin pulses remain visible at all zoom levels
        stim_pen = pg.mkPen(46, 80, 144, 100, width=1)  # Subtle border for antialiasing
        for (t0, t1) in (spans_s or []):
            if t1 > t0:
                region = pg.LinearRegionItem(
                    values=[t0, t1],
                    brush=pg.mkBrush(46, 80, 144, 60),  # #2E5090 with alpha
                    pen=stim_pen,  # Border helps with aliasing on thin regions
                    movable=False
                )
                region.setZValue(-10)  # Behind trace
                main_plot.addItem(region)
                self._span_items.append(region)

        # Add zero line
        zero_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen('#666666', width=1, style=Qt.PenStyle.DashLine)
        )
        zero_line.setZValue(-5)
        main_plot.addItem(zero_line)
        self._span_items.append(zero_line)

        # Set title and labels
        main_plot.setTitle(title, color='#d4d4d4' if self._current_theme == 'dark' else '#000000')
        main_plot.setLabel('left', ylabel)
        main_plot.setLabel('bottom', 'Time (s)')

        # Restore or auto-scale view
        if prev_xlim is not None:
            main_plot.setXRange(prev_xlim[0], prev_xlim[1], padding=0)
            # Also restore Y if preserved
            if prev_ylim is not None:
                main_plot.setYRange(prev_ylim[0], prev_ylim[1], padding=0)
        else:
            main_plot.autoRange()

        # Store view
        self._store_view()

    def show_multi_grid(self, traces, title: str = "", max_points_per_trace: int = None):
        """
        Display multiple traces in a grid layout.

        Args:
            traces: List of (t, y, label) tuples
            title: Plot title
            max_points_per_trace: Max points per trace for downsampling
        """
        prev_xlim = self._last_grid["xlim"] if self._preserve_x else None

        # Clear and rebuild layout
        safe_clear_graphics_layout(self.graphics_layout)
        self._subplots = []

        n = len(traces)
        if n == 0:
            return

        # Get trace color
        trace_color = '#d4d4d4' if self._current_theme == 'dark' else '#000000'
        text_color = '#d4d4d4' if self._current_theme == 'dark' else '#000000'

        # Create subplots
        for i, (t, y, label) in enumerate(traces):
            plot = self.graphics_layout.addPlot(row=i, col=0)
            plot.showGrid(x=False, y=False)
            plot.setLabel('left', label)

            # Hide x-axis on all but bottom panel (they share x-axis via setXLink)
            if i < n - 1:
                plot.hideAxis('bottom')

            # Style axes (only style bottom axis if it's visible)
            if i == n - 1:
                plot.getAxis('bottom').setTextPen(text_color)
                plot.getAxis('bottom').setPen(text_color)
            plot.getAxis('left').setTextPen(text_color)
            plot.getAxis('left').setPen(text_color)

            # Set fixed Y-axis width to ensure all panels align visually
            # This is critical for x-axis synchronization - setXLink() only syncs data range,
            # not visual positioning. Different label widths would cause misalignment.
            plot.getAxis('left').setWidth(70)

            # Configure mouse behavior: X-axis only, shift required for pan
            self._configure_plot_mouse(plot)

            # Disable PyQtGraph's built-in ViewBox context menu
            plot.vb.setMenuEnabled(False)
            plot.setMenuEnabled(False)

            # Link x-axis to first plot
            if i > 0:
                plot.setXLink(self._subplots[0])

            # Downsample and plot
            t_ds, y_ds = self._downsample(t, y, max_points_per_trace)
            plot.plot(t_ds, y_ds, pen=pg.mkPen(trace_color, width=1))

            self._subplots.append(plot)

        # Set title on first plot
        if self._subplots:
            self._subplots[0].setTitle(title, color=text_color)

        # Set x label on last plot
        if self._subplots:
            self._subplots[-1].setLabel('bottom', 'Time (s)')

        # Restore view
        if prev_xlim is not None:
            self._subplots[0].setXRange(prev_xlim[0], prev_xlim[1], padding=0)

        # Update main plot reference
        self.plot_widget = self._subplots[0] if self._subplots else None
        self.ax_main = self.plot_widget

    # ------- Marker Updates -------
    def update_peaks(self, t_peaks, y_peaks, size=24):
        """Update peak markers (red dots)."""
        if t_peaks is None or y_peaks is None or len(t_peaks) == 0:
            self.clear_peaks()
            return

        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        if self._peak_scatter is not None:
            try:
                main_plot.removeItem(self._peak_scatter)
            except:
                pass

        self._peak_scatter = pg.ScatterPlotItem(
            x=np.asarray(t_peaks),
            y=np.asarray(y_peaks),
            size=size // 2,  # pyqtgraph sizes are different
            brush=pg.mkBrush(255, 0, 0),
            pen=None
        )
        self._peak_scatter.setZValue(10)
        main_plot.addItem(self._peak_scatter)

    def clear_peaks(self):
        """Remove peak markers."""
        if self._peak_scatter is not None:
            main_plot = self._get_main_plot()
            try:
                if main_plot:
                    main_plot.removeItem(self._peak_scatter)
            except:
                pass
            self._peak_scatter = None

    def update_breath_markers(self, t_on=None, y_on=None,
                              t_off=None, y_off=None,
                              t_exp=None, y_exp=None,
                              t_exoff=None, y_exoff=None,
                              size=30):
        """Update breath event markers."""
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        marker_size = size // 3  # Adjust for pyqtgraph

        def _update_scatter(attr_name, t, y, color, symbol):
            scatter = getattr(self, attr_name, None)
            if scatter is not None:
                try:
                    main_plot.removeItem(scatter)
                except:
                    pass

            if t is not None and y is not None and len(t) > 0:
                new_scatter = pg.ScatterPlotItem(
                    x=np.asarray(t),
                    y=np.asarray(y),
                    size=marker_size,
                    brush=pg.mkBrush(*color),
                    symbol=symbol,
                    pen=None
                )
                new_scatter.setZValue(8)
                main_plot.addItem(new_scatter)
                setattr(self, attr_name, new_scatter)
            else:
                setattr(self, attr_name, None)

        _update_scatter('_onset_scatter', t_on, y_on, (46, 204, 113), 't')  # Triangle up
        _update_scatter('_offset_scatter', t_off, y_off, (243, 156, 18), 't1')  # Triangle down
        _update_scatter('_expmin_scatter', t_exp, y_exp, (31, 120, 180), 's')  # Square
        _update_scatter('_expoff_scatter', t_exoff, y_exoff, (155, 89, 182), 'd')  # Diamond

    def clear_breath_markers(self):
        """Remove all breath markers."""
        main_plot = self._get_main_plot()
        for attr in ('_onset_scatter', '_offset_scatter', '_expmin_scatter', '_expoff_scatter'):
            scatter = getattr(self, attr, None)
            if scatter is not None:
                try:
                    if main_plot:
                        main_plot.removeItem(scatter)
                except:
                    pass
                setattr(self, attr, None)

    def update_sighs(self, t, y, size=90, color="#FFD700", edge=None, filled=True):
        """Update sigh markers (stars)."""
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        if self._sigh_scatter is not None:
            try:
                main_plot.removeItem(self._sigh_scatter)
            except:
                pass

        if t is None or y is None or len(t) == 0:
            self._sigh_scatter = None
            return

        self._sigh_scatter = pg.ScatterPlotItem(
            x=np.asarray(t),
            y=np.asarray(y),
            size=size // 5,
            brush=pg.mkBrush(color) if filled else None,
            pen=pg.mkPen(edge or color, width=2),
            symbol='star'
        )
        self._sigh_scatter.setZValue(15)
        main_plot.addItem(self._sigh_scatter)

    def clear_sighs(self):
        """Remove sigh markers."""
        if self._sigh_scatter is not None:
            main_plot = self._get_main_plot()
            try:
                if main_plot:
                    main_plot.removeItem(self._sigh_scatter)
            except:
                pass
            self._sigh_scatter = None

    # ------- Threshold Line -------
    def update_threshold_line(self, threshold_value):
        """Draw horizontal threshold line with drag support and histogram display."""
        main_plot = self._get_main_plot()

        # Clear existing threshold line
        if self._threshold_line is not None:
            try:
                if main_plot:
                    main_plot.removeItem(self._threshold_line)
            except:
                pass
            self._threshold_line = None

        if threshold_value is None or main_plot is None:
            return

        # Ensure main_plot has a valid ViewBox before proceeding
        if not hasattr(main_plot, 'vb') or main_plot.vb is None:
            print("[threshold] Warning: main_plot has no valid ViewBox, skipping threshold line")
            return

        # Create draggable threshold line
        self._threshold_line = pg.InfiniteLine(
            pos=threshold_value,
            angle=0,
            pen=pg.mkPen('red', width=1.5, style=Qt.PenStyle.DashLine),
            movable=True
        )
        self._threshold_line.setZValue(20)

        try:
            main_plot.addItem(self._threshold_line)
        except Exception as e:
            print(f"[threshold] Warning: Failed to add threshold line: {e}")
            self._threshold_line = None
            return

        # Store reference for callbacks
        plot_host = self

        # Connect drag signals
        def on_threshold_dragging():
            """Update histogram colors during drag. Show histogram if not yet visible."""
            if plot_host._threshold_line is not None:
                new_val = plot_host._threshold_line.value()
                if new_val > 0:
                    # Show histogram if not yet visible (first drag event)
                    if not hasattr(plot_host, '_histogram_items') or not plot_host._histogram_items:
                        print("[threshold] First drag event - showing histogram")
                        plot_host._show_threshold_histogram()
                    else:
                        # Update colors for existing histogram
                        plot_host._update_threshold_histogram_colors(new_val)

        def on_threshold_drag_finished():
            """Update main window threshold and clear histogram."""
            if plot_host._threshold_line is None:
                return
            new_val = plot_host._threshold_line.value()
            if new_val > 0:
                # Update main window threshold
                main_window = plot_host._find_main_window()
                if main_window:
                    main_window.peak_height_threshold = new_val
                    main_window.peak_prominence = new_val
                    print(f"[threshold] Updated to {new_val:.4f}")

                    # Enable detect button so user can re-run detection
                    if hasattr(main_window, 'ApplyPeakFindPushButton'):
                        main_window.ApplyPeakFindPushButton.setEnabled(True)

                    # Sync with Analysis Options dialog if open
                    try:
                        prom_dialog = getattr(main_window, 'prominence_dialog', None)
                        if prom_dialog and hasattr(prom_dialog, 'update_threshold_from_external'):
                            prom_dialog.update_threshold_from_external(new_val)
                    except:
                        pass

            # Clear histogram
            plot_host._clear_threshold_histogram()

        # sigDragged fires during drag, sigPositionChangeFinished when released
        self._threshold_line.sigDragged.connect(on_threshold_dragging)
        self._threshold_line.sigPositionChangeFinished.connect(on_threshold_drag_finished)

        # Show histogram on click (before drag starts)
        def on_threshold_clicked(line, ev):
            print("[threshold] Line clicked - showing histogram")
            plot_host._show_threshold_histogram()

        self._threshold_line.sigClicked.connect(on_threshold_clicked)

    def clear_threshold_line(self):
        """Remove threshold line and histogram."""
        if self._threshold_line is not None:
            main_plot = self._get_main_plot()
            try:
                if main_plot:
                    main_plot.removeItem(self._threshold_line)
            except:
                pass
            self._threshold_line = None
        self._clear_threshold_histogram()

    def _find_main_window(self):
        """Find the main application window."""
        widget = self
        while widget is not None:
            if hasattr(widget, 'state') and hasattr(widget, 'peak_height_threshold'):
                return widget
            widget = getattr(widget, 'parent', lambda: None)()
        return None

    def _show_threshold_histogram(self):
        """Show sideways histogram of peak heights on the main plot (like matplotlib version)."""
        import numpy as np

        print("[histogram] _show_threshold_histogram called")

        main_plot = self._get_main_plot()
        if main_plot is None:
            print("[histogram] No main plot available")
            return

        # Ensure we have a proper plot with ViewBox (not just GraphicsLayoutWidget)
        if not hasattr(main_plot, 'vb') or main_plot.vb is None:
            print("[histogram] main_plot has no valid ViewBox")
            return

        # Additional check: ensure main_plot is a PlotItem, not GraphicsLayoutWidget
        if not hasattr(main_plot, 'addItem'):
            print("[histogram] main_plot doesn't support addItem")
            return

        main_window = self._find_main_window()
        if main_window is None:
            print("[histogram] main_window not found")
            return
        if not hasattr(main_window, 'state'):
            print("[histogram] main_window has no 'state' attribute")
            return

        print(f"[histogram] Found main_window, has state: {hasattr(main_window, 'state')}")

        st = main_window.state

        # Collect peak heights from all_peak_heights if cached, otherwise from peaks
        if hasattr(main_window, 'all_peak_heights') and main_window.all_peak_heights is not None:
            peak_heights = main_window.all_peak_heights
        else:
            # Collect from detected peaks
            all_heights = []
            for s in range(len(st.peaks_by_sweep)):
                pks = st.peaks_by_sweep.get(s, None)
                if pks is not None and len(pks) > 0:
                    y_proc = main_window._get_processed_for(st.analyze_chan, s)
                    if y_proc is not None:
                        all_heights.extend(y_proc[pks])

            if not all_heights:
                return

            peak_heights = np.array(all_heights)
            main_window.all_peak_heights = peak_heights

        if len(peak_heights) == 0:
            return

        # Get histogram settings
        num_bins = getattr(main_window, 'histogram_num_bins', 200)
        percentile_cutoff = getattr(main_window, 'histogram_percentile_cutoff', 99)

        # Calculate percentile cutoff
        if percentile_cutoff < 100:
            percentile_val = np.percentile(peak_heights, percentile_cutoff)
            peaks_for_hist = peak_heights[peak_heights <= percentile_val]
            hist_range = (peaks_for_hist.min(), percentile_val)
        else:
            peaks_for_hist = peak_heights
            hist_range = None

        if len(peaks_for_hist) == 0:
            return

        # Get x-axis limits to position histogram at left edge
        x_range = main_plot.viewRange()[0]
        x_min = x_range[0]
        x_span = x_range[1] - x_range[0]

        # Get current threshold
        current_threshold = getattr(main_window, 'peak_height_threshold', None)
        if current_threshold is None:
            return

        # Create histogram
        counts, bins = np.histogram(peaks_for_hist, bins=num_bins, range=hist_range)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_height = bins[1] - bins[0]

        # Scale histogram to 10% of x-axis range
        max_count = np.max(counts)
        if max_count <= 0:
            return

        scale_factor = (0.1 * x_span) / max_count
        scaled_counts = counts * scale_factor

        # Cache data for fast color updates
        self._histogram_bin_centers = bin_centers
        self._histogram_scaled_counts = scaled_counts
        self._histogram_x_min = x_min
        self._histogram_bin_height = bin_height

        # Clear any existing histogram
        self._clear_threshold_histogram()

        # Draw histogram as horizontal bars (sideways histogram along Y-axis)
        # Each bar extends from x_min to x_min + count, at height bin_center
        # Create ALL bars once; during threshold drag we just update pen/brush (no recreate)
        self._histogram_items = []
        self._histogram_bar_centers = []  # bin center for each bar item

        gray_pen = pg.mkPen(128, 128, 128, 200, width=0.5)
        gray_brush = pg.mkBrush(128, 128, 128, 180)
        red_pen = pg.mkPen(220, 60, 60, 200, width=0.5)
        red_brush = pg.mkBrush(220, 60, 60, 180)

        for y_center, width in zip(bin_centers, scaled_counts):
            if width > 0:
                y_bottom = y_center - bin_height / 2
                y_top = y_center + bin_height / 2
                x_pts = [x_min, x_min + width, x_min + width, x_min, x_min]
                y_pts = [y_bottom, y_bottom, y_top, y_top, y_bottom]
                is_above = y_center >= current_threshold
                bar = pg.PlotCurveItem(x_pts, y_pts,
                                      pen=red_pen if is_above else gray_pen,
                                      brush=red_brush if is_above else gray_brush,
                                      fillLevel=None)
                bar.setZValue(99)
                main_plot.addItem(bar)
                self._histogram_items.append(bar)
                self._histogram_bar_centers.append(y_center)

        print(f"[histogram] Created histogram with {len(self._histogram_items)} bars")

    def _update_threshold_histogram_colors(self, new_threshold):
        """Update histogram bar colors in-place based on new threshold (no object recreation)."""
        if not hasattr(self, '_histogram_items') or not self._histogram_items:
            return
        if not hasattr(self, '_histogram_bar_centers') or not self._histogram_bar_centers:
            return

        gray_pen = pg.mkPen(128, 128, 128, 200, width=0.5)
        gray_brush = pg.mkBrush(128, 128, 128, 180)
        red_pen = pg.mkPen(220, 60, 60, 200, width=0.5)
        red_brush = pg.mkBrush(220, 60, 60, 180)

        for bar, y_center in zip(self._histogram_items, self._histogram_bar_centers):
            if y_center >= new_threshold:
                bar.setPen(red_pen)
                bar.setBrush(red_brush)
            else:
                bar.setPen(gray_pen)
                bar.setBrush(gray_brush)

    def _clear_threshold_histogram(self):
        """Remove threshold histogram from plot."""
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        if hasattr(self, '_histogram_items') and self._histogram_items:
            for item in self._histogram_items:
                try:
                    main_plot.removeItem(item)
                except:
                    pass
            self._histogram_items = []

        # Clear cached data
        self._histogram_bin_centers = None
        self._histogram_scaled_counts = None
        self._histogram_x_min = None

    # ------- Y2 Axis (Secondary Y-axis) -------
    def _get_main_plot(self):
        """Get the main plot widget safely, avoiding stale references."""
        # Priority: ax_main > first subplot > plot_widget
        # Also validate that the plot has a proper ViewBox to avoid 'autoRangeEnabled' errors
        candidates = [self.ax_main]
        if self._subplots and len(self._subplots) > 0:
            candidates.append(self._subplots[0])
        if self.plot_widget is not None:
            candidates.append(self.plot_widget)

        for plot in candidates:
            if plot is not None and hasattr(plot, 'vb') and plot.vb is not None:
                return plot

        # Fallback to any non-None candidate (may not have ViewBox yet during init)
        for plot in candidates:
            if plot is not None:
                return plot

        return None

    def add_or_update_y2(self, t, y2, label: str = "Y2",
                         max_points: int = None, color: str = "#39FF14"):
        """Add or update secondary Y axis - simplified overlay approach."""
        t_ds, y_ds = self._downsample(t, y2, max_points)

        # Get the main plot safely
        main_plot = self._get_main_plot()
        if main_plot is None:
            print("[PyQtGraph] Warning: No main plot available for Y2")
            return

        # Remove existing Y2 line only (keep it simple)
        if self._y2_line is not None:
            try:
                main_plot.removeItem(self._y2_line)
            except:
                pass
            self._y2_line = None

        # Convert hex color to RGB
        qcolor = QColor(color)
        pen = pg.mkPen(qcolor.red(), qcolor.green(), qcolor.blue(), width=1.5)

        # Plot Y2 as an overlay line on the main plot
        # Note: This shares the Y-axis scale, but is simpler and more reliable
        self._y2_line = main_plot.plot(t_ds, y_ds, pen=pen, name=label)
        self._y2_line.setZValue(5)

    def clear_y2(self):
        """Remove Y2 axis line."""
        if self._y2_line is not None:
            try:
                main_plot = self._get_main_plot()
                if main_plot:
                    main_plot.removeItem(self._y2_line)
            except:
                pass
            self._y2_line = None

    # ------- Region Overlays -------
    def update_region_overlays(self, t, eupnea_mask, apnea_mask,
                               outlier_mask=None, failure_mask=None,
                               sniff_regions=None, eupnea_regions=None, state=None):
        """Add region overlays for eupnea, apnea, etc."""
        self.clear_region_overlays()

        # Get display modes
        eupnea_shade = getattr(state, 'eupnea_use_shade', False) if state else False
        sniffing_shade = getattr(state, 'sniffing_use_shade', True) if state else True
        apnea_shade = getattr(state, 'apnea_use_shade', False) if state else False

        # Draw eupnea regions (green)
        regions_to_draw = None
        if eupnea_regions is not None and len(eupnea_regions) > 0:
            regions_to_draw = eupnea_regions
        elif eupnea_mask is not None and len(eupnea_mask) == len(t):
            regions_to_draw = self._extract_regions(t, eupnea_mask)

        if regions_to_draw:
            self._draw_regions(regions_to_draw, '#2e7d32', eupnea_shade)

        # Draw sniffing regions (purple)
        if sniff_regions is not None and len(sniff_regions) > 0:
            self._draw_regions(sniff_regions, 'purple', sniffing_shade)

        # Draw apnea regions (red)
        if apnea_mask is not None and len(apnea_mask) == len(t):
            apnea_regions = self._extract_regions(t, apnea_mask)
            self._draw_regions(apnea_regions, 'red', apnea_shade)

        # Draw outlier regions (orange)
        if outlier_mask is not None and len(outlier_mask) == len(t):
            outlier_regions = self._extract_regions(t, outlier_mask)
            self._draw_regions(outlier_regions, '#FFA500', True)

    def _draw_regions(self, regions, color, use_shade):
        """Draw regions as shaded areas."""
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        qcolor = QColor(color)
        for start_t, end_t in regions:
            if use_shade:
                region = pg.LinearRegionItem(
                    values=[start_t, end_t],
                    brush=pg.mkBrush(qcolor.red(), qcolor.green(), qcolor.blue(), 60),
                    movable=False
                )
                region.setZValue(-8)
                main_plot.addItem(region)
                self._region_overlays.append(region)

    def _extract_regions(self, t, mask):
        """Extract continuous regions from binary mask."""
        regions = []
        if len(mask) == 0 or len(t) == 0:
            return regions

        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            return regions

        diff_mask = np.diff(mask.astype(int))
        starts = np.where(diff_mask == 1)[0] + 1
        ends = np.where(diff_mask == -1)[0] + 1

        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        for start_idx, end_idx in zip(starts, ends):
            start_t = float(t[int(start_idx)])
            end_t = float(t[min(int(end_idx) - 1, len(t) - 1)])
            if end_t > start_t:
                regions.append((start_t, end_t))

        return regions

    def clear_region_overlays(self):
        """Remove all region overlays."""
        main_plot = self._get_main_plot()
        for item in self._region_overlays:
            try:
                if main_plot:
                    main_plot.removeItem(item)
            except:
                pass
        self._region_overlays.clear()

    # ------- Click and Drag Handling -------
    def set_click_callback(self, fn):
        """Set callback for plot clicks: fn(xdata, ydata, event).
        Also disables pan/zoom to allow click-based editing."""
        self._external_click_cb = fn
        # Disable mouse pan/drag when edit callback is active
        if fn is not None:
            self._set_mouse_mode_edit()
        else:
            self._set_mouse_mode_normal()

    def clear_click_callback(self):
        """Remove click callback and restore normal mouse behavior."""
        self._external_click_cb = None
        self._set_mouse_mode_normal()

    def set_drag_callback(self, fn):
        """Set callback for drag events: fn(event_type, xdata, ydata, event).
        event_type is 'press', 'move', or 'release'.
        Also disables pan/zoom to allow drag-based editing."""
        self._drag_callback = fn
        if fn is not None:
            self._set_mouse_mode_edit()
            # Install event filter for mouse press/release detection
            self.graphics_layout.scene().installEventFilter(self)
        else:
            # Only restore normal mode if no click callback either
            if self._external_click_cb is None:
                self._set_mouse_mode_normal()
            self.graphics_layout.scene().removeEventFilter(self)
            self._clear_drag_visual()

    def clear_drag_callback(self):
        """Remove drag callback."""
        self._drag_callback = None
        try:
            self.graphics_layout.scene().removeEventFilter(self)
        except:
            pass
        self._clear_drag_visual()
        # Only restore normal mode if no click callback either
        if self._external_click_cb is None:
            self._set_mouse_mode_normal()

    def eventFilter(self, obj, event):
        """Event filter for capturing mouse press/release for drag operations."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent

        if self._drag_callback is None:
            return False

        main_plot = self._get_main_plot()
        if main_plot is None:
            return False

        if event.type() == QEvent.Type.GraphicsSceneMousePress:
            # Only handle left button for drag
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.scenePos()
                if main_plot.sceneBoundingRect().contains(pos):
                    # Convert to data coordinates
                    mouse_point = main_plot.vb.mapSceneToView(pos)
                    x_data = mouse_point.x()
                    y_data = mouse_point.y()

                    # Create wrapped event
                    wrapped = _MatplotlibCompatEvent(event, main_plot, x_data, y_data)

                    # Call drag callback FIRST to check if it wants to handle specially
                    # (e.g., Ctrl+Shift+click for full sweep omit should NOT start a drag)
                    result = self._drag_callback('press', x_data, y_data, wrapped)

                    # If callback returns 'handled', don't start drag operation
                    if result == 'handled':
                        # Explicitly clear any drag state to prevent accidental drags
                        self._dragging = False
                        self._drag_start_pos = None
                        self._clear_drag_visual()
                        return True  # Consume the event but don't start drag

                    # Normal case - start drag operation
                    self._drag_start_pos = pos
                    self._dragging = True
                    return True  # Consume the event

        elif event.type() == QEvent.Type.GraphicsSceneMouseRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._dragging:
                self._dragging = False
                pos = event.scenePos()
                # Convert to data coordinates
                mouse_point = main_plot.vb.mapSceneToView(pos)
                x_data = mouse_point.x()
                y_data = mouse_point.y()
                # Create wrapped event
                wrapped = _MatplotlibCompatEvent(event, main_plot, x_data, y_data)
                self._drag_callback('release', x_data, y_data, wrapped)
                self._drag_start_pos = None
                self._clear_drag_visual()
                return True  # Consume the event

        return False  # Don't consume other events

    def _on_scene_mouse_moved(self, pos):
        """Handle scene mouse movement for drag operations."""
        if not self._dragging or self._drag_callback is None:
            return

        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        # pos is already in scene coordinates
        if main_plot.sceneBoundingRect().contains(pos):
            mouse_point = main_plot.vb.mapSceneToView(pos)
            x_data = mouse_point.x()
            y_data = mouse_point.y()
            # Create a simple event wrapper (no Qt event available here)
            class SimpleDragEvent:
                def __init__(self, x, y, plot):
                    self.xdata = x
                    self.ydata = y
                    self.inaxes = plot
                    self.button = 1  # Left button
            wrapped = SimpleDragEvent(x_data, y_data, main_plot)
            self._drag_callback('move', x_data, y_data, wrapped)

    def add_drag_visual(self, x_start, x_end, color=(255, 255, 0, 50)):
        """Add or update a visual selection rectangle during drag."""
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        self._clear_drag_visual()

        # Create selection region
        self._drag_visual = pg.LinearRegionItem(
            values=[x_start, x_end],
            brush=pg.mkBrush(*color),
            pen=pg.mkPen(color[:3], width=1),
            movable=False
        )
        self._drag_visual.setZValue(100)  # On top of everything
        main_plot.addItem(self._drag_visual)

    def _clear_drag_visual(self):
        """Remove drag visual indicator."""
        if self._drag_visual is not None:
            try:
                main_plot = self._get_main_plot()
                if main_plot:
                    main_plot.removeItem(self._drag_visual)
            except:
                pass
            self._drag_visual = None

    # ------- Keyboard Event Handling -------
    def set_key_callback(self, fn):
        """Set callback for keyboard events: fn(key, modifiers).
        key is a string like 'enter', 'escape', 'left', 'right', etc.
        modifiers is a dict like {'shift': bool, 'ctrl': bool, 'alt': bool}
        """
        self._key_callback = fn
        if fn is not None:
            self.setFocus()  # Take focus to receive key events

    def clear_key_callback(self):
        """Remove keyboard callback."""
        self._key_callback = None

    def keyPressEvent(self, event):
        """Handle keyboard events and forward to callback."""
        from PyQt6.QtCore import Qt as QtCore_Qt

        # Ctrl+E: Export plot
        if (event.modifiers() & QtCore_Qt.KeyboardModifier.ControlModifier and
            event.key() == QtCore_Qt.Key.Key_E):
            self.show_export_dialog()
            event.accept()
            return

        if self._key_callback is not None:
            # Map Qt key to string
            key_map = {
                QtCore_Qt.Key.Key_Return: 'enter',
                QtCore_Qt.Key.Key_Enter: 'enter',
                QtCore_Qt.Key.Key_Escape: 'escape',
                QtCore_Qt.Key.Key_Left: 'left',
                QtCore_Qt.Key.Key_Right: 'right',
                QtCore_Qt.Key.Key_Up: 'up',
                QtCore_Qt.Key.Key_Down: 'down',
                QtCore_Qt.Key.Key_Space: 'space',
                QtCore_Qt.Key.Key_Delete: 'delete',
                QtCore_Qt.Key.Key_Backspace: 'backspace',
            }

            key = key_map.get(event.key(), None)
            if key is None:
                # Try to get character for letter keys
                text = event.text()
                if text and text.isalpha():
                    key = text.lower()

            if key is not None:
                modifiers = {
                    'shift': bool(event.modifiers() & QtCore_Qt.KeyboardModifier.ShiftModifier),
                    'ctrl': bool(event.modifiers() & QtCore_Qt.KeyboardModifier.ControlModifier),
                    'alt': bool(event.modifiers() & QtCore_Qt.KeyboardModifier.AltModifier),
                }
                self._key_callback(key, modifiers)
                event.accept()
                return

        # Pass to parent if not handled
        super().keyPressEvent(event)

    def _configure_plot_mouse(self, plot):
        """Configure mouse behavior for a plot.

        Mouse controls:
        - Wheel: Zoom X-axis
        - Shift+Wheel: Pan X-axis (horizontal scroll)
        - Click+drag on plot: No action (prevents accidental pan)
        - Shift+Click+drag: Pan X-axis
        - Click+drag on X-axis: Pan X-axis
        - Click+drag on Y-axis: Pan Y-axis (scale adjustment)
        """
        from PyQt6.QtCore import Qt as QtCore_Qt

        vb = plot.vb

        # Only allow X-axis mouse interaction (no Y zooming/panning) by default
        vb.setMouseEnabled(x=True, y=False)

        # Store original mouseDragEvent - need to bind to avoid closure issues
        original_drag = vb.mouseDragEvent.__func__ if hasattr(vb.mouseDragEvent, '__func__') else vb.mouseDragEvent

        def shift_only_drag(ev, axis=None, _vb=vb, _orig=original_drag):
            """Only allow drag if Shift is held."""
            modifiers = ev.modifiers() if hasattr(ev, 'modifiers') else QtCore_Qt.KeyboardModifier.NoModifier
            if modifiers & QtCore_Qt.KeyboardModifier.ShiftModifier:
                # Call original with proper binding
                if hasattr(_orig, '__self__'):
                    _orig(ev, axis)
                else:
                    _orig(_vb, ev, axis)
            else:
                # Accept the event but don't do anything - this prevents pan
                ev.accept()

        vb.mouseDragEvent = shift_only_drag

        # Override wheel event: normal = zoom X, shift = pan X
        original_wheel = vb.wheelEvent

        def smart_wheel(ev, axis=None, _vb=vb, _orig=original_wheel):
            """Wheel = zoom X-axis, Shift+Wheel = pan X-axis (horizontal scroll)."""
            modifiers = ev.modifiers() if hasattr(ev, 'modifiers') else QtCore_Qt.KeyboardModifier.NoModifier

            if modifiers & QtCore_Qt.KeyboardModifier.ShiftModifier:
                # Shift+Wheel: Pan horizontally
                delta = ev.delta()
                # Get current view range
                view_range = _vb.viewRange()
                x_min, x_max = view_range[0]
                x_span = x_max - x_min

                # Pan amount: 10% of visible range per wheel notch
                # delta is typically 120 per notch
                pan_fraction = 0.1 * (delta / 120.0)
                pan_amount = x_span * pan_fraction

                # Pan the view (negative because wheel up = scroll left feels natural)
                _vb.translateBy(x=-pan_amount, y=0)
                ev.accept()
            else:
                # Normal wheel: Zoom X-axis only
                # Force axis=0 (X-axis) regardless of mouse position
                _orig(ev, axis=0)

        vb.wheelEvent = smart_wheel

        # Enable axis dragging for panning
        # Click+drag on X-axis = pan X, click+drag on Y-axis = pan Y
        self._setup_axis_drag(plot)

    def _setup_axis_drag(self, plot):
        """Setup drag-to-pan on axis items.

        Allows user to click and drag on the X or Y axis to pan the view,
        providing an intuitive way to navigate without keyboard modifiers.
        """
        from PyQt6.QtCore import Qt as QtCore_Qt
        from PyQt6.QtGui import QCursor

        vb = plot.vb

        # Get axis items
        x_axis = plot.getAxis('bottom')
        y_axis = plot.getAxis('left')

        # Track drag state
        drag_state = {'axis': None, 'start_pos': None, 'start_range': None}

        # Store original event handlers
        original_x_mouse_drag = x_axis.mouseDragEvent if hasattr(x_axis, 'mouseDragEvent') else None
        original_y_mouse_drag = y_axis.mouseDragEvent if hasattr(y_axis, 'mouseDragEvent') else None

        def x_axis_drag(ev, _axis=x_axis, _vb=vb, _state=drag_state):
            """Handle drag on X-axis to pan horizontally."""
            if ev.button() != QtCore_Qt.MouseButton.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                _state['axis'] = 'x'
                _state['start_pos'] = ev.pos()
                _state['start_range'] = _vb.viewRange()[0]
                _axis.setCursor(QCursor(QtCore_Qt.CursorShape.ClosedHandCursor))
                ev.accept()
            elif ev.isFinish():
                _state['axis'] = None
                _axis.setCursor(QCursor(QtCore_Qt.CursorShape.SizeHorCursor))
                ev.accept()
            else:
                if _state['axis'] == 'x' and _state['start_range'] is not None:
                    # Calculate pan amount based on drag distance
                    delta = ev.pos() - _state['start_pos']
                    # Convert pixel delta to data coordinates
                    view_range = _state['start_range']
                    x_span = view_range[1] - view_range[0]
                    # Get axis width in pixels
                    axis_width = _axis.geometry().width()
                    if axis_width > 0:
                        pan_amount = (delta.x() / axis_width) * x_span
                        new_min = view_range[0] - pan_amount
                        new_max = view_range[1] - pan_amount
                        _vb.setXRange(new_min, new_max, padding=0)
                    ev.accept()

        def y_axis_drag(ev, _axis=y_axis, _vb=vb, _state=drag_state):
            """Handle drag on Y-axis to pan vertically."""
            if ev.button() != QtCore_Qt.MouseButton.LeftButton:
                ev.ignore()
                return

            if ev.isStart():
                _state['axis'] = 'y'
                _state['start_pos'] = ev.pos()
                _state['start_range'] = _vb.viewRange()[1]
                _axis.setCursor(QCursor(QtCore_Qt.CursorShape.ClosedHandCursor))
                ev.accept()
            elif ev.isFinish():
                _state['axis'] = None
                _axis.setCursor(QCursor(QtCore_Qt.CursorShape.SizeVerCursor))
                ev.accept()
            else:
                if _state['axis'] == 'y' and _state['start_range'] is not None:
                    # Calculate pan amount based on drag distance
                    delta = ev.pos() - _state['start_pos']
                    # Convert pixel delta to data coordinates
                    view_range = _state['start_range']
                    y_span = view_range[1] - view_range[0]
                    # Get axis height in pixels
                    axis_height = _axis.geometry().height()
                    if axis_height > 0:
                        # Invert because Y increases upward in data but downward in pixels
                        pan_amount = (delta.y() / axis_height) * y_span
                        new_min = view_range[0] + pan_amount
                        new_max = view_range[1] + pan_amount
                        _vb.setYRange(new_min, new_max, padding=0)
                    ev.accept()

        # Set custom drag handlers
        x_axis.mouseDragEvent = x_axis_drag
        y_axis.mouseDragEvent = y_axis_drag

        # Set initial cursors to indicate draggability
        x_axis.setCursor(QCursor(QtCore_Qt.CursorShape.SizeHorCursor))
        y_axis.setCursor(QCursor(QtCore_Qt.CursorShape.SizeVerCursor))

    def _set_mouse_mode_edit(self):
        """Disable pan/zoom for click-based editing."""
        for plot in self._subplots:
            try:
                # Disable left-button pan and right-button zoom
                plot.vb.setMouseEnabled(x=False, y=False)
            except:
                pass

    def _set_mouse_mode_normal(self):
        """Re-enable normal pan/zoom behavior."""
        for plot in self._subplots:
            try:
                plot.vb.setMouseEnabled(x=True, y=False)  # Keep Y disabled
            except:
                pass

    def _on_mouse_clicked(self, event):
        """Handle mouse click events."""
        from PyQt6.QtCore import Qt as QtCore_Qt

        pos = event.scenePos()

        # Check for right-click when NOT in edit mode - show context menu
        if event.button() == QtCore_Qt.MouseButton.RightButton and not self._edit_mode_active:
            # Find which plot was clicked
            clicked_plot = self._find_plot_at_pos(pos)
            if clicked_plot is not None:
                self._show_context_menu(event, clicked_plot)
            return

        # For left/middle clicks, pass to external callback
        if self._external_click_cb is None:
            return

        # Get the main plot for coordinate mapping
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        # Get click position in data coordinates
        if main_plot.sceneBoundingRect().contains(pos):
            mouse_point = main_plot.vb.mapSceneToView(pos)
            x_data = mouse_point.x()
            y_data = mouse_point.y()

            # Create matplotlib-compatible event wrapper
            wrapped_event = _MatplotlibCompatEvent(event, main_plot, x_data, y_data)
            self._external_click_cb(x_data, y_data, wrapped_event)

    def _find_plot_at_pos(self, scene_pos):
        """Find which subplot was clicked."""
        for plot in self._subplots:
            if plot.sceneBoundingRect().contains(scene_pos):
                return plot
        return None

    def _show_context_menu(self, event, plot):
        """Show context menu for plot panel."""
        from PyQt6.QtWidgets import QMenu, QApplication
        from PyQt6.QtGui import QCursor

        menu = QMenu()

        # Get the plot's label (channel name)
        try:
            plot_label = plot.titleLabel.text if hasattr(plot, 'titleLabel') else None
            if not plot_label:
                # Try to get from axis label
                left_axis = plot.getAxis('left')
                if left_axis:
                    plot_label = left_axis.labelText
        except:
            plot_label = None

        # Auto Scale Y for this panel
        action_auto_y = menu.addAction("Auto Scale Y")
        action_auto_y.setToolTip("Scale Y-axis to fit visible data")

        # Auto Scale All (all panels)
        if len(self._subplots) > 1:
            action_auto_all = menu.addAction("Auto Scale All Y")
            action_auto_all.setToolTip("Scale Y-axis on all panels to fit visible data")
        else:
            action_auto_all = None

        menu.addSeparator()

        # Reset View (full auto range)
        action_reset = menu.addAction("Reset View")
        action_reset.setToolTip("Reset to show all data")

        menu.addSeparator()

        # Performance mode submenu
        perf_menu = menu.addMenu("Performance Mode")
        _mw = self._find_main_window()
        pm = _mw.plot_manager if _mw and hasattr(_mw, 'plot_manager') else None
        if pm:
            override = pm._downsample_override
            action_perf_auto = perf_menu.addAction("Auto (Recommended)")
            action_perf_auto.setCheckable(True)
            action_perf_auto.setChecked(override is None)

            action_perf_on = perf_menu.addAction("Fast (Downsample On)")
            action_perf_on.setCheckable(True)
            action_perf_on.setChecked(override is True)

            action_perf_off = perf_menu.addAction("Full Resolution (Downsample Off)")
            action_perf_off.setCheckable(True)
            action_perf_off.setChecked(override is False)

            last_ms = pm._last_redraw_ms
            if last_ms > 0:
                perf_menu.addSeparator()
                info_action = perf_menu.addAction(f"Last redraw: {last_ms:.0f}ms")
                info_action.setEnabled(False)
        else:
            action_perf_auto = action_perf_on = action_perf_off = None

        menu.addSeparator()

        # Export Plot
        action_export = menu.addAction("Export Plot...")
        action_export.setToolTip("Export plot to PDF, SVG, or PNG (vector graphics)")

        # Execute menu at cursor position
        action = menu.exec(QCursor.pos())

        if action == action_auto_y:
            self._auto_scale_y_for_plot(plot)
        elif action == action_auto_all:
            self._auto_scale_y_all_panels()
        elif action == action_reset:
            plot.autoRange()
        elif action == action_export:
            self.show_export_dialog()
        elif pm and action == action_perf_auto:
            pm._downsample_override = None
            msg = "Performance mode: auto"
            if hasattr(_mw, '_log_status_message'):
                _mw._log_status_message(msg, 4000)
            pm.redraw_main_plot()
        elif pm and action == action_perf_on:
            pm._downsample_override = True
            msg = "Performance mode: fast (peak-preserving downsampling)"
            if hasattr(_mw, '_log_status_message'):
                _mw._log_status_message(msg, 4000)
            pm.redraw_main_plot()
        elif pm and action == action_perf_off:
            pm._downsample_override = False
            msg = "Performance mode: full resolution"
            if hasattr(_mw, '_log_status_message'):
                _mw._log_status_message(msg, 4000)
            pm.redraw_main_plot()

    def _auto_scale_y_for_plot(self, plot):
        """Auto-scale Y-axis for a specific plot to fit visible X range data."""
        try:
            # Get current X view range
            view_range = plot.viewRange()
            x_min, x_max = view_range[0]

            # Find data items in this plot
            for item in plot.listDataItems():
                if hasattr(item, 'getData'):
                    x_data, y_data = item.getData()
                    if x_data is None or y_data is None:
                        continue

                    # Find indices within visible X range
                    mask = (x_data >= x_min) & (x_data <= x_max)
                    if not np.any(mask):
                        continue

                    visible_y = y_data[mask]
                    if len(visible_y) == 0:
                        continue

                    # Calculate Y range with small padding
                    y_min = np.nanmin(visible_y)
                    y_max = np.nanmax(visible_y)
                    y_range = y_max - y_min
                    padding = y_range * 0.05 if y_range > 0 else 0.1

                    # Set Y range, keeping X range fixed
                    plot.setYRange(y_min - padding, y_max + padding, padding=0)
                    print(f"[PyQtGraph] Auto-scaled Y: [{y_min:.3f}, {y_max:.3f}]")
                    return

        except Exception as e:
            print(f"[PyQtGraph] Auto-scale Y error: {e}")

    def _auto_scale_y_all_panels(self):
        """Auto-scale Y-axis for all panels."""
        for plot in self._subplots:
            self._auto_scale_y_for_plot(plot)

    # ------- Toolbar Compatibility -------
    def set_toolbar_callback(self, callback):
        """Compatibility method - not used in pyqtgraph."""
        pass

    def turn_off_toolbar_modes(self):
        """Compatibility method - not used in pyqtgraph."""
        pass

    # ------- Edit Mode Control -------
    def set_edit_mode(self, active: bool):
        """Enable/disable edit mode. When active, right-click is used for editing, not context menu."""
        self._edit_mode_active = active
        # Context menu is always disabled (set in __init__), but this flag
        # can be used for other edit-mode-specific behaviors if needed

    def is_edit_mode_active(self) -> bool:
        """Check if edit mode is currently active."""
        return self._edit_mode_active

    # ------- Helper Methods -------
    def _downsample(self, t, y, max_points):
        """Downsample data using min-max envelope to preserve visual peaks/valleys.

        For each bucket, keeps both the min and max points, ensuring that peaks and
        valleys are never missed (unlike naive stride which can skip extrema).
        Falls back to no downsampling if data fits within max_points.
        """
        t = np.asarray(t)
        y = np.asarray(y)
        n = len(t)
        if max_points is None or max_points <= 0 or n <= max_points:
            return t, y

        # Min-max envelope: split into buckets, keep min and max from each
        # This gives 2 points per bucket, so use half as many buckets
        n_buckets = max(1, max_points // 2)
        # Compute bucket boundaries as integer indices
        boundaries = np.linspace(0, n, n_buckets + 1, dtype=np.intp)

        # Pre-allocate output (2 points per bucket + first and last)
        out_idx = np.empty(n_buckets * 2 + 2, dtype=np.intp)
        out_idx[0] = 0  # Always keep first point
        pos = 1

        for b in range(n_buckets):
            start = boundaries[b]
            end = boundaries[b + 1]
            if start >= end:
                continue

            seg = y[start:end]
            i_min = start + np.argmin(seg)
            i_max = start + np.argmax(seg)

            # Add in time order to preserve correct line drawing
            if i_min <= i_max:
                out_idx[pos] = i_min
                out_idx[pos + 1] = i_max
            else:
                out_idx[pos] = i_max
                out_idx[pos + 1] = i_min
            pos += 2

        out_idx[pos] = n - 1  # Always keep last point
        pos += 1

        # np.unique sorts and deduplicates — correct since indices are in ascending order
        indices = np.unique(out_idx[:pos])

        return t[indices], y[indices]

    def _clear_all_items(self):
        """Clear all plot items."""
        self.clear_peaks()
        self.clear_breath_markers()
        self.clear_sighs()
        self.clear_threshold_line()
        self.clear_y2()
        self.clear_region_overlays()

        # Use _get_main_plot() to avoid stale references after layout.clear()
        main_plot = self._get_main_plot()

        # Clear spans
        for item in self._span_items:
            try:
                if main_plot:
                    main_plot.removeItem(item)
            except:
                pass
        self._span_items.clear()

        # Clear main trace
        if self._main_trace is not None:
            try:
                if main_plot:
                    main_plot.removeItem(self._main_trace)
            except:
                pass
            self._main_trace = None

    def _store_view(self):
        """Store current view state."""
        try:
            main_plot = self._get_main_plot()
            if main_plot is not None:
                view_range = main_plot.viewRange()
                self._last_single["xlim"] = tuple(view_range[0])
                self._last_single["ylim"] = tuple(view_range[1])
        except:
            pass

    # ------- Compatibility Methods -------
    def clear(self):
        """Clear all plots - compatibility with matplotlib figure.clear()."""
        self._clear_all_items()
        # Clear and rebuild graphics layout for multi-panel support
        safe_clear_graphics_layout(self.graphics_layout)
        self._subplots = []
        # Recreate main plot widget
        self.plot_widget = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_widget.showGrid(x=False, y=False)
        self.ax_main = self.plot_widget
        self._subplots = [self.plot_widget]
        # Configure mouse behavior: X-axis only, shift required for pan
        self._configure_plot_mouse(self.plot_widget)
        # Disable PyQtGraph's built-in ViewBox context menu
        self.plot_widget.vb.setMenuEnabled(False)
        self.plot_widget.setMenuEnabled(False)
        # Reconnect click handler
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def add_subplot(self, gs_item):
        """Compatibility method for matplotlib add_subplot - returns a PlotItem.

        Note: This is a simplified compatibility layer. PyQtGraph doesn't use
        GridSpec, so we just add plots to the layout sequentially.
        """
        row = len(self._subplots)

        # Hide x-axis on previous bottom panel (it's no longer the bottom)
        if self._subplots:
            self._subplots[-1].hideAxis('bottom')

        plot = self.graphics_layout.addPlot(row=row, col=0)
        plot.showGrid(x=False, y=False)
        # Link x-axis to first plot for synchronized panning
        if self._subplots:
            plot.setXLink(self._subplots[0])
        # Configure mouse behavior: X-axis only, shift required for pan
        self._configure_plot_mouse(plot)
        self._subplots.append(plot)
        return plot

    def draw_idle(self):
        """Compatibility method - pyqtgraph updates automatically."""
        pass

    def push_current(self):
        """Compatibility method for toolbar home button."""
        pass

    @property
    def axes(self):
        """Return list of axes for compatibility."""
        return self._subplots

    def tight_layout(self, *args, **kwargs):
        """Compatibility method - pyqtgraph handles layout automatically."""
        pass

    def subplots_adjust(self, *args, **kwargs):
        """Compatibility method - pyqtgraph handles layout automatically."""
        pass

    # ------- Vector Export Methods -------
    def export_to_file(self, filepath: str, width: int = None, height: int = None):
        """Export the current plot to a file (PDF, SVG, or PNG).

        Args:
            filepath: Output file path. Extension determines format:
                     .pdf - Vector PDF (best for publications, all elements as vectors)
                     .svg - Vector SVG (editable in Illustrator/Inkscape)
                     .png - Raster PNG (high resolution)
            width: Output width in pixels (for PNG) or points (for PDF/SVG).
                   If None, uses current widget size.
            height: Output height in pixels/points. If None, uses current widget size.

        Returns:
            True if export succeeded, False otherwise.
        """
        from pathlib import Path

        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        # Get the graphics layout (contains all subplots)
        scene = self.graphics_layout.scene()
        if scene is None:
            print("[Export] Error: No scene to export")
            return False

        # Determine size - use actual scene size for accurate export
        source_rect = scene.sceneRect()
        if width is None:
            width = source_rect.width()
        if height is None:
            height = source_rect.height()

        try:
            if ext == '.pdf':
                # Direct PDF export using Qt's PDF writer - ALL elements as vectors
                return self._export_pdf_vector(filepath, scene, source_rect, width, height)

            elif ext == '.svg':
                # SVG export using pyqtgraph's exporter
                import pyqtgraph.exporters as exporters
                exporter = exporters.SVGExporter(scene)
                exporter.parameters()['width'] = width
                exporter.export(str(filepath))
                print(f"[Export] Saved SVG: {filepath}")
                return True

            elif ext == '.png':
                # High-resolution PNG export
                return self._export_png_hires(filepath, scene, source_rect, width, height)

            else:
                print(f"[Export] Unsupported format: {ext}. Use .pdf, .svg, or .png")
                return False

        except Exception as e:
            print(f"[Export] Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _export_pdf_vector(self, filepath, scene, source_rect, width, height):
        """Export to PDF with all elements as true vectors by manually drawing data."""
        from PyQt6.QtGui import QPainter, QPageSize, QPdfWriter, QPen, QColor, QPainterPath, QBrush
        from PyQt6.QtCore import QSizeF, QRectF, QMarginsF, QPointF, Qt
        import pyqtgraph as pg

        # Get page dimensions
        page_width = source_rect.width()
        page_height = source_rect.height()

        # Create PDF writer
        pdf_writer = QPdfWriter(str(filepath))
        pdf_writer.setResolution(72)  # 72 DPI = 1 point per pixel

        page_size = QPageSize(QSizeF(page_width, page_height), QPageSize.Unit.Point)
        pdf_writer.setPageSize(page_size)
        pdf_writer.setPageMargins(QMarginsF(0, 0, 0, 0))

        # Create painter
        painter = QPainter(pdf_writer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Fill background
        bg_color = self.graphics_layout.backgroundBrush().color()
        painter.fillRect(QRectF(0, 0, page_width, page_height), bg_color)

        # For each subplot, manually draw the data
        for plot_idx, plot in enumerate(self._subplots):
            # Get plot geometry in scene coordinates
            plot_rect = plot.sceneBoundingRect()

            # Get the view range (data coordinates)
            view_range = plot.viewRange()
            x_min, x_max = view_range[0]
            y_min, y_max = view_range[1]

            # Transform from data coords to page coords
            def data_to_page(x_data, y_data):
                # Map data to plot rect
                x_norm = (x_data - x_min) / (x_max - x_min) if x_max != x_min else 0.5
                y_norm = (y_data - y_min) / (y_max - y_min) if y_max != y_min else 0.5
                # Plot rect in page coordinates
                px = plot_rect.left() + x_norm * plot_rect.width()
                py = plot_rect.bottom() - y_norm * plot_rect.height()  # Y is inverted
                return px, py

            # Draw each data item
            for item in plot.listDataItems():
                if hasattr(item, 'getData'):
                    x_data, y_data = item.getData()
                    if x_data is None or y_data is None or len(x_data) == 0:
                        continue

                    # Get pen color
                    pen = item.opts.get('pen', None)
                    if pen is None:
                        pen = QPen(QColor(255, 255, 255), 1.5)
                    elif not isinstance(pen, QPen):
                        pen = pg.mkPen(pen)

                    # Ensure minimum width for Illustrator
                    export_pen = QPen(pen)
                    export_pen.setWidthF(max(pen.widthF(), 1.5))
                    export_pen.setCosmetic(False)
                    painter.setPen(export_pen)

                    # Build path from data points
                    path = QPainterPath()
                    first_point = True

                    # Downsample if too many points (for PDF size)
                    max_points = 50000
                    step = max(1, len(x_data) // max_points)

                    for i in range(0, len(x_data), step):
                        x, y = x_data[i], y_data[i]
                        if np.isnan(x) or np.isnan(y):
                            first_point = True
                            continue

                        px, py = data_to_page(x, y)

                        if first_point:
                            path.moveTo(px, py)
                            first_point = False
                        else:
                            path.lineTo(px, py)

                    painter.drawPath(path)

            # Draw scatter plots (markers)
            for item in plot.items:
                if isinstance(item, pg.ScatterPlotItem):
                    points = item.data
                    if points is None or len(points) == 0:
                        continue

                    # Get marker properties
                    brush = item.opts.get('brush', QBrush(QColor(255, 0, 0)))
                    if not isinstance(brush, QBrush):
                        brush = pg.mkBrush(brush)
                    size = item.opts.get('size', 8)

                    painter.setBrush(brush)
                    painter.setPen(Qt.PenStyle.NoPen)

                    for pt in points:
                        x, y = pt[0], pt[1]
                        if np.isnan(x) or np.isnan(y):
                            continue
                        px, py = data_to_page(x, y)
                        painter.drawEllipse(QPointF(px, py), size/2, size/2)

        # Draw axis labels and other text by rendering the non-data parts of the scene
        # This captures axis labels, titles, etc.
        painter.end()

        print(f"[Export] Saved vector PDF ({int(page_width)}x{int(page_height)} pts): {filepath}")
        return True

    def _export_png_hires(self, filepath, scene, source_rect, width, height):
        """Export to high-resolution PNG."""
        from PyQt6.QtGui import QPainter, QImage
        from PyQt6.QtCore import QRectF, Qt

        # Scale factor for high resolution (2x or 3x)
        scale_factor = 3

        # Output size in pixels
        out_width = int(width * scale_factor)
        out_height = int(height * scale_factor)

        # Create high-resolution image
        image = QImage(out_width, out_height, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)

        # Create painter
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Scale to output size
        scale_x = out_width / source_rect.width() if source_rect.width() > 0 else 1
        scale_y = out_height / source_rect.height() if source_rect.height() > 0 else 1
        painter.scale(scale_x, scale_y)

        # Render scene
        scene.render(painter, QRectF(), source_rect)
        painter.end()

        # Save image
        image.save(str(filepath), "PNG")

        print(f"[Export] Saved PNG ({out_width}x{out_height} pixels): {filepath}")
        return True

    def _export_svg_legacy(self, filepath, scene, width):
        """Legacy SVG export using pyqtgraph's exporter."""
        import pyqtgraph.exporters as exporters
        import tempfile
        import os

        # Create temp SVG
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            tmp_svg = tmp.name

        # Export to SVG first
        exporter = exporters.SVGExporter(scene)
        exporter.parameters()['width'] = width
        exporter.export(tmp_svg)

        # Convert SVG to PDF using Qt's PDF writer
        try:
            from PyQt6.QtSvg import QSvgRenderer
            from PyQt6.QtGui import QPainter, QPageSize
            from PyQt6.QtCore import QSizeF, QRectF
            from PyQt6.QtPrintSupport import QPrinter

            # Load SVG
            renderer = QSvgRenderer(tmp_svg)
            svg_size = renderer.defaultSize()

            # Create PDF with proper page size
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(str(filepath))

            # Set page size to match SVG aspect ratio
            page_size = QPageSize(QSizeF(svg_size.width(), svg_size.height()), QPageSize.Unit.Point)
            printer.setPageSize(page_size)
            printer.setPageMargins(0, 0, 0, 0, QPrinter.Unit.Point)

            # Render SVG to PDF
            painter = QPainter(printer)
            renderer.render(painter, QRectF(0, 0, svg_size.width(), svg_size.height()))
            painter.end()

            print(f"[Export] Saved PDF: {filepath}")
            return True

        except ImportError as e:
            print(f"[Export] PDF conversion requires PyQt6-WebEngine or cairosvg: {e}")
            # Fallback: just save the SVG with .pdf extension note
            import shutil
            svg_fallback = filepath.with_suffix('.svg')
            shutil.copy(tmp_svg, svg_fallback)
            print(f"[Export] Saved as SVG instead: {svg_fallback}")
            return True
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_svg)
            except:
                pass

    def show_export_dialog(self):
        """Show a file dialog to export the plot."""
        from PyQt6.QtWidgets import QFileDialog

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Plot",
            "",
            "PDF Files (*.pdf);;SVG Files (*.svg);;PNG Files (*.png);;All Files (*)"
        )

        if filepath:
            self.export_to_file(filepath)
            return True
        return False
