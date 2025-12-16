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
    PYQTGRAPH_AVAILABLE = True

    # Monkey-patch PyQtGraph to suppress 'autoRangeEnabled' AttributeError
    # This error occurs when PlotDataItems try to access their view during reparenting
    # (e.g., when GraphicsLayoutWidget.clear() is called and items reference stale views)
    # The error doesn't affect functionality, just causes console noise
    if hasattr(pg, 'PlotDataItem'):
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
        self.graphics_layout.setBackground('#1e1e1e')  # Dark background

        # Main plot widget
        self.plot_widget = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.setLabel('bottom', 'Time (s)')
        self.plot_widget.setLabel('left', 'Signal')

        # Enable auto-range initially
        self.plot_widget.enableAutoRange()

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
    def set_plot_theme(self, theme_name):
        """Apply dark or light theme."""
        self._current_theme = theme_name

        if theme_name == 'dark':
            bg_color = '#1e1e1e'
            text_color = '#d4d4d4'
            trace_color = '#d4d4d4'
        else:
            bg_color = '#ffffff'
            text_color = '#000000'
            trace_color = '#000000'

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

    # ------- View Preservation -------
    def set_preserve(self, x: bool = True, y: bool = False):
        """Set view preservation mode."""
        self._preserve_x = bool(x)
        self._preserve_y = bool(y)

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
        for (t0, t1) in (spans_s or []):
            if t1 > t0:
                region = pg.LinearRegionItem(
                    values=[t0, t1],
                    brush=pg.mkBrush(46, 80, 144, 60),  # #2E5090 with alpha
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
        self.graphics_layout.clear()
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

            # Style axes
            plot.getAxis('bottom').setTextPen(text_color)
            plot.getAxis('left').setTextPen(text_color)
            plot.getAxis('bottom').setPen(text_color)
            plot.getAxis('left').setPen(text_color)

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
        self._histogram_items = []

        below_mask = bin_centers < current_threshold
        above_mask = bin_centers >= current_threshold

        def draw_horizontal_bars(bins_y, counts_x, color):
            """Draw horizontal bar chart - bars extend rightward from x_min."""
            items = []
            for y_center, width in zip(bins_y, counts_x):
                if width > 0:
                    # Create a rectangle: x from x_min to x_min+width, y from y_center-h/2 to y_center+h/2
                    y_bottom = y_center - bin_height / 2
                    y_top = y_center + bin_height / 2
                    # Draw as a closed polygon (rectangle)
                    x_pts = [x_min, x_min + width, x_min + width, x_min, x_min]
                    y_pts = [y_bottom, y_bottom, y_top, y_top, y_bottom]
                    bar = pg.PlotCurveItem(x_pts, y_pts,
                                          pen=pg.mkPen(color[0], color[1], color[2], 200, width=0.5),
                                          brush=pg.mkBrush(*color),
                                          fillLevel=None)
                    bar.setZValue(99)
                    main_plot.addItem(bar)
                    items.append(bar)
            return items

        # Gray bars for below threshold
        if np.any(below_mask):
            below_bins = bin_centers[below_mask]
            below_counts = scaled_counts[below_mask]
            gray_items = draw_horizontal_bars(below_bins, below_counts, (128, 128, 128, 180))
            self._histogram_items.extend(gray_items)

        # Red bars for above threshold
        if np.any(above_mask):
            above_bins = bin_centers[above_mask]
            above_counts = scaled_counts[above_mask]
            red_items = draw_horizontal_bars(above_bins, above_counts, (220, 60, 60, 180))
            self._histogram_items.extend(red_items)

        print(f"[histogram] Created histogram with {len(self._histogram_items)} items, "
              f"{np.sum(below_mask)} below threshold, {np.sum(above_mask)} above threshold")

    def _update_threshold_histogram_colors(self, new_threshold):
        """Update histogram colors based on new threshold (fast update during drag)."""
        import numpy as np

        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        # Check if we have cached histogram data
        if (not hasattr(self, '_histogram_bin_centers') or
            self._histogram_bin_centers is None):
            return

        bin_centers = self._histogram_bin_centers
        scaled_counts = self._histogram_scaled_counts
        x_min = self._histogram_x_min
        bin_height = self._histogram_bin_height

        # Clear existing histogram items
        self._clear_threshold_histogram()

        # Redraw with new colors using horizontal bars
        self._histogram_items = []

        below_mask = bin_centers < new_threshold
        above_mask = bin_centers >= new_threshold

        def draw_horizontal_bars(bins_y, counts_x, color):
            """Draw horizontal bar chart - bars extend rightward from x_min."""
            items = []
            for y_center, width in zip(bins_y, counts_x):
                if width > 0:
                    y_bottom = y_center - bin_height / 2
                    y_top = y_center + bin_height / 2
                    x_pts = [x_min, x_min + width, x_min + width, x_min, x_min]
                    y_pts = [y_bottom, y_bottom, y_top, y_top, y_bottom]
                    bar = pg.PlotCurveItem(x_pts, y_pts,
                                          pen=pg.mkPen(color[0], color[1], color[2], 200, width=0.5),
                                          brush=pg.mkBrush(*color),
                                          fillLevel=None)
                    bar.setZValue(99)
                    main_plot.addItem(bar)
                    items.append(bar)
            return items

        # Gray bars for below threshold
        if np.any(below_mask):
            below_bins = bin_centers[below_mask]
            below_counts = scaled_counts[below_mask]
            gray_items = draw_horizontal_bars(below_bins, below_counts, (128, 128, 128, 180))
            self._histogram_items.extend(gray_items)

        # Red bars for above threshold
        if np.any(above_mask):
            above_bins = bin_centers[above_mask]
            above_counts = scaled_counts[above_mask]
            red_items = draw_horizontal_bars(above_bins, above_counts, (220, 60, 60, 180))
            self._histogram_items.extend(red_items)

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
                plot.vb.setMouseEnabled(x=True, y=True)
            except:
                pass

    def _on_mouse_clicked(self, event):
        """Handle mouse click events."""
        if self._external_click_cb is None:
            return

        # Get the main plot for coordinate mapping
        main_plot = self._get_main_plot()
        if main_plot is None:
            return

        # Get click position in data coordinates
        pos = event.scenePos()
        if main_plot.sceneBoundingRect().contains(pos):
            mouse_point = main_plot.vb.mapSceneToView(pos)
            x_data = mouse_point.x()
            y_data = mouse_point.y()

            # Create matplotlib-compatible event wrapper
            wrapped_event = _MatplotlibCompatEvent(event, main_plot, x_data, y_data)
            self._external_click_cb(x_data, y_data, wrapped_event)

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
        """Downsample data for performance."""
        if max_points is None or max_points <= 0 or len(t) <= max_points:
            return np.asarray(t), np.asarray(y)

        step = max(1, len(t) // max_points)
        return np.asarray(t)[::step], np.asarray(y)[::step]

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
        self.graphics_layout.clear()
        self._subplots = []
        # Recreate main plot widget
        self.plot_widget = self.graphics_layout.addPlot(row=0, col=0)
        self.plot_widget.showGrid(x=False, y=False)
        self.ax_main = self.plot_widget
        self._subplots = [self.plot_widget]
        # Reconnect click handler
        self.plot_widget.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    def add_subplot(self, gs_item):
        """Compatibility method for matplotlib add_subplot - returns a PlotItem.

        Note: This is a simplified compatibility layer. PyQtGraph doesn't use
        GridSpec, so we just add plots to the layout sequentially.
        """
        row = len(self._subplots)
        plot = self.graphics_layout.addPlot(row=row, col=0)
        plot.showGrid(x=False, y=False)
        # Link x-axis to first plot for synchronized panning
        if self._subplots:
            plot.setXLink(self._subplots[0])
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
