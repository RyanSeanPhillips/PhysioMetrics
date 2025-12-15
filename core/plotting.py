# core/plotting.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy, QGraphicsOpacityEffect
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtCore
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import matplotlib.patheffects as pe
from core.plot_themes import PlotThemeManager

# colors
PLETH_COLOR = "k"           # main trace
PEAK_COLOR  = "red"         # inspiratory peaks
ONSET_COLOR = "#2ecc71"     # green (inspiratory onset)
OFFSET_COLOR= "#f39c12"     # orange (inspiratory offset/expiratory onset)
EXPMIN_COLOR= "#1f78b4"     # blue (expiratory minimum)
EXPOFF_COLOR = "#9b59b6"   # purple (expiratory offset)


class TransparentToolbar(NavigationToolbar):
    """Custom toolbar with opacity animation on hover."""

    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)

        # Set up opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.1)  # Default: 10% visible

        # Animation for smooth transition
        self.opacity_animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.opacity_animation.setDuration(200)  # 200ms transition
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        # Hide the coordinate label
        if hasattr(self, 'locLabel'):
            self.locLabel.setVisible(False)

        # Hide separators
        for action in self.actions():
            if action.isSeparator():
                action.setVisible(False)

        # Make the extension/overflow button (black bar) fully transparent
        from PyQt6.QtWidgets import QToolButton
        from PyQt6.QtCore import Qt
        for child in self.findChildren(QToolButton):
            # Extension button has an arrow and no default action
            if child.arrowType() != Qt.ArrowType.NoArrow or not child.defaultAction():
                # Make it completely transparent
                button_opacity = QGraphicsOpacityEffect(child)
                button_opacity.setOpacity(0.0)
                child.setGraphicsEffect(button_opacity)

    def enterEvent(self, event):
        """Mouse enters toolbar - fade to full opacity."""
        self.opacity_animation.stop()
        self.opacity_animation.setStartValue(self.opacity_effect.opacity())
        self.opacity_animation.setEndValue(1.0)  # 100% visible
        self.opacity_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Mouse leaves toolbar - fade to transparent."""
        self.opacity_animation.stop()
        self.opacity_animation.setStartValue(self.opacity_effect.opacity())
        self.opacity_animation.setEndValue(0.1)  # 10% visible
        self.opacity_animation.start()
        super().leaveEvent(event)


class PlotHost(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Figure/canvas
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        # IMPORTANT: Connect our button handler BEFORE creating toolbar
        # This ensures our handler is registered first and gets events before toolbar handlers
        self._cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._cid_button = self.canvas.mpl_connect('button_press_event', self._on_button)

        # Qt-level failsafe: Override mousePressEvent to ensure editing modes always work
        # This provides a backup mechanism if matplotlib's callback chain is blocked
        # Normally not needed, but provides robustness against edge cases
        self._original_mouse_press = self.canvas.mousePressEvent
        self.canvas.mousePressEvent = self._qt_mouse_press_override

        # Toolbar (clean, dark buttons, no bar background)
        self.toolbar = TransparentToolbar(self.canvas, self)
        self.toolbar.setObjectName("PlotNavToolbar")
        self.toolbar.setIconSize(QtCore.QSize(15, 15))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("""
        QToolBar#PlotNavToolbar {
            background: transparent;
            border: 0px solid transparent;
            padding: 0px;
            spacing: 0px;
        }
        QToolBar#PlotNavToolbar:hover {
            border: 0px solid transparent;
        }
        QToolBar#PlotNavToolbar::separator { background: transparent; width: 0px; height: 0px; }
        QToolBar#PlotNavToolbar::handle { image: none; width: 0px; height: 0px; }
        QToolBar#PlotNavToolbar QToolButton {
            background: #434b5d; color: #eef2f8;
            border: 1px solid #5a6580; border-radius: 8px;
            padding: 5px 8px; margin: 2px;
            outline: none;
        }
        QToolBar#PlotNavToolbar QToolButton:hover {
            background: #515c72;
            border-color: #6a7694;
            outline: none;
        }
        QToolBar#PlotNavToolbar QToolButton:pressed {
            background: #5f6d88;
            border-color: #7886a6;
            outline: none;
        }
        QToolBar#PlotNavToolbar QToolButton:checked {
            background: #4A90E2;
            color: white;
            border-color: #357ABD;
            outline: none;
        }
        QToolBar#PlotNavToolbar QToolButton:disabled {
            background: #353b4a; border-color: #444d60; color: #8691a8;
        }
        QToolBar#PlotNavToolbar QToolButton:focus {
            outline: none;
            border-bottom: none;
        }""")

        # Connect toolbar actions to turn off edit modes
        self._toolbar_callback = None
        self._block_toolbar_callback = False  # Flag to prevent feedback loops
        for action in self.toolbar.actions():
            if action.isCheckable():
                action.toggled.connect(self._on_toolbar_action)

        #Per breath metrics
        self.ax_main = None
        self.scatter_peaks = None
        self.scatter_onsets = None
        self.scatter_offsets = None
        self.scatter_expmins = None
        self.scatter_expoffs = None
        self._sigh_artist = None

        # Line for height threshold visualization
        self.threshold_line = None
        self._dragging_threshold = False  # Track if threshold line is being dragged
        self._threshold_drag_cid_motion = None
        self._threshold_drag_cid_release = None
        self._threshold_histogram = None  # Histogram shown during threshold dragging

        # Cache histogram data for fast updates during drag
        self._histogram_bin_centers = None
        self._histogram_scaled_counts = None
        self._histogram_x_min = None

        # Separate storage for fills and lines (for correct draw order)
        self._histogram_fills = None
        self._histogram_lines = None

        # Background cache for blitting (threshold drag optimization)
        self._blit_background = None

        #Continuious Breath metrics
        self.ax_y2 = None
        self.line_y2 = None
        self.line_y2_secondary = None  # For second Y2 line (e.g., eupnea confidence)

        # Add mode for external click callback
        self._external_click_cb = None

        # Plot theme manager (dark/light mode)
        self.theme_manager = PlotThemeManager(default_theme='dark')  # Start with dark theme
        self._current_theme = 'dark'  # Default to dark theme






        
        # Layout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.canvas)  # Only canvas in layout

        # Position toolbar as overlay at top-right of canvas
        self.toolbar.setParent(self.canvas)  # Make it a child of canvas
        self._position_toolbar_right()  # Position at top-right
        self.toolbar.raise_()  # Ensure toolbar is on top

        # Preserve view flags & storage
        self._preserve_x = True
        self._preserve_y = False
        self._last_single = {"xlim": None, "ylim": None}
        self._last_grid   = {"xlim": None, "ylims": []}

        # Stateful single-panel bits (for cheap peak/span updates)
        self.ax_main = None
        self.line_main = None
        self.scatter_peaks = None
        self._span_patches = []

        # Event connections already registered above (before toolbar creation)

    def _position_toolbar_right(self):
        """Position the toolbar at the top-right corner of the canvas."""
        if hasattr(self, 'toolbar') and hasattr(self, 'canvas'):
            # Ensure toolbar is properly sized
            self.toolbar.adjustSize()

            canvas_width = self.canvas.width()
            # Use actual width, with sizeHint as fallback
            toolbar_width = self.toolbar.width()
            if toolbar_width <= 0:
                toolbar_width = self.toolbar.sizeHint().width()

            # Add extra margin to ensure all buttons are visible
            margin = 20
            x_pos = canvas_width - toolbar_width - margin
            self.toolbar.move(max(10, x_pos), 10)  # Ensure minimum 10px from left

    def set_plot_theme(self, theme_name):
        """
        Apply a plot theme (dark or light mode) to the current plot.

        Args:
            theme_name: 'dark' or 'light'
        """
        self._current_theme = theme_name

        # IMPORTANT: Update theme_manager FIRST so get_color() returns correct colors
        self.theme_manager.current_theme = theme_name

        # Always set figure background color, even if no axes exist yet
        theme = self.theme_manager.themes[theme_name]
        self.fig.patch.set_facecolor(theme['figure_facecolor'])

        # Get new trace color (now returns correct theme colors)
        trace_color = self.theme_manager.get_color('trace_color')
        text_color = self.theme_manager.get_color('text_color')

        # Apply theme to ALL existing axes (multi-channel mode needs all axes themed)
        if self.fig.axes:
            for ax in self.fig.axes:
                self.theme_manager.apply_theme(ax, self.fig, theme_name)

                # Update existing trace line colors (but not scatter plots for peaks/events)
                for line in ax.get_lines():
                    # Only update the main trace lines (not threshold lines or other overlays)
                    # Main traces typically have linewidth around 0.9-1.2
                    if line.get_linewidth() >= 0.8 and line.get_linewidth() <= 1.5:
                        # Check if this is a main trace color that should be updated
                        if self._is_trace_color(line.get_color()):
                            line.set_color(trace_color)

                # Update title color if it exists
                if ax.get_title():
                    ax.title.set_color(text_color)

            # Update Y2 axis colors if it exists
            if self.ax_y2 is not None:
                tick_color = self.theme_manager.get_color('tick_color')
                label_color = self.theme_manager.get_color('label_color')
                edge_color = self.theme_manager.get_color('axes_edgecolor')
                self.ax_y2.spines['right'].set_edgecolor(edge_color)
                self.ax_y2.tick_params(colors=tick_color, which='both')
                self.ax_y2.yaxis.label.set_color(label_color)

            # Update sigh markers (stars) colors based on theme
            if self._sigh_artist is not None:
                is_dark = (theme_name == 'dark')
                sigh_color = "#FFD700"  # Gold
                sigh_edge = "#FFD700" if is_dark else "black"
                self._sigh_artist.set_facecolors(sigh_color)
                self._sigh_artist.set_edgecolors(sigh_edge)

        self.canvas.draw()

    def _apply_current_theme(self):
        """Apply the current theme to ax_main after creating it."""
        if self.ax_main is not None:
            self.theme_manager.apply_theme(self.ax_main, self.fig, self._current_theme)

    def _is_trace_color(self, color):
        """
        Check if a color represents a main trace color that should be updated on theme change.

        This handles various color formats that matplotlib might return:
        - Named colors: 'k', 'black', 'w', 'white'
        - Hex strings: '#000000', '#ffffff', '#d4d4d4', '#cccccc'
        - RGB/RGBA tuples: (0, 0, 0), (1, 1, 1), etc.

        Returns True if the color is black, white, or gray-ish (typical trace colors).
        """
        import matplotlib.colors as mcolors

        # Known trace colors (named)
        named_trace_colors = {'k', 'black', 'w', 'white'}

        # Check named colors
        if isinstance(color, str):
            if color.lower() in named_trace_colors:
                return True
            # Check hex colors
            if color.startswith('#'):
                hex_trace_colors = {'#000000', '#ffffff', '#d4d4d4', '#cccccc'}
                return color.lower() in hex_trace_colors

        # Handle RGB/RGBA tuples or arrays
        try:
            # Convert any color format to RGBA
            rgba = mcolors.to_rgba(color)
            r, g, b = rgba[0], rgba[1], rgba[2]

            # Check if it's black (all values close to 0)
            if r < 0.1 and g < 0.1 and b < 0.1:
                return True

            # Check if it's white (all values close to 1)
            if r > 0.9 and g > 0.9 and b > 0.9:
                return True

            # Check if it's gray-ish (R, G, B are similar and in gray range)
            # #d4d4d4 = (0.831, 0.831, 0.831)
            # #cccccc = (0.8, 0.8, 0.8)
            if abs(r - g) < 0.05 and abs(g - b) < 0.05:
                if 0.7 < r < 0.95:  # Gray range
                    return True

        except Exception:
            pass

        return False

    def _clean_spines(self, ax, has_y2=False):
        """
        Clean up axis spines for a minimal, modern look.
        Only show left and bottom spines (and right if Y2 axis exists).

        Args:
            ax: The matplotlib axis to clean
            has_y2: If True, show the right spine for Y2 axis
        """
        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        if not has_y2:
            ax.spines['right'].set_visible(False)

        # Keep left and bottom spines visible
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # Only show ticks on left and bottom (not on Y2 spine from this axis)
        ax.tick_params(top=False, right=False, left=True, bottom=True)

    # ------- public API -------
    def set_preserve(self, x: bool = True, y: bool = False):
        self._preserve_x = bool(x)
        self._preserve_y = bool(y)

    def clear_saved_view(self, mode: str | None = None):
        if mode is None or mode == "single":
            self._last_single = {"xlim": None, "ylim": None}
        if mode is None or mode == "grid":
            self._last_grid   = {"xlim": None, "ylims": []}

    def set_toolbar_callback(self, callback):
        """Set callback to be called when toolbar actions are triggered."""
        self._toolbar_callback = callback

    def turn_off_toolbar_modes(self):
        """Turn off all matplotlib toolbar modes (zoom, pan)."""
        # Block callback to prevent feedback loop
        self._block_toolbar_callback = True

        # Explicitly turn off matplotlib's internal mode
        # This prevents matplotlib from hijacking mouse events
        if hasattr(self.toolbar, 'mode') and self.toolbar.mode != '':
            # There's an active mode - turn it off by calling the same method again (toggle)
            if self.toolbar.mode == 'pan/zoom':
                self.toolbar.pan()
            elif self.toolbar.mode == 'zoom rect':
                self.toolbar.zoom()

        # Uncheck all checkable toolbar actions
        for action in self.toolbar.actions():
            if action.isCheckable() and action.isChecked():
                action.setChecked(False)

        # Re-enable callback
        self._block_toolbar_callback = False

    # ------- interactions -------
    def _on_toolbar_action(self, checked):
        """Called when any toolbar action (zoom, pan, etc.) is toggled."""
        # Only call callback if not blocked (prevents feedback loops)
        if checked and self._toolbar_callback is not None and not self._block_toolbar_callback:
            self._toolbar_callback()

    def _on_button(self, event):
        if event.inaxes is None:
            return

        # Double-click: autoscale (keep your behavior)
        if event.dblclick:
            ax = event.inaxes
            ax.autoscale()
            self.canvas.draw_idle()
            self._store_from_axes(mode=("single" if len(self.fig.axes) == 1 else "grid"))
            return

        # Single-click (left or right): forward to external callback if present
        # The Y2 axis has mouse events disabled, so clicks should only come from main axis
        if self._external_click_cb is not None and event.xdata is not None:
            self._external_click_cb(event.xdata, event.ydata, event)

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        base = 1.2
        scale = (1 / base) if event.button == 'up' else base
        key = (event.key or "").lower()
        zoom_y = ('shift' in key)
        zoom_x = not zoom_y

        if zoom_x:
            x0, x1 = ax.get_xlim()
            cx = event.xdata
            ax.set_xlim(cx - (cx - x0) * scale, cx + (x1 - cx) * scale)

        if zoom_y:
            y0, y1 = ax.get_ylim()
            cy = event.ydata
            ax.set_ylim(cy - (cy - y0) * scale, cy + (y1 - cy) * scale)

        self.canvas.draw_idle()
        self._store_from_axes(mode=("single" if len(self.fig.axes) == 1 else "grid"))

    def _qt_mouse_press_override(self, event):
        """
        Qt-level failsafe for editing mode clicks.

        This override ensures editing modes (Mark Sniff, Add/Delete Peaks, etc.) continue
        working even if matplotlib's callback chain is blocked by other handlers. It manually
        converts click coordinates and calls our callback directly when an editing mode is active.

        Normally not triggered since our matplotlib handler has priority, but provides robust
        fallback behavior for edge cases.
        """
        # Always call the original handler first to let matplotlib/toolbar process normally
        self._original_mouse_press(event)

        # If we have an editing mode callback AND matplotlib didn't trigger our _on_button,
        # manually convert coordinates and call our callback
        if self._external_click_cb is not None and self.fig.axes:
            try:
                # Get click position in Qt coordinates
                x_pixel = event.pos().x()
                y_pixel = event.pos().y()

                # Convert to display coordinates (matplotlib uses different origin)
                # Get the figure's pixel dimensions
                dpi = self.fig.dpi
                fig_width, fig_height = self.fig.get_size_inches()
                fig_height_pixels = fig_height * dpi

                # Matplotlib has origin at bottom-left, Qt has origin at top-left
                # Invert y coordinate
                y_display = fig_height_pixels - y_pixel

                # Try to find which axes contains this point and convert to data coordinates
                for ax in self.fig.axes:
                    # Check if click is within this axes
                    bbox = ax.get_window_extent()
                    if bbox.contains(x_pixel, y_display):
                        # Convert display coordinates to data coordinates
                        inv = ax.transData.inverted()
                        x_data, y_data = inv.transform((x_pixel, y_display))

                        # Call our callback directly
                        self._external_click_cb(x_data, y_data, event)
                        break

            except Exception:
                # Silently ignore conversion errors - matplotlib handler will catch valid clicks
                pass

    # ------- remember/restore helpers -------
    def _attach_limit_listeners(self, axes, mode: str):
        def on_change(_):
            self._store_from_axes(mode)
        for ax in axes:
            ax.callbacks.connect('xlim_changed', on_change)
            ax.callbacks.connect('ylim_changed', on_change)

    def _store_from_axes(self, mode: str):
        if mode == "single" and self.fig.axes:
            ax = self.fig.axes[0]
            self._last_single["xlim"] = ax.get_xlim()
            self._last_single["ylim"] = ax.get_ylim()
        elif mode == "grid" and self.fig.axes:
            axes = list(self.fig.axes)
            self._last_grid["xlim"] = axes[-1].get_xlim()  # shared x
            self._last_grid["ylims"] = [ax.get_ylim() for ax in axes]


    # set x-limits programmatically
    def set_xlim(self, x0: float, x1: float):
        """Set x-limits on current plot (single or grid) and remember that view."""
        if not self.fig.axes:
            return
        if len(self.fig.axes) == 1:
            ax = self.fig.axes[0]
            ax.set_xlim(x0, x1)
            self._store_from_axes("single")
        else:
            for ax in self.fig.axes:
                ax.set_xlim(x0, x1)
            self._store_from_axes("grid")
        self.canvas.draw_idle()


    # ------- helpers -------
    def _downsample_even(self, t: np.ndarray, y: np.ndarray, max_points: int = 2000):
        n = len(t)
        if max_points is None or max_points <= 0 or n <= max_points:
            return t, y
        step = max(1, ceil(n / max_points))
        return t[::step], y[::step]


    # def show_trace_with_spans(self, t, y, spans_s, title: str = "", max_points: int = 2000):
    #     prev_xlim = self._last_single["xlim"] if self._preserve_x else None
    #     prev_ylim = self._last_single["ylim"] if self._preserve_y else None

    #     self.fig.clear()
    #     self.ax_main = self.fig.add_subplot(111)

    #     # new Axes => any old scatter is invalid
    #     # Reset old overlay references (figure was cleared)


    #     tds, yds = self._downsample_even(t, y, max_points=max_points if max_points else len(t))
    #     self.ax_main.plot(tds, yds, linewidth=0.75, color=PLETH_COLOR)

    #     for (t0, t1) in spans_s or []:
    #         if t1 > t0:
    #             self.ax_main.axvspan(t0, t1, alpha=0.15)

    #     if title:
    #         self.ax_main.set_title(title)
    #     self.ax_main.set_xlabel("Time (s)")
    #     self.ax_main.set_ylabel("Signal")
    #     self.ax_main.grid(True, alpha=0.2)

    #     if prev_xlim is not None:
    #     if prev_ylim is not None:
    #         self.ax_main.set_xlim(prev_xlim)
    #         self.ax_main.set_ylim(prev_ylim)

    #     self._attach_limit_listeners([self.ax_main], mode="single")
    #     self._store_from_axes(mode="single")
    #     self.canvas.draw_idle()

    def show_trace_with_spans(self, t, y, spans_s, title: str = "", max_points: int = 2000, ylabel: str = "Signal", state=None):
        prev_xlim = self._last_single["xlim"] if self._preserve_x else None
        prev_ylim = self._last_single["ylim"] if self._preserve_y else None

        self.fig.clear()
        self.ax_main = self.fig.add_subplot(111)

        # Apply current theme (dark/light mode)
        self._apply_current_theme()

        # Any old scatter is invalid after clearing
        self.clear_peaks()
        self.scatter_peaks = None
        self.scatter_onsets = None
        self.scatter_offsets = None
        self.scatter_expmins = None

        # Clear threshold line
        self.threshold_line = None

        # IMPORTANT: Clear Y2 axis references after fig.clear()
        # This ensures that when Y2 is recreated, it properly sets up mouse event blocking
        self.ax_y2 = None
        self.line_y2 = None
        self.line_y2_secondary = None

        # Plot main trace (color depends on theme)
        tds, yds = self._downsample_even(t, y, max_points=max_points if max_points else len(t))
        trace_color = self.theme_manager.get_color('trace_color')
        self.ax_main.plot(tds, yds, linewidth=0.9, color=trace_color)

        # Dashed baseline at y=0 (behind traces)
        self.ax_main.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Optional shaded spans
        for (t0, t1) in (spans_s or []):
            if t1 > t0:
                self.ax_main.axvspan(t0, t1, color="#2E5090", alpha=0.25)

        if title:
            title_color = self.theme_manager.get_color('text_color')
            self.ax_main.set_title(title, color=title_color)
        self.ax_main.set_xlabel("Time (s)")
        self.ax_main.set_ylabel(ylabel)

        # Single-panel: no grid
        self.ax_main.grid(False)

        # Intelligent y-axis scaling: percentile or full range based on user preference
        # First, filter out NaN/Inf values to avoid crashes
        y_valid = y[np.isfinite(y)]

        if len(y_valid) == 0:
            # No valid data - use default range
            print("[Plot] Warning: No valid (finite) data for Y-axis scaling, using default range")
            y_min, y_max, padding = -1.0, 1.0, 0.1
            mode_str = "default (no valid data)"
        elif state and hasattr(state, 'use_percentile_autoscale') and state.use_percentile_autoscale:
            # Percentile mode: Use 99th percentile + configurable padding to avoid artifacts
            y_min = np.percentile(y_valid, 1)  # 1st percentile for lower bound
            y_max = np.percentile(y_valid, 99)  # 99th percentile for upper bound
            y_range = y_max - y_min
            # Handle case where all values are the same
            if y_range == 0 or not np.isfinite(y_range):
                y_range = abs(y_min) if y_min != 0 else 1.0
            padding_factor = getattr(state, 'autoscale_padding', 0.25)
            padding = padding_factor * y_range
            mode_str = f"99th percentile, {padding_factor*100:.0f}% padding"
        else:
            # Full range mode: Use absolute min/max
            y_min = np.min(y_valid)
            y_max = np.max(y_valid)
            y_range = y_max - y_min
            # Handle case where all values are the same
            if y_range == 0 or not np.isfinite(y_range):
                y_range = abs(y_min) if y_min != 0 else 1.0
            padding = 0.05 * y_range  # Small padding to avoid clipping edges
            mode_str = "full range (min/max)"

        # Final validation before setting limits
        y_lower = y_min - padding
        y_upper = y_max + padding
        if np.isfinite(y_lower) and np.isfinite(y_upper) and y_lower < y_upper:
            self.ax_main.set_ylim(y_lower, y_upper)
            print(f"[Plot] Auto-scaled Y-axis: {y_lower:.3f} to {y_upper:.3f} ({mode_str})")
        else:
            # Fallback to matplotlib's autoscale
            print(f"[Plot] Warning: Invalid Y limits ({y_lower}, {y_upper}), using autoscale")

        # Restore preserved X view if desired (but always auto-scale Y)
        if prev_xlim is not None:
            self.ax_main.set_xlim(prev_xlim)

        self._attach_limit_listeners([self.ax_main], mode="single")
        self._store_from_axes(mode="single")

        # Clean up spines for minimal look
        self._clean_spines(self.ax_main, has_y2=False)

        # Use tight layout with minimal padding to maximize plot area
        self.fig.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
        # Further adjust to use full width
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)
        self.canvas.draw_idle()

        # Update matplotlib toolbar's home view to current (maximized) limits
        # This ensures the home button will always reset to the full x-axis view
        self.toolbar.push_current()



    def _update_spans(self, spans_s):
        # Remove old patches
        for p in self._span_patches:
            try: p.remove()
            except Exception: pass
        self._span_patches = []

        if not spans_s or self.ax_main is None:
            self.canvas.draw_idle()
            return

        trans = self.ax_main.get_xaxis_transform()  # x in data, y in axes (0..1)
        for (t0, t1) in spans_s:
            if t1 > t0:
                p = self.ax_main.axvspan(t0, t1, ymin=0.0, ymax=1.0, color="#2E5090", alpha=0.25, transform=trans)
                self._span_patches.append(p)
        self.canvas.draw_idle()

    #         try:
    #         except Exception:
    #             pass
    #     if self.scatter_peaks is not None:
    #             self.scatter_peaks.remove()
    #         self.scatter_peaks = None
    #         self.canvas.draw_idle()

    def clear_breath_events(self):
        for a in (self.scatter_onsets, self.scatter_offsets, self.scatter_expmins):
            if a is not None:
                a.remove()
        self.scatter_onsets = self.scatter_offsets = self.scatter_expmins = None
        self.canvas.draw_idle()


    # ------- multi-panel (downsampled) -------
    def show_multi_grid(self, traces, title: str = "", max_points_per_trace: int = 2000):
        prev_xlim = self._last_grid["xlim"] if self._preserve_x else None
        prev_ylims = self._last_grid["ylims"] if self._preserve_y else None

        n = len(traces)
        if n == 0:
            return
        self.fig.clear()

        # Clear threshold line when switching to multi-plot view
        self.threshold_line = None

        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        # Apply theme to all subplots first
        for ax in axes:
            self.theme_manager.apply_theme(ax, self.fig, self._current_theme)

        # Get trace color after theme is applied
        trace_color = self.theme_manager.get_color('trace_color')

        for ax, (t, y, label) in zip(axes, traces):
            tds, yds = self._downsample_even(t, y, max_points=max_points_per_trace)
            ax.plot(tds, yds, linewidth=1, color=trace_color)
            ax.set_ylabel(label)
            # Clean spines for minimal look
            self._clean_spines(ax, has_y2=False)

        axes[-1].set_xlabel("Time (s)")
        if title:
            title_color = self.theme_manager.get_color('text_color')
            axes[0].set_title(title, color=title_color)

        if prev_xlim is not None:
            for ax in axes:
                ax.set_xlim(prev_xlim)
        if prev_ylims is not None and len(prev_ylims) == len(axes):
            for ax, yl in zip(axes, prev_ylims):
                ax.set_ylim(yl)

        self._attach_limit_listeners(axes, mode="grid")
        self._store_from_axes(mode="grid")
        self.canvas.draw_idle()

        # Update matplotlib toolbar's home view to current (maximized) limits
        # This ensures the home button will always reset to the full x-axis view
        self.toolbar.push_current()

    # ------- Qt hook -------
    def resizeEvent(self, event):
        # keep layout tight in single-panel mode
        if len(self.fig.axes) == 1:
            self.fig.tight_layout()
        self.canvas.draw_idle()
        # Keep toolbar at top-right
        if hasattr(self, 'toolbar'):
            self._position_toolbar_right()
        return super().resizeEvent(event)


    def update_peaks(self, t_peaks, y_peaks, size=24):
        """Create/update red dots for inspiratory peaks."""
        if not self.fig.axes:
            return
        # Use ax_main (primary Pleth axis) if available, otherwise fall back to first axis
        ax = self.ax_main if self.ax_main is not None else self.fig.axes[0]
        import numpy as np
        if t_peaks is None or y_peaks is None or len(t_peaks) == 0:
            self.clear_peaks()
            return
        pts = np.column_stack([t_peaks, y_peaks])
        if self.scatter_peaks is None or self.scatter_peaks.axes is not ax:
            self.scatter_peaks = ax.scatter(pts[:, 0], pts[:, 1], s=size, c="red", marker="o", zorder=3)
        else:
            self.scatter_peaks.set_offsets(pts)
        self.canvas.draw_idle()

    def clear_peaks(self):
        if self.scatter_peaks is not None:
            try:
                self.scatter_peaks.remove()
            except Exception:
                pass
            self.scatter_peaks = None
            self.canvas.draw_idle()

    def update_threshold_line(self, threshold_value):
        """Draw/update a red dashed horizontal line at the height threshold."""
        if not self.fig.axes:
            return
        # Use ax_main (primary Pleth axis) if available, otherwise fall back to first axis
        ax = self.ax_main if self.ax_main is not None else self.fig.axes[0]

        if threshold_value is None:
            # Remove threshold line if no value
            self.clear_threshold_line()
            return

        if self.threshold_line is None or self.threshold_line.axes is not ax:
            # Create new line (make it pickable for dragging)
            self.threshold_line = ax.axhline(
                threshold_value,
                linestyle=(0, (3, 3)),  # Smaller dashes: (offset, (on, off))
                linewidth=1.2,  # Thinner
                color="red",
                alpha=0.8,
                zorder=1000,  # Very high z-order to be on top of Y2 axis and always pickable
                label="Height Threshold",
                picker=5  # Pickable within 5 pixels
            )
            # Connect pick event for dragging
            self.canvas.mpl_connect('pick_event', self._on_threshold_pick)
        else:
            # Update existing line
            self.threshold_line.set_ydata([threshold_value, threshold_value])

        self.canvas.draw_idle()

    def clear_threshold_line(self):
        """Remove the threshold line from the plot."""
        if self.threshold_line is not None:
            try:
                self.threshold_line.remove()
            except Exception:
                pass
            self.threshold_line = None
            self.canvas.draw_idle()

    def _on_threshold_pick(self, event):
        """Handle pick event on threshold line to start dragging."""
        print(f"[MainPlot] Threshold line picked!")
        if event.artist == self.threshold_line:
            print(f"[MainPlot] Starting threshold drag...")
            self._dragging_threshold = True
            # Connect motion and release events for dragging
            if self._threshold_drag_cid_motion is None:
                self._threshold_drag_cid_motion = self.canvas.mpl_connect(
                    'motion_notify_event', self._on_threshold_drag
                )
            if self._threshold_drag_cid_release is None:
                self._threshold_drag_cid_release = self.canvas.mpl_connect(
                    'button_release_event', self._on_threshold_release
                )
            # Change cursor to indicate dragging
            from PyQt6.QtCore import Qt
            self.setCursor(Qt.CursorShape.SizeVerCursor)

            # Show histogram of all peak heights
            self._show_peak_height_histogram()

            # Set threshold line and histogram as animated BEFORE caching background
            # This ensures they're NOT included in the cached background
            if self.threshold_line:
                self.threshold_line.set_animated(True)
            if self._threshold_histogram:
                for artist in self._threshold_histogram:
                    artist.set_animated(True)

            # Cache background for ultra-fast blitting (without threshold line/histogram)
            self.canvas.draw()
            self.canvas.flush_events()
            self._blit_background = self.canvas.copy_from_bbox(self.fig.bbox)

    def _on_threshold_drag(self, event):
        """Handle mouse motion while dragging threshold line (ultra-optimized with blitting)."""
        if not self._dragging_threshold or event.inaxes is None or self._blit_background is None:
            return

        # Update threshold line position
        new_threshold = event.ydata
        if new_threshold is not None and new_threshold > 0:
            self.threshold_line.set_ydata([new_threshold, new_threshold])

            # Store threshold but DON'T update SpinBox during drag (too slow!)
            main_window = self._find_main_window()
            if main_window:
                main_window.peak_height_threshold = new_threshold

            # Update histogram colors based on new threshold
            self._update_histogram_colors_for_blitting(new_threshold)

            # ULTRA-OPTIMIZED: Use blitting to only redraw threshold line + histogram
            # Restore cached background (everything except animated artists)
            self.canvas.restore_region(self._blit_background)

            # Redraw only the animated artists (threshold line + histogram)
            ax = self.threshold_line.axes

            # Draw histogram artists (fills first, then lines on top)
            if self._histogram_fills:
                for fill in self._histogram_fills:
                    try:
                        ax.draw_artist(fill)
                    except:
                        pass

            if self._histogram_lines:
                for line in self._histogram_lines:
                    try:
                        ax.draw_artist(line)
                    except:
                        pass

            # Draw threshold line on top of everything
            ax.draw_artist(self.threshold_line)

            # Blit the updated region (super fast!)
            self.canvas.blit(self.fig.bbox)
            self.canvas.flush_events()

    def _on_threshold_release(self, event):
        """Handle mouse release to stop dragging threshold line."""
        print(f"[MainPlot] Mouse released, dragging={self._dragging_threshold}")
        if self._dragging_threshold:
            print(f"[MainPlot] Inside dragging block")
            self._dragging_threshold = False
            # Restore cursor
            from PyQt6.QtCore import Qt
            self.setCursor(Qt.CursorShape.ArrowCursor)

            # Turn off animation on threshold line and histogram
            if self.threshold_line:
                self.threshold_line.set_animated(False)
            if self._threshold_histogram:
                for artist in self._threshold_histogram:
                    try:
                        artist.set_animated(False)
                    except:
                        pass

            # Clear background cache
            self._blit_background = None

            # Update stored prominence value and sync with Analysis Options dialog
            main_window = self._find_main_window()
            print(f"[MainPlot] Found main_window: {main_window is not None}")
            if main_window:
                current_threshold = getattr(main_window, 'peak_height_threshold', None)
                print(f"[MainPlot] current_threshold: {current_threshold}")
                if current_threshold is not None:
                    # Update stored prominence value
                    main_window.peak_prominence = current_threshold

                    # Update Analysis Options dialog histogram if it's open
                    print(f"[MainPlot] Threshold released at {current_threshold:.4f}")
                    if hasattr(main_window, '_analysis_options_dialog'):
                        print(f"[MainPlot] - Has _analysis_options_dialog: {main_window._analysis_options_dialog is not None}")
                        if main_window._analysis_options_dialog is not None:
                            try:
                                is_visible = main_window._analysis_options_dialog.isVisible()
                                print(f"[MainPlot] - Dialog visible: {is_visible}")
                                if is_visible:
                                    # Update the ProminenceThresholdDialog in the Peak Detection tab
                                    has_prom = hasattr(main_window._analysis_options_dialog, 'prominence_dialog')
                                    print(f"[MainPlot] - Has prominence_dialog: {has_prom}")
                                    if has_prom:
                                        prom_dialog = main_window._analysis_options_dialog.prominence_dialog
                                        print(f"[MainPlot] - prominence_dialog is not None: {prom_dialog is not None}")
                                        if prom_dialog is not None:
                                            has_method = hasattr(prom_dialog, 'update_threshold_from_external')
                                            print(f"[MainPlot] - Has update method: {has_method}")
                                            if has_method:
                                                print(f"[MainPlot] Syncing threshold {current_threshold:.4f} to histogram dialog")
                                                prom_dialog.update_threshold_from_external(current_threshold)
                            except RuntimeError as e:
                                # Dialog was deleted
                                print(f"[MainPlot] Dialog deleted, skipping sync: {e}")
                                pass
                            except Exception as e:
                                print(f"[MainPlot] Error syncing to histogram: {e}")
                    else:
                        print("[MainPlot] - No _analysis_options_dialog attribute")

            # Clear histogram now that dragging is done
            self._clear_peak_height_histogram()

            # Enable Apply button
            if main_window:
                main_window.ApplyPeakFindPushButton.setEnabled(True)

    def _find_main_window(self):
        """Walk up parent chain to find the main window (has ApplyPeakFindPushButton attribute)."""
        widget = self
        while widget is not None:
            # Look for a unique main window attribute (ApplyPeakFindPushButton or state)
            if hasattr(widget, 'ApplyPeakFindPushButton') or hasattr(widget, 'state'):
                return widget
            widget = widget.parent()
        return None

    def _show_peak_height_histogram(self):
        """Show a horizontal histogram of all peak heights while dragging threshold.

        NOTE: Known limitation - histogram shows peaks from last detection:
        - If dialog was used: Shows full peak distribution (threshold=0)
        - If only Apply was used: Shows peaks passing current threshold
        This causes inconsistency where dragging histogram may differ from dialog histogram.

        Considered solutions:
        1. Re-detect peaks with threshold=0 on first drag (may cause lag on long recordings)
        2. Always cache threshold=0 peaks in background (memory overhead)
        3. Wait for ML refactor where all peaks are kept and labeled (planned approach)

        Current approach: Simple and fast - use cached peaks from last detection.
        """
        try:
            if not self.fig.axes:
                return

            ax = self.fig.axes[0]
            main_window = self._find_main_window()

            if not main_window or not hasattr(main_window, 'state'):
                return

            import numpy as np

            st = main_window.state

            # Use cached peak heights if available (from dialog or previous detection)
            # Otherwise collect from currently detected peaks
            if not hasattr(main_window, 'all_peak_heights') or main_window.all_peak_heights is None:
                # Collect peak heights from currently detected peaks
                all_heights = []

                for s in range(len(st.peaks_by_sweep)):
                    pks = st.peaks_by_sweep.get(s, None)
                    if pks is not None and len(pks) > 0:
                        # Get processed data for this sweep
                        y_proc = main_window._get_processed_for(st.analyze_chan, s)
                        all_heights.extend(y_proc[pks])

                if len(all_heights) == 0:
                    print("[Histogram] No peaks detected yet")
                    return

                main_window.all_peak_heights = np.array(all_heights)

            peak_heights = main_window.all_peak_heights

            # Calculate percentile cutoff if not already available
            num_bins = getattr(main_window, 'histogram_num_bins', 200)
            percentile_cutoff = getattr(main_window, 'histogram_percentile_cutoff', 99)

            if not hasattr(main_window, 'percentile_95') or main_window.percentile_95 is None:
                if percentile_cutoff < 100:
                    main_window.percentile_95 = np.percentile(peak_heights, percentile_cutoff)
                    print(f"[Histogram] Calculated {percentile_cutoff}th percentile on-the-fly: {main_window.percentile_95:.3f}")
                else:
                    main_window.percentile_95 = None

            percentile_95 = main_window.percentile_95
            print(f"[Histogram] Using {len(peak_heights)} peaks, {num_bins} bins, {percentile_cutoff}% cutoff")

            if percentile_95 is not None:
                # Filter peaks for histogram to match dialog display
                peaks_for_hist = peak_heights[peak_heights <= percentile_95]
                hist_range = (peaks_for_hist.min(), percentile_95)
                n_excluded = len(peak_heights) - len(peaks_for_hist)
                pct_excluded = 100 * n_excluded / len(peak_heights) if len(peak_heights) > 0 else 0
                print(f"[Histogram] Excluding {n_excluded} outliers ({pct_excluded:.1f}%) above percentile ({percentile_95:.3f})")
            else:
                peaks_for_hist = peak_heights
                hist_range = None
                n_excluded = 0

            # Get current x-axis limits to position histogram at left edge
            xlim = ax.get_xlim()
            x_min = xlim[0]
            x_range = xlim[1] - xlim[0]

            # Get current threshold for coloring
            current_threshold = getattr(main_window, 'peak_height_threshold', None)
            if current_threshold is None:
                return

            # Create histogram data with SAME bins and range as dialog
            counts, bins = np.histogram(peaks_for_hist, bins=num_bins, range=hist_range)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Scale histogram to 10% of x-axis range
            max_count = np.max(counts)
            if max_count > 0:
                scale_factor = (0.1 * x_range) / max_count
                scaled_counts = counts * scale_factor

                # Cache histogram data for fast updates during drag
                self._histogram_bin_centers = bin_centers
                self._histogram_scaled_counts = scaled_counts
                self._histogram_x_min = x_min

                # Plot as line with fills
                # Split into below and above threshold
                below_mask = bin_centers < current_threshold
                above_mask = bin_centers >= current_threshold

                # Gray fill for below threshold (black outline, gray fill)
                if np.any(below_mask):
                    line_below = ax.plot(x_min + scaled_counts[below_mask], bin_centers[below_mask],
                                        'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                    fill_below = ax.fill_betweenx(bin_centers[below_mask],
                                                  x_min, x_min + scaled_counts[below_mask],
                                                  alpha=0.7, color='gray', zorder=99)
                    self._threshold_histogram = [line_below, fill_below]
                else:
                    self._threshold_histogram = []

                # Red fill for above threshold (black outline, red fill)
                if np.any(above_mask):
                    line_above = ax.plot(x_min + scaled_counts[above_mask], bin_centers[above_mask],
                                        'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                    fill_above = ax.fill_betweenx(bin_centers[above_mask],
                                                  x_min, x_min + scaled_counts[above_mask],
                                                  alpha=0.7, color='red', zorder=99)
                    if self._threshold_histogram is None:
                        self._threshold_histogram = []
                    self._threshold_histogram.extend([line_above, fill_above])

            # CRITICAL: Restore original x-limits to prevent autoscaling from expanding the view
            ax.set_xlim(xlim)

            print(f"[Histogram] Created line histogram with colored fills")
            self.canvas.draw_idle()

        except Exception as e:
            print(f"[Histogram] Error: {e}")
            import traceback
            traceback.print_exc()

    def _clear_peak_height_histogram(self):
        """Remove the peak height histogram."""
        if self._threshold_histogram is not None:
            for artist in self._threshold_histogram:
                try:
                    artist.remove()
                except:
                    pass
            self._threshold_histogram = None
            # Clear cached data
            self._histogram_bin_centers = None
            self._histogram_scaled_counts = None
            self._histogram_x_min = None
            self._histogram_fills = None
            self._histogram_lines = None
            self.canvas.draw_idle()

    def _update_histogram_colors_fast(self, new_threshold):
        """
        Fast update of histogram colors without redrawing everything.
        Only updates the fill colors based on new threshold.
        """
        # If histogram not created yet or no cached data, do full redraw
        if (self._threshold_histogram is None or
            self._histogram_bin_centers is None or
            self._histogram_scaled_counts is None):
            # First time - create histogram
            self._clear_peak_height_histogram()
            self._show_peak_height_histogram()
            return

        try:
            import numpy as np

            # Get the axis
            if not self.fig.axes:
                return
            ax = self.fig.axes[0]

            # Remove old histogram artists
            for artist in self._threshold_histogram:
                try:
                    artist.remove()
                except:
                    pass

            # Recreate with new threshold coloring
            bin_centers = self._histogram_bin_centers
            scaled_counts = self._histogram_scaled_counts
            x_min = self._histogram_x_min

            # Split into below and above threshold
            below_mask = bin_centers < new_threshold
            above_mask = bin_centers >= new_threshold

            self._threshold_histogram = []

            # Gray fill for below threshold (black outline, gray fill)
            if np.any(below_mask):
                line_below = ax.plot(x_min + scaled_counts[below_mask], bin_centers[below_mask],
                                    'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                fill_below = ax.fill_betweenx(bin_centers[below_mask],
                                              x_min, x_min + scaled_counts[below_mask],
                                              alpha=0.7, color='gray', zorder=99)
                self._threshold_histogram.extend([line_below, fill_below])

            # Red fill for above threshold (black outline, red fill)
            if np.any(above_mask):
                line_above = ax.plot(x_min + scaled_counts[above_mask], bin_centers[above_mask],
                                    'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                fill_above = ax.fill_betweenx(bin_centers[above_mask],
                                              x_min, x_min + scaled_counts[above_mask],
                                              alpha=0.7, color='red', zorder=99)
                self._threshold_histogram.extend([line_above, fill_above])

        except Exception as e:
            print(f"[Fast Histogram Update] Error: {e}")
            # Fall back to full redraw
            self._clear_peak_height_histogram()
            self._show_peak_height_histogram()

    def _update_histogram_colors_for_blitting(self, new_threshold):
        """
        Update histogram colors during blitting (no draw calls).
        Creates new colored histogram artists and marks them as animated.
        """
        # If no cached data, skip
        if (self._histogram_bin_centers is None or
            self._histogram_scaled_counts is None):
            return

        try:
            import numpy as np

            # Get the axis
            if not self.fig.axes:
                return
            ax = self.fig.axes[0]

            # Remove old histogram artists
            if self._threshold_histogram:
                for artist in self._threshold_histogram:
                    try:
                        artist.remove()
                    except:
                        pass

            # Recreate with new threshold coloring
            bin_centers = self._histogram_bin_centers
            scaled_counts = self._histogram_scaled_counts
            x_min = self._histogram_x_min

            # Split into below and above threshold
            below_mask = bin_centers < new_threshold
            above_mask = bin_centers >= new_threshold

            self._threshold_histogram = []
            self._histogram_fills = []
            self._histogram_lines = []

            # Gray fill for below threshold (black outline, gray fill)
            if np.any(below_mask):
                line_below = ax.plot(x_min + scaled_counts[below_mask], bin_centers[below_mask],
                                    'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                fill_below = ax.fill_betweenx(bin_centers[below_mask],
                                              x_min, x_min + scaled_counts[below_mask],
                                              alpha=0.7, color='gray', zorder=99)
                # Mark as animated for blitting
                line_below.set_animated(True)
                fill_below.set_animated(True)
                self._histogram_fills.append(fill_below)
                self._histogram_lines.append(line_below)
                self._threshold_histogram.extend([line_below, fill_below])

            # Red fill for above threshold (black outline, red fill)
            if np.any(above_mask):
                line_above = ax.plot(x_min + scaled_counts[above_mask], bin_centers[above_mask],
                                    'k-', linewidth=1.5, alpha=1.0, zorder=100)[0]
                fill_above = ax.fill_betweenx(bin_centers[above_mask],
                                              x_min, x_min + scaled_counts[above_mask],
                                              alpha=0.7, color='red', zorder=99)
                # Mark as animated for blitting
                line_above.set_animated(True)
                fill_above.set_animated(True)
                self._histogram_fills.append(fill_above)
                self._histogram_lines.append(line_above)
                self._threshold_histogram.extend([line_above, fill_above])

        except Exception as e:
            print(f"[Blitting Histogram Update] Error: {e}")

    #                         t_on=None, y_on=None,
    #                         t_off=None, y_off=None,
    #                         t_exp=None, y_exp=None,
    #                         size=30):
    #     """
    #     Onsets:  green triangles up
    #     Offsets: orange triangles down
    #     Exp. min: blue squares
    #     """
    #         return
    # def update_breath_markers(self,
    #     if not self.fig.axes:
    #     ax = self.fig.axes[0]

    #         if tx is not None and ty is not None and len(tx) > 0:
    #         if pts is None:
    #             if sc is not None:
    #                 try:
    #                     sc.remove()
    #                 except Exception:
    #                     pass
    #                 setattr(self, scatter_attr, None)
    #             return
    #         if sc is None or sc.axes is not ax:
    #             setattr(self, scatter_attr,
    #                     ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, marker=marker, zorder=3))
    #         else:
    #             sc.set_offsets(pts)

    def update_breath_markers(
            self,
            t_on=None,  y_on=None,
            t_off=None, y_off=None,
            t_exp=None, y_exp=None,
            t_exoff=None, y_exoff=None,
            size=30):
        """
        Plot breath event markers at actual data points (no vertical offsets).

        Onsets  : green triangles up
        Offsets : orange triangles down
        Exp. min: blue squares
        Exp. off: purple diamonds
        """
        if not self.fig.axes:
            return
        # Use ax_main (primary Pleth axis) if available, otherwise fall back to first axis
        ax = self.ax_main if self.ax_main is not None else self.fig.axes[0]

        def _upd(scatter_attr, tx, ty, color, marker):
            import numpy as np
            pts = None
            if tx is not None and ty is not None and len(tx) > 0:
                # No vertical offset - plot at actual data points
                pts = np.column_stack([tx, ty])
            sc = getattr(self, scatter_attr)
            if pts is None:
                if sc is not None:
                    try: sc.remove()
                    except Exception: pass
                    setattr(self, scatter_attr, None)
                return
            if sc is None or sc.axes is not ax:
                setattr(self, scatter_attr,
                        ax.scatter(pts[:, 0], pts[:, 1], s=size, c=color, marker=marker,
                                 zorder=5, edgecolors='none', linewidths=0))
            else:
                sc.set_offsets(pts)

        # Plot markers at actual data points (no vertical offsets)
        _upd("scatter_onsets",  t_on,   y_on,   "limegreen", "^")
        _upd("scatter_offsets", t_off,  y_off,  "orange",    "v")
        _upd("scatter_expmins", t_exp,  y_exp,  "blue",      "s")
        _upd("scatter_expoffs", t_exoff,y_exoff,EXPOFF_COLOR,"D")

        self.canvas.draw_idle()

    def clear_breath_markers(self):
        for attr in ("scatter_onsets", "scatter_offsets", "scatter_expmins", "scatter_expoffs"):
            sc = getattr(self, attr, None)
            if sc is not None:
                try:
                    sc.remove()
                except Exception:
                    pass
                setattr(self, attr, None)
        self.canvas.draw_idle()


    #     """Create/update a right-axis (Y2) line over the main trace."""
    #         return
    # def add_or_update_y2(self, t, y2, label: str = "Y2", max_points: int | None = None):
    #     if not self.fig.axes:
    #     ax = self.fig.axes[0]

    #     # ensure twin axis exists
    #     if self.ax_y2 is None or self.ax_y2.axes is not ax:
    #         self.ax_y2 = ax.twinx()
    #         self.line_y2 = None

    #     # optional downsample

    #     else:

    #     self.canvas.draw_idle()

    def add_or_update_y2(self, t, y2, label: str = "Y2",
                        max_points: int | None = None,
                        color: str = "#39FF14"):
        """Create/update a right-axis (Y2) line over the main trace."""
        if not self.fig.axes:
            return
        # Use ax_main (primary Pleth axis) if available, otherwise fall back to first axis
        ax = self.ax_main if self.ax_main is not None else self.fig.axes[0]

        # ensure twin axis exists (robust to missing attrs)
        ax_y2 = getattr(self, "ax_y2", None)
        if ax_y2 is None or ax_y2.axes is not ax:
            self.ax_y2 = ax.twinx()
            self.line_y2 = None
            ax_y2 = self.ax_y2

            # IMPORTANT: Disable mouse events on Y2 axis to prevent blocking clicks on main axis
            # This allows editing modes (Mark Sniff, Add Peaks, etc.) to work when Y2 is displayed
            ax_y2.set_navigate(False)  # Disable navigation (zoom/pan)
            ax_y2.patch.set_visible(False)  # Make background transparent/non-interactive
            ax_y2.patch.set_picker(None)  # Explicitly disable picking on patch

            # Clean spines - now main axis needs right spine for Y2
            self._clean_spines(ax, has_y2=True)
            # Clean Y2 axis spines - only show right spine
            ax_y2.spines['top'].set_visible(False)
            ax_y2.spines['left'].set_visible(False)
            ax_y2.spines['bottom'].set_visible(False)
            ax_y2.spines['right'].set_visible(True)

            # Apply theme colors to Y2 axis
            tick_color = self.theme_manager.get_color('tick_color')
            label_color = self.theme_manager.get_color('label_color')
            edge_color = self.theme_manager.get_color('axes_edgecolor')
            ax_y2.spines['right'].set_edgecolor(edge_color)
            ax_y2.tick_params(colors=tick_color, which='both')
            ax_y2.yaxis.label.set_color(label_color)

            # Only show ticks/labels on the right for Y2
            ax_y2.tick_params(top=False, left=False, bottom=False, right=True,
                             labeltop=False, labelleft=False, labelbottom=False, labelright=True)

        # optional downsample
        import numpy as np
        from math import ceil
        def _down(t_, y_, m):
            if m is None or m <= 0 or len(t_) <= m:
                return t_, y_
            step = max(1, ceil(len(t_) / m))
            return t_[::step], y_[::step]

        tds, yds = _down(np.asarray(t), np.asarray(y2),
                        max_points if max_points else len(t))

        line_y2 = getattr(self, "line_y2", None)
        if line_y2 is None or line_y2.axes is not ax_y2:
            (self.line_y2,) = ax_y2.plot(tds, yds, linewidth=1.2, alpha=0.95, color=color, zorder=6)
            ax_y2.set_ylabel(label)
            # Nice white outline for visibility; safe no-op if unavailable
            try:
                import matplotlib.patheffects as pe
                self.line_y2.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
            except Exception:
                pass
        else:
            self.line_y2.set_data(tds, yds)
            self.line_y2.set_color(color)
            if ax_y2.get_ylabel() != label:
                ax_y2.set_ylabel(label)

        # rescale Y2 to the new data
        ax_y2.relim()
        ax_y2.autoscale_view()

        self.canvas.draw_idle()

    def add_or_update_y2_secondary(self, t, y2, color: str = "#2ecc71", max_points: int | None = None):
        """Add a secondary line to the Y2 axis (for combined plotting like eupnea+sniffing confidence)."""
        if not self.fig.axes or self.ax_y2 is None:
            return

        # Ensure Y2 axis exists (should be created by add_or_update_y2)
        ax_y2 = self.ax_y2

        # optional downsample
        import numpy as np
        from math import ceil
        def _down(t_, y_, m):
            if m is None or m <= 0 or len(t_) <= m:
                return t_, y_
            step = max(1, ceil(len(t_) / m))
            return t_[::step], y_[::step]

        tds, yds = _down(np.asarray(t), np.asarray(y2),
                        max_points if max_points else len(t))

        line_y2_sec = getattr(self, "line_y2_secondary", None)
        if line_y2_sec is None or line_y2_sec.axes is not ax_y2:
            (self.line_y2_secondary,) = ax_y2.plot(tds, yds, linewidth=1.2, alpha=0.95, color=color, zorder=6)
            # Add white outline for visibility
            try:
                import matplotlib.patheffects as pe
                self.line_y2_secondary.set_path_effects([pe.Stroke(linewidth=2.0, foreground="white"), pe.Normal()])
            except Exception:
                pass
        else:
            self.line_y2_secondary.set_data(tds, yds)
            self.line_y2_secondary.set_color(color)

        # Rescale Y2 to accommodate both lines
        ax_y2.relim()
        ax_y2.autoscale_view()

        self.canvas.draw_idle()

    def clear_y2(self):
        """Remove the Y2 axis/line(s) if present."""
        if self.line_y2 is not None:
            try: self.line_y2.remove()
            except Exception: pass
            self.line_y2 = None
        if self.line_y2_secondary is not None:
            try: self.line_y2_secondary.remove()
            except Exception: pass
            self.line_y2_secondary = None
        if self.ax_y2 is not None:
            try: self.ax_y2.remove()
            except Exception: pass
            self.ax_y2 = None
        self.canvas.draw_idle()


    def set_click_callback(self, fn):
        """fn will be called as fn(xdata, ydata, event) on single left-clicks."""
        self._external_click_cb = fn

    def clear_click_callback(self):
        self._external_click_cb = None


    #     if event.inaxes is None:
    #         return
    # def _on_button(self, event):

    #     # Forward single left-clicks to external callback if present
    #         # event.xdata / event.ydata are in data coords for the clicked Axes
    #         return
    #     if (self._external_click_cb is not None) and (event.button == 1) and (not event.dblclick):
    #         self._external_click_cb(event.xdata, event.ydata, event)

    #     # Keep your existing double-click autoscale behavior
    #     if event.dblclick:
    #         ax.autoscale()
    #         ax = event.inaxes
    #         self.canvas.draw_idle()
    #         self._store_from_axes(mode=("single" if len(self.fig.axes) == 1 else "grid"))

    def _on_button(self, event):
        if event.inaxes is None:
            return

        # Double-click: autoscale (keep your behavior)
        if event.dblclick:
            ax = event.inaxes
            ax.autoscale()
            self.canvas.draw_idle()
            self._store_from_axes(mode=("single" if len(self.fig.axes) == 1 else "grid"))
            return

        # Single-click (left or right): forward to external callback if present
        if self._external_click_cb is not None and event.xdata is not None:
            self._external_click_cb(event.xdata, event.ydata, event)




    #     """Draw yellow stars slightly above given (t,y) points."""
    #         return
    # def update_sighs(self, t, y, size=140, offset_frac=0.03):
    #     if not self.fig.axes:
    #     ax = self.fig.axes[0]

    #     # compute small vertical offset relative to current y-span
    #     ymin, ymax = ax.get_ylim()
    #     dy = (ymax - ymin) * float(offset_frac if offset_frac is not None else 0.03)
    #     y_plot = (np.asarray(y, dtype=float) + dy)

    #             np.asarray(t, dtype=float),
    #             y_plot,
    #             s=size,
    #             marker="*",
    #             color="yellow",
    #             edgecolors="black",
    #             linewidths=0.6,
    #             zorder=6,
    #             clip_on=False,
    #         )
    #     else:
    #     if self.scatter_sighs is None or self.scatter_sighs.axes is not ax:
    #         self.scatter_sighs = ax.scatter(
    #         self.scatter_sighs.set_offsets(np.c_[np.asarray(t, dtype=float), y_plot])

    #     # keep stars “above” even if limits change later
    #     ax.figure.canvas.draw_idle()

    #         try:
    #         except Exception:
    #             pass



    #     """
    #     Draw star markers at given points on the main axis.
    #     """
    #         return
    # def update_sighs(self, t, y, size=90, color="#f7c948"):
    #     if not hasattr(self, "fig") or not self.fig.axes:
    #     ax = self.fig.axes[0]

    #     # remove previous
    #     try:
    #     except Exception:
    #         pass
    #         if self._sigh_artist is not None:
    #             self._sigh_artist.remove()
    #     self._sigh_artist = None

    #     # draw hollow star for visibility over traces
    #         t, y,
    #         marker="*",
    #         s=size,
    #         facecolors="none",
    #         edgecolors=color,
    #         linewidths=1.8,
    #         zorder=7,
    #     )
    #     self._sigh_artist = ax.scatter(
    #     self.canvas.draw_idle()


    def update_sighs(self, t, y, size=90, color="#ff9f1a", edge=None, filled=True):
        """
        Draw star markers at given points on the main axis.
        """
        if not hasattr(self, "fig") or not self.fig.axes:
            return
        # Use ax_main (primary Pleth axis) if available, otherwise fall back to first axis
        ax = self.ax_main if self.ax_main is not None else self.fig.axes[0]

        # remove previous
        try:
            if self._sigh_artist is not None:
                self._sigh_artist.remove()
        except Exception:
            pass
        self._sigh_artist = None

        face = (color if filled else "none")
        if edge is None:
            edge = color

        self._sigh_artist = ax.scatter(
            t, y,
            marker="*",
            s=size,
            facecolors=face,
            edgecolors=edge,
            linewidths=1.5,
            zorder=50,  # High z-order to ensure stars are on top of other plot elements
        )
        self.canvas.draw_idle()

    def clear_sighs(self):
        try:
            if self._sigh_artist is not None:
                self._sigh_artist.remove()
        except Exception:
            pass
        self._sigh_artist = None
        if hasattr(self, "canvas"):
            self.canvas.draw_idle()

    # ------- Region overlays (eupnea/apnea/problems) -------
    def _draw_regions_with_mode(self, regions, use_shade, color, y_position_fraction=0.99):
        """
        Draw regions as either lines or shaded spans based on user preference.

        Args:
            regions: List of (start_t, end_t) tuples
            use_shade: If True, draw axvspan shading; if False, draw line at specified Y position
            color: Color string (e.g., '#2e7d32', 'red', 'purple', '#FFA500')
            y_position_fraction: For line mode, Y position as fraction of range (0.0 to 1.0)
                                0.01 = bottom, 0.50 = middle, 0.99 = top
        """
        if use_shade:
            # Full-height shaded background
            for start_t, end_t in regions:
                span = self.ax_main.axvspan(
                    start_t, end_t,
                    color=color, alpha=0.25,
                    zorder=1, linewidth=0
                )
                self._region_overlays.append(span)
        else:
            # Thin line at specified Y position
            y_lim = self.ax_main.get_ylim()
            y_val = y_lim[0] + y_position_fraction * (y_lim[1] - y_lim[0])

            for start_t, end_t in regions:
                line = self.ax_main.plot(
                    [start_t, end_t], [y_val, y_val],
                    color=color, linewidth=1.5,
                    alpha=0.8, zorder=10
                )[0]
                self._region_overlays.append(line)

    def update_region_overlays(self, t, eupnea_mask, apnea_mask, outlier_mask=None, failure_mask=None, sniff_regions=None, eupnea_regions=None, state=None):
        """
        Add overlays for eupnea, sniffing, apnea, outliers, and calculation failures.
        Display mode (line vs shade) is controlled by state variables.

        Args:
            t: time array
            eupnea_mask: binary array (0/1) indicating eupnic regions (frequency mode) OR None
            apnea_mask: binary array (0/1) indicating apneic regions
            outlier_mask: binary array (0/1) indicating outlier breath cycles (orange)
            failure_mask: binary array (0/1) indicating calculation failure cycles (red)
            sniff_regions: list of (start_t, end_t) tuples for sniffing regions (purple)
            eupnea_regions: list of (start_t, end_t) tuples for eupnea regions (green, GMM mode)
        """
        self.clear_region_overlays()

        if self.ax_main is None:
            return

        # Storage for overlay artists
        if not hasattr(self, '_region_overlays'):
            self._region_overlays = []

        # Get display modes from state (defaults to current behavior if not set)
        eupnea_shade = getattr(state, 'eupnea_use_shade', False) if state else False
        sniffing_shade = getattr(state, 'sniffing_use_shade', True) if state else True
        apnea_shade = getattr(state, 'apnea_use_shade', False) if state else False
        outliers_shade = getattr(state, 'outliers_use_shade', True) if state else True

        # Draw eupnea regions (green) - support both mask (frequency mode) and regions (GMM mode)
        regions_to_draw = None
        if eupnea_regions is not None and len(eupnea_regions) > 0:
            # GMM mode: use provided time regions directly
            regions_to_draw = eupnea_regions
        elif eupnea_mask is not None and len(eupnea_mask) == len(t):
            # Frequency mode: convert mask to regions
            regions_to_draw = self._extract_regions(t, eupnea_mask)

        if regions_to_draw:
            self._draw_regions_with_mode(
                regions_to_draw,
                use_shade=eupnea_shade,
                color='#2e7d32',  # Dark green
                y_position_fraction=0.99  # Top of plot
            )

        # Draw sniffing regions (purple) - same Y position as eupnea since they don't overlap
        if sniff_regions is not None and len(sniff_regions) > 0:
            self._draw_regions_with_mode(
                sniff_regions,
                use_shade=sniffing_shade,
                color='purple',
                y_position_fraction=0.99  # Top of plot (same as eupnea)
            )

        # Draw apnea regions (red)
        if apnea_mask is not None and len(apnea_mask) == len(t):
            apnea_regions = self._extract_regions(t, apnea_mask)
            self._draw_regions_with_mode(
                apnea_regions,
                use_shade=apnea_shade,
                color='red',
                y_position_fraction=0.01  # Bottom of plot
            )

        # Draw outlier regions (orange)
        if outlier_mask is not None and len(outlier_mask) == len(t):
            outlier_regions = self._extract_regions(t, outlier_mask)
            # Ensure minimum width for visibility (0.1 seconds)
            adjusted_regions = []
            for start_t, end_t in outlier_regions:
                width = end_t - start_t
                if width < 0.1:
                    mid = (start_t + end_t) / 2
                    start_t = mid - 0.05
                    end_t = mid + 0.05
                adjusted_regions.append((start_t, end_t))

            self._draw_regions_with_mode(
                adjusted_regions,
                use_shade=outliers_shade,
                color='#FFA500',  # Orange
                y_position_fraction=0.02  # Near bottom, just above apnea line
            )

        # Add failure regions (red background) - full height, visible rectangles
        if failure_mask is not None and len(failure_mask) == len(t):
            failure_regions = self._extract_regions(t, failure_mask)
            print(f"Debug: Found {len(failure_regions)} calculation failure regions")
            for start_t, end_t in failure_regions:
                width = end_t - start_t
                # Ensure minimum width for visibility (0.1 seconds)
                if width < 0.1:
                    mid = (start_t + end_t) / 2
                    start_t = mid - 0.05
                    end_t = mid + 0.05
                print(f"  Failure region: {start_t:.3f} to {end_t:.3f} (width={end_t-start_t:.3f}s)")
                # Full-height rectangle with red color (stronger than outliers)
                span = self.ax_main.axvspan(start_t, end_t,
                                          color='#FF0000', alpha=0.30,
                                          zorder=2, linewidth=0)  # zorder=2 so red shows over orange
                self._region_overlays.append(span)

        self.canvas.draw_idle()

    def _extract_regions(self, t, mask):
        """Extract continuous regions where mask == 1 or mask == True."""
        import numpy as np
        regions = []

        if len(mask) == 0 or len(t) == 0:
            return regions

        # Ensure mask is boolean
        mask = np.asarray(mask, dtype=bool)

        # Check if there are any True values
        if not np.any(mask):
            return regions

        # Find transitions
        diff_mask = np.diff(mask.astype(int))
        starts = np.where(diff_mask == 1)[0] + 1  # Start of True regions
        ends = np.where(diff_mask == -1)[0] + 1   # End of True regions

        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        # Convert indices to time values with bounds checking
        for start_idx, end_idx in zip(starts, ends):
            start_idx = int(start_idx)
            end_idx = int(end_idx)

            if start_idx < 0 or start_idx >= len(t):
                continue
            if end_idx < 0 or end_idx > len(t):
                end_idx = len(t)

            # Get time values (use end_idx-1 since end_idx is exclusive)
            start_t = float(t[start_idx])
            end_t = float(t[min(end_idx - 1, len(t) - 1)])

            if end_t > start_t:  # Ensure valid region
                regions.append((start_t, end_t))

        return regions

    def clear_region_overlays(self):
        """Clear all region overlay lines."""
        if hasattr(self, '_region_overlays'):
            for artist in self._region_overlays:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._region_overlays.clear()
