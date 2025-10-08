# core/plotting.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from PyQt6 import QtCore
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import matplotlib.patheffects as pe

# colors
PLETH_COLOR = "k"           # main trace
PEAK_COLOR  = "red"         # inspiratory peaks
ONSET_COLOR = "#2ecc71"     # green (inspiratory onset)
OFFSET_COLOR= "#f39c12"     # orange (inspiratory offset/expiratory onset)
EXPMIN_COLOR= "#1f78b4"     # blue (expiratory minimum)
EXPOFF_COLOR = "#9b59b6"   # purple (expiratory offset)


class PlotHost(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Figure/canvas
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)

        # Toolbar (clean, dark buttons, no bar background)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setObjectName("PlotNavToolbar")
        self.toolbar.setIconSize(QtCore.QSize(15, 15))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("""
        QToolBar#PlotNavToolbar { background: transparent; border: none; padding: 0px; }
        QToolBar#PlotNavToolbar::separator { background: transparent; width: 0px; height: 0px; }
        QToolBar#PlotNavToolbar::handle { image: none; width: 0px; height: 0px; }
        QToolBar#PlotNavToolbar QToolButton {
            background: #434b5d; color: #eef2f8;
            border: 1px solid #5a6580; border-radius: 8px;
            padding: 5px 8px; margin: 2px;
        }
        QToolBar#PlotNavToolbar QToolButton:hover { background: #515c72; border-color: #6a7694; }
        QToolBar#PlotNavToolbar QToolButton:pressed { background: #5f6d88; border-color: #7886a6; }
        QToolBar#PlotNavToolbar QToolButton:checked {
            background: #4A90E2;
            color: white;
            border-color: #357ABD;
        }
        QToolBar#PlotNavToolbar QToolButton:disabled {
            background: #353b4a; border-color: #444d60; color: #8691a8;
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

        #Continuious Breath metrics
        self.ax_y2 = None
        self.line_y2 = None

        # Add mode for external click callback
        self._external_click_cb = None



        


        
        # Layout
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

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

        # Interactions (optional helpers)
        self._cid_scroll = self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._cid_button = self.canvas.mpl_connect('button_press_event', self._on_button)

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
        if event.dblclick and event.inaxes is not None:
            ax = event.inaxes
            ax.autoscale()
            self.canvas.draw_idle()
            self._store_from_axes(mode=("single" if len(self.fig.axes) == 1 else "grid"))

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
    #     self.clear_peaks()
    #     # Reset old overlay references (figure was cleared)
    #     self.scatter_peaks = None
    #     self.scatter_onsets = None
    #     self.scatter_offsets = None
    #     self.scatter_expmins = None


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
    #         self.ax_main.set_xlim(prev_xlim)
    #     if prev_ylim is not None:
    #         self.ax_main.set_ylim(prev_ylim)

    #     self._attach_limit_listeners([self.ax_main], mode="single")
    #     self._store_from_axes(mode="single")
    #     self.canvas.draw_idle()

    def show_trace_with_spans(self, t, y, spans_s, title: str = "", max_points: int = 2000, ylabel: str = "Signal"):
        prev_xlim = self._last_single["xlim"] if self._preserve_x else None
        prev_ylim = self._last_single["ylim"] if self._preserve_y else None

        self.fig.clear()
        self.ax_main = self.fig.add_subplot(111)

        # Any old scatter is invalid after clearing
        self.clear_peaks()
        self.scatter_peaks = None
        self.scatter_onsets = None
        self.scatter_offsets = None
        self.scatter_expmins = None

        # Plot main trace in black
        tds, yds = self._downsample_even(t, y, max_points=max_points if max_points else len(t))
        self.ax_main.plot(tds, yds, linewidth=0.9, color=PLETH_COLOR)

        # Dashed baseline at y=0 (behind traces)
        self.ax_main.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Optional shaded spans
        for (t0, t1) in (spans_s or []):
            if t1 > t0:
                self.ax_main.axvspan(t0, t1, color="#2E5090", alpha=0.25)

        if title:
            self.ax_main.set_title(title)
        self.ax_main.set_xlabel("Time (s)")
        self.ax_main.set_ylabel(ylabel)

        # Single-panel: no grid
        self.ax_main.grid(False)

        # Restore preserved view if desired
        if prev_xlim is not None:
            self.ax_main.set_xlim(prev_xlim)
        if prev_ylim is not None:
            self.ax_main.set_ylim(prev_ylim)

        self._attach_limit_listeners([self.ax_main], mode="single")
        self._store_from_axes(mode="single")

        # Keep layout tight always
        self.fig.tight_layout()
        self.canvas.draw_idle()



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

    #     if self.scatter_peaks is not None:
    #         try:
    #             self.scatter_peaks.remove()
    #         except Exception:
    #             pass
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
        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]

        for ax, (t, y, label) in zip(axes, traces):
            tds, yds = self._downsample_even(t, y, max_points=max_points_per_trace)
            ax.plot(tds, yds, linewidth=1, color=PLETH_COLOR)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (s)")
        if title:
            axes[0].set_title(title)

        if prev_xlim is not None:
            for ax in axes:
                ax.set_xlim(prev_xlim)
        if prev_ylims is not None and len(prev_ylims) == len(axes):
            for ax, yl in zip(axes, prev_ylims):
                ax.set_ylim(yl)

        self._attach_limit_listeners(axes, mode="grid")
        self._store_from_axes(mode="grid")
        self.canvas.draw_idle()

    # ------- Qt hook -------
    def resizeEvent(self, event):
        # keep layout tight in single-panel mode
        if len(self.fig.axes) == 1:
            self.fig.tight_layout()
        self.canvas.draw_idle()
        return super().resizeEvent(event)


    def update_peaks(self, t_peaks, y_peaks, size=24):
        """Create/update red dots for inspiratory peaks."""
        if not self.fig.axes:
            return
        ax = self.fig.axes[0]
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

    # def update_breath_markers(self,
    #                         t_on=None, y_on=None,
    #                         t_off=None, y_off=None,
    #                         t_exp=None, y_exp=None,
    #                         size=30):
    #     """
    #     Onsets:  green triangles up
    #     Offsets: orange triangles down
    #     Exp. min: blue squares
    #     """
    #     if not self.fig.axes:
    #         return
    #     ax = self.fig.axes[0]

    #     def _upd(scatter_attr, tx, ty, color, marker):
    #         import numpy as np
    #         pts = None
    #         if tx is not None and ty is not None and len(tx) > 0:
    #             pts = np.column_stack([tx, ty])
    #         sc = getattr(self, scatter_attr)
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
        Onsets  : green triangles up (at signal value)
        Offsets : orange triangles down (at signal value)
        Exp. min: blue squares (vertically offset below signal)
        Exp. off: purple diamonds (vertically offset further below signal)

        Vertical offsetting helps visualize when markers overlap at same location.
        """
        if not self.fig.axes:
            return
        ax = self.fig.axes[0]

        # Get y-axis range for computing vertical offsets
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]

        # Vertical offset amounts (as fraction of y-range)
        # Just enough to distinguish overlapping markers
        offset_expmin = -0.001 * y_range   # 0.1% below signal
        offset_expoff = +0.001 * y_range   # 0.1% above signal

        def _upd(scatter_attr, tx, ty, color, marker, y_offset=0):
            import numpy as np
            pts = None
            if tx is not None and ty is not None and len(tx) > 0:
                # Apply vertical offset
                ty_offset = np.asarray(ty) + y_offset
                pts = np.column_stack([tx, ty_offset])
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
                                 zorder=5, edgecolors='white', linewidths=0.5))
            else:
                sc.set_offsets(pts)

        # Plot markers with vertical offsets
        _upd("scatter_onsets",  t_on,   y_on,   "limegreen", "^", y_offset=0)           # No offset
        _upd("scatter_offsets", t_off,  y_off,  "orange",    "v", y_offset=0)           # No offset
        _upd("scatter_expmins", t_exp,  y_exp,  "blue",      "s", y_offset=offset_expmin)  # 3% below
        _upd("scatter_expoffs", t_exoff,y_exoff,EXPOFF_COLOR,"D", y_offset=offset_expoff)  # 6% below

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


    # def add_or_update_y2(self, t, y2, label: str = "Y2", max_points: int | None = None):
    #     """Create/update a right-axis (Y2) line over the main trace."""
    #     if not self.fig.axes:
    #         return
    #     ax = self.fig.axes[0]

    #     # ensure twin axis exists
    #     if self.ax_y2 is None or self.ax_y2.axes is not ax:
    #         self.ax_y2 = ax.twinx()
    #         self.line_y2 = None

    #     # optional downsample
    #     import numpy as np
    #     from math import ceil
    #     def _down(t_, y_, m):
    #         if m is None or m <= 0 or len(t_) <= m:
    #             return t_, y_
    #         step = max(1, ceil(len(t_) / m))
    #         return t_[::step], y_[::step]

    #     tds, yds = _down(np.asarray(t), np.asarray(y2), max_points if max_points else len(t))
    #     if self.line_y2 is None or self.line_y2.axes is not self.ax_y2:
    #         (self.line_y2,) = self.ax_y2.plot(tds, yds, linewidth=1.0)
    #         self.ax_y2.set_ylabel(label)
    #     else:
    #         self.line_y2.set_data(tds, yds)

    #     self.canvas.draw_idle()

    def add_or_update_y2(self, t, y2, label: str = "Y2",
                        max_points: int | None = None,
                        color: str = "#39FF14"):
        """Create/update a right-axis (Y2) line over the main trace."""
        if not self.fig.axes:
            return
        ax = self.fig.axes[0]

        # ensure twin axis exists (robust to missing attrs)
        ax_y2 = getattr(self, "ax_y2", None)
        if ax_y2 is None or ax_y2.axes is not ax:
            self.ax_y2 = ax.twinx()
            self.line_y2 = None
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



    def clear_y2(self):
        """Remove the Y2 axis/line if present."""
        if self.line_y2 is not None:
            try: self.line_y2.remove()
            except Exception: pass
            self.line_y2 = None
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


    # def _on_button(self, event):
    #     if event.inaxes is None:
    #         return

    #     # Forward single left-clicks to external callback if present
    #     if (self._external_click_cb is not None) and (event.button == 1) and (not event.dblclick):
    #         # event.xdata / event.ydata are in data coords for the clicked Axes
    #         self._external_click_cb(event.xdata, event.ydata, event)
    #         return

    #     # Keep your existing double-click autoscale behavior
    #     if event.dblclick:
    #         ax = event.inaxes
    #         ax.autoscale()
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




    # def update_sighs(self, t, y, size=140, offset_frac=0.03):
    #     """Draw yellow stars slightly above given (t,y) points."""
    #     if not self.fig.axes:
    #         return
    #     ax = self.fig.axes[0]

    #     # compute small vertical offset relative to current y-span
    #     ymin, ymax = ax.get_ylim()
    #     dy = (ymax - ymin) * float(offset_frac if offset_frac is not None else 0.03)
    #     y_plot = (np.asarray(y, dtype=float) + dy)

    #     if self.scatter_sighs is None or self.scatter_sighs.axes is not ax:
    #         self.scatter_sighs = ax.scatter(
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
    #         self.scatter_sighs.set_offsets(np.c_[np.asarray(t, dtype=float), y_plot])

    #     # keep stars “above” even if limits change later
    #     ax.figure.canvas.draw_idle()

    # def clear_sighs(self):
    #     if self.scatter_sighs is not None:
    #         try:
    #             self.scatter_sighs.remove()
    #         except Exception:
    #             pass
    #         self.scatter_sighs = None
    #         if self.fig:
    #             self.fig.canvas.draw_idle()



    # def update_sighs(self, t, y, size=90, color="#f7c948"):
    #     """
    #     Draw star markers at given points on the main axis.
    #     """
    #     if not hasattr(self, "fig") or not self.fig.axes:
    #         return
    #     ax = self.fig.axes[0]

    #     # remove previous
    #     try:
    #         if self._sigh_artist is not None:
    #             self._sigh_artist.remove()
    #     except Exception:
    #         pass
    #     self._sigh_artist = None

    #     # draw hollow star for visibility over traces
    #     self._sigh_artist = ax.scatter(
    #         t, y,
    #         marker="*",
    #         s=size,
    #         facecolors="none",
    #         edgecolors=color,
    #         linewidths=1.8,
    #         zorder=7,
    #     )
    #     self.canvas.draw_idle()


    def update_sighs(self, t, y, size=90, color="#ff9f1a", edge=None, filled=True):
        """
        Draw star markers at given points on the main axis.
        """
        if not hasattr(self, "fig") or not self.fig.axes:
            return
        ax = self.fig.axes[0]

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
            zorder=8,
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
    def update_region_overlays(self, t, eupnea_mask, apnea_mask, outlier_mask=None, failure_mask=None):
        """
        Add horizontal line overlays for eupnea (green lines), apnea (red lines),
        outliers (orange background), and calculation failures (red background).

        Args:
            t: time array
            eupnea_mask: binary array (0/1) indicating eupnic regions
            apnea_mask: binary array (0/1) indicating apneic regions
            outlier_mask: binary array (0/1) indicating outlier breath cycles (orange)
            failure_mask: binary array (0/1) indicating calculation failure cycles (red)
        """
        self.clear_region_overlays()

        if self.ax_main is None:
            return

        # Get current y-axis limits to position the overlay lines
        ylim = self.ax_main.get_ylim()
        y_eupnea = ylim[1] - 0.01 * (ylim[1] - ylim[0])  # 99% of y-range (near top)
        y_apnea = ylim[0] + 0.01 * (ylim[1] - ylim[0])   # 1% of y-range (near bottom)

        # Storage for overlay artists
        if not hasattr(self, '_region_overlays'):
            self._region_overlays = []

        # Add eupnea regions (thin black lines at top)
        if eupnea_mask is not None and len(eupnea_mask) == len(t):
            eupnea_regions = self._extract_regions(t, eupnea_mask)
            for start_t, end_t in eupnea_regions:
                line = self.ax_main.plot([start_t, end_t], [y_eupnea, y_eupnea],
                                       color='#2e7d32', linewidth=1.5, alpha=0.8, zorder=10)[0]
                self._region_overlays.append(line)

        # Add apnea regions (thin red lines at bottom)
        if apnea_mask is not None and len(apnea_mask) == len(t):
            apnea_regions = self._extract_regions(t, apnea_mask)
            for start_t, end_t in apnea_regions:
                line = self.ax_main.plot([start_t, end_t], [y_apnea, y_apnea],
                                       color='red', linewidth=1.5, alpha=0.8, zorder=10)[0]
                self._region_overlays.append(line)

        # Add outlier regions (orange background) - full height, visible rectangles
        if outlier_mask is not None and len(outlier_mask) == len(t):
            outlier_regions = self._extract_regions(t, outlier_mask)
            print(f"Debug: Found {len(outlier_regions)} outlier regions")
            for start_t, end_t in outlier_regions:
                width = end_t - start_t
                # Ensure minimum width for visibility (0.1 seconds)
                if width < 0.1:
                    mid = (start_t + end_t) / 2
                    start_t = mid - 0.05
                    end_t = mid + 0.05
                print(f"  Outlier region: {start_t:.3f} to {end_t:.3f} (width={end_t-start_t:.3f}s)")
                # Full-height rectangle with orange color
                span = self.ax_main.axvspan(start_t, end_t,
                                          color='#FFA500', alpha=0.25,
                                          zorder=1, linewidth=0)
                self._region_overlays.append(span)

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
