"""
Navigation bar widgets for recording navigation below the plot.

Provides two interchangeable navigation styles:
- ScrollBarNavigation: Slim scrollbar with proportional thumb
- MinimapNavigation: Thin waveform overview with draggable viewport

Both emit view_range_requested(x_min, x_max) for the plot host to consume.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollBar, QSizePolicy,
    QPushButton, QLabel,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt, pyqtSignal, QSettings, QPropertyAnimation, QEasingCurve, pyqtProperty, QTimer


class NavigationBarBase(QWidget):
    """Abstract base for navigation widgets below the plot."""

    view_range_requested = pyqtSignal(float, float)  # x_min, x_max

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data_min = 0.0
        self._data_max = 1.0
        self._guard = False  # Re-entrancy guard
        self._hover_mode = False  # Show only on hover
        self._hover_visible = False  # Currently showing due to hover

    def set_data_range(self, t_min: float, t_max: float):
        """Update the full extent of the recording."""
        self._data_min = t_min
        self._data_max = t_max

    def set_view_range(self, x_min: float, x_max: float):
        """Update the visible viewport (called when plot pans/zooms)."""
        raise NotImplementedError

    def set_data(self, t_array, signal_array):
        """Provide waveform data for minimap rendering. Scrollbar ignores this."""
        pass


class ScrollBarNavigation(NavigationBarBase):
    """Plain QScrollBar with proportional thumb — slim dark scrubber style."""

    # Fixed resolution to avoid integer overflow for any recording length
    _RESOLUTION = 1_000_000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal, self)
        self._scrollbar.setRange(0, self._RESOLUTION)
        self._scrollbar.setSingleStep(max(1, self._RESOLUTION // 200))
        self._scrollbar.setPageStep(self._RESOLUTION // 10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._scrollbar)

        self.setFixedHeight(14)

        # Dark theme style
        self._scrollbar.setStyleSheet("""
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 14px;
                border: none;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background-color: #555555;
                border-radius: 5px;
                min-width: 30px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #007acc;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                width: 0px;
            }
            QScrollBar::add-page, QScrollBar::sub-page {
                background: none;
            }
        """)

        self._scrollbar.valueChanged.connect(self._on_value_changed)

    def set_data_range(self, t_min: float, t_max: float):
        super().set_data_range(t_min, t_max)

    def set_view_range(self, x_min: float, x_max: float):
        if self._guard:
            return
        data_span = self._data_max - self._data_min
        if data_span <= 0:
            return
        view_span = x_max - x_min

        # Convert to scrollbar units
        page = max(1, int((view_span / data_span) * self._RESOLUTION))
        val = int(((x_min - self._data_min) / data_span) * self._RESOLUTION)

        self._scrollbar.blockSignals(True)
        self._scrollbar.setPageStep(page)
        self._scrollbar.setMaximum(max(0, self._RESOLUTION - page))
        self._scrollbar.setValue(max(0, min(val, self._scrollbar.maximum())))
        self._scrollbar.blockSignals(False)

    def _on_value_changed(self, value):
        if self._guard:
            return
        data_span = self._data_max - self._data_min
        if data_span <= 0:
            return

        max_val = self._scrollbar.maximum()
        page = self._scrollbar.pageStep()
        total = max_val + page
        if total <= 0:
            return

        frac = value / total
        view_span = (page / total) * data_span
        x_min = self._data_min + frac * data_span
        x_max = x_min + view_span

        self._guard = True
        try:
            self.view_range_requested.emit(x_min, x_max)
        finally:
            self._guard = False


class MinimapNavigation(NavigationBarBase):
    """Collapsible minimap: thin position bar that expands to waveform on hover."""

    COLLAPSED_HEIGHT = 6
    EXPANDED_HEIGHT = 36
    NAV_BAR_HEIGHT = 34  # Height for the navigation bar below minimap

    def __init__(self, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        self._expanded = True
        self._expandable = False  # Whether hover-expand is enabled

        self._plot = pg.PlotWidget(background='#0a0a0a')
        self._plot.hideAxis('bottom')
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setMenuEnabled(False)
        self._plot.getPlotItem().getViewBox().setDefaultPadding(0)
        # Match main plot's left axis width for alignment
        self._plot.getPlotItem().getAxis('left').setWidth(70)
        self._plot.getPlotItem().getAxis('left').setStyle(showValues=False)
        self._plot.getPlotItem().getAxis('left').setPen(pg.mkPen(None))

        # Dark overlays outside viewport — dims non-visible regions
        self._left_shade = pg.LinearRegionItem(
            values=[0, 0], movable=False,
            brush=pg.mkBrush(0, 0, 0, 140),
            pen=pg.mkPen(None),
        )
        self._right_shade = pg.LinearRegionItem(
            values=[1, 1], movable=False,
            brush=pg.mkBrush(0, 0, 0, 140),
            pen=pg.mkPen(None),
        )
        self._left_shade.setZValue(8)
        self._right_shade.setZValue(8)
        self._plot.addItem(self._left_shade)
        self._plot.addItem(self._right_shade)

        # Viewport region — green to distinguish from blue stim markers
        self._region = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(46, 204, 113, 35),       # subtle green fill
            pen=pg.mkPen('#2ecc71', width=2),          # green edge lines
            hoverBrush=pg.mkBrush(46, 204, 113, 55),   # brighter on hover
            hoverPen=pg.mkPen('#5dde9e', width=3),     # thicker edges on hover
            movable=True,
        )
        self._region.setZValue(10)
        self._plot.addItem(self._region)

        # Style edge lines: arrows pointing outward + resize cursor
        left_line, right_line = self._region.lines
        left_line.setCursor(Qt.CursorShape.SizeHorCursor)
        left_line.addMarker('|>', position=0.5, size=10)   # arrow pointing right (inward)
        right_line.setCursor(Qt.CursorShape.SizeHorCursor)
        right_line.addMarker('<|', position=0.5, size=10)  # arrow pointing left (inward)

        # Sync dark overlays when region moves
        self._region.sigRegionChanged.connect(self._update_shade_regions)

        # Duration label — shown while dragging edges
        self._duration_label = pg.TextItem(
            text='', color='#ffffff', anchor=(0.5, 1.0),
        )
        self._duration_label.setZValue(20)
        self._duration_label.setVisible(False)
        font = self._duration_label.textItem.font()
        font.setPointSize(8)
        font.setBold(True)
        self._duration_label.setFont(font)
        self._plot.addItem(self._duration_label)

        # Track edge dragging for duration tooltip
        self._dragging_edge = False
        for line in self._region.lines:
            line.sigDragged.connect(self._on_edge_dragged)
        self._region.sigRegionChangeFinished.connect(self._on_edge_drag_finished)

        # Thin center line visible in collapsed state
        self._center_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen('#444444', width=1),
        )
        self._center_line.setZValue(5)
        self._plot.addItem(self._center_line)

        self._traces = None  # List of PlotDataItem references
        self._marker_items = []  # Event marker visual items

        # Saved view state for home button toggle
        self._saved_view = None  # (x_min, x_max) before home was pressed
        self._is_home = False    # True when showing full-sweep autoscale view

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._plot)

        # Toolbar below minimap:
        # [edit icons] | ... [sweep] [Prev] [Home] [Next] [mode] ... | [save] [preview] [export] [...]
        self._nav_bar = QWidget(self)
        nav_layout = QHBoxLayout(self._nav_bar)
        nav_layout.setContentsMargins(70, 0, 4, 0)  # Left margin matches axis width
        nav_layout.setSpacing(5)

        sz = 30  # Button size

        nav_btn = (
            f"QPushButton {{ background: rgba(30,30,30,200); color: #ccc; "
            f"border: 1px solid #555; border-radius: 4px; padding: 2px 6px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: rgba(0,122,204,220); color: white; }}"
        )
        lbl_style = "color: #999; font-size: 11px; background: transparent;"
        sep_style = "color: #444; background: transparent; font-size: 14px;"

        def _edit_btn(text, tip, bg, fg="white", border=None, checkable=False):
            b = QPushButton(text)
            b.setFixedSize(sz, sz)
            b._icon_text = text  # Preserve icon text from setText overrides
            border_css = f"border: 1px solid {border}; " if border else ""
            b.setStyleSheet(
                f"QPushButton {{ background: {bg}; color: {fg}; {border_css}"
                f"border-radius: 4px; font-size: 15px; font-weight: bold; }}"
                f"QPushButton:hover {{ opacity: 0.8; }}"
                f"QPushButton:checked {{ "
                f"background: #00bcd4; color: #000; "
                f"border: 2px solid #00e5ff; }}"
            )
            b.setToolTip(tip)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if checkable:
                b.setCheckable(True)
            # Override setText to preserve icon — EditingModes calls setText("Merge (ON)")
            _orig_setText = b.setText
            def _guard_setText(txt, _orig=_orig_setText, _btn=b):
                _orig(_btn._icon_text)  # Always keep the icon character
            b.setText = _guard_setText
            return b

        def _sep():
            s = QLabel("\u2502")
            s.setStyleSheet(sep_style)
            s.setFixedWidth(10)
            s.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return s

        # === Left: Editing buttons ===
        self.btn_add_peaks = _edit_btn("\u00B1", "Add/Delete Peaks \u2014 click trace to add, near peak to delete", "#4CAF50")
        self.btn_merge = _edit_btn("M", "Merge Breaths \u2014 click near peaks to merge", "#FFEBEE", fg="#C62828", border="#E53935")
        self.btn_sniff = _edit_btn("S", "Mark Sniff \u2014 click and drag to highlight sniffing", "#9B59B6", border="#8E44AD", checkable=True)
        self.btn_move = _edit_btn("\u21c6", "Move Point \u2014 click marker, arrows to move, Enter to save", "#FFA500", fg="#1F2937", border="#FF8C00")
        self.btn_omit = _edit_btn("\u2298", "Omit Sweep \u2014 exclude from analysis", "#ECEFF4", fg="#1F2937", border="#C7D0DD")
        self.btn_editor = _edit_btn("\u270e", "Advanced Peak Editor", "transparent", fg="#4A9EFF", border="#4A9EFF")

        for btn in [self.btn_add_peaks, self.btn_merge, self.btn_sniff,
                    self.btn_move, self.btn_omit, self.btn_editor]:
            nav_layout.addWidget(btn)

        nav_layout.addWidget(_sep())
        nav_layout.addStretch(1)

        # === Center: Navigation (sweep counter + prev/home/next + mode) ===
        self._sweep_label = QLabel("")
        self._sweep_label.setStyleSheet(lbl_style)
        self._sweep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sweep_label.setFixedHeight(sz)
        nav_layout.addWidget(self._sweep_label)

        nav_color_btn = (
            f"QPushButton {{ background: rgba(44,62,80,220); color: #ddd; "
            f"border: 1px solid #5a7a9a; border-radius: 4px; padding: 2px 6px; font-size: 13px; }}"
            f"QPushButton:hover {{ background: rgba(52,152,219,220); color: white; border-color: #3498db; }}"
        )
        home_btn = (
            f"QPushButton {{ background: rgba(44,62,80,220); color: #f0c040; "
            f"border: 1px solid #5a7a9a; border-radius: 4px; padding: 2px 6px; font-size: 15px; }}"
            f"QPushButton:hover {{ background: rgba(52,152,219,220); color: #ffe066; border-color: #3498db; }}"
        )

        self._btn_prev = QPushButton("\u25C0")
        self._btn_prev.setFixedSize(sz, sz)
        self._btn_prev.setStyleSheet(nav_color_btn)
        self._btn_prev.setToolTip("Previous sweep/window")
        self._btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)

        self._btn_home = QPushButton("\U0001F3E0")
        self._btn_home.setFixedSize(sz, sz)
        self._btn_home.setStyleSheet(home_btn)
        self._btn_home.setToolTip("Reset view (click again to restore previous zoom)")
        self._btn_home.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_home.clicked.connect(self._on_home_clicked)

        self._btn_next = QPushButton("\u25B6")
        self._btn_next.setFixedSize(sz, sz)
        self._btn_next.setStyleSheet(nav_color_btn)
        self._btn_next.setToolTip("Next sweep/window")
        self._btn_next.setCursor(Qt.CursorShape.PointingHandCursor)

        nav_layout.addWidget(self._btn_prev)
        nav_layout.addWidget(self._btn_home)
        nav_layout.addWidget(self._btn_next)

        self._btn_mode = QPushButton("Sweep")
        self._btn_mode.setFixedHeight(sz)
        self._btn_mode.setStyleSheet(nav_color_btn)
        self._btn_mode.setToolTip("Toggle Sweep/Window mode")
        self._btn_mode.setCursor(Qt.CursorShape.PointingHandCursor)
        nav_layout.addWidget(self._btn_mode)

        nav_layout.addStretch(1)

        # Balance spacer — same width as left editing section so nav stays centered
        _right_balance = QWidget()
        _right_balance.setFixedWidth(6 * (sz + 5) + 10)
        _right_balance.setStyleSheet("background: transparent;")
        nav_layout.addWidget(_right_balance)

        # Signals for external wiring
        self.navigate_prev = self._btn_prev.clicked
        self.navigate_next = self._btn_next.clicked
        self.toggle_mode = self._btn_mode.clicked

        layout.addWidget(self._nav_bar)

        # Start expanded (always visible)
        self._apply_height(self.EXPANDED_HEIGHT + self.NAV_BAR_HEIGHT)
        self._set_expanded_visuals(True)

        self._region.sigRegionChanged.connect(self._on_region_changed)
        self.setMouseTracking(True)
        self._plot.setMouseTracking(True)

        # Animation for smooth expand/collapse
        self._anim = QPropertyAnimation(self, b"animHeight")
        self._anim.setDuration(150)  # ms
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        # Debounce timer for leave events
        self._leave_timer = QTimer(self)
        self._leave_timer.setSingleShot(True)
        self._leave_timer.setInterval(200)  # ms delay before collapsing
        self._leave_timer.timeout.connect(self._do_collapse)

    def _get_anim_height(self):
        return self.height()

    def _set_anim_height(self, h):
        self._apply_height(int(h))

    animHeight = pyqtProperty(int, _get_anim_height, _set_anim_height)

    def _apply_height(self, h):
        plot_h = max(self.COLLAPSED_HEIGHT, h - self.NAV_BAR_HEIGHT)
        self._plot.setFixedHeight(plot_h)
        self.setFixedHeight(h)
        self._nav_bar.setVisible(h > self.COLLAPSED_HEIGHT + 5)

    def _set_expanded_visuals(self, expanded):
        """Toggle trace/marker visibility for expanded vs collapsed state."""
        self._expanded = expanded
        if self._traces:
            for item in self._traces:
                item.setVisible(expanded)
        self._center_line.setVisible(not expanded)
        # Marker items are always visible (ticks in collapsed, full in expanded)
        if expanded and self._traces:
            n_ch = len(self._traces)
            self._plot.setYRange(-0.1, n_ch + 0.1, padding=0)
        else:
            self._plot.setYRange(0, 1, padding=0)

    def _on_anim_finished(self):
        """After animation completes, update visuals if collapsing."""
        if not self._expanded:
            self._set_expanded_visuals(False)

    def _animate_to(self, target_height):
        self._anim.stop()
        self._anim.setStartValue(self.height())
        self._anim.setEndValue(target_height)
        self._anim.start()

    def _do_expand(self):
        self._leave_timer.stop()
        target = self.EXPANDED_HEIGHT + self.NAV_BAR_HEIGHT
        if self._expanded and self.height() == target:
            return
        self._set_expanded_visuals(True)
        self._animate_to(target)

    def _do_collapse(self):
        if not self._expanded:
            return
        self._expanded = False
        self._animate_to(self.COLLAPSED_HEIGHT)
        # Visuals update when animation finishes (_on_anim_finished)

    def set_expandable(self, expandable):
        """Enable/disable hover-to-expand behavior.

        When disabled, the minimap stays permanently expanded.
        When enabled, it collapses and only expands on hover.
        """
        self._expandable = expandable
        if expandable and self._expanded:
            # Switching to hover mode — start collapsed
            self._do_collapse()
        elif not expandable and not self._expanded:
            # Switching to always-visible — expand now
            self._do_expand()

    def enterEvent(self, event):
        super().enterEvent(event)
        if self._expandable:
            self._do_expand()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        if self._expandable:
            self._leave_timer.start()

    def _update_shade_regions(self):
        """Sync dark overlays to the current viewport region."""
        x_min, x_max = self._region.getRegion()
        self._left_shade.blockSignals(True)
        self._right_shade.blockSignals(True)
        self._left_shade.setRegion([self._data_min - 1, x_min])
        self._right_shade.setRegion([x_max, self._data_max + 1])
        self._left_shade.blockSignals(False)
        self._right_shade.blockSignals(False)

    def _on_edge_dragged(self):
        """Show duration tooltip while an edge is being dragged."""
        x_min, x_max = self._region.getRegion()
        duration = x_max - x_min
        if duration >= 60:
            text = f"{duration / 60:.1f} min"
        else:
            text = f"{duration:.1f}s"
        center_x = (x_min + x_max) / 2.0
        # Position at top of the plot area
        y_range = self._plot.getPlotItem().getViewBox().viewRange()[1]
        self._duration_label.setText(text)
        self._duration_label.setPos(center_x, y_range[1])
        self._duration_label.setVisible(True)
        self._dragging_edge = True

    def _on_edge_drag_finished(self):
        """Hide duration tooltip when drag ends."""
        if self._dragging_edge:
            # Hide after a short delay so user can read the final value
            QTimer.singleShot(800, self._hide_duration_label)
            self._dragging_edge = False

    def _hide_duration_label(self):
        self._duration_label.setVisible(False)

    def set_sweep_info(self, current: int, total: int):
        """Update the sweep counter label (1-indexed for display)."""
        if total <= 1:
            self._sweep_label.setText("")
        else:
            self._sweep_label.setText(f"{current + 1}/{total}")

    def set_mode_label(self, mode: str):
        """Update the mode toggle button text."""
        self._btn_mode.setText("Sweep" if mode == "sweep" else "Window")

    def _on_home_clicked(self):
        """Home button: toggle between full-view autoscale and saved zoom.

        First click: save current view, zoom out to show full data range.
        Second click: restore saved view.
        If user scrolls/zooms after going home, the saved view is forgotten.
        """
        if self._is_home and self._saved_view is not None:
            # Restore previous zoom
            x_min, x_max = self._saved_view
            self._saved_view = None
            self._is_home = False
            self._guard = True
            try:
                self._region.setRegion([x_min, x_max])
                self.view_range_requested.emit(float(x_min), float(x_max))
            finally:
                self._guard = False
        else:
            # Save current view, go to full range
            current = self._region.getRegion()
            data_span = self._data_max - self._data_min
            current_span = current[1] - current[0]
            # Only save if we're actually zoomed in (not already at full range)
            if data_span > 0 and current_span < data_span * 0.95:
                self._saved_view = (current[0], current[1])
            else:
                self._saved_view = None
            self._is_home = True
            self._guard = True
            try:
                self._region.setRegion([self._data_min, self._data_max])
                self.view_range_requested.emit(float(self._data_min), float(self._data_max))
            finally:
                self._guard = False

    def _invalidate_saved_view(self):
        """Called when user manually scrolls/zooms while in home view."""
        if self._is_home:
            self._saved_view = None
            self._is_home = False

    def set_data_range(self, t_min: float, t_max: float):
        super().set_data_range(t_min, t_max)
        # Add 3% padding so edge handles are visible at full zoom
        # (main plot uses 2% padding, so minimap needs slightly more)
        span = t_max - t_min
        pad = span * 0.03 if span > 0 else 0.5
        self._plot.setXRange(t_min - pad, t_max + pad, padding=0)
        self._update_shade_regions()

    def set_markers(self, markers):
        """Draw event markers on the minimap.

        Args:
            markers: List of dicts with keys:
                start_time, end_time (or None), color
        """
        import pyqtgraph as pg

        # Remove old marker items
        if self._marker_items:
            for item in self._marker_items:
                self._plot.removeItem(item)
            self._marker_items.clear()

        for m in markers:
            color = m.get('color', '#ffffff')
            qc = QColor(color)
            start = m['start_time']
            end = m.get('end_time')

            if end is not None and end > start:
                # Paired marker — shaded region behind traces
                region = pg.LinearRegionItem(
                    values=[start, end],
                    brush=pg.mkBrush(qc.red(), qc.green(), qc.blue(), 80),
                    pen=pg.mkPen(color, width=1),
                    movable=False,
                )
                region.setZValue(-5)
                self._plot.addItem(region)
                self._marker_items.append(region)
            else:
                # Single marker — vertical line behind traces
                line = pg.InfiniteLine(
                    pos=start, angle=90,
                    pen=pg.mkPen(color, width=1, style=Qt.PenStyle.DashLine),
                )
                line.setZValue(-5)
                self._plot.addItem(line)
                self._marker_items.append(line)

    def set_data(self, t_array, signal_array):
        """Render a single waveform (legacy — prefer set_multi_data)."""
        self.set_multi_data(t_array, [(signal_array, '#4ec9b0')])

    def set_multi_data(self, t_array, channel_data):
        """Render multiple channels in the minimap.

        Args:
            t_array: Time array (shared across channels)
            channel_data: List of (y_array, color_hex) tuples
        """
        import pyqtgraph as pg

        if t_array is None or not channel_data:
            return

        t_full = np.asarray(t_array)
        if len(t_full) == 0:
            return

        # Remove old traces if channel count changed
        if self._traces is not None and len(self._traces) != len(channel_data):
            for item in self._traces:
                self._plot.removeItem(item)
            self._traces = None

        n_ch = len(channel_data)

        for i, (y_raw, color) in enumerate(channel_data):
            if y_raw is None or len(y_raw) == 0:
                continue

            t, y = self._downsample_minimap(t_full, np.asarray(y_raw))

            # Normalize each channel into its own vertical band
            # Channel 0 occupies [n_ch-1, n_ch], channel 1 occupies [n_ch-2, n_ch-1], etc.
            y_min, y_max = float(np.min(y)), float(np.max(y))
            y_range = y_max - y_min
            if y_range > 0:
                y_norm = (y - y_min) / y_range  # 0..1
            else:
                y_norm = np.zeros_like(y)
            # Map to stacked band with small gap
            band_bottom = (n_ch - 1 - i)
            y_stacked = y_norm * 0.9 + band_bottom  # 0.9 leaves 10% gap

            # Parse color for fill brush
            qc = QColor(color)
            fill_brush = pg.mkBrush(qc.red(), qc.green(), qc.blue(), 25)

            if self._traces is None or i >= len(self._traces):
                if self._traces is None:
                    self._traces = []
                item = self._plot.plot(
                    t, y_stacked,
                    pen=pg.mkPen(color, width=1.2),
                    fillLevel=float(band_bottom),
                    fillBrush=fill_brush,
                )
                self._traces.append(item)
            else:
                self._traces[i].setData(t, y_stacked)

        self._plot.setYRange(-0.1, n_ch + 0.1, padding=0)

    @staticmethod
    def _downsample_minimap(t, y, target=1000):
        """Min-max envelope downsample for minimap."""
        n = len(t)
        if n <= target:
            return t, y
        n_buckets = target // 2
        boundaries = np.linspace(0, n, n_buckets + 1, dtype=np.intp)
        out_idx = np.empty(n_buckets * 2 + 2, dtype=np.intp)
        out_idx[0] = 0
        pos = 1
        for b in range(n_buckets):
            s, e = boundaries[b], boundaries[b + 1]
            if s >= e:
                continue
            seg = y[s:e]
            i_min, i_max = s + np.argmin(seg), s + np.argmax(seg)
            if i_min <= i_max:
                out_idx[pos], out_idx[pos + 1] = i_min, i_max
            else:
                out_idx[pos], out_idx[pos + 1] = i_max, i_min
            pos += 2
        out_idx[pos] = n - 1
        pos += 1
        indices = np.unique(out_idx[:pos])
        return t[indices], y[indices]

    def set_view_range(self, x_min: float, x_max: float):
        if self._guard:
            return
        self._guard = True
        try:
            self._region.blockSignals(True)
            self._region.setRegion([x_min, x_max])
            self._region.blockSignals(False)
            self._update_shade_regions()
        finally:
            self._guard = False

    def wheelEvent(self, event):
        """Scroll wheel zooms the viewport in/out on the minimap."""
        self._invalidate_saved_view()
        delta = event.angleDelta().y()
        if delta == 0:
            return

        x_min, x_max = self._region.getRegion()
        view_span = x_max - x_min
        data_span = self._data_max - self._data_min
        if data_span <= 0 or view_span <= 0:
            return

        # Zoom factor: scroll up = zoom in, scroll down = zoom out
        factor = 0.8 if delta > 0 else 1.25

        # Get mouse x position in data coordinates for zoom center
        plot_item = self._plot.getPlotItem()
        vb = plot_item.getViewBox()
        mouse_point = vb.mapSceneToView(event.position())
        center = mouse_point.x()

        # Clamp center to current view
        center = max(x_min, min(center, x_max))

        # Compute new span and bounds
        new_span = max(view_span * factor, data_span * 0.005)  # min ~0.5% of data
        new_span = min(new_span, data_span)  # max = full data

        # Keep center anchored
        frac = (center - x_min) / view_span if view_span > 0 else 0.5
        new_min = center - frac * new_span
        new_max = center + (1 - frac) * new_span

        # Clamp to data bounds
        if new_min < self._data_min:
            new_min = self._data_min
            new_max = new_min + new_span
        if new_max > self._data_max:
            new_max = self._data_max
            new_min = new_max - new_span

        self._guard = True
        try:
            self._region.setRegion([new_min, new_max])
            self.view_range_requested.emit(float(new_min), float(new_max))
        finally:
            self._guard = False

        event.accept()

    def mouseDoubleClickEvent(self, event):
        """Double-click centers the viewport on the clicked position."""
        plot_item = self._plot.getPlotItem()
        vb = plot_item.getViewBox()
        mouse_point = vb.mapSceneToView(event.position())
        center = mouse_point.x()

        x_min, x_max = self._region.getRegion()
        half_span = (x_max - x_min) / 2.0

        new_min = center - half_span
        new_max = center + half_span

        # Clamp to data bounds
        if new_min < self._data_min:
            new_min = self._data_min
            new_max = new_min + 2 * half_span
        if new_max > self._data_max:
            new_max = self._data_max
            new_min = new_max - 2 * half_span

        self._guard = True
        try:
            self._region.setRegion([new_min, new_max])
            self.view_range_requested.emit(float(new_min), float(new_max))
        finally:
            self._guard = False

        event.accept()

    def _on_region_changed(self):
        if self._guard:
            return
        # User manually dragged the region — forget saved home view
        self._invalidate_saved_view()
        x_min, x_max = self._region.getRegion()
        self._guard = True
        try:
            self.view_range_requested.emit(float(x_min), float(x_max))
        finally:
            self._guard = False


def load_nav_settings():
    """Load navigation preferences from QSettings."""
    s = QSettings('PhysioMetrics', 'Navigation')
    return {
        'bar_visible': s.value('bar_visible', True, type=bool),
        'bar_mode': s.value('bar_mode', 'minimap', type=str),
        'wheel_mode': s.value('wheel_mode', 'zoom', type=str),
        'expandable': s.value('expandable', False, type=bool),
    }


def save_nav_settings(bar_visible=None, bar_mode=None, wheel_mode=None, expandable=None):
    """Save navigation preferences to QSettings."""
    s = QSettings('PhysioMetrics', 'Navigation')
    if bar_visible is not None:
        s.setValue('bar_visible', bar_visible)
    if bar_mode is not None:
        s.setValue('bar_mode', bar_mode)
    if wheel_mode is not None:
        s.setValue('wheel_mode', wheel_mode)
    if expandable is not None:
        s.setValue('expandable', expandable)
