"""
Plot context menu extensions for display, Y2 metric, and view controls.

These submenus replace the bottom toolbar checkboxes (shading, dark mode,
Y-axis autoscale, padding) and the Y2 dropdown, moving them into the
right-click context menu for a cleaner UI.
"""

from typing import Optional, List, Tuple, Callable
from PyQt6.QtWidgets import QMenu, QWidgetAction, QWidget, QHBoxLayout, QLabel, QDoubleSpinBox
from PyQt6.QtGui import QAction, QIcon, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, pyqtSignal, QObject


def _dot_icon(hex_color: str, size: int = 10) -> QIcon:
    """Small colored circle icon for menu items."""
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(hex_color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.end()
    return QIcon(pix)


class PlotDisplayMenuBuilder:
    """Builds Display, Y2 Metric, and View submenus for the right-click context menu.

    Usage:
        builder = PlotDisplayMenuBuilder(state_getter, callbacks)
        builder.add_to_menu(parent_menu)
    """

    def __init__(
        self,
        # State accessors (callables that return current values)
        get_eupnea_shade: Callable[[], bool],
        get_sniffing_shade: Callable[[], bool],
        get_apnea_shade: Callable[[], bool],
        get_outliers_shade: Callable[[], bool],
        get_dark_mode: Callable[[], bool],
        get_percentile_autoscale: Callable[[], bool],
        get_autoscale_padding: Callable[[], float],
        get_y2_metric_key: Callable[[], Optional[str]],
        # Metric specs: list of (label, key) tuples
        metric_specs: List[Tuple[str, str]],
        # Callbacks for when user changes a setting
        on_eupnea_toggled: Callable[[bool], None],
        on_sniffing_toggled: Callable[[bool], None],
        on_apnea_toggled: Callable[[bool], None],
        on_outliers_toggled: Callable[[bool], None],
        on_dark_mode_toggled: Callable[[bool], None],
        on_autoscale_toggled: Callable[[bool], None],
        on_padding_changed: Callable[[float], None],
        on_y2_metric_changed: Callable[[Optional[str]], None],
    ):
        self._get_eupnea = get_eupnea_shade
        self._get_sniffing = get_sniffing_shade
        self._get_apnea = get_apnea_shade
        self._get_outliers = get_outliers_shade
        self._get_dark = get_dark_mode
        self._get_autoscale = get_percentile_autoscale
        self._get_padding = get_autoscale_padding
        self._get_y2_key = get_y2_metric_key
        self._metric_specs = metric_specs
        self._on_eupnea = on_eupnea_toggled
        self._on_sniffing = on_sniffing_toggled
        self._on_apnea = on_apnea_toggled
        self._on_outliers = on_outliers_toggled
        self._on_dark = on_dark_mode_toggled
        self._on_autoscale = on_autoscale_toggled
        self._on_padding = on_padding_changed
        self._on_y2 = on_y2_metric_changed

    def add_to_menu(self, menu: QMenu) -> None:
        """Append Display, Y2 Metric, and View submenus to the given menu."""
        menu.addSeparator()
        self._build_display_submenu(menu)
        self._build_y2_submenu(menu)
        self._build_view_submenu(menu)

    def _build_display_submenu(self, parent: QMenu) -> None:
        """Display submenu: shading toggles + dark mode."""
        display = parent.addMenu("Display")

        # Shading toggles
        items = [
            ("Eupnea Shading", "#4CAF50", self._get_eupnea, self._on_eupnea),
            ("Sniffing Shading", "#9C27B0", self._get_sniffing, self._on_sniffing),
            ("Apnea Shading", "#F44336", self._get_apnea, self._on_apnea),
            ("Outlier Shading", "#FF9800", self._get_outliers, self._on_outliers),
        ]
        for label, color, getter, callback in items:
            action = display.addAction(_dot_icon(color), label)
            action.setCheckable(True)
            action.setChecked(getter())
            action.triggered.connect(callback)

        display.addSeparator()

        # Dark mode
        dark_action = display.addAction("Dark Mode")
        dark_action.setCheckable(True)
        dark_action.setChecked(self._get_dark())
        dark_action.triggered.connect(self._on_dark)

    def _build_y2_submenu(self, parent: QMenu) -> None:
        """Y2 Metric submenu: select secondary Y-axis metric."""
        y2 = parent.addMenu("Y2 Metric")
        current_key = self._get_y2_key()

        # "None" option
        action_none = y2.addAction("None")
        action_none.setCheckable(True)
        action_none.setChecked(current_key is None)
        action_none.triggered.connect(lambda: self._on_y2(None))

        y2.addSeparator()

        # Group metrics into categories for easier navigation
        for label, key in self._metric_specs:
            action = y2.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == current_key)
            action.triggered.connect(lambda checked, k=key: self._on_y2(k))

    def _build_view_submenu(self, parent: QMenu) -> None:
        """View submenu: autoscale and padding controls."""
        view = parent.addMenu("View Settings")

        # Percentile autoscale toggle
        autoscale_action = view.addAction("Y1 Percentile Scaling")
        autoscale_action.setCheckable(True)
        autoscale_action.setChecked(self._get_autoscale())
        autoscale_action.triggered.connect(self._on_autoscale)

        # Padding as a submenu with preset values
        padding_menu = view.addMenu("Y-Axis Padding")
        current_padding = self._get_padding()
        for pct, val in [("0%", 0.0), ("5%", 0.05), ("10%", 0.10),
                         ("15%", 0.15), ("20%", 0.20), ("30%", 0.30)]:
            action = padding_menu.addAction(pct)
            action.setCheckable(True)
            action.setChecked(abs(current_padding - val) < 0.001)
            action.triggered.connect(lambda checked, v=val: self._on_padding(v))
