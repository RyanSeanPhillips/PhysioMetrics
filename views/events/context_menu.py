"""
Event marker context menu.

This module provides the right-click context menu for adding and
managing event markers in the plot.
"""

from typing import Optional, Callable, Tuple, List
from PyQt6.QtWidgets import (
    QMenu, QWidgetAction, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QFrame,
)
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap, QPainter
from PyQt6.QtCore import Qt, pyqtSignal, QObject

from viewmodels.event_marker_viewmodel import EventMarkerViewModel
from core.domain.events import EventCategory, MarkerType


class TypeSelectorWidget(QWidget):
    """
    Widget for the sticky type selector dropdown in the context menu.
    """

    type_changed = pyqtSignal(str, str)  # category, label

    def __init__(self, viewmodel: EventMarkerViewModel, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._setup_ui()
        self._populate_types()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)

        label = QLabel("Marker Type:")
        layout.addWidget(label)

        self._combo = QComboBox()
        self._combo.setMinimumWidth(150)
        self._combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._combo, 1)

    def _populate_types(self) -> None:
        """Populate the combo box with available types."""
        self._combo.blockSignals(True)
        self._combo.clear()

        categories = self._viewmodel.get_categories()
        current_cat, current_label = self._viewmodel.selected_type
        current_index = 0
        index = 0

        for cat in sorted(categories, key=lambda c: c.name):
            # Add category header (disabled item)
            self._combo.addItem(f"-- {cat.display_name} --", None)
            # Make header item non-selectable
            model = self._combo.model()
            item = model.item(self._combo.count() - 1)
            item.setEnabled(False)

            # Add labels
            for label in cat.labels:
                display = cat.get_display_label(label)
                self._combo.addItem(f"    {display}", (cat.name, label))
                index = self._combo.count() - 1
                if cat.name == current_cat and label == current_label:
                    current_index = index

            # Add "Create New..." option for custom category
            if cat.name == 'custom':
                self._combo.addItem("    + Create New Type...", ('__create__', '__create__'))

        self._combo.setCurrentIndex(current_index)
        self._combo.blockSignals(False)

    def _on_type_changed(self, index: int) -> None:
        """Handle type selection change."""
        data = self._combo.currentData()
        if data is None:
            return

        category, label = data

        if category == '__create__':
            # TODO: Show create type dialog
            # For now, revert to previous selection
            self._populate_types()
            return

        self._viewmodel.set_selected_type(category, label)
        self.type_changed.emit(category, label)

    def refresh(self) -> None:
        """Refresh the combo box to reflect current selection."""
        self._populate_types()


class EventMarkerContextMenu(QMenu):
    """
    Context menu for event markers.

    Provides options for:
    - Adding single/paired markers (with keyboard shortcut hints)
    - Auto-detection on channels
    - Marker settings

    Also preserves existing plot options (Auto Scale Y, Reset View).
    """

    # Signals
    add_single_requested = pyqtSignal(float)  # time
    add_paired_requested = pyqtSignal(float)  # start_time
    auto_detect_requested = pyqtSignal(str)   # channel_name
    settings_requested = pyqtSignal()
    delete_all_requested = pyqtSignal()  # delete all markers everywhere
    delete_all_type_requested = pyqtSignal(str, str)  # category, label
    delete_all_sweep_requested = pyqtSignal()  # delete all in current sweep
    delete_category_sweep_requested = pyqtSignal(str)  # category - delete category in current sweep
    delete_category_all_requested = pyqtSignal(str)    # category - delete category in all sweeps
    derivative_toggle_changed = pyqtSignal(bool)  # show_derivative_on_drag toggle
    generate_cta_requested = pyqtSignal()  # generate photometry CTA dialog

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        click_time: float,
        parent: Optional[QWidget] = None,
        existing_actions: Optional[List[QAction]] = None,
        channel_names: Optional[List[str]] = None,
        show_derivative_on_drag: bool = False,
        clicked_channel: Optional[str] = None,
    ):
        """
        Initialize the context menu.

        Args:
            viewmodel: The event marker view model
            click_time: Time position where user right-clicked
            parent: Parent widget
            existing_actions: Existing actions to preserve (Auto Scale Y, etc.)
            channel_names: Available channel names for auto-detection
            show_derivative_on_drag: Current state of derivative overlay toggle
            clicked_channel: Channel name of the subplot that was right-clicked
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._click_time = click_time
        self._existing_actions = existing_actions or []
        self._channel_names = channel_names or []
        self._show_derivative_on_drag = show_derivative_on_drag
        self._clicked_channel = clicked_channel

        self._build_menu()

    def _build_menu(self) -> None:
        """Build the menu structure."""
        # Type selector widget
        type_widget = TypeSelectorWidget(self._viewmodel)
        type_action = QWidgetAction(self)
        type_action.setDefaultWidget(type_widget)
        self.addAction(type_action)

        self.addSeparator()

        # Add Single Marker
        add_single = self.addAction("Add Single Marker (S+Click)")
        add_single.triggered.connect(lambda: self.add_single_requested.emit(self._click_time))

        # Add Paired Marker (D for Double)
        add_paired = self.addAction("Add Paired Marker (D+Click)")
        add_paired.triggered.connect(lambda: self.add_paired_requested.emit(self._click_time))

        self.addSeparator()

        # Auto-Detect Events (uses the clicked channel directly)
        if self._channel_names:
            target_channel = self._clicked_channel or self._channel_names[0]
            auto_detect = self.addAction(f"Auto-Detect on {target_channel}")
            auto_detect.setToolTip(f"Open detection dialog for channel: {target_channel}")
            auto_detect.triggered.connect(lambda: self.auto_detect_requested.emit(target_channel))

        # Separator before existing actions
        if self._existing_actions:
            self.addSeparator()

            # Add existing actions (Auto Scale Y, Reset View, etc.)
            for action in self._existing_actions:
                self.addAction(action)

        self.addSeparator()

        # Delete submenu (only show if there are markers)
        if self._viewmodel.marker_count > 0:
            delete_menu = self.addMenu("Delete Markers...")

            total_count = self._viewmodel.marker_count

            # Delete all markers everywhere
            delete_all = delete_menu.addAction(f"Delete All ({total_count})")
            delete_all.triggered.connect(self.delete_all_requested.emit)

            # Delete all in this sweep
            delete_sweep = delete_menu.addAction("Delete All (This Sweep)")
            delete_sweep.triggered.connect(self.delete_all_sweep_requested.emit)

            delete_menu.addSeparator()

            # Delete by Category submenu
            delete_by_cat_menu = delete_menu.addMenu("Delete by Category...")

            # This Sweep submenu
            this_sweep_menu = delete_by_cat_menu.addMenu("This Sweep")

            # All Sweeps submenu
            all_sweeps_menu = delete_by_cat_menu.addMenu("All Sweeps")

            # Get all unique categories that have markers
            existing_categories = set()
            for m in self._viewmodel.store.all():
                existing_categories.add(m.category)

            if existing_categories:
                for cat_name in sorted(existing_categories):
                    # Get display name for category
                    cat_obj = self._viewmodel.service.registry.get(cat_name)
                    cat_display = cat_obj.display_name if cat_obj else cat_name.title()

                    # Count markers in this category
                    cat_markers = self._viewmodel.store.get_by_category(cat_name)
                    count = len(cat_markers)

                    # Add to "This Sweep" menu
                    action_sweep = this_sweep_menu.addAction(f"{cat_display} ({count})")
                    action_sweep.triggered.connect(
                        lambda checked, c=cat_name: self.delete_category_sweep_requested.emit(c)
                    )

                    # Add to "All Sweeps" menu
                    action_all = all_sweeps_menu.addAction(f"{cat_display} ({count})")
                    action_all.triggered.connect(
                        lambda checked, c=cat_name: self.delete_category_all_requested.emit(c)
                    )
            else:
                # No markers - disable menus
                this_sweep_menu.setEnabled(False)
                all_sweeps_menu.setEnabled(False)

        self.addSeparator()

        # Generate CTA (only show if there are markers and photometry data is available)
        if self._viewmodel.marker_count > 0:
            generate_cta = self.addAction("Generate Photometry CTA...")
            generate_cta.setToolTip(
                "Generate Condition-Triggered Averages aligned to event markers.\n"
                "Useful for analyzing photometry signals around behavioral events."
            )
            generate_cta.triggered.connect(self.generate_cta_requested.emit)

        self.addSeparator()

        # View Options
        view_menu = self.addMenu("View Options")

        # Derivative overlay toggle
        derivative_toggle = view_menu.addAction("Show Derivative on Drag")
        derivative_toggle.setCheckable(True)
        derivative_toggle.setChecked(self._show_derivative_on_drag)
        derivative_toggle.setToolTip(
            "Show smoothed derivative (dV/dt) overlay when dragging markers.\n"
            "Tip: You can also hold Shift while dragging to show it temporarily."
        )
        derivative_toggle.triggered.connect(
            lambda checked: self.derivative_toggle_changed.emit(checked)
        )

        self.addSeparator()

        # Settings
        settings = self.addAction("Marker Settings...")
        settings.triggered.connect(self.settings_requested.emit)


class MarkerContextMenu(QMenu):
    """
    Context menu when right-clicking on an existing marker.

    Provides options for:
    - Edit marker
    - Change category/label
    - Set color
    - Add note
    - Convert type
    - Delete
    """

    edit_requested = pyqtSignal(str)          # marker_id
    category_changed = pyqtSignal(str, str)    # marker_id, new_category
    label_changed = pyqtSignal(str, str)       # marker_id, new_label
    color_requested = pyqtSignal(str)          # marker_id
    note_requested = pyqtSignal(str)           # marker_id
    line_width_changed = pyqtSignal(str, int)  # marker_id, width
    grab_width_changed = pyqtSignal(str, int)  # marker_id, grab_width_px
    convert_requested = pyqtSignal(str, str)   # marker_id, new_type ('single' or 'paired')
    delete_requested = pyqtSignal(str)         # marker_id
    delete_all_requested = pyqtSignal()  # delete all markers everywhere
    delete_all_type_requested = pyqtSignal(str, str)  # category, label
    delete_all_sweep_requested = pyqtSignal()  # delete all in current sweep
    delete_category_sweep_requested = pyqtSignal(str)  # category - delete category in current sweep
    delete_category_all_requested = pyqtSignal(str)    # category - delete category in all sweeps
    edit_in_detector_requested = pyqtSignal(str)  # marker_id — open detection dialog with this marker's category

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        marker_id: str,
        parent: Optional[QWidget] = None,
    ):
        """
        Initialize the marker context menu.

        Args:
            viewmodel: The event marker view model
            marker_id: ID of the marker that was clicked
            parent: Parent widget
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._marker_id = marker_id

        marker = viewmodel.store.get(marker_id)
        if marker:
            self._build_menu(marker)

    def _build_menu(self, marker) -> None:
        """Build the menu structure for an existing marker."""
        # Edit Marker
        edit = self.addAction("Edit Marker...")
        edit.triggered.connect(lambda: self.edit_requested.emit(self._marker_id))

        # Change Category submenu
        cat_menu = self.addMenu("Change Category")
        for cat in self._viewmodel.get_categories():
            action = cat_menu.addAction(cat.display_name)
            action.setCheckable(True)
            action.setChecked(cat.name == marker.category)
            action.triggered.connect(
                lambda checked, c=cat.name: self.category_changed.emit(self._marker_id, c)
            )

        # Change Label submenu
        current_cat = self._viewmodel.service.registry.get_or_default(marker.category)
        if current_cat.labels:
            label_menu = self.addMenu("Change Label")
            for label in current_cat.labels:
                action = label_menu.addAction(current_cat.get_display_label(label))
                action.setCheckable(True)
                action.setChecked(label == marker.label)
                action.triggered.connect(
                    lambda checked, l=label: self.label_changed.emit(self._marker_id, l)
                )

        # Set Color
        color = self.addAction("Set Color...")
        color.triggered.connect(lambda: self.color_requested.emit(self._marker_id))

        # Line Thickness submenu
        thickness_menu = self.addMenu("Line Thickness")
        current_width = marker.line_width  # None means use default (0 = cosmetic/thinnest)
        for label_text, width in [("Thinnest (cosmetic)", 0), ("Thin (1px)", 1), ("Medium (2px)", 2), ("Thick (3px)", 3)]:
            action = thickness_menu.addAction(label_text)
            action.setCheckable(True)
            # Width 0 is the default when marker.line_width is None
            action.setChecked(
                (current_width is not None and current_width == width) or
                (current_width is None and width == 0)
            )
            action.triggered.connect(
                lambda checked, w=width: self.line_width_changed.emit(self._marker_id, w)
            )

        # Edge Grab Width submenu (only for paired markers)
        if marker.is_paired:
            grab_menu = self.addMenu("Edge Grab Width")
            for label_text, grab_w in [("Narrow (10px)", 10), ("Medium (20px)", 20), ("Wide (35px)", 35)]:
                action = grab_menu.addAction(label_text)
                action.triggered.connect(
                    lambda checked, gw=grab_w: self.grab_width_changed.emit(self._marker_id, gw)
                )

        # Add Note
        note = self.addAction("Add Note...")
        note.triggered.connect(lambda: self.note_requested.emit(self._marker_id))

        self.addSeparator()

        # Edit Category in Detector
        edit_detector = self.addAction("Edit Category in Detector...")
        edit_detector.setToolTip("Open detection dialog with all markers of this category loaded for editing")
        edit_detector.triggered.connect(lambda: self.edit_in_detector_requested.emit(self._marker_id))

        self.addSeparator()

        # Convert type
        if marker.is_paired:
            convert = self.addAction("Convert to Single")
            convert.triggered.connect(
                lambda: self.convert_requested.emit(self._marker_id, 'single')
            )
        else:
            convert = self.addAction("Convert to Paired")
            convert.triggered.connect(
                lambda: self.convert_requested.emit(self._marker_id, 'paired')
            )

        self.addSeparator()

        # Delete this marker
        delete = self.addAction("Delete Marker")
        delete.triggered.connect(lambda: self.delete_requested.emit(self._marker_id))

        # Delete submenu for bulk operations
        delete_menu = self.addMenu("Delete Multiple...")

        total_count = len(self._viewmodel.store)

        # Delete all markers everywhere
        delete_all = delete_menu.addAction(f"Delete All ({total_count})")
        delete_all.triggered.connect(self.delete_all_requested.emit)

        # Delete all in this sweep
        delete_all_sweep = delete_menu.addAction("Delete All (This Sweep)")
        delete_all_sweep.triggered.connect(self.delete_all_sweep_requested.emit)

        delete_menu.addSeparator()

        # Delete all of this exact type (category + label)
        # Use marker's actual category and label for display
        marker_cat_display = current_cat.display_name if current_cat else marker.category.title()
        marker_label_display = current_cat.get_display_label(marker.label) if current_cat else marker.label.title()
        type_count = len(self._viewmodel.store.get_by_label(marker.category, marker.label))
        delete_all_type = delete_menu.addAction(f"Delete All [{marker_cat_display}: {marker_label_display}] ({type_count})")
        delete_all_type.triggered.connect(
            lambda: self.delete_all_type_requested.emit(marker.category, marker.label)
        )

        delete_menu.addSeparator()

        # Delete by Category submenu
        delete_by_cat_menu = delete_menu.addMenu("Delete by Category...")

        # This Sweep submenu
        this_sweep_menu = delete_by_cat_menu.addMenu("This Sweep")

        # All Sweeps submenu
        all_sweeps_menu = delete_by_cat_menu.addMenu("All Sweeps")

        # Get all unique categories that have markers
        existing_categories = set()
        for m in self._viewmodel.store.all():
            existing_categories.add(m.category)

        if existing_categories:
            for cat_name in sorted(existing_categories):
                # Get display name for category (don't use get_or_default to avoid "Custom" fallback)
                cat_obj = self._viewmodel.service.registry.get(cat_name)
                cat_display = cat_obj.display_name if cat_obj else cat_name.title()

                # Count markers in this category
                cat_markers = self._viewmodel.store.get_by_category(cat_name)
                count = len(cat_markers)

                # Add to "This Sweep" menu
                action_sweep = this_sweep_menu.addAction(f"{cat_display} ({count})")
                action_sweep.triggered.connect(
                    lambda checked, c=cat_name: self.delete_category_sweep_requested.emit(c)
                )

                # Add to "All Sweeps" menu
                action_all = all_sweeps_menu.addAction(f"{cat_display} ({count})")
                action_all.triggered.connect(
                    lambda checked, c=cat_name: self.delete_category_all_requested.emit(c)
                )
        else:
            # No markers - disable menus
            this_sweep_menu.setEnabled(False)
            all_sweeps_menu.setEnabled(False)


def create_color_icon(color: str, size: int = 16) -> QIcon:
    """Create a small square icon of the given color."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(2, 2, size - 4, size - 4, 2, 2)
    painter.end()

    return QIcon(pixmap)
