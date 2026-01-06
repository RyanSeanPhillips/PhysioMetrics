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

    def __init__(
        self,
        viewmodel: EventMarkerViewModel,
        click_time: float,
        parent: Optional[QWidget] = None,
        existing_actions: Optional[List[QAction]] = None,
        channel_names: Optional[List[str]] = None,
    ):
        """
        Initialize the context menu.

        Args:
            viewmodel: The event marker view model
            click_time: Time position where user right-clicked
            parent: Parent widget
            existing_actions: Existing actions to preserve (Auto Scale Y, etc.)
            channel_names: Available channel names for auto-detection
        """
        super().__init__(parent)
        self._viewmodel = viewmodel
        self._click_time = click_time
        self._existing_actions = existing_actions or []
        self._channel_names = channel_names or []

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
        add_single = self.addAction("Add Single Marker")
        add_single.setShortcut("S")  # Show shortcut hint
        add_single.triggered.connect(lambda: self.add_single_requested.emit(self._click_time))

        # Add Paired Marker
        add_paired = self.addAction("Add Paired Marker")
        add_paired.setShortcut("P")  # Show shortcut hint
        add_paired.triggered.connect(lambda: self.add_paired_requested.emit(self._click_time))

        self.addSeparator()

        # Auto-Detect submenu
        if self._channel_names:
            auto_menu = self.addMenu("Auto-Detect on Channel...")
            for channel in self._channel_names:
                action = auto_menu.addAction(channel)
                action.triggered.connect(lambda checked, ch=channel: self.auto_detect_requested.emit(ch))
            auto_menu.addSeparator()
            auto_menu.addAction("Configure Defaults...")

        # Separator before existing actions
        if self._existing_actions:
            self.addSeparator()
            # Add a visual divider
            separator_label = QAction(self)
            separator_label.setSeparator(True)
            self.addAction(separator_label)

            # Add existing actions (Auto Scale Y, Reset View, etc.)
            for action in self._existing_actions:
                self.addAction(action)

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
    convert_requested = pyqtSignal(str, str)   # marker_id, new_type ('single' or 'paired')
    delete_requested = pyqtSignal(str)         # marker_id
    delete_all_type_requested = pyqtSignal(str, str)  # category, label

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

        # Add Note
        note = self.addAction("Add Note...")
        note.triggered.connect(lambda: self.note_requested.emit(self._marker_id))

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

        # Delete
        delete = self.addAction("Delete Marker")
        delete.triggered.connect(lambda: self.delete_requested.emit(self._marker_id))

        # Delete all of type
        cat_display = current_cat.get_display_label(marker.label)
        delete_all = self.addAction(f"Delete All [{cat_display}]")
        delete_all.triggered.connect(
            lambda: self.delete_all_type_requested.emit(marker.category, marker.label)
        )


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
