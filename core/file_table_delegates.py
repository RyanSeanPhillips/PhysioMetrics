"""
File Table Delegates - Custom item delegates for Project Builder table.

Provides custom rendering and interaction for special column types:
- ButtonDelegate: Action buttons (analyze, delete, etc.)
- StatusDelegate: Status indicators with icons
- ExportsDelegate: Export status display
- AutoCompleteDelegate: Text editing with autocomplete

Usage:
    from core.file_table_delegates import ButtonDelegate, AutoCompleteDelegate

    button_delegate = ButtonDelegate(parent)
    table_view.setItemDelegateForColumn(actions_col, button_delegate)
"""

from typing import List, Callable, Optional
from PyQt6.QtWidgets import (
    QStyledItemDelegate, QWidget, QStyleOptionViewItem,
    QStyle, QApplication, QPushButton, QHBoxLayout,
    QLineEdit, QCompleter, QStyleOptionButton
)
from PyQt6.QtCore import (
    Qt, QModelIndex, QRect, QSize, QEvent, QPoint,
    pyqtSignal, QAbstractItemModel, QStringListModel
)
from PyQt6.QtGui import (
    QPainter, QIcon, QColor, QPen, QBrush, QMouseEvent,
    QPalette
)


class ButtonDelegate(QStyledItemDelegate):
    """
    Delegate for rendering action buttons in cells.

    Renders a row of icon buttons and emits signals when clicked.
    """

    # Signals emitted when buttons are clicked
    analyze_clicked = pyqtSignal(int)  # row
    delete_clicked = pyqtSignal(int)   # row
    info_clicked = pyqtSignal(int)     # row

    def __init__(self, parent=None):
        super().__init__(parent)
        self._button_width = 24
        self._button_spacing = 2
        self._buttons = [
            ('analyze', '▶', 'Analyze this file'),
            ('delete', '✕', 'Remove from list'),
            ('info', 'ℹ', 'Show file info'),
        ]
        self._hover_row = -1
        self._hover_button = -1
        self._pressed_button = -1

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the button cells."""
        painter.save()

        # Draw background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())

        row = index.row()

        # Calculate button positions
        x = option.rect.left() + 4
        y = option.rect.top() + (option.rect.height() - self._button_width) // 2

        for i, (action, icon_text, tooltip) in enumerate(self._buttons):
            btn_rect = QRect(x, y, self._button_width, self._button_width)

            # Determine button state
            is_hover = (row == self._hover_row and i == self._hover_button)
            is_pressed = (row == self._hover_row and i == self._pressed_button)

            # Draw button background
            if is_pressed:
                painter.fillRect(btn_rect, QColor(80, 80, 80))
            elif is_hover:
                painter.fillRect(btn_rect, QColor(60, 60, 60))
            else:
                painter.fillRect(btn_rect, QColor(50, 50, 50))

            # Draw border
            painter.setPen(QPen(QColor(100, 100, 100), 1))
            painter.drawRect(btn_rect)

            # Draw icon/text
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, icon_text)

            x += self._button_width + self._button_spacing

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Return the preferred size for the button cell."""
        width = len(self._buttons) * (self._button_width + self._button_spacing) + 8
        return QSize(width, 30)

    def editorEvent(self, event: QEvent, model: QAbstractItemModel,
                    option: QStyleOptionViewItem, index: QModelIndex) -> bool:
        """Handle mouse events for button clicks."""
        if not index.isValid():
            return False

        row = index.row()

        if event.type() == QEvent.Type.MouseMove:
            # Track hover state
            mouse_event = event
            pos = mouse_event.position().toPoint()
            button_idx = self._get_button_at_pos(option.rect, pos)

            if self._hover_row != row or self._hover_button != button_idx:
                self._hover_row = row
                self._hover_button = button_idx
                # Request repaint
                return True

        elif event.type() == QEvent.Type.MouseButtonPress:
            mouse_event = event
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                pos = mouse_event.position().toPoint()
                button_idx = self._get_button_at_pos(option.rect, pos)
                if button_idx >= 0:
                    self._pressed_button = button_idx
                    return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            mouse_event = event
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                pos = mouse_event.position().toPoint()
                button_idx = self._get_button_at_pos(option.rect, pos)

                if button_idx >= 0 and button_idx == self._pressed_button:
                    # Button was clicked
                    action = self._buttons[button_idx][0]
                    if action == 'analyze':
                        self.analyze_clicked.emit(row)
                    elif action == 'delete':
                        self.delete_clicked.emit(row)
                    elif action == 'info':
                        self.info_clicked.emit(row)

                self._pressed_button = -1
                return True

        elif event.type() == QEvent.Type.Leave:
            self._hover_row = -1
            self._hover_button = -1
            self._pressed_button = -1

        return False

    def _get_button_at_pos(self, cell_rect: QRect, pos: QPoint) -> int:
        """Get the button index at a position, or -1 if none."""
        x = cell_rect.left() + 4
        y = cell_rect.top() + (cell_rect.height() - self._button_width) // 2

        # Adjust pos to be relative to cell
        rel_x = pos.x()
        rel_y = pos.y()

        for i in range(len(self._buttons)):
            btn_rect = QRect(x, y, self._button_width, self._button_width)
            if btn_rect.contains(pos):
                return i
            x += self._button_width + self._button_spacing

        return -1


class StatusDelegate(QStyledItemDelegate):
    """
    Delegate for rendering status indicators.

    Shows colored icons/text based on status value.
    """

    STATUS_STYLES = {
        'pending': ('⏳', QColor(150, 150, 150)),
        'in_progress': ('🔄', QColor(200, 200, 100)),
        'completed': ('✓', QColor(100, 200, 100)),
        'error': ('❌', QColor(200, 100, 100)),
        'skipped': ('⏭', QColor(150, 150, 150)),
        'conflict': ('⚠', QColor(200, 150, 50)),
    }

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the status indicator."""
        painter.save()

        # Draw background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Get status value
        value = index.data(Qt.ItemDataRole.DisplayRole)
        status = str(value).lower() if value else 'pending'

        # Get style for status
        icon, color = self.STATUS_STYLES.get(status, ('?', QColor(150, 150, 150)))

        # Draw status
        painter.setPen(QPen(color))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, icon)

        painter.restore()


class ExportsDelegate(QStyledItemDelegate):
    """
    Delegate for rendering export status.

    Shows which export types have been completed.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the exports display."""
        painter.save()

        # Draw background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())

        # Get exports value
        value = index.data(Qt.ItemDataRole.DisplayRole)

        if value:
            # Draw export indicators
            painter.setPen(QPen(QColor(150, 200, 150)))
            painter.drawText(
                option.rect.adjusted(4, 0, -4, 0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                str(value)
            )

        painter.restore()


class AutoCompleteDelegate(QStyledItemDelegate):
    """
    Delegate for text editing with autocomplete.

    Uses a QCompleter with history from QSettings.
    """

    def __init__(self, parent=None, history_key: str = ""):
        super().__init__(parent)
        self._history_key = history_key
        self._history: List[str] = []
        self._load_history()

    def _load_history(self):
        """Load autocomplete history from QSettings."""
        if self._history_key:
            from PyQt6.QtCore import QSettings
            settings = QSettings("PhysioMetrics", "BreathAnalysis")
            history = settings.value(f"autocomplete/{self._history_key}", [])
            self._history = history if isinstance(history, list) else []

    def _save_history(self):
        """Save autocomplete history to QSettings."""
        if self._history_key:
            from PyQt6.QtCore import QSettings
            settings = QSettings("PhysioMetrics", "BreathAnalysis")
            settings.setValue(f"autocomplete/{self._history_key}", self._history[:50])

    def add_to_history(self, value: str):
        """Add a value to the autocomplete history."""
        if value and value not in self._history:
            self._history.insert(0, value)
            self._history = self._history[:50]  # Keep max 50 items
            self._save_history()

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem,
                     index: QModelIndex) -> QWidget:
        """Create the editor widget with autocomplete."""
        editor = QLineEdit(parent)

        if self._history:
            completer = QCompleter(self._history, editor)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            editor.setCompleter(completer)

        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        """Set the editor data from the model."""
        value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(editor, QLineEdit):
            editor.setText(str(value) if value else "")

    def setModelData(self, editor: QWidget, model: QAbstractItemModel,
                     index: QModelIndex):
        """Set the model data from the editor."""
        if isinstance(editor, QLineEdit):
            value = editor.text()
            model.setData(index, value, Qt.ItemDataRole.EditRole)
            # Add to history
            if value:
                self.add_to_history(value)

    def updateEditorGeometry(self, editor: QWidget, option: QStyleOptionViewItem,
                             index: QModelIndex):
        """Set the editor geometry."""
        editor.setGeometry(option.rect)


class NotesActionsDelegate(QStyledItemDelegate):
    """
    Delegate for rendering action buttons in notes table.

    Renders compact icon buttons for folder location, open file, and preview.
    """

    # Signals emitted when buttons are clicked
    folder_clicked = pyqtSignal(int)   # row - open containing folder
    open_clicked = pyqtSignal(int)     # row - open file externally
    preview_clicked = pyqtSignal(int)  # row - preview content

    def __init__(self, parent=None):
        super().__init__(parent)
        self._button_width = 22
        self._button_spacing = 2
        self._buttons = [
            ('folder', '📁', 'Open containing folder'),
            ('open', '↗', 'Open in external app'),
            ('preview', '👁', 'Preview content'),
        ]
        self._hover_row = -1
        self._hover_button = -1
        self._pressed_button = -1

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the button cells."""
        painter.save()

        # Draw background
        if option.state & QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
        else:
            painter.fillRect(option.rect, option.palette.base())

        row = index.row()

        # Calculate button positions - center vertically
        total_width = len(self._buttons) * (self._button_width + self._button_spacing) - self._button_spacing
        x = option.rect.left() + (option.rect.width() - total_width) // 2
        y = option.rect.top() + (option.rect.height() - self._button_width) // 2

        for i, (action, icon_text, tooltip) in enumerate(self._buttons):
            btn_rect = QRect(x, y, self._button_width, self._button_width)

            # Determine button state
            is_hover = (row == self._hover_row and i == self._hover_button)
            is_pressed = (row == self._hover_row and i == self._pressed_button)

            # Draw button background
            if is_pressed:
                painter.fillRect(btn_rect, QColor(80, 80, 80))
            elif is_hover:
                painter.fillRect(btn_rect, QColor(60, 60, 60))
            else:
                painter.fillRect(btn_rect, QColor(45, 45, 45))

            # Draw border
            painter.setPen(QPen(QColor(80, 80, 80), 1))
            painter.drawRoundedRect(btn_rect, 3, 3)

            # Draw icon/text
            painter.setPen(QPen(QColor(200, 200, 200)))
            painter.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, icon_text)

            x += self._button_width + self._button_spacing

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        """Return the preferred size for the button cell."""
        width = len(self._buttons) * (self._button_width + self._button_spacing) + 8
        return QSize(width, 26)

    def editorEvent(self, event: QEvent, model: QAbstractItemModel,
                    option: QStyleOptionViewItem, index: QModelIndex) -> bool:
        """Handle mouse events for button clicks."""
        if not index.isValid():
            return False

        row = index.row()

        if event.type() == QEvent.Type.MouseMove:
            # Track hover state
            mouse_event = event
            pos = mouse_event.position().toPoint()
            button_idx = self._get_button_at_pos(option.rect, pos)

            if self._hover_row != row or self._hover_button != button_idx:
                self._hover_row = row
                self._hover_button = button_idx
                return True

        elif event.type() == QEvent.Type.MouseButtonPress:
            mouse_event = event
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                pos = mouse_event.position().toPoint()
                button_idx = self._get_button_at_pos(option.rect, pos)
                if button_idx >= 0:
                    self._pressed_button = button_idx
                    return True

        elif event.type() == QEvent.Type.MouseButtonRelease:
            mouse_event = event
            if mouse_event.button() == Qt.MouseButton.LeftButton:
                pos = mouse_event.position().toPoint()
                button_idx = self._get_button_at_pos(option.rect, pos)

                if button_idx >= 0 and button_idx == self._pressed_button:
                    # Button was clicked
                    action = self._buttons[button_idx][0]
                    if action == 'folder':
                        self.folder_clicked.emit(row)
                    elif action == 'open':
                        self.open_clicked.emit(row)
                    elif action == 'preview':
                        self.preview_clicked.emit(row)

                self._pressed_button = -1
                return True

        elif event.type() == QEvent.Type.Leave:
            self._hover_row = -1
            self._hover_button = -1
            self._pressed_button = -1

        return False

    def _get_button_at_pos(self, cell_rect: QRect, pos: QPoint) -> int:
        """Get the button index at a position, or -1 if none."""
        total_width = len(self._buttons) * (self._button_width + self._button_spacing) - self._button_spacing
        x = cell_rect.left() + (cell_rect.width() - total_width) // 2
        y = cell_rect.top() + (cell_rect.height() - self._button_width) // 2

        for i in range(len(self._buttons)):
            btn_rect = QRect(x, y, self._button_width, self._button_width)
            if btn_rect.contains(pos):
                return i
            x += self._button_width + self._button_spacing

        return -1


class ComboBoxDelegate(QStyledItemDelegate):
    """
    Delegate for dropdown selection.

    Provides a combobox editor with predefined options.
    """

    def __init__(self, parent=None, options: List[str] = None):
        super().__init__(parent)
        self._options = options or []

    def set_options(self, options: List[str]):
        """Set the dropdown options."""
        self._options = options

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem,
                     index: QModelIndex) -> QWidget:
        """Create a combobox editor."""
        from PyQt6.QtWidgets import QComboBox

        editor = QComboBox(parent)
        editor.addItems(self._options)
        editor.setEditable(True)  # Allow custom values
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        """Set the editor data from the model."""
        from PyQt6.QtWidgets import QComboBox

        value = index.data(Qt.ItemDataRole.EditRole)
        if isinstance(editor, QComboBox):
            idx = editor.findText(str(value) if value else "")
            if idx >= 0:
                editor.setCurrentIndex(idx)
            else:
                editor.setCurrentText(str(value) if value else "")

    def setModelData(self, editor: QWidget, model: QAbstractItemModel,
                     index: QModelIndex):
        """Set the model data from the editor."""
        from PyQt6.QtWidgets import QComboBox

        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor: QWidget, option: QStyleOptionViewItem,
                             index: QModelIndex):
        """Set the editor geometry."""
        editor.setGeometry(option.rect)
