"""
File tree widget for the Projects tab.

Displays a folder tree using QFileSystemModel. When the user clicks a folder,
emits folder_selected(path) so the experiments table can filter by path prefix.
"""

from PyQt6.QtCore import pyqtSignal, QDir
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QLabel, QHBoxLayout, QPushButton
from PyQt6.QtGui import QFileSystemModel


class FileTreeWidget(QWidget):
    """Folder browser for scoping the experiments table to a subtree."""

    folder_selected = pyqtSignal(str)   # Emitted with folder path (or "" to clear)
    folder_cleared = pyqtSignal()       # Emitted when selection is cleared

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_path = ""

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Header bar
        header = QHBoxLayout()
        header.setContentsMargins(4, 4, 4, 0)
        self._title_label = QLabel("Folders")
        self._title_label.setStyleSheet("font-weight: bold; color: #cccccc; font-size: 11px;")
        header.addWidget(self._title_label)
        header.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setFixedHeight(20)
        self._clear_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #569cd6; border: none;
                font-size: 10px; padding: 0 6px;
            }
            QPushButton:hover { color: #9cdcfe; text-decoration: underline; }
        """)
        self._clear_btn.setToolTip("Clear folder filter — show all experiments")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        self._clear_btn.hide()
        header.addWidget(self._clear_btn)
        layout.addLayout(header)

        # File system model — directories only
        self._fs_model = QFileSystemModel()
        self._fs_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self._fs_model.setNameFilterDisables(False)

        # Tree view
        self._tree = QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setHeaderHidden(True)
        # Hide Size, Type, Date Modified columns — show only Name
        for col in (1, 2, 3):
            self._tree.setColumnHidden(col, True)
        self._tree.setAnimated(True)
        self._tree.setIndentation(16)
        self._tree.setStyleSheet("""
            QTreeView {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3e3e42;
                font-size: 11px;
            }
            QTreeView::item:selected {
                background-color: #094771;
                color: #ffffff;
            }
            QTreeView::item:hover {
                background-color: #2a2d2e;
            }
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {
                image: none;
                border-image: none;
            }
        """)
        self._tree.clicked.connect(self._on_item_clicked)
        layout.addWidget(self._tree)

    def set_root(self, path: str):
        """Set the root folder for the tree. Call this when a data directory is known."""
        if not path:
            return
        self._root_path = path
        root_index = self._fs_model.setRootPath(path)
        self._tree.setRootIndex(root_index)
        self._title_label.setToolTip(path)

    def root_path(self) -> str:
        return self._root_path

    def _on_item_clicked(self, index):
        """Handle folder click — emit the selected path."""
        path = self._fs_model.filePath(index)
        if path:
            self._clear_btn.show()
            self.folder_selected.emit(path)

    def _on_clear_clicked(self):
        """Clear the current selection and filter."""
        self._tree.clearSelection()
        self._clear_btn.hide()
        self.folder_cleared.emit()
        self.folder_selected.emit("")
