"""
Folder Tree ViewModel — QObject wrapper for FolderTreeService.

Manages the tree state, folder selection, and navigation history.
"""

from typing import List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from core.services.folder_tree_service import FolderTreeService, FolderNode


class FolderTreeViewModel(QObject):
    """ViewModel for the DB-backed folder tree."""

    tree_updated = pyqtSignal()                  # Tree was rebuilt
    folder_selected = pyqtSignal(str)            # path (or "" to clear)
    nav_changed = pyqtSignal(bool, bool)         # can_back, can_forward

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = FolderTreeService()
        self._root: Optional[FolderNode] = None
        self._root_path: str = ""

        # Navigation history
        self._history: List[str] = []
        self._history_pos: int = -1
        self._current_path: str = ""

    @property
    def root_node(self) -> Optional[FolderNode]:
        return self._root

    @property
    def root_path(self) -> str:
        return self._root_path

    @property
    def current_path(self) -> str:
        return self._current_path

    def rebuild(self, file_paths: List[str], root_path: str = ""):
        """Rebuild the tree from DB file paths."""
        self._root = self._service.build_tree(file_paths, root_path)
        if self._root:
            self._root_path = self._root.full_path
        else:
            self._root_path = ""
        self.tree_updated.emit()

    def select_folder(self, path: str):
        """Select a folder, adding to nav history."""
        if path == self._current_path:
            return
        # Truncate forward history
        if self._history_pos < len(self._history) - 1:
            self._history = self._history[:self._history_pos + 1]
        self._history.append(path)
        self._history_pos = len(self._history) - 1
        self._current_path = path
        self.folder_selected.emit(path)
        self._emit_nav()

    def clear_selection(self):
        """Clear folder selection."""
        self.select_folder("")

    def go_back(self):
        """Navigate back in folder history."""
        if self._history_pos > 0:
            self._history_pos -= 1
            self._current_path = self._history[self._history_pos]
            self.folder_selected.emit(self._current_path)
            self._emit_nav()

    def go_forward(self):
        """Navigate forward in folder history."""
        if self._history_pos < len(self._history) - 1:
            self._history_pos += 1
            self._current_path = self._history[self._history_pos]
            self.folder_selected.emit(self._current_path)
            self._emit_nav()

    def can_go_back(self) -> bool:
        return self._history_pos > 0

    def can_go_forward(self) -> bool:
        return self._history_pos < len(self._history) - 1

    def _emit_nav(self):
        self.nav_changed.emit(self.can_go_back(), self.can_go_forward())
