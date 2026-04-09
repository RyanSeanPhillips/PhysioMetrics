"""
File tree widget for the Projects tab.

Displays a folder tree built from DB experiment paths (instant, no filesystem I/O).
Supports navigation history (back/forward), file count badges, and hybrid
filesystem discovery for unindexed folders.
"""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QLabel, QHBoxLayout,
    QPushButton, QMenu, QStyle,
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QColor, QIcon, QPixmap, QPainter

from core.services.folder_tree_service import FolderNode


class FileTreeWidget(QWidget):
    """Folder browser for scoping the experiments table to a subtree."""

    folder_selected = pyqtSignal(str)   # Emitted with folder path (or "" to clear)
    folder_cleared = pyqtSignal()       # Emitted when selection is cleared
    scan_requested = pyqtSignal(str)    # Emitted when user wants to scan an unindexed folder

    # Custom data roles
    PATH_ROLE = Qt.ItemDataRole.UserRole + 1
    INDEXED_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._discovered_folders = set()  # Folders already checked for unindexed children

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Nav buttons — hidden until history exists
        self._nav_widget = QWidget()
        nav_row = QHBoxLayout(self._nav_widget)
        nav_row.setContentsMargins(6, 2, 4, 0)
        nav_row.setSpacing(2)

        self._back_btn = QPushButton("<")
        self._back_btn.setFixedSize(18, 18)
        self._back_btn.setToolTip("Go back")
        self._back_btn.setEnabled(False)
        self._back_btn.setStyleSheet(self._nav_btn_style())
        nav_row.addWidget(self._back_btn)

        self._forward_btn = QPushButton(">")
        self._forward_btn.setFixedSize(18, 18)
        self._forward_btn.setToolTip("Go forward")
        self._forward_btn.setEnabled(False)
        self._forward_btn.setStyleSheet(self._nav_btn_style())
        nav_row.addWidget(self._forward_btn)

        nav_row.addStretch()
        self._nav_widget.hide()  # Only show when there's history
        layout.addWidget(self._nav_widget)

        # Hidden label — count is shown in the root tree node itself
        self._title_label = QLabel()
        self._title_label.hide()

        # Tree model (QStandardItemModel — instant, no filesystem I/O)
        self._model = QStandardItemModel()

        # Tree view
        self._tree = QTreeView()
        self._tree.setModel(self._model)
        self._tree.setHeaderHidden(True)
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
        """)
        self._tree.clicked.connect(self._on_item_clicked)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        self._tree.expanded.connect(self._on_item_expanded)
        layout.addWidget(self._tree)

    def set_viewmodel(self, vm):
        """Connect to a FolderTreeViewModel."""
        self._vm = vm
        vm.tree_updated.connect(self._rebuild_tree)
        vm.nav_changed.connect(self._on_nav_changed)
        self._back_btn.clicked.connect(vm.go_back)
        self._forward_btn.clicked.connect(vm.go_forward)

    def _rebuild_tree(self):
        """Rebuild the QStandardItemModel from the viewmodel's tree."""
        self._model.clear()
        self._discovered_folders.clear()
        root = self._vm.root_node
        if root is None:
            self._title_label.setText("Folders")
            return

        self._title_label.setText(f"Folders ({root.total_count})")
        self._title_label.setToolTip(root.full_path or "All locations")

        # If there's a single child under the virtual root, skip the virtual root
        # and show the child directly to reduce nesting
        if root.name == "All Locations" and len(root.children) == 1:
            only_child = next(iter(root.children.values()))
            root_item = self._make_item(only_child)
        else:
            root_item = self._make_item(root)

        self._model.appendRow(root_item)
        self._tree.expand(self._model.indexFromItem(root_item))

    def _folder_icon(self) -> QIcon:
        """Get the standard folder icon (cached)."""
        if not hasattr(self, '_cached_folder_icon'):
            self._cached_folder_icon = self.style().standardIcon(
                QStyle.StandardPixmap.SP_DirIcon)
        return self._cached_folder_icon

    def _folder_icon_gray(self) -> QIcon:
        """Get a grayed-out folder icon for unindexed folders (cached)."""
        if not hasattr(self, '_cached_folder_icon_gray'):
            # Render the normal icon's Disabled mode into a new icon
            base = self._folder_icon()
            pm = base.pixmap(16, 16, QIcon.Mode.Disabled)
            self._cached_folder_icon_gray = QIcon(pm)
        return self._cached_folder_icon_gray

    def _make_item(self, node: FolderNode) -> QStandardItem:
        """Create a QStandardItem from a FolderNode, recursively."""
        label = f"{node.name}  ({node.total_count})" if node.total_count else node.name
        item = QStandardItem(label)
        item.setIcon(self._folder_icon() if node.is_indexed else self._folder_icon_gray())
        item.setData(node.full_path, self.PATH_ROLE)
        item.setData(node.is_indexed, self.INDEXED_ROLE)
        item.setEditable(False)

        if not node.is_indexed:
            # Gray + italic for unindexed
            item.setForeground(QColor(120, 120, 120))
            font = item.font()
            font.setItalic(True)
            item.setFont(font)

        for child in node.sorted_children():
            item.appendRow(self._make_item(child))
        return item

    def add_unindexed_children(self, parent_item: QStandardItem, unindexed_names: list, parent_path: str):
        """Add grayed-out items for unindexed filesystem folders."""
        import os
        for name in unindexed_names:
            full_path = os.path.join(parent_path, name)
            item = QStandardItem(name)
            item.setIcon(self._folder_icon_gray())
            item.setData(full_path, self.PATH_ROLE)
            item.setData(False, self.INDEXED_ROLE)
            item.setEditable(False)
            item.setForeground(QColor(120, 120, 120))
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            parent_item.appendRow(item)

    def _on_item_clicked(self, index):
        """Handle folder click — toggle expand and emit the selected path."""
        item = self._model.itemFromIndex(index)
        if item is None:
            return

        # Toggle expand/collapse on click
        if item.hasChildren():
            self._tree.setExpanded(index, not self._tree.isExpanded(index))

        path = item.data(self.PATH_ROLE)
        if path:
            if hasattr(self, '_vm'):
                self._vm.select_folder(path)
            else:
                self.folder_selected.emit(path)
        else:
            # Clicked virtual root or empty path — clear filter
            if hasattr(self, '_vm'):
                self._vm.clear_selection()
            else:
                self.folder_selected.emit("")

    def _on_nav_changed(self, can_back: bool, can_forward: bool):
        self._back_btn.setEnabled(can_back)
        self._forward_btn.setEnabled(can_forward)
        # Only show nav bar when there's history to navigate
        self._nav_widget.setVisible(can_back or can_forward)

    def _on_item_expanded(self, index):
        """When a folder is expanded, discover unindexed siblings from filesystem."""
        item = self._model.itemFromIndex(index)
        if item is None:
            return
        path = item.data(self.PATH_ROLE)
        if not path or path in self._discovered_folders:
            return
        self._discovered_folders.add(path)

        # Collect indexed child names
        indexed_names = set()
        for i in range(item.rowCount()):
            child = item.child(i)
            if child and child.data(self.INDEXED_ROLE):
                child_path = child.data(self.PATH_ROLE) or ""
                if child_path:
                    import os
                    indexed_names.add(os.path.basename(child_path))

        # Discover unindexed via single os.scandir (read-only)
        from core.services.folder_tree_service import FolderTreeService
        svc = FolderTreeService()
        unindexed = svc.discover_unindexed(path, indexed_names)
        if unindexed:
            self.add_unindexed_children(item, unindexed, path)

    def _on_context_menu(self, pos):
        """Right-click context menu for folder items."""
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return
        item = self._model.itemFromIndex(index)
        if item is None:
            return

        path = item.data(self.PATH_ROLE) or ""
        is_indexed = item.data(self.INDEXED_ROLE)

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2b2b2b; color: #e0e0e0; border: 1px solid #555;
            }
            QMenu::item:selected { background-color: #094771; }
        """)

        if is_indexed:
            action = menu.addAction("Scan for new files")
            action.triggered.connect(lambda: self.scan_requested.emit(path))
        else:
            action = menu.addAction("Scan this folder")
            action.triggered.connect(lambda: self.scan_requested.emit(path))

        menu.addSeparator()
        copy_action = menu.addAction("Copy path")
        copy_action.triggered.connect(lambda: self._copy_path(path))

        menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _copy_path(self, path: str):
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(path)

    @staticmethod
    def _nav_btn_style() -> str:
        return """
            QPushButton {
                background: #2a2a2a; color: #999;
                border: 1px solid #555; border-radius: 3px;
                font-size: 10px; font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover { color: #fff; border-color: #777; background: #3a3a3a; }
            QPushButton:disabled { color: #444; border-color: #3e3e42; background: transparent; }
        """
