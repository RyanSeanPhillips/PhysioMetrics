"""
FileListManager - Handles curation tab list box operations.

Manages moving items between file lists and search/filter functionality.
Extracted from NavigationManager (navigation logic moved to MVVM).
"""

from PyQt6.QtCore import Qt


class FileListManager:
    """Manages file list operations in the curation tab."""

    def __init__(self, main_window):
        self.main = main_window
        self._connect_signals()

    def _connect_signals(self):
        """Connect curation tab list navigation buttons and filter."""
        self.main.moveAllRight.clicked.connect(self.on_move_all_right)
        self.main.moveSingleRight.clicked.connect(self.on_move_selected_right)
        self.main.moveSingleLeft.clicked.connect(self.on_move_selected_left)
        self.main.moveAllLeft.clicked.connect(self.on_move_all_left)
        self.main.FileListSearchBox.textChanged.connect(self._filter_file_list)

    ##################################################
    ## List Box Navigation (Curation Tab)           ##
    ##################################################

    def _list_has_key(self, lw, key: str) -> bool:
        """True if any item in lw has the same group key."""
        for i in range(lw.count()):
            it = lw.item(i)
            if not it:
                continue
            meta = it.data(Qt.ItemDataRole.UserRole) or {}
            if isinstance(meta, dict) and meta.get("key", "").lower() == key.lower():
                return True
        return False

    def _move_items(self, src_lw, dst_lw, rows_to_move: list[int]):
        """
        Move grouped items by root from src_lw to dst_lw.
        Duplicate check is by 'key' (dir+root), not by file path.
        """
        if not rows_to_move:
            return 0, 0

        plan = []
        for r in rows_to_move:
            it = src_lw.item(r)
            if it is None:
                continue
            meta = it.data(Qt.ItemDataRole.UserRole) or {}
            key = (meta.get("key") or "").lower()
            is_dup = self._list_has_key(dst_lw, key)
            plan.append((r, is_dup))

        taken = []
        skipped_dups = 0
        for r, is_dup in sorted(plan, key=lambda x: x[0], reverse=True):
            if is_dup:
                skipped_dups += 1
                continue
            it = src_lw.takeItem(r)
            if it is not None:
                taken.append((r, it))

        moved = 0
        for _, it in sorted(taken, key=lambda x: x[0]):
            dst_lw.addItem(it)
            moved += 1

        src_lw.sortItems()
        dst_lw.sortItems()
        return moved, skipped_dups

    def on_move_selected_right(self):
        """Move selected from left (FileList) to right (FilestoConsolidateList)."""
        src = self.main.FileList
        dst = self.main.FilestoConsolidateList
        rows = [src.row(it) for it in src.selectedItems()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} item(s) to right. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_all_right(self):
        """Move ALL VISIBLE from left to right."""
        src = self.main.FileList
        dst = self.main.FilestoConsolidateList
        rows = [i for i in range(src.count()) if not src.item(i).isHidden()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} visible item(s) to right. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_selected_left(self):
        """Move selected from right back to left."""
        src = self.main.FilestoConsolidateList
        dst = self.main.FileList
        rows = [src.row(it) for it in src.selectedItems()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} item(s) to left. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_all_left(self):
        """Move ALL VISIBLE from right back to left."""
        src = self.main.FilestoConsolidateList
        dst = self.main.FileList
        rows = [i for i in range(src.count()) if not src.item(i).isHidden()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} visible item(s) to left. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    ##################################################
    ## File List Filter (Curation Tab)              ##
    ##################################################

    def _filter_file_list(self, text: str):
        """Show/hide items in FileList based on search text.

        Supports multiple search modes:
        - Single keyword: 'gfp' - shows files containing 'gfp'
        - Multiple keywords (AND): 'gfp 2.5mW' - shows files containing BOTH 'gfp' AND '2.5mW'
        - Multiple keywords (OR): 'gfp, chr2' - shows files containing EITHER 'gfp' OR 'chr2'
        """
        search_text = text.strip().lower()

        # Determine search mode
        if ',' in search_text:
            keywords = [k.strip() for k in search_text.split(',') if k.strip()]
            search_mode = 'OR'
        else:
            keywords = [k.strip() for k in search_text.split() if k.strip()]
            search_mode = 'AND'

        for i in range(self.main.FileList.count()):
            item = self.main.FileList.item(i)
            if not item:
                continue

            item_text = item.text().lower()
            tooltip = (item.toolTip() or "").lower()
            combined_text = f"{item_text} {tooltip}"

            if not keywords:
                item.setHidden(False)
                continue

            if search_mode == 'AND':
                matches = all(kw in combined_text for kw in keywords)
            else:
                matches = any(kw in combined_text for kw in keywords)

            item.setHidden(not matches)
