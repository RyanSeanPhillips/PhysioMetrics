"""
Event Selection Dialog.

Allows users to select which detected events to keep before creating markers.
Particularly useful for Hargreaves experiments where detection finds 30 events
but only 10 (one animal's stims) are needed.
"""

from typing import List, Tuple
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QLabel, QSpinBox, QCheckBox,
)
from PyQt6.QtCore import Qt


class EventSelectionDialog(QDialog):
    """Dialog for selecting which detected events to create as markers."""

    def __init__(
        self,
        events: List[Tuple[float, float]],
        parent=None,
    ):
        super().__init__(parent)
        self._events = events
        self._setup_ui()
        self._populate_table()

    def _setup_ui(self) -> None:
        self.setWindowTitle(f"Select Events ({len(self._events)} detected)")
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout(self)

        # Info label
        info = QLabel(
            f"{len(self._events)} events detected. "
            "Uncheck events you don't want to create as markers."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["#", "Start (s)", "End (s)", "Duration (s)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        layout.addWidget(self._table)

        # Button row: Select All, Deselect All, Select First N
        btn_layout = QHBoxLayout()

        btn_select_all = QPushButton("Select All")
        btn_select_all.clicked.connect(self._select_all)
        btn_layout.addWidget(btn_select_all)

        btn_deselect_all = QPushButton("Deselect All")
        btn_deselect_all.clicked.connect(self._deselect_all)
        btn_layout.addWidget(btn_deselect_all)

        btn_layout.addStretch()

        btn_layout.addWidget(QLabel("Select First:"))
        self._spin_n = QSpinBox()
        self._spin_n.setMinimum(1)
        self._spin_n.setMaximum(len(self._events))
        self._spin_n.setValue(min(10, len(self._events)))
        btn_layout.addWidget(self._spin_n)

        btn_select_n = QPushButton("Apply")
        btn_select_n.clicked.connect(self._select_first_n)
        btn_layout.addWidget(btn_select_n)

        layout.addLayout(btn_layout)

        # OK / Cancel
        ok_cancel = QHBoxLayout()
        ok_cancel.addStretch()
        btn_ok = QPushButton("OK — Create Selected")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        ok_cancel.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        ok_cancel.addWidget(btn_cancel)
        layout.addLayout(ok_cancel)

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._events))
        self._checkboxes: List[QCheckBox] = []

        for i, (start, end) in enumerate(self._events):
            # Checkbox in first column
            cb = QCheckBox(f"  {i + 1}")
            cb.setChecked(True)
            self._checkboxes.append(cb)
            self._table.setCellWidget(i, 0, cb)

            # Start time
            item_start = QTableWidgetItem(f"{start:.3f}")
            item_start.setFlags(item_start.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 1, item_start)

            # End time
            item_end = QTableWidgetItem(f"{end:.3f}")
            item_end.setFlags(item_end.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 2, item_end)

            # Duration
            duration = end - start
            item_dur = QTableWidgetItem(f"{duration:.3f}")
            item_dur.setFlags(item_dur.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(i, 3, item_dur)

    def _select_all(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(True)

    def _deselect_all(self) -> None:
        for cb in self._checkboxes:
            cb.setChecked(False)

    def _select_first_n(self) -> None:
        n = self._spin_n.value()
        for i, cb in enumerate(self._checkboxes):
            cb.setChecked(i < n)

    def get_selected_events(self) -> List[Tuple[float, float]]:
        """Return only the events whose checkboxes are checked."""
        return [
            self._events[i]
            for i, cb in enumerate(self._checkboxes)
            if cb.isChecked()
        ]
