"""
Provenance Viewer Dialog — shows how metadata values were determined.

Displays a color-coded table of provenance records for a file:
- Green:  user-confirmed values
- Blue:   values from notes files
- Yellow: values from folder structure
- Orange: values from pattern matching
- Red:    uncertain / low-confidence values

Shows correction chains when a value has been superseded.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QComboBox,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush

from typing import List, Dict, Any, Optional


# Source type → display color
SOURCE_COLORS = {
    "user":       QColor(144, 238, 144),  # Light green
    "notes":      QColor(135, 206, 250),  # Light blue
    "folder":     QColor(255, 255, 150),  # Light yellow
    "pattern":    QColor(255, 200, 100),  # Orange
    "inspection": QColor(200, 200, 255),  # Light purple
    "image":      QColor(255, 182, 193),  # Light pink
}

DEFAULT_COLOR = QColor(255, 200, 200)  # Light red for uncertain/unknown


class ProvenanceViewerDialog(QDialog):
    """
    Dialog showing provenance records for a file's metadata.

    Args:
        file_path: Relative path to the file.
        provenance_records: List of provenance dicts from ProjectService.get_provenance().
        parent: Parent widget.
    """

    def __init__(
        self,
        file_path: str,
        provenance_records: List[Dict[str, Any]],
        parent=None,
    ):
        super().__init__(parent)
        self._file_path = file_path
        self._records = provenance_records
        self._setup_ui()
        self._populate()

    def _setup_ui(self):
        self.setWindowTitle(f"Provenance — {self._file_path}")
        self.setMinimumSize(700, 400)
        self.resize(800, 500)

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(f"<b>Provenance for:</b> {self._file_path}")
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        # Filter bar
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter by field:"))
        self._field_combo = QComboBox()
        self._field_combo.addItem("All fields")
        # Collect unique fields
        fields = sorted(set(r.get("field", "") for r in self._records))
        for f in fields:
            self._field_combo.addItem(f)
        self._field_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self._field_combo)
        filter_layout.addStretch()

        # Legend
        legend = QLabel(
            '<span style="background:#90EE90;">&nbsp;User&nbsp;</span> '
            '<span style="background:#87CEFA;">&nbsp;Notes&nbsp;</span> '
            '<span style="background:#FFFF96;">&nbsp;Folder&nbsp;</span> '
            '<span style="background:#FFC864;">&nbsp;Pattern&nbsp;</span> '
            '<span style="background:#C8C8FF;">&nbsp;Inspection&nbsp;</span>'
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        filter_layout.addWidget(legend)
        layout.addLayout(filter_layout)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels([
            "Field", "Value", "Source", "Detail", "Confidence", "Date",
        ])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # Stats bar
        self._stats_label = QLabel()
        layout.addWidget(self._stats_label)

        # Close button
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def _populate(self, field_filter: Optional[str] = None):
        """Fill the table with provenance records."""
        records = self._records
        if field_filter and field_filter != "All fields":
            records = [r for r in records if r.get("field") == field_filter]

        self._table.setRowCount(len(records))

        for row, rec in enumerate(records):
            source_type = rec.get("source_type", "unknown")
            color = SOURCE_COLORS.get(source_type, DEFAULT_COLOR)
            brush = QBrush(color)

            # Superseded records get dimmed
            is_superseded = any(
                r.get("supersedes") == rec.get("prov_id")
                for r in self._records
            )

            items = [
                rec.get("field", ""),
                rec.get("value", ""),
                source_type,
                rec.get("source_detail", "") or rec.get("reason", ""),
                f"{rec.get('confidence', 0):.0%}",
                (rec.get("created_at", ""))[:16],  # Trim to datetime
            ]

            for col, text in enumerate(items):
                item = QTableWidgetItem(str(text))
                item.setBackground(brush)
                if is_superseded:
                    item.setForeground(QBrush(QColor(128, 128, 128)))
                    font = item.font()
                    font.setStrikeOut(True)
                    item.setFont(font)
                self._table.setItem(row, col, item)

        # Stats
        by_source = {}
        for r in records:
            st = r.get("source_type", "unknown")
            by_source[st] = by_source.get(st, 0) + 1
        active = len([r for r in records if not any(
            r2.get("supersedes") == r.get("prov_id") for r2 in self._records
        )])
        parts = [f"{k}: {v}" for k, v in sorted(by_source.items())]
        self._stats_label.setText(
            f"{len(records)} records ({active} active) — " + ", ".join(parts)
        )

    def _on_filter_changed(self, text: str):
        self._populate(field_filter=text)
