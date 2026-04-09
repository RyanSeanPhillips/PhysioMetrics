"""
Marker edit dialog.

Simple dialog for editing event marker properties (notes, color, condition).
"""

from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QTextEdit, QPushButton, QLineEdit, QDialogButtonBox,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt


class MarkerEditDialog(QDialog):
    """Dialog for editing an event marker's properties."""

    def __init__(self, marker, viewmodel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Marker")
        self.setMinimumWidth(380)
        self._marker = marker
        self._viewmodel = viewmodel
        self._selected_color: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Header — marker identity
        cat_obj = self._viewmodel.service.registry.get_or_default(self._marker.category)
        cat_display = cat_obj.display_name if cat_obj else self._marker.category.title()
        label_display = cat_obj.get_display_label(self._marker.label) if cat_obj else self._marker.label.title()
        header = QLabel(f"<b>{cat_display}: {label_display}</b>  ({self._marker.id})")
        layout.addWidget(header)

        # Time info
        if self._marker.is_paired and self._marker.end_time is not None:
            time_text = f"{self._marker.start_time:.3f}s - {self._marker.end_time:.3f}s ({self._marker.duration:.3f}s)"
        else:
            time_text = f"{self._marker.start_time:.3f}s"
        layout.addWidget(QLabel(time_text))

        # Form
        form = QFormLayout()

        # Condition
        self._condition_edit = QLineEdit(self._marker.condition or "")
        self._condition_edit.setPlaceholderText("e.g. baseline, iso, awake")
        form.addRow("Condition:", self._condition_edit)

        # Color
        color_row = QHBoxLayout()
        current_color = self._viewmodel.get_color_for_marker(self._marker)
        self._color_label = QLabel(current_color)
        self._color_label.setFixedWidth(80)
        self._color_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_color_preview(current_color)
        color_row.addWidget(self._color_label)

        self._color_btn = QPushButton("Choose...")
        self._color_btn.clicked.connect(self._pick_color)
        color_row.addWidget(self._color_btn)

        if self._marker.color_override:
            reset_btn = QPushButton("Reset")
            reset_btn.setToolTip("Reset to category default color")
            reset_btn.clicked.connect(self._reset_color)
            color_row.addWidget(reset_btn)

        color_row.addStretch()
        form.addRow("Color:", color_row)

        # Notes
        self._notes_edit = QTextEdit()
        self._notes_edit.setPlainText(self._marker.notes or "")
        self._notes_edit.setPlaceholderText("Add notes about this marker...")
        self._notes_edit.setMaximumHeight(120)
        form.addRow("Notes:", self._notes_edit)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self):
        from PyQt6.QtWidgets import QColorDialog
        current = QColor(self._color_label.text())
        color = QColorDialog.getColor(current, self, "Set Marker Color")
        if color.isValid():
            self._selected_color = color.name()
            self._update_color_preview(self._selected_color)

    def _reset_color(self):
        self._selected_color = ""  # empty string signals "clear override"
        cat_obj = self._viewmodel.service.registry.get_or_default(self._marker.category)
        default_color = cat_obj.color if cat_obj else "#888888"
        self._update_color_preview(default_color)

    def _update_color_preview(self, color_hex: str):
        self._color_label.setText(color_hex)
        self._color_label.setStyleSheet(
            f"background-color: {color_hex}; color: white; border: 1px solid #555; padding: 2px;"
        )

    def get_changes(self) -> Dict[str, Any]:
        """Return a dict of changed fields (kwargs for update_marker)."""
        changes: Dict[str, Any] = {}

        # Notes
        new_notes = self._notes_edit.toPlainText().strip()
        old_notes = self._marker.notes or ""
        if new_notes != old_notes:
            changes['notes'] = new_notes

        # Color
        if self._selected_color is not None:
            changes['color_override'] = self._selected_color

        # Condition
        new_condition = self._condition_edit.text().strip()
        old_condition = self._marker.condition or ""
        if new_condition != old_condition:
            changes['condition'] = new_condition

        return changes
