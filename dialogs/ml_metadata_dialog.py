"""
ML Metadata Dialog

Prompts user to enter metadata for ML training data exports.
Captures quality score and optional user name (system info captured automatically).
"""

import sys
import os
import socket
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QLabel, QPushButton, QMessageBox, QComboBox, QTextEdit
)
from PyQt6.QtCore import Qt

from dialogs.export_mixin import ExportMixin


class MLMetadataDialog(ExportMixin, QDialog):
    """Dialog for entering ML training data metadata."""

    def __init__(self, parent=None, last_user_name=None):
        """
        Initialize the ML metadata dialog.

        Args:
            parent: Parent widget
            last_user_name: Last entered user name (for consistency across files in same session)

        Note: Experimental conditions (state, drug, gas) are NOT auto-filled to prevent
              accidental mislabeling when conditions change between recordings.
        """
        super().__init__(parent)
        self.setWindowTitle("ML Training Data - Metadata")
        self.resize(550, 600)

        self.last_user_name = last_user_name

        # Get system info to display
        self.system_username = os.getenv('USERNAME') or os.getenv('USER') or 'unknown'
        self.computer_name = socket.gethostname()

        self._setup_ui()
        self._enable_dark_title_bar()
        self.setup_export_menu()

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            "Enter metadata for this labeled dataset.\n"
            "System info and timestamp captured automatically."
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(header)

        # Form layout for inputs
        form = QFormLayout()

        # System info (read-only display)
        system_info_label = QLabel(
            f"<b>System:</b> {self.system_username}@{self.computer_name}"
        )
        system_info_label.setStyleSheet("color: #666; font-size: 10px;")
        form.addRow("", system_info_label)

        # Optional user name field (for filtering datasets)
        self.user_name_input = QLineEdit()
        self.user_name_input.setPlaceholderText("Optional: Your name or initials for dataset filtering")
        if self.last_user_name:
            self.user_name_input.setText(self.last_user_name)

        user_name_label = QLabel("User Name (optional):")
        user_name_help = QLabel("Additional label for filtering datasets. System info always saved separately.")
        user_name_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(user_name_label, self.user_name_input)
        form.addRow("", user_name_help)

        # Animal state dropdown (optional) - always starts at "(not specified)"
        self.state_input = QComboBox()
        self.state_input.addItem("(not specified)", userData="")
        self.state_input.addItem("Awake", userData="awake")
        self.state_input.addItem("Anesthetized", userData="anesthetized")

        state_label = QLabel("Animal State (optional):")
        state_help = QLabel("Select for each recording to avoid mislabeling.")
        state_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(state_label, self.state_input)
        form.addRow("", state_help)

        # Anesthetic type field (optional) - always starts blank
        self.anesthetic_input = QLineEdit()
        self.anesthetic_input.setPlaceholderText("e.g., Isoflurane, Urethane, Ketamine/Xylazine")

        anesthetic_label = QLabel("Anesthetic Type (optional):")
        anesthetic_help = QLabel("If anesthetized, specify the anesthetic used.")
        anesthetic_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(anesthetic_label, self.anesthetic_input)
        form.addRow("", anesthetic_help)

        # Drug/treatment field (optional) - always starts blank
        self.drug_input = QLineEdit()
        self.drug_input.setPlaceholderText("e.g., Morphine, Fentanyl, CNO, Saline")

        drug_label = QLabel("Drug/Treatment (optional):")
        drug_help = QLabel("Any drug or treatment administered during recording.")
        drug_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(drug_label, self.drug_input)
        form.addRow("", drug_help)

        # Drug concentration/dose field (optional)
        self.concentration_input = QLineEdit()
        self.concentration_input.setPlaceholderText("e.g., 5mg/kg, 2%, 10uM")

        concentration_label = QLabel("Dose/Concentration (optional):")
        form.addRow(concentration_label, self.concentration_input)

        # Gas condition dropdown (optional) - always starts at "(not specified)"
        self.gas_input = QComboBox()
        self.gas_input.addItem("(not specified)", userData="")
        self.gas_input.addItem("Room Air", userData="room_air")
        self.gas_input.addItem("Hypoxia", userData="hypoxia")
        self.gas_input.addItem("Hypercapnia", userData="hypercapnia")

        gas_label = QLabel("Gas Condition (optional):")
        gas_help = QLabel("Select for each recording to avoid mislabeling.")
        gas_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(gas_label, self.gas_input)
        form.addRow("", gas_help)

        # Notes field (optional)
        self.notes_input = QTextEdit()
        self.notes_input.setPlaceholderText("Optional notes about this dataset (experimental details, issues, etc.)")
        self.notes_input.setMaximumHeight(80)

        notes_label = QLabel("Notes (optional):")
        form.addRow(notes_label, self.notes_input)

        # Quality score field (starts blank - forces user to think about it)
        self.quality_input = QSpinBox()
        self.quality_input.setMinimum(0)  # 0 = not set
        self.quality_input.setMaximum(10)
        self.quality_input.setValue(0)  # Start blank
        self.quality_input.setSpecialValueText("(not rated)")  # Display text when value is 0
        self.quality_input.setSuffix(" / 10")

        quality_label = QLabel("Quality Score:")
        quality_help = QLabel(
            "How confident are you in these labels?\n"
            "1-3 = Low (uncertain labels, noisy signal)\n"
            "4-7 = Medium (decent quality, some issues)\n"
            "8-10 = High (confident labels, clean signal)"
        )
        quality_help.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")

        form.addRow(quality_label, self.quality_input)
        form.addRow("", quality_help)

        layout.addLayout(form)

        # Explanation section
        explanation = QLabel(
            "<b>Why this matters:</b><br>"
            "• <b>Timestamp</b>: Automatically saved when file is created<br>"
            "• <b>System info</b>: Computer + username for record-keeping<br>"
            "• <b>Experimental conditions</b>: State, drugs, gas - enables condition-specific analysis<br>"
            "• <b>Quality score</b>: Filter low-quality data during training (required)"
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;")
        layout.addWidget(explanation)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("Save with Metadata")
        save_btn.clicked.connect(self._on_save_clicked)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _on_save_clicked(self):
        """Validate input before accepting."""
        # Check if quality score was set (not 0)
        if self.quality_input.value() == 0:
            QMessageBox.warning(
                self,
                "Quality Score Required",
                "Please rate the quality of this dataset (1-10).\n\n"
                "This helps filter data during model training."
            )
            self.quality_input.setFocus()
            return

        # All good, accept the dialog
        self.accept()

    def get_metadata(self):
        """
        Get the entered metadata including system info and timestamp.

        Returns:
            dict: Metadata with keys:
                - 'timestamp': ISO format timestamp when saved (automatic)
                - 'system_username': Computer username (automatic, always saved)
                - 'computer_name': Computer hostname (automatic, always saved)
                - 'user_name': Optional user-entered name (empty string if not provided)
                - 'animal_state': Awake/Anesthetized (empty string if not specified)
                - 'anesthetic_type': Type of anesthetic used (empty string if not provided)
                - 'drug': Drug/treatment administered (empty string if not provided)
                - 'drug_concentration': Dose or concentration (empty string if not provided)
                - 'gas_condition': room_air/hypoxia/hypercapnia (empty string if not specified)
                - 'notes': Optional notes (empty string if not provided)
                - 'quality_score': Quality rating (1-10)
        """
        user_name = self.user_name_input.text().strip()
        animal_state = self.state_input.currentData() or ""
        anesthetic_type = self.anesthetic_input.text().strip()
        drug = self.drug_input.text().strip()
        drug_concentration = self.concentration_input.text().strip()
        gas_condition = self.gas_input.currentData() or ""
        notes = self.notes_input.toPlainText().strip()

        return {
            'timestamp': datetime.now().isoformat(),  # ISO 8601 format: 2025-11-14T12:34:56
            'system_username': self.system_username,
            'computer_name': self.computer_name,
            'user_name': user_name,  # Empty string if not provided
            'animal_state': animal_state,  # Empty string if not specified
            'anesthetic_type': anesthetic_type,  # Empty string if not provided
            'drug': drug,  # Empty string if not provided
            'drug_concentration': drug_concentration,  # Empty string if not provided
            'gas_condition': gas_condition,  # Empty string if not specified
            'notes': notes,  # Empty string if not provided
            'quality_score': self.quality_input.value()
        }
