"""
Report Issue dialog for PhysioMetrics.

Allows users to manually report bugs, request features, or provide feedback
without needing a crash to occur.
"""

import webbrowser
from urllib.parse import urlencode
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QGroupBox, QApplication, QComboBox, QLineEdit,
    QCheckBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

from version_info import VERSION_STRING
import sys
import platform


# GitHub repository for issue submission
GITHUB_REPO = "RyanSeanPhillips/PhysioMetrics"


class ReportIssueDialog(QDialog):
    """
    Dialog for manually reporting issues, requesting features, or providing feedback.

    Features:
    - Report type selector (Bug, Feature Request, Question/Feedback)
    - Title and description fields
    - Option to include debug log
    - "Submit to GitHub" button - opens browser to pre-filled issue
    - "Copy to Clipboard" button
    """

    # Report types with their GitHub labels and templates
    REPORT_TYPES = {
        "Bug Report": {
            "label": "bug",
            "icon": "",
            "placeholder": "Please describe the bug:\n\n"
                          "What happened?\n\n\n"
                          "What did you expect to happen?\n\n\n"
                          "Steps to reproduce:\n"
                          "1. \n"
                          "2. \n"
                          "3. \n",
        },
        "Feature Request": {
            "label": "enhancement",
            "icon": "",
            "placeholder": "Please describe the feature you'd like:\n\n"
                          "What problem would this solve?\n\n\n"
                          "How would you like it to work?\n\n\n"
                          "Any alternatives you've considered?\n",
        },
        "Question / Feedback": {
            "label": "question",
            "icon": "",
            "placeholder": "What would you like to know or share?\n\n\n"
                          "Any additional context?\n",
        },
    }

    def __init__(self, parent=None, prefill_title: str = None, prefill_description: str = None):
        super().__init__(parent)
        self.settings = QSettings("PhysioMetrics", "ReportIssueDialog")

        self.setWindowTitle("Report Issue / Request Feature")
        self.setModal(True)
        self.resize(600, 500)

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()
        self._restore_geometry()

        # Pre-fill if provided
        if prefill_title:
            self.title_edit.setText(prefill_title)
        if prefill_description:
            self.description_edit.setText(prefill_description)

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("<h3 style='margin: 0;'>Report an Issue or Request a Feature</h3>")
        header.setWordWrap(True)
        layout.addWidget(header)

        subheader = QLabel(
            "Your feedback helps improve PhysioMetrics! "
            "This will open a GitHub issue in your browser."
        )
        subheader.setWordWrap(True)
        subheader.setStyleSheet("color: #aaa; margin-bottom: 10px;")
        layout.addWidget(subheader)

        # Report type selector
        type_layout = QHBoxLayout()
        type_label = QLabel("Report Type:")
        type_label.setMinimumWidth(80)
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.REPORT_TYPES.keys())
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_layout.addWidget(type_label)
        type_layout.addWidget(self.type_combo, 1)
        layout.addLayout(type_layout)

        # Title field
        title_layout = QHBoxLayout()
        title_label = QLabel("Title:")
        title_label.setMinimumWidth(80)
        self.title_edit = QLineEdit()
        self.title_edit.setPlaceholderText("Brief summary of the issue or request")
        self.title_edit.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
            QLineEdit:focus {
                border-color: #2a7fff;
            }
        """)
        title_layout.addWidget(title_label)
        title_layout.addWidget(self.title_edit, 1)
        layout.addLayout(title_layout)

        # Description field
        desc_group = QGroupBox("Description")
        desc_layout = QVBoxLayout()

        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText(
            self.REPORT_TYPES["Bug Report"]["placeholder"]
        )
        self.description_edit.setMinimumHeight(180)
        self.description_edit.setStyleSheet("""
            QTextEdit {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
            QTextEdit:focus {
                border-color: #2a7fff;
            }
        """)
        desc_layout.addWidget(self.description_edit)

        desc_group.setLayout(desc_layout)
        layout.addWidget(desc_group)

        # Options
        options_layout = QHBoxLayout()

        self.include_debug_log = QCheckBox("Include recent activity log (last 20 actions)")
        self.include_debug_log.setChecked(True)
        self.include_debug_log.setToolTip(
            "Includes a log of your recent actions to help diagnose the issue"
        )
        options_layout.addWidget(self.include_debug_log)

        self.include_system_info = QCheckBox("Include system info")
        self.include_system_info.setChecked(True)
        self.include_system_info.setToolTip(
            "Includes app version, platform, and Python version"
        )
        options_layout.addWidget(self.include_system_info)

        options_layout.addStretch()
        layout.addLayout(options_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        # Submit to GitHub button (primary action)
        self.submit_btn = QPushButton("Submit to GitHub")
        self.submit_btn.setDefault(True)
        self.submit_btn.clicked.connect(self._on_submit_github)
        self.submit_btn.setMinimumHeight(36)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a7fff;
                color: white;
                font-weight: bold;
                padding: 8px 20px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3d8eff;
            }
            QPushButton:pressed {
                background-color: #1a6fef;
            }
        """)
        button_layout.addWidget(self.submit_btn)

        # Copy to clipboard
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self._on_copy_clipboard)
        self.copy_btn.setMinimumHeight(36)
        button_layout.addWidget(self.copy_btn)

        # Cancel
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setMinimumHeight(36)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def _on_type_changed(self, report_type: str):
        """Update placeholder when report type changes."""
        if report_type in self.REPORT_TYPES:
            self.description_edit.setPlaceholderText(
                self.REPORT_TYPES[report_type]["placeholder"]
            )

    def _get_debug_log_text(self) -> str:
        """Get formatted debug log for inclusion in report."""
        try:
            from core import error_reporting
            entries = error_reporting.get_debug_log(limit=20)

            if not entries:
                return "No recent activity recorded."

            lines = ["Recent Activity:"]
            for entry in entries:
                time = entry.get('timestamp', '')
                if 'T' in time:
                    time = time.split('T')[1].split('.')[0]  # HH:MM:SS
                action = entry.get('action', '?')
                context = entry.get('context', '')
                session = entry.get('session_id', '')[:4]

                if context:
                    lines.append(f"  [{time}] ({session}) {action} - {context}")
                else:
                    lines.append(f"  [{time}] ({session}) {action}")

            return "\n".join(lines)
        except Exception as e:
            return f"Could not retrieve debug log: {e}"

    def _get_system_info(self) -> str:
        """Get system information for inclusion in report."""
        return f"""**App Version:** {VERSION_STRING}
**Platform:** {sys.platform}
**Python:** {platform.python_version()}
**OS:** {platform.platform()}"""

    def _format_report_body(self) -> str:
        """Format the full report body."""
        report_type = self.type_combo.currentText()
        description = self.description_edit.toPlainText().strip()

        body_parts = []

        # System info
        if self.include_system_info.isChecked():
            body_parts.append(self._get_system_info())
            body_parts.append("")

        # Description
        body_parts.append("## Description")
        body_parts.append(description if description else "_No description provided_")
        body_parts.append("")

        # Debug log
        if self.include_debug_log.isChecked():
            body_parts.append("<details>")
            body_parts.append("<summary>Recent Activity Log (click to expand)</summary>")
            body_parts.append("")
            body_parts.append("```")
            body_parts.append(self._get_debug_log_text())
            body_parts.append("```")
            body_parts.append("</details>")
            body_parts.append("")

        # Footer
        body_parts.append("---")
        body_parts.append("*Submitted via PhysioMetrics Report Issue dialog*")

        return "\n".join(body_parts)

    def _on_submit_github(self):
        """Open browser to GitHub with pre-filled issue."""
        report_type = self.type_combo.currentText()
        title = self.title_edit.text().strip()

        if not title:
            title = f"{report_type}: (please add title)"

        # Add prefix based on type
        if report_type == "Bug Report" and not title.startswith("[Bug]"):
            title = f"[Bug] {title}"
        elif report_type == "Feature Request" and not title.startswith("[Feature]"):
            title = f"[Feature] {title}"
        elif report_type == "Question / Feedback" and not title.startswith("[Question]"):
            title = f"[Question] {title}"

        body = self._format_report_body()
        label = self.REPORT_TYPES[report_type]["label"]

        # Build GitHub URL
        params = {
            "title": title,
            "body": body,
            "labels": label,
        }

        url = f"https://github.com/{GITHUB_REPO}/issues/new?{urlencode(params)}"
        webbrowser.open(url)

        self.accept()

    def _on_copy_clipboard(self):
        """Copy report to clipboard."""
        report_type = self.type_combo.currentText()
        title = self.title_edit.text().strip() or "(no title)"

        full_report = f"# {report_type}: {title}\n\n{self._format_report_body()}"

        clipboard = QApplication.clipboard()
        clipboard.setText(full_report)

        # Show feedback
        self.copy_btn.setText("Copied!")
        self.copy_btn.setEnabled(False)

        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self._reset_copy_button)

    def _reset_copy_button(self):
        """Reset copy button to original state."""
        self.copy_btn.setText("Copy to Clipboard")
        self.copy_btn.setEnabled(True)

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
                background-color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QComboBox {
                background-color: #2a2a2a;
                color: #e0e0e0;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 6px 10px;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #555555;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #e0e0e0;
                selection-background-color: #2a7fff;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #555555;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QPushButton {
                background-color: #404040;
                color: #e0e0e0;
                border: 1px solid #555555;
                padding: 6px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #666666;
            }
            QPushButton:pressed {
                background-color: #353535;
            }
            QPushButton:disabled {
                background-color: #353535;
                color: #888888;
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows."""
        try:
            import ctypes
            hwnd = int(self.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd,
                DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(value),
                ctypes.sizeof(value)
            )
        except Exception:
            pass

    def _restore_geometry(self):
        """Restore window position from settings."""
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))

    def closeEvent(self, event):
        """Save geometry on close."""
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def show_report_issue_dialog(parent=None, prefill_title: str = None,
                              prefill_description: str = None) -> bool:
    """
    Show the report issue dialog.

    Args:
        parent: Parent widget
        prefill_title: Optional pre-filled title
        prefill_description: Optional pre-filled description

    Returns:
        True if user submitted the report, False if cancelled
    """
    dialog = ReportIssueDialog(
        parent=parent,
        prefill_title=prefill_title,
        prefill_description=prefill_description
    )
    result = dialog.exec()
    return result == QDialog.DialogCode.Accepted
