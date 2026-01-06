"""
Crash report dialog for PhysioMetrics.

Shows crash details and offers to submit as GitHub issue.
"""

import webbrowser
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QGroupBox, QApplication, QSizePolicy
)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QFont

from core.error_reporting import CrashReport, generate_github_issue_url, mark_submitted


class CrashReportDialog(QDialog):
    """
    Dialog shown after a crash or on startup if previous session crashed.

    Features:
    - Shows crash details (error type, message, traceback)
    - "Submit to GitHub" button - opens browser to pre-filled issue
    - "Copy to Clipboard" button - copies full report
    - "Dismiss" button
    """

    def __init__(self, crash_report: CrashReport, on_startup: bool = False, parent=None):
        super().__init__(parent)
        self.crash_report = crash_report
        self.on_startup = on_startup
        self.settings = QSettings("PhysioMetrics", "CrashReportDialog")

        self.setWindowTitle("Previous Session Crashed" if on_startup else "Crash Report")
        self.setModal(True)
        self.resize(650, 550)

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()
        self._restore_geometry()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header
        if self.on_startup:
            header_text = "PhysioMetrics detected that the previous session ended unexpectedly."
        else:
            header_text = "PhysioMetrics encountered an unexpected error."

        header = QLabel(f"<h3 style='margin: 0;'>{header_text}</h3>")
        header.setWordWrap(True)
        layout.addWidget(header)

        # Subheader with request
        subheader = QLabel(
            "Submitting this report helps improve PhysioMetrics. "
            "No personal data or file contents are included."
        )
        subheader.setWordWrap(True)
        subheader.setStyleSheet("color: #aaa; margin-bottom: 10px;")
        layout.addWidget(subheader)

        # Error details group
        error_group = QGroupBox("Error Details")
        error_layout = QVBoxLayout()
        error_layout.setSpacing(8)

        # Error type and message
        error_type_label = QLabel(f"<b>Error Type:</b> {self.crash_report.error_type}")
        error_layout.addWidget(error_type_label)

        # Truncate long messages
        error_msg = self.crash_report.error_message
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."

        error_msg_label = QLabel(f"<b>Message:</b> {error_msg}")
        error_msg_label.setWordWrap(True)
        error_msg_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        error_layout.addWidget(error_msg_label)

        # Traceback (scrollable)
        traceback_label = QLabel("<b>Stack Trace:</b>")
        error_layout.addWidget(traceback_label)

        self.traceback_edit = QTextEdit()
        self.traceback_edit.setPlainText(self.crash_report.traceback_str)
        self.traceback_edit.setReadOnly(True)
        self.traceback_edit.setMinimumHeight(150)
        self.traceback_edit.setMaximumHeight(200)
        self.traceback_edit.setFont(QFont("Consolas", 9))
        self.traceback_edit.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        error_layout.addWidget(self.traceback_edit)

        error_group.setLayout(error_layout)
        layout.addWidget(error_group)

        # Context info
        context_group = QGroupBox("Context")
        context_layout = QVBoxLayout()

        context_text = f"""
        <p style='margin: 0; line-height: 1.6;'>
        <b>Last Action:</b> {self.crash_report.last_action}<br>
        <b>App Version:</b> {self.crash_report.app_version}<br>
        <b>Platform:</b> {self.crash_report.platform_name}<br>
        <b>Files Analyzed:</b> {self.crash_report.files_analyzed}<br>
        <b>Total Breaths:</b> {self.crash_report.total_breaths}
        </p>
        """
        context_label = QLabel(context_text)
        context_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        context_layout.addWidget(context_label)

        context_group.setLayout(context_layout)
        layout.addWidget(context_group)

        # User input field - what were you doing?
        user_input_group = QGroupBox("What were you doing? (Optional but helpful)")
        user_input_layout = QVBoxLayout()

        self.user_description = QTextEdit()
        self.user_description.setPlaceholderText(
            "Please describe what you were doing when this happened...\n\n"
            "For example:\n"
            "- Loading a file\n"
            "- Running peak detection\n"
            "- Exporting data\n"
            "- The app became unresponsive"
        )
        self.user_description.setMinimumHeight(80)
        self.user_description.setMaximumHeight(100)
        self.user_description.setStyleSheet("""
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
        user_input_layout.addWidget(self.user_description)

        user_input_group.setLayout(user_input_layout)
        layout.addWidget(user_input_group)

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

        # Dismiss
        self.dismiss_btn = QPushButton("Dismiss")
        self.dismiss_btn.clicked.connect(self.reject)
        self.dismiss_btn.setMinimumHeight(36)
        button_layout.addWidget(self.dismiss_btn)

        layout.addLayout(button_layout)

    def _on_submit_github(self):
        """Open browser to GitHub with pre-filled issue."""
        # Get user's description if provided
        user_desc = self.user_description.toPlainText().strip()

        # Generate URL with user description included
        url = generate_github_issue_url(
            self.crash_report,
            user_description=user_desc if user_desc else None
        )
        webbrowser.open(url)

        # Mark as submitted
        mark_submitted(self.crash_report.report_id)

        self.accept()

    def _on_copy_clipboard(self):
        """Copy full crash report to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self._format_full_report())

        # Show feedback
        self.copy_btn.setText("Copied!")
        self.copy_btn.setEnabled(False)

        # Reset button after delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self._reset_copy_button())

    def _reset_copy_button(self):
        """Reset copy button to original state."""
        self.copy_btn.setText("Copy to Clipboard")
        self.copy_btn.setEnabled(True)

    def _format_full_report(self) -> str:
        """Format the full crash report for clipboard."""
        user_desc = self.user_description.toPlainText().strip()
        user_section = f"\nUser Description:\n{user_desc}\n" if user_desc else ""

        return f"""PhysioMetrics Crash Report
==========================

Error Type: {self.crash_report.error_type}
Message: {self.crash_report.error_message}

Stack Trace:
{self.crash_report.traceback_str}

Context:
- Last Action: {self.crash_report.last_action}
- App Version: {self.crash_report.app_version}
- Platform: {self.crash_report.platform_name}
- Python: {self.crash_report.python_version}
- OS: {self.crash_report.os_version}
- Files Analyzed: {self.crash_report.files_analyzed}
- Total Breaths: {self.crash_report.total_breaths}
{user_section}
Report ID: {self.crash_report.report_id}
Timestamp: {self.crash_report.timestamp}
"""

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
            from ctypes import wintypes

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
            pass  # Not on Windows or API not available

    def _restore_geometry(self):
        """Restore window position from settings."""
        if self.settings.contains("geometry"):
            self.restoreGeometry(self.settings.value("geometry"))

    def closeEvent(self, event):
        """Save geometry on close."""
        self.settings.setValue("geometry", self.saveGeometry())
        super().closeEvent(event)


def show_crash_report_dialog(crash_report: CrashReport, on_startup: bool = False, parent=None) -> bool:
    """
    Show the crash report dialog.

    Args:
        crash_report: The crash report to display
        on_startup: Whether this is shown on app startup (vs immediate crash)
        parent: Parent widget

    Returns:
        True if user submitted the report, False if dismissed
    """
    dialog = CrashReportDialog(crash_report, on_startup=on_startup, parent=parent)
    result = dialog.exec()
    return result == QDialog.DialogCode.Accepted
