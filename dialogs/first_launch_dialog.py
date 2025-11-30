"""
First-launch dialog for PhysioMetrics.

Shows welcome message and telemetry opt-in/opt-out on first run.
"""

import sys
import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QTextBrowser, QGroupBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from version_info import VERSION_STRING


class FirstLaunchDialog(QDialog):
    """
    Welcome dialog shown on first launch.

    Explains telemetry and lets user opt out.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to PhysioMetrics")
        self.setModal(True)
        self.resize(600, 500)

        # Default values (opt-out model - checkboxes pre-checked)
        self.telemetry_enabled = True
        self.crash_reports_enabled = True

        self._setup_ui()
        self._apply_dark_theme()
        self._enable_dark_title_bar()

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
        """Set up the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Header
        header = QLabel(f"<h2 style='color: #2a7fff;'>Welcome to PhysioMetrics {VERSION_STRING}</h2>")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        # Splash image
        image_path = Path(__file__).parent.parent / 'images' / 'plethapp_splash_dark-01.png'
        if image_path.exists():
            image_label = QLabel()
            pixmap = QPixmap(str(image_path))
            scaled_pixmap = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)

        # What's New section
        whats_new = QLabel(
            "<p style='font-size: 11pt; color: #2a7fff; font-weight: bold;'>What's New in v1.0.10:</p>"
            "<ul style='font-size: 10pt; margin-left: 25px;'>"
            "<li><b>Omit Regions:</b> Mark artifacts/noise to exclude from analysis</li>"
            "<li><b>Improved Peak Detection:</b> Better auto-thresholding with baseline filtering</li>"
            "<li><b>Draggable Threshold:</b> Click and drag the red line to adjust threshold</li>"
            "<li><b>Fine-tuned Controls:</b> Prominence spinbox with 0.01 increments</li>"
            "<li><b>Pulse Analysis:</b> 25ms stimulus + PRC (⚠ in development)</li>"
            "</ul>"
        )
        whats_new.setWordWrap(True)
        layout.addWidget(whats_new)

        # Telemetry explanation group
        telemetry_group = QGroupBox("Help Improve PhysioMetrics")
        telemetry_layout = QVBoxLayout()

        # Intro text
        intro_label = QLabel(
            "<p>Share anonymous usage statistics to help improve PhysioMetrics. "
            "Your data is completely anonymous and helps prioritize features and fix bugs.</p>"
        )
        intro_label.setWordWrap(True)
        telemetry_layout.addWidget(intro_label)

        # Telemetry checkbox
        self.telemetry_checkbox = QCheckBox("Share anonymous usage data")
        self.telemetry_checkbox.setChecked(True)  # Opt-out model
        self.telemetry_checkbox.setStyleSheet("font-size: 10pt;")
        telemetry_layout.addWidget(self.telemetry_checkbox)

        # What's collected
        collected_label = QLabel(
            "<ul style='margin-top: 5px; margin-left: 20px;'>"
            "<li>Number of files analyzed (no file names or paths)</li>"
            "<li>Features used (GMM clustering, manual editing, etc.)</li>"
            "<li>App version, OS, and Python version</li>"
            "<li>Anonymous user ID (random UUID)</li>"
            "</ul>"
        )
        telemetry_layout.addWidget(collected_label)

        # Crash reports checkbox
        self.crash_reports_checkbox = QCheckBox("Send crash reports")
        self.crash_reports_checkbox.setChecked(True)  # Opt-out model
        self.crash_reports_checkbox.setStyleSheet("font-size: 10pt;")
        telemetry_layout.addWidget(self.crash_reports_checkbox)

        crash_label = QLabel(
            "<ul style='margin-top: 5px; margin-left: 20px;'>"
            "<li>Error stack traces to help fix bugs</li>"
            "<li>No personal or experimental data</li>"
            "</ul>"
        )
        telemetry_layout.addWidget(crash_label)

        # Learn more link
        learn_more_label = QLabel('<a href="#details" style="color: #2a7fff; font-size: 10pt;">Learn more about what data is collected...</a>')
        learn_more_label.setOpenExternalLinks(False)
        learn_more_label.linkActivated.connect(self._show_details)
        learn_more_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        telemetry_layout.addWidget(learn_more_label)

        telemetry_group.setLayout(telemetry_layout)
        telemetry_group.setMaximumWidth(500)  # Make the box smaller
        layout.addWidget(telemetry_group, alignment=Qt.AlignmentFlag.AlignCenter)

        # Note about changing preferences
        note_label = QLabel(
            "<p style='font-size: 9pt; color: #888;'>"
            "You can change these preferences anytime in Help → About."
            "</p>"
        )
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(note_label)

        # Spacer
        layout.addStretch()

        # Continue button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        continue_btn = QPushButton("Continue")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self._on_continue)
        continue_btn.setMinimumWidth(120)
        continue_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a7fff;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4090ff;
            }
        """)
        button_layout.addWidget(continue_btn)

        layout.addLayout(button_layout)

    def _apply_dark_theme(self):
        """Apply dark theme styling."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QGroupBox {
                background-color: #2a2a2a;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #2a7fff;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 10px;
                height: 10px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #2a2a2a;
                border: 1px solid #555;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border: 1px solid #2a7fff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #e0e0e0;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)

    def _on_continue(self):
        """Handle Continue button click."""
        # Save checkbox states
        self.telemetry_enabled = self.telemetry_checkbox.isChecked()
        self.crash_reports_enabled = self.crash_reports_checkbox.isChecked()

        # Close dialog with accept
        self.accept()

    def _show_details(self):
        """Show detailed information about telemetry."""
        details_dialog = QDialog(self)
        details_dialog.setWindowTitle("Telemetry Details")
        details_dialog.resize(500, 450)

        layout = QVBoxLayout(details_dialog)

        # Text browser with details
        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml("""
            <h3 style="color: #2a7fff;">What Data is Collected?</h3>

            <h4>Usage Data</h4>
            <p>When you use PhysioMetrics, we collect anonymous usage statistics:</p>
            <ul>
                <li><b>File types:</b> Whether you load ABF, SMRX, or EDF files (not file names)</li>
                <li><b>Features used:</b> Which tools you use (GMM, manual editing, spectral analysis)</li>
                <li><b>Export types:</b> What you export (PDF, CSV, NPZ)</li>
                <li><b>Session duration:</b> How long you use the app</li>
                <li><b>System info:</b> OS, Python version, PhysioMetrics version</li>
            </ul>

            <h4>Crash Reports</h4>
            <p>If the app crashes, we collect:</p>
            <ul>
                <li><b>Error messages:</b> What went wrong</li>
                <li><b>Stack traces:</b> Where the error occurred in the code</li>
                <li><b>No data:</b> Your files and data are NEVER included</li>
            </ul>

            <h4>Anonymous User ID</h4>
            <p>A random UUID (like "a3f2e8c9-4b7d-...") is generated once and stored on your computer.
            This lets us count unique users without knowing who you are.</p>

            <p style="color: #FFD700; font-weight: bold; margin-top: 15px;">What we NEVER collect:</p>
            <ul>
                <li>File names, paths, or directory structure</li>
                <li>Animal metadata (strain, virus, injection site)</li>
                <li>Actual breathing data (frequencies, amplitudes, metrics)</li>
                <li>Your name, email, or institution</li>
                <li>Computer name or network information</li>
            </ul>

            <h4>Why Opt-Out by Default?</h4>
            <p>PhysioMetrics uses an "opt-out" model (checkboxes pre-checked) because:</p>
            <ul>
                <li>This is academic research software (not commercial spyware)</li>
                <li>All data is completely anonymous</li>
                <li>Usage statistics help prioritize bug fixes and features</li>
                <li>You can easily disable it anytime</li>
            </ul>

            <p style="color: #FFD700; font-weight: bold;">You have complete control.</p>
            <p>You can disable telemetry at any time in Help → About.</p>

            <h4>Precedents</h4>
            <p>Many trusted tools use opt-out telemetry:</p>
            <ul>
                <li>VS Code (Microsoft)</li>
                <li>Anaconda Navigator</li>
                <li>JupyterLab extensions</li>
            </ul>
        """)
        layout.addWidget(browser)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(details_dialog.close)
        layout.addWidget(close_btn)

        # Apply dark theme
        details_dialog.setStyleSheet(self.styleSheet())

        details_dialog.exec()

    def get_preferences(self):
        """
        Get user's telemetry preferences.

        Returns:
            tuple: (telemetry_enabled, crash_reports_enabled)
        """
        return (self.telemetry_enabled, self.crash_reports_enabled)
