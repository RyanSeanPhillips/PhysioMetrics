"""
Log viewer dialog — shows startup log + live stdout capture.

Opens with Ctrl+Shift+L. Non-blocking dialog for debugging.
"""

import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPlainTextEdit,
    QPushButton, QApplication,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QTextCursor


class LogViewerDialog(QDialog):
    """Non-blocking log viewer that shows startup.log + live stdout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PhysioMetrics — Log Viewer")
        self.setMinimumSize(700, 400)
        self.resize(900, 500)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )

        self._build_ui()
        self._load_startup_log()
        self._install_stdout_hook()

        # Poll for new log entries periodically
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._flush_pending)
        self._poll_timer.start(250)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Consolas", 9))
        self._text.setMaximumBlockCount(5000)
        self._text.setStyleSheet(
            "QPlainTextEdit { background-color: #1a1a1a; color: #d4d4d4; "
            "border: 1px solid #333; }"
        )
        layout.addWidget(self._text)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(self._copy_all)
        btn_row.addWidget(copy_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._text.clear)
        btn_row.addWidget(clear_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)

        layout.addLayout(btn_row)

    def _load_startup_log(self):
        """Load the startup.log file contents."""
        if sys.platform == 'win32':
            log_path = Path(__import__('os').environ.get('APPDATA', '')) / 'PhysioMetrics' / 'startup.log'
        else:
            log_path = Path.home() / '.config' / 'PhysioMetrics' / 'startup.log'

        if log_path.exists():
            try:
                text = log_path.read_text(encoding='utf-8', errors='replace')
                self._text.setPlainText(text)
                self._text.moveCursor(QTextCursor.MoveOperation.End)
            except Exception as e:
                self._text.setPlainText(f"Could not read startup.log: {e}")
        else:
            self._text.setPlainText("(No startup.log found)")

        self._text.appendPlainText("\n--- Live output ---\n")

    def _install_stdout_hook(self):
        """Tee stdout/stderr to the log viewer."""
        self._pending_lines = []
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        viewer = self

        class TeeStream:
            def __init__(self, original):
                self._original = original

            def write(self, text):
                if self._original:
                    try:
                        self._original.write(text)
                    except Exception:
                        pass
                if text.strip():
                    viewer._pending_lines.append(text.rstrip('\n'))

            def flush(self):
                if self._original:
                    try:
                        self._original.flush()
                    except Exception:
                        pass

        sys.stdout = TeeStream(self._original_stdout)
        sys.stderr = TeeStream(self._original_stderr)

    def _flush_pending(self):
        """Append any pending lines to the text widget."""
        if not self._pending_lines:
            return
        lines = self._pending_lines[:]
        self._pending_lines.clear()
        for line in lines:
            self._text.appendPlainText(line)

    def _copy_all(self):
        """Copy all log text to clipboard."""
        QApplication.clipboard().setText(self._text.toPlainText())

    def closeEvent(self, event):
        """Restore stdout/stderr on close."""
        self._poll_timer.stop()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        super().closeEvent(event)
