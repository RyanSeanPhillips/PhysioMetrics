"""
Shared pytest fixtures for PhysioMetrics tests.

pytest-qt provides the `qtbot` fixture automatically.
We add fixtures for MainWindow and test data.

Dialog handling: Instead of monkeypatching dialogs away, we use a
DialogWatcher that detects popup dialogs, logs them (so we can see
what the user would see), and auto-responds after a brief delay.

Test data files:
    tests/data/26402007.abf  — 7-channel ABF (pleth + EKG + stim)
        Protocol: 1-chamber 15s continuous opto stim
        10 sweeps, 880s each, 10kHz
        Ch 1 (IN 1): pleth (breathing)
        Ch 6 (IN 7): EKG (heart rate)
        Ch 0 (IN 0): stim trigger (IN 4, IN 5 are duplicates)
        Ch 2,3:      noise/empty
    examples/25121004.abf    — legacy test file (used by PLETHAPP_TESTING auto-load)
"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import pytest

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set working directory to project root (MainWindow expects relative paths)
os.chdir(ROOT)

# Suppress telemetry / crash reporting during tests
os.environ["PLETHAPP_TESTING"] = "1"


# ── Test data definitions ────────────────────────────────────────

@dataclass
class TestFileInfo:
    """Metadata about a test ABF file and its known channel roles."""
    path: Path
    pleth_channels: List[str] = field(default_factory=list)     # channel names for breathing
    ekg_channels: List[str] = field(default_factory=list)       # channel names for EKG
    stim_channels: List[str] = field(default_factory=list)      # channel names for stim trigger
    protocol: str = ""
    n_sweeps: int = 0
    sample_rate: int = 0


TEST_DATA_DIR = ROOT / "tests" / "data"
EXAMPLES_DIR = ROOT / "examples"

# Primary test file: single-chamber pleth + EKG + stim
MULTI_CHANNEL_ABF = TestFileInfo(
    path=TEST_DATA_DIR / "26402007.abf",
    pleth_channels=["IN 1"],                # pleth (breathing)
    ekg_channels=["IN 7"],                  # EKG (heart rate)
    stim_channels=["IN 0"],                 # stim trigger (IN 4, IN 5 are duplicates)
    protocol="1-chamber 15s continuous opto stim",
    n_sweeps=10,
    sample_rate=10000,
)

# Legacy test file (smaller, used by PLETHAPP_TESTING auto-load)
LEGACY_ABF = TestFileInfo(
    path=EXAMPLES_DIR / "25121004.abf",
    pleth_channels=["IN 0"],
    stim_channels=["IN 7"],
    protocol="30Hz stim",
    n_sweeps=1,
    sample_rate=10000,
)


@pytest.fixture
def multi_channel_abf():
    """Multi-channel ABF with pleth + EKG + stim; skips if missing."""
    if not MULTI_CHANNEL_ABF.path.exists():
        pytest.skip(f"Test file not found: {MULTI_CHANNEL_ABF.path}")
    return MULTI_CHANNEL_ABF


@pytest.fixture
def sample_abf():
    """Path to legacy sample ABF file; skips if missing."""
    if not LEGACY_ABF.path.exists():
        pytest.skip(f"Sample ABF not found: {LEGACY_ABF.path}")
    return LEGACY_ABF.path


@pytest.fixture
def tmp_output(tmp_path):
    """Temporary output directory."""
    return tmp_path


# ── Dialog Watcher ───────────────────────────────────────────────

class DialogWatcher:
    """
    Watches for modal dialogs (QMessageBox, QDialog) and auto-responds.

    Logs every dialog it sees so tests can assert on what was shown.
    Uses a QTimer to periodically check for active modal widgets.
    """

    def __init__(self):
        self.seen_dialogs = []  # list of {'type', 'title', 'text', 'response'}
        self._timer = None

    def start(self):
        from PyQt6.QtCore import QTimer
        self._timer = QTimer()
        self._timer.timeout.connect(self._check_for_dialogs)
        self._timer.start(100)  # check every 100ms

    def stop(self):
        if self._timer:
            self._timer.stop()
            self._timer = None

    def clear(self):
        """Clear seen dialogs (useful between test phases)."""
        self.seen_dialogs.clear()

    # Dialog titles that should be LEFT ALONE (progress dialogs, they close themselves)
    _IGNORE_TITLES = {
        'opening abf file', 'opening smrx file', 'opening edf file',
        'opening mat file', 'loading...', 'loading physiometrics',
        'saving session...', 'computing z-score', 'preparing data',
        'physiometrics',  # generic progress dialog title
    }

    # QMessageBox titles where we should click YES (proceed/continue)
    _ACCEPT_TITLES = {
        'data validation', 'validation warning', 'overwrite',
        'batch analyze', 'clear event markers',
    }

    # QDialog titles where we should ACCEPT (click OK/Continue)
    _DIALOG_ACCEPT_TITLES = {
        'save options', 'loading physiometrics session',
    }

    def _check_for_dialogs(self):
        from PyQt6.QtWidgets import (QApplication, QMessageBox, QDialog,
                                      QProgressDialog, QFileDialog)

        active = QApplication.activeModalWidget()
        if active is None:
            return

        title = active.windowTitle()
        title_lower = title.lower()

        # Skip progress dialogs — they close themselves when the operation finishes
        if isinstance(active, QProgressDialog):
            # Only log once per title
            if not any(d['title'] == title for d in self.seen_dialogs[-3:]):
                print(f"\n[DIALOG] QProgressDialog: '{title}' (ignoring — will close itself)")
                self.seen_dialogs.append({
                    'type': 'QProgressDialog',
                    'title': title,
                    'text': '',
                    'response': 'ignored',
                })
            return

        # Skip file dialogs — these are handled by test mocks
        if isinstance(active, QFileDialog):
            print(f"\n[DIALOG] QFileDialog: '{title}' (cancelling)")
            self.seen_dialogs.append({
                'type': 'QFileDialog',
                'title': title,
                'text': '',
                'response': 'cancel',
            })
            active.reject()
            return

        if isinstance(active, QMessageBox):
            text = active.text()
            detail = active.informativeText()
            full_text = f"{text}\n{detail}".strip() if detail else text

            print(f"\n[DIALOG] QMessageBox: '{title}'")
            print(f"  Text: {full_text[:200]}")

            self.seen_dialogs.append({
                'type': 'QMessageBox',
                'title': title,
                'text': full_text,
            })

            no_btn = active.button(QMessageBox.StandardButton.No)
            yes_btn = active.button(QMessageBox.StandardButton.Yes)
            ok_btn = active.button(QMessageBox.StandardButton.Ok)
            cancel_btn = active.button(QMessageBox.StandardButton.Cancel)

            # Decide response based on title/content
            should_accept = any(kw in title_lower for kw in self._ACCEPT_TITLES)

            if should_accept and yes_btn:
                print(f"  -> Auto-clicking 'Yes'")
                self.seen_dialogs[-1]['response'] = 'Yes'
                yes_btn.click()
            elif no_btn:
                print(f"  -> Auto-clicking 'No'")
                self.seen_dialogs[-1]['response'] = 'No'
                no_btn.click()
            elif ok_btn:
                print(f"  -> Auto-clicking 'OK'")
                self.seen_dialogs[-1]['response'] = 'OK'
                ok_btn.click()
            elif cancel_btn:
                print(f"  -> Auto-clicking 'Cancel'")
                self.seen_dialogs[-1]['response'] = 'Cancel'
                cancel_btn.click()
            else:
                print(f"  -> Force-closing (no standard buttons found)")
                self.seen_dialogs[-1]['response'] = 'force_close'
                active.close()

        elif isinstance(active, QDialog):
            print(f"\n[DIALOG] QDialog: '{title}'")
            self.seen_dialogs.append({
                'type': 'QDialog',
                'title': title,
                'text': '',
            })

            # Check if this is a dialog we should accept vs reject
            should_ignore = any(kw in title_lower for kw in self._IGNORE_TITLES)
            should_accept = any(kw in title_lower for kw in self._DIALOG_ACCEPT_TITLES)

            if should_ignore:
                print(f"  -> Ignoring (progress/loading dialog)")
                self.seen_dialogs[-1]['response'] = 'ignored'
                # Don't reject — let it close naturally
            elif should_accept:
                print(f"  -> Auto-accepting")
                self.seen_dialogs[-1]['response'] = 'accept'
                active.accept()
            else:
                print(f"  -> Auto-rejecting (cancel)")
                self.seen_dialogs[-1]['response'] = 'reject'
                active.reject()
            active.reject()


# ── Helper: load file into running MainWindow ────────────────────

def load_file_and_wait(main_window, file_path, channel_name=None, timeout=30):
    """
    Load a file into MainWindow and wait for data to be ready.

    Args:
        main_window: The MainWindow instance
        file_path: Path to the ABF file
        channel_name: Optional channel name to select after loading
        timeout: Max seconds to wait for load
    """
    from PyQt6.QtWidgets import QApplication
    import time

    main_window.load_file(Path(file_path))

    deadline = time.time() + timeout
    while time.time() < deadline:
        QApplication.processEvents()
        if (main_window.state.sweeps or
                main_window.state.in_path is not None):
            for _ in range(20):
                QApplication.processEvents()
                time.sleep(0.05)
            break
        time.sleep(0.1)

    if channel_name:
        # Find and select the channel in AnalyzeChanSelect
        combo = getattr(main_window, "AnalyzeChanSelect", None)
        if combo:
            for i in range(combo.count()):
                if channel_name in combo.itemText(i):
                    combo.setCurrentIndex(i)
                    QApplication.processEvents()
                    break


# ── GUI fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def app(qapp):
    """Session-scoped QApplication (alias for pytest-qt's qapp)."""
    return qapp


_main_window_instance = None
_dialog_watcher = None


@pytest.fixture(scope="session")
def dialog_watcher():
    """Access the dialog watcher to check what dialogs were shown."""
    return _dialog_watcher


@pytest.fixture(scope="session")
def main_window(qapp):
    """
    Create a single MainWindow instance for the entire test session.

    A DialogWatcher runs in the background to detect and auto-respond
    to any popup dialogs, logging them for inspection.

    PLETHAPP_TESTING=1 makes it auto-load examples/25121004.abf.
    """
    global _main_window_instance, _dialog_watcher

    if _main_window_instance is None:
        from PyQt6.QtWidgets import QApplication
        from main import MainWindow

        # Start dialog watcher BEFORE creating the window
        _dialog_watcher = DialogWatcher()
        _dialog_watcher.start()

        _main_window_instance = MainWindow()
        _main_window_instance.show()

        # Process events until the auto-loaded file is ready
        import time
        deadline = time.time() + 30  # max 30s wait
        while time.time() < deadline:
            QApplication.processEvents()
            if (_main_window_instance.state.sweeps or
                    _main_window_instance.state.in_path is not None):
                # Give a bit more time for post-load setup
                for _ in range(20):
                    QApplication.processEvents()
                    time.sleep(0.05)
                break
            time.sleep(0.1)

    return _main_window_instance
