"""
Testing utilities — screenshot comparison, dialog inspection, and timing helpers.

These utilities enable:
1. Visual regression: compare widget screenshots against baselines
2. Dialog inspection: read dialog text, buttons, verify content makes sense
3. Benchmarking: time operations and report performance

Usage in tests:
    from test_utils import capture_widget, compare_to_baseline, inspect_dialog, time_operation
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES_DIR = ROOT / "tests" / "output" / "baselines"
SCREENSHOTS_DIR = ROOT / "tests" / "output" / "screenshots"


# ── Screenshot & Visual Regression ───────────────────────────────

def capture_widget(widget, name: str, save_dir: Optional[Path] = None) -> Path:
    """
    Capture a screenshot of a QWidget to PNG.

    Args:
        widget: Any QWidget (MainWindow, dialog, plot, etc.)
        name: Filename stem (e.g. 'main_window', 'peak_dialog')
        save_dir: Directory to save to (default: tests/output/screenshots/)

    Returns:
        Path to the saved PNG
    """
    save_dir = save_dir or SCREENSHOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    pixmap = widget.grab()
    out_path = save_dir / f"{name}.png"
    pixmap.save(str(out_path), "PNG")
    return out_path


def compare_to_baseline(current_path: Path, name: str,
                        tolerance: float = 0.02) -> dict:
    """
    Compare a screenshot against a saved baseline.

    On first run (no baseline exists), saves the current as baseline.
    On subsequent runs, computes pixel-level difference.

    Args:
        current_path: Path to current screenshot
        name: Baseline name (matches filename in baselines/)
        tolerance: Max fraction of differing pixels (0.02 = 2%)

    Returns:
        dict with keys: 'match', 'diff_percent', 'baseline_created', 'diff_path'
    """
    from PIL import Image

    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = BASELINES_DIR / f"{name}.png"

    if not baseline_path.exists():
        # First run — save as baseline
        import shutil
        shutil.copy2(current_path, baseline_path)
        return {
            'match': True,
            'diff_percent': 0.0,
            'baseline_created': True,
            'diff_path': None,
            'message': f"Baseline created: {baseline_path.name}"
        }

    # Load both images
    baseline = np.array(Image.open(baseline_path).convert('RGB'))
    current = np.array(Image.open(current_path).convert('RGB'))

    # Handle size mismatch (window may have resized)
    if baseline.shape != current.shape:
        return {
            'match': False,
            'diff_percent': 100.0,
            'baseline_created': False,
            'diff_path': None,
            'message': f"Size mismatch: baseline={baseline.shape}, current={current.shape}"
        }

    # Pixel difference (per-channel, threshold of 10 intensity levels)
    diff = np.abs(baseline.astype(int) - current.astype(int))
    changed_pixels = np.any(diff > 10, axis=2)  # any channel differs by >10
    diff_percent = 100.0 * np.mean(changed_pixels)

    # Save diff image if there are differences
    diff_path = None
    if diff_percent > 0:
        diff_img = np.zeros_like(current)
        diff_img[changed_pixels] = [255, 0, 0]  # red for changed pixels
        diff_img[~changed_pixels] = current[~changed_pixels] // 2  # dim unchanged
        diff_path = SCREENSHOTS_DIR / f"{name}_diff.png"
        Image.fromarray(diff_img.astype(np.uint8)).save(str(diff_path))

    return {
        'match': diff_percent <= tolerance * 100,
        'diff_percent': diff_percent,
        'baseline_created': False,
        'diff_path': diff_path,
        'message': f"{diff_percent:.1f}% pixels changed (tolerance: {tolerance*100:.0f}%)"
    }


def update_baseline(name: str):
    """
    Replace a baseline with the current screenshot.
    Call this when you've intentionally changed the UI.
    """
    current = SCREENSHOTS_DIR / f"{name}.png"
    baseline = BASELINES_DIR / f"{name}.png"
    if current.exists():
        import shutil
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(current, baseline)
        return True
    return False


# ── Dialog Inspection ────────────────────────────────────────────

def inspect_dialog(dialog) -> dict:
    """
    Extract all readable information from a QDialog or QMessageBox.

    Returns dict with: title, text, buttons, widgets, size, etc.
    """
    from PyQt6.QtWidgets import (QMessageBox, QDialog, QPushButton, QLabel,
                                  QLineEdit, QComboBox, QCheckBox, QSpinBox,
                                  QDoubleSpinBox, QTextEdit, QPlainTextEdit)

    info = {
        'class': type(dialog).__name__,
        'title': dialog.windowTitle(),
        'size': (dialog.width(), dialog.height()),
        'visible': dialog.isVisible(),
    }

    if isinstance(dialog, QMessageBox):
        info['text'] = dialog.text()
        info['informative_text'] = dialog.informativeText()
        info['detailed_text'] = dialog.detailedText()
        info['icon'] = str(dialog.icon())
        info['buttons'] = []
        for btn in dialog.buttons():
            role = dialog.buttonRole(btn)
            info['buttons'].append({
                'text': btn.text(),
                'role': str(role),
            })
    else:
        # Generic QDialog — find all child widgets
        info['labels'] = []
        info['buttons'] = []
        info['inputs'] = []
        info['checkboxes'] = []
        info['combos'] = []

        for label in dialog.findChildren(QLabel):
            text = label.text()
            if text and text.strip():
                info['labels'].append(text[:200])

        for btn in dialog.findChildren(QPushButton):
            info['buttons'].append({
                'text': btn.text(),
                'enabled': btn.isEnabled(),
                'visible': btn.isVisible(),
            })

        for edit in dialog.findChildren(QLineEdit):
            info['inputs'].append({
                'text': edit.text(),
                'placeholder': edit.placeholderText(),
                'name': edit.objectName(),
            })

        for cb in dialog.findChildren(QCheckBox):
            info['checkboxes'].append({
                'text': cb.text(),
                'checked': cb.isChecked(),
                'name': cb.objectName(),
            })

        for combo in dialog.findChildren(QComboBox):
            items = [combo.itemText(i) for i in range(combo.count())]
            info['combos'].append({
                'name': combo.objectName(),
                'current': combo.currentText(),
                'items': items[:20],
            })

    return info


def inspect_all_dialogs(dialog_watcher) -> list:
    """Get detailed info about all dialogs the watcher has seen."""
    return list(dialog_watcher.seen_dialogs)


# ── Benchmarking ─────────────────────────────────────────────────

class Timer:
    """
    Simple context manager for timing operations.

    Usage:
        with Timer("peak detection") as t:
            main_window._apply_peak_detection()
        print(t.result)  # "peak detection: 1.23s"
    """
    def __init__(self, label: str = ""):
        self.label = label
        self.start_time = None
        self.elapsed = None
        self.result = ""

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        self.result = f"{self.label}: {self.elapsed:.3f}s" if self.label else f"{self.elapsed:.3f}s"


class ResourceMonitor:
    """
    Monitors CPU and memory usage during an operation.

    Usage:
        with ResourceMonitor("peak detection") as mon:
            main_window._apply_peak_detection()
        print(mon.result)
    """

    def __init__(self, label: str = ""):
        import psutil
        self.label = label
        self.process = psutil.Process()
        self.start_mem = None
        self.end_mem = None
        self.peak_mem = None
        self.cpu_percent = None
        self.elapsed = None
        self.result = ""
        self._samples = []
        self._timer = None

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_mem = self.start_mem
        self.process.cpu_percent()  # prime the measurement
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import psutil
        self.elapsed = time.perf_counter() - self._start_time
        self.end_mem = self.process.memory_info().rss / (1024 * 1024)
        self.cpu_percent = self.process.cpu_percent()
        self.peak_mem = max(self.peak_mem, self.end_mem)
        mem_delta = self.end_mem - self.start_mem

        self.result = (
            f"{self.label}: {self.elapsed:.3f}s, "
            f"mem: {self.start_mem:.0f}->{self.end_mem:.0f}MB "
            f"(delta: {mem_delta:+.1f}MB), "
            f"CPU: {self.cpu_percent:.0f}%"
        )


def get_process_info() -> dict:
    """Get current process memory and CPU stats."""
    import psutil
    proc = psutil.Process()
    mem = proc.memory_info()
    return {
        'rss_mb': mem.rss / (1024 * 1024),
        'vms_mb': mem.vms / (1024 * 1024),
        'cpu_percent': proc.cpu_percent(),
        'num_threads': proc.num_threads(),
    }


def benchmark_operation(func, *args, n_runs=3, label="", **kwargs) -> dict:
    """
    Run a function multiple times and report timing stats.

    Returns dict with: mean, min, max, std, runs, label
    """
    from PyQt6.QtWidgets import QApplication

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        QApplication.processEvents()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    return {
        'label': label or func.__name__,
        'mean': float(np.mean(times)),
        'min': float(np.min(times)),
        'max': float(np.max(times)),
        'std': float(np.std(times)),
        'runs': n_runs,
        'times': times.tolist(),
    }
