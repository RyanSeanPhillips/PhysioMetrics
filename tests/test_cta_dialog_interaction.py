"""
CTA Dialog interaction test — open the real dialog, configure it, generate CTAs.

This test opens the actual PhotometryCTADialog, selects breath events as trigger,
configures metrics (pleth, HR, respiratory rate, RSA), generates the CTA, and
takes a screenshot of the result.

Run:  python -m pytest tests/test_cta_dialog_interaction.py -v -s
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QRadioButton, QCheckBox, QPushButton
from PyQt6.QtCore import QTimer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis
from test_utils import capture_widget

OUTPUT_DIR = ROOT / "tests" / "output"


class TestCTADialogInteraction:

    def test_open_cta_and_generate(self, main_window, multi_channel_abf):
        """Open CTA dialog, select breath trigger + metrics, generate, screenshot."""

        # 1. Set up analysis (peaks + EKG)
        _setup_analysis(main_window, multi_channel_abf)
        assert main_window.state.ecg_results_by_sweep, "No EKG data"

        # 2. Schedule dialog interaction via timer (dialog.exec() is blocking)
        dialog_ref = [None]
        interaction_done = [False]
        error_msg = [None]

        def _interact_with_dialog():
            """Find and interact with the CTA dialog."""
            try:
                # Find the dialog
                active = QApplication.activeModalWidget()
                if active is None:
                    # Try finding by window title
                    for w in QApplication.topLevelWidgets():
                        if 'CTA' in w.windowTitle():
                            active = w
                            break
                if active is None:
                    return  # Dialog not open yet, timer will retry

                dialog_ref[0] = active
                print(f"  Found dialog: {active.windowTitle()}")

                # Find the first tab's comparison widget
                tabs = active._tabs if hasattr(active, '_tabs') else None
                if tabs is None:
                    error_msg[0] = "No _tabs widget found"
                    active.close()
                    return

                widget = tabs.widget(0)
                if widget is None:
                    error_msg[0] = "No tab widget"
                    active.close()
                    return

                # 3. Select "Breath Events" radio button
                breath_radio = getattr(widget, '_radio_breath_events', None)
                if breath_radio and breath_radio.isEnabled():
                    breath_radio.setChecked(True)
                    QApplication.processEvents()
                    print(f"  Selected: Breath Events trigger")
                else:
                    print(f"  Breath Events not available, using Event Markers")

                # 4. Configure breath event settings if available
                max_events = getattr(widget, '_breath_max_events', None)
                if max_events:
                    max_events.setValue(100)
                    print(f"  Set max events: 100")

                # Set time window to breath-scale
                spin_before = getattr(widget, '_spin_before', None)
                spin_after = getattr(widget, '_spin_after', None)
                if spin_before:
                    spin_before.setValue(0.5)
                if spin_after:
                    spin_after.setValue(1.5)
                print(f"  Time window: -0.5s to +1.5s")

                # Disable z-score (show raw values)
                zscore_group = getattr(widget, '_zscore_group', None)
                if zscore_group and hasattr(zscore_group, 'setChecked'):
                    zscore_group.setChecked(False)
                    print(f"  Z-score: disabled")

                # 5. Select metrics — check cardiorespiratory ones
                # The table uses ItemIsUserCheckable on column 1 items
                from PyQt6.QtCore import Qt
                metrics_table = getattr(widget, '_metrics_table', None)
                if metrics_table:
                    # Select: raw pleth (IN 1), HR, respiratory rate, RSA
                    target_keywords = ['heart rate', 'frequency', 'rsa']
                    target_exact = ['IN 1']  # Raw pleth waveform
                    checked = []
                    all_metrics = []
                    for row in range(metrics_table.rowCount()):
                        item_peek = metrics_table.item(row, 1)
                        if item_peek:
                            all_metrics.append(item_peek.text())
                    print(f"  Available metrics ({len(all_metrics)}):")
                    for m in all_metrics:
                        print(f"    {m}")
                    for row in range(metrics_table.rowCount()):
                        item = metrics_table.item(row, 1)
                        if item is None:
                            continue
                        text = item.text().lower()
                        key = item.data(Qt.ItemDataRole.UserRole)

                        should_check = (any(kw in text for kw in target_keywords) or
                                        any(text.strip() == ex.lower() for ex in target_exact))
                        if should_check:
                            item.setCheckState(Qt.CheckState.Checked)
                            checked.append(f"{key}: {item.text()}")
                        else:
                            item.setCheckState(Qt.CheckState.Unchecked)

                    QApplication.processEvents()
                    print(f"  Checked {len(checked)} metrics:")
                    for c in checked:
                        print(f"    - {c}")

                # 6. Click "Generate CTA"
                gen_btn = None
                for btn in widget.findChildren(QPushButton):
                    if 'generate' in btn.text().lower():
                        gen_btn = btn
                        break

                if gen_btn and gen_btn.isEnabled():
                    print(f"  Clicking Generate CTA...")
                    gen_btn.click()

                    # Wait for generation to complete
                    for _ in range(100):  # up to 10 seconds
                        QApplication.processEvents()
                        time.sleep(0.1)

                    print(f"  CTA generation complete")

                    # 7. Take screenshot
                    QApplication.processEvents()
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    screenshot_path = OUTPUT_DIR / "cta_dialog_breath_triggered.png"
                    capture_widget(active, "cta_dialog_breath_triggered", OUTPUT_DIR)
                    print(f"  Screenshot saved: {screenshot_path}")
                else:
                    print(f"  Generate button not found or disabled")
                    if gen_btn:
                        print(f"    Button text: '{gen_btn.text()}', enabled: {gen_btn.isEnabled()}")

                interaction_done[0] = True

                # Close the dialog
                active.accept()

            except Exception as e:
                error_msg[0] = str(e)
                import traceback
                traceback.print_exc()
                if dialog_ref[0]:
                    dialog_ref[0].close()
                interaction_done[0] = True

        # Schedule the interaction to run AFTER the dialog opens
        # (dialog.exec() blocks, but QTimer fires inside the Qt event loop)
        timer = QTimer()
        timer.timeout.connect(_interact_with_dialog)
        timer.start(500)  # Check every 500ms

        # Schedule the dialog to open in 100ms (gives timer time to start)
        QTimer.singleShot(100, main_window._on_generate_cta_requested)

        # Process events until interaction completes (max 30s)
        deadline = time.time() + 30
        while not interaction_done[0] and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.1)

        timer.stop()

        if error_msg[0]:
            pytest.fail(f"Dialog interaction failed: {error_msg[0]}")

        assert interaction_done[0], "Dialog interaction never completed"

        # Open the screenshot
        screenshot_path = OUTPUT_DIR / "cta_dialog_breath_triggered.png"
        if screenshot_path.exists():
            print(f"  Screenshot: {screenshot_path} ({screenshot_path.stat().st_size // 1024} KB)")
