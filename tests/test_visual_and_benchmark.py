"""
Visual regression, dialog inspection, and benchmarking tests.

These tests:
1. Take screenshots of the app and compare against baselines
2. Open dialogs, read their text/buttons, verify content makes sense
3. Time key operations and report performance

First run creates baselines. Subsequent runs compare against them.

Run:  python -m pytest tests/test_visual_and_benchmark.py -v -s
"""

import sys
import json
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import QTimer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_utils import (capture_widget, compare_to_baseline, inspect_dialog,
                         Timer, benchmark_operation, ResourceMonitor,
                         get_process_info)
from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis, _select_channel


# ── Visual Regression Tests ──────────────────────────────────────


class TestVisualRegression:
    """Screenshot comparison tests — detect unintended UI changes."""

    def test_main_window_screenshot(self, main_window):
        """Capture main window and compare to baseline."""
        QApplication.processEvents()
        path = capture_widget(main_window, "main_window")
        result = compare_to_baseline(path, "main_window", tolerance=0.10)
        print(f"  {result['message']}")
        if result['diff_path']:
            print(f"  Diff image: {result['diff_path']}")
        # Don't assert on first run (baseline creation)
        if not result['baseline_created']:
            assert result['match'], \
                f"Visual regression: {result['diff_percent']:.1f}% pixels changed"

    def test_plot_with_data_screenshot(self, main_window, multi_channel_abf):
        """Capture plot area after loading data + peaks."""
        _setup_analysis(main_window, multi_channel_abf)
        QApplication.processEvents()

        # Capture just the plot area
        plot_widget = main_window.plot_host
        path = capture_widget(plot_widget, "plot_with_peaks")
        result = compare_to_baseline(path, "plot_with_peaks", tolerance=0.15)
        print(f"  {result['message']}")
        if not result['baseline_created']:
            # Plot screenshots have more variance (rendering differences)
            # so use a higher tolerance
            assert result['match'], \
                f"Plot regression: {result['diff_percent']:.1f}% changed"

    def test_each_tab_screenshot(self, main_window):
        """Capture each main tab and compare to baseline."""
        tabs = main_window.Tabs
        original = tabs.currentIndex()

        for i in range(tabs.count()):
            tab_name = tabs.tabText(i).replace(" ", "_").replace("/", "_").lower()
            tabs.setCurrentIndex(i)
            QApplication.processEvents()

            path = capture_widget(main_window, f"tab_{tab_name}")
            result = compare_to_baseline(path, f"tab_{tab_name}")
            print(f"  Tab '{tabs.tabText(i)}': {result['message']}")

        tabs.setCurrentIndex(original)
        QApplication.processEvents()


# ── Dialog Inspection Tests ──────────────────────────────────────


class TestDialogInspection:
    """Open dialogs, inspect their content, verify text is sensible."""

    def test_help_dialog_content(self, main_window):
        """Help dialog should have version info and useful content."""
        # Schedule the dialog to be inspected before it blocks
        dialog_info = {}

        def _inspect_and_close():
            active = QApplication.activeModalWidget()
            if active:
                dialog_info.update(inspect_dialog(active))
                active.close()

        QTimer.singleShot(500, _inspect_and_close)

        # Open help dialog
        help_fn = getattr(main_window, 'on_help_clicked', None)
        if help_fn is None:
            pytest.skip("on_help_clicked not found")
        help_fn()
        QApplication.processEvents()

        if dialog_info:
            print(f"  Dialog: {dialog_info.get('class', '?')}")
            print(f"  Title: {dialog_info.get('title', '?')}")
            labels = dialog_info.get('labels', [])
            print(f"  Labels: {len(labels)} text elements")
            for label in labels[:5]:
                print(f"    - {label[:80]}")
            buttons = dialog_info.get('buttons', [])
            print(f"  Buttons: {[b['text'] for b in buttons]}")

    def test_channel_manager_content(self, main_window, multi_channel_abf):
        """Channel manager should list all channels with correct types."""
        if not main_window.state.sweeps:
            load_file_and_wait(main_window, multi_channel_abf.path)

        cm = main_window.channel_manager
        channels = cm.get_channels()

        print(f"  Channel manager: {len(channels)} channels")
        for ch_name, ch_cfg in channels.items():
            print(f"    {ch_name}: type={ch_cfg.channel_type}, "
                  f"visible={ch_cfg.visible}, order={ch_cfg.order}")

        # Verify expected channels are present
        expected = multi_channel_abf.pleth_channels + multi_channel_abf.ekg_channels
        for ch in expected:
            assert any(ch in name for name in channels.keys()), \
                f"Expected channel '{ch}' not in channel manager"

    def test_combo_box_contents(self, main_window, multi_channel_abf):
        """Verify combo boxes have sensible items."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        combos_to_check = {
            'AnalyzeChanSelect': 'Analysis channel',
            'StimChanSelect': 'Stim channel',
            'peak_detec_combo': 'Peak detection algorithm',
        }

        for combo_name, label in combos_to_check.items():
            combo = getattr(main_window, combo_name, None)
            if combo is None:
                print(f"  {label}: not found")
                continue

            items = [combo.itemText(i) for i in range(combo.count())]
            current = combo.currentText()
            print(f"  {label} ({combo_name}): {items} [current: '{current}']")
            assert len(items) > 0, f"{combo_name} is empty"

    def test_dialog_watcher_log(self, main_window, dialog_watcher):
        """Report all dialogs seen during the test session."""
        if dialog_watcher is None:
            pytest.skip("No dialog watcher")

        seen = dialog_watcher.seen_dialogs
        print(f"\n  Total dialogs seen this session: {len(seen)}")
        for i, d in enumerate(seen):
            print(f"    {i+1}. [{d['type']}] '{d['title']}' -> {d.get('response', '?')}")
            if d.get('text'):
                print(f"       Text: {d['text'][:100]}")


# ── Benchmarking Tests ───────────────────────────────────────────


class TestBenchmarks:
    """Time key operations to detect performance regressions."""

    def test_benchmark_baseline_memory(self, main_window):
        """Record baseline memory before any operations."""
        info = get_process_info()
        print(f"  Baseline: RSS={info['rss_mb']:.0f}MB, "
              f"VMS={info['vms_mb']:.0f}MB, "
              f"threads={info['num_threads']}")

    def test_benchmark_file_load(self, main_window, multi_channel_abf):
        """Time and measure memory for loading an ABF file."""
        with ResourceMonitor("file load") as mon:
            load_file_and_wait(main_window, multi_channel_abf.path, timeout=60)

        print(f"  {mon.result}")
        assert mon.elapsed < 30, f"File load too slow: {mon.elapsed:.1f}s"

    def test_benchmark_peak_detection(self, main_window, multi_channel_abf):
        """Time and measure memory for peak detection."""
        if not main_window.state.sweeps:
            load_file_and_wait(main_window, multi_channel_abf.path)

        _select_channel(main_window, "AnalyzeChanSelect",
                        multi_channel_abf.pleth_channels[0])

        if not main_window.peak_prominence:
            main_window.peak_prominence = 0.05
            main_window.peak_height_threshold = 0.05
            btn = getattr(main_window, "ApplyPeakFindPushButton", None)
            if btn:
                btn.setEnabled(True)

        with ResourceMonitor("peak detection") as mon:
            main_window._apply_peak_detection()
            QApplication.processEvents()

        print(f"  {mon.result}")
        assert mon.elapsed < 30, f"Peak detection too slow: {mon.elapsed:.1f}s"

    def test_benchmark_plot_redraw(self, main_window, multi_channel_abf):
        """Time plot redraw with memory tracking."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        with ResourceMonitor("plot redraw") as mon:
            for _ in range(3):
                main_window.redraw_main_plot()
                QApplication.processEvents()

        mean_ms = mon.elapsed / 3 * 1000
        print(f"  plot redraw: {mean_ms:.0f}ms mean (3 runs), {mon.result.split(',', 1)[1]}")
        assert mean_ms < 2000, f"Plot redraw too slow: {mean_ms:.0f}ms mean"

    def test_benchmark_ekg_detection(self, main_window, multi_channel_abf):
        """Time EKG R-peak detection with resource tracking."""
        if not main_window.state.sweeps:
            load_file_and_wait(main_window, multi_channel_abf.path)

        main_window.state.ekg_chan = multi_channel_abf.ekg_channels[0]
        QApplication.processEvents()

        ekg_fn = getattr(main_window, "_auto_detect_ekg_current_sweep", None)
        if ekg_fn is None:
            pytest.skip("EKG detection not available")

        with ResourceMonitor("EKG detection") as mon:
            ekg_fn()
            QApplication.processEvents()

        n_peaks = 0
        results = main_window.state.ecg_results_by_sweep
        if results:
            sweep = results.get(0) or results.get(main_window.state.sweep_idx)
            if sweep and hasattr(sweep, 'r_peaks'):
                n_peaks = len(sweep.r_peaks)

        print(f"  {mon.result} ({n_peaks} R-peaks)")
        assert mon.elapsed < 10, f"EKG detection too slow: {mon.elapsed:.1f}s"

    def test_benchmark_save_session(self, main_window, multi_channel_abf, tmp_path):
        """Time session save with resource tracking."""
        import shutil
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            with ResourceMonitor("session save") as mon:
                main_window._save_session_pmx()
                QApplication.processEvents()

            pmx_files = list((tmp_path / "physiometrics").glob("*.pmx"))
            size_kb = pmx_files[0].stat().st_size // 1024 if pmx_files else 0
            print(f"  {mon.result} ({size_kb} KB)")
            assert mon.elapsed < 15, f"Session save too slow: {mon.elapsed:.1f}s"
        finally:
            main_window.state.in_path = original

    def test_benchmark_summary(self, main_window):
        """Print final memory state."""
        info = get_process_info()
        print(f"\n  === RESOURCE SUMMARY ===")
        print(f"  Final: RSS={info['rss_mb']:.0f}MB, "
              f"VMS={info['vms_mb']:.0f}MB, "
              f"threads={info['num_threads']}")
        print(f"  Thresholds: load<30s, peaks<30s, redraw<2s, EKG<10s, save<15s")
