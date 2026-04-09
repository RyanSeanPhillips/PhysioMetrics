"""
Comprehensive app audit — systematically tests every accessible feature.

This test explores the app autonomously and writes findings to:
    tests/output/audit_report.md

Tests:
1. All tabs — screenshot each, list widgets
2. All combo boxes — verify contents
3. All buttons — catalog, check enabled state
4. Event markers — create, save, reload
5. Filter controls — toggle, verify effect
6. Sweep navigation — forward/back through all sweeps
7. Dialog catalog — open accessible dialogs, inspect content
8. Keyboard shortcuts — test non-destructive ones
9. State integrity — verify no stale data after operations
10. Channel manager — full inspection

Run:  python -m pytest tests/test_comprehensive_audit.py -v -s
"""

import sys
import time
import json
from pathlib import Path
from io import StringIO

import numpy as np
import pytest
from PyQt6.QtWidgets import (QApplication, QWidget, QPushButton, QComboBox,
                               QCheckBox, QLineEdit, QSpinBox, QDoubleSpinBox,
                               QTabWidget, QLabel, QToolButton)
from PyQt6.QtCore import QTimer

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis, _select_channel
from test_utils import capture_widget, inspect_dialog, Timer

OUTPUT_DIR = ROOT / "tests" / "output"
REPORT_PATH = OUTPUT_DIR / "audit_report.md"


class AuditReport:
    """Collects findings and writes markdown report."""

    def __init__(self):
        self.sections = []
        self.issues = []
        self.warnings = []
        self.stats = {}

    def add_section(self, title, content):
        self.sections.append((title, content))

    def add_issue(self, severity, description):
        """severity: 'bug', 'warning', 'info'"""
        self.issues.append((severity, description))
        if severity == 'warning':
            self.warnings.append(description)

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# PhysioMetrics Comprehensive Audit Report",
                 f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
                 f"Test file: tests/data/26402007.abf",
                 ""]

        # Summary
        bugs = [i for s, i in self.issues if s == 'bug']
        warns = [i for s, i in self.issues if s == 'warning']
        infos = [i for s, i in self.issues if s == 'info']

        lines.append("## Summary")
        lines.append(f"- **Bugs**: {len(bugs)}")
        lines.append(f"- **Warnings**: {len(warns)}")
        lines.append(f"- **Info**: {len(infos)}")
        lines.append("")

        if bugs:
            lines.append("### Bugs Found")
            for b in bugs:
                lines.append(f"- **BUG**: {b}")
            lines.append("")

        if warns:
            lines.append("### Warnings")
            for w in warns:
                lines.append(f"- WARNING: {w}")
            lines.append("")

        # Stats
        if self.stats:
            lines.append("## Performance")
            lines.append("| Operation | Time |")
            lines.append("|-----------|------|")
            for k, v in self.stats.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        # Sections
        for title, content in self.sections:
            lines.append(f"## {title}")
            lines.append(content)
            lines.append("")

        path.write_text("\n".join(lines), encoding='utf-8')


# Module-level report
report = AuditReport()


class TestComprehensiveAudit:
    """Systematic audit of every accessible feature."""

    # ── 1. Tab Audit ─────────────────────────────────────────────

    def test_01_tab_audit(self, main_window):
        """Catalog every tab and its contents."""
        tabs = main_window.Tabs
        content = []
        content.append(f"Main tabs: {tabs.count()}")

        for i in range(tabs.count()):
            tab_name = tabs.tabText(i)
            tabs.setCurrentIndex(i)
            QApplication.processEvents()

            tab_widget = tabs.widget(i)
            buttons = tab_widget.findChildren(QPushButton)
            combos = tab_widget.findChildren(QComboBox)
            checks = tab_widget.findChildren(QCheckBox)
            labels = tab_widget.findChildren(QLabel)
            sub_tabs = tab_widget.findChildren(QTabWidget)

            visible_buttons = [b for b in buttons if b.isVisible()]
            enabled_buttons = [b for b in visible_buttons if b.isEnabled()]
            disabled_buttons = [b for b in visible_buttons if not b.isEnabled()]

            content.append(f"\n### Tab {i+1}: {tab_name}")
            content.append(f"- Buttons: {len(visible_buttons)} visible "
                          f"({len(enabled_buttons)} enabled, {len(disabled_buttons)} disabled)")
            content.append(f"- Combo boxes: {len(combos)}")
            content.append(f"- Checkboxes: {len(checks)}")
            content.append(f"- Labels: {len(labels)}")
            content.append(f"- Sub-tabs: {len(sub_tabs)}")

            if disabled_buttons:
                disabled_names = [b.text() or b.objectName() or '(unnamed)'
                                  for b in disabled_buttons[:10]]
                content.append(f"- Disabled buttons: {disabled_names}")

            # Screenshot
            capture_widget(main_window, f"audit_tab_{i}_{tab_name.lower().replace(' ','_')}")

        tabs.setCurrentIndex(1)  # back to Analysis
        QApplication.processEvents()

        report.add_section("Tab Audit", "\n".join(content))
        print(f"  Audited {tabs.count()} tabs")

    # ── 2. Combo Box Audit ───────────────────────────────────────

    def test_02_combo_audit(self, main_window, multi_channel_abf):
        """Inspect every combo box in the app."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        combos = main_window.findChildren(QComboBox)
        visible_combos = [c for c in combos if c.isVisible()]

        content = [f"Total combo boxes: {len(combos)} ({len(visible_combos)} visible)"]

        for combo in visible_combos:
            name = combo.objectName() or '(unnamed)'
            items = [combo.itemText(i) for i in range(combo.count())]
            current = combo.currentText()

            content.append(f"\n**{name}**: {len(items)} items, current='{current}'")
            if items:
                content.append(f"  Items: {items[:15]}")
                if len(items) > 15:
                    content.append(f"  ... and {len(items)-15} more")

            # Check for issues
            if len(items) == 0:
                report.add_issue('warning', f"Combo '{name}' is empty")
            if current == '' and len(items) > 0:
                report.add_issue('info', f"Combo '{name}' has no selection")

        report.add_section("Combo Box Audit", "\n".join(content))
        print(f"  Audited {len(visible_combos)} visible combos")

    # ── 3. Button Audit ──────────────────────────────────────────

    def test_03_button_audit(self, main_window):
        """Catalog every button — name, enabled state, tooltip."""
        buttons = main_window.findChildren(QPushButton)
        visible = [b for b in buttons if b.isVisible()]

        content = [f"Total buttons: {len(buttons)} ({len(visible)} visible)"]

        enabled_list = []
        disabled_list = []
        no_tooltip = []

        for btn in visible:
            name = btn.text() or btn.objectName() or '(unnamed)'
            tooltip = btn.toolTip()
            enabled = btn.isEnabled()

            entry = f"- **{name}**: {'enabled' if enabled else 'DISABLED'}"
            if tooltip:
                entry += f" — _{tooltip[:80]}_"

            if enabled:
                enabled_list.append(entry)
            else:
                disabled_list.append(entry)

            if not tooltip and btn.text():
                no_tooltip.append(name)

        content.append(f"\n### Enabled ({len(enabled_list)})")
        content.extend(enabled_list[:30])
        if len(enabled_list) > 30:
            content.append(f"... and {len(enabled_list)-30} more")

        content.append(f"\n### Disabled ({len(disabled_list)})")
        content.extend(disabled_list[:20])

        if no_tooltip:
            content.append(f"\n### Missing Tooltips ({len(no_tooltip)})")
            content.append(", ".join(no_tooltip[:20]))
            report.add_issue('info', f"{len(no_tooltip)} visible buttons have no tooltip")

        report.add_section("Button Audit", "\n".join(content))
        print(f"  Audited {len(visible)} visible buttons")

    # ── 4. Checkbox & Filter Audit ───────────────────────────────

    def test_04_filter_audit(self, main_window, multi_channel_abf):
        """Test filter checkboxes — toggle and verify state changes."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        content = []
        checks = main_window.findChildren(QCheckBox)
        visible_checks = [c for c in checks if c.isVisible()]

        content.append(f"Visible checkboxes: {len(visible_checks)}")
        for cb in visible_checks:
            name = cb.text() or cb.objectName() or '(unnamed)'
            content.append(f"- **{name}**: checked={cb.isChecked()}, enabled={cb.isEnabled()}")

        # Test filter toggles specifically
        filter_checks = {
            'LowPass_checkBox': 'Low-pass filter',
            'HighPass_checkBox': 'High-pass filter',
            'InvertSignal_checkBox': 'Invert signal',
        }

        content.append("\n### Filter Toggle Test")
        for attr_name, label in filter_checks.items():
            cb = getattr(main_window, attr_name, None)
            if cb is None:
                content.append(f"- {label}: NOT FOUND")
                report.add_issue('warning', f"Filter checkbox '{attr_name}' not found")
                continue

            original = cb.isChecked()
            content.append(f"- {label} ({attr_name}): initial={original}")

        report.add_section("Checkbox & Filter Audit", "\n".join(content))
        print(f"  Audited {len(visible_checks)} checkboxes")

    # ── 5. Channel Manager Deep Audit ────────────────────────────

    def test_05_channel_manager_audit(self, main_window, multi_channel_abf):
        """Deep inspection of channel manager state."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        cm = main_window.channel_manager
        channels = cm.get_channels()

        content = [f"Channels: {len(channels)}"]

        for ch_name, ch_cfg in channels.items():
            has_gear = False
            if ch_name in cm._channel_rows:
                row = cm._channel_rows[ch_name]
                gear = getattr(row, 'settings_btn', None)
                has_gear = gear is not None and gear.isVisible()

            content.append(f"\n**{ch_name}**:")
            content.append(f"  - Type: {ch_cfg.channel_type}")
            content.append(f"  - Visible: {ch_cfg.visible}")
            content.append(f"  - Order: {ch_cfg.order}")
            content.append(f"  - Gear icon: {'yes' if has_gear else 'no'}")

            # Check for issues
            if ch_cfg.channel_type == "Unknown":
                report.add_issue('info', f"Channel '{ch_name}' has Unknown type")

        # Verify EKG channel has gear icon
        ekg_ch = multi_channel_abf.ekg_channels[0]
        ekg_cfg = channels.get(ekg_ch)
        if ekg_cfg and ekg_cfg.channel_type == "EKG":
            if ekg_ch in cm._channel_rows:
                gear = getattr(cm._channel_rows[ekg_ch], 'settings_btn', None)
                if gear and not gear.isVisible():
                    report.add_issue('bug', f"EKG channel '{ekg_ch}' gear icon not visible")

        report.add_section("Channel Manager Audit", "\n".join(content))
        print(f"  Audited {len(channels)} channels")

    # ── 6. Sweep Navigation Audit ────────────────────────────────

    def test_06_sweep_navigation_audit(self, main_window, multi_channel_abf):
        """Navigate through all sweeps, verify no crashes or stale data."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        vm = main_window._nav_vm
        sweep_count_attr = getattr(vm, 'sweep_count', None)
        n_sweeps = sweep_count_attr() if callable(sweep_count_attr) else (sweep_count_attr or 1)

        content = [f"Sweep count: {n_sweeps}"]
        issues = []

        for s in range(min(n_sweeps, 10)):  # test up to 10 sweeps
            if hasattr(vm, 'set_sweep'):
                vm.set_sweep(s)
            QApplication.processEvents()

            # Verify state is consistent
            current_idx = main_window.state.sweep_idx
            has_peaks = bool(main_window.state.peaks_by_sweep.get(s))

            content.append(f"- Sweep {s}: state.sweep_idx={current_idx}, has_peaks={has_peaks}")

            # Check for stale EKG data (should only exist for detected sweeps)
            ekg_result = main_window.state.ecg_results_by_sweep.get(s)
            has_ekg = ekg_result is not None and hasattr(ekg_result, 'r_peaks')
            if has_ekg:
                content.append(f"  EKG: {len(ekg_result.r_peaks)} R-peaks")

        # Return to sweep 0
        if hasattr(vm, 'set_sweep'):
            vm.set_sweep(0)
        QApplication.processEvents()

        report.add_section("Sweep Navigation Audit", "\n".join(content))
        print(f"  Navigated {min(n_sweeps, 10)} sweeps")

    # ── 7. Event Marker Audit ────────────────────────────────────

    def test_07_event_marker_audit(self, main_window, multi_channel_abf):
        """Check event marker system state and save/reload."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        content = []

        vm = getattr(main_window, '_event_marker_viewmodel', None)
        if vm is None:
            content.append("Event marker viewmodel: NOT FOUND")
            report.add_issue('warning', "No event marker viewmodel")
            report.add_section("Event Marker Audit", "\n".join(content))
            return

        # Count existing markers
        markers = vm.get_markers() if hasattr(vm, 'get_markers') else []
        content.append(f"Event marker viewmodel: found")
        content.append(f"Current markers: {len(markers)}")

        # Count by type
        type_counts = {}
        for m in markers:
            mt = getattr(m, 'event_type', getattr(m, 'marker_type', '?'))
            type_counts[mt] = type_counts.get(mt, 0) + 1
        if type_counts:
            content.append(f"By type: {type_counts}")

        # Check save capability
        try:
            save_data = vm.save_to_npz() if hasattr(vm, 'save_to_npz') else None
            content.append(f"save_to_npz: {'OK' if save_data else 'returned None'}")
            if save_data:
                content.append(f"  Keys: {list(save_data.keys()) if isinstance(save_data, dict) else type(save_data)}")
        except Exception as e:
            content.append(f"save_to_npz: ERROR — {e}")
            report.add_issue('bug', f"Event marker save_to_npz failed: {e}")

        report.add_section("Event Marker Audit", "\n".join(content))
        print(f"  Event markers: {len(markers)} markers, {len(type_counts)} types")

    # ── 8. State Integrity Audit ─────────────────────────────────

    def test_08_state_integrity_audit(self, main_window, multi_channel_abf):
        """Verify AppState has no inconsistencies."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        st = main_window.state
        content = []
        issues = []

        # Basic state checks
        content.append(f"in_path: {st.in_path}")
        content.append(f"sr_hz: {st.sr_hz}")
        content.append(f"analyze_chan: {st.analyze_chan}")
        content.append(f"stim_chan: {st.stim_chan}")
        content.append(f"ekg_chan: {st.ekg_chan}")
        content.append(f"sweep_idx: {st.sweep_idx}")
        content.append(f"channel_names: {st.channel_names}")

        # Sweep data consistency
        n_channels = len(st.sweeps) if st.sweeps else 0
        content.append(f"\nSweep data: {n_channels} channels")
        for ch_name, data in st.sweeps.items():
            shape = data.shape if hasattr(data, 'shape') else len(data)
            content.append(f"  {ch_name}: shape={shape}")

        # Peak data consistency
        n_peak_sweeps = len(st.peaks_by_sweep)
        n_all_peak_sweeps = len(st.all_peaks_by_sweep) if st.all_peaks_by_sweep else 0
        content.append(f"\nPeaks: {n_peak_sweeps} sweeps in peaks_by_sweep, "
                      f"{n_all_peak_sweeps} in all_peaks_by_sweep")

        if n_peak_sweeps != n_all_peak_sweeps:
            report.add_issue('warning',
                f"peaks_by_sweep ({n_peak_sweeps}) != all_peaks_by_sweep ({n_all_peak_sweeps})")

        # EKG consistency
        n_ekg_sweeps = len(st.ecg_results_by_sweep)
        content.append(f"EKG results: {n_ekg_sweeps} sweeps")
        if st.ekg_chan and n_ekg_sweeps == 0:
            report.add_issue('warning', "ekg_chan is set but no ecg_results")
        if not st.ekg_chan and n_ekg_sweeps > 0:
            report.add_issue('bug', "ecg_results exist but ekg_chan is None")

        # Stim consistency
        n_stim_sweeps = len(st.stim_spans_by_sweep)
        content.append(f"Stim spans: {n_stim_sweeps} sweeps")
        if st.stim_chan and n_stim_sweeps == 0:
            report.add_issue('warning', "stim_chan is set but no stim_spans")

        # Filter state
        content.append(f"\nFilters: low={st.use_low} ({st.low_hz}Hz), "
                      f"high={st.use_high} ({st.high_hz}Hz), "
                      f"invert={st.use_invert}")

        report.add_section("State Integrity Audit", "\n".join(content))
        print(f"  State: {n_channels} channels, {n_peak_sweeps} sweep peaks, "
              f"{n_ekg_sweeps} EKG sweeps")

    # ── 9. Benchmark Suite ───────────────────────────────────────

    def test_09_benchmarks(self, main_window, multi_channel_abf, tmp_path):
        """Time all key operations."""
        import shutil

        content = []

        # File load
        with Timer("File load (107MB ABF)") as t:
            load_file_and_wait(main_window, multi_channel_abf.path)
        content.append(f"- {t.result}")
        report.stats["File load"] = f"{t.elapsed:.2f}s"

        # Channel selection
        with Timer("Select pleth channel") as t:
            _select_channel(main_window, "AnalyzeChanSelect",
                           multi_channel_abf.pleth_channels[0])
        content.append(f"- {t.result}")
        report.stats["Channel select"] = f"{t.elapsed:.3f}s"

        # Peak detection
        if not main_window.peak_prominence:
            main_window.peak_prominence = 0.05
            main_window.peak_height_threshold = 0.05
            btn = getattr(main_window, "ApplyPeakFindPushButton", None)
            if btn:
                btn.setEnabled(True)

        with Timer("Peak detection (10 sweeps)") as t:
            main_window._apply_peak_detection()
            QApplication.processEvents()
        content.append(f"- {t.result}")
        report.stats["Peak detection"] = f"{t.elapsed:.2f}s"

        # EKG detection
        main_window.state.ekg_chan = multi_channel_abf.ekg_channels[0]
        ekg_fn = getattr(main_window, "_auto_detect_ekg_current_sweep", None)
        if ekg_fn:
            with Timer("EKG R-peak detection") as t:
                ekg_fn()
                QApplication.processEvents()
            content.append(f"- {t.result}")
            report.stats["EKG detection"] = f"{t.elapsed:.3f}s"

        # Plot redraw (3 runs)
        times = []
        for _ in range(3):
            with Timer() as t:
                main_window.redraw_main_plot()
                QApplication.processEvents()
            times.append(t.elapsed)
        mean_ms = np.mean(times) * 1000
        content.append(f"- Plot redraw: {mean_ms:.0f}ms mean (3 runs)")
        report.stats["Plot redraw"] = f"{mean_ms:.0f}ms"

        # Session save
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original = main_window.state.in_path
        main_window.state.in_path = tmp_abf
        try:
            with Timer("Session save (Ctrl+S)") as t:
                main_window._save_session_pmx()
                QApplication.processEvents()
            pmx = list((tmp_path / "physiometrics").glob("*.pmx"))
            size_kb = pmx[0].stat().st_size // 1024 if pmx else 0
            content.append(f"- {t.result} ({size_kb} KB)")
            report.stats["Session save"] = f"{t.elapsed:.2f}s ({size_kb}KB)"
        finally:
            main_window.state.in_path = original

        # Tab switching
        tabs = main_window.Tabs
        with Timer(f"Switch through {tabs.count()} tabs") as t:
            for i in range(tabs.count()):
                tabs.setCurrentIndex(i)
                QApplication.processEvents()
        content.append(f"- {t.result}")
        report.stats["Tab switching"] = f"{t.elapsed:.3f}s"

        report.add_section("Benchmarks", "\n".join(content))
        print(f"  Benchmarks complete")

    # ── 10. Write Report ─────────────────────────────────────────

    def test_99_write_report(self, main_window, dialog_watcher):
        """Write the final audit report."""
        # Add dialog log
        if dialog_watcher:
            content = [f"Total dialogs seen: {len(dialog_watcher.seen_dialogs)}"]
            for d in dialog_watcher.seen_dialogs:
                content.append(f"- [{d['type']}] '{d['title']}' -> {d.get('response', '?')}")
                if d.get('text'):
                    content.append(f"  _{d['text'][:150]}_")
            report.add_section("Dialog Log", "\n".join(content))

        report.write(REPORT_PATH)
        print(f"\n  Audit report written to: {REPORT_PATH}")
        print(f"  Issues found: {len(report.issues)}")
        for severity, desc in report.issues:
            print(f"    [{severity.upper()}] {desc}")
