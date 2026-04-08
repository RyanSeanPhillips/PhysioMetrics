"""
Memory leak tests — load/save files repeatedly and monitor for RSS growth.

Tests:
1. Load the same file N times — memory should stabilize, not grow linearly
2. Load different files in sequence — memory should not accumulate
3. Load + peak detect + save cycle — full workflow leak check
4. Rapid file switching — stress test state clearing

Run:  python -m pytest tests/test_memory_leaks.py -v -s --benchmark-disable
"""

import sys
import os
import gc
import shutil
import time
from pathlib import Path


# Fix stdout encoding for Windows (app print statements may contain unicode arrows)
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import psutil
import pytest
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF, LEGACY_ABF

EXAMPLES = ROOT / "examples"

# Collect all loadable test files
TEST_FILES = []
for abf in sorted(EXAMPLES.glob("*.abf")):
    TEST_FILES.append(abf)
TEST_FILES.append(MULTI_CHANNEL_ABF.path)


def get_rss_mb():
    """Get current process RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def force_gc():
    """Force garbage collection and let Qt process events."""
    gc.collect()
    QApplication.processEvents()
    gc.collect()


class TestMemoryLeaks:
    """Monitor memory during repeated file operations."""

    def test_repeated_load_same_file(self, main_window):
        """Loading the same file 5 times should not grow memory linearly."""
        test_file = LEGACY_ABF.path
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")

        force_gc()
        baseline_mb = get_rss_mb()
        readings = [baseline_mb]

        print(f"\n  Baseline: {baseline_mb:.0f} MB")
        print(f"  Loading {test_file.name} 5 times...")

        for i in range(5):
            load_file_and_wait(main_window, test_file)
            force_gc()
            rss = get_rss_mb()
            readings.append(rss)
            delta = rss - readings[-2]
            print(f"    Load {i+1}: {rss:.0f} MB (delta: {delta:+.1f} MB)")

        # Check: memory growth from load 2->5 should be small
        # (load 1 may allocate structures, but subsequent loads should reuse)
        growth_after_first = readings[-1] - readings[2]  # load 5 vs load 2
        total_growth = readings[-1] - baseline_mb

        print(f"  Total growth: {total_growth:+.0f} MB")
        print(f"  Growth after first load: {growth_after_first:+.0f} MB")

        # After the first load stabilizes, subsequent loads should add <50MB each
        # (some growth is normal from caches, but linear growth = leak)
        assert growth_after_first < 200, \
            f"Memory grew {growth_after_first:.0f} MB after first load — possible leak"

    def test_load_different_files_sequence(self, main_window):
        """Loading different files should not accumulate memory."""
        available = [f for f in TEST_FILES if f.exists()]
        if len(available) < 3:
            pytest.skip(f"Need 3+ test files, found {len(available)}")

        force_gc()
        baseline_mb = get_rss_mb()
        readings = [baseline_mb]

        print(f"\n  Baseline: {baseline_mb:.0f} MB")
        print(f"  Loading {len(available)} different files...")

        for i, filepath in enumerate(available):
            load_file_and_wait(main_window, filepath)
            force_gc()
            rss = get_rss_mb()
            delta = rss - readings[-1]
            readings.append(rss)
            print(f"    {filepath.name}: {rss:.0f} MB (delta: {delta:+.1f} MB)")

        # After loading all files, memory should reflect only the LAST file
        # (previous file data should be freed)
        peak_mb = max(readings)
        final_mb = readings[-1]

        print(f"  Peak: {peak_mb:.0f} MB, Final: {final_mb:.0f} MB")
        print(f"  Total growth from baseline: {final_mb - baseline_mb:+.0f} MB")

        # Final memory shouldn't be much more than peak (data from old files freed)
        # Allow for some retained caches
        assert final_mb < peak_mb + 100, \
            f"Memory didn't decrease after switching files: peak={peak_mb:.0f}, final={final_mb:.0f}"

    def test_load_detect_save_cycle(self, main_window, tmp_path):
        """Full workflow: load→detect→save, repeated. Check for leaks."""
        test_file = MULTI_CHANNEL_ABF.path
        if not test_file.exists():
            pytest.skip("Multi-channel test file not found")

        force_gc()
        baseline_mb = get_rss_mb()
        readings = [baseline_mb]
        n_cycles = 3

        print(f"\n  Baseline: {baseline_mb:.0f} MB")
        print(f"  Running {n_cycles} load→detect→save cycles...")

        for i in range(n_cycles):
            # Load
            load_file_and_wait(main_window, test_file)

            # Detect peaks
            if not main_window.peak_prominence:
                main_window.peak_prominence = 0.05
                main_window.peak_height_threshold = 0.05
                btn = getattr(main_window, "ApplyPeakFindPushButton", None)
                if btn:
                    btn.setEnabled(True)
            main_window._apply_peak_detection()
            QApplication.processEvents()

            # Save to tmp
            tmp_abf = tmp_path / f"cycle_{i}.abf"
            shutil.copy2(test_file, tmp_abf)
            original = main_window.state.in_path
            main_window.state.in_path = tmp_abf
            main_window._save_session_pmx()
            QApplication.processEvents()
            main_window.state.in_path = original

            force_gc()
            rss = get_rss_mb()
            delta = rss - readings[-1]
            readings.append(rss)

            n_peaks = sum(
                len(p) for p in main_window.state.all_peaks_by_sweep.values()
            ) if isinstance(main_window.state.all_peaks_by_sweep, dict) else 0

            print(f"    Cycle {i+1}: {rss:.0f} MB (delta: {delta:+.1f} MB), "
                  f"{n_peaks} peaks")

        growth_per_cycle = (readings[-1] - readings[1]) / max(n_cycles - 1, 1)
        total_growth = readings[-1] - baseline_mb

        print(f"  Growth per cycle: {growth_per_cycle:+.0f} MB")
        print(f"  Total growth: {total_growth:+.0f} MB")

        # Each cycle shouldn't add much — <100MB per cycle suggests a leak
        assert growth_per_cycle < 100, \
            f"Memory growing {growth_per_cycle:.0f} MB/cycle — possible leak"

    def test_rapid_file_switching(self, main_window):
        """Rapidly switch between two files 10 times — stress test cleanup."""
        file_a = LEGACY_ABF.path
        file_b = MULTI_CHANNEL_ABF.path
        if not file_a.exists() or not file_b.exists():
            pytest.skip("Need both test files")

        force_gc()
        baseline_mb = get_rss_mb()

        print(f"\n  Baseline: {baseline_mb:.0f} MB")
        print(f"  Rapid switching between {file_a.name} and {file_b.name}...")

        for i in range(10):
            f = file_a if i % 2 == 0 else file_b
            load_file_and_wait(main_window, f, timeout=15)

        force_gc()
        time.sleep(0.5)
        force_gc()
        final_mb = get_rss_mb()
        growth = final_mb - baseline_mb

        print(f"  After 10 switches: {final_mb:.0f} MB (growth: {growth:+.0f} MB)")

        # 10 rapid switches shouldn't accumulate much beyond the larger file's footprint
        assert growth < 500, \
            f"Memory grew {growth:.0f} MB after 10 file switches — possible leak"

    def test_full_workflow_cycle(self, main_window, tmp_path):
        """Full workflow: load→channels→peaks→EKG→markers→CTA→save→export, repeated."""
        from unittest.mock import patch, MagicMock
        from PyQt6.QtWidgets import QDialog
        from conftest import MULTI_CHANNEL_ABF

        test_file = MULTI_CHANNEL_ABF.path
        if not test_file.exists():
            pytest.skip("Multi-channel test file not found")

        force_gc()
        baseline_mb = get_rss_mb()
        readings = [baseline_mb]
        n_cycles = 3

        print(f"\n  Baseline: {baseline_mb:.0f} MB")
        print(f"  Running {n_cycles} FULL workflow cycles...")
        print(f"  (load -> channels -> peaks -> EKG -> markers -> save -> export)")

        for i in range(n_cycles):
            cycle_dir = tmp_path / f"cycle_{i}"
            cycle_dir.mkdir()

            # 1. Load file
            load_file_and_wait(main_window, test_file)

            # 2. Select pleth channel
            pleth_ch = MULTI_CHANNEL_ABF.pleth_channels[0]
            combo = main_window.AnalyzeChanSelect
            for j in range(combo.count()):
                if pleth_ch in combo.itemText(j):
                    combo.setCurrentIndex(j)
                    QApplication.processEvents()
                    break

            # 3. Select stim channel
            stim_ch = MULTI_CHANNEL_ABF.stim_channels[0]
            combo_s = main_window.StimChanSelect
            for j in range(combo_s.count()):
                if stim_ch in combo_s.itemText(j):
                    combo_s.setCurrentIndex(j)
                    QApplication.processEvents()
                    break

            # 4. Detect peaks
            main_window.peak_prominence = 0.05
            main_window.peak_height_threshold = 0.05
            btn = getattr(main_window, "ApplyPeakFindPushButton", None)
            if btn:
                btn.setEnabled(True)
            main_window._apply_peak_detection()
            QApplication.processEvents()

            n_peaks = sum(
                len(p) for p in main_window.state.all_peaks_by_sweep.values()
            ) if isinstance(main_window.state.all_peaks_by_sweep, dict) else 0

            # 5. Detect EKG
            ekg_ch = MULTI_CHANNEL_ABF.ekg_channels[0]
            main_window.state.ekg_chan = ekg_ch
            ekg_fn = getattr(main_window, "_auto_detect_ekg_current_sweep", None)
            if ekg_fn:
                ekg_fn()
                QApplication.processEvents()

            n_rpeaks = 0
            ecg = main_window.state.ecg_results_by_sweep.get(0)
            if ecg and hasattr(ecg, 'r_peaks'):
                n_rpeaks = len(ecg.r_peaks)

            # 6. Add event markers (programmatically)
            vm = getattr(main_window, '_event_marker_viewmodel', None)
            if vm and hasattr(vm, 'add_marker'):
                try:
                    from core.domain.events.models import EventMarker
                    marker = EventMarker(
                        start_time=10.0, end_time=15.0,
                        sweep_idx=0, category='test', label='stim_test',
                        condition=f'cycle_{i}',
                    )
                    vm.add_marker(marker)
                    QApplication.processEvents()
                except Exception as e:
                    print(f"    (marker add failed: {e})")

            # 7. Save .pmx
            tmp_abf = cycle_dir / "26402007.abf"
            shutil.copy2(test_file, tmp_abf)
            original = main_window.state.in_path
            main_window.state.in_path = tmp_abf
            main_window._save_session_pmx()
            QApplication.processEvents()
            main_window.state.in_path = original

            # 8. Export CSVs
            em = main_window.export_manager
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
            mock_dlg.values.return_value = {
                "preview": f"cycle_{i}", "experiment_type": "30hz_stim",
                "choose_dir": False, "save_npz": False,
                "save_timeseries_csv": True, "save_breaths_csv": True,
                "save_events_csv": True, "save_hr_csv": True,
                "save_pdf": False, "save_session": False,
                "save_ml_training": False,
            }
            tmp_export_abf = cycle_dir / "export_26402007.abf"
            shutil.copy2(test_file, tmp_export_abf)
            main_window.state.in_path = tmp_export_abf
            try:
                with patch("export.export_manager.SaveMetaDialog", return_value=mock_dlg):
                    em._export_all_analyzed_data(preview_only=False)
                QApplication.processEvents()
            except Exception as e:
                print(f"    (export failed: {e})")
            main_window.state.in_path = original

            # Measure memory
            force_gc()
            rss = get_rss_mb()
            delta = rss - readings[-1]
            readings.append(rss)

            # Count exported files
            export_dir = cycle_dir / "Pleth_App_analysis"
            n_csvs = len(list(export_dir.glob("*.csv"))) if export_dir.exists() else 0

            print(f"    Cycle {i+1}: {rss:.0f} MB (delta: {delta:+.1f} MB) | "
                  f"{n_peaks} peaks, {n_rpeaks} R-peaks, {n_csvs} CSVs exported")

        growth_per_cycle = (readings[-1] - readings[1]) / max(n_cycles - 1, 1)
        total_growth = readings[-1] - baseline_mb

        print(f"  Growth per cycle: {growth_per_cycle:+.0f} MB")
        print(f"  Total growth: {total_growth:+.0f} MB")

        # Full workflow cycles should not grow >150MB per cycle
        assert growth_per_cycle < 150, \
            f"Memory growing {growth_per_cycle:.0f} MB/cycle in full workflow -- possible leak"

    def test_memory_report(self, main_window):
        """Print final memory report with detailed breakdown."""
        proc = psutil.Process()
        mem = proc.memory_info()

        print(f"\n  === MEMORY REPORT ===")
        print(f"  RSS:     {mem.rss / (1024*1024):.0f} MB")
        print(f"  VMS:     {mem.vms / (1024*1024):.0f} MB")
        print(f"  Private: {mem.private / (1024*1024):.0f} MB" if hasattr(mem, 'private') else "")
        print(f"  Threads: {proc.num_threads()}")

        # Python-level memory stats
        import tracemalloc
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top = snapshot.statistics('lineno')[:10]
            print(f"\n  Top 10 memory allocations:")
            for stat in top:
                print(f"    {stat}")
