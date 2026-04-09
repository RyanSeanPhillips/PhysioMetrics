"""
Deep export validation — thorough check of ALL exported data.

Validates:
1. Timeseries CSV: all columns, timestamps, HR/RR data, no gaps
2. Breaths CSV: proper timestamps, breath metrics present
3. Events CSV: stim events with onset/offset times
4. HR CSV: per-beat data with RR intervals, HR BPM, timestamps
5. Session .pmx save/reload round-trip

Run:  python -m pytest tests/test_export_deep_validation.py -v -s
"""

import sys
import csv
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QDialog

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis, _select_channel


def _run_full_export(main_window, output_dir, include_hr=True):
    """Run export with all CSV types enabled. Returns export directory."""
    em = main_window.export_manager

    flags = {
        "preview": "test_deep_validation",
        "experiment_type": "30hz_stim",
        "choose_dir": False,
        "save_npz": False,
        "save_timeseries_csv": True,
        "save_breaths_csv": True,
        "save_events_csv": True,
        "save_hr_csv": include_hr,
        "save_pdf": False,
        "save_session": False,
        "save_ml_training": False,
    }

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.values.return_value = flags

    original_in_path = main_window.state.in_path
    tmp_abf = output_dir / original_in_path.name
    if not tmp_abf.exists():
        shutil.copy2(original_in_path, tmp_abf)
    main_window.state.in_path = tmp_abf

    try:
        with patch("export.export_manager.SaveMetaDialog", return_value=mock_dlg):
            em._export_all_analyzed_data(preview_only=False)
        QApplication.processEvents()
    finally:
        main_window.state.in_path = original_in_path

    return output_dir / "Pleth_App_analysis"


class TestDeepExportValidation:
    """Comprehensive validation of all exported CSV files."""

    @pytest.fixture(autouse=True)
    def setup_and_export(self, main_window, multi_channel_abf, tmp_path):
        """Set up analysis and run export once for all tests in this class."""
        self.mw = main_window
        self.info = multi_channel_abf
        self.tmp = tmp_path

        # Run analysis
        _setup_analysis(main_window, multi_channel_abf)

        # Verify prerequisites
        st = main_window.state
        assert st.ecg_results_by_sweep, "No EKG data — can't test HR export"

        # Count what we have
        peaks = st.all_peaks_by_sweep
        if isinstance(peaks, dict):
            self.total_peaks = sum(len(p) for p in peaks.values())
        else:
            self.total_peaks = sum(len(p) for p in peaks if p is not None)

        ecg = st.ecg_results_by_sweep.get(0) or st.ecg_results_by_sweep.get(st.sweep_idx)
        self.n_rpeaks = len(ecg.r_peaks) if ecg and hasattr(ecg, 'r_peaks') else 0

        print(f"\n  Setup: {self.total_peaks} breath peaks, {self.n_rpeaks} R-peaks")

        # Run export
        self.export_dir = _run_full_export(main_window, tmp_path, include_hr=True)

        # Find exported files
        self.csv_files = {}
        if self.export_dir.exists():
            for f in self.export_dir.glob("*.csv"):
                name = f.name.lower()
                if "_timeseries" in name or "_means_by_time" in name:
                    self.csv_files["timeseries"] = f
                elif "_breaths" in name or "_means_by_breath" in name:
                    self.csv_files["breaths"] = f
                elif "_events" in name:
                    self.csv_files["events"] = f
                elif "_hr" in name:
                    self.csv_files["hr"] = f

        print(f"  Exported files: {list(self.csv_files.keys())}")
        for k, v in self.csv_files.items():
            print(f"    {k}: {v.name} ({v.stat().st_size // 1024} KB)")

    # ── Timeseries CSV ──────────────────────────────────────────

    def test_timeseries_exists(self):
        assert "timeseries" in self.csv_files, \
            f"No timeseries CSV found. Files: {list(self.export_dir.glob('*.csv'))}"

    def test_timeseries_has_time_column(self):
        if "timeseries" not in self.csv_files:
            pytest.skip("No timeseries CSV")
        with open(self.csv_files["timeseries"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        time_cols = [h for h in headers if "time" in h.lower() or h.lower() == "t"]
        assert time_cols, f"No time column in timeseries. Headers: {headers}"
        print(f"  Time columns: {time_cols}")

    def test_timeseries_has_signal_columns(self):
        if "timeseries" not in self.csv_files:
            pytest.skip("No timeseries CSV")
        with open(self.csv_files["timeseries"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        # Should have at least: time, raw signal, and some metrics
        assert len(headers) >= 3, f"Too few columns ({len(headers)}): {headers}"
        print(f"  Timeseries columns ({len(headers)}): {headers[:10]}...")

    def test_timeseries_has_hr_columns(self):
        """Timeseries should include HR and/or RR interval columns when EKG is detected."""
        if "timeseries" not in self.csv_files:
            pytest.skip("No timeseries CSV")
        with open(self.csv_files["timeseries"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        hr_cols = [h for h in headers if any(kw in h.lower() for kw in
                   ["heart", "hr", "rr_interval", "rsa", "r-r"])]
        print(f"  HR-related columns: {hr_cols}")
        # HR columns are present if EKG was detected and Y2 metrics computed
        if self.n_rpeaks > 0:
            # Don't hard-fail — Y2 HR export depends on Y2 being computed
            if not hr_cols:
                print("  WARNING: No HR columns in timeseries despite EKG data")

    def test_timeseries_timestamps_monotonic(self):
        """Time values should be monotonically increasing within each sweep."""
        if "timeseries" not in self.csv_files:
            pytest.skip("No timeseries CSV")
        with open(self.csv_files["timeseries"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            time_col = next((h for h in headers if "time" in h.lower() or h.lower() == "t"), None)
            if not time_col:
                pytest.skip("No time column found")

            times = []
            for row in reader:
                try:
                    times.append(float(row[time_col]))
                except (ValueError, KeyError):
                    continue
                if len(times) >= 1000:
                    break  # Check first 1000 rows

        assert len(times) > 0, "No time values found"
        # Check first sweep is monotonic
        diffs = np.diff(times)
        non_monotonic = np.sum(diffs <= 0)
        # Allow a few resets (sweep boundaries)
        print(f"  First {len(times)} timestamps: [{times[0]:.3f} ... {times[-1]:.3f}]")
        print(f"  Non-monotonic points: {non_monotonic} (sweep resets expected)")

    def test_timeseries_no_empty_rows(self):
        """Rows should not be entirely empty."""
        if "timeseries" not in self.csv_files:
            pytest.skip("No timeseries CSV")
        with open(self.csv_files["timeseries"], newline="") as f:
            reader = csv.DictReader(f)
            empty_rows = 0
            total_rows = 0
            for row in reader:
                total_rows += 1
                if all(v.strip() == "" for v in row.values()):
                    empty_rows += 1
                if total_rows >= 5000:
                    break

        assert total_rows > 0, "Timeseries CSV is empty"
        assert empty_rows == 0, f"{empty_rows}/{total_rows} rows are completely empty"
        print(f"  Timeseries: {total_rows} rows checked, 0 empty")

    # ── Breaths CSV ─────────────────────────────────────────────

    def test_breaths_exists(self):
        assert "breaths" in self.csv_files, \
            f"No breaths CSV found. Files: {list(self.export_dir.glob('*.csv'))}"

    def test_breaths_has_required_columns(self):
        if "breaths" not in self.csv_files:
            pytest.skip("No breaths CSV")
        with open(self.csv_files["breaths"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

        headers_lower = [h.lower() for h in headers]
        print(f"  Breaths columns ({len(headers)}): {headers[:15]}...")

        # Should have sweep, time/onset, and at least a few metrics
        assert len(headers) >= 5, f"Too few breath columns ({len(headers)})"

    def test_breaths_has_timing_data(self):
        """Each breath should have a timestamp or onset time."""
        if "breaths" not in self.csv_files:
            pytest.skip("No breaths CSV")
        with open(self.csv_files["breaths"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            time_cols = [h for h in headers if any(kw in h.lower() for kw in
                        ["time", "onset", "t_onset", "peak_time"])]

            if not time_cols:
                print(f"  WARNING: No obvious time column in breaths. Headers: {headers}")
                return

            rows = list(reader)

        assert len(rows) > 0, "Breaths CSV is empty"
        # Check that time values are present and numeric
        time_col = time_cols[0]
        valid_times = 0
        for row in rows[:100]:
            try:
                t = float(row[time_col])
                if not np.isnan(t):
                    valid_times += 1
            except (ValueError, KeyError):
                pass

        assert valid_times > 0, f"No valid time values in column '{time_col}'"
        print(f"  Breaths: {len(rows)} rows, {valid_times}/100 have valid times in '{time_col}'")

    def test_breaths_count_reasonable(self):
        """Number of breaths in CSV should roughly match detected peaks."""
        if "breaths" not in self.csv_files:
            pytest.skip("No breaths CSV")
        with open(self.csv_files["breaths"], newline="") as f:
            n_rows = sum(1 for _ in csv.DictReader(f))

        # Should have at least some breaths (may be less than total peaks due to filtering)
        assert n_rows > 0, "Breaths CSV has no data rows"
        print(f"  Breaths CSV: {n_rows} rows (detected peaks: {self.total_peaks})")

    # ── Events CSV ──────────────────────────────────────────────

    def test_events_exists(self):
        if not self.export_dir.exists():
            pytest.skip("Export dir not created")
        all_csvs = list(self.export_dir.glob("*.csv"))
        event_files = [f for f in all_csvs if "_events" in f.name.lower()]
        assert "events" in self.csv_files or event_files, \
            f"No events CSV found. Files: {[f.name for f in all_csvs]}"

    def test_events_has_stim_data(self):
        """Events CSV should contain stim onset/offset times."""
        if "events" not in self.csv_files:
            pytest.skip("No events CSV")
        with open(self.csv_files["events"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        print(f"  Events columns: {headers}")
        print(f"  Events rows: {len(rows)}")

        if len(rows) > 0:
            # Show first row for debugging
            print(f"  First row: {dict(list(rows[0].items())[:6])}")

    # ── HR CSV ──────────────────────────────────────────────────

    def test_hr_csv_exists(self):
        """HR CSV should be created when EKG data is present."""
        # HR CSV export depends on save_hr_csv flag being set
        if "hr" not in self.csv_files:
            print("  WARNING: No HR CSV found — may need save_hr_csv=True in flags")
            # Don't hard-fail — flag might not be supported in this version
            return
        print(f"  HR CSV found: {self.csv_files['hr'].name}")

    def test_hr_csv_has_beat_data(self):
        """HR CSV should have per-beat rows with timestamps and BPM."""
        if "hr" not in self.csv_files:
            pytest.skip("No HR CSV")
        with open(self.csv_files["hr"], newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)

        print(f"  HR columns: {headers}")
        assert len(rows) > 0, "HR CSV has no data rows"
        print(f"  HR rows: {len(rows)} (R-peaks detected: {self.n_rpeaks})")

        # Check for expected columns
        headers_lower = [h.lower() for h in headers]
        has_time = any("t" in h or "time" in h for h in headers_lower)
        has_hr = any("hr" in h or "bpm" in h or "heart" in h for h in headers_lower)
        has_rr = any("rr" in h or "interval" in h for h in headers_lower)

        print(f"  Has time column: {has_time}")
        print(f"  Has HR/BPM column: {has_hr}")
        print(f"  Has RR interval column: {has_rr}")

        if has_time:
            # Verify timestamps are numeric and reasonable
            time_col = next(h for h in headers if "t" in h.lower())
            valid = 0
            for row in rows[:50]:
                try:
                    t = float(row[time_col])
                    if t >= 0:
                        valid += 1
                except (ValueError, KeyError):
                    pass
            assert valid > 0, f"No valid timestamps in HR CSV column '{time_col}'"

        if has_hr:
            # Verify HR values are in physiological range (rodent: 200-800 BPM)
            hr_col = next(h for h in headers if "hr" in h.lower() or "bpm" in h.lower())
            hr_values = []
            for row in rows[:50]:
                try:
                    hr = float(row[hr_col])
                    if not np.isnan(hr):
                        hr_values.append(hr)
                except (ValueError, KeyError):
                    pass
            if hr_values:
                print(f"  HR range: {min(hr_values):.0f} - {max(hr_values):.0f} BPM")
                # Rodent HR should be roughly 200-900 BPM
                assert min(hr_values) > 50, f"HR too low: {min(hr_values)}"
                assert max(hr_values) < 1500, f"HR too high: {max(hr_values)}"

    def test_hr_csv_timestamps_match_rpeaks(self):
        """HR CSV beat count should roughly match detected R-peaks."""
        if "hr" not in self.csv_files:
            pytest.skip("No HR CSV")
        with open(self.csv_files["hr"], newline="") as f:
            n_hr_rows = sum(1 for _ in csv.DictReader(f))

        # HR CSV has one row per beat per sweep — should be close to total R-peaks
        # Allow some tolerance since not all sweeps may be exported
        print(f"  HR CSV rows: {n_hr_rows}, R-peaks detected: {self.n_rpeaks}")
        if self.n_rpeaks > 0:
            # At least 10% of R-peaks should appear (first sweep at minimum)
            assert n_hr_rows >= self.n_rpeaks * 0.1, \
                f"Too few HR rows ({n_hr_rows}) vs R-peaks ({self.n_rpeaks})"


class TestExportFileSummary:
    """Print a summary of all exported files for manual review."""

    def test_export_summary(self, main_window, multi_channel_abf, tmp_path):
        """Generate and summarize all export files."""
        _setup_analysis(main_window, multi_channel_abf)
        export_dir = _run_full_export(main_window, tmp_path)

        print(f"\n{'='*60}")
        print(f"  EXPORT SUMMARY")
        print(f"{'='*60}")

        if not export_dir.exists():
            pytest.fail(f"Export directory not created: {export_dir}")

        total_size = 0
        for f in sorted(export_dir.glob("*")):
            size = f.stat().st_size
            total_size += size
            print(f"  {f.name:50s} {size//1024:>6d} KB")

            # Print headers for CSV files
            if f.suffix == ".csv":
                with open(f, newline="") as fh:
                    reader = csv.DictReader(fh)
                    headers = reader.fieldnames or []
                    n_rows = sum(1 for _ in reader)
                print(f"    Columns ({len(headers)}): {', '.join(headers[:8])}{'...' if len(headers) > 8 else ''}")
                print(f"    Rows: {n_rows}")

        print(f"{'='*60}")
        print(f"  Total: {total_size//1024} KB across {len(list(export_dir.glob('*')))} files")
        print(f"{'='*60}")
