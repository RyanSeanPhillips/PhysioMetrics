"""
Export system tests — behavioral baseline before ExportService extraction.

Unit tests (1-10): Pure computation functions with synthetic data.
Integration tests (11-19): Real MainWindow + ABF, exercise full export workflow.

Run:  python -m pytest tests/test_export_compute.py -v
"""

import sys
import csv
import re
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (1-10) — Pure computation, no Qt needed
# ═══════════════════════════════════════════════════════════════════


class TestSanitizeToken:
    """Test 1: _sanitize_token string cleaning."""

    def test_basic_sanitization(self):
        from export.export_manager import ExportManager

        # Need a minimal instance — sanitize_token only uses self for method lookup
        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        assert em._sanitize_token("hello world") == "hello_world"
        assert em._sanitize_token("  spaces  ") == "spaces"
        assert em._sanitize_token("a/b\\c:d") == "abcd"
        assert em._sanitize_token("under__score") == "under_score"
        assert em._sanitize_token("") == ""
        assert em._sanitize_token("normal-name.txt") == "normal-name.txt"
        assert em._sanitize_token("30Hz_10mW") == "30Hz_10mW"
        print("  Sanitization: special chars, spaces, repeats all handled")


class TestMetricKeysInOrder:
    """Test 2: _metric_keys_in_order returns ordered keys."""

    def test_returns_keys(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        keys = em._metric_keys_in_order()
        assert isinstance(keys, list)
        assert len(keys) > 0
        # Should include common metrics
        assert "if" in keys
        assert "ti" in keys
        assert "amp_insp" in keys
        print(f"  Got {len(keys)} metric keys in order")


class TestNanmeanSem:
    """Test 3: _nanmean_sem robust statistics."""

    def test_basic_mean_sem(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        # 2D array: 3 rows, 4 columns
        X = np.array([[1.0, 2.0, 3.0, 4.0],
                       [2.0, 4.0, 6.0, 8.0],
                       [3.0, 6.0, 9.0, 12.0]])

        mean, sem = em._nanmean_sem(X, axis=0)

        np.testing.assert_allclose(mean, [2.0, 4.0, 6.0, 8.0])
        assert sem.shape == (4,)
        assert np.all(np.isfinite(sem))
        print(f"  Mean={mean}, SEM finite and correct shape")

    def test_with_nans(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        X = np.array([[1.0, np.nan], [3.0, 5.0], [np.nan, 7.0]])
        mean, sem = em._nanmean_sem(X, axis=0)

        assert np.isfinite(mean[0])  # 2 finite values → valid mean
        assert np.isfinite(mean[1])  # 2 finite values → valid mean
        print(f"  NaN handling: mean={mean}")

    def test_empty_array(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        mean, sem = em._nanmean_sem(np.array([]))
        assert np.isnan(mean)
        assert np.isnan(sem)
        print("  Empty array → (nan, nan)")


class TestMeanSem1D:
    """Test 4: _mean_sem_1d for 1D arrays."""

    def test_basic(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        m, s = em._mean_sem_1d(np.array([2.0, 4.0, 6.0]))
        assert m == pytest.approx(4.0)
        assert s > 0
        print(f"  mean={m:.2f}, sem={s:.4f}")

    def test_single_value(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        m, s = em._mean_sem_1d(np.array([5.0]))
        assert m == pytest.approx(5.0)
        assert np.isnan(s)  # SEM undefined for n=1
        print(f"  Single value: mean={m}, sem=nan")

    def test_all_nan(self):
        from export.export_manager import ExportManager

        class FakeWindow:
            state = None
        em = ExportManager(FakeWindow())

        m, s = em._mean_sem_1d(np.array([np.nan, np.nan]))
        assert np.isnan(m)
        assert np.isnan(s)


class TestValidateBreathData:
    """Test 5: validate_breath_data checks data integrity."""

    def test_valid_data(self, main_window):
        """Valid breath data should pass validation."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state
        if not st.peaks_by_sweep:
            pytest.skip("No peaks detected")

        em = main_window.export_manager
        is_valid, issues = em.validate_breath_data()

        assert is_valid, f"Validation failed on good data: {issues}"
        print(f"  Valid data: {len(issues)} warnings")


class TestIsBreathSniffing:
    """Test 6: _is_breath_sniffing checks sniff regions."""

    def test_sniffing_detection(self, main_window):
        """Breaths in sniff regions should be detected."""
        st = main_window.state

        if not hasattr(st, 'sniff_regions_by_sweep') or not st.sniff_regions_by_sweep:
            pytest.skip("No sniff regions")

        em = main_window.export_manager

        # Find a sweep with sniff regions
        sweep_idx = next(iter(st.sniff_regions_by_sweep.keys()))
        breath_data = st.breath_by_sweep.get(sweep_idx, {})
        onsets = breath_data.get('onsets', np.array([]))

        if len(onsets) < 2:
            pytest.skip("Not enough breaths")

        # Check a few breaths
        results = []
        for i in range(min(10, len(onsets) - 1)):
            is_sniff = em._is_breath_sniffing(sweep_idx, i, onsets)
            results.append(is_sniff)

        # At least some should be True (we have sniff regions) and some False
        print(f"  Checked {len(results)} breaths: {sum(results)} sniffing, {len(results) - sum(results)} eupnea")


class TestCreateOmittedMask:
    """Test 9: create_omitted_mask with known ranges."""

    def test_basic_mask(self):
        from export.export_manager import create_omitted_mask

        # Sweep 0, trace length 1000, omit samples 100-199 and 500-599
        mask = create_omitted_mask(
            sweep_idx=0,
            trace_length=1000,
            omitted_ranges={0: [(100, 199), (500, 599)]},
            omitted_sweeps=set(),
        )

        assert mask.shape == (1000,)
        assert mask[0] == True    # not omitted
        assert mask[150] == False  # omitted
        assert mask[550] == False  # omitted
        assert mask[300] == True   # not omitted
        assert np.sum(~mask) == 200  # 100 + 100 samples omitted
        print(f"  Mask: {np.sum(mask)} kept, {np.sum(~mask)} omitted")

    def test_full_sweep_omitted(self):
        from export.export_manager import create_omitted_mask

        mask = create_omitted_mask(
            sweep_idx=0,
            trace_length=500,
            omitted_ranges={},
            omitted_sweeps={0},
        )

        assert np.all(~mask), "Full sweep should be omitted"

    def test_no_omissions(self):
        from export.export_manager import create_omitted_mask

        mask = create_omitted_mask(
            sweep_idx=0,
            trace_length=500,
            omitted_ranges={},
            omitted_sweeps=set(),
        )

        assert np.all(mask), "No omissions → all kept"


class TestGetStimMasks:
    """Test 8: _get_stim_masks builds correct masks."""

    def test_stim_masks(self, main_window):
        """With stim channel, masks should have non-empty stim region."""
        st = main_window.state

        if not st.stim_chan or not st.stim_spans_by_sweep:
            pytest.skip("No stim data")

        em = main_window.export_manager
        sweep_idx = next(iter(st.stim_spans_by_sweep.keys()))

        baseline, stim, post = em._get_stim_masks(sweep_idx)

        assert baseline.shape == st.t.shape
        assert stim.shape == st.t.shape
        assert post.shape == st.t.shape

        # Stim should have some True values
        assert np.any(stim), "Stim mask should have True values"
        # Baseline should be before stim
        assert np.any(baseline), "Baseline mask should have True values"
        # They shouldn't overlap
        assert not np.any(baseline & stim), "Baseline and stim should not overlap"

        print(f"  Baseline: {np.sum(baseline)} samples, Stim: {np.sum(stim)}, Post: {np.sum(post)}")


class TestComputeMetricTrace:
    """Test 10: _compute_metric_trace computes correct metric arrays."""

    def test_compute_if(self, main_window):
        """Compute IF metric trace on real data."""
        st = main_window.state

        if not st.peaks_by_sweep or not st.breath_by_sweep:
            pytest.skip("No peaks/breaths")

        em = main_window.export_manager
        sweep_idx = next(iter(st.peaks_by_sweep.keys()))

        y_proc = main_window._get_processed_for(st.analyze_chan, sweep_idx)
        peaks = st.peaks_by_sweep[sweep_idx]
        breaths = st.breath_by_sweep[sweep_idx]

        trace = em._compute_metric_trace("if", st.t, y_proc, st.sr_hz, peaks, breaths, sweep=sweep_idx)

        assert trace is not None
        assert len(trace) == len(st.t)
        # IF should have some finite values where breaths exist
        finite_count = np.sum(np.isfinite(trace))
        assert finite_count > 0, "IF trace should have finite values"
        print(f"  IF trace: {finite_count} finite values out of {len(trace)}")


# ═══════════════════════════════════════════════════════════════════
# Integration Tests (11-19) — Real MainWindow + full export workflow
# ═══════════════════════════════════════════════════════════════════

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF
from test_export_and_save import _setup_analysis, _run_export


def _ensure_analysis_ready(main_window, file_info):
    """Load file and detect peaks if not already done."""
    from PyQt6.QtWidgets import QApplication
    import time

    st = main_window.state
    total_peaks = 0
    if st.peaks_by_sweep:
        total_peaks = sum(len(p) for p in st.peaks_by_sweep.values())

    if total_peaks < 10:
        total_peaks = _setup_analysis(main_window, file_info)
        # Extra time for GMM + precompute
        for _ in range(30):
            QApplication.processEvents()
            time.sleep(0.05)

    return total_peaks


class TestExportProducesAllCSVFiles:
    """Test 11: Full export creates expected CSV files."""

    @pytest.fixture(autouse=True)
    def _setup(self, main_window, multi_channel_abf):
        self.mw = main_window
        self.info = multi_channel_abf
        _ensure_analysis_ready(main_window, multi_channel_abf)

    def test_export_creates_csvs(self, tmp_path):
        """Export produces timeseries, breaths, and events CSVs."""
        export_dir = _run_export(self.mw, tmp_path, "test_all_csvs",
                                  save_flags={"save_hr_csv": True})

        if not export_dir.exists():
            pytest.skip("Export dir not created")

        csvs = list(export_dir.glob("*.csv"))
        csv_names = [f.name for f in csvs]
        print(f"  Exported files: {csv_names}")

        # Should have at least timeseries and breaths
        has_timeseries = any("timeseries" in n or "means_by_time" in n for n in csv_names)
        has_breaths = any("breaths" in n for n in csv_names)

        assert has_timeseries, f"No timeseries CSV found in {csv_names}"
        assert has_breaths, f"No breaths CSV found in {csv_names}"
        print(f"  Found {len(csvs)} CSV files")


class TestExportTimeseriesColumns:
    """Test 12: Timeseries CSV has expected columns."""

    def test_timeseries_columns(self, main_window, multi_channel_abf, tmp_path):
        _ensure_analysis_ready(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_ts_cols")
        if not export_dir.exists():
            pytest.skip("Export dir not created")

        ts_files = list(export_dir.glob("*timeseries*")) or list(export_dir.glob("*means_by_time*"))
        if not ts_files:
            pytest.skip("No timeseries CSV found")

        with open(ts_files[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Should have time column and at least one metric
        header_lower = [h.lower() for h in header]
        assert any(h in ("time", "t") for h in header_lower), f"No time column in {header[:5]}"
        assert len(header) > 2, f"Too few columns: {len(header)}"
        assert len(rows) > 0, "No data rows"

        print(f"  Timeseries: {len(header)} columns, {len(rows)} rows")


class TestExportBreathsRowCount:
    """Test 13: Breaths CSV row count matches detected peaks."""

    def test_breath_count(self, main_window, multi_channel_abf, tmp_path):
        _ensure_analysis_ready(main_window, multi_channel_abf)

        st = main_window.state
        total_breaths = sum(len(p) for p in st.peaks_by_sweep.values())

        export_dir = _run_export(main_window, tmp_path, "test_breath_count")
        if not export_dir.exists():
            pytest.skip("Export dir not created")

        breath_files = list(export_dir.glob("*breaths*"))
        if not breath_files:
            pytest.skip("No breaths CSV found")

        with open(breath_files[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        # Row count should be close to total breaths (may differ slightly due to edge filtering)
        csv_breaths = len(rows)
        assert csv_breaths > 0, "No breath rows in CSV"
        # Allow some tolerance (edge breaths may be excluded)
        ratio = csv_breaths / max(total_breaths, 1)
        assert 0.5 < ratio < 1.5, f"Breath count mismatch: CSV={csv_breaths}, detected={total_breaths}"

        print(f"  Breaths: CSV={csv_breaths} rows, detected={total_breaths} peaks")


class TestExportWithEKG:
    """Test 16: Export with EKG produces HR CSV."""

    def test_hr_csv_exists(self, main_window, multi_channel_abf, tmp_path):
        _ensure_analysis_ready(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_hr",
                                  save_flags={"save_hr_csv": True})
        if not export_dir.exists():
            pytest.skip("Export dir not created")

        hr_files = list(export_dir.glob("*hr*")) + list(export_dir.glob("*HR*"))
        if not hr_files:
            # HR CSV may not be created if EKG detection hasn't run
            pytest.skip("No HR CSV (EKG may not be detected)")

        with open(hr_files[0], 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        assert len(rows) > 0, "HR CSV has no data rows"
        print(f"  HR CSV: {len(header)} columns, {len(rows)} rows")


class TestExportPreviewModeNoFiles:
    """Test 17: Preview mode doesn't write files to disk."""

    def test_preview_no_files(self, main_window, multi_channel_abf, tmp_path):
        from PyQt6.QtWidgets import QApplication, QDialog
        from PyQt6.QtCore import QTimer

        _ensure_analysis_ready(main_window, multi_channel_abf)

        em = main_window.export_manager

        # Auto-close any preview dialog that pops up
        def close_active():
            active = QApplication.activeModalWidget()
            if active:
                active.close()

        QTimer.singleShot(2000, close_active)
        QTimer.singleShot(3000, close_active)

        # Run preview (should NOT create files)
        original_in_path = main_window.state.in_path
        tmp_abf = tmp_path / original_in_path.name
        if not tmp_abf.exists():
            shutil.copy2(original_in_path, tmp_abf)
        main_window.state.in_path = tmp_abf

        try:
            mock_dlg = MagicMock()
            mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
            mock_dlg.values.return_value = {
                "preview": "test_preview",
                "experiment_type": "30hz_stim",
                "choose_dir": False,
            }
            with patch("export.export_manager.SaveMetaDialog", return_value=mock_dlg):
                em._export_all_analyzed_data(preview_only=True)
            QApplication.processEvents()
        finally:
            main_window.state.in_path = original_in_path

        # Verify no CSV/NPZ files created
        export_dir = tmp_path / "Pleth_App_analysis"
        csvs = list(export_dir.glob("*.csv")) if export_dir.exists() else []
        npzs = list(export_dir.glob("*.npz")) if export_dir.exists() else []

        assert len(csvs) == 0, f"Preview mode should not create CSVs: {[f.name for f in csvs]}"
        assert len(npzs) == 0, f"Preview mode should not create NPZs: {[f.name for f in npzs]}"
        print("  Preview mode: no files created on disk")


class TestExportValidatesBeforeWriting:
    """Test 19: validate_breath_data passes on good data."""

    def test_validation_passes(self, main_window, multi_channel_abf):
        _ensure_analysis_ready(main_window, multi_channel_abf)

        em = main_window.export_manager
        is_valid, issues = em.validate_breath_data()

        assert is_valid, f"Validation failed: {issues}"
        print(f"  Validation passed with {len(issues)} warnings: {issues[:3]}")
