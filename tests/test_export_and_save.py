"""
Export & Save tests — verify CSV export includes HR data and Ctrl+S round-trips.

Tests the full workflow:
1. Load ABF -> select pleth + EKG channels -> detect peaks -> detect EKG
2. Export to CSV -> verify HR/RR columns in timeseries CSV
3. Ctrl+S save (.pmx) -> verify file created with all data
4. Reload .pmx -> verify peaks, EKG results, channel assignments restored

Uses tests/data/26402007.abf:
    IN 1: pleth (breathing)
    IN 7: EKG (heart rate)
    IN 0: stim trigger

Run:  python -m pytest tests/test_export_and_save.py -v -s
"""

import sys
import csv
import json
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


# ── Helpers ──────────────────────────────────────────────────────

def _select_channel(main_window, combo_name, channel_name):
    """Select a channel by name in a combo box. Returns True if found."""
    combo = getattr(main_window, combo_name, None)
    if combo is None:
        return False
    for i in range(combo.count()):
        if channel_name in combo.itemText(i):
            combo.setCurrentIndex(i)
            QApplication.processEvents()
            return True
    return False


def _setup_analysis(main_window, file_info):
    """Load file, select channels, detect peaks + EKG. Returns peak count."""
    load_file_and_wait(main_window, file_info.path)

    pleth_ch = file_info.pleth_channels[0]
    _select_channel(main_window, "AnalyzeChanSelect", pleth_ch)

    stim_ch = file_info.stim_channels[0]
    _select_channel(main_window, "StimChanSelect", stim_ch)

    auto_fn = getattr(main_window, "_auto_detect_prominence_silent", None)
    if auto_fn:
        auto_fn()
        QApplication.processEvents()

    if not main_window.peak_prominence:
        main_window.peak_prominence = 0.05
        main_window.peak_height_threshold = 0.05
        btn = getattr(main_window, "ApplyPeakFindPushButton", None)
        if btn:
            btn.setEnabled(True)

    main_window._apply_peak_detection()
    QApplication.processEvents()

    peaks = main_window.state.all_peaks_by_sweep
    if isinstance(peaks, dict):
        total_peaks = sum(len(p) for p in peaks.values())
    else:
        total_peaks = sum(len(p) for p in peaks if p is not None)

    ekg_ch = file_info.ekg_channels[0]
    main_window.state.ekg_chan = ekg_ch
    QApplication.processEvents()

    ekg_fn = getattr(main_window, "_auto_detect_ekg_current_sweep", None)
    if ekg_fn:
        ekg_fn()
        QApplication.processEvents()

    return total_peaks


def _run_export(main_window, output_dir, base_name, save_flags=None):
    """
    Run the export pipeline, bypassing the SaveMetaDialog.

    Patches SaveMetaDialog to auto-accept with given flags.
    Saves to output_dir/Pleth_App_analysis/ (the standard export location).
    Returns the actual output directory.
    """
    em = main_window.export_manager

    flags = {
        "preview": base_name,
        "experiment_type": "30hz_stim",
        "choose_dir": False,
        "save_npz": False,
        "save_timeseries_csv": True,
        "save_breaths_csv": True,
        "save_events_csv": True,
        "save_pdf": False,
        "save_session": False,
        "save_ml_training": False,
    }
    if save_flags:
        flags.update(save_flags)

    mock_dlg = MagicMock()
    mock_dlg.exec.return_value = QDialog.DialogCode.Accepted
    mock_dlg.values.return_value = flags

    # The export creates Pleth_App_analysis/ under the ABF's parent dir.
    # Temporarily point in_path to a copy in output_dir so files go there.
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

    # Files go into output_dir/Pleth_App_analysis/
    export_dir = output_dir / "Pleth_App_analysis"
    return export_dir


# ── Test: Full analysis setup ────────────────────────────────────


class TestAnalysisSetup:
    """Set up pleth + EKG analysis (runs first, other tests depend on it)."""

    def test_setup_analysis(self, main_window, multi_channel_abf):
        """Load file, detect peaks and EKG R-peaks."""
        total_peaks = _setup_analysis(main_window, multi_channel_abf)
        assert total_peaks > 0, "No breathing peaks detected"
        print(f"  Pleth: {total_peaks} peaks detected")

    def test_ekg_results_present(self, main_window):
        """EKG results should be populated after detection."""
        results = main_window.state.ecg_results_by_sweep
        assert results, "No EKG results in state"
        sweep_0 = results.get(0) or results.get(main_window.state.sweep_idx)
        assert sweep_0 is not None, "No EKG result for current sweep"
        assert hasattr(sweep_0, 'r_peaks'), "ECGResult missing r_peaks"
        n_rpeaks = len(sweep_0.r_peaks)
        assert n_rpeaks > 0, "No R-peaks detected"
        print(f"  EKG: {n_rpeaks} R-peaks on sweep {main_window.state.sweep_idx}")


# ── Test: CSV Export ─────────────────────────────────────────────


class TestCSVExport:
    """Test that CSV export includes HR data."""

    def test_export_creates_files(self, main_window, multi_channel_abf, tmp_path):
        """Export should create timeseries, breaths, events, and HR CSVs."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_export",
                                 {"save_hr_csv": True})

        # Check files exist
        timeseries = list(export_dir.glob("*_timeseries.csv")) or list(export_dir.glob("*_means_by_time.csv"))
        breaths = list(export_dir.glob("*_breaths.csv"))
        events = list(export_dir.glob("*_events.csv"))
        hr = list(export_dir.glob("*_hr.csv"))

        all_files = list(export_dir.iterdir()) if export_dir.exists() else []
        print(f"  Export dir: {export_dir}")
        print(f"  Files: {[f.name for f in all_files]}")

        assert len(timeseries) > 0, f"No timeseries CSV. Files: {[f.name for f in all_files]}"
        assert len(breaths) > 0, f"No breaths CSV. Files: {[f.name for f in all_files]}"
        assert len(events) > 0, f"No events CSV. Files: {[f.name for f in all_files]}"
        assert len(hr) > 0, f"No HR CSV. Files: {[f.name for f in all_files]}"

    def test_hr_csv_has_beats(self, main_window, multi_channel_abf, tmp_path):
        """HR CSV should have per-beat R-peak data with BPM and RR intervals."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_hr_csv",
                                 {"save_hr_csv": True, "save_timeseries_csv": False,
                                  "save_breaths_csv": False, "save_events_csv": False})

        hr_files = list(export_dir.glob("*_hr.csv"))
        assert hr_files, "No HR CSV created"

        with open(hr_files[0], newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "HR CSV is empty"
        assert "sweep" in rows[0], f"Missing 'sweep'. Cols: {list(rows[0].keys())}"
        assert "beat" in rows[0], f"Missing 'beat'"
        assert "t" in rows[0], f"Missing 't'"
        assert "rr_interval_ms" in rows[0], f"Missing 'rr_interval_ms'"
        assert "hr_bpm" in rows[0], f"Missing 'hr_bpm'"
        assert "sample_index" in rows[0], f"Missing 'sample_index'"

        bpm_values = [float(r["hr_bpm"]) for r in rows if r["hr_bpm"]]
        print(f"  HR CSV: {len(rows)} beats, BPM range: {min(bpm_values):.0f}-{max(bpm_values):.0f}")
        assert len(bpm_values) > 0

    def test_timeseries_csv_has_hr_columns(self, main_window, multi_channel_abf, tmp_path):
        """Timeseries CSV should contain HR and RR interval columns."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_hr",
                                 {"save_breaths_csv": False, "save_events_csv": False})

        timeseries_files = list(export_dir.glob("*_timeseries.csv")) or list(export_dir.glob("*_means_by_time.csv"))
        assert timeseries_files, "Timeseries CSV not created"
        timeseries_csv = timeseries_files[0]

        with open(timeseries_csv, newline='') as f:
            reader = csv.reader(f)
            headers = next(reader)

        hr_cols = [h for h in headers if 'hr' in h.lower()]
        rr_cols = [h for h in headers if 'rr_interval' in h.lower()]

        print(f"  HR columns: {hr_cols[:5]}")
        print(f"  RR columns: {rr_cols[:5]}")

        assert len(hr_cols) > 0, \
            f"No HR columns in timeseries CSV. Headers: {headers[:20]}..."
        assert len(rr_cols) > 0, \
            f"No RR interval columns. Headers: {headers[:20]}..."

    def test_timeseries_csv_hr_has_values(self, main_window, multi_channel_abf, tmp_path):
        """HR column should have actual numeric values, not all NaN/empty."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_hr_vals",
                                 {"save_breaths_csv": False, "save_events_csv": False})

        timeseries_files = list(export_dir.glob("*_timeseries.csv")) or list(export_dir.glob("*_means_by_time.csv"))
        assert timeseries_files, "Timeseries CSV not created"

        with open(timeseries_files[0], newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Find an HR mean column
        hr_mean_col = None
        for col in rows[0].keys():
            if 'hr_mean' in col.lower() and 'norm' not in col.lower():
                hr_mean_col = col
                break

        if hr_mean_col is None:
            pytest.fail(f"No hr_mean column found. Columns: {list(rows[0].keys())[:20]}")

        values = []
        for row in rows:
            val = row[hr_mean_col]
            if val and val.strip() and val.lower() != 'nan':
                try:
                    values.append(float(val))
                except ValueError:
                    pass

        print(f"  {hr_mean_col}: {len(values)}/{len(rows)} rows with values")
        if values:
            print(f"  HR range: {min(values):.0f} - {max(values):.0f} BPM")

        assert len(values) > 0, \
            f"HR column '{hr_mean_col}' has no numeric values (all NaN/empty)"

    def test_breaths_csv_content(self, main_window, multi_channel_abf, tmp_path):
        """Breaths CSV should have per-breath metrics with proper columns and values."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_breaths",
                                 {"save_timeseries_csv": False, "save_events_csv": False})

        breaths_files = list(export_dir.glob("*_breaths.csv"))
        assert breaths_files, "No breaths CSV created"

        with open(breaths_files[0], newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "Breaths CSV is empty"

        # Check required columns
        required = ['sweep', 'breath', 't', 'region', 'is_sigh', 'is_sniffing',
                     'is_eupnea', 'is_apnea', 'if', 'amp_insp', 'amp_exp',
                     'ti', 'te', 'vent_proxy', 'area_insp', 'area_exp']
        cols = list(rows[0].keys())
        missing = [c for c in required if c not in cols]
        assert not missing, f"Missing columns in breaths CSV: {missing}"

        # Check values are numeric and reasonable
        sweep1 = [r for r in rows if r.get('sweep') == '1']
        assert len(sweep1) > 0, "No rows for sweep 1"

        # Breathing rate should be positive
        if_vals = [float(r['if']) for r in sweep1 if r['if'] and r['if'] != 'nan']
        assert len(if_vals) > 0, "No breathing rate values in sweep 1"
        assert all(v > 0 for v in if_vals), "Negative breathing rate found"

        # ti and te should be positive durations
        ti_vals = [float(r['ti']) for r in sweep1 if r['ti'] and r['ti'] != 'nan']
        te_vals = [float(r['te']) for r in sweep1 if r['te'] and r['te'] != 'nan']
        assert len(ti_vals) > 0, "No ti values"
        assert len(te_vals) > 0, "No te values"

        # Check multiple sweeps present
        sweeps = set(r['sweep'] for r in rows)
        print(f"  Breaths CSV: {len(rows)} rows, {len(sweeps)} sweeps, "
              f"{len(sweep1)} breaths in sweep 1")
        print(f"  IF range: {min(if_vals):.1f}-{max(if_vals):.1f} Hz")
        print(f"  Ti range: {min(ti_vals)*1000:.0f}-{max(ti_vals)*1000:.0f} ms")

    def test_events_csv_content(self, main_window, multi_channel_abf, tmp_path):
        """Events CSV should have stim, eupnea, sniffing, and apnea events."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        export_dir = _run_export(main_window, tmp_path, "test_events",
                                 {"save_timeseries_csv": False, "save_breaths_csv": False})

        events_files = list(export_dir.glob("*_events.csv"))
        assert events_files, "No events CSV created"

        with open(events_files[0], newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "Events CSV is empty"

        # Check required columns
        required = ['sweep', 'event_type', 'start_time', 'end_time', 'duration']
        cols = list(rows[0].keys())
        missing = [c for c in required if c not in cols]
        assert not missing, f"Missing columns in events CSV: {missing}"

        # Count event types
        types = {}
        for r in rows:
            et = r.get('event_type', '?')
            types[et] = types.get(et, 0) + 1

        print(f"  Events CSV: {len(rows)} rows, types: {types}")

        # Should have at least stimulus events
        assert 'stimulus' in types, f"No stimulus events. Types: {types}"

        # Stim duration should be ~15s for this protocol
        stim_rows = [r for r in rows if r['event_type'] == 'stimulus']
        if stim_rows:
            dur = float(stim_rows[0]['duration'])
            assert 10 < dur < 20, f"Stim duration {dur}s unexpected (expected ~15s)"
            print(f"  Stim duration: {dur:.1f}s")


# ── Test: Ctrl+S Save (.pmx) ────────────────────────────────────


class TestSessionSave:
    """Test that Ctrl+S saves all pleth + EKG data to .pmx."""

    def test_save_session_creates_file(self, main_window, multi_channel_abf, tmp_path):
        """Ctrl+S should create a .pmx file."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        original_in_path = main_window.state.in_path
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_dir = tmp_path / "physiometrics"
            pmx_files = list(pmx_dir.glob("*.pmx")) if pmx_dir.exists() else []

            assert len(pmx_files) > 0, \
                f"No .pmx created. Contents: {list(tmp_path.rglob('*'))}"

            pmx_path = pmx_files[0]
            print(f"  Saved: {pmx_path.name} ({pmx_path.stat().st_size / 1024:.0f} KB)")
            self.__class__._saved_pmx_path = pmx_path
            self.__class__._saved_tmp_abf = tmp_abf
        finally:
            main_window.state.in_path = original_in_path

    def test_pmx_contains_ekg_data(self):
        """The saved .pmx should contain EKG results."""
        pmx_path = getattr(self.__class__, '_saved_pmx_path', None)
        if pmx_path is None or not pmx_path.exists():
            pytest.skip("No .pmx from previous test")

        data = np.load(pmx_path, allow_pickle=True)
        keys = list(data.keys())

        assert 'ekg_chan' in keys, f"ekg_chan not in .pmx. Keys: {keys[:20]}"
        assert 'ecg_results_json' in keys, f"ecg_results_json not in .pmx"

        ekg_chan = str(data['ekg_chan'])
        assert ekg_chan != 'None', "ekg_chan saved as 'None'"
        print(f"  ekg_chan: {ekg_chan}")

        ecg_json = json.loads(str(data['ecg_results_json']))
        assert len(ecg_json) > 0, "ecg_results_json is empty"
        first_sweep = next(iter(ecg_json.values()))
        assert 'r_peaks' in first_sweep
        print(f"  ECG: {len(ecg_json)} sweep(s), {len(first_sweep['r_peaks'])} R-peaks")
        data.close()

    def test_pmx_contains_peaks(self):
        """The saved .pmx should contain breathing peaks."""
        pmx_path = getattr(self.__class__, '_saved_pmx_path', None)
        if pmx_path is None or not pmx_path.exists():
            pytest.skip("No .pmx from previous test")

        data = np.load(pmx_path, allow_pickle=True)
        peak_keys = [k for k in data.keys() if 'peak' in k.lower() or 'onset' in k.lower()]
        print(f"  Peak keys: {peak_keys[:10]}")
        assert len(peak_keys) > 0
        data.close()

    def test_pmx_contains_channel_config(self):
        """The saved .pmx should remember channel assignments."""
        pmx_path = getattr(self.__class__, '_saved_pmx_path', None)
        if pmx_path is None or not pmx_path.exists():
            pytest.skip("No .pmx from previous test")

        data = np.load(pmx_path, allow_pickle=True)
        assert 'analyze_chan' in data.keys()
        analyze_chan = str(data['analyze_chan'])
        assert analyze_chan != 'None'
        print(f"  analyze_chan: {analyze_chan}")

        if 'channel_config_json' in data.keys():
            ch_config = json.loads(str(data['channel_config_json']))
            print(f"  Channel config: {list(ch_config.keys())[:5]}")
        data.close()


# ── Test: Reload .pmx ────────────────────────────────────────────


class TestSessionReload:
    """Test that reloading a .pmx restores all data."""

    def test_reload_restores_everything(self, main_window, multi_channel_abf, tmp_path):
        """Save -> reload should restore peaks, EKG, and channel assignments."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        # Save
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original_in_path = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        # Record pre-save state
        pre_peaks = sum(
            len(p) for p in (main_window.state.all_peaks_by_sweep.values()
                             if isinstance(main_window.state.all_peaks_by_sweep, dict)
                             else main_window.state.all_peaks_by_sweep)
        )
        pre_ekg_chan = main_window.state.ekg_chan
        pre_analyze_chan = main_window.state.analyze_chan

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_dir = tmp_path / "physiometrics"
            pmx_files = list(pmx_dir.glob("*.pmx"))
            assert pmx_files, "No .pmx created"
            pmx_path = pmx_files[0]

            # Reload
            main_window.load_npz_state(pmx_path)
            QApplication.processEvents()

            import time
            deadline = time.time() + 10
            while time.time() < deadline:
                QApplication.processEvents()
                if main_window.state.all_peaks_by_sweep:
                    break
                time.sleep(0.1)

            # Verify peaks
            post_peaks_data = main_window.state.all_peaks_by_sweep
            assert post_peaks_data is not None
            post_peaks = sum(
                len(p) for p in (post_peaks_data.values()
                                 if isinstance(post_peaks_data, dict)
                                 else post_peaks_data)
            )
            print(f"  Peaks: {pre_peaks} saved -> {post_peaks} restored")
            assert post_peaks > 0

            # Verify EKG
            post_ekg = main_window.state.ecg_results_by_sweep
            assert post_ekg, "EKG results not restored"
            for s, r in post_ekg.items():
                if hasattr(r, 'r_peaks'):
                    print(f"  EKG sweep {s}: {len(r.r_peaks)} R-peaks restored")

            # Verify channels
            assert main_window.state.analyze_chan is not None
            print(f"  analyze_chan: {main_window.state.analyze_chan}")
            print(f"  ekg_chan: {main_window.state.ekg_chan}")
            if pre_ekg_chan:
                assert main_window.state.ekg_chan == pre_ekg_chan, \
                    f"ekg_chan changed: {pre_ekg_chan} -> {main_window.state.ekg_chan}"
        finally:
            main_window.state.in_path = original_in_path

    def test_reload_restores_channel_visibility(self, main_window, multi_channel_abf, tmp_path):
        """After reload, only previously-visible channels should be visible."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        # Record which channels are visible before save
        pre_visible = {}
        if hasattr(main_window, 'channel_manager') and main_window.channel_manager:
            for ch_name, ch_cfg in main_window.channel_manager.get_channels().items():
                pre_visible[ch_name] = ch_cfg.visible

        print(f"  Pre-save visible: {[k for k,v in pre_visible.items() if v]}")

        # Save
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original_in_path = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_path = list((tmp_path / "physiometrics").glob("*.pmx"))[0]

            main_window.load_npz_state(pmx_path)
            import time
            for _ in range(50):
                QApplication.processEvents()
                time.sleep(0.05)

            post_visible = {}
            for ch_name, ch_cfg in main_window.channel_manager.get_channels().items():
                post_visible[ch_name] = ch_cfg.visible

            print(f"  Post-reload visible: {[k for k,v in post_visible.items() if v]}")

            for ch_name, was_visible in pre_visible.items():
                if ch_name in post_visible:
                    assert post_visible[ch_name] == was_visible, \
                        f"'{ch_name}' visibility changed: {was_visible} -> {post_visible[ch_name]}"
        finally:
            main_window.state.in_path = original_in_path

    def test_reload_restores_ekg_channel_type(self, main_window, multi_channel_abf, tmp_path):
        """After reload, the EKG channel should still have type 'EKG'."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        ekg_ch = multi_channel_abf.ekg_channels[0]
        if hasattr(main_window, 'channel_manager'):
            main_window.channel_manager.set_channel_type(ekg_ch, "EKG")
            QApplication.processEvents()

        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original_in_path = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_path = list((tmp_path / "physiometrics").glob("*.pmx"))[0]

            main_window.load_npz_state(pmx_path)
            import time
            for _ in range(50):
                QApplication.processEvents()
                time.sleep(0.05)

            channels = main_window.channel_manager.get_channels()
            ekg_cfg = channels.get(ekg_ch)
            if ekg_cfg:
                print(f"  '{ekg_ch}' type after reload: {ekg_cfg.channel_type}")
                assert ekg_cfg.channel_type == "EKG", \
                    f"EKG type not restored: {ekg_cfg.channel_type}"
            assert main_window.state.ekg_chan == ekg_ch, \
                f"state.ekg_chan = {main_window.state.ekg_chan}, expected {ekg_ch}"
        finally:
            main_window.state.in_path = original_in_path

    def test_loading_new_file_clears_ekg_state(self, main_window, multi_channel_abf):
        """Loading a different file should clear EKG state completely."""
        if not main_window.state.ecg_results_by_sweep:
            _setup_analysis(main_window, multi_channel_abf)

        assert main_window.state.ekg_chan is not None
        assert len(main_window.state.ecg_results_by_sweep) > 0

        # Load the legacy file (no EKG)
        from conftest import LEGACY_ABF
        load_file_and_wait(main_window, LEGACY_ABF.path)

        print(f"  After new file: ekg_chan={main_window.state.ekg_chan}, "
              f"ecg_results={len(main_window.state.ecg_results_by_sweep)}")

        assert main_window.state.ekg_chan is None, \
            f"ekg_chan not cleared: {main_window.state.ekg_chan}"
        assert len(main_window.state.ecg_results_by_sweep) == 0, \
            f"ecg_results not cleared: {len(main_window.state.ecg_results_by_sweep)} entries"

        # Reload test file for subsequent tests
        _setup_analysis(main_window, multi_channel_abf)
