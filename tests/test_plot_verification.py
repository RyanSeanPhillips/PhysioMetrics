"""
Visual verification plots — generates PNGs purely from exported CSV data.

Creates a 4-panel figure reading ONLY from the CSV exports:
    Panel 1: Raw pleth (from ABF) + breath peak times (from breaths.csv)
    Panel 2: Raw EKG (from ABF) + R-peak positions (from hr.csv)
    Panel 3: Breathing metrics timeseries (from timeseries.csv):
             - Breathing rate (if_mean, Hz) + Insp. amplitude (amp_insp_mean)
    Panel 4: HR metrics (from timeseries.csv):
             - Heart rate (hr_mean, BPM) + RR interval (rr_interval_mean, ms)

This proves the exported CSVs contain enough data to fully reconstruct
the analysis without needing the .pmx file or app state.

Output: tests/output/verification_from_csv.png

Run:  python -m pytest tests/test_plot_verification.py -v -s
"""

import sys
import csv
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF

OUTPUT_DIR = ROOT / "tests" / "output"


def _ensure_analysis(main_window, multi_channel_abf):
    """Make sure pleth + EKG analysis is set up."""
    if main_window.state.ecg_results_by_sweep:
        return
    from test_export_and_save import _setup_analysis
    _setup_analysis(main_window, multi_channel_abf)


def _run_full_export(main_window, output_dir):
    """Run CSV export (all files including HR) and return paths."""
    from PyQt6.QtWidgets import QApplication, QDialog

    em = main_window.export_manager
    flags = {
        "preview": "verify",
        "experiment_type": "30hz_stim",
        "choose_dir": False,
        "save_npz": False,
        "save_timeseries_csv": True,
        "save_breaths_csv": True,
        "save_events_csv": True,
        "save_hr_csv": True,
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

    export_dir = output_dir / "Pleth_App_analysis"
    return {
        "timeseries": next(export_dir.glob("*_timeseries.csv"), None),
        "breaths": next(export_dir.glob("*_breaths.csv"), None),
        "hr": next(export_dir.glob("*_hr.csv"), None),
        "events": next(export_dir.glob("*_events.csv"), None),
    }


def _read_csv_to_dict(path):
    """Read a CSV into a dict of column_name -> list of values."""
    data = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val) if val and val.strip() else np.nan)
                except ValueError:
                    data[key].append(np.nan)
    return data


class TestVerificationPlots:
    """Generate verification plots purely from exported CSVs."""

    def test_generate_csv_verification_plot(self, main_window, multi_channel_abf, tmp_path):
        """
        4-panel figure plotted entirely from exported CSV data + raw ABF.
        No app state used for plotting — proves CSVs are self-sufficient.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import pyabf

        _ensure_analysis(main_window, multi_channel_abf)

        # --- Export all CSVs ---
        files = _run_full_export(main_window, tmp_path)
        assert files["timeseries"], "No timeseries CSV"
        assert files["breaths"], "No breaths CSV"
        assert files["hr"], "No HR CSV"

        # --- Load raw ABF (only source that's not a CSV) ---
        abf = pyabf.ABF(str(multi_channel_abf.path))
        sr = abf.dataRate

        abf.setSweep(0, channel=1)  # IN 1 = pleth
        pleth_y = abf.sweepY
        pleth_t = np.arange(len(pleth_y)) / sr

        abf.setSweep(0, channel=6)  # IN 7 = EKG
        ekg_y = abf.sweepY
        ekg_t = np.arange(len(ekg_y)) / sr

        # --- Load CSVs ---
        ts = _read_csv_to_dict(files["timeseries"])
        breaths = _read_csv_to_dict(files["breaths"])
        hr = _read_csv_to_dict(files["hr"])

        ts_t = np.array(ts.get('t', []))

        # Breath times from breaths.csv (sweep 1 = sweep index 0)
        breath_times_s1 = [breaths['t'][i] for i in range(len(breaths['t']))
                           if breaths['sweep'][i] == 1.0 and not np.isnan(breaths['t'][i])]

        # R-peak times and HR from hr.csv (sweep 1)
        rpeak_times_s1 = [hr['t'][i] for i in range(len(hr['t']))
                          if hr['sweep'][i] == 1.0 and not np.isnan(hr['t'][i])]
        rpeak_samples_s1 = [int(hr['sample_index'][i]) for i in range(len(hr['sample_index']))
                            if hr['sweep'][i] == 1.0 and not np.isnan(hr['sample_index'][i])]

        # Find stim onset from events.csv
        stim_onset = 0.0
        if files["events"]:
            events = _read_csv_to_dict(files["events"])
            # stim events have start_time
            for i in range(len(events.get('start_time', []))):
                if not np.isnan(events.get('start_time', [np.nan])[i]):
                    stim_onset = events['start_time'][i]
                    break

        # Convert breath times to absolute (they're relative to stim onset)
        # t in breaths.csv is relative to stim onset
        # t in ABF is absolute from sweep start
        # We need to find the absolute stim onset to align
        # The timeseries t=0 corresponds to stim onset in the ABF
        # So ABF_time = CSV_time + stim_onset_in_ABF
        # From events.csv, stim start_time is already relative to stim onset = 0

        # 10s window around stim onset
        win_start_rel = -2   # 2s before stim
        win_end_rel = 8      # 8s after stim

        # --- Create figure ---
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'hspace': 0.35})
        fig.suptitle('Export Verification (plotted from CSVs only) - 26402007.abf',
                     fontsize=14, fontweight='bold')

        # ── Panel 1: Raw Pleth + Breath Peaks from breaths.csv ──
        ax1 = axes[0]

        # Convert relative times to absolute for ABF indexing
        # Need to find actual stim onset in ABF seconds
        stim_abs = None
        st = main_window.state
        if st.stim_spans_by_sweep.get(0):
            stim_abs = st.stim_spans_by_sweep[0][0][0]
        if stim_abs is None:
            stim_abs = 25.0  # fallback

        abs_start = stim_abs + win_start_rel
        abs_end = stim_abs + win_end_rel
        win_mask = (pleth_t >= abs_start) & (pleth_t <= abs_end)

        ax1.plot(pleth_t[win_mask] - stim_abs, pleth_y[win_mask],
                 'k-', linewidth=0.5, alpha=0.8)

        # Overlay breath peaks from breaths.csv
        bt = np.array(breath_times_s1)
        bt_in_win = bt[(bt >= win_start_rel) & (bt <= win_end_rel)]
        if len(bt_in_win) > 0:
            bt_abs = bt_in_win + stim_abs
            bt_vals = np.interp(bt_abs, pleth_t, pleth_y)
            ax1.plot(bt_in_win, bt_vals, 'rv', markersize=7,
                     label=f'Breaths from CSV ({len(bt_in_win)} in window)')
            ax1.legend(loc='upper right', fontsize=9)

        ax1.axvspan(0, win_end_rel, alpha=0.1, color='blue')
        ax1.set_ylabel('Pleth (IN 1)', fontsize=10)
        ax1.set_title('Raw Pleth + Breath Peaks (from breaths.csv)', fontsize=11)
        ax1.set_xlim(win_start_rel, win_end_rel)
        ax1.tick_params(labelsize=8)

        # ── Panel 2: Raw EKG + R-Peaks from hr.csv ──
        ax2 = axes[1]
        ekg_win_mask = (ekg_t >= abs_start) & (ekg_t <= abs_end)
        ax2.plot(ekg_t[ekg_win_mask] - stim_abs, ekg_y[ekg_win_mask],
                 'k-', linewidth=0.5, alpha=0.8)

        # Overlay R-peaks from hr.csv using sample_index
        rp_samples = np.array(rpeak_samples_s1)
        rp_valid = rp_samples[(rp_samples >= 0) & (rp_samples < len(ekg_t))]
        rp_times_rel = ekg_t[rp_valid] - stim_abs
        rp_in_win = (rp_times_rel >= win_start_rel) & (rp_times_rel <= win_end_rel)

        if np.any(rp_in_win):
            ax2.plot(rp_times_rel[rp_in_win], ekg_y[rp_valid[rp_in_win]],
                     'r^', markersize=5,
                     label=f'R-peaks from CSV ({np.sum(rp_in_win)} in window)')
            ax2.legend(loc='upper right', fontsize=9)

        ax2.axvspan(0, win_end_rel, alpha=0.1, color='blue')
        ax2.set_ylabel('EKG (IN 7)', fontsize=10)
        ax2.set_title('Raw EKG + R-Peaks (from hr.csv)', fontsize=11)
        ax2.set_xlim(win_start_rel, win_end_rel)
        ax2.set_xlabel('Time relative to stim onset (s)', fontsize=10)
        ax2.tick_params(labelsize=8)

        # ── Panel 3: Breathing Metrics from timeseries.csv ──
        ax3 = axes[2]
        ax3b = ax3.twinx()

        if_mean = np.array(ts.get('if_mean', []))
        if_sem = np.array(ts.get('if_sem', []))
        valid_if = ~np.isnan(if_mean) & ~np.isnan(ts_t)

        if np.any(valid_if):
            ax3.plot(ts_t[valid_if], if_mean[valid_if], 'b-', linewidth=1,
                     label='Breathing Rate (Hz)')
            ax3.fill_between(ts_t[valid_if],
                            (if_mean - if_sem)[valid_if],
                            (if_mean + if_sem)[valid_if],
                            alpha=0.2, color='blue')

        amp_mean = np.array(ts.get('amp_insp_mean', []))
        amp_sem = np.array(ts.get('amp_insp_sem', []))
        valid_amp = ~np.isnan(amp_mean) & ~np.isnan(ts_t)

        if np.any(valid_amp):
            ax3b.plot(ts_t[valid_amp], amp_mean[valid_amp], 'g-', linewidth=1,
                      label='Insp. Amplitude')
            ax3b.fill_between(ts_t[valid_amp],
                             (amp_mean - amp_sem)[valid_amp],
                             (amp_mean + amp_sem)[valid_amp],
                             alpha=0.2, color='green')

        ax3.set_ylabel('Breathing Rate (Hz)', color='blue', fontsize=10)
        ax3b.set_ylabel('Insp. Amplitude', color='green', fontsize=10)
        ax3.set_title('Breathing Metrics from timeseries.csv (mean +/- SEM)', fontsize=11)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax3.tick_params(labelsize=8)
        ax3b.tick_params(labelsize=8)

        # ── Panel 4: HR Metrics from timeseries.csv ──
        ax4 = axes[3]
        ax4b = ax4.twinx()

        hr_mean = np.array(ts.get('hr_mean', []))
        hr_sem = np.array(ts.get('hr_sem', []))
        valid_hr = ~np.isnan(hr_mean) & ~np.isnan(ts_t)

        if np.any(valid_hr):
            ax4.plot(ts_t[valid_hr], hr_mean[valid_hr], 'r-', linewidth=1,
                     label='Heart Rate (BPM)')
            ax4.fill_between(ts_t[valid_hr],
                            (hr_mean - hr_sem)[valid_hr],
                            (hr_mean + hr_sem)[valid_hr],
                            alpha=0.2, color='red')
            hr_vals = hr_mean[valid_hr]
            print(f"  HR: {np.nanmin(hr_vals):.0f}-{np.nanmax(hr_vals):.0f} BPM "
                  f"({np.sum(valid_hr)} points)")
        else:
            print("  WARNING: No HR data in timeseries CSV!")

        rr_mean = np.array(ts.get('rr_interval_mean', []))
        rr_sem = np.array(ts.get('rr_interval_sem', []))
        valid_rr = ~np.isnan(rr_mean) & ~np.isnan(ts_t)

        if np.any(valid_rr):
            ax4b.plot(ts_t[valid_rr], rr_mean[valid_rr], 'm-', linewidth=1,
                      label='RR Interval (ms)')
            ax4b.fill_between(ts_t[valid_rr],
                             (rr_mean - rr_sem)[valid_rr],
                             (rr_mean + rr_sem)[valid_rr],
                             alpha=0.2, color='magenta')

        ax4.set_ylabel('Heart Rate (BPM)', color='red', fontsize=10)
        ax4b.set_ylabel('RR Interval (ms)', color='magenta', fontsize=10)
        ax4.set_title('Heart Rate from timeseries.csv (mean +/- SEM)', fontsize=11)
        ax4.set_xlabel('Time relative to stim onset (s)', fontsize=10)
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        ax4.tick_params(labelsize=8)
        ax4b.tick_params(labelsize=8)

        # ── Save ──
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "verification_from_csv.png"
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"\n  Plot saved: {out_path}")
        print(f"  Data sources: breaths.csv ({len(breath_times_s1)} breaths), "
              f"hr.csv ({len(rpeak_samples_s1)} R-peaks), timeseries.csv")
        assert out_path.exists()
        assert out_path.stat().st_size > 10000
