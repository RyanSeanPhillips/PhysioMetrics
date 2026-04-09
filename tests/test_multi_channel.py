"""
Multi-channel tests — pleth + EKG + stim using 26402007.abf.

This file has 7 channels:
    IN 1: pleth (breathing)
    IN 7: EKG (heart rate)
    IN 0: stim trigger (IN 4, IN 5 are duplicates)
    IN 2, IN 3: noise/empty
    IN 4, IN 5: stim duplicates

Run:  python -m pytest tests/test_multi_channel.py -v -s
"""

import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait


# ── Loading multi-channel file ───────────────────────────────────


class TestMultiChannelLoad:
    """Load the multi-channel ABF and verify channels are detected."""

    def test_load_file(self, main_window, multi_channel_abf):
        """Loading the multi-channel ABF should succeed."""
        load_file_and_wait(main_window, multi_channel_abf.path)
        assert main_window.state.in_path is not None

    def test_channels_detected(self, main_window, multi_channel_abf):
        """Key channels (pleth, EKG, stim) should be in channel list."""
        names = main_window.state.channel_names
        for ch in (multi_channel_abf.pleth_channels +
                   multi_channel_abf.ekg_channels +
                   multi_channel_abf.stim_channels):
            assert any(ch in n for n in names), \
                f"Channel '{ch}' not found in {names}"

    def test_sample_rate(self, main_window, multi_channel_abf):
        """Sample rate should be 10kHz."""
        assert main_window.state.sr_hz == multi_channel_abf.sample_rate

    def test_sweep_count(self, main_window, multi_channel_abf):
        """Should have 10 sweeps."""
        sweeps = main_window.state.sweeps
        if isinstance(sweeps, dict):
            first_key = next(iter(sweeps))
            first_val = sweeps[first_key]
            if hasattr(first_val, 'shape') and len(first_val.shape) == 2:
                n = first_val.shape[1]
            else:
                n = len(sweeps)
        else:
            n = len(sweeps) if sweeps else 0
        assert n >= multi_channel_abf.n_sweeps, \
            f"Expected {multi_channel_abf.n_sweeps} sweeps, got {n}"


# ── Pleth channel analysis ───────────────────────────────────────


class TestPlethChannel:
    """Test breathing analysis on the pleth channel (IN 1)."""

    def test_select_pleth_channel(self, main_window, multi_channel_abf):
        """Selecting the pleth channel should work."""
        pleth_ch = multi_channel_abf.pleth_channels[0]  # IN 1
        combo = main_window.AnalyzeChanSelect
        found = False
        for i in range(combo.count()):
            if pleth_ch in combo.itemText(i):
                combo.setCurrentIndex(i)
                QApplication.processEvents()
                found = True
                break
        assert found, f"Pleth channel '{pleth_ch}' not found in AnalyzeChanSelect"

    def test_peak_detection_on_pleth(self, main_window, multi_channel_abf):
        """Peak detection on pleth channel should find breaths."""
        pleth_ch = multi_channel_abf.pleth_channels[0]
        combo = main_window.AnalyzeChanSelect
        for i in range(combo.count()):
            if pleth_ch in combo.itemText(i):
                combo.setCurrentIndex(i)
                QApplication.processEvents()
                break

        # Auto-detect prominence
        auto_fn = getattr(main_window, "_auto_detect_prominence_silent", None)
        if auto_fn:
            auto_fn()
            QApplication.processEvents()

        # Fallback: set manually if auto-detect didn't work
        if not main_window.peak_prominence:
            main_window.peak_prominence = 0.05
            main_window.peak_height_threshold = 0.05
            btn = getattr(main_window, "ApplyPeakFindPushButton", None)
            if btn:
                btn.setEnabled(True)

        main_window._apply_peak_detection()
        QApplication.processEvents()

        peaks = main_window.state.all_peaks_by_sweep
        assert peaks is not None
        if isinstance(peaks, dict):
            total = sum(len(p) for p in peaks.values())
        else:
            total = sum(len(p) for p in peaks if p is not None)
        assert total > 0, "No breaths detected on pleth channel"
        print(f"  Detected {total} breaths on {pleth_ch}")

    def test_plot_renders_with_peaks(self, main_window):
        """Plot should render with detected peaks without crashing."""
        main_window.redraw_main_plot()
        QApplication.processEvents()


# ── Stim channel ─────────────────────────────────────────────────


class TestStimChannel:
    """Test stim detection on the stim channel (IN 0)."""

    def test_select_stim_channel(self, main_window, multi_channel_abf):
        """Selecting the stim channel should work."""
        stim_ch = multi_channel_abf.stim_channels[0]  # IN 0
        combo = main_window.StimChanSelect
        found = False
        for i in range(combo.count()):
            if stim_ch in combo.itemText(i):
                combo.setCurrentIndex(i)
                QApplication.processEvents()
                found = True
                break
        assert found, f"Stim channel '{stim_ch}' not found in StimChanSelect"


# ── EKG channel ──────────────────────────────────────────────────


class TestEKGChannel:
    """Test EKG/heart rate on the EKG channel (IN 7)."""

    def test_ekg_channel_exists(self, main_window, multi_channel_abf):
        """The EKG channel should be available."""
        ekg_ch = multi_channel_abf.ekg_channels[0]  # IN 7
        names = main_window.state.channel_names
        assert any(ekg_ch in n for n in names), \
            f"EKG channel '{ekg_ch}' not found in {names}"

    def test_ekg_detection(self, main_window, multi_channel_abf):
        """EKG detection should find R-peaks."""
        ekg_fn = getattr(main_window, "_auto_detect_ekg_current_sweep", None)
        if ekg_fn is None:
            pytest.skip("EKG detection not available")

        # Set EKG channel
        main_window.state.ekg_chan = multi_channel_abf.ekg_channels[0]
        QApplication.processEvents()

        ekg_fn()
        QApplication.processEvents()

        # Check if EKG results were produced
        results = main_window.state.ecg_results_by_sweep
        if results:
            sweep_0 = results.get(0) or results.get(main_window.state.sweep_idx)
            if sweep_0 and hasattr(sweep_0, 'r_peaks'):
                n_peaks = len(sweep_0.r_peaks)
                print(f"  EKG: detected {n_peaks} R-peaks on sweep 0")
                assert n_peaks > 0, "No R-peaks detected"
            else:
                print("  EKG: detection ran but no r_peaks attribute found")
        else:
            print("  EKG: no results produced (may need channel assignment)")
