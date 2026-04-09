"""
GUI smoke tests — verify the app launches and basic interactions work.

The MainWindow is session-scoped (created once, reused).
PLETHAPP_TESTING=1 auto-loads examples/25121004.abf at startup.

Run:  python -m pytest tests/test_gui_smoke.py -v
"""

import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── App Launch ───────────────────────────────────────────────────


class TestAppLaunch:
    """Verify the app starts without crashing."""

    def test_window_created(self, main_window):
        assert main_window is not None
        assert main_window.isVisible()

    def test_window_title(self, main_window):
        assert "PhysioMetrics" in main_window.windowTitle()

    def test_app_state_initialized(self, main_window):
        assert main_window.state is not None

    def test_plot_host_exists(self, main_window):
        assert main_window.plot_host is not None

    def test_settings_loaded(self, main_window):
        assert main_window.settings is not None


# ── Tab Navigation ───────────────────────────────────────────────


class TestTabNavigation:
    """Test switching between main tabs (widget name is 'Tabs' in .ui)."""

    def test_main_tabs_exist(self, main_window):
        tabs = main_window.Tabs
        assert tabs is not None
        assert tabs.count() >= 2

    def test_switch_tabs_no_crash(self, main_window):
        tabs = main_window.Tabs
        original = tabs.currentIndex()
        for i in range(tabs.count()):
            tabs.setCurrentIndex(i)
            QApplication.processEvents()
            assert tabs.currentIndex() == i
        # Restore
        tabs.setCurrentIndex(original)
        QApplication.processEvents()


# ── Data Loaded (via PLETHAPP_TESTING auto-load) ─────────────────


class TestDataLoaded:
    """Tests that depend on the auto-loaded ABF file."""

    def test_data_present(self, main_window):
        has_data = (bool(main_window.state.sweeps) or
                    main_window.state.in_path is not None)
        assert has_data, "No data in state — auto-load may have failed"

    def test_sample_rate_set(self, main_window):
        assert main_window.state.sr_hz is not None
        assert main_window.state.sr_hz > 0

    def test_time_array_exists(self, main_window):
        assert main_window.state.t is not None
        assert len(main_window.state.t) > 0

    def test_plot_renders(self, main_window):
        """Redrawing the plot should not crash."""
        main_window.redraw_main_plot()
        QApplication.processEvents()


# ── Peak Detection ───────────────────────────────────────────────


class TestPeakDetection:
    """Test that peak detection runs on loaded data."""

    def test_auto_detect_then_peak_detection(self, main_window):
        """Auto-detect prominence, then run peak detection.

        NOTE: Auto-detect skips 'Raw Signal' channels. The TESTING MODE
        channel setup may leave the channel in raw mode. If prominence
        can't be auto-detected, we set a reasonable default and proceed.
        """
        # Try auto-detect first
        auto_fn = getattr(main_window, "_auto_detect_prominence_silent", None)
        if auto_fn is not None:
            auto_fn()
            QApplication.processEvents()

        # If auto-detect didn't set a threshold, set one manually
        if not main_window.peak_prominence:
            main_window.peak_prominence = 0.05
            main_window.peak_height_threshold = 0.05
            # Enable the apply button
            btn = getattr(main_window, "ApplyPeakFindPushButton", None)
            if btn is not None:
                btn.setEnabled(True)
            QApplication.processEvents()

        main_window._apply_peak_detection()
        QApplication.processEvents()

        peaks = main_window.state.all_peaks_by_sweep
        assert peaks is not None, "all_peaks_by_sweep is None after detection"
        if isinstance(peaks, dict):
            total = sum(len(p) for p in peaks.values())
        else:
            total = sum(len(p) for p in peaks if p is not None)
        assert total > 0, "No peaks detected"


# ── Sweep Navigation ────────────────────────────────────────────


class TestSweepNavigation:
    """Test navigating between sweeps via the navigation viewmodel."""

    def test_has_navigation(self, main_window):
        assert hasattr(main_window, "_nav_vm"), "NavigationViewModel not found"

    def test_sweep_count(self, main_window):
        """Should have at least one sweep after loading."""
        attr = getattr(main_window._nav_vm, "sweep_count", None)
        n = attr() if callable(attr) else attr
        if n is None:
            # Fallback: check state
            n = len(main_window.state.sweeps) if main_window.state.sweeps else 0
        assert n > 0, "No sweeps available"

    def test_navigate_sweep(self, main_window):
        """Changing sweep index should not crash."""
        vm = main_window._nav_vm
        if hasattr(vm, "set_sweep"):
            vm.set_sweep(0)
        elif hasattr(vm, "sweep_idx"):
            vm.sweep_idx = 0
        QApplication.processEvents()


# ── Export Manager ───────────────────────────────────────────────


class TestExportManager:

    def test_export_manager_accessible(self, main_window):
        em = (main_window._get_export_manager()
              if hasattr(main_window, "_get_export_manager")
              else getattr(main_window, "export_manager", None))
        assert em is not None, "Export manager not available"


# ── Channel Manager ─────────────────────────────────────────────


class TestChannelManager:

    def test_channel_manager_exists(self, main_window):
        assert hasattr(main_window, "channel_manager"), "Channel manager not found"

    def test_channels_detected(self, main_window):
        """After loading, channel names should be populated."""
        assert len(main_window.state.channel_names) > 0, "No channels detected"
