"""Tests for editing button wiring from minimap nav bar.

Verifies that all editing mode buttons (add/del peaks, merge, move point,
omit) are properly aliased from the minimap and connected to their handlers.
"""
import sys
import os
import pytest

os.environ["PLETHAPP_TESTING"] = "1"

from PyQt6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def main_window():
    from main import MainWindow
    w = MainWindow()
    yield w
    w.close()


class TestEditingButtonWiring:
    """Verify buttons exist, are checkable, and are the same objects in EditingModes."""

    def test_buttons_exist(self, main_window):
        for name in ["addPeaksButton", "MergeBreathsButton", "movePointButton",
                      "OmitSweepButton", "editor_pushButton"]:
            btn = getattr(main_window, name, None)
            assert btn is not None, f"{name} not found on MainWindow"

    def test_buttons_are_checkable(self, main_window):
        for name in ["addPeaksButton", "MergeBreathsButton", "movePointButton",
                      "OmitSweepButton"]:
            btn = getattr(main_window, name)
            assert btn.isCheckable(), f"{name} is not checkable"

    def test_buttons_match_editing_modes(self, main_window):
        em = main_window.editing_modes
        assert em._btn_add_peaks is main_window.addPeaksButton
        assert em._btn_merge is main_window.MergeBreathsButton
        assert em._btn_move_point is main_window.movePointButton
        assert em._btn_omit is main_window.OmitSweepButton

    def test_buttons_come_from_minimap(self, main_window):
        minimap = getattr(main_window.plot_host, "_minimap_nav", None)
        assert minimap is not None
        assert main_window.addPeaksButton is minimap.btn_add_peaks
        assert main_window.MergeBreathsButton is minimap.btn_merge
        assert main_window.movePointButton is minimap.btn_move
        assert main_window.OmitSweepButton is minimap.btn_omit


class TestEditingModeToggle:
    """Verify that clicking buttons actually toggles the editing mode flags."""

    def test_add_peaks_toggle(self, main_window):
        em = main_window.editing_modes
        btn = main_window.addPeaksButton
        assert not em._add_peaks_mode
        btn.click()
        assert em._add_peaks_mode, "Add peaks mode did not activate"
        btn.click()
        assert not em._add_peaks_mode, "Add peaks mode did not deactivate"

    def test_merge_toggle(self, main_window):
        em = main_window.editing_modes
        btn = main_window.MergeBreathsButton
        assert not em._merge_peaks_mode
        btn.click()
        assert em._merge_peaks_mode, "Merge mode did not activate"
        btn.click()
        assert not em._merge_peaks_mode, "Merge mode did not deactivate"

    def test_move_point_toggle(self, main_window):
        em = main_window.editing_modes
        btn = main_window.movePointButton
        assert not em._move_point_mode
        btn.click()
        assert em._move_point_mode, "Move point mode did not activate"
        btn.click()
        assert not em._move_point_mode, "Move point mode did not deactivate"

    def test_mutual_exclusion(self, main_window):
        """Activating one mode should deactivate others."""
        em = main_window.editing_modes
        main_window.addPeaksButton.click()
        assert em._add_peaks_mode
        # Activate merge — should deactivate add peaks
        main_window.MergeBreathsButton.click()
        assert em._merge_peaks_mode
        assert not em._add_peaks_mode, "Add peaks was not deactivated by merge"
        # Clean up
        main_window.MergeBreathsButton.click()
