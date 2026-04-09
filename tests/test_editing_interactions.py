"""Tests for editing mode PLOT INTERACTIONS — not just button wiring.

Simulates actual mouse events on the plot to verify that omit regions
get created, peaks get added/deleted, points get moved, etc.
"""
import sys
import os
import time

# Ensure pyqt6 root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PLETHAPP_TESTING"] = "1"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF, QTimer, QEventLoop
from PyQt6.QtTest import QTest
from PyQt6.QtGui import QMouseEvent

app = QApplication.instance() or QApplication(sys.argv)


def wait_ms(ms=200):
    """Process events for a short time."""
    deadline = time.time() + ms / 1000.0
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def get_main_window():
    """Create MainWindow and wait for test file to load."""
    from main import MainWindow
    w = MainWindow()
    w.show()
    # Wait for auto-load to complete
    wait_ms(6000)
    return w


def _data_to_widget_pos(plot_host, x_frac, y_frac=0.5):
    """Convert fractional plot position to widget pixel coordinates."""
    main_plot = plot_host._get_main_plot()
    vb = main_plot.vb
    view_range = vb.viewRange()
    x_min, x_max = view_range[0]
    y_min, y_max = view_range[1]

    x_data = x_min + (x_max - x_min) * x_frac
    y_data = y_min + (y_max - y_min) * y_frac

    # Data -> scene -> view (widget) coordinates
    scene_pos = vb.mapViewToScene(QPointF(x_data, y_data))
    # scene -> graphics view widget coordinates
    gv = plot_host.graphics_layout.scene().views()[0]
    view_pos = gv.mapFromScene(scene_pos)

    return view_pos, x_data, gv


def simulate_drag_on_plot(plot_host, x_start_frac, x_end_frac, y_frac=0.5):
    """Simulate a mouse drag using QTest on the graphics view widget."""
    start_pos, x_start, gv = _data_to_widget_pos(plot_host, x_start_frac, y_frac)
    end_pos, x_end, _ = _data_to_widget_pos(plot_host, x_end_frac, y_frac)

    print(f"  Drag: widget ({start_pos.x()},{start_pos.y()}) -> ({end_pos.x()},{end_pos.y()})")
    print(f"  Data: x={x_start:.3f} -> {x_end:.3f}")

    # Use QTest to simulate mouse press, move, release on the QGraphicsView
    QTest.mousePress(gv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start_pos)
    wait_ms(50)

    # Intermediate moves
    for frac in [0.25, 0.5, 0.75]:
        mid_pos, _, _ = _data_to_widget_pos(plot_host, x_start_frac + (x_end_frac - x_start_frac) * frac, y_frac)
        QTest.mouseMove(gv.viewport(), mid_pos)
        wait_ms(30)

    QTest.mouseMove(gv.viewport(), end_pos)
    wait_ms(50)
    QTest.mouseRelease(gv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end_pos)
    wait_ms(200)

    return True


def simulate_click_on_plot(plot_host, x_frac, y_frac=0.5, modifiers=Qt.KeyboardModifier.NoModifier):
    """Simulate a single mouse click using QTest."""
    pos, x_data, gv = _data_to_widget_pos(plot_host, x_frac, y_frac)
    print(f"  Click: widget ({pos.x()},{pos.y()}), data x={x_data:.3f}")
    QTest.mouseClick(gv.viewport(), Qt.MouseButton.LeftButton, modifiers, pos)
    wait_ms(200)
    return True


def test_omit_region_drag():
    """Test: activate omit mode, drag a region, verify it was created."""
    print("\n" + "="*60)
    print("TEST: Omit Region Drag")
    print("="*60)

    w = get_main_window()
    em = w.editing_modes
    ph = w.plot_host
    st = w.state

    sweep_count = w._nav_vm.sweep_count()
    print(f"Sweep count: {sweep_count}")
    if sweep_count == 0:
        print("SKIP: No file loaded")
        w.close()
        return

    # Count existing omit regions
    omit_before = len(getattr(st, 'omit_regions_by_sweep', {}).get(st.sweep_idx, []))
    print(f"Omit regions before: {omit_before}")

    # Activate omit mode
    w.OmitSweepButton.click()
    wait_ms(200)
    print(f"Omit mode active: {em._omit_region_mode}")
    print(f"Drag callback set: {ph._drag_callback is not None}")
    print(f"Cursor: {ph.cursor().shape()}")

    # Try drag on plot
    print("Attempting drag from 30% to 50% of plot width...")
    ok = simulate_drag_on_plot(ph, 0.3, 0.5)
    wait_ms(500)

    omit_after = len(getattr(st, 'omit_regions_by_sweep', {}).get(st.sweep_idx, []))
    print(f"Omit regions after: {omit_after}")

    if omit_after > omit_before:
        print("PASS: Omit region was created!")
    else:
        print("FAIL: No omit region created")
        # Debug: check what the drag callback received
        print(f"  _omit_region_start_x: {em._omit_region_start_x}")
        print(f"  _omit_region_mode still: {em._omit_region_mode}")

    # Cleanup
    w.OmitSweepButton.click()
    w.close()


def test_add_peak():
    """Test: activate add/del mode, click on plot, verify peak was added."""
    print("\n" + "="*60)
    print("TEST: Add Peak Click")
    print("="*60)

    w = get_main_window()
    em = w.editing_modes
    ph = w.plot_host
    st = w.state

    if w._nav_vm.sweep_count() == 0:
        print("SKIP: No file loaded")
        w.close()
        return

    # Count peaks before
    peaks_before = len(st.all_peaks_by_sweep.get(st.sweep_idx, []))
    print(f"Peaks before: {peaks_before}")

    # Activate add/del mode
    w.addPeaksButton.click()
    wait_ms(200)
    print(f"Add peaks mode: {em._add_peaks_mode}")
    print(f"Click callback set: {ph._external_click_cb is not None}")

    # Click on plot at 40% x position
    print("Clicking at 40% of plot width...")
    ok = simulate_click_on_plot(ph, 0.4)
    wait_ms(500)

    peaks_after = len(st.all_peaks_by_sweep.get(st.sweep_idx, []))
    print(f"Peaks after: {peaks_after}")

    if peaks_after != peaks_before:
        print(f"PASS: Peak count changed ({peaks_before} -> {peaks_after})")
    else:
        print("FAIL: Peak count unchanged")
        print(f"  _add_peaks_mode still: {em._add_peaks_mode}")
        print(f"  _external_click_cb: {ph._external_click_cb}")

    w.addPeaksButton.click()
    w.close()


def test_event_dispatch():
    """Test: verify that mouse events actually reach the eventFilter."""
    print("\n" + "="*60)
    print("TEST: Event Dispatch Verification")
    print("="*60)

    w = get_main_window()
    em = w.editing_modes
    ph = w.plot_host

    if w._nav_vm.sweep_count() == 0:
        print("SKIP: No file loaded")
        w.close()
        return

    # Patch the drag callback to log calls
    calls = []
    original_cb = None

    w.OmitSweepButton.click()
    wait_ms(200)

    original_cb = ph._drag_callback
    def logging_cb(event_type, x, y, event):
        calls.append((event_type, x, y))
        print(f"  DRAG CB: {event_type} at x={x:.3f}", flush=True)
        if original_cb:
            return original_cb(event_type, x, y, event)
    ph._drag_callback = logging_cb

    print("Drag callback patched, attempting drag...")
    simulate_drag_on_plot(ph, 0.3, 0.5)
    wait_ms(300)

    print(f"Drag callback was called {len(calls)} times: {[c[0] for c in calls]}")

    if len(calls) == 0:
        print("FAIL: Drag callback never called — events not reaching eventFilter")
        # Try sending events directly to the scene instead
        print("\nTrying alternative: direct scene event...")
        scene = ph.graphics_layout.scene()
        print(f"  Scene: {scene}")
        print(f"  Scene items: {len(scene.items())}")

        # Check if eventFilter is installed
        main_plot = ph._get_main_plot()
        print(f"  Main plot: {main_plot}")
        print(f"  Plot scene bounding rect: {main_plot.sceneBoundingRect()}")
    else:
        print("PASS: Events are reaching the drag callback")

    w.OmitSweepButton.click()
    w.close()


if __name__ == "__main__":
    print("Running editing interaction tests...")
    print(f"Testing OLD version (git stash active): check git status")

    test_event_dispatch()
    test_omit_region_drag()
    test_add_peak()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
