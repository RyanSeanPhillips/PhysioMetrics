"""
NavigationViewModel - Qt integration for sweep/window navigation.

Wraps NavigationService with pyqtSignals. MainWindow injects state/xlim
providers via callbacks, eliminating direct MainWindow coupling.
"""

from typing import Callable, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from core.services.navigation_service import NavigationService, NavigationResult


class NavigationViewModel(QObject):
    """ViewModel for sweep/window navigation.

    Signals:
        sweep_changed(int): New sweep index — main.py sets state.sweep_idx
        window_changed(float, float): New (left, right) x-axis bounds
        mode_changed(str): "sweep" or "window"
        needs_redraw(): Request main plot redraw
        needs_stim_recompute(): Stim spans need recomputing for new sweep
        snap_to_sweep_requested(): Clear saved view + redraw (snap to full sweep)
    """

    sweep_changed = pyqtSignal(int)
    window_changed = pyqtSignal(float, float)
    mode_changed = pyqtSignal(str)
    needs_redraw = pyqtSignal()
    needs_stim_recompute = pyqtSignal()
    snap_to_sweep_requested = pyqtSignal()

    def __init__(self, service: NavigationService, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._service = service

        # Injected providers (set by main.py, replaces self.mw access)
        self._state_provider: Optional[Callable] = None
        self._xlim_provider: Optional[Callable] = None
        self._window_seconds_provider: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Provider injection (called once during main.py setup)
    # ------------------------------------------------------------------

    def set_state_provider(self, fn: Callable):
        """fn() -> (sweep_idx, sweeps, t, stim_chan, stim_spans_by_sweep)"""
        self._state_provider = fn

    def set_xlim_provider(self, fn: Callable):
        """fn() -> float (current left x-axis limit)"""
        self._xlim_provider = fn

    def set_window_seconds_provider(self, fn: Callable):
        """fn() -> float (window duration from UI text field)"""
        self._window_seconds_provider = fn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._service.mode

    @property
    def service(self) -> NavigationService:
        return self._service

    # ------------------------------------------------------------------
    # Helper: get current state
    # ------------------------------------------------------------------

    def _get_state(self):
        """Get navigation state from provider."""
        if not self._state_provider:
            return None
        return self._state_provider()

    def _get_xlim_left(self) -> float:
        if self._xlim_provider:
            return self._xlim_provider()
        return 0.0

    def _parse_window_seconds(self) -> float:
        if self._window_seconds_provider:
            try:
                val = float(self._window_seconds_provider())
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
        return 20.0

    def _current_t_plot(self, sweep_idx, sweeps, t, stim_chan, stim_spans_by_sweep):
        """Compute display time axis for current sweep."""
        count = NavigationService.sweep_count(sweeps)
        return NavigationService.compute_t_plot(t, sweep_idx, count, stim_chan, stim_spans_by_sweep)

    # ------------------------------------------------------------------
    # Commands (connected to buttons)
    # ------------------------------------------------------------------

    def navigate_prev(self):
        """Unified previous: dispatches to sweep or window based on mode."""
        if self._service.mode == "sweep":
            self._do_prev_sweep()
        else:
            self._do_prev_window()

    def navigate_next(self):
        """Unified next: dispatches to sweep or window based on mode."""
        if self._service.mode == "sweep":
            self._do_next_sweep()
        else:
            self._do_next_window()

    def toggle_mode(self):
        """Toggle sweep/window mode and emit signals."""
        new_mode = self._service.toggle_mode()
        self.mode_changed.emit(new_mode)

        if new_mode == "window":
            self._do_snap_to_window()
        else:
            self._do_snap_to_sweep()

    def snap(self):
        """Snap to sweep or window depending on current mode."""
        if self._service.mode == "sweep":
            self._do_snap_to_sweep()
        else:
            self._do_snap_to_window()

    def set_window(self, left: float, width: float):
        """Public API for external callers (e.g. PeakNavigatorDialog)."""
        bounds = self._service.set_window(left, width)
        self._apply_window_bounds(bounds)

    def reset_window_state(self):
        """Reset window state (called on file load)."""
        self._service.reset_window_state()

    # ------------------------------------------------------------------
    # Delegated to service: sweep count helper (used by main.py)
    # ------------------------------------------------------------------

    def sweep_count(self) -> int:
        """Return sweep count from current state."""
        state = self._get_state()
        if not state:
            return 0
        _, sweeps, _, _, _ = state
        return NavigationService.sweep_count(sweeps)

    # ------------------------------------------------------------------
    # Internal: sweep navigation
    # ------------------------------------------------------------------

    def _do_prev_sweep(self):
        state = self._get_state()
        if not state:
            return
        sweep_idx, sweeps, _, stim_chan, _ = state
        count = NavigationService.sweep_count(sweeps)
        result = self._service.prev_sweep(sweep_idx, count, bool(stim_chan))
        self._apply_result(result)

    def _do_next_sweep(self):
        state = self._get_state()
        if not state:
            return
        sweep_idx, sweeps, _, stim_chan, _ = state
        count = NavigationService.sweep_count(sweeps)
        result = self._service.next_sweep(sweep_idx, count, bool(stim_chan))
        self._apply_result(result)

    # ------------------------------------------------------------------
    # Internal: window navigation
    # ------------------------------------------------------------------

    def _do_snap_to_sweep(self):
        self._service.reset_window_state()
        self.snap_to_sweep_requested.emit()

    def _do_snap_to_window(self):
        state = self._get_state()
        if not state:
            return
        sweep_idx, sweeps, t, stim_chan, stim_spans = state
        t_plot = self._current_t_plot(sweep_idx, sweeps, t, stim_chan, stim_spans)
        W = self._parse_window_seconds()
        result = self._service.snap_to_window(t_plot, W)
        if result:
            self._apply_window_bounds(result.window_bounds)

    def _do_next_window(self):
        state = self._get_state()
        if not state:
            return
        sweep_idx, sweeps, t, stim_chan, stim_spans = state
        t_plot = self._current_t_plot(sweep_idx, sweeps, t, stim_chan, stim_spans)
        count = NavigationService.sweep_count(sweeps)
        W = self._parse_window_seconds()
        xlim_left = self._get_xlim_left()

        result = self._service.next_window(t_plot, W, xlim_left, sweep_idx, count, bool(stim_chan))
        if not result:
            return

        if result.sweep_changed:
            self._apply_result(result)
            # After sweep change, set window at start of new sweep
            new_state = self._get_state()
            if new_state:
                new_idx, new_sweeps, new_t, new_stim, new_spans = new_state
                t2 = self._current_t_plot(new_idx, new_sweeps, new_t, new_stim, new_spans)
                if t2 is not None and t2.size > 0:
                    dur2 = float(t2[-1] - t2[0])
                    W_eff2 = min(W, max(1e-6, dur2))
                    bounds = self._service.set_window(float(t2[0]), W_eff2)
                    self._apply_window_bounds(bounds)
        else:
            self._apply_window_bounds(result.window_bounds)

    def _do_prev_window(self):
        state = self._get_state()
        if not state:
            return
        sweep_idx, sweeps, t, stim_chan, stim_spans = state
        t_plot = self._current_t_plot(sweep_idx, sweeps, t, stim_chan, stim_spans)
        count = NavigationService.sweep_count(sweeps)
        W = self._parse_window_seconds()
        xlim_left = self._get_xlim_left()

        result = self._service.prev_window(t_plot, W, xlim_left, sweep_idx, count, bool(stim_chan))
        if not result:
            return

        if result.sweep_changed:
            self._apply_result(result)
            # After sweep change, set window at end of new sweep
            new_state = self._get_state()
            if new_state:
                new_idx, new_sweeps, new_t, new_stim, new_spans = new_state
                t2 = self._current_t_plot(new_idx, new_sweeps, new_t, new_stim, new_spans)
                if t2 is not None and t2.size > 0:
                    dur2 = float(t2[-1] - t2[0])
                    W_eff2 = min(W, max(1e-6, dur2))
                    last_left = max(float(t2[0]), float(t2[-1]) - W_eff2)
                    bounds = self._service.set_window(last_left, W_eff2)
                    self._apply_window_bounds(bounds)
        else:
            self._apply_window_bounds(result.window_bounds)

    # ------------------------------------------------------------------
    # Internal: apply results → emit signals
    # ------------------------------------------------------------------

    def _apply_result(self, result: NavigationResult):
        """Emit appropriate signals from a NavigationResult."""
        if result.sweep_changed:
            self.sweep_changed.emit(result.new_sweep_idx)
        if result.needs_stim_recompute:
            self.needs_stim_recompute.emit()
        if result.window_bounds:
            self._apply_window_bounds(result.window_bounds)
        if result.needs_redraw:
            self.needs_redraw.emit()

    def _apply_window_bounds(self, bounds):
        """Apply window bounds to the plot via signal."""
        if bounds:
            left, right = bounds
            self.window_changed.emit(left, right)
