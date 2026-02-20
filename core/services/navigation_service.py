"""
NavigationService - Pure Python navigation logic for sweep/window navigation.

No Qt dependencies. Extracted from legacy NavigationManager as part of MVVM migration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NavigationResult:
    """Result of a navigation action."""
    new_sweep_idx: int
    sweep_changed: bool = False
    window_bounds: Optional[tuple] = None  # (left, right)
    needs_stim_recompute: bool = False
    needs_redraw: bool = False


class NavigationService:
    """Pure navigation logic — no Qt, no MainWindow references.

    Manages sweep/window navigation modes, computing new positions
    and returning NavigationResult objects that the ViewModel translates
    into Qt signals.
    """

    def __init__(self):
        self.mode: str = "sweep"  # "sweep" or "window"
        self._win_left: Optional[float] = None
        self._win_overlap_frac: float = 0.10
        self._win_min_overlap_s: float = 0.50

    def reset_window_state(self):
        """Reset window navigation state (called when loading new files)."""
        self._win_left = None

    def toggle_mode(self) -> str:
        """Toggle between sweep and window modes. Returns new mode."""
        self.mode = "window" if self.mode == "sweep" else "sweep"
        return self.mode

    # ------------------------------------------------------------------
    # Sweep navigation
    # ------------------------------------------------------------------

    def prev_sweep(self, sweep_idx: int, sweep_count: int, has_stim: bool) -> NavigationResult:
        """Navigate to previous sweep."""
        if sweep_count == 0 or sweep_idx <= 0:
            return NavigationResult(new_sweep_idx=sweep_idx)
        new_idx = sweep_idx - 1
        return NavigationResult(
            new_sweep_idx=new_idx,
            sweep_changed=True,
            needs_stim_recompute=has_stim,
            needs_redraw=True,
        )

    def next_sweep(self, sweep_idx: int, sweep_count: int, has_stim: bool) -> NavigationResult:
        """Navigate to next sweep."""
        if sweep_count == 0 or sweep_idx >= sweep_count - 1:
            return NavigationResult(new_sweep_idx=sweep_idx)
        new_idx = sweep_idx + 1
        return NavigationResult(
            new_sweep_idx=new_idx,
            sweep_changed=True,
            needs_stim_recompute=has_stim,
            needs_redraw=True,
        )

    # ------------------------------------------------------------------
    # Window navigation
    # ------------------------------------------------------------------

    def _window_step(self, W: float) -> float:
        """Step size when paging windows: W - overlap."""
        overlap = max(self._win_min_overlap_s, self._win_overlap_frac * W)
        step = max(0.0, W - overlap)
        if step <= 0:
            step = 0.9 * W
        return step

    def snap_to_window(self, t_plot, window_secs: float) -> Optional[NavigationResult]:
        """Jump to start of current sweep's time axis."""
        if t_plot is None or t_plot.size == 0:
            return None
        left = float(t_plot[0])
        bounds = self.set_window(left, window_secs)
        return NavigationResult(
            new_sweep_idx=-1,  # no sweep change
            window_bounds=bounds,
        )

    def next_window(self, t_plot, window_secs: float, current_xlim_left: float,
                    sweep_idx: int, sweep_count: int, has_stim: bool) -> Optional[NavigationResult]:
        """Step forward one window. Returns NavigationResult or None if no data."""
        if t_plot is None or t_plot.size == 0:
            return None

        W = window_secs
        step = self._window_step(W)

        # Initialize left edge if needed
        if self._win_left is None:
            self._win_left = current_xlim_left

        dur = float(t_plot[-1] - t_plot[0])
        W_eff = min(W, max(1e-6, dur))
        max_left = float(t_plot[-1]) - W_eff
        eps = 1e-9

        # Normal step within this sweep
        if self._win_left + step <= max_left + eps:
            bounds = self.set_window(self._win_left + step, W_eff)
            return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

        # Not enough room: show last full window first
        if self._win_left < max_left - eps:
            bounds = self.set_window(max_left, W_eff)
            return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

        # Already at last window: hop to next sweep if possible
        if sweep_idx < sweep_count - 1:
            return NavigationResult(
                new_sweep_idx=sweep_idx + 1,
                sweep_changed=True,
                needs_stim_recompute=has_stim,
                needs_redraw=True,
                # window_bounds set by caller after sweep change + t_plot refresh
                window_bounds=None,
            )

        # No next sweep: stay clamped
        bounds = self.set_window(max_left, W_eff)
        return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

    def prev_window(self, t_plot, window_secs: float, current_xlim_left: float,
                    sweep_idx: int, sweep_count: int, has_stim: bool) -> Optional[NavigationResult]:
        """Step backward one window."""
        if t_plot is None or t_plot.size == 0:
            return None

        W = window_secs
        step = self._window_step(W)

        if self._win_left is None:
            self._win_left = current_xlim_left

        dur = float(t_plot[-1] - t_plot[0])
        W_eff = min(W, max(1e-6, dur))
        min_left = float(t_plot[0])
        eps = 1e-9

        # Normal step within this sweep
        if self._win_left - step >= min_left - eps:
            bounds = self.set_window(self._win_left - step, W_eff)
            return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

        # Not enough room: show first full window
        if self._win_left > min_left + eps:
            bounds = self.set_window(min_left, W_eff)
            return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

        # Already at first window: hop to previous sweep
        if sweep_idx > 0:
            return NavigationResult(
                new_sweep_idx=sweep_idx - 1,
                sweep_changed=True,
                needs_stim_recompute=has_stim,
                needs_redraw=True,
                window_bounds=None,
            )

        # No previous sweep: stay clamped
        bounds = self.set_window(min_left, W_eff)
        return NavigationResult(new_sweep_idx=sweep_idx, window_bounds=bounds)

    def set_window(self, left: float, width: float) -> tuple:
        """Set window position. Returns (left, right) bounds."""
        right = left + max(0.01, float(width))
        self._win_left = float(left)
        return (float(left), right)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_t_plot(t, sweep_idx: int, sweep_count: int, stim_chan, stim_spans_by_sweep: dict):
        """Compute display time axis (normalized if stim spans exist)."""
        if t is None:
            return None
        s = max(0, min(sweep_idx, sweep_count - 1))
        spans = stim_spans_by_sweep.get(s, []) if stim_chan else []
        if stim_chan and spans:
            t0 = spans[0][0]
            return t - t0
        return t

    @staticmethod
    def sweep_count(sweeps: dict) -> int:
        """Return total sweep count from the first channel."""
        if not sweeps:
            return 0
        first = next(iter(sweeps.values()))
        return int(first.shape[1]) if first is not None else 0
