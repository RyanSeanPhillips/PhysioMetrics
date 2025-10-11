"""
NavigationManager - Handles all navigation functionality for PlethApp.

This module manages:
- Sweep navigation (prev/next sweep)
- Window navigation (time-based windowing with overlap)
- Unified navigation (mode toggle between sweep and window)
- List box navigation (move items between lists in curation tab)
- File list filtering (search/filter functionality)
"""

from PyQt6.QtCore import Qt


class NavigationManager:
    """Manages all navigation operations for the main window."""

    def __init__(self, main_window):
        """Initialize navigation manager.

        Args:
            main_window: Reference to MainWindow instance
        """
        self.main = main_window
        self.state = main_window.state

        # Navigation mode
        self.navigation_mode = "sweep"  # "sweep" or "window"

        # Window navigation state
        self._win_overlap_frac = 0.10   # 10% of the window length
        self._win_min_overlap_s = 0.50  # but at least 0.5 s overlap
        self._win_left = None  # Track current window left edge (in "display time" coordinates)

        # Connect signals
        self._connect_signals()

    def _connect_signals(self):
        """Connect all navigation-related buttons and widgets."""
        # Unified navigation buttons
        self.main.PrevButton.clicked.connect(self.on_unified_prev)
        self.main.NextButton.clicked.connect(self.on_unified_next)
        self.main.ViewModeToggleButton.clicked.connect(self.on_toggle_view_mode)

        # Curation tab list navigation buttons
        self.main.moveAllRight.clicked.connect(self.on_move_all_right)
        self.main.moveSingleRight.clicked.connect(self.on_move_selected_right)
        self.main.moveSingleLeft.clicked.connect(self.on_move_selected_left)
        self.main.moveAllLeft.clicked.connect(self.on_move_all_left)

        # File list filter
        self.main.FileListSearchBox.textChanged.connect(self._filter_file_list)

    def reset_window_state(self):
        """Reset window navigation state (called when loading new files)."""
        self._win_left = None

    ##################################################
    ## Sweep Navigation                             ##
    ##################################################

    def _num_sweeps(self) -> int:
        """Return total sweep count from the first channel (0 if no data)."""
        st = self.state
        if not st.sweeps:
            return 0
        first = next(iter(st.sweeps.values()))
        return int(first.shape[1]) if first is not None else 0

    def on_prev_sweep(self):
        """Navigate to previous sweep."""
        st = self.state
        n = self._num_sweeps()
        if n == 0:
            return
        if st.sweep_idx > 0:
            st.sweep_idx -= 1
            # recompute stim spans for this sweep if a stim channel is selected
            if st.stim_chan:
                self.main._compute_stim_for_current_sweep()
            self.main._refresh_omit_button_label()
            self.main.redraw_main_plot()

    def on_next_sweep(self):
        """Navigate to next sweep."""
        st = self.state
        n = self._num_sweeps()
        if n == 0:
            return
        if st.sweep_idx < n - 1:
            st.sweep_idx += 1
            if st.stim_chan:
                self.main._compute_stim_for_current_sweep()
            self.main._refresh_omit_button_label()
            self.main.redraw_main_plot()

    def on_snap_to_sweep(self):
        """Snap to full sweep view (clear saved zoom)."""
        # clear saved zoom so the next draw autoscales to full sweep range
        self.main.plot_host.clear_saved_view("single" if self.main.single_panel_mode else "grid")
        self.main._refresh_omit_button_label()
        self.main.redraw_main_plot()

    ##################################################
    ## Window Navigation (relative to current window)##
    ##################################################

    def _parse_window_seconds(self) -> float:
        """Read WindowRangeValue (seconds). Returns a positive float, default 20."""
        try:
            val = float(self.main.WindowRangeValue.text().strip())
            if val > 0:
                return val
        except Exception:
            pass
        return 20.0

    def _window_step(self, W: float) -> float:
        """
        Step size when paging windows: W - overlap,
        where overlap is max(min_overlap, frac * W).
        """
        overlap = max(self._win_min_overlap_s, self._win_overlap_frac * W)
        step = max(0.0, W - overlap)
        # avoid zero step for tiny windows
        if step <= 0:
            step = 0.9 * W
        return step

    def on_snap_to_window(self):
        """Jump to start of current sweep (normalized domain if applicable)."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return
        W = self._parse_window_seconds()
        left = float(t[0])
        self._set_window(left=left, width=W)

    def on_next_window(self):
        """Step forward; if stepping past end, first show the last full window once,
        then on the next press hop to the first window of the next sweep."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return

        W = self._parse_window_seconds()
        step = self._window_step(W)

        # Initialize left edge if needed
        if self._win_left is None:
            ax = self.main.plot_host.fig.axes[0] if self.main.plot_host.fig.axes else None
            self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

        # Use an effective width that never exceeds this sweep's duration
        dur = float(t[-1] - t[0])
        W_eff = min(W, max(1e-6, dur))
        max_left = float(t[-1]) - W_eff
        eps = 1e-9

        # Normal step within this sweep?
        if self._win_left + step <= max_left + eps:
            self._set_window(left=self._win_left + step, width=W_eff)
            return

        # Not enough room for a full step:
        # 1) If we're not yet at the last full window, show it once.
        if self._win_left < max_left - eps:
            self._set_window(left=max_left, width=W_eff)
            return

        # 2) Already at last full window -> hop to next sweep if possible
        s_count = self._sweep_count()
        if self.state.sweep_idx < s_count - 1:
            self.state.sweep_idx += 1
            if self.state.stim_chan:
                self.main._compute_stim_for_current_sweep()
            self.main.redraw_main_plot()

            t2 = self._current_t_plot()
            if t2 is None or t2.size == 0:
                return
            dur2 = float(t2[-1] - t2[0])
            W_eff2 = min(W, max(1e-6, dur2))
            self._set_window(left=float(t2[0]), width=W_eff2)
        else:
            # No next sweep: stay clamped at the last full window
            self._set_window(left=max_left, width=W_eff)

    def on_prev_window(self):
        """Step backward; if stepping before start, first show the first full window once,
        then on the next press hop to the last window of the previous sweep."""
        t = self._current_t_plot()
        if t is None or t.size == 0:
            return

        W = self._parse_window_seconds()
        step = self._window_step(W)

        if self._win_left is None:
            ax = self.main.plot_host.fig.axes[0] if self.main.plot_host.fig.axes else None
            self._win_left = float(ax.get_xlim()[0]) if ax else float(t[0])

        dur = float(t[-1] - t[0])
        W_eff = min(W, max(1e-6, dur))
        min_left = float(t[0])
        eps = 1e-9

        # Normal step within this sweep?
        if self._win_left - step >= min_left - eps:
            self._set_window(left=self._win_left - step, width=W_eff)
            return

        # Not enough room for a full step:
        # 1) If we're not yet at the first full window, show it once.
        if self._win_left > min_left + eps:
            self._set_window(left=min_left, width=W_eff)
            return

        # 2) Already at first window -> hop to previous sweep if possible
        if self.state.sweep_idx > 0:
            self.state.sweep_idx -= 1
            if self.state.stim_chan:
                self.main._compute_stim_for_current_sweep()
            self.main.redraw_main_plot()

            t2 = self._current_t_plot()
            if t2 is None or t2.size == 0:
                return
            dur2 = float(t2[-1] - t2[0])
            W_eff2 = min(W, max(1e-6, dur2))
            last_left = max(float(t2[0]), float(t2[-1]) - W_eff2)
            self._set_window(left=last_left, width=W_eff2)
        else:
            # No previous sweep: stay clamped at the first window
            self._set_window(left=min_left, width=W_eff)

    ##################################################
    ## Unified Navigation (Sweep/Window Toggle)     ##
    ##################################################

    def on_toggle_view_mode(self):
        """Toggle between sweep and window navigation modes."""
        if self.navigation_mode == "sweep":
            self.navigation_mode = "window"
            self.main.ViewModeToggleButton.setText("Mode: Window View")
            self.main.PrevButton.setToolTip("Move to the previous time window")
            self.main.NextButton.setToolTip("Move to the next time window")
            # Snap to window view (show current position as windowed)
            self.on_snap_to_window()
        else:
            self.navigation_mode = "sweep"
            self.main.ViewModeToggleButton.setText("Mode: Sweep View")
            self.main.PrevButton.setToolTip("Navigate to the previous sweep")
            self.main.NextButton.setToolTip("Navigate to the next sweep")
            # Snap to sweep view (show full sweep)
            self.on_snap_to_sweep()

    def on_unified_prev(self):
        """Unified previous button: dispatches to sweep or window based on mode."""
        if self.navigation_mode == "sweep":
            self.on_prev_sweep()
        else:
            self.on_prev_window()

    def on_unified_next(self):
        """Unified next button: dispatches to sweep or window based on mode."""
        if self.navigation_mode == "sweep":
            self.on_next_sweep()
        else:
            self.on_next_window()

    def _sweep_count(self) -> int:
        """Return total sweep count."""
        st = self.state
        if not st.sweeps:
            return 0
        any_ch = next(iter(st.sweeps.values()))
        return any_ch.shape[1]

    def _current_t_plot(self):
        """Time axis exactly like the one used in redraw (normalized if stim spans exist)."""
        st = self.state
        if st.t is None:
            return None
        s = max(0, min(st.sweep_idx, self._sweep_count()-1))
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            return st.t - t0
        return st.t

    def _set_window(self, left: float, width: float):
        """Apply x-limits and remember left edge for subsequent steps."""
        ax = self.main.plot_host.fig.axes[0] if self.main.plot_host.fig.axes else None
        if ax is None:
            return
        right = left + max(0.01, float(width))
        ax.set_xlim(left, right)
        self._win_left = float(left)
        self.main.plot_host.fig.tight_layout()
        self.main.plot_host.canvas.draw_idle()

    ##################################################
    ## List Box Navigation (Curation Tab)          ##
    ##################################################

    def _list_has_key(self, lw, key: str) -> bool:
        """True if any item in lw has the same group key."""
        for i in range(lw.count()):
            it = lw.item(i)
            if not it:
                continue
            meta = it.data(Qt.ItemDataRole.UserRole) or {}
            if isinstance(meta, dict) and meta.get("key", "").lower() == key.lower():
                return True
        return False

    def _move_items(self, src_lw, dst_lw, rows_to_move: list[int]):
        """
        Move grouped items by root from src_lw to dst_lw.
        Duplicate check is by 'key' (dir+root), not by file path.
        """
        if not rows_to_move:
            return 0, 0

        plan = []
        for r in rows_to_move:
            it = src_lw.item(r)
            if it is None:
                continue
            meta = it.data(Qt.ItemDataRole.UserRole) or {}
            key = (meta.get("key") or "").lower()
            is_dup = self._list_has_key(dst_lw, key)
            plan.append((r, is_dup))

        taken = []
        skipped_dups = 0
        for r, is_dup in sorted(plan, key=lambda x: x[0], reverse=True):
            if is_dup:
                skipped_dups += 1
                continue
            it = src_lw.takeItem(r)
            if it is not None:
                taken.append((r, it))

        moved = 0
        for _, it in sorted(taken, key=lambda x: x[0]):
            dst_lw.addItem(it)
            moved += 1

        src_lw.sortItems()
        dst_lw.sortItems()
        return moved, skipped_dups

    def on_move_selected_right(self):
        """Move selected from left (FileList) to right (FilestoConsolidateList)."""
        src = self.main.FileList
        dst = self.main.FilestoConsolidateList
        rows = [src.row(it) for it in src.selectedItems()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} item(s) to right. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_all_right(self):
        """Move ALL VISIBLE from left to right."""
        src = self.main.FileList
        dst = self.main.FilestoConsolidateList
        # Only move visible (non-hidden) items
        rows = [i for i in range(src.count()) if not src.item(i).isHidden()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} visible item(s) to right. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_selected_left(self):
        """Move selected from right back to left."""
        src = self.main.FilestoConsolidateList
        dst = self.main.FileList
        rows = [src.row(it) for it in src.selectedItems()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} item(s) to left. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    def on_move_all_left(self):
        """Move ALL VISIBLE from right back to left."""
        src = self.main.FilestoConsolidateList
        dst = self.main.FileList
        # Only move visible (non-hidden) items
        rows = [i for i in range(src.count()) if not src.item(i).isHidden()]
        moved, skipped = self._move_items(src, dst, rows)
        try:
            if moved or skipped:
                self.main.statusbar.showMessage(f"Moved {moved} visible item(s) to left. Skipped {skipped} duplicate(s).", 3000)
        except Exception:
            pass

    ##################################################
    ## File List Filter (Curation Tab)             ##
    ##################################################

    def _filter_file_list(self, text: str):
        """Show/hide items in FileList based on search text.

        Supports multiple search modes:
        - Single keyword: 'gfp' - shows files containing 'gfp'
        - Multiple keywords (AND): 'gfp 2.5mW' - shows files containing BOTH 'gfp' AND '2.5mW'
        - Multiple keywords (OR): 'gfp, chr2' - shows files containing EITHER 'gfp' OR 'chr2'
        """
        search_text = text.strip().lower()

        # Determine search mode
        if ',' in search_text:
            # OR mode: split by comma
            keywords = [k.strip() for k in search_text.split(',') if k.strip()]
            search_mode = 'OR'
        else:
            # AND mode: split by whitespace
            keywords = [k.strip() for k in search_text.split() if k.strip()]
            search_mode = 'AND'

        for i in range(self.main.FileList.count()):
            item = self.main.FileList.item(i)
            if not item:
                continue

            # Get the display text
            item_text = item.text().lower()

            # Also search in tooltip (which contains full path)
            tooltip = (item.toolTip() or "").lower()
            combined_text = f"{item_text} {tooltip}"

            # Show item if search text is empty
            if not keywords:
                item.setHidden(False)
                continue

            # Apply search logic
            if search_mode == 'AND':
                # ALL keywords must be present
                matches = all(kw in combined_text for kw in keywords)
            else:  # OR mode
                # ANY keyword must be present
                matches = any(kw in combined_text for kw in keywords)

            item.setHidden(not matches)
