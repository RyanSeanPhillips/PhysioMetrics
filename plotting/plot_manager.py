"""
PlotManager - Orchestrates all plotting operations for the main window.

This module extracts plotting logic from main.py to improve maintainability.
"""

import numpy as np
from core import metrics


class PlotManager:
    """Manages all plotting operations for the main application window."""

    def __init__(self, main_window):
        """
        Initialize the PlotManager.

        Args:
            main_window: Reference to MainWindow instance for accessing state and UI
        """
        self.window = main_window
        self.plot_host = main_window.plot_host
        self.state = main_window.state

        # Threshold line artists (for manual removal)
        self._thresh_line_artists = []

    def redraw_main_plot(self):
        """
        Main plot orchestration method.
        Determines whether to show single-panel or multi-channel grid mode.
        """
        st = self.state
        if st.t is None:
            return

        if self.window.single_panel_mode:
            self._draw_single_panel_plot()
            # Restore editing mode connections after redraw
            self._restore_editing_mode_connections()
        else:
            # Multi-channel grid mode - show all channels for current sweep
            self.plot_all_channels()

    def _draw_single_panel_plot(self):
        """Draw the main single-panel plot with all overlays."""
        st = self.state
        t, y = self.window._current_trace()
        if t is None:
            return

        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []

        # Normalize time to first stim onset if available
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = t - t0
            spans_plot = [(a - t0, b - t0) for (a, b) in spans]
        else:
            t0 = 0.0
            t_plot = t
            spans_plot = spans

        # Build title with file info for multi-file loading
        title = self._build_plot_title(s)

        # Check if we need dual subplot layout for event channel
        use_event_subplot = (st.event_channel is not None and st.event_channel in st.sweeps)

        if use_event_subplot:
            # Create dual subplot layout
            self._draw_dual_subplot_plot(t, y, t_plot, spans_plot, title, s, t0)
        else:
            # Standard single panel plot
            # Clear event subplot reference
            self.plot_host.ax_event = None
            # Draw base trace with stimulus spans
            self.plot_host.show_trace_with_spans(
                t_plot, y, spans_plot,
                title=title,
                max_points=None,
                ylabel=st.analyze_chan or "Signal"
            )

        # Clear any existing region overlays (will be recomputed if needed)
        self.plot_host.clear_region_overlays()

        # Draw peak markers
        self._draw_peak_markers(s, t_plot, y)

        # Draw sigh markers (stars)
        self._draw_sigh_markers(s, t_plot, y)

        # Draw breath event markers (onsets, offsets, expiratory mins/offs)
        self._draw_breath_markers(s, t_plot, y)

        # Update sniff region overlays
        self.window.editing_modes.update_sniff_artists(t_plot, s)

        # Draw Y2 metric if selected
        self._draw_y2_metric(s, t, t_plot)

        # Draw automatic region overlays (eupnea, apnea, outliers)
        self._draw_region_overlays(s, t, y, t_plot)

        # Draw omitted region overlays
        self._draw_omitted_regions(s, t_plot)

        # Refresh threshold lines
        self.refresh_threshold_lines()

        # Set y-limits based on non-omitted data only (exclude omitted regions from autoscaling)
        self._set_ylim_excluding_omitted(s, y, t_plot)

        # Note: Full sweep dimming now handled in _draw_omitted_regions() for consistency

    def _set_ylim_excluding_omitted(self, sweep_idx, y, t_plot):
        """Calculate and set y-limits based only on non-omitted data and peaks."""
        st = self.state
        ax = self.plot_host.ax_main

        # If view is preserved, don't change ylim
        if self.plot_host._preserve_y:
            return

        # If full sweep is omitted, use default autoscale
        if sweep_idx in st.omitted_sweeps:
            return

        # Collect all y-values to consider (non-omitted only)
        y_values = []

        # 1. Add non-omitted trace data
        if sweep_idx in st.omitted_ranges:
            # Create mask for non-omitted samples
            mask = np.ones(len(y), dtype=bool)
            for (start_idx, end_idx) in st.omitted_ranges[sweep_idx]:
                start_idx = max(0, min(start_idx, len(y) - 1))
                end_idx = max(0, min(end_idx, len(y) - 1))
                mask[start_idx:end_idx+1] = False
            y_non_omit = y[mask]
        else:
            # No omitted regions, use all data
            y_non_omit = y

        if len(y_non_omit) > 0:
            y_values.extend(y_non_omit)

        # 2. Add non-omitted peak markers
        pks = st.peaks_by_sweep.get(sweep_idx, [])
        if len(pks) > 0:
            for pk in pks:
                if not self._is_peak_in_omitted_region(sweep_idx, pk):
                    y_values.append(y[pk])

        # 3. Add non-omitted breath markers (onsets, offsets, expmins, expoffs)
        br = st.breath_by_sweep.get(sweep_idx, {})
        for key in ['onsets', 'offsets', 'expmins', 'expoffs']:
            indices = br.get(key, [])
            if len(indices) > 0:
                for idx in indices:
                    if not self._is_peak_in_omitted_region(sweep_idx, idx):
                        y_values.append(y[idx])

        # 4. Add non-omitted sigh markers (they have vertical offset, but we'll approximate)
        sigh_idx = getattr(st, "sigh_by_sweep", {}).get(sweep_idx, [])
        if len(sigh_idx) > 0:
            for idx in sigh_idx:
                if not self._is_peak_in_omitted_region(sweep_idx, idx):
                    # Sigh markers have ~7% offset, so add that for ylim calculation
                    y_values.append(y[idx])

        # Calculate ylim with percentile-based scaling (matching core/plotting.py autoscale)
        if len(y_values) > 0:
            y_values = np.array(y_values)
            # Use percentiles to avoid artifacts and huge outliers
            y_min = np.percentile(y_values, 1)   # 1st percentile
            y_max = np.percentile(y_values, 99)  # 99th percentile

            # Add 25% padding to avoid clipping peaks (matching core/plotting.py)
            y_range = y_max - y_min
            if y_range > 0:
                padding = 0.25 * y_range
                ax.set_ylim(y_min - padding, y_max + padding)
                print(f"[Plot Manager] Auto-scaled Y-axis (excluding omitted): {y_min - padding:.3f} to {y_max + padding:.3f}")

    def _draw_dual_subplot_plot(self, t_full, y_pleth, t_plot, spans_plot, title, sweep_idx, t0):
        """Draw dual subplot layout with pleth trace on top and event channel on bottom."""
        from matplotlib.gridspec import GridSpec
        st = self.state

        # Save previous view if needed
        prev_xlim = self.plot_host._last_single["xlim"] if self.plot_host._preserve_x else None
        prev_ylim = self.plot_host._last_single["ylim"] if self.plot_host._preserve_y else None

        # Clear figure and create GridSpec layout
        self.plot_host.fig.clear()
        gs = GridSpec(2, 1, height_ratios=[0.7, 0.3], hspace=0.05, figure=self.plot_host.fig)
        ax_pleth = self.plot_host.fig.add_subplot(gs[0])
        ax_event = self.plot_host.fig.add_subplot(gs[1], sharex=ax_pleth)

        # Hide x-axis labels on top plot
        ax_pleth.tick_params(labelbottom=False)

        # Set plot_host.ax_main to top subplot for compatibility with existing overlay code
        self.plot_host.ax_main = ax_pleth
        self.plot_host.ax_event = ax_event

        # Clear old scatter references
        self.plot_host.scatter_peaks = None
        self.plot_host.scatter_onsets = None
        self.plot_host.scatter_offsets = None
        self.plot_host.scatter_expmins = None
        self.plot_host.scatter_expoffs = None
        self.plot_host._sigh_artist = None
        self.plot_host.ax_y2 = None
        self.plot_host.line_y2 = None
        self.plot_host.line_y2_secondary = None

        # Plot pleth trace on top subplot
        ax_pleth.plot(t_plot, y_pleth, linewidth=0.9, color='k')
        ax_pleth.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Add stim spans to top plot
        for (t0_span, t1_span) in (spans_plot or []):
            if t1_span > t0_span:
                ax_pleth.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

        # Set title and labels for top plot
        if title:
            ax_pleth.set_title(title)
        ax_pleth.set_ylabel(st.analyze_chan or "Signal")
        ax_pleth.grid(False)

        # Apply percentile-based Y-axis autoscaling (matching single-panel behavior)
        if len(y_pleth) > 0:
            y_min = np.percentile(y_pleth, 1)   # 1st percentile
            y_max = np.percentile(y_pleth, 99)  # 99th percentile
            y_range = y_max - y_min
            padding = 0.25 * y_range  # 25% padding
            ax_pleth.set_ylim(y_min - padding, y_max + padding)
            print(f"[Dual Plot] Auto-scaled Y-axis: {y_min - padding:.3f} to {y_max + padding:.3f} (99th percentile)")

        # Plot event trace on bottom subplot
        self._plot_event_trace(ax_event, t_full, t_plot, spans_plot, t0)

        # Plot bout annotations on both subplots
        if sweep_idx in st.bout_annotations and st.bout_annotations[sweep_idx]:
            self._plot_bout_annotations(ax_pleth, ax_event, st.bout_annotations[sweep_idx], t0)

        # Restore preserved view if desired
        if prev_xlim is not None:
            ax_pleth.set_xlim(prev_xlim)
        if prev_ylim is not None:
            ax_pleth.set_ylim(prev_ylim)

        # Set up limit listeners
        self.plot_host._attach_limit_listeners([ax_pleth, ax_event], mode="single")
        self.plot_host._store_from_axes(mode="single")

        # Keep layout tight
        self.plot_host.fig.tight_layout()
        self.plot_host.canvas.draw_idle()

    def _plot_event_trace(self, ax, t_full, t_plot, spans_plot, t0):
        """Plot event channel trace on given axis."""
        st = self.state
        swp = st.sweep_idx

        if not st.event_channel or st.event_channel not in st.sweeps:
            return

        # Get event channel data for current sweep
        event_data_full = st.sweeps[st.event_channel][:, swp]

        # Apply time normalization if needed (same as pleth trace)
        event_plot = event_data_full

        # Plot continuous trace
        ax.plot(t_plot, event_plot, 'b-', linewidth=1, label=st.event_channel)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Add threshold line if event detection dialog exists and has a threshold set
        if hasattr(self.window, '_event_detection_dialog') and self.window._event_detection_dialog is not None:
            try:
                threshold = self.window._event_detection_dialog.threshold_spin.value()
                ax.axhline(threshold, linestyle=':', linewidth=1.5, color='red', alpha=0.7,
                          label=f'Threshold ({threshold:.2f})', zorder=5)
            except:
                pass

        # Add stim spans to event plot
        for (t0_span, t1_span) in (spans_plot or []):
            if t1_span > t0_span:
                ax.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

        # Set labels
        ax.set_ylabel(f'{st.event_channel}', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(False)

    def _plot_bout_annotations(self, ax_pleth, ax_event, bouts, t0):
        """Plot bout annotations as shaded regions on both subplots with legend."""
        if not bouts:
            return

        # Check if shading is enabled
        shade_enabled = True  # Default to True for backward compatibility
        labels_enabled = True  # Default to True
        hargreaves_mode = False
        if hasattr(self.window, '_event_detection_dialog') and self.window._event_detection_dialog is not None:
            try:
                shade_enabled = self.window._event_detection_dialog.shade_events_check.isChecked()
                labels_enabled = self.window._event_detection_dialog.show_labels_check.isChecked()
                hargreaves_mode = self.window._event_detection_dialog.hargreaves_radio.isChecked()
            except:
                pass

        # Check if a boundary is being dragged (to hide it during drag)
        dragging_region_idx = None
        dragging_edge = None
        try:
            from editing.event_marking_mode import get_dragging_boundary
            dragging_region_idx, dragging_edge, _ = get_dragging_boundary()
        except:
            pass

        # Track if we've already added legend entries (to avoid duplicates)
        added_legend = {'onset': False, 'offset': False, 'region': False}

        for i, bout in enumerate(bouts):
            start = bout['start_time'] - t0  # Normalize time
            end = bout['end_time'] - t0

            # Shaded region on both subplots (only if enabled)
            if shade_enabled:
                if not added_legend['region']:
                    ax_pleth.axvspan(start, end, alpha=0.2, color='cyan', label='Event Region')
                    added_legend['region'] = True
                else:
                    ax_pleth.axvspan(start, end, alpha=0.2, color='cyan')
                ax_event.axvspan(start, end, alpha=0.2, color='cyan')

            # Vertical lines at boundaries
            for ax in [ax_pleth, ax_event]:
                # Check if this boundary is being dragged - if so, skip drawing it
                skip_start = (dragging_region_idx == i and dragging_edge == 'start')
                skip_end = (dragging_region_idx == i and dragging_edge == 'end')

                # Draw start boundary (unless being dragged)
                if not skip_start:
                    if not added_legend['onset']:
                        ax.axvline(start, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Event Onset')
                        added_legend['onset'] = True
                    else:
                        ax.axvline(start, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

                # Draw end boundary (unless being dragged)
                if not skip_end:
                    if not added_legend['offset']:
                        ax.axvline(end, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Event Offset')
                        added_legend['offset'] = True
                    else:
                        ax.axvline(end, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

            # Add labels if enabled (only on pleth subplot to avoid clutter)
            if labels_enabled:
                # Calculate actual times (add t0 back)
                start_time = bout['start_time']
                end_time = bout['end_time']
                duration = end_time - start_time

                # Get y-position for labels (top of axis)
                ylim = ax_pleth.get_ylim()
                label_y = ylim[1] * 0.95  # 95% of the way up

                if hargreaves_mode:
                    # Hargreaves mode: "Heat onset, t=Xs" and "Withdrawal, t=Xs, Latency=Xs"
                    if not skip_start:
                        ax_pleth.text(start, label_y, f'Heat onset\nt={start_time:.2f}s',
                                    fontsize=8, ha='left', va='top', color='green',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
                    if not skip_end:
                        ax_pleth.text(end, label_y, f'Withdrawal\nt={end_time:.2f}s\nLatency={duration:.2f}s',
                                    fontsize=8, ha='right', va='top', color='red',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))
                else:
                    # Normal mode: "event onset t=Xs" and "event offset, t=Xs, dur=Xs"
                    if not skip_start:
                        ax_pleth.text(start, label_y, f'Event onset\nt={start_time:.2f}s',
                                    fontsize=8, ha='left', va='top', color='green',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='green'))
                    if not skip_end:
                        ax_pleth.text(end, label_y, f'Event offset\nt={end_time:.2f}s\ndur={duration:.2f}s',
                                    fontsize=8, ha='right', va='top', color='red',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='red'))

        # Add legend to top subplot
        if any(added_legend.values()):
            ax_pleth.legend(loc='upper right', fontsize=9, framealpha=0.9)

    def _build_plot_title(self, sweep_idx):
        """Build plot title with channel name, sweep number, and file info."""
        st = self.state
        title_parts = [st.analyze_chan or '']

        # Add sweep number
        title_parts.append(f"sweep {sweep_idx+1}")

        # Add file info if multiple files loaded
        if len(st.file_info) > 1:
            for i, info in enumerate(st.file_info):
                if info['sweep_start'] <= sweep_idx <= info['sweep_end']:
                    file_num = i + 1
                    total_files = len(st.file_info)
                    title_parts.append(f"file {file_num}/{total_files} ({info['path'].name})")
                    break

        return " | ".join(title_parts)

    def _draw_peak_markers(self, sweep_idx, t_plot, y):
        """Draw peak markers for current sweep, with gray markers for omitted regions."""
        st = self.state
        pks = getattr(st, "peaks_by_sweep", {}).get(sweep_idx, None)

        if pks is not None and len(pks):
            # Separate peaks into normal and omitted groups
            normal_pks = []
            omitted_pks = []

            for pk in pks:
                if self._is_peak_in_omitted_region(sweep_idx, pk):
                    omitted_pks.append(pk)
                else:
                    normal_pks.append(pk)

            # Draw normal peaks in red
            if len(normal_pks):
                t_normal = t_plot[normal_pks]
                y_normal = y[normal_pks]
                self.plot_host.update_peaks(t_normal, y_normal, size=24)
            else:
                self.plot_host.clear_peaks()

            # Draw omitted peaks in gray
            if len(omitted_pks):
                t_omitted = t_plot[omitted_pks]
                y_omitted = y[omitted_pks]
                ax = self.plot_host.ax_main
                if ax is not None:
                    ax.scatter(t_omitted, y_omitted, s=24, c='gray', alpha=0.5, marker='o', zorder=5, edgecolors='darkgray', linewidths=0.5)
        else:
            self.plot_host.clear_peaks()

    def _draw_sigh_markers(self, sweep_idx, t_plot, y):
        """Draw sigh markers (orange stars) for current sweep."""
        st = self.state
        sigh_idx = getattr(st, "sigh_by_sweep", {}).get(sweep_idx, None)

        if sigh_idx is not None and len(sigh_idx):
            t_sigh = t_plot[sigh_idx]

            # Add vertical offset (7% of y-span by default)
            offset_frac = float(getattr(self.window, "_sigh_offset_frac", 0.07))
            try:
                y_span = float(np.nanmax(y) - np.nanmin(y))
                y_off = offset_frac * (y_span if np.isfinite(y_span) and y_span > 0 else 1.0)
            except Exception:
                y_off = offset_frac
            y_sigh = y[sigh_idx] + y_off

            # Filled orange star with darker edge
            self.plot_host.update_sighs(
                t_sigh, y_sigh,
                size=110,
                color="#ff9f1a",   # warm orange fill
                edge="#a35400",    # slightly darker edge
                filled=True
            )
        else:
            self.plot_host.clear_sighs()

    def _draw_breath_markers(self, sweep_idx, t_plot, y):
        """Draw breath event markers (onsets, offsets, expiratory mins/offs) with gray for omitted regions."""
        st = self.state
        br = getattr(st, "breath_by_sweep", {}).get(sweep_idx, None)

        if br:
            on_idx = br.get("onsets", [])
            off_idx = br.get("offsets", [])
            ex_idx = br.get("expmins", [])
            exoff_idx = br.get("expoffs", [])

            # Separate into normal and omitted markers
            on_normal, on_omit = [], []
            off_normal, off_omit = [], []
            ex_normal, ex_omit = [], []
            exoff_normal, exoff_omit = [], []

            for idx in on_idx:
                (on_omit if self._is_peak_in_omitted_region(sweep_idx, idx) else on_normal).append(idx)
            for idx in off_idx:
                (off_omit if self._is_peak_in_omitted_region(sweep_idx, idx) else off_normal).append(idx)
            for idx in ex_idx:
                (ex_omit if self._is_peak_in_omitted_region(sweep_idx, idx) else ex_normal).append(idx)
            for idx in exoff_idx:
                (exoff_omit if self._is_peak_in_omitted_region(sweep_idx, idx) else exoff_normal).append(idx)

            # Draw normal markers
            t_on = t_plot[on_normal] if len(on_normal) else None
            y_on = y[on_normal] if len(on_normal) else None
            t_off = t_plot[off_normal] if len(off_normal) else None
            y_off = y[off_normal] if len(off_normal) else None
            t_exp = t_plot[ex_normal] if len(ex_normal) else None
            y_exp = y[ex_normal] if len(ex_normal) else None
            t_exof = t_plot[exoff_normal] if len(exoff_normal) else None
            y_exof = y[exoff_normal] if len(exoff_normal) else None

            self.plot_host.update_breath_markers(
                t_on=t_on, y_on=y_on,
                t_off=t_off, y_off=y_off,
                t_exp=t_exp, y_exp=y_exp,
                t_exoff=t_exof, y_exoff=y_exof,
                size=36
            )

            # Draw omitted markers in gray
            ax = self.plot_host.ax_main
            if ax is not None:
                if len(on_omit):
                    ax.scatter(t_plot[on_omit], y[on_omit], s=36, c='gray', alpha=0.5, marker='^', zorder=4, edgecolors='darkgray', linewidths=0.5)
                if len(off_omit):
                    ax.scatter(t_plot[off_omit], y[off_omit], s=36, c='gray', alpha=0.5, marker='v', zorder=4, edgecolors='darkgray', linewidths=0.5)
                if len(ex_omit):
                    ax.scatter(t_plot[ex_omit], y[ex_omit], s=36, c='gray', alpha=0.5, marker='s', zorder=4, edgecolors='darkgray', linewidths=0.5)
                if len(exoff_omit):
                    ax.scatter(t_plot[exoff_omit], y[exoff_omit], s=36, c='gray', alpha=0.5, marker='D', zorder=4, edgecolors='darkgray', linewidths=0.5)
        else:
            self.plot_host.clear_breath_markers()

    def _draw_y2_metric(self, sweep_idx, t, t_plot):
        """Draw Y2 axis metric if selected and available."""
        st = self.state
        key = getattr(st, "y2_metric_key", None)

        if key:
            # Get metric data for current sweep
            arr = st.y2_values_by_sweep.get(sweep_idx, None)
            if arr is not None and len(arr) == len(t):
                # Determine label and color based on metric type
                if key == "if":
                    label = "IF (Hz)"
                    color = "#39FF14"  # Bright green
                elif key == "sniff_conf":
                    label = "Sniffing Confidence"
                    color = "#9b59b6"  # Purple
                elif key == "eupnea_conf":
                    label = "Eupnea Confidence"
                    color = "#2ecc71"  # Green
                else:
                    label = key
                    color = "#39FF14"  # Default bright green

                self.plot_host.add_or_update_y2(t_plot, arr, label=label, color=color, max_points=None)
                self.plot_host.fig.tight_layout()
            else:
                self.plot_host.clear_y2()
                self.plot_host.fig.tight_layout()
        else:
            self.plot_host.clear_y2()
            self.plot_host.fig.tight_layout()

    def _draw_region_overlays(self, sweep_idx, t, y, t_plot):
        """Draw automatic region overlays (eupnea, apnea, outliers)."""
        st = self.state

        try:
            br = getattr(st, "breath_by_sweep", {}).get(sweep_idx, None)
            if br and len(t) > 100:  # Only if we have breath data and sufficient points
                # Extract breath events
                pks = getattr(st, "peaks_by_sweep", {}).get(sweep_idx, None)
                on_idx = br.get("onsets", [])
                off_idx = br.get("offsets", [])
                ex_idx = br.get("expmins", [])
                exoff_idx = br.get("expoffs", [])

                # Get thresholds
                eupnea_thresh = self.window.eupnea_freq_threshold
                apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5

                # Compute eupnea regions based on selected mode
                if self.window.eupnea_detection_mode == "gmm":
                    # GMM-based: Eupnea = breaths NOT classified as sniffing
                    eupnea_mask = self.window._compute_eupnea_from_gmm(sweep_idx, len(y))
                else:
                    # Frequency-based (legacy): Use threshold method
                    sniff_regions = self.state.sniff_regions_by_sweep.get(sweep_idx, [])
                    eupnea_mask = metrics.detect_eupnic_regions(
                        t, y, st.sr_hz, pks, on_idx, off_idx, ex_idx, exoff_idx,
                        freq_threshold_hz=eupnea_thresh,
                        min_duration_sec=self.window.eupnea_min_duration,
                        sniff_regions=sniff_regions
                    )

                # Compute apnea regions
                apnea_mask = metrics.detect_apneas(
                    t, y, st.sr_hz, pks, on_idx, off_idx, ex_idx, exoff_idx,
                    min_apnea_duration_sec=apnea_thresh
                )

                # Identify problematic breaths using outlier detection
                outlier_mask, failure_mask = self._compute_outlier_masks(
                    sweep_idx, t, y, pks, on_idx, off_idx, ex_idx, exoff_idx
                )

                # Apply the overlays
                self.plot_host.update_region_overlays(t_plot, eupnea_mask, apnea_mask, outlier_mask, failure_mask)
            else:
                self.plot_host.clear_region_overlays()
        except Exception as e:
            print(f"Warning: Could not compute region overlays: {e}")
            self.plot_host.clear_region_overlays()

    def _compute_outlier_masks(self, sweep_idx, t, y, pks, on_idx, off_idx, ex_idx, exoff_idx):
        """Compute outlier and failure masks for problematic breath detection."""
        st = self.state
        outlier_mask = None
        failure_mask = None

        try:
            from core.breath_outliers import identify_problematic_breaths, compute_global_metric_statistics

            # Convert indices to arrays
            peaks_arr = np.array(pks) if pks is not None and len(pks) > 0 else np.array([])
            onsets_arr = np.array(on_idx) if on_idx is not None and len(on_idx) > 0 else np.array([])
            offsets_arr = np.array(off_idx) if off_idx is not None and len(off_idx) > 0 else np.array([])
            expmins_arr = np.array(ex_idx) if ex_idx is not None and len(ex_idx) > 0 else np.array([])
            expoffs_arr = np.array(exoff_idx) if exoff_idx is not None and len(exoff_idx) > 0 else np.array([])

            # Get all computed metrics for this sweep
            metrics_dict = {}
            for metric_key in ["if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"]:
                if metric_key in metrics.METRICS:
                    metric_arr = metrics.METRICS[metric_key](
                        t, y, st.sr_hz, peaks_arr, onsets_arr, offsets_arr, expmins_arr, expoffs_arr
                    )
                    metrics_dict[metric_key] = metric_arr

            # Store metrics for this sweep (for cross-sweep outlier detection)
            self.window.metrics_by_sweep[sweep_idx] = metrics_dict
            self.window.onsets_by_sweep[sweep_idx] = onsets_arr

            # Compute global statistics across all sweeps (if we have multiple sweeps analyzed)
            if len(self.window.metrics_by_sweep) >= 2:
                self.window.global_outlier_stats = compute_global_metric_statistics(
                    self.window.metrics_by_sweep,
                    self.window.onsets_by_sweep,
                    self.window.outlier_metrics
                )
            else:
                self.window.global_outlier_stats = None

            # Get outlier threshold from UI
            outlier_sd = self.window._parse_float(self.window.OutlierSD) or 3.0

            # Identify problematic breaths
            outlier_mask, failure_mask = identify_problematic_breaths(
                t, y, st.sr_hz, peaks_arr, onsets_arr, offsets_arr,
                expmins_arr, expoffs_arr, metrics_dict, outlier_threshold=outlier_sd,
                outlier_metrics=self.window.outlier_metrics,
                global_stats=self.window.global_outlier_stats
            )

        except Exception as outlier_error:
            print(f"Warning: Could not detect breath outliers: {outlier_error}")
            import traceback
            traceback.print_exc()

        return outlier_mask, failure_mask

    def _is_peak_in_omitted_region(self, sweep_idx, peak_idx):
        """Check if a peak index falls within an omitted region for the given sweep."""
        st = self.state
        if sweep_idx not in st.omitted_ranges:
            return False

        for (start_idx, end_idx) in st.omitted_ranges[sweep_idx]:
            if start_idx <= peak_idx <= end_idx:
                return True
        return False

    def _draw_omitted_regions(self, sweep_idx, t_plot):
        """Draw semi-transparent overlays for omitted regions, or full sweep if sweep is omitted."""
        st = self.state
        print(f"[_draw_omitted_regions] Called for sweep {sweep_idx}")

        ax = self.plot_host.ax_main
        if ax is None:
            print(f"[_draw_omitted_regions] No axes found!")
            return

        # Check if full sweep is omitted
        if sweep_idx in st.omitted_sweeps:
            print(f"[_draw_omitted_regions] Full sweep {sweep_idx} is omitted - drawing full overlay")
            # Draw gray overlay over entire visible time range
            xlim = ax.get_xlim()
            ax.axvspan(xlim[0], xlim[1], color='gray', alpha=0.4, zorder=100,
                      linewidth=1, edgecolor='darkgray', linestyle='--')

            # Add "omitted" label in center
            t_center = (xlim[0] + xlim[1]) / 2.0
            y_limits = ax.get_ylim()
            y_center = (y_limits[0] + y_limits[1]) / 2.0
            ax.text(t_center, y_center, 'omitted',
                   ha='center', va='center',
                   fontsize=10, color='darkgray',
                   weight='bold', alpha=0.7, zorder=101,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))
            return

        # Otherwise draw smaller omitted regions
        print(f"[_draw_omitted_regions] omitted_ranges keys: {list(st.omitted_ranges.keys())}")

        if sweep_idx not in st.omitted_ranges:
            print(f"[_draw_omitted_regions] No omitted ranges for sweep {sweep_idx}")
            return

        print(f"[_draw_omitted_regions] Found {len(st.omitted_ranges[sweep_idx])} regions: {st.omitted_ranges[sweep_idx]}")

        # Convert sample indices to time for plotting
        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        for (i_start, i_end) in st.omitted_ranges[sweep_idx]:
            t_start = i_start / sr_hz
            t_end = i_end / sr_hz

            # Adjust times if plot is normalized to stim onset
            if st.stim_chan:
                s = max(0, min(sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))
                spans = st.stim_spans_by_sweep.get(s, [])
                if spans:
                    t0 = spans[0][0]
                    t_start -= t0
                    t_end -= t0

            print(f"[_draw_omitted_regions] Drawing region at {t_start:.3f} - {t_end:.3f}s")

            # Draw semi-transparent gray overlay for omitted region (high zorder to appear on top)
            ax.axvspan(t_start, t_end, color='gray', alpha=0.4, zorder=100,
                      linewidth=1, edgecolor='darkgray', linestyle='--')

            # Add "omitted" text label in center of region
            t_center = (t_start + t_end) / 2.0
            y_limits = ax.get_ylim()
            y_center = (y_limits[0] + y_limits[1]) / 2.0
            ax.text(t_center, y_center, 'omitted',
                   ha='center', va='center',
                   fontsize=10, color='darkgray',
                   weight='bold', alpha=0.7, zorder=101,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))

        print(f"[_draw_omitted_regions] Finished drawing {len(st.omitted_ranges[sweep_idx])} regions")

    def _apply_omitted_dimming(self):
        """Dim the plot and hide markers for omitted sweeps."""
        fig = self.plot_host.fig
        if fig and fig.axes:
            ax = fig.axes[0]
            # Hide detail markers so the dimming reads clearly
            self.plot_host.clear_peaks()
            self.plot_host.clear_breath_markers()
            self.dim_axes_for_omitted(ax, label=True)
            self.plot_host.fig.tight_layout()
            self.plot_host.canvas.draw_idle()

    def plot_all_channels(self):
        """Plot the current sweep for every channel, one panel per channel (grid mode)."""
        st = self.state
        if not st.channel_names or st.t is None:
            return

        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))

        traces = []
        for ch_name in st.channel_names:
            Y = st.sweeps[ch_name]  # (n_samples, n_sweeps)
            y = Y[:, s]  # current sweep (1D)
            # No filtering in preview - show raw data to avoid distorting stimulus channels
            traces.append((st.t, y, ch_name))

        # Build title with sweep and file info
        title_parts = ["All channels"]
        title_parts.append(f"sweep {s+1}")

        # Add file info if multiple files loaded
        if len(st.file_info) > 1:
            for i, info in enumerate(st.file_info):
                if info['sweep_start'] <= s <= info['sweep_end']:
                    file_num = i + 1
                    total_files = len(st.file_info)
                    title_parts.append(f"file {file_num}/{total_files} ({info['path'].name})")
                    break

        title = " | ".join(title_parts)

        # Adaptive downsampling: only for very long traces
        max_pts = None if len(st.t) < 100000 else 50000

        self.plot_host.show_multi_grid(
            traces,
            title=title,
            max_points_per_trace=max_pts
        )

    def refresh_threshold_lines(self):
        """
        (Re)draw the dashed threshold line on the current visible axes.
        Called after redraws that clear the axes.
        """
        # Use the new PlotHost threshold line system
        threshold_value = getattr(self.window, "peak_height_threshold", None)
        if threshold_value is not None:
            self.plot_host.update_threshold_line(threshold_value)
        else:
            self.plot_host.clear_threshold_line()

    def dim_axes_for_omitted(self, ax, label=True):
        """Grey overlay + 'OMITTED' watermark on given axes."""
        x0, x1 = ax.get_xlim()
        ax.axvspan(x0, x1, ymin=0.0, ymax=1.0, color="#6c7382", alpha=0.22, zorder=50)
        if label:
            ax.text(0.5, 0.5, "OMITTED",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=22, weight="bold", color="#d7dce7", alpha=0.65, zorder=60)

    def _restore_editing_mode_connections(self):
        """Restore matplotlib event connections for active editing modes after redraw."""
        editing = self.window.editing_modes

        # DON'T reconnect _cid_button - it survives fig.clear() and reconnecting creates duplicates!
        # The canvas-level connection is persistent, we just need to restore the callback registration.

        # Check if Mark Sniff mode is active and needs reconnection
        if getattr(editing, "_mark_sniff_mode", False):
            # Re-register the click callback (this is the key!)
            self.window.plot_host.set_click_callback(editing._on_plot_click_mark_sniff)

            # Disconnect old connections if they exist
            if editing._motion_cid is not None:
                try:
                    self.window.plot_host.canvas.mpl_disconnect(editing._motion_cid)
                except:
                    pass
            if editing._release_cid is not None:
                try:
                    self.window.plot_host.canvas.mpl_disconnect(editing._release_cid)
                except:
                    pass

            # Reconnect with fresh connections
            editing._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', editing._on_sniff_drag)
            editing._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', editing._on_sniff_release)

        # Check for other active editing modes that need click callback restoration
        if getattr(editing, "_add_peaks_mode", False):
            self.window.plot_host.set_click_callback(editing._on_plot_click_add_peak)
        elif getattr(editing, "_delete_peaks_mode", False):
            self.window.plot_host.set_click_callback(editing._on_plot_click_delete_peak)
        elif getattr(editing, "_add_sigh_mode", False):
            self.window.plot_host.set_click_callback(editing._on_plot_click_add_sigh)

        # Check if Move Point mode is active and needs reconnection
        if getattr(editing, "_move_point_mode", False):
            self.window.plot_host.set_click_callback(editing._on_plot_click_move_point)

            # Disconnect old connections
            if editing._key_press_cid is not None:
                try:
                    self.window.plot_host.canvas.mpl_disconnect(editing._key_press_cid)
                except:
                    pass
            if editing._motion_cid is not None:
                try:
                    self.window.plot_host.canvas.mpl_disconnect(editing._motion_cid)
                except:
                    pass
            if editing._release_cid is not None:
                try:
                    self.window.plot_host.canvas.mpl_disconnect(editing._release_cid)
                except:
                    pass

            # Reconnect with fresh connections
            editing._key_press_cid = self.window.plot_host.canvas.mpl_connect('key_press_event', editing._on_canvas_key_press)
            editing._motion_cid = self.window.plot_host.canvas.mpl_connect('motion_notify_event', editing._on_canvas_motion)
            editing._release_cid = self.window.plot_host.canvas.mpl_connect('button_release_event', editing._on_canvas_release)
