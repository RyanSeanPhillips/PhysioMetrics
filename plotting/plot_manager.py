"""
PlotManager - Orchestrates all plotting operations for the main window.

This module extracts plotting logic from main.py to improve maintainability.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from core import metrics


@dataclass
class ChannelPanelConfig:
    """Configuration for a single channel panel in multi-channel plotting.

    This standardizes channel configuration across all plotting code paths,
    enabling unified handling of 1-N channel displays.
    """
    name: str                          # Channel name
    channel_type: str                  # "Pleth", "Opto Stim", "Event", "Raw Signal"
    height_ratio: float = 0.25         # Relative height (0.0-1.0, normalized later)
    show_overlays: bool = False        # Show peak/breath/region overlays (Pleth only)
    is_primary_pleth: bool = False     # Is this the primary Pleth for editing/analysis
    is_event_channel: bool = False     # Show bout annotations on this panel
    show_stim_spans: bool = False      # Show blue stim background spans
    apply_filtering: bool = False      # Apply filtering (Pleth channels only)

    @classmethod
    def from_channel_manager(cls, name: str, config, is_primary: bool = False) -> 'ChannelPanelConfig':
        """Create ChannelPanelConfig from a ChannelManager ChannelConfig.

        Args:
            name: Channel name
            config: ChannelConfig from channel_manager.py
            is_primary: Whether this is the primary Pleth channel for analysis
        """
        channel_type = config.channel_type

        # Determine height ratio based on channel type
        if channel_type == "Pleth":
            height_ratio = 0.6
        elif channel_type == "Opto Stim":
            height_ratio = 0.15
        elif channel_type == "Event":
            height_ratio = 0.25
        else:  # Raw Signal
            height_ratio = 0.25

        return cls(
            name=name,
            channel_type=channel_type,
            height_ratio=height_ratio,
            show_overlays=(channel_type == "Pleth" and is_primary),
            is_primary_pleth=is_primary,
            is_event_channel=(channel_type == "Event"),
            show_stim_spans=(channel_type == "Pleth"),
            apply_filtering=(channel_type == "Pleth"),
        )

    @classmethod
    def for_single_panel(cls, name: str) -> 'ChannelPanelConfig':
        """Create config for single-panel mode (legacy mode without channel manager)."""
        return cls(
            name=name,
            channel_type="Pleth",
            height_ratio=1.0,
            show_overlays=True,
            is_primary_pleth=True,
            is_event_channel=False,
            show_stim_spans=True,
            apply_filtering=True,
        )

    @classmethod
    def for_grid_preview(cls, name: str) -> 'ChannelPanelConfig':
        """Create config for grid preview mode (all channels, no overlays)."""
        return cls(
            name=name,
            channel_type="Raw Signal",  # All channels treated as raw in preview
            height_ratio=1.0,  # Equal height for all in grid
            show_overlays=False,
            is_primary_pleth=False,
            is_event_channel=False,
            show_stim_spans=False,
            apply_filtering=False,  # No filtering in preview
        )


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

    def redraw_main_plot(self, use_unified: bool = True):
        """
        Main plot orchestration method.
        Determines whether to show single-panel, multi-channel, or grid mode.

        Args:
            use_unified: If True, use the new unified draw_channels() method.
                        If False, use legacy methods (for backward compatibility during transition).
        """
        st = self.state
        if st.t is None:
            return

        if use_unified:
            # NEW: Use unified draw_channels() for all modes
            configs = self._build_channel_configs_from_manager()

            if configs:
                # Channel manager has visible channels - use unified drawing
                self.draw_channels(configs, grid_mode=False)
                self._restore_editing_mode_connections()
            elif self.window.single_panel_mode:
                # No channel manager config - create single Pleth config
                single_config = ChannelPanelConfig.for_single_panel(st.analyze_chan or "Signal")

                # Check for event channel and add it if present
                if st.event_channel and st.event_channel in st.sweeps:
                    event_config = ChannelPanelConfig(
                        name=st.event_channel,
                        channel_type="Event",
                        height_ratio=0.3,
                        show_overlays=False,
                        is_primary_pleth=False,
                        is_event_channel=True,
                        show_stim_spans=False,
                        apply_filtering=False,
                    )
                    # Adjust Pleth height for dual layout
                    single_config.height_ratio = 0.7
                    self.draw_channels([single_config, event_config], grid_mode=False)
                else:
                    self.draw_channels([single_config], grid_mode=False)

                self._restore_editing_mode_connections()
            else:
                # Grid preview mode - all channels, equal size, no overlays
                grid_configs = [
                    ChannelPanelConfig.for_grid_preview(ch_name)
                    for ch_name in st.channel_names
                ]
                self.draw_channels(grid_configs, grid_mode=True)
        else:
            # LEGACY: Use old methods (for transition/debugging)
            channel_config = self._get_channel_manager_config()
            visible_channels = channel_config['visible_channels']

            if len(visible_channels) >= 1:
                self._draw_multi_channel_plot(channel_config)
                self._restore_editing_mode_connections()
            elif self.window.single_panel_mode:
                self._draw_single_panel_plot()
                self._restore_editing_mode_connections()
            else:
                self.plot_all_channels()

    def draw_channels(self, channel_configs: List[ChannelPanelConfig], grid_mode: bool = False):
        """
        Unified multi-channel plotting method that handles 1-N channels.

        This is the core drawing method that all plotting modes should use.
        It handles:
        - Single channel (1 config)
        - Dual channels (2 configs, e.g., Pleth + Event)
        - Multi-channel (N configs from Channel Manager)
        - Grid preview mode (all channels, equal size, no overlays)

        Args:
            channel_configs: List of ChannelPanelConfig objects defining what to plot
            grid_mode: If True, use grid layout instead of stacked panels
        """
        st = self.state

        # Check backend and route to appropriate implementation
        if st.plotting_backend == 'pyqtgraph':
            self._draw_channels_pyqtgraph(channel_configs, grid_mode)
            return

        # Matplotlib implementation
        from matplotlib.gridspec import GridSpec

        if not channel_configs:
            return

        n_panels = len(channel_configs)

        # Get current sweep info
        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))

        # Get time normalization (stim offset)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = st.t - t0
            spans_plot = [(a - t0, b - t0) for (a, b) in spans]
        else:
            t0 = 0.0
            t_plot = st.t
            spans_plot = spans

        # Extract Opto Stim spans from any Opto Stim channel (for blue backgrounds)
        opto_spans_plot = []
        for config in channel_configs:
            if config.channel_type == "Opto Stim" and config.name in st.sweeps:
                opto_data = st.sweeps[config.name][:, s]
                opto_spans_plot = self._extract_stim_spans_from_channel(st.t, opto_data, t0)
                break  # Use first opto channel found

        # Also check hidden opto channels (from channel manager)
        if not opto_spans_plot and hasattr(self.window, 'channel_manager'):
            channels = self.window.channel_manager.get_channels()
            for name, cm_config in channels.items():
                if cm_config.channel_type == "Opto Stim" and name in st.sweeps:
                    opto_data = st.sweeps[name][:, s]
                    opto_spans_plot = self._extract_stim_spans_from_channel(st.t, opto_data, t0)
                    break

        # Calculate height ratios (normalize)
        if grid_mode:
            # Grid mode: equal heights
            height_ratios = [1.0] * n_panels
        else:
            height_ratios = [cfg.height_ratio for cfg in channel_configs]
            total = sum(height_ratios)
            if total > 0:
                height_ratios = [h/total for h in height_ratios]

        # Save previous view if needed
        prev_xlim = self.plot_host._last_single["xlim"] if self.plot_host._preserve_x else None
        prev_ylim = self.plot_host._last_single["ylim"] if self.plot_host._preserve_y else None

        # Clear figure and create layout
        self.plot_host.fig.clear()

        if grid_mode and n_panels > 1:
            # Grid layout: calculate rows and columns
            import math
            n_cols = min(3, n_panels)  # Max 3 columns
            n_rows = math.ceil(n_panels / n_cols)
            gs = GridSpec(n_rows, n_cols, hspace=0.3, wspace=0.3, figure=self.plot_host.fig)
        else:
            # Stacked layout: single column
            gs = GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.05, figure=self.plot_host.fig)

        # Create subplots
        axes = []
        ax_main = None  # Primary Pleth axis for overlays
        ax_event = None  # Event channel axis for bout annotations

        for i, config in enumerate(channel_configs):
            if grid_mode and n_panels > 1:
                row = i // n_cols
                col = i % n_cols
                ax = self.plot_host.fig.add_subplot(gs[row, col])
            else:
                if i == 0:
                    ax = self.plot_host.fig.add_subplot(gs[i])
                else:
                    ax = self.plot_host.fig.add_subplot(gs[i], sharex=axes[0])
            axes.append(ax)

            # Track primary Pleth axis
            if config.is_primary_pleth:
                ax_main = ax
                self.plot_host.ax_main = ax

            # Track Event channel axis
            if config.is_event_channel:
                ax_event = ax
                self.plot_host.ax_event = ax
                # Sync event channel to state
                if config.name != st.event_channel:
                    st.event_channel = config.name

            # Hide x-axis labels except on bottom (for stacked layout)
            if not grid_mode and i < n_panels - 1:
                ax.tick_params(labelbottom=False)

            # Apply theme
            self.plot_host.theme_manager.apply_theme(ax, self.plot_host.fig, self.plot_host._current_theme)

        # Get theme colors
        trace_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['trace_color']
        text_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['text_color']

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
        if ax_event is None:
            self.plot_host.ax_event = None

        # Plot each channel
        primary_y_data = None  # Store primary Pleth data for overlays

        for i, (config, ax) in enumerate(zip(channel_configs, axes)):
            # Get channel data
            if config.name not in st.sweeps:
                continue

            y_data = st.sweeps[config.name][:, s]

            # Apply filtering if configured (Pleth channels)
            if config.apply_filtering and config.name == st.analyze_chan:
                _, y_filtered = self.window._current_trace()
                y_data = y_filtered

            # Store primary Pleth data for overlays
            if config.is_primary_pleth:
                primary_y_data = y_data

            # Plot trace
            ax.plot(t_plot, y_data, linewidth=0.9, color=trace_color)
            ax.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

            # Add stim spans (blue background) if configured
            if config.show_stim_spans:
                # Prefer Opto Stim channel spans
                all_spans = opto_spans_plot if opto_spans_plot else list(spans_plot)
                for (t0_span, t1_span) in all_spans:
                    if t1_span > t0_span:
                        ax.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

            # Set labels
            ax.set_ylabel(config.name, fontsize=10)
            ax.grid(False)

            # Apply Y-axis autoscaling
            if not grid_mode:
                self._autoscale_axis(ax, y_data)

        # Set title on first axis
        title = self._build_plot_title(s)
        if axes:
            axes[0].set_title(title, color=text_color)

        # Set xlabel on bottom axis (for stacked layout)
        if axes and not grid_mode:
            axes[-1].set_xlabel('Time (s)', fontsize=10)

        # Draw overlays on primary Pleth channel (if configured)
        if ax_main is not None and primary_y_data is not None:
            # Find the primary config
            primary_config = next((c for c in channel_configs if c.is_primary_pleth), None)

            if primary_config and primary_config.show_overlays:
                # Clear region overlays (will be redrawn)
                self.plot_host.clear_region_overlays()

                # Draw peak markers
                self._draw_peak_markers(s, t_plot, primary_y_data)

                # Draw sigh markers
                self._draw_sigh_markers(s, t_plot, primary_y_data)

                # Draw breath markers
                self._draw_breath_markers(s, t_plot, primary_y_data)

                # Draw Y2 metric if selected
                self._draw_y2_metric(s, st.t, t_plot)

                # Draw omitted region overlays
                self._draw_omitted_regions(s, t_plot)

                # Refresh threshold lines
                self.refresh_threshold_lines()

                # Set y-limits excluding omitted
                self._set_ylim_excluding_omitted(s, primary_y_data, t_plot)

                # Draw region overlays (eupnea, apnea, etc.)
                self._draw_region_overlays(s, st.t, primary_y_data, t_plot, t0)

        # Draw bout annotations on Event channel (if present)
        if ax_event is not None and ax_main is not None:
            if s in st.bout_annotations and st.bout_annotations[s]:
                self._plot_bout_annotations(ax_main, ax_event, st.bout_annotations[s], t0)

        # Restore preserved view or set default xlim
        if axes:
            if prev_xlim is not None:
                axes[0].set_xlim(prev_xlim)
            elif not grid_mode:
                # Set default xlim to full time range
                axes[0].set_xlim(t_plot[0], t_plot[-1])
        if prev_ylim is not None and ax_main is not None:
            ax_main.set_ylim(prev_ylim)

        # Set up limit listeners
        if not grid_mode:
            self.plot_host._attach_limit_listeners(axes, mode="single")
            self.plot_host._store_from_axes(mode="single")

        # Layout adjustments
        if grid_mode:
            self.plot_host.fig.tight_layout(pad=1.0)
        else:
            self.plot_host.fig.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
            self.plot_host.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)

        self.plot_host.canvas.draw_idle()

    def _draw_channels_pyqtgraph(self, channel_configs: List[ChannelPanelConfig], grid_mode: bool = False):
        """
        PyQtGraph-specific implementation of draw_channels().

        Uses native PyQtGraph APIs for high-performance rendering.
        """
        import pyqtgraph as pg
        st = self.state

        if not channel_configs:
            return

        n_panels = len(channel_configs)

        # Get current sweep info
        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))

        # Get time normalization (stim offset)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = st.t - t0
            spans_plot = [(a - t0, b - t0) for (a, b) in spans]
        else:
            t0 = 0.0
            t_plot = st.t
            spans_plot = spans

        # Extract Opto Stim spans from any Opto Stim channel (for blue backgrounds)
        opto_spans_plot = []
        for config in channel_configs:
            if config.channel_type == "Opto Stim" and config.name in st.sweeps:
                opto_data = st.sweeps[config.name][:, s]
                opto_spans_plot = self._extract_stim_spans_from_channel(st.t, opto_data, t0)
                break

        # Also check hidden opto channels
        if not opto_spans_plot and hasattr(self.window, 'channel_manager'):
            channels = self.window.channel_manager.get_channels()
            for name, cm_config in channels.items():
                if cm_config.channel_type == "Opto Stim" and name in st.sweeps:
                    opto_data = st.sweeps[name][:, s]
                    opto_spans_plot = self._extract_stim_spans_from_channel(st.t, opto_data, t0)
                    break

        # Get theme colors
        is_dark = self.plot_host._current_theme == 'dark'
        trace_color = '#d4d4d4' if is_dark else '#000000'
        opto_trace_color = '#4682E0' if is_dark else '#2563EB'  # Blue for opto stim channel
        text_color = '#d4d4d4' if is_dark else '#000000'
        bg_color = '#1e1e1e' if is_dark else '#ffffff'

        # Check if we should force auto-range (e.g., after snap to sweep)
        force_autorange = getattr(self.plot_host, '_force_autorange', False)
        if force_autorange:
            self.plot_host._force_autorange = False  # Reset flag

        # Save current view range to prevent jumpiness on sweep change
        # Skip if force_autorange is set (user wants full view)
        prev_x_range = None
        prev_y_range = None
        if not force_autorange and hasattr(self.plot_host, '_subplots') and self.plot_host._subplots:
            try:
                first_plot = self.plot_host._subplots[0]
                if self.plot_host._preserve_x:
                    prev_x_range = first_plot.viewRange()[0]
                if self.plot_host._preserve_y:
                    prev_y_range = first_plot.viewRange()[1]
            except:
                pass

        # IMPORTANT: Clear Y2 elements BEFORE clearing graphics_layout
        # Because Y2 viewbox is added to the scene separately and won't be cleared by layout.clear()
        self._cleanup_y2_elements_pyqtgraph()

        # Reset ALL stale references BEFORE clearing layout to avoid GraphicsLayoutWidget AttributeError
        # (items try to access their old parent during clear() and callbacks fire with stale refs)
        self.plot_host.ax_main = None
        self.plot_host.ax_event = None
        self.plot_host.plot_widget = None
        self.plot_host._main_trace = None
        self.plot_host._peak_scatter = None
        self.plot_host._onset_scatter = None
        self.plot_host._offset_scatter = None
        self.plot_host._expmin_scatter = None
        self.plot_host._expoff_scatter = None
        self.plot_host._sigh_scatter = None
        self.plot_host._threshold_line = None
        self.plot_host._threshold_line_neg = None
        self.plot_host._y2_line = None
        self.plot_host._y2_viewbox = None
        self.plot_host._y2_axis = None
        self.plot_host._y2_update_connection = None
        self.plot_host._span_items = []
        self.plot_host._region_overlays = []
        self.plot_host._subplots = []

        # Clear and rebuild layout
        # Suppress PyQtGraph's internal 'autoRangeEnabled' AttributeError during clear()
        # This happens when PlotDataItems try to access their view during reparenting
        # It's a known PyQtGraph issue that doesn't affect functionality
        import sys
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()  # Temporarily suppress stderr
        try:
            self.plot_host.graphics_layout.clear()
        finally:
            sys.stderr = old_stderr  # Restore stderr
        self.plot_host.graphics_layout.setBackground(bg_color)

        # Create subplots
        plots = []
        ax_main = None
        ax_event = None
        primary_y_data = None

        for i, config in enumerate(channel_configs):
            # Create plot widget
            plot = self.plot_host.graphics_layout.addPlot(row=i, col=0)
            plot.showGrid(x=False, y=False)

            # Disable context menu for right-click editing support
            plot.setMenuEnabled(False)
            plot.vb.setMenuEnabled(False)  # Also disable on ViewBox

            # Style axes
            plot.getAxis('bottom').setTextPen(text_color)
            plot.getAxis('left').setTextPen(text_color)
            plot.getAxis('bottom').setPen(text_color)
            plot.getAxis('left').setPen(text_color)

            # Link x-axis to first plot
            if i > 0 and plots:
                plot.setXLink(plots[0])

            # Hide x-axis labels except on bottom
            if i < n_panels - 1:
                plot.getAxis('bottom').setStyle(showValues=False)

            plots.append(plot)
            self.plot_host._subplots.append(plot)

            # Track primary Pleth axis
            if config.is_primary_pleth:
                ax_main = plot
                self.plot_host.ax_main = plot

            # Track Event channel axis
            if config.is_event_channel:
                ax_event = plot
                self.plot_host.ax_event = plot
                if config.name != st.event_channel:
                    st.event_channel = config.name

            # Get channel data
            if config.name not in st.sweeps:
                continue

            y_data = st.sweeps[config.name][:, s]

            # Apply filtering if configured (Pleth channels)
            if config.apply_filtering and config.name == st.analyze_chan:
                _, y_filtered = self.window._current_trace()
                y_data = y_filtered

            # Store primary Pleth data for overlays
            if config.is_primary_pleth:
                primary_y_data = y_data

            # Choose trace color based on channel type
            channel_trace_color = opto_trace_color if config.channel_type == "Opto Stim" else trace_color

            # Plot trace with optimal settings for signal fidelity
            # clipToView=True only sends visible data to GPU (main performance optimization)
            # autoDownsample=False gives full resolution for visible data
            # Use width=1.2 for slightly thicker lines that render better
            plot.plot(t_plot, y_data, pen=pg.mkPen(channel_trace_color, width=1.2),
                      clipToView=True, autoDownsample=False)

            # Add zero line
            zero_line = pg.InfiniteLine(pos=0, angle=0,
                                        pen=pg.mkPen('#666666', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
            zero_line.setZValue(-5)
            plot.addItem(zero_line)

            # Add stim spans (blue background) if configured
            if config.show_stim_spans:
                all_spans = opto_spans_plot if opto_spans_plot else list(spans_plot)
                # Use subtle border pen to ensure thin pulses remain visible at all zoom levels
                stim_pen = pg.mkPen(70, 130, 220, 150, width=1)
                for (t0_span, t1_span) in all_spans:
                    if t1_span > t0_span:
                        region = pg.LinearRegionItem(
                            values=[t0_span, t1_span],
                            brush=pg.mkBrush(70, 130, 220, 100),  # Brighter blue with alpha
                            pen=stim_pen,  # Border helps with aliasing on thin regions
                            movable=False
                        )
                        region.setZValue(-10)
                        plot.addItem(region)

            # Set labels
            plot.setLabel('left', config.name)

        # Set title on first plot
        if plots:
            title = self._build_plot_title(s)
            plots[0].setTitle(title, color=text_color)

        # Set xlabel on bottom plot
        if plots:
            plots[-1].setLabel('bottom', 'Time (s)')

        # Update main plot reference
        self.plot_host.plot_widget = plots[0] if plots else None

        # Draw overlays on primary Pleth channel (if configured and we have data)
        if ax_main is not None and primary_y_data is not None:
            primary_config = next((c for c in channel_configs if c.is_primary_pleth), None)

            if primary_config and primary_config.show_overlays:
                # Draw peak markers
                self._draw_peaks_pyqtgraph(ax_main, s, t_plot, primary_y_data)

                # Draw breath markers
                self._draw_breath_markers_pyqtgraph(ax_main, s, t_plot, primary_y_data)

                # Draw sigh markers (stars)
                self._draw_sigh_markers_pyqtgraph(ax_main, s, t_plot, primary_y_data)

                # Draw region overlays (eupnea, apnea, etc.)
                self._draw_region_overlays_pyqtgraph(ax_main, s, t_plot, t0)

                # Draw omitted region overlays
                self._draw_omitted_regions_pyqtgraph(ax_main, s, t_plot, t0)

                # Draw threshold line
                self.refresh_threshold_lines()

                # Draw Y2 metric if configured
                self._draw_y2_metric_pyqtgraph(s, t_plot)

        # View range handling:
        # - If we have previous view range, restore it and disable auto-range
        # - If force_autorange, let auto-range work and then disable for future edits
        # - Otherwise (first draw), let auto-range complete
        view_was_restored = False
        if plots and (prev_x_range is not None or prev_y_range is not None):
            try:
                first_plot = plots[0]
                if prev_x_range is not None:
                    first_plot.setXRange(prev_x_range[0], prev_x_range[1], padding=0)
                if prev_y_range is not None:
                    first_plot.setYRange(prev_y_range[0], prev_y_range[1], padding=0)
                view_was_restored = True
            except:
                pass

        # Disable auto-range after view is set to prevent jumpiness during edits
        # Only disable if we successfully restored a view OR if force_autorange completed
        if plots and (view_was_restored or force_autorange):
            if hasattr(self.plot_host, 'disable_autorange'):
                self.plot_host.disable_autorange()

    def _draw_peaks_pyqtgraph(self, plot, sweep_idx, t_plot, y_data):
        """Draw peak markers on PyQtGraph plot, with gray markers for omitted regions."""
        import pyqtgraph as pg
        st = self.state

        if sweep_idx not in st.peaks_by_sweep:
            return

        peak_idxs = st.peaks_by_sweep[sweep_idx]
        if len(peak_idxs) == 0:
            return

        # Separate peaks into normal and omitted groups
        normal_pks = []
        omitted_pks = []
        for pk in peak_idxs:
            if self._is_peak_in_omitted_region(sweep_idx, pk):
                omitted_pks.append(pk)
            else:
                normal_pks.append(pk)

        # Draw normal peaks in red
        if normal_pks:
            normal_pks = np.array(normal_pks)
            t_peaks = t_plot[normal_pks]
            y_peaks = y_data[normal_pks]
            scatter = pg.ScatterPlotItem(
                x=t_peaks, y=y_peaks,
                size=10, brush=pg.mkBrush(255, 0, 0),
                pen=None
            )
            scatter.setZValue(10)
            plot.addItem(scatter)

        # Draw omitted peaks in gray
        if omitted_pks:
            omitted_pks = np.array(omitted_pks)
            t_peaks_omit = t_plot[omitted_pks]
            y_peaks_omit = y_data[omitted_pks]
            scatter_omit = pg.ScatterPlotItem(
                x=t_peaks_omit, y=y_peaks_omit,
                size=10, brush=pg.mkBrush(128, 128, 128, 150),
                pen=None
            )
            scatter_omit.setZValue(9)  # Slightly below normal peaks
            plot.addItem(scatter_omit)

    def _draw_breath_markers_pyqtgraph(self, plot, sweep_idx, t_plot, y_data):
        """Draw breath event markers on PyQtGraph plot, with gray for omitted regions."""
        import pyqtgraph as pg
        st = self.state

        if sweep_idx not in st.breath_by_sweep:
            return

        breath = st.breath_by_sweep[sweep_idx]
        gray_color = (128, 128, 128, 150)  # Semi-transparent gray for omitted

        def add_markers(indices, color, symbol):
            if indices is None or len(indices) == 0:
                return
            valid = indices[indices < len(t_plot)]
            if len(valid) == 0:
                return

            # Separate into normal and omitted
            normal_idx = []
            omit_idx = []
            for idx in valid:
                if self._is_peak_in_omitted_region(sweep_idx, idx):
                    omit_idx.append(idx)
                else:
                    normal_idx.append(idx)

            # Draw normal markers with original color
            if normal_idx:
                normal_idx = np.array(normal_idx)
                scatter = pg.ScatterPlotItem(
                    x=t_plot[normal_idx], y=y_data[normal_idx],
                    size=8, brush=pg.mkBrush(*color),
                    symbol=symbol, pen=None
                )
                scatter.setZValue(8)
                plot.addItem(scatter)

            # Draw omitted markers in gray
            if omit_idx:
                omit_idx = np.array(omit_idx)
                scatter_omit = pg.ScatterPlotItem(
                    x=t_plot[omit_idx], y=y_data[omit_idx],
                    size=8, brush=pg.mkBrush(*gray_color),
                    symbol=symbol, pen=None
                )
                scatter_omit.setZValue(7)  # Slightly below normal markers
                plot.addItem(scatter_omit)

        # Onsets (green triangles)
        add_markers(breath.get('onsets'), (46, 204, 113), 't')
        # Offsets (orange triangles)
        add_markers(breath.get('offsets'), (243, 156, 18), 't1')
        # Exp mins (blue squares)
        add_markers(breath.get('expmins'), (31, 120, 180), 's')

    def _draw_sigh_markers_pyqtgraph(self, plot, sweep_idx, t_plot, y_data):
        """Draw sigh markers (gold stars) on PyQtGraph plot."""
        import pyqtgraph as pg
        st = self.state

        sigh_idx = getattr(st, "sigh_by_sweep", {}).get(sweep_idx, None)
        if sigh_idx is None or len(sigh_idx) == 0:
            return

        # Filter to valid indices
        valid_idx = np.array([i for i in sigh_idx if i < len(t_plot)])
        if len(valid_idx) == 0:
            return

        t_sigh = t_plot[valid_idx]

        # Add vertical offset (7% of y-span)
        offset_frac = float(getattr(self.window, "_sigh_offset_frac", 0.07))
        try:
            y_span = float(np.nanmax(y_data) - np.nanmin(y_data))
            y_off = offset_frac * (y_span if np.isfinite(y_span) and y_span > 0 else 1.0)
        except Exception:
            y_off = offset_frac
        y_sigh = y_data[valid_idx] + y_off

        # Theme-aware colors
        is_dark_mode = self.plot_host._current_theme == 'dark'
        if is_dark_mode:
            sigh_color = (255, 215, 0)  # Gold RGB
            edge_color = (255, 215, 0)
        else:
            sigh_color = (255, 215, 0)  # Gold RGB
            edge_color = (0, 0, 0)      # Black outline for light mode

        # Create star scatter plot
        scatter = pg.ScatterPlotItem(
            x=t_sigh, y=y_sigh,
            size=16,  # Larger size for visibility
            brush=pg.mkBrush(*sigh_color),
            symbol='star',
            pen=pg.mkPen(edge_color, width=1.5)
        )
        scatter.setZValue(15)  # Above other markers
        plot.addItem(scatter)

    def _draw_region_overlays_pyqtgraph(self, plot, sweep_idx, t_plot, t0):
        """Draw region overlays (eupnea, apnea, sniffing, outliers, failures) on PyQtGraph plot.

        Supports both shaded backgrounds and horizontal line indicators depending on user preference.
        Computes apnea, outlier, and failure regions dynamically (same as matplotlib version).
        """
        import pyqtgraph as pg
        from core import metrics
        st = self.state

        # Get Y data range from the actual trace data (not view range which may be incorrect)
        # We'll compute this from t_plot bounds and current data
        y_min, y_max = None, None
        try:
            # Get the primary Y data for this sweep to determine actual data range
            if st.analyze_chan and st.analyze_chan in st.sweeps:
                y_data = st.sweeps[st.analyze_chan][:, sweep_idx]
                y_valid = y_data[np.isfinite(y_data)]
                if len(y_valid) > 0:
                    y_min = float(np.min(y_valid))
                    y_max = float(np.max(y_valid))
        except:
            pass

        # Fallback to view range if we couldn't get data range
        if y_min is None or y_max is None:
            y_range = plot.viewRange()[1]
            y_min, y_max = y_range[0], y_range[1]

        def add_shaded_region(start_t, end_t, color_rgba):
            """Helper to add a shaded region with proper offset and no border."""
            start_adj = start_t - t0
            end_adj = end_t - t0
            if end_adj > start_adj:
                region = pg.LinearRegionItem(
                    values=[start_adj, end_adj],
                    brush=pg.mkBrush(*color_rgba),
                    pen=pg.mkPen(None),  # No border lines
                    movable=False
                )
                region.setZValue(-8)
                plot.addItem(region)

        def add_line_indicator(start_t, end_t, color_rgb, y_fraction):
            """Helper to add a horizontal line indicator at a specific Y position.

            Args:
                start_t, end_t: Time range (in plot time, already adjusted for t0)
                color_rgb: RGB tuple (e.g., (46, 125, 50) for green)
                y_fraction: Y position as fraction of VIEW range (0.0=bottom, 1.0=top of visible area)
            """
            if end_t > start_t:
                # Get current VIEW range (what's actually visible on screen)
                # This ensures lines appear at top/bottom of visible area, not data range
                view_y_range = plot.viewRange()[1]
                view_y_min, view_y_max = view_y_range[0], view_y_range[1]

                # If view range is still default [0, 1], use data range as fallback
                if view_y_min == 0 and view_y_max == 1:
                    view_y_min, view_y_max = y_min, y_max

                # Calculate Y position from VIEW range
                y_val = view_y_min + y_fraction * (view_y_max - view_y_min)

                # Draw horizontal line
                line = pg.PlotCurveItem(
                    x=[start_t, end_t],
                    y=[y_val, y_val],
                    pen=pg.mkPen(color_rgb[0], color_rgb[1], color_rgb[2], 200, width=2)
                )
                line.setZValue(10)  # Above traces
                plot.addItem(line)

        # Get display mode preferences
        eupnea_shade = getattr(st, 'eupnea_use_shade', False)
        sniffing_shade = getattr(st, 'sniffing_use_shade', False)
        apnea_shade = getattr(st, 'apnea_use_shade', False)
        outliers_shade = getattr(st, 'outliers_use_shade', False)

        # Draw eupnea regions (green) - from GMM clustering or detection
        eupnea_regions = []
        if hasattr(st, 'eupnea_regions_by_sweep') and sweep_idx in st.eupnea_regions_by_sweep:
            eupnea_regions = st.eupnea_regions_by_sweep[sweep_idx]

        if eupnea_regions:
            if eupnea_shade:
                for start_t, end_t in eupnea_regions:
                    add_shaded_region(start_t, end_t, (46, 125, 50, 80))  # Green shade
            else:
                for start_t, end_t in eupnea_regions:
                    # Adjust for t0 offset
                    add_line_indicator(start_t - t0, end_t - t0, (46, 125, 50), 0.99)  # Green line at top

        # Draw sniffing regions (purple) - same Y position as eupnea since they don't overlap
        sniff_regions = st.sniff_regions_by_sweep.get(sweep_idx, [])
        if sniff_regions:
            if sniffing_shade:
                for start_t, end_t in sniff_regions:
                    add_shaded_region(start_t, end_t, (128, 0, 128, 80))  # Purple shade
            else:
                for start_t, end_t in sniff_regions:
                    add_line_indicator(start_t - t0, end_t - t0, (128, 0, 128), 0.99)  # Purple line at top

        # Compute apnea, outlier, and failure regions dynamically (same as matplotlib version)
        try:
            br = getattr(st, "breath_by_sweep", {}).get(sweep_idx, None)
            t = st.t
            if br and t is not None and len(t) > 100:
                # Get breath data
                pks = getattr(st, "peaks_by_sweep", {}).get(sweep_idx, None)
                on_idx = br.get("onsets", [])
                off_idx = br.get("offsets", [])
                ex_idx = br.get("expmins", [])
                exoff_idx = br.get("expoffs", [])

                # Get processed Y data
                y = self.window._get_processed_for(st.analyze_chan, sweep_idx) if hasattr(self, 'window') else None
                if y is None:
                    y = st.sweeps[st.analyze_chan][:, sweep_idx]

                # Get apnea threshold
                apnea_thresh = self.window._parse_float(self.window.ApneaThresh) or 0.5 if hasattr(self, 'window') else 0.5

                # Compute apnea mask and convert to regions
                apnea_mask = metrics.detect_apneas(
                    t, y, st.sr_hz, pks, on_idx, off_idx, ex_idx, exoff_idx,
                    min_apnea_duration_sec=apnea_thresh
                )
                apnea_regions = self._extract_regions_from_mask(t, apnea_mask)

                if apnea_regions:
                    if apnea_shade:
                        for start_t, end_t in apnea_regions:
                            add_shaded_region(start_t, end_t, (255, 0, 0, 60))  # Red shade
                    else:
                        for start_t, end_t in apnea_regions:
                            add_line_indicator(start_t - t0, end_t - t0, (255, 0, 0), 0.02)  # Red line at bottom

                # Compute outlier and failure masks
                outlier_mask, failure_mask = self._compute_outlier_masks(
                    sweep_idx, t, y, pks, on_idx, off_idx, ex_idx, exoff_idx
                )

                # Draw outlier regions (orange)
                if outlier_mask is not None:
                    outlier_regions = self._extract_regions_from_mask(t, outlier_mask)
                    # Ensure minimum width for visibility
                    adjusted_regions = []
                    for start_t, end_t in outlier_regions:
                        width = end_t - start_t
                        if width < 0.1:
                            mid = (start_t + end_t) / 2
                            start_t = mid - 0.05
                            end_t = mid + 0.05
                        adjusted_regions.append((start_t, end_t))

                    if adjusted_regions:
                        if outliers_shade:
                            for start_t, end_t in adjusted_regions:
                                add_shaded_region(start_t, end_t, (255, 165, 0, 80))  # Orange shade
                        else:
                            for start_t, end_t in adjusted_regions:
                                add_line_indicator(start_t - t0, end_t - t0, (255, 165, 0), 0.04)  # Orange line near bottom

                # Draw failure regions (red shade) - always shaded
                if failure_mask is not None:
                    failure_regions = self._extract_regions_from_mask(t, failure_mask)
                    # Ensure minimum width
                    for start_t, end_t in failure_regions:
                        width = end_t - start_t
                        if width < 0.1:
                            mid = (start_t + end_t) / 2
                            start_t = mid - 0.05
                            end_t = mid + 0.05
                        add_shaded_region(start_t, end_t, (255, 0, 0, 76))  # Red shade

        except Exception as e:
            print(f"[PyQtGraph] Warning: Could not compute region overlays: {e}")

    def _extract_regions_from_mask(self, t, mask):
        """Extract continuous regions where mask == 1 or mask == True.

        This is a copy of the matplotlib version's _extract_regions for use in PyQtGraph.
        """
        regions = []

        if mask is None or len(mask) == 0 or len(t) == 0:
            return regions

        # Ensure mask is boolean
        mask = np.asarray(mask, dtype=bool)

        # Check if there are any True values
        if not np.any(mask):
            return regions

        # Find transitions
        diff_mask = np.diff(mask.astype(int))
        starts = np.where(diff_mask == 1)[0] + 1  # Start of True regions
        ends = np.where(diff_mask == -1)[0] + 1   # End of True regions

        # Handle edge cases
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])

        # Convert indices to time values with bounds checking
        for start_idx, end_idx in zip(starts, ends):
            start_idx = int(start_idx)
            end_idx = int(end_idx)

            if start_idx < 0 or start_idx >= len(t):
                continue
            if end_idx < 0 or end_idx > len(t):
                end_idx = len(t)

            # Get time values (use end_idx-1 since end_idx is exclusive)
            start_t = float(t[start_idx])
            end_t = float(t[min(end_idx - 1, len(t) - 1)])

            if end_t > start_t:  # Ensure valid region
                regions.append((start_t, end_t))

        return regions

    def _draw_omitted_regions_pyqtgraph(self, plot, sweep_idx, t_plot, t0):
        """Draw omitted region overlays on PyQtGraph plot."""
        import pyqtgraph as pg
        st = self.state

        # Check if full sweep is omitted (like matplotlib version)
        if sweep_idx in st.omitted_sweeps:
            # Draw gray overlay over entire data range (not view range, which may not be set yet)
            # Use t_plot which contains the actual time data for this sweep
            if t_plot is not None and len(t_plot) > 0:
                t_min = float(t_plot[0])
                t_max = float(t_plot[-1])
            else:
                # Fallback to view range if no time data
                x_range = plot.viewRange()[0]
                t_min, t_max = x_range[0], x_range[1]

            region = pg.LinearRegionItem(
                values=[t_min, t_max],
                brush=pg.mkBrush(128, 128, 128, 100),  # Gray with alpha
                pen=pg.mkPen('darkgray', width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
                movable=False
            )
            region.setZValue(100)
            plot.addItem(region)

            # Add "omitted" label in center
            t_center = (t_min + t_max) / 2.0
            y_range = plot.viewRange()[1]
            # If y_range is default [0, 1], estimate from data
            if y_range[0] == 0 and y_range[1] == 1:
                y_center = 0.0  # Default to zero line
            else:
                y_center = (y_range[0] + y_range[1]) / 2.0

            text = pg.TextItem("omitted", color='darkgray', anchor=(0.5, 0.5))
            text.setPos(t_center, y_center)
            text.setZValue(101)
            plot.addItem(text)
            return

        # Otherwise draw smaller omitted regions
        omitted_ranges = st.omitted_ranges.get(sweep_idx, [])
        if not omitted_ranges:
            return

        sr_hz = st.sr_hz if st.sr_hz else 1000.0

        for (i_start, i_end) in omitted_ranges:
            # Convert sample indices to plot time
            t_start = i_start / sr_hz - t0
            t_end = i_end / sr_hz - t0

            if t_end > t_start:
                region = pg.LinearRegionItem(
                    values=[t_start, t_end],
                    brush=pg.mkBrush(80, 80, 80, 100),  # Dark gray with alpha
                    pen=pg.mkPen('darkgray', width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
                    movable=False
                )
                region.setZValue(100)  # Above trace but below markers
                plot.addItem(region)

                # Add "omitted" label in center of region
                t_center = (t_start + t_end) / 2.0
                y_range = plot.viewRange()[1]
                y_center = (y_range[0] + y_range[1]) / 2.0

                text = pg.TextItem("omitted", color='darkgray', anchor=(0.5, 0.5))
                text.setPos(t_center, y_center)
                text.setZValue(101)
                plot.addItem(text)

    def _cleanup_y2_elements_pyqtgraph(self):
        """Clean up Y2 axis elements before redrawing or clearing layout."""
        if not hasattr(self.plot_host, '_y2_viewbox'):
            return

        # Disconnect sigResized signal if connected
        if hasattr(self.plot_host, '_y2_update_connection') and self.plot_host._y2_update_connection is not None:
            try:
                main_plot = self.plot_host._get_main_plot()
                if main_plot is not None:
                    main_plot.vb.sigResized.disconnect(self.plot_host._y2_update_connection)
            except:
                pass
            self.plot_host._y2_update_connection = None

        # Remove Y2 line from viewbox
        if self.plot_host._y2_line is not None:
            try:
                if self.plot_host._y2_viewbox is not None:
                    self.plot_host._y2_viewbox.removeItem(self.plot_host._y2_line)
            except:
                pass
            self.plot_host._y2_line = None

        # Remove Y2 viewbox from scene
        if self.plot_host._y2_viewbox is not None:
            try:
                main_plot = self.plot_host._get_main_plot()
                if main_plot is not None and main_plot.scene() is not None:
                    main_plot.scene().removeItem(self.plot_host._y2_viewbox)
            except:
                pass
            self.plot_host._y2_viewbox = None

        # Remove Y2 axis from layout
        if self.plot_host._y2_axis is not None:
            try:
                main_plot = self.plot_host._get_main_plot()
                if main_plot is not None and main_plot.layout is not None:
                    main_plot.layout.removeItem(self.plot_host._y2_axis)
            except:
                pass
            self.plot_host._y2_axis = None

    def _draw_y2_metric_pyqtgraph(self, sweep_idx, t_plot):
        """Draw Y2 metric on PyQtGraph plot with independent Y axis."""
        import pyqtgraph as pg
        st = self.state
        key = getattr(st, "y2_metric_key", None)

        # Get the main plot
        main_plot = self.plot_host._get_main_plot()
        if main_plot is None:
            return

        # Clear existing Y2 elements (use centralized cleanup)
        self._cleanup_y2_elements_pyqtgraph()

        if not key:
            return

        # Get metric data for current sweep
        arr = st.y2_values_by_sweep.get(sweep_idx, None)
        if arr is None or len(arr) != len(t_plot):
            return

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

        # Create a new ViewBox for Y2 data
        y2_viewbox = pg.ViewBox()
        self.plot_host._y2_viewbox = y2_viewbox

        # Create Y2 axis on the right
        is_dark = self.plot_host._current_theme == 'dark'
        text_color = '#d4d4d4' if is_dark else '#000000'

        y2_axis = pg.AxisItem('right')
        y2_axis.setLabel(label, color=color)
        y2_axis.setTextPen(color)
        y2_axis.setPen(color)
        self.plot_host._y2_axis = y2_axis

        # Add the Y2 axis and viewbox to the plot layout
        main_plot.layout.addItem(y2_axis, 2, 2)  # row 2, col 2 (right side)
        main_plot.scene().addItem(y2_viewbox)
        y2_axis.linkToView(y2_viewbox)

        # Link X axes
        y2_viewbox.setXLink(main_plot.vb)

        # Plot the Y2 data
        qcolor = pg.mkColor(color)
        pen = pg.mkPen(qcolor, width=1.5)
        y2_line = pg.PlotDataItem(t_plot, arr, pen=pen)
        y2_viewbox.addItem(y2_line)
        self.plot_host._y2_line = y2_line

        # Update viewbox geometry when main plot changes
        def update_views():
            try:
                y2_viewbox.setGeometry(main_plot.vb.sceneBoundingRect())
                y2_viewbox.linkedViewChanged(main_plot.vb, y2_viewbox.XAxis)
            except RuntimeError:
                # ViewBox may have been deleted
                pass

        main_plot.vb.sigResized.connect(update_views)
        self.plot_host._y2_update_connection = update_views  # Store reference for cleanup
        update_views()

        # Auto-range the Y2 axis
        y2_viewbox.autoRange()

    def _build_channel_configs_from_manager(self) -> List[ChannelPanelConfig]:
        """Build ChannelPanelConfig list from channel manager.

        Returns:
            List of ChannelPanelConfig for visible channels
        """
        configs = []
        st = self.state

        if not hasattr(self.window, 'channel_manager'):
            return configs

        channels = self.window.channel_manager.get_channels()

        # Collect visible channels sorted by order
        visible_list = []
        for name, cm_config in channels.items():
            if cm_config.visible:
                visible_list.append((name, cm_config))

        # Sort by order attribute
        visible_list.sort(key=lambda x: x[1].order)

        # Find primary Pleth channel (first Pleth or the analyze_chan)
        primary_pleth_name = st.analyze_chan

        for name, cm_config in visible_list:
            is_primary = (cm_config.channel_type == "Pleth" and name == primary_pleth_name)
            config = ChannelPanelConfig.from_channel_manager(name, cm_config, is_primary)
            configs.append(config)

        # If no primary was found but there are Pleth channels, make first one primary
        pleth_configs = [c for c in configs if c.channel_type == "Pleth"]
        if pleth_configs and not any(c.is_primary_pleth for c in configs):
            pleth_configs[0].is_primary_pleth = True
            pleth_configs[0].show_overlays = True

        return configs

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
                ylabel=st.analyze_chan or "Signal",
                state=self.state
            )

        # Clear any existing region overlays (will be recomputed if needed)
        self.plot_host.clear_region_overlays()

        # Draw peak markers
        self._draw_peak_markers(s, t_plot, y)

        # Draw sigh markers (stars)
        self._draw_sigh_markers(s, t_plot, y)

        # Draw breath event markers (onsets, offsets, expiratory mins/offs)
        self._draw_breath_markers(s, t_plot, y)

        # Sniff region overlays are now handled by _draw_region_overlays() via unified update_region_overlays()
        # self.window.editing_modes.update_sniff_artists(t_plot, s)  # REMOVED: Causes duplicate shading

        # Draw Y2 metric if selected
        self._draw_y2_metric(s, t, t_plot)

        # Draw omitted region overlays
        self._draw_omitted_regions(s, t_plot)

        # Refresh threshold lines
        self.refresh_threshold_lines()

        # Set y-limits based on non-omitted data only (exclude omitted regions from autoscaling)
        # IMPORTANT: This must happen BEFORE drawing region overlays so the y-position is consistent
        self._set_ylim_excluding_omitted(s, y, t_plot)

        # Draw automatic region overlays (eupnea, apnea, outliers)
        # AFTER y-limits are finalized so the line position is consistent with lightweight refresh
        self._draw_region_overlays(s, t, y, t_plot, t0)

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

            # Remove NaN values before calculating percentiles
            y_valid = y_values[~np.isnan(y_values)]

            if len(y_valid) > 0:
                # Use percentile or full range based on user preference
                if st.use_percentile_autoscale:
                    # Percentile mode: Use percentiles to avoid artifacts and huge outliers
                    y_min = np.percentile(y_valid, 1)   # 1st percentile
                    y_max = np.percentile(y_valid, 99)  # 99th percentile
                    y_range = y_max - y_min

                    # Handle edge case where all values are the same
                    if y_range == 0 or np.isnan(y_range):
                        y_range = abs(y_min) if y_min != 0 else 1.0

                    # Add configurable padding
                    padding = st.autoscale_padding * y_range
                    mode_str = f"percentile, {st.autoscale_padding*100:.0f}% padding"
                else:
                    # Full range mode: Use absolute min/max
                    y_min = np.min(y_valid)
                    y_max = np.max(y_valid)
                    y_range = y_max - y_min

                    # Handle edge case where all values are the same
                    if y_range == 0 or np.isnan(y_range):
                        y_range = abs(y_min) if y_min != 0 else 1.0

                    # Add small padding
                    padding = 0.05 * y_range
                    mode_str = "full range"

                # Final validation
                y_lower = y_min - padding
                y_upper = y_max + padding

                # Account for sigh marker vertical offset if present
                if len(sigh_idx) > 0:
                    offset_frac = float(getattr(self.window, "_sigh_offset_frac", 0.07))
                    sigh_offset = offset_frac * y_range
                    y_upper += sigh_offset  # Extend y_max to include sigh markers

                if np.isfinite(y_lower) and np.isfinite(y_upper):
                    ax.set_ylim(y_lower, y_upper)
                    print(f"[Plot Manager] Auto-scaled Y-axis (excluding omitted, {mode_str}): {y_lower:.3f} to {y_upper:.3f}")
                else:
                    print(f"[Plot Manager] Warning: Invalid Y-axis limits (NaN/Inf), using default")
            else:
                print(f"[Plot Manager] Warning: No valid (non-NaN) data for Y-axis scaling")

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

        # Apply theme to both subplots
        self.plot_host.theme_manager.apply_theme(ax_pleth, self.plot_host.fig, self.plot_host._current_theme)
        self.plot_host.theme_manager.apply_theme(ax_event, self.plot_host.fig, self.plot_host._current_theme)

        # Get trace color from the current theme explicitly
        trace_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['trace_color']
        text_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['text_color']

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

        # Plot pleth trace on top subplot (use theme color)
        ax_pleth.plot(t_plot, y_pleth, linewidth=0.9, color=trace_color)
        ax_pleth.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Add stim spans to top plot
        for (t0_span, t1_span) in (spans_plot or []):
            if t1_span > t0_span:
                ax_pleth.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

        # Set title and labels for top plot
        if title:
            ax_pleth.set_title(title, color=text_color)
        ax_pleth.set_ylabel(st.analyze_chan or "Signal")
        ax_pleth.grid(False)

        # Apply Y-axis autoscaling based on user preference (matching single-panel behavior)
        if len(y_pleth) > 0:
            # Remove NaN values before calculating percentiles
            y_valid = y_pleth[~np.isnan(y_pleth)]

            if len(y_valid) > 0:
                if st.use_percentile_autoscale:
                    # Percentile mode
                    y_min = np.percentile(y_valid, 1)   # 1st percentile
                    y_max = np.percentile(y_valid, 99)  # 99th percentile
                    y_range = y_max - y_min

                    # Handle edge case where all values are the same
                    if y_range == 0 or np.isnan(y_range):
                        y_range = abs(y_min) if y_min != 0 else 1.0

                    padding = st.autoscale_padding * y_range
                    mode_str = f"percentile, {st.autoscale_padding*100:.0f}% padding"
                else:
                    # Full range mode
                    y_min = np.min(y_valid)
                    y_max = np.max(y_valid)
                    y_range = y_max - y_min

                    # Handle edge case where all values are the same
                    if y_range == 0 or np.isnan(y_range):
                        y_range = abs(y_min) if y_min != 0 else 1.0

                    padding = 0.05 * y_range
                    mode_str = "full range"

                # Final validation
                y_lower = y_min - padding
                y_upper = y_max + padding

                if np.isfinite(y_lower) and np.isfinite(y_upper):
                    ax_pleth.set_ylim(y_lower, y_upper)
                    print(f"[Dual Plot] Auto-scaled Y-axis ({mode_str}): {y_lower:.3f} to {y_upper:.3f}")
                else:
                    print(f"[Dual Plot] Warning: Invalid Y-axis limits (NaN/Inf), using default")
            else:
                print(f"[Dual Plot] Warning: No valid (non-NaN) data for Y-axis scaling")

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

        # Use tight layout with minimal padding to maximize plot area
        self.plot_host.fig.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
        # Further adjust to use full width
        self.plot_host.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)
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

        # Plot continuous trace (use theme color explicitly from current theme)
        trace_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['trace_color']
        ax.plot(t_plot, event_plot, color=trace_color, linewidth=1)
        ax.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

        # Add threshold line if event detection dialog exists and has a threshold set
        if hasattr(self.window, '_event_detection_dialog') and self.window._event_detection_dialog is not None:
            try:
                threshold = self.window._event_detection_dialog.threshold_spin.value()
                ax.axhline(threshold, linestyle=':', linewidth=1.5, color='red', alpha=0.7, zorder=5)
            except:
                pass

        # Add stim spans to event plot
        for (t0_span, t1_span) in (spans_plot or []):
            if t1_span > t0_span:
                ax.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

        # Set labels
        ax.set_ylabel(f'{st.event_channel}', fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=11)
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
                self.plot_host.update_peaks(t_normal, y_normal, size=18)  # Reduced from 24 (25% smaller)
            else:
                self.plot_host.clear_peaks()

            # Draw omitted peaks in gray
            if len(omitted_pks):
                t_omitted = t_plot[omitted_pks]
                y_omitted = y[omitted_pks]
                ax = self.plot_host.ax_main
                if ax is not None:
                    ax.scatter(t_omitted, y_omitted, s=18, c='gray', alpha=0.5, marker='o', zorder=5, edgecolors='none', linewidths=0)
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

            # Theme-aware sigh markers
            # Dark mode: gold star with no outline (cleaner look)
            # Light mode: gold star with black outline (better visibility)
            is_dark_mode = self.plot_host._current_theme == 'dark'
            if is_dark_mode:
                sigh_color = "#FFD700"  # Gold
                sigh_edge = "#FFD700"   # Same as fill (no visible outline)
            else:
                sigh_color = "#FFD700"  # Gold
                sigh_edge = "black"     # Black outline for light backgrounds

            self.plot_host.update_sighs(
                t_sigh, y_sigh,
                size=82,  # Reduced from 110 (25% smaller)
                color=sigh_color,
                edge=sigh_edge,
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
                size=27  # Reduced from 36 (25% smaller)
            )

            # Draw omitted markers in gray
            ax = self.plot_host.ax_main
            if ax is not None:
                if len(on_omit):
                    ax.scatter(t_plot[on_omit], y[on_omit], s=27, c='gray', alpha=0.5, marker='^', zorder=4, edgecolors='none', linewidths=0)
                if len(off_omit):
                    ax.scatter(t_plot[off_omit], y[off_omit], s=27, c='gray', alpha=0.5, marker='v', zorder=4, edgecolors='none', linewidths=0)
                if len(ex_omit):
                    ax.scatter(t_plot[ex_omit], y[ex_omit], s=27, c='gray', alpha=0.5, marker='s', zorder=4, edgecolors='none', linewidths=0)
                if len(exoff_omit):
                    ax.scatter(t_plot[exoff_omit], y[exoff_omit], s=27, c='gray', alpha=0.5, marker='D', zorder=4, edgecolors='none', linewidths=0)
        else:
            self.plot_host.clear_breath_markers()

    def _draw_y2_metric(self, sweep_idx, t, t_plot):
        """Draw Y2 axis metric if selected and available (matplotlib version)."""
        st = self.state
        is_pyqtgraph = getattr(st, 'plotting_backend', 'matplotlib') == 'pyqtgraph'

        # Skip if using PyQtGraph (handled separately by _draw_y2_metric_pyqtgraph)
        if is_pyqtgraph:
            return

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

                # Force Y-axis range for onset_height_ratio to start at 0
                if key == "onset_height_ratio":
                    ax_y2 = getattr(self.plot_host, 'ax_y2', None)
                    if ax_y2 is not None:
                        import numpy as np
                        valid_data = arr[~np.isnan(arr)]
                        if len(valid_data) > 0:
                            max_val = np.max(valid_data)
                            ax_y2.set_ylim(0, max_val * 1.1)  # 0 to max + 10% padding

                self.plot_host.fig.tight_layout()
            else:
                self.plot_host.clear_y2()
                self.plot_host.fig.tight_layout()
        else:
            self.plot_host.clear_y2()
            self.plot_host.fig.tight_layout()

    def _draw_region_overlays(self, sweep_idx, t, y, t_plot, t0=0.0):
        """Draw automatic region overlays (eupnea, apnea, outliers).

        Args:
            sweep_idx: Current sweep index
            t: Time array (absolute time)
            y: Signal array
            t_plot: Time array for plotting (normalized if stimulus active)
            t0: Time offset for normalization (0.0 if no stimulus)
        """
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

                # Get sniff and eupnea regions from state (stored in absolute time)
                sniff_regions_raw = self.state.sniff_regions_by_sweep.get(sweep_idx, [])
                eupnea_regions_raw = getattr(self.state, 'eupnea_regions_by_sweep', {}).get(sweep_idx, [])

                # Normalize region times to match t_plot (subtract t0 if stimulus active)
                sniff_regions = [(start - t0, end - t0) for start, end in sniff_regions_raw]
                eupnea_regions = [(start - t0, end - t0) for start, end in eupnea_regions_raw]

                # For backward compatibility, also support mask-based eupnea from frequency mode
                eupnea_mask = None
                if self.window.eupnea_detection_mode != "gmm":
                    # Frequency-based (legacy): Use threshold method to create mask
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

                # Apply the overlays (including sniff_regions and eupnea_regions for unified display control)
                self.plot_host.update_region_overlays(t_plot, eupnea_mask, apnea_mask, outlier_mask, failure_mask, sniff_regions, eupnea_regions, state=self.state)
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
        """Check if a peak index falls within an omitted region for the given sweep.

        Returns True if:
        1. The entire sweep is omitted (sweep_idx in omitted_sweeps), OR
        2. The peak falls within a specific omitted range
        """
        st = self.state

        # Check if entire sweep is omitted
        if sweep_idx in st.omitted_sweeps:
            return True

        # Check if peak is in a specific omitted range
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

    def _get_channel_manager_config(self):
        """Get visible channels and their types from the channel manager.

        Returns:
            dict: {
                'pleth_channels': [(name, config), ...],  # Channels marked as Pleth
                'opto_stim_channels': [(name, config), ...],  # All Opto Stim channels (for blue spans)
                'visible_opto_channels': [(name, config), ...],  # Visible Opto Stim channels (for panels)
                'event_channels': [(name, config), ...],  # Channels marked as Event
                'raw_channels': [(name, config), ...],  # Channels marked as Raw Signal
                'visible_channels': [(name, config), ...],  # All visible channels in order
            }
        """
        result = {
            'pleth_channels': [],
            'opto_stim_channels': [],  # All opto channels (for blue spans even if hidden)
            'visible_opto_channels': [],  # Visible opto channels (for panels)
            'event_channels': [],  # Channels marked as Event (for bout annotations)
            'raw_channels': [],
            'visible_channels': []
        }

        # Check if channel manager exists
        if not hasattr(self.window, 'channel_manager'):
            return result

        channels = self.window.channel_manager.get_channels()

        # Collect all opto stim channels (even hidden ones) for blue spans
        for name, config in channels.items():
            if config.channel_type == "Opto Stim":
                result['opto_stim_channels'].append((name, config))

        # Collect visible channels sorted by order
        visible_list = []
        for name, config in channels.items():
            if config.visible:
                visible_list.append((name, config))

        # Sort by order attribute to preserve channel manager order
        visible_list.sort(key=lambda x: x[1].order)

        for name, config in visible_list:
            result['visible_channels'].append((name, config))

            if config.channel_type == "Pleth":
                result['pleth_channels'].append((name, config))
            elif config.channel_type == "Opto Stim":
                result['visible_opto_channels'].append((name, config))
            elif config.channel_type == "Event":
                result['event_channels'].append((name, config))
            else:  # "Raw Signal"
                result['raw_channels'].append((name, config))

        return result

    def _draw_multi_channel_plot(self, channel_config):
        """Draw multi-channel plot based on channel manager configuration.

        Args:
            channel_config: Dict from _get_channel_manager_config()
        """
        from matplotlib.gridspec import GridSpec
        st = self.state

        # Use visible_channels which preserves channel manager order
        # This includes all visible channels: Pleth, Raw Signal, and Opto Stim
        plot_channels = channel_config['visible_channels']

        # Get Pleth channels for overlay drawing
        pleth_channels = channel_config['pleth_channels']

        # Get all opto channels (even hidden) for blue span extraction
        opto_channels = channel_config['opto_stim_channels']

        if not plot_channels:
            return

        n_panels = len(plot_channels)

        # Calculate height ratios - Pleth gets more space, others get less
        height_ratios = []
        for name, config in plot_channels:
            if config.channel_type == "Pleth":
                height_ratios.append(0.6)  # Pleth gets 60%
            elif config.channel_type == "Opto Stim":
                height_ratios.append(0.15)  # Opto Stim panels are small (TTL signal)
            elif config.channel_type == "Event":
                height_ratios.append(0.25)  # Event channels get 25% (like dual subplot)
            else:
                height_ratios.append(0.25)  # Raw channels get 25% each

        # Normalize ratios
        total = sum(height_ratios)
        height_ratios = [h/total for h in height_ratios]

        # Get current sweep info
        s = max(0, min(st.sweep_idx, next(iter(st.sweeps.values())).shape[1] - 1))

        # Get time normalization (stim offset)
        spans = st.stim_spans_by_sweep.get(s, []) if st.stim_chan else []
        if st.stim_chan and spans:
            t0 = spans[0][0]
            t_plot = st.t - t0
            spans_plot = [(a - t0, b - t0) for (a, b) in spans]
        else:
            t0 = 0.0
            t_plot = st.t
            spans_plot = spans

        # Check for Opto Stim channel - extract stimulus spans from it
        opto_spans_plot = []
        if opto_channels:
            opto_name, opto_config = opto_channels[0]  # Use first Opto Stim channel
            if opto_name in st.sweeps:
                opto_data = st.sweeps[opto_name][:, s]
                opto_spans_plot = self._extract_stim_spans_from_channel(st.t, opto_data, t0)

        # Save previous view if needed
        prev_xlim = self.plot_host._last_single["xlim"] if self.plot_host._preserve_x else None
        prev_ylim = self.plot_host._last_single["ylim"] if self.plot_host._preserve_y else None

        # Clear figure and create GridSpec layout
        self.plot_host.fig.clear()
        gs = GridSpec(n_panels, 1, height_ratios=height_ratios, hspace=0.05, figure=self.plot_host.fig)

        # Create subplots
        axes = []
        ax_main = None

        for i, (ch_name, config) in enumerate(plot_channels):
            if i == 0:
                ax = self.plot_host.fig.add_subplot(gs[i])
            else:
                ax = self.plot_host.fig.add_subplot(gs[i], sharex=axes[0])
            axes.append(ax)

            # First Pleth channel becomes ax_main for overlays
            if config.channel_type == "Pleth" and ax_main is None:
                ax_main = ax
                self.plot_host.ax_main = ax

            # Hide x-axis labels except on bottom
            if i < n_panels - 1:
                ax.tick_params(labelbottom=False)

            # Apply theme
            self.plot_host.theme_manager.apply_theme(ax, self.plot_host.fig, self.plot_host._current_theme)

        # Get theme colors
        trace_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['trace_color']
        text_color = self.plot_host.theme_manager.themes[self.plot_host._current_theme]['text_color']

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
        self.plot_host.ax_event = None

        # Track Event channel axis for bout annotations
        ax_event = None
        event_channel_name = None

        # Plot each channel
        for i, ((ch_name, config), ax) in enumerate(zip(plot_channels, axes)):
            # Get channel data
            if ch_name not in st.sweeps:
                continue

            y_data = st.sweeps[ch_name][:, s]

            # Apply filtering only to Pleth channels
            if config.channel_type == "Pleth":
                # Use filtered data via _current_trace if this is the analyze channel
                if ch_name == st.analyze_chan:
                    _, y_filtered = self.window._current_trace()
                    y_data = y_filtered

            # Track Event channel axis
            if config.channel_type == "Event":
                ax_event = ax
                event_channel_name = ch_name
                self.plot_host.ax_event = ax

            # Plot trace
            ax.plot(t_plot, y_data, linewidth=0.9, color=trace_color)
            ax.axhline(0.0, linestyle="--", linewidth=0.8, color="#666666", alpha=0.9, zorder=0)

            # Add stim spans (blue background) to Pleth channels
            if config.channel_type == "Pleth":
                # Prefer Opto Stim channel spans from channel manager over old stim spans
                # This avoids duplication when both are available
                if opto_spans_plot:
                    all_spans = opto_spans_plot
                else:
                    all_spans = list(spans_plot)
                for (t0_span, t1_span) in all_spans:
                    if t1_span > t0_span:
                        ax.axvspan(t0_span, t1_span, color="#2E5090", alpha=0.25)

            # Set ylabel
            ax.set_ylabel(ch_name, fontsize=10)
            ax.grid(False)

            # Apply Y-axis autoscaling
            self._autoscale_axis(ax, y_data)

        # Sync Event channel to state (for event detection dialog)
        if event_channel_name and event_channel_name != st.event_channel:
            st.event_channel = event_channel_name

        # Set title on first axis
        title = self._build_plot_title(s)
        if axes:
            axes[0].set_title(title, color=text_color)

        # Set xlabel on bottom axis
        if axes:
            axes[-1].set_xlabel('Time (s)', fontsize=10)

        # Now draw overlays on the Pleth channel (ax_main)
        if ax_main is not None and pleth_channels:
            pleth_name, pleth_config = pleth_channels[0]

            # Get filtered Pleth data
            if pleth_name == st.analyze_chan:
                _, y_pleth = self.window._current_trace()
            else:
                y_pleth = st.sweeps[pleth_name][:, s]

            # Clear region overlays (will be redrawn)
            self.plot_host.clear_region_overlays()

            # Draw peak markers
            self._draw_peak_markers(s, t_plot, y_pleth)

            # Draw sigh markers
            self._draw_sigh_markers(s, t_plot, y_pleth)

            # Draw breath markers
            self._draw_breath_markers(s, t_plot, y_pleth)

            # Draw Y2 metric if selected
            self._draw_y2_metric(s, st.t, t_plot)

            # Draw omitted region overlays
            self._draw_omitted_regions(s, t_plot)

            # Refresh threshold lines
            self.refresh_threshold_lines()

            # Set y-limits excluding omitted
            self._set_ylim_excluding_omitted(s, y_pleth, t_plot)

            # Draw region overlays (eupnea, apnea, etc.)
            self._draw_region_overlays(s, st.t, y_pleth, t_plot, t0)

        # Draw bout annotations on Event channel (if present)
        if ax_event is not None and ax_main is not None:
            if s in st.bout_annotations and st.bout_annotations[s]:
                self._plot_bout_annotations(ax_main, ax_event, st.bout_annotations[s], t0)

        # Restore preserved view or set default xlim to full data range
        if axes:
            if prev_xlim is not None:
                axes[0].set_xlim(prev_xlim)
            else:
                # Set default xlim to full time range
                axes[0].set_xlim(t_plot[0], t_plot[-1])
        if prev_ylim is not None and ax_main is not None:
            ax_main.set_ylim(prev_ylim)

        # Set up limit listeners
        self.plot_host._attach_limit_listeners(axes, mode="single")
        self.plot_host._store_from_axes(mode="single")

        # Layout adjustments
        self.plot_host.fig.tight_layout(pad=0.5, h_pad=0.1, w_pad=0.1)
        self.plot_host.fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08)
        self.plot_host.canvas.draw_idle()

    def _extract_stim_spans_from_channel(self, t, data, t0=0.0, threshold=0.5):
        """Extract stimulus spans from an Opto Stim channel.

        Args:
            t: Time array
            data: Channel data array
            t0: Time offset for normalization
            threshold: Threshold for detecting stimulus on/off (default 0.5)

        Returns:
            List of (start, end) tuples in normalized time
        """
        # Simple threshold-based detection
        above = data > threshold

        # Find transitions
        spans = []
        in_stim = False
        start_time = None

        for i, val in enumerate(above):
            if val and not in_stim:
                # Start of stim
                in_stim = True
                start_time = t[i] - t0
            elif not val and in_stim:
                # End of stim
                in_stim = False
                end_time = t[i] - t0
                spans.append((start_time, end_time))

        # Handle case where stim continues to end
        if in_stim and start_time is not None:
            spans.append((start_time, t[-1] - t0))

        return spans

    def _autoscale_axis(self, ax, y_data):
        """Apply Y-axis autoscaling to an axis."""
        st = self.state

        if len(y_data) == 0:
            return

        # Remove NaN values
        y_valid = y_data[~np.isnan(y_data)]

        if len(y_valid) == 0:
            return

        if st.use_percentile_autoscale:
            y_min = np.percentile(y_valid, 1)
            y_max = np.percentile(y_valid, 99)
            y_range = y_max - y_min

            if y_range == 0 or np.isnan(y_range):
                y_range = abs(y_min) if y_min != 0 else 1.0

            padding = st.autoscale_padding * y_range
        else:
            y_min = np.min(y_valid)
            y_max = np.max(y_valid)
            y_range = y_max - y_min

            if y_range == 0 or np.isnan(y_range):
                y_range = abs(y_min) if y_min != 0 else 1.0

            padding = 0.05 * y_range

        y_lower = y_min - padding
        y_upper = y_max + padding

        if np.isfinite(y_lower) and np.isfinite(y_upper):
            ax.set_ylim(y_lower, y_upper)

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
