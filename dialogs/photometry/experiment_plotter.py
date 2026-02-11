"""
Experiment Plotter - PyQtGraph plotting for photometry experiments.

Handles all visualization logic for fiber photometry data including:
- Raw signal display (Isosbestic + GCaMP)
- Fitted isosbestic overlay
- dF/F computation results with detrending curves
- AI channel display
- Interactive fit regions

This module is designed to be reusable and independent of the dialog UI logic.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget


# Color palette for photometry plots
COLORS = {
    'isosbestic': '#9370DB',      # Medium purple
    'gcamp': '#00cc00',           # Bright green
    'fitted_iso': '#ff6b6b',      # Red (for fitted curves)
    'dff_raw': '#00cc00',         # Bright green (same as GCaMP)
    'detrend': '#ff6b6b',         # Red (for detrend curve)
    'final_dff': '#00cc00',       # Bright green (same as GCaMP)
    'ai_channels': [              # Multiple AI channel colors (matches All Channels tab)
        '#ff9900',                # Orange
        '#00cccc',                # Cyan
        '#ff66ff',                # Pink/Magenta
        '#ffff00',                # Yellow
        '#ff6666',                # Light red
    ],
    'axis': '#3e3e42',            # Dark gray
    'text': '#cccccc',            # Light gray
    'fit_region_brush': (100, 100, 255, 50),  # Semi-transparent blue
    'fit_region_pen': '#6464ff',  # Blue
}


class ExperimentPlotter:
    """
    Handles all PyQtGraph plotting for photometry experiments.

    This class manages the creation and styling of plots for:
    - Raw photometry signals (Iso + GCaMP on same axis)
    - Fitted isosbestic overlay
    - dF/F with detrending curve
    - Final dF/F
    - AI analog channels

    Usage:
        plotter = ExperimentPlotter()
        plots = plotter.plot_experiment(
            exp_idx=0,
            fiber_data=fiber_data,
            dff_results=results,
            ai_data=ai_df,
            ai_time=timestamps,
            show_intermediates=True,
            graphics_layout=layout
        )
    """

    def __init__(self):
        """Initialize the plotter."""
        self._fit_regions: Dict[int, List] = {}
        self._updating_regions = False

    def plot_experiment(
        self,
        exp_idx: int,
        fiber_col: str,
        fiber_data: Dict,
        dff_results: Optional[Dict],
        ai_data,  # DataFrame or None
        ai_time: Optional[np.ndarray],
        ai_channels: List[Tuple[int, str]],  # List of (col_idx, channel_name)
        show_intermediates: bool,
        graphics_layout: GraphicsLayoutWidget,
        fit_region_enabled: bool = False,
        fit_start: float = 0.0,
        fit_end: float = 10.0,
        on_region_changed: Optional[Callable] = None,
    ) -> Tuple[List, int]:
        """
        Plot all panels for a single fiber in an experiment.

        Args:
            exp_idx: Experiment index (for tracking fit regions)
            fiber_col: Fiber column name (e.g., "G1")
            fiber_data: Dict with keys 'iso_time', 'iso', 'gcamp_time', 'gcamp'
            dff_results: Dict with 'time', 'dff', 'intermediates' from compute_dff_full
            ai_data: DataFrame with AI channels (or None)
            ai_time: AI timestamps array in seconds (or None)
            ai_channels: List of (column_index, channel_name) tuples for this experiment
            show_intermediates: Whether to show intermediate processing plots
            graphics_layout: PyQtGraph GraphicsLayoutWidget to add plots to
            fit_region_enabled: Whether to show interactive fit region
            fit_start: Fit region start time in minutes
            fit_end: Fit region end time in minutes
            on_region_changed: Callback when fit region is changed

        Returns:
            Tuple of (list of plot items, next available row number)
        """
        plot_items = []
        ai_plots = []  # Initialize to empty list (populated later if AI data exists)
        row = 0
        first_plot = None

        # Extract fiber data
        iso_time = fiber_data.get('iso_time', np.array([]))
        iso_signal = fiber_data.get('iso', np.array([]))
        gcamp_time = fiber_data.get('gcamp_time', np.array([]))
        gcamp_signal = fiber_data.get('gcamp', np.array([]))

        # Convert to minutes for display
        iso_t_min = iso_time / 60.0
        gcamp_t_min = gcamp_time / 60.0

        # Get dF/F results if available
        if dff_results:
            common_time = dff_results.get('time', np.array([]))
            dff = dff_results.get('dff', np.array([]))
            intermediates = dff_results.get('intermediates', {})
            t_plot_min = common_time / 60.0 if len(common_time) > 0 else np.array([])
        else:
            common_time = np.array([])
            dff = np.array([])
            intermediates = {}
            t_plot_min = np.array([])

        # ========== Panel 1: Raw Signals (Iso + GCaMP on same Y-axis) ==========
        plot_raw = graphics_layout.addPlot(row=row, col=0)
        plot_items.append(plot_raw)
        first_plot = plot_raw
        self._style_plot(plot_raw)

        # Plot both signals on the same Y-axis (with downsampling for performance)
        plot_raw.plot(iso_t_min, iso_signal, pen=pg.mkPen(COLORS['isosbestic'], width=1), name='Iso',
                      clipToView=True, autoDownsample=True, downsampleMethod='subsample')
        plot_raw.plot(gcamp_t_min, gcamp_signal, pen=pg.mkPen(COLORS['gcamp'], width=1), name='GCaMP',
                      clipToView=True, autoDownsample=True, downsampleMethod='subsample')
        plot_raw.setLabel('left', 'Raw Signal (V)', color=COLORS['text'])
        plot_raw.setTitle(f'{fiber_col} - Raw Signals (Iso + GCaMP)', color=COLORS['text'], size='9pt')

        # Add legend
        legend = plot_raw.addLegend(offset=(10, 10))
        legend.setLabelTextColor(COLORS['text'])

        row += 1

        # ========== Intermediate plots (only when show_intermediates is True) ==========
        if show_intermediates and dff_results:
            # Panel 2: GCaMP + Fitted Isosbestic overlay
            if intermediates.get('gcamp_aligned') is not None:
                plot_fit = graphics_layout.addPlot(row=row, col=0)
                plot_items.append(plot_fit)
                self._style_plot(plot_fit, first_plot)
                plot_fit.setXLink(first_plot)

                # Plot GCaMP aligned (with downsampling for performance)
                plot_fit.plot(t_plot_min, intermediates['gcamp_aligned'],
                             pen=pg.mkPen(COLORS['gcamp'], width=1), name='GCaMP',
                             clipToView=True, autoDownsample=True, downsampleMethod='subsample')

                # Plot fitted isosbestic (dashed red)
                if intermediates.get('fitted_iso') is not None:
                    plot_fit.plot(t_plot_min, intermediates['fitted_iso'],
                                 pen=pg.mkPen(COLORS['fitted_iso'], width=1.5,
                                             style=Qt.PenStyle.DashLine),
                                 name='Fitted Iso',
                                 clipToView=True, autoDownsample=True, downsampleMethod='subsample')

                plot_fit.setLabel('left', 'Signal (V)', color=COLORS['text'])
                plot_fit.setTitle('GCaMP + Fitted Isosbestic', color='#888888', size='8pt')

                # Add legend
                legend = plot_fit.addLegend(offset=(10, 10))
                legend.setLabelTextColor(COLORS['text'])

                row += 1

            # Panel 3: dF/F Raw + Detrending curve overlay
            if intermediates.get('dff_raw') is not None:
                plot_dff_raw = graphics_layout.addPlot(row=row, col=0)
                plot_items.append(plot_dff_raw)
                self._style_plot(plot_dff_raw, first_plot)
                plot_dff_raw.setXLink(first_plot)

                # Plot dF/F raw (with downsampling for performance)
                plot_dff_raw.plot(t_plot_min, intermediates['dff_raw'],
                                 pen=pg.mkPen(COLORS['dff_raw'], width=1), name='dF/F raw',
                                 clipToView=True, autoDownsample=True, downsampleMethod='subsample')

                # Plot detrending curve (if present)
                if intermediates.get('detrend_curve') is not None:
                    plot_dff_raw.plot(t_plot_min, intermediates['detrend_curve'],
                                     pen=pg.mkPen(COLORS['detrend'], width=1.5),
                                     name='Detrend',
                                     clipToView=True, autoDownsample=True, downsampleMethod='subsample')

                plot_dff_raw.setLabel('left', 'dF/F (%)', color=COLORS['text'])
                plot_dff_raw.setTitle('dF/F Raw + Detrend Curve', color='#888888', size='8pt')

                # Add legend
                legend = plot_dff_raw.addLegend(offset=(10, 10))
                legend.setLabelTextColor(COLORS['text'])

                row += 1

        # ========== Final dF/F (always shown when results available) ==========
        if dff_results and len(dff) > 0:
            plot_final = graphics_layout.addPlot(row=row, col=0)
            plot_items.append(plot_final)
            self._style_plot(plot_final, first_plot)
            plot_final.setXLink(first_plot)

            plot_final.plot(t_plot_min, dff, pen=pg.mkPen(COLORS['final_dff'], width=1),
                          clipToView=True, autoDownsample=True, downsampleMethod='subsample')
            plot_final.setLabel('left', 'dF/F (%)', color=COLORS['text'])
            plot_final.setTitle(f'{fiber_col} - Final dF/F', color=COLORS['text'], size='9pt')

            row += 1

        # ========== AI Channels ==========
        if ai_data is not None and ai_time is not None and ai_channels:
            ai_plots = self._plot_ai_channels(
                ai_data, ai_time, ai_channels, graphics_layout, row, first_plot
            )
            plot_items.extend(ai_plots)
            row += len(ai_plots)

        # ========== Finalize layout ==========
        # Show X tick labels only on bottom plot
        if plot_items:
            for plot in plot_items[:-1]:
                plot.getAxis('bottom').setStyle(showValues=False)
            plot_items[-1].getAxis('bottom').setStyle(showValues=True)
            plot_items[-1].setLabel('bottom', 'Time (min)', color=COLORS['text'])

        # Auto-range all plots
        for plot in plot_items:
            plot.enableAutoRange()
            plot.autoRange()

        # Add fit regions if enabled (to photometry plots only, not AI)
        photometry_plots = [p for p in plot_items if p not in ai_plots]
        if fit_region_enabled and len(photometry_plots) > 0:
            self.add_fit_regions(exp_idx, photometry_plots, fit_start, fit_end, on_region_changed)

        return plot_items, row

    def _plot_ai_channels(
        self,
        ai_data,
        ai_time: np.ndarray,
        ai_channels: List[Tuple[int, str]],
        graphics_layout: GraphicsLayoutWidget,
        start_row: int,
        first_plot
    ) -> List:
        """
        Plot AI analog channels.

        Args:
            ai_data: Either a DataFrame with AI columns, or a dict of preprocessed
                     AI arrays like {0: np.ndarray, 1: np.ndarray, ...}
            ai_time: Timestamps in seconds (on same time base as ai_data)
            ai_channels: List of (column_index, channel_name) tuples
            graphics_layout: Layout to add plots to
            start_row: Starting row number
            first_plot: First plot for X-axis linking

        Returns:
            List of created plot items
        """
        ai_plots = []

        if ai_data is None or ai_time is None:
            return ai_plots

        # Convert AI time to minutes
        ai_time_min = ai_time / 60.0

        row = start_row
        for col_idx, channel_name in ai_channels:
            # Normalize col_idx - may be string when loaded from NPZ
            col_idx_int = int(col_idx) if isinstance(col_idx, str) else col_idx
            col_idx_str = str(col_idx)

            # Get AI signal - handle both DataFrame and dict formats
            if isinstance(ai_data, dict):
                # Preprocessed format: {col_idx: np.ndarray} - keys may be int or str
                if col_idx in ai_data:
                    ai_signal = ai_data[col_idx]
                elif col_idx_str in ai_data:
                    ai_signal = ai_data[col_idx_str]
                elif col_idx_int in ai_data:
                    ai_signal = ai_data[col_idx_int]
                else:
                    continue
            else:
                # DataFrame format (legacy)
                if col_idx_int >= ai_data.shape[1]:
                    continue
                ai_signal = ai_data.iloc[:, col_idx_int].values

            # Ensure time and signal lengths match
            min_len = min(len(ai_time_min), len(ai_signal))
            if min_len < 10:
                continue

            plot = graphics_layout.addPlot(row=row, col=0)
            ai_plots.append(plot)
            self._style_plot(plot, first_plot)

            if first_plot:
                plot.setXLink(first_plot)

            # Use column index for consistent colors across all views
            color = COLORS['ai_channels'][col_idx_int % len(COLORS['ai_channels'])]

            plot.plot(ai_time_min[:min_len], ai_signal[:min_len], pen=pg.mkPen(color, width=1),
                     clipToView=True, autoDownsample=True, downsampleMethod='subsample')
            plot.setLabel('left', channel_name, color=color)
            plot.setTitle(channel_name, color='#888888', size='8pt')

            row += 1

        return ai_plots

    def _style_plot(self, plot, first_plot=None):
        """
        Apply consistent dark theme styling to a plot.

        Args:
            plot: PyQtGraph PlotItem
            first_plot: Optional first plot for X-axis linking
        """
        plot.setMouseEnabled(x=True, y=True)
        plot.vb.setMouseMode(pg.ViewBox.PanMode)

        # Custom wheel event: shift+scroll = X zoom, otherwise ignore (let scroll area handle)
        # Note: pyqtgraph's AxisItem passes axis=0 kwarg, so we must accept it
        def handle_wheel(event, p=plot, axis=None):
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                # Shift+scroll = zoom X axis only
                delta = event.delta()
                scale_factor = 1.1 if delta > 0 else 0.9
                p.vb.scaleBy((scale_factor, 1.0))
                event.accept()
            else:
                # Regular scroll = pass to parent (scroll area)
                event.ignore()

        plot.vb.wheelEvent = handle_wheel

        # Axis styling
        plot.getAxis('left').setPen(COLORS['axis'])
        plot.getAxis('left').setTextPen(COLORS['text'])
        plot.getAxis('bottom').setPen(COLORS['axis'])
        plot.getAxis('bottom').setTextPen(COLORS['text'])
        plot.showGrid(x=False, y=False)

    def add_fit_regions(
        self,
        exp_idx: int,
        plot_items: List,
        fit_start: float,
        fit_end: float,
        on_region_changed: Optional[Callable] = None
    ) -> List:
        """
        Add synchronized fit region selectors to plots.

        Args:
            exp_idx: Experiment index
            plot_items: List of plot items to add regions to
            fit_start: Start time in minutes
            fit_end: End time in minutes
            on_region_changed: Callback when region changes (receives start, end)

        Returns:
            List of LinearRegionItem objects
        """
        # Remove existing regions
        self.clear_fit_regions(exp_idx)

        if not plot_items:
            return []

        regions = []
        for plot in plot_items:
            region = pg.LinearRegionItem(
                values=[fit_start, fit_end],
                brush=pg.mkBrush(*COLORS['fit_region_brush']),
                pen=pg.mkPen(COLORS['fit_region_pen'], width=2),
                movable=True
            )
            # Add wider hover zone and cursor for boundary lines
            for line in region.lines:
                line.setHoverPen(pg.mkPen(COLORS['fit_region_pen'], width=6))
                line.setCursor(Qt.CursorShape.SizeHorCursor)
            plot.addItem(region)
            regions.append(region)

        self._fit_regions[exp_idx] = regions

        # Connect first region to sync all others
        # Note: Use try/except to handle case where C++ objects are deleted
        def on_region_change():
            if self._updating_regions:
                return
            self._updating_regions = True
            try:
                min_val, max_val = regions[0].getRegion()

                # Sync all other regions
                for region in regions[1:]:
                    region.blockSignals(True)
                    region.setRegion([min_val, max_val])
                    region.blockSignals(False)
            except RuntimeError:
                pass  # C++ object deleted
            finally:
                self._updating_regions = False

        def on_region_change_finished():
            if on_region_changed:
                try:
                    min_val, max_val = regions[0].getRegion()
                    on_region_changed(min_val, max_val)
                except RuntimeError:
                    pass  # C++ object deleted

        regions[0].sigRegionChanged.connect(on_region_change)
        regions[0].sigRegionChangeFinished.connect(on_region_change_finished)

        # Connect other regions to sync back to first
        for region in regions[1:]:
            def sync_to_first(r=region):
                if self._updating_regions:
                    return
                self._updating_regions = True
                try:
                    min_val, max_val = r.getRegion()
                    regions[0].setRegion([min_val, max_val])
                except RuntimeError:
                    pass  # C++ object deleted
                finally:
                    self._updating_regions = False
            region.sigRegionChanged.connect(sync_to_first)
            region.sigRegionChangeFinished.connect(on_region_change_finished)

        return regions

    def clear_fit_regions(self, exp_idx: int):
        """Remove fit regions for an experiment."""
        if exp_idx in self._fit_regions:
            # Disconnect signals from all regions to prevent callbacks after deletion
            for region in self._fit_regions[exp_idx]:
                try:
                    region.sigRegionChanged.disconnect()
                    region.sigRegionChangeFinished.disconnect()
                except (TypeError, RuntimeError):
                    pass  # Already disconnected or C++ object deleted
            del self._fit_regions[exp_idx]

    def update_fit_regions(self, exp_idx: int, fit_start: float, fit_end: float):
        """Update fit region positions programmatically."""
        if exp_idx not in self._fit_regions:
            return

        self._updating_regions = True
        for region in self._fit_regions[exp_idx]:
            region.setRegion([fit_start, fit_end])
        self._updating_regions = False

    @staticmethod
    def subsample(time: np.ndarray, signal: np.ndarray, max_points: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subsample data for faster preview plotting.

        Args:
            time: Time array
            signal: Signal array
            max_points: Maximum number of points to return

        Returns:
            Tuple of (subsampled_time, subsampled_signal)
        """
        if len(time) <= max_points:
            return time, signal
        step = len(time) // max_points
        return time[::step], signal[::step]

    @staticmethod
    def detect_time_unit(time: np.ndarray) -> str:
        """
        Detect if time is in seconds or milliseconds.

        Args:
            time: Time array

        Returns:
            'seconds' or 'milliseconds'
        """
        if len(time) < 10:
            return 'seconds'

        time_range = time[-1] - time[0]
        approx_sr = len(time) / time_range

        # If sampling rate is > 500, time is probably in milliseconds
        if approx_sr > 500:
            return 'milliseconds'
        return 'seconds'

    @staticmethod
    def convert_time_to_minutes(time: np.ndarray) -> np.ndarray:
        """
        Convert time to minutes, auto-detecting if input is in seconds or milliseconds.

        Args:
            time: Time array in seconds or milliseconds

        Returns:
            Time array in minutes
        """
        unit = ExperimentPlotter.detect_time_unit(time)
        if unit == 'milliseconds':
            return time / 60000.0
        return time / 60.0
