"""
Spectral Analysis Dialog for PhysioMetrics.

This dialog provides comprehensive spectral analysis tools for identifying
and filtering oscillatory noise contamination in respiratory signals.
"""

import sys
import numpy as np
from scipy import signal
import traceback

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox,
    QPushButton, QMessageBox, QCheckBox
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt


class SpectralAnalysisDialog(QDialog):
    def __init__(self, parent=None, t=None, y=None, sr_hz=None, stim_spans=None, parent_window=None, use_zscore=True):
        super().__init__(parent)
        self.setWindowTitle("Spectral Analysis & Notch Filter")
        self.resize(1400, 900)

        self.parent_window = parent_window  # Reference to main window for sweep navigation
        self.t = t
        self.y = y
        self.sr_hz = sr_hz
        self.stim_spans = stim_spans  # List of (start, end) tuples for stimulation periods
        self.notch_lower = None
        self.notch_upper = None
        self.initial_zscore = use_zscore  # Store initial z-score state

        # Normalize time to stim onset if stim available
        self.t_offset = 0
        if stim_spans and len(stim_spans) > 0:
            self.t_offset = stim_spans[0][0]  # First stim onset

        # Main layout with tight spacing
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Control panel at top
        control_layout = QHBoxLayout()
        control_layout.setSpacing(5)

        # Sweep navigation controls
        if parent_window:
            self.prev_sweep_btn = QPushButton("◄ Prev Sweep")
            self.prev_sweep_btn.clicked.connect(self.on_prev_sweep)
            control_layout.addWidget(self.prev_sweep_btn)

            self.sweep_label = QLabel(f"Sweep: {getattr(parent_window.state, 'sweep_idx', 0) + 1}")
            self.sweep_label.setStyleSheet("color: #2a7fff; font-size: 11pt; font-weight: bold; background-color: #2a2a2a; padding: 4px 10px; border-radius: 3px; border: 1px solid #3e3e42;")
            control_layout.addWidget(self.sweep_label)

            self.next_sweep_btn = QPushButton("Next Sweep ►")
            self.next_sweep_btn.clicked.connect(self.on_next_sweep)
            control_layout.addWidget(self.next_sweep_btn)

            control_layout.addWidget(QLabel("  |  "))  # Separator

        # Notch filter controls
        control_layout.addWidget(QLabel("Notch Filter (Hz):"))

        self.lower_freq_spin = QDoubleSpinBox()
        self.lower_freq_spin.setRange(0.0, sr_hz/2 if sr_hz else 100)
        self.lower_freq_spin.setValue(0.0)
        self.lower_freq_spin.setDecimals(2)
        self.lower_freq_spin.setSuffix(" Hz")
        control_layout.addWidget(QLabel("Lower:"))
        control_layout.addWidget(self.lower_freq_spin)

        self.upper_freq_spin = QDoubleSpinBox()
        self.upper_freq_spin.setRange(0.0, sr_hz/2 if sr_hz else 100)
        self.upper_freq_spin.setValue(0.0)
        self.upper_freq_spin.setDecimals(2)
        self.upper_freq_spin.setSuffix(" Hz")
        control_layout.addWidget(QLabel("Upper:"))
        control_layout.addWidget(self.upper_freq_spin)

        self.apply_filter_btn = QPushButton("Apply Filter")
        self.apply_filter_btn.clicked.connect(self.on_apply_filter)
        control_layout.addWidget(self.apply_filter_btn)

        self.reset_filter_btn = QPushButton("Reset Filter")
        self.reset_filter_btn.clicked.connect(self.on_reset_filter)
        control_layout.addWidget(self.reset_filter_btn)

        # Add separator
        control_layout.addWidget(QLabel("  |  "))

        # Mean Subtraction controls
        self.mean_subtract_cb = QCheckBox("Mean Subtraction")
        self.mean_subtract_cb.setChecked(parent_window.state.use_mean_sub if parent_window else False)
        self.mean_subtract_cb.toggled.connect(self.on_mean_subtract_toggled)
        control_layout.addWidget(self.mean_subtract_cb)

        self.mean_window_spin = QDoubleSpinBox()
        self.mean_window_spin.setRange(0.1, 100.0)
        # Use mean_val attribute which is the window size in seconds
        self.mean_window_spin.setValue(parent_window.state.mean_val if (parent_window and parent_window.state.mean_val) else 10.0)
        self.mean_window_spin.setDecimals(1)
        self.mean_window_spin.setSuffix(" s")
        self.mean_window_spin.setEnabled(self.mean_subtract_cb.isChecked())
        self.mean_window_spin.valueChanged.connect(self.on_mean_window_changed)
        control_layout.addWidget(self.mean_window_spin)

        # Add separator
        control_layout.addWidget(QLabel("  |  "))

        # Z-Score Normalization checkbox
        self.zscore_cb = QCheckBox("Z-Score Normalization")
        self.zscore_cb.setChecked(use_zscore)
        self.zscore_cb.setToolTip("Normalize signal to zero mean and unit standard deviation")
        self.zscore_cb.toggled.connect(self.on_zscore_toggled)
        control_layout.addWidget(self.zscore_cb)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # Plot area with matplotlib
        # Create figure with dark background
        self.figure = Figure(figsize=(14, 10), facecolor='#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        main_layout.addWidget(self.canvas)

        # Buttons at bottom
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        main_layout.addLayout(btn_layout)

        # Initial plot
        self.update_plots()

        # Apply dark theme and title bar
        self._apply_dark_theme()
        self._enable_dark_title_bar()

    def _apply_dark_theme(self):
        """Apply dark theme styling to match main application."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
                background-color: transparent;
            }
            QCheckBox {
                color: #d4d4d4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QCheckBox::indicator:hover {
                border-color: #2a7fff;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
            }
            QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #2a7fff;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: #d4d4d4;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border-color: #2a7fff;
            }
            QPushButton:pressed {
                background-color: #2a7fff;
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform == "win32":
            try:
                from ctypes import windll, byref, sizeof, c_int
                DWMWA_USE_IMMERSIVE_DARK_MODE = 20
                hwnd = int(self.winId())
                value = c_int(1)
                windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, byref(value), sizeof(value)
                )
            except Exception:
                pass

    def update_plots(self):
        """Generate power spectrum and wavelet plots."""
        if self.y is None or len(self.y) == 0:
            return

        self.figure.clear()

        # Create subplots with aligned axes: power spectrum on top, wavelet on bottom
        # Let matplotlib handle spacing automatically with tight_layout
        gs = GridSpec(2, 1, figure=self.figure)
        ax1 = self.figure.add_subplot(gs[0])
        ax2 = self.figure.add_subplot(gs[1])

        # Power Spectrum (Welch method)
        if self.sr_hz:
            # Increase resolution 5x: longer nperseg for smoother curves
            nperseg = min(163840, len(self.y)//2)  # 5x increase from 32768 to 163840
            noverlap = int(nperseg * 0.90)  # 90% overlap for smooth estimate

            # All sweeps concatenated (if parent window available)
            if self.parent_window and hasattr(self.parent_window.state, 'sweeps') and self.parent_window.state.analyze_chan:
                try:
                    sweeps_dict = self.parent_window.state.sweeps
                    analyze_chan = self.parent_window.state.analyze_chan

                    if analyze_chan in sweeps_dict:
                        sweeps_data = sweeps_dict[analyze_chan]  # Shape: (n_samples, n_sweeps)

                        # Concatenate all sweeps
                        all_sweeps_concat = []
                        for sweep_idx in range(sweeps_data.shape[1]):
                            all_sweeps_concat.append(sweeps_data[:, sweep_idx])

                        if all_sweeps_concat:
                            concatenated = np.concatenate(all_sweeps_concat)

                            # Compute Welch PSD on concatenated data
                            freqs_all, psd_all = signal.welch(concatenated, fs=self.sr_hz, nperseg=nperseg, noverlap=noverlap)
                            mask_all = freqs_all <= 30
                            ax1.plot(freqs_all[mask_all], psd_all[mask_all], 'magenta', linewidth=2, label='All Sweeps', alpha=0.8)
                except Exception as e:
                    print(f"[spectral] Failed to compute all-sweeps spectrum: {e}")

            # Full trace spectrum
            freqs, psd = signal.welch(self.y, fs=self.sr_hz, nperseg=nperseg, noverlap=noverlap)

            # Only plot frequencies up to 30 Hz (respiratory range)
            mask = freqs <= 30
            freqs_plot = freqs[mask]
            psd_plot = psd[mask]

            ax1.plot(freqs_plot, psd_plot, 'cyan', linewidth=2, label='Current Trace')

            # If stim spans provided, compute spectrum during and after stim
            if self.stim_spans and len(self.stim_spans) > 0:
                # During stim: from first laser onset to last laser offset
                first_stim_start = self.stim_spans[0][0]
                last_stim_end = self.stim_spans[-1][1]

                stim_start_idx = np.searchsorted(self.t, first_stim_start)
                stim_end_idx = np.searchsorted(self.t, last_stim_end)

                stim_data = None
                post_stim_data = None

                if stim_end_idx > stim_start_idx:
                    stim_data = self.y[stim_start_idx:stim_end_idx]

                # Post-stim: everything after last laser offset
                if stim_end_idx < len(self.y):
                    post_stim_data = self.y[stim_end_idx:]

                # During stim spectrum (use adaptive nperseg if data is short)
                if stim_data is not None and len(stim_data) > 256:
                    nperseg_stim = min(nperseg, len(stim_data)//2)
                    noverlap_stim = int(nperseg_stim * 0.90)
                    freqs_stim, psd_stim = signal.welch(stim_data, fs=self.sr_hz, nperseg=nperseg_stim, noverlap=noverlap_stim)
                    mask_stim = freqs_stim <= 30
                    ax1.plot(freqs_stim[mask_stim], psd_stim[mask_stim], 'orange', linewidth=2, label='During Stim', alpha=0.8)

                # Post-stim spectrum (use adaptive nperseg if data is short)
                if post_stim_data is not None and len(post_stim_data) > 256:
                    nperseg_post = min(nperseg, len(post_stim_data)//2)
                    noverlap_post = int(nperseg_post * 0.90)
                    freqs_post, psd_post = signal.welch(post_stim_data, fs=self.sr_hz, nperseg=nperseg_post, noverlap=noverlap_post)
                    mask_post = freqs_post <= 30
                    ax1.plot(freqs_post[mask_post], psd_post[mask_post], 'lime', linewidth=2, label='Post-Stim', alpha=0.8)

            # Add labels with padding to prevent cutoff
            ax1.set_xlabel('Frequency (Hz)', color='#cccccc', fontsize=10, labelpad=8)
            ax1.set_ylabel('Power Spectral Density', color='#cccccc', fontsize=10, labelpad=8)
            ax1.set_title('Power Spectrum (Welch Method)', color='#e0e0e0', fontsize=11, pad=10)
            ax1.set_xlim([0, 30])
            ax1.grid(True, alpha=0.3, color='gray', linestyle='--')
            ax1.set_facecolor('#1e1e1e')
            ax1.tick_params(colors='#cccccc', labelsize=9, width=1, length=4, pad=5)

            # Set gray spines
            for spine in ax1.spines.values():
                spine.set_edgecolor('#555555')
                spine.set_linewidth(1)

            # Highlight notch filter region if set
            if self.notch_lower is not None and self.notch_upper is not None:
                ax1.axvspan(self.notch_lower, self.notch_upper, alpha=0.3, color='red', label='Notch Filter')

            ax1.legend(facecolor='#2a2a2a', edgecolor='#555555', labelcolor='#cccccc', fontsize=9)

        # Wavelet Analysis (Continuous Wavelet Transform)
        try:
            if self.sr_hz and self.t is not None and len(self.y) > 0:
                print(f"[wavelet] Computing CWT for {len(self.y)} samples at {self.sr_hz} Hz")

                # Downsample for faster computation if signal is very long
                downsample_factor = 1
                if len(self.y) > 100000:  # If more than 100k samples
                    downsample_factor = max(1, len(self.y) // 50000)
                    y_ds = self.y[::downsample_factor]
                    t_ds = self.t[::downsample_factor]
                    print(f"[wavelet] Downsampling by factor {downsample_factor} to {len(y_ds)} samples")
                else:
                    y_ds = self.y
                    t_ds = self.t

                # Create frequency array from 0.5 Hz to 30 Hz (fewer frequencies for speed)
                frequencies = np.linspace(0.5, 30, 50)  # Restricted to respiratory range

                # Compute CWT using FFT-based convolution for speed
                cwtmatr = np.zeros((len(frequencies), len(y_ds)))

                for i, freq in enumerate(frequencies):
                    # Create Complex Morlet wavelet for this frequency
                    w = 6.0  # Standard Morlet parameter
                    sigma = w / (2 * np.pi * freq)  # Time domain width

                    # Limit wavelet length for speed
                    max_wavelet_samples = min(int(10 * sigma * self.sr_hz / downsample_factor), len(y_ds) // 2)
                    wavelet_time = np.arange(-max_wavelet_samples, max_wavelet_samples) / (self.sr_hz / downsample_factor)

                    # Complex Morlet wavelet (optimal for oscillatory respiratory signals)
                    wavelet = np.exp(2j * np.pi * freq * wavelet_time) * np.exp(-wavelet_time**2 / (2 * sigma**2))
                    wavelet = wavelet / np.sqrt(sigma * np.sqrt(np.pi))  # Normalize

                    # FFT-based convolution (much faster)
                    convolved = signal.fftconvolve(y_ds, wavelet, mode='same')
                    cwtmatr[i, :] = np.abs(convolved)

                print(f"[wavelet] CWT matrix shape: {cwtmatr.shape}, min={cwtmatr.min():.2e}, max={cwtmatr.max():.2e}")

                # Use percentile-based color scaling to handle bright transients (like sniffing bouts)
                # This prevents one bright spot from washing out the rest of the signal
                vmin = 0
                vmax = np.percentile(cwtmatr, 95)  # Use 95th percentile instead of max
                print(f"[wavelet] Color scale: vmin={vmin:.2e}, vmax (95th percentile)={vmax:.2e}, actual max={cwtmatr.max():.2e}")

                # Plot scalogram (use downsampled time array, normalized to stim onset)
                t_plot_start = t_ds[0] - self.t_offset
                t_plot_end = t_ds[-1] - self.t_offset
                im = ax2.imshow(cwtmatr, extent=[t_plot_start, t_plot_end, frequencies[0], frequencies[-1]],
                           cmap='hot', aspect='auto', interpolation='bilinear', origin='lower',
                           vmin=vmin, vmax=vmax)

                # Add vertical lines for stim onset and offset
                if self.stim_spans and len(self.stim_spans) > 0:
                    stim_start_rel = self.stim_spans[0][0] - self.t_offset
                    stim_end_rel = self.stim_spans[-1][1] - self.t_offset  # Use last span's end time
                    ax2.axvline(x=stim_start_rel, color='lime', linewidth=2, linestyle='--', alpha=0.9)
                    ax2.axvline(x=stim_end_rel, color='lime', linewidth=2, linestyle='--', alpha=0.9)
                    ax2.legend(['Stim On/Offset'], facecolor='#2a2a2a', edgecolor='#555555', labelcolor='#cccccc', fontsize=9, loc='upper right')

                # Add labels with padding to prevent cutoff
                ax2.set_xlabel('Time (s, rel. to stim onset)', color='#cccccc', fontsize=10, labelpad=8)
                ax2.set_ylabel('Frequency (Hz)', color='#cccccc', fontsize=10, labelpad=8)
                ax2.set_title('Wavelet Analysis (Scalogram)', color='#e0e0e0', fontsize=11, pad=10)
                ax2.set_ylim([0, 30])
                ax2.set_facecolor('#1e1e1e')
                ax2.tick_params(colors='#cccccc', labelsize=9, width=1, length=4, pad=5)

                # Set gray spines
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#555555')
                    spine.set_linewidth(1)

                # Add colorbar
                cbar = self.figure.colorbar(im, ax=ax2, pad=0.02)
                cbar.set_label('Magnitude', color='#cccccc', fontsize=9, labelpad=8)
                cbar.ax.tick_params(colors='#cccccc', labelsize=8)
                # Make colorbar outline gray
                cbar.outline.set_edgecolor('#555555')
                cbar.outline.set_linewidth(1)

                print("[wavelet] Scalogram plotted successfully")

        except Exception as e:
            error_msg = f'Wavelet analysis error: {str(e)}'
            print(f"[wavelet] ERROR: {error_msg}")
            traceback.print_exc()

            ax2.text(0.5, 0.5, error_msg,
                    ha='center', va='center', transform=ax2.transAxes, color='#cccccc', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='#8b0000', alpha=0.7))
            ax2.set_facecolor('#1e1e1e')
            ax2.set_xlabel('Time (s)', color='#cccccc', fontsize=10)
            ax2.set_ylabel('Frequency (Hz)', color='#cccccc', fontsize=10)
            ax2.set_title('Wavelet Analysis (Error)', color='#e0e0e0', fontsize=11)
            ax2.tick_params(colors='#cccccc', labelsize=9, width=1, length=4)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#555555')
                spine.set_linewidth(1)

        # Use tight_layout with padding to prevent text cutoff
        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def on_apply_filter(self):
        """Store the notch filter settings."""
        self.notch_lower = self.lower_freq_spin.value()
        self.notch_upper = self.upper_freq_spin.value()

        if self.notch_lower >= self.notch_upper:
            QMessageBox.warning(self, "Invalid Range", "Lower frequency must be less than upper frequency.")
            return

        self.update_plots()
        QMessageBox.information(self, "Filter Applied",
                              f"Notch filter set to {self.notch_lower:.2f} - {self.notch_upper:.2f} Hz.\n"
                              "Close this dialog to apply the filter to your signal.")

    def on_reset_filter(self):
        """Reset the notch filter."""
        self.notch_lower = None
        self.notch_upper = None
        self.lower_freq_spin.setValue(0.0)
        self.upper_freq_spin.setValue(0.0)
        self.update_plots()

    def on_mean_subtract_toggled(self, checked):
        """Handle mean subtraction checkbox toggle."""
        if self.parent_window:
            self.parent_window.state.use_mean_sub = checked
            self.mean_window_spin.setEnabled(checked)
            # Update main window and this dialog
            self.parent_window.update_and_redraw()
            self._load_sweep_data()
            self.update_plots()

    def on_mean_window_changed(self, value):
        """Handle mean subtraction window size change."""
        if self.parent_window:
            self.parent_window.state.mean_val = value
            # Update main window and this dialog
            self.parent_window.update_and_redraw()
            self._load_sweep_data()
            self.update_plots()

    def on_zscore_toggled(self, checked):
        """Handle z-score normalization checkbox toggle."""
        if self.parent_window:
            self.parent_window.use_zscore_normalization = checked
            # Clear z-score cache to recompute with new setting
            self.parent_window.zscore_global_mean = None
            self.parent_window.zscore_global_std = None
            # Update main window and this dialog
            self.parent_window.update_and_redraw()
            self._load_sweep_data()
            self.update_plots()

    def get_filter_params(self):
        """Return the notch filter parameters."""
        return self.notch_lower, self.notch_upper

    def on_prev_sweep(self):
        """Navigate to previous sweep."""
        if not self.parent_window:
            return

        # Move to previous sweep
        if self.parent_window.state.sweep_idx > 0:
            self.parent_window.state.sweep_idx -= 1
            # Re-extract data for new sweep
            self._load_sweep_data()
            self.update_plots()

    def on_next_sweep(self):
        """Navigate to next sweep."""
        if not self.parent_window:
            return

        # Move to next sweep
        sweep_count = self.parent_window.navigation_manager._sweep_count()
        if self.parent_window.state.sweep_idx < sweep_count - 1:
            self.parent_window.state.sweep_idx += 1
            # Re-extract data for new sweep
            self._load_sweep_data()
            self.update_plots()

    def _load_sweep_data(self):
        """Reload data for current sweep from parent window."""
        if not self.parent_window:
            return

        # Get current sweep data (already filtered by _current_trace)
        t_all, y_all = self.parent_window._current_trace()
        if t_all is None or y_all is None:
            return

        # Update instance variables
        self.t = t_all
        self.y = y_all

        # Get stim spans for this sweep
        if hasattr(self.parent_window.state, 'stim_markers') and self.parent_window.state.stim_markers:
            sweep_idx = self.parent_window.state.sweep_idx
            if sweep_idx in self.parent_window.state.stim_markers:
                self.stim_spans = self.parent_window.state.stim_markers[sweep_idx]
            else:
                self.stim_spans = None

        # Update sweep label if it exists
        if hasattr(self, 'sweep_label'):
            self.sweep_label.setText(f"Sweep: {self.parent_window.state.sweep_idx + 1}")

        # Calculate stim onset offset for time normalization
        if self.stim_spans and len(self.stim_spans) > 0:
            self.t_offset = self.stim_spans[0][0]  # Use first stim onset
        else:
            self.t_offset = 0.0
