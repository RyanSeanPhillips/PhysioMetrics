"""
Prominence Threshold Detection Dialog

Interactive dialog using Otsu's method to auto-detect optimal prominence threshold.
Provides histogram visualization and manual threshold adjustment via draggable line.

Otsu's Method:
    - Detects all peaks with minimal prominence
    - Calculates histogram of peak prominences
    - Finds threshold that maximizes inter-class variance
    - Separates "noise peaks" from "breath peaks" optimally
"""

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QLabel, QGroupBox, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.signal import find_peaks


class ProminenceThresholdDialog(QDialog):
    """Interactive prominence threshold detection using Otsu's method."""

    def __init__(self, parent=None, y_data=None, sr_hz=None, current_prom=None, current_min_dist=None, current_height_threshold=None,
                 percentile_cutoff=99, num_bins=200):
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect Prominence Threshold")
        self.resize(1000, 700)

        self.y_data = y_data
        self.sr_hz = sr_hz or 1000.0
        self.current_prom = current_prom or 0.1
        self.current_min_dist = current_min_dist or 0.05
        self.user_threshold = current_height_threshold  # Previously set threshold

        # Cached peak detection
        self.all_peaks = None
        self.all_peak_heights = None

        # Auto-calculated threshold
        self.auto_threshold = None
        self.current_threshold = None  # User-adjusted value
        self.local_min_threshold = None  # Local minimum threshold
        self.inter_class_variance_curve = None  # For plotting

        # Separation metric for display
        self.separation_metric = 1.0

        # Draggable line (only vertical, no horizontal to avoid obscuring labels)
        self.threshold_vline = None
        self.otsu_reference_line = None  # Fixed line showing Otsu's calculated threshold
        self.is_dragging = False

        # Y2 axis mode toggle
        self.y2_mode = "peak_count"  # or "variance"

        # Histogram controls - use passed values or defaults
        self.percentile_cutoff = percentile_cutoff
        self.num_bins = num_bins

        # Apply dark theme
        self._apply_dark_theme()

        self._setup_ui()

        # Detect peaks and calculate threshold
        if y_data is not None and len(y_data) > 0:
            self._detect_all_peaks()
            self._calculate_otsu_threshold()

            # Ensure canvas is sized before plotting
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()

            self._plot_histogram()

            # Process events again to ensure plot is rendered
            QApplication.processEvents()

    def _apply_dark_theme(self):
        """Apply dark theme styling to match main application."""
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
            }
            QGroupBox {
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #d4d4d4;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 5px 15px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border: 1px solid #505050;
            }
            QPushButton:pressed {
                background-color: #505050;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666666;
                border: 1px solid #2d2d2d;
            }
            QLineEdit {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
                selection-background-color: #3e3e42;
            }
            QLineEdit:focus {
                border: 1px solid #2a7fff;
            }
            QDialogButtonBox QPushButton {
                min-width: 80px;
            }
        """)

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Title and description
        title = QLabel("<h2>Prominence Threshold Detection (Otsu's Method)</h2>")
        layout.addWidget(title)

        desc = QLabel(
            "Automatically detects optimal prominence threshold by analyzing the distribution "
            "of peak prominences. The threshold separates noise from real breaths by maximizing "
            "inter-class variance."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Threshold display and controls with peak count
        threshold_group = QGroupBox("Detected Threshold")
        threshold_layout = QHBoxLayout()

        threshold_layout.addWidget(QLabel("Optimal Prominence:"))

        self.lbl_threshold = QLabel("Calculating...")
        self.lbl_threshold.setStyleSheet("font-weight: bold; font-size: 14pt; color: #2a7fff;")
        threshold_layout.addWidget(self.lbl_threshold)

        threshold_layout.addStretch()

        self.lbl_peak_count = QLabel("Peaks detected: 0")
        threshold_layout.addWidget(self.lbl_peak_count)

        self.btn_reset = QPushButton("Reset to Auto")
        self.btn_reset.setToolTip("Reset threshold to auto-detected value")
        self.btn_reset.clicked.connect(self._reset_threshold)
        self.btn_reset.setEnabled(False)
        threshold_layout.addWidget(self.btn_reset)

        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # Interactive plot with toggle
        plot_header = QHBoxLayout()
        plot_label = QLabel("<b>Interactive Histogram</b> - <span style='color: #ff6666;'>Drag the red line to adjust threshold</span>")
        plot_label.setStyleSheet("color: #666; font-size: 10pt;")
        plot_header.addWidget(plot_label)

        plot_header.addStretch()

        # "Set to Otsu" button
        self.btn_set_otsu = QPushButton("→ Otsu")
        self.btn_set_otsu.setToolTip("Set threshold to Otsu's calculated value")
        self.btn_set_otsu.clicked.connect(self._set_to_otsu)
        plot_header.addWidget(self.btn_set_otsu)

        # "Set to Local Min" button
        self.btn_set_local_min = QPushButton("→ Local Min")
        self.btn_set_local_min.setToolTip("Set threshold to first local minimum")
        self.btn_set_local_min.clicked.connect(self._set_to_local_min)
        self.btn_set_local_min.setEnabled(False)  # Enabled only if local min exists
        plot_header.addWidget(self.btn_set_local_min)

        # Y2 axis toggle button (initial text shows what you'll see if you CLICK it)
        self.btn_toggle_y2 = QPushButton("Show: Inter-Class Variance")
        self.btn_toggle_y2.setToolTip("Toggle between Peak Count and Inter-Class Variance")
        self.btn_toggle_y2.setMaximumWidth(220)
        self.btn_toggle_y2.clicked.connect(self._toggle_y2_axis)
        plot_header.addWidget(self.btn_toggle_y2)

        layout.addLayout(plot_header)

        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.canvas)

        # Add matplotlib navigation toolbar for zoom/pan
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2d2d2d;
                border: 1px solid #3e3e42;
                spacing: 3px;
            }
            QToolButton {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
            }
            QToolButton:hover {
                background-color: #3e3e42;
            }
            QToolButton:pressed {
                background-color: #505050;
            }
        """)
        layout.addWidget(self.toolbar)

        # Connect drag events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        # Advanced parameters (always visible)
        params_group = QGroupBox("Advanced Parameters")
        params_layout = QFormLayout()

        self.le_min_dist = QLineEdit(str(self.current_min_dist))
        self.le_min_dist.setToolTip("Minimum time between peaks (seconds)")
        params_layout.addRow("Min Peak Distance (s):", self.le_min_dist)

        self.le_threshold_height = QLineEdit("")  # Will be populated after threshold calculation
        self.le_threshold_height.setToolTip("Absolute height threshold - auto-populated from Otsu's method")
        params_layout.addRow("Height Threshold:", self.le_threshold_height)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Histogram controls
        histogram_controls_group = QGroupBox("Histogram Controls")
        histogram_controls_layout = QHBoxLayout()

        # Percentile cutoff control
        from PyQt6.QtWidgets import QSpinBox, QComboBox
        histogram_controls_layout.addWidget(QLabel("Outlier Cutoff:"))
        self.percentile_combo = QComboBox()
        self.percentile_combo.addItems(["90%", "95%", "99%", "100% (None)"])
        # Set to stored value
        if self.percentile_cutoff == 90:
            self.percentile_combo.setCurrentText("90%")
        elif self.percentile_cutoff == 95:
            self.percentile_combo.setCurrentText("95%")
        elif self.percentile_cutoff == 99:
            self.percentile_combo.setCurrentText("99%")
        else:
            self.percentile_combo.setCurrentText("100% (None)")
        self.percentile_combo.setToolTip("Exclude outliers above this percentile from histogram")
        self.percentile_combo.currentTextChanged.connect(self._on_percentile_changed)
        histogram_controls_layout.addWidget(self.percentile_combo)

        histogram_controls_layout.addSpacing(20)

        # Bin count control
        histogram_controls_layout.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(20, 500)  # Increased max to 500
        self.bins_spin.setValue(self.num_bins)  # Use stored value
        self.bins_spin.setSingleStep(10)
        self.bins_spin.setToolTip("Number of bins in histogram")
        self.bins_spin.valueChanged.connect(self._on_bins_changed)
        histogram_controls_layout.addWidget(self.bins_spin)

        histogram_controls_layout.addStretch()

        histogram_controls_group.setLayout(histogram_controls_layout)
        layout.addWidget(histogram_controls_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _detect_all_peaks(self):
        """Run peak detection once with minimal threshold."""
        print("[Prominence Dialog] Detecting all peaks...")
        import time
        t_start = time.time()

        try:
            min_dist_samples = int(self.current_min_dist * self.sr_hz)

            # Find ALL peaks with very low prominence AND above baseline (height > 0)
            # height=0 filters out rebound peaks below baseline, giving cleaner 2-population model
            peaks, props = find_peaks(self.y_data, height=0, prominence=0.001, distance=min_dist_samples)

            self.all_peaks = peaks
            self.all_peak_heights = self.y_data[peaks]  # Use peak heights instead of prominences

            # Calculate percentile to exclude large artifacts
            if len(self.all_peak_heights) > 0:
                self.percentile_95 = np.percentile(self.all_peak_heights, self.percentile_cutoff)

                # Count outliers above percentile
                outliers = np.sum(self.all_peak_heights > self.percentile_95)

                print(f"[Prominence Dialog] Peak height range: {self.all_peak_heights.min():.3f} - {self.all_peak_heights.max():.3f}")
                print(f"[Prominence Dialog] {self.percentile_cutoff}th percentile: {self.percentile_95:.3f}")
                print(f"[Prominence Dialog] Outliers excluded from histogram: {outliers} ({100*outliers/len(self.all_peak_heights):.1f}%)")
            else:
                self.percentile_95 = None

            t_elapsed = time.time() - t_start
            print(f"[Prominence Dialog] Found {len(self.all_peaks)} peaks in {t_elapsed:.2f}s")

            self.lbl_peak_count.setText(f"Peaks detected: {len(self.all_peaks)}")

        except Exception as e:
            print(f"[Prominence Dialog] Error: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_otsu_threshold(self):
        """Calculate optimal height threshold using Otsu's method."""
        if self.all_peak_heights is None or len(self.all_peak_heights) < 10:
            return

        try:
            # Filter out outliers above 95th percentile to focus on real breath distribution
            if self.percentile_95 is not None:
                peak_heights = self.all_peak_heights[self.all_peak_heights <= self.percentile_95]
                print(f"[Otsu] Using {len(peak_heights)} peaks (excluded {len(self.all_peak_heights) - len(peak_heights)} outliers)")
            else:
                peak_heights = self.all_peak_heights

            if len(peak_heights) < 10:
                print("[Otsu] Not enough peaks after filtering outliers")
                return

            # Normalize peak heights to [0, 255] for Otsu
            heights_norm = ((peak_heights - peak_heights.min()) /
                        (peak_heights.max() - peak_heights.min()) * 255).astype(np.uint8)

            # Compute histogram
            hist, bin_edges = np.histogram(heights_norm, bins=256, range=(0, 256))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Otsu's method: maximize inter-class variance
            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]

            mean1 = np.cumsum(hist * bin_centers) / (weight1 + 1e-10)
            mean2 = (np.cumsum((hist * bin_centers)[::-1]) / (weight2 + 1e-10))[::-1]

            # Inter-class variance
            variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

            # Find threshold that maximizes variance
            optimal_bin = np.argmax(variance)
            optimal_thresh_norm = bin_centers[optimal_bin]

            # Convert back to original scale
            self.auto_threshold = float((optimal_thresh_norm / 255.0 *
                            (peak_heights.max() - peak_heights.min()) +
                            peak_heights.min()))

            # Store maximum variance for quality assessment
            self.max_inter_class_variance = float(np.max(variance))

            # Store inter-class variance curve for plotting
            # Convert bin_centers back to original scale for x-axis
            thresh_values = bin_centers[:-1] / 255.0 * (peak_heights.max() - peak_heights.min()) + peak_heights.min()
            self.inter_class_variance_curve = (thresh_values, variance)

            # Calculate local minimum threshold (first local min to the right of Otsu's threshold)
            self.local_min_threshold = self._calculate_local_minimum_threshold(peak_heights, self.auto_threshold)
            if self.local_min_threshold is not None:
                print(f"[Local Min] First local minimum after Otsu: {self.local_min_threshold:.4f}")
                self.btn_set_local_min.setEnabled(True)  # Enable button if local min exists
            else:
                self.btn_set_local_min.setEnabled(False)

            # Use user's previously set threshold if available, otherwise DEFAULT TO OTSU
            if self.user_threshold is not None:
                self.current_threshold = self.user_threshold
                print(f"[Otsu] Using user threshold: {self.user_threshold:.4f} (Otsu calculated: {self.auto_threshold:.4f})")
            else:
                self.current_threshold = self.auto_threshold  # DEFAULT TO OTSU
                print(f"[Otsu] Auto-detected height threshold: {self.auto_threshold:.4f}")

            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")

            # Also populate the height threshold field
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")

        except Exception as e:
            print(f"[Otsu] Error: {e}")
            import traceback
            traceback.print_exc()


    def _calculate_local_minimum_threshold(self, peak_heights, otsu_threshold):
        """
        Calculate first local minimum of histogram to the right of Otsu's threshold.

        This provides an alternative threshold that may better separate noise from breaths
        when there's a clear valley between the two distributions.
        """
        try:
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import argrelmin

            # Create histogram with same bins as visual display
            if self.percentile_95 is not None:
                hist_range = (peak_heights.min(), self.percentile_95)
                peaks_for_hist = peak_heights[peak_heights <= self.percentile_95]
            else:
                hist_range = None
                peaks_for_hist = peak_heights

            if len(peaks_for_hist) < 10:
                print("[Local Min] Not enough peaks for calculation")
                return None

            counts, bins = np.histogram(peaks_for_hist, bins=self.num_bins, range=hist_range)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Smooth histogram to reduce noise (adaptive sigma based on bin count)
            sigma = max(1.5, self.num_bins / 50)  # Scale sigma with bin count
            smoothed_counts = gaussian_filter1d(counts.astype(float), sigma=sigma)

            # Find local minima with adaptive order
            order = max(2, int(self.num_bins / 40))  # Adaptive order based on bins
            local_mins = argrelmin(smoothed_counts, order=order)[0]

            print(f"[Local Min] Found {len(local_mins)} local minima in histogram")
            print(f"[Local Min] Otsu threshold: {otsu_threshold:.4f}")
            print(f"[Local Min] Histogram range: {bin_centers.min():.4f} - {bin_centers.max():.4f}")

            if len(local_mins) == 0:
                print("[Local Min] No local minima found in smoothed histogram")
                return None

            # Find first local minimum to the right of Otsu threshold
            candidates = []
            for min_idx in local_mins:
                min_value = bin_centers[min_idx]
                if min_value > otsu_threshold:
                    candidates.append((min_value, smoothed_counts[min_idx]))
                    print(f"[Local Min] Candidate: {min_value:.4f} (count: {smoothed_counts[min_idx]:.1f})")

            if len(candidates) == 0:
                print("[Local Min] No local minima found to the right of Otsu threshold")
                return None

            # Return the first (leftmost) local minimum
            best_threshold = candidates[0][0]
            print(f"[Local Min] Selected: {best_threshold:.4f}")
            return float(best_threshold)

        except Exception as e:
            print(f"[Local Min] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _on_percentile_changed(self, text):
        """Callback when percentile cutoff changes."""
        # Parse percentile from text (e.g., "95%" -> 95)
        if "100%" in text:
            self.percentile_cutoff = 100
        else:
            self.percentile_cutoff = int(text.replace("%", ""))

        # Recalculate percentile value
        if self.all_peak_heights is not None and len(self.all_peak_heights) > 0:
            if self.percentile_cutoff < 100:
                self.percentile_95 = np.percentile(self.all_peak_heights, self.percentile_cutoff)
            else:
                self.percentile_95 = None  # No filtering

        # Recalculate Otsu with new filtered range
        self._calculate_otsu_threshold()

        # Redraw histogram
        self._plot_histogram()

    def _on_bins_changed(self, value):
        """Callback when bin count changes."""
        self.num_bins = value
        self._plot_histogram()

    def _set_to_otsu(self):
        """Set threshold to Otsu's calculated value."""
        if self.auto_threshold is not None:
            self.current_threshold = self.auto_threshold
            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")
            self.btn_reset.setEnabled(False)  # No need to reset if already at Otsu
            self._plot_histogram()
            print(f"[Threshold] Set to Otsu: {self.current_threshold:.4f}")

    def _set_to_local_min(self):
        """Set threshold to local minimum value."""
        if self.local_min_threshold is not None:
            self.current_threshold = self.local_min_threshold
            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")
            self.btn_reset.setEnabled(True)  # Enable reset since we moved away from Otsu
            self._plot_histogram()

            # Real-time update to main plot
            if self.parent() is not None:
                try:
                    self.parent().plot_host.update_threshold_line(self.current_threshold)
                    self.parent().plot_host.canvas.draw_idle()
                except Exception:
                    pass

            print(f"[Threshold] Set to Local Min: {self.current_threshold:.4f}")

    def _toggle_y2_axis(self):
        """Toggle between peak count and inter-class variance on y2 axis."""
        if self.y2_mode == "peak_count":
            self.y2_mode = "variance"
            self.btn_toggle_y2.setText("Show: Peak Count")  # Show what you'll see if you click again
        else:
            self.y2_mode = "peak_count"
            self.btn_toggle_y2.setText("Show: Inter-Class Variance")  # Show what you'll see if you click again

        # Redraw plot with new y2 axis
        self._plot_histogram()

    def _plot_histogram(self):
        """Plot peak height histogram with draggable threshold lines."""
        try:
            self.fig.clear()
            ax1 = self.fig.add_subplot(111)

            # Apply dark theme to axes
            ax1.set_facecolor('#2d2d2d')
            ax1.spines['bottom'].set_color('#666666')
            ax1.spines['top'].set_color('#666666')
            ax1.spines['left'].set_color('#666666')
            ax1.spines['right'].set_color('#666666')
            ax1.tick_params(axis='x', colors='#d4d4d4')
            ax1.tick_params(axis='y', colors='#d4d4d4')

            if self.all_peak_heights is None:
                ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', color='#d4d4d4')
                self.canvas.draw()
                return

            peak_heights = self.all_peak_heights

            # Use percentile as upper range to exclude large artifacts
            if self.percentile_95 is not None:
                hist_range = (peak_heights.min(), self.percentile_95)
                # Filter peaks for histogram display (but keep all for threshold line positioning)
                peaks_for_hist = peak_heights[peak_heights <= self.percentile_95]
                n_excluded = len(peak_heights) - len(peaks_for_hist)
                pct_excluded = 100 * n_excluded / len(peak_heights) if len(peak_heights) > 0 else 0
            else:
                hist_range = None
                peaks_for_hist = peak_heights
                n_excluded = 0
                pct_excluded = 0

            # Plot histogram with user-controlled bin count and restricted range
            n, bins, patches = ax1.hist(peaks_for_hist, bins=self.num_bins, color='steelblue',
                                       alpha=0.7, edgecolor='#1e1e1e',
                                       label=f'Peak Height Distribution (n={len(peaks_for_hist)})',
                                       range=hist_range)

            # Color bars based on threshold (gray = below, red = above)
            if self.current_threshold is not None:
                for i, patch in enumerate(patches):
                    if bins[i] < self.current_threshold:
                        patch.set_facecolor('gray')  # Gray for "noise"
                        patch.set_alpha(0.5)
                    else:
                        patch.set_facecolor('red')  # Red for "breaths"
                        patch.set_alpha(0.5)

            ax1.set_xlabel('Peak Height', fontsize=11, color='#d4d4d4')
            ax1.set_ylabel('Frequency (count)', fontsize=11, color='#d4d4d4')

            # Title with outlier count if applicable
            if n_excluded > 0:
                title_text = f'Peak Height Distribution (Otsu\'s Method)\n{n_excluded} outlier peaks excluded (above {self.percentile_cutoff}th percentile)'
            else:
                title_text = 'Peak Height Distribution (Otsu\'s Method)'
            ax1.set_title(title_text, fontsize=12, fontweight='bold', color='#d4d4d4')
            ax1.grid(True, alpha=0.2, axis='y', color='#666666')

            # Draw FIXED Otsu reference line (thin, gray, dashed)
            if self.auto_threshold is not None:
                self.otsu_reference_line = ax1.axvline(self.auto_threshold, color='#888888',
                                                       linestyle=':', linewidth=1.0,
                                                       label=f'Otsu = {self.auto_threshold:.4f}',
                                                       zorder=1)  # Behind draggable line

            # Draw local minimum line (thin, cyan, dotted)
            if hasattr(self, 'local_min_threshold') and self.local_min_threshold is not None:
                ax1.axvline(self.local_min_threshold, color='#00cccc',
                           linestyle=':', linewidth=1.0,
                           label=f'Local Min = {self.local_min_threshold:.4f}',
                           zorder=1)

            # Draw draggable threshold line (red, thicker)
            if self.current_threshold is not None:
                self.threshold_vline = ax1.axvline(self.current_threshold, color='red',
                                                   linestyle='--', linewidth=1.5,
                                                   label=f'Current = {self.current_threshold:.4f}',
                                                   picker=5, zorder=2)  # Pickable within 5 pixels

            # Secondary y-axis: Toggle between peak count and inter-class variance
            ax2 = ax1.twinx()

            # Apply dark theme to second y-axis
            ax2.spines['right'].set_color('#666666')
            ax2.spines['left'].set_color('#666666')
            ax2.spines['top'].set_color('#666666')
            ax2.spines['bottom'].set_color('#666666')

            if self.y2_mode == "peak_count":
                # Compute peak count vs threshold (using filtered range)
                if self.percentile_95 is not None:
                    thresh_range = np.linspace(peak_heights.min(), self.percentile_95, 100)
                else:
                    thresh_range = np.linspace(peak_heights.min(), peak_heights.max(), 100)

                peak_counts = [np.sum(peak_heights >= t) for t in thresh_range]

                ax2.plot(thresh_range, peak_counts, color='#66cc66', linewidth=2, alpha=0.6,
                        label='Peaks Above Threshold')
                ax2.set_ylabel('Peaks Above Threshold', fontsize=11, color='#66cc66')
                ax2.tick_params(axis='y', labelcolor='#66cc66')

                # Mark current peak count (no horizontal line to avoid obscuring label)
                if self.current_threshold is not None:
                    current_count = np.sum(peak_heights >= self.current_threshold)
                    ax2.plot(self.current_threshold, current_count, 'ro', markersize=8)
                    ax2.text(self.current_threshold, current_count, f'  {current_count} peaks',
                            va='center', fontsize=9, color='#ff6666')

            else:  # variance mode
                # Plot inter-class variance
                if self.inter_class_variance_curve is not None:
                    var_thresh, var_values = self.inter_class_variance_curve
                    ax2.plot(var_thresh, var_values, color='#cc66cc', linewidth=2, alpha=0.6,
                            label='Inter-Class Variance')
                    ax2.set_ylabel('Inter-Class Variance', fontsize=11, color='#cc66cc')
                    ax2.tick_params(axis='y', labelcolor='#cc66cc')

                    # Mark maximum variance point
                    max_idx = np.argmax(var_values)
                    ax2.plot(var_thresh[max_idx], var_values[max_idx], color='#cc66cc',
                            marker='o', markersize=8)

                    # Mark current variance point (no horizontal line to avoid obscuring labels)
                    if self.current_threshold is not None:
                        # Find variance at current threshold
                        closest_idx = np.argmin(np.abs(var_thresh - self.current_threshold))
                        current_var = var_values[closest_idx]
                        ax2.plot(self.current_threshold, current_var, 'ro', markersize=8)
                        ax2.text(self.current_threshold, current_var, f'  Var={current_var:.0f}',
                                va='center', fontsize=9, color='#ff6666')

            # Combine legends with dark theme styling
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
                               fontsize=9, facecolor='#2d2d2d', edgecolor='#666666')
            legend.get_frame().set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color('#d4d4d4')

            # Use tight_layout with extra padding to ensure labels are visible
            # This gets called on every redraw, so labels become visible when dragging
            self.fig.tight_layout(rect=[0, 0.03, 1, 0.97])

            # Draw the canvas
            self.canvas.draw()

        except Exception as e:
            print(f"[Plot] Error: {e}")
            import traceback
            traceback.print_exc()

    def _on_mouse_press(self, event):
        """Handle mouse click on threshold lines."""
        if event.inaxes is None:
            return

        # Check if click is near vertical threshold line
        if event.button == 1:  # Left click
            if self.threshold_vline is not None:
                contains, _ = self.threshold_vline.contains(event)
                if contains:
                    self.is_dragging = True
                    self.canvas.setCursor(Qt.CursorShape.SizeHorCursor)
                    return

    def _on_mouse_move(self, event):
        """Handle mouse drag to adjust threshold."""
        if not self.is_dragging or event.inaxes is None:
            return

        # Update threshold to mouse x position
        new_threshold = event.xdata
        if new_threshold is not None and new_threshold > 0:
            self.current_threshold = float(new_threshold)
            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")

            # Enable reset button if threshold changed
            if abs(self.current_threshold - self.auto_threshold) > 0.001:
                self.btn_reset.setEnabled(True)
            else:
                self.btn_reset.setEnabled(False)

            # Update height threshold field to match dragged threshold
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")

            # Real-time update to main plot
            if self.parent() is not None:
                try:
                    self.parent().plot_host.update_threshold_line(self.current_threshold)
                    self.parent().plot_host.canvas.draw_idle()
                except Exception as e:
                    pass  # Silently fail if main plot update doesn't work

            # Redraw plot
            self._plot_histogram()

    def _on_mouse_release(self, event):
        """Handle mouse release after drag."""
        if self.is_dragging:
            self.is_dragging = False
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

    def _reset_threshold(self):
        """Reset threshold to auto-detected value."""
        self.current_threshold = self.auto_threshold
        self.lbl_threshold.setText(f"{self.auto_threshold:.4f}")
        self.le_threshold_height.setText(f"{self.auto_threshold:.4f}")  # Reset height threshold too
        self.btn_reset.setEnabled(False)
        self._plot_histogram()

    def get_values(self):
        """Get the current parameter values."""
        try:
            min_dist = float(self.le_min_dist.text())
        except ValueError:
            min_dist = self.current_min_dist

        # Get optional height threshold
        try:
            height_thresh = float(self.le_threshold_height.text()) if self.le_threshold_height.text().strip() else None
        except ValueError:
            height_thresh = None

        return {
            'prominence': self.current_threshold if self.current_threshold else self.auto_threshold,
            'min_dist': min_dist,
            'height_threshold': height_thresh,  # Now always populated from Otsu's method
            'percentile_95': self.percentile_95,  # Pass to main window for consistent histogram range
            'all_peak_heights': self.all_peak_heights,  # Pass to main window for histogram display
            'histogram_num_bins': self.num_bins,  # Pass bin count for matching histograms
            'percentile_cutoff': self.percentile_cutoff  # Pass cutoff for remembering setting
        }
