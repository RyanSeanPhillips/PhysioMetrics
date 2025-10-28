"""
Peak Detection Options Dialog

This dialog provides advanced peak detection parameters and experimental
auto-threshold detection methods for testing and comparison.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QLabel, QGroupBox, QDialogButtonBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class PeakDetectionDialog(QDialog):
    """Advanced peak detection options and auto-threshold testing dialog."""

    def __init__(self, parent=None, y_data=None, sr_hz=None, current_thresh=None, current_prom=None, current_min_dist=None):
        super().__init__(parent)
        self.setWindowTitle("Peak Detection Options")
        self.resize(900, 700)

        self.y_data = y_data
        self.sr_hz = sr_hz or 1000.0
        self.current_thresh = current_thresh or 0.1
        self.current_prom = current_prom or 0.1
        self.current_min_dist = current_min_dist or 0.05

        # Auto-calculated thresholds
        self.auto_thresh_elbow = None
        self.auto_thresh_snr = None
        self.auto_thresh_otsu = None

        # Cached peak detection results (calculate once, reuse for all methods)
        self.all_peaks = None
        self.all_prominences = None
        self.peak_count_vs_threshold = None  # Pre-computed curve

        self._setup_ui()

        # Calculate auto-thresholds if data provided
        if y_data is not None and len(y_data) > 0:
            self._detect_all_peaks_once()
            self._calculate_auto_thresholds()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)

        # Basic parameters section
        params_group = QGroupBox("Basic Parameters")
        params_layout = QFormLayout()

        self.le_thresh = QLineEdit(str(self.current_thresh))
        self.le_thresh.setToolTip("Minimum amplitude threshold for peak detection")
        params_layout.addRow("Threshold:", self.le_thresh)

        self.le_min_dist = QLineEdit(str(self.current_min_dist))
        self.le_min_dist.setToolTip("Minimum time between peaks (seconds)")
        params_layout.addRow("Min Peak Distance (s):", self.le_min_dist)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Auto-threshold methods tabs
        self.tabs = QTabWidget()

        # Method A: Elbow Detection
        self.tab_elbow = QWidget()
        self._setup_elbow_tab(self.tab_elbow)
        self.tabs.addTab(self.tab_elbow, "Method A: Elbow Detection")

        # Method B: Peak Count vs Threshold
        self.tab_snr = QWidget()
        self._setup_snr_tab(self.tab_snr)
        self.tabs.addTab(self.tab_snr, "Method B: SNR Optimization")

        # Method C: Otsu's Method
        self.tab_otsu = QWidget()
        self._setup_otsu_tab(self.tab_otsu)
        self.tabs.addTab(self.tab_otsu, "Method C: Otsu's Method")

        layout.addWidget(self.tabs)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _setup_elbow_tab(self, tab):
        """Setup elbow detection tab."""
        layout = QVBoxLayout(tab)

        # Description
        desc = QLabel(
            "<b>Elbow Detection Method</b><br>"
            "Plots number of detected peaks vs prominence threshold and finds the 'elbow' - "
            "the point of maximum curvature where increasing threshold yields diminishing "
            "reduction in peak count. Uses perpendicular distance from a reference line to "
            "identify this optimal threshold."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Result display
        result_layout = QHBoxLayout()
        self.lbl_elbow_result = QLabel("Calculated threshold: (calculating...)")
        self.lbl_elbow_result.setStyleSheet("font-weight: bold; color: #2a7fff;")
        result_layout.addWidget(self.lbl_elbow_result)

        result_layout.addStretch()

        self.btn_use_elbow = QPushButton("Use This Value")
        self.btn_use_elbow.clicked.connect(lambda: self._use_threshold(self.auto_thresh_elbow))
        self.btn_use_elbow.setEnabled(False)
        result_layout.addWidget(self.btn_use_elbow)

        layout.addLayout(result_layout)

        # Visualization
        self.fig_elbow = Figure(figsize=(8, 4), dpi=100)
        self.canvas_elbow = FigureCanvas(self.fig_elbow)
        layout.addWidget(self.canvas_elbow)

    def _setup_snr_tab(self, tab):
        """Setup SNR optimization tab."""
        layout = QVBoxLayout(tab)

        # Description
        desc = QLabel(
            "<b>Signal-to-Noise Ratio Optimization</b><br>"
            "Analyzes the number of detected peaks as a function of threshold. "
            "The optimal threshold is where the rate of peak detection stabilizes, "
            "indicating we've captured true breaths while filtering out noise."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Result display
        result_layout = QHBoxLayout()
        self.lbl_snr_result = QLabel("Calculated threshold: (calculating...)")
        self.lbl_snr_result.setStyleSheet("font-weight: bold; color: #2a7fff;")
        result_layout.addWidget(self.lbl_snr_result)

        result_layout.addStretch()

        self.btn_use_snr = QPushButton("Use This Value")
        self.btn_use_snr.clicked.connect(lambda: self._use_threshold(self.auto_thresh_snr))
        self.btn_use_snr.setEnabled(False)
        result_layout.addWidget(self.btn_use_snr)

        layout.addLayout(result_layout)

        # Visualization
        self.fig_snr = Figure(figsize=(8, 4), dpi=100)
        self.canvas_snr = FigureCanvas(self.fig_snr)
        layout.addWidget(self.canvas_snr)

    def _setup_otsu_tab(self, tab):
        """Setup Otsu's method tab."""
        layout = QVBoxLayout(tab)

        # Description
        desc = QLabel(
            "<b>Otsu's Method (Histogram-Based)</b><br>"
            "Calculates optimal threshold by maximizing inter-class variance "
            "in the amplitude histogram. Originally designed for image segmentation, "
            "adapted here for separating true peaks from noise."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Result display
        result_layout = QHBoxLayout()
        self.lbl_otsu_result = QLabel("Calculated threshold: (calculating...)")
        self.lbl_otsu_result.setStyleSheet("font-weight: bold; color: #2a7fff;")
        result_layout.addWidget(self.lbl_otsu_result)

        result_layout.addStretch()

        self.btn_use_otsu = QPushButton("Use This Value")
        self.btn_use_otsu.clicked.connect(lambda: self._use_threshold(self.auto_thresh_otsu))
        self.btn_use_otsu.setEnabled(False)
        result_layout.addWidget(self.btn_use_otsu)

        layout.addLayout(result_layout)

        # Visualization
        self.fig_otsu = Figure(figsize=(8, 4), dpi=100)
        self.canvas_otsu = FigureCanvas(self.fig_otsu)
        layout.addWidget(self.canvas_otsu)

    def _detect_all_peaks_once(self):
        """
        Run peak detection ONCE with minimal threshold to find all peaks.
        Cache results for reuse by all three methods. This is much faster than
        running find_peaks() 200+ times.
        """
        print("[Peak Dialog] Running peak detection once with minimal threshold...")
        import time
        t_start = time.time()

        try:
            y = self.y_data
            min_dist_samples = int(0.01 * self.sr_hz)

            # Find ALL peaks with very low prominence threshold
            peaks, props = find_peaks(y, prominence=0.001, distance=min_dist_samples)

            self.all_peaks = peaks
            self.all_prominences = props['prominences']

            # Pre-compute peak count vs threshold curve (for Elbow and SNR methods)
            y_range = np.max(y) - np.min(y)
            thresholds = np.linspace(0.001 * y_range, 0.5 * y_range, 100)
            peak_counts = []

            for thresh in thresholds:
                # Just count peaks with prominence >= thresh (no re-detection!)
                count = np.sum(self.all_prominences >= thresh)
                peak_counts.append(count)

            self.peak_count_vs_threshold = (thresholds, np.array(peak_counts))

            t_elapsed = time.time() - t_start
            print(f"[Peak Dialog] Found {len(self.all_peaks)} peaks in {t_elapsed:.2f}s (will reuse for all methods)")

        except Exception as e:
            print(f"[Peak Dialog] Error detecting peaks: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_auto_thresholds(self):
        """Calculate all three auto-threshold methods."""
        if self.y_data is None or len(self.y_data) == 0:
            return

        # Method A: Elbow Detection
        self.auto_thresh_elbow = self._method_elbow_detection()
        if self.auto_thresh_elbow is not None:
            self.lbl_elbow_result.setText(f"Calculated threshold: {self.auto_thresh_elbow:.4f}")
            self.btn_use_elbow.setEnabled(True)
            self._plot_elbow()

        # Method B: SNR Optimization
        self.auto_thresh_snr = self._method_snr_optimization()
        if self.auto_thresh_snr is not None:
            self.lbl_snr_result.setText(f"Calculated threshold: {self.auto_thresh_snr:.4f}")
            self.btn_use_snr.setEnabled(True)
            self._plot_snr()

        # Method C: Otsu's Method
        self.auto_thresh_otsu = self._method_otsu()
        if self.auto_thresh_otsu is not None:
            self.lbl_otsu_result.setText(f"Calculated threshold: {self.auto_thresh_otsu:.4f}")
            self.btn_use_otsu.setEnabled(True)
            self._plot_otsu()

    def _method_elbow_detection(self):
        """
        Method A: Elbow Detection

        Find the elbow point in the prominence vs peak count curve.
        Algorithm:
        1. Use pre-computed peak count vs threshold curve
        2. Find elbow point using perpendicular distance method
        """
        try:
            if self.peak_count_vs_threshold is None:
                return None

            # Use pre-computed curve (already cached!)
            thresholds, peak_counts = self.peak_count_vs_threshold

            if len(peak_counts) < 3:
                return None

            # Find elbow using perpendicular distance method
            # Line from first point (lowest threshold) to last point (highest threshold)
            x = thresholds
            y_counts = peak_counts

            p1 = np.array([x[0], y_counts[0]])
            p2 = np.array([x[-1], y_counts[-1]])

            # Calculate perpendicular distances from each point to the line
            distances = np.zeros(len(x))
            for i in range(len(x)):
                point = np.array([x[i], y_counts[i]])
                # Distance from point to line defined by p1 and p2
                distances[i] = np.abs(np.cross(p2 - p1, p1 - point)) / np.linalg.norm(p2 - p1)

            # Elbow is at maximum distance from the line
            elbow_idx = np.argmax(distances)
            elbow_threshold = thresholds[elbow_idx]

            print(f"[Elbow] Found {peak_counts[0]} total peaks, elbow at prominence={elbow_threshold:.4f} ({peak_counts[elbow_idx]} peaks)")

            return float(elbow_threshold)

        except Exception as e:
            print(f"[Elbow] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _method_snr_optimization(self):
        """
        Method B: SNR Optimization

        Analyze peak count vs threshold curve to find optimal SNR.
        Algorithm:
        1. Use pre-computed peak count vs threshold curve
        2. Find threshold where count stabilizes (derivative minimum)
        """
        try:
            if self.peak_count_vs_threshold is None:
                return None

            # Use pre-computed curve (already cached!)
            thresholds, peak_counts = self.peak_count_vs_threshold

            if len(peak_counts) < 3:
                return None

            # Find where derivative stabilizes (elbow in peak count curve)
            # Use second derivative to find inflection point
            first_deriv = np.diff(peak_counts)
            second_deriv = np.diff(first_deriv)

            # Smooth derivatives to avoid noise
            if len(second_deriv) > 5:
                second_deriv_smooth = gaussian_filter1d(second_deriv.astype(float), sigma=2)

                # Find point where second derivative crosses zero (inflection)
                # Or use maximum of second derivative (steepest decline)
                optimal_idx = np.argmax(np.abs(second_deriv_smooth))
                optimal_idx = min(optimal_idx + 1, len(thresholds) - 1)  # Account for diff

                optimal_thresh = float(thresholds[optimal_idx])
                print(f"[SNR] Found {peak_counts[0]} total peaks, optimal at prominence={optimal_thresh:.4f} ({peak_counts[optimal_idx]} peaks)")
                return optimal_thresh
            else:
                # Fallback: use middle of range
                median_thresh = float(np.median(thresholds))
                print(f"[SNR] Insufficient data for derivative analysis, using median: {median_thresh:.4f}")
                return median_thresh

        except Exception as e:
            print(f"[SNR] Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _method_otsu(self):
        """
        Method C: Otsu's Method

        Histogram-based threshold selection from image processing.
        Algorithm:
        1. Use pre-computed peak prominences
        2. Compute prominence histogram
        3. Find threshold that maximizes inter-class variance
        """
        try:
            if self.all_prominences is None or len(self.all_prominences) < 10:
                return None

            # Use cached prominences
            prominences = self.all_prominences

            # Normalize prominences to [0, 255] for Otsu
            prom_norm = ((prominences - prominences.min()) /
                        (prominences.max() - prominences.min()) * 255).astype(np.uint8)

            # Compute histogram
            hist, bin_edges = np.histogram(prom_norm, bins=256, range=(0, 256))
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
            optimal_thresh = (optimal_thresh_norm / 255.0 *
                            (prominences.max() - prominences.min()) +
                            prominences.min())

            return float(optimal_thresh)

        except Exception as e:
            print(f"[Otsu] Error: {e}")
            return None

    def _plot_elbow(self):
        """Plot elbow detection visualization."""
        try:
            self.fig_elbow.clear()
            ax = self.fig_elbow.add_subplot(111)

            if self.peak_count_vs_threshold is None:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                self.canvas_elbow.draw()
                return

            # Use cached data (no re-computation!)
            thresholds, peak_counts = self.peak_count_vs_threshold

            # Plot prominence vs peak count
            ax.plot(thresholds, peak_counts, 'b-', linewidth=2, label='Peak Count vs Prominence')

            # Draw the line from first to last point
            ax.plot([thresholds[0], thresholds[-1]],
                   [peak_counts[0], peak_counts[-1]],
                   'g--', alpha=0.3, linewidth=1, label='Reference Line')

            # Mark elbow point
            if self.auto_thresh_elbow is not None:
                elbow_idx = np.argmin(np.abs(thresholds - self.auto_thresh_elbow))
                ax.plot(self.auto_thresh_elbow, peak_counts[elbow_idx], 'ro', markersize=10,
                       label=f'Elbow (thresh={self.auto_thresh_elbow:.4f}, {peak_counts[elbow_idx]} peaks)')
                ax.axvline(self.auto_thresh_elbow, color='r', linestyle='--', alpha=0.5)

            ax.set_xlabel('Prominence Threshold')
            ax.set_ylabel('Number of Detected Peaks')
            ax.set_title('Elbow Detection: Prominence vs Peak Count')
            ax.legend()
            ax.grid(True, alpha=0.3)

            self.canvas_elbow.draw()

        except Exception as e:
            print(f"[Plot Elbow] Error: {e}")
            import traceback
            traceback.print_exc()

    def _plot_snr(self):
        """Plot SNR optimization visualization."""
        try:
            self.fig_snr.clear()
            ax = self.fig_snr.add_subplot(111)

            if self.peak_count_vs_threshold is None:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                self.canvas_snr.draw()
                return

            # Use cached data (no re-computation!)
            thresholds, peak_counts = self.peak_count_vs_threshold

            # Plot peak count vs threshold
            ax.plot(thresholds, peak_counts, 'b-', linewidth=2, label='Peak Count')

            # Mark optimal point
            if self.auto_thresh_snr is not None:
                opt_idx = np.argmin(np.abs(thresholds - self.auto_thresh_snr))
                ax.plot(self.auto_thresh_snr, peak_counts[opt_idx], 'ro', markersize=10,
                       label=f'Optimal (thresh={self.auto_thresh_snr:.4f})')
                ax.axvline(self.auto_thresh_snr, color='r', linestyle='--', alpha=0.5)

            ax.set_xlabel('Threshold (Prominence)')
            ax.set_ylabel('Number of Detected Peaks')
            ax.set_title('Peak Count vs Threshold (SNR Optimization)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            self.canvas_snr.draw()

        except Exception as e:
            print(f"[Plot SNR] Error: {e}")

    def _plot_otsu(self):
        """Plot Otsu's method visualization."""
        try:
            self.fig_otsu.clear()
            ax = self.fig_otsu.add_subplot(111)

            if self.all_prominences is None:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                self.canvas_otsu.draw()
                return

            # Use cached prominences (no re-computation!)
            prominences = self.all_prominences

            # Plot histogram
            ax.hist(prominences, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

            # Mark Otsu threshold
            if self.auto_thresh_otsu is not None:
                ax.axvline(self.auto_thresh_otsu, color='r', linestyle='--', linewidth=2,
                          label=f'Otsu Threshold = {self.auto_thresh_otsu:.4f}')

            ax.set_xlabel('Prominence')
            ax.set_ylabel('Frequency')
            ax.set_title('Prominence Histogram (Otsu\'s Method)')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            self.canvas_otsu.draw()

        except Exception as e:
            print(f"[Plot Otsu] Error: {e}")

    def _use_threshold(self, value):
        """Apply selected auto-threshold to the threshold field."""
        if value is not None:
            self.le_thresh.setText(f"{value:.4f}")

    def get_values(self):
        """Get the current parameter values."""
        try:
            thresh = float(self.le_thresh.text())
        except ValueError:
            thresh = self.current_thresh

        try:
            min_dist = float(self.le_min_dist.text())
        except ValueError:
            min_dist = self.current_min_dist

        return {
            'threshold': thresh,
            'min_dist': min_dist
        }
