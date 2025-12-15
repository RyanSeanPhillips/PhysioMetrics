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

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLineEdit, QPushButton,
    QLabel, QGroupBox, QDialogButtonBox, QWidget
)
from PyQt6.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from scipy.signal import find_peaks


class ProminenceThresholdDialog(QDialog):
    """Interactive prominence threshold detection using Otsu's method."""

    def __init__(self, parent=None, y_data=None, sr_hz=None, current_prom=None, current_min_dist=None, current_height_threshold=None,
                 percentile_cutoff=99, num_bins=200, skip_detection=False, cached_peaks=None, cached_peak_heights=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect Prominence Threshold")
        self.resize(1200, 700)  # 20% wider (1000 -> 1200)

        # Store parent for later access
        self.parent_ref = parent

        self.y_data = y_data
        self.sr_hz = sr_hz or 1000.0
        self.current_prom = current_prom or 0.1
        self.current_min_dist = current_min_dist or 0.05
        self.user_threshold = current_height_threshold  # Previously set threshold

        # Cached peak detection - use provided data if available
        self.all_peaks = cached_peaks
        self.all_peak_heights = cached_peak_heights

        # Auto-calculated threshold
        self.auto_threshold = None
        self.current_threshold = None  # User-adjusted value
        self.local_min_threshold = None  # Local minimum threshold (valley location)
        self.valley_depth_score = 0.0  # Signal quality metric (0-1, higher = better)
        self.inter_class_variance_curve = None  # For plotting
        self.exp_gauss_params = None  # Exponential + Gaussian fit parameters

        # Separation metric for display
        self.separation_metric = 1.0

        # Draggable line (only vertical, no horizontal to avoid obscuring labels)
        self.threshold_vline = None
        self.otsu_reference_line = None  # Fixed line showing Otsu's calculated threshold
        self.is_dragging = False

        # Store histogram patches and bins for fast updating
        self.histogram_patches = None
        self.histogram_bins = None

        # Y2 axis mode toggle
        self.y2_mode = "peak_count"  # or "variance"

        # Histogram controls - use passed values or defaults
        self.percentile_cutoff = percentile_cutoff
        self.num_bins = num_bins

        # Apply dark theme
        self._apply_dark_theme()
        self._enable_dark_title_bar()

        self._setup_ui()

        # Detect peaks and calculate threshold (unless skip_detection=True for embedded display-only mode)
        # Run if we have y_data (for detection) OR cached_peak_heights (skip detection, just plot)
        has_data_to_plot = (y_data is not None and len(y_data) > 0) or (self.all_peak_heights is not None and len(self.all_peak_heights) > 0)

        if not skip_detection and has_data_to_plot:
            if self.all_peak_heights is None:
                # No cached data - need to detect peaks (requires y_data)
                if y_data is not None and len(y_data) > 0:
                    self._detect_all_peaks()
                else:
                    print("[Prominence Dialog] WARNING: No y_data and no cached peaks - cannot plot histogram")
                    return
            else:
                # Cached data provided - calculate percentile for histogram filtering
                import time
                t_start = time.time()
                print(f"[Prominence Dialog] Using {len(self.all_peak_heights)} cached peaks (skipping detection)")
                if len(self.all_peak_heights) > 0:
                    self.percentile_95 = np.percentile(self.all_peak_heights, self.percentile_cutoff)
                else:
                    self.percentile_95 = 1.0
                print(f"[Prominence Dialog] Cached data processed in {(time.time() - t_start)*1000:.1f}ms")
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
            QComboBox {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
            }
            QComboBox:focus {
                border: 1px solid #2a7fff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                color: #d4d4d4;
                selection-background-color: #3e3e42;
            }
            QSpinBox {
                background-color: #2d2d2d;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
            }
            QSpinBox:focus {
                border: 1px solid #2a7fff;
            }
            QDialogButtonBox QPushButton {
                min-width: 80px;
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

    def _setup_ui(self):
        """Build the dialog UI."""
        main_layout = QVBoxLayout(self)

        # Title and description
        title = QLabel("<h2>Auto-Threshold Detection</h2>")
        main_layout.addWidget(title)

        desc = QLabel(
            "Automatically detects optimal threshold by analyzing peak height distribution. "
            "Drag the red line to adjust the threshold interactively."
        )
        desc.setWordWrap(True)
        main_layout.addWidget(desc)

        # Create horizontal layout for left sidebar + right plot
        content_layout = QHBoxLayout()

        # LEFT SIDEBAR - Controls (350px wide)
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Threshold display group
        threshold_group = QGroupBox("Detected Threshold")
        threshold_layout = QVBoxLayout()

        # Threshold value row
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Optimal Threshold:"))
        self.lbl_threshold = QLabel("Calculating...")
        self.lbl_threshold.setStyleSheet("font-weight: bold; font-size: 14pt; color: #2a7fff;")
        thresh_row.addWidget(self.lbl_threshold)
        thresh_row.addStretch()
        threshold_layout.addLayout(thresh_row)

        # Peak count row
        count_row = QHBoxLayout()
        self.lbl_peak_count = QLabel("Peaks detected: 0")
        count_row.addWidget(self.lbl_peak_count)
        count_row.addStretch()
        threshold_layout.addLayout(count_row)

        # Signal quality indicator (HIDDEN)
        self.lbl_signal_quality = QLabel("Signal Quality: Calculating...")
        self.lbl_signal_quality.setStyleSheet("font-size: 10pt;")
        self.lbl_signal_quality.setVisible(False)  # Hidden - not working well yet
        threshold_layout.addWidget(self.lbl_signal_quality)

        # Button row - single Reset button to snap to valley threshold
        btn_row = QHBoxLayout()
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset threshold to valley (local minimum)")
        self.btn_reset.clicked.connect(self._reset_to_valley)
        self.btn_reset.setEnabled(False)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch()
        threshold_layout.addLayout(btn_row)

        threshold_group.setLayout(threshold_layout)
        left_layout.addWidget(threshold_group)

        # Advanced parameters
        params_group = QGroupBox("Advanced Parameters")
        params_layout = QFormLayout()

        self.le_min_dist = QLineEdit(str(self.current_min_dist))
        self.le_min_dist.setToolTip("Minimum time between peaks (seconds)")
        self.le_min_dist.setMaximumWidth(100)
        params_layout.addRow("Min Peak Distance (s):", self.le_min_dist)

        self.le_threshold_height = QLineEdit("")
        self.le_threshold_height.setToolTip("Absolute height threshold")
        self.le_threshold_height.setMaximumWidth(100)
        params_layout.addRow("Height Threshold:", self.le_threshold_height)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Classifier Selection (if state is available)
        if hasattr(self.parent_ref, 'state') and self.parent_ref.state is not None:
            classifier_group = QGroupBox("Breath Classifier")
            classifier_layout = QVBoxLayout()

            classifier_row = QHBoxLayout()
            classifier_row.addWidget(QLabel("Algorithm:"))

            from PyQt6.QtWidgets import QComboBox
            self.classifier_combo = QComboBox()
            self.classifier_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])

            # Set current selection based on state
            if self.parent_ref.state.active_classifier == 'threshold':
                self.classifier_combo.setCurrentText("Threshold")
            elif self.parent_ref.state.active_classifier == 'xgboost':
                self.classifier_combo.setCurrentText("XGBoost")
            elif self.parent_ref.state.active_classifier == 'rf':
                self.classifier_combo.setCurrentText("Random Forest")
            elif self.parent_ref.state.active_classifier == 'mlp':
                self.classifier_combo.setCurrentText("MLP")

            self.classifier_combo.setToolTip("Select which classifier to use for breath detection")
            self.classifier_combo.currentTextChanged.connect(lambda text: self._on_classifier_changed(text, self.parent_ref))
            self.classifier_combo.setMaximumWidth(150)
            classifier_row.addWidget(self.classifier_combo)
            classifier_row.addStretch()
            classifier_layout.addLayout(classifier_row)

            # Status label for ML models
            self.lbl_classifier_status = QLabel()
            self.lbl_classifier_status.setWordWrap(True)
            self.lbl_classifier_status.setStyleSheet("font-size: 9pt; color: #888;")
            self._update_classifier_status(self.parent_ref)
            classifier_layout.addWidget(self.lbl_classifier_status)

            classifier_group.setLayout(classifier_layout)
            left_layout.addWidget(classifier_group)
        else:
            self.classifier_combo = None

        # Histogram controls
        histogram_controls_group = QGroupBox("Histogram Controls")
        histogram_controls_layout = QVBoxLayout()

        # Percentile cutoff control
        from PyQt6.QtWidgets import QSpinBox, QComboBox
        perc_row = QHBoxLayout()
        perc_row.addWidget(QLabel("Outlier Cutoff:"))
        self.percentile_combo = QComboBox()
        self.percentile_combo.addItems(["90%", "95%", "99%", "100% (None)"])
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
        self.percentile_combo.setMaximumWidth(130)
        perc_row.addWidget(self.percentile_combo)
        perc_row.addStretch()
        histogram_controls_layout.addLayout(perc_row)

        # Bin count control
        bins_row = QHBoxLayout()
        bins_row.addWidget(QLabel("Bins:"))
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(20, 500)
        self.bins_spin.setValue(self.num_bins)
        self.bins_spin.setSingleStep(10)
        self.bins_spin.setToolTip("Number of bins in histogram")
        self.bins_spin.valueChanged.connect(self._on_bins_changed)
        self.bins_spin.setMaximumWidth(80)
        bins_row.addWidget(self.bins_spin)
        bins_row.addStretch()
        histogram_controls_layout.addLayout(bins_row)

        # Y2 toggle - COMMENTED OUT (not needed)
        # y2_row = QHBoxLayout()
        # self.btn_toggle_y2 = QPushButton("Show: Inter-Class Variance")
        # self.btn_toggle_y2.setToolTip("Toggle between Peak Count and Inter-Class Variance")
        # self.btn_toggle_y2.clicked.connect(self._toggle_y2_axis)
        # y2_row.addWidget(self.btn_toggle_y2)
        # y2_row.addStretch()
        # histogram_controls_layout.addLayout(y2_row)

        histogram_controls_group.setLayout(histogram_controls_layout)
        left_layout.addWidget(histogram_controls_group)

        left_layout.addStretch()
        content_layout.addWidget(left_panel)

        # RIGHT PANEL - Plot area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")

        # Add navigation toolbar BELOW canvas with blue styling
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setObjectName("ProminenceDialogToolbar")
        self.toolbar.setIconSize(QSize(15, 15))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setStyleSheet("""
        QToolBar#ProminenceDialogToolbar { background: transparent; border: none; padding: 0px; }
        QToolBar#ProminenceDialogToolbar::separator { background: transparent; width: 0px; height: 0px; }
        QToolBar#ProminenceDialogToolbar::handle { image: none; width: 0px; height: 0px; }
        QToolBar#ProminenceDialogToolbar QToolButton {
            background: #434b5d; color: #eef2f8;
            border: 1px solid #5a6580; border-radius: 8px;
            padding: 5px 8px; margin: 2px;
        }
        QToolBar#ProminenceDialogToolbar QToolButton:hover { background: #515c72; border-color: #6a7694; }
        QToolBar#ProminenceDialogToolbar QToolButton:pressed { background: #5f6d88; border-color: #7886a6; }
        QToolBar#ProminenceDialogToolbar QToolButton:checked {
            background: #4A90E2; color: white; border-color: #357ABD;
        }
        QToolBar#ProminenceDialogToolbar QToolButton:disabled {
            background: #353b4a; border-color: #444d60; color: #8691a8;
        }""")

        right_layout.addWidget(self.canvas)
        right_layout.addWidget(self.toolbar)

        # Connect drag events
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

        content_layout.addWidget(right_panel, stretch=1)

        main_layout.addLayout(content_layout)

        # Dialog buttons at bottom
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

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

            # Calculate valley threshold (independent of Otsu) and quality metric
            self.local_min_threshold, self.valley_depth_score = self._calculate_local_minimum_threshold(peak_heights)
            if self.local_min_threshold is not None:
                print(f"[Valley] Natural valley threshold: {self.local_min_threshold:.4f}")
                print(f"[Valley] Signal quality score: {self.valley_depth_score:.3f}")
            else:
                self.valley_depth_score = 0.0

            # Update signal quality display
            self._update_quality_label()

            # Use user's previously set threshold if available, otherwise DEFAULT TO LOCAL MINIMUM
            if self.user_threshold is not None:
                self.current_threshold = self.user_threshold
                print(f"[Threshold] Using user threshold: {self.user_threshold:.4f} (Otsu: {self.auto_threshold:.4f}, Valley: {self.local_min_threshold:.4f if self.local_min_threshold else 'N/A'})")
            elif self.local_min_threshold is not None:
                self.current_threshold = self.local_min_threshold  # DEFAULT TO LOCAL MINIMUM
                print(f"[Threshold] Using valley threshold: {self.local_min_threshold:.4f} (Otsu: {self.auto_threshold:.4f})")
            else:
                self.current_threshold = self.auto_threshold  # FALLBACK TO OTSU
                print(f"[Threshold] Using Otsu threshold: {self.auto_threshold:.4f} (no valley found)")

            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")

            # Also populate the height threshold field
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")

        except Exception as e:
            print(f"[Otsu] Error: {e}")
            import traceback
            traceback.print_exc()


    def _calculate_local_minimum_threshold(self, peak_heights, otsu_threshold=None):
        """
        Calculate valley threshold using exponential + Gaussian mixture model.

        Tries TWO models in order:
        1. Exp + 2 Gaussians (for eupnea + sniffing)
        2. Exp + 1 Gaussian (fallback for simpler distributions)

        Returns (valley_location, valley_depth_score) where:
        - valley_location: Minimum between 0 and first Gaussian peak
        - valley_depth_score: Quality metric based on fit quality and separation
        """
        try:
            from scipy.optimize import curve_fit

            # Create histogram with same bins as visual display
            if self.percentile_95 is not None:
                hist_range = (peak_heights.min(), self.percentile_95)
                peaks_for_hist = peak_heights[peak_heights <= self.percentile_95]
            else:
                hist_range = None
                peaks_for_hist = peak_heights

            if len(peaks_for_hist) < 10:
                print("[Valley] Not enough peaks for calculation")
                return None, 0.0

            counts, bins = np.histogram(peaks_for_hist, bins=self.num_bins, range=hist_range)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

            # Normalize histogram to PDF (density, not counts)
            total_area = np.sum(counts) * bin_width
            density = counts / total_area if total_area > 0 else counts

            # Try 2-Gaussian model first (eupnea + sniffing)
            result = self._fit_exp_2gauss(bin_centers, density, counts)

            if result is not None:
                print("[Valley Fit] Using 2-Gaussian model (eupnea + sniffing)")
                return result

            # Fallback to 1-Gaussian model
            print("[Valley Fit] Falling back to 1-Gaussian model")
            result = self._fit_exp_1gauss(bin_centers, density, counts)

            return result if result is not None else (None, 0.0)

        except Exception as e:
            print(f"[Valley] Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0

    def _fit_exp_2gauss(self, bin_centers, density, counts):
        """Fit exponential + 2 Gaussians (noise + eupnea + sniffing)."""
        try:
            from scipy.optimize import curve_fit

            def exp_2gauss_model(x, lambda_exp, mu1, sigma1, mu2, sigma2, w_exp, w_g1):
                """
                Exponential + 2 Gaussians
                w_exp: weight of exponential
                w_g1: weight of first Gaussian
                Remaining weight (1 - w_exp - w_g1) goes to second Gaussian
                """
                exp_comp = lambda_exp * np.exp(-lambda_exp * x)
                gauss1 = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
                gauss2 = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

                w_g2 = max(0, 1 - w_exp - w_g1)  # Ensure non-negative
                return w_exp * exp_comp + w_g1 * gauss1 + w_g2 * gauss2

            # Initial guesses
            left_third = bin_centers[bin_centers < np.percentile(bin_centers, 33)]
            lambda_init = 1.0 / np.mean(left_third) if len(left_third) > 0 else 1.0

            # Two Gaussian peaks: one at ~40th percentile (eupnea), one at ~70th (sniffing)
            mu1_init = np.percentile(bin_centers, 40)
            mu2_init = np.percentile(bin_centers, 70)
            sigma1_init = (mu2_init - mu1_init) / 4  # Rough estimate
            sigma2_init = sigma1_init

            w_exp_init = 0.25
            w_g1_init = 0.40  # Eupnea typically more common than sniffing

            print(f"[2-Gauss] Initial: λ={lambda_init:.3f}, μ1={mu1_init:.3f}, μ2={mu2_init:.3f}")

            # Bounds
            bounds = ([0.001, bin_centers.min(), 0.001, mu1_init, 0.001, 0.0, 0.0],
                     [100.0, mu2_init, bin_centers.max(), bin_centers.max(), bin_centers.max(), 0.8, 0.8])

            popt, pcov = curve_fit(
                exp_2gauss_model,
                bin_centers,
                density,
                p0=[lambda_init, mu1_init, sigma1_init, mu2_init, sigma2_init, w_exp_init, w_g1_init],
                bounds=bounds,
                maxfev=10000
            )

            lambda_fit, mu1_fit, sigma1_fit, mu2_fit, sigma2_fit, w_exp_fit, w_g1_fit = popt
            w_g2_fit = max(0, 1 - w_exp_fit - w_g1_fit)

            # Calculate R²
            fitted = exp_2gauss_model(bin_centers, *popt)
            ss_res = np.sum((density - fitted) ** 2)
            ss_tot = np.sum((density - np.mean(density)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"[2-Gauss] Fitted: λ={lambda_fit:.3f}, μ1={mu1_fit:.3f}, μ2={mu2_fit:.3f}, R²={r_squared:.3f}")
            print(f"[2-Gauss] Weights: exp={w_exp_fit:.3f}, g1={w_g1_fit:.3f}, g2={w_g2_fit:.3f}")

            # Reject if R² too low or weights are weird
            if r_squared < 0.7 or w_g1_fit < 0.05 or w_g2_fit < 0.05:
                print("[2-Gauss] Poor fit or degenerate weights, rejecting")
                return None

            # Find valley between 0 and first Gaussian peak
            search_range = (bin_centers >= 0) & (bin_centers <= mu1_fit)
            if not np.any(search_range):
                return None

            search_x = bin_centers[search_range]
            search_y = exp_2gauss_model(search_x, *popt)

            valley_idx = np.argmin(search_y)
            valley_location = search_x[valley_idx]
            valley_value = search_y[valley_idx]

            # Quality score
            exp_at_valley = w_exp_fit * lambda_fit * np.exp(-lambda_fit * valley_location)
            valley_depth = 1.0 - (valley_value / exp_at_valley) if exp_at_valley > 0 else 0.0
            quality_score = r_squared * valley_depth
            quality_score = max(0.0, min(1.0, quality_score))

            print(f"[2-Gauss] Valley at {valley_location:.3f}, quality={quality_score:.3f}")

            # Store parameters including bin_centers and fitted curve for plotting
            self.exp_gauss_params = {
                'model': '2gauss',
                'lambda': lambda_fit,
                'mu1': mu1_fit,
                'sigma1': sigma1_fit,
                'mu2': mu2_fit,
                'sigma2': sigma2_fit,
                'w_exp': w_exp_fit,
                'w_g1': w_g1_fit,
                'w_g2': w_g2_fit,
                'r_squared': r_squared,
                'bin_centers': bin_centers,
                'fitted_curve': fitted,
                'density': density
            }

            # Store parameters in metrics module for probability metrics
            self._store_model_params_for_probability_metrics()

            return float(valley_location), float(quality_score)

        except Exception as e:
            print(f"[2-Gauss] Fit failed: {e}")
            return None

    def _fit_exp_1gauss(self, bin_centers, density, counts):
        """Fit exponential + 1 Gaussian (noise + breaths)."""
        try:
            from scipy.optimize import curve_fit

            def exp_gauss_model(x, lambda_exp, mu_gauss, sigma_gauss, w_exp):
                exp_component = lambda_exp * np.exp(-lambda_exp * x)
                gauss_component = (1 / (np.sqrt(2 * np.pi) * sigma_gauss)) * np.exp(-0.5 * ((x - mu_gauss) / sigma_gauss) ** 2)
                return w_exp * exp_component + (1 - w_exp) * gauss_component

            # Initial guesses
            left_half = bin_centers[bin_centers < np.median(bin_centers)]
            lambda_init = 1.0 / np.mean(left_half) if len(left_half) > 0 else 1.0
            mu_init = np.average(bin_centers, weights=counts + 1)
            sigma_init = np.sqrt(np.average((bin_centers - mu_init)**2, weights=counts + 1))
            w_exp_init = 0.3

            print(f"[1-Gauss] Initial: λ={lambda_init:.3f}, μ={mu_init:.3f}, σ={sigma_init:.3f}")

            bounds = ([0.001, bin_centers.min(), 0.001, 0.0],
                     [100.0, bin_centers.max(), bin_centers.max(), 1.0])

            popt, pcov = curve_fit(
                exp_gauss_model,
                bin_centers,
                density,
                p0=[lambda_init, mu_init, sigma_init, w_exp_init],
                bounds=bounds,
                maxfev=5000
            )

            lambda_fit, mu_fit, sigma_fit, w_exp_fit = popt

            # Calculate R²
            fitted = exp_gauss_model(bin_centers, *popt)
            ss_res = np.sum((density - fitted) ** 2)
            ss_tot = np.sum((density - np.mean(density)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            print(f"[1-Gauss] Fitted: λ={lambda_fit:.3f}, μ={mu_fit:.3f}, σ={sigma_fit:.3f}, R²={r_squared:.3f}")

            # Find valley
            search_range = (bin_centers >= 0) & (bin_centers <= mu_fit)
            if not np.any(search_range):
                return None

            search_x = bin_centers[search_range]
            search_y = exp_gauss_model(search_x, *popt)

            valley_idx = np.argmin(search_y)
            valley_location = search_x[valley_idx]
            valley_value = search_y[valley_idx]

            # Quality
            exp_at_valley = w_exp_fit * lambda_fit * np.exp(-lambda_fit * valley_location)
            valley_depth = 1.0 - (valley_value / exp_at_valley) if exp_at_valley > 0 else 0.0
            quality_score = r_squared * valley_depth
            quality_score = max(0.0, min(1.0, quality_score))

            print(f"[1-Gauss] Valley at {valley_location:.3f}, quality={quality_score:.3f}")

            # Store parameters for plotting
            self.exp_gauss_params = {
                'model': '1gauss',
                'lambda': lambda_fit,
                'mu': mu_fit,
                'sigma': sigma_fit,
                'w_exp': w_exp_fit,
                'r_squared': r_squared,
                'bin_centers': bin_centers,
                'fitted_curve': fitted,
                'density': density
            }

            # Store parameters in metrics module for probability metrics
            self._store_model_params_for_probability_metrics()

            return float(valley_location), float(quality_score)

        except Exception as e:
            print(f"[1-Gauss] Fit failed: {e}")
            return None

    def _store_model_params_for_probability_metrics(self):
        """
        Store auto-threshold model parameters in metrics module.

        This enables P(noise) and P(breath) metrics to evaluate probabilities
        based on the fitted exponential + Gaussian model.
        """
        from core import metrics as core_metrics

        if not hasattr(self, 'exp_gauss_params') or self.exp_gauss_params is None:
            return

        params = self.exp_gauss_params
        model_type = params.get('model')

        if model_type == '2gauss':
            # Exponential + 2 Gaussians
            model_params = {
                'lambda_exp': float(params['lambda']),
                'mu1': float(params['mu1']),
                'sigma1': float(params['sigma1']),
                'mu2': float(params['mu2']),
                'sigma2': float(params['sigma2']),
                'w_exp': float(params['w_exp']),
                'w_g1': float(params['w_g1']),
                'w_g2': float(params['w_g2'])
            }
        elif model_type == '1gauss':
            # Exponential + 1 Gaussian (treat as 2-gauss with second Gaussian disabled)
            model_params = {
                'lambda_exp': float(params['lambda']),
                'mu1': float(params['mu']),
                'sigma1': float(params['sigma']),
                'mu2': float(params['mu']),  # Same as mu1 (won't be used much)
                'sigma2': float(params['sigma']),  # Same as sigma1
                'w_exp': float(params['w_exp']),
                'w_g1': float(1.0 - params['w_exp']),  # All non-exp weight to first Gaussian
                'w_g2': 0.0  # No second Gaussian
            }
        else:
            print(f"[Probability] Unknown model type: {model_type}")
            return

        core_metrics.set_threshold_model_params(model_params)
        print(f"[Probability] Stored {model_type} model params for P(noise) and P(breath) metrics")

    def _update_quality_label(self):
        """Update the signal quality label with color-coded quality rating."""
        score = self.valley_depth_score

        # Determine quality category and color
        if score >= 0.8:
            quality = "Excellent"
            color = "#00aa00"  # Green
            tooltip = "Very clear separation between noise and signal peaks"
        elif score >= 0.5:
            quality = "Good"
            color = "#55aa00"  # Yellow-green
            tooltip = "Good separation, minimal overlap between noise and signal"
        elif score >= 0.3:
            quality = "Fair"
            color = "#aaaa00"  # Yellow
            tooltip = "Moderate overlap - may need manual review of low-amplitude breaths"
        elif score >= 0.1:
            quality = "Poor"
            color = "#aa5500"  # Orange
            tooltip = "Significant overlap - manual labeling recommended"
        else:
            quality = "Very Poor"
            color = "#aa0000"  # Red
            tooltip = "Heavy overlap - extensive manual review required"

        # Update label with score and quality
        self.lbl_signal_quality.setText(f"Signal Quality: {quality} ({score:.2f})")
        self.lbl_signal_quality.setStyleSheet(f"font-size: 10pt; font-weight: bold; color: {color};")
        self.lbl_signal_quality.setToolTip(tooltip)

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

    def _reset_to_valley(self):
        """Reset threshold to valley (local minimum) value."""
        if self.local_min_threshold is not None:
            self.current_threshold = self.local_min_threshold
            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")
            self.btn_reset.setEnabled(False)  # Disable after resetting
            self._plot_histogram()

            # Real-time update to main plot
            # Use main_window reference if available (when embedded), otherwise use parent()
            main_window = getattr(self, 'main_window', None) or self.parent()
            if main_window is not None:
                try:
                    if hasattr(main_window, 'plot_host'):
                        main_window.plot_host.update_threshold_line(self.current_threshold)
                        main_window.plot_host.canvas.draw_idle()
                except Exception:
                    pass

            print(f"[Threshold] Reset to Valley: {self.current_threshold:.4f}")

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

    def _on_classifier_changed(self, text, parent):
        """Handle classifier selection change."""
        # Map display text to internal classifier name
        classifier_map = {
            "Threshold": "threshold",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp"
        }

        new_classifier = classifier_map.get(text, "threshold")

        # Update state
        if parent and hasattr(parent, 'state'):
            parent.state.active_classifier = new_classifier
            print(f"[Classifier] Switched to: {new_classifier}")

            # Update status label
            self._update_classifier_status(parent)

            # Trigger re-plot in parent if available
            if hasattr(parent, 'plot_current_data'):
                parent.plot_current_data()

    def _update_classifier_status(self, parent):
        """Update the classifier status label."""
        if not hasattr(self, 'lbl_classifier_status'):
            return

        if not parent or not hasattr(parent, 'state'):
            self.lbl_classifier_status.setText("")
            return

        state = parent.state

        if state.active_classifier == 'threshold':
            self.lbl_classifier_status.setText("Using amplitude-based threshold")
            self.lbl_classifier_status.setStyleSheet("font-size: 9pt; color: #888;")
        else:
            # Check if ML models are loaded
            if state.loaded_ml_models:
                model_key = f'model1_{state.active_classifier}'
                if model_key in state.loaded_ml_models:
                    metadata = state.loaded_ml_models[model_key]['metadata']
                    accuracy = metadata.get('test_accuracy', 0)
                    self.lbl_classifier_status.setText(f"ML model loaded ({accuracy:.1%} accuracy)")
                    self.lbl_classifier_status.setStyleSheet("font-size: 9pt; color: #4ec9b0;")
                else:
                    self.lbl_classifier_status.setText(f"No {state.active_classifier.upper()} models loaded")
                    self.lbl_classifier_status.setStyleSheet("font-size: 9pt; color: #ce9178;")
            else:
                self.lbl_classifier_status.setText("No ML models loaded")
                self.lbl_classifier_status.setStyleSheet("font-size: 9pt; color: #ce9178;")

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

            # Store patches and bins for fast updating during drag
            self.histogram_patches = patches
            self.histogram_bins = bins

            # Color bars based on threshold (gray = below, red = above)
            if self.current_threshold is not None:
                for i, patch in enumerate(patches):
                    if bins[i] < self.current_threshold:
                        patch.set_facecolor('gray')  # Gray for "noise"
                        patch.set_alpha(0.5)
                    else:
                        patch.set_facecolor('red')  # Red for "breaths"
                        patch.set_alpha(0.5)

            # Plot fitted curve components if available
            if hasattr(self, 'exp_gauss_params') and self.exp_gauss_params is not None:
                params = self.exp_gauss_params
                bin_centers = params.get('bin_centers')
                fitted_curve = params.get('fitted_curve')
                r_squared = params.get('r_squared', 0.0)
                model_type = params.get('model', 'unknown')

                if bin_centers is not None and fitted_curve is not None:
                    # Scale factor to convert density to counts
                    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
                    scale = len(peaks_for_hist) * bin_width

                    # Plot individual components
                    x = bin_centers

                    if model_type == '2gauss':
                        # Extract 2-Gaussian model parameters
                        lambda_exp = params.get('lambda')
                        mu1 = params.get('mu1')
                        sigma1 = params.get('sigma1')
                        mu2 = params.get('mu2')
                        sigma2 = params.get('sigma2')
                        w_exp = params.get('w_exp')
                        w_g1 = params.get('w_g1')
                        w_g2 = params.get('w_g2')

                        # Calculate individual components (in density units)
                        exp_comp = lambda_exp * np.exp(-lambda_exp * x)
                        gauss1_comp = (1 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((x - mu1) / sigma1) ** 2)
                        gauss2_comp = (1 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((x - mu2) / sigma2) ** 2)

                        # Scale to counts
                        exp_counts = w_exp * exp_comp * scale
                        gauss1_counts = w_g1 * gauss1_comp * scale
                        gauss2_counts = w_g2 * gauss2_comp * scale

                        # Plot individual components with different styles
                        ax1.plot(x, exp_counts, color='#ffaa33', linewidth=1.5,
                                linestyle='--', alpha=0.6, label='Exponential (noise)', zorder=3)
                        ax1.plot(x, gauss1_counts, color='#00ff88', linewidth=1.5,
                                linestyle='--', alpha=0.6, label=f'Gaussian 1 (μ={mu1:.2f})', zorder=3)
                        ax1.plot(x, gauss2_counts, color='#bb66ff', linewidth=1.5,
                                linestyle='--', alpha=0.6, label=f'Gaussian 2 (μ={mu2:.2f})', zorder=3)

                        # Plot combined fit (thinner line)
                        fitted_counts = fitted_curve * scale
                        ax1.plot(x, fitted_counts, color='#ff00ff', linewidth=2.0,
                                linestyle='-', alpha=0.9, label=f'Combined Fit (R²={r_squared:.3f})', zorder=4)

                    else:  # 1gauss model
                        # Extract 1-Gaussian model parameters
                        lambda_exp = params.get('lambda')
                        mu = params.get('mu')
                        sigma = params.get('sigma')
                        w_exp = params.get('w_exp')

                        # Calculate individual components (in density units)
                        exp_comp = lambda_exp * np.exp(-lambda_exp * x)
                        gauss_comp = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

                        # Scale to counts
                        exp_counts = w_exp * exp_comp * scale
                        gauss_counts = (1 - w_exp) * gauss_comp * scale

                        # Plot individual components
                        ax1.plot(x, exp_counts, color='#ffaa33', linewidth=1.5,
                                linestyle='--', alpha=0.6, label='Exponential (noise)', zorder=3)
                        ax1.plot(x, gauss_counts, color='#00ff88', linewidth=1.5,
                                linestyle='--', alpha=0.6, label=f'Gaussian (μ={mu:.2f})', zorder=3)

                        # Plot combined fit (thinner line)
                        fitted_counts = fitted_curve * scale
                        ax1.plot(x, fitted_counts, color='#00ffff', linewidth=2.0,
                                linestyle='-', alpha=0.9, label=f'Combined Fit (R²={r_squared:.3f})', zorder=4)

            ax1.set_xlabel('Peak Height', fontsize=11, color='#d4d4d4')
            ax1.set_ylabel('Frequency (count)', fontsize=11, color='#d4d4d4')

            # Title with outlier count if applicable
            if n_excluded > 0:
                title_text = f'Peak Height Distribution\n{n_excluded} outlier peaks excluded (above {self.percentile_cutoff}th percentile)'
            else:
                title_text = 'Peak Height Distribution'
            ax1.set_title(title_text, fontsize=12, fontweight='bold', color='#d4d4d4')
            ax1.grid(True, alpha=0.2, axis='y', color='#666666')

            # COMMENTED OUT - Otsu reference line not needed
            # # Draw FIXED Otsu reference line (thin, gray, dashed) - no label to simplify legend
            # if self.auto_threshold is not None:
            #     self.otsu_reference_line = ax1.axvline(self.auto_threshold, color='#888888',
            #                                            linestyle=':', linewidth=1.0,
            #                                            zorder=1)  # Behind draggable line

            # Draw local minimum line (thin, cyan, dotted)
            if hasattr(self, 'local_min_threshold') and self.local_min_threshold is not None:
                ax1.axvline(self.local_min_threshold, color='#00cccc',
                           linestyle=':', linewidth=1.0,
                           label=f'Local Min = {self.local_min_threshold:.4f}',
                           zorder=1)

            # Draw draggable threshold line (red, thicker) - no label to simplify legend
            if self.current_threshold is not None:
                self.threshold_vline = ax1.axvline(self.current_threshold, color='red',
                                                   linestyle='--', linewidth=1.5,
                                                   picker=5, zorder=2)  # Pickable within 5 pixels

            # COMMENTED OUT - Y2 axis not needed
            # # Secondary y-axis: Toggle between peak count and inter-class variance
            # ax2 = ax1.twinx()
            #
            # # Apply dark theme to second y-axis
            # ax2.spines['right'].set_color('#666666')
            # ax2.spines['left'].set_color('#666666')
            # ax2.spines['top'].set_color('#666666')
            # ax2.spines['bottom'].set_color('#666666')
            #
            # if self.y2_mode == "peak_count":
            #     # Compute peak count vs threshold (using filtered range)
            #     if self.percentile_95 is not None:
            #         thresh_range = np.linspace(peak_heights.min(), self.percentile_95, 100)
            #     else:
            #         thresh_range = np.linspace(peak_heights.min(), peak_heights.max(), 100)
            #
            #     peak_counts = [np.sum(peak_heights >= t) for t in thresh_range]
            #
            #     # No label to simplify legend
            #     ax2.plot(thresh_range, peak_counts, color='#66cc66', linewidth=2, alpha=0.6)
            #     ax2.set_ylabel('Peaks Above Threshold', fontsize=11, color='#66cc66')
            #     ax2.tick_params(axis='y', labelcolor='#66cc66')
            #
            #     # Mark current peak count (no horizontal line to avoid obscuring label)
            #     if self.current_threshold is not None:
            #         current_count = np.sum(peak_heights >= self.current_threshold)
            #         ax2.plot(self.current_threshold, current_count, 'ro', markersize=8)
            #         ax2.text(self.current_threshold, current_count, f'  {current_count} peaks',
            #                 va='center', fontsize=9, color='#ff6666')
            #
            # else:  # variance mode
            #     # Plot inter-class variance
            #     if self.inter_class_variance_curve is not None:
            #         var_thresh, var_values = self.inter_class_variance_curve
            #         ax2.plot(var_thresh, var_values, color='#cc66cc', linewidth=2, alpha=0.6,
            #                 label='Inter-Class Variance')
            #         ax2.set_ylabel('Inter-Class Variance', fontsize=11, color='#cc66cc')
            #         ax2.tick_params(axis='y', labelcolor='#cc66cc')
            #
            #         # Mark maximum variance point
            #         max_idx = np.argmax(var_values)
            #         ax2.plot(var_thresh[max_idx], var_values[max_idx], color='#cc66cc',
            #                 marker='o', markersize=8)
            #
            #         # Mark current variance point (no horizontal line to avoid obscuring labels)
            #         if self.current_threshold is not None:
            #             # Find variance at current threshold
            #             closest_idx = np.argmin(np.abs(var_thresh - self.current_threshold))
            #             current_var = var_values[closest_idx]
            #             ax2.plot(self.current_threshold, current_var, 'ro', markersize=8)
            #             ax2.text(self.current_threshold, current_var, f'  Var={current_var:.0f}',
            #                     va='center', fontsize=9, color='#ff6666')

            # Legend with dark theme styling
            lines1, labels1 = ax1.get_legend_handles_labels()
            legend = ax1.legend(lines1, labels1, loc='upper right',
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
        """Handle mouse drag to adjust threshold (optimized - no full replot)."""
        if not self.is_dragging or event.inaxes is None:
            return

        # Update threshold to mouse x position
        new_threshold = event.xdata
        if new_threshold is not None and new_threshold > 0:
            self.current_threshold = float(new_threshold)
            self.lbl_threshold.setText(f"{self.current_threshold:.4f}")

            # Enable reset button if threshold changed from valley
            if self.local_min_threshold is not None:
                if abs(self.current_threshold - self.local_min_threshold) > 0.001:
                    self.btn_reset.setEnabled(True)
                else:
                    self.btn_reset.setEnabled(False)

            # Update height threshold field to match dragged threshold
            self.le_threshold_height.setText(f"{self.current_threshold:.4f}")

            # OPTIMIZED: Just update line position and bar colors, don't replot everything
            # Don't update main plot during drag - too slow!
            self._update_threshold_line_fast(self.current_threshold)

    def _on_mouse_release(self, event):
        """Handle mouse release after drag."""
        if self.is_dragging:
            self.is_dragging = False
            self.canvas.setCursor(Qt.CursorShape.ArrowCursor)

            # Update main plot ONCE after drag is complete
            # Use main_window reference if available (when embedded), otherwise use parent()
            main_window = getattr(self, 'main_window', None) or self.parent()

            if main_window is not None:
                try:
                    # Update main plot threshold line
                    if hasattr(main_window, 'plot_host'):
                        main_window.plot_host.update_threshold_line(self.current_threshold)
                        main_window.plot_host.canvas.draw_idle()

                    # Update stored prominence value in main window
                    if hasattr(main_window, 'peak_prominence'):
                        main_window.peak_prominence = self.current_threshold
                    if hasattr(main_window, 'peak_height_threshold'):
                        main_window.peak_height_threshold = self.current_threshold
                except Exception as e:
                    print(f"[ProminenceDialog] Error updating main plot: {e}")  # Debug

    def _update_threshold_line_fast(self, new_threshold):
        """
        Fast update of threshold line position and bar colors.
        This is MUCH faster than calling _plot_histogram() on every mouse move.
        """
        if self.threshold_vline is None or self.histogram_patches is None:
            return

        try:
            # Update threshold line position
            self.threshold_vline.set_xdata([new_threshold, new_threshold])

            # Recolor histogram bars based on new threshold
            bins = self.histogram_bins
            for i, patch in enumerate(self.histogram_patches):
                if bins[i] < new_threshold:
                    patch.set_facecolor('gray')  # Gray for "noise"
                    patch.set_alpha(0.5)
                else:
                    patch.set_facecolor('red')  # Red for "breaths"
                    patch.set_alpha(0.5)

            # Fast redraw without clearing figure
            # This avoids the overhead of tight_layout, axis clearing, etc.
            self.canvas.draw_idle()

        except Exception as e:
            # If fast update fails, fall back to full replot
            print(f"[Fast Update] Failed: {e}")
            self._plot_histogram()

    def update_threshold_from_external(self, new_threshold):
        """
        Update threshold from an external source (e.g., main plot dragging).
        Syncs the histogram display with changes made elsewhere.
        """
        if new_threshold is not None and new_threshold > 0:
            self.current_threshold = new_threshold
            # Update UI labels
            self.lbl_threshold.setText(f"{new_threshold:.4f}")
            self.le_threshold_height.setText(f"{new_threshold:.4f}")
            # Update histogram visualization
            self._update_threshold_line_fast(new_threshold)
            # Enable reset button if threshold changed from auto-detected value
            if hasattr(self, 'auto_threshold') and abs(new_threshold - self.auto_threshold) > 1e-6:
                if hasattr(self, 'btn_reset'):
                    self.btn_reset.setEnabled(True)

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
