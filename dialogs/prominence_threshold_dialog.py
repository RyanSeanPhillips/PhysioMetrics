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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import find_peaks


class ProminenceThresholdDialog(QDialog):
    """Interactive prominence threshold detection using Otsu's method."""

    def __init__(self, parent=None, y_data=None, sr_hz=None, current_prom=None, current_min_dist=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Detect Prominence Threshold")
        self.resize(1000, 700)

        self.y_data = y_data
        self.sr_hz = sr_hz or 1000.0
        self.current_prom = current_prom or 0.1
        self.current_min_dist = current_min_dist or 0.05

        # Cached peak detection
        self.all_peaks = None
        self.all_prominences = None

        # Auto-calculated threshold
        self.auto_threshold = None
        self.current_threshold = None  # User-adjusted value
        self.inter_class_variance_curve = None  # For plotting

        # Separation metric for display
        self.separation_metric = 1.0

        # Draggable line (only vertical, no horizontal to avoid obscuring labels)
        self.threshold_vline = None
        self.is_dragging = False

        # Y2 axis mode toggle
        self.y2_mode = "peak_count"  # or "variance"

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

            # Find ALL peaks with very low prominence
            peaks, props = find_peaks(self.y_data, prominence=0.001, distance=min_dist_samples)

            self.all_peaks = peaks
            self.all_prominences = props['prominences']

            t_elapsed = time.time() - t_start
            print(f"[Prominence Dialog] Found {len(self.all_peaks)} peaks in {t_elapsed:.2f}s")

            self.lbl_peak_count.setText(f"Peaks detected: {len(self.all_peaks)}")

        except Exception as e:
            print(f"[Prominence Dialog] Error: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_otsu_threshold(self):
        """Calculate optimal prominence threshold using Otsu's method."""
        if self.all_prominences is None or len(self.all_prominences) < 10:
            return

        try:
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
            self.auto_threshold = float((optimal_thresh_norm / 255.0 *
                            (prominences.max() - prominences.min()) +
                            prominences.min()))

            # Store maximum variance for quality assessment
            self.max_inter_class_variance = float(np.max(variance))

            # Store inter-class variance curve for plotting
            # Convert bin_centers back to original scale for x-axis
            thresh_values = bin_centers[:-1] / 255.0 * (prominences.max() - prominences.min()) + prominences.min()
            self.inter_class_variance_curve = (thresh_values, variance)

            self.current_threshold = self.auto_threshold

            print(f"[Otsu] Auto-detected prominence threshold: {self.auto_threshold:.4f}")
            self.lbl_threshold.setText(f"{self.auto_threshold:.4f}")

            # Also populate the height threshold field with the same value
            self.le_threshold_height.setText(f"{self.auto_threshold:.4f}")

        except Exception as e:
            print(f"[Otsu] Error: {e}")
            import traceback
            traceback.print_exc()


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
        """Plot prominence histogram with draggable threshold lines."""
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

            if self.all_prominences is None:
                ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', color='#d4d4d4')
                self.canvas.draw()
                return

            prominences = self.all_prominences

            # Plot histogram with MORE BINS (100 instead of 50)
            n, bins, patches = ax1.hist(prominences, bins=100, color='steelblue',
                                       alpha=0.7, edgecolor='#1e1e1e', label='Prominence Distribution')

            # Color bars based on threshold
            if self.current_threshold is not None:
                for i, patch in enumerate(patches):
                    if bins[i] < self.current_threshold:
                        patch.set_facecolor('#cc8866')  # Orange for "noise"
                    else:
                        patch.set_facecolor('#6699cc')  # Blue for "breaths"

            ax1.set_xlabel('Prominence', fontsize=11, color='#d4d4d4')
            ax1.set_ylabel('Frequency (count)', fontsize=11, color='#d4d4d4')
            ax1.set_title('Peak Prominence Distribution (Otsu\'s Method)', fontsize=12,
                         fontweight='bold', color='#d4d4d4')
            ax1.grid(True, alpha=0.2, axis='y', color='#666666')

            # Draw THINNER threshold lines (draggable)
            if self.current_threshold is not None:
                # Vertical line (thinner: linewidth=1.5 instead of 2.5)
                self.threshold_vline = ax1.axvline(self.current_threshold, color='red',
                                                   linestyle='--', linewidth=1.5,
                                                   label=f'Threshold = {self.current_threshold:.4f}',
                                                   picker=5)  # Pickable within 5 pixels

            # Secondary y-axis: Toggle between peak count and inter-class variance
            ax2 = ax1.twinx()

            # Apply dark theme to second y-axis
            ax2.spines['right'].set_color('#666666')
            ax2.spines['left'].set_color('#666666')
            ax2.spines['top'].set_color('#666666')
            ax2.spines['bottom'].set_color('#666666')

            if self.y2_mode == "peak_count":
                # Compute peak count vs threshold
                thresh_range = np.linspace(prominences.min(), prominences.max(), 100)
                peak_counts = [np.sum(prominences >= t) for t in thresh_range]

                ax2.plot(thresh_range, peak_counts, color='#66cc66', linewidth=2, alpha=0.6,
                        label='Peaks Above Threshold')
                ax2.set_ylabel('Peaks Above Threshold', fontsize=11, color='#66cc66')
                ax2.tick_params(axis='y', labelcolor='#66cc66')

                # Mark current peak count (no horizontal line to avoid obscuring label)
                if self.current_threshold is not None:
                    current_count = np.sum(prominences >= self.current_threshold)
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
            'height_threshold': height_thresh  # Now always populated from Otsu's method
        }
