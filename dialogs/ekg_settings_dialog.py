"""
EKG Detection Settings Dialog

Allows the user to adjust R-peak detection parameters with a live
preview of the bandpass-filtered signal, threshold, and detected peaks.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QSlider, QPushButton, QGroupBox, QFormLayout, QFrame,
    QSplitter, QWidget, QCheckBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class EKGSettingsDialog(QDialog):
    """EKG detection settings with live signal preview.

    Shows a matplotlib preview of the raw signal, bandpass-filtered signal,
    the Pan-Tompkins integrated signal with threshold, and detected R-peaks.
    Updates in real time as the user adjusts sliders.
    """

    detection_requested = pyqtSignal()  # re-run detection on main plot

    def __init__(self, ecg_config, signal=None, sr_hz=1000.0, parent=None):
        """
        Parameters
        ----------
        ecg_config : ECGConfig
            The config object (modified in-place by sliders).
        signal : ndarray or None
            Raw EKG signal for the current sweep.  If None, preview is disabled.
        sr_hz : float
            Sampling rate in Hz.
        parent : QWidget
            Parent window (MainWindow).
        """
        super().__init__(parent)
        self.setWindowTitle("EKG Detection Settings")
        self.setMinimumSize(700, 600)
        self._config = ecg_config
        self._signal = signal
        self._sr_hz = sr_hz
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self._run_preview)
        self._setup_ui()
        self._load_from_config()
        # Initial preview
        QTimer.singleShot(50, self._run_preview)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Live feedback stats ──
        self._stats_label = QLabel("Detecting...")
        self._stats_label.setStyleSheet(
            "background: #1a1a2e; color: #FF9800; padding: 6px; "
            "border-radius: 4px; font-size: 12px; font-weight: bold;"
        )
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._stats_label)

        # ── Preview plot ──
        self._fig = Figure(figsize=(7, 3.5), dpi=100, facecolor='#1a1a2e')
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(250)
        layout.addWidget(self._canvas, stretch=1)

        # ── Controls in a compact horizontal layout ──
        controls = QWidget()
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(10)

        # Left column: Species + Filter
        left_group = QGroupBox("Filter / Species")
        left_layout = QFormLayout(left_group)
        left_layout.setSpacing(4)

        self._species_combo = QComboBox()
        self._species_combo.addItems(["mouse", "rat", "human"])
        self._species_combo.currentTextChanged.connect(self._on_species_changed)
        left_layout.addRow("Species:", self._species_combo)

        # Polarity: Auto / Upright / Inverted
        self._polarity_combo = QComboBox()
        self._polarity_combo.addItems(["Auto", "Upright (R-peak up)", "Inverted (R-peak down)"])
        self._polarity_combo.currentIndexChanged.connect(self._on_polarity_changed)
        left_layout.addRow("Polarity:", self._polarity_combo)

        self._bp_low_slider = QSlider(Qt.Orientation.Horizontal)
        self._bp_low_slider.setRange(1, 50)
        self._bp_low_label = QLabel()
        self._bp_low_label.setFixedWidth(40)
        self._bp_low_slider.valueChanged.connect(self._on_bp_low_changed)
        row = QHBoxLayout()
        row.addWidget(self._bp_low_slider, 1)
        row.addWidget(self._bp_low_label)
        left_layout.addRow("BP low (Hz):", row)

        self._bp_high_slider = QSlider(Qt.Orientation.Horizontal)
        self._bp_high_slider.setRange(20, 500)
        self._bp_high_label = QLabel()
        self._bp_high_label.setFixedWidth(40)
        self._bp_high_slider.valueChanged.connect(self._on_bp_high_changed)
        row2 = QHBoxLayout()
        row2.addWidget(self._bp_high_slider, 1)
        row2.addWidget(self._bp_high_label)
        left_layout.addRow("BP high (Hz):", row2)

        ctrl_layout.addWidget(left_group)

        # Right column: Detection thresholds
        right_group = QGroupBox("Detection")
        right_layout = QFormLayout(right_group)
        right_layout.setSpacing(4)

        self._thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self._thresh_slider.setRange(1, 80)
        self._thresh_label = QLabel()
        self._thresh_label.setFixedWidth(40)
        self._thresh_slider.valueChanged.connect(self._on_thresh_changed)
        row3 = QHBoxLayout()
        row3.addWidget(self._thresh_slider, 1)
        row3.addWidget(self._thresh_label)
        right_layout.addRow("Threshold:", row3)

        self._prom_slider = QSlider(Qt.Orientation.Horizontal)
        self._prom_slider.setRange(1, 30)
        self._prom_label = QLabel()
        self._prom_label.setFixedWidth(40)
        self._prom_slider.valueChanged.connect(self._on_prom_changed)
        row4 = QHBoxLayout()
        row4.addWidget(self._prom_slider, 1)
        row4.addWidget(self._prom_label)
        right_layout.addRow("Prominence:", row4)

        self._refract_slider = QSlider(Qt.Orientation.Horizontal)
        self._refract_slider.setRange(20, 300)
        self._refract_label = QLabel()
        self._refract_label.setFixedWidth(45)
        self._refract_slider.valueChanged.connect(self._on_refract_changed)
        row5 = QHBoxLayout()
        row5.addWidget(self._refract_slider, 1)
        row5.addWidget(self._refract_label)
        right_layout.addRow("Min RR (ms):", row5)

        ctrl_layout.addWidget(right_group)

        layout.addWidget(controls)

        # ── Buttons ──
        btn_layout = QHBoxLayout()
        self._reset_btn = QPushButton("Reset Defaults")
        self._reset_btn.clicked.connect(self._on_reset_clicked)
        self._apply_btn = QPushButton("Apply to Main Plot")
        self._apply_btn.setStyleSheet(
            "background-color: #1565C0; color: white; padding: 6px 16px;"
        )
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        btn_layout.addWidget(self._reset_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._apply_btn)
        layout.addLayout(btn_layout)

    # ── Load / Labels ──

    def _load_from_config(self):
        for w in (self._thresh_slider, self._prom_slider, self._refract_slider,
                  self._bp_low_slider, self._bp_high_slider):
            w.blockSignals(True)

        c = self._config
        idx = self._species_combo.findText(c.species)
        if idx >= 0:
            self._species_combo.blockSignals(True)
            self._species_combo.setCurrentIndex(idx)
            self._species_combo.blockSignals(False)

        # Polarity combo
        force_inv = getattr(c, 'force_inverted', None)
        self._polarity_combo.blockSignals(True)
        if force_inv is None:
            self._polarity_combo.setCurrentIndex(0)
        elif force_inv is False:
            self._polarity_combo.setCurrentIndex(1)
        else:
            self._polarity_combo.setCurrentIndex(2)
        self._polarity_combo.blockSignals(False)
        self._thresh_slider.setValue(int(c.threshold_fraction * 100))
        prom = getattr(c, 'prominence_fraction', 0.05)
        self._prom_slider.setValue(int(prom * 100))
        self._refract_slider.setValue(int(c.refractory_ms))
        self._bp_low_slider.setValue(int(c.bandpass_low))
        self._bp_high_slider.setValue(int(c.bandpass_high))

        for w in (self._thresh_slider, self._prom_slider, self._refract_slider,
                  self._bp_low_slider, self._bp_high_slider):
            w.blockSignals(False)

        self._update_labels()

    def _update_labels(self):
        self._thresh_label.setText(f"{self._thresh_slider.value()}%")
        self._prom_label.setText(f"{self._prom_slider.value()}%")
        self._refract_label.setText(f"{self._refract_slider.value()} ms")
        self._bp_low_label.setText(f"{self._bp_low_slider.value()} Hz")
        self._bp_high_label.setText(f"{self._bp_high_slider.value()} Hz")

    # ── Slider handlers ──

    def _on_thresh_changed(self, val):
        self._config.threshold_fraction = val / 100.0
        self._thresh_label.setText(f"{val}%")
        self._schedule_preview()

    def _on_prom_changed(self, val):
        self._config.prominence_fraction = val / 100.0
        self._prom_label.setText(f"{val}%")
        self._schedule_preview()

    def _on_refract_changed(self, val):
        self._config.refractory_ms = float(val)
        self._refract_label.setText(f"{val} ms")
        self._schedule_preview()

    def _on_bp_low_changed(self, val):
        self._config.bandpass_low = float(val)
        self._bp_low_label.setText(f"{val} Hz")
        self._schedule_preview()

    def _on_bp_high_changed(self, val):
        self._config.bandpass_high = float(val)
        self._bp_high_label.setText(f"{val} Hz")
        self._schedule_preview()

    def _on_polarity_changed(self, idx):
        if idx == 0:
            self._config.force_inverted = None  # Auto
        elif idx == 1:
            self._config.force_inverted = False  # Upright
        else:
            self._config.force_inverted = True   # Inverted
        self._schedule_preview()

    def _on_species_changed(self, species):
        self._config.species = species
        self._config.apply_species_preset()
        self._load_from_config()
        self._schedule_preview()

    def _on_reset_clicked(self):
        self._config.apply_species_preset()
        self._load_from_config()
        self._schedule_preview()

    def _on_apply_clicked(self):
        self.detection_requested.emit()

    def _schedule_preview(self):
        self._debounce_timer.start()

    # ── Preview rendering ──

    def _run_preview(self):
        """Run detection on the signal and render the preview plot."""
        if self._signal is None or len(self._signal) == 0:
            self._stats_label.setText("No signal data available for preview")
            return

        from scipy.signal import butter, sosfiltfilt, find_peaks

        signal = self._signal
        sr = self._sr_hz
        config = self._config
        n = len(signal)
        t = np.arange(n) / sr

        # 1. Bandpass filter
        nyq = sr / 2.0
        low = max(config.bandpass_low, 0.5)
        high = min(config.bandpass_high, nyq - 1.0)
        if low >= high:
            low, high = 0.5, nyq - 1.0
        try:
            sos = butter(config.filter_order, [low, high], btype='band',
                         fs=sr, output='sos')
            filtered = sosfiltfilt(sos, signal)
        except Exception:
            filtered = signal.copy()

        # 2. Pan-Tompkins: derivative + squaring + MWI
        h = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (sr / 8.0)
        derivative = np.convolve(filtered, h, mode='same')
        squared = derivative ** 2
        win_size = max(1, int(config.mwi_window_ms * sr / 1000.0))
        kernel = np.ones(win_size) / win_size
        integrated = np.convolve(squared, kernel, mode='same')

        # 3. Threshold
        pct_val = np.percentile(integrated, config.threshold_percentile)
        if pct_val <= 0:
            pct_val = np.percentile(integrated, 99)
        height_thresh = config.threshold_fraction * pct_val
        prom_frac = getattr(config, 'prominence_fraction', 0.05)
        prom_thresh = prom_frac * pct_val
        min_dist = max(1, int(config.refractory_ms * sr / 1000.0))

        # 4. Find peaks
        candidates, _ = find_peaks(
            integrated, distance=min_dist,
            height=height_thresh, prominence=prom_thresh,
        )

        # 5. Refine to filtered signal
        search = max(1, int(config.qrs_search_ms * sr / 1000.0))
        force_inv = getattr(config, 'force_inverted', None)
        baseline = np.mean(filtered)
        r_peaks = []
        n_neg = 0
        for c in candidates:
            lo = max(0, c - search)
            hi = min(n, c + search + 1)
            window = filtered[lo:hi]
            if force_inv is True:
                # User says inverted — find minimum
                r_peaks.append(lo + np.argmin(window))
                n_neg += 1
            elif force_inv is False:
                # User says upright — find maximum
                r_peaks.append(lo + np.argmax(window))
            else:
                # Auto — pick largest absolute deflection from baseline
                idx_max = np.argmax(window)
                idx_min = np.argmin(window)
                if abs(window[idx_max] - baseline) >= abs(window[idx_min] - baseline):
                    r_peaks.append(lo + idx_max)
                else:
                    r_peaks.append(lo + idx_min)
                    n_neg += 1
        r_peaks = np.unique(r_peaks)
        is_inv = force_inv if force_inv is not None else (
            n_neg > len(candidates) // 2 if len(candidates) > 0 else False
        )

        # ── Update stats label ──
        if len(r_peaks) >= 2:
            rr_sec = np.diff(r_peaks) / sr
            mean_hr = 60.0 / np.mean(rr_sec)
        else:
            mean_hr = 0.0
        inv_str = "  (inverted)" if is_inv else ""
        n_beats = len(r_peaks)

        if n_beats == 0:
            self._stats_label.setText("No beats detected -- lower thresholds")
            self._stats_label.setStyleSheet(
                "background: #1a1a2e; color: #E53935; padding: 6px; "
                "border-radius: 4px; font-size: 12px; font-weight: bold;"
            )
        else:
            self._stats_label.setText(
                f"{n_beats} beats  |  {mean_hr:.0f} BPM{inv_str}"
            )
            self._stats_label.setStyleSheet(
                "background: #1a1a2e; color: #4CAF50; padding: 6px; "
                "border-radius: 4px; font-size: 12px; font-weight: bold;"
            )

        # ── Draw preview ──
        self._fig.clear()
        ax_style = dict(facecolor='#1a1a2e')

        def _style_ax(ax):
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='#888888', labelsize=6)
            for spine in ax.spines.values():
                spine.set_color('#333333')
            ax.grid(True, alpha=0.15, color='#555555')

        # Pick a 2-second window in the middle of the signal for detail view
        mid = n // 2
        win_samples = int(2.0 * sr)
        i0 = max(0, mid - win_samples // 2)
        i1 = min(n, i0 + win_samples)
        t_win = t[i0:i1]

        # ── Panel 1: raw (faint) + filtered + detected peaks ──
        ax1 = self._fig.add_subplot(3, 1, 1)
        ax1.plot(t_win, signal[i0:i1], linewidth=0.4, color='#555555',
                 alpha=0.4, label='Raw')
        ax1.plot(t_win, filtered[i0:i1], linewidth=1.0, color='#42A5F5',
                 label='Filtered')

        win_peaks = r_peaks[(r_peaks >= i0) & (r_peaks < i1)]
        if len(win_peaks):
            ax1.scatter(t[win_peaks], filtered[win_peaks],
                        s=30, c='#E53935', marker='v', zorder=6,
                        label=f'R-peaks ({n_beats} total)')

        ax1.set_ylabel('Signal', fontsize=7, color='#aaaaaa')
        ax1.legend(fontsize=6, loc='upper right',
                   facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#aaaaaa')
        _style_ax(ax1)

        # ── Panel 2: integrated signal + threshold ──
        ax2 = self._fig.add_subplot(3, 1, 2, sharex=ax1)
        ax2.plot(t_win, integrated[i0:i1], linewidth=0.8, color='#66BB6A',
                 label='Integrated')
        ax2.axhline(height_thresh, color='#FF9800', linewidth=1.0,
                     linestyle='--', label=f'Threshold ({config.threshold_fraction*100:.0f}%)',
                     alpha=0.9)

        win_cand = candidates[(candidates >= i0) & (candidates < i1)]
        if len(win_cand):
            ax2.scatter(t[win_cand], integrated[win_cand],
                        s=15, c='#E53935', marker='o', zorder=6)

        ax2.set_xlabel('Time (s)', fontsize=7, color='#aaaaaa')
        ax2.set_ylabel('Integrated', fontsize=7, color='#aaaaaa')
        ax2.legend(fontsize=6, loc='upper right',
                   facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#aaaaaa')
        _style_ax(ax2)

        # ── Panel 3: Cycle-Triggered Average (CTA) ──
        ax3 = self._fig.add_subplot(3, 1, 3)
        if len(r_peaks) >= 5:
            # Extract beats aligned to R-peak from filtered signal
            rr_median = int(np.median(np.diff(r_peaks)))
            half_before = rr_median // 2
            half_after = rr_median // 2

            beats = []
            for pk in r_peaks:
                lo = pk - half_before
                hi = pk + half_after
                if lo >= 0 and hi < len(filtered):
                    beats.append(filtered[lo:hi])

            if len(beats) >= 3:
                min_len = min(len(b) for b in beats)
                beats = [b[:min_len] for b in beats]
                beat_matrix = np.array(beats)
                t_beat_ms = (np.arange(min_len) - half_before) * (1000.0 / sr)

                # Individual beats (faint)
                max_show = min(50, len(beats))
                for b in beats[:max_show]:
                    ax3.plot(t_beat_ms, b, linewidth=0.3, color='#42A5F5', alpha=0.15)

                # Mean + SEM
                mean_beat = np.mean(beat_matrix, axis=0)
                sem_beat = np.std(beat_matrix, axis=0) / np.sqrt(len(beats))
                ax3.plot(t_beat_ms, mean_beat, linewidth=1.5, color='#FF9800',
                         label=f'Mean (n={len(beats)})')
                ax3.fill_between(t_beat_ms, mean_beat - sem_beat, mean_beat + sem_beat,
                                 color='#FF9800', alpha=0.2)
                ax3.axvline(0, color='#E53935', linewidth=0.8, linestyle='--',
                            alpha=0.7, label='R-peak')
            else:
                ax3.text(0.5, 0.5, 'Too few clean beats for CTA',
                         transform=ax3.transAxes, ha='center', va='center',
                         color='#888888', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Need 5+ beats for CTA',
                     transform=ax3.transAxes, ha='center', va='center',
                     color='#888888', fontsize=9)

        ax3.set_xlabel('Time from R-peak (ms)', fontsize=7, color='#aaaaaa')
        ax3.set_ylabel('Amplitude', fontsize=7, color='#aaaaaa')
        ax3.legend(fontsize=6, loc='upper right',
                   facecolor='#1a1a2e', edgecolor='#333333', labelcolor='#aaaaaa')
        _style_ax(ax3)

        self._fig.tight_layout(pad=0.3, h_pad=0.4)
        self._canvas.draw_idle()

    # ── Public API ──

    def update_stats(self, n_beats: int, hr_bpm: float, quality_pct: int,
                     is_inverted: bool = False):
        """Update stats from main window (called after Apply)."""
        inv = "  (inverted)" if is_inverted else ""
        if n_beats == 0:
            self._stats_label.setText("No beats detected -- try lowering thresholds")
            self._stats_label.setStyleSheet(
                "background: #1a1a2e; color: #E53935; padding: 6px; "
                "border-radius: 4px; font-size: 12px; font-weight: bold;"
            )
        else:
            self._stats_label.setText(
                f"{n_beats} beats  |  {hr_bpm:.0f} BPM  |  Quality: {quality_pct}%{inv}"
            )
            color = "#4CAF50" if quality_pct >= 70 else "#FF9800"
            self._stats_label.setStyleSheet(
                f"background: #1a1a2e; color: {color}; padding: 6px; "
                "border-radius: 4px; font-size: 12px; font-weight: bold;"
            )

    @property
    def config(self):
        return self._config
