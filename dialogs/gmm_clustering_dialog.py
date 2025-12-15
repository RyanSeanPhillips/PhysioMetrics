"""
GMM Clustering Dialog for PhysioMetrics.

This dialog provides an interface for Gaussian Mixture Model (GMM) clustering
of breath patterns to identify eupnea and sniffing behaviors.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton,
    QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox, QRadioButton,
    QButtonGroup, QDoubleSpinBox, QScrollArea, QWidget, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from core import metrics, filters


class GMMClusteringDialog(QDialog):
    def __init__(self, parent=None, main_window=None):
        from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSpinBox, QPushButton, QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox, QRadioButton, QButtonGroup, QDoubleSpinBox
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QSizePolicy
        import numpy as np

        super().__init__(parent)
        self.setWindowTitle("Eupnea/Sniffing Detection")
        self.resize(1200, 800)  # Adjusted to fit laptop screens comfortably

        # Apply VS Code Dark+ theme to match main window
        self.setStyleSheet("""
            /* VS Code Dark+ Theme */
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
            }
            QGroupBox {
                color: #d4d4d4;
                border: 1px solid white;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 5px;
                color: #4A9EFF;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid white;
                border-radius: 2px;
                background-color: #252526;
            }
            QCheckBox::indicator:checked {
                background-color: #4A9EFF;
                border: 1px solid white;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #6BB6FF;
            }
            QCheckBox:disabled {
                color: #6a6a6a;
            }
            QCheckBox::indicator:disabled {
                border: 1px solid #6a6a6a;
                background-color: #1e1e1e;
            }
            QRadioButton {
                color: #cccccc;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 3px;
            }
            QPushButton {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3e3e42;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QTableWidget {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                gridline-color: #3e3e42;
            }
            QTableWidget::item {
                color: #cccccc;
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #094771;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 5px;
                font-weight: bold;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 14px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #3e3e42;
                border-radius: 7px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #4e4e52;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.main_window = main_window
        self.cluster_labels = None  # Will store cluster assignments
        self.gmm_model = None
        self.feature_data = None  # Per-breath feature matrix
        self.breath_cycles = []  # List of (sweep_idx, breath_idx) tuples
        self.sniffing_cluster_id = None  # Which cluster represents sniffing
        self.cluster_colors = {}  # Map cluster_id -> color (purple for sniffing, green for eupnea)
        self.had_quality_warning = False  # Track if clustering quality was questionable
        self.cluster_custom_labels = {}  # Store user-customized cluster labels (cluster_id -> custom_label)

        # Main layout
        main_layout = QHBoxLayout(self)

        # Left panel: Controls - compact spacing
        left_panel = QVBoxLayout()
        left_panel.setSpacing(6)  # Reduced from 10 to 6

        # Help/Info button at top
        help_layout = QHBoxLayout()
        help_btn = QPushButton("ℹ️ What is GMM Clustering?")
        help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        help_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                background: transparent;
                text-decoration: underline;
                color: #4A9EFF;
                font-size: 10pt;
            }
            QPushButton:hover {
                color: #6BB6FF;
            }
        """)
        help_btn.clicked.connect(self.show_help)
        help_layout.addWidget(help_btn)
        help_layout.addStretch()
        left_panel.addLayout(help_layout)

        # Feature selection group
        feature_group = QGroupBox("Select Features for Clustering")
        feature_layout = QGridLayout()  # Changed from QVBoxLayout to QGridLayout for multi-column layout

        self.feature_checkboxes = {}
        available_features = ["if", "ti", "te", "amp_insp", "amp_exp", "area_insp", "area_exp", "max_dinsp", "max_dexp"]
        default_features = ["if", "ti", "amp_insp", "max_dinsp"]  # Streamlined defaults for eupnea/sniffing separation

        # Arrange features in 3 columns
        num_cols = 3
        for idx, feature in enumerate(available_features):
            cb = QCheckBox(feature)
            cb.setChecked(feature in default_features)
            self.feature_checkboxes[feature] = cb
            row = idx // num_cols
            col = idx % num_cols
            feature_layout.addWidget(cb, row, col)

        feature_group.setLayout(feature_layout)
        left_panel.addWidget(feature_group)

        # Clustering controls - compact layout with button on same row
        cluster_controls_layout = QHBoxLayout()
        cluster_controls_layout.addWidget(QLabel("Clusters:"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 10)
        self.n_clusters_spin.setValue(2)  # Start with 2 for eupnea/sniffing
        self.n_clusters_spin.setMaximumWidth(60)
        cluster_controls_layout.addWidget(self.n_clusters_spin)

        cluster_controls_layout.addSpacing(10)

        # Run GMM button - compact, on same row
        self.run_gmm_btn = QPushButton("Run GMM")
        self.run_gmm_btn.clicked.connect(self.on_run_gmm)
        self.run_gmm_btn.setMinimumHeight(30)
        # Override theme style for this button with blue highlight
        self.run_gmm_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A9EFF;
                color: white;
                font-weight: bold;
                border-radius: 3px;
                border: 1px solid #3A8EEF;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #6BB6FF;
                border: 1px solid #4A9EFF;
            }
            QPushButton:pressed {
                background-color: #3A8EEF;
                border: 1px solid #2A7EDF;
            }
        """)
        cluster_controls_layout.addWidget(self.run_gmm_btn)
        cluster_controls_layout.addStretch()
        left_panel.addLayout(cluster_controls_layout)

        # Results table
        results_label = QLabel("Cluster Statistics:")
        results_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        left_panel.addWidget(results_label)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Cluster", "Count", "Percentage", "Avg Confidence"])

        # Configure table to fit content without horizontal scrolling
        from PyQt6.QtWidgets import QHeaderView
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Set initial height for empty table (will be adjusted dynamically)
        self.results_table.setMaximumHeight(150)

        # Connect signal to capture user edits to cluster labels
        self.results_table.itemChanged.connect(self._on_table_item_changed)

        left_panel.addWidget(self.results_table)

        # Quality Metrics Display - compact 2-column layout
        quality_group = QGroupBox("Quality Metrics")
        quality_layout = QGridLayout()

        # Row 0: Silhouette and Cohen's d side by side
        self.silhouette_label = QLabel("Silhouette: —")
        self.silhouette_label.setToolTip(
            "Cluster separation (-1 to +1)\n"
            ">0.5=Excellent, 0.25-0.5=Good\n"
            "0.15-0.25=Warning, <0.15=Reject"
        )
        quality_layout.addWidget(self.silhouette_label, 0, 0)

        self.cohens_d_label = QLabel("Cohen's d: —")
        self.cohens_d_label.setToolTip(
            "Effect size (0 to ∞)\n"
            ">0.8=Large, 0.5-0.8=Medium\n"
            "<0.5=Small"
        )
        quality_layout.addWidget(self.cohens_d_label, 0, 1)

        # Row 1: Verdict spans both columns
        self.quality_verdict_label = QLabel("Verdict: —")
        self.quality_verdict_label.setWordWrap(True)
        quality_layout.addWidget(self.quality_verdict_label, 1, 0, 1, 2)  # Span 2 columns

        quality_group.setLayout(quality_layout)
        left_panel.addWidget(quality_group)

        # Confidence Threshold Control - compact
        confidence_group = QGroupBox("Confidence Threshold")
        confidence_layout = QHBoxLayout()

        confidence_layout.addWidget(QLabel("Min prob:"))

        self.confidence_threshold_slider = QDoubleSpinBox()
        self.confidence_threshold_slider.setRange(0.50, 0.95)
        self.confidence_threshold_slider.setValue(0.50)  # Default: >50% probability = sniffing
        self.confidence_threshold_slider.setSingleStep(0.05)
        self.confidence_threshold_slider.setDecimals(2)
        self.confidence_threshold_slider.setMaximumWidth(70)
        self.confidence_threshold_slider.setToolTip(
            "Minimum probability to mark breath as sniffing\n"
            "Higher = more conservative\n"
            "Lower = more liberal"
        )
        confidence_layout.addWidget(self.confidence_threshold_slider)

        self.confidence_threshold_label = QLabel("(50%)")
        self.confidence_threshold_label.setStyleSheet("color: #4A9EFF; font-weight: bold;")
        confidence_layout.addWidget(self.confidence_threshold_label)

        # Connect slider to update label
        self.confidence_threshold_slider.valueChanged.connect(
            lambda val: self.confidence_threshold_label.setText(f"({int(val*100)}%)")
        )

        confidence_layout.addStretch()
        confidence_group.setLayout(confidence_layout)
        left_panel.addWidget(confidence_group)

        # Status label
        self.status_label = QLabel("Select features and click 'Run GMM Clustering' to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #888; font-style: italic;")
        left_panel.addWidget(self.status_label)

        # Info label about what this dialog does
        info_label = QLabel("Note: Sniffing breaths will be marked with purple background, eupnea with green overlay.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #888; font-size: 9pt; font-style: italic;")
        left_panel.addWidget(info_label)

        # Waveform plotting controls
        waveform_group = QGroupBox("Waveform Visualization")
        waveform_layout = QVBoxLayout()

        self.plot_waveforms_cb = QCheckBox("Plot Mean Waveforms")
        self.plot_waveforms_cb.setChecked(True)  # Enabled by default
        self.plot_waveforms_cb.setToolTip("Enable to plot mean waveforms with variability bands")
        waveform_layout.addWidget(self.plot_waveforms_cb)

        n_breaths_layout = QHBoxLayout()
        n_breaths_layout.addWidget(QLabel("Max breaths per group:"))
        self.n_breaths_spin = QSpinBox()
        self.n_breaths_spin.setRange(10, 500)
        self.n_breaths_spin.setValue(25)
        self.n_breaths_spin.setToolTip("Number of breaths to include from each group (eupnea/sniffing)")
        n_breaths_layout.addWidget(self.n_breaths_spin)
        n_breaths_layout.addStretch()
        waveform_layout.addLayout(n_breaths_layout)

        # Variability display mode
        variability_label = QLabel("Variability bands:")
        waveform_layout.addWidget(variability_label)

        self.variability_button_group = QButtonGroup(self)
        self.sem_radio = QRadioButton("SEM (Standard Error)")
        self.sem_radio.setChecked(True)  # Default
        self.sem_radio.setToolTip("Show mean ± standard error (measures precision of mean estimate)")
        self.std_radio = QRadioButton("STD (Standard Deviation)")
        self.std_radio.setToolTip("Show mean ± standard deviation (measures data spread)")
        self.minmax_radio = QRadioButton("Min/Max Range")
        self.minmax_radio.setToolTip("Show full range of variation (min to max values)")

        self.variability_button_group.addButton(self.sem_radio, 0)
        self.variability_button_group.addButton(self.std_radio, 1)
        self.variability_button_group.addButton(self.minmax_radio, 2)

        # Arrange radio buttons in horizontal layout
        variability_radios_layout = QHBoxLayout()
        variability_radios_layout.addWidget(self.sem_radio)
        variability_radios_layout.addWidget(self.std_radio)
        variability_radios_layout.addWidget(self.minmax_radio)
        variability_radios_layout.addStretch()
        waveform_layout.addLayout(variability_radios_layout)

        # Apply button to re-plot without re-running GMM
        self.apply_variability_btn = QPushButton("Apply")
        self.apply_variability_btn.setToolTip("Re-plot waveforms with current settings (variability mode and number of breaths)")
        self.apply_variability_btn.clicked.connect(self._on_apply_variability)
        waveform_layout.addWidget(self.apply_variability_btn)

        # Connect n_breaths spinner to trigger re-plot via Apply button logic
        # (User can change value and click Apply to see updated plot)
        # Note: We don't auto-update on every spinner change to avoid performance issues

        waveform_group.setLayout(waveform_layout)
        left_panel.addWidget(waveform_group)

        # Sniffing Application Controls
        sniff_app_group = QGroupBox("Sniffing Region Application")
        sniff_app_layout = QVBoxLayout()

        # Checkbox to enable/disable eupnea/sniffing detection application
        self.apply_sniffing_cb = QCheckBox("Apply Eupnea/Sniffing Detection to Plot")
        self.apply_sniffing_cb.setChecked(True)  # Default to ON
        self.apply_sniffing_cb.setToolTip("When enabled, detected eupnea (green) and sniffing (purple) regions will be marked on the main plot.\n"
                                          "Turn off if baseline respiratory rate is high or plot is too cluttered.")
        sniff_app_layout.addWidget(self.apply_sniffing_cb)

        # Checkbox to enable/disable automatic GMM update when peaks are manually edited
        self.auto_update_gmm_cb = QCheckBox("Auto-Update GMM on Peak Edits")
        self.auto_update_gmm_cb.setChecked(False)  # Default to OFF for performance
        self.auto_update_gmm_cb.setToolTip("When enabled, GMM clustering will automatically rerun and apply sniffing regions\n"
                                           "whenever you manually add or delete peaks.\n"
                                           "Disable for faster editing with large files.")
        sniff_app_layout.addWidget(self.auto_update_gmm_cb)

        # Button to clear existing sniffing regions
        self.clear_sniffing_btn = QPushButton("Clear All Sniffing Regions")
        self.clear_sniffing_btn.setToolTip("Remove all sniffing region markings from the main plot")
        self.clear_sniffing_btn.clicked.connect(self._on_clear_sniffing_regions)
        sniff_app_layout.addWidget(self.clear_sniffing_btn)

        sniff_app_group.setLayout(sniff_app_layout)
        left_panel.addWidget(sniff_app_group)

        # Eupnea detection mode selection (moved to bottom)
        mode_group = QGroupBox("Eupnea Detection Method")
        mode_layout = QVBoxLayout()

        self.detection_mode_button_group = QButtonGroup(self)
        self.gmm_mode_radio = QRadioButton("GMM-Based (Automatic)")
        self.gmm_mode_radio.setToolTip("Use GMM clustering to identify eupnea automatically.\n"
                                       "Breaths NOT classified as sniffing are marked as eupnea.")
        self.freq_mode_radio = QRadioButton("Frequency-Based (Manual Threshold)")
        self.freq_mode_radio.setToolTip("Use manual frequency threshold to identify eupnea.\n"
                                        "Breaths below frequency threshold are marked as eupnea.")

        self.detection_mode_button_group.addButton(self.gmm_mode_radio, 0)
        self.detection_mode_button_group.addButton(self.freq_mode_radio, 1)

        # Set initial selection based on main window's current mode
        if hasattr(main_window, 'eupnea_detection_mode') and main_window.eupnea_detection_mode == "frequency":
            self.freq_mode_radio.setChecked(True)
        else:
            self.gmm_mode_radio.setChecked(True)

        mode_layout.addWidget(self.gmm_mode_radio)
        gmm_info = QLabel("  → Automatic classification from clustering")
        gmm_info.setStyleSheet("color: #888; font-size: 9pt; margin-left: 20px;")
        mode_layout.addWidget(gmm_info)

        mode_layout.addWidget(self.freq_mode_radio)

        # Frequency-based parameters (enabled/disabled based on mode selection)
        freq_params_layout = QVBoxLayout()
        freq_params_layout.setContentsMargins(30, 5, 0, 0)  # Indent for visual grouping

        freq_thresh_layout = QHBoxLayout()
        freq_thresh_layout.addWidget(QLabel("Frequency Threshold (Hz):"))
        self.freq_threshold_spin = QDoubleSpinBox()
        self.freq_threshold_spin.setRange(0.1, 20.0)
        self.freq_threshold_spin.setValue(getattr(main_window, 'eupnea_freq_threshold', 5.0))
        self.freq_threshold_spin.setDecimals(1)
        self.freq_threshold_spin.setSingleStep(0.5)
        self.freq_threshold_spin.setToolTip("Breathing below this frequency is considered eupneic")
        freq_thresh_layout.addWidget(self.freq_threshold_spin)
        freq_thresh_layout.addStretch()
        freq_params_layout.addLayout(freq_thresh_layout)

        min_dur_layout = QHBoxLayout()
        min_dur_layout.addWidget(QLabel("Minimum Duration (s):"))
        self.min_duration_spin = QDoubleSpinBox()
        self.min_duration_spin.setRange(0.5, 10.0)
        self.min_duration_spin.setValue(main_window.eupnea_min_duration if hasattr(main_window, 'eupnea_min_duration') else 2.0)
        self.min_duration_spin.setDecimals(1)
        self.min_duration_spin.setSingleStep(0.5)
        self.min_duration_spin.setToolTip("Region must sustain criteria for at least this long")
        min_dur_layout.addWidget(self.min_duration_spin)
        min_dur_layout.addStretch()
        freq_params_layout.addLayout(min_dur_layout)

        mode_layout.addLayout(freq_params_layout)

        mode_group.setLayout(mode_layout)
        left_panel.addWidget(mode_group)

        # Connect mode radio buttons to auto-apply changes
        self.gmm_mode_radio.toggled.connect(self._on_detection_mode_changed)
        self.freq_mode_radio.toggled.connect(self._on_detection_mode_changed)

        # Connect frequency parameters to auto-apply changes
        self.freq_threshold_spin.valueChanged.connect(self._on_frequency_params_changed)
        self.min_duration_spin.valueChanged.connect(self._on_frequency_params_changed)

        # Connect sniffing application checkbox to apply/remove sniffing immediately
        self.apply_sniffing_cb.toggled.connect(self._on_sniffing_application_toggled)

        # Connect auto-update GMM checkbox to main window state
        self.auto_update_gmm_cb.toggled.connect(self._on_auto_update_gmm_toggled)

        # Load current auto-update state from main window
        if hasattr(self.main_window, 'auto_gmm_enabled'):
            self.auto_update_gmm_cb.setChecked(self.main_window.auto_gmm_enabled)

        # Close button - styled for visibility
        button_layout = QHBoxLayout()
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(35)
        close_btn.clicked.connect(self.accept)
        # Style to make it stand out
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d2d30;
                color: #ffffff;
                font-weight: bold;
                border: 2px solid #4A9EFF;
                border-radius: 3px;
                padding: 5px 15px;
                font-size: 11pt;
            }
            QPushButton:hover {
                background-color: #3e3e42;
                border: 2px solid #6BB6FF;
            }
            QPushButton:pressed {
                background-color: #094771;
                border: 2px solid #4A9EFF;
            }
        """)
        button_layout.addWidget(close_btn)
        left_panel.addLayout(button_layout)

        left_panel.addStretch()

        # Wrap left panel in a scroll area
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll_area.setMinimumWidth(380)  # Minimum width to fit content
        left_scroll_area.setMaximumWidth(400)  # Maximum width, plots get more space

        # Container widget for left panel controls
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setMaximumWidth(380)  # Match scroll area to prevent overflow
        left_scroll_area.setWidget(left_container)

        main_layout.addWidget(left_scroll_area, 1)

        # Right panel: Scrollable Visualizations
        # Note: QScrollArea, QWidget, FigureCanvasQTAgg, Figure already imported at top

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # No horizontal scroll
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Enable mouse wheel scrolling

        # Container widget for plots
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib figure for visualizations (will contain both histograms and scatter plots)
        # Fit width to window, tall for scrolling
        self.figure = Figure(figsize=(9, 25))  # Narrower to fit width, tall for vertical scroll
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(2000)  # Tall enough to trigger vertical scrolling

        # Install event filter to forward mouse wheel events to scroll area
        self.canvas.installEventFilter(self)

        plot_layout.addWidget(self.canvas)

        self.scroll_area.setWidget(plot_container)
        main_layout.addWidget(self.scroll_area, 2)

        # Auto-run GMM visualization if results already exist (from automatic clustering)
        # This happens after peak detection, so the dialog can immediately show results
        if hasattr(main_window.state, 'gmm_sniff_probabilities') and main_window.state.gmm_sniff_probabilities:
            print("[gmm-dialog] Found existing GMM results, auto-loading visualization...")
            self._load_existing_gmm_results()

        # Enable dark title bar on Windows
        self._enable_dark_title_bar()

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

    def _load_existing_gmm_results(self):
        """Load and display existing GMM results that were computed automatically."""
        import numpy as np

        try:
            # Check if main window has cached GMM results
            if (hasattr(self.main_window, '_cached_gmm_results') and
                self.main_window._cached_gmm_results is not None):

                print("[gmm-dialog] Loading cached GMM results (fast path)...")

                # Unpack cached results
                cached = self.main_window._cached_gmm_results
                self.cluster_labels = cached['cluster_labels']
                self.cluster_probabilities = cached['cluster_probabilities']
                self.feature_data = cached['feature_matrix']
                self.breath_cycles = cached['breath_cycles']
                self.sniffing_cluster_id = cached['sniffing_cluster_id']
                selected_features = cached['feature_keys']

                # Assign colors based on cached sniffing cluster
                for cluster_id in np.unique(self.cluster_labels):
                    if cluster_id == self.sniffing_cluster_id:
                        self.cluster_colors[cluster_id] = 'purple'
                    else:
                        self.cluster_colors[cluster_id] = 'green'

                # Calculate and display quality metrics
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                feature_matrix_scaled = scaler.fit_transform(self.feature_data)

                quality_status, silhouette, cohens_d, quality_warning = self._check_clustering_quality(
                    feature_matrix_scaled, selected_features
                )
                self._update_quality_metrics_display(quality_status, silhouette, cohens_d)

                # Update results table
                self._update_results_table()

                # Plot results (only plotting, no re-computation)
                self._plot_clusters(self.feature_data, self.cluster_labels, selected_features)

                self.status_label.setText(f"✓ Loaded cached results: {len(self.feature_data)} breaths classified into {len(np.unique(self.cluster_labels))} clusters.")
                self.status_label.setStyleSheet("color: green; font-weight: bold;")

            else:
                print("[gmm-dialog] No cached results found, user needs to run GMM manually")
                self.status_label.setText("No existing results. Click 'Run GMM Clustering' to analyze breaths.")
                self.status_label.setStyleSheet("color: #888; font-style: italic;")

            # Scroll to top
            self.scroll_area.verticalScrollBar().setValue(0)

        except Exception as e:
            print(f"[gmm-dialog] Error loading existing results: {e}")
            import traceback
            traceback.print_exc()

    def eventFilter(self, obj, event):
        """Forward wheel events from canvas to scroll area."""
        from PyQt6.QtCore import QEvent
        if obj == self.canvas and event.type() == QEvent.Type.Wheel:
            # Forward wheel event to scroll area
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - event.angleDelta().y() // 2
            )
            return True
        return super().eventFilter(obj, event)

    def _on_detection_mode_changed(self):
        """Enable/disable frequency parameters and auto-apply detection mode change."""
        gmm_mode = self.gmm_mode_radio.isChecked()
        self.freq_threshold_spin.setEnabled(not gmm_mode)
        self.min_duration_spin.setEnabled(not gmm_mode)

        # Auto-apply detection mode change to main window
        selected_mode = "gmm" if gmm_mode else "frequency"
        if self.main_window.eupnea_detection_mode != selected_mode:
            self.main_window.eupnea_detection_mode = selected_mode
            print(f"[gmm-dialog] Auto-applied eupnea detection mode: {selected_mode}")

            # Update main window's eupnea regions and redraw
            self.main_window.redraw_main_plot()

    def _on_frequency_params_changed(self):
        """Auto-apply frequency parameter changes to main window."""
        # Only apply if in frequency mode
        if self.freq_mode_radio.isChecked():
            # Update main window parameters
            self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
            self.main_window.eupnea_min_duration = self.min_duration_spin.value()

            print(f"[gmm-dialog] Auto-applied frequency params: thresh={self.main_window.eupnea_freq_threshold} Hz, "
                  f"min_dur={self.main_window.eupnea_min_duration} s")

            # Redraw to apply new parameters
            self.main_window.redraw_main_plot()

    def _on_clear_sniffing_regions(self):
        """Clear all eupnea and sniffing regions from the main plot."""
        regions_cleared = False

        # Clear all sniffing regions
        if hasattr(self.main_window.state, 'sniff_regions_by_sweep'):
            self.main_window.state.sniff_regions_by_sweep.clear()
            regions_cleared = True

        # Clear all eupnea regions
        if hasattr(self.main_window.state, 'eupnea_regions_by_sweep'):
            self.main_window.state.eupnea_regions_by_sweep.clear()
            regions_cleared = True

        # Clear GMM probabilities as well
        if hasattr(self.main_window.state, 'gmm_sniff_probabilities'):
            self.main_window.state.gmm_sniff_probabilities.clear()

        if regions_cleared:
            print("[gmm-dialog] Cleared all eupnea and sniffing regions")
        else:
            print("[gmm-dialog] No eupnea/sniffing regions to clear")

        # Redraw main plot to remove green/purple markings
        self.main_window.redraw_main_plot()

    def _on_sniffing_application_toggled(self, checked):
        """Handle toggling of the Apply Eupnea/Sniffing checkbox.

        When checked: Apply eupnea and sniffing detection to main plot immediately
        When unchecked: Clear all eupnea and sniffing regions from main plot
        """
        if checked:
            # User wants to apply eupnea/sniffing detection
            # Only apply if we have GMM results
            if self.cluster_labels is not None and self.sniffing_cluster_id is not None:
                print("[gmm-dialog] Eupnea/sniffing application enabled - applying to plot")
                self._apply_sniffing_to_plot()
            else:
                print("[gmm-dialog] Eupnea/sniffing application enabled, but no GMM results available yet")
        else:
            # User wants to remove eupnea/sniffing detection
            print("[gmm-dialog] Eupnea/sniffing application disabled - clearing regions")
            self._on_clear_sniffing_regions()

    def _on_auto_update_gmm_toggled(self, checked):
        """Handle toggling of the Auto-Update GMM checkbox.

        Synchronizes the state with the main window so editing_modes can check it.
        """
        if hasattr(self.main_window, 'auto_gmm_enabled'):
            self.main_window.auto_gmm_enabled = checked
            status = "enabled" if checked else "disabled"
            print(f"[gmm-dialog] Auto-Update GMM {status}")
            self.main_window.statusBar().showMessage(f"Auto-Update GMM {status}", 2000)

    def _apply_sniffing_to_plot(self):
        """Apply sniffing regions to main plot using shared core.gmm_clustering functions.

        Uses the confidence threshold slider to determine which breaths to mark as sniffing.
        """
        import numpy as np
        from core import gmm_clustering

        if self.sniffing_cluster_id is None:
            print("[gmm-dialog] Cannot apply: no sniffing cluster identified")
            return

        # Get confidence threshold from slider
        confidence_threshold = self.confidence_threshold_slider.value()
        print(f"[gmm-dialog] Using confidence threshold: {confidence_threshold:.2f}")

        # Store GMM classifications in all_peaks_by_sweep using shared function
        n_classified = gmm_clustering.store_gmm_classifications_in_peaks(
            self.main_window.state,
            self.breath_cycles,
            self.cluster_labels,
            self.sniffing_cluster_id,
            self.cluster_probabilities,
            confidence_threshold=confidence_threshold
        )

        # Build BOTH eupnea AND sniffing regions using shared function
        results = gmm_clustering.build_eupnea_sniffing_regions(
            self.main_window.state,
            verbose=True,
            log_prefix="[gmm-dialog]"
        )

        # Redraw main plot to show eupnea and sniffing regions
        self.main_window.redraw_main_plot()

        # Log summary results
        n_total_sniff_cluster = np.sum(self.cluster_labels == self.sniffing_cluster_id)
        n_sweeps_with_sniffing = len(self.main_window.state.sniff_regions_by_sweep)
        n_sweeps_with_eupnea = len(self.main_window.state.eupnea_regions_by_sweep)
        print(f"[gmm-dialog] Total breaths in sniffing cluster: {n_total_sniff_cluster}")
        print(f"[gmm-dialog] Classified {n_classified} breaths using threshold {confidence_threshold:.2f}")
        print(f"[gmm-dialog] Sniffing breaths: {results['n_sniffing']}")
        print(f"[gmm-dialog] Eupnea breaths: {results['n_eupnea']}")
        print(f"[gmm-dialog] Created {results['total_sniff_regions']} sniffing regions across {n_sweeps_with_sniffing} sweep(s)")
        print(f"[gmm-dialog] Created {results['total_eupnea_regions']} eupnea regions across {n_sweeps_with_eupnea} sweep(s)")

    def _on_apply_variability(self):
        """Re-plot waveforms with new variability mode and number of breaths without re-running GMM."""
        import numpy as np

        # Check if we have clustering results to plot
        if self.cluster_labels is None or self.feature_data is None:
            print("[gmm-dialog] No clustering results to re-plot")
            return

        # Get selected features from previous GMM run
        # We need to reconstruct the feature list from cached results
        if hasattr(self.main_window, '_cached_gmm_results') and self.main_window._cached_gmm_results is not None:
            selected_features = self.main_window._cached_gmm_results['feature_keys']
        else:
            # Fallback: use currently selected features (may not match)
            selected_features = [f for f, cb in self.feature_checkboxes.items() if cb.isChecked()]

        # Determine which variability mode is selected
        if self.minmax_radio.isChecked():
            mode_name = "Min/Max"
        elif self.std_radio.isChecked():
            mode_name = "STD"
        else:
            mode_name = "SEM"

        print(f"[gmm-dialog] Re-plotting with variability mode: {mode_name}, max breaths: {self.n_breaths_spin.value()}")

        # Re-plot with current variability settings and number of breaths
        self._plot_clusters(self.feature_data, self.cluster_labels, selected_features)

        # Scroll to top after re-plotting
        self.scroll_area.verticalScrollBar().setValue(0)

    def on_run_gmm(self):
        """Run GMM clustering on breath metrics."""
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Get selected features
        selected_features = [f for f, cb in self.feature_checkboxes.items() if cb.isChecked()]
        if len(selected_features) < 2:
            self.status_label.setText("Error: Please select at least 2 features.")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return

        # Collect breath metrics from all analyzed sweeps
        self.status_label.setText("Collecting breath metrics...")
        self.status_label.setStyleSheet("color: blue;")

        try:
            feature_matrix, breath_cycles = self._collect_breath_features(selected_features)

            if len(feature_matrix) < self.n_clusters_spin.value():
                self.status_label.setText(f"Error: Not enough breaths ({len(feature_matrix)}) for {self.n_clusters_spin.value()} clusters.")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")
                return

            # Standardize features
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            # Fit GMM
            n_clusters = self.n_clusters_spin.value()
            self.status_label.setText(f"Fitting GMM with {n_clusters} clusters...")

            self.gmm_model = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
            self.cluster_labels = self.gmm_model.fit_predict(feature_matrix_scaled)
            self.cluster_probabilities = self.gmm_model.predict_proba(feature_matrix_scaled)
            self.feature_data = feature_matrix
            self.breath_cycles = breath_cycles

            # Identify sniffing cluster (must be done before quality check to compute Cohen's d)
            self._identify_sniffing_cluster(feature_matrix, selected_features, None)

            # Check clustering quality and get metrics
            quality_status, silhouette, cohens_d, quality_warning = self._check_clustering_quality(
                feature_matrix_scaled, selected_features
            )

            # Update quality metrics display
            self._update_quality_metrics_display(quality_status, silhouette, cohens_d)

            # Handle quality status
            if quality_status == "REJECT":
                # Show rejection message
                QMessageBox.critical(
                    self,
                    "Clustering Rejected - No Clear Separation",
                    quality_warning
                )
                self.status_label.setText("❌ Clustering rejected - no clear sniffing pattern detected")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")

                # Still show plots for inspection, but don't apply
                self._update_results_table()
                self._plot_clusters(feature_matrix, self.cluster_labels, selected_features)
                self.scroll_area.verticalScrollBar().setValue(0)
                return  # Don't auto-apply

            elif quality_status == "WARNING":
                # Show warning
                self.had_quality_warning = True
                from PyQt6.QtWidgets import QMessageBox
                msg = QMessageBox(self)
                msg.setWindowTitle("Clustering Quality Warning")
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setText(quality_warning)
                msg.setInformativeText("You can still explore the results, but be cautious about applying them to the plot.")
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
            else:
                self.had_quality_warning = False

            # Update results table
            self._update_results_table()

            # Plot results (histograms + scatter plots)
            self._plot_clusters(feature_matrix, self.cluster_labels, selected_features)

            self.status_label.setText(f"✓ Clustering complete: {len(feature_matrix)} breaths classified into {n_clusters} clusters.")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")

            # Auto-apply results to main plot (only if quality is good or user accepted warning)
            self._auto_apply_gmm_results()

            # Scroll to top after plotting
            self.scroll_area.verticalScrollBar().setValue(0)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            import traceback
            traceback.print_exc()

    def _check_clustering_quality(self, feature_matrix_scaled, feature_keys):
        """Check if clustering is meaningful (not just splitting homogeneous data).

        Returns: (status, silhouette, cohens_d, warning_message)
            status: "REJECT", "WARNING", or "GOOD"
            silhouette: float (-1 to 1)
            cohens_d: float (0 to infinity)
            warning_message: str or None
        """
        import numpy as np
        from sklearn.metrics import silhouette_score

        # Silhouette score measures how well-separated clusters are
        # Range: -1 (poor) to +1 (excellent), ~0 means overlapping clusters
        if len(np.unique(self.cluster_labels)) > 1:
            silhouette = silhouette_score(feature_matrix_scaled, self.cluster_labels)
        else:
            silhouette = -1

        # Calculate Cohen's d (effect size) for IF separation
        cohens_d = 0.0
        if_idx = feature_keys.index('if') if 'if' in feature_keys else None

        if if_idx is not None and self.sniffing_cluster_id is not None:
            sniff_mask = self.cluster_labels == self.sniffing_cluster_id
            eupnea_mask = ~sniff_mask

            sniff_if = feature_matrix_scaled[sniff_mask, if_idx]
            eupnea_if = feature_matrix_scaled[eupnea_mask, if_idx]

            if len(sniff_if) > 0 and len(eupnea_if) > 0:
                mean_diff = abs(np.mean(sniff_if) - np.mean(eupnea_if))
                pooled_std = np.sqrt((np.var(sniff_if) + np.var(eupnea_if)) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        print(f"[gmm-clustering] Silhouette score: {silhouette:.3f}, Cohen's d: {cohens_d:.3f}")

        # Quality decision logic
        if silhouette < 0.15 and cohens_d < 0.5:
            # REJECT: No clear separation
            status = "REJECT"
            warning = (
                f"⚠️ **CLUSTERING REJECTED** ⚠️\n\n"
                f"Silhouette score: {silhouette:.3f} (< 0.15)\n"
                f"Effect size (Cohen's d): {cohens_d:.3f} (< 0.5)\n\n"
                f"This suggests there is NO clear sniffing pattern in this data.\n"
                f"GMM is forcing a split where none exists.\n\n"
                f"**Possible reasons:**\n"
                f"• This recording has no sniffing bouts\n"
                f"• Baseline respiratory rate is too high to distinguish sniffing\n"
                f"• All breathing is of similar pattern\n\n"
                f"**Recommendation:** Do not apply these results to the plot."
            )
        elif silhouette < 0.25 or cohens_d < 0.8:
            # WARNING: Borderline separation
            status = "WARNING"
            warning = (
                f"⚠️ Low cluster separation detected\n\n"
                f"Silhouette score: {silhouette:.3f}\n"
                f"Effect size (Cohen's d): {cohens_d:.3f}\n\n"
                f"The breathing patterns are somewhat similar.\n"
                f"The identified 'sniffing' cluster may just be normal variation,\n"
                f"not distinct sniffing behavior.\n\n"
                f"Consider reviewing the results carefully before applying."
            )
        else:
            # GOOD: Clear separation
            status = "GOOD"
            warning = None

        return (status, silhouette, cohens_d, warning)

    def _update_quality_metrics_display(self, quality_status, silhouette, cohens_d):
        """Update the quality metrics display panel with current values."""
        # Update silhouette label with color coding
        silhouette_text = f"Silhouette: {silhouette:.3f}"
        if silhouette >= 0.5:
            silhouette_color = "#4CAF50"  # Green - excellent
        elif silhouette >= 0.25:
            silhouette_color = "#8BC34A"  # Light green - good
        elif silhouette >= 0.15:
            silhouette_color = "#FFC107"  # Orange - warning
        else:
            silhouette_color = "#F44336"  # Red - reject

        self.silhouette_label.setText(silhouette_text)
        self.silhouette_label.setStyleSheet(f"color: {silhouette_color}; font-weight: bold;")

        # Update Cohen's d label with color coding
        cohens_d_text = f"Cohen's d: {cohens_d:.3f}"
        if cohens_d >= 0.8:
            cohens_d_color = "#4CAF50"  # Green - large effect
        elif cohens_d >= 0.5:
            cohens_d_color = "#FFC107"  # Orange - medium effect
        else:
            cohens_d_color = "#F44336"  # Red - small effect

        self.cohens_d_label.setText(cohens_d_text)
        self.cohens_d_label.setStyleSheet(f"color: {cohens_d_color}; font-weight: bold;")

        # Update quality verdict (compact)
        if quality_status == "GOOD":
            verdict_text = "✓ GOOD"
            verdict_color = "#4CAF50"
        elif quality_status == "WARNING":
            verdict_text = "⚠ WARNING"
            verdict_color = "#FFC107"
        else:  # REJECT
            verdict_text = "✗ REJECTED"
            verdict_color = "#F44336"

        self.quality_verdict_label.setText(verdict_text)
        self.quality_verdict_label.setStyleSheet(f"color: {verdict_color}; font-weight: bold; font-size: 11pt;")

    def _identify_sniffing_cluster(self, feature_matrix, feature_keys, quality_warning=None):
        """Identify which cluster represents sniffing and assign colors.

        Sniffing characteristics:
        - Highest mean instantaneous frequency (if)
        - Lowest mean inspiratory time (ti)

        Args:
            quality_warning: Warning message from quality check (if any)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels)

        # Get indices of IF and Ti features
        if_idx = feature_keys.index('if') if 'if' in feature_keys else None
        ti_idx = feature_keys.index('ti') if 'ti' in feature_keys else None

        if if_idx is None and ti_idx is None:
            # Can't identify sniffing without IF or Ti, use default colors
            print("[gmm-clustering] Warning: Cannot identify sniffing cluster without 'if' or 'ti' features. Using default colors.")
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            self.cluster_colors = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_labels)}
            self.sniffing_cluster_id = None
            return

        # Compute mean IF and Ti for each cluster
        cluster_stats = {}
        for cluster_id in unique_labels:
            mask = self.cluster_labels == cluster_id
            stats = {}

            if if_idx is not None:
                stats['mean_if'] = np.mean(feature_matrix[mask, if_idx])
            if ti_idx is not None:
                stats['mean_ti'] = np.mean(feature_matrix[mask, ti_idx])

            cluster_stats[cluster_id] = stats

        # Identify sniffing: highest IF and/or lowest Ti
        # Score = (normalized IF rank) + (normalized Ti inverse rank)
        cluster_scores = {}
        for cluster_id in unique_labels:
            score = 0

            if if_idx is not None:
                # Higher IF = more likely sniffing
                if_vals = [cluster_stats[c]['mean_if'] for c in unique_labels]
                if_rank = sorted(if_vals).index(cluster_stats[cluster_id]['mean_if'])
                score += if_rank / (n_clusters - 1) if n_clusters > 1 else 0

            if ti_idx is not None:
                # Lower Ti = more likely sniffing
                ti_vals = [cluster_stats[c]['mean_ti'] for c in unique_labels]
                ti_rank = sorted(ti_vals, reverse=True).index(cluster_stats[cluster_id]['mean_ti'])
                score += ti_rank / (n_clusters - 1) if n_clusters > 1 else 0

            cluster_scores[cluster_id] = score

        # Cluster with highest score is sniffing
        self.sniffing_cluster_id = max(cluster_scores, key=cluster_scores.get)

        # Validate that the identified cluster actually looks like sniffing
        # Sniffing typically has IF > 5-6 Hz, eupnea is usually 2-4 Hz
        sniff_stats = cluster_stats[self.sniffing_cluster_id]
        if if_idx is not None:
            sniff_if = sniff_stats['mean_if']
            # Check if "sniffing" cluster actually has high enough frequency
            if sniff_if < 5.0:
                # This doesn't look like real sniffing
                physiological_warning = (
                    f"⚠️ The identified 'sniffing' cluster has low mean IF ({sniff_if:.2f} Hz).\n\n"
                    "True sniffing typically has IF > 5-6 Hz.\n"
                    "This may just be variation in normal breathing, not distinct sniffing.\n\n"
                    "The cluster separation may not represent meaningful patterns."
                )
                if quality_warning:
                    quality_warning = quality_warning + "\n\n" + physiological_warning
                else:
                    quality_warning = physiological_warning

        # Assign colors: purple for sniffing, green for others (if 2 clusters), otherwise tab10
        if n_clusters == 2:
            # Simple case: sniffing = purple, other = green
            for cluster_id in unique_labels:
                if cluster_id == self.sniffing_cluster_id:
                    self.cluster_colors[cluster_id] = 'purple'
                else:
                    self.cluster_colors[cluster_id] = 'green'
        else:
            # Multiple clusters: sniffing = purple, others = tab10 colors (avoiding purple)
            tab10_colors = plt.cm.tab10(np.linspace(0, 1, 10))
            color_idx = 0
            for cluster_id in unique_labels:
                if cluster_id == self.sniffing_cluster_id:
                    self.cluster_colors[cluster_id] = 'purple'
                else:
                    # Use green as first alternative color, then other tab10 colors
                    if color_idx == 0:
                        self.cluster_colors[cluster_id] = 'green'
                    else:
                        # Skip purple-ish colors in tab10 (index 4)
                        skip_indices = [4]
                        actual_idx = color_idx
                        while actual_idx in skip_indices and actual_idx < 10:
                            actual_idx += 1
                        self.cluster_colors[cluster_id] = tab10_colors[actual_idx % 10]
                    color_idx += 1

        print(f"[gmm-clustering] Identified cluster {self.sniffing_cluster_id} as sniffing")
        for cluster_id in unique_labels:
            stats_str = ", ".join([f"{k}={v:.3f}" for k, v in cluster_stats[cluster_id].items()])
            print(f"  Cluster {cluster_id}: {stats_str}, color={self.cluster_colors[cluster_id]}")

        # Note: Warning dialogs now handled in on_run_gmm() after quality check

    def _collect_breath_features(self, feature_keys):
        """Collect per-breath features from all analyzed sweeps."""
        import numpy as np
        from core import metrics, filters

        feature_matrix = []
        breath_cycles = []

        # Iterate through all analyzed sweeps
        st = self.main_window.state

        for sweep_idx in sorted(st.breath_by_sweep.keys()):
            breath_data = st.breath_by_sweep[sweep_idx]

            # Get peaks from peaks_by_sweep (they're stored separately)
            if sweep_idx not in st.peaks_by_sweep:
                continue

            peaks = st.peaks_by_sweep[sweep_idx]

            # Get time, signal, and event indices for this sweep
            t = st.t
            y_raw = st.sweeps[st.analyze_chan][:, sweep_idx]

            # Apply filters to get processed signal (replicate _current_trace logic)
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=self.main_window.filter_order
            )

            # Apply notch filter if configured
            if self.main_window.notch_filter_lower is not None and self.main_window.notch_filter_upper is not None:
                y = self.main_window._apply_notch_filter(y, st.sr_hz,
                                                          self.main_window.notch_filter_lower,
                                                          self.main_window.notch_filter_upper)

            # Apply z-score normalization if enabled (using global statistics)
            if self.main_window.use_zscore_normalization:
                # Compute global stats if not cached
                if self.main_window.zscore_global_mean is None or self.main_window.zscore_global_std is None:
                    self.main_window.zscore_global_mean, self.main_window.zscore_global_std = self.main_window._compute_global_zscore_stats()
                y = filters.zscore_normalize(y, self.main_window.zscore_global_mean, self.main_window.zscore_global_std)

            # Get breath event indices
            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            expmins = breath_data.get('expmins', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            # Skip if no breath events detected
            if len(onsets) == 0:
                continue

            # Compute metrics for this sweep
            metrics_dict = {}
            for feature_key in feature_keys:
                if feature_key in metrics.METRICS:
                    metric_arr = metrics.METRICS[feature_key](
                        t, y, st.sr_hz, peaks, onsets, offsets, expmins, expoffs
                    )
                    metrics_dict[feature_key] = metric_arr

            # Extract per-breath values (one value per breath cycle)
            n_breaths = len(onsets)
            for breath_idx in range(n_breaths):
                start = int(onsets[breath_idx])
                breath_features = []

                # Extract metric value for this breath
                valid_breath = True
                for feature_key in feature_keys:
                    if feature_key not in metrics_dict:
                        valid_breath = False
                        break

                    metric_arr = metrics_dict[feature_key]
                    if start < len(metric_arr):
                        val = metric_arr[start]
                        if np.isnan(val) or not np.isfinite(val):
                            valid_breath = False
                            break
                        breath_features.append(val)
                    else:
                        valid_breath = False
                        break

                if valid_breath and len(breath_features) == len(feature_keys):
                    feature_matrix.append(breath_features)
                    breath_cycles.append((sweep_idx, breath_idx))

        return np.array(feature_matrix), breath_cycles

    def _update_results_table(self):
        """Update the results table with cluster statistics."""
        import numpy as np
        from PyQt6.QtWidgets import QTableWidgetItem
        from PyQt6.QtGui import QColor
        from PyQt6.QtCore import Qt

        unique_clusters = np.unique(self.cluster_labels)

        # Block signals while updating to prevent triggering itemChanged
        self.results_table.blockSignals(True)
        self.results_table.setRowCount(len(unique_clusters))

        for i, cluster_id in enumerate(unique_clusters):
            count = np.sum(self.cluster_labels == cluster_id)
            percentage = 100.0 * count / len(self.cluster_labels)

            # Calculate average confidence (probability of assigned cluster)
            mask = self.cluster_labels == cluster_id
            if hasattr(self, 'cluster_probabilities') and self.cluster_probabilities is not None:
                avg_confidence = np.mean(self.cluster_probabilities[mask, cluster_id])
            else:
                avg_confidence = 1.0  # Perfect confidence if no probabilities

            # Determine label: custom label > default labeling
            if cluster_id in self.cluster_custom_labels:
                # Use custom label if user has edited it
                label = self.cluster_custom_labels[cluster_id]
            elif cluster_id == self.sniffing_cluster_id:
                # Sniffing cluster
                label = f"Cluster {cluster_id} (Sniffing)"
            elif cluster_id == 0:
                # Cluster 0 defaults to "Eupnea" (unless it's the sniffing cluster)
                label = f"Cluster {cluster_id} (Eupnea)"
            else:
                # Other clusters get generic label
                label = f"Cluster {cluster_id}"

            cluster_item = QTableWidgetItem(label)
            # Make cluster label editable
            cluster_item.setFlags(cluster_item.flags() | Qt.ItemFlag.ItemIsEditable)

            count_item = QTableWidgetItem(str(count))
            pct_item = QTableWidgetItem(f"{percentage:.1f}%")
            conf_item = QTableWidgetItem(f"{avg_confidence:.3f}")

            # Make other columns non-editable
            count_item.setFlags(count_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            pct_item.setFlags(pct_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

            # Color-code the row
            if cluster_id in self.cluster_colors:
                color = self.cluster_colors[cluster_id]
                if isinstance(color, str):
                    # Named color
                    bg_color = QColor(color)
                else:
                    # RGBA tuple from colormap
                    bg_color = QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255))

                bg_color.setAlpha(80)  # Make it semi-transparent
                cluster_item.setBackground(bg_color)
                count_item.setBackground(bg_color)
                pct_item.setBackground(bg_color)
                conf_item.setBackground(bg_color)

            self.results_table.setItem(i, 0, cluster_item)
            self.results_table.setItem(i, 1, count_item)
            self.results_table.setItem(i, 2, pct_item)
            self.results_table.setItem(i, 3, conf_item)

        # Re-enable signals after table is populated
        self.results_table.blockSignals(False)

        # Dynamically adjust table height to fit number of clusters (no scrolling needed)
        # Calculate height: header + rows + some padding
        row_height = self.results_table.rowHeight(0) if len(unique_clusters) > 0 else 30
        header_height = self.results_table.horizontalHeader().height()
        total_height = header_height + (row_height * len(unique_clusters)) + 5  # +5 for padding
        self.results_table.setMaximumHeight(total_height)
        self.results_table.setMinimumHeight(total_height)

    def _on_table_item_changed(self, item):
        """Handle user edits to cluster labels in the table."""
        import numpy as np

        # Only process edits to the first column (cluster labels)
        if item.column() != 0:
            return

        # Get the cluster_id from the row index
        row = item.row()
        unique_clusters = np.unique(self.cluster_labels)

        if row < len(unique_clusters):
            cluster_id = unique_clusters[row]
            new_label = item.text().strip()

            if new_label:
                # Store custom label
                self.cluster_custom_labels[cluster_id] = new_label
                print(f"[gmm-dialog] User set custom label for cluster {cluster_id}: '{new_label}'")
            else:
                # If user clears the label, remove custom label and revert to default
                if cluster_id in self.cluster_custom_labels:
                    del self.cluster_custom_labels[cluster_id]
                # Refresh table to restore default label
                self._update_results_table()

    def _plot_waveform_overlays(self, total_rows, max_cols, labels, colors):
        """Plot waveform overlays: Column 1 = mean +/- SEM waveforms, Column 2 = mean trajectory line plot."""
        import numpy as np
        from core import filters

        if not self.main_window or not hasattr(self.main_window, 'state'):
            return

        st = self.main_window.state

        # Get current processed signal (with all filters applied)
        if not st.analyze_chan or st.analyze_chan not in st.sweeps:
            return

        # Check if waveform plotting is enabled
        plot_waveforms = self.plot_waveforms_cb.isChecked()
        max_breaths = self.n_breaths_spin.value()

        # Collect waveform cutouts for each cluster (now restricted to onset -> expoff)
        eupnea_waveforms = []  # List of (time_array, signal_array) tuples
        sniffing_waveforms = []

        # Also collect signal and derivative for mean trajectory plot
        eupnea_trajectories = []  # List of (signal_array, derivative_array) tuples
        sniffing_trajectories = []

        # Only collect data if plotting is enabled
        if plot_waveforms:
            for breath_idx, (sweep_idx, breath_i) in enumerate(self.breath_cycles):
                # Early exit if both groups have enough breaths
                if len(eupnea_waveforms) >= max_breaths and len(sniffing_waveforms) >= max_breaths:
                    break

                cluster_id = labels[breath_idx]

                # Skip if this cluster already has enough breaths
                if cluster_id == self.sniffing_cluster_id:
                    if len(sniffing_waveforms) >= max_breaths:
                        continue
                else:
                    if len(eupnea_waveforms) >= max_breaths:
                        continue

                # Get breath event data for this sweep
                breath_data = st.breath_by_sweep.get(sweep_idx)
                if breath_data is None:
                    continue

                onsets = breath_data.get('onsets', None)
                expoffs = breath_data.get('expoffs', None)
                expmins = breath_data.get('expmins', None)
                if onsets is None or expoffs is None or len(onsets) == 0 or len(expoffs) == 0:
                    continue

                if breath_i >= len(onsets) or breath_i >= len(expoffs):
                    continue

                # Get processed signal for this sweep
                Y = st.sweeps[st.analyze_chan]
                y_raw = Y[:, sweep_idx]

                # Apply all filters (same as main plot)
                y = filters.apply_all_1d(
                    y_raw,
                    sr_hz=st.sr_hz,
                    use_low=st.use_low,
                    low_hz=st.low_hz,
                    use_high=st.use_high,
                    high_hz=st.high_hz,
                    use_mean=st.use_mean_sub,
                    mean_param=st.mean_val,
                    use_inv=st.use_invert,
                    order=self.main_window.filter_order
                )

                # Apply notch filter if configured
                if self.main_window.notch_filter_lower is not None and self.main_window.notch_filter_upper is not None:
                    y = self.main_window._apply_notch_filter(y, st.sr_hz)

                # Apply z-score if enabled
                if self.main_window.use_zscore_normalization:
                    if self.main_window.zscore_global_mean is None or self.main_window.zscore_global_std is None:
                        self.main_window.zscore_global_mean, self.main_window.zscore_global_std = self.main_window._compute_global_zscore_stats()
                    y = filters.zscore_normalize(y, self.main_window.zscore_global_mean, self.main_window.zscore_global_std)

                t = st.t

                # Extract indices for breath cycle
                onset_idx = int(onsets[breath_i])
                expoff_idx = int(expoffs[breath_i])

                # For waveform plot: onset -> expiratory offset
                # Validate indices
                if onset_idx < 0 or expoff_idx >= len(y) or onset_idx >= expoff_idx:
                    continue

                # Extract waveform segment (onset -> expoff) with t=0 at onset
                waveform_t = t[onset_idx:expoff_idx+1] - t[onset_idx]  # Time relative to onset
                waveform_y = y[onset_idx:expoff_idx+1]

                # Downsample waveform for faster computation (reduce to ~1/3rd of original sampling rate)
                downsample_factor = 3
                if len(waveform_y) > downsample_factor * 2:  # Only downsample if enough samples
                    from scipy import signal
                    waveform_y = signal.decimate(waveform_y, downsample_factor, zero_phase=True)
                    waveform_t = waveform_t[::downsample_factor]  # Downsample time axis to match

                # For trajectory plot: Find first derivative zero-crossing before onset, then extend to expoff
                # Start by searching backwards from onset for derivative zero-crossing
                trajectory_start_idx = onset_idx  # Default fallback

                # Search backwards from onset (up to 100 samples or beginning of trace)
                search_window = min(100, onset_idx)  # Don't search beyond trace start
                if search_window > 10:  # Only search if we have enough samples
                    # Compute derivative in the region before onset
                    search_start = onset_idx - search_window
                    y_before_onset = y[search_start:onset_idx]

                    if len(y_before_onset) > 1:
                        # Compute derivative using np.gradient
                        dt_search = (t[onset_idx] - t[search_start]) / len(y_before_onset) if len(y_before_onset) > 1 else (1.0 / st.sr_hz)
                        dy_before_onset = np.gradient(y_before_onset, dt_search)

                        # Find zero-crossings: where derivative changes sign
                        sign_changes = np.where(np.diff(np.sign(dy_before_onset)))[0]

                        if len(sign_changes) > 0:
                            # Use the LAST (most recent) zero-crossing before onset
                            # sign_changes[-1] gives the index where sign changes, meaning zero is between i and i+1
                            # Find which of these two points is closer to zero
                            zero_crossing_idx = sign_changes[-1]

                            # Check both points around the sign change
                            if zero_crossing_idx + 1 < len(dy_before_onset):
                                val_before = abs(dy_before_onset[zero_crossing_idx])
                                val_after = abs(dy_before_onset[zero_crossing_idx + 1])
                                # Use whichever is closer to zero
                                if val_after < val_before:
                                    zero_crossing_idx += 1

                            # Convert to global index
                            trajectory_start_idx = search_start + zero_crossing_idx

                # End at current expiratory offset
                trajectory_end_idx = expoff_idx

                # Validate trajectory indices
                if trajectory_start_idx < 0 or trajectory_end_idx >= len(y) or trajectory_start_idx >= trajectory_end_idx:
                    # Fallback to onset -> expoff if invalid
                    trajectory_start_idx = onset_idx
                    trajectory_end_idx = expoff_idx

                # Extract trajectory segment (derivative zero-crossing -> current expoff) with t=0 at current onset
                trajectory_t = t[trajectory_start_idx:trajectory_end_idx+1] - t[onset_idx]
                trajectory_y = y[trajectory_start_idx:trajectory_end_idx+1]

                # Downsample trajectory for performance
                if len(trajectory_y) > downsample_factor * 2:
                    from scipy import signal
                    trajectory_y = signal.decimate(trajectory_y, downsample_factor, zero_phase=True)
                    trajectory_t = trajectory_t[::downsample_factor]

                # Compute first derivative for trajectory (using downsampled data)
                dt = trajectory_t[1] - trajectory_t[0] if len(trajectory_t) > 1 else (1.0 / st.sr_hz)
                trajectory_dy = np.gradient(trajectory_y, dt)

                # Add to appropriate cluster (limit to max_breaths)
                if cluster_id == self.sniffing_cluster_id:
                    if len(sniffing_waveforms) < max_breaths:
                        sniffing_waveforms.append((waveform_t, waveform_y))
                        sniffing_trajectories.append((trajectory_y, trajectory_dy))
                else:
                    if len(eupnea_waveforms) < max_breaths:
                        eupnea_waveforms.append((waveform_t, waveform_y))
                        eupnea_trajectories.append((trajectory_y, trajectory_dy))

        # ========================================
        # Column 1: Mean +/- SEM Waveforms (optional)
        # ========================================
        if plot_waveforms:
            ax1 = self.figure.add_subplot(total_rows, max_cols, 1)

            # Helper function to pad waveforms to longest duration (preserving time alignment)
            def align_and_compute_stats(waveforms, variability_mode='sem'):
                """Pad waveforms to longest duration, then compute mean +/- variability.
                Preserves time alignment where t=0 is at breath onset.

                Args:
                    waveforms: List of (time, signal) tuples
                    variability_mode: One of 'sem', 'std', 'minmax'

                Returns:
                    t_common: Common time axis
                    mean_wf: Mean waveform
                    var_lower: Lower variability bound
                    var_upper: Upper variability bound
                """
                if len(waveforms) == 0:
                    return None, None, None, None

                # Find the waveform with the longest duration
                longest_wf_t, longest_wf_y = max(waveforms, key=lambda x: len(x[0]))
                max_len = len(longest_wf_t)

                # Use the time axis from the longest waveform (preserves t=0 alignment)
                t_common = longest_wf_t

                # Pad all waveforms to max_len with NaN
                padded_signals = []
                for wf_t, wf_y in waveforms:
                    if len(wf_y) < max_len:
                        # Pad with NaN
                        padded = np.full(max_len, np.nan)
                        padded[:len(wf_y)] = wf_y
                        padded_signals.append(padded)
                    else:
                        padded_signals.append(wf_y)

                # Stack and compute statistics (ignoring NaN)
                wf_matrix = np.vstack(padded_signals)
                mean_wf = np.nanmean(wf_matrix, axis=0)

                if variability_mode == 'minmax':
                    # Compute min and max across all waveforms
                    var_lower = np.nanmin(wf_matrix, axis=0)
                    var_upper = np.nanmax(wf_matrix, axis=0)
                elif variability_mode == 'std':
                    # Compute standard deviation
                    std_wf = np.nanstd(wf_matrix, axis=0)
                    var_lower = mean_wf - std_wf
                    var_upper = mean_wf + std_wf
                else:  # 'sem' (default)
                    # Compute SEM
                    sem_wf = np.nanstd(wf_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(wf_matrix), axis=0))
                    var_lower = mean_wf - sem_wf
                    var_upper = mean_wf + sem_wf

                return t_common, mean_wf, var_lower, var_upper

            # Check which variability mode is selected
            if self.minmax_radio.isChecked():
                variability_mode = 'minmax'
                variability_label = "Min/Max"
            elif self.std_radio.isChecked():
                variability_mode = 'std'
                variability_label = "STD"
            else:
                variability_mode = 'sem'
                variability_label = "SEM"

            # Plot eupnea mean +/- variability (green)
            if eupnea_waveforms:
                t_eup, mean_eup, var_lower_eup, var_upper_eup = align_and_compute_stats(eupnea_waveforms, variability_mode=variability_mode)
                if t_eup is not None:
                    ax1.plot(t_eup, mean_eup, color='green', linewidth=2.5,
                            label=f'Eupnea (n={len(eupnea_waveforms)})', alpha=0.9)
                    ax1.fill_between(t_eup, var_lower_eup, var_upper_eup,
                                    color='green', alpha=0.25, linewidth=0)

            # Plot sniffing mean +/- variability (purple)
            if sniffing_waveforms:
                t_snf, mean_snf, var_lower_snf, var_upper_snf = align_and_compute_stats(sniffing_waveforms, variability_mode=variability_mode)
                if t_snf is not None:
                    ax1.plot(t_snf, mean_snf, color='purple', linewidth=2.5,
                            label=f'Sniffing (n={len(sniffing_waveforms)})', alpha=0.9)
                    ax1.fill_between(t_snf, var_lower_snf, var_upper_snf,
                                    color='purple', alpha=0.25, linewidth=0)

            ax1.set_xlabel('Time from Onset (s)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Signal Amplitude', fontsize=11, fontweight='bold')
            ax1.set_title(f'Mean +/- {variability_label} Breath Waveforms', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            # Placeholder when waveforms disabled
            ax1 = self.figure.add_subplot(total_rows, max_cols, 1)
            ax1.text(0.5, 0.5, 'Waveform plotting disabled\n(Enable checkbox to view)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')

        # ========================================
        # Column 2: Mean Trajectory (Signal vs Derivative) - optional
        # ========================================
        if plot_waveforms:
            ax2 = self.figure.add_subplot(total_rows, max_cols, 2)

            # Helper function to compute mean trajectory with variability
            def compute_mean_trajectory_with_variability(trajectories, variability_mode='sem'):
                """Pad trajectories to longest, then compute mean +/- variability.

                Args:
                    trajectories: List of (signal, derivative) tuples
                    variability_mode: One of 'sem', 'std', 'minmax'

                Returns:
                    mean_sig, mean_deriv: Mean signal and derivative
                    var_sig_lower, var_sig_upper: Lower and upper bounds for signal
                    var_deriv_lower, var_deriv_upper: Lower and upper bounds for derivative
                """
                if len(trajectories) == 0:
                    return None, None, None, None, None, None

                # Find longest trajectory
                max_len = max(len(sig) for sig, deriv in trajectories)

                # Pad all trajectories to max_len with NaN
                padded_sigs = []
                padded_derivs = []
                for sig, deriv in trajectories:
                    if len(sig) < max_len:
                        padded_sig = np.full(max_len, np.nan)
                        padded_deriv = np.full(max_len, np.nan)
                        padded_sig[:len(sig)] = sig
                        padded_deriv[:len(deriv)] = deriv
                        padded_sigs.append(padded_sig)
                        padded_derivs.append(padded_deriv)
                    else:
                        padded_sigs.append(sig)
                        padded_derivs.append(deriv)

                # Stack and compute mean (ignoring NaN)
                sig_matrix = np.vstack(padded_sigs)
                deriv_matrix = np.vstack(padded_derivs)
                mean_sig = np.nanmean(sig_matrix, axis=0)
                mean_deriv = np.nanmean(deriv_matrix, axis=0)

                if variability_mode == 'minmax':
                    # Compute min/max
                    var_sig_lower = np.nanmin(sig_matrix, axis=0)
                    var_sig_upper = np.nanmax(sig_matrix, axis=0)
                    var_deriv_lower = np.nanmin(deriv_matrix, axis=0)
                    var_deriv_upper = np.nanmax(deriv_matrix, axis=0)
                elif variability_mode == 'std':
                    # Compute standard deviation
                    std_sig = np.nanstd(sig_matrix, axis=0)
                    std_deriv = np.nanstd(deriv_matrix, axis=0)
                    var_sig_lower = mean_sig - std_sig
                    var_sig_upper = mean_sig + std_sig
                    var_deriv_lower = mean_deriv - std_deriv
                    var_deriv_upper = mean_deriv + std_deriv
                else:  # 'sem' (default)
                    # Compute SEM
                    sem_sig = np.nanstd(sig_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(sig_matrix), axis=0))
                    sem_deriv = np.nanstd(deriv_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(deriv_matrix), axis=0))
                    var_sig_lower = mean_sig - sem_sig
                    var_sig_upper = mean_sig + sem_sig
                    var_deriv_lower = mean_deriv - sem_deriv
                    var_deriv_upper = mean_deriv + sem_deriv

                return mean_sig, mean_deriv, var_sig_lower, var_sig_upper, var_deriv_lower, var_deriv_upper

            # Helper function to calculate perpendicular box intersections with neighbor-aware clipping
            def calculate_perpendicular_intersections(mean_sig, mean_deriv, sig_lower, sig_upper, deriv_lower, deriv_upper):
                """Calculate boundary using perpendicular lines clipped by neighbors and box edges.

                When neighboring perpendicular lines intersect before reaching their boxes,
                use the intersection point instead to prevent crossing boundaries.

                Args:
                    mean_sig, mean_deriv: Mean trajectory coordinates
                    sig_lower, sig_upper: X-axis (signal) error bounds
                    deriv_lower, deriv_upper: Y-axis (derivative) error bounds

                Returns:
                    upper_points, lower_points: Lists of (x, y) tuples forming the boundary
                """
                # First pass: calculate perpendicular lines for all points
                perp_lines = []  # List of dicts or None

                for i in range(len(mean_sig)):
                    # Skip NaN values
                    if (np.isnan(mean_sig[i]) or np.isnan(mean_deriv[i]) or
                        np.isnan(sig_lower[i]) or np.isnan(sig_upper[i]) or
                        np.isnan(deriv_lower[i]) or np.isnan(deriv_upper[i])):
                        perp_lines.append(None)
                        continue

                    # Calculate tangent vector
                    if i < len(mean_sig) - 1:
                        dx = mean_sig[i+1] - mean_sig[i]
                        dy = mean_deriv[i+1] - mean_deriv[i]
                    elif i > 0:
                        dx = mean_sig[i] - mean_sig[i-1]
                        dy = mean_deriv[i] - mean_deriv[i-1]
                    else:
                        perp_lines.append(None)
                        continue

                    # Normalize tangent
                    length = np.sqrt(dx**2 + dy**2)
                    if length < 1e-10:
                        perp_lines.append(None)
                        continue

                    dx /= length
                    dy /= length

                    # Perpendicular vector (rotate 90 degrees)
                    perp_x = -dy
                    perp_y = dx

                    perp_lines.append({
                        'point': (mean_sig[i], mean_deriv[i]),
                        'direction': (perp_x, perp_y),
                        'box': (sig_lower[i], sig_upper[i], deriv_lower[i], deriv_upper[i])
                    })

                # Helper: find intersection between two perpendicular lines
                def line_intersection(p1, d1, p2, d2):
                    """Find intersection of two lines."""
                    det = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(det) < 1e-10:
                        return None
                    dp = (p2[0] - p1[0], p2[1] - p1[1])
                    s = (dp[0] * d2[1] - dp[1] * d2[0]) / det
                    return (p1[0] + s * d1[0], p1[1] + s * d1[1])

                # Helper: find box intersections
                def box_intersections(point, direction, box):
                    box_left, box_right, box_bottom, box_top = box
                    intersections = []
                    perp_x, perp_y = direction
                    px, py = point

                    if abs(perp_x) > 1e-10:
                        for x_edge in [box_left, box_right]:
                            t = (x_edge - px) / perp_x
                            y = py + t * perp_y
                            if box_bottom <= y <= box_top:
                                intersections.append((t, x_edge, y))

                    if abs(perp_y) > 1e-10:
                        for y_edge in [box_bottom, box_top]:
                            t = (y_edge - py) / perp_y
                            x = px + t * perp_x
                            if box_left <= x <= box_right:
                                intersections.append((t, x, y_edge))

                    intersections.sort(key=lambda item: item[0])
                    return intersections

                # Second pass: calculate boundary points considering neighbors
                upper_points = []
                lower_points = []

                for i, line_info in enumerate(perp_lines):
                    if line_info is None:
                        continue

                    point = line_info['point']
                    direction = line_info['direction']
                    box = line_info['box']

                    # Find box intersections
                    box_ints = box_intersections(point, direction, box)
                    if len(box_ints) < 2:
                        upper_points.append((box[1], box[3]))
                        lower_points.append((box[0], box[2]))
                        continue

                    lower_candidate = (box_ints[0][1], box_ints[0][2])
                    upper_candidate = (box_ints[-1][1], box_ints[-1][2])

                    # Check neighboring perpendiculars
                    for neighbor_idx in [i-1, i+1]:
                        if 0 <= neighbor_idx < len(perp_lines) and perp_lines[neighbor_idx] is not None:
                            neighbor = perp_lines[neighbor_idx]
                            isect = line_intersection(point, direction,
                                                     neighbor['point'], neighbor['direction'])
                            if isect is not None:
                                # Use neighbor intersection if closer than box edge
                                dist_isect = np.sqrt((isect[0]-point[0])**2 + (isect[1]-point[1])**2)
                                dist_lower = np.sqrt((lower_candidate[0]-point[0])**2 + (lower_candidate[1]-point[1])**2)
                                dist_upper = np.sqrt((upper_candidate[0]-point[0])**2 + (upper_candidate[1]-point[1])**2)

                                if dist_isect < dist_lower:
                                    lower_candidate = isect
                                if dist_isect < dist_upper:
                                    upper_candidate = isect

                    lower_points.append(lower_candidate)
                    upper_points.append(upper_candidate)

                return upper_points, lower_points

            # Plot mean trajectories as lines with variability shading (no markers)
            if eupnea_trajectories:
                result = compute_mean_trajectory_with_variability(eupnea_trajectories, variability_mode=variability_mode)
                if result[0] is not None:
                    mean_sig_eup, mean_deriv_eup, sig_lower_eup, sig_upper_eup, deriv_lower_eup, deriv_upper_eup = result

                    # Calculate perpendicular intersections
                    upper_pts, lower_pts = calculate_perpendicular_intersections(
                        mean_sig_eup, mean_deriv_eup, sig_lower_eup, sig_upper_eup, deriv_lower_eup, deriv_upper_eup
                    )

                    if len(upper_pts) > 0 and len(lower_pts) > 0:
                        # Create polygon from upper points forward + lower points backward
                        upper_x = [pt[0] for pt in upper_pts]
                        upper_y = [pt[1] for pt in upper_pts]
                        lower_x = [pt[0] for pt in lower_pts]
                        lower_y = [pt[1] for pt in lower_pts]

                        poly_x = upper_x + lower_x[::-1]
                        poly_y = upper_y + lower_y[::-1]

                        ax2.fill(poly_x, poly_y, color='green', alpha=0.15, linewidth=0)

                    # Plot mean trajectory line (no markers, just line) - on top
                    ax2.plot(mean_sig_eup, mean_deriv_eup, color='green', linewidth=2.5,
                            label=f'Eupnea Mean Trajectory', alpha=0.9, zorder=10)

            if sniffing_trajectories:
                result = compute_mean_trajectory_with_variability(sniffing_trajectories, variability_mode=variability_mode)
                if result[0] is not None:
                    mean_sig_snf, mean_deriv_snf, sig_lower_snf, sig_upper_snf, deriv_lower_snf, deriv_upper_snf = result

                    # Calculate perpendicular intersections
                    upper_pts, lower_pts = calculate_perpendicular_intersections(
                        mean_sig_snf, mean_deriv_snf, sig_lower_snf, sig_upper_snf, deriv_lower_snf, deriv_upper_snf
                    )

                    if len(upper_pts) > 0 and len(lower_pts) > 0:
                        # Create polygon
                        upper_x = [pt[0] for pt in upper_pts]
                        upper_y = [pt[1] for pt in upper_pts]
                        lower_x = [pt[0] for pt in lower_pts]
                        lower_y = [pt[1] for pt in lower_pts]

                        poly_x = upper_x + lower_x[::-1]
                        poly_y = upper_y + lower_y[::-1]

                        ax2.fill(poly_x, poly_y, color='purple', alpha=0.15, linewidth=0)

                    # Plot mean trajectory line on top
                    ax2.plot(mean_sig_snf, mean_deriv_snf, color='purple', linewidth=2.5,
                            label=f'Sniffing Mean Trajectory', alpha=0.9, zorder=10)

            ax2.set_xlabel('Signal Amplitude', fontsize=11, fontweight='bold')
            ax2.set_ylabel('First Derivative (dy/dt)', fontsize=11, fontweight='bold')
            ax2.set_title('Mean Trajectory (Signal vs Derivative)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax2.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        else:
            # Placeholder when trajectories disabled
            ax2 = self.figure.add_subplot(total_rows, max_cols, 2)
            ax2.text(0.5, 0.5, 'Trajectory plotting disabled\n(Enable checkbox to view)',
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')

    def _plot_clusters(self, feature_matrix, labels, feature_keys):
        """Plot histograms for each feature + 2D scatter plots of clusters + waveform overlays."""
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        self.figure.clear()

        n_features = len(feature_keys)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        # Use pre-assigned colors (purple for sniffing, green for eupnea, etc.)
        colors = self.cluster_colors

        # Calculate layout: NEW WAVEFORM ROW at top, then histograms in 2 columns, scatter plots below
        waveform_rows = 1  # Add 1 row at top for waveform overlays
        hist_cols = 2
        hist_rows = int(np.ceil(n_features / hist_cols))

        n_scatter_plots = min(6, (n_features * (n_features - 1)) // 2)  # Max 6 scatter plots

        if n_scatter_plots > 0:
            scatter_rows = int(np.ceil(np.sqrt(n_scatter_plots)))
            scatter_cols = int(np.ceil(n_scatter_plots / scatter_rows))
        else:
            scatter_rows = 0
            scatter_cols = 2

        total_rows = waveform_rows + hist_rows + scatter_rows
        max_cols = 2  # Fixed 2 columns for cleaner layout

        # Fixed number of bins for all histograms
        n_bins = 50

        # ========================================
        # Section 0: NEW WAVEFORM OVERLAYS (Top Row)
        # ========================================
        self._plot_waveform_overlays(total_rows, max_cols, labels, colors)

        # ========================================
        # Section 1: Feature Histograms (as line plots)
        # ========================================
        for feat_idx, feature_key in enumerate(feature_keys):
            row = waveform_rows + (feat_idx // hist_cols)  # Offset by waveform_rows
            col = feat_idx % hist_cols
            ax = self.figure.add_subplot(total_rows, max_cols, row * max_cols + col + 1)

            feature_data = feature_matrix[:, feat_idx]

            # Calculate histogram bins (same for all overlays)
            data_min = np.min(feature_data)
            data_max = np.max(feature_data)
            bins = np.linspace(data_min, data_max, n_bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # Plot overall distribution (thicker gray line)
            counts_all, _ = np.histogram(feature_data, bins=bins)
            ax.plot(bin_centers, counts_all, color='gray', linewidth=2.5,
                   label='All Breaths', alpha=0.7)

            # Overlay each cluster's distribution
            for cluster_id in unique_labels:
                mask = labels == cluster_id
                cluster_data = feature_data[mask]
                counts_cluster, _ = np.histogram(cluster_data, bins=bins)

                # Label with pattern type if identified
                if cluster_id == self.sniffing_cluster_id:
                    cluster_label = f'Cluster {cluster_id} (Sniffing)'
                else:
                    cluster_label = f'Cluster {cluster_id}'

                ax.plot(bin_centers, counts_cluster, color=colors[cluster_id],
                       linewidth=2, label=cluster_label, alpha=0.8)

            ax.set_xlabel(feature_key, fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{feature_key} Distribution', fontsize=11, fontweight='bold')

        # ========================================
        # Section 2: 2D Scatter Plots
        # ========================================
        if n_scatter_plots > 0:
            plot_idx = 0

            for i in range(n_features):
                for j in range(i + 1, n_features):
                    if plot_idx >= n_scatter_plots:
                        break

                    # Calculate position in grid (below histograms, offset by waveform_rows)
                    row = waveform_rows + hist_rows + (plot_idx // scatter_cols)
                    col = plot_idx % scatter_cols
                    ax = self.figure.add_subplot(total_rows, max_cols, row * max_cols + col + 1)

                    # Plot each cluster with different color
                    for cluster_id in unique_labels:
                        mask = labels == cluster_id

                        # Label with pattern type if identified
                        if cluster_id == self.sniffing_cluster_id:
                            cluster_label = f"Cluster {cluster_id} (Sniffing)"
                        else:
                            cluster_label = f"Cluster {cluster_id}"

                        ax.scatter(feature_matrix[mask, i], feature_matrix[mask, j],
                                 c=[colors[cluster_id]], label=cluster_label,
                                 alpha=0.6, s=30, edgecolors='black', linewidth=0.5)

                    ax.set_xlabel(feature_keys[i], fontsize=10)
                    ax.set_ylabel(feature_keys[j], fontsize=10)
                    ax.legend(loc='best', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_title(f'{feature_keys[i]} vs {feature_keys[j]}', fontsize=10, fontweight='bold')

                    plot_idx += 1

                if plot_idx >= n_scatter_plots:
                    break

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()

    def show_help(self):
        """Display help dialog explaining GMM clustering."""
        from PyQt6.QtWidgets import QMessageBox

        help_text = """
<b>Gaussian Mixture Model (GMM) Breath Clustering</b>

<p><b>What is it?</b><br>
GMM is an unsupervised machine learning technique that automatically identifies distinct breathing patterns by grouping breaths with similar characteristics.</p>

<p><b>How it works:</b><br>
1. <b>Select Features:</b> Choose which breath metrics to use for clustering (e.g., frequency, timing, amplitude)<br>
2. <b>Set Number of Clusters:</b> Specify how many breathing patterns to identify (2-10)<br>
3. <b>Run GMM:</b> The algorithm analyzes all breaths from analyzed sweeps and groups them into clusters<br>
4. <b>View Results:</b> Examine cluster statistics and 2D scatter plots showing pattern separation</p>

<p><b>Typical Breathing Patterns (4 clusters recommended):</b><br>
• <b>Normal Breathing (Eupnea):</b> Regular frequency and amplitude<br>
• <b>Apnea/Pauses:</b> Extended inter-breath intervals<br>
• <b>Sighs:</b> Large amplitude, longer duration<br>
• <b>Sniffing:</b> High frequency, short duration, small amplitude</p>

<p><b>Tips:</b><br>
• Start with default features (IF, Ti, Amp Insp) and 4 clusters<br>
• Use at least 2-3 features for meaningful separation<br>
• More analyzed sweeps = better clustering results<br>
• Experiment with different feature combinations to find optimal separation</p>

<p><b>Recommended Settings:</b><br>
• <b>For basic pattern detection:</b> IF, Ti, Amp Insp (3 features, 4 clusters)<br>
• <b>For detailed analysis:</b> All 7 features, 5-6 clusters<br>
• <b>For sigh detection:</b> Include Amp Insp, Area Insp (high amplitude patterns)</p>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("GMM Clustering Help")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setText(help_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _auto_apply_gmm_results(self):
        """Automatically apply GMM clustering results to main plot (called after successful GMM run).

        NOTE: This now only applies if the user has the "Apply Eupnea/Sniffing Detection" checkbox enabled.
        If the checkbox is checked when GMM completes, eupnea and sniffing regions will be applied automatically.
        """
        import numpy as np
        
        print(f"[gmm-dialog] _auto_apply_gmm_results() called. Checkbox state: {self.apply_sniffing_cb.isChecked()}")

        if self.sniffing_cluster_id is None:
            print("[gmm-dialog] Cannot auto-apply: no sniffing cluster identified")
            return

        # Update eupnea detection mode and parameters based on user selection
        selected_mode = "gmm" if self.gmm_mode_radio.isChecked() else "frequency"
        self.main_window.eupnea_detection_mode = selected_mode

        # Save frequency-based parameters (used when mode is "frequency")
        if not hasattr(self.main_window, 'eupnea_freq_threshold'):
            self.main_window.eupnea_freq_threshold = 5.0
        self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
        self.main_window.eupnea_min_duration = self.min_duration_spin.value()

        print(f"[gmm-dialog] Auto-apply: Eupnea detection mode: {selected_mode}")
        print(f"[gmm-dialog] Auto-apply: Frequency threshold: {self.main_window.eupnea_freq_threshold} Hz, Min duration: {self.main_window.eupnea_min_duration} s")

        # Only apply eupnea/sniffing regions if the checkbox is enabled
        if self.apply_sniffing_cb.isChecked():
            print("[gmm-dialog] Auto-apply: Checkbox is enabled, applying eupnea and sniffing regions")
            self._apply_sniffing_to_plot()
        else:
            print("[gmm-dialog] Auto-apply: Checkbox is disabled, NOT applying eupnea/sniffing regions")
            # Just update eupnea mode, but don't apply regions
            self.main_window.redraw_main_plot()

    def on_apply_to_plot(self):
        """Apply GMM clustering results to main plot by marking sniffing breaths."""
        from PyQt6.QtWidgets import QMessageBox
        import numpy as np

        if self.sniffing_cluster_id is None:
            QMessageBox.warning(
                self,
                "No Sniffing Cluster",
                "Cannot identify sniffing cluster. Make sure you include 'if' or 'ti' features."
            )
            return

        # Warn again if clustering quality was poor
        if self.had_quality_warning:
            reply = QMessageBox.question(
                self,
                "Apply Despite Quality Warning?",
                "The clustering quality was flagged as questionable.\n\n"
                "The identified 'sniffing' breaths may just be normal variation,\n"
                "not true sniffing behavior.\n\n"
                "Are you sure you want to apply these results to the plot?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # Update eupnea detection mode and parameters based on user selection
        selected_mode = "gmm" if self.gmm_mode_radio.isChecked() else "frequency"
        self.main_window.eupnea_detection_mode = selected_mode

        # Save frequency-based parameters (used when mode is "frequency")
        if not hasattr(self.main_window, 'eupnea_freq_threshold'):
            self.main_window.eupnea_freq_threshold = 5.0
        self.main_window.eupnea_freq_threshold = self.freq_threshold_spin.value()
        self.main_window.eupnea_min_duration = self.min_duration_spin.value()

        print(f"[gmm-clustering] Updated eupnea detection mode to: {selected_mode}")
        print(f"[gmm-clustering] Frequency threshold: {self.main_window.eupnea_freq_threshold} Hz, Min duration: {self.main_window.eupnea_min_duration} s")

        # Store GMM classification directly in all_peaks_by_sweep
        # Add 'breath_type_class' field: 0=eupnea, 1=sniffing, -1=unclassified
        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            # Get peak sample index for this breath
            peaks = self.main_window.state.peaks_by_sweep.get(sweep_idx)
            if peaks is None or breath_idx >= len(peaks):
                continue
            peak_sample_idx = int(peaks[breath_idx])

            # Get all_peaks for this sweep
            all_peaks = self.main_window.state.all_peaks_by_sweep.get(sweep_idx)
            if all_peaks is None:
                continue

            # Initialize arrays if they don't exist or are None
            if 'breath_type_class' not in all_peaks or all_peaks['breath_type_class'] is None:
                all_peaks['breath_type_class'] = np.full(len(all_peaks['indices']), -1, dtype=np.int8)
            if 'gmm_class_ro' not in all_peaks or all_peaks['gmm_class_ro'] is None:
                all_peaks['gmm_class_ro'] = np.full(len(all_peaks['indices']), -1, dtype=np.int8)
            if 'eupnea_sniff_source' not in all_peaks or all_peaks['eupnea_sniff_source'] is None:
                all_peaks['eupnea_sniff_source'] = np.array(['auto'] * len(all_peaks['indices']))

            # Find this peak in all_peaks_by_sweep
            peak_mask = all_peaks['indices'] == peak_sample_idx
            if not peak_mask.any():
                continue
            peak_pos = np.where(peak_mask)[0][0]

            # Get probability and classify
            sniff_prob = self.cluster_probabilities[i, self.sniffing_cluster_id]
            gmm_prediction = 1 if sniff_prob >= confidence_threshold else 0

            # Store GMM prediction as read-only (for classifier switching)
            all_peaks['gmm_class_ro'][peak_pos] = gmm_prediction

            # Copy to user-editable array (will be used for display/export)
            all_peaks['breath_type_class'][peak_pos] = gmm_prediction
            all_peaks['eupnea_sniff_source'][peak_pos] = 'gmm'

        # Collect all sniffing breaths (just breath indices, not time ranges yet)
        sniffing_breaths_by_sweep = {}  # sweep_idx -> list of breath indices

        for i, (sweep_idx, breath_idx) in enumerate(self.breath_cycles):
            cluster_id = self.cluster_labels[i]

            if cluster_id == self.sniffing_cluster_id:
                # Store the breath index
                if sweep_idx not in sniffing_breaths_by_sweep:
                    sniffing_breaths_by_sweep[sweep_idx] = []
                sniffing_breaths_by_sweep[sweep_idx].append(breath_idx)

        # Convert breath indices to merged time ranges (merge consecutive breaths)
        total_regions = 0
        for sweep_idx, breath_indices in sniffing_breaths_by_sweep.items():
            if not breath_indices:
                continue

            # Sort breath indices
            breath_indices = sorted(breath_indices)

            # Group consecutive breath indices into runs
            runs = []
            current_run = [breath_indices[0]]

            for idx in breath_indices[1:]:
                if idx == current_run[-1] + 1:  # Consecutive breath
                    current_run.append(idx)
                else:  # Gap in breath indices
                    runs.append(current_run)
                    current_run = [idx]
            runs.append(current_run)  # Add the last run

            # Convert each run to a time range
            breath_data = self.main_window.state.breath_by_sweep.get(sweep_idx)
            if breath_data is None:
                continue

            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            t = self.main_window.state.t

            if sweep_idx not in self.main_window.state.sniff_regions_by_sweep:
                self.main_window.state.sniff_regions_by_sweep[sweep_idx] = []

            for run in runs:
                # Start time = onset of first breath in run
                first_breath = run[0]
                last_breath = run[-1]

                if first_breath >= len(onsets):
                    continue

                start_time = t[int(onsets[first_breath])]

                # End time = offset of last breath in run
                if last_breath < len(offsets):
                    end_idx = int(offsets[last_breath])
                else:
                    # Fallback: use next onset or end of trace
                    if last_breath + 1 < len(onsets):
                        end_idx = int(onsets[last_breath + 1])
                    else:
                        end_idx = len(t) - 1

                end_time = t[end_idx]

                # Add merged region
                self.main_window.state.sniff_regions_by_sweep[sweep_idx].append((start_time, end_time))
                print(f"[gmm-clustering] Created merged region from breaths {run[0]}-{run[-1]}: {start_time:.3f} - {end_time:.3f} s")

            total_regions += len(self.main_window.state.sniff_regions_by_sweep[sweep_idx])

        # Redraw main plot to show sniffing regions AND eupnea (based on selected mode)
        self.main_window.redraw_main_plot()

        # Show success message
        n_sniffing_breaths = np.sum(self.cluster_labels == self.sniffing_cluster_id)
        mode_msg = "GMM-based" if selected_mode == "gmm" else "Frequency-based"
        QMessageBox.information(
            self,
            "Applied to Plot",
            f"Marked {n_sniffing_breaths} sniffing breaths across {len(sniffing_breaths_by_sweep)} sweep(s).\n\n"
            f"After merging consecutive breaths: {total_regions} total sniffing region(s).\n\n"
            f"Purple background regions are now visible on the main plot.\n\n"
            f"Eupnea detection mode: {mode_msg}"
        )

        print(f"[gmm-clustering] Applied {n_sniffing_breaths} sniffing breaths to main plot")
        print(f"[gmm-clustering] Created {total_regions} merged sniffing regions across {len(sniffing_breaths_by_sweep)} sweeps")

