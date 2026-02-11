"""
Advanced Peak Editor Dialog for PhysioMetrics

Provides efficient review of edge cases and model disagreements:
- Tab 1: Breath vs Noise edge case review (using P_edge, ML disagreements, ML probabilities)
- Tab 2: Eupnea vs Sniff classification (future)
- Tab 3: Sigh vs Normal breath detection (future)

Features embedded matplotlib plotting with light/dark theme support.
"""

import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QComboBox,
    QPushButton, QLabel, QRadioButton, QButtonGroup, QCheckBox,
    QWidget, QMessageBox, QSplitter, QTabWidget, QSpinBox,
    QDoubleSpinBox, QScrollArea, QGridLayout, QProgressDialog,
    QApplication, QFrame, QSlider, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QColor, QBrush

# PyQtGraph imports for UMAP visualization
import pyqtgraph as pg

# Configuration: Set to True to force-disable 3D mode (useful for troubleshooting OpenGL issues)
FORCE_DISABLE_3D = False

try:
    import pyqtgraph.opengl as gl
    OPENGL_AVAILABLE = True and not FORCE_DISABLE_3D
except ImportError:
    OPENGL_AVAILABLE = False
    print("[Warning] PyQtGraph OpenGL not available. 3D UMAP disabled.")

if FORCE_DISABLE_3D:
    print("[Info] 3D UMAP disabled by FORCE_DISABLE_3D flag.")

# UMAP wrapper (optional dependency)
from core.umap_wrapper import compute_embedding, get_umap_availability, UMAP_AVAILABLE

# Matplotlib imports
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Import theme manager
from core.plot_themes import PlotThemeManager

from dialogs.export_mixin import ExportMixin


class ClickableGLViewWidget(gl.GLViewWidget if OPENGL_AVAILABLE else object):
    """Custom GLViewWidget that emits signals on mouse clicks for point selection."""

    def __init__(self, parent=None):
        if not OPENGL_AVAILABLE:
            return
        super().__init__(parent)
        self._click_callback = None
        self._coords = None  # Normalized coordinates for hit detection

    def set_click_callback(self, callback):
        """Set callback function(point_idx) for click events."""
        self._click_callback = callback

    def set_coords(self, coords):
        """Set coordinates for hit detection."""
        self._coords = coords

    def mouseReleaseEvent(self, event):
        """Detect clicks and find closest point."""
        super().mouseReleaseEvent(event)

        if self._click_callback is None or self._coords is None:
            return

        # Only handle left clicks (not drags)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        # Get click position
        x, y = event.pos().x(), event.pos().y()

        # Project 3D points to screen coordinates and find closest
        closest_idx = self._find_closest_point(x, y)

        if closest_idx is not None:
            self._click_callback(closest_idx)

    def _find_closest_point(self, screen_x, screen_y):
        """Find the closest point to screen coordinates using ray casting."""
        if self._coords is None or len(self._coords) == 0:
            return None

        try:
            # Get view and projection info
            view = self.viewMatrix()
            proj = self.projectionMatrix()

            # Get widget size
            w, h = self.width(), self.height()

            # Calculate screen positions for all points
            min_dist = float('inf')
            closest_idx = None

            for i, (px, py, pz) in enumerate(self._coords):
                # Transform point through view and projection matrices
                # This is a simplified screen projection
                point = np.array([px, py, pz, 1.0])

                # Apply view matrix
                view_array = np.array([view.row(j) for j in range(4)])
                view_point = view_array @ point

                # Apply projection matrix
                proj_array = np.array([proj.row(j) for j in range(4)])
                clip_point = proj_array @ view_point

                # Perspective division
                if clip_point[3] != 0:
                    ndc = clip_point[:3] / clip_point[3]
                else:
                    continue

                # NDC to screen coordinates
                sx = (ndc[0] + 1) * w / 2
                sy = (1 - ndc[1]) * h / 2  # Flip Y

                # Calculate distance
                dist = np.sqrt((sx - screen_x)**2 + (sy - screen_y)**2)

                if dist < min_dist and dist < 30:  # 30 pixel threshold
                    min_dist = dist
                    closest_idx = i

            return closest_idx

        except Exception as e:
            print(f"[3D Click] Error in hit detection: {e}")
            return None


class AdvancedPeakEditorDialog(ExportMixin, QDialog):
    """Dialog for reviewing edge cases in peak classification."""

    def __init__(self, main_window, parent=None):
        """
        Initialize the Advanced Peak Editor dialog.

        Args:
            main_window: Reference to MainWindow instance
            parent: Parent widget (optional)
        """
        super().__init__(parent)
        self.main_window = main_window

        # Edge case data
        self.edge_cases = []  # List of edge case dictionaries
        self.current_edge_idx = -1  # Index in edge_cases list
        self.reviewed_peaks = set()  # Set of (sweep, peak_idx) tuples that were reviewed

        # Tab 2: UMAP state variables
        self._umap_coords = None           # np.ndarray (N, 2) or (N, 3) - current display
        self._umap_coords_2d = None        # Cached 2D embedding
        self._umap_coords_3d = None        # Cached 3D embedding
        self._umap_breath_metadata = []    # List[Dict] with sweep, breath_idx, peak info
        self._umap_feature_matrix = None   # np.ndarray (N, F) - scaled features
        self._umap_selected_idx = None     # Currently selected point index
        self._umap_is_3d = False           # Start with 2D (more compatible)
        self._umap_color_mode = 'class'    # 'class' or 'metric'
        self._umap_color_metric = None     # Metric key if color_mode == 'metric'
        self._umap_method = None           # 'umap' or 'pca' (actual method used)
        self._umap_reviewed_breaths = set()  # Set of (sweep, peak_idx) tuples reviewed in Tab 2
        self._umap_changes_count = 0       # Counter for classification changes
        self._umap_cache_params = None     # Cached parameters to detect changes

        # Tab 2: Decision boundary overlay state
        self._overlay_type = 'None'        # 'None', 'Boundary Lines', 'Probability Heatmap', 'Both'
        self._overlay_classifier = None    # Classifier object for decision boundaries
        self._overlay_classifier_type = None  # 'gmm', 'noise', 'xgboost', 'rf', 'mlp'
        self._overlay_heatmap_item = None  # PyQtGraph ImageItem for heatmap
        self._overlay_boundary_items = []  # List of PyQtGraph PlotCurveItems for boundaries
        self._overlay_opacity = 0.5        # Opacity for overlays (0.0-1.0)
        self._available_classifiers = {}   # Dict of available classifiers {name: (classifier, type)}

        # Tab 2: Animation state
        self._anim_timer = None            # QTimer for animation
        self._anim_playing = False         # Is animation currently playing
        self._anim_time_sorted_indices = None  # Indices sorted by time
        self._anim_current_step = 0        # Current step in animation
        self._anim_speed_ms = 200          # Milliseconds between frames
        self._anim_mode = 'Highlight'      # 'Highlight', 'Cumulative', 'Trail'
        self._anim_trail_length = 20       # Number of points in trail mode
        self._anim_visible_indices = None  # Set of indices visible in animation

        # Tab 2: Selection highlight
        self._selection_highlight_item = None  # PyQtGraph ScatterPlotItem for highlight ring
        self._selection_highlight_3d = None    # GL scatter for 3D highlight
        self._scatter_selection_highlights = {}  # Dict of scatter plot key -> highlight item

        # Tab 2: Confidence navigation state
        self._sorted_breath_indices = []       # List of indices into _umap_breath_metadata, sorted by confidence
        self._sorted_nav_position = -1         # Current position in _sorted_breath_indices

        # Tab 2: UMAP color scheme
        self.UMAP_COLORS = {
            'eupnea': (46, 204, 113, 255),      # Green #2ecc71
            'sniffing': (155, 89, 182, 255),    # Purple #9b59b6
            'sigh': (241, 196, 15, 255),        # Gold #f1c40f
            'noise': (192, 57, 43, 255),        # Red #c0392b
            'unclassified': (127, 140, 141, 128),  # Gray with transparency
        }

        self.setWindowTitle("Advanced Peak Editor")

        # Make dialog resizable with maximize button and non-modal (can interact with main window)
        self.setWindowFlags(
            Qt.WindowType.Window |  # Make it a regular window, not a dialog
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )
        self.setModal(False)  # Non-blocking - can interact with main window
        self.setSizeGripEnabled(True)  # Add size grip for easier resizing

        # Set size based on screen dimensions (80% of screen, max 1400x900)
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen:
            screen_size = screen.availableGeometry()
            width = min(int(screen_size.width() * 0.8), 1400)
            height = min(int(screen_size.height() * 0.85), 900)
            self.resize(width, height)
            # Center on screen
            x = (screen_size.width() - width) // 2
            y = (screen_size.height() - height) // 2
            self.move(x, y)
        else:
            self.resize(1200, 800)  # Fallback size

        # Apply dark theme to entire dialog
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 3px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                color: #cccccc;
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
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #6a6a6a;
                border-color: #2d2d30;
            }
            QRadioButton {
                color: #cccccc;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #666;
                border-radius: 8px;
                background-color: #2a2a2a;
            }
            QRadioButton::indicator:checked {
                background-color: #2a7fff;
                border-color: #2a7fff;
            }
            QRadioButton::indicator:hover {
                border-color: #2a7fff;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border: 2px solid #666;
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
            QComboBox {
                background-color: #252526;
                color: #cccccc;
                border: 1px solid #3e3e42;
                padding: 3px 6px;
                border-radius: 3px;
            }
            QComboBox:hover {
                background-color: #2d2d30;
            }
            QComboBox::drop-down {
                background: #2d2d30;
                border-left: 1px solid #3e3e42;
                width: 18px;
            }
            QComboBox QAbstractItemView {
                background: #252526;
                color: #cccccc;
                selection-background-color: #094771;
                selection-color: #ffffff;
                border: 1px solid #3e3e42;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e42;
                background: #1e1e1e;
            }
            QTabBar::tab {
                background: #2d2d30;
                color: #cccccc;
                padding: 8px 16px;
                border: 1px solid #3e3e42;
                border-bottom: none;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #1e1e1e;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background: #3e3e42;
            }
        """)

        self._init_ui()
        self._connect_signals()
        self._enable_dark_title_bar()
        self.setup_export_menu()

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

    def keyPressEvent(self, event):
        """Handle key press events - ESC to close, Ctrl+W to close."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        elif event.key() == Qt.Key.Key_W and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self.close()
        else:
            super().keyPressEvent(event)

    def _init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()

        # Tab widget for different review types
        self.tab_widget = QTabWidget()

        # Tab 1: Breath/Noise (hidden by default)
        self.breath_noise_tab = self._create_breath_noise_tab()
        self.tab_widget.addTab(self.breath_noise_tab, "Breath/Noise")
        self.tab_widget.setTabVisible(0, False)  # Hide Breath/Noise tab

        # Tab 2: Eupnea/Sniff with UMAP visualization (main tab)
        self.eupnea_sniff_tab = self._create_eupnea_sniff_tab()
        self.tab_widget.addTab(self.eupnea_sniff_tab, "Breath Review")

        # Tab 3: Sigh/Normal (hidden by default)
        self.sigh_normal_tab = QWidget()
        sigh_layout = QVBoxLayout()
        sigh_layout.addWidget(QLabel("Sigh/Normal review - Coming soon"))
        self.sigh_normal_tab.setLayout(sigh_layout)
        self.tab_widget.addTab(self.sigh_normal_tab, "Sigh/Normal")
        self.tab_widget.setTabVisible(2, False)  # Hide Sigh/Normal tab

        # Tab 4: UMAP Settings
        self.umap_settings_tab = self._create_umap_settings_tab()
        self.tab_widget.addTab(self.umap_settings_tab, "UMAP Settings")

        # Start on the Breath Review tab (index 1)
        self.tab_widget.setCurrentIndex(1)

        main_layout.addWidget(self.tab_widget)

        self.setLayout(main_layout)

    def _create_breath_noise_tab(self) -> QWidget:
        """Create the Breath/Noise review tab (Tab 1)."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Main splitter (left controls, right plot)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel (15% width)
        left_panel = self._create_left_panel_tab1()
        splitter.addWidget(left_panel)

        # Right panel (85% width)
        right_panel = self._create_plot_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (15% left, 85% right)
        splitter.setSizes([150, 850])

        layout.addWidget(splitter)

        # Bottom action bar
        action_bar = self._create_action_bar_tab1()
        layout.addWidget(action_bar)

        tab.setLayout(layout)
        return tab

    def _create_left_panel_tab1(self) -> QWidget:
        """Create the left control panel for Tab 1."""
        widget = QWidget()
        layout = QVBoxLayout()

        # === Edge Case Type ===
        edge_group = QGroupBox("Edge Case Type")
        edge_layout = QVBoxLayout()

        self.edge_type_group = QButtonGroup()

        self.edge_p_edge_radio = QRadioButton("Threshold P_edge")
        self.edge_p_edge_radio.setChecked(True)
        self.edge_type_group.addButton(self.edge_p_edge_radio, 0)
        edge_layout.addWidget(self.edge_p_edge_radio)

        self.edge_ml_disagree_radio = QRadioButton("ML Disagreement")
        self.edge_type_group.addButton(self.edge_ml_disagree_radio, 1)
        edge_layout.addWidget(self.edge_ml_disagree_radio)

        # Model comparison dropdowns for disagreement detection
        ml_compare_layout = QHBoxLayout()
        ml_compare_layout.addWidget(QLabel("  Compare:"))
        self.ml_model_a_combo = QComboBox()
        self.ml_model_a_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])
        ml_compare_layout.addWidget(self.ml_model_a_combo)
        ml_compare_layout.addWidget(QLabel("vs"))
        self.ml_model_b_combo = QComboBox()
        self.ml_model_b_combo.addItems(["Threshold", "XGBoost", "Random Forest", "MLP"])
        self.ml_model_b_combo.setCurrentIndex(1)  # Default to XGBoost
        ml_compare_layout.addWidget(self.ml_model_b_combo)
        ml_compare_layout.addStretch()
        edge_layout.addLayout(ml_compare_layout)

        self.edge_ml_prob_radio = QRadioButton("ML Probability")
        self.edge_type_group.addButton(self.edge_ml_prob_radio, 2)
        edge_layout.addWidget(self.edge_ml_prob_radio)

        edge_group.setLayout(edge_layout)
        layout.addWidget(edge_group)

        # === Sort Order ===
        sort_group = QGroupBox("Sort")
        sort_layout = QVBoxLayout()

        self.sort_order_group = QButtonGroup()

        self.sort_asc_radio = QRadioButton("Ascending")
        self.sort_order_group.addButton(self.sort_asc_radio, 0)
        sort_layout.addWidget(self.sort_asc_radio)

        self.sort_desc_radio = QRadioButton("Descending")
        self.sort_desc_radio.setChecked(True)
        self.sort_order_group.addButton(self.sort_desc_radio, 1)
        sort_layout.addWidget(self.sort_desc_radio)

        sort_group.setLayout(sort_layout)
        layout.addWidget(sort_group)

        # === Display Options ===
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout()

        self.show_events_checkbox = QCheckBox("Show Events")
        self.show_events_checkbox.setChecked(True)
        display_layout.addWidget(self.show_events_checkbox)

        self.show_peaks_checkbox = QCheckBox("Show All Peaks")
        self.show_peaks_checkbox.setChecked(True)
        display_layout.addWidget(self.show_peaks_checkbox)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # === Plot Controls ===
        plot_group = QGroupBox("Plot")
        plot_layout = QVBoxLayout()

        plot_layout.addWidget(QLabel("Window:"))
        self.window_size_combo = QComboBox()
        self.window_size_combo.addItems(["±0.5s", "±1s", "±2s", "±5s", "±10s"])
        self.window_size_combo.setCurrentText("±2s")
        plot_layout.addWidget(self.window_size_combo)

        plot_layout.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        plot_layout.addWidget(self.theme_combo)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # === Detect Button ===
        self.detect_btn = QPushButton("Detect Edge Cases")
        self.detect_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self.detect_btn)

        layout.addStretch()

        widget.setLayout(layout)
        return widget

    def _create_plot_panel(self) -> QWidget:
        """Create the plot panel with matplotlib canvas."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Matplotlib canvas
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Initialize theme manager and apply default theme
        self.theme_manager = PlotThemeManager(default_theme='dark')
        self.theme_manager.apply_theme(self.ax, self.fig, 'dark')
        self.canvas.draw()

        # Info label below plot
        self.info_label = QLabel("No edge cases loaded. Click 'Detect Edge Cases' to begin.")
        self.info_label.setStyleSheet("font-size: 11pt; padding: 5px;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        widget.setLayout(layout)
        return widget

    def _create_action_bar_tab1(self) -> QGroupBox:
        """Create the bottom action bar for Tab 1."""
        group = QGroupBox("Actions")
        layout = QVBoxLayout()

        # Navigation row
        nav_layout = QHBoxLayout()

        self.first_btn = QPushButton("◀◀ First")
        self.first_btn.setEnabled(False)
        nav_layout.addWidget(self.first_btn)

        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setEnabled(False)
        nav_layout.addWidget(self.next_btn)

        self.last_btn = QPushButton("Last ▶▶")
        self.last_btn.setEnabled(False)
        nav_layout.addWidget(self.last_btn)

        nav_layout.addStretch()

        self.progress_label = QLabel("Progress: 0/0")
        self.progress_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        nav_layout.addWidget(self.progress_label)

        layout.addLayout(nav_layout)

        # Classification row
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Classification:"))

        self.mark_breath_btn = QPushButton("Mark as BREATH")
        self.mark_breath_btn.setEnabled(False)
        self.mark_breath_btn.setStyleSheet("background-color: #2d5016; padding: 8px;")
        class_layout.addWidget(self.mark_breath_btn)

        self.mark_noise_btn = QPushButton("Mark as NOISE")
        self.mark_noise_btn.setEnabled(False)
        self.mark_noise_btn.setStyleSheet("background-color: #5a1a1a; padding: 8px;")
        class_layout.addWidget(self.mark_noise_btn)

        self.skip_btn = QPushButton("Skip/Uncertain")
        self.skip_btn.setEnabled(False)
        self.skip_btn.setStyleSheet("padding: 8px;")
        class_layout.addWidget(self.skip_btn)

        class_layout.addStretch()

        layout.addLayout(class_layout)

        # Utility row
        util_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Edge Cases CSV")
        self.export_btn.setEnabled(False)
        util_layout.addWidget(self.export_btn)

        self.jump_main_btn = QPushButton("Jump to Main Window")
        self.jump_main_btn.setEnabled(False)
        util_layout.addWidget(self.jump_main_btn)

        util_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        util_layout.addWidget(close_btn)

        layout.addLayout(util_layout)

        group.setLayout(layout)
        return group

    # ========== Tab 2: Eupnea/Sniff with UMAP ==========

    def _create_eupnea_sniff_tab(self) -> QWidget:
        """Create the Eupnea/Sniff UMAP visualization tab (Tab 2)."""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Main horizontal splitter (left controls 25%, right plots 75%)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel with controls (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMaximumWidth(320)
        left_panel = self._create_left_panel_tab2()
        left_scroll.setWidget(left_panel)
        main_splitter.addWidget(left_scroll)

        # Right panel container with vertical splitter for resizable trace/plots
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Vertical splitter between trace panel and bottom plots
        self._vertical_splitter = QSplitter(Qt.Orientation.Vertical)

        # 1. Trace panel with zoom buttons (TOP - resizable)
        trace_container = self._create_trace_panel_with_zoom()
        self._vertical_splitter.addWidget(trace_container)

        # Bottom section (classification bar + plots + info)
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(2)

        # 2. Classification button bar
        class_bar = self._create_classification_bar()
        bottom_layout.addWidget(class_bar)

        # 3. Scrollable plots area (UMAP + scatter plots)
        scroll_plots = self._create_scrollable_plots_area()
        bottom_layout.addWidget(scroll_plots, stretch=1)

        # 4. Info label at bottom
        self._info_label_tab2 = QLabel("Click 'Compute UMAP' to visualize breath types.")
        self._info_label_tab2.setStyleSheet("font-size: 10pt; padding: 3px; color: #cccccc;")
        bottom_layout.addWidget(self._info_label_tab2)

        bottom_widget.setLayout(bottom_layout)
        self._vertical_splitter.addWidget(bottom_widget)

        # Set initial split ratio (35% trace, 65% bottom)
        self._vertical_splitter.setSizes([350, 650])

        right_layout.addWidget(self._vertical_splitter)

        right_widget.setLayout(right_layout)
        main_splitter.addWidget(right_widget)

        # Set horizontal split ratio (25% left, 75% right)
        main_splitter.setSizes([250, 750])

        layout.addWidget(main_splitter)

        tab.setLayout(layout)
        return tab

    def _create_left_panel_tab2(self) -> QWidget:
        """Create the left control panel for Tab 2 (UMAP settings)."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # === Feature Selection ===
        feature_group = QGroupBox("Features for UMAP")
        feature_layout = QGridLayout()

        # Available features (common ones for breath classification)
        self._feature_checkboxes = {}
        features = [
            ('if', 'Inst. Freq'),
            ('ti', 'Ti'),
            ('te', 'Te'),
            ('amp_insp', 'Amp Insp'),
            ('amp_exp', 'Amp Exp'),
            ('max_dinsp', 'Max dInsp'),
            ('area_insp', 'Area Insp'),
            ('ibi', 'IBI'),
        ]

        # Default features checked
        default_features = {'if', 'ti', 'amp_insp', 'max_dinsp'}

        for i, (key, label) in enumerate(features):
            cb = QCheckBox(label)
            cb.setChecked(key in default_features)
            self._feature_checkboxes[key] = cb
            row, col = divmod(i, 2)
            feature_layout.addWidget(cb, row, col)

        feature_group.setLayout(feature_layout)
        layout.addWidget(feature_group)

        # === UMAP Parameters ===
        umap_group = QGroupBox("UMAP Parameters")
        umap_layout = QVBoxLayout()

        # n_neighbors
        nn_layout = QHBoxLayout()
        nn_layout.addWidget(QLabel("n_neighbors:"))
        self._umap_n_neighbors = QSpinBox()
        self._umap_n_neighbors.setRange(2, 100)
        self._umap_n_neighbors.setValue(15)
        nn_layout.addWidget(self._umap_n_neighbors)
        nn_layout.addStretch()
        umap_layout.addLayout(nn_layout)

        # min_dist
        md_layout = QHBoxLayout()
        md_layout.addWidget(QLabel("min_dist:"))
        self._umap_min_dist = QDoubleSpinBox()
        self._umap_min_dist.setRange(0.0, 1.0)
        self._umap_min_dist.setSingleStep(0.05)
        self._umap_min_dist.setValue(0.1)
        md_layout.addWidget(self._umap_min_dist)
        md_layout.addStretch()
        umap_layout.addLayout(md_layout)

        # 2D/3D toggle
        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dimensions:"))
        self._umap_dim_group = QButtonGroup()
        self._umap_2d_radio = QRadioButton("2D")
        self._umap_2d_radio.setChecked(True)
        self._umap_3d_radio = QRadioButton("3D")
        self._umap_dim_group.addButton(self._umap_2d_radio, 2)
        self._umap_dim_group.addButton(self._umap_3d_radio, 3)
        dim_layout.addWidget(self._umap_2d_radio)
        dim_layout.addWidget(self._umap_3d_radio)
        dim_layout.addStretch()
        umap_layout.addLayout(dim_layout)

        # Disable 3D if OpenGL not available
        if not OPENGL_AVAILABLE:
            self._umap_3d_radio.setEnabled(False)
            self._umap_3d_radio.setToolTip("PyQtGraph OpenGL not available")

        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group)

        # === Scatter Comparisons ===
        scatter_group = QGroupBox("Scatter Comparisons")
        scatter_layout = QVBoxLayout()

        self._scatter_checkboxes = {}
        scatter_pairs = [
            ('if_ti', 'IF vs Ti'),
            ('if_amp', 'IF vs Amp'),
            ('ti_te', 'Ti vs Te'),
            ('amp_dinsp', 'Amp vs dInsp'),
        ]

        for key, label in scatter_pairs:
            cb = QCheckBox(label)
            cb.setChecked(key in {'if_ti', 'if_amp'})  # Default first two checked
            self._scatter_checkboxes[key] = cb
            scatter_layout.addWidget(cb)

        scatter_group.setLayout(scatter_layout)
        layout.addWidget(scatter_group)

        # === Color Settings ===
        color_group = QGroupBox("Color By")
        color_layout = QVBoxLayout()

        self._color_mode_group = QButtonGroup()
        self._color_class_radio = QRadioButton("Class")
        self._color_class_radio.setChecked(True)
        self._color_mode_group.addButton(self._color_class_radio, 0)
        color_layout.addWidget(self._color_class_radio)

        self._color_confidence_radio = QRadioButton("Confidence (alpha)")
        self._color_mode_group.addButton(self._color_confidence_radio, 1)
        color_layout.addWidget(self._color_confidence_radio)

        metric_layout = QHBoxLayout()
        self._color_metric_radio = QRadioButton("Metric:")
        self._color_mode_group.addButton(self._color_metric_radio, 2)
        metric_layout.addWidget(self._color_metric_radio)

        self._color_metric_combo = QComboBox()
        self._color_metric_combo.addItems(['if', 'ti', 'te', 'amp_insp', 'amp_exp', 'ibi'])
        self._color_metric_combo.setEnabled(False)
        metric_layout.addWidget(self._color_metric_combo)
        metric_layout.addStretch()
        color_layout.addLayout(metric_layout)

        # Color by Time option
        self._color_time_radio = QRadioButton("Time (temporal order)")
        self._color_mode_group.addButton(self._color_time_radio, 3)
        color_layout.addWidget(self._color_time_radio)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # === Animation Controls ===
        animation_group = QGroupBox("Animation")
        animation_layout = QVBoxLayout()

        # Play/Pause controls
        anim_btn_layout = QHBoxLayout()
        self._anim_prev_btn = QPushButton("◀")
        self._anim_prev_btn.setFixedWidth(30)
        self._anim_prev_btn.setToolTip("Previous breath (by time)")
        anim_btn_layout.addWidget(self._anim_prev_btn)

        self._anim_play_btn = QPushButton("▶")
        self._anim_play_btn.setFixedWidth(30)
        self._anim_play_btn.setToolTip("Play/Pause animation")
        self._anim_play_btn.setCheckable(True)
        anim_btn_layout.addWidget(self._anim_play_btn)

        self._anim_next_btn = QPushButton("▶▶")
        self._anim_next_btn.setFixedWidth(30)
        self._anim_next_btn.setToolTip("Next breath (by time)")
        anim_btn_layout.addWidget(self._anim_next_btn)

        anim_btn_layout.addStretch()
        animation_layout.addLayout(anim_btn_layout)

        # Speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self._anim_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._anim_speed_slider.setRange(1, 10)  # 1 = slow, 10 = fast
        self._anim_speed_slider.setValue(5)
        self._anim_speed_slider.setMaximumWidth(80)
        speed_layout.addWidget(self._anim_speed_slider)
        self._anim_speed_label = QLabel("5x")
        speed_layout.addWidget(self._anim_speed_label)
        animation_layout.addLayout(speed_layout)

        # Animation mode selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self._anim_mode_combo = QComboBox()
        self._anim_mode_combo.addItems(["Highlight", "Cumulative", "Trail"])
        self._anim_mode_combo.setToolTip(
            "Highlight: Move highlight through points\n"
            "Cumulative: Points appear and stay as animation progresses\n"
            "Trail: Show last N points with fading opacity"
        )
        mode_layout.addWidget(self._anim_mode_combo)
        animation_layout.addLayout(mode_layout)

        # Trail length slider (for Trail mode)
        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail:"))
        self._anim_trail_slider = QSlider(Qt.Orientation.Horizontal)
        self._anim_trail_slider.setRange(5, 50)  # 5-50 points in trail
        self._anim_trail_slider.setValue(20)
        self._anim_trail_slider.setMaximumWidth(80)
        trail_layout.addWidget(self._anim_trail_slider)
        self._anim_trail_label = QLabel("20")
        trail_layout.addWidget(self._anim_trail_label)
        animation_layout.addLayout(trail_layout)

        # Time range info
        self._anim_time_label = QLabel("Time: --")
        self._anim_time_label.setStyleSheet("font-size: 9pt; color: #888888;")
        animation_layout.addWidget(self._anim_time_label)

        animation_group.setLayout(animation_layout)
        layout.addWidget(animation_group)

        # === Display Options ===
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()

        self._show_events_cb_tab2 = QCheckBox("Show breath events")
        self._show_events_cb_tab2.setChecked(True)
        display_layout.addWidget(self._show_events_cb_tab2)

        # Visibility toggles for UMAP scatter
        self._show_noise_cb = QCheckBox("Show noise points")
        self._show_noise_cb.setChecked(True)
        self._show_noise_cb.setToolTip("Show breaths classified as noise in UMAP scatter")
        display_layout.addWidget(self._show_noise_cb)

        self._show_unclassified_cb = QCheckBox("Show unclassified points")
        self._show_unclassified_cb.setChecked(True)
        self._show_unclassified_cb.setToolTip("Show breaths without classification in UMAP scatter")
        display_layout.addWidget(self._show_unclassified_cb)

        # 3D rendering options
        self._3d_opaque_cb = QCheckBox("3D Opaque Mode (fix white dots)")
        self._3d_opaque_cb.setChecked(True)  # Default to opaque to avoid the white dot issue
        self._3d_opaque_cb.setToolTip(
            "Enable opaque rendering in 3D mode to fix dots appearing white.\n"
            "Uncheck to enable transparency (may cause color issues with overlapping points)."
        )
        display_layout.addWidget(self._3d_opaque_cb)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # === Decision Boundary Overlays ===
        overlay_group = QGroupBox("Overlays")
        overlay_layout = QVBoxLayout()

        # Overlay type selector
        overlay_type_layout = QHBoxLayout()
        overlay_type_layout.addWidget(QLabel("Boundary:"))
        self._overlay_type_combo = QComboBox()
        self._overlay_type_combo.addItems(["None", "Boundary Lines", "Probability Heatmap", "Both"])
        self._overlay_type_combo.setCurrentText("None")
        overlay_type_layout.addWidget(self._overlay_type_combo)
        overlay_layout.addLayout(overlay_type_layout)

        # Classifier selector
        classifier_layout = QHBoxLayout()
        classifier_layout.addWidget(QLabel("Classifier:"))
        self._overlay_classifier_combo = QComboBox()
        # Start with empty combo - will be populated by _refresh_available_classifiers()
        self._overlay_classifier_combo.addItem("(No models available)")
        classifier_layout.addWidget(self._overlay_classifier_combo)
        overlay_layout.addLayout(classifier_layout)

        # Classifier status label
        self._overlay_classifier_status = QLabel("")
        self._overlay_classifier_status.setStyleSheet("color: #888888; font-size: 9pt;")
        overlay_layout.addWidget(self._overlay_classifier_status)

        # Refresh button to detect loaded models
        self._refresh_models_btn = QPushButton("Refresh Models")
        self._refresh_models_btn.setToolTip("Detect loaded ML models for decision boundaries")
        overlay_layout.addWidget(self._refresh_models_btn)

        # Overlay opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self._overlay_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._overlay_opacity_slider.setRange(10, 100)
        self._overlay_opacity_slider.setValue(50)
        self._overlay_opacity_slider.setMaximumWidth(100)
        opacity_layout.addWidget(self._overlay_opacity_slider)
        self._overlay_opacity_label = QLabel("50%")
        opacity_layout.addWidget(self._overlay_opacity_label)
        overlay_layout.addLayout(opacity_layout)

        overlay_group.setLayout(overlay_layout)
        layout.addWidget(overlay_group)

        # === Status / UMAP availability ===
        is_available, msg = get_umap_availability()
        status_text = "UMAP: Available" if is_available else "PCA fallback"
        self._umap_status_label = QLabel(status_text)
        self._umap_status_label.setStyleSheet(
            "color: #2ecc71;" if is_available else "color: #f39c12;"
        )
        layout.addWidget(self._umap_status_label)

        # === Compute UMAP Button ===
        self._compute_umap_btn = QPushButton("Compute UMAP")
        self._compute_umap_btn.setStyleSheet("font-weight: bold; padding: 10px;")
        layout.addWidget(self._compute_umap_btn)

        # === Actions Group ===
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()

        self._export_btn_tab2 = QPushButton("Export CSV")
        self._export_btn_tab2.setEnabled(False)
        actions_layout.addWidget(self._export_btn_tab2)

        self._jump_main_btn_tab2 = QPushButton("Jump to Main")
        self._jump_main_btn_tab2.setEnabled(False)
        actions_layout.addWidget(self._jump_main_btn_tab2)

        actions_group.setLayout(actions_layout)
        layout.addWidget(actions_group)

        layout.addStretch()

        # Close button at bottom
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        widget.setLayout(layout)
        return widget

    def _create_trace_panel_with_zoom(self) -> QWidget:
        """Create the trace view panel with zoom buttons (matches main plot rendering)."""
        container = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Trace plot area (matplotlib canvas)
        trace_widget = QWidget()
        trace_layout = QVBoxLayout()
        trace_layout.setContentsMargins(0, 0, 0, 0)

        # Matplotlib canvas for trace
        self._fig_tab2 = Figure(figsize=(10, 3))
        self._ax_tab2 = self._fig_tab2.add_subplot(111)
        self._canvas_tab2 = FigureCanvas(self._fig_tab2)
        trace_layout.addWidget(self._canvas_tab2)

        # Info label below trace (window dropdown moved to classification bar)
        info_layout = QHBoxLayout()
        self._trace_info_label_tab2 = QLabel("Select a point in UMAP to view trace")
        self._trace_info_label_tab2.setStyleSheet("color: #888888;")
        info_layout.addWidget(self._trace_info_label_tab2)
        info_layout.addStretch()
        trace_layout.addLayout(info_layout)

        # Apply theme
        self.theme_manager.apply_theme(self._ax_tab2, self._fig_tab2, 'dark')
        self._canvas_tab2.draw()

        trace_widget.setLayout(trace_layout)
        main_layout.addWidget(trace_widget, stretch=1)

        # Zoom buttons (stacked vertically on right side)
        zoom_widget = QWidget()
        zoom_widget.setMaximumWidth(40)
        zoom_layout = QVBoxLayout()
        zoom_layout.setContentsMargins(2, 2, 2, 2)
        zoom_layout.setSpacing(4)

        zoom_layout.addStretch()

        self._zoom_out_btn = QPushButton("−")
        self._zoom_out_btn.setToolTip("Zoom out (wider window)")
        self._zoom_out_btn.setFixedSize(32, 32)
        self._zoom_out_btn.setStyleSheet("font-size: 16pt; font-weight: bold;")
        zoom_layout.addWidget(self._zoom_out_btn)

        self._zoom_in_btn = QPushButton("+")
        self._zoom_in_btn.setToolTip("Zoom in (narrower window)")
        self._zoom_in_btn.setFixedSize(32, 32)
        self._zoom_in_btn.setStyleSheet("font-size: 16pt; font-weight: bold;")
        zoom_layout.addWidget(self._zoom_in_btn)

        zoom_layout.addStretch()

        zoom_widget.setLayout(zoom_layout)
        main_layout.addWidget(zoom_widget)

        container.setLayout(main_layout)
        return container

    def _create_classification_bar(self) -> QWidget:
        """Create the unified classification and info bar (single compact row)."""
        widget = QWidget()
        widget.setMaximumHeight(40)  # Compact height
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(4)  # Tighter spacing

        # === CONFIDENCE NAVIGATION SECTION (left side) ===
        btn_style = "padding: 3px 8px; font-size: 9pt;"

        # Navigation buttons FIRST
        self._prev_breath_btn = QPushButton("◀ Prev")
        self._prev_breath_btn.setEnabled(False)
        self._prev_breath_btn.setStyleSheet(btn_style)
        self._prev_breath_btn.setToolTip("Go to previous breath in sorted order")
        layout.addWidget(self._prev_breath_btn)

        self._next_breath_btn = QPushButton("Next ▶")
        self._next_breath_btn.setEnabled(False)
        self._next_breath_btn.setStyleSheet(btn_style)
        self._next_breath_btn.setToolTip("Go to next breath in sorted order")
        layout.addWidget(self._next_breath_btn)

        # Position indicator
        self._nav_position_label = QLabel("--/--")
        self._nav_position_label.setStyleSheet("color: #888888; font-size: 9pt; min-width: 50px;")
        self._nav_position_label.setToolTip("Current position in sorted list")
        layout.addWidget(self._nav_position_label)

        # Sort by confidence dropdown AFTER nav buttons
        layout.addWidget(QLabel("Sort:"))
        self._confidence_sort_combo = QComboBox()
        self._confidence_sort_combo.addItems([
            "Time Order",
            "Noise/Breath (Low→High)",
            "Noise/Breath (High→Low)",
            "Eupnea/Sniff (Low→High)",
            "Eupnea/Sniff (High→Low)",
            "Breath/Sigh (Low→High)",
            "Breath/Sigh (High→Low)"
        ])
        self._confidence_sort_combo.setToolTip("Sort breaths by classifier confidence")
        self._confidence_sort_combo.setMaximumWidth(160)
        layout.addWidget(self._confidence_sort_combo)

        # Gap between navigation and labeling sections
        spacer = QWidget()
        spacer.setFixedWidth(20)
        layout.addWidget(spacer)

        # Separator
        sep_nav = QLabel("|")
        sep_nav.setStyleSheet("color: #555555;")
        layout.addWidget(sep_nav)

        spacer2 = QWidget()
        spacer2.setFixedWidth(10)
        layout.addWidget(spacer2)

        # === LABELING BUTTONS SECTION ===
        self._mark_noise_btn = QPushButton("Noise")
        self._mark_noise_btn.setEnabled(False)
        self._mark_noise_btn.setStyleSheet(f"background-color: #5a1a1a; {btn_style}")
        layout.addWidget(self._mark_noise_btn)

        self._mark_eupnea_btn = QPushButton("Eupnea")
        self._mark_eupnea_btn.setEnabled(False)
        self._mark_eupnea_btn.setStyleSheet(f"background-color: #1d6f42; {btn_style}")
        layout.addWidget(self._mark_eupnea_btn)

        self._mark_sniffing_btn = QPushButton("Sniff")
        self._mark_sniffing_btn.setEnabled(False)
        self._mark_sniffing_btn.setStyleSheet(f"background-color: #5b3a71; {btn_style}")
        layout.addWidget(self._mark_sniffing_btn)

        self._mark_sigh_btn = QPushButton("Sigh")
        self._mark_sigh_btn.setEnabled(False)
        self._mark_sigh_btn.setStyleSheet(f"background-color: #8a7520; {btn_style}")
        layout.addWidget(self._mark_sigh_btn)

        self._mark_none_btn = QPushButton("None")
        self._mark_none_btn.setEnabled(False)
        self._mark_none_btn.setStyleSheet(btn_style)
        layout.addWidget(self._mark_none_btn)

        self._skip_btn_tab2 = QPushButton("→")
        self._skip_btn_tab2.setEnabled(False)
        self._skip_btn_tab2.setStyleSheet(btn_style)
        self._skip_btn_tab2.setToolTip("Skip to next breath without changing classification")
        self._skip_btn_tab2.setFixedWidth(30)
        layout.addWidget(self._skip_btn_tab2)

        # Separator
        sep = QLabel("|")
        sep.setStyleSheet("color: #555555;")
        layout.addWidget(sep)

        # Window size dropdown (compact)
        layout.addWidget(QLabel("Win:"))
        self._window_size_combo_tab2 = QComboBox()
        self._window_size_combo_tab2.addItems(["±0.5s", "±1s", "±2s", "±5s", "±10s"])
        self._window_size_combo_tab2.setCurrentText("±2s")
        self._window_size_combo_tab2.setMaximumWidth(65)
        layout.addWidget(self._window_size_combo_tab2)

        # Separator
        sep2 = QLabel("|")
        sep2.setStyleSheet("color: #555555;")
        layout.addWidget(sep2)

        # Breath info label (compact)
        self._breath_info_label_tab2 = QLabel("Breath: --")
        self._breath_info_label_tab2.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        layout.addWidget(self._breath_info_label_tab2)

        layout.addStretch()

        # Changes counter
        self._changes_label_tab2 = QLabel("Changes: 0")
        self._changes_label_tab2.setStyleSheet("font-weight: bold; color: #cccccc; font-size: 9pt;")
        layout.addWidget(self._changes_label_tab2)

        widget.setLayout(layout)
        return widget

    def _create_scrollable_plots_area(self) -> QScrollArea:
        """Create the scrollable area containing UMAP and scatter plots in a tiled grid."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container for all plots in a grid (3 columns: UMAP takes 1/3, scatter plots tile)
        container = QWidget()
        self._plots_container_layout = QGridLayout()
        self._plots_container_layout.setContentsMargins(0, 0, 0, 0)
        self._plots_container_layout.setSpacing(10)

        # Set column stretch so each column is roughly equal (1/3 each)
        self._plots_container_layout.setColumnStretch(0, 1)
        self._plots_container_layout.setColumnStretch(1, 1)
        self._plots_container_layout.setColumnStretch(2, 1)

        # UMAP panel (1/3 width - first column, row 0)
        self._umap_full_width = False  # Start in tiled mode
        self._umap_container = self._create_umap_plot_widget()
        self._plots_container_layout.addWidget(self._umap_container, 0, 0)  # Row 0, Col 0

        # Store scatter plot widgets (will be added in remaining grid cells)
        self._scatter_plots = {}
        self._scatter_grid_next_position = (0, 1)  # Start at row 0, col 1

        container.setLayout(self._plots_container_layout)
        scroll_area.setWidget(container)

        return scroll_area

    def _create_umap_plot_widget(self) -> QWidget:
        """Create the UMAP plot widget (can be full or half width)."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create PyQtGraph PlotWidget for 2D (default)
        self._umap_plot_widget = pg.PlotWidget()
        self._umap_plot_widget.setBackground('#1e1e1e')
        self._umap_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._umap_plot_widget.setLabel('bottom', 'UMAP 1')
        self._umap_plot_widget.setLabel('left', 'UMAP 2')
        self._umap_plot_widget.setMinimumHeight(300)
        self._umap_plot_widget.setMinimumWidth(300)

        # Don't lock aspect - allow flexible resizing
        self._umap_plot_widget.setAspectLocked(False)

        # Create scatter plot item
        self._umap_scatter = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen(None),
            hoverable=True,
            hoverSize=12,
            hoverPen=pg.mkPen('w', width=2)
        )
        self._umap_plot_widget.addItem(self._umap_scatter)

        # Connect click handler
        self._umap_scatter.sigClicked.connect(self._on_umap_point_clicked)

        # Double-click to toggle full/half width
        self._umap_plot_widget.scene().sigMouseClicked.connect(self._on_umap_double_click)

        layout.addWidget(self._umap_plot_widget)

        # Placeholder for 3D widget (created on demand)
        self._umap_gl_widget = None
        self._umap_gl_scatter = None

        # Control buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(5)

        # Recenter button
        self._umap_recenter_btn = QPushButton("⟲ Recenter")
        self._umap_recenter_btn.setFixedWidth(80)
        self._umap_recenter_btn.setToolTip("Reset UMAP view to show all points")
        self._umap_recenter_btn.clicked.connect(self._recenter_umap_view)
        btn_row.addWidget(self._umap_recenter_btn)

        # Pop-out button
        self._umap_popout_btn = QPushButton("⧉ Pop Out")
        self._umap_popout_btn.setFixedWidth(80)
        self._umap_popout_btn.setToolTip("Open UMAP in separate window")
        self._umap_popout_btn.clicked.connect(self._popout_umap_window)
        btn_row.addWidget(self._umap_popout_btn)

        btn_row.addStretch()

        # UMAP is now always in tiled mode (1/3 width)
        # No toggle label needed

        layout.addLayout(btn_row)

        widget.setLayout(layout)
        return widget

    def _create_umap_settings_tab(self) -> QWidget:
        """Create the UMAP Settings tab for feature selection and parameters."""
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # === LEFT COLUMN: Feature Selection ===
        left_column = QVBoxLayout()

        # Feature Selection Group
        feature_group = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout()

        # Auto-select checkbox
        self._auto_select_features_cb = QCheckBox("Auto-select features")
        self._auto_select_features_cb.setChecked(True)
        self._auto_select_features_cb.stateChanged.connect(self._on_auto_select_changed)
        feature_layout.addWidget(self._auto_select_features_cb)

        # Auto-select method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self._feature_select_method = QComboBox()
        self._feature_select_method.addItems([
            "Combined (recommended)",
            "PCA (variance-based)",
            "ML: Noise vs Breath",
            "ML: Eupnea vs Sniff",
            "ML: Sigh vs Normal",
            "Union of all ML"
        ])
        self._feature_select_method.currentTextChanged.connect(self._on_feature_method_changed)
        method_layout.addWidget(self._feature_select_method)
        method_layout.addStretch()
        feature_layout.addLayout(method_layout)

        # Top N features
        topn_layout = QHBoxLayout()
        topn_layout.addWidget(QLabel("Top N features:"))
        self._top_n_features = QSpinBox()
        self._top_n_features.setRange(3, 20)
        self._top_n_features.setValue(12)
        topn_layout.addWidget(self._top_n_features)
        topn_layout.addStretch()
        feature_layout.addLayout(topn_layout)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("background-color: #3e3e42;")
        feature_layout.addWidget(sep1)

        # Manual feature selection label
        manual_label = QLabel("Manual Feature Selection:")
        manual_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        feature_layout.addWidget(manual_label)
        override_label = QLabel("(overrides auto when unchecked)")
        override_label.setStyleSheet("font-size: 9pt; color: #888888;")
        feature_layout.addWidget(override_label)

        # Feature checkboxes with importance scores
        self._settings_feature_checkboxes = {}
        self._feature_importance_labels = {}

        all_features = [
            ('if', 'Inst. Freq (IF)'),
            ('ti', 'Insp. Time (Ti)'),
            ('te', 'Exp. Time (Te)'),
            ('amp_insp', 'Insp. Amplitude'),
            ('amp_exp', 'Exp. Amplitude'),
            ('max_dinsp', 'Max dInsp'),
            ('area_insp', 'Insp. Area'),
            ('area_exp', 'Exp. Area'),
            ('ibi', 'Inter-breath Interval'),
            ('pif', 'Peak Insp. Flow'),
            ('pef', 'Peak Exp. Flow'),
            ('duty_cycle', 'Duty Cycle'),
        ]

        features_scroll = QScrollArea()
        features_scroll.setWidgetResizable(True)
        features_scroll.setMaximumHeight(250)
        features_widget = QWidget()
        features_grid = QGridLayout()
        features_grid.setSpacing(4)

        for i, (key, label) in enumerate(all_features):
            cb = QCheckBox(label)
            cb.setChecked(key in {'if', 'ti', 'amp_insp', 'max_dinsp'})  # Default selection
            self._settings_feature_checkboxes[key] = cb

            importance_label = QLabel("--")
            importance_label.setStyleSheet("color: #888888; font-size: 9pt;")
            importance_label.setFixedWidth(50)
            self._feature_importance_labels[key] = importance_label

            features_grid.addWidget(cb, i, 0)
            features_grid.addWidget(importance_label, i, 1)

        features_widget.setLayout(features_grid)
        features_scroll.setWidget(features_widget)
        feature_layout.addWidget(features_scroll)

        # Legend
        legend_label = QLabel("★ = auto-selected  ○ = available")
        legend_label.setStyleSheet("font-size: 9pt; color: #888888; margin-top: 5px;")
        feature_layout.addWidget(legend_label)

        feature_group.setLayout(feature_layout)
        left_column.addWidget(feature_group)

        # Run PCA Analysis button
        self._run_pca_btn = QPushButton("Run PCA Analysis")
        self._run_pca_btn.clicked.connect(self._run_pca_analysis)
        left_column.addWidget(self._run_pca_btn)

        left_column.addStretch()
        main_layout.addLayout(left_column, stretch=1)

        # === MIDDLE COLUMN: PCA Analysis Results ===
        middle_column = QVBoxLayout()

        pca_group = QGroupBox("PCA Analysis")
        pca_layout = QVBoxLayout()

        # Explained variance bars
        var_label = QLabel("Explained Variance by Component:")
        var_label.setStyleSheet("font-weight: bold;")
        pca_layout.addWidget(var_label)

        self._pca_variance_bars = {}
        for i in range(5):
            bar_layout = QHBoxLayout()
            bar_layout.addWidget(QLabel(f"PC{i+1}"))
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%p%")
            bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #3e3e42;
                    border-radius: 3px;
                    background-color: #2d2d30;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4ec9b0;
                }
            """)
            self._pca_variance_bars[i] = bar
            bar_layout.addWidget(bar)
            pca_layout.addLayout(bar_layout)

        # Cumulative variance
        cum_layout = QHBoxLayout()
        cum_layout.addWidget(QLabel("Cumulative:"))
        self._pca_cumulative_bar = QProgressBar()
        self._pca_cumulative_bar.setRange(0, 100)
        self._pca_cumulative_bar.setValue(0)
        self._pca_cumulative_bar.setFormat("%p%")
        self._pca_cumulative_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #2d2d30;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2a7fff;
            }
        """)
        cum_layout.addWidget(self._pca_cumulative_bar)
        pca_layout.addLayout(cum_layout)

        # Top features by loading
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #3e3e42; margin: 10px 0;")
        pca_layout.addWidget(sep2)

        top_label = QLabel("Top Features by Loading:")
        top_label.setStyleSheet("font-weight: bold;")
        pca_layout.addWidget(top_label)

        # Scrollable area for PCA results
        pca_scroll = QScrollArea()
        pca_scroll.setWidgetResizable(True)
        pca_scroll.setMaximumHeight(150)
        pca_scroll.setMinimumHeight(100)
        pca_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pca_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._pca_top_features_label = QLabel("Run PCA analysis to see results")
        self._pca_top_features_label.setStyleSheet("color: #888888;")
        self._pca_top_features_label.setWordWrap(True)
        self._pca_top_features_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        pca_scroll.setWidget(self._pca_top_features_label)
        pca_layout.addWidget(pca_scroll)

        pca_layout.addStretch()
        pca_group.setLayout(pca_layout)
        middle_column.addWidget(pca_group)

        # ML Feature Importance Group
        ml_group = QGroupBox("ML Feature Importance")
        ml_layout = QVBoxLayout()

        # Sort by model dropdown
        sort_layout = QHBoxLayout()
        sort_layout.addWidget(QLabel("Sort by:"))
        self._ml_sort_combo = QComboBox()
        self._ml_sort_combo.addItems(["All models (union)", "XGBoost", "Random Forest", "MLP"])
        self._ml_sort_combo.currentTextChanged.connect(self._refresh_ml_importance)
        sort_layout.addWidget(self._ml_sort_combo)
        sort_layout.addStretch()
        ml_layout.addLayout(sort_layout)

        # Scrollable area for ML importance
        ml_scroll = QScrollArea()
        ml_scroll.setWidgetResizable(True)
        ml_scroll.setMaximumHeight(150)
        ml_scroll.setMinimumHeight(100)
        ml_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        ml_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self._ml_importance_label = QLabel("Load ML models to see feature importance")
        self._ml_importance_label.setStyleSheet("color: #888888;")
        self._ml_importance_label.setWordWrap(True)
        self._ml_importance_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        ml_scroll.setWidget(self._ml_importance_label)
        ml_layout.addWidget(ml_scroll)

        self._refresh_ml_importance_btn = QPushButton("Refresh from Loaded Models")
        self._refresh_ml_importance_btn.clicked.connect(self._refresh_ml_importance)
        ml_layout.addWidget(self._refresh_ml_importance_btn)

        ml_layout.addStretch()
        ml_group.setLayout(ml_layout)
        middle_column.addWidget(ml_group)

        main_layout.addLayout(middle_column, stretch=1)

        # === RIGHT COLUMN: UMAP Parameters & Cache ===
        right_column = QVBoxLayout()

        # UMAP Parameters Group
        params_group = QGroupBox("UMAP Parameters")
        params_layout = QVBoxLayout()

        # n_neighbors
        nn_layout = QHBoxLayout()
        nn_layout.addWidget(QLabel("n_neighbors:"))
        self._settings_n_neighbors = QSpinBox()
        self._settings_n_neighbors.setRange(2, 100)
        self._settings_n_neighbors.setValue(15)
        nn_layout.addWidget(self._settings_n_neighbors)
        nn_layout.addStretch()
        params_layout.addLayout(nn_layout)

        # min_dist
        md_layout = QHBoxLayout()
        md_layout.addWidget(QLabel("min_dist:"))
        self._settings_min_dist = QDoubleSpinBox()
        self._settings_min_dist.setRange(0.0, 1.0)
        self._settings_min_dist.setSingleStep(0.05)
        self._settings_min_dist.setValue(0.1)
        md_layout.addWidget(self._settings_min_dist)
        md_layout.addStretch()
        params_layout.addLayout(md_layout)

        # metric
        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("metric:"))
        self._settings_metric = QComboBox()
        self._settings_metric.addItems(['euclidean', 'manhattan', 'cosine', 'correlation'])
        metric_layout.addWidget(self._settings_metric)
        metric_layout.addStretch()
        params_layout.addLayout(metric_layout)

        # Dimensions
        dim_layout = QHBoxLayout()
        dim_layout.addWidget(QLabel("Dimensions:"))
        self._settings_dim_group = QButtonGroup()
        self._settings_2d_radio = QRadioButton("2D")
        self._settings_2d_radio.setChecked(True)
        self._settings_3d_radio = QRadioButton("3D")
        if not OPENGL_AVAILABLE:
            self._settings_3d_radio.setEnabled(False)
            self._settings_3d_radio.setToolTip("PyQtGraph OpenGL not available")
        self._settings_dim_group.addButton(self._settings_2d_radio, 2)
        self._settings_dim_group.addButton(self._settings_3d_radio, 3)
        dim_layout.addWidget(self._settings_2d_radio)
        dim_layout.addWidget(self._settings_3d_radio)
        dim_layout.addStretch()
        params_layout.addLayout(dim_layout)

        # Random seed
        seed_layout = QHBoxLayout()
        seed_layout.addWidget(QLabel("Random seed:"))
        self._settings_seed = QSpinBox()
        self._settings_seed.setRange(0, 9999)
        self._settings_seed.setValue(42)
        seed_layout.addWidget(self._settings_seed)
        seed_layout.addStretch()
        params_layout.addLayout(seed_layout)

        params_group.setLayout(params_layout)
        right_column.addWidget(params_group)

        # Cache Status Group
        cache_group = QGroupBox("Cache Status")
        cache_layout = QVBoxLayout()

        self._cache_2d_label = QLabel("2D Embedding: Not computed")
        self._cache_2d_label.setStyleSheet("color: #888888;")
        cache_layout.addWidget(self._cache_2d_label)

        self._cache_3d_label = QLabel("3D Embedding: Not computed")
        self._cache_3d_label.setStyleSheet("color: #888888;")
        cache_layout.addWidget(self._cache_3d_label)

        self._cache_features_label = QLabel("Feature matrix: Not computed")
        self._cache_features_label.setStyleSheet("color: #888888;")
        cache_layout.addWidget(self._cache_features_label)

        self._clear_cache_btn = QPushButton("Clear Cache")
        self._clear_cache_btn.clicked.connect(self._clear_umap_cache)
        cache_layout.addWidget(self._clear_cache_btn)

        cache_group.setLayout(cache_layout)
        right_column.addWidget(cache_group)

        # Data Scope Group
        scope_group = QGroupBox("Data Scope")
        scope_layout = QVBoxLayout()

        scope_label = QLabel("Include in UMAP:")
        scope_label.setStyleSheet("font-weight: bold;")
        scope_layout.addWidget(scope_label)

        self._include_breaths_cb = QCheckBox("Real breaths (labeled 1)")
        self._include_breaths_cb.setChecked(True)
        scope_layout.addWidget(self._include_breaths_cb)

        self._include_noise_cb = QCheckBox("Noise peaks (labeled 0)")
        self._include_noise_cb.setChecked(True)
        scope_layout.addWidget(self._include_noise_cb)

        self._include_unclassified_cb = QCheckBox("Unclassified peaks")
        self._include_unclassified_cb.setChecked(True)
        scope_layout.addWidget(self._include_unclassified_cb)

        # Statistics
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setStyleSheet("background-color: #3e3e42;")
        scope_layout.addWidget(sep3)

        self._data_stats_label = QLabel("Load data to see statistics")
        self._data_stats_label.setStyleSheet("color: #888888;")
        scope_layout.addWidget(self._data_stats_label)

        scope_group.setLayout(scope_layout)
        right_column.addWidget(scope_group)

        right_column.addStretch()

        # Compute UMAP Button
        self._settings_compute_btn = QPushButton("Compute UMAP")
        self._settings_compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #094771;
                color: #ffffff;
                border: 2px solid #0a5a8a;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0a5a8a;
            }
        """)
        self._settings_compute_btn.clicked.connect(self._compute_umap_from_settings)
        right_column.addWidget(self._settings_compute_btn)

        main_layout.addLayout(right_column, stretch=1)

        tab.setLayout(main_layout)
        return tab

    def _on_auto_select_changed(self, state):
        """Handle auto-select checkbox change."""
        auto_enabled = state == Qt.CheckState.Checked.value
        self._feature_select_method.setEnabled(auto_enabled)
        self._top_n_features.setEnabled(auto_enabled)

        # Enable/disable manual checkboxes
        for cb in self._settings_feature_checkboxes.values():
            cb.setEnabled(not auto_enabled)

    def _on_feature_method_changed(self, method):
        """Handle feature selection method change."""
        if self._auto_select_features_cb.isChecked():
            self._update_auto_selected_features()

    def _update_auto_selected_features(self):
        """Update feature checkboxes based on auto-selection method."""
        # This will be implemented to actually select features based on PCA or ML importance
        pass

    def _run_pca_analysis(self):
        """Run PCA analysis on all available features."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        st = self.main_window.state

        # Check for any peak data
        has_peaks = (hasattr(st, 'peaks_by_sweep') and st.peaks_by_sweep) or \
                    (hasattr(st, 'all_peaks_by_sweep') and st.all_peaks_by_sweep)
        has_breaths = hasattr(st, 'breath_by_sweep') and st.breath_by_sweep

        if not has_peaks and not has_breaths:
            QMessageBox.warning(self, "No Data", "No peak or breath data available. Run peak detection first.")
            return

        # Get all features for all breaths
        all_features = list(self._settings_feature_checkboxes.keys())

        print(f"[PCA] Attempting PCA with {len(all_features)} features")
        print(f"[PCA] Has peaks_by_sweep: {hasattr(st, 'peaks_by_sweep') and bool(st.peaks_by_sweep)}")
        print(f"[PCA] Has all_peaks_by_sweep: {hasattr(st, 'all_peaks_by_sweep') and bool(st.all_peaks_by_sweep)}")
        print(f"[PCA] Has breath_by_sweep: {has_breaths}")

        try:
            feature_matrix, metadata = self._collect_all_breath_features_tab2(all_features)
            print(f"[PCA] Collected {len(feature_matrix)} valid peaks with {len(all_features)} features each")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to collect features:\n{str(e)}")
            return

        if len(feature_matrix) == 0:
            # Try with just basic features that are more likely to work
            basic_features = ['if', 'ti', 'te', 'amp_insp']
            available_basic = [f for f in basic_features if f in all_features]
            print(f"[PCA] No data with all features. Trying basic features: {available_basic}")

            try:
                feature_matrix, metadata = self._collect_all_breath_features_tab2(available_basic)
                all_features = available_basic  # Use reduced feature set
                print(f"[PCA] With basic features: {len(feature_matrix)} valid peaks")
            except Exception as e:
                pass

        if len(feature_matrix) < 5:
            # Show more helpful error
            msg = f"Need at least 5 breaths for PCA analysis.\n\n"
            msg += f"Found: {len(feature_matrix)} valid data points.\n\n"
            msg += "Possible causes:\n"
            msg += "• Peaks detected but breath features not computed\n"
            msg += "• Missing onset/offset detection\n"
            msg += "• Try running 'Detect Breath Features' first"
            QMessageBox.warning(self, "Insufficient Data", msg)
            return

        # Normalize
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feature_matrix)

        # PCA
        pca = PCA()
        pca.fit(scaled)

        # Update variance bars
        explained = pca.explained_variance_ratio_ * 100
        cumulative = 0
        for i in range(min(5, len(explained))):
            self._pca_variance_bars[i].setValue(int(explained[i]))
            cumulative += explained[i]

        self._pca_cumulative_bar.setValue(int(cumulative))

        # Get top features by loading magnitude
        loadings = np.abs(pca.components_[0])  # First PC loadings
        feature_importance = list(zip(all_features, loadings))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # Update top features label - show ALL features since we have scrollable area
        top_text = "\n".join([f"{i+1}. {f} ({v:.2f})" for i, (f, v) in enumerate(feature_importance)])
        self._pca_top_features_label.setText(top_text)
        self._pca_top_features_label.setStyleSheet("color: #cccccc;")

        # Update importance labels
        for feat, importance in feature_importance:
            if feat in self._feature_importance_labels:
                self._feature_importance_labels[feat].setText(f"{importance:.2f}")
                if importance > 0.5:
                    self._feature_importance_labels[feat].setStyleSheet("color: #4ec9b0; font-weight: bold;")
                else:
                    self._feature_importance_labels[feat].setStyleSheet("color: #888888;")

        # Store for auto-selection
        self._pca_feature_ranking = feature_importance

    def _refresh_ml_importance(self):
        """Refresh ML feature importance from loaded models."""
        st = self.main_window.state

        if not hasattr(st, 'loaded_ml_models') or not st.loaded_ml_models:
            self._ml_importance_label.setText("No ML models loaded.\nLoad models in Analysis Options dialog.")
            self._ml_importance_label.setStyleSheet("color: #f39c12;")
            return

        # Get sort preference
        sort_by = self._ml_sort_combo.currentText() if hasattr(self, '_ml_sort_combo') else "All models (union)"

        importance_text = []
        self._ml_feature_importance = {}

        # Map sort dropdown to model types
        type_filter = None
        if "XGBoost" in sort_by:
            type_filter = 'xgboost'
        elif "Random Forest" in sort_by:
            type_filter = 'rf'
        elif "MLP" in sort_by:
            type_filter = 'mlp'

        for model_key, model_data in st.loaded_ml_models.items():
            model = model_data.get('model')
            metadata = model_data.get('metadata', {})
            feature_names = metadata.get('feature_names', [])
            model_type = metadata.get('model_type', 'unknown')

            if model is None or not feature_names:
                continue

            # Filter by model type if specified
            if type_filter and model_type.lower() != type_filter:
                continue

            # Extract feature importance
            importance = self._extract_feature_importance(model, feature_names)
            if importance is not None:
                importance_text.append(f"\n{model_key} ({model_type}):")

                # Sort by importance - show ALL features for better insight
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_imp:  # Show all features
                    importance_text.append(f"  {feat}: {imp:.3f}")

                self._ml_feature_importance[model_key] = importance

        if importance_text:
            self._ml_importance_label.setText("\n".join(importance_text))
            self._ml_importance_label.setStyleSheet("color: #cccccc;")
        else:
            msg = f"No {sort_by} models found." if type_filter else "Could not extract importance from loaded models."
            self._ml_importance_label.setText(msg)
            self._ml_importance_label.setStyleSheet("color: #f39c12;")

    def _extract_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importance from a trained model."""
        try:
            # sklearn RF and XGBoost have feature_importances_
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                return dict(zip(feature_names, importance))

            # MLP - use absolute mean weights from first layer
            if hasattr(model, 'coefs_') and len(model.coefs_) > 0:
                importance = np.abs(model.coefs_[0]).mean(axis=1)
                if len(importance) == len(feature_names):
                    # Normalize
                    importance = importance / importance.sum()
                    return dict(zip(feature_names, importance))

            return None
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None

    def _clear_umap_cache(self):
        """Clear cached UMAP embeddings."""
        self._umap_coords = None
        self._umap_coords_2d = None
        self._umap_coords_3d = None
        self._umap_feature_matrix = None
        self._umap_breath_metadata = []

        self._cache_2d_label.setText("2D Embedding: Cleared")
        self._cache_2d_label.setStyleSheet("color: #888888;")
        self._cache_3d_label.setText("3D Embedding: Cleared")
        self._cache_3d_label.setStyleSheet("color: #888888;")
        self._cache_features_label.setText("Feature matrix: Cleared")
        self._cache_features_label.setStyleSheet("color: #888888;")

    def _compute_umap_from_settings(self):
        """Compute UMAP using settings from the UMAP Settings tab."""
        # Get selected features
        if self._auto_select_features_cb.isChecked():
            # Use auto-selected features
            selected = self._get_auto_selected_features()
        else:
            # Use manually checked features
            selected = [k for k, cb in self._settings_feature_checkboxes.items() if cb.isChecked()]

        if len(selected) < 2:
            QMessageBox.warning(self, "Insufficient Features", "Please select at least 2 features.")
            return

        # Update the main Tab 2 feature checkboxes to match
        for key, cb in self._feature_checkboxes.items():
            cb.setChecked(key in selected)

        # Update UMAP parameters
        self._umap_n_neighbors.setValue(self._settings_n_neighbors.value())
        self._umap_min_dist.setValue(self._settings_min_dist.value())

        if self._settings_2d_radio.isChecked():
            self._umap_2d_radio.setChecked(True)
        else:
            self._umap_3d_radio.setChecked(True)

        # Clear cache to force recomputation
        self._umap_coords_2d = None
        self._umap_coords_3d = None

        # Switch to Eupnea/Sniff tab and compute
        self.tab_widget.setCurrentIndex(1)  # Switch to Tab 2
        self._compute_umap()

        # Update cache status
        self._update_cache_status()

    def _get_auto_selected_features(self) -> List[str]:
        """Get automatically selected features based on current method."""
        method = self._feature_select_method.currentText()
        n = self._top_n_features.value()

        print(f"[AutoSelect] Method: {method}, n: {n}")

        # Feature name mapping (common alternate names -> standard keys)
        feature_aliases = {
            'instantaneous_frequency': 'if',
            'inst_freq': 'if',
            'frequency': 'if',
            'inspiratory_time': 'ti',
            'insp_time': 'ti',
            'expiratory_time': 'te',
            'exp_time': 'te',
            'inspiratory_amplitude': 'amp_insp',
            'insp_amplitude': 'amp_insp',
            'insp_amp': 'amp_insp',
            'expiratory_amplitude': 'amp_exp',
            'exp_amplitude': 'amp_exp',
            'exp_amp': 'amp_exp',
            'max_inspiratory_rate': 'max_dinsp',
            'max_insp_rate': 'max_dinsp',
            'max_expiratory_rate': 'max_dexp',
            'max_exp_rate': 'max_dexp',
            'inspiratory_area': 'area_insp',
            'insp_area': 'area_insp',
            'expiratory_area': 'area_exp',
            'exp_area': 'area_exp',
            'inter_breath_interval': 'ibi',
        }

        def normalize_feature_name(name: str) -> str:
            """Normalize feature name to match checkbox keys."""
            normalized = name.lower().strip()
            return feature_aliases.get(normalized, name)

        if "PCA" in method:
            if hasattr(self, '_pca_feature_ranking') and self._pca_feature_ranking:
                features = [f for f, _ in self._pca_feature_ranking[:n]]
                print(f"[AutoSelect] PCA selected: {features}")
                return features
            else:
                print("[AutoSelect] PCA ranking not available - run PCA analysis first")

        elif "ML:" in method or "Union" in method:
            if hasattr(self, '_ml_feature_importance') and self._ml_feature_importance:
                # Combine importance from relevant models
                combined = {}
                all_ml_features = set()

                for model_key, importance in self._ml_feature_importance.items():
                    print(f"[AutoSelect] ML model {model_key}: {list(importance.keys())}")
                    for feat, imp in importance.items():
                        all_ml_features.add(feat)
                        # Try normalized name first
                        normalized_feat = normalize_feature_name(feat)
                        if normalized_feat in self._settings_feature_checkboxes:
                            combined[normalized_feat] = combined.get(normalized_feat, 0) + imp
                        elif feat in self._settings_feature_checkboxes:
                            combined[feat] = combined.get(feat, 0) + imp

                if combined:
                    sorted_features = sorted(combined.items(), key=lambda x: x[1], reverse=True)
                    features = [f for f, _ in sorted_features[:n]]
                    print(f"[AutoSelect] ML Union selected: {features}")
                    return features
                else:
                    # Show helpful debug info
                    available_keys = list(self._settings_feature_checkboxes.keys()) if hasattr(self, '_settings_feature_checkboxes') else []
                    print(f"[AutoSelect] No matching features from ML models")
                    print(f"[AutoSelect] ML features: {all_ml_features}")
                    print(f"[AutoSelect] Available checkbox keys: {available_keys}")
                    # Show warning to user
                    QMessageBox.warning(self, "Feature Mismatch",
                        f"ML model features don't match available feature names.\n\n"
                        f"ML features: {', '.join(sorted(all_ml_features)[:10])}\n\n"
                        f"Available: {', '.join(sorted(available_keys)[:10])}\n\n"
                        f"Falling back to default features.")
            else:
                print("[AutoSelect] ML importance not available - refresh from loaded models first")
                QMessageBox.warning(self, "ML Models Not Loaded",
                    "ML feature importance not available.\n\n"
                    "Click 'Refresh from Loaded Models' in the UMAP Settings tab first,\n"
                    "or load ML models from the Analysis Options dialog.")

        elif "Combined" in method:
            # Combine PCA and ML
            combined_scores = {}

            if hasattr(self, '_pca_feature_ranking') and self._pca_feature_ranking:
                for rank, (feat, score) in enumerate(self._pca_feature_ranking):
                    if feat in self._settings_feature_checkboxes:
                        combined_scores[feat] = combined_scores.get(feat, 0) + score

            if hasattr(self, '_ml_feature_importance') and self._ml_feature_importance:
                for importance in self._ml_feature_importance.values():
                    for feat, imp in importance.items():
                        normalized_feat = normalize_feature_name(feat)
                        if normalized_feat in self._settings_feature_checkboxes:
                            combined_scores[normalized_feat] = combined_scores.get(normalized_feat, 0) + imp
                        elif feat in self._settings_feature_checkboxes:
                            combined_scores[feat] = combined_scores.get(feat, 0) + imp

            if combined_scores:
                sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                features = [f for f, _ in sorted_features[:n]]
                print(f"[AutoSelect] Combined selected: {features}")
                return features

        # Fallback to defaults
        print("[AutoSelect] Using fallback defaults")
        return ['if', 'ti', 'amp_insp', 'max_dinsp']

    def _update_cache_status(self):
        """Update cache status labels."""
        if self._umap_coords_2d is not None:
            n = len(self._umap_coords_2d)
            self._cache_2d_label.setText(f"2D Embedding: ✓ Cached ({n} pts)")
            self._cache_2d_label.setStyleSheet("color: #4ec9b0;")
        else:
            self._cache_2d_label.setText("2D Embedding: Not computed")
            self._cache_2d_label.setStyleSheet("color: #888888;")

        if self._umap_coords_3d is not None:
            n = len(self._umap_coords_3d)
            self._cache_3d_label.setText(f"3D Embedding: ✓ Cached ({n} pts)")
            self._cache_3d_label.setStyleSheet("color: #4ec9b0;")
        else:
            self._cache_3d_label.setText("3D Embedding: Not computed")
            self._cache_3d_label.setStyleSheet("color: #888888;")

        if self._umap_feature_matrix is not None:
            n_samples, n_features = self._umap_feature_matrix.shape
            self._cache_features_label.setText(f"Feature matrix: ✓ ({n_features} features)")
            self._cache_features_label.setStyleSheet("color: #4ec9b0;")
        else:
            self._cache_features_label.setText("Feature matrix: Not computed")
            self._cache_features_label.setStyleSheet("color: #888888;")

    def _create_scatter_plot(self, title: str, x_label: str, y_label: str) -> pg.PlotWidget:
        """Create a scatter plot widget for metric comparisons."""
        plot = pg.PlotWidget()
        plot.setBackground('#1e1e1e')
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel('bottom', x_label)
        plot.setLabel('left', y_label)
        plot.setTitle(title, color='#cccccc', size='9pt')
        plot.setMinimumHeight(200)
        plot.setMinimumWidth(200)

        # Create scatter item
        scatter = pg.ScatterPlotItem(
            size=6,
            pen=pg.mkPen(None),
            hoverable=True,
            hoverSize=10,
            hoverPen=pg.mkPen('w', width=2)
        )
        plot.addItem(scatter)

        # Connect click handler
        scatter.sigClicked.connect(lambda p, s, ev: self._on_scatter_point_clicked(p, s, ev))

        return plot, scatter

    def _on_umap_double_click(self, ev):
        """Handle double-click on UMAP - disabled (now always tiled)."""
        # Double-click toggle disabled - UMAP always in tiled mode (1/3 width)
        pass

    def _recenter_umap_view(self):
        """Reset UMAP view to show all data points."""
        if self._umap_coords is None or len(self._umap_coords) == 0:
            return

        if self._umap_is_3d and self._umap_gl_widget is not None:
            # Reset 3D camera position
            self._umap_gl_widget.setCameraPosition(distance=20, elevation=30, azimuth=45)
        else:
            # Reset 2D view - autorange to fit all data
            self._umap_plot_widget.getViewBox().autoRange()

        print("[UMAP] View recentered")

    def _popout_umap_window(self):
        """Open UMAP visualization in a separate pop-out window.

        For 3D mode, this moves the existing GL widget to the pop-out dialog
        to avoid OpenGL context conflicts. The widget is moved back when closed.
        """
        if self._umap_coords is None or len(self._umap_coords) < 3:
            QMessageBox.warning(self, "No Data", "Compute UMAP first before popping out.")
            return

        # Check if we can use 3D (only if we have the GL widget already)
        use_3d = self._umap_is_3d and OPENGL_AVAILABLE and self._umap_gl_widget is not None

        # Store reference to main panel's container layout for restoration
        main_container_layout = self._umap_container.layout() if hasattr(self, '_umap_container') else None

        # Create a custom dialog class with ESC key handling
        class PopoutDialog(QDialog):
            def __init__(dialog_self, parent=None):
                super().__init__(parent)
                dialog_self.fullscreen_btn = None
                dialog_self.btn_container = None
                dialog_self.gl_widget_borrowed = None  # Track if we borrowed the GL widget
                dialog_self.original_layout = None

            def keyPressEvent(dialog_self, event):
                if event.key() == Qt.Key.Key_Escape:
                    if dialog_self.isFullScreen():
                        dialog_self.showNormal()
                        dialog_self._update_fullscreen_btn()
                    else:
                        dialog_self.close()
                else:
                    super().keyPressEvent(event)

            def toggle_fullscreen(dialog_self):
                if dialog_self.isFullScreen():
                    dialog_self.showNormal()
                else:
                    dialog_self.showFullScreen()
                dialog_self._update_fullscreen_btn()

            def _update_fullscreen_btn(dialog_self):
                if dialog_self.fullscreen_btn:
                    if dialog_self.isFullScreen():
                        dialog_self.fullscreen_btn.setText("⮌ Exit Fullscreen (ESC)")
                    else:
                        dialog_self.fullscreen_btn.setText("⛶ Fullscreen")

            def closeEvent(dialog_self, event):
                # Return the GL widget to the main panel when closing
                if dialog_self.gl_widget_borrowed and dialog_self.original_layout:
                    gl_widget = dialog_self.gl_widget_borrowed
                    # Remove from popout layout
                    dialog_self.layout().removeWidget(gl_widget)
                    # Add back to main panel at position 0
                    dialog_self.original_layout.insertWidget(0, gl_widget)
                    gl_widget.setVisible(True)
                    print("[UMAP] 3D widget returned to main panel")
                super().closeEvent(event)

        popout = PopoutDialog(self)
        popout.setWindowTitle("UMAP 3D Visualization" if use_3d else "UMAP Visualization")
        popout.setWindowFlags(
            popout.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )
        popout.resize(900, 800)

        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        if use_3d:
            # MOVE the existing GL widget to the pop-out (don't create a new one)
            # This avoids OpenGL context conflicts
            print("[UMAP] Moving 3D widget to pop-out window")

            # Remove from main panel
            if main_container_layout:
                main_container_layout.removeWidget(self._umap_gl_widget)

            # Add to pop-out
            layout.addWidget(self._umap_gl_widget, stretch=1)
            self._umap_gl_widget.setVisible(True)

            # Store references for restoration
            popout.gl_widget_borrowed = self._umap_gl_widget
            popout.original_layout = main_container_layout

            # Info label for 3D interaction
            help_label = QLabel("Drag to rotate • Scroll to zoom • Middle-click to pan • Click to select point")
            help_label.setStyleSheet("color: #888888; font-size: 9pt; padding: 2px;")
            help_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(help_label)

        else:
            # 2D mode - create a new plot widget (no context issues with 2D)
            if self._umap_is_3d:
                info_label = QLabel("Note: 3D widget not available. Showing 2D projection.")
                info_label.setStyleSheet("color: #f39c12; padding: 5px;")
                info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(info_label)

            plot_widget = pg.PlotWidget()
            plot_widget.setBackground('#1e1e1e')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setLabel('bottom', 'UMAP 1')
            plot_widget.setLabel('left', 'UMAP 2')

            # Copy scatter data
            colors = self._get_point_colors()
            scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
            spots = []
            for i, (x, y) in enumerate(self._umap_coords[:, :2]):
                r, g, b, a = colors[i]
                brush = pg.mkBrush(int(r), int(g), int(b), int(a))
                spots.append({'pos': (x, y), 'brush': brush, 'data': i})
            scatter.setData(spots)

            # Connect click handler
            scatter.sigClicked.connect(self._on_umap_point_clicked)
            plot_widget.addItem(scatter)

            layout.addWidget(plot_widget, stretch=1)

        # Button row (always visible)
        btn_container = QWidget()
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(5, 5, 5, 5)

        fullscreen_btn = QPushButton("⛶ Fullscreen")
        fullscreen_btn.clicked.connect(popout.toggle_fullscreen)
        popout.fullscreen_btn = fullscreen_btn
        btn_row.addWidget(fullscreen_btn)

        btn_row.addStretch()

        # ESC hint label
        hint_label = QLabel("Press ESC to exit fullscreen or close")
        hint_label.setStyleSheet("color: #888888; font-size: 9pt;")
        btn_row.addWidget(hint_label)

        btn_row.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(popout.close)
        btn_row.addWidget(close_btn)

        btn_container.setLayout(btn_row)
        popout.btn_container = btn_container
        layout.addWidget(btn_container)

        popout.setLayout(layout)
        popout.show()

    def _update_plots_layout(self):
        """Update the layout of UMAP and scatter plots in a 3-column tiled grid."""
        # Clear the current layout
        while self._plots_container_layout.count():
            item = self._plots_container_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Always use 3-column grid layout with UMAP in first position
        col = 0
        row = 0

        # Add UMAP first (1/3 width)
        self._plots_container_layout.addWidget(self._umap_container, row, col)
        col += 1

        # Add scatter plots in remaining cells (tiled 3 per row)
        for key, (plot_widget, scatter_item) in self._scatter_plots.items():
            if self._scatter_checkboxes.get(key, None) and self._scatter_checkboxes[key].isChecked():
                if col >= 3:
                    col = 0
                    row += 1
                self._plots_container_layout.addWidget(plot_widget, row, col)
                col += 1

    def _connect_signals(self):
        """Connect all signal handlers."""
        # Detect button
        self.detect_btn.clicked.connect(self._detect_edge_cases)

        # Navigation buttons
        self.first_btn.clicked.connect(self._navigate_first)
        self.prev_btn.clicked.connect(self._navigate_previous)
        self.next_btn.clicked.connect(self._navigate_next)
        self.last_btn.clicked.connect(self._navigate_last)

        # Classification buttons
        self.mark_breath_btn.clicked.connect(self._mark_as_breath)
        self.mark_noise_btn.clicked.connect(self._mark_as_noise)
        self.skip_btn.clicked.connect(self._skip_uncertain)

        # Jump to main
        self.jump_main_btn.clicked.connect(self._jump_to_main_window)

        # Export
        self.export_btn.clicked.connect(self._export_edge_cases)

        # Plot controls
        self.window_size_combo.currentTextChanged.connect(self._refresh_plot)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.show_events_checkbox.stateChanged.connect(self._refresh_plot)
        self.show_peaks_checkbox.stateChanged.connect(self._refresh_plot)

        # === Tab 2 signals ===
        self._compute_umap_btn.clicked.connect(self._compute_umap)

        # Dimension toggle
        self._umap_2d_radio.toggled.connect(self._on_umap_dimension_changed)
        self._umap_3d_radio.toggled.connect(self._on_umap_dimension_changed)

        # Color mode
        self._color_class_radio.toggled.connect(self._on_color_mode_changed)
        self._color_confidence_radio.toggled.connect(self._on_color_mode_changed)
        self._color_metric_radio.toggled.connect(self._on_color_mode_changed)
        self._color_metric_combo.currentTextChanged.connect(self._on_color_metric_changed)

        # Confidence navigation controls
        self._confidence_sort_combo.currentTextChanged.connect(self._on_confidence_sort_changed)
        self._prev_breath_btn.clicked.connect(self._go_to_prev_breath)
        self._next_breath_btn.clicked.connect(self._go_to_next_breath)

        # Classification buttons (new order: Noise, Eupnea, Sniff, Sigh, None)
        self._mark_noise_btn.clicked.connect(self._mark_as_noise_tab2)
        self._mark_eupnea_btn.clicked.connect(self._mark_as_eupnea)
        self._mark_sniffing_btn.clicked.connect(self._mark_as_sniffing)
        self._mark_sigh_btn.clicked.connect(self._mark_as_sigh)
        self._mark_none_btn.clicked.connect(self._mark_as_none)
        self._skip_btn_tab2.clicked.connect(self._skip_to_next_point)

        # Utility buttons
        self._jump_main_btn_tab2.clicked.connect(self._jump_to_main_window_tab2)
        self._export_btn_tab2.clicked.connect(self._export_classifications_tab2)

        # Trace controls and zoom
        self._window_size_combo_tab2.currentTextChanged.connect(self._refresh_trace_tab2)
        self._zoom_in_btn.clicked.connect(self._zoom_in_trace)
        self._zoom_out_btn.clicked.connect(self._zoom_out_trace)

        # Scatter comparison checkboxes
        for key, cb in self._scatter_checkboxes.items():
            cb.stateChanged.connect(self._on_scatter_checkbox_changed)

        # Display options
        self._show_events_cb_tab2.stateChanged.connect(self._refresh_trace_tab2)
        self._show_noise_cb.stateChanged.connect(self._on_visibility_changed)
        self._show_unclassified_cb.stateChanged.connect(self._on_visibility_changed)
        self._3d_opaque_cb.stateChanged.connect(self._on_3d_opaque_changed)

        # Decision boundary overlay controls
        self._overlay_type_combo.currentTextChanged.connect(self._on_overlay_type_changed)
        self._overlay_classifier_combo.currentTextChanged.connect(self._on_overlay_classifier_changed)
        self._refresh_models_btn.clicked.connect(self._refresh_available_classifiers)
        self._overlay_opacity_slider.valueChanged.connect(self._on_overlay_opacity_changed)

        # Color by time option
        self._color_time_radio.toggled.connect(self._on_color_time_changed)

        # Animation controls
        self._anim_prev_btn.clicked.connect(self._anim_step_prev)
        self._anim_play_btn.clicked.connect(self._anim_toggle_play)
        self._anim_next_btn.clicked.connect(self._anim_step_next)
        self._anim_speed_slider.valueChanged.connect(self._on_anim_speed_changed)
        self._anim_mode_combo.currentTextChanged.connect(self._on_anim_mode_changed)
        self._anim_trail_slider.valueChanged.connect(self._on_anim_trail_changed)

    def _detect_edge_cases(self):
        """Detect edge cases based on selected method."""
        print("\n===== Detecting Edge Cases =====")

        # Check if peaks have been detected
        if not hasattr(self.main_window.state, 'all_peaks_by_sweep') or not self.main_window.state.all_peaks_by_sweep:
            error_msg = "Please detect peaks first using 'Find Peaks & Events' button in the main window.\n\nThe Advanced Peak Editor needs detected peaks to find edge cases."
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("No Peaks Detected")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            print("[Edge Detection] No peaks detected yet")
            print("===== Detection Aborted =====\n")
            return

        # Get selected edge case type
        edge_type_id = self.edge_type_group.checkedId()

        if edge_type_id == 0:
            print("Method: Threshold P_edge")
            self._detect_p_edge_cases()
        elif edge_type_id == 1:
            print("Method: ML Disagreement")
            self._detect_ml_disagreements()
        elif edge_type_id == 2:
            print("Method: ML Probability")
            self._detect_ml_probability_edges()

        # Sort results
        self._sort_edge_cases()

        # Update UI
        self._update_ui_state()

        # Show first edge case
        if self.edge_cases:
            self.current_edge_idx = 0
            self._update_plot(0)
        else:
            # No edge cases found - show message in plot
            self._show_no_edge_cases_message()

        print(f"Total edge cases found: {len(self.edge_cases)}")
        print("===== Detection Complete =====\n")

    def _detect_p_edge_cases(self):
        """Detect edge cases using threshold P_edge metric."""
        self.edge_cases = []

        # Get sample rate
        sr_hz = self.main_window.state.sr_hz
        if sr_hz is None or sr_hz == 0:
            print("Error: Sample rate not available")
            return

        # Check if ANY sweep has P_edge data
        has_p_edge = any('P_edge' in peak_data for peak_data in self.main_window.state.all_peaks_by_sweep.values())

        if not has_p_edge:
            error_msg = ("P_edge data is not available in the current detection results.\n\n"
                        "P_edge is only computed by threshold-based detection.\n\n"
                        "Try using 'ML Disagreement' instead to find edge cases.")
            print(f"[P_edge Detection] {error_msg}")

            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Information)
            msg_box.setWindowTitle("P_edge Not Available")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        for sweep_idx, peak_data in self.main_window.state.all_peaks_by_sweep.items():
            if 'P_edge' not in peak_data:
                print(f"Skipping sweep {sweep_idx}: No P_edge data")
                continue

            p_edge = peak_data['P_edge']
            indices = peak_data['indices']  # Sample indices

            # Convert indices to times
            times = indices / sr_hz

            # Find peaks with high P_edge (close to decision boundary)
            # Lower threshold to 0.3 to catch more edge cases
            threshold = 0.3
            edge_mask = p_edge > threshold

            edge_indices = np.where(edge_mask)[0]
            print(f"Sweep {sweep_idx}: {len(edge_indices)} edge cases (P_edge > {threshold})")

            for peak_idx in edge_indices:
                self.edge_cases.append({
                    'sweep': int(sweep_idx),
                    'peak_idx': int(peak_idx),
                    'time': float(times[peak_idx]),
                    'sample_idx': int(indices[peak_idx]),
                    'p_edge': float(p_edge[peak_idx]),
                    'method': 'p_edge',
                    'current_label': int(peak_data['labels'][peak_idx]),
                })

    def _detect_ml_disagreements(self):
        """Detect edge cases where ML models disagree."""
        self.edge_cases = []

        # Get selected models
        model_a_name = self.ml_model_a_combo.currentText()
        model_b_name = self.ml_model_b_combo.currentText()

        if model_a_name == model_b_name:
            error_msg = "Please select two different models to compare."
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("Same Model Selected")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        # Map names to array keys
        model_map = {
            "Threshold": "labels_threshold_ro",
            "XGBoost": "labels_xgboost_ro",
            "Random Forest": "labels_rf_ro",
            "MLP": "labels_mlp_ro"
        }

        key_a = model_map[model_a_name]
        key_b = model_map[model_b_name]

        print(f"Comparing {model_a_name} vs {model_b_name}")

        # Get sample rate
        sr_hz = self.main_window.state.sr_hz
        if sr_hz is None or sr_hz == 0:
            print("Error: Sample rate not available")
            return

        for sweep_idx, peak_data in self.main_window.state.all_peaks_by_sweep.items():
            # Check if predictions exist
            if key_a not in peak_data or key_b not in peak_data:
                print(f"Skipping sweep {sweep_idx}: Missing predictions for {model_a_name} or {model_b_name}")
                continue

            labels_a = peak_data[key_a]
            labels_b = peak_data[key_b]
            indices = peak_data['indices']  # Sample indices

            # Convert indices to times
            times = indices / sr_hz

            # Find disagreements
            disagree_mask = (labels_a != labels_b)
            disagree_indices = np.where(disagree_mask)[0]

            print(f"Sweep {sweep_idx}: {len(disagree_indices)} disagreements")

            for peak_idx in disagree_indices:
                self.edge_cases.append({
                    'sweep': int(sweep_idx),
                    'peak_idx': int(peak_idx),
                    'time': float(times[peak_idx]),
                    'sample_idx': int(indices[peak_idx]),
                    'model_a_name': model_a_name,
                    'model_b_name': model_b_name,
                    'model_a_pred': int(labels_a[peak_idx]),
                    'model_b_pred': int(labels_b[peak_idx]),
                    'method': 'ml_disagree',
                    'current_label': int(peak_data['labels'][peak_idx]),
                })

    def _detect_ml_probability_edges(self):
        """Detect edge cases using ML model probability scores."""
        # TODO: Implement when probability scores are available
        error_msg = ("ML Probability edge detection requires probability scores.\n\n"
                    "This feature will be implemented when models output probabilities.\n\n"
                    "Try using 'ML Disagreement' instead to find edge cases.")

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setWindowTitle("Not Implemented")
        msg_box.setText(error_msg)
        msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_box.exec()

        self.edge_cases = []

    def _sort_edge_cases(self):
        """Sort edge cases based on selected order."""
        if not self.edge_cases:
            return

        sort_desc = self.sort_desc_radio.isChecked()

        # Sort by appropriate metric
        if self.edge_cases[0]['method'] == 'p_edge':
            self.edge_cases.sort(key=lambda x: x['p_edge'], reverse=sort_desc)
        elif self.edge_cases[0]['method'] == 'ml_disagree':
            # For disagreements, just keep original order or sort by time
            self.edge_cases.sort(key=lambda x: x['time'], reverse=sort_desc)

    def _update_ui_state(self):
        """Update UI button states based on edge cases."""
        has_cases = len(self.edge_cases) > 0

        # Enable/disable navigation
        self.first_btn.setEnabled(has_cases)
        self.prev_btn.setEnabled(has_cases)
        self.next_btn.setEnabled(has_cases)
        self.last_btn.setEnabled(has_cases)

        # Enable/disable classification
        self.mark_breath_btn.setEnabled(has_cases)
        self.mark_noise_btn.setEnabled(has_cases)
        self.skip_btn.setEnabled(has_cases)

        # Enable/disable utilities
        self.export_btn.setEnabled(has_cases)
        self.jump_main_btn.setEnabled(has_cases)

        # Update progress
        reviewed = len(self.reviewed_peaks)
        total = len(self.edge_cases)
        self.progress_label.setText(f"Progress: {reviewed}/{total}")

    def _update_plot(self, edge_idx):
        """
        Update the plot to show the selected edge case.

        Args:
            edge_idx: Index in self.edge_cases list
        """
        if edge_idx < 0 or edge_idx >= len(self.edge_cases):
            return

        edge = self.edge_cases[edge_idx]
        sweep_idx = edge['sweep']
        peak_idx = edge['peak_idx']
        peak_time = edge['time']

        # Get data
        peak_data = self.main_window.state.all_peaks_by_sweep[sweep_idx]

        # Get sweep data using the main window's method
        # Switch to the correct sweep first
        old_sweep = self.main_window.state.sweep_idx
        self.main_window.state.sweep_idx = sweep_idx

        # Get processed trace data
        times, trace = self.main_window._current_trace()

        # Restore original sweep
        self.main_window.state.sweep_idx = old_sweep

        if times is None or trace is None:
            # Can't plot without sweep data
            error_msg = f"Cannot display plot: Sweep {sweep_idx} data not available.\n\nPlease ensure data is loaded and the analyze channel is selected."
            print(f"Error: {error_msg}")

            # Create a copyable error dialog
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setWindowTitle("No Sweep Data")
            msg_box.setText(error_msg)
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            return

        # Get window bounds - center on peak
        window_text = self.window_size_combo.currentText()
        half_window = float(window_text.replace('±', '').replace('s', ''))

        t_min = peak_time - half_window
        t_max = peak_time + half_window

        # Find indices within window
        mask = (times >= t_min) & (times <= t_max)
        window_times = times[mask]
        window_trace = trace[mask]

        # Clear and redraw
        self.ax.clear()

        # Apply theme
        theme_name = 'dark' if self.theme_combo.currentText() == 'Dark' else 'light'
        self.theme_manager.apply_theme(self.ax, self.fig, theme_name)

        # Plot trace
        trace_color = self.theme_manager.get_color('trace_color')
        self.ax.plot(window_times, window_trace,
                     color=trace_color,
                     linewidth=self.theme_manager.get_value('trace_linewidth'),
                     label='Respiratory Signal', zorder=1)

        # Plot disagreement peak (highlighted)
        # Use transparency if marked as noise by active classifier
        disagree_color = self.theme_manager.get_color('disagreement_peak_color')
        disagree_size = self.theme_manager.get_value('disagreement_peak_size')

        # Find y-value at peak time
        peak_time_idx = np.argmin(np.abs(times - peak_time))
        peak_y_value = trace[peak_time_idx]

        # Check if current peak is marked as breath or noise by active classifier
        current_peak_label = peak_data['labels'][peak_idx]  # 0=noise, 1=breath
        is_breath = current_peak_label == 1

        # Use transparency for noise peaks
        peak_alpha = 1.0 if is_breath else 0.4
        peak_label = 'Edge Case Peak (Breath)' if is_breath else 'Edge Case Peak (Noise)'

        self.ax.plot(peak_time, peak_y_value,
                     marker=self.theme_manager.get_value('disagreement_peak_marker'),
                     color=disagree_color,
                     markersize=disagree_size,
                     alpha=peak_alpha,
                     label=peak_label,
                     zorder=5)

        # Plot other peaks in window (if enabled)
        # Only show peaks that the ACTIVE classifier marks as breaths
        if self.show_peaks_checkbox.isChecked():
            # Convert peak indices to times
            sr_hz = self.main_window.state.sr_hz
            all_peak_indices = peak_data['indices']
            all_peak_times = all_peak_indices / sr_hz

            # Get active classifier labels (only show peaks marked as breath by active classifier)
            active_labels = peak_data['labels']  # This is the currently displayed labels

            # Filter to window AND breaths only
            peak_mask = (all_peak_times >= t_min) & (all_peak_times <= t_max) & (active_labels == 1)
            other_peak_times = all_peak_times[peak_mask]

            # Remove current peak (use small tolerance for float comparison)
            tolerance = 0.001  # 1ms tolerance
            other_peak_times = other_peak_times[np.abs(other_peak_times - peak_time) > tolerance]

            if len(other_peak_times) > 0:
                # Find y-values for other peaks
                other_peak_y = []
                for pt in other_peak_times:
                    idx = np.argmin(np.abs(times - pt))
                    other_peak_y.append(trace[idx])

                peak_color = self.theme_manager.get_color('peak_color')
                self.ax.plot(other_peak_times, other_peak_y,
                             marker=self.theme_manager.get_value('peak_marker'),
                             color=peak_color,
                             linestyle='',
                             markersize=self.theme_manager.get_value('peak_size'),
                             label='Breaths (active classifier)',
                             zorder=3)

        # Plot event markers (onset/offset/expmin/expoff) if enabled
        # Only show events for peaks that active classifier marks as breaths
        if self.show_events_checkbox.isChecked():
            breath_data = self.main_window.state.all_breaths_by_sweep.get(sweep_idx)
            if breath_data:
                sr_hz = self.main_window.state.sr_hz

                # Get active classifier labels and peak times for filtering
                active_labels = peak_data['labels']  # Currently displayed labels
                all_peak_indices = peak_data['indices']
                all_peak_times = all_peak_indices / sr_hz

                # Helper function: Check if event belongs to a breath peak
                def is_breath_event(event_time):
                    """Check if event time is associated with a breath peak."""
                    # Find nearest peak to this event time
                    peak_dists = np.abs(all_peak_times - event_time)
                    nearest_peak_idx = np.argmin(peak_dists)

                    # Only show event if:
                    # 1. There's a peak nearby (within 0.5s)
                    # 2. That peak is marked as breath by active classifier
                    if peak_dists[nearest_peak_idx] < 0.5:
                        return active_labels[nearest_peak_idx] == 1
                    return False

                # Inspiratory onsets (green upward triangles)
                if 'onsets' in breath_data and len(breath_data['onsets']) > 0:
                    onset_samples = breath_data['onsets']
                    onset_times = onset_samples / sr_hz
                    onset_mask = (onset_times >= t_min) & (onset_times <= t_max)
                    onset_times_in_window = onset_times[onset_mask]

                    # Filter by breath classification
                    onset_y_vals = []
                    onset_times_filtered = []
                    for ot in onset_times_in_window:
                        if is_breath_event(ot):
                            idx = np.argmin(np.abs(times - ot))
                            onset_y_vals.append(trace[idx])
                            onset_times_filtered.append(ot)

                    if len(onset_times_filtered) > 0:
                        self.ax.scatter(onset_times_filtered, onset_y_vals,
                                       marker='^',
                                       color=self.theme_manager.get_color('onset_color'),
                                       s=50,
                                       zorder=4)

                # Inspiratory offsets (orange downward triangles)
                if 'offsets' in breath_data and len(breath_data['offsets']) > 0:
                    offset_samples = breath_data['offsets']
                    offset_times = offset_samples / sr_hz
                    offset_mask = (offset_times >= t_min) & (offset_times <= t_max)
                    offset_times_in_window = offset_times[offset_mask]

                    # Filter by breath classification
                    offset_y_vals = []
                    offset_times_filtered = []
                    for ot in offset_times_in_window:
                        if is_breath_event(ot):
                            idx = np.argmin(np.abs(times - ot))
                            offset_y_vals.append(trace[idx])
                            offset_times_filtered.append(ot)

                    if len(offset_times_filtered) > 0:
                        self.ax.scatter(offset_times_filtered, offset_y_vals,
                                       marker='v',
                                       color=self.theme_manager.get_color('offset_color'),
                                       s=50,
                                       zorder=4)

                # Expiratory minima (blue squares)
                if 'expmins' in breath_data and len(breath_data['expmins']) > 0:
                    expmin_samples = breath_data['expmins']
                    expmin_times = expmin_samples / sr_hz
                    expmin_mask = (expmin_times >= t_min) & (expmin_times <= t_max)
                    expmin_times_in_window = expmin_times[expmin_mask]

                    # Filter by breath classification
                    expmin_y_vals = []
                    expmin_times_filtered = []
                    for et in expmin_times_in_window:
                        if is_breath_event(et):
                            idx = np.argmin(np.abs(times - et))
                            expmin_y_vals.append(trace[idx])
                            expmin_times_filtered.append(et)

                    if len(expmin_times_filtered) > 0:
                        self.ax.scatter(expmin_times_filtered, expmin_y_vals,
                                       marker='s',
                                       color=self.theme_manager.get_color('expmin_color'),
                                       s=50,
                                       zorder=4)

                # Expiratory offsets (purple diamonds)
                if 'expoffs' in breath_data and len(breath_data['expoffs']) > 0:
                    expoff_samples = breath_data['expoffs']
                    expoff_times = expoff_samples / sr_hz
                    expoff_mask = (expoff_times >= t_min) & (expoff_times <= t_max)
                    expoff_times_in_window = expoff_times[expoff_mask]

                    # Filter by breath classification
                    expoff_y_vals = []
                    expoff_times_filtered = []
                    for et in expoff_times_in_window:
                        if is_breath_event(et):
                            idx = np.argmin(np.abs(times - et))
                            expoff_y_vals.append(trace[idx])
                            expoff_times_filtered.append(et)

                    if len(expoff_times_filtered) > 0:
                        self.ax.scatter(expoff_times_filtered, expoff_y_vals,
                                       marker='D',
                                       color=self.theme_manager.get_color('expoff_color'),
                                       s=50,
                                       zorder=4)

        # Highlight disagreement span
        span_color = self.theme_manager.get_color('disagreement_span_color')
        self.ax.axvspan(peak_time - 0.1, peak_time + 0.1,
                        alpha=0.3,
                        color=span_color,
                        zorder=2)

        # Labels and title
        self.ax.set_xlabel('Time (s)', color=self.theme_manager.get_color('label_color'))
        self.ax.set_ylabel('Amplitude', color=self.theme_manager.get_color('label_color'))

        # Build title
        if edge['method'] == 'p_edge':
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s | P_edge: {edge['p_edge']:.3f}"
        elif edge['method'] == 'ml_disagree':
            # Use dynamic model names
            pred_a_text = "Breath" if edge['model_a_pred'] == 1 else "Noise"
            pred_b_text = "Breath" if edge['model_b_pred'] == 1 else "Noise"
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s | {edge['model_a_name']}: {pred_a_text} vs {edge['model_b_name']}: {pred_b_text}"
        else:
            title = f"Peak #{peak_idx} @ {peak_time:.2f}s"

        self.ax.set_title(title, color=self.theme_manager.get_color('text_color'))

        # Legend
        self.ax.legend(loc='upper right')

        # Ensure window is exactly centered on peak
        # Set explicit x-axis limits AFTER all plotting is done
        self.ax.set_xlim(t_min, t_max)

        # Refresh canvas
        self.canvas.draw()

        # Update info label
        current_label_text = "BREATH" if edge['current_label'] == 1 else "NOISE"
        reviewed_text = "✓ Reviewed" if (edge['sweep'], edge['peak_idx']) in self.reviewed_peaks else "Pending"

        info_text = (f"Peak #{edge['peak_idx']} / {len(self.edge_cases)} edge cases | "
                    f"Sweep: {edge['sweep']} | Time: {edge['time']:.2f}s | "
                    f"Current: {current_label_text} | {reviewed_text}")

        if edge['method'] == 'p_edge':
            info_text += f" | P_edge: {edge['p_edge']:.3f}"

        self.info_label.setText(info_text)

    def _show_no_edge_cases_message(self):
        """Show a message in the plot when no edge cases are found."""
        self.ax.clear()

        # Apply theme
        theme_name = 'dark' if self.theme_combo.currentText() == 'Dark' else 'light'
        self.theme_manager.apply_theme(self.ax, self.fig, theme_name)

        # Add centered text message
        self.ax.text(0.5, 0.5, 'No edge cases found.\n\nTry:\n• Lowering the P_edge threshold\n• Using a different detection method\n• Loading ML models for disagreement detection',
                    ha='center', va='center',
                    transform=self.ax.transAxes,
                    fontsize=14,
                    color=self.theme_manager.get_color('text_color'))

        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas.draw()

        self.info_label.setText("No edge cases found with current settings.")

    def _refresh_plot(self):
        """Refresh the current plot with new settings."""
        if self.current_edge_idx >= 0:
            self._update_plot(self.current_edge_idx)

    def _on_theme_changed(self, theme_text):
        """Handle theme dropdown change."""
        self._refresh_plot()

    def _navigate_first(self):
        """Navigate to first edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = 0
        self._update_plot(self.current_edge_idx)

    def _navigate_previous(self):
        """Navigate to previous edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = max(0, self.current_edge_idx - 1)
        self._update_plot(self.current_edge_idx)

    def _navigate_next(self):
        """Navigate to next edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = min(len(self.edge_cases) - 1, self.current_edge_idx + 1)
        self._update_plot(self.current_edge_idx)

    def _navigate_last(self):
        """Navigate to last edge case."""
        if not self.edge_cases:
            return
        self.current_edge_idx = len(self.edge_cases) - 1
        self._update_plot(self.current_edge_idx)

    def _mark_as_breath(self):
        """Mark current peak as BREATH."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self._apply_classification(edge, label=1, label_name="BREATH")

    def _mark_as_noise(self):
        """Mark current peak as NOISE."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self._apply_classification(edge, label=0, label_name="NOISE")

    def _apply_classification(self, edge: dict, label: int, label_name: str):
        """
        Apply classification to a peak.

        Args:
            edge: Edge case dictionary
            label: 0=noise, 1=breath
            label_name: Human-readable label
        """
        sweep = edge['sweep']
        peak_idx = edge['peak_idx']

        print(f"\n===== Applying Classification =====")
        print(f"Sweep: {sweep}, Peak: {peak_idx}")
        print(f"Marking as: {label_name} (label={label})")

        peak_data = self.main_window.state.all_peaks_by_sweep[sweep]
        peak_data['labels'][peak_idx] = label
        peak_data['label_source'][peak_idx] = 'user'

        # Mark as reviewed
        self.reviewed_peaks.add((sweep, peak_idx))

        # Update edge case current_label
        edge['current_label'] = label

        # Update progress
        self._update_ui_state()

        # Refresh plot if on current sweep
        if self.main_window.current_sweep == sweep:
            self.main_window.plot_sweep()

        print("===== Classification Applied =====\n")

        # Auto-advance to next
        self._navigate_next()

    def _skip_uncertain(self):
        """Skip current peak (mark as reviewed but don't change label)."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]
        self.reviewed_peaks.add((edge['sweep'], edge['peak_idx']))

        # Update progress
        self._update_ui_state()

        # Auto-advance
        self._navigate_next()

    def _jump_to_main_window(self):
        """Jump to current peak in main window."""
        if self.current_edge_idx < 0:
            return

        edge = self.edge_cases[self.current_edge_idx]

        print(f"\n===== Jumping to Main Window =====")
        print(f"Sweep: {edge['sweep']}, Time: {edge['time']:.2f}s")

        # Switch to sweep
        self.main_window.CurrentSweep.setValue(edge['sweep'])

        # Center view on peak
        window_half_width = 2.0
        xlim_left = max(0, edge['time'] - window_half_width)
        xlim_right = edge['time'] + window_half_width

        self.main_window.ax.set_xlim(xlim_left, xlim_right)
        self.main_window.canvas.draw()

        print("===== Jump Complete =====\n")

    def _export_edge_cases(self):
        """Export edge cases to CSV."""
        from PyQt6.QtWidgets import QFileDialog
        import csv

        if not self.edge_cases:
            QMessageBox.warning(self, "No Data", "No edge cases to export.")
            return

        # Get save filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Edge Cases",
            "edge_cases.csv",
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header
                writer.writerow([
                    'Sweep', 'PeakIndex', 'Time_s', 'Method',
                    'P_edge', 'Current_Label', 'Reviewed'
                ])

                # Data rows
                for edge in self.edge_cases:
                    reviewed = "Yes" if (edge['sweep'], edge['peak_idx']) in self.reviewed_peaks else "No"
                    p_edge_val = edge.get('p_edge', '')

                    writer.writerow([
                        edge['sweep'],
                        edge['peak_idx'],
                        f"{edge['time']:.3f}",
                        edge['method'],
                        f"{p_edge_val:.3f}" if p_edge_val else '',
                        edge['current_label'],
                        reviewed
                    ])

            QMessageBox.information(self, "Export Complete",
                                  f"Edge cases exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                               f"Failed to export edge cases:\n{str(e)}")

    # ========== Tab 2: UMAP Implementation Methods ==========

    def _compute_umap(self):
        """Compute UMAP embedding from breath features."""
        from sklearn.preprocessing import StandardScaler
        from core import metrics, filters

        print("\n===== Computing UMAP Embedding =====")

        # Check if breath data exists
        st = self.main_window.state
        if not hasattr(st, 'breath_by_sweep') or not st.breath_by_sweep:
            QMessageBox.warning(self, "No Data",
                              "No breath data available.\nPlease run peak detection first.")
            return

        # Check if auto-select is enabled in UMAP Settings tab
        if hasattr(self, '_auto_select_features_cb') and self._auto_select_features_cb.isChecked():
            # Use features from UMAP Settings tab auto-selection
            selected_features = self._get_auto_selected_features()
            print(f"Using auto-selected features from Settings tab: {selected_features}")

            # Sync the Tab 2 checkboxes to match
            for key, cb in self._feature_checkboxes.items():
                cb.setChecked(key in selected_features)
        else:
            # Get selected features from Tab 2 checkboxes
            selected_features = [key for key, cb in self._feature_checkboxes.items() if cb.isChecked()]

        if len(selected_features) < 2:
            QMessageBox.warning(self, "Insufficient Features",
                              "Please select at least 2 features for UMAP.\n\n"
                              "Tip: Run PCA Analysis or load ML models in the UMAP Settings tab,\n"
                              "then enable 'Auto-select features' to use the best features automatically.")
            return

        print(f"Selected features: {selected_features}")

        # Show progress dialog
        progress = QProgressDialog("Collecting breath features...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(10)
        QApplication.processEvents()

        # Collect features using data scope settings (if available)
        include_breaths = True
        include_noise = True
        include_unclassified = True

        # Check for data scope checkboxes from UMAP Settings tab
        if hasattr(self, '_include_breaths_cb'):
            include_breaths = self._include_breaths_cb.isChecked()
        if hasattr(self, '_include_noise_cb'):
            include_noise = self._include_noise_cb.isChecked()
        if hasattr(self, '_include_unclassified_cb'):
            include_unclassified = self._include_unclassified_cb.isChecked()

        try:
            feature_matrix, breath_metadata = self._collect_all_breath_features_tab2(
                selected_features,
                include_breaths=include_breaths,
                include_noise=include_noise,
                include_unclassified=include_unclassified
            )
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"Failed to collect features:\n{str(e)}")
            return

        if len(feature_matrix) < 5:
            progress.close()
            QMessageBox.warning(self, "Insufficient Data",
                              f"Only {len(feature_matrix)} valid breaths found.\nNeed at least 5 for UMAP.")
            return

        print(f"Collected {len(feature_matrix)} breaths with {len(selected_features)} features each")
        progress.setLabelText(f"Computing embedding for {len(feature_matrix)} breaths...")
        progress.setValue(40)
        QApplication.processEvents()

        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix)

        # Get UMAP parameters
        n_neighbors = self._umap_n_neighbors.value()
        min_dist = self._umap_min_dist.value()
        n_components = 3 if self._umap_3d_radio.isChecked() else 2

        progress.setLabelText("Computing UMAP embedding...")
        progress.setValue(60)
        QApplication.processEvents()

        # Compute embedding
        try:
            embedding, method = compute_embedding(
                scaled_features,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                verbose=True
            )
        except Exception as e:
            progress.close()
            QMessageBox.critical(self, "Error", f"UMAP computation failed:\n{str(e)}")
            return

        # Store results
        self._umap_coords = embedding
        self._umap_breath_metadata = breath_metadata
        self._umap_feature_matrix = scaled_features
        self._umap_method = method
        self._umap_is_3d = (n_components == 3)

        # Cache the embedding for quick 2D/3D switching
        if n_components == 2:
            self._umap_coords_2d = embedding
        else:
            self._umap_coords_3d = embedding

        # Store selected features for scatter plots
        self._umap_selected_features = selected_features

        print(f"Embedding computed using {method}: shape {embedding.shape}")

        progress.setLabelText("Updating visualization...")
        progress.setValue(80)
        QApplication.processEvents()

        # Update UMAP scatter plot
        self._update_umap_scatter()

        # Create/update metric scatter plots
        self._update_scatter_plots()

        # Refresh available classifiers for overlays
        self._refresh_available_classifiers()

        # Update decision boundary overlay if enabled
        self._update_decision_boundary_overlay()

        # Update UI state
        self._update_ui_state_tab2()

        # Update layout
        self._update_plots_layout()

        # Update status
        self._umap_status_label.setText(f"Method: {method.upper()} | {len(breath_metadata)} breaths")
        self._info_label_tab2.setText(f"Click on a point to view the breath waveform.")

        # Initialize confidence-based sorting
        current_sort = self._confidence_sort_combo.currentText()
        self._sort_breaths_by_confidence(current_sort)
        self._update_nav_buttons_state()

        # Auto-select the first breath in sorted order
        if self._sorted_breath_indices:
            self._sorted_nav_position = 0
            breath_idx = self._sorted_breath_indices[0]
            self._select_umap_point(breath_idx)
            self._update_nav_position_label()

        progress.setValue(100)
        progress.close()

        print("===== UMAP Computation Complete =====\n")

    def _collect_all_breath_features_tab2(
        self,
        feature_keys: List[str],
        include_breaths: bool = True,
        include_noise: bool = True,
        include_unclassified: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Collect features from ALL detected peaks for UMAP (universal mode).

        Args:
            feature_keys: List of metric keys to extract
            include_breaths: Include peaks labeled as real breaths (label=1)
            include_noise: Include peaks labeled as noise (label=0)
            include_unclassified: Include peaks without labels

        Returns:
            Tuple of (feature_matrix, breath_metadata)
        """
        from core import metrics, filters

        feature_matrix = []
        breath_metadata = []

        st = self.main_window.state

        # Determine if we have all_peaks_by_sweep (ML-labeled) or just regular peaks
        use_all_peaks = hasattr(st, 'all_peaks_by_sweep') and st.all_peaks_by_sweep

        # Debug: print available data
        print(f"[Features] use_all_peaks={use_all_peaks}")
        print(f"[Features] Requesting features: {feature_keys}")

        # Get sweep keys from available data
        if use_all_peaks:
            sweep_keys = sorted(st.all_peaks_by_sweep.keys())
            print(f"[Features] Using all_peaks_by_sweep: {len(sweep_keys)} sweeps")
        elif hasattr(st, 'peaks_by_sweep') and st.peaks_by_sweep:
            sweep_keys = sorted(st.peaks_by_sweep.keys())
            print(f"[Features] Using peaks_by_sweep: {len(sweep_keys)} sweeps")
        elif hasattr(st, 'breath_by_sweep') and st.breath_by_sweep:
            sweep_keys = sorted(st.breath_by_sweep.keys())
            print(f"[Features] Using breath_by_sweep: {len(sweep_keys)} sweeps")
        else:
            print("[Features] No data available!")
            return np.array([]), []  # No data available

        for sweep_idx in sweep_keys:
            # Get peak indices - try all_peaks first, then fall back to peaks_by_sweep
            if use_all_peaks:
                all_peaks = st.all_peaks_by_sweep.get(sweep_idx, {})
                if not all_peaks or 'indices' not in all_peaks:
                    continue
                all_peak_indices = all_peaks.get('indices', np.array([]))
                labels = all_peaks.get('labels', None)
            else:
                # Fall back to regular peaks (from peaks_by_sweep)
                all_peaks = {}
                all_peak_indices = st.peaks_by_sweep.get(sweep_idx, np.array([]))
                labels = None  # No noise/breath labels without ML

            if len(all_peak_indices) == 0:
                continue

            # Get breath data for metrics computation
            breath_data = st.breath_by_sweep.get(sweep_idx, {})
            peaks = st.peaks_by_sweep.get(sweep_idx, np.array([]))

            # Get time and signal
            t = st.t
            y_raw = st.sweeps[st.analyze_chan][:, sweep_idx]

            # Apply filters
            y = filters.apply_all_1d(
                y_raw, st.sr_hz,
                st.use_low, st.low_hz,
                st.use_high, st.high_hz,
                st.use_mean_sub, st.mean_val,
                st.use_invert,
                order=getattr(self.main_window, 'filter_order', 4)
            )

            # Get breath events (used for metric computation)
            onsets = breath_data.get('onsets', np.array([]))
            offsets = breath_data.get('offsets', np.array([]))
            expmins = breath_data.get('expmins', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            # Debug: print first sweep info
            if sweep_idx == sweep_keys[0]:
                print(f"[Features] Sweep {sweep_idx}: {len(all_peak_indices)} peaks, "
                      f"{len(peaks)} breath peaks, {len(onsets)} onsets, {len(offsets)} offsets")

            # Compute metrics for this sweep using breath peaks
            metrics_dict = {}
            if len(onsets) > 0 and len(peaks) > 0:
                for feature_key in feature_keys:
                    if feature_key in metrics.METRICS:
                        try:
                            metric_arr = metrics.METRICS[feature_key](
                                t, y, st.sr_hz, peaks, onsets, offsets, expmins, expoffs
                            )
                            metrics_dict[feature_key] = metric_arr
                        except Exception as e:
                            if sweep_idx == sweep_keys[0]:
                                print(f"[Features] Failed to compute {feature_key}: {e}")
            elif sweep_idx == sweep_keys[0]:
                print(f"[Features] Cannot compute metrics: onsets={len(onsets)}, peaks={len(peaks)}")

            # Get sigh info for this sweep
            sigh_samples = st.sigh_by_sweep.get(sweep_idx, np.array([])) if hasattr(st, 'sigh_by_sweep') else np.array([])

            # Calculate sweep duration for absolute time
            sweep_duration = len(t) / st.sr_hz if len(t) > 0 else 0

            # Extract features for ALL peaks (not just breaths)
            for peak_idx, peak_sample in enumerate(all_peak_indices):
                peak_sample = int(peak_sample)
                peak_time = peak_sample / st.sr_hz
                # Absolute time includes sweep offset (for proper time ordering)
                absolute_time = sweep_idx * sweep_duration + peak_time

                # Check label for filtering
                if labels is not None and peak_idx < len(labels):
                    label = labels[peak_idx]
                    if label == 1 and not include_breaths:
                        continue
                    if label == 0 and not include_noise:
                        continue
                elif not include_unclassified:
                    continue

                # Determine is_noise for metadata
                is_noise = False
                if labels is not None and peak_idx < len(labels):
                    is_noise = labels[peak_idx] == 0

                # Get breath_type_class (eupnea/sniff classification)
                current_class = -1  # unclassified
                if is_noise:
                    current_class = 2  # Mark as noise
                elif 'breath_type_class' in all_peaks and all_peaks['breath_type_class'] is not None:
                    if peak_idx < len(all_peaks['breath_type_class']):
                        current_class = int(all_peaks['breath_type_class'][peak_idx])

                # Find nearest onset sample for metric lookup
                if len(onsets) > 0:
                    # Find closest onset to this peak
                    onset_diffs = np.abs(onsets - peak_sample)
                    nearest_onset_idx = np.argmin(onset_diffs)
                    onset_sample = int(onsets[nearest_onset_idx])
                else:
                    onset_sample = peak_sample  # Use peak itself if no onsets

                # Extract features
                peak_features = []
                valid_peak = True

                for feature_key in feature_keys:
                    if feature_key not in metrics_dict:
                        valid_peak = False
                        break

                    metric_arr = metrics_dict[feature_key]
                    if onset_sample < len(metric_arr):
                        val = metric_arr[onset_sample]
                        if np.isnan(val) or not np.isfinite(val):
                            valid_peak = False
                            break
                        peak_features.append(val)
                    else:
                        valid_peak = False
                        break

                if valid_peak and len(peak_features) == len(feature_keys):
                    # Check if sigh
                    is_sigh = peak_sample in sigh_samples if len(sigh_samples) > 0 else False

                    feature_matrix.append(peak_features)
                    breath_metadata.append({
                        'sweep': sweep_idx,
                        'breath_idx': peak_idx,  # Now represents peak_idx in all_peaks
                        'peak_sample': peak_sample,
                        'peak_time': peak_time,
                        'absolute_time': absolute_time,  # For proper time ordering across sweeps
                        'current_class': current_class,
                        'is_sigh': is_sigh,
                        'is_noise': is_noise,
                        'umap_idx': len(breath_metadata),
                    })

        return np.array(feature_matrix), breath_metadata

    def _update_umap_scatter(self):
        """Update the UMAP scatter plot with current data."""
        if self._umap_coords is None or len(self._umap_coords) == 0:
            return

        # Get colors for points
        colors = self._get_point_colors()

        if self._umap_is_3d and OPENGL_AVAILABLE:
            self._update_umap_scatter_3d(colors)
        else:
            self._update_umap_scatter_2d(colors)

    def _update_umap_scatter_2d(self, colors: np.ndarray):
        """Update 2D scatter plot."""
        # Ensure 2D widget is visible
        self._umap_plot_widget.setVisible(True)
        if self._umap_gl_widget is not None:
            self._umap_gl_widget.setVisible(False)

        # Convert colors to brushes
        spots = []
        for i, (x, y) in enumerate(self._umap_coords[:, :2]):
            r, g, b, a = colors[i]
            brush = pg.mkBrush(int(r), int(g), int(b), int(a))
            spots.append({
                'pos': (x, y),
                'brush': brush,
                'data': i,  # Store index for click lookup
            })

        self._umap_scatter.setData(spots)

        # Add selection marker for selected point
        if self._umap_selected_idx is not None and self._umap_selected_idx < len(self._umap_coords):
            sel_x, sel_y = self._umap_coords[self._umap_selected_idx, :2]
            # Highlight is handled by hover, but we can add extra emphasis if needed

    def _update_umap_scatter_3d(self, colors: np.ndarray):
        """Update 3D scatter plot using OpenGL."""
        if not OPENGL_AVAILABLE:
            return

        # Create GL widget if needed
        if self._umap_gl_widget is None:
            self._umap_gl_widget = ClickableGLViewWidget()
            self._umap_gl_widget.setBackgroundColor('#1e1e1e')
            self._umap_gl_widget.setMinimumHeight(300)
            self._umap_gl_widget.setMinimumWidth(300)

            # Set up click callback for point selection
            self._umap_gl_widget.set_click_callback(self._on_3d_point_clicked)

            # Set camera position for better initial view
            self._umap_gl_widget.setCameraPosition(distance=20, elevation=30, azimuth=45)

            # Add grid
            grid = gl.GLGridItem()
            grid.setSize(x=10, y=10, z=10)
            grid.setSpacing(x=1, y=1, z=1)
            self._umap_gl_widget.addItem(grid)

            # Add to layout
            layout = self._umap_container.layout()
            layout.insertWidget(0, self._umap_gl_widget)

        # Show 3D, hide 2D
        self._umap_gl_widget.setVisible(True)
        self._umap_plot_widget.setVisible(False)

        # Ensure GL widget has proper size (force resize)
        self._umap_gl_widget.updateGeometry()

        # Remove old scatter
        if self._umap_gl_scatter is not None:
            self._umap_gl_widget.removeItem(self._umap_gl_scatter)

        # Create new scatter
        # Normalize coordinates for display
        coords_normalized = self._umap_coords.copy()
        for i in range(3):
            col = coords_normalized[:, i]
            col_range = col.max() - col.min()
            if col_range > 0:
                coords_normalized[:, i] = (col - col.min()) / col_range * 10 - 5

        # Check if opaque mode is enabled (fixes white dots issue)
        use_opaque = self._3d_opaque_cb.isChecked()

        if use_opaque:
            # Opaque mode: set alpha to 1.0 for all points, disable blending
            colors_normalized = colors.copy() / 255.0
            colors_normalized[:, 3] = 1.0  # Full opacity
            gl_options = 'opaque'
        else:
            # Transparent mode: use original alpha values (may cause white dots)
            colors_fixed = colors.copy()
            colors_fixed[:, 3] = np.maximum(colors_fixed[:, 3], 200)
            colors_normalized = colors_fixed / 255.0
            gl_options = 'translucent'

        # Store normalized coords for click detection
        self._umap_gl_coords_normalized = coords_normalized

        # Pass coordinates to clickable widget for hit detection
        self._umap_gl_widget.set_coords(coords_normalized)

        self._umap_gl_scatter = gl.GLScatterPlotItem(
            pos=coords_normalized,
            color=colors_normalized,
            size=8,  # Larger for better visibility
            pxMode=True,
            glOptions=gl_options
        )
        self._umap_gl_widget.addItem(self._umap_gl_scatter)

        # Update 3D selection highlight if a point is selected
        self._update_3d_selection_highlight()

    def _update_scatter_plots(self):
        """Create/update the metric comparison scatter plots."""
        if self._umap_feature_matrix is None or len(self._umap_feature_matrix) == 0:
            return

        if not hasattr(self, '_umap_selected_features') or self._umap_selected_features is None:
            return

        # Get colors for points
        colors = self._get_point_colors()

        # Define scatter plot pairs based on available features
        scatter_configs = {
            'if_ti': ('if', 'ti', 'IF', 'Ti'),
            'if_amp': ('if', 'amp_insp', 'IF', 'Amp Insp'),
            'ti_te': ('ti', 'te', 'Ti', 'Te'),
            'amp_dinsp': ('amp_insp', 'max_dinsp', 'Amp Insp', 'Max dInsp'),
        }

        features = self._umap_selected_features

        for key, (x_feat, y_feat, x_label, y_label) in scatter_configs.items():
            # Check if both features are available
            if x_feat not in features or y_feat not in features:
                # Remove plot if it exists but features aren't available
                if key in self._scatter_plots:
                    plot_widget, _ = self._scatter_plots[key]
                    plot_widget.setParent(None)
                    del self._scatter_plots[key]
                continue

            x_idx = features.index(x_feat)
            y_idx = features.index(y_feat)

            x_data = self._umap_feature_matrix[:, x_idx]
            y_data = self._umap_feature_matrix[:, y_idx]

            # Create plot if it doesn't exist
            if key not in self._scatter_plots:
                plot_widget, scatter_item = self._create_scatter_plot(
                    f"{x_label} vs {y_label}", x_label, y_label
                )
                self._scatter_plots[key] = (plot_widget, scatter_item)
            else:
                plot_widget, scatter_item = self._scatter_plots[key]

            # Update scatter data
            spots = []
            for i in range(len(x_data)):
                r, g, b, a = colors[i]
                brush = pg.mkBrush(int(r), int(g), int(b), int(a))
                spots.append({
                    'pos': (x_data[i], y_data[i]),
                    'brush': brush,
                    'data': i,  # Store index for click lookup
                })

            scatter_item.setData(spots)

    def _get_point_colors(self) -> np.ndarray:
        """Get colors for each point based on current color mode."""
        n_points = len(self._umap_breath_metadata)
        colors = np.zeros((n_points, 4), dtype=np.float32)

        if self._color_class_radio.isChecked():
            # Color by class
            for i, meta in enumerate(self._umap_breath_metadata):
                if meta['is_sigh']:
                    colors[i] = self.UMAP_COLORS['sigh']
                elif meta.get('is_noise', False) or meta['current_class'] == 2:
                    colors[i] = self.UMAP_COLORS['noise']
                elif meta['current_class'] == 0:
                    colors[i] = self.UMAP_COLORS['eupnea']
                elif meta['current_class'] == 1:
                    colors[i] = self.UMAP_COLORS['sniffing']
                else:
                    colors[i] = self.UMAP_COLORS['unclassified']

        elif self._color_confidence_radio.isChecked():
            # Color by class with alpha based on confidence
            # Get confidence values from GMM or ML probabilities
            for i, meta in enumerate(self._umap_breath_metadata):
                sweep = meta['sweep']
                peak_sample = meta['peak_sample']

                # Get base color from class
                if meta['is_sigh']:
                    base_color = list(self.UMAP_COLORS['sigh'])
                elif meta.get('is_noise', False) or meta['current_class'] == 2:
                    base_color = list(self.UMAP_COLORS['noise'])
                elif meta['current_class'] == 0:
                    base_color = list(self.UMAP_COLORS['eupnea'])
                elif meta['current_class'] == 1:
                    base_color = list(self.UMAP_COLORS['sniffing'])
                else:
                    base_color = list(self.UMAP_COLORS['unclassified'])

                # Try to get confidence from GMM or ML probabilities
                confidence = 1.0  # Default to full confidence
                st = self.main_window.state

                if hasattr(st, 'all_peaks_by_sweep'):
                    all_peaks = st.all_peaks_by_sweep.get(sweep)
                    if all_peaks is not None:
                        # Find peak index in all_peaks
                        peak_indices = all_peaks.get('indices', np.array([]))
                        peak_mask = peak_indices == peak_sample
                        if np.any(peak_mask):
                            peak_pos = np.where(peak_mask)[0][0]

                            # Check for GMM probabilities
                            if 'gmm_eupnea_prob' in all_peaks:
                                eupnea_prob = all_peaks['gmm_eupnea_prob']
                                if peak_pos < len(eupnea_prob):
                                    prob = eupnea_prob[peak_pos]
                                    # Confidence is how far from 0.5 (decision boundary)
                                    confidence = abs(prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

                            # Check for ML probabilities (if available)
                            elif 'ml_breath_prob' in all_peaks:
                                prob = all_peaks['ml_breath_prob']
                                if peak_pos < len(prob):
                                    confidence = abs(prob[peak_pos] - 0.5) * 2

                # Apply confidence to alpha (min 80 for visibility, max 255)
                alpha = int(80 + 175 * confidence)
                base_color[3] = alpha
                colors[i] = base_color

        elif self._color_metric_radio.isChecked():
            # Color by metric using viridis colormap
            from matplotlib import cm

            metric_key = self._color_metric_combo.currentText()

            # Extract metric values from stored feature matrix
            feature_keys = [key for key, cb in self._feature_checkboxes.items() if cb.isChecked()]
            if metric_key in feature_keys:
                metric_idx = feature_keys.index(metric_key)
                metric_values = self._umap_feature_matrix[:, metric_idx]

                # Normalize
                vmin, vmax = np.nanmin(metric_values), np.nanmax(metric_values)
                if vmax - vmin > 1e-10:
                    normalized = (metric_values - vmin) / (vmax - vmin)
                else:
                    normalized = np.zeros_like(metric_values)

                # Apply colormap
                cmap = cm.get_cmap('viridis')
                for i, norm_val in enumerate(normalized):
                    rgba = cmap(norm_val)
                    colors[i] = [rgba[0] * 255, rgba[1] * 255, rgba[2] * 255, 255]
            else:
                # Fallback to gray
                colors[:] = self.UMAP_COLORS['unclassified']

        elif self._color_time_radio.isChecked():
            # Color by time using plasma colormap (blue -> purple -> yellow)
            from matplotlib import cm

            # Extract absolute times from metadata (includes sweep offset)
            times = np.array([meta.get('absolute_time', meta['peak_time']) for meta in self._umap_breath_metadata])

            # Normalize times to 0-1
            t_min, t_max = np.nanmin(times), np.nanmax(times)
            if t_max - t_min > 1e-10:
                normalized = (times - t_min) / (t_max - t_min)
            else:
                normalized = np.zeros_like(times)

            # Apply plasma colormap (good for temporal data)
            cmap = cm.get_cmap('plasma')
            for i, norm_val in enumerate(normalized):
                rgba = cmap(norm_val)
                colors[i] = [rgba[0] * 255, rgba[1] * 255, rgba[2] * 255, 255]

            # Store sorted indices for animation (using absolute time for proper ordering)
            self._anim_time_sorted_indices = np.argsort(times)

        else:
            # Fallback to gray
            colors[:] = self.UMAP_COLORS['unclassified']

        # Apply visibility filtering (hide points by setting alpha to 0)
        show_noise = self._show_noise_cb.isChecked()
        show_unclassified = self._show_unclassified_cb.isChecked()

        if not show_noise or not show_unclassified:
            for i, meta in enumerate(self._umap_breath_metadata):
                is_noise = meta.get('is_noise', False) or meta['current_class'] == 2
                is_unclassified = meta['current_class'] is None or (
                    meta['current_class'] not in [0, 1, 2] and not meta['is_sigh']
                )

                if not show_noise and is_noise:
                    colors[i, 3] = 0  # Hide by setting alpha to 0
                elif not show_unclassified and is_unclassified:
                    colors[i, 3] = 0  # Hide by setting alpha to 0

        # Apply animation visibility (for cumulative/trail modes)
        if self._anim_visible_indices is not None:
            n_points = len(self._umap_breath_metadata)

            if self._anim_mode == 'Cumulative':
                # Hide points not yet revealed
                for i in range(n_points):
                    if i not in self._anim_visible_indices:
                        colors[i, 3] = 0

            elif self._anim_mode == 'Trail':
                # Apply fading for trail - most recent is brightest
                current_step = self._anim_current_step
                for i in range(n_points):
                    if i not in self._anim_visible_indices:
                        colors[i, 3] = 0
                    else:
                        # Find how far back this point is in the trail
                        if self._anim_time_sorted_indices is not None:
                            pos = np.where(self._anim_time_sorted_indices == i)[0]
                            if len(pos) > 0:
                                step = pos[0]
                                distance_from_current = current_step - step
                                if distance_from_current >= 0:
                                    # Fade from 255 (current) to 80 (oldest in trail)
                                    fade_factor = 1.0 - (distance_from_current / self._anim_trail_length)
                                    alpha = int(80 + 175 * fade_factor)
                                    colors[i, 3] = min(colors[i, 3], alpha)

        return colors

    def _on_umap_point_clicked(self, plot_item, spots, ev):
        """Handle click on UMAP scatter point."""
        if len(spots) == 0:
            return

        # Get clicked point data (index)
        point_data = spots[0].data()
        if point_data is not None:
            self._select_umap_point(int(point_data))

    def _select_umap_point(self, point_idx: int):
        """Select a UMAP point and update trace panel."""
        if point_idx < 0 or point_idx >= len(self._umap_breath_metadata):
            return

        self._umap_selected_idx = point_idx
        meta = self._umap_breath_metadata[point_idx]

        print(f"Selected breath: sweep={meta['sweep']}, idx={meta['breath_idx']}, time={meta['peak_time']:.2f}s")

        # Update navigation position if this point is in the sorted list
        if point_idx in self._sorted_breath_indices:
            self._sorted_nav_position = self._sorted_breath_indices.index(point_idx)
            self._update_nav_position_label()

        # Update trace panel
        self._update_trace_tab2(meta['sweep'], meta['breath_idx'], meta['peak_time'])

        # Enable ALL classification buttons
        self._mark_noise_btn.setEnabled(True)
        self._mark_eupnea_btn.setEnabled(True)
        self._mark_sniffing_btn.setEnabled(True)
        self._mark_sigh_btn.setEnabled(True)
        self._mark_none_btn.setEnabled(True)
        self._skip_btn_tab2.setEnabled(True)
        self._jump_main_btn_tab2.setEnabled(True)

        # Get classification info
        class_names = {-1: 'Unclassified', 0: 'Eupnea', 1: 'Sniffing', 2: 'Noise'}
        class_name = 'Sigh' if meta['is_sigh'] else class_names.get(meta['current_class'], 'Unknown')

        # Update status info label (bottom of dialog)
        self._info_label_tab2.setText(
            f"Point {point_idx + 1}/{len(self._umap_breath_metadata)} | "
            f"Sweep {meta['sweep']} | Time {meta['peak_time']:.2f}s | Class: {class_name}"
        )

        # Update breath info label in classification bar (compact format)
        self._breath_info_label_tab2.setText(
            f"#{meta['breath_idx']+1} Sw{meta['sweep']+1} @{meta['peak_time']:.1f}s [{class_name}]"
        )

        # Update selection highlight in UMAP scatter
        self._update_selection_highlight()

    def _update_selection_highlight(self):
        """Update the visual highlight for the currently selected point in UMAP and scatter plots."""
        # Remove old 2D highlight
        if self._selection_highlight_item is not None:
            try:
                self._umap_plot_widget.removeItem(self._selection_highlight_item)
            except:
                pass
            self._selection_highlight_item = None

        # Remove old scatter plot highlights
        for key, highlight_item in list(self._scatter_selection_highlights.items()):
            if key in self._scatter_plots:
                try:
                    plot_widget, _ = self._scatter_plots[key]
                    plot_widget.removeItem(highlight_item)
                except:
                    pass
        self._scatter_selection_highlights.clear()

        if self._umap_selected_idx is None or self._umap_coords is None:
            return

        if self._umap_selected_idx >= len(self._umap_coords):
            return

        # Handle 3D highlight
        if self._umap_is_3d:
            self._update_3d_selection_highlight()
        else:
            # Get selected point coordinates for UMAP
            sel_x, sel_y = self._umap_coords[self._umap_selected_idx, :2]

            # Create a ring highlight (larger circle around selected point)
            self._selection_highlight_item = pg.ScatterPlotItem(
                pos=[(sel_x, sel_y)],
                size=25,  # Larger than normal points
                symbol='o',
                pen=pg.mkPen(color=(255, 255, 255, 255), width=3),  # White border
                brush=pg.mkBrush(None),  # Hollow (no fill)
            )
            self._selection_highlight_item.setZValue(100)  # Above all other items
            self._umap_plot_widget.addItem(self._selection_highlight_item)

        # Update scatter plot highlights
        self._update_scatter_selection_highlights()

    def _update_scatter_selection_highlights(self):
        """Update selection highlights in all scatter plots."""
        if self._umap_selected_idx is None:
            return

        if self._umap_feature_matrix is None or len(self._umap_feature_matrix) == 0:
            return

        if not hasattr(self, '_umap_selected_features') or self._umap_selected_features is None:
            return

        # Get the feature indices for each scatter plot
        features = self._umap_selected_features
        scatter_configs = {
            'if_ti': ('if', 'ti'),
            'if_amp': ('if', 'amp_insp'),
            'ti_te': ('ti', 'te'),
            'amp_dinsp': ('amp_insp', 'max_dinsp'),
        }

        for key, (x_feat, y_feat) in scatter_configs.items():
            if key not in self._scatter_plots:
                continue

            if x_feat not in features or y_feat not in features:
                continue

            plot_widget, scatter_item = self._scatter_plots[key]

            x_idx = features.index(x_feat)
            y_idx = features.index(y_feat)

            # Get selected point coordinates in this feature space
            sel_x = self._umap_feature_matrix[self._umap_selected_idx, x_idx]
            sel_y = self._umap_feature_matrix[self._umap_selected_idx, y_idx]

            # Create highlight marker
            highlight = pg.ScatterPlotItem(
                pos=[(sel_x, sel_y)],
                size=20,
                symbol='o',
                pen=pg.mkPen(color=(255, 255, 255, 255), width=3),
                brush=pg.mkBrush(None),
            )
            highlight.setZValue(100)
            plot_widget.addItem(highlight)
            self._scatter_selection_highlights[key] = highlight

    def _on_3d_point_clicked(self, point_idx: int):
        """Handle click on a point in the 3D UMAP view."""
        if point_idx is not None and point_idx < len(self._umap_breath_metadata):
            self._select_umap_point(point_idx)

    def _update_3d_selection_highlight(self):
        """Update the 3D selection highlight for the currently selected point."""
        if not OPENGL_AVAILABLE or self._umap_gl_widget is None:
            return

        # Remove old 3D highlight
        if self._selection_highlight_3d is not None:
            try:
                self._umap_gl_widget.removeItem(self._selection_highlight_3d)
            except:
                pass
            self._selection_highlight_3d = None

        if self._umap_selected_idx is None:
            return

        if not hasattr(self, '_umap_gl_coords_normalized') or self._umap_gl_coords_normalized is None:
            return

        if self._umap_selected_idx >= len(self._umap_gl_coords_normalized):
            return

        # Get selected point coordinates (normalized)
        sel_coords = self._umap_gl_coords_normalized[self._umap_selected_idx]

        # Create a larger, brighter point as highlight
        self._selection_highlight_3d = gl.GLScatterPlotItem(
            pos=np.array([sel_coords]),
            color=np.array([[1.0, 1.0, 1.0, 1.0]]),  # White
            size=20,  # Larger than normal points
            pxMode=True
        )
        self._umap_gl_widget.addItem(self._selection_highlight_3d)

    def _update_trace_tab2(self, sweep: int, breath_idx: int, peak_time: float):
        """Update Tab 2 trace panel to show selected breath."""
        st = self.main_window.state

        # Get processed trace
        old_sweep = st.sweep_idx
        st.sweep_idx = sweep
        times, trace = self.main_window._current_trace()
        st.sweep_idx = old_sweep

        if times is None or trace is None:
            self._trace_info_label_tab2.setText(f"Error: Could not load sweep {sweep}")
            return

        # Get window bounds
        window_text = self._window_size_combo_tab2.currentText()
        half_window = float(window_text.replace('±', '').replace('s', ''))

        t_min = peak_time - half_window
        t_max = peak_time + half_window

        # Extract window
        mask = (times >= t_min) & (times <= t_max)
        window_times = times[mask]
        window_trace = trace[mask]

        # Clear and redraw
        self._ax_tab2.clear()
        self.theme_manager.apply_theme(self._ax_tab2, self._fig_tab2, 'dark')

        # Get metadata for current breath
        meta = self._umap_breath_metadata[self._umap_selected_idx]
        peak_sample = meta['peak_sample']

        # Find breath boundaries (inspiratory onset to expiratory offset) for highlighting
        breath_onset_time = None
        breath_expoff_time = None
        breath_data = st.all_breaths_by_sweep.get(sweep)
        if breath_data:
            sr_hz = st.sr_hz
            onsets = breath_data.get('onsets', np.array([]))
            expoffs = breath_data.get('expoffs', np.array([]))

            if len(onsets) > 0:
                # Find the onset closest to (and before) the peak
                onset_diffs = peak_sample - onsets
                valid_onsets = onset_diffs >= 0  # Only onsets before the peak
                if np.any(valid_onsets):
                    closest_onset_idx = np.argmin(onset_diffs[valid_onsets])
                    actual_idx = np.where(valid_onsets)[0][closest_onset_idx]
                    breath_onset_time = onsets[actual_idx] / sr_hz

                    # Get corresponding expiratory offset (end of full breath cycle)
                    if len(expoffs) > 0 and actual_idx < len(expoffs):
                        breath_expoff_time = expoffs[actual_idx] / sr_hz

        # Add background highlight for the current breath region (onset to expiratory offset)
        if breath_onset_time is not None and breath_expoff_time is not None:
            # Cyan/teal highlight for the breath region
            self._ax_tab2.axvspan(
                breath_onset_time, breath_expoff_time,
                alpha=0.25, color='#00CED1', zorder=0,
                label='Current Breath'
            )

        # Plot trace - use different color within breath region
        trace_color = self.theme_manager.get_color('trace_color')

        if breath_onset_time is not None and breath_expoff_time is not None:
            # Split trace into three parts: before, during, after breath
            before_mask = (window_times < breath_onset_time)
            during_mask = (window_times >= breath_onset_time) & (window_times <= breath_expoff_time)
            after_mask = (window_times > breath_expoff_time)

            # Plot trace segments with different colors
            if np.any(before_mask):
                self._ax_tab2.plot(window_times[before_mask], window_trace[before_mask],
                                   color=trace_color, linewidth=0.8)
            if np.any(during_mask):
                # Highlighted segment - brighter cyan color
                self._ax_tab2.plot(window_times[during_mask], window_trace[during_mask],
                                   color='#00FFFF', linewidth=1.5, zorder=2)
            if np.any(after_mask):
                self._ax_tab2.plot(window_times[after_mask], window_trace[after_mask],
                                   color=trace_color, linewidth=0.8)
        else:
            # No breath boundaries found - plot normally
            self._ax_tab2.plot(window_times, window_trace, color=trace_color, linewidth=0.8)

        # Plot peak marker
        peak_idx = np.argmin(np.abs(times - peak_time))
        peak_y = trace[peak_idx]
        self._ax_tab2.plot(peak_time, peak_y, 'rv', markersize=12, label='Selected Breath')

        # Plot breath events if enabled
        if self._show_events_cb_tab2.isChecked():
            breath_data = st.all_breaths_by_sweep.get(sweep)
            if breath_data:
                sr_hz = st.sr_hz

                # Onsets
                if 'onsets' in breath_data:
                    onset_times = breath_data['onsets'] / sr_hz
                    onset_mask = (onset_times >= t_min) & (onset_times <= t_max)
                    for ot in onset_times[onset_mask]:
                        idx = np.argmin(np.abs(times - ot))
                        self._ax_tab2.scatter([ot], [trace[idx]], marker='^', color='#2ecc71', s=50, zorder=4)

                # Offsets (end of inspiration)
                if 'offsets' in breath_data:
                    offset_times = breath_data['offsets'] / sr_hz
                    offset_mask = (offset_times >= t_min) & (offset_times <= t_max)
                    for ot in offset_times[offset_mask]:
                        idx = np.argmin(np.abs(times - ot))
                        self._ax_tab2.scatter([ot], [trace[idx]], marker='v', color='#f39c12', s=50, zorder=4)

                # Expiratory minimums
                if 'expmins' in breath_data:
                    expmin_times = breath_data['expmins'] / sr_hz
                    expmin_mask = (expmin_times >= t_min) & (expmin_times <= t_max)
                    for et in expmin_times[expmin_mask]:
                        idx = np.argmin(np.abs(times - et))
                        self._ax_tab2.scatter([et], [trace[idx]], marker='d', color='#9b59b6', s=40, zorder=4)

                # Expiratory offsets (end of breath)
                if 'expoffs' in breath_data:
                    expoff_times = breath_data['expoffs'] / sr_hz
                    expoff_mask = (expoff_times >= t_min) & (expoff_times <= t_max)
                    for et in expoff_times[expoff_mask]:
                        idx = np.argmin(np.abs(times - et))
                        self._ax_tab2.scatter([et], [trace[idx]], marker='s', color='#e74c3c', s=40, zorder=4)

        # Labels
        self._ax_tab2.set_xlabel('Time (s)', color='white')
        self._ax_tab2.set_ylabel('Amplitude', color='white')

        # Get classification for title (meta already fetched above)
        class_names = {-1: 'Unclassified', 0: 'Eupnea', 1: 'Sniffing', 2: 'Noise'}
        class_name = 'Sigh' if meta['is_sigh'] else class_names.get(meta['current_class'], 'Unknown')

        self._ax_tab2.set_title(f"Sweep {sweep} | Time {peak_time:.2f}s | {class_name}", color='white', fontsize=11)
        self._ax_tab2.set_xlim(t_min, t_max)

        self._canvas_tab2.draw()

        # Update info label
        self._trace_info_label_tab2.setText(
            f"Breath #{breath_idx} in Sweep {sweep} | Peak at {peak_time:.2f}s | Current: {class_name}"
        )

    def _refresh_trace_tab2(self):
        """Refresh trace panel with current settings."""
        if self._umap_selected_idx is not None and self._umap_selected_idx < len(self._umap_breath_metadata):
            meta = self._umap_breath_metadata[self._umap_selected_idx]
            self._update_trace_tab2(meta['sweep'], meta['breath_idx'], meta['peak_time'])

    def _update_ui_state_tab2(self):
        """Update Tab 2 UI button states."""
        has_data = self._umap_coords is not None and len(self._umap_coords) > 0
        has_selection = self._umap_selected_idx is not None

        # Enable/disable based on data
        self._export_btn_tab2.setEnabled(has_data)

        # Navigation buttons - enabled when we have breath data (sorted list)
        has_breaths = len(self._sorted_breath_indices) > 0
        self._prev_breath_btn.setEnabled(has_breaths)
        self._next_breath_btn.setEnabled(has_breaths)

        # Classification buttons require selection
        self._mark_noise_btn.setEnabled(has_selection)
        self._mark_eupnea_btn.setEnabled(has_selection)
        self._mark_sniffing_btn.setEnabled(has_selection)
        self._mark_sigh_btn.setEnabled(has_selection)
        self._mark_none_btn.setEnabled(has_selection)
        self._jump_main_btn_tab2.setEnabled(has_selection)

        # Update changes counter
        self._changes_label_tab2.setText(f"Changes: {self._umap_changes_count}")

    def _on_umap_dimension_changed(self):
        """Handle 2D/3D dimension toggle - use cached embeddings if available."""
        want_3d = self._umap_3d_radio.isChecked()

        # Check if we have a cached embedding for the requested dimension
        if want_3d and self._umap_coords_3d is not None:
            # Use cached 3D
            self._umap_coords = self._umap_coords_3d
            self._umap_is_3d = True
            self._update_umap_scatter()
            self._update_scatter_plots()
            print("[UMAP] Switched to cached 3D embedding")
        elif not want_3d and self._umap_coords_2d is not None:
            # Use cached 2D
            self._umap_coords = self._umap_coords_2d
            self._umap_is_3d = False
            self._update_umap_scatter()
            self._update_scatter_plots()
            print("[UMAP] Switched to cached 2D embedding")
        elif self._umap_feature_matrix is not None:
            # Need to compute for this dimension
            self._compute_umap()

    def _on_color_mode_changed(self):
        """Handle color mode change (class vs metric)."""
        is_metric_mode = self._color_metric_radio.isChecked()
        self._color_metric_combo.setEnabled(is_metric_mode)

        # Update scatter colors
        if self._umap_coords is not None:
            self._update_umap_scatter()

    def _on_visibility_changed(self):
        """Handle visibility toggle for noise/unclassified points."""
        if self._umap_coords is not None:
            self._update_umap_scatter()

    def _on_3d_opaque_changed(self):
        """Handle 3D opaque mode toggle - re-renders 3D scatter with new settings."""
        if self._umap_coords is not None and self._umap_is_3d:
            self._update_umap_scatter()

    def _on_color_metric_changed(self):
        """Handle color metric selection change."""
        if self._color_metric_radio.isChecked() and self._umap_coords is not None:
            self._update_umap_scatter()

    # ========== Decision Boundary Overlay Methods ==========

    def _on_overlay_type_changed(self, overlay_type: str):
        """Handle overlay type selection change."""
        self._overlay_type = overlay_type
        self._update_decision_boundary_overlay()

    def _on_overlay_classifier_changed(self, classifier_name: str):
        """Handle classifier selection change."""
        if classifier_name in self._available_classifiers:
            self._overlay_classifier, self._overlay_classifier_type = self._available_classifiers[classifier_name]
        else:
            self._overlay_classifier = None
            self._overlay_classifier_type = None
        self._update_decision_boundary_overlay()

    def _on_overlay_opacity_changed(self, value: int):
        """Handle overlay opacity slider change."""
        self._overlay_opacity = value / 100.0
        self._overlay_opacity_label.setText(f"{value}%")
        self._update_decision_boundary_overlay()

    def _refresh_available_classifiers(self):
        """Refresh the list of available classifiers for decision boundaries."""
        self._available_classifiers = {}
        st = self.main_window.state

        # Built-in classifiers
        # 1. GMM for Eupnea/Sniff (if computed)
        if hasattr(st, 'gmm_model') and st.gmm_model is not None:
            # Store model with scaler and features for proper prediction
            gmm_info = {
                'model': st.gmm_model,
                'scaler': getattr(st, 'gmm_scaler', None),
                'features': getattr(st, 'gmm_features', None),
                'sniffing_cluster_id': getattr(st, 'gmm_sniffing_cluster_id', None)
            }
            self._available_classifiers["GMM (Eupnea/Sniff)"] = (gmm_info, 'gmm')

        # 2. Noise detector model (if loaded)
        if hasattr(st, 'ml_breath_model') and st.ml_breath_model is not None:
            self._available_classifiers["Noise Detector"] = (st.ml_breath_model, 'noise')

        # 3. Loaded ML models from state
        if hasattr(st, 'loaded_ml_models') and st.loaded_ml_models:
            for model_name, model_data in st.loaded_ml_models.items():
                model = model_data.get('model')
                metadata = model_data.get('metadata', {})
                if model is not None:
                    # Determine model type from metadata or class name
                    model_class = type(model).__name__
                    if 'XGB' in model_class or 'xgb' in model_class.lower():
                        model_type = 'xgboost'
                    elif 'RandomForest' in model_class:
                        model_type = 'rf'
                    elif 'MLP' in model_class:
                        model_type = 'mlp'
                    else:
                        model_type = 'sklearn'
                    self._available_classifiers[f"ML: {model_name}"] = (model, model_type)

        # Update the combo box
        current_selection = self._overlay_classifier_combo.currentText()
        self._overlay_classifier_combo.clear()

        n_models = len(self._available_classifiers)
        if n_models > 0:
            self._overlay_classifier_combo.addItems(list(self._available_classifiers.keys()))

            # Restore selection if still available
            if current_selection in self._available_classifiers:
                self._overlay_classifier_combo.setCurrentText(current_selection)
            else:
                first_name = list(self._available_classifiers.keys())[0]
                self._overlay_classifier_combo.setCurrentText(first_name)
                self._overlay_classifier, self._overlay_classifier_type = self._available_classifiers[first_name]

            # Update status
            self._overlay_classifier_status.setText(f"{n_models} classifier(s) available")
            self._overlay_classifier_status.setStyleSheet("color: #2ecc71; font-size: 9pt;")
        else:
            self._overlay_classifier_combo.addItem("(No models available)")
            self._overlay_classifier = None
            self._overlay_classifier_type = None

            # Show help message
            self._overlay_classifier_status.setText("Run GMM clustering or load ML model first")
            self._overlay_classifier_status.setStyleSheet("color: #f39c12; font-size: 9pt;")

        print(f"[Overlays] Refreshed classifiers: {list(self._available_classifiers.keys())}")

    def _update_decision_boundary_overlay(self):
        """Update decision boundary overlay on UMAP and scatter plots."""
        print(f"[Overlays] _update_decision_boundary_overlay called, type={self._overlay_type}")

        if self._overlay_type == 'None':
            self._clear_overlay_items()
            return

        if self._umap_coords is None or len(self._umap_coords) < 3:
            print("[Overlays] No UMAP coords available")
            return

        if self._overlay_classifier is None:
            # Try to get classifier from current selection
            classifier_name = self._overlay_classifier_combo.currentText()
            print(f"[Overlays] Trying to get classifier: {classifier_name}")
            if classifier_name in self._available_classifiers:
                self._overlay_classifier, self._overlay_classifier_type = self._available_classifiers[classifier_name]
                print(f"[Overlays] Got classifier type: {self._overlay_classifier_type}")
            else:
                # No classifier available - show message
                print(f"[Overlays] Classifier not found in available: {list(self._available_classifiers.keys())}")
                self._info_label_tab2.setText("No classifier available. Run GMM clustering or load ML model first.")
                return

        # Only support 2D overlays for now
        if self._umap_is_3d:
            self._info_label_tab2.setText("Decision boundaries only supported in 2D mode")
            return

        try:
            print(f"[Overlays] Drawing overlay type: {self._overlay_type}")
            if self._overlay_type in ['Probability Heatmap', 'Both']:
                self._draw_probability_heatmap()

            if self._overlay_type in ['Boundary Lines', 'Both']:
                self._draw_boundary_lines()

            print("[Overlays] Overlay drawing completed")

        except Exception as e:
            print(f"[Overlays] Error drawing decision boundary: {e}")
            import traceback
            traceback.print_exc()

    def _clear_overlay_items(self):
        """Remove all overlay items from plots."""
        # Clear heatmap
        if self._overlay_heatmap_item is not None:
            try:
                self._umap_plot_widget.removeItem(self._overlay_heatmap_item)
            except:
                pass
            self._overlay_heatmap_item = None

        # Clear boundary lines
        for item in self._overlay_boundary_items:
            try:
                self._umap_plot_widget.removeItem(item)
            except:
                pass
        self._overlay_boundary_items = []

    def _draw_probability_heatmap(self):
        """Draw probability heatmap as background on UMAP plot."""
        if self._umap_coords is None or len(self._umap_coords) < 3:
            return

        # Get data bounds
        x_coords = self._umap_coords[:, 0]
        y_coords = self._umap_coords[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Add padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # Create grid
        grid_resolution = 50
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Get predictions on grid
        try:
            probs = self._predict_on_grid(grid_points)
            if probs is None:
                return
            probs = probs.reshape(xx.shape)
        except Exception as e:
            print(f"[Overlays] Error predicting on grid: {e}")
            return

        # Remove old heatmap
        if self._overlay_heatmap_item is not None:
            try:
                self._umap_plot_widget.removeItem(self._overlay_heatmap_item)
            except:
                pass

        # Create colormap: green (eupnea, low prob) to purple (sniffing, high prob)
        from matplotlib import cm
        from matplotlib.colors import LinearSegmentedColormap

        # Custom colormap: green -> gray -> purple
        colors = [
            (46/255, 204/255, 113/255),   # Eupnea green
            (127/255, 140/255, 141/255),  # Gray (boundary)
            (155/255, 89/255, 182/255)    # Sniffing purple
        ]
        cmap = LinearSegmentedColormap.from_list('eupnea_sniff', colors, N=256)

        # Apply colormap
        rgba = cmap(probs)
        rgba[:, :, 3] = self._overlay_opacity  # Set opacity

        # Convert to 8-bit RGBA
        img_data = (rgba * 255).astype(np.uint8)

        # Create ImageItem
        self._overlay_heatmap_item = pg.ImageItem(img_data)

        # Set transform to position correctly
        scale_x = (x_max - x_min) / grid_resolution
        scale_y = (y_max - y_min) / grid_resolution
        self._overlay_heatmap_item.setTransform(
            pg.QtGui.QTransform()
            .translate(x_min, y_min)
            .scale(scale_x, scale_y)
        )

        # Add to plot (behind scatter points)
        self._overlay_heatmap_item.setZValue(-10)
        self._umap_plot_widget.addItem(self._overlay_heatmap_item)

    def _draw_boundary_lines(self):
        """Draw decision boundary contour lines on UMAP plot."""
        if self._umap_coords is None or len(self._umap_coords) < 3:
            return

        # Get data bounds
        x_coords = self._umap_coords[:, 0]
        y_coords = self._umap_coords[:, 1]
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        # Add padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        # Create grid
        grid_resolution = 100
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Get predictions on grid
        try:
            probs = self._predict_on_grid(grid_points)
            if probs is None:
                return
            probs = probs.reshape(xx.shape)
        except Exception as e:
            print(f"[Overlays] Error predicting on grid: {e}")
            return

        # Clear old boundary lines
        for item in self._overlay_boundary_items:
            try:
                self._umap_plot_widget.removeItem(item)
            except:
                pass
        self._overlay_boundary_items = []

        # Find contour at 0.5 (decision boundary)
        try:
            from skimage import measure
            contours = measure.find_contours(probs, 0.5)

            for contour in contours:
                # Convert contour indices to data coordinates
                x_line = x_min + contour[:, 1] * (x_max - x_min) / grid_resolution
                y_line = y_min + contour[:, 0] * (y_max - y_min) / grid_resolution

                # Create line item
                pen = pg.mkPen(color=(255, 255, 255, int(255 * self._overlay_opacity)), width=2)
                line_item = pg.PlotCurveItem(x_line, y_line, pen=pen)
                line_item.setZValue(5)  # Above heatmap, below scatter
                self._umap_plot_widget.addItem(line_item)
                self._overlay_boundary_items.append(line_item)

        except ImportError:
            # Fall back to matplotlib contour if skimage not available
            print("[Overlays] skimage not available, using matplotlib for contours")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            cs = ax.contour(xx, yy, probs, levels=[0.5])
            plt.close(fig)

            for collection in cs.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    x_line = vertices[:, 0]
                    y_line = vertices[:, 1]

                    pen = pg.mkPen(color=(255, 255, 255, int(255 * self._overlay_opacity)), width=2)
                    line_item = pg.PlotCurveItem(x_line, y_line, pen=pen)
                    line_item.setZValue(5)
                    self._umap_plot_widget.addItem(line_item)
                    self._overlay_boundary_items.append(line_item)

    def _predict_on_grid(self, grid_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities for grid points in UMAP space.

        This maps UMAP coordinates back to feature space using inverse distance weighting
        from the original data points, then applies the classifier.

        Args:
            grid_points: (N, 2) array of UMAP coordinates

        Returns:
            (N,) array of probabilities (probability of class 1 / sniffing)
        """
        if self._overlay_classifier is None:
            print("[Overlays] No classifier available")
            return None

        if self._umap_feature_matrix is None or len(self._umap_feature_matrix) == 0:
            print("[Overlays] No UMAP feature matrix available")
            return None

        # Get original UMAP coordinates and features
        umap_coords_2d = self._umap_coords[:, :2]
        features = self._umap_feature_matrix

        # For each grid point, interpolate features using inverse distance weighting
        n_grid = len(grid_points)
        n_neighbors = min(10, len(umap_coords_2d))  # Use up to 10 nearest neighbors

        grid_features = np.zeros((n_grid, features.shape[1]))

        from scipy.spatial.distance import cdist
        distances = cdist(grid_points, umap_coords_2d)

        for i in range(n_grid):
            # Get nearest neighbors
            nn_indices = np.argsort(distances[i])[:n_neighbors]
            nn_dists = distances[i, nn_indices]

            # Inverse distance weights (with small epsilon to avoid division by zero)
            weights = 1.0 / (nn_dists + 1e-8)
            weights /= weights.sum()

            # Weighted average of features
            grid_features[i] = np.average(features[nn_indices], axis=0, weights=weights)

        # Apply classifier
        try:
            if self._overlay_classifier_type == 'gmm':
                # GMM is stored as a dict with model, scaler, features
                gmm_info = self._overlay_classifier
                model = gmm_info.get('model')
                scaler = gmm_info.get('scaler')
                gmm_features = gmm_info.get('features')
                sniffing_cluster_id = gmm_info.get('sniffing_cluster_id', 1)

                if model is None:
                    print("[Overlays] GMM model not found in classifier info")
                    return None

                # Check feature compatibility
                umap_features = getattr(self, '_umap_selected_features', None)
                if gmm_features and umap_features and set(gmm_features) != set(umap_features):
                    print(f"[Overlays] Warning: GMM features {gmm_features} don't match UMAP features {umap_features}")
                    # Continue anyway - the interpolated features are already scaled by UMAP scaler
                    # We'll use them directly (may not be perfectly accurate but gives visualization)

                # The grid_features are already scaled by the UMAP scaler (StandardScaler)
                # GMM was also trained on StandardScaler output, so should be compatible
                probs = model.predict_proba(grid_features)

                # Return probability of sniffing cluster
                if sniffing_cluster_id is not None and sniffing_cluster_id < probs.shape[1]:
                    return probs[:, sniffing_cluster_id]
                else:
                    # Fallback to class 1
                    return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            elif hasattr(self._overlay_classifier, 'predict_proba'):
                probs = self._overlay_classifier.predict_proba(grid_features)
                return probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            elif hasattr(self._overlay_classifier, 'predict'):
                # For classifiers without predict_proba
                preds = self._overlay_classifier.predict(grid_features)
                return preds.astype(float)

            else:
                print(f"[Overlays] Classifier type {self._overlay_classifier_type} not supported")
                return None

        except Exception as e:
            print(f"[Overlays] Error in classifier prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ========== Animation Methods ==========

    def _on_color_time_changed(self, checked: bool):
        """Handle color by time radio button toggle."""
        if checked and self._umap_coords is not None:
            self._update_umap_scatter()
            self._update_scatter_plots()

            # Update time range label
            if len(self._umap_breath_metadata) > 0:
                times = [meta['peak_time'] for meta in self._umap_breath_metadata]
                t_min, t_max = min(times), max(times)
                self._anim_time_label.setText(f"Time: {t_min:.1f}s - {t_max:.1f}s")

    def _on_anim_speed_changed(self, value: int):
        """Handle animation speed slider change."""
        self._anim_speed_label.setText(f"{value}x")
        # Map 1-10 to 500ms-50ms (inverse relationship)
        self._anim_speed_ms = int(550 - value * 50)

        # Update timer if playing
        if self._anim_timer is not None and self._anim_playing:
            self._anim_timer.setInterval(self._anim_speed_ms)

    def _on_anim_mode_changed(self, mode: str):
        """Handle animation mode change."""
        self._anim_mode = mode
        # Reset animation state when mode changes
        self._anim_visible_indices = None

        # Update scatter to reset visibility
        if self._umap_coords is not None and not self._anim_playing:
            self._update_umap_scatter()

    def _on_anim_trail_changed(self, value: int):
        """Handle trail length slider change."""
        self._anim_trail_length = value
        self._anim_trail_label.setText(str(value))

        # Update scatter if in trail mode and playing
        if self._anim_mode == 'Trail' and self._anim_playing:
            self._update_animation_visibility()
            self._update_umap_scatter()

    def _update_animation_visibility(self):
        """Update which points are visible based on animation mode and current step."""
        if self._anim_time_sorted_indices is None or len(self._anim_time_sorted_indices) == 0:
            self._anim_visible_indices = None
            return

        n_points = len(self._anim_time_sorted_indices)
        current = self._anim_current_step

        if self._anim_mode == 'Highlight':
            # Only current point is special (handled by selection highlight)
            self._anim_visible_indices = None

        elif self._anim_mode == 'Cumulative':
            # All points from 0 to current step are visible
            visible_steps = self._anim_time_sorted_indices[:current + 1]
            self._anim_visible_indices = set(visible_steps)

        elif self._anim_mode == 'Trail':
            # Last N points are visible (with the fading handled in colors)
            start_step = max(0, current - self._anim_trail_length + 1)
            visible_steps = self._anim_time_sorted_indices[start_step:current + 1]
            self._anim_visible_indices = set(visible_steps)

    def _anim_toggle_play(self):
        """Toggle animation play/pause."""
        if self._umap_coords is None or len(self._umap_breath_metadata) == 0:
            self._anim_play_btn.setChecked(False)
            return

        self._anim_playing = self._anim_play_btn.isChecked()

        if self._anim_playing:
            # Start animation
            self._anim_play_btn.setText("⏸")

            # Ensure time-sorted indices are available (use absolute_time for proper ordering)
            if self._anim_time_sorted_indices is None:
                times = np.array([meta.get('absolute_time', meta['peak_time']) for meta in self._umap_breath_metadata])
                self._anim_time_sorted_indices = np.argsort(times)

            # Find current position in time order
            if self._umap_selected_idx is not None:
                # Find where current selection is in time order
                positions = np.where(self._anim_time_sorted_indices == self._umap_selected_idx)[0]
                if len(positions) > 0:
                    self._anim_current_step = positions[0]
                else:
                    self._anim_current_step = 0
            else:
                self._anim_current_step = 0

            # Initialize visibility for cumulative/trail modes
            if self._anim_mode in ['Cumulative', 'Trail']:
                self._update_animation_visibility()
                self._update_umap_scatter()

            # Create and start timer
            if self._anim_timer is None:
                self._anim_timer = QTimer()
                self._anim_timer.timeout.connect(self._anim_step_forward)

            self._anim_timer.setInterval(self._anim_speed_ms)
            self._anim_timer.start()

        else:
            # Stop animation
            self._anim_play_btn.setText("▶")
            if self._anim_timer is not None:
                self._anim_timer.stop()

            # Clear animation visibility to show all points again
            self._anim_visible_indices = None
            if self._umap_coords is not None:
                self._update_umap_scatter()

    def _anim_step_forward(self):
        """Advance animation by one step."""
        if self._anim_time_sorted_indices is None or len(self._anim_time_sorted_indices) == 0:
            return

        self._anim_current_step = (self._anim_current_step + 1) % len(self._anim_time_sorted_indices)
        point_idx = self._anim_time_sorted_indices[self._anim_current_step]

        # Update visibility for cumulative/trail modes
        if self._anim_mode in ['Cumulative', 'Trail']:
            self._update_animation_visibility()
            self._update_umap_scatter()

        # Select the current point (updates trace viewer and highlight)
        self._select_umap_point(point_idx)

        # Update time label with sweep info
        meta = self._umap_breath_metadata[point_idx]
        self._anim_time_label.setText(
            f"Sweep {meta['sweep']+1} @ {meta['peak_time']:.2f}s ({self._anim_current_step + 1}/{len(self._anim_time_sorted_indices)})"
        )

    def _anim_step_prev(self):
        """Step to previous breath in time order."""
        if self._umap_coords is None or len(self._umap_breath_metadata) == 0:
            return

        # Ensure time-sorted indices (use absolute_time for proper ordering)
        if self._anim_time_sorted_indices is None:
            times = np.array([meta.get('absolute_time', meta['peak_time']) for meta in self._umap_breath_metadata])
            self._anim_time_sorted_indices = np.argsort(times)

        if self._umap_selected_idx is not None:
            # Find current position and step back
            positions = np.where(self._anim_time_sorted_indices == self._umap_selected_idx)[0]
            if len(positions) > 0:
                self._anim_current_step = (positions[0] - 1) % len(self._anim_time_sorted_indices)
            else:
                self._anim_current_step = len(self._anim_time_sorted_indices) - 1
        else:
            self._anim_current_step = len(self._anim_time_sorted_indices) - 1

        point_idx = self._anim_time_sorted_indices[self._anim_current_step]
        self._select_umap_point(point_idx)

        # Update time label with sweep info
        meta = self._umap_breath_metadata[point_idx]
        self._anim_time_label.setText(
            f"Sweep {meta['sweep']+1} @ {meta['peak_time']:.2f}s ({self._anim_current_step + 1}/{len(self._anim_time_sorted_indices)})"
        )

    def _anim_step_next(self):
        """Step to next breath in time order."""
        if self._umap_coords is None or len(self._umap_breath_metadata) == 0:
            return

        # Ensure time-sorted indices (use absolute_time for proper ordering)
        if self._anim_time_sorted_indices is None:
            times = np.array([meta.get('absolute_time', meta['peak_time']) for meta in self._umap_breath_metadata])
            self._anim_time_sorted_indices = np.argsort(times)

        if self._umap_selected_idx is not None:
            # Find current position and step forward
            positions = np.where(self._anim_time_sorted_indices == self._umap_selected_idx)[0]
            if len(positions) > 0:
                self._anim_current_step = (positions[0] + 1) % len(self._anim_time_sorted_indices)
            else:
                self._anim_current_step = 0
        else:
            self._anim_current_step = 0

        point_idx = self._anim_time_sorted_indices[self._anim_current_step]
        self._select_umap_point(point_idx)

        # Update time label with sweep info
        meta = self._umap_breath_metadata[point_idx]
        self._anim_time_label.setText(
            f"Sweep {meta['sweep']+1} @ {meta['peak_time']:.2f}s ({self._anim_current_step + 1}/{len(self._anim_time_sorted_indices)})"
        )

    def _mark_as_eupnea(self):
        """Mark selected breath as eupnea."""
        self._apply_breath_classification(class_label=0, class_name='Eupnea', is_sigh=False)

    def _mark_as_sniffing(self):
        """Mark selected breath as sniffing."""
        self._apply_breath_classification(class_label=1, class_name='Sniffing', is_sigh=False)

    def _mark_as_sigh(self):
        """Mark selected breath as sigh."""
        self._apply_breath_classification(class_label=0, class_name='Sigh', is_sigh=True)

    def _apply_breath_classification(self, class_label: int, class_name: str, is_sigh: bool):
        """Apply classification to the selected breath."""
        if self._umap_selected_idx is None:
            return

        meta = self._umap_breath_metadata[self._umap_selected_idx]
        sweep = meta['sweep']
        peak_sample = meta['peak_sample']

        st = self.main_window.state

        print(f"\n===== Applying Classification =====")
        print(f"Sweep: {sweep}, Peak: {peak_sample}")
        print(f"Marking as: {class_name}")

        # Update breath_type_class in all_peaks_by_sweep
        all_peaks = st.all_peaks_by_sweep.get(sweep)
        if all_peaks is not None:
            peak_indices = all_peaks.get('indices', np.array([]))
            peak_mask = peak_indices == peak_sample

            if np.any(peak_mask):
                peak_pos = np.where(peak_mask)[0][0]

                # Initialize breath_type_class if needed
                if 'breath_type_class' not in all_peaks or all_peaks['breath_type_class'] is None:
                    all_peaks['breath_type_class'] = np.full(len(peak_indices), -1, dtype=np.int8)

                all_peaks['breath_type_class'][peak_pos] = class_label

                # Update eupnea_sniff_source
                if 'eupnea_sniff_source' not in all_peaks or all_peaks['eupnea_sniff_source'] is None:
                    all_peaks['eupnea_sniff_source'] = np.array(['auto'] * len(peak_indices), dtype=object)
                all_peaks['eupnea_sniff_source'][peak_pos] = 'user'

        # Handle sigh marking
        if is_sigh:
            if not hasattr(st, 'sigh_by_sweep'):
                st.sigh_by_sweep = {}
            if sweep not in st.sigh_by_sweep:
                st.sigh_by_sweep[sweep] = np.array([], dtype=np.int64)

            # Add to sighs if not already there
            if peak_sample not in st.sigh_by_sweep[sweep]:
                st.sigh_by_sweep[sweep] = np.append(st.sigh_by_sweep[sweep], peak_sample)
        else:
            # Remove from sighs if it was a sigh
            if hasattr(st, 'sigh_by_sweep') and sweep in st.sigh_by_sweep:
                st.sigh_by_sweep[sweep] = st.sigh_by_sweep[sweep][st.sigh_by_sweep[sweep] != peak_sample]

        # Update metadata
        meta['current_class'] = class_label
        meta['is_sigh'] = is_sigh

        # Track reviewed and increment changes counter
        self._umap_reviewed_breaths.add((sweep, peak_sample))
        self._umap_changes_count += 1

        # Update scatter colors
        self._update_umap_scatter()

        # Update UI
        self._update_ui_state_tab2()

        # Refresh main window if on same sweep
        if self.main_window.current_sweep == sweep:
            self.main_window.plot_sweep()

        print("===== Classification Applied =====\n")

    def _skip_breath_tab2(self):
        """Skip current breath without changing classification."""
        if self._umap_selected_idx is None:
            return

        meta = self._umap_breath_metadata[self._umap_selected_idx]
        self._umap_reviewed_breaths.add((meta['sweep'], meta['peak_sample']))
        self._update_ui_state_tab2()

    # =========================================================================
    # CONFIDENCE-BASED NAVIGATION
    # =========================================================================

    def _on_confidence_sort_changed(self, sort_text: str):
        """Handle confidence sort dropdown change."""
        self._sort_breaths_by_confidence(sort_text)
        self._update_nav_buttons_state()

        # Auto-select first breath if none selected and we have data
        if self._umap_selected_idx is None and self._sorted_breath_indices:
            self._sorted_nav_position = 0
            breath_idx = self._sorted_breath_indices[0]
            self._select_umap_point(breath_idx)
            self._update_nav_position_label()

    def _sort_breaths_by_confidence(self, sort_text: str):
        """Sort breaths by the selected confidence metric."""
        if not self._umap_breath_metadata:
            self._sorted_breath_indices = []
            self._sorted_nav_position = -1
            return

        # Get confidence scores for all breaths
        confidences = []
        for i, meta in enumerate(self._umap_breath_metadata):
            conf = self._get_breath_confidence(meta, sort_text)
            confidences.append((i, conf))

        # Sort based on selection
        if sort_text == "Time Order":
            # Sort by absolute time (original order)
            confidences.sort(key=lambda x: self._umap_breath_metadata[x[0]].get('absolute_time', 0))
        elif "Low→High" in sort_text:
            # Sort by confidence ascending (lowest confidence first - most uncertain)
            confidences.sort(key=lambda x: x[1])
        elif "High→Low" in sort_text:
            # Sort by confidence descending (highest confidence first - most certain)
            confidences.sort(key=lambda x: x[1], reverse=True)

        # Store sorted indices
        self._sorted_breath_indices = [idx for idx, _ in confidences]
        self._sorted_nav_position = -1

        # If a breath is currently selected, find its position in the sorted list
        if self._umap_selected_idx is not None and self._umap_selected_idx in self._sorted_breath_indices:
            self._sorted_nav_position = self._sorted_breath_indices.index(self._umap_selected_idx)

        self._update_nav_position_label()
        print(f"[Confidence Nav] Sorted {len(self._sorted_breath_indices)} breaths by '{sort_text}'")

    def _get_breath_confidence(self, meta: dict, sort_text: str) -> float:
        """
        Get the confidence score for a breath based on the sort type.

        Returns a value from 0 (most uncertain) to 1 (most certain).
        """
        st = self.main_window.state
        sweep = meta['sweep']
        peak_sample = meta['peak_sample']

        # Get all_peaks for this sweep
        all_peaks = st.all_peaks_by_sweep.get(sweep, {})
        if not all_peaks or 'indices' not in all_peaks:
            return 0.5  # Default to uncertain

        # Find peak position in all_peaks
        peak_indices = all_peaks.get('indices', np.array([]))
        peak_mask = peak_indices == peak_sample
        if not np.any(peak_mask):
            return 0.5
        peak_pos = np.where(peak_mask)[0][0]

        # Get confidence based on sort type
        if "Noise/Breath" in sort_text:
            # Use noise/breath classifier probability
            # Check for stored probabilities from ML classifiers
            for prob_key in ['labels_xgboost_proba', 'labels_rf_proba', 'labels_mlp_proba', 'ml_breath_prob']:
                if prob_key in all_peaks and all_peaks[prob_key] is not None:
                    prob_arr = all_peaks[prob_key]
                    if peak_pos < len(prob_arr):
                        prob = prob_arr[peak_pos]
                        # Confidence is distance from decision boundary (0.5)
                        return abs(prob - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1

            # Fallback: use label source (user-edited = high confidence)
            if 'label_source' in all_peaks:
                source_arr = all_peaks['label_source']
                if peak_pos < len(source_arr):
                    return 1.0 if source_arr[peak_pos] == 'user' else 0.5

            return 0.5  # No probability info available

        elif "Eupnea/Sniff" in sort_text:
            # Use eupnea/sniff classifier probability (GMM or ML)
            if 'gmm_eupnea_prob' in all_peaks and all_peaks['gmm_eupnea_prob'] is not None:
                prob_arr = all_peaks['gmm_eupnea_prob']
                if peak_pos < len(prob_arr):
                    prob = prob_arr[peak_pos]
                    return abs(prob - 0.5) * 2

            # Check for ML classifier probabilities
            for prob_key in ['eupnea_sniff_xgboost_proba', 'eupnea_sniff_rf_proba', 'eupnea_sniff_mlp_proba']:
                if prob_key in all_peaks and all_peaks[prob_key] is not None:
                    prob_arr = all_peaks[prob_key]
                    if peak_pos < len(prob_arr):
                        prob = prob_arr[peak_pos]
                        return abs(prob - 0.5) * 2

            # Use state-level GMM probabilities if available
            if hasattr(st, 'gmm_sniff_probabilities') and sweep in st.gmm_sniff_probabilities:
                gmm_probs = st.gmm_sniff_probabilities[sweep]
                if peak_pos < len(gmm_probs):
                    prob = gmm_probs[peak_pos]
                    return abs(prob - 0.5) * 2

            return 0.5

        elif "Breath/Sigh" in sort_text:
            # Use sigh classifier probability
            for prob_key in ['sigh_xgboost_proba', 'sigh_rf_proba', 'sigh_mlp_proba']:
                if prob_key in all_peaks and all_peaks[prob_key] is not None:
                    prob_arr = all_peaks[prob_key]
                    if peak_pos < len(prob_arr):
                        prob = prob_arr[peak_pos]
                        return abs(prob - 0.5) * 2

            # Check if manually marked as sigh (high confidence)
            if meta.get('is_sigh', False):
                return 1.0

            return 0.5

        return 0.5  # Default

    def _go_to_prev_breath(self):
        """Navigate to the previous breath in the sorted list."""
        if not self._sorted_breath_indices:
            return

        if self._sorted_nav_position <= 0:
            # Already at first or not started, go to last
            self._sorted_nav_position = len(self._sorted_breath_indices) - 1
        else:
            self._sorted_nav_position -= 1

        # Select the breath at this position
        breath_idx = self._sorted_breath_indices[self._sorted_nav_position]
        self._select_umap_point(breath_idx)
        self._update_nav_position_label()

    def _go_to_next_breath(self):
        """Navigate to the next breath in the sorted list."""
        if not self._sorted_breath_indices:
            return

        if self._sorted_nav_position >= len(self._sorted_breath_indices) - 1:
            # Already at last, go to first
            self._sorted_nav_position = 0
        else:
            self._sorted_nav_position += 1

        # Select the breath at this position
        breath_idx = self._sorted_breath_indices[self._sorted_nav_position]
        self._select_umap_point(breath_idx)
        self._update_nav_position_label()

    def _update_nav_position_label(self):
        """Update the navigation position label."""
        if not self._sorted_breath_indices:
            self._nav_position_label.setText("--/--")
        elif self._sorted_nav_position < 0:
            self._nav_position_label.setText(f"--/{len(self._sorted_breath_indices)}")
        else:
            self._nav_position_label.setText(
                f"{self._sorted_nav_position + 1}/{len(self._sorted_breath_indices)}"
            )

    def _update_nav_buttons_state(self):
        """Update the enabled state of navigation buttons."""
        has_breaths = len(self._sorted_breath_indices) > 0
        self._prev_breath_btn.setEnabled(has_breaths)
        self._next_breath_btn.setEnabled(has_breaths)
        self._update_nav_position_label()

    def _jump_to_main_window_tab2(self):
        """Jump to selected breath in main window."""
        if self._umap_selected_idx is None:
            return

        meta = self._umap_breath_metadata[self._umap_selected_idx]

        print(f"\n===== Jumping to Main Window =====")
        print(f"Sweep: {meta['sweep']}, Time: {meta['peak_time']:.2f}s")

        # Switch to sweep
        self.main_window.CurrentSweep.setValue(meta['sweep'])

        # Center view on peak
        window_half_width = 2.0
        xlim_left = max(0, meta['peak_time'] - window_half_width)
        xlim_right = meta['peak_time'] + window_half_width

        self.main_window.ax.set_xlim(xlim_left, xlim_right)
        self.main_window.canvas.draw()

        print("===== Jump Complete =====\n")

    def _export_classifications_tab2(self):
        """Export breath classifications to CSV."""
        from PyQt6.QtWidgets import QFileDialog
        import csv

        if not self._umap_breath_metadata:
            QMessageBox.warning(self, "No Data", "No breath data to export.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Classifications",
            "breath_classifications.csv",
            "CSV Files (*.csv)"
        )

        if not filename:
            return

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Sweep', 'BreathIndex', 'PeakSample', 'PeakTime_s',
                    'Class', 'ClassName', 'IsSigh', 'Reviewed'
                ])

                class_names = {-1: 'Unclassified', 0: 'Eupnea', 1: 'Sniffing', 2: 'Noise'}

                for meta in self._umap_breath_metadata:
                    reviewed = (meta['sweep'], meta['peak_sample']) in self._umap_reviewed_breaths
                    class_name = 'Sigh' if meta['is_sigh'] else class_names.get(meta['current_class'], 'Unknown')

                    writer.writerow([
                        meta['sweep'],
                        meta['breath_idx'],
                        meta['peak_sample'],
                        f"{meta['peak_time']:.3f}",
                        meta['current_class'],
                        class_name,
                        'Yes' if meta['is_sigh'] else 'No',
                        'Yes' if reviewed else 'No'
                    ])

            QMessageBox.information(self, "Export Complete",
                                  f"Classifications exported to:\n{filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Failed",
                               f"Failed to export classifications:\n{str(e)}")

    # ===== New Tab 2 Handler Methods =====

    def _mark_as_noise_tab2(self):
        """Mark selected breath as noise (exclude from analysis)."""
        self._apply_breath_classification(class_label=2, class_name='Noise', is_sigh=False)

    def _mark_as_none(self):
        """Clear classification for selected breath (reset to unclassified)."""
        if self._umap_selected_idx is None:
            return

        meta = self._umap_breath_metadata[self._umap_selected_idx]
        sweep = meta['sweep']
        peak_sample = meta['peak_sample']

        st = self.main_window.state

        print(f"\n===== Clearing Classification =====")
        print(f"Sweep: {sweep}, Peak: {peak_sample}")

        # Reset breath_type_class to unclassified (-1)
        all_peaks = st.all_peaks_by_sweep.get(sweep)
        if all_peaks is not None:
            peak_indices = all_peaks.get('indices', np.array([]))
            peak_mask = peak_indices == peak_sample

            if np.any(peak_mask):
                peak_pos = np.where(peak_mask)[0][0]

                if 'breath_type_class' in all_peaks and all_peaks['breath_type_class'] is not None:
                    all_peaks['breath_type_class'][peak_pos] = -1

                if 'eupnea_sniff_source' in all_peaks and all_peaks['eupnea_sniff_source'] is not None:
                    all_peaks['eupnea_sniff_source'][peak_pos] = 'auto'

        # Remove from sighs if it was a sigh
        if hasattr(st, 'sigh_by_sweep') and sweep in st.sigh_by_sweep:
            st.sigh_by_sweep[sweep] = st.sigh_by_sweep[sweep][st.sigh_by_sweep[sweep] != peak_sample]

        # Update metadata
        meta['current_class'] = -1
        meta['is_sigh'] = False

        # Track changes
        self._umap_changes_count += 1

        # Update scatter colors
        self._update_umap_scatter()

        # Update UI
        self._update_ui_state_tab2()

        # Refresh trace
        self._refresh_trace_tab2()

        # Refresh main window if on same sweep
        if self.main_window.current_sweep == sweep:
            self.main_window.plot_sweep()

        print("===== Classification Cleared =====\n")

    def _skip_to_next_point(self):
        """Skip to the next unclassified point without changing classification."""
        if self._umap_selected_idx is None or len(self._umap_breath_metadata) == 0:
            return

        n_points = len(self._umap_breath_metadata)
        start_idx = self._umap_selected_idx

        # Search for next unclassified point (wrapping around)
        for offset in range(1, n_points + 1):
            check_idx = (start_idx + offset) % n_points
            meta = self._umap_breath_metadata[check_idx]

            # Skip if already classified (and not sigh which is a special category)
            if meta['current_class'] == -1 or check_idx == start_idx:
                # Found an unclassified point or we've wrapped around
                self._select_umap_point(check_idx)
                break

    def _zoom_in_trace(self):
        """Zoom in on trace (narrower window) and update dropdown."""
        combo = self._window_size_combo_tab2
        current_idx = combo.currentIndex()
        # Zoom in means smaller window, which is lower index
        if current_idx > 0:
            combo.setCurrentIndex(current_idx - 1)
            # Refresh is triggered by currentTextChanged signal

    def _zoom_out_trace(self):
        """Zoom out on trace (wider window) and update dropdown."""
        combo = self._window_size_combo_tab2
        current_idx = combo.currentIndex()
        # Zoom out means larger window, which is higher index
        if current_idx < combo.count() - 1:
            combo.setCurrentIndex(current_idx + 1)
            # Refresh is triggered by currentTextChanged signal

    def _on_scatter_checkbox_changed(self):
        """Handle scatter plot checkbox changes to show/hide plots."""
        # Update the layout with currently visible scatter plots
        self._update_plots_layout()

    def _on_scatter_point_clicked(self, plot_item, spots, ev):
        """Handle click on scatter plot points for cross-selection."""
        if len(spots) == 0:
            return

        # Get clicked point index
        spot = spots[0]
        idx = spot.index()

        if idx is not None and idx < len(self._umap_breath_metadata):
            # Update selection
            self._umap_selected_idx = idx

            # Update all plots to highlight this point
            self._update_umap_scatter()

            # Update trace view
            meta = self._umap_breath_metadata[idx]
            self._update_trace_tab2(meta['sweep'], meta['breath_idx'], meta['peak_time'])

            # Update UI state
            self._update_ui_state_tab2()

            # Update info label
            class_names = {-1: 'Unclassified', 0: 'Eupnea', 1: 'Sniffing', 2: 'Noise'}
            class_name = 'Sigh' if meta['is_sigh'] else class_names.get(meta['current_class'], 'Unknown')
            self._info_label_tab2.setText(
                f"Sweep {meta['sweep']} | Time {meta['peak_time']:.2f}s | Class: {class_name}"
            )
