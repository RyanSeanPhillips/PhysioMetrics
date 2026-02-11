"""
Processing Widget - Tab 2 of Photometry Import Dialog

Handles:
- Load NPZ files or receive data from Tab 1
- dF/F computation configuration (method, detrending, filtering)
- Processed signal preview with intermediates
- Load into main application
"""

import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.uic import loadUi

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from core import photometry


class ProcessingWidget(QWidget):
    """
    Processing tab for photometry import.

    Signals:
        data_ready: Emitted when processed data is ready for main app
    """

    data_ready = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Load UI from .ui file
        ui_path = Path(__file__).parent.parent.parent / "ui" / "photometry_processing.ui"
        if ui_path.exists():
            loadUi(ui_path, self)
        else:
            self._create_fallback_ui()

        # Data storage
        self._raw_data: Optional[Dict] = None  # From Tab 1 or NPZ
        self._npz_path: Optional[Path] = None
        self._processed_results: Dict = {}  # Stores dF/F results per fiber

        # Setup matplotlib canvas
        self._setup_matplotlib()

        # Connect signals
        self._setup_connections()

    def _create_fallback_ui(self):
        """Create minimal UI if .ui file not found."""
        from PyQt6.QtWidgets import QLabel
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Error: photometry_processing.ui not found"))

    def _setup_matplotlib(self):
        """Setup matplotlib figure and canvas."""
        self.figure = Figure(figsize=(10, 6), dpi=100, facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)

        # Add canvas to container
        if hasattr(self, 'canvas_container'):
            layout = QVBoxLayout(self.canvas_container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.canvas)

        # Add navigation toolbar
        if hasattr(self, 'toolbar_container'):
            toolbar_layout = QHBoxLayout(self.toolbar_container)
            toolbar_layout.setContentsMargins(0, 0, 0, 0)
            self.nav_toolbar = NavigationToolbar(self.canvas, self)
            self.nav_toolbar.setStyleSheet("""
                QToolBar { background: transparent; border: none; }
                QToolButton { background: transparent; color: #888888; border: none; padding: 4px; }
                QToolButton:hover { background-color: #3e3e42; color: #ffffff; }
            """)
            toolbar_layout.addWidget(self.nav_toolbar)

    def _setup_connections(self):
        """Connect UI signals to handlers."""
        # Browse NPZ button
        if hasattr(self, 'btn_browse_npz'):
            self.btn_browse_npz.clicked.connect(self._browse_npz)

        # dF/F method change
        if hasattr(self, 'combo_dff_method'):
            self.combo_dff_method.currentIndexChanged.connect(self._on_settings_changed)
            self.combo_dff_method.currentIndexChanged.connect(self._update_method_description)

        # Detrend method change
        if hasattr(self, 'combo_detrend_method'):
            self.combo_detrend_method.currentIndexChanged.connect(self._on_settings_changed)

        # Fit range controls
        if hasattr(self, 'chk_use_fit_range'):
            self.chk_use_fit_range.stateChanged.connect(self._on_settings_changed)
        if hasattr(self, 'spin_fit_start'):
            self.spin_fit_start.valueChanged.connect(self._on_settings_changed)
        if hasattr(self, 'spin_fit_end'):
            self.spin_fit_end.valueChanged.connect(self._on_settings_changed)

        # Filtering controls
        if hasattr(self, 'chk_lowpass'):
            self.chk_lowpass.stateChanged.connect(self._on_settings_changed)
        if hasattr(self, 'spin_lowpass_hz'):
            self.spin_lowpass_hz.valueChanged.connect(self._on_settings_changed)
        if hasattr(self, 'spin_exclude_start'):
            self.spin_exclude_start.valueChanged.connect(self._on_settings_changed)

        # Display options
        if hasattr(self, 'chk_show_intermediates'):
            self.chk_show_intermediates.stateChanged.connect(self._update_preview)

        # Update preview button
        if hasattr(self, 'btn_update_preview'):
            self.btn_update_preview.clicked.connect(self._update_preview)

        # Cancel button
        if hasattr(self, 'btn_cancel'):
            self.btn_cancel.clicked.connect(self._on_cancel)

        # Load into app button
        if hasattr(self, 'btn_load_into_app'):
            self.btn_load_into_app.clicked.connect(self._on_load_into_app)

    def _browse_npz(self):
        """Browse for NPZ file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photometry Data File",
            "", "NPZ Files (*.npz);;All Files (*)"
        )

        if file_path:
            self.load_npz(Path(file_path))

    def load_npz(self, path: Path):
        """Load photometry data from NPZ file."""
        try:
            data = np.load(path, allow_pickle=True)

            self._raw_data = {key: data[key] for key in data.files}
            self._npz_path = path

            if hasattr(self, 'npz_path_edit'):
                self.npz_path_edit.setText(str(path))

            self._update_data_info()
            self._update_preview()

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load photometry data:\n{str(e)}"
            )

    def set_raw_data(self, data_dict: Dict):
        """Set raw data from Tab 1 (when user switches tabs)."""
        if data_dict is None:
            return

        self._raw_data = data_dict
        self._npz_path = None

        if hasattr(self, 'npz_path_edit'):
            self.npz_path_edit.setText("(Data from Tab 1)")

        self._update_data_info()
        self._update_preview()

    def _update_data_info(self):
        """Update data info label."""
        if not hasattr(self, 'data_info_label'):
            return

        if self._raw_data is None:
            self.data_info_label.setText("No data loaded")
            return

        # Count fibers and samples
        fiber_data = self._raw_data.get('fiber_data', {})
        n_fibers = len(fiber_data)

        if n_fibers > 0:
            first_fiber = list(fiber_data.values())[0]
            n_samples = len(first_fiber.get('gcamp', []))
            self.data_info_label.setText(
                f"{n_fibers} fiber(s), ~{n_samples:,} samples"
            )
        else:
            self.data_info_label.setText("Data loaded (checking structure...)")

    def _update_method_description(self):
        """Update method description based on selection."""
        if not hasattr(self, 'method_description') or not hasattr(self, 'combo_dff_method'):
            return

        method = self.combo_dff_method.currentText()
        if 'Fitted' in method:
            self.method_description.setText(
                "Fitted: Linear regression of isosbestic to GCaMP for motion correction"
            )
        else:
            self.method_description.setText(
                "Simple: Normalized subtraction (GCaMP - Iso) / Iso"
            )

    def _on_settings_changed(self):
        """Handle any processing settings change."""
        # Could auto-update preview or just wait for button click
        pass

    def _get_processing_params(self) -> Dict:
        """Get current processing parameters from UI."""
        params = {
            'method': 'fitted',
            'detrend_method': 'none',
            'lowpass_hz': None,
            'exclude_start_min': 0.0,
            'fit_start': 0.0,
            'fit_end': 0.0,
            'show_intermediates': False
        }

        if hasattr(self, 'combo_dff_method'):
            params['method'] = 'fitted' if 'Fitted' in self.combo_dff_method.currentText() else 'simple'

        if hasattr(self, 'combo_detrend_method'):
            detrend_text = self.combo_detrend_method.currentText().lower()
            if 'linear' in detrend_text:
                params['detrend_method'] = 'linear'
            elif 'biexp' in detrend_text:
                params['detrend_method'] = 'biexponential'
            elif 'exp' in detrend_text:
                params['detrend_method'] = 'exponential'
            else:
                params['detrend_method'] = 'none'

        if hasattr(self, 'chk_lowpass') and self.chk_lowpass.isChecked():
            if hasattr(self, 'spin_lowpass_hz'):
                params['lowpass_hz'] = self.spin_lowpass_hz.value()

        if hasattr(self, 'spin_exclude_start'):
            params['exclude_start_min'] = self.spin_exclude_start.value()

        if hasattr(self, 'chk_use_fit_range') and self.chk_use_fit_range.isChecked():
            if hasattr(self, 'spin_fit_start'):
                params['fit_start'] = self.spin_fit_start.value()
            if hasattr(self, 'spin_fit_end'):
                params['fit_end'] = self.spin_fit_end.value()

        if hasattr(self, 'chk_show_intermediates'):
            params['show_intermediates'] = self.chk_show_intermediates.isChecked()

        return params

    def _update_preview(self):
        """Update processed signal preview using core/photometry functions."""
        if self._raw_data is None:
            return

        self._show_progress("Processing signals...")

        try:
            params = self._get_processing_params()
            fiber_data = self._raw_data.get('fiber_data', {})

            if not fiber_data:
                self._show_no_data_message()
                return

            self.figure.clear()
            self._processed_results.clear()

            n_fibers = len(fiber_data)
            show_intermediates = params['show_intermediates']

            # Calculate number of subplots
            if show_intermediates:
                n_rows = n_fibers * 3  # Raw, dF/F raw, dF/F detrended per fiber
            else:
                n_rows = n_fibers

            for i, (fiber_col, data) in enumerate(fiber_data.items()):
                iso_time = data.get('iso_time', np.array([]))
                iso_signal = data.get('iso', np.array([]))
                gcamp_time = data.get('gcamp_time', np.array([]))
                gcamp_signal = data.get('gcamp', np.array([]))

                if len(iso_time) == 0 or len(gcamp_time) == 0:
                    continue

                # Compute dF/F using core/photometry functions
                t0 = time.perf_counter()
                common_time, dff, intermediates = photometry.compute_dff_full(
                    iso_time, iso_signal,
                    gcamp_time, gcamp_signal,
                    method=params['method'],
                    detrend_method=params['detrend_method'],
                    lowpass_hz=params['lowpass_hz'],
                    exclude_start_min=params['exclude_start_min'],
                    fit_start=params['fit_start'],
                    fit_end=params['fit_end'],
                    return_intermediates=True
                )
                print(f"[Timing] dF/F computation for {fiber_col}: {time.perf_counter() - t0:.3f}s")

                # Store results
                self._processed_results[fiber_col] = {
                    'time': common_time,
                    'dff': dff,
                    'intermediates': intermediates
                }

                # Plot
                if show_intermediates:
                    # Plot 1: Raw signals
                    ax1 = self.figure.add_subplot(n_rows, 1, i * 3 + 1)
                    ax1.set_facecolor('#1e1e1e')
                    ax1.plot(iso_time, iso_signal, color='#9370DB', label='Iso', linewidth=0.5)
                    ax1.plot(gcamp_time, gcamp_signal, color='#4ec9b0', label='GCaMP', linewidth=0.5)
                    ax1.set_ylabel(f'{fiber_col}\nRaw', color='#d4d4d4', fontsize=9)
                    ax1.tick_params(colors='#888888', labelsize=8)
                    ax1.legend(loc='upper right', fontsize=7)
                    self._style_axes(ax1)

                    # Plot 2: dF/F raw (before detrend)
                    ax2 = self.figure.add_subplot(n_rows, 1, i * 3 + 2)
                    ax2.set_facecolor('#1e1e1e')
                    if intermediates and intermediates.get('dff_raw') is not None:
                        ax2.plot(common_time, intermediates['dff_raw'], color='#dcdcaa', linewidth=0.5)
                        if intermediates.get('detrend_curve') is not None:
                            ax2.plot(common_time, intermediates['detrend_curve'],
                                    color='#ff6b6b', linewidth=1, linestyle='--', label='Trend')
                            ax2.legend(loc='upper right', fontsize=7)
                    ax2.set_ylabel('dF/F raw (%)', color='#d4d4d4', fontsize=9)
                    ax2.tick_params(colors='#888888', labelsize=8)
                    self._style_axes(ax2)

                    # Plot 3: dF/F final
                    ax3 = self.figure.add_subplot(n_rows, 1, i * 3 + 3)
                    ax3.set_facecolor('#1e1e1e')
                    ax3.plot(common_time, dff, color='#4ec9b0', linewidth=0.5)
                    ax3.set_ylabel('dF/F (%)', color='#d4d4d4', fontsize=9)
                    ax3.tick_params(colors='#888888', labelsize=8)
                    self._style_axes(ax3)

                    if i == n_fibers - 1:
                        ax3.set_xlabel('Time (min)', color='#d4d4d4')

                else:
                    # Just plot final dF/F
                    ax = self.figure.add_subplot(n_rows, 1, i + 1)
                    ax.set_facecolor('#1e1e1e')
                    ax.plot(common_time, dff, color='#4ec9b0', linewidth=0.5)
                    ax.set_ylabel(f'{fiber_col}\ndF/F (%)', color='#d4d4d4', fontsize=9)
                    ax.tick_params(colors='#888888', labelsize=8)
                    self._style_axes(ax)

                    if i == n_fibers - 1:
                        ax.set_xlabel('Time (min)', color='#d4d4d4')

            # Update fit parameters display
            self._update_fit_params_display()

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"[Photometry] Error in preview: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self._hide_progress()

    def _style_axes(self, ax):
        """Apply dark theme styling to axes."""
        for spine in ax.spines.values():
            spine.set_color('#3e3e42')

    def _show_no_data_message(self):
        """Show message when no data available."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.text(0.5, 0.5, "No fiber data available",
               ha='center', va='center', color='#888888', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        self._style_axes(ax)
        self.canvas.draw()

    def _update_fit_params_display(self):
        """Update fit parameters label with results."""
        if not hasattr(self, 'fit_params_label'):
            return

        if not self._processed_results:
            self.fit_params_label.setText("No fit performed yet")
            return

        lines = []
        for fiber_col, results in self._processed_results.items():
            intermediates = results.get('intermediates', {})
            fit_params = intermediates.get('fit_params', {}) if intermediates else {}

            if fit_params:
                lines.append(f"--- {fiber_col} ---")

                method = fit_params.get('method', 'unknown')
                if method == 'fitted':
                    lines.append(f"  Method: Fitted regression")
                    lines.append(f"  Slope: {fit_params.get('slope', 0):.4f}")
                    lines.append(f"  R²: {fit_params.get('r_squared', 0):.4f}")
                else:
                    lines.append(f"  Method: Simple subtraction")
                    lines.append(f"  Iso mean: {fit_params.get('iso_mean', 0):.2f}")

                detrend = fit_params.get('detrend_method', 'none')
                lines.append(f"  Detrend: {detrend}")

                if 'exp_tau' in fit_params:
                    lines.append(f"  Tau: {fit_params['exp_tau']:.2f} min")

        self.fit_params_label.setText("\n".join(lines) if lines else "No parameters")

    def _show_progress(self, message: str):
        """Show processing progress message."""
        if hasattr(self, 'processing_label'):
            self.processing_label.setText(message)
            self.processing_label.setVisible(True)
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

    def _hide_progress(self):
        """Hide processing progress message."""
        if hasattr(self, 'processing_label'):
            self.processing_label.setVisible(False)

    def _on_cancel(self):
        """Handle cancel button click."""
        # Find parent dialog and reject
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'reject'):
                parent.reject()
                return
            parent = parent.parent()

    def _on_load_into_app(self):
        """Handle load into app button click."""
        if not self._processed_results:
            QMessageBox.warning(
                self, "No Data",
                "Please process data first by clicking 'Update Preview'."
            )
            return

        # Prepare data for main application
        result_data = self._prepare_result_data()

        if result_data:
            self.data_ready.emit(result_data)

    def _prepare_result_data(self) -> Optional[Dict]:
        """Prepare processed data for main application."""
        if not self._processed_results:
            return None

        # Get first fiber's data (for single-fiber case)
        first_fiber = list(self._processed_results.keys())[0]
        first_result = self._processed_results[first_fiber]

        params = self._get_processing_params()

        return {
            'time': first_result['time'],
            'dff': first_result['dff'],
            'fiber_results': self._processed_results,
            'processing_params': params,
            'source_path': self._npz_path,
            'raw_data': self._raw_data
        }

    def get_result_data(self) -> Optional[Dict]:
        """Get result data (called by parent dialog)."""
        return self._prepare_result_data()
