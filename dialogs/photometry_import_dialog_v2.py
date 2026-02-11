"""
Photometry Import Dialog (Refactored v2)

Single-page dialog for importing and processing fiber photometry data.
All functionality is now in DataAssemblyWidget:
- File loading (FP_data, AI_data, timestamps)
- Column mapping
- Channel assignment to experiments
- dF/F computation and visualization
- Save to NPZ for later reopening

The dF/F computation uses core/photometry.py functions.
Visualization is handled by ExperimentPlotter.
"""

import sys
from pathlib import Path
from typing import Optional, Dict

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt

from .photometry import DataAssemblyWidget

from dialogs.export_mixin import ExportMixin


class PhotometryImportDialog(ExportMixin, QDialog):
    """
    Dialog for importing and processing fiber photometry data.

    This single-page dialog handles:
    - File selection (FP_data CSV, AI_data CSV, timestamps)
    - Column mapping (time, LED state, fiber columns)
    - Channel assignment to experiments
    - dF/F computation with live preview
    - Save to NPZ file for reopening

    The dialog can be opened with:
    - initial_path: Path to FP_data CSV file
    - photometry_npz_path: Path to previously saved NPZ file
    """

    data_ready = pyqtSignal(dict)

    def __init__(
        self,
        parent=None,
        initial_path: Optional[Path] = None,
        photometry_npz_path: Optional[Path] = None,
        initial_params: Optional[Dict] = None,
        current_experiment: Optional[Dict] = None,
        cached_photometry_data: Optional[Dict] = None
    ):
        super().__init__(parent)

        self.setWindowTitle("Import Photometry Data")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Enable minimize/maximize buttons (standard window controls)
        self.setWindowFlags(
            Qt.WindowType.Window |
            Qt.WindowType.WindowMinimizeButtonHint |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowCloseButtonHint
        )

        # Store initial parameters
        self._initial_path = initial_path
        self._npz_path = photometry_npz_path
        self._initial_params = initial_params
        self._current_experiment = current_experiment  # For edit mode (gear icon)
        self._cached_photometry_data = cached_photometry_data  # For instant reopening

        # Result data to return
        self._result_data: Optional[Dict] = None

        # Setup UI
        self._setup_ui()

        # Apply dark theme
        self._apply_dark_theme()

        # Enable dark title bar on Windows
        self._enable_dark_title_bar()
        self.setup_export_menu()

        # Connect signals
        self._setup_connections()

        # Handle initial data
        self._apply_initial_data()

        # Enable edit mode if current_experiment is provided
        if self._current_experiment:
            self._enable_edit_mode()

    def _setup_ui(self):
        """Setup the dialog UI - single page with DataAssemblyWidget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main content - DataAssemblyWidget
        # Buttons are inside the widget's scroll area (defined in UI file)
        self.data_assembly = DataAssemblyWidget(self)
        layout.addWidget(self.data_assembly)

    def _apply_dark_theme(self):
        """Apply dark theme stylesheet."""
        self.setStyleSheet("""
            QDialog, QWidget {
                background-color: #1e1e1e;
                color: #d4d4d4;
            }
            QGroupBox {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 8px;
                font-weight: bold;
                color: #d4d4d4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #d4d4d4;
            }
            QLabel {
                color: #d4d4d4;
                background-color: transparent;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QLineEdit:read-only {
                background-color: #2d2d30;
                color: #888888;
            }
            QLineEdit:focus {
                border: 1px solid #007acc;
            }
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 6px 12px;
                color: #d4d4d4;
            }
            QPushButton:hover {
                background-color: #4e4e52;
            }
            QPushButton:pressed {
                background-color: #094771;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px 8px;
                color: #d4d4d4;
            }
            QComboBox:hover {
                border: 1px solid #007acc;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d30;
                color: #d4d4d4;
                selection-background-color: #094771;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                padding: 4px;
                color: #d4d4d4;
            }
            QCheckBox {
                color: #d4d4d4;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #3e3e42;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #007acc;
            }
            QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3e3e42;
                gridline-color: #3e3e42;
                color: #d4d4d4;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #094771;
            }
            QHeaderView::section {
                background-color: #2d2d30;
                color: #d4d4d4;
                border: 1px solid #3e3e42;
                padding: 4px;
            }
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #1e1e1e;
                width: 12px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 4px;
                min-height: 30px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #787878;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 12px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 4px;
                min-width: 30px;
                margin: 2px;
            }
            QFrame {
                color: #d4d4d4;
            }
            QSplitter::handle {
                background-color: #3e3e42;
            }
            QSplitter::handle:hover {
                background-color: #007acc;
            }
        """)

    def _enable_dark_title_bar(self):
        """Enable dark title bar on Windows 10/11."""
        if sys.platform != 'win32':
            return

        try:
            import ctypes
            hwnd = int(self.winId())
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int)
            )
        except Exception:
            pass

    def _setup_connections(self):
        """Connect signals."""
        # When data is saved, track the path
        self.data_assembly.npz_saved.connect(self._on_npz_saved)

        # When user clicks "Save & Load", handle loading into the app
        self.data_assembly.load_into_app_requested.connect(self._on_load_into_app)

        # Note: Button connections are already made in DataAssemblyWidget.__init__
        # Don't connect them again here to avoid double save dialogs

    def _apply_initial_data(self):
        """Apply any initial data passed to the dialog."""
        if self._cached_photometry_data is not None:
            # Use cached data for instant opening (from gear icon)
            # Pass session state so dialog knows n_experiments, etc.
            session_state = None
            if self._current_experiment:
                session_state = {
                    'n_experiments': self._current_experiment.get('n_experiments', 1),
                    'experiment_index': self._current_experiment.get('experiment_index', 0),
                    'animal_id': self._current_experiment.get('animal_id', ''),
                }
            success = self.data_assembly.load_from_cached_data(
                self._cached_photometry_data,
                npz_path=self._npz_path,
                initial_params=self._initial_params,  # Restore dF/F params
                session_state=session_state,  # Pass n_experiments, etc.
            )
            if not success:
                print("[Photometry] Failed to load cached data")
        elif self._npz_path:
            # Load existing NPZ file into the dialog
            success = self.data_assembly.load_from_npz(self._npz_path)
            if not success:
                print(f"[Photometry] Failed to load NPZ: {self._npz_path}")
        elif self._initial_path:
            # Start with initial file
            self.data_assembly.file_paths['fp_data'] = self._initial_path

            # Also look for companion files (AI data, timestamps)
            from core import photometry
            companions = photometry.find_companion_files(self._initial_path)
            if companions.get('ai_data'):
                self.data_assembly.file_paths['ai_data'] = companions['ai_data']
            if companions.get('timestamps'):
                self.data_assembly.file_paths['timestamps'] = companions['timestamps']

            self.data_assembly._update_file_edits()
            self.data_assembly._load_and_preview_data()

    def _on_npz_saved(self, path: Path):
        """Handle NPZ save."""
        self._npz_path = path
        print(f"[Photometry] Data saved to: {path}")

    def _on_load_into_app(self, saved_path: Path):
        """Handle Save & Load - data was saved, now load into app."""
        # Check how many experiments are available
        n_experiments = 1
        if hasattr(self.data_assembly, 'spin_num_experiments'):
            n_experiments = self.data_assembly.spin_num_experiments.value()

        # If multiple experiments, prompt user to select which one with buttons
        exp_idx = getattr(self, '_current_exp_idx', 0)
        if n_experiments > 1:
            exp_idx = self._show_experiment_selection_dialog(n_experiments, exp_idx)
            if exp_idx is None:
                return  # User cancelled

        # Get processed data in format expected by main app
        result_data = self.data_assembly.get_data_for_main_app(exp_idx=exp_idx, npz_path=saved_path)

        if result_data:
            self._result_data = result_data
            self.data_ready.emit(result_data)
            self.accept()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Data",
                "Please load photometry data before loading into the app."
            )

    def _on_save_data(self):
        """Handle Save Data button click - save NPZ without loading into app."""
        saved_path = self.data_assembly.save_to_npz()
        if saved_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Data Saved",
                f"Photometry data saved to:\n{saved_path}"
            )

    def _on_save_and_load(self):
        """Handle Save & Load button click - save NPZ and load into app."""
        # First save the data
        saved_path = self.data_assembly.save_to_npz()
        if not saved_path:
            return

        # Then get result data and close dialog
        result_data = self.data_assembly.get_raw_data()

        if result_data:
            # Add the saved NPZ path to the result
            result_data['photometry_npz_path'] = saved_path
            self._result_data = result_data
            self.data_ready.emit(result_data)
            self.accept()
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Data",
                "Please load photometry data before loading into the app."
            )

    def _show_experiment_selection_dialog(self, n_experiments: int, default_idx: int = 0) -> Optional[int]:
        """Show a dialog with buttons to select which experiment to load.

        Args:
            n_experiments: Total number of experiments available
            default_idx: Default experiment index (0-indexed)

        Returns:
            Selected experiment index (0-indexed) or None if cancelled
        """
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                     QPushButton, QSizePolicy)
        from PyQt6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("Select Experiment")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        # Header label
        header = QLabel("Which experiment would you like to load?")
        header.setStyleSheet("font-size: 12px; font-weight: bold; color: #d4d4d4;")
        layout.addWidget(header)

        # Get animal IDs from the data assembly widget if available
        animal_ids = {}
        if hasattr(self.data_assembly, '_dff_controls'):
            for i in range(n_experiments):
                controls_key = f'exp_{i}'
                if controls_key in self.data_assembly._dff_controls:
                    controls = self.data_assembly._dff_controls[controls_key]
                    if 'animal_id_edit' in controls:
                        animal_ids[i] = controls['animal_id_edit'].text()

        # Create a button for each experiment
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)

        selected_idx = [None]  # Use list to allow modification in closure

        def make_click_handler(idx):
            def handler():
                selected_idx[0] = idx
                dialog.accept()
            return handler

        for i in range(n_experiments):
            btn = QPushButton()

            # Build button text with animal ID if available
            if i in animal_ids and animal_ids[i].strip():
                btn_text = f"Exp {i+1}\n{animal_ids[i]}"
            else:
                btn_text = f"Experiment {i+1}"

            btn.setText(btn_text)
            btn.setMinimumSize(100, 60)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            # Style the default button differently
            if i == default_idx:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #0e639c;
                        border: 2px solid #007acc;
                        border-radius: 6px;
                        padding: 8px 16px;
                        color: white;
                        font-weight: bold;
                        font-size: 11px;
                    }
                    QPushButton:hover { background-color: #1177bb; }
                    QPushButton:pressed { background-color: #094771; }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #3c3c3c;
                        border: 1px solid #3e3e42;
                        border-radius: 6px;
                        padding: 8px 16px;
                        color: #d4d4d4;
                        font-size: 11px;
                    }
                    QPushButton:hover { background-color: #4a4a4a; }
                    QPushButton:pressed { background-color: #2d2d30; }
                """)

            btn.clicked.connect(make_click_handler(i))
            buttons_layout.addWidget(btn)

        layout.addLayout(buttons_layout)

        # Cancel button
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #3c3c3c;
                border: 1px solid #3e3e42;
                border-radius: 4px;
                padding: 6px 20px;
                color: #888888;
            }
            QPushButton:hover { background-color: #4a4a4a; color: #d4d4d4; }
        """)
        cancel_btn.clicked.connect(dialog.reject)

        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()
        cancel_layout.addWidget(cancel_btn)
        cancel_layout.addStretch()
        layout.addLayout(cancel_layout)

        # Set dialog size based on number of experiments
        dialog.setMinimumWidth(max(300, 110 * n_experiments))

        # Apply dark theme to dialog
        dialog.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #d4d4d4;
            }
        """)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return selected_idx[0]
        return None

    def get_result_data(self) -> Optional[Dict]:
        """Get the processed result data."""
        return self._result_data

    def get_saved_path(self) -> Optional[Path]:
        """Get the path to saved NPZ file (if any)."""
        return self._npz_path

    def get_selected_files(self) -> Dict[str, Optional[Path]]:
        """Get the selected file paths."""
        return self.data_assembly.file_paths.copy()

    def _enable_edit_mode(self):
        """
        Enable edit mode when dialog is opened from gear icon.

        In edit mode:
        - Current experiment is tracked for reload
        - "Load Different Experiment" button is shown if n_experiments > 1
        """
        exp_info = self._current_experiment
        if not exp_info:
            return

        # Store current experiment for reload functionality
        self._current_exp_idx = exp_info.get('experiment_index', 0)
        n_experiments = exp_info.get('n_experiments', 1)

        # Pass edit mode info to DataAssemblyWidget
        self.data_assembly.set_edit_mode(self._current_exp_idx, n_experiments)

        # Connect the "load different" signal if widget supports it
        if hasattr(self.data_assembly, 'load_different_requested'):
            self.data_assembly.load_different_requested.connect(self._on_load_different)

    def _on_load_different(self):
        """Handle request to load a different experiment."""
        from dialogs.photometry.chooser_dialog import PhotometryChooserDialog
        from core import photometry

        if not self._npz_path:
            return

        # Get experiment info from NPZ
        npz_info = photometry.get_npz_experiment_info(self._npz_path)
        if not npz_info:
            return

        # Show chooser dialog
        dialog = PhotometryChooserDialog(
            parent=self,
            npz_info=npz_info,
            raw_path=None
        )

        result = dialog.exec()
        if result == QDialog.DialogCode.Accepted:
            code, exp_idx = dialog.get_result()
            if code == PhotometryChooserDialog.LOAD_EXPERIMENT and exp_idx is not None:
                # Load the selected experiment directly
                result_data = photometry.load_experiment_from_npz(self._npz_path, exp_idx)
                if result_data:
                    self._result_data = result_data
                    self.data_ready.emit(result_data)
                    self.accept()


def show_photometry_import_dialog(
    parent=None,
    initial_path: Optional[Path] = None
) -> Optional[Dict]:
    """
    Show the photometry import dialog and return result.

    Args:
        parent: Parent widget
        initial_path: Optional initial file path to load

    Returns:
        Dict with processed data if accepted, None if cancelled
    """
    dialog = PhotometryImportDialog(parent=parent, initial_path=initial_path)

    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.get_result_data()

    return None
