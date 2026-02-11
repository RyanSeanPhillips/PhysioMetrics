"""
Photometry Chooser Dialog

When a user tries to load photometry data and an NPZ file already exists,
this dialog lets them choose to:
1. Modify the data (opens the full import dialog)
2. Load a specific experiment directly into the app

This provides a streamlined workflow for returning to previously processed data.
"""

from pathlib import Path
from typing import Optional, Dict, Callable

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QSizePolicy, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class PhotometryChooserDialog(QDialog):
    """
    Dialog for choosing how to handle photometry data when an NPZ already exists.

    Shows options:
    - Modify Data: Opens the full import dialog for editing
    - Load Experiment N (Animal ID): Loads that experiment directly
    """

    # Result codes
    CANCELLED = 0
    MODIFY = 1
    LOAD_EXPERIMENT = 2
    CREATE_NEW = 3  # No NPZ exists, create new
    REPROCESS = 4  # Reprocess from raw data files

    def __init__(
        self,
        parent=None,
        npz_info: Optional[Dict] = None,
        raw_path: Optional[Path] = None
    ):
        """
        Args:
            parent: Parent widget
            npz_info: Dict from get_npz_experiment_info() with experiment metadata
            raw_path: Path to raw FP_data file (for display)
        """
        super().__init__(parent)
        self.npz_info = npz_info
        self.raw_path = raw_path
        self.selected_experiment = None
        self.result_code = self.CANCELLED

        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        self.setWindowTitle("Load Photometry Data")
        self.setModal(True)
        self.setMinimumWidth(400)

        # Dark theme styling
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #d4d4d4;
            }
            QPushButton {
                background-color: #0e639c;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 13px;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
            QPushButton:pressed {
                background-color: #0d5a8c;
            }
            QPushButton#modifyBtn {
                background-color: #3c3c3c;
                border: 1px solid #555555;
            }
            QPushButton#modifyBtn:hover {
                background-color: #4a4a4a;
            }
            QPushButton#experimentBtn {
                background-color: #2d7d46;
            }
            QPushButton#experimentBtn:hover {
                background-color: #38a058;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QLabel("Processed photometry data found")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: #569cd6;")
        layout.addWidget(header)

        # File info
        if self.npz_info:
            npz_path = Path(self.npz_info.get('npz_path', ''))
            info_text = f"File: {npz_path.name}"
            n_exp = self.npz_info.get('n_experiments', 1)
            info_text += f"\nExperiments: {n_exp}"

            info_label = QLabel(info_text)
            info_label.setStyleSheet("color: #888888; font-size: 11px;")
            layout.addWidget(info_label)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3e3e42;")
        layout.addWidget(sep)

        # Modify button
        modify_label = QLabel("Edit processing settings:")
        modify_label.setStyleSheet("color: #999999; font-size: 11px; margin-top: 8px;")
        layout.addWidget(modify_label)

        modify_btn = QPushButton("Modify Data")
        modify_btn.setObjectName("modifyBtn")
        modify_btn.setToolTip("Open the import dialog to change processing settings")
        modify_btn.clicked.connect(self._on_modify)
        layout.addWidget(modify_btn)

        # Reprocess button
        reprocess_btn = QPushButton("Reprocess from Raw Files")
        reprocess_btn.setObjectName("modifyBtn")  # Same style as modify
        reprocess_btn.setToolTip("Discard saved NPZ and reprocess from original CSV files")
        reprocess_btn.clicked.connect(self._on_reprocess)
        layout.addWidget(reprocess_btn)

        # Experiment buttons section
        if self.npz_info and self.npz_info.get('experiments'):
            exp_label = QLabel("Or load an experiment directly:")
            exp_label.setStyleSheet("color: #999999; font-size: 11px; margin-top: 16px;")
            layout.addWidget(exp_label)

            for exp in self.npz_info['experiments']:
                exp_idx = exp['index']
                animal_id = exp.get('animal_id', '').strip()
                fibers = exp.get('fibers', [])

                # Build button text
                if animal_id:
                    btn_text = f"Load Experiment {exp_idx + 1}: {animal_id}"
                else:
                    btn_text = f"Load Experiment {exp_idx + 1}"

                # Add fiber info as tooltip
                tooltip = f"Fibers: {', '.join(fibers)}" if fibers else ""

                btn = QPushButton(btn_text)
                btn.setObjectName("experimentBtn")
                btn.setToolTip(tooltip)
                btn.clicked.connect(lambda checked, idx=exp_idx: self._on_load_experiment(idx))
                layout.addWidget(btn)

        # Spacer
        layout.addStretch()

        # Cancel button
        cancel_layout = QHBoxLayout()
        cancel_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888888;
                border: 1px solid #555555;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #333333;
                color: #cccccc;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        cancel_layout.addWidget(cancel_btn)

        layout.addLayout(cancel_layout)

    def _on_modify(self):
        """Handle Modify button click."""
        self.result_code = self.MODIFY
        self.accept()

    def _on_reprocess(self):
        """Handle Reprocess button click."""
        self.result_code = self.REPROCESS
        self.accept()

    def _on_load_experiment(self, exp_idx: int):
        """Handle experiment load button click."""
        self.selected_experiment = exp_idx
        self.result_code = self.LOAD_EXPERIMENT
        self.accept()

    def get_result(self) -> tuple:
        """
        Get the dialog result.

        Returns:
            Tuple of (result_code, selected_experiment_index or None)
        """
        return (self.result_code, self.selected_experiment)


def show_photometry_chooser(
    parent=None,
    file_path: Path = None
) -> tuple:
    """
    Show the photometry chooser dialog if an NPZ exists.

    Args:
        parent: Parent widget
        file_path: Path to FP_data CSV or NPZ file

    Returns:
        Tuple of (action, data) where:
        - action is one of: 'cancel', 'modify', 'load', 'create_new', 'reprocess'
        - data is the experiment index for 'load', npz_path for 'modify', or None otherwise
    """
    from core import photometry

    # Check for existing NPZ
    npz_path = photometry.find_existing_photometry_npz(file_path)

    if npz_path is None:
        # No NPZ exists - go straight to import dialog
        return ('create_new', None)

    # Get experiment info from NPZ
    npz_info = photometry.get_npz_experiment_info(npz_path)

    if npz_info is None:
        # Couldn't read NPZ - go to import dialog
        return ('create_new', None)

    # Show chooser dialog
    dialog = PhotometryChooserDialog(
        parent=parent,
        npz_info=npz_info,
        raw_path=file_path
    )

    result = dialog.exec()

    if result == QDialog.DialogCode.Accepted:
        code, exp_idx = dialog.get_result()
        if code == PhotometryChooserDialog.MODIFY:
            return ('modify', npz_path)
        elif code == PhotometryChooserDialog.LOAD_EXPERIMENT:
            return ('load', {'npz_path': npz_path, 'experiment_index': exp_idx})
        elif code == PhotometryChooserDialog.REPROCESS:
            return ('reprocess', None)

    return ('cancel', None)
