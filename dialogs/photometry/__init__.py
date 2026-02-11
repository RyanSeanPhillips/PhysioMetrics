"""
Photometry Import Dialog Components

This package contains the refactored photometry import dialog:
- DataAssemblyWidget: File loading, column mapping, experiment configuration
- ExperimentPlotter: PyQtGraph plotting for photometry experiments
- PhotometryChooserDialog: Quick-access dialog when NPZ already exists
- ProcessingWidget: (Legacy) Tab 2 - kept for reference during refactoring
"""

from .data_assembly_widget import DataAssemblyWidget
from .experiment_plotter import ExperimentPlotter
from .chooser_dialog import PhotometryChooserDialog, show_photometry_chooser
from .processing_widget import ProcessingWidget

__all__ = [
    'DataAssemblyWidget',
    'ExperimentPlotter',
    'PhotometryChooserDialog',
    'show_photometry_chooser',
    'ProcessingWidget'
]
