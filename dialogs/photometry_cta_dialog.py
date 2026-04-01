"""
Photometry CTA (Condition-Triggered Average) Dialog.

Provides a tabbed workspace for configuring and comparing multiple CTA analyses,
each with independent trigger source, time windows, metrics, and plots.
"""

from typing import Optional, List, Dict
from pathlib import Path
import numpy as np

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTabWidget, QToolButton,
    QWidget, QMessageBox, QFileDialog, QInputDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from viewmodels.cta_viewmodel import CTAViewModel
from core.domain.events import EventMarker

from dialogs.export_mixin import ExportMixin
from dialogs.cta_comparison_widget import CTAComparisonWidget


class PhotometryCTADialog(ExportMixin, QDialog):
    """
    Tabbed CTA workspace dialog.

    Each tab is a CTAComparisonWidget with its own trigger source,
    configuration, viewmodel, and plot area.
    """

    cta_generated = pyqtSignal()

    def __init__(
        self,
        parent=None,
        viewmodel: Optional[CTAViewModel] = None,
        markers: Optional[List[EventMarker]] = None,
        signals: Optional[Dict[str, np.ndarray]] = None,
        time_array: Optional[np.ndarray] = None,
        metric_labels: Optional[Dict[str, str]] = None,
        channel_colors: Optional[Dict[str, str]] = None,
        breath_data: Optional[Dict] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Photometry CTA Generator")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)
        self.setSizeGripEnabled(True)
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )

        # Store data for creating new tabs
        self._markers = markers or []
        self._signals = signals or {}
        self._time_array = time_array
        self._metric_labels = metric_labels or {}
        self._channel_colors = channel_colors or {}
        self._breath_data = breath_data
        self._first_viewmodel = viewmodel  # Reuse for first tab

        # Derive source file stem and export dir from parent's state
        self._source_stem = ""
        self._source_dir = ""
        if parent and hasattr(parent, 'state') and hasattr(parent.state, 'in_path') and parent.state.in_path:
            p = Path(parent.state.in_path)
            self._source_dir = str(p.parent)

            # Build a rich stem: {file_stem}.{exp_id}
            # e.g., "3_30_2026_B__recovered_photometry.exp0" or "3_30_2026_B.L_280671"
            stem_parts = [p.stem]
            st = parent.state

            # Add experiment index
            exp_idx = getattr(st, 'photometry_experiment_index', None)
            if exp_idx is not None:
                stem_parts.append(f"exp{exp_idx}")

            # Add animal ID if available
            animal_id = ""
            active_row = getattr(parent, '_active_master_list_row', None)
            if active_row is not None and hasattr(parent, '_master_file_list'):
                if active_row < len(parent._master_file_list):
                    animal_id = parent._master_file_list[active_row].get('animal_id', '') or ''
            if animal_id:
                stem_parts.append(animal_id)

            self._source_stem = ".".join(stem_parts)

        self._init_ui()
        self._apply_dark_theme()
        self.setup_export_menu()

        # Add initial tab
        self._add_tab(viewmodel=self._first_viewmodel)

    def _init_ui(self):
        """Build the dialog UI — tab widget + bottom bar."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Tab widget
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        self._tabs.tabBarDoubleClicked.connect(self._rename_tab)

        # "+" button as corner widget
        add_btn = QToolButton()
        add_btn.setText("+")
        add_btn.setToolTip("Add new CTA comparison tab")
        add_btn.setStyleSheet(
            "QToolButton { font-size: 16px; font-weight: bold; padding: 2px 10px; "
            "background: #1a6b1a; border: 1px solid #2a8a2a; border-radius: 3px; color: #ddd; }"
            "QToolButton:hover { background: #228b22; color: #fff; border-color: #33aa33; }"
        )
        add_btn.clicked.connect(lambda: self._add_tab())
        self._tabs.setCornerWidget(add_btn, Qt.Corner.TopRightCorner)

        layout.addWidget(self._tabs, 1)

        # Bottom bar
        bottom = QHBoxLayout()
        bottom.setSpacing(8)

        self._btn_export_all = QPushButton("Export All CSVs")
        self._btn_export_all.setToolTip("Export CSV from every tab into a folder")
        self._btn_export_all.setStyleSheet(
            "QPushButton { background-color: #1a4a6b; border: 1px solid #2a6a8a; }"
            "QPushButton:hover { background-color: #22628b; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._btn_export_all.clicked.connect(self._on_export_all)
        bottom.addWidget(self._btn_export_all)

        bottom.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(
            "QPushButton { background-color: #6b1a1a; border: 1px solid #8a2a2a; }"
            "QPushButton:hover { background-color: #8b2222; border-color: #aa3333; }"
            "QPushButton:pressed { background-color: #4d0d0d; }"
        )
        btn_close.clicked.connect(self.close)
        bottom.addWidget(btn_close)

        layout.addLayout(bottom)

    # -------------------------------------------------------------------------
    # Tab Management
    # -------------------------------------------------------------------------

    def _add_tab(self, viewmodel: Optional[CTAViewModel] = None):
        """Add a new CTA comparison tab."""
        vm = viewmodel or CTAViewModel(self)

        has_peaks = bool(self._breath_data and self._breath_data.get('all_peaks', {}).get('indices') is not None)
        print(f"[CTA] _add_tab: breath_data={'present' if self._breath_data else 'None'}, has_peaks={has_peaks}")

        widget = CTAComparisonWidget(
            parent=self,
            viewmodel=vm,
            markers=self._markers,
            signals=self._signals,
            time_array=self._time_array,
            metric_labels=self._metric_labels,
            channel_colors=self._channel_colors,
            breath_data=self._breath_data,
            source_stem=self._source_stem,
        )
        widget._default_export_dir = self._source_dir
        widget.name_changed.connect(
            lambda name, w=widget: self._on_tab_name_changed(w, name)
        )
        widget.cta_generated.connect(self.cta_generated.emit)

        tab_name = widget.tab_name or f"CTA {self._tabs.count() + 1}"
        idx = self._tabs.addTab(widget, tab_name)
        self._tabs.setCurrentIndex(idx)

    def _close_tab(self, index: int):
        """Close a tab (minimum 1 tab enforced)."""
        if self._tabs.count() <= 1:
            QMessageBox.information(self, "Cannot Close", "At least one tab must remain.")
            return
        widget = self._tabs.widget(index)
        self._tabs.removeTab(index)
        if widget:
            widget.deleteLater()

    def _rename_tab(self, index: int):
        """Rename a tab on double-click."""
        if index < 0:
            return
        current_name = self._tabs.tabText(index)
        new_name, ok = QInputDialog.getText(
            self, "Rename Tab", "Tab name:", text=current_name
        )
        if ok and new_name.strip():
            self._tabs.setTabText(index, new_name.strip())

    def _on_tab_name_changed(self, widget, name: str):
        """Handle tab name change from a comparison widget."""
        idx = self._tabs.indexOf(widget)
        if idx >= 0:
            self._tabs.setTabText(idx, name)

    # -------------------------------------------------------------------------
    # Export All
    # -------------------------------------------------------------------------

    def _on_export_all(self):
        """Export CSV from every tab into a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if not folder:
            return

        folder_path = Path(folder)
        exported = 0
        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            if not isinstance(widget, CTAComparisonWidget):
                continue
            vm = widget._viewmodel
            tab_name = self._tabs.tabText(i).replace(' ', '_').replace('/', '_')

            # Export per-tab CSV
            has_condition_data = bool(vm.condition_collections)
            has_main = vm.current_collection is not None
            if has_condition_data or has_main:
                stem = self._source_stem or "CTA"
                filepath = str(folder_path / f"{stem}_CTA_{tab_name}.csv")
                try:
                    vm.export_to_csv_wide(filepath)
                    exported += 1
                except Exception as e:
                    print(f"[CTA Export All] Error exporting tab {i}: {e}")

        if exported:
            QMessageBox.information(
                self, "Export Complete",
                f"Exported {exported} tab(s) to:\n{folder}"
            )
        else:
            QMessageBox.warning(self, "No Data", "No tabs have CTA data to export.")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_data(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
    ):
        """Set or update data for all tabs."""
        self._markers = markers
        self._signals = signals
        self._time_array = time_array
        self._metric_labels = metric_labels or {}

        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            if isinstance(widget, CTAComparisonWidget):
                widget.set_data(markers, signals, time_array, metric_labels)

    def refresh_data(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
        channel_colors: Optional[Dict[str, str]] = None,
    ):
        """Refresh data on all tabs."""
        self._markers = markers
        self._signals = signals
        self._time_array = time_array
        if metric_labels:
            self._metric_labels = metric_labels
        if channel_colors:
            self._channel_colors = channel_colors

        for i in range(self._tabs.count()):
            widget = self._tabs.widget(i)
            if isinstance(widget, CTAComparisonWidget):
                widget.set_data(markers, signals, time_array, metric_labels)

    # -------------------------------------------------------------------------
    # Theme
    # -------------------------------------------------------------------------

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #e0e0e0; }
            QTabWidget::pane {
                background-color: #1e1e1e; border: 1px solid #3a3a3a;
                border-top: none;
            }
            QTabBar::tab {
                background-color: #2a2a2a; color: #aaa; border: 1px solid #3a3a3a;
                border-bottom: none; padding: 6px 16px; margin-right: 2px;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #1e1e1e; color: #e0e0e0;
                border-bottom: 1px solid #1e1e1e;
            }
            QTabBar::tab:hover:!selected { background-color: #333; color: #ddd; }
            QTabBar::close-button {
                image: none; subcontrol-position: right;
                border: none; padding: 2px;
            }
            QTabBar::close-button:hover { background-color: #c44; border-radius: 2px; }
            QGroupBox {
                font-weight: bold; border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 8px; padding-top: 12px; background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #88aaff;
            }
            QGroupBox::indicator { width: 14px; height: 14px; }
            QGroupBox::indicator:unchecked {
                border: 1px solid #555; border-radius: 3px; background-color: #2a2a2a;
            }
            QGroupBox::indicator:checked {
                border: 1px solid #2a7fff; border-radius: 3px; background-color: #2a7fff;
            }
            QLabel { color: #e0e0e0; background-color: transparent; }
            QListWidget {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
            }
            QListWidget::item:selected { background-color: #2a7fff; color: white; }
            QTableWidget {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px;
                gridline-color: #3a3a3a;
            }
            QTableWidget::item:selected { background-color: transparent; color: #e0e0e0; }
            QHeaderView::section {
                background-color: #333; color: #ccc; border: 1px solid #3a3a3a;
                padding: 3px; font-size: 11px;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px;
            }
            QComboBox {
                background-color: #2a2a2a; color: #e0e0e0;
                border: 1px solid #3a3a3a; border-radius: 4px; padding: 4px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a; color: #e0e0e0;
                selection-background-color: #2a7fff;
            }
            QRadioButton { color: #e0e0e0; }
            QRadioButton::indicator {
                width: 14px; height: 14px; border: 1px solid #555;
                border-radius: 7px; background-color: #2a2a2a;
            }
            QRadioButton::indicator:checked { background-color: #2a7fff; border-color: #2a7fff; }
            QCheckBox { color: #e0e0e0; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border: 1px solid #555;
                border-radius: 3px; background-color: #2a2a2a;
            }
            QCheckBox::indicator:checked { background-color: #2a7fff; border-color: #2a7fff; }
            QPushButton {
                background-color: #3a3a3a; color: #e0e0e0;
                border: 1px solid #555; border-radius: 4px; padding: 6px 16px;
            }
            QPushButton:hover { background-color: #4a4a4a; border-color: #2a7fff; }
            QPushButton:pressed { background-color: #2a7fff; }
            QPushButton:disabled { background-color: #2a2a2a; color: #666; }
            QProgressBar {
                border: 1px solid #3a3a3a; border-radius: 4px;
                background-color: #2a2a2a; text-align: center;
            }
            QProgressBar::chunk { background-color: #2a7fff; border-radius: 3px; }
            QScrollArea { background-color: #1e1e1e; border: none; }
            QScrollArea > QWidget > QWidget { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; }
            QToolBar { background-color: #2d2d2d; border: none; spacing: 3px; padding: 2px; }
            QToolButton {
                background-color: #3d3d3d; border: 1px solid #4d4d4d;
                border-radius: 3px; padding: 4px; color: #d4d4d4;
            }
            QToolButton:hover { background-color: #4d4d4d; border: 1px solid #5d5d5d; }
            QToolButton:pressed { background-color: #5d5d5d; }
            QToolButton:checked { background-color: #2a7fff; border-color: #2a7fff; }
        """)
