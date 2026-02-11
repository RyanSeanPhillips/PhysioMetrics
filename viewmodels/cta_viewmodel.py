"""
CTA (Condition-Triggered Average) view model.

This module provides Qt integration for the CTA system,
exposing signals for UI updates and commands for user actions.
"""

from typing import List, Optional, Dict, Callable
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np

from core.domain.cta import CTAConfig, CTAResult, CTACollection
from core.domain.events import EventMarker
from core.services import CTAService


class CTAViewModel(QObject):
    """
    View model for CTA calculation and display.

    Provides Qt signals for UI binding and methods for calculating,
    previewing, and exporting CTAs.

    Signals:
        calculation_started: Emitted when calculation begins
        calculation_progress: Emitted during calculation (0-100)
        calculation_complete: Emitted when calculation finishes
        preview_ready: Emitted when preview data is ready
        export_complete: Emitted when export finishes (filepath)
        error_occurred: Emitted on error (error message)
    """

    # Signals
    calculation_started = pyqtSignal()
    calculation_progress = pyqtSignal(int)
    calculation_complete = pyqtSignal()
    preview_ready = pyqtSignal()
    export_complete = pyqtSignal(str)  # filepath
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, parent: Optional[QObject] = None):
        """
        Initialize the view model.

        Args:
            parent: Optional parent QObject
        """
        super().__init__(parent)

        self._service = CTAService()
        self._config = CTAConfig()
        self._current_collection: Optional[CTACollection] = None

        # Selected metrics for CTA
        self._selected_metrics: List[str] = []
        self._available_metrics: Dict[str, str] = {}  # key -> label

        # Selected marker categories
        self._selected_categories: List[str] = []

    @property
    def service(self) -> CTAService:
        """Get the underlying CTA service."""
        return self._service

    @property
    def config(self) -> CTAConfig:
        """Get the current CTA configuration."""
        return self._config

    @config.setter
    def config(self, value: CTAConfig) -> None:
        """Set the CTA configuration."""
        self._config = value

    @property
    def window_before(self) -> float:
        """Get window before event (seconds)."""
        return self._config.window_before

    @window_before.setter
    def window_before(self, value: float) -> None:
        """Set window before event (seconds)."""
        self._config.window_before = max(0.1, value)

    @property
    def window_after(self) -> float:
        """Get window after event (seconds)."""
        return self._config.window_after

    @window_after.setter
    def window_after(self, value: float) -> None:
        """Set window after event (seconds)."""
        self._config.window_after = max(0.1, value)

    @property
    def include_withdrawal(self) -> bool:
        """Get whether to include withdrawal CTAs."""
        return self._config.include_withdrawal

    @include_withdrawal.setter
    def include_withdrawal(self, value: bool) -> None:
        """Set whether to include withdrawal CTAs."""
        self._config.include_withdrawal = value

    @property
    def current_collection(self) -> Optional[CTACollection]:
        """Get the current CTA collection (result of last calculation)."""
        return self._current_collection

    @property
    def selected_metrics(self) -> List[str]:
        """Get selected metric keys."""
        return self._selected_metrics

    @selected_metrics.setter
    def selected_metrics(self, value: List[str]) -> None:
        """Set selected metric keys."""
        self._selected_metrics = list(value)

    @property
    def available_metrics(self) -> Dict[str, str]:
        """Get available metrics (key -> label)."""
        return self._available_metrics

    def set_available_metrics(self, metrics: Dict[str, str]) -> None:
        """
        Set available metrics for selection.

        Args:
            metrics: Dictionary of metric_key -> metric_label
        """
        self._available_metrics = dict(metrics)

    @property
    def selected_categories(self) -> List[str]:
        """Get selected marker categories."""
        return self._selected_categories

    @selected_categories.setter
    def selected_categories(self, value: List[str]) -> None:
        """Set selected marker categories."""
        self._selected_categories = list(value)

    def generate_preview(
        self,
        markers: List[EventMarker],
        signals: Dict[str, np.ndarray],
        time_array: np.ndarray,
        metric_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Generate CTA preview.

        Args:
            markers: List of markers to use
            signals: Dictionary of signals keyed by metric
            time_array: Time array for signals
            metric_labels: Optional metric labels
        """
        self.calculation_started.emit()

        try:
            # Filter markers by selected categories
            if self._selected_categories:
                filtered_markers = [
                    m for m in markers
                    if f"{m.category}:{m.label}" in self._selected_categories
                    or m.category in self._selected_categories
                ]
            else:
                filtered_markers = markers

            # Filter signals by selected metrics
            if self._selected_metrics:
                filtered_signals = {
                    k: v for k, v in signals.items()
                    if k in self._selected_metrics
                }
            else:
                filtered_signals = signals

            if not filtered_markers:
                self.error_occurred.emit("No markers selected for CTA")
                return

            if not filtered_signals:
                self.error_occurred.emit("No metrics selected for CTA")
                return

            # Calculate CTAs
            self._current_collection = self._service.calculate_for_markers(
                markers=filtered_markers,
                signals=filtered_signals,
                time_array=time_array,
                metric_labels=metric_labels or self._available_metrics,
                config=self._config,
                progress_callback=lambda p: self.calculation_progress.emit(p),
            )

            self.calculation_complete.emit()
            self.preview_ready.emit()

        except Exception as e:
            self.error_occurred.emit(f"CTA calculation failed: {str(e)}")

    def get_result(
        self,
        category: str,
        label: str,
        alignment: str,
        metric_key: str
    ) -> Optional[CTAResult]:
        """
        Get a specific CTA result.

        Args:
            category: Marker category
            label: Marker label
            alignment: 'onset' or 'withdrawal'
            metric_key: Metric key

        Returns:
            CTAResult or None
        """
        if self._current_collection is None:
            return None
        return self._current_collection.get_result(category, label, alignment, metric_key)

    def get_all_results(self) -> Dict[str, CTAResult]:
        """Get all CTA results from current collection."""
        if self._current_collection is None:
            return {}
        return self._current_collection.results

    def get_marker_types(self) -> List[str]:
        """
        Get unique marker types from current collection.

        Returns:
            List of 'category:label' strings
        """
        if self._current_collection is None:
            return []

        types = set()
        for result in self._current_collection.results.values():
            types.add(f"{result.category}:{result.label}")
        return sorted(types)

    def export_to_csv(self, filepath: str, include_traces: bool = True) -> None:
        """
        Export CTA data to CSV.

        Args:
            filepath: Path to save CSV
            include_traces: Whether to include individual traces
        """
        if self._current_collection is None:
            self.error_occurred.emit("No CTA data to export")
            return

        try:
            self._service.export_to_csv(
                self._current_collection,
                filepath,
                include_individual_traces=include_traces,
            )
            self.export_complete.emit(filepath)
        except Exception as e:
            self.error_occurred.emit(f"CSV export failed: {str(e)}")

    def export_to_npz(self, filepath: str) -> None:
        """
        Export CTA data to NPZ format.

        Args:
            filepath: Path to save NPZ
        """
        if self._current_collection is None:
            self.error_occurred.emit("No CTA data to export")
            return

        try:
            npz_data = self._service.to_npz_dict(self._current_collection)
            np.savez_compressed(filepath, **npz_data)
            self.export_complete.emit(filepath)
        except Exception as e:
            self.error_occurred.emit(f"NPZ export failed: {str(e)}")

    def load_from_npz(self, filepath: str) -> bool:
        """
        Load CTA data from NPZ file.

        Args:
            filepath: Path to NPZ file

        Returns:
            True if loaded successfully
        """
        try:
            data = dict(np.load(filepath, allow_pickle=True))
            collection = self._service.from_npz_dict(data)
            if collection is not None:
                self._current_collection = collection
                self.preview_ready.emit()
                return True
            return False
        except Exception as e:
            self.error_occurred.emit(f"NPZ load failed: {str(e)}")
            return False

    def clear(self) -> None:
        """Clear current CTA data."""
        self._current_collection = None
