"""
ClassifierViewModel — Qt bridge for classifier operations.

Wraps ClassifierService with pyqtSignals so the UI can react to
classifier changes without the service knowing about Qt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.services import classifier_service as svc


class ClassifierViewModel(QObject):
    """ViewModel for ML classifier selection and prediction.

    Signals:
        models_loaded: Emitted when ML models are loaded, with available algorithms dict.
        predictions_changed: Emitted when labels are updated (classifier switch or new predictions).
        classifier_switched: Emitted with (tier, algorithm) when user switches classifier.
        status_message: Emitted with (message, duration_ms) for status bar.
    """

    models_loaded = pyqtSignal(dict)        # {model1: [algos], model2: [...], model3: [...]}
    predictions_changed = pyqtSignal()       # labels updated, redraw needed
    classifier_switched = pyqtSignal(str, str)  # (tier, algorithm)
    status_message = pyqtSignal(str, int)    # (message, duration_ms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_models: Dict[str, Any] = {}

    @property
    def loaded_models(self) -> Dict[str, Any]:
        return self._loaded_models

    def load_models(self, models_dir: Path) -> int:
        """Load ML models from directory. Returns count loaded."""
        self._loaded_models = svc.load_models(models_dir)
        avail = svc.available_algorithms(self._loaded_models)
        self.models_loaded.emit(avail)
        self.status_message.emit(
            f"Loaded {len(self._loaded_models)} ML models", 3000
        )
        return len(self._loaded_models)

    def get_available_algorithms(self) -> Dict[str, List[str]]:
        return svc.available_algorithms(self._loaded_models)

    # ── Model 1: Breath vs Noise ─────────────────────────────────

    def switch_breath_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        get_peak_metrics_fn=None,
    ) -> bool:
        """Switch Model 1 classifier and update labels.

        Returns True if successful (predictions available or computed).
        """
        if algorithm != "threshold" and not self._loaded_models:
            self.status_message.emit("No ML models loaded", 5000)
            return False

        if algorithm not in ("threshold",):
            # Check model exists
            prefix = f"model1_{algorithm}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                self.status_message.emit(f"{algorithm} model not loaded", 5000)
                return False

            # Compute if needed
            self.status_message.emit(f"Computing {algorithm} predictions...", 0)
            svc.predict_breath_vs_noise(
                all_peaks_by_sweep, self._loaded_models, algorithm,
                get_peak_metrics_fn,
            )

        fallback = svc.apply_classifier_labels(all_peaks_by_sweep, algorithm)
        self.classifier_switched.emit("model1", algorithm)
        self.predictions_changed.emit()

        if fallback and algorithm != "threshold":
            self.status_message.emit(
                f"{algorithm} predictions unavailable, using threshold", 5000
            )
        return not fallback

    # ── Model 3: Eupnea/Sniff ────────────────────────────────────

    def switch_eupnea_sniff_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        active_classifier: str = "threshold",
        get_peak_metrics_fn=None,
    ):
        """Switch eupnea/sniff classifier."""
        if algorithm == "all_eupnea":
            svc.set_all_eupnea_sniff(all_peaks_by_sweep, 0, "all_eupnea")
        elif algorithm == "none":
            svc.clear_eupnea_sniff(all_peaks_by_sweep)
        elif algorithm == "gmm":
            svc.apply_eupnea_sniff_labels(all_peaks_by_sweep, "gmm")
        else:
            prefix = f"model3_{algorithm}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                self.status_message.emit(f"{algorithm} model not loaded", 5000)
                return
            svc.predict_eupnea_sniff(
                all_peaks_by_sweep, self._loaded_models, algorithm,
                active_classifier, get_peak_metrics_fn,
            )
            svc.apply_eupnea_sniff_labels(all_peaks_by_sweep, algorithm)

        self.classifier_switched.emit("model3", algorithm)
        self.predictions_changed.emit()

    # ── Model 2: Sigh ────────────────────────────────────────────

    def switch_sigh_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        sigh_by_sweep: Dict[int, Any],
        active_classifier: str = "threshold",
        get_peak_metrics_fn=None,
    ):
        """Switch sigh classifier."""
        if algorithm == "none":
            svc.clear_sighs(all_peaks_by_sweep, sigh_by_sweep)
        elif algorithm in ("manual",):
            svc.apply_sigh_labels(all_peaks_by_sweep, sigh_by_sweep, "manual")
        else:
            prefix = f"model2_{algorithm}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                self.status_message.emit(f"{algorithm} model not loaded", 5000)
                return
            svc.predict_sighs(
                all_peaks_by_sweep, self._loaded_models, algorithm,
                active_classifier, get_peak_metrics_fn,
            )
            svc.apply_sigh_labels(all_peaks_by_sweep, sigh_by_sweep, algorithm)

        self.classifier_switched.emit("model2", algorithm)
        self.predictions_changed.emit()

    # ── GMM ──────────────────────────────────────────────────────

    def run_gmm(
        self,
        feature_matrix,
        feature_keys: List[str],
        n_clusters: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Run GMM clustering. Returns result dict or None."""
        result = svc.run_gmm(feature_matrix, feature_keys, n_clusters)
        if result:
            self.status_message.emit("GMM clustering complete", 2000)
        return result
