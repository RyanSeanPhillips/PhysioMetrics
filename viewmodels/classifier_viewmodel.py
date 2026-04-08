"""
ClassifierViewModel — Qt bridge for classifier operations.

Wraps ClassifierService with pyqtSignals so the UI can react to
classifier changes without the service knowing about Qt.

Extended in Step 4B of the MVVM refactoring to replace ClassifierManager.
See _internal/docs/PLANNING/GMM_MVVM_EXTRACTION.md (same pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from core.services import classifier_service as svc


# ── Text-to-algorithm maps (moved from ClassifierManager) ───────

CLASSIFIER_MAP = {
    "Threshold": "threshold",
    "XGBoost": "xgboost",
    "Random Forest": "rf",
    "MLP": "mlp",
}

EUPNEA_SNIFF_MAP = {
    "GMM": "gmm",
    "XGBoost": "xgboost",
    "Random Forest": "rf",
    "MLP": "mlp",
    "All Eupnea": "all_eupnea",
    "None (Clear)": "none",
}

SIGH_MAP = {
    "Manual": "manual",
    "XGBoost": "xgboost",
    "Random Forest": "rf",
    "MLP": "mlp",
}


class ClassifierViewModel(QObject):
    """ViewModel for ML classifier selection and prediction.

    Signals:
        models_loaded: Emitted when ML models are loaded, with available algorithms dict.
        predictions_changed: Emitted when labels are updated (classifier switch or new predictions).
        classifier_switched: Emitted with (tier, algorithm) when user switches classifier.
        classifier_fallback: Emitted with (tier, default_text) when combo needs resetting.
        dropdown_availability_changed: Emitted with enable/disable map after model load.
        gmm_requested: Emitted when GMM clustering needs to run.
        status_message: Emitted with (message, duration_ms) for status bar.
    """

    models_loaded = pyqtSignal(dict)              # {model1: [algos], model2: [...], model3: [...]}
    predictions_changed = pyqtSignal()             # labels updated, redraw needed
    classifier_switched = pyqtSignal(str, str)     # (tier, algorithm)
    classifier_fallback = pyqtSignal(str, str)     # (tier, default_combo_text) — View resets combo
    dropdown_availability_changed = pyqtSignal(dict)  # {model1: {algo: bool}, ...}
    gmm_requested = pyqtSignal()                   # View triggers GMM run
    status_message = pyqtSignal(str, int)          # (message, duration_ms)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded_models: Dict[str, Any] = {}

        # Provider callbacks (set by main.py wiring)
        self._state_provider: Optional[Callable] = None
        self._get_processed_fn: Optional[Callable] = None
        self._get_peak_metrics_fn: Optional[Callable] = None

    # ── Provider setup ──────────────────────────────────────────

    def set_state_provider(self, fn: Callable):
        """Set callback that returns the current AppState."""
        self._state_provider = fn

    def set_get_processed_fn(self, fn: Callable):
        """Set callback: fn(channel, sweep_idx) -> ndarray (filtered signal)."""
        self._get_processed_fn = fn

    def set_get_peak_metrics_fn(self, fn: Callable):
        """Set callback: fn(sweep_idx, all_peaks_data) -> peak_metrics list."""
        self._get_peak_metrics_fn = fn

    def _state(self):
        return self._state_provider() if self._state_provider else None

    # ── Model loading ───────────────────────────────────────────

    @property
    def loaded_models(self) -> Dict[str, Any]:
        return self._loaded_models

    def load_models(self, models_dir: Path) -> int:
        """Load ML models from directory. Returns count loaded."""
        self._loaded_models = svc.load_models(models_dir)

        # Also store on state for backward compat
        state = self._state()
        if state is not None:
            state.loaded_ml_models = self._loaded_models

        avail = svc.available_algorithms(self._loaded_models)
        self.models_loaded.emit(avail)
        self._emit_dropdown_availability()
        self.status_message.emit(
            f"Loaded {len(self._loaded_models)} ML models", 3000
        )
        return len(self._loaded_models)

    def auto_load_ml_models(self, models_dir: Optional[Path]) -> int:
        """Load models from a directory (replaces settings-reading version).

        Args:
            models_dir: Path to models directory. If None, skips silently.

        Returns:
            Number of models loaded.
        """
        if models_dir is None or not Path(models_dir).exists():
            return 0
        return self.load_models(Path(models_dir))

    def get_available_algorithms(self) -> Dict[str, List[str]]:
        return svc.available_algorithms(self._loaded_models)

    def _emit_dropdown_availability(self):
        """Compute and emit which dropdown items should be enabled/disabled."""
        models = self._loaded_models
        avail = {
            'model1': {
                'threshold': True,
                'xgboost': any(k.startswith('model1_xgboost') for k in models),
                'rf': any(k.startswith('model1_rf') for k in models),
                'mlp': any(k.startswith('model1_mlp') for k in models),
            },
            'model3': {
                'none': True,
                'all_eupnea': True,
                'gmm': True,
                'xgboost': any(k.startswith('model3_xgboost') for k in models),
                'rf': any(k.startswith('model3_rf') for k in models),
                'mlp': any(k.startswith('model3_mlp') for k in models),
            },
            'model2': {
                'manual': True,
                'xgboost': any(k.startswith('model2_xgboost') for k in models),
                'rf': any(k.startswith('model2_rf') for k in models),
                'mlp': any(k.startswith('model2_mlp') for k in models),
            },
        }
        self.dropdown_availability_changed.emit(avail)

    # ── Model 1: Breath vs Noise ─────────────────────────────────

    def on_classifier_changed(self, text: str):
        """Handle classifier dropdown change (Model 1 — breath vs noise).

        Validates model availability, computes predictions if needed,
        applies labels, updates peaks/breaths, and emits signals.
        """
        state = self._state()
        if state is None:
            return

        new_classifier = CLASSIFIER_MAP.get(text, "threshold")

        if state.active_classifier == new_classifier:
            return

        state.active_classifier = new_classifier
        print(f"[Classifier] Switched to: {new_classifier}")

        # Validate ML model availability
        if new_classifier != 'threshold':
            if not self._loaded_models:
                print(f"[Classifier] ERROR: No ML models loaded!")
                self.status_message.emit("WARNING: No ML models loaded.", 5000)
                state.active_classifier = "threshold"
                self.classifier_fallback.emit("model1", "Threshold")
                return

            prefix = f"model1_{new_classifier}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                print(f"[Classifier] ERROR: No model matching {prefix} found!")
                self.status_message.emit(f"WARNING: {text} models not loaded.", 5000)
                state.active_classifier = "threshold"
                self.classifier_fallback.emit("model1", "Threshold")
                return

        # Apply classifier labels
        if state.analyze_chan and state.all_peaks_by_sweep:
            self.status_message.emit(f"Switching to {text} classifier...", 2000)

            # Compute predictions on-demand if needed
            if new_classifier in ('xgboost', 'rf', 'mlp'):
                svc.predict_breath_vs_noise(
                    state.all_peaks_by_sweep,
                    self._loaded_models,
                    new_classifier,
                    self._get_peak_metrics_fn,
                )

            fallback = svc.apply_classifier_labels(state.all_peaks_by_sweep, new_classifier)

            # Update peaks_by_sweep and breath_by_sweep
            svc.update_peaks_and_breaths_from_labels(state, self._get_processed_fn)

            if fallback and new_classifier != 'threshold':
                self.status_message.emit(
                    f"WARNING: {new_classifier.upper()} predictions failed. Using threshold.", 5000
                )
            else:
                self.status_message.emit(
                    f"Switched to {new_classifier.upper()} classifier.", 3000
                )

            self.classifier_switched.emit("model1", new_classifier)
            self.predictions_changed.emit()
        else:
            self.status_message.emit(
                f"Classifier set to {text}. Run peak detection to see results.", 3000
            )

    def switch_breath_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        get_peak_metrics_fn=None,
    ) -> bool:
        """Switch Model 1 classifier and update labels (direct API).

        Returns True if successful (predictions available or computed).
        """
        if algorithm != "threshold" and not self._loaded_models:
            self.status_message.emit("No ML models loaded", 5000)
            return False

        if algorithm not in ("threshold",):
            prefix = f"model1_{algorithm}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                self.status_message.emit(f"{algorithm} model not loaded", 5000)
                return False

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

    def on_eupnea_sniff_classifier_changed(self, text: str):
        """Handle eupnea/sniff dropdown change (Model 3).

        Validates, computes predictions, applies labels, rebuilds regions.
        """
        import core.gmm_clustering as gmm_clustering

        state = self._state()
        if state is None:
            return

        new_classifier = EUPNEA_SNIFF_MAP.get(text, "gmm")
        old_classifier = state.active_eupnea_sniff_classifier
        state.active_eupnea_sniff_classifier = new_classifier

        if old_classifier != new_classifier:
            print(f"[Eupnea/Sniff Classifier] Switched to: {new_classifier}")

        if new_classifier == 'all_eupnea':
            svc.set_all_eupnea_sniff(state.all_peaks_by_sweep, 0, "all_eupnea")
        elif new_classifier == 'none':
            svc.clear_eupnea_sniff(state.all_peaks_by_sweep)
        elif new_classifier == 'gmm':
            # Check if GMM has been run
            first_sweep_peaks = state.all_peaks_by_sweep.get(0, {})
            if 'gmm_class_ro' not in first_sweep_peaks or first_sweep_peaks['gmm_class_ro'] is None:
                print(f"[Eupnea/Sniff Classifier] GMM not run yet - requesting...")
                self.status_message.emit("Running GMM clustering...", 2000)
                self.gmm_requested.emit()
            svc.apply_eupnea_sniff_labels(state.all_peaks_by_sweep, "gmm")
        else:
            # ML classifier (xgboost, rf, mlp)
            prefix = f"model3_{new_classifier}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                print(f"[Eupnea/Sniff Classifier] ERROR: Model {prefix} not found!")
                self.status_message.emit(f"WARNING: {text} model not loaded.", 5000)
                state.active_eupnea_sniff_classifier = "gmm"
                self.classifier_fallback.emit("model3", "GMM")
                return

            # Compute Model 3 predictions on-demand
            svc.predict_eupnea_sniff(
                state.all_peaks_by_sweep,
                self._loaded_models,
                new_classifier,
                state.active_classifier,
                self._get_peak_metrics_fn,
            )
            svc.apply_eupnea_sniff_labels(state.all_peaks_by_sweep, new_classifier)

        # Rebuild regions for display
        try:
            gmm_clustering.build_eupnea_sniffing_regions(state, verbose=False)
        except Exception as e:
            print(f"[Eupnea/Sniff Classifier] Warning: Could not rebuild regions: {e}")

        self.classifier_switched.emit("model3", new_classifier)
        self.predictions_changed.emit()

    def switch_eupnea_sniff_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        active_classifier: str = "threshold",
        get_peak_metrics_fn=None,
    ):
        """Switch eupnea/sniff classifier (direct API)."""
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

    def on_sigh_classifier_changed(self, text: str):
        """Handle sigh dropdown change (Model 2)."""
        state = self._state()
        if state is None:
            return

        new_classifier = SIGH_MAP.get(text, "manual")
        old_classifier = state.active_sigh_classifier
        state.active_sigh_classifier = new_classifier

        if old_classifier != new_classifier:
            print(f"[Sigh Classifier] Switched to: {new_classifier}")

        if new_classifier == 'manual':
            svc.apply_sigh_labels(
                state.all_peaks_by_sweep, state.sigh_by_sweep, "manual"
            )
        elif new_classifier in ('xgboost', 'rf', 'mlp'):
            prefix = f"model2_{new_classifier}"
            if not any(k.startswith(prefix) for k in self._loaded_models):
                print(f"[Sigh Classifier] ERROR: Model {prefix} not found!")
                self.status_message.emit(f"WARNING: {text} model not loaded.", 5000)
                state.active_sigh_classifier = "manual"
                self.classifier_fallback.emit("model2", "Manual")
                return

            svc.predict_sighs(
                state.all_peaks_by_sweep,
                self._loaded_models,
                new_classifier,
                state.active_classifier,
                self._get_peak_metrics_fn,
            )
            svc.apply_sigh_labels(
                state.all_peaks_by_sweep, state.sigh_by_sweep, new_classifier
            )

        self.classifier_switched.emit("model2", new_classifier)
        self.predictions_changed.emit()

    def switch_sigh_classifier(
        self,
        algorithm: str,
        all_peaks_by_sweep: Dict[int, Dict],
        sigh_by_sweep: Dict[int, Any],
        active_classifier: str = "threshold",
        get_peak_metrics_fn=None,
    ):
        """Switch sigh classifier (direct API)."""
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

    # ── Peaks/Breath Update ──────────────────────────────────────

    def update_displayed_peaks_from_classifier(self):
        """Update peaks_by_sweep and breath_by_sweep from active classifier labels."""
        state = self._state()
        if state is None:
            return

        svc.update_peaks_and_breaths_from_labels(state, self._get_processed_fn)

    # ── Precomputation ───────────────────────────────────────────

    def precompute_remaining_classifiers(self):
        """Precompute predictions for all available Model 1 classifiers."""
        state = self._state()
        if state is None:
            return

        if not self._loaded_models or not state.all_peaks_by_sweep:
            return

        computed = svc.precompute_remaining_classifiers(
            state.all_peaks_by_sweep,
            self._loaded_models,
            self._get_peak_metrics_fn,
        )

        if computed:
            self.status_message.emit("Classifier precomputation complete", 2000)

    # ── Eupnea/Sniff/Sigh helpers (delegate to service) ─────────

    def update_eupnea_sniff_from_classifier(self):
        """Copy active eupnea/sniff predictions to breath_type_class."""
        state = self._state()
        if state is None:
            return
        svc.apply_eupnea_sniff_labels(
            state.all_peaks_by_sweep,
            state.active_eupnea_sniff_classifier,
        )

    def set_all_breaths_eupnea_sniff_class(self, class_value: int):
        """Set all breaths to a specific eupnea/sniff class."""
        state = self._state()
        if state is None:
            return
        source = state.active_eupnea_sniff_classifier
        svc.set_all_eupnea_sniff(state.all_peaks_by_sweep, class_value, source)

    def clear_all_eupnea_sniff_labels(self):
        """Clear all eupnea/sniff labels."""
        state = self._state()
        if state is None:
            return
        svc.clear_eupnea_sniff(state.all_peaks_by_sweep)

    def update_sigh_from_classifier(self):
        """Copy active sigh predictions to sigh_class."""
        state = self._state()
        if state is None:
            return
        svc.apply_sigh_labels(
            state.all_peaks_by_sweep,
            state.sigh_by_sweep,
            state.active_sigh_classifier,
        )

    def clear_all_sigh_labels(self):
        """Clear all sigh labels."""
        state = self._state()
        if state is None:
            return
        svc.clear_sighs(state.all_peaks_by_sweep, state.sigh_by_sweep)
