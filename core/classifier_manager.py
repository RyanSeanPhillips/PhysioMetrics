"""
ClassifierManager - Handles ML classifier selection and predictions.

Extracted from main.py to improve code maintainability.
Manages breath detection classifiers (Model 1), eupnea/sniff classifiers (Model 3),
and sigh classifiers (Model 2).
"""

from pathlib import Path
import numpy as np


class ClassifierManager:
    """Manages ML classifier operations for breath analysis."""

    def __init__(self, main_window):
        """
        Args:
            main_window: Reference to MainWindow for state and UI access
        """
        self.mw = main_window

    @property
    def state(self):
        return self.mw.state

    # =========================================================================
    # Dropdown Management
    # =========================================================================

    def update_classifier_dropdowns(self):
        """Update dropdown options based on loaded models."""
        models = self.state.loaded_ml_models or {}

        # Check which models are available (use prefix matching for accuracy suffix)
        has_model1_xgboost = any(k.startswith('model1_xgboost') for k in models.keys())
        has_model1_rf = any(k.startswith('model1_rf') for k in models.keys())
        has_model1_mlp = any(k.startswith('model1_mlp') for k in models.keys())

        has_model3_xgboost = any(k.startswith('model3_xgboost') for k in models.keys())
        has_model3_rf = any(k.startswith('model3_rf') for k in models.keys())
        has_model3_mlp = any(k.startswith('model3_mlp') for k in models.keys())

        has_model2_xgboost = any(k.startswith('model2_xgboost') for k in models.keys())
        has_model2_rf = any(k.startswith('model2_rf') for k in models.keys())
        has_model2_mlp = any(k.startswith('model2_mlp') for k in models.keys())

        # Update Model 1 (Breath Detection) dropdown
        model = self.mw.peak_detec_combo.model()
        model.item(0).setEnabled(True)  # Threshold always available
        model.item(1).setEnabled(has_model1_xgboost)
        model.item(2).setEnabled(has_model1_rf)
        model.item(3).setEnabled(has_model1_mlp)

        # Update Model 3 (Eupnea/Sniff) dropdown
        model = self.mw.eup_sniff_combo.model()
        model.item(0).setEnabled(True)   # "None (Clear)" always available
        model.item(1).setEnabled(True)   # "All Eupnea" always available
        model.item(2).setEnabled(True)   # "GMM" always available
        model.item(3).setEnabled(has_model3_xgboost)
        model.item(4).setEnabled(has_model3_rf)
        model.item(5).setEnabled(has_model3_mlp)

        # Update Model 2 (Sigh) dropdown
        model = self.mw.digh_combo.model()
        model.item(0).setEnabled(True)  # Manual always available
        model.item(1).setEnabled(has_model2_xgboost)
        model.item(2).setEnabled(has_model2_rf)
        model.item(3).setEnabled(has_model2_mlp)

        # If current selection is disabled, fall back to default
        self.fallback_disabled_classifiers()

    def fallback_disabled_classifiers(self):
        """Check if current classifier selections are disabled and fall back to defaults."""
        # Model 1 (Breath Detection)
        current_text = self.mw.peak_detec_combo.currentText()
        current_index = self.mw.peak_detec_combo.currentIndex()
        if current_index >= 0:
            model = self.mw.peak_detec_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 1 classifier '{current_text}' not available, falling back to Threshold")
                self.mw.peak_detec_combo.setCurrentText("Threshold")
                self.state.active_classifier = "threshold"

        # Model 3 (Eupnea/Sniff)
        current_text = self.mw.eup_sniff_combo.currentText()
        current_index = self.mw.eup_sniff_combo.currentIndex()
        if current_index >= 0:
            model = self.mw.eup_sniff_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 3 classifier '{current_text}' not available, falling back to GMM")
                self.mw.eup_sniff_combo.setCurrentText("GMM")
                self.state.active_eupnea_sniff_classifier = "gmm"

        # Model 2 (Sigh)
        current_text = self.mw.digh_combo.currentText()
        current_index = self.mw.digh_combo.currentIndex()
        if current_index >= 0:
            model = self.mw.digh_combo.model()
            if not model.item(current_index).isEnabled():
                print(f"[Dropdown Update] Model 2 classifier '{current_text}' not available, falling back to Manual")
                self.mw.digh_combo.setCurrentText("Manual")
                self.state.active_sigh_classifier = "manual"

    # =========================================================================
    # Model Loading
    # =========================================================================

    def auto_load_ml_models_on_startup(self):
        """Silently load ML models from last used directory on startup."""
        import core.ml_prediction as ml_prediction

        # Get last used models directory from settings
        last_models_dir = self.mw.settings.value("ml_models_path", None)

        if not last_models_dir:
            print("[Auto-load] No saved models directory")
            return

        models_path = Path(last_models_dir)

        if not models_path.exists() or not models_path.is_dir():
            print(f"[Auto-load] Saved models directory no longer exists: {models_path}")
            return

        try:
            model_files = list(models_path.glob("model*.pkl"))

            if not model_files:
                print(f"[Auto-load] No model files found in: {models_path}")
                return

            loaded_models = {}
            for model_file in model_files:
                try:
                    model, metadata = ml_prediction.load_model(model_file)
                    model_key = model_file.stem
                    loaded_models[model_key] = {
                        'model': model,
                        'metadata': metadata,
                        'path': str(model_file)
                    }
                except Exception as e:
                    print(f"[Auto-load] Warning: Failed to load {model_file.name}: {e}")

            if loaded_models:
                self.state.loaded_ml_models = loaded_models
                self.update_classifier_dropdowns()
                print(f"[Auto-load] Successfully loaded {len(loaded_models)} models from {models_path}")
                self.mw.statusBar().showMessage(f"Auto-loaded {len(loaded_models)} ML models", 3000)
            else:
                print(f"[Auto-load] Failed to load any models from {models_path}")

        except Exception as e:
            print(f"[Auto-load] Error loading models: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Model 1: Breath Detection Classifier
    # =========================================================================

    def on_classifier_changed(self, text: str):
        """Handle classifier selection change from main window dropdown."""
        classifier_map = {
            "Threshold": "threshold",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp"
        }

        new_classifier = classifier_map.get(text, "threshold")

        if self.state.active_classifier != new_classifier:
            self.state.active_classifier = new_classifier
            print(f"[Classifier] Switched to: {new_classifier}")

            # Check if ML models are loaded for this classifier
            if new_classifier != 'threshold':
                if not self.state.loaded_ml_models:
                    print(f"[Classifier] ERROR: No ML models loaded at all!")
                    self.mw.statusBar().showMessage(
                        f"WARNING: No ML models loaded. Load models from ML Training tab first.", 5000)
                    self.mw.peak_detec_combo.blockSignals(True)
                    self.mw.peak_detec_combo.setCurrentText("Threshold")
                    self.mw.peak_detec_combo.blockSignals(False)
                    self.state.active_classifier = "threshold"
                    return
                else:
                    model_key_prefix = f'model1_{new_classifier}'
                    matching_keys = [k for k in self.state.loaded_ml_models.keys()
                                     if k.startswith(model_key_prefix)]

                    if not matching_keys:
                        print(f"[Classifier] ERROR: No model matching {model_key_prefix} found!")
                        self.mw.statusBar().showMessage(
                            f"WARNING: {text} models not loaded. Load models from ML Training tab first.", 5000)
                        self.mw.peak_detec_combo.blockSignals(True)
                        self.mw.peak_detec_combo.setCurrentText("Threshold")
                        self.mw.peak_detec_combo.blockSignals(False)
                        self.state.active_classifier = "threshold"
                        return
                    else:
                        print(f"[Classifier] Model {matching_keys[0]} found! Proceeding...")

            # Re-run peak detection to apply new classifier
            if self.state.analyze_chan and self.state.peaks_by_sweep:
                self.mw.statusBar().showMessage(f"Switching to {text} classifier...", 2000)
                self.update_displayed_peaks_from_classifier()
                if hasattr(self.mw, 'plot_manager'):
                    self.mw.redraw_main_plot()

    def update_displayed_peaks_from_classifier(self):
        """Update peaks_by_sweep and breath_by_sweep based on active classifier."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks_data = st.all_peaks_by_sweep[s]

            active_labels_key_ro = f'labels_{st.active_classifier}_ro'
            if active_labels_key_ro in all_peaks_data and all_peaks_data[active_labels_key_ro] is not None:
                all_peaks_data['labels'] = all_peaks_data[active_labels_key_ro].copy()
                all_peaks_data['label_source'] = np.array(['auto'] * len(all_peaks_data['labels']))
                print(f"[Classifier Update] Sweep {s}: Copied {active_labels_key_ro} to 'labels', "
                      f"found {np.sum(all_peaks_data['labels'] == 1)} breaths")
            else:
                if 'labels_threshold_ro' in all_peaks_data and all_peaks_data['labels_threshold_ro'] is not None:
                    all_peaks_data['labels'] = all_peaks_data['labels_threshold_ro'].copy()
                    all_peaks_data['label_source'] = np.array(['auto'] * len(all_peaks_data['labels']))
                    print(f"[Classifier Update] Sweep {s}: Falling back to threshold")
                else:
                    print(f"[Classifier Update] Sweep {s}: WARNING - No predictions available!")
                    continue

            labeled_mask = all_peaks_data['labels'] == 1
            labeled_indices = all_peaks_data['indices'][labeled_mask]
            st.peaks_by_sweep[s] = labeled_indices
            print(f"[Classifier Update] Sweep {s}: Updated peaks_by_sweep with {len(labeled_indices)} peaks")

            # Recompute breath events
            y_proc = self.mw._get_processed_for(st.analyze_chan, s)
            import core.peaks as peakdet
            breaths = peakdet.compute_breath_events(y_proc, labeled_indices, sr_hz=st.sr_hz, exclude_sec=0.030)
            st.breath_by_sweep[s] = breaths

    # =========================================================================
    # Model 3: Eupnea/Sniff Classifier
    # =========================================================================

    def on_eupnea_sniff_classifier_changed(self, text: str):
        """Handle eupnea/sniff classifier selection change."""
        classifier_map = {
            "GMM": "gmm",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp",
            "All Eupnea": "all_eupnea",
            "None (Clear)": "none"
        }

        new_classifier = classifier_map.get(text, "gmm")
        old_classifier = self.state.active_eupnea_sniff_classifier
        self.state.active_eupnea_sniff_classifier = new_classifier

        if old_classifier != new_classifier:
            print(f"[Eupnea/Sniff Classifier] Switched to: {new_classifier}")

        if new_classifier == 'all_eupnea':
            self.set_all_breaths_eupnea_sniff_class(0)
            print(f"[Eupnea/Sniff Classifier] Set all breaths to eupnea")
        elif new_classifier == 'none':
            self.clear_all_eupnea_sniff_labels()
            print(f"[Eupnea/Sniff Classifier] Cleared all eupnea/sniff labels")
        elif new_classifier == 'gmm':
            first_sweep_peaks = self.state.all_peaks_by_sweep.get(0, {})
            if 'gmm_class_ro' not in first_sweep_peaks or first_sweep_peaks['gmm_class_ro'] is None:
                print(f"[Eupnea/Sniff Classifier] GMM not run yet - running now...")
                self.mw.statusBar().showMessage(f"Running GMM clustering...", 2000)
                self.mw._run_automatic_gmm_clustering()
            self.update_eupnea_sniff_from_classifier()
        else:
            model_key_prefix = f'model3_{new_classifier}'
            matching_keys = [k for k in self.state.loaded_ml_models.keys()
                           if k.startswith(model_key_prefix)]

            if not matching_keys:
                print(f"[Eupnea/Sniff Classifier] ERROR: Model {model_key_prefix} not found!")
                self.mw.statusBar().showMessage(f"WARNING: {text} model not loaded.", 5000)
                self.mw.eup_sniff_combo.blockSignals(True)
                self.mw.eup_sniff_combo.setCurrentText("GMM")
                self.mw.eup_sniff_combo.blockSignals(False)
                self.state.active_eupnea_sniff_classifier = "gmm"
                return

            self.update_eupnea_sniff_from_classifier()

        # Rebuild regions for display
        try:
            import core.gmm_clustering as gmm_clustering
            gmm_clustering.build_eupnea_sniffing_regions(self.state, verbose=False)
        except Exception as e:
            print(f"[Eupnea/Sniff Classifier] Warning: Could not rebuild regions: {e}")

        if hasattr(self.mw, 'plot_manager'):
            self.mw.redraw_main_plot()

    def update_eupnea_sniff_from_classifier(self):
        """Copy selected classifier's predictions to breath_type_class array."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if st.active_eupnea_sniff_classifier == 'gmm':
                source_key = 'gmm_class_ro'
            else:
                source_key = f'eupnea_sniff_{st.active_eupnea_sniff_classifier}_ro'

            if source_key in all_peaks and all_peaks[source_key] is not None:
                all_peaks['breath_type_class'] = all_peaks[source_key].copy()
                all_peaks['eupnea_sniff_source'] = np.array(
                    [st.active_eupnea_sniff_classifier] * len(all_peaks['indices']))

                if s == 0:
                    n_eupnea = np.sum(all_peaks['breath_type_class'] == 0)
                    n_sniff = np.sum(all_peaks['breath_type_class'] == 1)
                    print(f"[Eupnea/Sniff Update] Sweep {s}: Eupnea: {n_eupnea}, Sniffing: {n_sniff}")
            else:
                print(f"[Eupnea/Sniff Update] Sweep {s}: WARNING - No predictions for {source_key}")

    def set_all_breaths_eupnea_sniff_class(self, class_value: int):
        """Set all breaths to a specific eupnea/sniff class (0=eupnea, 1=sniffing)."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if 'indices' in all_peaks and all_peaks['indices'] is not None:
                n_breaths = len(all_peaks['indices'])
                all_peaks['breath_type_class'] = np.full(n_breaths, class_value, dtype=int)
                all_peaks['eupnea_sniff_source'] = np.array(
                    [st.active_eupnea_sniff_classifier] * n_breaths)

    def clear_all_eupnea_sniff_labels(self):
        """Clear all eupnea/sniff labels."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if 'breath_type_class' in all_peaks:
                del all_peaks['breath_type_class']
            if 'eupnea_sniff_source' in all_peaks:
                del all_peaks['eupnea_sniff_source']

        st.sniff_regions_by_sweep.clear()

    # =========================================================================
    # Model 2: Sigh Classifier
    # =========================================================================

    def on_sigh_classifier_changed(self, text: str):
        """Handle sigh classifier selection change."""
        classifier_map = {
            "Manual": "manual",
            "XGBoost": "xgboost",
            "Random Forest": "rf",
            "MLP": "mlp",
            "None (Clear)": "none"
        }

        new_classifier = classifier_map.get(text, "manual")
        old_classifier = self.state.active_sigh_classifier
        self.state.active_sigh_classifier = new_classifier

        if old_classifier != new_classifier:
            print(f"[Sigh Classifier] Switched to: {new_classifier}")

        if new_classifier == 'none':
            self.clear_all_sigh_labels()
        elif new_classifier == 'manual':
            self.update_sigh_from_classifier()
        else:
            model_key_prefix = f'model2_{new_classifier}'
            matching_keys = [k for k in self.state.loaded_ml_models.keys()
                           if k.startswith(model_key_prefix)]

            if not matching_keys:
                print(f"[Sigh Classifier] ERROR: Model {model_key_prefix} not found!")
                self.mw.statusBar().showMessage(f"WARNING: {text} model not loaded.", 5000)
                self.mw.digh_combo.blockSignals(True)
                self.mw.digh_combo.setCurrentText("Manual")
                self.mw.digh_combo.blockSignals(False)
                self.state.active_sigh_classifier = "manual"
                return

            self.update_sigh_from_classifier()

        if hasattr(self.mw, 'plot_manager'):
            self.mw.redraw_main_plot()

    def update_sigh_from_classifier(self):
        """Copy selected classifier's predictions to sigh_class array."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if st.active_sigh_classifier == 'manual':
                if 'sigh_manual_ro' in all_peaks and all_peaks['sigh_manual_ro'] is not None:
                    all_peaks['sigh_class'] = all_peaks['sigh_manual_ro'].copy()
                    all_peaks['sigh_source'] = np.array(['manual'] * len(all_peaks['indices']))
                    if s in st.sigh_by_sweep:
                        manual_sigh_mask = all_peaks['sigh_manual_ro'] == 1
                        st.sigh_by_sweep[s] = all_peaks['indices'][manual_sigh_mask].tolist()
                else:
                    n_breaths = len(all_peaks['indices']) if 'indices' in all_peaks else 0
                    if n_breaths > 0:
                        all_peaks['sigh_class'] = np.zeros(n_breaths, dtype=np.int8)
                        if 'labels' in all_peaks and all_peaks['labels'] is not None:
                            all_peaks['sigh_class'][all_peaks['labels'] == 0] = -1
                        all_peaks['sigh_source'] = np.array(['manual'] * n_breaths)
                    if s in st.sigh_by_sweep:
                        st.sigh_by_sweep[s] = []
                continue

            source_key = f'sigh_{st.active_sigh_classifier}_ro'

            if source_key in all_peaks and all_peaks[source_key] is not None:
                all_peaks['sigh_class'] = all_peaks[source_key].copy()
                all_peaks['sigh_source'] = np.array(
                    [st.active_sigh_classifier] * len(all_peaks['indices']))

                sigh_mask = all_peaks['sigh_class'] == 1
                st.sigh_by_sweep[s] = all_peaks['indices'][sigh_mask].tolist()

                if s == 0:
                    n_sighs = np.sum(sigh_mask)
                    print(f"[Sigh Update] Sweep {s}: Found {n_sighs} sighs from {source_key}")
            else:
                print(f"[Sigh Update] Sweep {s}: WARNING - No predictions for {source_key}")

    def clear_all_sigh_labels(self):
        """Clear all sigh labels."""
        st = self.state

        for s in st.all_peaks_by_sweep.keys():
            all_peaks = st.all_peaks_by_sweep[s]

            if 'sigh_class' in all_peaks:
                del all_peaks['sigh_class']
            if 'sigh_source' in all_peaks:
                del all_peaks['sigh_source']

        st.sigh_by_sweep.clear()

    # =========================================================================
    # Async Precomputation
    # =========================================================================

    def precompute_remaining_classifiers_async(self):
        """
        Precompute predictions for all available classifiers in the background.
        This allows instant switching between classifiers.
        """
        from PyQt6.QtCore import QThread, pyqtSignal
        import core.ml_prediction as ml_pred

        st = self.state

        if not st.loaded_ml_models:
            print("[Precompute] No models loaded, skipping")
            return

        if not st.all_peaks_by_sweep:
            print("[Precompute] No peaks detected, skipping")
            return

        # Get list of model keys to precompute (skip already computed)
        models_to_compute = []
        for model_key in st.loaded_ml_models.keys():
            if model_key.startswith('model1_'):
                classifier_type = model_key.split('_')[1].split('_')[0]
                ro_key = f'labels_{classifier_type}_ro'
                first_sweep = st.all_peaks_by_sweep.get(0, {})
                if ro_key not in first_sweep or first_sweep[ro_key] is None:
                    models_to_compute.append(model_key)

        if not models_to_compute:
            print("[Precompute] All classifiers already computed")
            return

        print(f"[Precompute] Computing predictions for: {models_to_compute}")

        # Run prediction for each model
        for model_key in models_to_compute:
            try:
                model_info = st.loaded_ml_models[model_key]
                model = model_info['model']
                metadata = model_info['metadata']

                classifier_type = model_key.split('_')[1].split('_')[0]
                ro_key = f'labels_{classifier_type}_ro'

                for s in st.all_peaks_by_sweep.keys():
                    all_peaks = st.all_peaks_by_sweep[s]
                    if 'features' not in all_peaks:
                        continue

                    features = all_peaks['features']
                    if features is None or len(features) == 0:
                        continue

                    predictions = model.predict(features)
                    all_peaks[ro_key] = predictions

                print(f"[Precompute] Computed {ro_key} for all sweeps")

            except Exception as e:
                print(f"[Precompute] Error computing {model_key}: {e}")

        self.mw.statusBar().showMessage("Classifier precomputation complete", 2000)
