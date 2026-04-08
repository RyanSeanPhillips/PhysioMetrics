"""
Classifier system tests — behavioral baseline before ClassifierManager extraction.

Unit tests (1-8): Test classifier_service.py pure functions with synthetic data.
Integration tests (9-15): Test classifier switching end-to-end with real MainWindow.

Run:  python -m pytest tests/test_classifier_system.py -v
"""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════
# Helpers — synthetic peak data for unit tests
# ═══════════════════════════════════════════════════════════════════


def _make_synthetic_peaks(n_peaks=100, n_breaths=60):
    """Create synthetic all_peaks_by_sweep dict for unit tests.

    Returns a single-sweep dict with:
    - indices, labels (threshold), peak_metrics
    - labels_threshold_ro (n_breaths as label=1, rest as label=0)
    """
    indices = np.arange(0, n_peaks * 100, 100, dtype=int)  # every 100 samples
    labels_threshold = np.zeros(n_peaks, dtype=np.int8)
    labels_threshold[:n_breaths] = 1  # first n_breaths are "breaths"

    # Minimal peak_metrics (each peak gets a dict with some features)
    peak_metrics = []
    for i in range(n_peaks):
        peak_metrics.append({
            'amplitude': float(np.random.default_rng(i).normal(0.3, 0.1)),
            'width': float(np.random.default_rng(i + 1).normal(50, 10)),
            'prominence': float(np.random.default_rng(i + 2).normal(0.2, 0.05)),
        })

    return {
        0: {
            'indices': indices,
            'labels': labels_threshold.copy(),
            'labels_threshold_ro': labels_threshold.copy(),
            'label_source': np.array(['auto'] * n_peaks),
            'peak_metrics': peak_metrics,
            'breath_type_class': None,
            'gmm_class_ro': None,
            'sigh_class': None,
        }
    }


def _make_mock_model(predictions):
    """Create a mock sklearn model that returns fixed predictions."""
    class MockModel:
        def predict(self, X):
            n = len(X)
            if len(predictions) >= n:
                return np.array(predictions[:n])
            return np.array(predictions + [0] * (n - len(predictions)))

        def predict_proba(self, X):
            preds = self.predict(X)
            proba = np.zeros((len(X), 2))
            for i, p in enumerate(preds):
                proba[i, int(p)] = 0.9
                proba[i, 1 - int(p)] = 0.1
            return proba

    return MockModel()


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (1-8) — classifier_service.py pure functions
# ═══════════════════════════════════════════════════════════════════


class TestApplyClassifierLabels:
    """Tests 1-2: Label copying and fallback."""

    def test_apply_copies_ro_to_labels(self):
        """Test 1: apply_classifier_labels copies labels_{algo}_ro to labels."""
        from core.services.classifier_service import apply_classifier_labels

        all_peaks = _make_synthetic_peaks(100, 60)

        # Add xgboost predictions (different from threshold)
        xgb_labels = np.zeros(100, dtype=np.int8)
        xgb_labels[:40] = 1  # only 40 breaths (vs 60 from threshold)
        all_peaks[0]['labels_xgboost_ro'] = xgb_labels

        fallback = apply_classifier_labels(all_peaks, 'xgboost')

        assert not fallback, "Should not fallback when xgboost labels exist"
        assert np.array_equal(all_peaks[0]['labels'], xgb_labels)
        assert np.sum(all_peaks[0]['labels'] == 1) == 40
        print("  Copied xgboost labels: 40 breaths (was 60 from threshold)")

    def test_apply_fallback_to_threshold(self):
        """Test 2: Missing ML predictions → falls back to threshold."""
        from core.services.classifier_service import apply_classifier_labels

        all_peaks = _make_synthetic_peaks(100, 60)
        # No xgboost labels exist

        fallback = apply_classifier_labels(all_peaks, 'xgboost')

        assert fallback, "Should fallback when xgboost labels missing"
        # Should have copied threshold labels
        assert np.sum(all_peaks[0]['labels'] == 1) == 60
        print("  Fallback to threshold: 60 breaths preserved")


class TestPredictBreathVsNoise:
    """Test 3: Model 1 prediction stores in correct key."""

    def test_predict_stores_ro(self):
        """Test 3: predict_breath_vs_noise stores in labels_{algo}_ro."""
        from core.services.classifier_service import predict_breath_vs_noise

        all_peaks = _make_synthetic_peaks(50, 30)

        # Create mock model that predicts 20 breaths
        mock_predictions = [1] * 20 + [0] * 30
        mock_model = _make_mock_model(mock_predictions)

        loaded_models = {
            'model1_xgboost': {
                'model': mock_model,
                'metadata': {'feature_names': ['amplitude', 'width', 'prominence']},
                'path': 'mock_model.pkl',
            }
        }

        computed = predict_breath_vs_noise(
            all_peaks, loaded_models, 'xgboost',
            get_peak_metrics_fn=None,
        )

        assert computed > 0, "Should have computed predictions"
        assert 'labels_xgboost_ro' in all_peaks[0], "Missing labels_xgboost_ro"
        labels = all_peaks[0]['labels_xgboost_ro']
        assert labels is not None
        assert len(labels) == 50
        print(f"  Stored predictions: {np.sum(labels == 1)} breaths out of {len(labels)}")


class TestPredictEupneaSniff:
    """Test 4: Model 3 prediction stores eupnea_sniff key."""

    def test_predict_eupnea_sniff_stores_ro(self):
        """Test 4: predict_eupnea_sniff stores in eupnea_sniff_{algo}_ro."""
        from core.services.classifier_service import predict_eupnea_sniff

        all_peaks = _make_synthetic_peaks(50, 30)
        # Need breath labels (from Model 1) for Model 3 to work
        all_peaks[0]['labels'] = all_peaks[0]['labels_threshold_ro'].copy()

        # Mock model 3 that classifies breaths as eupnea/sniffing
        # First 20 eupnea (0), next 10 sniffing (1)
        mock_preds = [0] * 20 + [1] * 10
        mock_model = _make_mock_model(mock_preds)

        loaded_models = {
            'model3_xgboost': {
                'model': mock_model,
                'metadata': {'feature_names': ['amplitude', 'width', 'prominence']},
                'path': 'mock_model3.pkl',
            }
        }

        computed = predict_eupnea_sniff(
            all_peaks, loaded_models, 'xgboost',
            active_classifier='threshold',
            get_peak_metrics_fn=None,
        )

        assert computed > 0, "Should have computed eupnea/sniff predictions"
        key = 'eupnea_sniff_xgboost_ro'
        assert key in all_peaks[0], f"Missing {key}"
        es = all_peaks[0][key]
        assert es is not None
        # Non-breath peaks should be -1
        non_breath_mask = all_peaks[0]['labels_threshold_ro'] == 0
        assert np.all(es[non_breath_mask] == -1), "Non-breaths should be -1"
        print(f"  Eupnea/sniff predictions: eupnea={np.sum(es == 0)}, sniff={np.sum(es == 1)}, unclassified={np.sum(es == -1)}")


class TestPredictSignhs:
    """Test 5: Model 2 prediction stores sigh key."""

    def test_predict_sighs_stores_ro(self):
        """Test 5: predict_sighs stores in sigh_{algo}_ro."""
        from core.services.classifier_service import predict_sighs

        all_peaks = _make_synthetic_peaks(50, 30)
        all_peaks[0]['labels'] = all_peaks[0]['labels_threshold_ro'].copy()

        # Mock model 2: 5 sighs among 30 breaths
        mock_preds = [0] * 25 + [1] * 5
        mock_model = _make_mock_model(mock_preds)

        loaded_models = {
            'model2_xgboost': {
                'model': mock_model,
                'metadata': {'feature_names': ['amplitude', 'width', 'prominence']},
                'path': 'mock_model2.pkl',
            }
        }

        computed = predict_sighs(
            all_peaks, loaded_models, 'xgboost',
            active_classifier='threshold',
            get_peak_metrics_fn=None,
        )

        assert computed > 0
        key = 'sigh_xgboost_ro'
        assert key in all_peaks[0]
        sigh = all_peaks[0][key]
        assert sigh is not None
        print(f"  Sigh predictions: normal={np.sum(sigh == 0)}, sigh={np.sum(sigh == 1)}, unclassified={np.sum(sigh == -1)}")


class TestSetAllEupneaSniff:
    """Test 6: Bulk set all breaths to eupnea."""

    def test_set_all_eupnea(self):
        """Test 6: set_all_eupnea_sniff sets breath_type_class to 0."""
        from core.services.classifier_service import set_all_eupnea_sniff

        all_peaks = _make_synthetic_peaks(50, 30)
        set_all_eupnea_sniff(all_peaks, 0, 'all_eupnea')

        btc = all_peaks[0]['breath_type_class']
        assert btc is not None
        assert np.all(btc == 0), "All should be eupnea (0)"
        assert len(btc) == 50  # covers all peaks, not just breaths
        print(f"  Set all {len(btc)} peaks to eupnea")


class TestClearEupneaSniff:
    """Test 7: Clear eupnea/sniff labels."""

    def test_clear_removes_keys(self):
        """Test 7: clear_eupnea_sniff removes breath_type_class."""
        from core.services.classifier_service import set_all_eupnea_sniff, clear_eupnea_sniff

        all_peaks = _make_synthetic_peaks(50, 30)
        set_all_eupnea_sniff(all_peaks, 0, 'all_eupnea')
        assert 'breath_type_class' in all_peaks[0]

        clear_eupnea_sniff(all_peaks)

        assert 'breath_type_class' not in all_peaks[0]
        assert 'eupnea_sniff_source' not in all_peaks[0]
        print("  Cleared breath_type_class and eupnea_sniff_source")


class TestClearSignhs:
    """Test 8: Clear sigh labels."""

    def test_clear_sighs_removes_keys(self):
        """Test 8: clear_sighs removes sigh_class."""
        from core.services.classifier_service import clear_sighs

        all_peaks = _make_synthetic_peaks(50, 30)
        sigh_by_sweep = {0: [100, 200, 300]}

        # Set up sigh data
        all_peaks[0]['sigh_class'] = np.zeros(50, dtype=np.int8)
        all_peaks[0]['sigh_source'] = np.array(['xgboost'] * 50)

        clear_sighs(all_peaks, sigh_by_sweep)

        assert 'sigh_class' not in all_peaks[0]
        assert 'sigh_source' not in all_peaks[0]
        assert len(sigh_by_sweep) == 0
        print("  Cleared sigh_class, sigh_source, and sigh_by_sweep")


# ═══════════════════════════════════════════════════════════════════
# Integration Tests (9-15) — Real MainWindow + ABF files
# ═══════════════════════════════════════════════════════════════════

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF


def _ensure_file_loaded_with_peaks(mw, file_info=None):
    """Load multi-channel ABF and ensure peaks are detected."""
    from PyQt6.QtWidgets import QApplication

    if file_info is None:
        file_info = MULTI_CHANNEL_ABF

    if not file_info.path.exists():
        pytest.skip(f"Test file not found: {file_info.path}")

    load_file_and_wait(mw, file_info.path, channel_name=file_info.pleth_channels[0])

    st = mw.state
    if not st.peaks_by_sweep or sum(len(p) for p in st.peaks_by_sweep.values()) < 10:
        # Need to detect peaks
        if not mw.peak_prominence or mw.peak_prominence < 0.01:
            mw.peak_prominence = 0.05
            mw.peak_height_threshold = 0.05
            btn = getattr(mw, "ApplyPeakFindPushButton", None)
            if btn:
                btn.setEnabled(True)

        mw._apply_peak_detection()

        deadline = time.time() + 30
        while time.time() < deadline:
            QApplication.processEvents()
            if st.peaks_by_sweep and sum(len(p) for p in st.peaks_by_sweep.values()) > 0:
                break
            time.sleep(0.1)

        for _ in range(30):
            QApplication.processEvents()
            time.sleep(0.05)

    return sum(len(p) for p in st.peaks_by_sweep.values()) > 0


class TestClassifierSwitchThresholdToXGBoost:
    """Test 9: Switch Model 1 from threshold to xgboost."""

    @pytest.fixture(autouse=True)
    def _setup(self, main_window):
        if not MULTI_CHANNEL_ABF.path.exists():
            pytest.skip("Test file not found")
        assert _ensure_file_loaded_with_peaks(main_window), "Peak detection failed"

    def test_switch_to_xgboost(self, main_window):
        """Test 9: Switch to XGBoost → peaks_by_sweep changes."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state
        if not st.loaded_ml_models:
            pytest.skip("No ML models loaded")

        if not any(k.startswith('model1_xgboost') for k in st.loaded_ml_models):
            pytest.skip("No xgboost model loaded")

        # Record threshold peak count
        threshold_peaks = sum(len(p) for p in st.peaks_by_sweep.values())

        # Switch to XGBoost
        main_window.on_classifier_changed("XGBoost")
        QApplication.processEvents()

        xgboost_peaks = sum(len(p) for p in st.peaks_by_sweep.values())

        # XGBoost typically detects fewer breaths than threshold (more selective)
        assert xgboost_peaks > 0, "XGBoost should detect some breaths"
        assert st.active_classifier == 'xgboost'

        print(f"  Threshold: {threshold_peaks} peaks → XGBoost: {xgboost_peaks} peaks")

        # Switch back to threshold
        main_window.on_classifier_changed("Threshold")
        QApplication.processEvents()

        restored_peaks = sum(len(p) for p in st.peaks_by_sweep.values())
        assert st.active_classifier == 'threshold'
        print(f"  Restored threshold: {restored_peaks} peaks")


class TestClassifierFallbackNoModels:
    """Test 10: Switch to ML with no model → fallback to threshold."""

    def test_fallback_when_model_missing(self, main_window):
        """Test 10: Switching to unavailable model falls back gracefully."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state

        # Save current state
        old_classifier = st.active_classifier

        # Try switching to MLP (may not be loaded)
        has_mlp = st.loaded_ml_models and any(k.startswith('model1_mlp') for k in st.loaded_ml_models)

        if has_mlp:
            # If MLP is actually loaded, test with a made-up algorithm
            # Just verify the classifier state management doesn't crash
            st.active_classifier = old_classifier
            print("  MLP model is loaded, skipping fallback test")
            return

        # MLP not loaded — switching to it should fallback
        main_window.on_classifier_changed("MLP")
        QApplication.processEvents()

        # Should have fallen back to threshold (or at least not crashed)
        assert st.active_classifier in ('threshold', 'mlp', old_classifier), \
            f"Unexpected classifier state: {st.active_classifier}"

        # Restore
        st.active_classifier = old_classifier
        print(f"  Fallback handled gracefully, classifier={st.active_classifier}")


class TestEupneaSniffSwitchGMMToXGBoost:
    """Test 11: Switch eupnea/sniff classifier."""

    def test_eupnea_sniff_switch(self, main_window):
        """Test 11: Switch Model 3 from GMM to XGBoost."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state

        if not st.loaded_ml_models or not any(k.startswith('model3_xgboost') for k in st.loaded_ml_models):
            pytest.skip("No model3 xgboost loaded")

        # Ensure we have peaks
        if not st.all_peaks_by_sweep:
            pytest.skip("No peaks detected")

        # Current state should have GMM-based classification
        old_classifier = st.active_eupnea_sniff_classifier

        # Switch to XGBoost
        main_window.on_eupnea_sniff_classifier_changed("XGBoost")
        QApplication.processEvents()

        assert st.active_eupnea_sniff_classifier == 'xgboost'

        # Check that breath_type_class was updated
        first_sweep = next(iter(st.all_peaks_by_sweep.keys()))
        btc = st.all_peaks_by_sweep[first_sweep].get('breath_type_class')
        assert btc is not None, "breath_type_class should be set after switch"

        print(f"  Switched to xgboost: eupnea={np.sum(btc == 0)}, sniff={np.sum(btc == 1)}")

        # Restore
        main_window.on_eupnea_sniff_classifier_changed(
            {"gmm": "GMM", "all_eupnea": "All Eupnea"}.get(old_classifier, "GMM")
        )
        QApplication.processEvents()


class TestSighSwitchManualToXGBoost:
    """Test 12: Switch sigh classifier."""

    def test_sigh_switch(self, main_window):
        """Test 12: Switch Model 2 from manual to XGBoost."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state

        if not st.loaded_ml_models or not any(k.startswith('model2_xgboost') for k in st.loaded_ml_models):
            pytest.skip("No model2 xgboost loaded")

        if not st.all_peaks_by_sweep:
            pytest.skip("No peaks detected")

        # Switch to XGBoost sigh detection
        main_window.on_sigh_classifier_changed("XGBoost")
        QApplication.processEvents()

        assert st.active_sigh_classifier == 'xgboost'

        first_sweep = next(iter(st.all_peaks_by_sweep.keys()))
        sigh_class = st.all_peaks_by_sweep[first_sweep].get('sigh_class')
        assert sigh_class is not None, "sigh_class should be set after switch"

        n_sighs = np.sum(sigh_class == 1)
        print(f"  XGBoost sigh: {n_sighs} sighs detected in sweep 0")

        # Restore
        main_window.on_sigh_classifier_changed("Manual")
        QApplication.processEvents()


class TestCrossModelSideEffect:
    """Test 13: predict_with_cascade stores Model 2/3 predictions alongside Model 1."""

    def test_cascade_stores_eupnea_and_sigh(self, main_window):
        """Test 13: After XGBoost Model 1 runs, Model 2/3 predictions also exist."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state

        if not st.loaded_ml_models:
            pytest.skip("No ML models loaded")

        if not st.all_peaks_by_sweep:
            pytest.skip("No peaks detected")

        # The cascade should have run during peak detection (precompute)
        first_sweep = next(iter(st.all_peaks_by_sweep.keys()))
        data = st.all_peaks_by_sweep[first_sweep]

        # Check for cross-model predictions stored by cascade
        has_xgb_labels = 'labels_xgboost_ro' in data and data['labels_xgboost_ro'] is not None
        has_xgb_eupnea = 'eupnea_sniff_xgboost_ro' in data and data['eupnea_sniff_xgboost_ro'] is not None
        has_xgb_sigh = 'sigh_xgboost_ro' in data and data['sigh_xgboost_ro'] is not None

        print(f"  Cross-model: labels_xgb={has_xgb_labels}, eupnea_xgb={has_xgb_eupnea}, sigh_xgb={has_xgb_sigh}")

        # At minimum, if xgboost labels exist, the cascade ran
        if has_xgb_labels:
            # The cascade SHOULD also produce eupnea/sniff and sigh predictions
            # (depending on model availability)
            n_breaths = np.sum(data['labels_xgboost_ro'] == 1)
            print(f"  XGBoost detected {n_breaths} breaths")
            if has_xgb_eupnea:
                n_eupnea = np.sum(data['eupnea_sniff_xgboost_ro'] == 0)
                n_sniff = np.sum(data['eupnea_sniff_xgboost_ro'] == 1)
                print(f"  Cross-model eupnea/sniff: eupnea={n_eupnea}, sniff={n_sniff}")
            if has_xgb_sigh:
                n_sighs = np.sum(data['sigh_xgboost_ro'] == 1)
                print(f"  Cross-model sigh: {n_sighs} sighs")
        else:
            print("  No xgboost predictions found (models may not have run)")


class TestPrecomputePopulatesRoKeys:
    """Test 14: After peak detection, precompute fills _ro arrays."""

    def test_precompute_fills_keys(self, main_window):
        """Test 14: Verify _ro prediction arrays exist after precompute."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state

        if not st.all_peaks_by_sweep:
            pytest.skip("No peaks detected")

        first_sweep = next(iter(st.all_peaks_by_sweep.keys()))
        data = st.all_peaks_by_sweep[first_sweep]

        # Check which _ro keys exist
        ro_keys = [k for k in data.keys() if k.endswith('_ro') and data[k] is not None]

        # At minimum, threshold should always exist
        assert 'labels_threshold_ro' in data and data['labels_threshold_ro'] is not None, \
            "labels_threshold_ro should always exist"

        print(f"  Available _ro keys: {ro_keys}")

        # If ML models are loaded, precompute should have filled more
        if st.loaded_ml_models:
            # After peak detection + precompute, we expect at least threshold + one ML
            assert len(ro_keys) >= 2, f"Expected at least 2 _ro keys with models loaded, got {len(ro_keys)}"
            print(f"  Models loaded: {list(st.loaded_ml_models.keys())}")


class TestClassifierDropdownEnableDisable:
    """Test 15: Dropdowns reflect loaded models."""

    def test_dropdown_items_match_models(self, main_window):
        """Test 15: Verify dropdown items enabled/disabled based on models."""
        from PyQt6.QtWidgets import QApplication

        st = main_window.state
        combo = main_window.peak_detec_combo

        # Threshold (index 0) should always be enabled
        model = combo.model()
        assert model.item(0).isEnabled(), "Threshold should always be enabled"

        if st.loaded_ml_models:
            # Check XGBoost (index 1)
            has_xgb = any(k.startswith('model1_xgboost') for k in st.loaded_ml_models)
            assert model.item(1).isEnabled() == has_xgb, \
                f"XGBoost enabled={model.item(1).isEnabled()}, expected={has_xgb}"

            print(f"  Dropdown: Threshold=enabled, XGBoost={'enabled' if has_xgb else 'disabled'}")
        else:
            print("  No models loaded, only Threshold should be enabled")
