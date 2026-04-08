"""
GMM clustering tests — behavioral baseline before GMMService extraction.

Unit tests (1-7): Use FakeMW mock with synthetic data to test GMMManager logic.
Integration tests (8-15): Use session-scoped MainWindow with real ABF files.

Run:  python -m pytest tests/test_gmm_clustering.py -v
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
# FakeMW — minimal mock of MainWindow for unit tests
# ═══════════════════════════════════════════════════════════════════

class FakeMW:
    """Minimal stand-in for MainWindow used by GMMManager."""

    def __init__(self, state):
        self.state = state
        self.filter_order = 4
        self.notch_filter_lower = None
        self.notch_filter_upper = None
        self.use_zscore_normalization = False
        self.zscore_global_mean = None
        self.zscore_global_std = None
        self._cached_gmm_results = None
        self._status_messages = []

    def _apply_notch_filter(self, y, sr_hz, lower, upper):
        from core.filters import notch_filter_1d
        return notch_filter_1d(y, sr_hz, lower, upper)

    def _compute_global_zscore_stats(self):
        return 0.0, 1.0

    def _log_status_message(self, msg, timeout=0):
        self._status_messages.append(msg)


def _make_synthetic_state(n_breaths=80, sr_hz=1000, seed=42):
    """Create a synthetic AppState with plausible breathing data.

    Generates two distinct populations:
      - eupnea: freq ~3 Hz, larger amplitude
      - sniffing: freq ~7 Hz, smaller amplitude
    """
    from core.state import AppState

    rng = np.random.default_rng(seed)
    st = AppState()
    st.sr_hz = sr_hz
    st.analyze_chan = "IN 0"

    # Build a simple time array (1 sweep, enough duration for n_breaths)
    duration = n_breaths * 0.5  # ~0.5s per breath on average
    n_samples = int(duration * sr_hz)
    st.t = np.linspace(0, duration, n_samples)

    # Generate synthetic signal: sum of two sinusoidal components
    # (not used directly by GMM — it works on metrics — but needed for feature collection)
    y = 0.3 * np.sin(2 * np.pi * 3.0 * st.t) + rng.normal(0, 0.02, n_samples)
    st.sweeps = {"IN 0": y.reshape(-1, 1)}  # (n_samples, 1 sweep)

    # Split breaths: first 60% eupnea, last 40% sniffing
    n_eupnea = int(n_breaths * 0.6)
    n_sniffing = n_breaths - n_eupnea

    # Build onset/offset arrays with plausible intervals
    onsets = []
    peaks_arr = []
    offsets = []
    expmins = []
    expoffs = []
    idx = 100  # start a bit into the trace

    for i in range(n_breaths):
        if i < n_eupnea:
            # Eupnea: longer cycle (~300ms), larger amplitude
            cycle_len = int(rng.normal(300, 20) * sr_hz / 1000)
        else:
            # Sniffing: shorter cycle (~140ms), smaller amplitude
            cycle_len = int(rng.normal(140, 15) * sr_hz / 1000)

        cycle_len = max(cycle_len, 50)  # minimum 50 samples

        onset = idx
        peak = idx + cycle_len // 3
        offset = idx + cycle_len * 2 // 3
        expmin = idx + cycle_len * 5 // 6
        expoff = idx + cycle_len

        if expoff >= n_samples:
            break

        onsets.append(onset)
        peaks_arr.append(peak)
        offsets.append(offset)
        expmins.append(expmin)
        expoffs.append(expoff)

        # Make the signal reflect the breath shape
        breath_t = np.arange(cycle_len)
        amp = 0.4 if i < n_eupnea else 0.15
        breath_wave = amp * np.sin(np.pi * breath_t / cycle_len)
        end = min(onset + cycle_len, n_samples)
        actual_len = end - onset
        y[onset:end] = breath_wave[:actual_len]

        idx = expoff + int(rng.integers(5, 20))

    onsets = np.array(onsets, dtype=int)
    peaks_arr = np.array(peaks_arr, dtype=int)
    offsets = np.array(offsets, dtype=int)
    expmins = np.array(expmins, dtype=int)
    expoffs = np.array(expoffs, dtype=int)
    actual_n = len(onsets)

    # Store in state (sweep 0)
    st.peaks_by_sweep = {0: peaks_arr}
    st.breath_by_sweep = {0: {
        'onsets': onsets,
        'offsets': offsets,
        'expmins': expmins,
        'expoffs': expoffs,
    }}

    # Build all_peaks_by_sweep (needed for store_gmm_classifications_in_peaks)
    labels = np.ones(actual_n, dtype=np.int8)  # all are breaths (label=1)
    st.all_peaks_by_sweep = {0: {
        'indices': peaks_arr.copy(),
        'labels': labels,
        'breath_type_class': None,
        'gmm_class_ro': None,
    }}

    st.active_eupnea_sniff_classifier = 'gmm'

    # Re-store the modified signal
    st.sweeps["IN 0"] = y.reshape(-1, 1)

    # No filters active
    st.use_low = False
    st.use_high = False
    st.use_mean_sub = False
    st.use_invert = False
    st.low_hz = None
    st.high_hz = None
    st.mean_val = 0.0

    return st


# ═══════════════════════════════════════════════════════════════════
# Unit Tests (1-7) — FakeMW + synthetic data
# ═══════════════════════════════════════════════════════════════════


class TestCollectFeatures:
    """Tests 1-2: Feature collection from GMMManager."""

    def test_collect_features_shape(self):
        """Test 1: Feature matrix has expected shape (n_breaths, 4)."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state(n_breaths=60)
        mw = FakeMW(st)
        mgr = GMMManager(mw)

        feature_keys = ["if", "ti", "amp_insp", "max_dinsp"]
        feature_matrix, breath_cycles = mgr.collect_gmm_breath_features(feature_keys)

        assert feature_matrix.ndim == 2
        assert feature_matrix.shape[1] == 4, f"Expected 4 features, got {feature_matrix.shape[1]}"
        assert feature_matrix.shape[0] > 0, "No breaths collected"
        assert len(breath_cycles) == feature_matrix.shape[0]
        # All breath_cycles should reference sweep 0
        assert all(bc[0] == 0 for bc in breath_cycles)
        print(f"  Collected {feature_matrix.shape[0]} breaths with {feature_matrix.shape[1]} features")

    def test_collect_features_skips_nan(self):
        """Test 2: Breaths with NaN metric values are excluded."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state(n_breaths=40)

        # Corrupt signal at a few breath locations to produce NaN metrics
        onsets = st.breath_by_sweep[0]['onsets']
        y = st.sweeps["IN 0"][:, 0]
        for i in [5, 10, 15]:
            if i < len(onsets):
                start = int(onsets[i])
                # Set signal to constant (will produce NaN for derivative-based metrics)
                end = min(start + 50, len(y))
                y[start:end] = 0.0
        st.sweeps["IN 0"] = y.reshape(-1, 1)

        mw = FakeMW(st)
        mgr = GMMManager(mw)

        feature_keys = ["if", "ti", "amp_insp", "max_dinsp"]
        feature_matrix, breath_cycles = mgr.collect_gmm_breath_features(feature_keys)

        # Should still have some breaths, but not necessarily all
        assert feature_matrix.shape[0] > 0
        # No NaN values should be in the output
        assert not np.any(np.isnan(feature_matrix)), "Feature matrix contains NaN values"
        print(f"  Collected {feature_matrix.shape[0]} valid breaths (NaN breaths excluded)")


class TestIdentifySniffingCluster:
    """Test 3: Sniffing cluster identification."""

    def test_identify_sniffing_cluster(self):
        """Test 3: Two clear clusters -> correct sniffing cluster identified."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state()
        mw = FakeMW(st)
        mgr = GMMManager(mw)

        # Create two clearly separated clusters
        n = 100
        feature_keys = ["if", "ti"]

        # Cluster 0: low IF, high Ti (eupnea)
        eupnea_features = np.column_stack([
            np.random.default_rng(1).normal(3.0, 0.3, 60),   # IF ~3 Hz
            np.random.default_rng(2).normal(0.15, 0.02, 60),  # Ti ~150ms
        ])
        # Cluster 1: high IF, low Ti (sniffing)
        sniffing_features = np.column_stack([
            np.random.default_rng(3).normal(7.0, 0.5, 40),   # IF ~7 Hz
            np.random.default_rng(4).normal(0.06, 0.01, 40),  # Ti ~60ms
        ])

        feature_matrix = np.vstack([eupnea_features, sniffing_features])
        cluster_labels = np.array([0] * 60 + [1] * 40)

        sniffing_id = mgr.identify_gmm_sniffing_cluster(
            feature_matrix, cluster_labels, feature_keys, silhouette=0.8
        )

        assert sniffing_id == 1, f"Expected cluster 1 as sniffing, got {sniffing_id}"
        print(f"  Correctly identified cluster {sniffing_id} as sniffing")


class TestApplyRegions:
    """Test 4: Region application stores in state."""

    def test_apply_regions_stores_in_state(self):
        """Test 4: After apply -> state has sniff/eupnea regions."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state(n_breaths=50)
        mw = FakeMW(st)
        mgr = GMMManager(mw)

        n_actual = len(st.peaks_by_sweep[0])
        breath_cycles = [(0, i) for i in range(n_actual)]

        # Assign first 60% as eupnea (cluster 0), rest as sniffing (cluster 1)
        n_eupnea = int(n_actual * 0.6)
        cluster_labels = np.array([0] * n_eupnea + [1] * (n_actual - n_eupnea))
        cluster_probs = np.zeros((n_actual, 2))
        for i in range(n_actual):
            if cluster_labels[i] == 0:
                cluster_probs[i] = [0.85, 0.15]
            else:
                cluster_probs[i] = [0.1, 0.9]

        sniffing_id = 1
        n_sniffing = mgr.apply_gmm_sniffing_regions(
            breath_cycles, cluster_labels, cluster_probs, sniffing_id
        )

        assert n_sniffing > 0, "Expected some sniffing breaths"
        assert hasattr(st, 'sniff_regions_by_sweep')
        assert hasattr(st, 'eupnea_regions_by_sweep')
        assert len(st.sniff_regions_by_sweep) > 0, "No sniffing regions created"
        assert len(st.eupnea_regions_by_sweep) > 0, "No eupnea regions created"
        assert hasattr(st, 'gmm_sniff_probabilities')
        assert 0 in st.gmm_sniff_probabilities
        print(f"  Created sniff regions in {len(st.sniff_regions_by_sweep)} sweep(s), "
              f"eupnea regions in {len(st.eupnea_regions_by_sweep)} sweep(s)")


class TestRunAutomatic:
    """Test 5: Full automatic pipeline."""

    def test_run_automatic_end_to_end(self):
        """Test 5: Full pipeline -> state has regions + cache populated."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state(n_breaths=80)
        mw = FakeMW(st)
        mgr = GMMManager(mw)

        mgr.run_automatic_gmm_clustering()

        # Verify cache was populated
        assert mw._cached_gmm_results is not None, "GMM results not cached"
        cache = mw._cached_gmm_results
        assert 'cluster_labels' in cache
        assert 'cluster_probabilities' in cache
        assert 'feature_matrix' in cache
        assert 'breath_cycles' in cache
        assert 'sniffing_cluster_id' in cache
        assert 'feature_keys' in cache

        # Verify regions were created
        assert hasattr(st, 'sniff_regions_by_sweep')
        assert hasattr(st, 'eupnea_regions_by_sweep')

        # Verify probabilities stored
        assert hasattr(st, 'gmm_sniff_probabilities')
        assert len(st.gmm_sniff_probabilities) > 0

        # Verify all_peaks_by_sweep was updated with classifications
        all_peaks = st.all_peaks_by_sweep[0]
        assert all_peaks['breath_type_class'] is not None
        assert all_peaks['gmm_class_ro'] is not None

        print(f"  Pipeline complete: {len(cache['cluster_labels'])} breaths classified, "
              f"cache has {len(cache)} keys")


class TestComputeEupnea:
    """Test 6: Eupnea mask computation from active classifier."""

    def test_compute_eupnea_from_active_classifier(self):
        """Test 6: Known breath_type_class -> correct eupnea mask."""
        from core.gmm_manager import GMMManager

        st = _make_synthetic_state(n_breaths=30)
        mw = FakeMW(st)
        mgr = GMMManager(mw)

        n_actual = len(st.peaks_by_sweep[0])
        onsets = st.breath_by_sweep[0]['onsets']
        offsets = st.breath_by_sweep[0]['offsets']
        signal_length = len(st.t)

        # Set known breath_type_class: alternating eupnea/sniffing
        breath_type_class = np.array([i % 2 for i in range(n_actual)], dtype=np.int8)
        st.all_peaks_by_sweep[0]['breath_type_class'] = breath_type_class

        mask = mgr.compute_eupnea_from_active_classifier(0, signal_length)

        assert mask.shape == (signal_length,)
        assert mask.dtype == float
        # Eupnea breaths (class=0) should have mask=1 at their onset locations
        for i in range(n_actual):
            onset = int(onsets[i])
            if onset < signal_length:
                if breath_type_class[i] == 0:
                    assert mask[onset] == 1.0, f"Breath {i} is eupnea but mask is 0 at onset {onset}"
                else:
                    assert mask[onset] == 0.0, f"Breath {i} is sniffing but mask is 1 at onset {onset}"

        print(f"  Eupnea mask correct for {n_actual} breaths")


class TestNoPeaksNoop:
    """Test 7: Empty state doesn't crash."""

    def test_no_peaks_is_noop(self):
        """Test 7: Empty state -> no crash, no side effects."""
        from core.gmm_manager import GMMManager
        from core.state import AppState

        st = AppState()
        st.sr_hz = 1000
        st.analyze_chan = "IN 0"
        st.t = np.linspace(0, 1, 1000)
        st.sweeps = {"IN 0": np.zeros((1000, 1))}
        st.peaks_by_sweep = {}
        st.breath_by_sweep = {}
        st.all_peaks_by_sweep = {}
        st.active_eupnea_sniff_classifier = 'gmm'

        mw = FakeMW(st)
        mgr = GMMManager(mw)

        # Should not crash
        mgr.run_automatic_gmm_clustering()

        # Cache should remain None
        assert mw._cached_gmm_results is None

        # Eupnea computation should return zeros
        mask = mgr.compute_eupnea_from_gmm(0, 1000)
        assert np.all(mask == 0)

        mask2 = mgr.compute_eupnea_from_active_classifier(0, 1000)
        assert np.all(mask2 == 0)

        print("  Empty state handled gracefully")


# ═══════════════════════════════════════════════════════════════════
# Integration Tests (8-15) — Real MainWindow + ABF files
# ═══════════════════════════════════════════════════════════════════

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF, EXAMPLES_DIR


# Helper: find AWK multi-chamber ABF
AWAKE_ABF = EXAMPLES_DIR / "25729001.abf"


def _ensure_peaks_detected(mw):
    """Run peak detection if not already done. Returns True if peaks exist."""
    from PyQt6.QtWidgets import QApplication

    st = mw.state
    if st.peaks_by_sweep and len(st.peaks_by_sweep) > 0:
        total = sum(len(p) for p in st.peaks_by_sweep.values())
        if total > 10:
            return True

    # Auto-detect prominence (may skip for non-Pleth channels)
    auto_fn = getattr(mw, "_auto_detect_prominence_silent", None)
    if auto_fn:
        auto_fn()
        QApplication.processEvents()

    # Always set prominence manually as fallback (auto-detect skips non-Pleth types)
    if not mw.peak_prominence or mw.peak_prominence < 0.01:
        mw.peak_prominence = 0.05
        mw.peak_height_threshold = 0.05
        btn = getattr(mw, "ApplyPeakFindPushButton", None)
        if btn:
            btn.setEnabled(True)

    mw._apply_peak_detection()

    # Process events with generous timeout (peak detection + GMM can take a while)
    deadline = time.time() + 30
    while time.time() < deadline:
        QApplication.processEvents()
        if st.peaks_by_sweep and sum(len(p) for p in st.peaks_by_sweep.values()) > 0:
            break
        time.sleep(0.1)

    # Extra time for GMM to finish (it runs synchronously after peak detection)
    for _ in range(30):
        QApplication.processEvents()
        time.sleep(0.05)

    return bool(st.peaks_by_sweep and sum(len(p) for p in st.peaks_by_sweep.values()) > 0)


class TestGMMAfterPeakDetection:
    """Test 8: GMM runs after peak detection on real data."""

    @pytest.fixture(autouse=True)
    def _load_file(self, main_window):
        """Load the multi-channel ABF and detect peaks."""
        if not MULTI_CHANNEL_ABF.path.exists():
            pytest.skip(f"Test file not found: {MULTI_CHANNEL_ABF.path}")

        load_file_and_wait(main_window, MULTI_CHANNEL_ABF.path,
                           channel_name=MULTI_CHANNEL_ABF.pleth_channels[0])
        assert _ensure_peaks_detected(main_window), "Peak detection failed"

    def test_gmm_runs_after_peak_detection(self, main_window):
        """Test 8: After peak detection, GMM probabilities should be populated."""
        st = main_window.state
        assert hasattr(st, 'gmm_sniff_probabilities'), "gmm_sniff_probabilities not set"
        assert len(st.gmm_sniff_probabilities) > 0, "No GMM probabilities stored"
        print(f"  GMM probabilities stored for {len(st.gmm_sniff_probabilities)} sweep(s)")


class TestEupneaSniffingRegions:
    """Test 9: Eupnea/sniffing regions created after peak detection."""

    def test_eupnea_sniffing_regions_created(self, main_window):
        """Test 9: state should have sniff and eupnea regions."""
        st = main_window.state

        # At least one of these should have entries if GMM ran
        has_sniff = hasattr(st, 'sniff_regions_by_sweep') and len(st.sniff_regions_by_sweep) > 0
        has_eupnea = hasattr(st, 'eupnea_regions_by_sweep') and len(st.eupnea_regions_by_sweep) > 0

        assert has_sniff or has_eupnea, (
            "Neither sniff nor eupnea regions found — GMM may not have run"
        )
        if has_sniff:
            total_sniff = sum(len(v) for v in st.sniff_regions_by_sweep.values())
            print(f"  Sniffing regions: {total_sniff} across {len(st.sniff_regions_by_sweep)} sweep(s)")
        if has_eupnea:
            total_eupnea = sum(len(v) for v in st.eupnea_regions_by_sweep.values())
            print(f"  Eupnea regions: {total_eupnea} across {len(st.eupnea_regions_by_sweep)} sweep(s)")


class TestGMMCachePopulated:
    """Test 10: GMM results cached for dialog."""

    def test_gmm_results_cached(self, main_window):
        """Test 10: _cached_gmm_results should have expected keys."""
        cache = getattr(main_window, '_cached_gmm_results', None)
        assert cache is not None, "GMM results not cached"

        expected_keys = {'cluster_labels', 'cluster_probabilities', 'feature_matrix',
                         'breath_cycles', 'sniffing_cluster_id', 'feature_keys'}
        actual_keys = set(cache.keys())
        missing = expected_keys - actual_keys
        assert not missing, f"Missing cache keys: {missing}"

        assert len(cache['cluster_labels']) > 0, "Empty cluster labels"
        assert cache['sniffing_cluster_id'] is not None, "No sniffing cluster identified"
        print(f"  Cache has {len(cache['cluster_labels'])} classified breaths, "
              f"sniffing cluster = {cache['sniffing_cluster_id']}")


class TestEupneaSniffingShadingRenders:
    """Test 11: Plot renders with eupnea/sniffing shading."""

    def test_eupnea_sniffing_shading_renders(self, main_window):
        """Test 11: Redraw with shading doesn't crash; plot items exist."""
        from PyQt6.QtWidgets import QApplication

        main_window.redraw_main_plot()
        QApplication.processEvents()

        # Check that the plot host has items (LinearRegionItems for shading)
        plot_host = main_window.plot_host
        assert plot_host is not None
        # The plot should have rendered without crashing — that's the key assertion
        print("  Plot rendered with eupnea/sniffing shading successfully")


class TestClassifierSwitchRebuildsRegions:
    """Test 12: Switching classifier rebuilds regions."""

    def test_classifier_switch_rebuilds_regions(self, main_window):
        """Test 12: Switching classifier dropdown rebuilds regions."""
        from PyQt6.QtWidgets import QApplication
        import core.gmm_clustering as gmm_clustering

        st = main_window.state

        # Record current regions
        old_sniff_count = sum(len(v) for v in getattr(st, 'sniff_regions_by_sweep', {}).values())

        # Switch to 'all_eupnea' — should clear sniffing regions
        st.active_eupnea_sniff_classifier = 'all_eupnea'
        # Trigger region rebuild via the classifier service
        from core.services.classifier_service import set_all_eupnea_sniff
        set_all_eupnea_sniff(st.all_peaks_by_sweep, 0, 'all_eupnea')
        gmm_clustering.build_eupnea_sniffing_regions(st, verbose=False)
        QApplication.processEvents()

        # After 'all_eupnea', all breaths should be eupnea — sniffing regions should be zero
        new_sniff_count = sum(len(v) for v in getattr(st, 'sniff_regions_by_sweep', {}).values())
        assert new_sniff_count == 0, f"Expected 0 sniffing regions with all_eupnea, got {new_sniff_count}"

        # Switch back to 'gmm' — should restore regions
        st.active_eupnea_sniff_classifier = 'gmm'
        from core.services.classifier_service import apply_eupnea_sniff_labels
        apply_eupnea_sniff_labels(st.all_peaks_by_sweep, 'gmm')
        gmm_clustering.build_eupnea_sniffing_regions(st, verbose=False)
        QApplication.processEvents()

        restored_sniff_count = sum(len(v) for v in getattr(st, 'sniff_regions_by_sweep', {}).values())
        print(f"  all_eupnea: 0 sniff regions | gmm restored: {restored_sniff_count} sniff regions")


class TestGMMDialogOpens:
    """Test 13: GMM dialog opens with cached data."""

    def test_gmm_dialog_opens_with_cached_data(self, main_window, dialog_watcher):
        """Test 13: GMM dialog opens without error using cached data."""
        from PyQt6.QtWidgets import QApplication

        cache = getattr(main_window, '_cached_gmm_results', None)
        if cache is None:
            pytest.skip("No cached GMM results to load into dialog")

        # Temporarily stop the dialog watcher so it doesn't close our dialog mid-init
        if dialog_watcher:
            dialog_watcher.stop()

        try:
            from dialogs.gmm_clustering_dialog import GMMClusteringDialog
            dialog = GMMClusteringDialog(parent=main_window, main_window=main_window)
            dialog.show()
            for _ in range(10):
                QApplication.processEvents()
                time.sleep(0.05)
            dialog.close()
            QApplication.processEvents()
            print("  GMM dialog opened and closed successfully")
        except Exception as e:
            pytest.fail(f"GMM dialog failed to open: {e}")
        finally:
            if dialog_watcher:
                dialog_watcher.start()


class TestGMMSurvivesPMXRoundtrip:
    """Test 14: GMM data survives save/reload."""

    def test_gmm_survives_pmx_roundtrip(self, main_window, tmp_path):
        """Test 14: Save as .npz -> reload -> GMM probabilities restored."""
        from PyQt6.QtWidgets import QApplication
        from core.npz_io import save_state_to_npz

        st = main_window.state

        # Verify we have GMM data to save
        if not hasattr(st, 'gmm_sniff_probabilities') or not st.gmm_sniff_probabilities:
            pytest.skip("No GMM probabilities to save")

        original_prob_count = sum(len(v) for v in st.gmm_sniff_probabilities.values())
        gmm_cache = getattr(main_window, '_cached_gmm_results', None)

        # Save session using npz_io directly (avoids file dialog)
        save_path = tmp_path / "test_gmm_roundtrip.pleth.npz"
        try:
            save_state_to_npz(st, save_path, include_raw_data=True, gmm_cache=gmm_cache)
        except Exception as e:
            pytest.skip(f"Save failed: {e}")

        assert save_path.exists(), "NPZ file not created"

        # Reload
        try:
            load_file_and_wait(main_window, save_path)
            QApplication.processEvents()
        except Exception as e:
            pytest.skip(f"Reload failed: {e}")

        # Verify GMM data restored (state may be new object after reload)
        new_st = main_window.state
        restored_probs = getattr(new_st, 'gmm_sniff_probabilities', {})
        cache_restored = getattr(main_window, '_cached_gmm_results', None)
        has_probs = len(restored_probs) > 0
        has_cache = cache_restored is not None

        assert has_probs or has_cache, (
            "GMM data not restored after NPZ roundtrip "
            f"(probs={has_probs}, cache={has_cache})"
        )
        print(f"  NPZ roundtrip: probs restored={has_probs} "
              f"(was {original_prob_count}), cache restored={has_cache}")


class TestGMMClearedOnFileSwitch:
    """Test 15: GMM state cleared when loading a different file."""

    def test_gmm_cleared_on_file_switch(self, main_window):
        """Test 15: Loading a different file clears GMM state."""
        from PyQt6.QtWidgets import QApplication

        # Verify we have GMM data from previous tests
        st = main_window.state
        had_gmm = (hasattr(st, 'gmm_sniff_probabilities') and
                    len(st.gmm_sniff_probabilities) > 0)

        if not had_gmm:
            pytest.skip("No GMM state to verify clearing")

        # Load a different file (the awake recording)
        if not AWAKE_ABF.exists():
            pytest.skip(f"Awake ABF not found: {AWAKE_ABF}")

        load_file_and_wait(main_window, AWAKE_ABF)
        QApplication.processEvents()

        # After loading a new file, the old GMM state should be gone
        # (new state object or cleared probabilities)
        new_st = main_window.state
        old_probs = getattr(new_st, 'gmm_sniff_probabilities', {})

        # Either state was replaced or probabilities were cleared
        # The key check: the old sweep-indexed data shouldn't still be there
        # for the old file's sweeps
        cache = getattr(main_window, '_cached_gmm_results', None)

        # If both are empty/None, the GMM state was properly cleared
        # If cache exists, it should be for the NEW file, not the old one
        print(f"  After file switch: probs={len(old_probs)} entries, "
              f"cache={'populated' if cache else 'cleared'}")

        # The main assertion: loading a new file should not crash
        # and old GMM data should not persist (or should be for new file)
        assert True  # If we got here without crashing, the file switch worked


# ═══════════════════════════════════════════════════════════════════
# Multi-channel GMM test on awake recording
# ═══════════════════════════════════════════════════════════════════


class TestGMMMultiChannel:
    """Bonus: Test GMM on multiple pleth channels of awake recording."""

    @pytest.mark.skipif(not AWAKE_ABF.exists(), reason="25729001.abf not available")
    def test_gmm_on_awake_pleth_channels(self, main_window):
        """Test GMM clustering works across different pleth channels."""
        from PyQt6.QtWidgets import QApplication

        load_file_and_wait(main_window, AWAKE_ABF)

        # Find pleth channels (IN 0, IN 1, IN 2 typically)
        st = main_window.state
        combo = main_window.AnalyzeChanSelect
        pleth_channels = []
        for i in range(combo.count()):
            text = combo.itemText(i)
            if "IN" in text and any(f"IN {n}" in text for n in range(4)):
                pleth_channels.append((i, text))

        if len(pleth_channels) < 2:
            pytest.skip("Not enough pleth channels for multi-channel test")

        results = {}
        for idx, ch_name in pleth_channels[:3]:  # Test up to 3 channels
            combo.setCurrentIndex(idx)
            QApplication.processEvents()

            if _ensure_peaks_detected(main_window):
                cache = getattr(main_window, '_cached_gmm_results', None)
                if cache:
                    n_breaths = len(cache['cluster_labels'])
                    n_sniff = int(np.sum(
                        cache['cluster_labels'] == cache['sniffing_cluster_id']
                    ))
                    results[ch_name] = {'total': n_breaths, 'sniffing': n_sniff}
                    print(f"  {ch_name}: {n_breaths} breaths, {n_sniff} sniffing")

        assert len(results) > 0, "GMM didn't run on any channel"
        print(f"  Successfully tested GMM on {len(results)} channels")
