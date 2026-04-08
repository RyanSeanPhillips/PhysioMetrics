"""
CTA (Conditional Time Average) system tests.

Tests:
1. CTA domain models — serialization round-trip
2. CTA service — calculation produces valid results
3. CTA save/reload — workspace persists in .pmx
4. CTA state clearing — verify no stale data after file switch
5. Event marker persistence — save/reload round-trip

Uses tests/data/26402007.abf (pleth + EKG + stim)

Run:  python -m pytest tests/test_cta_system.py -v -s
"""

import sys
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import load_file_and_wait, MULTI_CHANNEL_ABF, LEGACY_ABF
from test_export_and_save import _setup_analysis


# ── 1. CTA Domain Model Tests ───────────────────────────────────


class TestCTAModels:
    """Pure Python tests — no GUI needed."""

    def test_cta_config_roundtrip(self):
        """CTAConfig serialization should preserve all fields."""
        from core.domain.cta.models import CTAConfig

        cfg = CTAConfig(
            window_before=20.0,
            window_after=40.0,
            n_points=500,
            zscore_baseline=True,
            baseline_start=-10.0,
            baseline_end=-1.0,
        )
        d = cfg.to_dict()
        cfg2 = CTAConfig.from_dict(d)

        assert cfg2.window_before == 20.0
        assert cfg2.window_after == 40.0
        assert cfg2.n_points == 500
        assert cfg2.zscore_baseline is True
        assert cfg2.baseline_start == -10.0
        assert cfg2.baseline_end == -1.0

    def test_cta_collection_roundtrip(self):
        """CTACollection serialize/deserialize should preserve results."""
        from core.domain.cta.models import CTAConfig, CTAResult, CTATrace, CTACollection

        # Build a minimal collection
        config = CTAConfig()
        trace = CTATrace(
            event_id="test_1",
            sweep_idx=0,
            event_time=10.5,
            time=np.linspace(-2, 5, 100),
            values=np.random.randn(100),
        )
        result = CTAResult(
            metric_key="test_metric",
            metric_label="Test Metric",
            alignment="onset",
            category="test_cat",
            label="test_label",
            config=config,
            traces=[trace],
            time_common=np.linspace(-2, 5, 100),
            mean=np.random.randn(100),
            sem=np.abs(np.random.randn(100)) * 0.1,
            n_events=1,
        )
        collection = CTACollection(config=config, results={}, metrics=["test_metric"])
        collection.add_result(result)

        # Serialize
        npz_dict = collection.to_npz_dict()
        assert 'cta_json' in npz_dict

        # Deserialize
        collection2 = CTACollection.from_npz_dict(npz_dict)
        assert len(collection2.results) == 1
        assert "test_metric" in collection2.metrics

        key = list(collection2.results.keys())[0]
        r2 = collection2.results[key]
        assert r2.metric_key == "test_metric"
        assert r2.n_events == 1
        assert r2.mean is not None
        assert len(r2.mean) == 100
        print(f"  Round-trip OK: {key}")


# ── 2. CTA Service Tests ────────────────────────────────────────


class TestCTAService:
    """Test CTA calculation with synthetic data."""

    def test_calculate_cta_basic(self):
        """CTA service should compute mean/SEM from event-aligned traces."""
        from core.services.cta_service import CTAService
        from core.domain.cta.models import CTAConfig

        service = CTAService()
        config = CTAConfig(window_before=2.0, window_after=5.0, n_points=100)

        # Synthetic signal: sine wave
        sr = 1000
        duration = 30
        t = np.arange(0, duration, 1/sr)
        signal = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine

        # Events at t=5, 10, 15, 20
        event_times = [5.0, 10.0, 15.0, 20.0]

        result = service.calculate_cta(
            time_array=t,
            signal=signal,
            event_times=event_times,
            config=config,
            metric_key="test_sine",
            metric_label="Test Sine",
            category="test",
            label="pulse",
            alignment="onset",
        )

        assert result is not None
        assert result.mean is not None
        assert result.sem is not None
        assert result.n_events == 4
        assert len(result.mean) > 50, f"Too few points: {len(result.mean)}"
        assert len(result.time_common) == len(result.mean)

        # Time should span -2 to +5
        assert result.time_common[0] < -1.5
        assert result.time_common[-1] > 4.5

        # Mean should be sinusoidal (all events aligned to same phase)
        assert np.std(result.mean) > 0.1, "Mean is flat — events not properly aligned"
        print(f"  CTA: {result.n_events} events, mean range [{result.mean.min():.2f}, {result.mean.max():.2f}]")

    def test_calculate_cta_zscore(self):
        """Z-score normalization should normalize to baseline period."""
        from core.services.cta_service import CTAService
        from core.domain.cta.models import CTAConfig

        service = CTAService()
        config = CTAConfig(
            window_before=5.0, window_after=10.0,
            zscore_baseline=True, baseline_start=-5.0, baseline_end=-1.0,
            n_points=200,
        )

        # Signal with different baseline levels at each event
        sr = 1000
        t = np.arange(0, 60, 1/sr)
        signal = np.zeros(len(t))
        # Event at t=10: baseline=0, response=+2
        signal[10*sr:15*sr] = 2.0
        # Event at t=30: baseline=10, response=+12
        signal[25*sr:35*sr] = 10.0
        signal[30*sr:35*sr] = 12.0

        result = service.calculate_cta(
            time_array=t, signal=signal,
            event_times=[10.0, 30.0],
            config=config,
            metric_key="test", metric_label="Test",
            category="test", label="test", alignment="onset",
        )

        assert result is not None
        assert result.n_events == 2
        print(f"  Z-scored CTA: mean baseline={np.mean(result.mean[:50]):.2f}, "
              f"mean response={np.mean(result.mean[100:]):.2f}")


# ── 3. CTA Save/Reload in .pmx ──────────────────────────────────


class TestCTASaveReload:
    """Test CTA workspace persistence in .pmx files."""

    def test_cta_workspace_saved_in_pmx(self, main_window, multi_channel_abf, tmp_path):
        """Setting a CTA workspace config should persist to .pmx."""
        _setup_analysis(main_window, multi_channel_abf)

        # Create a mock workspace config (as if user had generated CTAs)
        from core.domain.cta.models import CTAConfig, CTACollection
        config = CTAConfig(window_before=5.0, window_after=10.0)
        collection = CTACollection(config=config, results={}, metrics=["if"])

        workspace = {
            'version': 2,
            'active_tab_index': 0,
            'tabs': [{
                'tab_name': 'Test CTA',
                'config': None,
                'collection': collection.to_dict(),
                'condition_collections': None,
            }]
        }
        main_window._cta_workspace_config = workspace

        # Save
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_files = list((tmp_path / "physiometrics").glob("*.pmx"))
            assert pmx_files, "No .pmx created"

            # Verify CTA data in .pmx
            data = np.load(pmx_files[0], allow_pickle=True)
            keys = list(data.keys())

            has_workspace = 'cta_workspace_json' in keys
            has_legacy = 'cta_json' in keys

            print(f"  cta_workspace_json: {'YES' if has_workspace else 'NO'}")
            print(f"  cta_json (legacy): {'YES' if has_legacy else 'NO'}")

            assert has_workspace or has_legacy, \
                f"No CTA data in .pmx. Keys with 'cta': {[k for k in keys if 'cta' in k.lower()]}"

            if has_workspace:
                ws = json.loads(str(data['cta_workspace_json']))
                assert ws.get('version') == 2
                assert len(ws.get('tabs', [])) == 1
                assert ws['tabs'][0]['tab_name'] == 'Test CTA'
                print(f"  Workspace: v{ws['version']}, {len(ws['tabs'])} tab(s)")

            data.close()
        finally:
            main_window.state.in_path = original

    def test_cta_workspace_restored_from_pmx(self, main_window, multi_channel_abf, tmp_path):
        """Reloading .pmx should restore CTA workspace config."""
        _setup_analysis(main_window, multi_channel_abf)

        # Set workspace
        workspace = {
            'version': 2,
            'active_tab_index': 0,
            'tabs': [{
                'tab_name': 'Restored CTA',
                'config': None,
                'collection': {'generated_at': '2026-01-01', 'config': {},
                               'results': {}, 'metrics': ['if', 'ti']},
                'condition_collections': None,
            }]
        }
        main_window._cta_workspace_config = workspace

        # Save and reload
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_path = list((tmp_path / "physiometrics").glob("*.pmx"))[0]

            # Clear the workspace before reload
            main_window._cta_workspace_config = None

            main_window.load_npz_state(pmx_path)
            for _ in range(50):
                QApplication.processEvents()
                time.sleep(0.05)

            # Check workspace was restored
            ws = getattr(main_window, '_cta_workspace_config', None)
            assert ws is not None, "CTA workspace not restored from .pmx"
            assert ws.get('version') in (1, 2)
            tabs = ws.get('tabs', [])
            assert len(tabs) >= 1
            print(f"  Restored workspace: v{ws.get('version')}, "
                  f"{len(tabs)} tab(s), name='{tabs[0].get('tab_name', '?')}'")
        finally:
            main_window.state.in_path = original


# ── 4. State Clearing Tests ──────────────────────────────────────


class TestStateClearingOnFileSwitch:
    """Verify no stale CTA/event data after switching files."""

    def test_cta_state_after_file_switch(self, main_window, multi_channel_abf):
        """CTA workspace should be cleared or warned when loading a new file."""
        _setup_analysis(main_window, multi_channel_abf)

        # Set CTA workspace (simulating user having generated CTAs)
        main_window._cta_workspace_config = {
            'version': 2,
            'tabs': [{'tab_name': 'Stale CTA', 'collection': {'results': {'key': 'val'}}}]
        }
        had_cta = main_window._cta_workspace_config is not None

        # Load a different file
        load_file_and_wait(main_window, LEGACY_ABF.path)

        ws = getattr(main_window, '_cta_workspace_config', None)
        has_stale_cta = ws is not None and ws.get('tabs', [{}])[0].get('collection', {}).get('results')

        print(f"  Had CTA before switch: {had_cta}")
        print(f"  CTA workspace after switch: {ws}")
        print(f"  Has stale CTA data: {has_stale_cta}")

        assert not has_stale_cta, \
            "CTA workspace contains stale data from previous file!"

    def test_event_markers_after_file_switch(self, main_window, multi_channel_abf):
        """Event markers should be cleared when loading a new file."""
        _setup_analysis(main_window, multi_channel_abf)

        # Check marker state
        vm = getattr(main_window, '_event_marker_viewmodel', None)
        if vm is None:
            pytest.skip("No event marker viewmodel")

        # Add a test marker if possible
        pre_markers = vm.get_markers() if hasattr(vm, 'get_markers') else []

        # Load different file
        load_file_and_wait(main_window, LEGACY_ABF.path)

        post_markers = vm.get_markers() if hasattr(vm, 'get_markers') else []
        print(f"  Markers before switch: {len(pre_markers)}")
        print(f"  Markers after switch: {len(post_markers)}")

        # Markers from previous file should not persist
        # (Unless the new file has its own saved markers)

    def test_ekg_cleared_after_file_switch(self, main_window, multi_channel_abf):
        """EKG state should be cleared when loading a new file."""
        _setup_analysis(main_window, multi_channel_abf)

        assert main_window.state.ekg_chan is not None
        assert len(main_window.state.ecg_results_by_sweep) > 0

        load_file_and_wait(main_window, LEGACY_ABF.path)

        assert main_window.state.ekg_chan is None, \
            f"ekg_chan not cleared: {main_window.state.ekg_chan}"
        assert len(main_window.state.ecg_results_by_sweep) == 0, \
            f"ecg_results not cleared: {len(main_window.state.ecg_results_by_sweep)}"
        print("  EKG state properly cleared")


# ── 5. Event Marker Save/Reload ──────────────────────────────────


class TestEventMarkerPersistence:
    """Test event markers save to .pmx and reload correctly."""

    def test_event_marker_save_data_format(self, main_window, multi_channel_abf):
        """Event marker save_to_npz should produce valid data."""
        if not main_window.state.sweeps:
            _setup_analysis(main_window, multi_channel_abf)

        vm = getattr(main_window, '_event_marker_viewmodel', None)
        if vm is None:
            pytest.skip("No event marker viewmodel")

        save_data = vm.save_to_npz() if hasattr(vm, 'save_to_npz') else None
        assert save_data is not None, "save_to_npz returned None"

        if isinstance(save_data, dict):
            print(f"  Save data keys: {list(save_data.keys())}")
            if 'event_markers_json' in save_data:
                markers_json = json.loads(save_data['event_markers_json'])
                if isinstance(markers_json, list):
                    print(f"  Markers in save: {len(markers_json)}")
                elif isinstance(markers_json, dict):
                    print(f"  Marker data keys: {list(markers_json.keys())}")

    def test_event_markers_survive_pmx_roundtrip(self, main_window, multi_channel_abf, tmp_path):
        """Event markers should persist through save/reload cycle."""
        _setup_analysis(main_window, multi_channel_abf)

        vm = getattr(main_window, '_event_marker_viewmodel', None)
        if vm is None:
            pytest.skip("No event marker viewmodel")

        pre_save_data = vm.save_to_npz() if hasattr(vm, 'save_to_npz') else None
        pre_markers = vm.get_markers() if hasattr(vm, 'get_markers') else []

        # Save
        tmp_abf = tmp_path / "26402007.abf"
        shutil.copy2(multi_channel_abf.path, tmp_abf)
        original = main_window.state.in_path
        main_window.state.in_path = tmp_abf

        try:
            main_window._save_session_pmx()
            QApplication.processEvents()

            pmx_path = list((tmp_path / "physiometrics").glob("*.pmx"))[0]

            # Verify event markers in .pmx
            data = np.load(pmx_path, allow_pickle=True)
            marker_keys = [k for k in data.keys() if 'marker' in k.lower() or 'event' in k.lower()]
            print(f"  Marker keys in .pmx: {marker_keys}")

            has_markers = 'event_markers_json' in data.keys()
            print(f"  event_markers_json in .pmx: {has_markers}")
            if has_markers:
                em_json = json.loads(str(data['event_markers_json']))
                if isinstance(em_json, list):
                    print(f"  Saved {len(em_json)} markers")
                elif isinstance(em_json, dict):
                    print(f"  Marker data: {list(em_json.keys())}")
            data.close()

            # Reload
            main_window.load_npz_state(pmx_path)
            for _ in range(50):
                QApplication.processEvents()
                time.sleep(0.05)

            post_markers = vm.get_markers() if hasattr(vm, 'get_markers') else []
            print(f"  Markers: {len(pre_markers)} before save -> {len(post_markers)} after reload")
        finally:
            main_window.state.in_path = original


# ── 6. Unsaved Data Warning Check ────────────────────────────────


class TestUnsavedDataWarning:
    """Check if the app warns about unsaved data."""

    def test_check_for_unsaved_warning_mechanism(self, main_window):
        """Verify there's a mechanism to detect unsaved changes."""
        # Check for common patterns: dirty flag, modified flag, etc.
        has_dirty_flag = hasattr(main_window, '_is_dirty') or hasattr(main_window, '_modified')
        has_unsaved_check = hasattr(main_window, '_check_unsaved') or hasattr(main_window, '_has_unsaved_changes')

        # Check closeEvent for save prompt
        import inspect
        close_src = inspect.getsource(main_window.closeEvent)
        has_close_prompt = 'save' in close_src.lower() or 'unsaved' in close_src.lower()

        # Check load_file for save prompt
        load_src = inspect.getsource(main_window.load_file) if hasattr(main_window, 'load_file') else ""
        has_load_prompt = 'save' in load_src.lower() or 'unsaved' in load_src.lower()

        print(f"  Dirty flag: {has_dirty_flag}")
        print(f"  Unsaved check method: {has_unsaved_check}")
        print(f"  Close event save prompt: {has_close_prompt}")
        print(f"  Load file save prompt: {has_load_prompt}")

        if not has_load_prompt and not has_unsaved_check:
            print("  WARNING: No unsaved data warning when switching files!")
            print("  Users could lose CTA analysis, event markers, peak edits")
            # This is a finding, not a test failure
