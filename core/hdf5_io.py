"""
HDF5 I/O Module - Save/Load PhysioMetrics analysis sessions in HDF5 format

This module provides parallel HDF5 save/load for PhysioMetrics session files,
mirroring the existing NPZ format in core/npz_io.py. HDF5 offers faster partial
reads (metadata-only), chunked compression for large arrays, and a hierarchical
group structure that maps cleanly to the session schema.

Schema version: 3 (HDF5-native layout)
"""

from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from core.state import AppState
from core.npz_io import OriginalFileNotFoundError


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 3
_CHUNK_THRESHOLD = 10_000  # Only chunk arrays longer than this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunked_kwargs(arr: np.ndarray) -> dict:
    """Return h5py dataset creation kwargs with chunking + lzf for large arrays.

    lzf is ~10x faster than gzip with reasonable compression ratio.
    """
    if arr.size > _CHUNK_THRESHOLD:
        return dict(chunks=True, compression='lzf')
    return {}


def _safe_write_dataset(group, name: str, data, **kwargs):
    """Write a dataset, converting object dtypes to JSON strings if needed."""
    import h5py
    if data is None:
        return
    arr = np.asarray(data)
    if arr.dtype == object:
        # Object arrays can't be stored natively in HDF5 — serialize as JSON
        try:
            json_str = json.dumps(arr.tolist())
            group.create_dataset(name + '_json', data=json_str,
                                 dtype=h5py.string_dtype())
        except (TypeError, ValueError):
            # Last resort: pickle to string
            import pickle, base64
            b64 = base64.b64encode(pickle.dumps(arr)).decode('ascii')
            group.create_dataset(name + '_pickle_b64', data=b64,
                                 dtype=h5py.string_dtype())
    else:
        group.create_dataset(name, data=arr, **kwargs)


def _write_string_dataset(group, name: str, value: str):
    """Write a Python string as a dataset — compressed for large strings."""
    import h5py
    encoded = value.encode('utf-8')
    if len(encoded) > 10_000:
        # Large strings: store as compressed byte array (much smaller on disk)
        arr = np.frombuffer(encoded, dtype=np.uint8)
        group.create_dataset(name, data=arr, compression='lzf')
        group[name].attrs['_encoding'] = 'utf8_bytes'
    else:
        group.create_dataset(name, data=value, dtype=h5py.string_dtype())


def _read_string(dataset) -> str:
    """Read a string dataset, handling bytes, numpy arrays, and compressed byte arrays."""
    # Check if this is a compressed byte array
    if dataset.attrs.get('_encoding') == 'utf8_bytes':
        return dataset[()].tobytes().decode('utf-8')
    val = dataset[()]
    if isinstance(val, bytes):
        return val.decode('utf-8')
    if isinstance(val, np.ndarray):
        val = val.item()
        if isinstance(val, bytes):
            return val.decode('utf-8')
    return str(val)


def _safe_attr_str(val, default='unknown') -> str:
    """Convert a value to a plain Python string, handling numpy scalars."""
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode('utf-8')
    if hasattr(val, 'item'):
        return str(val.item())
    return str(val)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_state_to_hdf5(
    state: AppState,
    hdf5_path: Path,
    include_raw_data: bool = False,
    gmm_cache: Optional[Dict] = None,
    app_settings: Optional[Dict] = None,
    event_markers: Optional[Dict] = None,
    cta_data: Optional[Dict] = None,
    channel_config: Optional[Dict] = None,
) -> None:
    """
    Save complete analysis state to an HDF5 file.

    Mirrors every field category saved by ``save_state_to_npz`` in npz_io.py,
    using HDF5 groups instead of flat NPZ keys.

    Args:
        state: AppState instance to save.
        hdf5_path: Destination path (e.g. ``session.pleth.h5``).
        include_raw_data: If True, embed raw sweeps (larger file, portable).
        gmm_cache: Optional GMM cache dict to preserve cluster assignments.
        app_settings: Optional dict with filter_order, zscore, notch, etc.
        event_markers: Optional dict from EventMarkerService.to_npz_dict().
        cta_data: Optional CTA workspace dict.
        channel_config: Optional channel visibility/type config dict.
    """
    import h5py

    if state.in_path is None:
        raise ValueError("Cannot save state: no data file loaded (in_path is None)")
    if state.analyze_chan is None:
        raise ValueError("Cannot save state: no channel selected (analyze_chan is None)")

    hdf5_path = Path(hdf5_path)
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(hdf5_path), 'w') as f:

        # ================================================================
        # ROOT ATTRIBUTES
        # ================================================================
        f.attrs['schema_version'] = _SCHEMA_VERSION
        f.attrs['app_version'] = '1.1.0'
        f.attrs['saved_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['original_file_path'] = str(state.in_path)

        # ================================================================
        # /metadata/
        # ================================================================
        meta = f.create_group('metadata')
        meta.attrs['analyze_chan'] = state.analyze_chan if state.analyze_chan else 'None'
        meta.attrs['stim_chan'] = state.stim_chan if state.stim_chan else 'None'
        meta.attrs['event_channel'] = state.event_channel if state.event_channel else 'None'
        meta.attrs['sr_hz'] = state.sr_hz if state.sr_hz is not None else 0.0

        # Multi-file info
        if state.file_info:
            file_info_serializable = []
            for fi in state.file_info:
                file_info_serializable.append({
                    'path': str(fi['path']),
                    'sweep_start': int(fi['sweep_start']),
                    'sweep_end': int(fi['sweep_end']),
                    'padded': fi.get('padded', False),
                    'original_samples': fi.get('original_samples', 0),
                    'padded_samples': fi.get('padded_samples', 0),
                })
            _write_string_dataset(meta, 'file_info_json', json.dumps(file_info_serializable))

        # Channel config
        if channel_config:
            _write_string_dataset(meta, 'channel_config_json', json.dumps(channel_config))

        # ================================================================
        # /filters/
        # ================================================================
        filt = f.create_group('filters')
        filt.attrs['use_low'] = bool(state.use_low)
        filt.attrs['use_high'] = bool(state.use_high)
        filt.attrs['use_mean_sub'] = bool(state.use_mean_sub)
        filt.attrs['use_invert'] = bool(state.use_invert)
        filt.attrs['low_hz'] = float(state.low_hz) if state.low_hz else 0.0
        filt.attrs['high_hz'] = float(state.high_hz) if state.high_hz else 0.0
        filt.attrs['mean_val'] = float(state.mean_val)

        # ================================================================
        # /navigation/
        # ================================================================
        nav = f.create_group('navigation')
        nav.attrs['sweep_idx'] = int(state.sweep_idx)
        nav.attrs['window_start_s'] = float(state.window_start_s)
        nav.attrs['window_dur_s'] = float(state.window_dur_s)

        # ================================================================
        # /raw_data/ (optional)
        # ================================================================
        # Always save time array and sample rate
        f.create_dataset('t', data=state.t, **_chunked_kwargs(state.t))
        f.attrs['sr_hz'] = state.sr_hz if state.sr_hz is not None else 0.0

        if include_raw_data:
            raw = f.create_group('raw_data')
            chan_names_arr = np.array(state.channel_names, dtype=object)
            raw.create_dataset('channel_names', data=chan_names_arr,
                               dtype=h5py.string_dtype())
            for chan_name, sweep_data in state.sweeps.items():
                safe_name = chan_name.replace(' ', '_').replace('/', '_')
                raw.create_dataset(f'sweeps_{safe_name}', data=sweep_data,
                                   **_chunked_kwargs(sweep_data))

        # ================================================================
        # /peaks/{sweep_idx}/
        # ================================================================
        sweep_indices = sorted(state.peaks_by_sweep.keys())
        peaks_grp = f.create_group('peaks')
        peaks_grp.create_dataset('sweep_indices', data=np.array(sweep_indices, dtype=int))

        for sidx in sweep_indices:
            arr = state.peaks_by_sweep[sidx]
            peaks_grp.create_dataset(str(sidx), data=arr, **_chunked_kwargs(arr))

        # ================================================================
        # /all_peaks/{sweep_idx}/ — master list with labels + _ro arrays
        # ================================================================
        if hasattr(state, 'all_peaks_by_sweep') and state.all_peaks_by_sweep:
            all_peaks_grp = f.create_group('all_peaks')
            ap_sweep_indices = sorted(state.all_peaks_by_sweep.keys())
            all_peaks_grp.create_dataset('sweep_indices',
                                         data=np.array(ap_sweep_indices, dtype=int))

            for sidx in ap_sweep_indices:
                sg = all_peaks_grp.create_group(str(sidx))
                apd = state.all_peaks_by_sweep[sidx]

                sg.create_dataset('indices', data=apd['indices'],
                                  **_chunked_kwargs(apd['indices']))
                sg.create_dataset('labels', data=apd['labels'],
                                  **_chunked_kwargs(apd['labels']))
                sg.create_dataset('label_source', data=apd['label_source'],
                                  dtype=h5py.string_dtype())

                # Breath type classification
                if 'breath_type_class' in apd and apd['breath_type_class'] is not None:
                    sg.create_dataset('breath_type_class', data=apd['breath_type_class'],
                                      **_chunked_kwargs(apd['breath_type_class']))

                # Read-only classifier predictions
                _ro_keys = [
                    'labels_threshold_ro', 'labels_xgboost_ro', 'labels_rf_ro', 'labels_mlp_ro',
                    'gmm_class_ro', 'eupnea_sniff_xgboost_ro', 'eupnea_sniff_rf_ro', 'eupnea_sniff_mlp_ro',
                    'sigh_xgboost_ro', 'sigh_rf_ro', 'sigh_mlp_ro',
                    'eupnea_sniff_source',
                    'prominences',
                ]
                for ro_key in _ro_keys:
                    if ro_key in apd and apd[ro_key] is not None:
                        val = np.asarray(apd[ro_key])
                        if val.dtype == object or ro_key == 'eupnea_sniff_source':
                            sg.create_dataset(ro_key, data=val,
                                              dtype=h5py.string_dtype())
                        else:
                            sg.create_dataset(ro_key, data=val,
                                              **_chunked_kwargs(val))

        # ================================================================
        # /peak_metrics/{sweep_idx}_json
        # ================================================================
        if hasattr(state, 'peak_metrics_by_sweep') and state.peak_metrics_by_sweep:
            pm_grp = f.create_group('peak_metrics')
            pm_indices = sorted(state.peak_metrics_by_sweep.keys())
            pm_grp.create_dataset('sweep_indices',
                                  data=np.array(pm_indices, dtype=int))
            for sidx in pm_indices:
                _write_string_dataset(pm_grp, f'{sidx}_json',
                                      json.dumps(state.peak_metrics_by_sweep[sidx]))

        # ================================================================
        # /current_peak_metrics/{sweep_idx}_json
        # ================================================================
        if hasattr(state, 'current_peak_metrics_by_sweep') and state.current_peak_metrics_by_sweep:
            cpm_grp = f.create_group('current_peak_metrics')
            cpm_indices = sorted(state.current_peak_metrics_by_sweep.keys())
            cpm_grp.create_dataset('sweep_indices',
                                   data=np.array(cpm_indices, dtype=int))
            for sidx in cpm_indices:
                _write_string_dataset(cpm_grp, f'{sidx}_json',
                                      json.dumps(state.current_peak_metrics_by_sweep[sidx]))

        # ================================================================
        # /breaths/{sweep_idx}/ — native int datasets (NO JSON)
        # ================================================================
        breath_sweep_indices = sorted(state.breath_by_sweep.keys())
        br_grp = f.create_group('breaths')
        br_grp.create_dataset('sweep_indices',
                              data=np.array(breath_sweep_indices, dtype=int))

        for sidx in breath_sweep_indices:
            bd = state.breath_by_sweep[sidx]
            sg = br_grp.create_group(str(sidx))
            for key in ('onsets', 'offsets', 'expmins', 'expoffs'):
                val = bd.get(key, [])
                arr = np.asarray(val, dtype=int) if not isinstance(val, np.ndarray) else val
                sg.create_dataset(key, data=arr, **_chunked_kwargs(arr))

        # ================================================================
        # /sighs/{sweep_idx}
        # ================================================================
        sigh_sweep_indices = sorted(state.sigh_by_sweep.keys())
        sigh_grp = f.create_group('sighs')
        sigh_grp.create_dataset('sweep_indices',
                                data=np.array(sigh_sweep_indices, dtype=int))
        for sidx in sigh_sweep_indices:
            arr = state.sigh_by_sweep[sidx]
            sigh_grp.create_dataset(str(sidx), data=arr, **_chunked_kwargs(arr))

        # ================================================================
        # /omissions/
        # ================================================================
        om_grp = f.create_group('omissions')

        # Omitted sweeps
        om_sweeps = np.array(sorted(state.omitted_sweeps), dtype=int) \
            if state.omitted_sweeps else np.array([], dtype=int)
        om_grp.create_dataset('sweeps', data=om_sweeps)

        # Omitted points (per-sweep)
        op_indices = sorted(state.omitted_points.keys())
        om_grp.create_dataset('points_indices',
                              data=np.array(op_indices, dtype=int))
        if op_indices:
            pts_grp = om_grp.create_group('points')
            for sidx in op_indices:
                pts_grp.create_dataset(str(sidx),
                                       data=np.array(state.omitted_points[sidx], dtype=int))

        # Omitted ranges (per-sweep, JSON)
        or_indices = sorted(state.omitted_ranges.keys())
        om_grp.create_dataset('ranges_indices',
                              data=np.array(or_indices, dtype=int))
        if or_indices:
            rng_grp = om_grp.create_group('ranges')
            for sidx in or_indices:
                _write_string_dataset(rng_grp, f'{sidx}_json',
                                      json.dumps(state.omitted_ranges[sidx]))

        # ================================================================
        # /sniff_regions/{sweep_idx}_json
        # ================================================================
        sniff_sweep_indices = sorted(state.sniff_regions_by_sweep.keys())
        sniff_grp = f.create_group('sniff_regions')
        sniff_grp.create_dataset('sweep_indices',
                                 data=np.array(sniff_sweep_indices, dtype=int))
        for sidx in sniff_sweep_indices:
            _write_string_dataset(sniff_grp, f'{sidx}_json',
                                  json.dumps(state.sniff_regions_by_sweep[sidx]))

        # ================================================================
        # /bout_annotations/{sweep_idx}_json
        # ================================================================
        bout_sweep_indices = sorted(state.bout_annotations.keys())
        bout_grp = f.create_group('bout_annotations')
        bout_grp.create_dataset('sweep_indices',
                                data=np.array(bout_sweep_indices, dtype=int))
        for sidx in bout_sweep_indices:
            _write_string_dataset(bout_grp, f'{sidx}_json',
                                  json.dumps(state.bout_annotations[sidx]))

        # ================================================================
        # /event_markers/
        # ================================================================
        if event_markers is not None:
            em_grp = f.create_group('event_markers')
            # event_markers dict contains event_markers_version (np.array) and
            # event_markers_json (str)
            if 'event_markers_version' in event_markers:
                ver = event_markers['event_markers_version']
                if isinstance(ver, np.ndarray):
                    ver = int(ver.item()) if ver.ndim == 0 else int(ver[0])
                em_grp.attrs['version'] = int(ver)
            if 'event_markers_json' in event_markers:
                _write_string_dataset(em_grp, 'json',
                                      str(event_markers['event_markers_json']))

        # ================================================================
        # /cta/
        # ================================================================
        if cta_data is not None:
            cta_grp = f.create_group('cta')
            if 'cta_version' in cta_data:
                ver = cta_data['cta_version']
                if isinstance(ver, np.ndarray):
                    ver = int(ver.item()) if ver.ndim == 0 else int(ver[0])
                cta_grp.attrs['version'] = int(ver)
            if 'cta_workspace_json' in cta_data:
                _write_string_dataset(cta_grp, 'workspace_json',
                                      str(cta_data['cta_workspace_json']))
            if 'cta_json' in cta_data:
                _write_string_dataset(cta_grp, 'legacy_json',
                                      str(cta_data['cta_json']))

        # ================================================================
        # /gmm/
        # ================================================================
        gmm_grp = f.create_group('gmm')

        # Per-sweep probabilities
        gmm_sweep_indices = sorted(state.gmm_sniff_probabilities.keys())
        gmm_grp.create_dataset('sweep_indices',
                               data=np.array(gmm_sweep_indices, dtype=int))
        if gmm_sweep_indices:
            prob_grp = gmm_grp.create_group('probabilities')
            for sidx in gmm_sweep_indices:
                arr = state.gmm_sniff_probabilities[sidx]
                _safe_write_dataset(prob_grp, str(sidx), arr,
                                    **_chunked_kwargs(np.asarray(arr) if not isinstance(arr, np.ndarray) else arr))

        # GMM cache
        if gmm_cache is not None:
            gmm_grp.attrs['has_cache'] = True
            _safe_write_dataset(gmm_grp, 'cluster_labels',
                                gmm_cache['cluster_labels'],
                                **_chunked_kwargs(np.asarray(gmm_cache['cluster_labels'])))
            _safe_write_dataset(gmm_grp, 'cluster_probabilities',
                                gmm_cache['cluster_probabilities'],
                                **_chunked_kwargs(np.asarray(gmm_cache['cluster_probabilities'])))
            gmm_grp.create_dataset('feature_matrix',
                                   data=gmm_cache['feature_matrix'],
                                   **_chunked_kwargs(gmm_cache['feature_matrix']))
            gmm_grp.attrs['sniffing_cluster_id'] = int(gmm_cache['sniffing_cluster_id'])
            _write_string_dataset(gmm_grp, 'feature_keys_json',
                                  json.dumps(gmm_cache['feature_keys']))
            breath_cycles_array = np.array(gmm_cache['breath_cycles'], dtype=int)
            gmm_grp.create_dataset('breath_cycles', data=breath_cycles_array)
        else:
            gmm_grp.attrs['has_cache'] = False

        # ================================================================
        # /app_settings/
        # ================================================================
        if app_settings is not None:
            as_grp = f.create_group('app_settings')
            as_grp.attrs['has_app_settings'] = True
            as_grp.attrs['filter_order'] = int(app_settings.get('filter_order', 4))
            as_grp.attrs['use_zscore_normalization'] = bool(
                app_settings.get('use_zscore_normalization', True))
            nl = app_settings.get('notch_filter_lower')
            nu = app_settings.get('notch_filter_upper')
            as_grp.attrs['notch_filter_lower'] = float(nl) if nl is not None else 0.0
            as_grp.attrs['notch_filter_upper'] = float(nu) if nu is not None else 0.0
            as_grp.attrs['apnea_threshold'] = float(
                app_settings.get('apnea_threshold', 0.5))
            as_grp.attrs['active_eupnea_sniff_classifier'] = str(
                app_settings.get('active_eupnea_sniff_classifier', 'gmm'))
            as_grp.attrs['active_classifier'] = str(
                app_settings.get('active_classifier', 'xgboost'))
            as_grp.attrs['active_sigh_classifier'] = str(
                app_settings.get('active_sigh_classifier', 'xgboost'))

        # ================================================================
        # /y2_metrics/
        # ================================================================
        if state.y2_metric_key:
            y2_grp = f.create_group('y2_metrics')
            y2_grp.attrs['metric_key'] = str(state.y2_metric_key)

            y2_sweep_indices = sorted(state.y2_values_by_sweep.keys())
            y2_grp.create_dataset('sweep_indices',
                                  data=np.array(y2_sweep_indices, dtype=int))
            for sidx in y2_sweep_indices:
                arr = state.y2_values_by_sweep[sidx]
                y2_grp.create_dataset(str(sidx), data=arr,
                                      **_chunked_kwargs(arr))

        # ================================================================
        # /stim/{sweep_idx}/ — onsets, offsets, spans_json, metrics_json
        # ================================================================
        stim_grp = f.create_group('stim')

        # Onsets
        stim_onset_indices = sorted(state.stim_onsets_by_sweep.keys())
        stim_grp.create_dataset('onset_sweep_indices',
                                data=np.array(stim_onset_indices, dtype=int))
        for sidx in stim_onset_indices:
            arr = state.stim_onsets_by_sweep[sidx]
            stim_grp.create_dataset(f'onsets_{sidx}', data=arr,
                                    **_chunked_kwargs(arr))

        # Offsets
        stim_offset_indices = sorted(state.stim_offsets_by_sweep.keys())
        stim_grp.create_dataset('offset_sweep_indices',
                                data=np.array(stim_offset_indices, dtype=int))
        for sidx in stim_offset_indices:
            arr = state.stim_offsets_by_sweep[sidx]
            stim_grp.create_dataset(f'offsets_{sidx}', data=arr,
                                    **_chunked_kwargs(arr))

        # Spans (JSON)
        stim_spans_indices = sorted(state.stim_spans_by_sweep.keys())
        stim_grp.create_dataset('spans_sweep_indices',
                                data=np.array(stim_spans_indices, dtype=int))
        for sidx in stim_spans_indices:
            _write_string_dataset(stim_grp, f'spans_{sidx}_json',
                                  json.dumps(state.stim_spans_by_sweep[sidx]))

        # Metrics (JSON)
        stim_metrics_indices = sorted(state.stim_metrics_by_sweep.keys())
        stim_grp.create_dataset('metrics_sweep_indices',
                                data=np.array(stim_metrics_indices, dtype=int))
        for sidx in stim_metrics_indices:
            _write_string_dataset(stim_grp, f'metrics_{sidx}_json',
                                  json.dumps(state.stim_metrics_by_sweep[sidx]))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_state_from_hdf5(
    hdf5_path: Path,
    reload_raw_data: bool = True,
    alternative_data_path: Path = None,
) -> Tuple[AppState, bool, Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load complete analysis state from an HDF5 session file.

    Returns the exact same 7-element tuple as ``load_state_from_npz``:

        (state, raw_data_loaded, gmm_cache, app_settings, event_markers,
         cta_data, channel_config)

    Args:
        hdf5_path: Path to the ``.pleth.h5`` file.
        reload_raw_data: If True, reload raw data from the original recording.
        alternative_data_path: Override path for the original data file.

    Raises:
        FileNotFoundError: If the HDF5 file does not exist.
        ValueError: If the file is corrupt or unreadable.
    """
    import h5py

    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    try:
        f = h5py.File(str(hdf5_path), 'r')
    except Exception as e:
        raise ValueError(f"Failed to open HDF5 file: {e}")

    try:
        return _load_from_handle(f, hdf5_path, reload_raw_data, alternative_data_path)
    finally:
        f.close()


def _load_from_handle(
    f,
    hdf5_path: Path,
    reload_raw_data: bool,
    alternative_data_path: Optional[Path],
) -> Tuple[AppState, bool, Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Core loader — operates on an open h5py.File handle."""

    state = AppState()

    # ================================================================
    # ROOT ATTRIBUTES / METADATA
    # ================================================================
    original_file_path_str = _safe_attr_str(f.attrs.get('original_file_path'))
    original_file_path = Path(original_file_path_str)

    # ================================================================
    # LOAD RAW DATA
    # ================================================================
    raw_data_loaded = False
    data_path_to_load = alternative_data_path if alternative_data_path else original_file_path

    # Search for original file relative to HDF5 location (mirrors npz_io.py)
    if reload_raw_data and not data_path_to_load.exists():
        original_filename = original_file_path.name
        print(f"[hdf5_io] Original path not found: {original_file_path}")
        print(f"[hdf5_io] Searching for {original_filename} relative to HDF5 location...")

        search_paths = [
            hdf5_path.parent.parent / original_filename,
            hdf5_path.parent.parent.parent / original_filename,
            hdf5_path.parent / original_filename,
        ]
        for candidate in search_paths:
            if candidate.exists():
                print(f"[hdf5_io] Found file at: {candidate}")
                data_path_to_load = candidate
                break
        else:
            print(f"[hdf5_io] Could not find {original_filename} in any expected location")

    if reload_raw_data and data_path_to_load.exists():
        from core.abf_io import load_data_file

        try:
            sr, sweeps_by_ch, ch_names, t, file_metadata = load_data_file(data_path_to_load)
            state.sr_hz = sr
            state.sweeps = sweeps_by_ch
            state.channel_names = ch_names
            state.t = t
            state.in_path = data_path_to_load
            raw_data_loaded = True
        except Exception as e:
            print(f"Warning: Could not reload from {data_path_to_load}: {e}")

            # Fallback: photometry NPZ
            if data_path_to_load.suffix == '.npz':
                try:
                    from core.photometry import load_experiment_from_npz
                    phot_data = load_experiment_from_npz(data_path_to_load, experiment_index=0)
                    if phot_data and 'sweeps' in phot_data:
                        state.sr_hz = phot_data['sr_hz']
                        state.sweeps = phot_data['sweeps']
                        state.channel_names = phot_data['channel_names']
                        state.t = phot_data['t']
                        state.in_path = data_path_to_load
                        state.photometry_raw = phot_data.get('photometry_raw')
                        state.photometry_npz_path = data_path_to_load
                        raw_data_loaded = True
                        print(f"[hdf5_io] Loaded raw data from photometry NPZ: "
                              f"{len(state.channel_names)} channels")
                except Exception as e2:
                    print(f"Warning: Photometry NPZ fallback also failed: {e2}")

            if not raw_data_loaded:
                print("Attempting to use embedded data from HDF5...")

    # Check for embedded raw data
    if not raw_data_loaded and 'raw_data' in f and 't' in f:
        state.sr_hz = float(f.attrs.get('sr_hz', f['metadata'].attrs.get('sr_hz', 0.0)))
        state.t = f['t'][:]
        raw_grp = f['raw_data']
        state.channel_names = [
            _safe_attr_str(n) for n in raw_grp['channel_names'][:]
        ]
        state.sweeps = {}
        for chan_name in state.channel_names:
            safe_name = chan_name.replace(' ', '_').replace('/', '_')
            key = f'sweeps_{safe_name}'
            if key in raw_grp:
                state.sweeps[chan_name] = raw_grp[key][:]
        state.in_path = original_file_path
        raw_data_loaded = True
    elif not raw_data_loaded and 't' in f:
        # Minimal fallback: time array present but no full raw data group
        sr_val = f.attrs.get('sr_hz', 0.0)
        if 'metadata' in f:
            sr_val = f['metadata'].attrs.get('sr_hz', sr_val)
        state.sr_hz = float(sr_val)
        state.t = f['t'][:]
        state.in_path = original_file_path
        # No sweeps available - still mark as not loaded so caller can prompt

    if not raw_data_loaded:
        raise OriginalFileNotFoundError(
            original_path=original_file_path,
            npz_path=hdf5_path,
            message=(
                f"Could not load raw data:\n"
                f"- Original file not found: {original_file_path}\n"
                f"- No embedded data in HDF5 file\n"
                f"Please locate the original data file."
            ),
        )

    # ================================================================
    # MULTI-FILE INFO
    # ================================================================
    if 'metadata' in f and 'file_info_json' in f['metadata']:
        file_info_str = _read_string(f['metadata']['file_info_json'])
        file_info_list = json.loads(file_info_str)
        state.file_info = []
        for fi in file_info_list:
            state.file_info.append({
                'path': Path(fi['path']),
                'sweep_start': fi['sweep_start'],
                'sweep_end': fi['sweep_end'],
                'padded': fi.get('padded', False),
                'original_samples': fi.get('original_samples', 0),
                'padded_samples': fi.get('padded_samples', 0),
            })
    else:
        # Single file
        n_sweeps = next(iter(state.sweeps.values())).shape[1]
        state.file_info = [{
            'path': original_file_path,
            'sweep_start': 0,
            'sweep_end': n_sweeps - 1,
        }]

    # ================================================================
    # CHANNEL SELECTIONS
    # ================================================================
    if 'metadata' in f:
        meta = f['metadata']
        ac = _safe_attr_str(meta.attrs.get('analyze_chan', 'None'))
        state.analyze_chan = ac if ac != 'None' else None

        sc = _safe_attr_str(meta.attrs.get('stim_chan', 'None'))
        state.stim_chan = sc if sc != 'None' else None

        ec = _safe_attr_str(meta.attrs.get('event_channel', 'None'))
        state.event_channel = ec if ec != 'None' else None

    # ================================================================
    # FILTER SETTINGS
    # ================================================================
    if 'filters' in f:
        fg = f['filters']
        state.use_low = bool(fg.attrs.get('use_low', False))
        state.use_high = bool(fg.attrs.get('use_high', False))
        state.use_mean_sub = bool(fg.attrs.get('use_mean_sub', False))
        state.use_invert = bool(fg.attrs.get('use_invert', False))
        lhz = float(fg.attrs.get('low_hz', 0.0))
        hhz = float(fg.attrs.get('high_hz', 0.0))
        state.low_hz = lhz if lhz != 0.0 else None
        state.high_hz = hhz if hhz != 0.0 else None
        state.mean_val = float(fg.attrs.get('mean_val', 0.0))

    # ================================================================
    # NAVIGATION STATE
    # ================================================================
    if 'navigation' in f:
        ng = f['navigation']
        state.sweep_idx = int(ng.attrs.get('sweep_idx', 0))
        state.window_start_s = float(ng.attrs.get('window_start_s', 0.0))
        state.window_dur_s = float(ng.attrs.get('window_dur_s', 10.0))

    # ================================================================
    # PEAKS (per-sweep)
    # ================================================================
    if 'peaks' in f:
        pg = f['peaks']
        if 'sweep_indices' in pg:
            for sidx in pg['sweep_indices'][:]:
                key = str(int(sidx))
                if key in pg:
                    state.peaks_by_sweep[int(sidx)] = pg[key][:]

    # ================================================================
    # ALL PEAKS (master list with labels)
    # ================================================================
    if 'all_peaks' in f:
        apg = f['all_peaks']
        if 'sweep_indices' in apg:
            for sidx in apg['sweep_indices'][:]:
                key = str(int(sidx))
                if key not in apg:
                    continue
                sg = apg[key]

                all_peaks_dict = {
                    'indices': sg['indices'][:],
                    'labels': sg['labels'][:],
                    'label_source': np.array([
                        _safe_attr_str(s) for s in sg['label_source'][:]
                    ]),
                }

                # Breath type classification
                if 'breath_type_class' in sg:
                    all_peaks_dict['breath_type_class'] = sg['breath_type_class'][:]

                # Read-only classifier arrays
                _ro_keys = [
                    'labels_threshold_ro', 'labels_xgboost_ro', 'labels_rf_ro', 'labels_mlp_ro',
                    'gmm_class_ro', 'eupnea_sniff_xgboost_ro', 'eupnea_sniff_rf_ro', 'eupnea_sniff_mlp_ro',
                    'sigh_xgboost_ro', 'sigh_rf_ro', 'sigh_mlp_ro',
                    'eupnea_sniff_source', 'prominences',
                ]
                for ro_key in _ro_keys:
                    if ro_key in sg:
                        if ro_key == 'eupnea_sniff_source':
                            all_peaks_dict[ro_key] = np.array([
                                _safe_attr_str(s) for s in sg[ro_key][:]
                            ])
                        else:
                            all_peaks_dict[ro_key] = sg[ro_key][:]

                state.all_peaks_by_sweep[int(sidx)] = all_peaks_dict

    # ================================================================
    # PEAK METRICS (for ML export)
    # ================================================================
    if 'peak_metrics' in f:
        pmg = f['peak_metrics']
        if 'sweep_indices' in pmg:
            for sidx in pmg['sweep_indices'][:]:
                key = f'{int(sidx)}_json'
                if key in pmg:
                    state.peak_metrics_by_sweep[int(sidx)] = json.loads(
                        _read_string(pmg[key]))

    # ================================================================
    # CURRENT PEAK METRICS (edited, for Y2 plotting)
    # ================================================================
    if 'current_peak_metrics' in f:
        cpmg = f['current_peak_metrics']
        if 'sweep_indices' in cpmg:
            for sidx in cpmg['sweep_indices'][:]:
                key = f'{int(sidx)}_json'
                if key in cpmg:
                    state.current_peak_metrics_by_sweep[int(sidx)] = json.loads(
                        _read_string(cpmg[key]))

    # ================================================================
    # BREATH FEATURES (per-sweep) — native int arrays
    # ================================================================
    if 'breaths' in f:
        bg = f['breaths']
        if 'sweep_indices' in bg:
            for sidx in bg['sweep_indices'][:]:
                key = str(int(sidx))
                if key in bg:
                    sg = bg[key]
                    state.breath_by_sweep[int(sidx)] = {
                        'onsets': sg['onsets'][:].astype(int),
                        'offsets': sg['offsets'][:].astype(int),
                        'expmins': sg['expmins'][:].astype(int),
                        'expoffs': sg['expoffs'][:].astype(int),
                    }

    # Populate all_breaths_by_sweep (mirrors npz_io.py logic)
    if state.breath_by_sweep and not state.all_breaths_by_sweep:
        for sidx, breath_dict in state.breath_by_sweep.items():
            state.all_breaths_by_sweep[sidx] = breath_dict.copy()

    # ================================================================
    # SIGHS (per-sweep)
    # ================================================================
    if 'sighs' in f:
        sg_grp = f['sighs']
        if 'sweep_indices' in sg_grp:
            for sidx in sg_grp['sweep_indices'][:]:
                key = str(int(sidx))
                if key in sg_grp:
                    state.sigh_by_sweep[int(sidx)] = sg_grp[key][:]

    # ================================================================
    # OMISSIONS
    # ================================================================
    if 'omissions' in f:
        og = f['omissions']
        if 'sweeps' in og:
            state.omitted_sweeps = set(int(x) for x in og['sweeps'][:])

        if 'points_indices' in og:
            for sidx in og['points_indices'][:]:
                key = str(int(sidx))
                if 'points' in og and key in og['points']:
                    state.omitted_points[int(sidx)] = list(og['points'][key][:])

        if 'ranges_indices' in og:
            for sidx in og['ranges_indices'][:]:
                key = f'{int(sidx)}_json'
                if 'ranges' in og and key in og['ranges']:
                    ranges = json.loads(_read_string(og['ranges'][key]))
                    state.omitted_ranges[int(sidx)] = [tuple(r) for r in ranges]

    # ================================================================
    # SNIFFING REGIONS (per-sweep)
    # ================================================================
    if 'sniff_regions' in f:
        srg = f['sniff_regions']
        if 'sweep_indices' in srg:
            for sidx in srg['sweep_indices'][:]:
                key = f'{int(sidx)}_json'
                if key in srg:
                    regions = json.loads(_read_string(srg[key]))
                    state.sniff_regions_by_sweep[int(sidx)] = [tuple(r) for r in regions]

    # ================================================================
    # BOUT ANNOTATIONS (per-sweep)
    # ================================================================
    if 'bout_annotations' in f:
        bag = f['bout_annotations']
        if 'sweep_indices' in bag:
            for sidx in bag['sweep_indices'][:]:
                key = f'{int(sidx)}_json'
                if key in bag:
                    state.bout_annotations[int(sidx)] = json.loads(
                        _read_string(bag[key]))

    # ================================================================
    # EVENT MARKERS
    # ================================================================
    event_markers_out = None
    if 'event_markers' in f:
        emg = f['event_markers']
        em_dict = {}
        if 'version' in emg.attrs:
            em_dict['event_markers_version'] = np.array(
                int(emg.attrs['version']))
        if 'json' in emg:
            em_dict['event_markers_json'] = _read_string(emg['json'])
        if em_dict:
            event_markers_out = em_dict

    # ================================================================
    # GMM PROBABILITIES (per-sweep)
    # ================================================================
    if 'gmm' in f:
        gmg = f['gmm']
        if 'sweep_indices' in gmg:
            for sidx in gmg['sweep_indices'][:]:
                key = str(int(sidx))
                if 'probabilities' in gmg and key in gmg['probabilities']:
                    state.gmm_sniff_probabilities[int(sidx)] = \
                        gmg['probabilities'][key][:]

    # ================================================================
    # GMM CACHE
    # ================================================================
    gmm_cache = None
    if 'gmm' in f and f['gmm'].attrs.get('has_cache', False):
        gmg = f['gmm']
        gmm_cache = {
            'cluster_labels': gmg['cluster_labels'][:],
            'cluster_probabilities': gmg['cluster_probabilities'][:],
            'feature_matrix': gmg['feature_matrix'][:],
            'sniffing_cluster_id': int(gmg.attrs['sniffing_cluster_id']),
            'feature_keys': json.loads(_read_string(gmg['feature_keys_json'])),
            'breath_cycles': [tuple(bc) for bc in gmg['breath_cycles'][:]],
        }

    # ================================================================
    # APP SETTINGS
    # ================================================================
    app_settings = None
    if 'app_settings' in f:
        asg = f['app_settings']
        if asg.attrs.get('has_app_settings', False):
            notch_lower = float(asg.attrs.get('notch_filter_lower', 0.0))
            notch_upper = float(asg.attrs.get('notch_filter_upper', 0.0))
            app_settings = {
                'filter_order': int(asg.attrs.get('filter_order', 4)),
                'use_zscore_normalization': bool(
                    asg.attrs.get('use_zscore_normalization', True)),
                'notch_filter_lower': notch_lower if notch_lower != 0.0 else None,
                'notch_filter_upper': notch_upper if notch_upper != 0.0 else None,
                'apnea_threshold': float(asg.attrs.get('apnea_threshold', 0.5)),
                'active_eupnea_sniff_classifier': _safe_attr_str(
                    asg.attrs.get('active_eupnea_sniff_classifier', 'gmm')),
                'active_classifier': _safe_attr_str(
                    asg.attrs.get('active_classifier', 'xgboost')),
                'active_sigh_classifier': _safe_attr_str(
                    asg.attrs.get('active_sigh_classifier', 'xgboost')),
            }
            # Mirror npz_io.py: set on state directly
            state.active_classifier = app_settings['active_classifier']
            state.active_sigh_classifier = app_settings['active_sigh_classifier']

    # ================================================================
    # Y2 METRICS
    # ================================================================
    if 'y2_metrics' in f:
        y2g = f['y2_metrics']
        state.y2_metric_key = _safe_attr_str(y2g.attrs.get('metric_key'))

        if 'sweep_indices' in y2g:
            for sidx in y2g['sweep_indices'][:]:
                key = str(int(sidx))
                if key in y2g:
                    state.y2_values_by_sweep[int(sidx)] = y2g[key][:]

    # ================================================================
    # STIMULUS DETECTION (per-sweep)
    # ================================================================
    if 'stim' in f:
        stg = f['stim']

        if 'onset_sweep_indices' in stg:
            for sidx in stg['onset_sweep_indices'][:]:
                key = f'onsets_{int(sidx)}'
                if key in stg:
                    state.stim_onsets_by_sweep[int(sidx)] = stg[key][:]

        if 'offset_sweep_indices' in stg:
            for sidx in stg['offset_sweep_indices'][:]:
                key = f'offsets_{int(sidx)}'
                if key in stg:
                    state.stim_offsets_by_sweep[int(sidx)] = stg[key][:]

        if 'spans_sweep_indices' in stg:
            for sidx in stg['spans_sweep_indices'][:]:
                key = f'spans_{int(sidx)}_json'
                if key in stg:
                    spans = json.loads(_read_string(stg[key]))
                    state.stim_spans_by_sweep[int(sidx)] = [tuple(s) for s in spans]

        if 'metrics_sweep_indices' in stg:
            for sidx in stg['metrics_sweep_indices'][:]:
                key = f'metrics_{int(sidx)}_json'
                if key in stg:
                    state.stim_metrics_by_sweep[int(sidx)] = json.loads(
                        _read_string(stg[key]))

    # ================================================================
    # CTA DATA
    # ================================================================
    cta_data = None
    if 'cta' in f:
        cg = f['cta']
        cta_data = {}
        if 'version' in cg.attrs:
            cta_data['cta_version'] = np.array(int(cg.attrs['version']))
        if 'workspace_json' in cg:
            cta_data['cta_workspace_json'] = _read_string(cg['workspace_json'])
        if 'legacy_json' in cg:
            cta_data['cta_json'] = _read_string(cg['legacy_json'])

    # ================================================================
    # CHANNEL CONFIG
    # ================================================================
    channel_config = None
    if 'metadata' in f and 'channel_config_json' in f['metadata']:
        try:
            channel_config = json.loads(
                _read_string(f['metadata']['channel_config_json']))
        except (json.JSONDecodeError, ValueError):
            pass

    return (state, raw_data_loaded, gmm_cache, app_settings,
            event_markers_out, cta_data, channel_config)


# ---------------------------------------------------------------------------
# Metadata (instant read)
# ---------------------------------------------------------------------------

def get_hdf5_metadata(hdf5_path: Path) -> Dict[str, Any]:
    """
    Read ONLY root attributes from an HDF5 session file.

    This is instant (no data loading) and returns the same dict format
    as ``get_npz_metadata``.

    Args:
        hdf5_path: Path to the ``.pleth.h5`` or ``.pmx.h5`` file.

    Returns:
        Dict with keys: version, saved_timestamp, modified_time, original_file,
        channel, n_peaks, n_sweeps, has_gmm, has_edits, and optionally
        schema_version, analysis_type, summary.
    """
    import h5py
    import os

    hdf5_path = Path(hdf5_path)
    try:
        with h5py.File(str(hdf5_path), 'r') as f:
            # Count peaks across all sweeps (reads only small index datasets)
            n_peaks = 0
            n_sweeps = 0
            if 'peaks' in f and 'sweep_indices' in f['peaks']:
                sweep_indices = f['peaks']['sweep_indices'][:]
                n_sweeps = len(sweep_indices)
                for sidx in sweep_indices:
                    key = str(int(sidx))
                    if key in f['peaks']:
                        n_peaks += f['peaks'][key].shape[0]

            # File modification time
            mtime = datetime.fromtimestamp(os.path.getmtime(hdf5_path))

            # GMM presence
            has_gmm = False
            if 'gmm' in f and 'sweep_indices' in f['gmm']:
                has_gmm = len(f['gmm']['sweep_indices']) > 0

            # Channel name from metadata
            channel = 'unknown'
            if 'metadata' in f:
                channel = _safe_attr_str(
                    f['metadata'].attrs.get('analyze_chan', 'unknown'))

            result = {
                'version': _safe_attr_str(f.attrs.get('app_version', 'unknown')),
                'saved_timestamp': _safe_attr_str(
                    f.attrs.get('saved_timestamp', 'unknown')),
                'modified_time': mtime.strftime('%Y-%m-%d %H:%M'),
                'original_file': _safe_attr_str(
                    f.attrs.get('original_file_path', 'unknown')),
                'channel': channel,
                'n_peaks': n_peaks,
                'n_sweeps': n_sweeps,
                'has_gmm': has_gmm,
                'has_edits': True,
            }

            # Schema version
            schema_ver = f.attrs.get('schema_version')
            if schema_ver is not None:
                result['schema_version'] = int(schema_ver)

            return result

    except Exception as e:
        return {
            'error': str(e),
            'modified_time': 'unknown',
            'n_peaks': 0,
            'n_sweeps': 0,
        }
