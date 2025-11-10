"""
NPZ I/O Module - Save/Load PhysioMetrics analysis sessions

This module handles saving and loading complete analysis states to/from .npz files.
Supports per-channel session files that link to original data (no duplication).
"""

from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from core.state import AppState


def get_npz_path_for_channel(data_path: Path, channel_name: str) -> Path:
    """
    Get .pleth.npz path for a specific channel.

    Session files are saved next to the original data file.

    Args:
        data_path: Path to original data file (.abf, .smrx, .edf)
        channel_name: Channel name (e.g., "Pleth", "Pleth_Animal1")

    Returns:
        Path to .pleth.npz file

    Examples:
        mouse01.abf + "Pleth" → mouse01.Pleth.pleth.npz
        mouse01.abf + "Pleth_Animal1" → mouse01.Pleth_Animal1.pleth.npz
    """
    base = data_path.stem  # "mouse01" from "mouse01.abf"

    # Sanitize channel name (replace spaces, slashes, special chars)
    safe_channel = channel_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

    return data_path.parent / f"{base}.{safe_channel}.pleth.npz"


def get_ml_npz_path_for_channel(data_path: Path, channel_name: str) -> Path:
    """
    Get .ml.npz path for ML training data export.

    ML training files are saved in ML/ subfolder next to data file (for testing).
    Future: May move to central ML training folder across all experiments.

    Args:
        data_path: Path to original data file (.abf, .smrx, .edf)
        channel_name: Channel name (e.g., "Pleth", "Pleth_Animal1")

    Returns:
        Path to .ml.npz file in ML/ subfolder

    Examples:
        Data/mouse01.abf + "Pleth" → Data/ML/mouse01.Pleth.ml.npz
    """
    base = data_path.stem
    safe_channel = channel_name.replace(' ', '_').replace('/', '_').replace('\\', '_')

    # Create ML subfolder if it doesn't exist
    ml_folder = data_path.parent / "ML"

    return ml_folder / f"{base}.{safe_channel}.ml.npz"


def save_state_to_npz(state: AppState, npz_path: Path, include_raw_data: bool = False, gmm_cache: Optional[Dict] = None, app_settings: Optional[Dict] = None) -> None:
    """
    Save complete analysis state to .pleth.npz file.

    By default, does NOT save raw signal data (links to original file instead).
    This keeps file size small (5-10 MB vs 65+ MB).

    Args:
        state: AppState instance to save
        npz_path: Path to save .pleth.npz file
        include_raw_data: If True, include raw sweeps (for portability, larger files)
        gmm_cache: Optional GMM cache dict to preserve user's cluster assignments
        app_settings: Optional dict with app-level settings (filter_order, zscore, notch, apnea_thresh)

    Raises:
        ValueError: If state is missing required fields
    """
    if state.in_path is None:
        raise ValueError("Cannot save state: no data file loaded (in_path is None)")

    if state.analyze_chan is None:
        raise ValueError("Cannot save state: no channel selected (analyze_chan is None)")

    # Build NPZ data dictionary
    data = {}

    # ===== METADATA =====
    data['version'] = '1.0.9'  # PhysioMetrics version
    data['saved_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data['original_file_path'] = str(state.in_path)

    # Multi-file info (if concatenated ABF files)
    if state.file_info:
        # Convert list of dicts to JSON (NPZ doesn't handle complex nested structures)
        file_info_serializable = []
        for fi in state.file_info:
            file_info_serializable.append({
                'path': str(fi['path']),
                'sweep_start': int(fi['sweep_start']),
                'sweep_end': int(fi['sweep_end']),
                'padded': fi.get('padded', False),
                'original_samples': fi.get('original_samples', 0),
                'padded_samples': fi.get('padded_samples', 0)
            })
        data['file_info_json'] = json.dumps(file_info_serializable)

    # ===== RAW DATA (optional - only if user wants portability) =====
    if include_raw_data:
        # Save all channel sweeps
        for chan_name, sweep_data in state.sweeps.items():
            # Use safe key name (NPZ doesn't like spaces/slashes)
            safe_name = chan_name.replace(' ', '_').replace('/', '_')
            data[f'sweeps_{safe_name}'] = sweep_data

        data['t'] = state.t
        data['channel_names'] = np.array(state.channel_names, dtype=object)

    data['sr_hz'] = state.sr_hz

    # ===== CHANNEL SELECTIONS =====
    data['analyze_chan'] = state.analyze_chan if state.analyze_chan else 'None'
    data['stim_chan'] = state.stim_chan if state.stim_chan else 'None'
    data['event_channel'] = state.event_channel if state.event_channel else 'None'

    # ===== FILTER SETTINGS =====
    data['use_low'] = state.use_low
    data['use_high'] = state.use_high
    data['use_mean_sub'] = state.use_mean_sub
    data['use_invert'] = state.use_invert
    data['low_hz'] = state.low_hz if state.low_hz else 0.0
    data['high_hz'] = state.high_hz if state.high_hz else 0.0
    data['mean_val'] = state.mean_val

    # ===== NAVIGATION STATE =====
    data['sweep_idx'] = state.sweep_idx
    data['window_start_s'] = state.window_start_s
    data['window_dur_s'] = state.window_dur_s

    # ===== PEAKS (per-sweep) =====
    # Save each sweep's peaks as separate array
    sweep_indices = sorted(state.peaks_by_sweep.keys())
    data['peak_sweep_indices'] = np.array(sweep_indices, dtype=int)

    for sweep_idx in sweep_indices:
        peaks = state.peaks_by_sweep[sweep_idx]
        data[f'peaks_sweep_{sweep_idx}'] = peaks

    # ===== BREATH FEATURES (per-sweep) =====
    # Each sweep's breath dict contains onsets, offsets, expmins, expoffs
    breath_sweep_indices = sorted(state.breath_by_sweep.keys())
    data['breath_sweep_indices'] = np.array(breath_sweep_indices, dtype=int)

    for sweep_idx in breath_sweep_indices:
        breath_dict = state.breath_by_sweep[sweep_idx]
        # Convert dict to JSON for NPZ storage
        breath_json = json.dumps({
            'onsets': breath_dict.get('onsets', []).tolist() if isinstance(breath_dict.get('onsets'), np.ndarray) else breath_dict.get('onsets', []),
            'offsets': breath_dict.get('offsets', []).tolist() if isinstance(breath_dict.get('offsets'), np.ndarray) else breath_dict.get('offsets', []),
            'expmins': breath_dict.get('expmins', []).tolist() if isinstance(breath_dict.get('expmins'), np.ndarray) else breath_dict.get('expmins', []),
            'expoffs': breath_dict.get('expoffs', []).tolist() if isinstance(breath_dict.get('expoffs'), np.ndarray) else breath_dict.get('expoffs', [])
        })
        data[f'breath_sweep_{sweep_idx}_json'] = breath_json

    # ===== SIGHS (per-sweep) =====
    sigh_sweep_indices = sorted(state.sigh_by_sweep.keys())
    data['sigh_sweep_indices'] = np.array(sigh_sweep_indices, dtype=int)

    for sweep_idx in sigh_sweep_indices:
        sighs = state.sigh_by_sweep[sweep_idx]
        data[f'sigh_sweep_{sweep_idx}'] = sighs

    # ===== OMISSIONS =====
    data['omitted_sweeps'] = np.array(sorted(state.omitted_sweeps), dtype=int) if state.omitted_sweeps else np.array([], dtype=int)

    # Omitted points (per-sweep)
    omitted_points_indices = sorted(state.omitted_points.keys())
    data['omitted_points_indices'] = np.array(omitted_points_indices, dtype=int)
    for sweep_idx in omitted_points_indices:
        data[f'omitted_points_sweep_{sweep_idx}'] = np.array(state.omitted_points[sweep_idx], dtype=int)

    # Omitted ranges (per-sweep) - save as JSON (list of tuples)
    omitted_ranges_indices = sorted(state.omitted_ranges.keys())
    data['omitted_ranges_indices'] = np.array(omitted_ranges_indices, dtype=int)
    for sweep_idx in omitted_ranges_indices:
        ranges = state.omitted_ranges[sweep_idx]
        data[f'omitted_ranges_sweep_{sweep_idx}_json'] = json.dumps(ranges)

    # ===== SNIFFING REGIONS (per-sweep) =====
    sniff_sweep_indices = sorted(state.sniff_regions_by_sweep.keys())
    data['sniff_sweep_indices'] = np.array(sniff_sweep_indices, dtype=int)

    for sweep_idx in sniff_sweep_indices:
        regions = state.sniff_regions_by_sweep[sweep_idx]
        # Save as JSON (list of (start, end) tuples)
        data[f'sniff_regions_sweep_{sweep_idx}_json'] = json.dumps(regions)

    # ===== BOUT ANNOTATIONS (per-sweep) =====
    bout_sweep_indices = sorted(state.bout_annotations.keys())
    data['bout_sweep_indices'] = np.array(bout_sweep_indices, dtype=int)

    for sweep_idx in bout_sweep_indices:
        bouts = state.bout_annotations[sweep_idx]
        # Save as JSON (list of dicts)
        data[f'bout_sweep_{sweep_idx}_json'] = json.dumps(bouts)

    # ===== GMM PROBABILITIES (per-sweep) =====
    gmm_sweep_indices = sorted(state.gmm_sniff_probabilities.keys())
    data['gmm_sweep_indices'] = np.array(gmm_sweep_indices, dtype=int)

    for sweep_idx in gmm_sweep_indices:
        probs = state.gmm_sniff_probabilities[sweep_idx]
        data[f'gmm_probs_sweep_{sweep_idx}'] = probs

    # ===== GMM CACHE (preserves user's cluster assignments and dialog data) =====
    if gmm_cache is not None:
        data['has_gmm_cache'] = True
        data['gmm_cluster_labels'] = gmm_cache['cluster_labels']
        data['gmm_cluster_probabilities'] = gmm_cache['cluster_probabilities']
        data['gmm_feature_matrix'] = gmm_cache['feature_matrix']
        data['gmm_sniffing_cluster_id'] = gmm_cache['sniffing_cluster_id']
        data['gmm_feature_keys_json'] = json.dumps(gmm_cache['feature_keys'])

        # Breath cycles is a list of tuples: (sweep_idx, breath_idx)
        breath_cycles_array = np.array(gmm_cache['breath_cycles'], dtype=int)
        data['gmm_breath_cycles'] = breath_cycles_array
    else:
        data['has_gmm_cache'] = False

    # ===== APP SETTINGS (filter order, zscore, notch, apnea threshold) =====
    if app_settings is not None:
        data['has_app_settings'] = True
        data['filter_order'] = app_settings.get('filter_order', 4)
        data['use_zscore_normalization'] = app_settings.get('use_zscore_normalization', True)
        data['notch_filter_lower'] = app_settings.get('notch_filter_lower', 0.0) if app_settings.get('notch_filter_lower') is not None else 0.0
        data['notch_filter_upper'] = app_settings.get('notch_filter_upper', 0.0) if app_settings.get('notch_filter_upper') is not None else 0.0
        data['apnea_threshold'] = app_settings.get('apnea_threshold', 0.5)
    else:
        data['has_app_settings'] = False

    # ===== Y2 METRICS =====
    if state.y2_metric_key:
        data['y2_metric_key'] = state.y2_metric_key

        y2_sweep_indices = sorted(state.y2_values_by_sweep.keys())
        data['y2_sweep_indices'] = np.array(y2_sweep_indices, dtype=int)

        for sweep_idx in y2_sweep_indices:
            y2_vals = state.y2_values_by_sweep[sweep_idx]
            data[f'y2_values_sweep_{sweep_idx}'] = y2_vals

    # ===== STIMULUS DETECTION (per-sweep) =====
    stim_onset_indices = sorted(state.stim_onsets_by_sweep.keys())
    data['stim_onset_sweep_indices'] = np.array(stim_onset_indices, dtype=int)
    for sweep_idx in stim_onset_indices:
        data[f'stim_onsets_sweep_{sweep_idx}'] = state.stim_onsets_by_sweep[sweep_idx]

    stim_offset_indices = sorted(state.stim_offsets_by_sweep.keys())
    data['stim_offset_sweep_indices'] = np.array(stim_offset_indices, dtype=int)
    for sweep_idx in stim_offset_indices:
        data[f'stim_offsets_sweep_{sweep_idx}'] = state.stim_offsets_by_sweep[sweep_idx]

    stim_spans_indices = sorted(state.stim_spans_by_sweep.keys())
    data['stim_spans_sweep_indices'] = np.array(stim_spans_indices, dtype=int)
    for sweep_idx in stim_spans_indices:
        spans = state.stim_spans_by_sweep[sweep_idx]
        data[f'stim_spans_sweep_{sweep_idx}_json'] = json.dumps(spans)

    stim_metrics_indices = sorted(state.stim_metrics_by_sweep.keys())
    data['stim_metrics_sweep_indices'] = np.array(stim_metrics_indices, dtype=int)
    for sweep_idx in stim_metrics_indices:
        metrics = state.stim_metrics_by_sweep[sweep_idx]
        data[f'stim_metrics_sweep_{sweep_idx}_json'] = json.dumps(metrics)

    # ===== SAVE TO NPZ =====
    np.savez_compressed(npz_path, **data)


def load_state_from_npz(npz_path: Path, reload_raw_data: bool = True) -> Tuple[AppState, bool, Optional[Dict], Optional[Dict]]:
    """
    Load complete analysis state from .pleth.npz file.

    Args:
        npz_path: Path to .pleth.npz file
        reload_raw_data: If True, reload raw data from original file
                        If False, only load if embedded in NPZ

    Returns:
        (state, raw_data_loaded, gmm_cache, app_settings)
        - state: Restored AppState instance
        - raw_data_loaded: True if raw data was loaded (either from file or NPZ)
        - gmm_cache: Restored GMM cache dict (or None if not saved)
        - app_settings: Restored app-level settings dict (or None if not saved)

    Raises:
        FileNotFoundError: If original data file not found (and not embedded)
        ValueError: If NPZ file is corrupted or invalid
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Load NPZ file
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file: {e}")

    # Create new AppState
    state = AppState()

    # ===== METADATA =====
    version = str(data.get('version', 'unknown'))
    # Convert numpy scalar to Python string, then to Path
    original_file_path_str = str(data['original_file_path'])
    if hasattr(data['original_file_path'], 'item'):
        original_file_path_str = str(data['original_file_path'].item())
    original_file_path = Path(original_file_path_str)

    # ===== LOAD RAW DATA =====
    raw_data_loaded = False

    if reload_raw_data and original_file_path.exists():
        # Reload from original file (preferred - ensures latest data)
        from core.abf_io import load_data_file

        try:
            sr, sweeps_by_ch, ch_names, t = load_data_file(original_file_path)
            state.sr_hz = sr
            state.sweeps = sweeps_by_ch
            state.channel_names = ch_names
            state.t = t
            state.in_path = original_file_path
            raw_data_loaded = True
        except Exception as e:
            # Original file couldn't be loaded - try NPZ embedded data
            print(f"Warning: Could not reload from {original_file_path}: {e}")
            print("Attempting to use embedded data from NPZ...")
            pass

    # Check if raw data is embedded in NPZ (fallback or if reload_raw_data=False)
    if not raw_data_loaded and 't' in data:
        state.sr_hz = float(data['sr_hz'])
        state.t = data['t']
        state.channel_names = list(data['channel_names'])

        # Reconstruct sweeps dict
        state.sweeps = {}
        for chan_name in state.channel_names:
            safe_name = chan_name.replace(' ', '_').replace('/', '_')
            key = f'sweeps_{safe_name}'
            if key in data:
                state.sweeps[chan_name] = data[key]

        state.in_path = original_file_path
        raw_data_loaded = True

    if not raw_data_loaded:
        raise ValueError(
            f"Could not load raw data:\n"
            f"- Original file not found: {original_file_path}\n"
            f"- No embedded data in NPZ file\n"
            f"Please locate the original data file."
        )

    # ===== MULTI-FILE INFO =====
    if 'file_info_json' in data:
        file_info_str = str(data['file_info_json'])
        file_info_list = json.loads(file_info_str)
        state.file_info = []
        for fi in file_info_list:
            state.file_info.append({
                'path': Path(fi['path']),
                'sweep_start': fi['sweep_start'],
                'sweep_end': fi['sweep_end'],
                'padded': fi.get('padded', False),
                'original_samples': fi.get('original_samples', 0),
                'padded_samples': fi.get('padded_samples', 0)
            })
    else:
        # Single file (no concatenation)
        n_sweeps = next(iter(state.sweeps.values())).shape[1]
        state.file_info = [{
            'path': original_file_path,
            'sweep_start': 0,
            'sweep_end': n_sweeps - 1
        }]

    # ===== CHANNEL SELECTIONS =====
    # Convert numpy scalars to Python strings
    analyze_chan_val = data.get('analyze_chan', 'None')
    if hasattr(analyze_chan_val, 'item'):
        analyze_chan_val = analyze_chan_val.item()
    state.analyze_chan = str(analyze_chan_val) if analyze_chan_val != 'None' else None

    stim_chan_val = data.get('stim_chan', 'None')
    if hasattr(stim_chan_val, 'item'):
        stim_chan_val = stim_chan_val.item()
    state.stim_chan = str(stim_chan_val) if stim_chan_val != 'None' else None

    event_channel_val = data.get('event_channel', 'None')
    if hasattr(event_channel_val, 'item'):
        event_channel_val = event_channel_val.item()
    state.event_channel = str(event_channel_val) if event_channel_val != 'None' else None

    # ===== FILTER SETTINGS =====
    state.use_low = bool(data['use_low'])
    state.use_high = bool(data['use_high'])
    state.use_mean_sub = bool(data['use_mean_sub'])
    state.use_invert = bool(data['use_invert'])
    state.low_hz = float(data['low_hz']) if data['low_hz'] != 0.0 else None
    state.high_hz = float(data['high_hz']) if data['high_hz'] != 0.0 else None
    state.mean_val = float(data['mean_val'])

    # ===== NAVIGATION STATE =====
    state.sweep_idx = int(data['sweep_idx'])
    state.window_start_s = float(data['window_start_s'])
    state.window_dur_s = float(data['window_dur_s'])

    # ===== PEAKS (per-sweep) =====
    if 'peak_sweep_indices' in data:
        sweep_indices = data['peak_sweep_indices']
        for sweep_idx in sweep_indices:
            peaks = data[f'peaks_sweep_{sweep_idx}']
            state.peaks_by_sweep[int(sweep_idx)] = peaks

    # ===== BREATH FEATURES (per-sweep) =====
    if 'breath_sweep_indices' in data:
        breath_indices = data['breath_sweep_indices']
        for sweep_idx in breath_indices:
            breath_json = str(data[f'breath_sweep_{sweep_idx}_json'])
            breath_dict = json.loads(breath_json)
            # Convert lists back to numpy arrays
            state.breath_by_sweep[int(sweep_idx)] = {
                'onsets': np.array(breath_dict['onsets'], dtype=int),
                'offsets': np.array(breath_dict['offsets'], dtype=int),
                'expmins': np.array(breath_dict['expmins'], dtype=int),
                'expoffs': np.array(breath_dict['expoffs'], dtype=int)
            }

    # ===== SIGHS (per-sweep) =====
    if 'sigh_sweep_indices' in data:
        sigh_indices = data['sigh_sweep_indices']
        for sweep_idx in sigh_indices:
            sighs = data[f'sigh_sweep_{sweep_idx}']
            state.sigh_by_sweep[int(sweep_idx)] = sighs

    # ===== OMISSIONS =====
    if 'omitted_sweeps' in data:
        state.omitted_sweeps = set(int(x) for x in data['omitted_sweeps'])

    if 'omitted_points_indices' in data:
        for sweep_idx in data['omitted_points_indices']:
            points = data[f'omitted_points_sweep_{sweep_idx}']
            state.omitted_points[int(sweep_idx)] = list(points)

    if 'omitted_ranges_indices' in data:
        for sweep_idx in data['omitted_ranges_indices']:
            ranges_json = str(data[f'omitted_ranges_sweep_{sweep_idx}_json'])
            ranges = json.loads(ranges_json)
            state.omitted_ranges[int(sweep_idx)] = [tuple(r) for r in ranges]

    # ===== SNIFFING REGIONS (per-sweep) =====
    if 'sniff_sweep_indices' in data:
        sniff_indices = data['sniff_sweep_indices']
        for sweep_idx in sniff_indices:
            regions_json = str(data[f'sniff_regions_sweep_{sweep_idx}_json'])
            regions = json.loads(regions_json)
            state.sniff_regions_by_sweep[int(sweep_idx)] = [tuple(r) for r in regions]

    # ===== BOUT ANNOTATIONS (per-sweep) =====
    if 'bout_sweep_indices' in data:
        bout_indices = data['bout_sweep_indices']
        for sweep_idx in bout_indices:
            bouts_json = str(data[f'bout_sweep_{sweep_idx}_json'])
            bouts = json.loads(bouts_json)
            state.bout_annotations[int(sweep_idx)] = bouts

    # ===== GMM PROBABILITIES (per-sweep) =====
    if 'gmm_sweep_indices' in data:
        gmm_indices = data['gmm_sweep_indices']
        for sweep_idx in gmm_indices:
            probs = data[f'gmm_probs_sweep_{sweep_idx}']
            state.gmm_sniff_probabilities[int(sweep_idx)] = probs

    # ===== GMM CACHE (restore user's cluster assignments and dialog data) =====
    gmm_cache = None
    if data.get('has_gmm_cache', False):
        gmm_cache = {
            'cluster_labels': data['gmm_cluster_labels'],
            'cluster_probabilities': data['gmm_cluster_probabilities'],
            'feature_matrix': data['gmm_feature_matrix'],
            'sniffing_cluster_id': int(data['gmm_sniffing_cluster_id']),
            'feature_keys': json.loads(str(data['gmm_feature_keys_json'])),
            'breath_cycles': [tuple(bc) for bc in data['gmm_breath_cycles']]
        }

    # ===== APP SETTINGS (restore filter order, zscore, notch, apnea threshold) =====
    app_settings = None
    if data.get('has_app_settings', False):
        notch_lower = float(data['notch_filter_lower'])
        notch_upper = float(data['notch_filter_upper'])
        app_settings = {
            'filter_order': int(data['filter_order']),
            'use_zscore_normalization': bool(data['use_zscore_normalization']),
            'notch_filter_lower': notch_lower if notch_lower != 0.0 else None,
            'notch_filter_upper': notch_upper if notch_upper != 0.0 else None,
            'apnea_threshold': float(data['apnea_threshold'])
        }

    # ===== Y2 METRICS =====
    if 'y2_metric_key' in data:
        y2_key = data['y2_metric_key']
        if hasattr(y2_key, 'item'):
            y2_key = y2_key.item()
        state.y2_metric_key = str(y2_key)

        if 'y2_sweep_indices' in data:
            y2_indices = data['y2_sweep_indices']
            for sweep_idx in y2_indices:
                y2_vals = data[f'y2_values_sweep_{sweep_idx}']
                state.y2_values_by_sweep[int(sweep_idx)] = y2_vals

    # ===== STIMULUS DETECTION (per-sweep) =====
    if 'stim_onset_sweep_indices' in data:
        for sweep_idx in data['stim_onset_sweep_indices']:
            state.stim_onsets_by_sweep[int(sweep_idx)] = data[f'stim_onsets_sweep_{sweep_idx}']

    if 'stim_offset_sweep_indices' in data:
        for sweep_idx in data['stim_offset_sweep_indices']:
            state.stim_offsets_by_sweep[int(sweep_idx)] = data[f'stim_offsets_sweep_{sweep_idx}']

    if 'stim_spans_sweep_indices' in data:
        for sweep_idx in data['stim_spans_sweep_indices']:
            spans_json = str(data[f'stim_spans_sweep_{sweep_idx}_json'])
            spans = json.loads(spans_json)
            state.stim_spans_by_sweep[int(sweep_idx)] = [tuple(s) for s in spans]

    if 'stim_metrics_sweep_indices' in data:
        for sweep_idx in data['stim_metrics_sweep_indices']:
            metrics_json = str(data[f'stim_metrics_sweep_{sweep_idx}_json'])
            metrics = json.loads(metrics_json)
            state.stim_metrics_by_sweep[int(sweep_idx)] = metrics

    return state, raw_data_loaded, gmm_cache, app_settings


def get_npz_metadata(npz_path: Path) -> Dict[str, Any]:
    """
    Get metadata from .pleth.npz file without loading full state.

    Useful for displaying info in load dialogs.

    Args:
        npz_path: Path to .pleth.npz file

    Returns:
        Dict with metadata: version, timestamp, n_peaks, n_sweeps, etc.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)

        # Count peaks across all sweeps
        n_peaks = 0
        if 'peak_sweep_indices' in data:
            for sweep_idx in data['peak_sweep_indices']:
                peaks = data[f'peaks_sweep_{sweep_idx}']
                n_peaks += len(peaks)

        # Count manual edits (rough estimate - would need edit history for exact count)
        # For now, just show total peaks

        # Get file modification time
        import os
        mtime = datetime.fromtimestamp(os.path.getmtime(npz_path))

        # Helper to convert numpy scalars
        def safe_str(val, default='unknown'):
            if val is None:
                return default
            if hasattr(val, 'item'):
                val = val.item()
            return str(val)

        return {
            'version': safe_str(data.get('version'), 'unknown'),
            'saved_timestamp': safe_str(data.get('saved_timestamp'), 'unknown'),
            'modified_time': mtime.strftime('%Y-%m-%d %H:%M'),
            'original_file': safe_str(data.get('original_file_path'), 'unknown'),
            'channel': safe_str(data.get('analyze_chan'), 'unknown'),
            'n_peaks': n_peaks,
            'n_sweeps': len(data.get('peak_sweep_indices', [])),
            'has_gmm': len(data.get('gmm_sweep_indices', [])) > 0,
            'has_edits': True  # Assume true if NPZ exists (could be more sophisticated)
        }
    except Exception as e:
        return {
            'error': str(e),
            'modified_time': 'unknown',
            'n_peaks': 0,
            'n_sweeps': 0
        }
