from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def load_data_file(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray, dict]:
    """
    Load data file - dispatches to appropriate loader based on file extension

    Supported formats:
    - .abf: Axon Binary Format (pyabf)
    - .smrx: Son64 format (CED Spike2) via Python 3.9 bridge
    - .edf: European Data Format (pyedflib)

    Returns (sr_hz, sweeps_by_channel, channel_names, t, metadata)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)
    metadata: dict with file-specific metadata (protocol, n_channels, etc.)

    progress_callback: optional function(current, total, message) for progress updates
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.abf':
        return load_abf(path, progress_callback)
    elif suffix == '.smrx':
        from core.io.son64_loader import load_son64
        sr, sweeps, names, t = load_son64(str(path), progress_callback)
        metadata = {'file_type': 'smrx', 'n_channels': len(names)}
        return sr, sweeps, names, t, metadata
    elif suffix == '.edf':
        from core.io.edf_loader import load_edf
        sr, sweeps, names, t = load_edf(path, progress_callback)
        metadata = {'file_type': 'edf', 'n_channels': len(names)}
        return sr, sweeps, names, t, metadata
    else:
        raise ValueError(f"Unsupported file format: {suffix}\nSupported formats: .abf, .smrx, .edf")


def load_abf(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray, dict]:
    """
    Returns (sr_hz, sweeps_by_channel, channel_names, t, metadata)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)
    metadata: dict with 'protocol', 'n_channels', 'n_sweeps' for ABF files (empty dict for other formats)

    progress_callback: optional function(current, total, message) for progress updates
    """
    try:
        import pyabf
    except ImportError:
        raise RuntimeError("pyabf not installed. pip install pyabf")

    if progress_callback:
        progress_callback(0, 100, "Opening ABF file...")

    abf = pyabf.ABF(str(path))
    sr_hz = float(abf.dataRate)
    chan_names = [abf.adcNames[c] for c in range(abf.channelCount)]

    # Extract ABF-specific metadata
    metadata = {
        'protocol': abf.protocol if hasattr(abf, 'protocol') else '',
        'n_channels': abf.channelCount,
        'n_sweeps': abf.sweepCount,
        'file_type': 'abf'
    }

    if progress_callback:
        progress_callback(10, 100, "Reading channel data...")

    # Collect sweeps into arrays per channel
    sweeps_by_channel: Dict[str, np.ndarray] = {}
    n_sweeps = abf.sweepCount
    n_samples = abf.sweepPointCount
    total_ops = abf.channelCount * n_sweeps

    op_count = 0
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        M = np.empty((n_samples, n_sweeps), dtype=float)
        for s in range(n_sweeps):
            abf.setSweep(sweepNumber=s, channel=ch)
            M[:, s] = abf.sweepY

            op_count += 1
            if progress_callback and op_count % 5 == 0:  # Update every 5 sweeps
                pct = 10 + int(80 * op_count / total_ops)
                progress_callback(pct, 100, f"Loading channel {ch+1}/{abf.channelCount}, sweep {s+1}/{n_sweeps}...")

        sweeps_by_channel[chan_names[ch]] = M

    if progress_callback:
        progress_callback(95, 100, "Finalizing...")

    # time vector for one sweep
    t = np.arange(n_samples, dtype=float) / sr_hz

    if progress_callback:
        progress_callback(100, 100, "Complete")

    return sr_hz, sweeps_by_channel, chan_names, t, metadata


def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract date from ABF filename (format: YYYYMMDD####.abf or similar)
    Returns date string (YYYYMMDD) or None if not found
    """
    # Try to find YYYYMMDD pattern in filename
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        # Validate it looks like a valid date
        year = int(date_str[:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            return date_str
    return None


def validate_files_for_concatenation(file_paths: List[Path]) -> Tuple[bool, List[str]]:
    """
    Validate that multiple files can be safely concatenated.

    Checks:
    - All files are same type (.abf or .smrx)
    - All files have same number of channels
    - All files have same channel names
    - All files have same sample rate
    - (Optional warning) Files have same date in filename

    Returns (valid, warnings_or_errors)
    warnings_or_errors: List of warning/error messages
    """
    errors = []
    warnings = []

    # Check all files exist
    for path in file_paths:
        if not path.exists():
            errors.append(f"File not found: {path.name}")

    if errors:
        return False, errors

    # Check all files are same type
    extensions = set(path.suffix.lower() for path in file_paths)
    if len(extensions) > 1:
        errors.append(f"Mixed file types detected: {', '.join(extensions)}. All files must be same type (.abf, .smrx, or .edf).")
        return False, errors

    file_type = extensions.pop()

    # Load metadata from all files (without loading full data)
    file_metadata = []

    try:
        if file_type == '.abf':
            import pyabf
            for path in file_paths:
                abf = pyabf.ABF(str(path))
                metadata = {
                    'path': path,
                    'n_channels': abf.channelCount,
                    'channel_names': [abf.adcNames[c] for c in range(abf.channelCount)],
                    'sample_rate': float(abf.dataRate),
                    'n_sweeps': abf.sweepCount,
                    'n_samples': abf.sweepPointCount,
                }
                file_metadata.append(metadata)
        elif file_type == '.smrx':
            # For SMRX files, we'd need to partially load them to get metadata
            # For now, we'll handle this more gracefully in the concatenation function
            errors.append("Multi-file concatenation for .smrx files not yet implemented.")
            return False, errors
        elif file_type == '.edf':
            # For EDF files, we'd need to partially load them to get metadata
            # For now, we'll handle this more gracefully in the concatenation function
            errors.append("Multi-file concatenation for .edf files not yet implemented.")
            return False, errors
        else:
            errors.append(f"Unsupported file type: {file_type}")
            return False, errors

    except Exception as e:
        errors.append(f"Error reading file metadata: {str(e)}")
        return False, errors

    # Compare metadata across files
    first = file_metadata[0]

    for i, meta in enumerate(file_metadata[1:], start=1):
        # Check number of channels
        if meta['n_channels'] != first['n_channels']:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has {meta['n_channels']} channels, "
                f"but file 1 ({first['path'].name}) has {first['n_channels']} channels."
            )

        # Check channel names
        if meta['channel_names'] != first['channel_names']:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has different channel names: "
                f"{meta['channel_names']} vs {first['channel_names']}"
            )

        # Check sample rate (allow small floating point differences)
        if abs(meta['sample_rate'] - first['sample_rate']) > 0.1:
            errors.append(
                f"File {i+1} ({meta['path'].name}) has sample rate {meta['sample_rate']} Hz, "
                f"but file 1 ({first['path'].name}) has {first['sample_rate']} Hz."
            )

        # Check sweep length (different lengths will be padded with NaN)
        if meta['n_samples'] != first['n_samples']:
            warnings.append(
                f"File {i+1} ({meta['path'].name}) has {meta['n_samples']} samples per sweep, "
                f"but file 1 ({first['path'].name}) has {first['n_samples']} samples. "
                f"Shorter sweeps will be padded with NaN values to match the longest sweep."
            )

    # Check date consistency (warning only)
    dates = []
    for meta in file_metadata:
        date = extract_date_from_filename(meta['path'].name)
        if date:
            dates.append(date)

    if len(dates) == len(file_metadata) and len(set(dates)) > 1:
        warnings.append(
            f"Files have different dates in filenames: {', '.join(set(dates))}. "
            f"This may indicate data from different recording sessions."
        )

    if errors:
        return False, errors
    elif warnings:
        return True, warnings
    else:
        return True, []


def load_and_concatenate_abf_files(file_paths: List[Path], progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray, List[Dict]]:
    """
    Load multiple ABF files and concatenate their sweeps.

    Returns (sr_hz, sweeps_by_channel, channel_names, t, file_info)
    sweeps_by_channel[channel] -> 2D array (n_samples, total_sweeps)
    file_info: List of dicts with 'path', 'sweep_start', 'sweep_end' for each file

    progress_callback: optional function(current, total, message) for progress updates
    """
    if not file_paths:
        raise ValueError("No files provided")

    if len(file_paths) == 1:
        # Just load single file normally
        sr, sweeps, channels, t = load_abf(file_paths[0], progress_callback)
        file_info = [{
            'path': file_paths[0],
            'sweep_start': 0,
            'sweep_end': next(iter(sweeps.values())).shape[1] - 1
        }]
        return sr, sweeps, channels, t, file_info

    # Validate files first
    valid, messages = validate_files_for_concatenation(file_paths)
    if not valid:
        raise ValueError("File validation failed:\n" + "\n".join(messages))

    # Load each file
    all_file_data = []
    total_files = len(file_paths)

    for file_idx, path in enumerate(file_paths):
        if progress_callback:
            pct = int(100 * file_idx / total_files)
            progress_callback(pct, 100, f"Loading file {file_idx+1}/{total_files}: {path.name}...")

        sr, sweeps, channels, t = load_abf(path, progress_callback=None)  # Disable internal progress for cleaner display
        all_file_data.append({
            'path': path,
            'sr': sr,
            'sweeps': sweeps,
            'channels': channels,
            't': t
        })

    # Use metadata from first file
    sr_hz = all_file_data[0]['sr']
    channel_names = all_file_data[0]['channels']

    # Find maximum sweep length across all files by checking actual sweep data
    # (not just 't' array, to handle any edge cases)
    max_samples = max(
        next(iter(file_data['sweeps'].values())).shape[0]
        for file_data in all_file_data
    )

    # Create time vector for longest sweep
    t = np.arange(max_samples, dtype=float) / sr_hz

    # Track which files were padded
    padded_files = []

    # Concatenate sweeps for each channel, padding shorter sweeps with NaN
    sweeps_by_channel = {}
    file_info = []
    current_sweep_idx = 0

    for channel in channel_names:
        channel_sweeps = []
        for file_idx, file_data in enumerate(all_file_data):
            file_sweep_data = file_data['sweeps'][channel]
            n_samples_this_file = file_sweep_data.shape[0]
            n_sweeps_this_file = file_sweep_data.shape[1]

            if n_samples_this_file < max_samples:
                # Pad with NaN to match longest sweep
                padding_size = max_samples - n_samples_this_file
                padding = np.full((padding_size, n_sweeps_this_file), np.nan)
                padded_sweep_data = np.vstack([file_sweep_data, padding])

                # Verify padding worked correctly
                if padded_sweep_data.shape[0] != max_samples:
                    raise ValueError(
                        f"Padding failed for file {file_data['path'].name}: "
                        f"expected {max_samples} samples, got {padded_sweep_data.shape[0]}"
                    )

                channel_sweeps.append(padded_sweep_data)

                # Track that this file was padded (only record once per file)
                if channel == channel_names[0]:  # Only record once
                    padded_files.append({
                        'file_idx': file_idx,
                        'path': file_data['path'],
                        'original_samples': n_samples_this_file,
                        'padded_samples': max_samples
                    })
            else:
                # File already at max length or longer
                channel_sweeps.append(file_sweep_data)

        # Verify all arrays have the same shape[0] before concatenating
        shapes = [arr.shape[0] for arr in channel_sweeps]
        if len(set(shapes)) > 1:
            raise ValueError(
                f"Cannot concatenate {channel}: arrays have different lengths {shapes}. "
                f"Expected all to be {max_samples} samples."
            )

        # Concatenate along sweep axis (axis=1)
        sweeps_by_channel[channel] = np.concatenate(channel_sweeps, axis=1)

    # Build file_info with sweep ranges
    current_sweep_idx = 0
    for file_data in all_file_data:
        n_sweeps = next(iter(file_data['sweeps'].values())).shape[1]
        file_info.append({
            'path': file_data['path'],
            'sweep_start': current_sweep_idx,
            'sweep_end': current_sweep_idx + n_sweeps - 1
        })
        current_sweep_idx += n_sweeps

    if progress_callback:
        message = f"Loaded {len(file_paths)} files with {current_sweep_idx} total sweeps"
        if padded_files:
            message += f" ({len(padded_files)} files padded with NaN)"
        progress_callback(100, 100, message)

    # Add padding info to file_info for display
    for pad_info in padded_files:
        file_info[pad_info['file_idx']]['padded'] = True
        file_info[pad_info['file_idx']]['original_samples'] = pad_info['original_samples']
        file_info[pad_info['file_idx']]['padded_samples'] = pad_info['padded_samples']

    return sr_hz, sweeps_by_channel, channel_names, t, file_info


def detect_stimulus_channel(sweeps_by_channel: Dict[str, np.ndarray], channel_names: List[str]) -> Optional[str]:
    """
    Auto-detect which channel is most likely a stimulus/TTL channel.

    Stimulus channels typically have:
    - Digital-like behavior: values cluster around low (0) and high (e.g., 5V)
    - Sharp transitions between discrete levels
    - Few unique value levels compared to analog signals

    Returns the name of the detected stimulus channel, or None if no clear candidate.
    """
    candidates = []

    for ch_name in channel_names:
        data = sweeps_by_channel[ch_name]

        # Flatten all sweeps for analysis
        flat_data = data.flatten()

        # Remove NaN values
        flat_data = flat_data[np.isfinite(flat_data)]

        if len(flat_data) < 100:
            continue

        # Check for TTL-like characteristics
        score = _score_stimulus_likelihood(flat_data)

        if score > 0:
            candidates.append((ch_name, score))

    if not candidates:
        return None

    # Sort by score (highest first) and return the best candidate
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Only return if the score is high enough (indicates likely stimulus channel)
    best_name, best_score = candidates[0]
    if best_score >= 50:  # Threshold for confidence
        return best_name

    return None


def _score_stimulus_likelihood(data: np.ndarray) -> float:
    """
    Score how likely a data array is from a stimulus/TTL channel.

    Returns a score from 0-100, where higher = more likely stimulus.
    """
    score = 0.0

    # 1. Check for bimodal distribution (low and high values)
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min

    if data_range < 0.1:
        # Nearly constant signal - not a stimulus channel
        return 0

    # Normalize to 0-1
    normalized = (data - data_min) / data_range

    # 2. Count how many values are near 0 or 1 (within 10% of range)
    near_low = np.sum(normalized < 0.1)
    near_high = np.sum(normalized > 0.9)
    total = len(normalized)

    bimodal_fraction = (near_low + near_high) / total

    if bimodal_fraction > 0.95:
        # Very bimodal - strong TTL indicator
        score += 50
    elif bimodal_fraction > 0.85:
        score += 35
    elif bimodal_fraction > 0.70:
        score += 20

    # 3. Check if the high value looks like TTL (around 5V typical)
    # Common TTL values: 5V, 3.3V, or similar discrete levels
    if 4.5 <= data_max <= 5.5:
        score += 20  # Looks like 5V TTL
    elif 3.0 <= data_max <= 3.6:
        score += 15  # Looks like 3.3V logic
    elif data_max > 2.0 and data_min < 0.5:
        score += 10  # Has reasonable on/off levels

    # 4. Check for sharp transitions (derivative has large spikes)
    if len(data) > 1000:
        # Sample for speed
        sample = data[::max(1, len(data) // 1000)]
    else:
        sample = data

    diff = np.abs(np.diff(sample))
    if len(diff) > 0:
        # Check if most differences are near zero (stable) with some large jumps
        small_diffs = np.sum(diff < data_range * 0.05)
        large_diffs = np.sum(diff > data_range * 0.5)

        if small_diffs / len(diff) > 0.95 and large_diffs > 0:
            # Mostly stable with some sharp transitions - TTL-like
            score += 20

    # 5. Penalty if the channel name suggests it's NOT a stimulus
    # (This won't work since we're just looking at data, but good to note)

    return score


def auto_select_channels(sweeps_by_channel: Dict[str, np.ndarray],
                         channel_names: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-select stimulus and analysis channels based on data patterns.

    Returns (stim_channel, analysis_channel):
    - stim_channel: Detected stimulus channel name, or None
    - analysis_channel: Suggested analysis channel, or None
    """
    # First, detect stimulus channel
    stim_channel = detect_stimulus_channel(sweeps_by_channel, channel_names)

    analysis_channel = None

    if stim_channel:
        # Find remaining channels (excluding stimulus)
        remaining = [ch for ch in channel_names if ch != stim_channel]

        # Only auto-select analysis channel if exactly one remains
        # If multiple channels exist, let user choose
        if len(remaining) == 1:
            analysis_channel = remaining[0]
        # If len(remaining) > 1, leave analysis_channel as None

    return stim_channel, analysis_channel
