"""
Project Builder - Batch processing workflow for PhysioMetrics

This module provides functionality for:
- Auto-discovering ABF, SMRX, EDF, and photometry files in a directory
- Extracting protocol information from ABF files
- Extracting metadata from Neurophotometrics photometry folders
- Parsing Excel files for experiment metadata
- Managing batch processing workflows
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pyabf
import pandas as pd


def discover_files(directory: str, recursive: bool = True,
                   file_types: List[str] = None) -> Dict[str, List[Path]]:
    """
    Discover data files and notes files in a directory.

    Args:
        directory: Path to directory to search
        recursive: If True, search subdirectories recursively
        file_types: Optional list of file types to scan for.
                    Options: 'abf', 'smrx', 'edf', 'photometry', 'notes'
                    If None, scans for all types.

    Returns:
        Dictionary with keys:
            'abf_files': List of Path objects for .abf files
            'smrx_files': List of Path objects for .smrx files
            'edf_files': List of Path objects for .edf files
            'photometry_files': List of Path objects for Neurophotometrics FP_data folders
            'notes_files': List of Path objects for notes (.xlsx, .xls, .docx, .doc, .txt)
            'excel_files': Legacy alias for notes_files (for backward compatibility)
    """
    directory_path = Path(directory)

    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not directory_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Default to all types if none specified
    if file_types is None:
        file_types = ['abf', 'smrx', 'edf', 'photometry', 'notes']

    abf_files = []
    smrx_files = []
    edf_files = []
    photometry_files = []
    notes_files = []

    # Define extensions for each category
    data_extensions = {
        'abf': ['.abf'],
        'smrx': ['.smrx'],
        'edf': ['.edf'],
    }
    notes_extensions = ['.xlsx', '.xls', '.docx', '.doc', '.txt']

    # Build list of extensions to search for
    target_extensions = set()
    if 'abf' in file_types:
        target_extensions.update(data_extensions['abf'])
    if 'smrx' in file_types:
        target_extensions.update(data_extensions['smrx'])
    if 'edf' in file_types:
        target_extensions.update(data_extensions['edf'])
    if 'notes' in file_types:
        target_extensions.update(notes_extensions)

    # Use extension-specific glob patterns to avoid stat() calls on non-target files
    # This is much faster on network drives
    for ext in target_extensions:
        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"

        # glob() with extension pattern is faster than checking is_file() on every file
        for file_path in directory_path.glob(pattern):
            # Only check is_file() if there's ambiguity (some systems allow dirs with extensions)
            # Skip this check for performance on network drives - glob with extension is reliable
            suffix_lower = file_path.suffix.lower()

            # Data files
            if suffix_lower in data_extensions.get('abf', []):
                abf_files.append(file_path)
            elif suffix_lower in data_extensions.get('smrx', []):
                smrx_files.append(file_path)
            elif suffix_lower in data_extensions.get('edf', []):
                edf_files.append(file_path)
            # Notes files
            elif suffix_lower in notes_extensions:
                notes_files.append(file_path)

    # Discover photometry files (Neurophotometrics FP_data folders)
    if 'photometry' in file_types:
        photometry_files = discover_photometry_files(directory_path, recursive=recursive)

    # Sort files by name for consistent ordering
    abf_files.sort()
    smrx_files.sort()
    edf_files.sort()
    photometry_files.sort()
    notes_files.sort()

    return {
        'abf_files': abf_files,
        'smrx_files': smrx_files,
        'edf_files': edf_files,
        'photometry_files': photometry_files,
        'notes_files': notes_files,
        'excel_files': notes_files,  # Legacy alias for backward compatibility
    }


def extract_abf_protocol_info(abf_file_path: Path) -> Optional[Dict]:
    """
    Extract protocol information from an ABF file.

    Args:
        abf_file_path: Path to the ABF file

    Returns:
        Dictionary containing:
            'protocol': Protocol name string
            'duration_sec': Recording duration in seconds
            'sample_rate': Sample rate in Hz
            'channels': List of channel names
            'sweeps': Number of sweeps/episodes
            'file_size_mb': File size in megabytes
            'creation_time': File creation timestamp

        Returns None if file cannot be read
    """
    try:
        abf = pyabf.ABF(str(abf_file_path), loadData=False)  # Don't load data, just metadata

        info = {
            'file_path': abf_file_path,
            'file_name': abf_file_path.name,
            'protocol': abf.protocol if hasattr(abf, 'protocol') else 'Unknown',
            'duration_sec': abf.dataLengthSec if hasattr(abf, 'dataLengthSec') else 0,
            'sample_rate': abf.dataRate if hasattr(abf, 'dataRate') else 0,
            'channels': [abf.adcNames[i] for i in range(abf.channelCount)] if hasattr(abf, 'adcNames') else [],
            'sweeps': abf.sweepCount if hasattr(abf, 'sweepCount') else 0,
            'file_size_mb': abf_file_path.stat().st_size / (1024 * 1024),
            'creation_time': abf.abfDateTime if hasattr(abf, 'abfDateTime') else None,
        }

        return info

    except Exception as e:
        print(f"[project-builder] Error reading ABF file {abf_file_path}: {e}")
        return None


def scan_directory_with_metadata(directory: str, recursive: bool = True, progress_callback=None) -> Dict:
    """
    Scan directory for files and extract metadata from ABF files.

    Args:
        directory: Path to directory to search
        recursive: If True, search subdirectories recursively
        progress_callback: Optional callback function called periodically (e.g., to update UI)

    Returns:
        Dictionary with keys:
            'abf_files': List of dicts with ABF metadata
            'excel_files': List of Path objects for Excel files
            'abf_count': Number of ABF files found
            'excel_count': Number of Excel files found
            'total_duration_sec': Total duration of all ABF files
            'protocols': Set of unique protocol names found
    """
    # Discover all files
    files = discover_files(directory, recursive=recursive)

    abf_metadata = []
    protocols = set()
    total_duration = 0

    # Extract metadata from each ABF file
    total_files = len(files['abf_files'])
    print(f"[project-builder] Scanning {total_files} ABF files...")

    for i, abf_path in enumerate(files['abf_files']):
        info = extract_abf_protocol_info(abf_path)
        if info:
            abf_metadata.append(info)
            protocols.add(info['protocol'])
            total_duration += info['duration_sec']

        # Call progress callback every 5 files to keep UI responsive
        if progress_callback and (i % 5 == 0 or i == total_files - 1):
            progress_callback(i + 1, total_files)

    print(f"[project-builder] Found {len(protocols)} unique protocols:")
    for protocol in sorted(protocols):
        print(f"  - {protocol}")

    return {
        'abf_files': abf_metadata,
        'excel_files': files['excel_files'],
        'abf_count': len(abf_metadata),
        'excel_count': len(files['excel_files']),
        'total_duration_sec': total_duration,
        'protocols': protocols
    }


def group_files_by_protocol(abf_metadata: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group ABF files by protocol name.

    Args:
        abf_metadata: List of ABF metadata dictionaries

    Returns:
        Dictionary mapping protocol names to lists of ABF metadata dicts
    """
    grouped = {}

    for abf_info in abf_metadata:
        protocol = abf_info['protocol']
        if protocol not in grouped:
            grouped[protocol] = []
        grouped[protocol].append(abf_info)

    return grouped


# =============================================================================
# Photometry File Discovery and Metadata Extraction
# =============================================================================

# Known LED wavelength mappings for Neurophotometrics systems
# LedState values may vary by system configuration
LED_WAVELENGTH_MAP = {
    1: '415nm (Isosbestic)',
    2: '470nm (GCaMP/GRABNE)',
    4: '560nm (RCaMP/jRGECO)',
    7: 'Multi-LED/Init',
}


def discover_photometry_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Discover Neurophotometrics photometry data files/folders.

    Neurophotometrics creates folders named FP_data_0, FP_data_1, etc.
    containing FP_data_X.csv files with the actual photometry data.

    Args:
        directory: Path to directory to search
        recursive: If True, search subdirectories recursively

    Returns:
        List of Path objects pointing to FP_data CSV files
    """
    photometry_files = []

    # Pattern to match FP_data folders (case-insensitive)
    # e.g., FP_data_0, FP_data_1, fp_data_0
    if recursive:
        pattern = "**/FP_data_*"
    else:
        pattern = "FP_data_*"

    # Find all FP_data folders/files
    seen_paths = set()  # Track to avoid duplicates

    for path in directory.glob(pattern):
        # Case-insensitive check for FP_data pattern
        name_lower = path.name.lower()
        if not name_lower.startswith('fp_data'):
            continue

        # If it's a directory, look for the CSV file inside
        if path.is_dir():
            # Look for FP_data_X.csv inside (same name as folder)
            csv_name = f"{path.name}.csv"
            csv_path = path / csv_name
            if csv_path.exists():
                resolved = csv_path.resolve()
                if resolved not in seen_paths:
                    photometry_files.append(csv_path)
                    seen_paths.add(resolved)
            else:
                # Try case-insensitive search for the CSV
                for f in path.iterdir():
                    if f.suffix.lower() == '.csv' and f.name.lower().startswith('fp_data'):
                        resolved = f.resolve()
                        if resolved not in seen_paths:
                            photometry_files.append(f)
                            seen_paths.add(resolved)
                        break

        # If it's a CSV file directly (standalone FP_data_0.csv)
        elif path.is_file() and path.suffix.lower() == '.csv':
            resolved = path.resolve()
            if resolved not in seen_paths:
                photometry_files.append(path)
                seen_paths.add(resolved)

    return photometry_files


def extract_photometry_info(fp_csv_path: Path) -> Optional[Dict]:
    """
    Extract metadata from a Neurophotometrics photometry CSV file.

    Args:
        fp_csv_path: Path to FP_data_X.csv file

    Returns:
        Dictionary containing:
            'file_path': Path to the FP_data CSV file
            'file_name': Display name (folder or file name)
            'file_type': 'photometry'
            'protocol': 'Neurophotometrics' or extracted from path
            'signal_columns': List of signal column names (e.g., ['G0', 'G1'])
            'region_count': Number of recorded regions/fibers
            'led_states': Set of unique LED state values found
            'led_info': Human-readable LED info string
            'row_count': Approximate number of data rows
            'duration_sec': Approximate duration in seconds
            'sample_rate': Approximate sample rate in Hz
            'has_ai_data': Whether companion AI_data file exists
            'ai_data_path': Path to AI_data file (if exists)
            'has_timestamps': Whether companion timestamps file exists
            'timestamps_path': Path to timestamps file (if exists)
            'has_npz': Whether processed .npz file exists
            'npz_path': Path to .npz file (if exists)
            'file_size_mb': File size in megabytes

        Returns None if file cannot be read
    """
    try:
        if not fp_csv_path.exists():
            return None

        # Determine the experiment folder (parent of FP_data folder or same folder)
        if fp_csv_path.parent.name.lower().startswith('fp_data'):
            experiment_folder = fp_csv_path.parent.parent
            display_name = fp_csv_path.parent.name  # e.g., "FP_data_0"
        else:
            experiment_folder = fp_csv_path.parent
            display_name = fp_csv_path.stem  # e.g., "FP_data_0"

        # Extract suffix (e.g., "_0" from "FP_data_0")
        match = re.search(r'fp_data(_\d+)?', display_name, re.IGNORECASE)
        suffix = match.group(1) if match and match.group(1) else ''

        # Read header and first chunk of data to extract metadata
        # Only read enough to determine structure (fast on network drives)
        df_sample = pd.read_csv(fp_csv_path, nrows=1000, low_memory=False)

        # Identify signal columns (typically G0, G1, R0, R1, etc.)
        # These are columns after the standard metadata columns
        standard_cols = {'framecounter', 'systemtimestamp', 'ledstate',
                         'computertimestamp', 'flags'}
        signal_columns = []
        for col in df_sample.columns:
            col_lower = col.lower()
            if col_lower not in standard_cols:
                # Check if it looks like a signal column (G0, G1, R0, R1, etc.)
                if re.match(r'^[A-Za-z]\d+$', col):
                    signal_columns.append(col)

        region_count = len(signal_columns)

        # Get LED states (skip first row if it looks like initialization)
        led_col = None
        for col in df_sample.columns:
            if col.lower() == 'ledstate':
                led_col = col
                break

        led_states = set()
        led_info = "Unknown"
        if led_col:
            # Convert to numeric, coercing errors
            led_values = pd.to_numeric(df_sample[led_col], errors='coerce')
            # Filter out NaN and initialization values (7 is often init)
            valid_led = led_values.dropna()
            led_states = set(int(v) for v in valid_led.unique() if v in [1, 2, 4])
            if not led_states:
                # Include all found values if none match known patterns
                led_states = set(int(v) for v in valid_led.unique())

            # Build human-readable LED info
            led_descriptions = []
            for state in sorted(led_states):
                if state in LED_WAVELENGTH_MAP:
                    led_descriptions.append(LED_WAVELENGTH_MAP[state])
                else:
                    led_descriptions.append(f"LED {state}")
            led_info = ", ".join(led_descriptions) if led_descriptions else "Unknown"

        # Estimate duration and sample rate from timestamps
        time_col = None
        for col in df_sample.columns:
            if col.lower() in ('systemtimestamp', 'computertimestamp'):
                time_col = col
                break

        duration_sec = 0.0
        sample_rate = 0.0
        if time_col and len(df_sample) > 10:
            times = pd.to_numeric(df_sample[time_col], errors='coerce').dropna()
            if len(times) > 1:
                # Estimate sample interval from time deltas (more reliable than absolute values)
                time_diffs = times.diff().dropna()
                if len(time_diffs) > 0:
                    avg_interval = time_diffs.median()
                    if avg_interval > 0:
                        # Determine time units from interval size:
                        # - If interval is ~0.01-0.1 sec, timestamps are in seconds (typical 10-100Hz)
                        # - If interval is ~10-100, timestamps are in milliseconds
                        # Neurophotometrics typically samples at 30-60 Hz per LED
                        if avg_interval > 1:  # Interval > 1 suggests milliseconds
                            sample_rate = 1000.0 / avg_interval  # Convert ms interval to Hz
                        else:
                            sample_rate = 1.0 / avg_interval  # Already in seconds

        # Get approximate row count from file size (faster than counting)
        file_size = fp_csv_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # Estimate rows from sample
        sample_bytes = len(df_sample.to_csv(index=False).encode())
        if sample_bytes > 0 and len(df_sample) > 0:
            bytes_per_row = sample_bytes / len(df_sample)
            row_count = int(file_size / bytes_per_row) if bytes_per_row > 0 else 0
        else:
            row_count = 0

        # Estimate duration if we have sample rate and row count
        if sample_rate > 0 and row_count > 0:
            # Each LED state is sampled separately, so divide by number of LED states
            n_led = len(led_states) if led_states else 2
            duration_sec = row_count / (sample_rate * n_led)

        # Find companion files
        ai_data_path = None
        timestamps_path = None
        npz_path = None

        # Look for AI_data file
        for folder in [experiment_folder, fp_csv_path.parent]:
            if not folder.exists():
                continue
            for pattern_name in [f'AI_data{suffix}.csv', f'AI data{suffix}.csv',
                                 f'ai_data{suffix}.csv', f'ai data{suffix}.csv']:
                candidate = folder / pattern_name
                if candidate.exists():
                    ai_data_path = candidate
                    break
            if ai_data_path:
                break

        # Look for timestamps file
        for folder in [experiment_folder, fp_csv_path.parent]:
            if not folder.exists():
                continue
            for pattern_name in [f'timestamps{suffix}.csv', f'Timestamps{suffix}.csv']:
                candidate = folder / pattern_name
                if candidate.exists():
                    timestamps_path = candidate
                    break
            if timestamps_path:
                break

        # Look for existing processed .npz file
        npz_pattern = f"{display_name}*_photometry.npz"
        npz_files = list(experiment_folder.glob(npz_pattern))
        if npz_files:
            # Get most recently modified
            npz_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            npz_path = npz_files[0]

        # Try to extract protocol/experiment info from path
        protocol = "Neurophotometrics"
        # Check parent folders for meaningful names
        for parent in [experiment_folder, experiment_folder.parent]:
            parent_name = parent.name
            # Skip generic folder names
            if parent_name.lower() not in ('data', 'raw', 'recordings', ''):
                if not parent_name.lower().startswith('fp_data'):
                    protocol = parent_name
                    break

        return {
            'file_path': fp_csv_path,
            'file_name': display_name,
            'file_type': 'photometry',
            'protocol': protocol,
            'signal_columns': signal_columns,
            'region_count': region_count,
            'led_states': led_states,
            'led_info': led_info,
            'row_count': row_count,
            'duration_sec': duration_sec,
            'sample_rate': round(sample_rate, 1),
            'has_ai_data': ai_data_path is not None,
            'ai_data_path': ai_data_path,
            'has_timestamps': timestamps_path is not None,
            'timestamps_path': timestamps_path,
            'has_npz': npz_path is not None,
            'npz_path': npz_path,
            'file_size_mb': round(file_size_mb, 2),
            'creation_time': None,  # Could add from file stat if needed
        }

    except Exception as e:
        print(f"[project-builder] Error reading photometry file {fp_csv_path}: {e}")
        return None


# Test function for development
if __name__ == "__main__":
    # Test with examples directory
    test_dir = Path(__file__).parent.parent / "examples"
    if test_dir.exists():
        print(f"Testing file discovery in: {test_dir}")
        results = scan_directory_with_metadata(str(test_dir))
        print(f"\nResults:")
        print(f"  ABF files: {results['abf_count']}")
        print(f"  Excel files: {results['excel_count']}")
        print(f"  Total duration: {results['total_duration_sec']:.1f} seconds")
        print(f"  Protocols found: {results['protocols']}")

    # Test photometry discovery
    print("\n--- Testing Photometry Discovery ---")
    photometry_test = Path(r"Z:\DATA\Jonathan\GRABNE Aug 2025 Cohort")
    if photometry_test.exists():
        print(f"Scanning for photometry files in: {photometry_test}")
        fp_files = discover_photometry_files(photometry_test, recursive=True)
        print(f"Found {len(fp_files)} photometry files")
        for fp in fp_files[:5]:  # Show first 5
            print(f"  - {fp}")
            info = extract_photometry_info(fp)
            if info:
                print(f"      Regions: {info['region_count']}, LEDs: {info['led_info']}")
                print(f"      AI data: {info['has_ai_data']}, NPZ: {info['has_npz']}")
