"""
Photometry Processing Module

Functions for loading, processing, and analyzing fiber photometry data.
Supports the standard CSV format from photometry acquisition systems.

File patterns:
- FP_data_*.csv: Raw photometry data (channel ID, time, fluorescence)
- AI data_*.csv: Analog inputs (thermal stim, pleth, etc.)
- timestamps_*.csv: Common time vector
"""

import ast
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# File Detection
# =============================================================================

def detect_photometry_file(path: Path) -> bool:
    """
    Check if a file looks like photometry data based on filename and structure.

    Args:
        path: Path to file to check

    Returns:
        True if file appears to be photometry data
    """
    if not path.exists() or not path.is_file():
        return False

    # Check filename patterns
    name = path.name.lower()

    # Primary pattern: FP_data*.csv
    if re.match(r'fp_data.*\.csv$', name, re.IGNORECASE):
        return True

    # Check if it's in a folder that looks like photometry data
    # e.g., "FP_data_0/FP_data_0.csv"
    if path.parent.name.lower().startswith('fp_data'):
        return True

    # If CSV, check structure
    if path.suffix.lower() == '.csv':
        try:
            # Read first few rows to check structure
            df = pd.read_csv(path, nrows=10, header=None)

            # Photometry files typically have 5+ columns
            # col3 contains channel IDs (1 or 2) - often called "LedState"
            # col4 contains timestamps (increasing values in ms)
            # col5 contains fluorescence values
            if df.shape[1] >= 5:
                # Check if first row is a header (contains "LedState" or similar)
                first_row = df.iloc[0].astype(str).str.lower()
                if 'ledstate' in first_row.values or 'led_state' in first_row.values:
                    return True

                # Also check if col3 contains channel identifiers (1 or 2)
                # Skip first row in case it's a header
                col3_vals = pd.to_numeric(df.iloc[1:, 2], errors='coerce').dropna().unique()
                if len(col3_vals) > 0 and set(col3_vals).issubset({1, 2, 1.0, 2.0}):
                    return True
        except Exception:
            pass

    return False


def find_companion_files(fp_path: Path) -> Dict[str, Optional[Path]]:
    """
    Find companion AI data and timestamps files in the same folder or parent folder.

    Neurophotometrics file structure:
        experiment_folder/
        ├── AI data_0.csv          ← Analog inputs (pleth, thermal stim, etc.)
        ├── timestamps_0.csv       ← Time vector (optional, for internal use)
        └── FP_data_0/
            └── FP_data_0.csv      ← Photometry data

    Args:
        fp_path: Path to FP_data file

    Returns:
        Dict with keys 'fp_data', 'ai_data', 'timestamps' and Path values (or None if not found)
    """
    result = {
        'fp_data': fp_path,
        'ai_data': None,
        'timestamps': None
    }

    # Extract the suffix (e.g., "_0" from "FP_data_0.csv")
    match = re.search(r'fp_data(_\d+)?', fp_path.stem, re.IGNORECASE)
    suffix = match.group(1) if match and match.group(1) else ''

    # Search locations: parent folder first (most common), then same folder
    search_dirs = []
    if fp_path.parent.name.lower().startswith('fp_data'):
        # FP_data is in its own subfolder - AI data is in parent
        search_dirs.append(fp_path.parent.parent)
    search_dirs.append(fp_path.parent)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Look for AI data file
        if result['ai_data'] is None:
            ai_patterns = [
                f'AI data{suffix}.csv',
                f'AI_data{suffix}.csv',
                f'ai data{suffix}.csv',
                f'ai_data{suffix}.csv',
            ]
            for pattern in ai_patterns:
                candidate = search_dir / pattern
                if candidate.exists():
                    result['ai_data'] = candidate
                    break

        # Look for timestamps file
        if result['timestamps'] is None:
            ts_patterns = [
                f'timestamps{suffix}.csv',
                f'Timestamps{suffix}.csv',
                f'TIMESTAMPS{suffix}.csv',
            ]
            for pattern in ts_patterns:
                candidate = search_dir / pattern
                if candidate.exists():
                    result['timestamps'] = candidate
                    break

    return result


def find_timestamps_file(fp_path: Path) -> Optional[Path]:
    """
    Find timestamps file for internal processing (not exposed in UI).

    Args:
        fp_path: Path to FP_data file

    Returns:
        Path to timestamps file, or None if not found
    """
    # Extract the suffix (e.g., "_0" from "FP_data_0.csv")
    match = re.search(r'fp_data(_\d+)?', fp_path.stem, re.IGNORECASE)
    suffix = match.group(1) if match and match.group(1) else ''

    # Search locations
    search_dirs = []
    if fp_path.parent.name.lower().startswith('fp_data'):
        search_dirs.append(fp_path.parent.parent)
    search_dirs.append(fp_path.parent)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        ts_patterns = [
            f'timestamps{suffix}.csv',
            f'Timestamps{suffix}.csv',
            f'time{suffix}.csv',
        ]
        for pattern in ts_patterns:
            candidate = search_dir / pattern
            if candidate.exists():
                return candidate

    return None


def find_existing_photometry_npz(fp_path: Path) -> Optional[Path]:
    """
    Check if a processed *_photometry.npz file exists for this raw data.

    When user opens FP_data_0.csv, check if a previously saved NPZ exists
    so we can offer to load it directly instead of reprocessing.

    Searches in order:
    1. NPZ registry (for files saved in different locations)
    2. Same folder as the source file
    3. Parent folder (legacy location)

    Args:
        fp_path: Path to raw FP_data file or NPZ file

    Returns:
        Path to existing NPZ file, or None if not found
    """
    # If already an NPZ file, return it directly
    if fp_path.suffix.lower() == '.npz':
        return fp_path if fp_path.exists() else None

    # First, check the NPZ registry for this source file
    # This handles cases where the NPZ was saved to a different location
    try:
        from core import config as app_config
        registry_entry = app_config.lookup_npz_for_source(fp_path)
        if registry_entry:
            npz_path = Path(registry_entry['npz_path'])
            if npz_path.exists():
                print(f"[Photometry] Found NPZ in registry: {npz_path.name}")
                return npz_path
    except Exception as e:
        print(f"[Photometry] Warning: Registry lookup failed: {e}")

    # Determine folder name for expected NPZ filename
    if fp_path.parent.name.lower().startswith('fp_data'):
        folder_name = fp_path.parent.parent.name  # e.g., "251212 Awake Hargreaves w Saline"
    else:
        folder_name = fp_path.parent.name

    # Search locations: same folder as fp_data first, then parent folder (legacy)
    search_dirs = [fp_path.parent]
    if fp_path.parent.name.lower().startswith('fp_data'):
        search_dirs.append(fp_path.parent.parent)

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        # Look for NPZ file with expected name: {folder_name}_photometry.npz
        expected_npz = search_dir / f"{folder_name}_photometry.npz"
        if expected_npz.exists():
            return expected_npz

        # Also check for any *_photometry.npz in case of custom naming
        npz_files = list(search_dir.glob("*_photometry.npz"))
        if npz_files:
            # Return most recently modified if multiple exist
            npz_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return npz_files[0]

    return None


def get_npz_experiment_info(npz_path: Path) -> Optional[Dict]:
    """
    Read experiment metadata from an NPZ file without loading all data.

    Args:
        npz_path: Path to photometry NPZ file

    Returns:
        Dict with:
            - n_experiments: Number of experiments
            - experiments: List of dicts with {index, animal_id, fibers}
            - file_paths: Original source file paths
        Or None if file can't be read
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # Get number of experiments
            n_experiments = int(data['n_experiments'][0]) if 'n_experiments' in data else 1

            # Get animal IDs
            animal_ids = {}
            if 'animal_ids' in data:
                animal_ids_str = str(data['animal_ids'][0])
                try:
                    animal_ids = ast.literal_eval(animal_ids_str)
                except Exception:
                    pass

            # Get fiber columns
            fiber_columns = []
            if 'fiber_columns' in data:
                fiber_columns = list(data['fiber_columns'])

            # Get experiment assignments
            assignments = {}
            if 'experiment_assignments' in data:
                try:
                    assignments = ast.literal_eval(str(data['experiment_assignments'][0]))
                except Exception:
                    pass

            # Get file paths
            file_paths = {}
            if 'file_paths' in data:
                try:
                    file_paths = ast.literal_eval(str(data['file_paths'][0]))
                except Exception:
                    pass

            # Build experiment info
            experiments = []
            for exp_idx in range(n_experiments):
                animal_id = animal_ids.get(exp_idx, '')
                # Find fibers assigned to this experiment
                exp_fibers = [fiber for fiber, exp in assignments.items()
                              if exp == exp_idx and not fiber.startswith('ai_')]
                experiments.append({
                    'index': exp_idx,
                    'animal_id': animal_id,
                    'fibers': exp_fibers
                })

            return {
                'n_experiments': n_experiments,
                'experiments': experiments,
                'file_paths': file_paths,
                'npz_path': str(npz_path)
            }

    except Exception as e:
        print(f"[Photometry] Error reading NPZ info: {e}")
        return None


def load_experiment_from_npz(npz_path: Path, exp_idx: int = 0) -> Optional[Dict]:
    """
    Load a specific experiment from an NPZ file, ready for the main app.

    Args:
        npz_path: Path to photometry NPZ file
        exp_idx: Which experiment to load (0-indexed)

    Returns:
        Dict in format expected by _load_photometry_data:
            - sweeps: Dict[str, np.ndarray] - channel_name -> (n_samples, 1) array
            - channel_names: List[str] - ordered list of channel names
            - channel_visibility: Dict[str, bool] - which channels visible by default
            - t: np.ndarray - time vector in seconds
            - sr_hz: float - sample rate
            - photometry_raw: Dict - raw data for recalculation
            - photometry_params: Dict - dF/F parameters
            - photometry_npz_path: Path - source file
            - dff_channel_name: str - name of the dF/F channel
            - experiment_index: int - which experiment this is
            - n_experiments: int - total number of experiments
            - animal_id: str - animal ID for this experiment
        Or None if loading fails
    """
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            # Get basic info
            n_experiments = int(data['n_experiments'][0]) if 'n_experiments' in data else 1
            sample_rate = float(data['sample_rate'][0]) if 'sample_rate' in data else 100.0
            common_time = data['common_time']  # Already in seconds

            # Get animal IDs
            animal_ids = {}
            if 'animal_ids' in data:
                try:
                    animal_ids = ast.literal_eval(str(data['animal_ids'][0]))
                except Exception:
                    pass
            animal_id = animal_ids.get(exp_idx, '')

            # Get dF/F parameters per experiment
            dff_params_all = {}
            if 'dff_params' in data:
                try:
                    dff_params_all = ast.literal_eval(str(data['dff_params'][0]))
                except Exception:
                    pass
            params = dff_params_all.get(exp_idx, {
                'method': 'fitted',
                'detrend_method': 'none',
                'lowpass_hz': None,
                'fit_start': 0,
                'fit_end': float(common_time[-1]) / 60.0 if len(common_time) > 0 else 0
            })

            # Get experiment assignments
            assignments = {}
            if 'experiment_assignments' in data:
                try:
                    assignments = ast.literal_eval(str(data['experiment_assignments'][0]))
                except Exception:
                    pass

            # Get fiber columns
            fiber_columns = list(data['fiber_columns']) if 'fiber_columns' in data else []

            print(f"[Photometry] Loading experiment {exp_idx}")
            print(f"[Photometry] Fiber columns from NPZ: {fiber_columns}")
            print(f"[Photometry] Assignments: {assignments}")

            # Find fibers assigned to this experiment
            # Assignments use channel names like "G0-GCaMP", not fiber column names like "G0"
            experiment_fibers = []
            for fiber_col in fiber_columns:
                # Check both the fiber column name and the channel name format
                gcamp_channel = f"{fiber_col}-GCaMP"
                assigned_exp = assignments.get(gcamp_channel, assignments.get(fiber_col, -1))
                print(f"[Photometry] Fiber {fiber_col}: assigned_exp={assigned_exp}")
                if assigned_exp == exp_idx or assigned_exp == -1:  # -1 means "All"
                    experiment_fibers.append(fiber_col)

            if not experiment_fibers and fiber_columns:
                # Fall back to all fibers
                print(f"[Photometry] No fibers matched, using all: {fiber_columns}")
                experiment_fibers = fiber_columns

            print(f"[Photometry] Experiment fibers to process: {experiment_fibers}")

            # Build sweeps and channels
            sweeps = {}
            channel_names = []
            channel_visibility = {}
            dff_channel_name = None

            # Process each fiber
            common_time_min = common_time / 60.0

            for fiber_col in experiment_fibers:
                # NPZ saves with 'fiber_' prefix
                iso_key = f'fiber_{fiber_col}_iso'
                gcamp_key = f'fiber_{fiber_col}_gcamp'

                if iso_key not in data or gcamp_key not in data:
                    print(f"[Photometry] Keys not found: {iso_key}, {gcamp_key}")
                    print(f"[Photometry] Available keys: {list(data.keys())}")
                    continue

                iso_signal = data[iso_key]
                gcamp_signal = data[gcamp_key]

                # Compute dF/F using the same approach as the dialog
                try:
                    method = params.get('method', 'fitted')
                    fit_start = params.get('fit_start', 0)
                    fit_end = params.get('fit_end', common_time_min[-1] if len(common_time_min) > 0 else 0)

                    # Compute dF/F based on method
                    if method == 'simple':
                        dff, fit_params = compute_dff_simple(gcamp_signal, iso_signal)
                        fitted_iso = iso_signal  # No fitting for simple method
                    else:
                        dff, fit_params = compute_dff_fitted(
                            gcamp_signal, iso_signal, common_time_min,
                            fit_start=fit_start,
                            fit_end=fit_end
                        )
                        fitted_iso = fit_params.get('fitted_iso', iso_signal)

                    # Apply detrending if requested
                    detrend_curve = None
                    detrend_method = params.get('detrend_method', 'none')
                    if detrend_method != 'none':
                        dff, detrend_curve, _ = detrend_signal(
                            dff, common_time_min,
                            method=detrend_method,
                            fit_start=fit_start,
                            fit_end=fit_end
                        )

                    # Apply lowpass filter if requested
                    lowpass_hz = params.get('lowpass_hz')
                    if lowpass_hz is not None and lowpass_hz > 0:
                        dff = lowpass_filter(dff, lowpass_hz, sample_rate)

                    # Add dF/F channel (VISIBLE by default)
                    dff_name = f'{fiber_col}-dF/F'
                    sweeps[dff_name] = dff.reshape(-1, 1)
                    channel_names.append(dff_name)
                    channel_visibility[dff_name] = True
                    if dff_channel_name is None:
                        dff_channel_name = dff_name

                    # Add raw signals (HIDDEN by default)
                    iso_name = f'{fiber_col}-Iso'
                    gcamp_name = f'{fiber_col}-GCaMP'
                    sweeps[iso_name] = iso_signal.reshape(-1, 1)
                    sweeps[gcamp_name] = gcamp_signal.reshape(-1, 1)
                    channel_names.extend([iso_name, gcamp_name])
                    channel_visibility[iso_name] = False
                    channel_visibility[gcamp_name] = False

                    # Add fitted isosbestic (HIDDEN by default)
                    if fitted_iso is not None:
                        fitted_name = f'{fiber_col}-Fitted'
                        sweeps[fitted_name] = fitted_iso.reshape(-1, 1)
                        channel_names.append(fitted_name)
                        channel_visibility[fitted_name] = False

                    # Add detrend curve if available (HIDDEN by default)
                    if detrend_curve is not None:
                        detrend_name = f'{fiber_col}-Detrend'
                        sweeps[detrend_name] = detrend_curve.reshape(-1, 1)
                        channel_names.append(detrend_name)
                        channel_visibility[detrend_name] = False

                except Exception as e:
                    print(f"[Photometry] Error computing dF/F for {fiber_col}: {e}")
                    import traceback
                    traceback.print_exc()

            # Add AI channels (VISIBLE by default)
            # NPZ saves with 'ai_channel_' prefix
            for key in data.files:
                if key.startswith('ai_channel_'):
                    ai_col = key.replace('ai_channel_', '')
                    ai_name = f'AI-{ai_col}'
                    sweeps[ai_name] = data[key].reshape(-1, 1)
                    channel_names.append(ai_name)
                    channel_visibility[ai_name] = True

            if not sweeps:
                print(f"[Photometry] No channels loaded from NPZ for experiment {exp_idx}")
                return None

            # Build raw data for recalculation
            fibers_raw = {}
            for fiber_col in fiber_columns:
                # NPZ saves with 'fiber_' prefix
                iso_key = f'fiber_{fiber_col}_iso'
                gcamp_key = f'fiber_{fiber_col}_gcamp'
                if iso_key in data and gcamp_key in data:
                    fibers_raw[fiber_col] = {
                        'iso': data[iso_key],
                        'gcamp': data[gcamp_key]
                    }

            ai_channels_raw = {}
            for key in data.files:
                # NPZ saves with 'ai_channel_' prefix
                if key.startswith('ai_channel_'):
                    ai_col = key.replace('ai_channel_', '')
                    ai_channels_raw[ai_col] = data[key]

            raw_photometry_data = {
                'fibers': fibers_raw,
                'common_time': common_time,
                'sample_rate': sample_rate,
                'ai_channels': ai_channels_raw,
            }

            print(f"[Photometry] Loaded experiment {exp_idx + 1} ({animal_id}) from NPZ")
            print(f"[Photometry]   {len(channel_names)} channels: {channel_names}")
            print(f"[Photometry]   Visible: {[c for c, v in channel_visibility.items() if v]}")

            return {
                'sweeps': sweeps,
                'channel_names': channel_names,
                'channel_visibility': channel_visibility,
                't': common_time,
                'sr_hz': sample_rate,
                'photometry_raw': raw_photometry_data,
                'photometry_params': params,
                'photometry_npz_path': npz_path,
                'dff_channel_name': dff_channel_name,
                'experiment_index': exp_idx,
                'n_experiments': n_experiments,
                'animal_id': animal_id,
            }

    except Exception as e:
        print(f"[Photometry] Error loading experiment from NPZ: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_file_preview(path: Path, n_rows: int = 5) -> Tuple[List[str], List[List[str]]]:
    """
    Get a preview of a CSV file (headers and first few rows).

    Shows actual column headers from the file when available.

    Args:
        path: Path to CSV file
        n_rows: Number of rows to preview

    Returns:
        Tuple of (column_names, rows_data)
    """
    if not path or not path.exists():
        return [], []

    try:
        # Read raw lines first to handle malformed CSVs
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            raw_lines = [f.readline() for _ in range(n_rows + 10)]  # Read extra for safety

        if not raw_lines:
            return [], []

        # Detect the number of columns from data rows (not header)
        n_cols = 2  # default
        for line in raw_lines[1:]:  # Skip first line (possible header)
            parts = line.strip().split(',')
            if not parts or not parts[0]:
                continue
            # Check if this looks like numeric data
            try:
                float(parts[0])
                n_cols = max(n_cols, len(parts))
                break
            except (ValueError, IndexError):
                continue

        # Check if first line is a header (non-numeric first value)
        first_line = raw_lines[0].strip()
        first_parts = first_line.split(',')
        has_header = False
        header_names = []

        if first_parts:
            first_val = first_parts[0].strip()
            try:
                float(first_val)
                # First value is numeric - no header row
                has_header = False
            except (ValueError, IndexError):
                # First value is not numeric - check if it's a valid header or malformed
                # Skip malformed Bonsai headers like 'ToString("G17"'
                if first_val.startswith('ToString') or first_val.startswith('"') or ')' in first_val:
                    has_header = False  # Treat as no header, use generic names
                else:
                    has_header = True
                    header_names = [p.strip().strip('"\'') for p in first_parts]

        # Build column names - use actual headers if available, pad with col# if needed
        col_names = []
        for i in range(n_cols):
            if has_header and i < len(header_names) and header_names[i]:
                # Use actual header name (truncate if too long)
                name = header_names[i][:20]
                col_names.append(name)
            else:
                # Fall back to generic name
                col_names.append(f'col{i+1}')

        # Parse data rows (skip header if present)
        rows = []
        start_idx = 1 if has_header else 0
        for line in raw_lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')

            # Check if first value is numeric (skip any additional header-like rows)
            try:
                float(parts[0])
            except (ValueError, IndexError):
                continue  # Skip non-numeric rows

            # Pad or truncate to n_cols
            row_vals = []
            for i in range(n_cols):
                if i < len(parts):
                    row_vals.append(parts[i][:15])  # Truncate long values
                else:
                    row_vals.append('')
            rows.append(row_vals)

            if len(rows) >= n_rows:
                break

        return col_names, rows
    except Exception as e:
        print(f"[photometry] Error reading {path}: {e}")
        return [], []


def get_file_info(path: Path, count_rows: bool = False) -> Dict:
    """
    Get basic info about a CSV file.

    Args:
        path: Path to CSV file
        count_rows: If True, count all rows (slow for large files on network drives)
                   If False, estimate rows from file size (fast)

    Returns:
        Dict with file info (exists, rows, cols, size_mb)
    """
    if not path or not path.exists():
        return {'exists': False, 'rows': 0, 'cols': 0, 'size_mb': 0}

    try:
        # Get file size (fast - just stat call)
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        # Get column count from first row (fast - only reads beginning of file)
        df = pd.read_csv(path, nrows=5, header=None)
        col_count = df.shape[1]

        if count_rows:
            # Count rows by reading entire file (slow for network drives)
            with open(path, 'rb') as f:
                row_count = sum(1 for _ in f)
        else:
            # Estimate rows from file size and average row size
            # Use the 5 rows we already read to estimate bytes per row
            sample_size = len(df.to_csv(index=False, header=False).encode())
            bytes_per_row = sample_size / 5 if sample_size > 0 else 100
            row_count = int(size_bytes / bytes_per_row)  # Approximate

        return {
            'exists': True,
            'rows': row_count,
            'rows_estimated': not count_rows,
            'cols': col_count,
            'size_mb': round(size_mb, 2)
        }
    except Exception as e:
        print(f"[photometry] Error getting file info for {path}: {e}")
        return {'exists': False, 'rows': 0, 'cols': 0, 'size_mb': 0, 'error': str(e)}


# =============================================================================
# Data Loading
# =============================================================================

def load_photometry_csv(path: Path, preserve_headers: bool = True) -> pd.DataFrame:
    """
    Load and parse a photometry CSV file.

    Args:
        path: Path to FP_data CSV file
        preserve_headers: If True, preserve actual column names from file header.
                         If False, use generic col1, col2, ... names (legacy behavior).

    Returns:
        DataFrame with photometry data
    """
    if preserve_headers:
        # Load with headers - standard Neurophotometrics format has headers
        df = pd.read_csv(path, low_memory=False)

        # Check if first row looks like data (all numeric) vs header
        # If headers are missing, the column names will be the first data row
        first_row_numeric = True
        for col in df.columns:
            try:
                float(col)
            except (ValueError, TypeError):
                first_row_numeric = False
                break

        if first_row_numeric:
            # No headers - reload without header and use generic names
            df = pd.read_csv(path, header=None, low_memory=False)
            df.columns = [f'col{i+1}' for i in range(df.shape[1])]

        # Convert data columns to numeric (skip if already numeric)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows that are all NaN
        df = df.dropna(how='all').reset_index(drop=True)

        return df
    else:
        # Legacy behavior: generic column names
        df = pd.read_csv(path, header=None, low_memory=False)
        df.columns = [f'col{i+1}' for i in range(df.shape[1])]

        # Ensure numeric (headers will become NaN)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows that are all NaN (header rows)
        df = df.dropna(how='all').reset_index(drop=True)

        return df


def load_ai_data_csv(path: Path, subsample: int = 1) -> pd.DataFrame:
    """
    Load analog input data CSV.

    Args:
        path: Path to AI data CSV file
        subsample: Load every Nth row (1 = all rows, 10 = every 10th row)

    Returns:
        DataFrame with generic column names
    """
    # Some Bonsai files have malformed headers (e.g., 'ToString("G17", )')
    # that have fewer columns than data rows. We need to detect the actual
    # column count from data rows, not from the header.

    # First, detect the number of columns from data rows
    n_cols = 2  # default
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # Skip header line
                if i > 10:
                    break  # Check first 10 data lines
                parts = line.strip().split(',')
                # Check if this looks like numeric data
                try:
                    float(parts[0])
                    n_cols = max(n_cols, len(parts))
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"[photometry] Error detecting columns in {path}: {e}")

    print(f"[photometry] AI data: detected {n_cols} columns")

    # Generate column names for the expected number of columns
    col_names = [f'col{i+1}' for i in range(n_cols)]

    # Read CSV with explicit column names to handle mismatched header
    try:
        if subsample > 1:
            df = pd.read_csv(path, header=None, names=col_names,
                             low_memory=False, on_bad_lines='warn',
                             skiprows=lambda i: i > 0 and i % subsample != 0)
        else:
            df = pd.read_csv(path, header=None, names=col_names,
                             low_memory=False, on_bad_lines='warn')
    except TypeError:
        # Older pandas versions use error_bad_lines instead of on_bad_lines
        if subsample > 1:
            df = pd.read_csv(path, header=None, names=col_names,
                             low_memory=False, warn_bad_lines=True, error_bad_lines=False,
                             skiprows=lambda i: i > 0 and i % subsample != 0)
        else:
            df = pd.read_csv(path, header=None, names=col_names,
                             low_memory=False, warn_bad_lines=True, error_bad_lines=False)

    # Ensure numeric (headers will become NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are all NaN (header rows)
    df = df.dropna(how='all').reset_index(drop=True)

    print(f"[photometry] AI data loaded: {len(df)} rows, {len(df.columns)} columns")

    return df


def load_timestamps_csv(path: Path, subsample: int = 1) -> np.ndarray:
    """
    Load timestamps CSV.

    Args:
        path: Path to timestamps CSV file
        subsample: Load every Nth row (1 = all rows, 10 = every 10th row)

    Returns:
        1D numpy array of timestamps in ms
    """
    if subsample > 1:
        df = pd.read_csv(path, header=None, low_memory=False,
                         skiprows=lambda i: i > 0 and i % subsample != 0)
    else:
        df = pd.read_csv(path, header=None, low_memory=False)

    timestamps = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    # Remove NaN values (from headers)
    return timestamps[~np.isnan(timestamps)]


# =============================================================================
# Fiber Detection (Multi-fiber support)
# =============================================================================

def detect_fiber_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect fiber signal columns in a photometry DataFrame.

    Neurophotometrics systems name fiber columns as G0, G1, G2... for green
    (GCaMP) and R0, R1, R2... for red indicators.

    Args:
        df: DataFrame with photometry data (with original column names)

    Returns:
        List of fiber column names found (e.g., ['G0', 'G1'] or ['col5', 'col6'])
    """
    fiber_cols = []

    for col in df.columns:
        # Match standard Neurophotometrics naming: G0, G1, R0, R1, etc.
        if re.match(r'^[GRgr]\d+$', str(col)):
            fiber_cols.append(col)

    # If no named fiber columns found, fall back to positional detection
    # Columns after FrameCounter, SystemTimestamp, LedState, ComputerTimestamp
    # are typically fiber signals
    if not fiber_cols:
        cols = list(df.columns)
        # Standard layout: col1=FrameCounter, col2=SysTime, col3=LedState, col4=CompTime, col5+=signals
        if len(cols) >= 5:
            # Check if columns 5+ contain numeric signal data
            for i in range(4, len(cols)):
                col = cols[i]
                try:
                    # Check if column contains numeric data in expected range for fluorescence
                    sample = pd.to_numeric(df[col].head(10), errors='coerce')
                    if sample.notna().any():
                        fiber_cols.append(col)
                except Exception:
                    pass

    return fiber_cols


def detect_led_states(df: pd.DataFrame, led_col: str = 'LedState') -> Dict[int, str]:
    """
    Detect and interpret LED states in photometry data.

    Standard Neurophotometrics LED states:
    - 1: Isosbestic (415nm) - motion/hemodynamic reference
    - 2: GCaMP (470nm) - calcium indicator
    - 4: Red channel (560nm) - for red indicators like jRGECO, dLight
    - 7: Sync/initialization frame (skip)

    Args:
        df: DataFrame with photometry data
        led_col: Name of the LED state column

    Returns:
        Dict mapping LED state value to interpretation string
    """
    # Find LED column (might be named or positional)
    if led_col not in df.columns:
        # Try to find by position (usually col3)
        cols = list(df.columns)
        if len(cols) >= 3:
            led_col = cols[2]
        else:
            return {}

    try:
        unique_states = df[led_col].dropna().unique()
        unique_states = [int(x) for x in unique_states if not np.isnan(x)]
    except Exception:
        return {}

    interpretations = {
        1: 'Isosbestic (415nm)',
        2: 'GCaMP (470nm)',
        4: 'Red (560nm)',
        7: 'Sync (skip)',
    }

    result = {}
    for state in sorted(unique_states):
        result[state] = interpretations.get(state, f'Unknown (LED {state})')

    return result


def get_fiber_type(fiber_col: str) -> str:
    """
    Determine if a fiber column is green (GCaMP) or red (jRGECO/dLight).

    Args:
        fiber_col: Column name like 'G0', 'R1', etc.

    Returns:
        'green' for G0/G1/etc, 'red' for R0/R1/etc, 'unknown' otherwise
    """
    col_upper = str(fiber_col).upper()
    if col_upper.startswith('G'):
        return 'green'
    elif col_upper.startswith('R'):
        return 'red'
    return 'unknown'


def get_signal_led_for_fiber(fiber_col: str, green_led: int = 2, red_led: int = 4) -> int:
    """
    Get the appropriate signal LED state for a fiber column.

    Args:
        fiber_col: Column name like 'G0', 'R1', etc.
        green_led: LED state for green/GCaMP signal (default 2 = 470nm)
        red_led: LED state for red signal (default 4 = 560nm)

    Returns:
        LED state integer for the signal channel
    """
    fiber_type = get_fiber_type(fiber_col)
    if fiber_type == 'red':
        return red_led
    return green_led  # Default to green


# Standard Neurophotometrics LED state interpretations
# LED states use bit flags: L415=1, L470=2, L560=4
# Combined states are sums (e.g., 3=L415+L470, 5=L415+L560, 7=all)
LED_STATE_NAMES = {
    0: 'Off/Unknown',
    1: 'L415 - 415nm (Isosbestic)',
    2: 'L470 - 470nm (GCaMP/Green)',
    3: 'L415+L470 (415nm + 470nm)',
    4: 'L560 - 560nm (Red/jRGECO)',
    5: 'L415+L560 (415nm + 560nm)',
    6: 'L470+L560 (470nm + 560nm)',
    7: 'Sync/Init (all LEDs)',
}

# LED state to channel type mapping
LED_STATE_TYPES = {
    1: 'iso',      # 415nm isosbestic reference
    2: 'green',    # 470nm for GCaMP, GFP, etc.
    4: 'red',      # 560nm for jRGECO, RCaMP, dLight
    7: 'sync',     # Sync frame - skip
}


def get_available_led_states(data: pd.DataFrame, led_col: str = 'LedState') -> Dict[int, Dict]:
    """
    Detect all available LED states in the data and return their info.

    Args:
        data: DataFrame with photometry data
        led_col: Name of the LED state column

    Returns:
        Dict mapping LED state value to info dict with:
            - 'name': Human-readable name
            - 'count': Number of samples with this state
            - 'is_sync': Whether this is a sync/skip state
    """
    # Find LED column
    if led_col not in data.columns:
        cols = list(data.columns)
        if len(cols) >= 3:
            led_col = cols[2]
        else:
            return {}

    try:
        led_counts = data[led_col].value_counts()
    except Exception:
        return {}

    result = {}
    for state, count in led_counts.items():
        try:
            state_int = int(state)
        except (ValueError, TypeError):
            continue

        # Skip NaN or invalid states
        if np.isnan(state) if isinstance(state, float) else False:
            continue

        result[state_int] = {
            'name': LED_STATE_NAMES.get(state_int, f'LED {state_int}'),
            'count': int(count),
            'is_sync': state_int == 7,  # State 7 is typically sync
        }

    return result


def auto_detect_led_mapping(data: pd.DataFrame, led_col: str = 'LedState') -> Dict[str, int]:
    """
    Auto-detect which LED states to use for isosbestic and signal channels.

    Logic:
    - State 1 is always isosbestic (415nm)
    - If state 2 exists, it's green signal (470nm GCaMP)
    - If state 4 exists, it's red signal (560nm)
    - If only two non-sync states exist, assume iso + signal

    Args:
        data: DataFrame with photometry data
        led_col: Name of the LED state column

    Returns:
        Dict with keys: 'iso', 'green', 'red' mapping to LED state integers
        Values are None if that LED type is not present
    """
    available = get_available_led_states(data, led_col)

    # Filter out sync states
    signal_states = {k: v for k, v in available.items() if not v['is_sync']}

    result = {
        'iso': None,
        'green': None,
        'red': None,
    }

    # State 1 is always isosbestic
    if 1 in signal_states:
        result['iso'] = 1

    # State 2 is green (470nm GCaMP)
    if 2 in signal_states:
        result['green'] = 2

    # State 4 is red (560nm)
    if 4 in signal_states:
        result['red'] = 4

    # Handle non-standard configurations
    # If we have exactly 2 states and one is 1 (iso), the other is signal
    if len(signal_states) == 2 and 1 in signal_states:
        other_state = [s for s in signal_states.keys() if s != 1][0]
        if result['green'] is None and result['red'] is None:
            # Guess based on state number
            if other_state == 4:
                result['red'] = other_state
            else:
                result['green'] = other_state

    print(f"[photometry] Auto-detected LED mapping: iso={result['iso']}, green={result['green']}, red={result['red']}")
    print(f"[photometry] Available LED states: {signal_states}")

    return result


def separate_channels_multi_fiber(
    data: pd.DataFrame,
    led_col: str,
    time_col: str,
    fiber_cols: List[str],
    iso_led: int = None,
    signal_led: int = None,
    red_led: int = None,
    auto_detect_type: bool = True,
    auto_detect_leds: bool = True,
    led_mapping: Dict[str, int] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Separate isosbestic and signal channels for multiple fibers.

    Supports both green (GCaMP, 470nm) and red (jRGECO/dLight, 560nm) indicators,
    as well as any custom LED state configuration.

    Fiber type is auto-detected from column names:
    - G0, G1, G2... = Green channels
    - R0, R1, R2... = Red channels

    LED states are auto-detected from the data:
    - Standard: 1=415nm iso, 2=470nm green, 4=560nm red, 7=sync
    - Non-standard: auto-detected based on what's present

    Args:
        data: DataFrame with photometry data
        led_col: Column containing LED state identifier
        time_col: Column containing timestamps
        fiber_cols: List of fiber signal column names (e.g., ['G0', 'G1', 'R0'])
        iso_led: LED state for isosbestic (default: auto-detect, typically 1)
        signal_led: LED state for green signal (default: auto-detect, typically 2)
        red_led: LED state for red signal (default: auto-detect, typically 4)
        auto_detect_type: If True, detect fiber type from column name (G=green, R=red)
        auto_detect_leds: If True, auto-detect LED states from data
        led_mapping: Optional explicit mapping {'iso': int, 'green': int, 'red': int}

    Returns:
        Dict mapping fiber name to:
            - 'iso_time': array of isosbestic timestamps
            - 'iso': array of isosbestic signal values
            - 'signal_time': array of signal timestamps
            - 'signal': array of signal values
            - 'fiber_type': 'green' or 'red'
            - 'signal_led': LED state used for this fiber's signal
    """
    result = {}

    # Auto-detect LED mapping if not provided
    if auto_detect_leds and (iso_led is None or signal_led is None or red_led is None):
        detected = auto_detect_led_mapping(data, led_col)
        if iso_led is None:
            iso_led = detected.get('iso', 1)
        if signal_led is None:
            signal_led = detected.get('green', 2)
        if red_led is None:
            red_led = detected.get('red', 4)

    # Override with explicit mapping if provided
    if led_mapping:
        if 'iso' in led_mapping and led_mapping['iso'] is not None:
            iso_led = led_mapping['iso']
        if 'green' in led_mapping and led_mapping['green'] is not None:
            signal_led = led_mapping['green']
        if 'red' in led_mapping and led_mapping['red'] is not None:
            red_led = led_mapping['red']

    # Use defaults if still None
    if iso_led is None:
        iso_led = 1
    if signal_led is None:
        signal_led = 2
    if red_led is None:
        red_led = 4

    # Create mask for isosbestic (shared by all fiber types)
    iso_mask = data[led_col] == iso_led
    iso_time = data.loc[iso_mask, time_col].values

    # Pre-filter DataFrame once per mask — column indexing on the result is cheap
    iso_data = data.loc[iso_mask]

    # Create masks for all signal LEDs we might use
    signal_times = {}
    signal_data = {}  # Pre-filtered DataFrames keyed by LED state

    for led_state in set([signal_led, red_led]):
        if led_state is not None:
            mask = data[led_col] == led_state
            signal_times[led_state] = data.loc[mask, time_col].values
            signal_data[led_state] = data.loc[mask]

    # Check which LED states are present
    has_green = signal_led in signal_times and len(signal_times[signal_led]) > 0
    has_red = red_led in signal_times and len(signal_times[red_led]) > 0

    print(f"[photometry] LED configuration: iso=LED {iso_led} ({len(iso_time)} samples)")
    if has_green:
        print(f"[photometry]   green=LED {signal_led} ({len(signal_times[signal_led])} samples)")
    if has_red:
        print(f"[photometry]   red=LED {red_led} ({len(signal_times[red_led])} samples)")

    for fiber_col in fiber_cols:
        if fiber_col not in data.columns:
            print(f"[photometry] Warning: Fiber column '{fiber_col}' not found")
            continue

        # Determine fiber type from column name
        if auto_detect_type:
            fiber_type = get_fiber_type(fiber_col)
        else:
            fiber_type = 'green'  # Default to green if not auto-detecting

        # Select appropriate signal LED based on fiber type and availability
        if fiber_type == 'red' and has_red:
            used_led = red_led
            print(f"[photometry] Fiber {fiber_col}: Using RED channel (LED {red_led})")
        elif fiber_type == 'green' and has_green:
            used_led = signal_led
            print(f"[photometry] Fiber {fiber_col}: Using GREEN channel (LED {signal_led})")
        elif has_green:
            # Fallback to green if red not available
            used_led = signal_led
            fiber_type = 'green'
            print(f"[photometry] Fiber {fiber_col}: RED not available, using GREEN (LED {signal_led})")
        elif has_red:
            # Fallback to red if green not available
            used_led = red_led
            fiber_type = 'red'
            print(f"[photometry] Fiber {fiber_col}: GREEN not available, using RED (LED {red_led})")
        else:
            print(f"[photometry] Warning: No signal data for fiber {fiber_col}")
            continue

        # Extract signals for this fiber (from pre-filtered DataFrames — no repeated mask application)
        signal_time = signal_times[used_led]

        iso_signal = iso_data[fiber_col].values
        signal_signal = signal_data[used_led][fiber_col].values

        result[fiber_col] = {
            'iso_time': iso_time,
            'iso': iso_signal,
            'signal_time': signal_time,
            'signal': signal_signal,
            'fiber_type': fiber_type,
            'signal_led': used_led,
            'iso_led': iso_led,
        }

        print(f"[photometry] Fiber {fiber_col} ({fiber_type}): {len(iso_signal)} iso samples, "
              f"{len(signal_signal)} signal samples (LED {used_led})")

    return result


# =============================================================================
# Signal Processing
# =============================================================================

def separate_channels(data: pd.DataFrame,
                      channel_col: str = 'col3',
                      time_col: str = 'col4',
                      signal_col: str = 'col5',
                      iso_id: int = 1,
                      ca_id: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate isosbestic and calcium channels from combined photometry data.

    Args:
        data: DataFrame with photometry data
        channel_col: Column containing channel identifier
        time_col: Column containing timestamps
        signal_col: Column containing fluorescence signal
        iso_id: Channel ID for isosbestic (default 1 = 415nm)
        ca_id: Channel ID for calcium (default 2 = 470nm)

    Returns:
        Tuple of (isosbestic_df, calcium_df) each with 'time' and 'signal' columns
    """
    iso_mask = data[channel_col] == iso_id
    ca_mask = data[channel_col] == ca_id

    iso_df = data.loc[iso_mask, [time_col, signal_col]].reset_index(drop=True)
    iso_df.columns = ['time', 'signal']

    ca_df = data.loc[ca_mask, [time_col, signal_col]].reset_index(drop=True)
    ca_df.columns = ['time', 'signal']

    return iso_df, ca_df


# =============================================================================
# dF/F Computation Functions
# =============================================================================

def interpolate_to_common_time(
    iso_time: np.ndarray, iso_signal: np.ndarray,
    gcamp_time: np.ndarray, gcamp_signal: np.ndarray,
    exclude_start_min: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Interpolate isosbestic and GCaMP signals to a common time base.

    Args:
        iso_time: Time vector for isosbestic channel (minutes)
        iso_signal: Isosbestic fluorescence signal
        gcamp_time: Time vector for GCaMP channel (minutes)
        gcamp_signal: GCaMP fluorescence signal
        exclude_start_min: Exclude this many minutes from the start

    Returns:
        Tuple of (common_time, iso_aligned, gcamp_aligned, sample_rate_hz)
    """
    if len(iso_time) == 0 or len(gcamp_time) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0

    # Find common time range
    t_min = max(np.min(iso_time), np.min(gcamp_time))
    t_max = min(np.max(iso_time), np.max(gcamp_time))

    # Apply exclusion of initial transient
    if exclude_start_min > 0:
        t_min = t_min + exclude_start_min

    if t_max <= t_min:
        print("[Photometry] Warning: No data remaining after exclusion")
        return np.array([]), np.array([]), np.array([]), 0.0

    # Create common time vector (use the shorter signal's sample rate)
    n_points = min(len(iso_time), len(gcamp_time))
    common_time = np.linspace(t_min, t_max, n_points)

    # Calculate sample rate (samples per minute -> Hz)
    if len(common_time) > 1:
        dt_min = (common_time[-1] - common_time[0]) / (len(common_time) - 1)
        fs = 1.0 / (dt_min * 60)  # Convert to Hz
    else:
        fs = 20.0  # Default assumption

    # Interpolate both signals to common time base
    # np.interp is faster than scipy.interp1d for linear interpolation (no object overhead)
    iso_aligned = np.interp(common_time, iso_time, iso_signal)
    gcamp_aligned = np.interp(common_time, gcamp_time, gcamp_signal)

    return common_time, iso_aligned, gcamp_aligned, fs


def lowpass_filter(signal: np.ndarray, cutoff_hz: float, sample_rate_hz: float,
                   order: int = 2) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to signal.

    Args:
        signal: Input signal
        cutoff_hz: Cutoff frequency in Hz
        sample_rate_hz: Sample rate in Hz
        order: Filter order (default 2)

    Returns:
        Filtered signal
    """
    from scipy import signal as scipy_signal

    nyq = sample_rate_hz / 2
    if cutoff_hz >= nyq:
        print(f"[Photometry] Cutoff {cutoff_hz} Hz >= Nyquist {nyq} Hz, skipping filter")
        return signal

    normalized_cutoff = cutoff_hz / nyq
    sos = scipy_signal.butter(order, normalized_cutoff, btype='low', output='sos')
    filtered = scipy_signal.sosfiltfilt(sos, signal)
    print(f"[Photometry] Applied {cutoff_hz} Hz low-pass filter (fs={sample_rate_hz:.1f} Hz)")
    return filtered


def compute_dff_simple(gcamp: np.ndarray, iso: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Compute ΔF/F using simple normalized subtraction.

    Formula: ΔF/F = (GCaMP_norm - Iso_norm) / Iso_norm * 100
    Where each signal is first normalized by its mean.

    Args:
        gcamp: GCaMP signal (aligned to common time)
        iso: Isosbestic signal (aligned to common time)

    Returns:
        Tuple of (dff_signal, params_dict)
    """
    iso_mean = np.mean(iso)
    gcamp_mean = np.mean(gcamp)

    iso_norm = iso / iso_mean
    gcamp_norm = gcamp / gcamp_mean

    epsilon = np.abs(iso_norm).mean() * 1e-6
    dff = (gcamp_norm - iso_norm) / (iso_norm + epsilon) * 100

    params = {
        'method': 'simple',
        'iso_mean': iso_mean,
        'gcamp_mean': gcamp_mean,
        'iso_normalized': iso_norm,
        'gcamp_normalized': gcamp_norm
    }

    print(f"[Photometry] Simple subtraction: iso_mean={iso_mean:.2f}, gcamp_mean={gcamp_mean:.2f}")
    return dff, params


def compute_dff_fitted(
    gcamp: np.ndarray, iso: np.ndarray, time: np.ndarray,
    fit_start: float = 0.0, fit_end: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    Compute ΔF/F using linear regression fitting.

    Fits isosbestic to GCaMP via linear regression, then:
    ΔF/F = (GCaMP - fitted_iso) / fitted_iso * 100

    Args:
        gcamp: GCaMP signal (aligned to common time)
        iso: Isosbestic signal (aligned to common time)
        time: Common time vector (minutes)
        fit_start: Start of fitting window (minutes from start, 0 = use all)
        fit_end: End of fitting window (minutes from start, 0 = use all)

    Returns:
        Tuple of (dff_signal, params_dict)
    """
    from scipy import stats

    t_normalized = time - time[0]  # Time starting at 0

    # Determine fit window
    if fit_end > fit_start:
        fit_mask = (t_normalized >= fit_start) & (t_normalized <= fit_end)
        if np.sum(fit_mask) < 10:  # Need at least 10 points
            fit_mask = np.ones(len(t_normalized), dtype=bool)
            print(f"[Photometry] Iso fit window too narrow, using all data")
        else:
            print(f"[Photometry] Fitting iso regression to {fit_start:.1f}-{fit_end:.1f} min window")
    else:
        fit_mask = np.ones(len(t_normalized), dtype=bool)

    # Fit regression only on the selected window
    iso_for_fit = iso[fit_mask]
    gcamp_for_fit = gcamp[fit_mask]
    slope, intercept, r_value, p_value, std_err = stats.linregress(iso_for_fit, gcamp_for_fit)

    # Apply fit to ALL data
    fitted_iso = slope * iso + intercept
    r_squared = r_value**2

    # Log fitting results with interpretation
    print(f"[Photometry] dF/F fitting: slope={slope:.4f}, intercept={intercept:.4f}, R^2={r_squared:.4f}")
    if r_squared < 0.1:
        print(f"[Photometry] Low R^2 indicates minimal shared artifacts - good signal quality!")
    elif r_squared > 0.5:
        print(f"[Photometry] High R^2 indicates significant shared artifacts - fitting will help")

    # ΔF/F = (GCaMP - fitted_iso) / fitted_iso
    epsilon = np.abs(fitted_iso).mean() * 1e-6
    dff = (gcamp - fitted_iso) / (fitted_iso + epsilon) * 100

    params = {
        'method': 'fitted',
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'fitted_iso': fitted_iso
    }

    return dff, params


def detrend_signal(
    signal: np.ndarray, time: np.ndarray,
    method: str = 'none',
    fit_start: float = 0.0, fit_end: float = 0.0
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """
    Apply detrending to remove slow drift from signal.

    Args:
        signal: Input signal (typically ΔF/F)
        time: Time vector (minutes)
        method: 'none', 'linear', 'exponential', or 'biexponential'
        fit_start: Start of fitting window (minutes from start)
        fit_end: End of fitting window (minutes from start)

    Returns:
        Tuple of (detrended_signal, trend_curve, params_dict)
    """
    from scipy.optimize import curve_fit

    t_normalized = time - time[0]  # Time starting at 0

    # Determine fit window
    if fit_end > fit_start:
        fit_mask = (t_normalized >= fit_start) & (t_normalized <= fit_end)
        if np.sum(fit_mask) < 10:
            fit_mask = np.ones(len(t_normalized), dtype=bool)
            print(f"[Photometry] Fit window too narrow, using all data")
        else:
            print(f"[Photometry] Fitting detrend to {fit_start:.1f}-{fit_end:.1f} min")
    else:
        fit_mask = np.ones(len(t_normalized), dtype=bool)

    t_for_fit = t_normalized[fit_mask]
    signal_for_fit = signal[fit_mask]

    if method == 'none':
        return signal.copy(), None, {'detrend_method': 'none'}

    elif method == 'linear':
        coeffs = np.polyfit(t_for_fit, signal_for_fit, deg=1)
        trend = np.polyval(coeffs, t_normalized)
        detrended = signal - trend
        print(f"[Photometry] Linear detrend: slope={coeffs[0]:.4f}/min, intercept={coeffs[1]:.2f}%")

        return detrended, trend, {
            'detrend_method': 'linear',
            'detrend_slope': coeffs[0],
            'detrend_intercept': coeffs[1]
        }

    elif method == 'exponential':
        def exp_decay(t, a, tau, b):
            return a * np.exp(-t / tau) + b

        try:
            n_pts = len(signal_for_fit)
            early_mean = np.mean(signal_for_fit[:max(1, n_pts//10)])
            late_mean = np.mean(signal_for_fit[-max(1, n_pts//10):])

            a0 = early_mean - late_mean
            tau0 = (t_for_fit[-1] - t_for_fit[0]) / 2 if len(t_for_fit) > 1 else 1.0
            b0 = late_mean

            if abs(a0) < 0.1:
                print(f"[Photometry] No exponential decay detected (delta={a0:.3f}%), skipping")
                return signal.copy(), None, {'detrend_method': 'none (no decay)'}

            popt, pcov = curve_fit(exp_decay, t_for_fit, signal_for_fit,
                                   p0=[a0, tau0, b0],
                                   maxfev=10000,
                                   bounds=([-100, 0.1, -100], [100, t_normalized[-1]*10, 100]))

            exp_fit = exp_decay(t_normalized, *popt)
            detrended = signal - exp_fit + popt[2]
            print(f"[Photometry] Exponential detrend: a={popt[0]:.2f}%, tau={popt[1]:.2f} min, b={popt[2]:.2f}%")

            return detrended, exp_fit, {
                'detrend_method': 'exponential',
                'exp_a': popt[0],
                'exp_tau': popt[1],
                'exp_b': popt[2]
            }

        except Exception as e:
            print(f"[Photometry] Exponential fit failed: {e}, using linear instead")
            coeffs = np.polyfit(t_for_fit, signal_for_fit, deg=1)
            trend = np.polyval(coeffs, t_normalized)
            return signal - trend, trend, {'detrend_method': 'linear (exp failed)'}

    elif method == 'biexponential':
        def biexp_decay(t, a1, tau1, a2, tau2, b):
            return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + b

        try:
            n_pts = len(signal_for_fit)
            early_mean = np.mean(signal_for_fit[:max(1, n_pts//10)])
            late_mean = np.mean(signal_for_fit[-max(1, n_pts//10):])
            total_decay = early_mean - late_mean

            t_range = (t_for_fit[-1] - t_for_fit[0]) if len(t_for_fit) > 1 else 1.0
            a1_0 = total_decay * 0.7
            tau1_0 = max(0.1, t_range / 10)
            a2_0 = total_decay * 0.3
            tau2_0 = max(0.5, t_range / 2)
            b0 = late_mean

            if abs(total_decay) < 0.1:
                print(f"[Photometry] No decay detected (delta={total_decay:.3f}%), skipping biexp")
                return signal.copy(), None, {'detrend_method': 'none (no decay)'}

            popt, pcov = curve_fit(biexp_decay, t_for_fit, signal_for_fit,
                                   p0=[a1_0, tau1_0, a2_0, tau2_0, b0],
                                   maxfev=20000,
                                   bounds=([-100, 0.01, -100, 0.1, -100],
                                           [100, t_normalized[-1]*5, 100, t_normalized[-1]*10, 100]))

            biexp_fit = biexp_decay(t_normalized, *popt)
            detrended = signal - biexp_fit + popt[4]
            print(f"[Photometry] Biexponential detrend: a1={popt[0]:.2f}%, tau1={popt[1]:.2f}min, "
                  f"a2={popt[2]:.2f}%, tau2={popt[3]:.2f}min, b={popt[4]:.2f}%")

            return detrended, biexp_fit, {
                'detrend_method': 'biexponential',
                'biexp_a1': popt[0],
                'biexp_tau1': popt[1],
                'biexp_a2': popt[2],
                'biexp_tau2': popt[3],
                'biexp_b': popt[4]
            }

        except Exception as e:
            print(f"[Photometry] Biexponential fit failed: {e}, using linear instead")
            coeffs = np.polyfit(t_for_fit, signal_for_fit, deg=1)
            trend = np.polyval(coeffs, t_normalized)
            return signal - trend, trend, {'detrend_method': 'linear (biexp failed)'}

    else:
        print(f"[Photometry] Unknown detrend method '{method}', skipping")
        return signal.copy(), None, {'detrend_method': 'none (unknown)'}


def compute_dff_full(
    iso_time: np.ndarray, iso_signal: np.ndarray,
    gcamp_time: np.ndarray, gcamp_signal: np.ndarray,
    method: str = 'fitted',
    detrend_method: str = 'none',
    lowpass_hz: Optional[float] = None,
    exclude_start_min: float = 0.0,
    fit_start: float = 0.0, fit_end: float = 0.0,
    return_intermediates: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Full ΔF/F computation pipeline.

    Processing steps:
    1. Interpolate both channels to common time base
    2. (Optional) Exclude initial transient period
    3. (Optional) Low-pass filter both signals
    4. Compute ΔF/F using selected method
    5. (Optional) Detrend ΔF/F to remove drift
    6. Clip extreme values to ±50%

    Args:
        iso_time, iso_signal: Isosbestic channel data
        gcamp_time, gcamp_signal: Calcium channel data
        method: 'fitted' (regression) or 'simple' (direct subtraction)
        detrend_method: 'none', 'linear', 'exponential', or 'biexponential'
        lowpass_hz: If provided, apply low-pass filter at this frequency
        exclude_start_min: Exclude this many minutes from the start
        fit_start, fit_end: Time window for fitting (detrend and iso regression)
        return_intermediates: If True, return dict with intermediate processing data

    Returns:
        If return_intermediates=False: (common_time, dff_signal, None)
        If return_intermediates=True: (common_time, dff_signal, intermediates_dict)
    """
    # Initialize intermediates
    intermediates = {
        'time': None,
        'iso_aligned': None,
        'gcamp_aligned': None,
        'iso_normalized': None,
        'gcamp_normalized': None,
        'fitted_iso': None,
        'dff_raw': None,
        'detrend_curve': None,
        'fit_params': {}
    }

    # Step 1: Interpolate to common time base
    common_time, iso_aligned, gcamp_aligned, fs = interpolate_to_common_time(
        iso_time, iso_signal, gcamp_time, gcamp_signal, exclude_start_min
    )

    if len(common_time) == 0:
        if return_intermediates:
            return np.array([]), np.array([]), intermediates
        return np.array([]), np.array([]), None

    intermediates['time'] = common_time.copy()
    intermediates['iso_aligned'] = iso_aligned.copy()
    intermediates['gcamp_aligned'] = gcamp_aligned.copy()

    # Step 2: Optional low-pass filter
    if lowpass_hz is not None and fs > 2 * lowpass_hz:
        iso_aligned = lowpass_filter(iso_aligned, lowpass_hz, fs)
        gcamp_aligned = lowpass_filter(gcamp_aligned, lowpass_hz, fs)

    # Step 3: Compute ΔF/F
    if method == 'fitted':
        dff, dff_params = compute_dff_fitted(gcamp_aligned, iso_aligned, common_time, fit_start, fit_end)
        intermediates['fitted_iso'] = dff_params.get('fitted_iso')
    else:
        dff, dff_params = compute_dff_simple(gcamp_aligned, iso_aligned)
        intermediates['iso_normalized'] = dff_params.get('iso_normalized')
        intermediates['gcamp_normalized'] = dff_params.get('gcamp_normalized')

    intermediates['fit_params'].update(dff_params)
    intermediates['dff_raw'] = dff.copy()

    # Step 4: Detrend
    dff, trend_curve, detrend_params = detrend_signal(dff, common_time, detrend_method, fit_start, fit_end)
    intermediates['detrend_curve'] = trend_curve
    intermediates['fit_params'].update(detrend_params)

    # Step 5: Clip extreme values
    dff_clipped = np.clip(dff, -50, 50)
    if not np.allclose(dff, dff_clipped):
        n_clipped = np.sum(dff != dff_clipped)
        print(f"[Photometry] Warning: Clipped {n_clipped} extreme values to ±50%")
        dff = dff_clipped

    if return_intermediates:
        return common_time, dff, intermediates
    return common_time, dff, None
