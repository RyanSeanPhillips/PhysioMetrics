"""
Photometry Processing Module

Functions for loading, processing, and analyzing fiber photometry data.
Supports the standard CSV format from photometry acquisition systems.

File patterns:
- FP_data_*.csv: Raw photometry data (channel ID, time, fluorescence)
- AI data_*.csv: Analog inputs (thermal stim, pleth, etc.)
- timestamps_*.csv: Common time vector
"""

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
    Find companion AI data file in the same folder or parent folder.

    Neurophotometrics file structure:
        experiment_folder/
        ├── AI data_0.csv          ← Analog inputs (pleth, thermal stim, etc.)
        ├── timestamps_0.csv       ← Time vector (optional, for internal use)
        └── FP_data_0/
            └── FP_data_0.csv      ← Photometry data

    Args:
        fp_path: Path to FP_data file

    Returns:
        Dict with keys 'fp_data', 'ai_data' and Path values (or None if not found)
    """
    result = {
        'fp_data': fp_path,
        'ai_data': None
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


def get_file_preview(path: Path, n_rows: int = 5) -> Tuple[List[str], List[List[str]]]:
    """
    Get a preview of a CSV file (headers and first few rows).

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
            raw_lines = [f.readline() for _ in range(n_rows + 1)]

        # Try to detect the number of columns from a data row (not header)
        # Find a row that looks like numeric data
        n_cols = 2  # default
        for line in raw_lines[1:]:  # Skip first line (possible header)
            parts = line.strip().split(',')
            # Check if this looks like numeric data
            try:
                for p in parts[:3]:  # Check first few values
                    float(p)
                n_cols = len(parts)
                break
            except (ValueError, IndexError):
                continue

        # Generate column names (col1, col2, etc.)
        col_names = [f'col{i+1}' for i in range(n_cols)]

        # Parse rows - split by comma and take first n_cols values
        # Skip non-numeric header rows
        rows = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')

            # Check if first value is numeric (skip header rows)
            try:
                float(parts[0])
            except (ValueError, IndexError):
                continue  # Skip non-numeric rows (likely headers)

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

def load_photometry_csv(path: Path) -> pd.DataFrame:
    """
    Load and parse a photometry CSV file.

    Args:
        path: Path to FP_data CSV file

    Returns:
        DataFrame with columns: col1, col2, col3 (channel), col4 (time_ms), col5 (signal)
    """
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
    if subsample > 1:
        # Read with subsampling - skip rows that aren't multiples of subsample
        # This is faster than reading all then slicing for large files
        df = pd.read_csv(path, header=None, low_memory=False,
                         skiprows=lambda i: i > 0 and i % subsample != 0)
    else:
        df = pd.read_csv(path, header=None, low_memory=False)

    df.columns = [f'col{i+1}' for i in range(df.shape[1])]

    # Ensure numeric (headers will become NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are all NaN (header rows)
    df = df.dropna(how='all').reset_index(drop=True)

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
# Signal Processing (to be expanded in later phases)
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


# Placeholder functions for later phases
def interpolate_to_uniform(signal: np.ndarray, times: np.ndarray,
                           target_hz: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Resample signal to uniform time base. (Phase 4)"""
    raise NotImplementedError("Will be implemented in Phase 4: Processing")


def compute_dff_simple(ca_norm: np.ndarray, iso_norm: np.ndarray) -> np.ndarray:
    """Simple subtraction ΔF/F. (Phase 4)"""
    raise NotImplementedError("Will be implemented in Phase 4: Processing")


def compute_dff_fitted(ca: np.ndarray, iso: np.ndarray,
                       method: str = 'linear') -> np.ndarray:
    """Fitted ΔF/F with isosbestic correction. (Phase 4)"""
    raise NotImplementedError("Will be implemented in Phase 4: Processing")
