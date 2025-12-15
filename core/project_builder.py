"""
Project Builder - Batch processing workflow for PhysioMetrics

This module provides functionality for:
- Auto-discovering ABF and Excel files in a directory
- Extracting protocol information from ABF files
- Parsing Excel files for experiment metadata
- Managing batch processing workflows
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pyabf


def discover_files(directory: str, recursive: bool = True,
                   file_types: List[str] = None) -> Dict[str, List[Path]]:
    """
    Discover data files and notes files in a directory.

    Args:
        directory: Path to directory to search
        recursive: If True, search subdirectories recursively
        file_types: Optional list of file types to scan for.
                    Options: 'abf', 'smrx', 'edf', 'notes'
                    If None, scans for all types.

    Returns:
        Dictionary with keys:
            'abf_files': List of Path objects for .abf files
            'smrx_files': List of Path objects for .smrx files
            'edf_files': List of Path objects for .edf files
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
        file_types = ['abf', 'smrx', 'edf', 'notes']

    abf_files = []
    smrx_files = []
    edf_files = []
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

    # Sort files by name for consistent ordering
    abf_files.sort()
    smrx_files.sort()
    edf_files.sort()
    notes_files.sort()

    return {
        'abf_files': abf_files,
        'smrx_files': smrx_files,
        'edf_files': edf_files,
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
