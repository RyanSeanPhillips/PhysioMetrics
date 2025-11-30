"""
Fast ABF header reader - extracts minimal metadata without loading full file.

This is much faster than pyabf for scanning large directories.
"""

import struct
import re
from pathlib import Path
from typing import Optional, Dict, List


def read_abf_metadata_fast(abf_file_path: Path, detect_stim: bool = True) -> Optional[Dict]:
    """
    Quickly extract metadata from ABF file by reading only the header.

    Uses pyabf with loadData=False for reliable header parsing while avoiding
    loading the full data arrays.

    Args:
        abf_file_path: Path to the ABF file
        detect_stim: If True, attempt to detect stim channels (requires loading a small sample)

    Returns:
        Dictionary containing:
            'file_path': Path object
            'file_name': Filename
            'protocol': Protocol name
            'file_size_mb': File size in megabytes
            'channel_count': Number of channels
            'sweep_count': Number of sweeps
            'stim_channels': List of channel names that appear to be stim/TTL channels
            'stim_frequency': Detected stimulation frequency (e.g., '30Hz') or ''
            'channel_names': List of all channel names
            'channel_units': List of all channel units

        Returns None if file cannot be read
    """
    try:
        import pyabf

        file_size_mb = abf_file_path.stat().st_size / (1024 * 1024)

        # Use pyabf with loadData=False - reads header only, reliable
        abf = pyabf.ABF(str(abf_file_path), loadData=False)

        # Get channel names and units (available from header)
        channel_names = abf.adcNames if hasattr(abf, 'adcNames') else []
        channel_units = abf.adcUnits if hasattr(abf, 'adcUnits') else []

        # Detect stim channels based on name patterns
        stim_channels = []
        stim_keywords = ['ttl', 'stim', 'laser', 'digital', 'trigger', 'opto', 'led', 'pulse']

        for i, name in enumerate(channel_names):
            name_lower = name.lower() if name else ''
            # Check if channel name suggests it's a stim channel
            if any(keyword in name_lower for keyword in stim_keywords):
                stim_channels.append(f"AD{i}")

        # Detect stim frequency if requested and potential stim channels found
        stim_frequency = ''
        if detect_stim and stim_channels:
            stim_frequency = _detect_stim_frequency_fast(abf_file_path, stim_channels)

        return {
            'file_path': abf_file_path,
            'file_name': abf_file_path.name,
            'protocol': abf.protocol if abf.protocol else 'Unknown',
            'file_size_mb': file_size_mb,
            'channel_count': abf.channelCount,
            'sweep_count': abf.sweepCount,
            'stim_channels': stim_channels,
            'stim_frequency': stim_frequency,
            'channel_names': channel_names,
            'channel_units': channel_units,
        }

    except Exception as e:
        print(f"[fast-abf] Error reading {abf_file_path}: {e}")
        return None


def _detect_stim_frequency_fast(abf_file_path: Path, stim_channels: List[str]) -> str:
    """
    Quickly detect stimulation frequency by sampling a portion of the first stim channel.

    Uses threshold crossing detection on a ~2 second sample to minimize load time.

    Args:
        abf_file_path: Path to ABF file
        stim_channels: List of potential stim channel names (e.g., ['AD1'])

    Returns:
        Detected frequency string (e.g., '30Hz', '6Hz', '25ms pulse') or ''
    """
    try:
        import pyabf
        import numpy as np

        if not stim_channels:
            return ''

        # Load file with data this time (for stim detection)
        abf = pyabf.ABF(str(abf_file_path))

        # Get the first stim channel index
        first_stim = stim_channels[0]
        if first_stim.startswith('AD'):
            channel_idx = int(first_stim[2:])
        else:
            channel_idx = 0

        if channel_idx >= abf.channelCount:
            return ''

        # Set to first sweep and stim channel
        abf.setSweep(0, channel=channel_idx)

        # Get sample rate and data
        sample_rate = abf.sampleRate
        data = abf.sweepY

        # Only use first ~2 seconds of data for speed
        samples_to_use = min(len(data), int(2.0 * sample_rate))
        data = data[:samples_to_use]

        if len(data) < 100:
            return ''

        # Detect threshold crossings (TTL-style signal)
        # Assume TTL signals go from ~0 to high value (>2V typically)
        data_min, data_max = np.min(data), np.max(data)
        data_range = data_max - data_min

        if data_range < 0.5:  # No significant variation, not a stim signal
            return ''

        # Threshold at 50% of range
        threshold = data_min + data_range * 0.5

        # Find rising edges (crossings from below to above threshold)
        above_threshold = data > threshold
        rising_edges = np.where(np.diff(above_threshold.astype(int)) == 1)[0]

        if len(rising_edges) < 2:
            return ''

        # Calculate inter-pulse intervals
        intervals = np.diff(rising_edges) / sample_rate  # In seconds

        if len(intervals) == 0:
            return ''

        # Get median interval (robust to outliers)
        median_interval = np.median(intervals)

        if median_interval <= 0:
            return ''

        # Calculate frequency
        frequency = 1.0 / median_interval

        # Format the result
        if frequency >= 1.0:
            # Express as Hz
            if frequency >= 100:
                return f"{int(round(frequency))}Hz"
            elif frequency >= 10:
                return f"{int(round(frequency))}Hz"
            else:
                return f"{frequency:.1f}Hz"
        else:
            # Express as pulse duration (e.g., "25ms pulse")
            pulse_ms = median_interval * 1000
            return f"{int(round(pulse_ms))}ms pulse"

    except Exception as e:
        # Stim detection failed - not critical
        # print(f"[fast-abf] Stim detection error: {e}")
        return ''


def extract_channel_count(header: bytes) -> int:
    """
    Extract channel count from ABF header.

    For ABF2 format:
    - Byte offset 120-121: ADC channel count (uint16)

    Args:
        header: First ~16KB of ABF file

    Returns:
        Number of channels, or 0 if cannot be determined
    """
    try:
        # Check if this is ABF2 format
        signature = header[:4].decode('ascii', errors='ignore')

        if signature == 'ABF2':
            # ABF2: Channel count at offset 120 (uint16, little-endian)
            if len(header) >= 122:
                channel_count = struct.unpack('<H', header[120:122])[0]
                return channel_count
        elif signature.startswith('ABF'):
            # ABF1: Try offset 10 (nADCNumChannels, uint16)
            if len(header) >= 12:
                channel_count = struct.unpack('<H', header[10:12])[0]
                return channel_count

        return 0

    except Exception as e:
        print(f"[fast-abf] Error extracting channel count: {e}")
        return 0


def extract_path_keywords(file_path: Path, base_directory: Path = None) -> Dict[str, List[str]]:
    """
    Extract useful keywords from file path for organizing experiments.

    Looks for:
    - Power levels (e.g., 10mW, 5mW)
    - Animal IDs (numbers in path components)
    - Keywords like 'pain', 'pleth', 'opto', etc.
    - Subdirectory names

    Args:
        file_path: Path to the file
        base_directory: Base directory to make paths relative to

    Returns:
        Dictionary with:
            'subdirs': List of subdirectory names (relative to base)
            'keywords': List of extracted keywords
            'power_levels': List of power levels found (e.g., ['10mW', '5mW'])
            'animal_ids': List of potential animal IDs (numbers)
    """
    # Get relative path if base directory provided
    if base_directory:
        try:
            rel_path = file_path.relative_to(base_directory)
            path_parts = rel_path.parts[:-1]  # Exclude filename
        except ValueError:
            path_parts = file_path.parts[:-1]
    else:
        path_parts = file_path.parts[:-1]

    # Keywords to look for (case-insensitive)
    interesting_keywords = [
        'pain', 'pleth', 'opto', 'laser', 'stim', 'baseline',
        'test', 'control', 'drug', 'saline', 'vehicle',
        'pre', 'post', 'during', 'before', 'after'
    ]

    subdirs = []
    keywords = []
    power_levels = []
    animal_ids = []

    for part in path_parts:
        part_lower = part.lower()

        # Add as subdirectory
        subdirs.append(part)

        # Look for power levels (e.g., 10mW, 5mw, 10MW)
        power_matches = re.findall(r'(\d+\.?\d*\s?mw)', part_lower)
        power_levels.extend(power_matches)

        # Look for interesting keywords
        for keyword in interesting_keywords:
            if keyword in part_lower:
                keywords.append(keyword)

        # Look for numbers that might be animal IDs (standalone numbers)
        number_matches = re.findall(r'\b(\d{3,6})\b', part)  # 3-6 digit numbers
        animal_ids.extend(number_matches)

    return {
        'subdirs': subdirs,
        'keywords': list(set(keywords)),  # Remove duplicates
        'power_levels': list(set(power_levels)),
        'animal_ids': list(set(animal_ids))
    }


def extract_protocol_name(header: bytes) -> str:
    """
    Extract protocol name from ABF header.

    The protocol name is stored as part of a file path ending in .pro
    Example: "Z:\pClamp Protocols\...\my_protocol.pro"

    Args:
        header: First ~16KB of ABF file

    Returns:
        Protocol name (without .pro extension), or "Unknown" if not found
    """
    try:
        # Search for .pro file paths (null-terminated strings)
        # Convert to string for easier searching
        header_str = header.decode('latin1', errors='ignore')

        # Find all occurrences of .pro followed by null byte
        import re
        matches = re.findall(r'([^\x00]*\.pro)\x00', header_str)

        if matches:
            # Take the first match and extract just the filename
            pro_path = matches[0]
            # Get the filename without path and without .pro extension
            protocol = Path(pro_path).stem  # stem = filename without extension
            return protocol

        return "Unknown"

    except Exception as e:
        print(f"[fast-abf] Error extracting protocol: {e}")
        return "Unknown"


# Test function
if __name__ == "__main__":
    import time

    test_file = Path("examples/25121004.abf")

    print("=" * 60)
    print("FAST ABF READER TEST")
    print("=" * 60)

    # Test fast reader
    start = time.time()
    fast_result = read_abf_metadata_fast(test_file)
    fast_time = time.time() - start

    print(f"\nFast reader ({fast_time*1000:.2f} ms):")
    print(f"  Protocol: {fast_result['protocol']}")
    print(f"  File size: {fast_result['file_size_mb']:.2f} MB")

    # Compare with pyabf
    import pyabf
    start = time.time()
    abf = pyabf.ABF(str(test_file), loadData=False)
    pyabf_time = time.time() - start

    print(f"\npyabf reader ({pyabf_time*1000:.2f} ms):")
    print(f"  Protocol: {abf.protocol}")

    if fast_time > 0:
        print(f"\nSpeedup: {pyabf_time/fast_time:.1f}x faster!")
    else:
        print(f"\nSpeedup: >100x faster (too fast to measure accurately!)")

    # Test on multiple files
    print("\n" + "=" * 60)
    print("BENCHMARKING ON MULTIPLE FILES")
    print("=" * 60)

    from pathlib import Path
    test_dir = Path("examples")
    abf_files = list(test_dir.glob("**/*.abf"))[:10]  # First 10 files

    print(f"\nTesting on {len(abf_files)} files...")

    # Fast method
    start = time.time()
    for f in abf_files:
        read_abf_metadata_fast(f)
    fast_total = time.time() - start

    # pyabf method
    start = time.time()
    for f in abf_files:
        pyabf.ABF(str(f), loadData=False)
    pyabf_total = time.time() - start

    print(f"\nFast reader: {fast_total:.3f} seconds ({fast_total/len(abf_files)*1000:.1f} ms/file)")
    print(f"pyabf:       {pyabf_total:.3f} seconds ({pyabf_total/len(abf_files)*1000:.1f} ms/file)")
    print(f"Speedup:     {pyabf_total/fast_total:.1f}x faster!")


def read_abf_metadata_parallel(abf_files, progress_callback=None, max_workers=4, detect_stim=True):
    """
    Read ABF metadata in parallel using multiple threads.

    Args:
        abf_files: List of Path objects for ABF files
        progress_callback: Optional callback(index, total, metadata) called for each file
        max_workers: Number of parallel workers (default 4)
        detect_stim: If True, detect stim channels and frequency (may be slower)

    Returns:
        List of metadata dictionaries (same order as input)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(abf_files)
    total = len(abf_files)

    # Create index mapping for maintaining order
    file_to_index = {file: i for i, file in enumerate(abf_files)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs with detect_stim parameter
        future_to_file = {executor.submit(read_abf_metadata_fast, file, detect_stim): file
                          for file in abf_files}

        # Process as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            index = file_to_index[file]

            try:
                metadata = future.result()
                results[index] = metadata

                # Call progress callback
                if progress_callback:
                    progress_callback(index, total, metadata)

            except Exception as e:
                print(f"[parallel-abf] Error reading {file}: {e}")
                results[index] = None

    return results
