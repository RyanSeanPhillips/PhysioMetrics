"""
Son64 (.smrx) file loader - Pure binary implementation
Direct binary reading without external dependencies

Author: Reverse-engineered from binary analysis
Compatible with PlethApp's data loading interface
"""
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_son64(file_path: str, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load a Son64 (.smrx) file using pure binary reading

    Args:
        file_path: Path to .smrx file
        progress_callback: Optional callback(current, total, message)

    Returns:
        (sr_hz, sweeps_by_channel, channel_names, t)
        Compatible with PlethApp's ABF loader interface
    """
    file_path = Path(file_path)

    if progress_callback:
        progress_callback(0, 100, "Opening Son64 file...")

    with open(file_path, 'rb') as fid:
        # Verify Son64 signature
        fid.seek(0)
        signature = fid.read(8)
        if not signature.startswith(b'S64'):
            raise ValueError(f"Not a Son64 file (signature: {signature})")

        if progress_callback:
            progress_callback(10, 100, "Scanning for channels...")

        # Find all channel descriptors by scanning for valid sample rates
        channels = _find_all_channel_descriptors(fid)

        if not channels:
            raise ValueError("No active channels found in file")

        if progress_callback:
            progress_callback(30, 100, f"Found {len(channels)} channels, locating data...")

        # Find actual data locations for each channel
        _locate_channel_data(fid, channels, progress_callback)

        # Load channel data
        sweeps_by_channel = {}
        channel_names = []
        sample_rates = []

        for idx, chan in enumerate(channels):
            if progress_callback:
                pct = 50 + int(40 * (idx + 1) / len(channels))
                progress_callback(pct, 100, f"Reading channel {idx+1}/{len(channels)}: {chan['name']}")

            data = _read_channel_data(fid, chan)

            if data is not None and len(data) > 0:
                chan_name = chan.get('name', f"Channel_{len(channel_names)+1}")
                sweeps_by_channel[chan_name] = data.reshape(-1, 1)
                channel_names.append(chan_name)
                sample_rates.append(chan['sample_rate'])

        if not channel_names:
            raise ValueError("No channel data could be read")

        # Use highest sample rate (most representative for time base)
        sr_hz = max(sample_rates)

        # Pad channels to match longest duration
        max_samples = max(arr.shape[0] for arr in sweeps_by_channel.values())
        for chan_name in list(sweeps_by_channel.keys()):
            current_samples = sweeps_by_channel[chan_name].shape[0]
            if current_samples < max_samples:
                # Pad with NaN
                padded = np.full((max_samples, 1), np.nan, dtype=np.float32)
                padded[:current_samples, 0] = sweeps_by_channel[chan_name][:, 0]
                sweeps_by_channel[chan_name] = padded

        # Create time vector
        t = np.arange(max_samples) / sr_hz

        if progress_callback:
            progress_callback(100, 100, "Complete")

        return sr_hz, sweeps_by_channel, channel_names, t


def _find_all_channel_descriptors(fid) -> List[Dict]:
    """
    Find all channel descriptors by scanning file for valid sample rates

    Returns list of channel info dicts sorted by descriptor offset
    """
    # Read first 20KB to find channel descriptors
    fid.seek(0)
    search_region = fid.read(20480)

    # Common sample rates to search for (expanded range)
    target_sample_rates = [100000.0, 50000.0, 20000.0, 15000.0, 12500.0, 10000.0,
                           5000.0, 2000.0, 1000.0, 961.5384615384617, 500.0, 250.0, 100.0]

    channels = []
    found_offsets = set()

    for target_sr in target_sample_rates:
        sr_bytes = struct.pack('<d', target_sr)
        offset = 0

        while True:
            idx = search_region.find(sr_bytes, offset)
            if idx == -1:
                break

            # Sample rate is at descriptor +72
            desc_offset = idx - 72

            # Avoid duplicates
            if desc_offset >= 0 and desc_offset not in found_offsets:
                # Try to read descriptor
                try:
                    fid.seek(desc_offset)
                    ch = _read_son64_channel_descriptor(fid)

                    # Validate
                    if (ch and ch['kind'] > 0 and
                        ch['data_pointer'] > 0 and
                        ch['data_pointer'] < 10000000000):  # Reasonable file offset

                        ch['descriptor_offset'] = desc_offset
                        channels.append(ch)
                        found_offsets.add(desc_offset)

                except Exception:
                    pass

            offset = idx + 1

    # Sort by descriptor offset to maintain channel order
    channels.sort(key=lambda c: c['descriptor_offset'])

    return channels


def _read_son64_channel_descriptor(fid) -> Optional[Dict]:
    """
    Read Son64 channel descriptor

    Structure (reverse-engineered):
    +0:   uint32  - size/flags
    +8:   uint64  - data block pointer (file offset)
    +16:  uint64  - number of samples (or related)
    +32:  uint32  - channel kind/type
    +72:  float64 - sample rate (Hz)
    +80:  float64 - scale factor 1
    +88:  float64 - zero value
    +96:  float64 - offset/min value
    +104: float64 - scale/max value
    """
    start_pos = fid.tell()

    try:
        # Read descriptor fields
        size_flags = struct.unpack('<I', fid.read(4))[0]
        fid.read(4)  # padding

        data_pointer = struct.unpack('<Q', fid.read(8))[0]
        n_samples_val = struct.unpack('<Q', fid.read(8))[0]

        fid.read(8)  # skip to kind

        kind = struct.unpack('<I', fid.read(4))[0]

        # Skip to sample rate
        fid.seek(start_pos + 72)
        sample_rate = struct.unpack('<d', fid.read(8))[0]

        # Read scale/offset values
        scale1 = struct.unpack('<d', fid.read(8))[0]
        zero_val = struct.unpack('<d', fid.read(8))[0]
        offset_val = struct.unpack('<d', fid.read(8))[0]
        scale_val = struct.unpack('<d', fid.read(8))[0]

        # Try to find channel name (look ahead a bit)
        fid.seek(start_pos + 112)
        name_region = fid.read(100)

        # Look for length-prefixed ASCII string
        name = None
        for i in range(len(name_region) - 10):
            str_len = name_region[i]
            if 3 <= str_len <= 32:
                try:
                    possible_name = name_region[i+1:i+1+str_len].decode('ascii', errors='ignore')
                    if all(32 <= ord(c) <= 126 or c == ' ' for c in possible_name):
                        name = possible_name.strip()
                        break
                except:
                    pass

        return {
            'kind': kind,
            'data_pointer': data_pointer,
            'n_samples': n_samples_val,
            'sample_rate': sample_rate,
            'scale': scale_val,
            'offset': offset_val,
            'name': name or f"Channel",
            'data_offset': None  # Will be filled by _locate_channel_data
        }

    except Exception:
        return None


def _locate_channel_data(fid, channels: List[Dict], progress_callback=None):
    """
    Locate actual data for each channel by searching for block-aligned offsets

    Data is stored in 65,536-byte (0x10000) blocks with 32-byte headers
    Data starts at offsets ending in 0x020 (e.g., 0x00010020, 0x00030020)
    """
    # Search for data at common block offsets
    fid.seek(0, 2)
    file_size = fid.tell()

    BLOCK_SIZE = 0x10000  # 65,536 bytes
    HEADER_SIZE = 0x20     # 32 bytes

    # Start searching from first block
    max_blocks_to_search = min(1000, file_size // BLOCK_SIZE)

    for chan in channels:
        # Try blocks in sequence
        for block_num in range(1, max_blocks_to_search):
            offset = block_num * BLOCK_SIZE + HEADER_SIZE

            if offset >= file_size:
                break

            fid.seek(offset)
            # Read first few samples to check if this looks like valid data
            try:
                test_data = np.frombuffer(fid.read(100), dtype=np.int16)
                # Check if data varies (not all zeros or constant)
                if len(set(test_data[:20].tolist())) > 2:
                    # This might be the channel's data
                    chan['data_offset'] = offset
                    break
            except:
                continue


def _read_channel_data(fid, chan_info: Dict) -> Optional[np.ndarray]:
    """
    Read Son64 channel data from located offset

    Data is stored as int16 values that need to be scaled
    """
    try:
        data_offset = chan_info.get('data_offset')

        if data_offset is None:
            return None

        # Calculate how much data to read
        fid.seek(0, 2)
        file_size = fid.tell()

        # Read up to next block or end of file
        BLOCK_SIZE = 0x10000
        current_block = (data_offset // BLOCK_SIZE) * BLOCK_SIZE
        next_block_candidate = current_block + BLOCK_SIZE

        # Read multiple blocks if needed (estimate based on sample rate)
        # For 12,500 Hz and 200s, we need ~2.5M samples = ~5MB = ~77 blocks
        estimated_blocks = 100  # Conservative estimate
        bytes_to_read = min(estimated_blocks * BLOCK_SIZE, file_size - data_offset)

        fid.seek(data_offset)
        raw_bytes = fid.read(bytes_to_read)

        # Convert to int16
        data = np.frombuffer(raw_bytes, dtype=np.int16)

        # Apply scale and offset
        scale = chan_info.get('scale', 1.0)
        offset = chan_info.get('offset', 0.0)

        # IMPORTANT: The scale in the descriptor might not be the final scale
        # We need to use a fixed scale factor based on reverse engineering
        # For CED files, typical scale is around 0.00015259 (1/6553.6)
        actual_scale = scale / 6553.6 if scale == 1.0 else scale

        data_float = data.astype(np.float32) * actual_scale + offset

        return data_float

    except Exception as e:
        print(f"Error reading channel {chan_info.get('name')}: {e}")
        return None


if __name__ == '__main__':
    # Test the loader
    test_file = r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\100225_000.smrx"
    print(f"Testing Son64 binary loader on: {test_file}\n")

    try:
        sr_hz, sweeps, chan_names, t = load_son64(test_file)

        print(f"\n" + "="*80)
        print(f"SUCCESS! Loaded Son64 file")
        print("="*80)
        print(f"  Sample rate: {sr_hz} Hz")
        print(f"  Channels: {chan_names}")
        print(f"  Duration: {t[-1]:.2f} seconds ({t[-1]/60:.1f} minutes)")

        for name in chan_names:
            data = sweeps[name]
            print(f"\n  {name}:")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]")
            print(f"    Mean: {np.nanmean(data):.6f}, Std: {np.nanstd(data):.6f}")
            print(f"    First 10 values: {data[:10, 0]}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
