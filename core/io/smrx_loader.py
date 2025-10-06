"""
CED Spike2 .smr/.smrx file loader
Reverse-engineered from MATLAB SON library

Supports:
- .smr files (SON 32-bit format)
- .smrx files (SON64 format)
- ADC waveform channels (kind=1)
- RealWave channels (kind=9)

Based on CED's SON file format specification
"""
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_smrx(file_path: str, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load a Spike2 .smr or .smrx file

    Args:
        file_path: Path to .smr or .smrx file
        progress_callback: Optional callback(current, total, message)

    Returns:
        (sr_hz, sweeps_by_channel, channel_names, t)
        - sr_hz: Sample rate in Hz
        - sweeps_by_channel: dict {channel_name: 2D array (n_samples, n_sweeps)}
        - channel_names: List of channel names
        - t: Time vector in seconds

    Compatible with PlethApp's ABF loader interface
    """
    file_path = Path(file_path)

    if progress_callback:
        progress_callback(0, 100, "Opening Spike2 file...")

    with open(file_path, 'rb') as fid:
        # Read file header
        header = _read_file_header(fid)

        if progress_callback:
            progress_callback(10, 100, f"Found {header['channels']} channels...")

        # Read channel info
        channels = []
        for i in range(1, header['channels'] + 1):
            chan_info = _read_channel_info(fid, i, header)
            print(f"  Channel {i}: kind={chan_info['kind']}, title={chan_info.get('title', 'N/A')}")
            if chan_info['kind'] > 0:  # Active channel
                channels.append(chan_info)

        if progress_callback:
            progress_callback(20, 100, f"Loading {len(channels)} active channels...")

        # Load channel data
        sweeps_by_channel = {}
        channel_names = []
        sample_rates = []

        for idx, chan in enumerate(channels):
            if progress_callback:
                pct = 20 + int(70 * (idx + 1) / len(channels))
                progress_callback(pct, 100, f"Loading channel {idx+1}/{len(channels)}: {chan['title']}")

            # Try to load any waveform-like channel (including unknown types)
            if chan['kind'] in [1, 9, 229]:  # ADC, RealWave, or unknown type 229
                data = _read_adc_channel(fid, chan, header)
                if data is not None and len(data) > 0:
                    # Store as 2D array (samples, sweeps) - for now single sweep
                    sweeps_by_channel[chan['title']] = data.reshape(-1, 1)
                    channel_names.append(chan['title'])
                    sample_rates.append(chan['sample_rate'])

        if not channel_names:
            raise ValueError("No ADC waveform channels found in file")

        # Use first channel's sample rate
        sr_hz = sample_rates[0]

        # Create time vector
        first_channel = list(sweeps_by_channel.values())[0]
        n_samples = first_channel.shape[0]
        t = np.arange(n_samples) / sr_hz

        if progress_callback:
            progress_callback(100, 100, "Complete")

        return sr_hz, sweeps_by_channel, channel_names, t


def _read_file_header(fid) -> Dict:
    """Read SON file header (512 bytes)"""
    fid.seek(0)

    header = {}
    header['systemID'] = struct.unpack('<h', fid.read(2))[0]

    # Skip copyright, creator fields
    fid.read(18)

    header['usPerTime'] = struct.unpack('<h', fid.read(2))[0]
    header['timePerADC'] = struct.unpack('<h', fid.read(2))[0]
    header['filestate'] = struct.unpack('<h', fid.read(2))[0]
    header['firstdata'] = struct.unpack('<i', fid.read(4))[0]
    header['channels'] = struct.unpack('<h', fid.read(2))[0]
    header['chansize'] = struct.unpack('<h', fid.read(2))[0]
    header['extraData'] = struct.unpack('<h', fid.read(2))[0]
    header['buffersize'] = struct.unpack('<h', fid.read(2))[0]
    header['osFormat'] = struct.unpack('<h', fid.read(2))[0]
    header['maxFTime'] = struct.unpack('<i', fid.read(4))[0]
    header['dTimeBase'] = struct.unpack('<d', fid.read(8))[0]

    # For old files, set default
    if header['systemID'] < 6:
        header['dTimeBase'] = 1e-6

    return header


def _read_channel_info(fid, chan: int, file_header: Dict) -> Dict:
    """
    Read channel header (140 bytes each)

    Offset: 512 + 140*(chan-1)
    """
    base = 512 + 140 * (chan - 1)
    fid.seek(base)

    info = {'number': chan}

    info['delSize'] = struct.unpack('<h', fid.read(2))[0]
    info['nextDelBlock'] = struct.unpack('<i', fid.read(4))[0]
    info['firstblock'] = struct.unpack('<i', fid.read(4))[0]
    info['lastblock'] = struct.unpack('<i', fid.read(4))[0]
    info['blocks'] = struct.unpack('<h', fid.read(2))[0]
    info['nExtra'] = struct.unpack('<h', fid.read(2))[0]
    info['preTrig'] = struct.unpack('<h', fid.read(2))[0]
    info['free0'] = struct.unpack('<h', fid.read(2))[0]
    info['phySz'] = struct.unpack('<h', fid.read(2))[0]
    info['maxData'] = struct.unpack('<h', fid.read(2))[0]

    # Read comment (variable length string)
    comment_len = struct.unpack('<B', fid.read(1))[0]
    pointer = fid.tell()
    info['comment'] = fid.read(comment_len).decode('ascii', errors='ignore')
    fid.seek(pointer + 71)

    info['maxChanTime'] = struct.unpack('<i', fid.read(4))[0]
    info['lChanDvd'] = struct.unpack('<i', fid.read(4))[0]
    info['phyChan'] = struct.unpack('<h', fid.read(2))[0]

    # Read title (variable length string)
    title_len = struct.unpack('<B', fid.read(1))[0]
    pointer = fid.tell()
    info['title'] = fid.read(title_len).decode('ascii', errors='ignore')
    if not info['title']:
        info['title'] = f"Channel_{chan}"
    fid.seek(pointer + 9)

    info['idealRate'] = struct.unpack('<f', fid.read(4))[0]
    info['kind'] = struct.unpack('<B', fid.read(1))[0]
    info['pad'] = struct.unpack('<b', fid.read(1))[0]

    # Calculate sample rate
    if info['kind'] in [1, 6, 7, 9]:  # Channels with sample rates
        info['sample_rate'] = info['idealRate']
    else:
        info['sample_rate'] = None

    # Read channel-type specific fields
    if info['kind'] in [1, 6]:  # ADC or ADC Marker
        info['scale'] = struct.unpack('<f', fid.read(4))[0]
        info['offset'] = struct.unpack('<f', fid.read(4))[0]
        units_len = struct.unpack('<B', fid.read(1))[0]
        pointer = fid.tell()
        info['units'] = fid.read(units_len).decode('ascii', errors='ignore')
        fid.seek(pointer + 5)

        if file_header['systemID'] < 6:
            info['divide'] = struct.unpack('<h', fid.read(2))[0]
        else:
            info['interleave'] = struct.unpack('<h', fid.read(2))[0]
    elif info['kind'] in [7, 9]:  # RealMarker or RealWave
        info['min'] = struct.unpack('<f', fid.read(4))[0]
        info['max'] = struct.unpack('<f', fid.read(4))[0]
        units_len = struct.unpack('<B', fid.read(1))[0]
        pointer = fid.tell()
        info['units'] = fid.read(units_len).decode('ascii', errors='ignore')
        fid.seek(pointer + 5)

        if file_header['systemID'] < 6:
            info['divide'] = struct.unpack('<h', fid.read(2))[0]
        else:
            info['interleave'] = struct.unpack('<h', fid.read(2))[0]

    return info


def _read_adc_channel(fid, chan_info: Dict, file_header: Dict) -> Optional[np.ndarray]:
    """
    Read ADC waveform data for a channel

    This reads continuous waveform data (kind=1, 9, or unknown types)
    """
    # Allow unknown channel types - try to read them anyway
    if chan_info['kind'] == 0:
        return None  # Empty channel

    # Navigate to first data block
    if chan_info['firstblock'] <= 0:
        return None  # No data

    # For continuous waveforms, read all blocks
    current_block = chan_info['firstblock']
    all_data = []

    while current_block > 0:
        fid.seek(current_block)

        # Read block header
        predBlock = struct.unpack('<i', fid.read(4))[0]
        succBlock = struct.unpack('<i', fid.read(4))[0]
        startTime = struct.unpack('<i', fid.read(4))[0]
        endTime = struct.unpack('<i', fid.read(4))[0]
        chanNumber = struct.unpack('<h', fid.read(2))[0]
        items = struct.unpack('<h', fid.read(2))[0]

        # Read data samples
        if chan_info['kind'] == 1:  # int16 ADC data
            samples = np.frombuffer(fid.read(items * 2), dtype=np.int16)
            # Apply scale and offset
            if 'scale' in chan_info and 'offset' in chan_info:
                samples = samples.astype(np.float32) * chan_info['scale'] / 6553.6 + chan_info['offset']
        elif chan_info['kind'] == 9:  # float32 RealWave data
            samples = np.frombuffer(fid.read(items * 4), dtype=np.float32)

        all_data.append(samples)

        # Move to next block
        current_block = succBlock

        # Safety check to avoid infinite loops
        if len(all_data) > 100000:
            break

    if not all_data:
        return None

    # Concatenate all blocks
    return np.concatenate(all_data)


if __name__ == '__main__':
    # Test the loader
    test_file = r"Z:\DATA\PhillipsRS\RVM manipulations\AH\ryan\R5\100225_001.smrx"
    print(f"Testing SMRX loader on: {test_file}\n")

    try:
        sr_hz, sweeps, chan_names, t = load_smrx(test_file)

        print(f"✓ Successfully loaded file!")
        print(f"  Sample rate: {sr_hz} Hz")
        print(f"  Channels: {chan_names}")
        print(f"  Duration: {t[-1]:.2f} seconds")

        for name in chan_names:
            data = sweeps[name]
            print(f"\n  {name}:")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{data.min():.3f}, {data.max():.3f}]")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
