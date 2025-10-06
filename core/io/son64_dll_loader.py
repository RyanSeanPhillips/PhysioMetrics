"""
Python ctypes wrapper for CED SON64 DLL.

This module provides a Python interface to the official CED CEDS64ML library
for reading .smrx (SON64 format) files.

The wrapper uses the ceds64int.dll which provides a C-compatible interface
to the underlying son64.dll library.
"""

import ctypes
from ctypes import c_int, c_char_p, c_double, c_longlong, POINTER, Structure, c_ubyte
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import platform


# Constants from the CED library
MAX_MATINT_FILES = 100
MAX_CHAN_NAME = 50  # Typical max channel name length

# Channel types (from SON64 documentation)
CHN_ADC = 1      # Waveform ADC channel
CHN_EVENT = 2    # Event channel
CHN_MARKER = 6   # Marker channel
CHN_ADCMARK = 7  # Waveform with markers


class S64Marker(Structure):
    """S64Marker structure matching the C definition."""
    _fields_ = [
        ("m_Time", c_longlong),     # 64-bit time stamp
        ("m_Code1", c_ubyte),       # First marker code
        ("m_Code2", c_ubyte),
        ("m_Code3", c_ubyte),
        ("m_Code4", c_ubyte),
    ]


class SON64DLL:
    """Wrapper for CED SON64 DLL functions."""

    def __init__(self, dll_path: Optional[str] = None):
        """Initialize the DLL wrapper.

        Args:
            dll_path: Path to ceds64int.dll. If None, uses default location.
        """
        if dll_path is None:
            # Default to the x64 version
            dll_path = r"C:\CEDMATLAB\CEDS64ML\x64\ceds64int.dll"

        if not Path(dll_path).exists():
            raise FileNotFoundError(f"CED DLL not found at: {dll_path}")

        # Load the DLL
        self.dll = ctypes.CDLL(dll_path)

        # Define function signatures
        self._define_functions()

    def _define_functions(self):
        """Define C function signatures for ctypes."""

        # File operations
        self.dll.S64Open.argtypes = [c_char_p, c_int]
        self.dll.S64Open.restype = c_int

        self.dll.S64Close.argtypes = [c_int]
        self.dll.S64Close.restype = c_int

        self.dll.S64CloseAll.argtypes = []
        self.dll.S64CloseAll.restype = c_int

        # Channel information
        self.dll.S64MaxChans.argtypes = [c_int]
        self.dll.S64MaxChans.restype = c_int

        self.dll.S64ChanType.argtypes = [c_int, c_int]  # Was S64ChanKind
        self.dll.S64ChanType.restype = c_int

        self.dll.S64GetChanTitle.argtypes = [c_int, c_int, c_char_p, c_int]
        self.dll.S64GetChanTitle.restype = c_int

        self.dll.S64GetChanUnits.argtypes = [c_int, c_int, c_char_p, c_int]  # Was S64GetChanYUnits
        self.dll.S64GetChanUnits.restype = c_int

        self.dll.S64GetIdealRate.argtypes = [c_int, c_int]  # Was S64GetChanRate
        self.dll.S64GetIdealRate.restype = c_double

        self.dll.S64GetChanScale.argtypes = [c_int, c_int, POINTER(c_double)]  # Returns via pointer
        self.dll.S64GetChanScale.restype = c_int

        self.dll.S64GetChanOffset.argtypes = [c_int, c_int, POINTER(c_double)]  # Returns via pointer
        self.dll.S64GetChanOffset.restype = c_int

        # Time information
        self.dll.S64GetTimeBase.argtypes = [c_int]
        self.dll.S64GetTimeBase.restype = c_double

        self.dll.S64ChanMaxTime.argtypes = [c_int, c_int]  # Was S64MaxTime
        self.dll.S64ChanMaxTime.restype = c_longlong

        self.dll.S64ChanDivide.argtypes = [c_int, c_int]  # Channel sample interval in ticks
        self.dll.S64ChanDivide.restype = c_longlong

        # Data reading - use float version (S64ReadWaveF reads as float, not double)
        self.dll.S64ReadWaveF.argtypes = [c_int, c_int, POINTER(ctypes.c_float),
                                          c_int, c_longlong, c_longlong, POINTER(c_longlong), c_int]
        self.dll.S64ReadWaveF.restype = c_int


class SON64Loader:
    """High-level interface for loading SON64 (.smrx) files."""

    def __init__(self, dll_path: Optional[str] = None):
        """Initialize the loader.

        Args:
            dll_path: Path to ceds64int.dll. If None, uses default location.
        """
        self.dll = SON64DLL(dll_path)
        self.file_handle = None
        self.filename = None

    def open(self, filename: str) -> int:
        """Open a SON64 file.

        Args:
            filename: Path to .smrx file

        Returns:
            File handle (1-100) or negative error code
        """
        # Convert to bytes for C
        filename_bytes = filename.encode('utf-8')

        # Open file (0 = read-only)
        handle = self.dll.dll.S64Open(filename_bytes, 0)

        if handle < 0:
            raise IOError(f"Failed to open file: {filename} (error code: {handle})")

        self.file_handle = handle
        self.filename = filename
        return handle

    def close(self):
        """Close the current file."""
        if self.file_handle is not None:
            self.dll.dll.S64Close(self.file_handle)
            self.file_handle = None
            self.filename = None

    def get_channel_count(self) -> int:
        """Get the maximum channel number (0-indexed).

        Returns:
            Maximum channel number + 1
        """
        if self.file_handle is None:
            raise RuntimeError("No file is open")

        return self.dll.dll.S64MaxChans(self.file_handle)

    def get_channel_info(self, channel: int) -> Dict:
        """Get information about a channel.

        Args:
            channel: Channel number (0-indexed)

        Returns:
            Dictionary with channel metadata
        """
        if self.file_handle is None:
            raise RuntimeError("No file is open")

        # Get channel kind/type
        kind = self.dll.dll.S64ChanType(self.file_handle, channel)

        if kind <= 0:
            # Channel doesn't exist or is empty
            return None

        # Get channel title
        title_buffer = ctypes.create_string_buffer(MAX_CHAN_NAME)
        self.dll.dll.S64GetChanTitle(self.file_handle, channel, title_buffer, MAX_CHAN_NAME)
        title = title_buffer.value.decode('utf-8', errors='ignore')

        # Get units
        units_buffer = ctypes.create_string_buffer(MAX_CHAN_NAME)
        self.dll.dll.S64GetChanUnits(self.file_handle, channel, units_buffer, MAX_CHAN_NAME)
        units = units_buffer.value.decode('utf-8', errors='ignore')

        # Get sample rate (Hz)
        rate = self.dll.dll.S64GetIdealRate(self.file_handle, channel)

        # Get scale and offset (returned via pointers)
        scale_val = c_double()
        self.dll.dll.S64GetChanScale(self.file_handle, channel, ctypes.byref(scale_val))
        scale = scale_val.value

        offset_val = c_double()
        self.dll.dll.S64GetChanOffset(self.file_handle, channel, ctypes.byref(offset_val))
        offset = offset_val.value

        # Get time base (seconds per tick)
        time_base = self.dll.dll.S64GetTimeBase(self.file_handle)

        # Get channel divide (sample interval in ticks) - THIS IS CRITICAL!
        chan_divide = self.dll.dll.S64ChanDivide(self.file_handle, channel)

        # Get max time (in ticks)
        max_time_ticks = self.dll.dll.S64ChanMaxTime(self.file_handle, channel)
        max_time_sec = max_time_ticks * time_base

        return {
            'channel': channel,
            'kind': kind,
            'title': title,
            'units': units,
            'sample_rate_hz': rate,
            'scale': scale,
            'offset': offset,
            'time_base': time_base,
            'chan_divide': chan_divide,  # Sample interval in ticks
            'max_time_ticks': max_time_ticks,
            'max_time_sec': max_time_sec,
            'duration_sec': max_time_sec,
        }

    def read_waveform(self, channel: int, start_time: float = 0,
                      end_time: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Read waveform data from a channel.

        Args:
            channel: Channel number (0-indexed)
            start_time: Start time in seconds (default: 0)
            end_time: End time in seconds (default: entire trace)

        Returns:
            Tuple of (time_array, data_array)
        """
        if self.file_handle is None:
            raise RuntimeError("No file is open")

        # Get channel info
        info = self.get_channel_info(channel)
        if info is None:
            raise ValueError(f"Channel {channel} does not exist or is empty")

        if info['kind'] not in [CHN_ADC, CHN_ADCMARK]:
            raise ValueError(f"Channel {channel} is not a waveform channel (kind={info['kind']})")

        # Convert times to ticks
        time_base = info['time_base']
        start_tick = int(start_time / time_base)

        if end_time is None:
            end_tick = info['max_time_ticks']
        else:
            end_tick = int(end_time / time_base)

        # Calculate maximum number of samples that could be in this time range
        # Use ChanDivide to calculate how many samples fit in the tick range
        chan_divide = info['chan_divide']  # Sample interval in ticks
        max_samples = int((end_tick - start_tick) / chan_divide) + 1000  # Add buffer for safety

        if max_samples <= 0:
            return np.array([]), np.array([])

        # SMRX files may have gaps - read all segments and concatenate
        # The key is to advance correctly after each segment using ChanDivide
        all_data = []
        all_times = []
        current_tick = start_tick

        while current_tick < end_tick:
            # Allocate buffer for this segment
            segment_buffer = (ctypes.c_float * max_samples)()
            first_time = c_longlong(0)

            # Read data segment (nMask=0 means no filtering)
            n_read = self.dll.dll.S64ReadWaveF(
                self.file_handle,
                channel,
                segment_buffer,
                max_samples,
                current_tick,
                end_tick,
                ctypes.byref(first_time),
                0  # nMask - no filtering
            )

            if n_read <= 0:
                # No more data or error
                break

            # Convert segment to numpy array
            segment_data = np.array(segment_buffer[:n_read], dtype=np.float64)
            all_data.append(segment_data)

            # Calculate time for this segment using ChanDivide
            # Each sample is chan_divide ticks apart
            # Sample i has tick time: first_time + (i * chan_divide)
            # Sample i has real time: (first_time + i * chan_divide) * time_base
            sample_ticks = first_time.value + np.arange(n_read) * chan_divide
            segment_times = sample_ticks * time_base
            all_times.append(segment_times)

            # Move to next segment: advance past the last sample
            # Last sample is at tick: first_time + (n_read - 1) * chan_divide
            last_sample_tick = first_time.value + (n_read - 1) * chan_divide

            # Next segment starts at the next possible sample time
            current_tick = last_sample_tick + chan_divide

        # Concatenate all segments
        if not all_data:
            return np.array([]), np.array([])

        data = np.concatenate(all_data)
        time = np.concatenate(all_times)

        return time, data

    def get_all_channels(self) -> List[Dict]:
        """Get information about all channels in the file.

        Returns:
            List of channel info dictionaries
        """
        max_chan = self.get_channel_count()
        channels = []

        for ch in range(max_chan):
            info = self.get_channel_info(ch)
            if info is not None:
                channels.append(info)

        return channels

    def load_file(self, filename: str) -> Dict:
        """Load a complete .smrx file.

        Args:
            filename: Path to .smrx file

        Returns:
            Dictionary with all channel data and metadata
        """
        self.open(filename)

        try:
            # Get all channels
            channels = self.get_all_channels()

            # Read waveform data
            data = {}
            for ch_info in channels:
                ch_num = ch_info['channel']

                if ch_info['kind'] in [CHN_ADC, CHN_ADCMARK]:
                    # Read waveform
                    time, waveform = self.read_waveform(ch_num)

                    data[ch_info['title']] = {
                        'time': time,
                        'data': waveform,
                        'sample_rate_hz': ch_info['sample_rate_hz'],
                        'units': ch_info['units'],
                        'channel': ch_num,
                        'kind': ch_info['kind'],
                    }

            return {
                'filename': filename,
                'channels': data,
                'channel_info': channels,
            }

        finally:
            self.close()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support."""
        self.close()


def load_smrx(filename: str, dll_path: Optional[str] = None) -> Dict:
    """Convenience function to load a .smrx file.

    Args:
        filename: Path to .smrx file
        dll_path: Path to ceds64int.dll (optional)

    Returns:
        Dictionary with all channel data and metadata
    """
    loader = SON64Loader(dll_path)
    return loader.load_file(filename)


# Example usage
if __name__ == "__main__":
    # Test with the example file
    test_file = r"C:\path\to\100225_000.smrx"

    if Path(test_file).exists():
        print(f"Loading {test_file}...")
        data = load_smrx(test_file)

        print(f"\nFile: {data['filename']}")
        print(f"\nChannels found: {len(data['channels'])}")

        for name, ch_data in data['channels'].items():
            print(f"\n  {name}:")
            print(f"    Channel: {ch_data['channel']}")
            print(f"    Sample rate: {ch_data['sample_rate_hz']} Hz")
            print(f"    Units: {ch_data['units']}")
            print(f"    Samples: {len(ch_data['data'])}")
            print(f"    Duration: {ch_data['time'][-1]:.2f} seconds")
            print(f"    Data range: [{ch_data['data'].min():.3f}, {ch_data['data'].max():.3f}]")
    else:
        print(f"Test file not found: {test_file}")
        print("\nTo test, update the test_file path and run this script.")
