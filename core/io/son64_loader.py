"""
SON64 (.smrx) file loader for PlethApp.

This module uses the CED DLL wrapper to load Spike2 .smrx files
and convert them to the format expected by PlethApp.
"""

from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from scipy import interpolate
from .son64_dll_loader import SON64Loader
from .s2rx_parser import get_hidden_channels


def load_son64(path: str, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load a SON64 (.smrx) file using the CED DLL.

    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    Note: .smrx files have continuous recording, so n_sweeps = 1

    progress_callback: optional function(current, total, message) for progress updates
    """
    if progress_callback:
        progress_callback(0, 100, "Opening .smrx file...")

    # Check for .s2rx visibility settings (get HIDDEN channels)
    hidden_channels = get_hidden_channels(path)

    # Load the file using the DLL wrapper
    loader = SON64Loader()

    try:
        loader.open(path)

        if progress_callback:
            progress_callback(10, 100, "Reading channel information...")

        # Get all waveform channels
        all_channels = loader.get_all_channels()

        # Filter to waveform channels only (kind 1 = ADC, 7 = ADCMARK)
        waveform_channels = [ch for ch in all_channels if ch['kind'] in [1, 7]]

        # If .s2rx file exists, filter out hidden channels
        if hidden_channels is not None:
            waveform_channels = [ch for ch in waveform_channels
                                if ch['channel'] not in hidden_channels]
            if progress_callback and hidden_channels:
                if len(hidden_channels) > 0:
                    progress_callback(15, 100,
                        f"Filtered {len(hidden_channels)} hidden channel(s) from .s2rx settings...")

        if not waveform_channels:
            raise ValueError("No waveform channels found in .smrx file")

        if progress_callback:
            progress_callback(20, 100, f"Found {len(waveform_channels)} waveform channels...")

        # Find the channel with the LOWEST sample rate to use as the reference
        # This avoids upsampling which causes interpolation artifacts with gapped data
        min_rate_channel = min(waveform_channels, key=lambda ch: ch['sample_rate_hz'])
        sr_hz = float(min_rate_channel['sample_rate_hz'])

        # Find maximum duration to determine target time range
        max_duration = max(ch['duration_sec'] for ch in waveform_channels)

        # Create common time grid at the target sample rate
        # Use the actual max duration from the data
        target_n_samples = int(np.ceil(max_duration * sr_hz))
        t_common = np.arange(target_n_samples) / sr_hz

        # Read all waveform data and interpolate to common time grid
        sweeps_by_channel: Dict[str, np.ndarray] = {}
        channel_names: List[str] = []

        for i, ch_info in enumerate(waveform_channels):
            ch_num = ch_info['channel']
            ch_name = ch_info['title']
            ch_rate = ch_info['sample_rate_hz']

            if not ch_name:
                ch_name = f"Channel_{ch_num}"

            channel_names.append(ch_name)

            if progress_callback:
                pct = 20 + int(70 * (i + 1) / len(waveform_channels))
                progress_callback(pct, 100, f"Reading {ch_name}...")

            # Read the waveform with its actual timestamps
            time_ch, data_ch = loader.read_waveform(ch_num)

            # Check if we need to resample
            if ch_rate != sr_hz:
                # Different sample rate - need to resample
                # Use linear interpolation to map data onto common time grid
                f_interp = interpolate.interp1d(
                    time_ch, data_ch,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(data_ch[0], data_ch[-1])  # Extend with edge values
                )
                data = f_interp(t_common)
            else:
                # Same sample rate - just pad to match target length
                # This avoids interpolation artifacts
                if len(data_ch) < target_n_samples:
                    # Pad with last value
                    padding = np.full(target_n_samples - len(data_ch), data_ch[-1])
                    data = np.concatenate([data_ch, padding])
                elif len(data_ch) > target_n_samples:
                    # Trim
                    data = data_ch[:target_n_samples]
                else:
                    data = data_ch

            # PlethApp expects shape (n_samples, n_sweeps)
            # .smrx files are continuous, so n_sweeps = 1
            sweeps_by_channel[ch_name] = data.reshape(-1, 1)

        # Use the common time grid
        t = t_common

        if progress_callback:
            progress_callback(100, 100, "Complete")

        return sr_hz, sweeps_by_channel, channel_names, t

    finally:
        loader.close()
