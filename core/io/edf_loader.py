"""
EDF/EDF+ file loader for PhysioMetrics.

This module loads European Data Format (EDF) files, commonly used for
physiological recordings (EEG, breathing, etc.).

Supported features:
- EDF and EDF+ formats
- Multi-rate channels (automatically resampled to lowest rate)
- Channel metadata (names, units, sample rates)
- Annotation channels (automatically filtered out)
- Progress callback support
"""

from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
from scipy import interpolate


def load_edf(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load an EDF/EDF+ file using pyedflib.

    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    Note: EDF files have continuous recording, so n_sweeps = 1

    progress_callback: optional function(current, total, message) for progress updates
    """
    try:
        import pyedflib
    except ImportError:
        raise RuntimeError(
            "pyedflib not installed. Install with: pip install pyedflib\n"
            "Note: pyedflib requires numpy<2.0. If you have numpy 2.0+, you may need to downgrade."
        )

    if progress_callback:
        progress_callback(0, 100, "Opening EDF file...")

    # Open and read the EDF file
    try:
        f = pyedflib.EdfReader(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open EDF file: {str(e)}")

    try:
        n_channels = f.signals_in_file

        if n_channels == 0:
            raise ValueError("EDF file contains no signals")

        if progress_callback:
            progress_callback(10, 100, "Reading channel information...")

        # Get channel metadata
        signal_labels = f.getSignalLabels()
        sample_rates = [f.getSampleFrequency(i) for i in range(n_channels)]
        signal_headers = [f.getSignalHeader(i) for i in range(n_channels)]

        # Filter out annotations channel (EDF+ specific)
        # EDF Annotations channel is identified by label "EDF Annotations" or empty label
        waveform_indices = []
        for i in range(n_channels):
            label = signal_labels[i].strip()
            # Skip annotation channels
            if label in ['EDF Annotations', '']:
                continue
            # Skip if sample rate is 0 (annotation channel indicator)
            if sample_rates[i] <= 0:
                continue
            waveform_indices.append(i)

        if not waveform_indices:
            raise ValueError("No waveform channels found in EDF file (only annotations or empty channels)")

        if progress_callback:
            progress_callback(20, 100, f"Found {len(waveform_indices)} waveform channels...")

        # Find the channel with the LOWEST sample rate to use as the reference
        # This avoids upsampling which can introduce artifacts
        min_rate = min(sample_rates[i] for i in waveform_indices)
        sr_hz = float(min_rate)

        # Determine maximum duration across all channels
        max_duration = 0
        for i in waveform_indices:
            n_samples = f.getNSamples()[i]
            duration = n_samples / sample_rates[i]
            max_duration = max(max_duration, duration)

        # Create common time grid at the target sample rate
        target_n_samples = int(np.ceil(max_duration * sr_hz))
        t_common = np.arange(target_n_samples) / sr_hz

        # Read all waveform data and interpolate to common time grid
        sweeps_by_channel: Dict[str, np.ndarray] = {}
        channel_names: List[str] = []

        for idx, i in enumerate(waveform_indices):
            ch_name = signal_labels[i].strip()
            ch_rate = sample_rates[i]
            ch_unit = signal_headers[i].get('dimension', '')

            # Create informative channel name with units if available
            if ch_unit and ch_unit.strip():
                display_name = f"{ch_name} ({ch_unit.strip()})"
            else:
                display_name = ch_name

            # Handle duplicate channel names
            if display_name in channel_names:
                display_name = f"{display_name}_{i}"

            channel_names.append(display_name)

            if progress_callback:
                pct = 20 + int(70 * (idx + 1) / len(waveform_indices))
                progress_callback(pct, 100, f"Reading {ch_name}...")

            # Read the raw signal data
            try:
                data_raw = f.readSignal(i)
            except Exception as e:
                raise RuntimeError(f"Failed to read channel '{ch_name}': {str(e)}")

            # Check if we need to resample
            if abs(ch_rate - sr_hz) > 0.01:  # Allow small floating point differences
                # Different sample rate - need to resample
                # Create original time axis
                time_orig = np.arange(len(data_raw)) / ch_rate

                # Use linear interpolation to map data onto common time grid
                f_interp = interpolate.interp1d(
                    time_orig, data_raw,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(data_raw[0], data_raw[-1])  # Extend with edge values
                )
                data = f_interp(t_common)
            else:
                # Same sample rate - just pad or trim to match target length
                # This avoids interpolation artifacts
                if len(data_raw) < target_n_samples:
                    # Pad with last value
                    padding = np.full(target_n_samples - len(data_raw), data_raw[-1])
                    data = np.concatenate([data_raw, padding])
                elif len(data_raw) > target_n_samples:
                    # Trim to target length
                    data = data_raw[:target_n_samples]
                else:
                    data = data_raw

            # PhysioMetrics expects shape (n_samples, n_sweeps)
            # EDF files are continuous, so n_sweeps = 1
            sweeps_by_channel[display_name] = data.reshape(-1, 1)

        # Use the common time grid
        t = t_common

        if progress_callback:
            progress_callback(100, 100, "Complete")

        return sr_hz, sweeps_by_channel, channel_names, t

    finally:
        # Always close the file
        f.close()
