"""
MATLAB .mat file loader for Spike2 exports
Compatible with PlethApp's data loading interface
"""
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_mat(file_path: str, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load a MATLAB .mat file exported from Spike2

    Args:
        file_path: Path to .mat file (HDF5 format, MATLAB v7.3)
        progress_callback: Optional callback(current, total, message)

    Returns:
        (sr_hz, sweeps_by_channel, channel_names, t)
        Compatible with PlethApp's ABF loader interface
    """
    file_path = Path(file_path)

    if progress_callback:
        progress_callback(0, 100, "Opening MAT file...")

    with h5py.File(file_path, 'r') as f:
        # Find all channel keys (exclude metadata)
        channel_keys = [k for k in f.keys() if not k.startswith('#') and k != 'file']

        if not channel_keys:
            raise ValueError("No channels found in MAT file")

        if progress_callback:
            progress_callback(10, 100, f"Found {len(channel_keys)} channels...")

        sweeps_by_channel = {}
        channel_names = []
        sample_rates = []

        for idx, ch_key in enumerate(sorted(channel_keys)):
            if progress_callback:
                pct = 10 + int(80 * (idx + 1) / len(channel_keys))
                progress_callback(pct, 100, f"Loading channel {idx+1}/{len(channel_keys)}")

            ch_group = f[ch_key]

            # Skip marker/keyboard channels (they have 'codes' instead of 'values')
            if 'values' not in ch_group:
                continue

            # Extract channel name from title
            if 'title' in ch_group:
                title_data = ch_group['title'][:]
                try:
                    # Decode ASCII title
                    ch_name = ''.join(chr(int(c[0])) for c in title_data if c[0] != 0)
                except:
                    ch_name = ch_key

            else:
                ch_name = ch_key

            # Get data
            values = ch_group['values'][0, :]  # Shape is (1, N), extract to (N,)

            # Get sample rate from interval
            if 'interval' in ch_group:
                interval = ch_group['interval'][0, 0]
                sr = 1.0 / interval
            else:
                sr = 1.0  # Default

            # Store
            sweeps_by_channel[ch_name] = values.reshape(-1, 1)
            channel_names.append(ch_name)
            sample_rates.append(sr)

        if not channel_names:
            raise ValueError("No waveform channels found (only markers?)")

        # Use highest sample rate for time base
        sr_hz = max(sample_rates)

        # Create time vector based on longest channel
        max_samples = max(data.shape[0] for data in sweeps_by_channel.values())
        t = np.arange(max_samples) / sr_hz

        if progress_callback:
            progress_callback(100, 100, "Complete")

        return sr_hz, sweeps_by_channel, channel_names, t


if __name__ == '__main__':
    # Test the loader
    test_file = r"C:\Users\rphil2\Dropbox\python scripts\breath_analysis\pyqt6\examples\100225_000.mat"
    print(f"Testing MAT loader on: {test_file}\n")

    try:
        sr_hz, sweeps, chan_names, t = load_mat(test_file)

        print(f"\n" + "="*80)
        print(f"SUCCESS! Loaded MAT file")
        print("="*80)
        print(f"  Sample rate: {sr_hz} Hz")
        print(f"  Channels: {chan_names}")
        print(f"  Duration: {t[-1]:.2f} seconds ({t[-1]/60:.1f} minutes)")

        for name in chan_names:
            data = sweeps[name]
            print(f"\n  {name}:")
            print(f"    Shape: {data.shape}")
            print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
            print(f"    Mean: {data.mean():.6f}, Std: {data.std():.6f}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
