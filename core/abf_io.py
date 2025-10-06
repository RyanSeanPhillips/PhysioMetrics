from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple


def load_data_file(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Load data file - dispatches to appropriate loader based on file extension

    Supported formats:
    - .abf: Axon Binary Format (pyabf)
    - .smrx: Son64 format (CED Spike2) via Python 3.9 bridge

    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    progress_callback: optional function(current, total, message) for progress updates
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.abf':
        return load_abf(path, progress_callback)
    elif suffix == '.smrx':
        from core.io.son64_loader import load_son64
        return load_son64(str(path), progress_callback)
    else:
        raise ValueError(f"Unsupported file format: {suffix}\nSupported formats: .abf, .smrx")


def load_abf(path: Path, progress_callback=None) -> Tuple[float, Dict[str, np.ndarray], List[str], np.ndarray]:
    """
    Returns (sr_hz, sweeps_by_channel, channel_names, t)
    sweeps_by_channel[channel] -> 2D array (n_samples, n_sweeps)

    progress_callback: optional function(current, total, message) for progress updates
    """
    try:
        import pyabf
    except ImportError:
        raise RuntimeError("pyabf not installed. pip install pyabf")

    if progress_callback:
        progress_callback(0, 100, "Opening ABF file...")

    abf = pyabf.ABF(str(path))
    sr_hz = float(abf.dataRate)
    chan_names = [abf.adcNames[c] for c in range(abf.channelCount)]

    if progress_callback:
        progress_callback(10, 100, "Reading channel data...")

    # Collect sweeps into arrays per channel
    sweeps_by_channel: Dict[str, np.ndarray] = {}
    n_sweeps = abf.sweepCount
    n_samples = abf.sweepPointCount
    total_ops = abf.channelCount * n_sweeps

    op_count = 0
    for ch in range(abf.channelCount):
        abf.setSweep(0, channel=ch)
        M = np.empty((n_samples, n_sweeps), dtype=float)
        for s in range(n_sweeps):
            abf.setSweep(sweepNumber=s, channel=ch)
            M[:, s] = abf.sweepY

            op_count += 1
            if progress_callback and op_count % 5 == 0:  # Update every 5 sweeps
                pct = 10 + int(80 * op_count / total_ops)
                progress_callback(pct, 100, f"Loading channel {ch+1}/{abf.channelCount}, sweep {s+1}/{n_sweeps}...")

        sweeps_by_channel[chan_names[ch]] = M

    if progress_callback:
        progress_callback(95, 100, "Finalizing...")

    # time vector for one sweep
    t = np.arange(n_samples, dtype=float) / sr_hz

    if progress_callback:
        progress_callback(100, 100, "Complete")

    return sr_hz, sweeps_by_channel, chan_names, t
