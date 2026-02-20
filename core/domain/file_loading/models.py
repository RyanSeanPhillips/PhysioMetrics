"""
File loading domain models.

Pure Python dataclasses — no Qt imports. These represent the results
of file loading operations and are passed between service, viewmodel,
and view layers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FileLoadResult:
    """Result from loading a single data file (ABF, SMRX, EDF)."""
    sr_hz: float
    sweeps_by_ch: dict                  # {ch_name: np.ndarray}
    channel_names: list[str]
    time_array: np.ndarray
    file_metadata: dict                 # protocol, n_channels, file_type, etc.
    source_path: Path
    load_duration_seconds: float = 0.0


@dataclass
class MultiFileLoadResult:
    """Result from loading and concatenating multiple data files."""
    sr_hz: float
    sweeps_by_ch: dict                  # {ch_name: np.ndarray}
    channel_names: list[str]
    time_array: np.ndarray
    file_info: list[dict]               # Per-file metadata (sweep ranges, padding info)
    source_paths: list[Path] = field(default_factory=list)


@dataclass
class NpzLoadResult:
    """Result from loading an NPZ session file."""
    new_state: object                   # AppState instance
    raw_data_loaded: bool
    gmm_cache: Optional[object] = None
    app_settings: Optional[dict] = None
    event_markers: Optional[dict] = None
    npz_path: Optional[Path] = None
    metadata: Optional[dict] = None     # NPZ header metadata


@dataclass
class ChannelAutoDetection:
    """Results of auto-detecting stim and analysis channels."""
    stim_channel: Optional[str] = None
    analysis_channel: Optional[str] = None
