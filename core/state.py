from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Tuple


@dataclass
class AppState:
    # File & channels
    in_path: Optional[Path] = None
    file_info: List[Dict] = field(default_factory=list)  # For multi-file concatenation: [{'path', 'sweep_start', 'sweep_end'}]
    channel_names: List[str] = field(default_factory=list)
    analyze_chan: Optional[str] = None
    stim_chan: Optional[str] = None
    event_channel: Optional[str] = None  # Channel for event trace (e.g., lick detector)
    # core/state.py  (inside AppState dataclass)
    stim_onsets_by_sweep: dict[int, np.ndarray] = field(default_factory=dict)
    stim_offsets_by_sweep: dict[int, np.ndarray] = field(default_factory=dict)
    stim_spans_by_sweep: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    stim_metrics_by_sweep: dict[int, dict] = field(default_factory=dict)


    # Raw & processed
    sr_hz: Optional[float] = None
    sweeps: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (n_samples, n_sweeps) or 2D
    t: Optional[np.ndarray] = None

    # Filters
    use_low: bool = False
    use_high: bool = False
    use_mean_sub: bool = False
    use_invert: bool = False
    low_hz: Optional[float] = None
    high_hz: Optional[float] = None
    mean_val: float = 0.0

    # Navigation
    sweep_idx: int = 0
    window_start_s: float = 0.0
    window_dur_s: float = 10.0

    # Peaks & edits
    peaks_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)
    breath_by_sweep: Dict[int, Dict] = field(default_factory=dict)  # sweep -> {'onsets', 'offsets', 'expmins', 'expoffs'}
    sigh_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)  # sweep -> peak indices marked as sighs

    # ML Training Data: ALL detected peaks with auto-labels
    all_peaks_by_sweep: Dict[int, Dict] = field(default_factory=dict)  # sweep -> {'indices', 'labels', 'label_source', 'prominences'}
    all_breaths_by_sweep: Dict[int, Dict] = field(default_factory=dict)  # sweep -> {'onsets', 'offsets', 'expmins'} for ALL peaks (including noise)
    omitted_points: Dict[int, List[int]] = field(default_factory=dict)       # sweep -> sample idxs
    omitted_ranges: Dict[int, List[Tuple[int,int]]] = field(default_factory=dict)  # sweep -> [(i0,i1), ...]
    omitted_sweeps: set = field(default_factory=set)  # set of sweep indices to exclude
    sniff_regions_by_sweep: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)  # sweep -> [(start_time, end_time), ...]
    bout_annotations: Dict[int, List[Dict]] = field(default_factory=dict)  # sweep -> [{'start_time': float, 'end_time': float, 'id': int}, ...]

    # Y2 axis metrics
    y2_metric_key: Optional[str] = None  # e.g., "if", "ti", "te", etc.
    y2_values_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)  # sweep -> y2 metric values

    # GMM clustering
    gmm_sniff_probabilities: Dict[int, np.ndarray] = field(default_factory=dict)  # sweep -> probability of being sniff

    # Cached processed data (per sweep)
    proc_cache: Dict[Tuple[str,int], np.ndarray] = field(default_factory=dict)  # (chan, sweep) -> y
