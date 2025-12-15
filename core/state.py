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

    # Display controls
    use_percentile_autoscale: bool = False  # Y-axis autoscaling: True = 99th percentile, False = min/max
    autoscale_padding: float = 0.25         # Padding multiplier for percentile autoscale (0.25 = 25%)
    eupnea_use_shade: bool = False          # Eupnea display: True = background shade, False = line at top
    sniffing_use_shade: bool = False        # Sniffing display: True = background shade, False = line at top
    apnea_use_shade: bool = False           # Apnea display: True = background shade, False = line at bottom
    outliers_use_shade: bool = False        # Outliers display: True = background shade, False = line

    # Peaks & edits
    peaks_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)
    breath_by_sweep: Dict[int, Dict] = field(default_factory=dict)  # sweep -> {'onsets', 'offsets', 'expmins', 'expoffs'}
    sigh_by_sweep: Dict[int, np.ndarray] = field(default_factory=dict)  # sweep -> peak indices marked as sighs

    # ML Training Data: ALL detected peaks with labels
    # Structure: sweep -> {
    #   'indices': np.ndarray,              # Peak sample indices
    #
    #   # Breath vs Noise classification
    #   'labels': np.ndarray,               # USER-EDITABLE labels (0=noise, 1=breath) - what gets displayed and edited
    #   'label_source': np.ndarray,         # 'auto' or 'user' - tracks editing provenance
    #   'labels_threshold_ro': np.ndarray,  # Read-only threshold predictions (for classifier switching)
    #   'labels_xgboost_ro': np.ndarray,    # Read-only XGBoost predictions (for classifier switching)
    #   'labels_rf_ro': np.ndarray,         # Read-only Random Forest predictions (for classifier switching)
    #   'labels_mlp_ro': np.ndarray,        # Read-only MLP predictions (for classifier switching)
    #
    #   # Eupnea/Sniffing classification (only for breaths where labels==1)
    #   'breath_type_class': np.ndarray,    # USER-EDITABLE: -1=unclassified, 0=eupnea, 1=sniffing (from active classifier)
    #   'eupnea_sniff_source': np.ndarray,  # 'gmm', 'user_region', 'xgboost', etc.
    #   'gmm_class_ro': np.ndarray,         # Read-only GMM predictions (for classifier switching)
    #   'eupnea_sniff_xgboost_ro': np.ndarray,  # Read-only XGBoost predictions
    #   'eupnea_sniff_rf_ro': np.ndarray,       # Read-only Random Forest predictions
    #   'eupnea_sniff_mlp_ro': np.ndarray,      # Read-only MLP predictions
    #
    #   'prominences': np.ndarray,
    # }
    all_peaks_by_sweep: Dict[int, Dict] = field(default_factory=dict)
    all_breaths_by_sweep: Dict[int, Dict] = field(default_factory=dict)  # sweep -> {'onsets', 'offsets', 'expmins'} for ALL peaks (including noise)
    peak_metrics_by_sweep: Dict[int, List[Dict]] = field(default_factory=dict)  # sweep -> list of metric dicts for ORIGINAL auto-detected peaks (NEVER modified, for ML training)
    current_peak_metrics_by_sweep: Dict[int, List[Dict]] = field(default_factory=dict)  # sweep -> list of metric dicts for CURRENT edited peaks (for Y2 plotting, updated after edits)
    user_merge_decisions: Dict[int, List[Dict]] = field(default_factory=dict)  # sweep -> [{'peak1_idx', 'peak2_idx', 'kept_idx', 'removed_idx', 'timestamp'}] - tracks user merge decisions for ML training

    # ML Models: Loaded models and their metadata
    loaded_ml_models: Dict[str, Dict] = field(default_factory=dict)  # {'model1_xgboost': {'model': model_obj, 'metadata': {...}}, ...}
    active_classifier: str = 'xgboost'  # Which classifier to use for display: 'threshold', 'xgboost', 'rf', 'mlp'
    active_eupnea_sniff_classifier: str = 'gmm'  # Which classifier for eupnea/sniff: 'gmm', 'xgboost', 'rf', 'mlp', 'all_eupnea', 'none'
    active_sigh_classifier: str = 'xgboost'  # Which classifier for sigh: 'manual', 'xgboost', 'rf', 'mlp'
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
