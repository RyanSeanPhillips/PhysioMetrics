"""
CTA (Condition-Triggered Average) domain models.

This module contains the core data structures for CTAs,
designed to be independent of any UI framework (Qt-free).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import numpy as np


@dataclass
class CTAConfig:
    """
    Configuration for CTA calculation (pure data).

    Attributes:
        window_before: Time window before event (seconds)
        window_after: Time window after event (seconds)
        n_points: Number of points in interpolated common time base
        include_withdrawal: Whether to compute withdrawal CTAs for paired markers
        zscore_baseline: Whether to z-score signals to baseline period
        baseline_start: Start of baseline period relative to event (seconds, negative = before)
        baseline_end: End of baseline period relative to event (seconds, typically 0)
    """
    window_before: float = 30.0
    window_after: float = 30.0
    n_points: int = 1000
    include_withdrawal: bool = True
    zscore_baseline: bool = True  # Z-score to baseline by default for photometry
    baseline_start: float = -5.0  # Baseline from -5s to -0.25s by default
    baseline_end: float = -0.25

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'window_before': self.window_before,
            'window_after': self.window_after,
            'n_points': self.n_points,
            'include_withdrawal': self.include_withdrawal,
            'zscore_baseline': self.zscore_baseline,
            'baseline_start': self.baseline_start,
            'baseline_end': self.baseline_end,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTAConfig':
        """Deserialize from dictionary."""
        return cls(
            window_before=float(d.get('window_before', 30.0)),
            window_after=float(d.get('window_after', 30.0)),
            n_points=int(d.get('n_points', 1000)),
            include_withdrawal=bool(d.get('include_withdrawal', True)),
            zscore_baseline=bool(d.get('zscore_baseline', True)),
            baseline_start=float(d.get('baseline_start', -5.0)),
            baseline_end=float(d.get('baseline_end', -0.25)),
        )


@dataclass
class CTATrace:
    """
    Single event-aligned trace (pure data).

    Represents one event's worth of data extracted and aligned to the event time.

    Attributes:
        event_id: ID of the source EventMarker
        sweep_idx: Sweep this trace came from
        event_time: Absolute time of the event in the original signal
        time: Time relative to event (e.g., -2 to +5 seconds)
        values: Signal values at each time point
    """
    event_id: str
    sweep_idx: int
    event_time: float
    time: np.ndarray
    values: np.ndarray
    paired_event_offset: Optional[float] = None  # Time offset to paired marker (onset→withdrawal or vice versa)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for JSON storage)."""
        d = {
            'event_id': self.event_id,
            'sweep_idx': self.sweep_idx,
            'event_time': float(self.event_time),
            'time': self.time.tolist() if isinstance(self.time, np.ndarray) else list(self.time),
            'values': self.values.tolist() if isinstance(self.values, np.ndarray) else list(self.values),
        }
        if self.paired_event_offset is not None:
            d['paired_event_offset'] = float(self.paired_event_offset)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTATrace':
        """Deserialize from dictionary."""
        return cls(
            event_id=str(d['event_id']),
            sweep_idx=int(d['sweep_idx']),
            event_time=float(d['event_time']),
            time=np.array(d['time'], dtype=np.float64),
            values=np.array(d['values'], dtype=np.float64),
            paired_event_offset=d.get('paired_event_offset'),
        )


@dataclass
class CTAResult:
    """
    Complete CTA result for one metric and alignment (pure data).

    Contains individual traces plus computed mean and SEM.

    Attributes:
        metric_key: Key identifying the metric (e.g., 'photometry_dff', 'IF', 'Ti')
        metric_label: Human-readable label for the metric
        alignment: 'onset' or 'withdrawal'
        category: Marker category this CTA is for (e.g., 'hargreaves')
        label: Marker label this CTA is for (e.g., 'heat_onset')
        config: Configuration used for calculation
        traces: List of individual event-aligned traces
        time_common: Common interpolated time base
        mean: Mean across all traces
        sem: Standard error of the mean
        n_events: Number of events included
    """
    metric_key: str
    metric_label: str
    alignment: str  # 'onset' or 'withdrawal'
    category: str
    label: str
    config: CTAConfig
    traces: List[CTATrace] = field(default_factory=list)
    time_common: Optional[np.ndarray] = None
    mean: Optional[np.ndarray] = None
    sem: Optional[np.ndarray] = None
    n_events: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'metric_key': self.metric_key,
            'metric_label': self.metric_label,
            'alignment': self.alignment,
            'category': self.category,
            'label': self.label,
            'config': self.config.to_dict(),
            'traces': [t.to_dict() for t in self.traces],
            'time_common': self.time_common.tolist() if self.time_common is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'sem': self.sem.tolist() if self.sem is not None else None,
            'n_events': self.n_events,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTAResult':
        """Deserialize from dictionary."""
        time_common = np.array(d['time_common'], dtype=np.float64) if d.get('time_common') else None
        mean = np.array(d['mean'], dtype=np.float64) if d.get('mean') else None
        sem = np.array(d['sem'], dtype=np.float64) if d.get('sem') else None

        return cls(
            metric_key=str(d['metric_key']),
            metric_label=str(d.get('metric_label', d['metric_key'])),
            alignment=str(d['alignment']),
            category=str(d.get('category', '')),
            label=str(d.get('label', '')),
            config=CTAConfig.from_dict(d.get('config', {})),
            traces=[CTATrace.from_dict(t) for t in d.get('traces', [])],
            time_common=time_common,
            mean=mean,
            sem=sem,
            n_events=int(d.get('n_events', 0)),
        )


@dataclass
class CTACollection:
    """
    Collection of CTA results for multiple metrics and alignments.

    This is the top-level container for all CTA data, designed for NPZ storage.

    Attributes:
        generated_at: Timestamp when CTAs were generated
        config: Default configuration used
        results: Dictionary of results keyed by '{category}:{label}:{alignment}:{metric}'
        metrics: List of metric keys included
    """
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    config: CTAConfig = field(default_factory=CTAConfig)
    results: Dict[str, CTAResult] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)

    def add_result(self, result: CTAResult) -> None:
        """Add a result to the collection."""
        key = f"{result.category}:{result.label}:{result.alignment}:{result.metric_key}"
        self.results[key] = result
        if result.metric_key not in self.metrics:
            self.metrics.append(result.metric_key)

    def get_result(self, category: str, label: str, alignment: str, metric_key: str) -> Optional[CTAResult]:
        """Get a specific result."""
        key = f"{category}:{label}:{alignment}:{metric_key}"
        return self.results.get(key)

    def get_results_for_marker_type(self, category: str, label: str) -> Dict[str, CTAResult]:
        """Get all results for a specific marker type."""
        prefix = f"{category}:{label}:"
        return {k: v for k, v in self.results.items() if k.startswith(prefix)}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for NPZ storage."""
        return {
            'generated_at': self.generated_at,
            'config': self.config.to_dict(),
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'metrics': self.metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTACollection':
        """Deserialize from dictionary."""
        return cls(
            generated_at=str(d.get('generated_at', '')),
            config=CTAConfig.from_dict(d.get('config', {})),
            results={k: CTAResult.from_dict(v) for k, v in d.get('results', {}).items()},
            metrics=list(d.get('metrics', [])),
        )

    def to_npz_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format suitable for NPZ storage.

        Returns dictionary with:
        - 'cta_version': Version number for format compatibility
        - 'cta_json': JSON string of the collection data
        """
        return {
            'cta_version': np.array([1]),
            'cta_json': json.dumps(self.to_dict()),
        }

    @classmethod
    def from_npz_dict(cls, data: Dict[str, Any]) -> Optional['CTACollection']:
        """
        Load from NPZ dictionary format.

        Args:
            data: Dictionary from NPZ file

        Returns:
            CTACollection or None if no CTA data present
        """
        if 'cta_json' not in data:
            return None

        try:
            cta_data = json.loads(str(data['cta_json']))
            return cls.from_dict(cta_data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[CTACollection] Warning: Failed to load CTA data: {e}")
            return None


@dataclass
class CTATabConfig:
    """
    Configuration for a single CTA comparison tab (pure data, no Qt).

    Captures all UI widget state needed to reconstruct a tab exactly
    as the user configured it.
    """
    tab_name: str = "CTA 1"
    trigger_source: str = "event_markers"  # "event_markers" or "breath_events"

    # Breath event config
    breath_trigger_point: str = "inspiratory_onset"
    breath_type_filter: str = "all"
    breath_max_events: int = 100
    breath_min_bout: int = 1
    breath_time_start: float = 0.0
    breath_time_end: float = 0.0

    # Time windows
    window_before: float = 30.0
    window_after: float = 30.0

    # Z-score normalization
    zscore_enabled: bool = True
    baseline_start: float = -5.0
    baseline_end: float = -0.25

    # Condition mode
    condition_mode: str = "overlay"

    # Selections
    selected_categories: List[str] = field(default_factory=list)
    selected_conditions: List[str] = field(default_factory=list)
    selected_metrics: List[str] = field(default_factory=list)
    metric_colors: Dict[str, str] = field(default_factory=dict)  # key -> hex

    # Display toggles
    show_paired_lines: bool = True
    show_baseline_zone: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tab_name': self.tab_name,
            'trigger_source': self.trigger_source,
            'breath_trigger_point': self.breath_trigger_point,
            'breath_type_filter': self.breath_type_filter,
            'breath_max_events': self.breath_max_events,
            'breath_min_bout': self.breath_min_bout,
            'breath_time_start': self.breath_time_start,
            'breath_time_end': self.breath_time_end,
            'window_before': self.window_before,
            'window_after': self.window_after,
            'zscore_enabled': self.zscore_enabled,
            'baseline_start': self.baseline_start,
            'baseline_end': self.baseline_end,
            'condition_mode': self.condition_mode,
            'selected_categories': self.selected_categories,
            'selected_conditions': self.selected_conditions,
            'selected_metrics': self.selected_metrics,
            'metric_colors': self.metric_colors,
            'show_paired_lines': self.show_paired_lines,
            'show_baseline_zone': self.show_baseline_zone,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTATabConfig':
        return cls(
            tab_name=str(d.get('tab_name', 'CTA 1')),
            trigger_source=str(d.get('trigger_source', 'event_markers')),
            breath_trigger_point=str(d.get('breath_trigger_point', 'inspiratory_onset')),
            breath_type_filter=str(d.get('breath_type_filter', 'all')),
            breath_max_events=int(d.get('breath_max_events', 100)),
            breath_min_bout=int(d.get('breath_min_bout', 1)),
            breath_time_start=float(d.get('breath_time_start', 0.0)),
            breath_time_end=float(d.get('breath_time_end', 0.0)),
            window_before=float(d.get('window_before', 30.0)),
            window_after=float(d.get('window_after', 30.0)),
            zscore_enabled=bool(d.get('zscore_enabled', True)),
            baseline_start=float(d.get('baseline_start', -5.0)),
            baseline_end=float(d.get('baseline_end', -0.25)),
            condition_mode=str(d.get('condition_mode', 'overlay')),
            selected_categories=list(d.get('selected_categories', [])),
            selected_conditions=list(d.get('selected_conditions', [])),
            selected_metrics=list(d.get('selected_metrics', [])),
            metric_colors=dict(d.get('metric_colors', {})),
            show_paired_lines=bool(d.get('show_paired_lines', True)),
            show_baseline_zone=bool(d.get('show_baseline_zone', True)),
        )


@dataclass
class CTAWorkspaceConfig:
    """
    Complete CTA workspace state — all tabs with configs and results.
    """
    active_tab_index: int = 0
    tabs: List[Dict[str, Any]] = field(default_factory=list)
    # Each tab entry: {'config': CTATabConfig.to_dict(), 'collection': CTACollection.to_dict() or None,
    #                   'condition_collections': {cond_name: CTACollection.to_dict()} or None}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': 2,
            'active_tab_index': self.active_tab_index,
            'tabs': self.tabs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CTAWorkspaceConfig':
        return cls(
            active_tab_index=int(d.get('active_tab_index', 0)),
            tabs=list(d.get('tabs', [])),
        )
