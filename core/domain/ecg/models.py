"""
ECG domain models -- pure Python, no Qt dependencies.

Dataclasses for ECG/EKG analysis configuration, per-sweep results,
and HRV (heart rate variability) metrics.  All are serialisable via
``to_dict()`` / ``from_dict()`` for session save/load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── Species presets ──────────────────────────────────────────────────

_SPECIES_PRESETS: Dict[str, Dict[str, Any]] = {
    "mouse": {
        "bandpass_low": 10.0,
        "bandpass_high": 200.0,
        "refractory_ms": 60.0,      # max ~1000 BPM
        "mwi_window_ms": 20.0,      # QRS ~10ms
        "qrs_search_ms": 10.0,
        "expected_hr_range": (400, 800),
        "pnnx_threshold_ms": 6.0,   # pNN6 for mice
        "hrv_lf_band": (0.15, 1.5),
        "hrv_hf_band": (1.5, 5.0),
    },
    "rat": {
        "bandpass_low": 5.0,
        "bandpass_high": 150.0,
        "refractory_ms": 100.0,     # max ~600 BPM
        "mwi_window_ms": 40.0,      # QRS ~20ms
        "qrs_search_ms": 20.0,
        "expected_hr_range": (250, 550),
        "pnnx_threshold_ms": 10.0,  # pNN10 for rats
        "hrv_lf_band": (0.04, 0.6),
        "hrv_hf_band": (0.6, 2.5),
    },
    "human": {
        "bandpass_low": 5.0,
        "bandpass_high": 15.0,
        "refractory_ms": 200.0,     # max ~300 BPM
        "mwi_window_ms": 150.0,     # QRS ~100ms
        "qrs_search_ms": 50.0,
        "expected_hr_range": (40, 200),
        "pnnx_threshold_ms": 50.0,  # classic pNN50
        "hrv_lf_band": (0.04, 0.15),
        "hrv_hf_band": (0.15, 0.4),
    },
}


# ── ECG configuration ───────────────────────────────────────────────


@dataclass
class ECGConfig:
    """Configuration for ECG R-peak detection and HRV analysis.

    Call ``apply_species_preset()`` after changing *species* to update
    all derived parameters to the species defaults.
    """

    species: str = "mouse"

    # Bandpass filter
    bandpass_low: float = 10.0   # Hz
    bandpass_high: float = 200.0  # Hz
    filter_order: int = 3

    # Peak detection
    refractory_ms: float = 60.0         # min inter-beat interval
    mwi_window_ms: float = 20.0         # moving-window integration width
    qrs_search_ms: float = 10.0         # refine search radius
    threshold_percentile: float = 90.0   # for initial threshold
    threshold_fraction: float = 0.15     # fraction of rolling max (low = more sensitive)
    prominence_fraction: float = 0.05    # min prominence as fraction of percentile

    # Polarity: None = auto-detect, True = inverted, False = upright
    force_inverted: Optional[bool] = None

    # Expected HR range for sanity checks (BPM)
    expected_hr_min: float = 400.0
    expected_hr_max: float = 800.0

    # HRV analysis
    pnnx_threshold_ms: float = 6.0   # pNNx threshold
    hrv_lf_band: tuple = (0.15, 1.5)  # LF frequency band (Hz)
    hrv_hf_band: tuple = (1.5, 5.0)   # HF frequency band (Hz)

    def apply_species_preset(self) -> None:
        """Overwrite detection/HRV params with species defaults."""
        preset = _SPECIES_PRESETS.get(self.species)
        if preset is None:
            return
        self.bandpass_low = preset["bandpass_low"]
        self.bandpass_high = preset["bandpass_high"]
        self.refractory_ms = preset["refractory_ms"]
        self.mwi_window_ms = preset["mwi_window_ms"]
        self.qrs_search_ms = preset["qrs_search_ms"]
        self.expected_hr_min, self.expected_hr_max = preset["expected_hr_range"]
        self.pnnx_threshold_ms = preset["pnnx_threshold_ms"]
        self.hrv_lf_band = tuple(preset["hrv_lf_band"])
        self.hrv_hf_band = tuple(preset["hrv_hf_band"])

    # ── serialisation ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "bandpass_low": self.bandpass_low,
            "bandpass_high": self.bandpass_high,
            "filter_order": self.filter_order,
            "refractory_ms": self.refractory_ms,
            "mwi_window_ms": self.mwi_window_ms,
            "qrs_search_ms": self.qrs_search_ms,
            "threshold_percentile": self.threshold_percentile,
            "threshold_fraction": self.threshold_fraction,
            "prominence_fraction": self.prominence_fraction,
            "force_inverted": self.force_inverted,
            "expected_hr_min": self.expected_hr_min,
            "expected_hr_max": self.expected_hr_max,
            "pnnx_threshold_ms": self.pnnx_threshold_ms,
            "hrv_lf_band": list(self.hrv_lf_band),
            "hrv_hf_band": list(self.hrv_hf_band),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ECGConfig:
        c = cls(species=d.get("species", "mouse"))
        c.bandpass_low = float(d.get("bandpass_low", c.bandpass_low))
        c.bandpass_high = float(d.get("bandpass_high", c.bandpass_high))
        c.filter_order = int(d.get("filter_order", c.filter_order))
        c.refractory_ms = float(d.get("refractory_ms", c.refractory_ms))
        c.mwi_window_ms = float(d.get("mwi_window_ms", c.mwi_window_ms))
        c.qrs_search_ms = float(d.get("qrs_search_ms", c.qrs_search_ms))
        c.threshold_percentile = float(
            d.get("threshold_percentile", c.threshold_percentile)
        )
        c.threshold_fraction = float(
            d.get("threshold_fraction", c.threshold_fraction)
        )
        c.prominence_fraction = float(
            d.get("prominence_fraction", c.prominence_fraction)
        )
        c.force_inverted = d.get("force_inverted", None)
        c.expected_hr_min = float(d.get("expected_hr_min", c.expected_hr_min))
        c.expected_hr_max = float(d.get("expected_hr_max", c.expected_hr_max))
        c.pnnx_threshold_ms = float(
            d.get("pnnx_threshold_ms", c.pnnx_threshold_ms)
        )
        lf = d.get("hrv_lf_band", c.hrv_lf_band)
        c.hrv_lf_band = (float(lf[0]), float(lf[1]))
        hf = d.get("hrv_hf_band", c.hrv_hf_band)
        c.hrv_hf_band = (float(hf[0]), float(hf[1]))
        return c

    @classmethod
    def for_species(cls, species: str) -> ECGConfig:
        """Create a config pre-filled with species defaults."""
        c = cls(species=species)
        c.apply_species_preset()
        return c


# ── Per-sweep ECG result ─────────────────────────────────────────────


@dataclass
class ECGResult:
    """R-peak detection + heart rate for one sweep.

    Stores only the compact data (peak indices, labels).  The full-length
    instantaneous HR trace is recomputed on demand via ``compute_hr_trace``.
    """

    r_peaks: np.ndarray              # sample indices of R-peaks
    rr_intervals_ms: np.ndarray      # inter-beat intervals (ms)
    labels: np.ndarray               # 1=valid beat, 0=artifact/noise
    label_source: np.ndarray         # 'auto' or 'user' per beat (object array)
    is_inverted: bool = False        # True if R-peaks are negative deflections
    quality_score: float = 1.0       # 0-1 overall signal quality

    def valid_r_peaks(self) -> np.ndarray:
        """Return only R-peak indices labelled as valid."""
        return self.r_peaks[self.labels == 1]

    def valid_rr_intervals_ms(self) -> np.ndarray:
        """RR intervals between successive *valid* beats only."""
        vp = self.valid_r_peaks()
        if len(vp) < 2:
            return np.array([], dtype=np.float64)
        return np.diff(vp).astype(np.float64)  # in samples — caller scales by sr

    def mean_hr_bpm(self, sr_hz: float) -> float:
        """Mean heart rate in BPM from valid beats."""
        vp = self.valid_r_peaks()
        if len(vp) < 2:
            return 0.0
        rr_sec = np.diff(vp) / sr_hz
        return float(60.0 / np.mean(rr_sec))

    def compute_hr_trace(self, sr_hz: float, signal_length: int) -> np.ndarray:
        """Instantaneous HR (BPM) interpolated to signal length.

        Uses stepwise-constant (zero-order hold) for each inter-beat
        interval, matching the respiratory metric convention.
        """
        vp = self.valid_r_peaks()
        if len(vp) < 2:
            return np.full(signal_length, np.nan, dtype=np.float64)

        rr_sec = np.diff(vp) / sr_hz
        hr_bpm = 60.0 / rr_sec

        # Stepwise-constant: each beat's HR extends from that R-peak
        # to the next R-peak.
        trace = np.full(signal_length, np.nan, dtype=np.float64)
        for i in range(len(hr_bpm)):
            start = int(vp[i])
            end = int(vp[i + 1]) if i + 1 < len(vp) else signal_length
            end = min(end, signal_length)
            trace[start:end] = hr_bpm[i]

        # Fill before first beat and after last beat
        if len(vp) > 0 and vp[0] > 0:
            trace[: int(vp[0])] = trace[int(vp[0])] if not np.isnan(
                trace[int(vp[0])]
            ) else np.nan
        if len(vp) > 1 and vp[-1] < signal_length:
            trace[int(vp[-1]):] = hr_bpm[-1]

        return trace

    # ── serialisation ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_peaks": self.r_peaks.tolist(),
            "rr_intervals_ms": self.rr_intervals_ms.tolist(),
            "labels": self.labels.tolist(),
            "label_source": self.label_source.tolist(),
            "is_inverted": self.is_inverted,
            "quality_score": self.quality_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ECGResult:
        return cls(
            r_peaks=np.asarray(d["r_peaks"], dtype=np.int64),
            rr_intervals_ms=np.asarray(d["rr_intervals_ms"], dtype=np.float64),
            labels=np.asarray(d["labels"], dtype=np.int8),
            label_source=np.asarray(d["label_source"], dtype=object),
            is_inverted=bool(d.get("is_inverted", False)),
            quality_score=float(d.get("quality_score", 1.0)),
        )


# ── HRV result ───────────────────────────────────────────────────────


@dataclass
class HRVResult:
    """Heart rate variability metrics for a time window.

    Time-domain metrics are always computed.  Frequency-domain and
    non-linear metrics are optional (computed on demand in the HRV dialog).
    """

    # Time-domain (always present)
    mean_hr_bpm: float = 0.0
    sdnn_ms: float = 0.0
    rmssd_ms: float = 0.0
    pnnx_pct: float = 0.0          # % of successive diffs > threshold
    pnnx_threshold_ms: float = 6.0  # which threshold was used
    min_hr_bpm: float = 0.0
    max_hr_bpm: float = 0.0
    sdsd_ms: float = 0.0           # std of successive differences
    cv: float = 0.0                # coefficient of variation (SDNN/meanRR)
    n_beats: int = 0

    # Frequency-domain (optional)
    lf_power_ms2: Optional[float] = None
    hf_power_ms2: Optional[float] = None
    lf_hf_ratio: Optional[float] = None
    total_power_ms2: Optional[float] = None

    # Poincare (optional)
    sd1_ms: Optional[float] = None
    sd2_ms: Optional[float] = None

    # Respiratory-cardiac coupling (optional, requires pleth data)
    rsa_amplitude_bpm: Optional[float] = None
    phase_coherence: Optional[float] = None

    # ── serialisation ────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is not None:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> HRVResult:
        h = cls()
        for k, v in d.items():
            if hasattr(h, k):
                setattr(h, k, v)
        return h
