"""
Analysis domain models — pure Python, no Qt dependencies.

These dataclasses capture all configuration needed to run peak detection
and signal processing headlessly (without MainWindow).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FilterConfig:
    """Signal filtering configuration.

    Captures every parameter needed by ``filters.apply_all_1d`` plus
    notch and z-score normalisation so that ``_get_processed_for`` can
    be called without MainWindow.
    """

    # Butterworth filters (from AppState)
    use_low: bool = False
    low_hz: Optional[float] = None
    use_high: bool = False
    high_hz: Optional[float] = None
    use_mean_sub: bool = False
    mean_val: float = 0.0
    use_invert: bool = False
    filter_order: int = 4

    # Notch (band-stop) — currently on MainWindow
    notch_lower: Optional[float] = None
    notch_upper: Optional[float] = None

    # Z-score normalisation — currently on MainWindow
    use_zscore: bool = True
    zscore_global_mean: Optional[float] = None
    zscore_global_std: Optional[float] = None

    # ── serialisation ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "use_low": self.use_low,
            "low_hz": self.low_hz,
            "use_high": self.use_high,
            "high_hz": self.high_hz,
            "use_mean_sub": self.use_mean_sub,
            "mean_val": self.mean_val,
            "use_invert": self.use_invert,
            "filter_order": self.filter_order,
            "notch_lower": self.notch_lower,
            "notch_upper": self.notch_upper,
            "use_zscore": self.use_zscore,
            "zscore_global_mean": self.zscore_global_mean,
            "zscore_global_std": self.zscore_global_std,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> FilterConfig:
        return cls(
            use_low=bool(d.get("use_low", False)),
            low_hz=d.get("low_hz"),
            use_high=bool(d.get("use_high", False)),
            high_hz=d.get("high_hz"),
            use_mean_sub=bool(d.get("use_mean_sub", False)),
            mean_val=float(d.get("mean_val", 0.0)),
            use_invert=bool(d.get("use_invert", False)),
            filter_order=int(d.get("filter_order", 4)),
            notch_lower=d.get("notch_lower"),
            notch_upper=d.get("notch_upper"),
            use_zscore=bool(d.get("use_zscore", True)),
            zscore_global_mean=d.get("zscore_global_mean"),
            zscore_global_std=d.get("zscore_global_std"),
        )

    @classmethod
    def from_app_state(cls, st: Any, mw: Any = None) -> FilterConfig:
        """Build from AppState + optional MainWindow for fields still on mw.

        Args:
            st: AppState instance (has use_low, low_hz, etc.)
            mw: MainWindow instance (has notch_filter_lower, use_zscore_normalization, etc.)
                If None, notch/zscore fields default.
        """
        cfg = cls(
            use_low=st.use_low,
            low_hz=st.low_hz,
            use_high=st.use_high,
            high_hz=st.high_hz,
            use_mean_sub=st.use_mean_sub,
            mean_val=st.mean_val,
            use_invert=st.use_invert,
        )
        if mw is not None:
            cfg.filter_order = getattr(mw, "filter_order", 4)
            cfg.notch_lower = getattr(mw, "notch_filter_lower", None)
            cfg.notch_upper = getattr(mw, "notch_filter_upper", None)
            cfg.use_zscore = getattr(mw, "use_zscore_normalization", True)
            cfg.zscore_global_mean = getattr(mw, "zscore_global_mean", None)
            cfg.zscore_global_std = getattr(mw, "zscore_global_std", None)
        return cfg


@dataclass
class PeakDetectionConfig:
    """Peak detection parameters.

    Currently live on MainWindow as ``self.peak_prominence``, etc.
    """

    prominence: Optional[float] = None
    height_threshold: Optional[float] = None
    min_dist_sec: float = 0.05
    direction: str = "up"

    # Eupnea / apnea thresholds
    apnea_threshold_sec: float = 0.5
    eupnea_freq_threshold: float = 5.0
    eupnea_min_duration: float = 2.0
    eupnea_detection_mode: str = "gmm"

    # Outlier detection
    outlier_sd: float = 3.0
    outlier_metrics: List[str] = field(
        default_factory=lambda: [
            "if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prominence": self.prominence,
            "height_threshold": self.height_threshold,
            "min_dist_sec": self.min_dist_sec,
            "direction": self.direction,
            "apnea_threshold_sec": self.apnea_threshold_sec,
            "eupnea_freq_threshold": self.eupnea_freq_threshold,
            "eupnea_min_duration": self.eupnea_min_duration,
            "eupnea_detection_mode": self.eupnea_detection_mode,
            "outlier_sd": self.outlier_sd,
            "outlier_metrics": list(self.outlier_metrics),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PeakDetectionConfig:
        return cls(
            prominence=d.get("prominence"),
            height_threshold=d.get("height_threshold"),
            min_dist_sec=float(d.get("min_dist_sec", 0.05)),
            direction=str(d.get("direction", "up")),
            apnea_threshold_sec=float(d.get("apnea_threshold_sec", 0.5)),
            eupnea_freq_threshold=float(d.get("eupnea_freq_threshold", 5.0)),
            eupnea_min_duration=float(d.get("eupnea_min_duration", 2.0)),
            eupnea_detection_mode=str(d.get("eupnea_detection_mode", "gmm")),
            outlier_sd=float(d.get("outlier_sd", 3.0)),
            outlier_metrics=list(d.get("outlier_metrics", [
                "if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"
            ])),
        )

    @classmethod
    def from_main_window(cls, mw: Any) -> PeakDetectionConfig:
        """Build from MainWindow instance attributes."""
        apnea_text = ""
        if hasattr(mw, "ApneaThresh"):
            apnea_text = mw.ApneaThresh.text().strip() if hasattr(mw.ApneaThresh, "text") else str(mw.ApneaThresh)
        try:
            apnea_val = float(apnea_text) if apnea_text else 0.5
        except ValueError:
            apnea_val = 0.5

        outlier_text = ""
        if hasattr(mw, "OutlierSD"):
            outlier_text = mw.OutlierSD.text().strip() if hasattr(mw.OutlierSD, "text") else str(mw.OutlierSD)
        try:
            outlier_val = float(outlier_text) if outlier_text else 3.0
        except ValueError:
            outlier_val = 3.0

        return cls(
            prominence=getattr(mw, "peak_prominence", None),
            height_threshold=getattr(mw, "peak_height_threshold", None),
            min_dist_sec=getattr(mw, "peak_min_dist", 0.05),
            direction="up",
            apnea_threshold_sec=apnea_val,
            eupnea_freq_threshold=getattr(mw, "eupnea_freq_threshold", 5.0),
            eupnea_min_duration=getattr(mw, "eupnea_min_duration", 2.0),
            eupnea_detection_mode=getattr(mw, "eupnea_detection_mode", "gmm"),
            outlier_sd=outlier_val,
            outlier_metrics=list(getattr(
                getattr(mw, "plot_manager", None), "_outlier_metrics",
                ["if", "amp_insp", "amp_exp", "ti", "te", "area_insp", "area_exp"]
            )),
        )


@dataclass
class AnalysisConfig:
    """Complete analysis configuration — everything needed for headless batch.

    Combines filter settings, peak detection settings, and classifier choice.
    """

    filter: FilterConfig = field(default_factory=FilterConfig)
    peak: PeakDetectionConfig = field(default_factory=PeakDetectionConfig)

    # Classifier settings (from AppState)
    active_classifier: str = "xgboost"
    active_eupnea_sniff_classifier: str = "gmm"
    active_sigh_classifier: str = "xgboost"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filter": self.filter.to_dict(),
            "peak": self.peak.to_dict(),
            "active_classifier": self.active_classifier,
            "active_eupnea_sniff_classifier": self.active_eupnea_sniff_classifier,
            "active_sigh_classifier": self.active_sigh_classifier,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AnalysisConfig:
        return cls(
            filter=FilterConfig.from_dict(d.get("filter", {})),
            peak=PeakDetectionConfig.from_dict(d.get("peak", {})),
            active_classifier=str(d.get("active_classifier", "xgboost")),
            active_eupnea_sniff_classifier=str(d.get("active_eupnea_sniff_classifier", "gmm")),
            active_sigh_classifier=str(d.get("active_sigh_classifier", "xgboost")),
        )

    @classmethod
    def from_app_state(cls, st: Any, mw: Any = None) -> AnalysisConfig:
        """Build from AppState + optional MainWindow."""
        return cls(
            filter=FilterConfig.from_app_state(st, mw),
            peak=PeakDetectionConfig.from_main_window(mw) if mw else PeakDetectionConfig(),
            active_classifier=getattr(st, "active_classifier", "xgboost"),
            active_eupnea_sniff_classifier=getattr(st, "active_eupnea_sniff_classifier", "gmm"),
            active_sigh_classifier=getattr(st, "active_sigh_classifier", "xgboost"),
        )


@dataclass
class AnalysisResult:
    """Result from analysing a single file.

    Produced by AnalysisService.analyze_file(), consumed by batch runner
    and stored on disk as session NPZ + results CSV.
    """

    file_path: Path
    n_sweeps: int = 0
    n_peaks_total: int = 0
    n_breaths_total: int = 0
    session_path: Optional[Path] = None   # NPZ path
    results_path: Optional[Path] = None   # CSV path
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": str(self.file_path),
            "n_sweeps": self.n_sweeps,
            "n_peaks_total": self.n_peaks_total,
            "n_breaths_total": self.n_breaths_total,
            "session_path": str(self.session_path) if self.session_path else None,
            "results_path": str(self.results_path) if self.results_path else None,
            "error": self.error,
            "warnings": self.warnings,
        }
