"""
AnalysisService — headless signal processing and peak detection.

Pure Python (no Qt). All functions take config + data in, return results out.
This is the core engine that both the GUI and batch runner use.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.domain.analysis.models import (
    AnalysisConfig, AnalysisResult, FilterConfig, PeakDetectionConfig,
)
import core.filters as filters
import core.peaks as peakdet


# ── Signal processing ────────────────────────────────────────────


def get_processed_signal(
    raw_sweep: np.ndarray,
    sr_hz: float,
    config: FilterConfig,
) -> np.ndarray:
    """Apply the full filter chain to a single sweep.

    Equivalent to MainWindow._get_processed_for but without caching or
    channel gating (caller decides which channels to filter).

    Args:
        raw_sweep: 1-D raw signal array
        sr_hz: Sample rate in Hz
        config: FilterConfig with all filter parameters

    Returns:
        Processed signal array (same length as raw_sweep)
    """
    y = filters.apply_all_1d(
        raw_sweep, sr_hz,
        config.use_low, config.low_hz,
        config.use_high, config.high_hz,
        config.use_mean_sub, config.mean_val,
        config.use_invert,
        order=config.filter_order,
    )

    # Notch filter
    if config.notch_lower is not None and config.notch_upper is not None:
        y = filters.notch_filter_1d(y, sr_hz, config.notch_lower, config.notch_upper)

    # Z-score normalisation
    if config.use_zscore:
        y = filters.zscore_normalize(y, config.zscore_global_mean, config.zscore_global_std)

    return y


def compute_normalization_stats(
    sweeps_2d: np.ndarray,
    sr_hz: float,
    config: FilterConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute global mean/std across all sweeps for z-score normalisation.

    Applies all filters *except* z-score to each sweep, then computes
    statistics across the concatenated result.

    Args:
        sweeps_2d: (n_samples, n_sweeps) raw data array
        sr_hz: Sample rate
        config: FilterConfig (use_zscore is ignored — we compute the stats)
        progress_callback: Optional (current, total) callback

    Returns:
        (global_mean, global_std) or (None, None) if all-NaN
    """
    n_samples, n_sweeps = sweeps_2d.shape

    # Build a config without z-score for the pre-normalisation pass
    cfg_no_z = FilterConfig(
        use_low=config.use_low, low_hz=config.low_hz,
        use_high=config.use_high, high_hz=config.high_hz,
        use_mean_sub=config.use_mean_sub, mean_val=config.mean_val,
        use_invert=config.use_invert, filter_order=config.filter_order,
        notch_lower=config.notch_lower, notch_upper=config.notch_upper,
        use_zscore=False,
    )

    all_data = np.empty(n_samples * n_sweeps, dtype=np.float64)

    for i in range(n_sweeps):
        y = get_processed_signal(sweeps_2d[:, i], sr_hz, cfg_no_z)
        all_data[i * n_samples:(i + 1) * n_samples] = y
        if progress_callback and i % 10 == 0:
            progress_callback(i, n_sweeps)

    if progress_callback:
        progress_callback(n_sweeps, n_sweeps)

    valid = ~np.isnan(all_data)
    if not np.any(valid):
        return None, None

    return float(np.mean(all_data[valid])), float(np.std(all_data[valid], ddof=1))


# ── Peak detection (single sweep) ────────────────────────────────


def detect_single_sweep(
    sweep_idx: int,
    y_proc: np.ndarray,
    t: np.ndarray,
    sr_hz: float,
    config: PeakDetectionConfig,
) -> Dict[str, Any]:
    """Core peak detection for one sweep — no state mutation, fully parallelisable.

    This is the pure-function equivalent of MainWindow._detect_single_sweep_core.

    Args:
        sweep_idx: Sweep index (passed through for result identification)
        y_proc: Pre-processed signal (already filtered)
        t: Time array
        sr_hz: Sample rate
        config: PeakDetectionConfig

    Returns:
        Dict with detection results matching the legacy format.
    """
    min_dist_samples = None
    if config.min_dist_sec is not None and config.min_dist_sec > 0:
        min_dist_samples = max(1, int(round(config.min_dist_sec * sr_hz)))

    thresh = config.height_threshold
    direction = config.direction

    # Step 1: Detect ALL peaks (no threshold filtering)
    all_peak_indices = peakdet.detect_peaks(
        y=y_proc, sr_hz=sr_hz,
        thresh=None,
        prominence=None,
        min_dist_samples=min_dist_samples,
        direction=direction,
        return_all=True,
    )

    # Step 2: Breath features for ALL peaks
    all_breaths = peakdet.compute_breath_events(
        y_proc, all_peak_indices, sr_hz=sr_hz, exclude_sec=0.030
    )

    # Step 3: Label by threshold
    all_peaks_data = peakdet.label_peaks_by_threshold(
        y=y_proc,
        peak_indices=all_peak_indices,
        thresh=thresh,
        direction=direction,
    )
    all_peaks_data["labels_threshold_ro"] = all_peaks_data["labels"].copy()
    all_peaks_data["labels_xgboost_ro"] = None
    all_peaks_data["labels_rf_ro"] = None
    all_peaks_data["labels_mlp_ro"] = None

    # Step 4: p_noise
    try:
        import core.metrics as metrics_mod
        on = all_breaths.get("onsets", np.array([]))
        off = all_breaths.get("offsets", np.array([]))
        exm = all_breaths.get("expmins", np.array([]))
        exo = all_breaths.get("expoffs", np.array([]))
        p_noise_all = metrics_mod.compute_p_noise(t, y_proc, sr_hz, all_peak_indices, on, off, exm, exo)
        p_breath_all = 1.0 - p_noise_all if p_noise_all is not None else None
    except Exception:
        p_noise_all = None
        p_breath_all = None

    # Step 5: Peak candidate metrics
    peak_metrics = peakdet.compute_peak_candidate_metrics(
        y=y_proc,
        all_peak_indices=all_peak_indices,
        breath_events=all_breaths,
        sr_hz=sr_hz,
        p_noise=p_noise_all,
        p_breath=p_breath_all,
    )

    # Step 6: Labeled peaks
    labeled_mask = all_peaks_data["labels"] == 1
    labeled_indices = all_peak_indices[labeled_mask]

    # Step 7: Breath events for labeled peaks
    labeled_breaths = peakdet.compute_breath_events(
        y_proc, labeled_indices, sr_hz=sr_hz, exclude_sec=0.030
    )

    # Step 8: Current metrics (using labeled peaks as neighbours)
    try:
        import core.metrics as metrics_mod
        p_noise_labeled = metrics_mod.compute_p_noise(
            t, y_proc, sr_hz, labeled_indices,
            labeled_breaths.get("onsets", np.array([])),
            labeled_breaths.get("offsets", np.array([])),
            labeled_breaths.get("expmins", np.array([])),
            labeled_breaths.get("expoffs", np.array([])),
        )
        p_breath_labeled = 1.0 - p_noise_labeled if p_noise_labeled is not None else None
        current_metrics = peakdet.compute_peak_candidate_metrics(
            y=y_proc,
            all_peak_indices=labeled_indices,
            breath_events=labeled_breaths,
            sr_hz=sr_hz,
            p_noise=p_noise_labeled,
            p_breath=p_breath_labeled,
        )
    except Exception:
        current_metrics = peak_metrics

    return {
        "sweep_idx": sweep_idx,
        "all_peak_indices": all_peak_indices,
        "all_breaths": all_breaths,
        "all_peaks_data": all_peaks_data,
        "peak_metrics": peak_metrics,
        "current_metrics": current_metrics,
        "labeled_indices": labeled_indices,
        "labeled_breaths": labeled_breaths,
        "p_noise_all": p_noise_all,
        "p_breath_all": p_breath_all,
    }


# ── Auto-threshold detection ────────────────────────────────────


def auto_detect_threshold(
    y_data: np.ndarray,
    sr_hz: float,
    min_dist_sec: float = 0.05,
) -> Optional[float]:
    """Auto-detect peak height threshold using Otsu + valley fitting.

    Pure function extracted from MainWindow._compute_auto_threshold +
    _calculate_local_minimum_threshold_silent.

    Args:
        y_data: 1-D processed signal (e.g. concatenated or single sweep)
        sr_hz: Sample rate
        min_dist_sec: Minimum inter-peak distance in seconds

    Returns:
        Optimal threshold value, or None if detection fails.
    """
    from scipy.signal import find_peaks

    min_dist_samples = max(1, int(round(min_dist_sec * sr_hz)))

    peaks, props = find_peaks(y_data, height=0, prominence=0.001, distance=min_dist_samples)
    peak_heights = y_data[peaks]

    if len(peak_heights) < 10:
        return None

    # Otsu's method
    heights_norm = (
        (peak_heights - peak_heights.min())
        / (peak_heights.max() - peak_heights.min())
        * 255
    ).astype(np.uint8)
    hist, bin_edges = np.histogram(heights_norm, bins=256, range=(0, 256))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    w1 = np.cumsum(hist)
    w2 = np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * bin_centers) / (w1 + 1e-10)
    m2 = (np.cumsum((hist * bin_centers)[::-1]) / (w2 + 1e-10))[::-1]
    variance = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    optimal_bin = np.argmax(variance)
    otsu_threshold = float(
        bin_centers[optimal_bin] / 255.0 * (peak_heights.max() - peak_heights.min())
        + peak_heights.min()
    )

    # Try valley fit (exponential + Gaussian mixture)
    valley = _valley_threshold(peak_heights)
    return valley if valley is not None else otsu_threshold


def _valley_threshold(peak_heights: np.ndarray) -> Optional[float]:
    """Find valley between noise exponential and signal Gaussian in peak height distribution."""
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        return None

    p95 = np.percentile(peak_heights, 99)
    h = peak_heights[peak_heights <= p95]
    if len(h) < 10:
        return None

    counts, bin_edges = np.histogram(h, bins=200, range=(h.min(), p95))
    bc = (bin_edges[:-1] + bin_edges[1:]) / 2
    bw = bin_edges[1] - bin_edges[0]
    density = counts / (len(h) * bw)

    # 2-Gaussian model
    def model_2g(x, lam, mu1, s1, mu2, s2, we, wg1):
        g1 = (1 / (np.sqrt(2 * np.pi) * s1)) * np.exp(-0.5 * ((x - mu1) / s1) ** 2)
        g2 = (1 / (np.sqrt(2 * np.pi) * s2)) * np.exp(-0.5 * ((x - mu2) / s2) ** 2)
        wg2 = max(0, 1 - we - wg1)
        return we * lam * np.exp(-lam * x) + wg1 * g1 + wg2 * g2

    try:
        p0 = [
            1.0 / np.mean(bc),
            np.percentile(bc, 40), np.std(bc) * 0.3,
            np.percentile(bc, 70), np.std(bc) * 0.3,
            0.3, 0.4,
        ]
        popt, _ = curve_fit(model_2g, bc, density, p0=p0, maxfev=5000)
        fitted = model_2g(bc, *popt)
        ss_res = np.sum((density - fitted) ** 2)
        ss_tot = np.sum((density - np.mean(density)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 >= 0.7 and popt[5] >= 0.05 and popt[6] >= 0.05:
            end = np.argmin(np.abs(bc - popt[1]))
            return float(bc[np.argmin(fitted[:end])])
    except Exception:
        pass

    # 1-Gaussian fallback
    def model_1g(x, lam, mu, s, we):
        g = (1 / (np.sqrt(2 * np.pi) * s)) * np.exp(-0.5 * ((x - mu) / s) ** 2)
        return we * lam * np.exp(-lam * x) + (1 - we) * g

    try:
        p0 = [1.0 / np.mean(bc), np.median(bc), np.std(bc), 0.3]
        popt, _ = curve_fit(model_1g, bc, density, p0=p0, maxfev=5000)
        fitted = model_1g(bc, *popt)
        ss_res = np.sum((density - fitted) ** 2)
        ss_tot = np.sum((density - np.mean(density)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        if r2 >= 0.7:
            end = np.argmin(np.abs(bc - popt[1]))
            return float(bc[np.argmin(fitted[:end])])
    except Exception:
        pass

    return None


# ── File loading helper ──────────────────────────────────────────


def load_data_file(
    path: Path,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Load a recording file and return its contents.

    Supports ABF, SMRX, EDF, MAT formats.

    Returns:
        Dict with keys: sr_hz, sweeps, channel_names, t, metadata
    """
    ext = path.suffix.lower()

    if ext == ".abf":
        from core.abf_io import load_abf
        sr, sweeps, ch_names, t, meta = load_abf(path, progress_callback=progress_callback)
    elif ext == ".smrx":
        from core.io.son64_loader import load_son64
        sr, sweeps, ch_names, t = load_son64(str(path), progress_callback=progress_callback)
        meta = {"file_type": "smrx"}
    elif ext == ".edf":
        from core.io.edf_loader import load_edf
        sr, sweeps, ch_names, t = load_edf(path, progress_callback=progress_callback)
        meta = {"file_type": "edf"}
    elif ext == ".mat":
        from core.io.mat_loader import load_mat
        sr, sweeps, ch_names, t = load_mat(str(path), progress_callback=progress_callback)
        meta = {"file_type": "mat"}
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    return {
        "sr_hz": sr,
        "sweeps": sweeps,
        "channel_names": ch_names,
        "t": t,
        "metadata": meta,
    }


# ── Summary statistics ────────────────────────────────────────────


def _safe_float(v) -> Optional[float]:
    """Convert to float, returning None for non-numeric or NaN values."""
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _compute_summary(
    metrics_rows: List[Dict[str, Any]],
    n_sweeps: int,
) -> Dict[str, Any]:
    """Compute one-row summary from per-breath metrics.

    Returns a dict with mean_freq, mean_ti, mean_te, mean_amp, n_breaths,
    n_sweeps, and other aggregate stats useful for grouping.
    """
    summary: Dict[str, Any] = {
        "n_breaths": len(metrics_rows),
        "n_sweeps": n_sweeps,
    }

    if not metrics_rows:
        return summary

    # Collect numeric arrays for common metrics (single pass per key)
    metric_keys = ["if", "ti", "te", "amp_insp", "amp_exp", "area_insp", "area_exp"]
    for key in metric_keys:
        values = []
        for r in metrics_rows:
            f = _safe_float(r.get(key))
            if f is not None:
                values.append(f)
        if values:
            arr = np.array(values)
            summary[f"mean_{key}"] = float(np.mean(arr))
            summary[f"std_{key}"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            summary[f"median_{key}"] = float(np.median(arr))

    # Frequency is 1/if for instantaneous frequency → derive mean_freq
    if "mean_if" in summary and summary["mean_if"] > 0:
        summary["mean_freq"] = 1.0 / summary["mean_if"]

    return summary


# ── Batch analysis ───────────────────────────────────────────────


def analyze_file(
    path: Path,
    config: AnalysisConfig,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    write_csv: bool = True,
    save_session: bool = False,
    analyze_channel: Optional[str] = None,
    animal_id: str = "",
    existing_event_markers: Optional[Dict] = None,
) -> AnalysisResult:
    """Headless analysis of a single recording file.

    Pipeline:
    1. Load file
    2. Auto-detect channels
    3. Compute z-score normalisation stats
    4. Auto-detect threshold (if not in config)
    5. Filter + detect peaks per sweep
    6. Compute per-breath metrics
    7. Save results CSV and/or .pmx session

    Args:
        path: Path to recording file (ABF, SMRX, EDF, MAT)
        config: Full analysis configuration
        output_dir: Where to write results. Defaults to same folder as input.
        progress_callback: Optional status message callback
        write_csv: If False, skip CSV writing (dry run). Still returns full metrics.
        save_session: If True, save .pmx session file in physiometrics/ subfolder.
        analyze_channel: If provided (e.g. "IN 0"), use that channel instead of auto-detect.
        animal_id: Animal ID for .pmx file naming and metadata.
        existing_event_markers: Event markers to carry over on re-analysis.

    Returns:
        AnalysisResult with paths to output files.
    """
    import core.metrics as metrics_mod

    result = AnalysisResult(file_path=path)

    if output_dir is None:
        output_dir = path.parent

    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    try:
        # 1. Load
        _log(f"Loading {path.name}...")
        data = load_data_file(path)
        sr_hz = data["sr_hz"]
        sweeps = data["sweeps"]
        ch_names = data["channel_names"]
        t = data["t"]

        # 2. Select analysis channel + detect stim
        from core.abf_io import auto_select_channels
        stim_ch, auto_analyze_ch = auto_select_channels(sweeps, ch_names)

        if analyze_channel and analyze_channel in ch_names:
            analyze_ch = analyze_channel
        elif analyze_channel and analyze_channel not in ch_names:
            result.error = f"Channel '{analyze_channel}' not found in {ch_names}"
            return result
        else:
            analyze_ch = auto_analyze_ch
            if analyze_ch is None:
                non_stim = [c for c in ch_names if c != stim_ch]
                analyze_ch = non_stim[0] if non_stim else ch_names[0]

        Y = sweeps[analyze_ch]  # (n_samples, n_sweeps)
        n_sweeps = Y.shape[1]
        result.n_sweeps = n_sweeps

        # 3. Z-score normalisation stats
        fc = config.filter
        if fc.use_zscore and fc.zscore_global_mean is None:
            _log("Computing normalisation stats...")
            mean, std = compute_normalization_stats(Y, sr_hz, fc)
            fc = FilterConfig(**{**fc.__dict__, "zscore_global_mean": mean, "zscore_global_std": std})

        # 4. Auto-detect threshold if not provided
        pc = config.peak
        if pc.height_threshold is None or pc.prominence is None:
            _log("Auto-detecting threshold...")
            # Process first sweep (or concatenate a few) for threshold detection
            n_sample_sweeps = min(3, n_sweeps)
            sample_signals = []
            for i in range(n_sample_sweeps):
                sample_signals.append(get_processed_signal(Y[:, i], sr_hz, fc))
            y_sample = np.concatenate(sample_signals)
            threshold = auto_detect_threshold(y_sample, sr_hz, pc.min_dist_sec)
            if threshold is None:
                result.error = "Could not auto-detect threshold"
                return result
            pc = PeakDetectionConfig(**{
                **pc.__dict__,
                "height_threshold": threshold,
                "prominence": threshold,
            })

        # 5. Detect peaks per sweep
        _log(f"Detecting peaks across {n_sweeps} sweeps...")
        all_results = {}
        all_metrics_rows = []

        for s in range(n_sweeps):
            y_proc = get_processed_signal(Y[:, s], sr_hz, fc)
            det = detect_single_sweep(s, y_proc, t, sr_hz, pc)
            all_results[s] = det

            labeled_idx = det["labeled_indices"]
            labeled_br = det["labeled_breaths"]
            result.n_peaks_total += len(det["all_peak_indices"])
            result.n_breaths_total += len(labeled_idx)

            # 6. Compute per-breath metrics for this sweep
            if len(labeled_idx) > 0:
                onsets = labeled_br.get("onsets", np.array([]))
                offsets = labeled_br.get("offsets", np.array([]))
                expmins = labeled_br.get("expmins", np.array([]))
                expoffs = labeled_br.get("expoffs", np.array([]))

                for i, pk in enumerate(labeled_idx):
                    row = {"file": path.name, "sweep": s, "peak_sample": int(pk)}
                    if i < len(onsets):
                        row["onset_sample"] = int(onsets[i])
                    if i < len(offsets):
                        row["offset_sample"] = int(offsets[i])
                    # Add time values
                    row["peak_time"] = float(t[pk]) if pk < len(t) else None

                    # Per-breath metrics from current_metrics
                    cm = det["current_metrics"]
                    if cm and i < len(cm):
                        for k, v in cm[i].items():
                            if isinstance(v, (int, float, np.integer, np.floating)):
                                row[k] = float(v)

                    all_metrics_rows.append(row)

        # 7. Compute continuous y2 metrics per sweep (for grouping/consolidation)
        y2_by_sweep = {}  # sweep_idx -> {metric_key: 1D array}
        _Y2_METRICS = {
            'if': 'compute_if',
            'ti': 'compute_ti',
            'te': 'compute_te',
            'amp_insp': 'compute_amp_insp',
            'amp_exp': 'compute_amp_exp',
            'area_insp': 'compute_area_insp',
            'area_exp': 'compute_area_exp',
            'vent_proxy': 'compute_vent_proxy',
        }
        try:
            from core import metrics as _metrics_mod
            for s in range(n_sweeps):
                det = all_results[s]
                labeled_idx = det["labeled_indices"]
                if len(labeled_idx) < 2:
                    continue
                labeled_br = det.get("labeled_breaths", {})
                onsets = labeled_br.get("onsets", np.array([]))
                offsets = labeled_br.get("offsets", np.array([]))
                expmins = labeled_br.get("expmins", np.array([]))
                expoffs = labeled_br.get("expoffs", np.array([]))
                if len(onsets) < 2:
                    continue
                y_proc = get_processed_signal(Y[:, s], sr_hz, fc)
                sweep_y2 = {}
                for key, fn_name in _Y2_METRICS.items():
                    fn = getattr(_metrics_mod, fn_name)
                    sweep_y2[key] = fn(t, y_proc, sr_hz, labeled_idx, onsets, offsets, expmins, expoffs)
                y2_by_sweep[s] = sweep_y2
        except Exception as e:
            import traceback
            _log(f"Warning: y2 metric computation failed: {e}\n{traceback.format_exc()}")

        # 8. Compute summary statistics
        summary = _compute_summary(all_metrics_rows, n_sweeps)
        result.summary = summary

        # 9. Save .pmx session file
        if save_session:
            from core.npz_io import get_pmx_path, save_batch_result

            pmx_path = get_pmx_path(
                path,
                analysis_type="pleth",
                animal_id=animal_id,
                channel=analyze_ch,
            )
            _log(f"Saving session to {pmx_path.name}...")
            save_batch_result(
                output_path=pmx_path,
                original_file=path,
                channel=analyze_ch,
                sr_hz=sr_hz,
                t=t,
                config=config,
                filter_config=fc,
                detection_results=all_results,
                metrics_rows=all_metrics_rows,
                summary=summary,
                event_markers=existing_event_markers,
                stim_chan=stim_ch if stim_ch else "None",
                y2_by_sweep=y2_by_sweep,
            )
            result.session_path = pmx_path
            _log(f"Saved {pmx_path.name}")

        # 10. Save results CSV (skip in dry-run mode)
        if all_metrics_rows and write_csv:
            import csv
            csv_path = output_dir / f"{path.stem}_results.csv"
            fieldnames = list(all_metrics_rows[0].keys())
            # Collect all keys across all rows
            for row in all_metrics_rows:
                for k in row:
                    if k not in fieldnames:
                        fieldnames.append(k)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(all_metrics_rows)

            result.results_path = csv_path
            _log(f"Wrote {len(all_metrics_rows)} breath rows to {csv_path.name}")

        return result

    except Exception as e:
        import traceback
        result.error = f"{type(e).__name__}: {e}"
        result.warnings.append(traceback.format_exc())
        return result


def analyze_folder(
    folder: Path,
    config: AnalysisConfig,
    pattern: str = "*.abf",
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[AnalysisResult]:
    """Batch-analyse all matching files in a folder.

    Args:
        folder: Directory containing recording files
        config: Analysis configuration (applied to all files)
        pattern: Glob pattern for file matching
        output_dir: Output directory (defaults to folder)
        progress_callback: Optional (current, total, message) callback

    Returns:
        List of AnalysisResult, one per file.
    """
    files = sorted(folder.glob(pattern))
    if not files:
        return []

    if output_dir is None:
        output_dir = folder

    results = []
    for i, f in enumerate(files):
        def _file_progress(msg):
            if progress_callback:
                progress_callback(i + 1, len(files), msg)

        _file_progress(f"Analysing {f.name}...")
        r = analyze_file(f, config, output_dir, _file_progress)
        results.append(r)

    return results
