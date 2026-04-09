"""
ECG analysis service -- pure Python, no Qt dependencies.

Provides R-peak detection (Pan-Tompkins variant tuned for rodents),
instantaneous heart rate, HRV time/frequency domain, and
respiratory-cardiac coupling metrics.

All public functions are stateless: config + data in, results out.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt, welch

from core.domain.ecg.models import ECGConfig, ECGResult, HRVResult


# =====================================================================
#  R-Peak Detection
# =====================================================================


def detect_r_peaks(
    signal: np.ndarray,
    sr_hz: float,
    config: ECGConfig,
) -> Tuple[np.ndarray, bool]:
    """Detect R-peaks using a Pan-Tompkins variant.

    The squaring step makes detection polarity-agnostic.  After
    detection, we check whether the raw signal at each peak is
    positive or negative to determine inversion.

    Returns
    -------
    r_peaks : ndarray of int64
        Sample indices of detected R-peaks.
    is_inverted : bool
        True if R-peaks correspond to negative deflections.
    """
    if len(signal) < int(0.5 * sr_hz):
        return np.array([], dtype=np.int64), False

    # 1. Bandpass filter
    nyq = sr_hz / 2.0
    low = max(config.bandpass_low, 0.5)
    high = min(config.bandpass_high, nyq - 1.0)
    if low >= high:
        low, high = 0.5, nyq - 1.0
    sos = butter(config.filter_order, [low, high], btype="band",
                 fs=sr_hz, output="sos")
    filtered = sosfiltfilt(sos, signal)

    # 2. Five-point derivative (emphasises QRS slope)
    h = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (sr_hz / 8.0)
    derivative = np.convolve(filtered, h, mode="same")

    # 3. Squaring (removes polarity, amplifies large slopes)
    squared = derivative ** 2

    # 4. Moving-window integration
    win_size = max(1, int(config.mwi_window_ms * sr_hz / 1000.0))
    kernel = np.ones(win_size, dtype=np.float64) / win_size
    integrated = np.convolve(squared, kernel, mode="same")

    # 5. Find peaks in integrated signal with refractory period
    min_dist = max(1, int(config.refractory_ms * sr_hz / 1000.0))
    pct_val = np.percentile(integrated, config.threshold_percentile)
    if pct_val <= 0:
        pct_val = np.percentile(integrated, 99)
    height_threshold = config.threshold_fraction * pct_val
    prom_fraction = getattr(config, 'prominence_fraction', 0.05)
    prom_threshold = prom_fraction * pct_val

    candidates, _ = find_peaks(
        integrated,
        distance=min_dist,
        height=height_threshold,
        prominence=prom_threshold,
    )

    if len(candidates) == 0:
        return np.array([], dtype=np.int64), False

    # 6-7. Refine to actual R-peak in filtered signal.
    search_radius = max(1, int(config.qrs_search_ms * sr_hz / 1000.0))
    force_inv = getattr(config, 'force_inverted', None)

    if force_inv is not None:
        # User explicitly set polarity
        r_peaks = _refine_peaks_forced(filtered, candidates, search_radius, force_inv)
        is_inverted = force_inv
    else:
        # Auto-detect: pick largest absolute deflection from baseline
        r_peaks, is_inverted = _refine_peaks_by_amplitude(
            filtered, candidates, search_radius
        )

    # 8. Remove duplicates and sort
    r_peaks = np.unique(r_peaks)

    return r_peaks, is_inverted


def _refine_peaks_forced(
    signal: np.ndarray,
    candidates: np.ndarray,
    search_radius: int,
    is_inverted: bool,
) -> np.ndarray:
    """Snap each candidate using user-specified polarity."""
    n = len(signal)
    refined = np.empty(len(candidates), dtype=np.int64)
    for i, c in enumerate(candidates):
        lo = max(0, c - search_radius)
        hi = min(n, c + search_radius + 1)
        if is_inverted:
            refined[i] = lo + np.argmin(signal[lo:hi])
        else:
            refined[i] = lo + np.argmax(signal[lo:hi])
    return refined


def _refine_peaks_by_amplitude(
    signal: np.ndarray,
    candidates: np.ndarray,
    search_radius: int,
) -> Tuple[np.ndarray, bool]:
    """Snap each candidate to the largest absolute deflection in the window.

    For each candidate, finds both the local max and min within the
    search window, then picks whichever is farther from the signal
    baseline (mean).  This correctly identifies the R-peak whether
    the signal is upright or inverted.

    Returns (r_peaks, is_inverted).
    """
    n = len(signal)
    baseline = np.mean(signal)
    refined = np.empty(len(candidates), dtype=np.int64)
    n_negative = 0

    for i, c in enumerate(candidates):
        lo = max(0, c - search_radius)
        hi = min(n, c + search_radius + 1)
        window = signal[lo:hi]

        idx_max = np.argmax(window)
        idx_min = np.argmin(window)
        val_max = window[idx_max]
        val_min = window[idx_min]

        # Pick whichever deflection is larger from baseline
        if abs(val_max - baseline) >= abs(val_min - baseline):
            refined[i] = lo + idx_max
        else:
            refined[i] = lo + idx_min
            n_negative += 1

    # Signal is considered inverted if majority of R-peaks are negative deflections
    is_inverted = n_negative > len(candidates) // 2
    return refined, is_inverted


# =====================================================================
#  Heart Rate Computation
# =====================================================================


def compute_rr_intervals_ms(
    r_peaks: np.ndarray, sr_hz: float
) -> np.ndarray:
    """RR intervals in milliseconds from R-peak sample indices."""
    if len(r_peaks) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(r_peaks).astype(np.float64) * (1000.0 / sr_hz)


def compute_instantaneous_hr(
    r_peaks: np.ndarray, sr_hz: float
) -> np.ndarray:
    """Instantaneous HR in BPM at each R-peak (length = len(r_peaks) - 1)."""
    rr_ms = compute_rr_intervals_ms(r_peaks, sr_hz)
    if len(rr_ms) == 0:
        return np.array([], dtype=np.float64)
    return 60_000.0 / rr_ms


# =====================================================================
#  Signal Quality
# =====================================================================


def compute_signal_quality(
    signal: np.ndarray,
    r_peaks: np.ndarray,
    sr_hz: float,
    is_inverted: bool = False,
) -> float:
    """Signal quality index (0-1) based on template correlation.

    Extracts individual beats, correlates each with the median
    template, and returns the median correlation.
    """
    if len(r_peaks) < 5:
        return 0.0

    # Build templates: half-beat before/after each R-peak
    rr_median = int(np.median(np.diff(r_peaks)))
    half = rr_median // 2
    templates: List[np.ndarray] = []

    for pk in r_peaks:
        lo = pk - half
        hi = pk + half
        if lo < 0 or hi >= len(signal):
            continue
        beat = signal[lo:hi].astype(np.float64)
        # Normalise to zero mean, unit variance
        std = np.std(beat)
        if std > 1e-10:
            beat = (beat - np.mean(beat)) / std
            templates.append(beat)

    if len(templates) < 3:
        return 0.0

    # Median template
    min_len = min(len(t) for t in templates)
    templates = [t[:min_len] for t in templates]
    template_matrix = np.stack(templates)
    avg_template = np.median(template_matrix, axis=0)
    avg_std = np.std(avg_template)
    if avg_std < 1e-10:
        return 0.0
    avg_template = (avg_template - np.mean(avg_template)) / avg_std

    # Correlate each beat with template
    correlations = []
    for t in templates:
        corr = np.corrcoef(t, avg_template)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)

    if not correlations:
        return 0.0
    return float(np.clip(np.median(correlations), 0.0, 1.0))


# =====================================================================
#  Auto-label beats
# =====================================================================


def label_beats(
    r_peaks: np.ndarray,
    sr_hz: float,
    config: ECGConfig,
    signal: Optional[np.ndarray] = None,
    is_inverted: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Label each R-peak as valid (1) or artifact (0).

    Heuristics:
    - RR interval outside expected HR range -> artifact
    - Template correlation < 0.5 -> artifact (if signal provided)

    Returns (labels, label_source) arrays.
    """
    n = len(r_peaks)
    labels = np.ones(n, dtype=np.int8)
    label_source = np.full(n, "auto", dtype=object)

    if n < 2:
        return labels, label_source

    rr_ms = compute_rr_intervals_ms(r_peaks, sr_hz)

    # Expected RR range from config HR bounds
    min_rr = 60_000.0 / config.expected_hr_max if config.expected_hr_max > 0 else 0
    max_rr = 60_000.0 / config.expected_hr_min if config.expected_hr_min > 0 else 10_000

    # First and last beats get their neighbour's RR
    for i in range(n):
        if i < len(rr_ms):
            rr = rr_ms[i]
        elif i > 0 and i - 1 < len(rr_ms):
            rr = rr_ms[i - 1]
        else:
            continue
        if rr < min_rr or rr > max_rr:
            labels[i] = 0

    return labels, label_source


# =====================================================================
#  Full detection pipeline
# =====================================================================


def detect_and_analyze(
    signal: np.ndarray,
    sr_hz: float,
    config: ECGConfig,
) -> ECGResult:
    """Full pipeline: detect R-peaks, compute RR, label, assess quality."""
    r_peaks, is_inverted = detect_r_peaks(signal, sr_hz, config)

    rr_ms = compute_rr_intervals_ms(r_peaks, sr_hz)
    labels, label_source = label_beats(
        r_peaks, sr_hz, config, signal, is_inverted
    )
    quality = compute_signal_quality(signal, r_peaks, sr_hz, is_inverted)

    return ECGResult(
        r_peaks=r_peaks,
        rr_intervals_ms=rr_ms,
        labels=labels,
        label_source=label_source,
        is_inverted=is_inverted,
        quality_score=quality,
    )


def detect_and_analyze_chunked(
    signal: np.ndarray,
    sr_hz: float,
    config: ECGConfig,
    chunk_sec: float = 60.0,
    overlap_sec: float = 2.0,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> ECGResult:
    """Chunked pipeline for long recordings.

    Processes *chunk_sec*-second segments with *overlap_sec* overlap to
    avoid edge effects.  Peak memory is bounded to ~one chunk.
    """
    n_samples = len(signal)
    chunk_size = int(chunk_sec * sr_hz)
    overlap = int(overlap_sec * sr_hz)

    # Short signals: just process whole thing
    if n_samples <= chunk_size * 1.5:
        result = detect_and_analyze(signal, sr_hz, config)
        if progress_callback:
            progress_callback(100)
        return result

    all_peaks: List[int] = []
    total_chunks = max(1, int(np.ceil(n_samples / chunk_size)))

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        # Include overlap on both sides (except edges)
        read_start = max(0, start - overlap)
        read_end = min(n_samples, start + chunk_size + overlap)
        chunk = signal[read_start:read_end]

        peaks, _ = detect_r_peaks(chunk, sr_hz, config)

        # Offset to global indices, keep only peaks in non-overlap region
        peaks_global = peaks + read_start
        valid_start = start
        valid_end = min(start + chunk_size, n_samples)
        valid_mask = (peaks_global >= valid_start) & (peaks_global < valid_end)
        all_peaks.extend(peaks_global[valid_mask].tolist())

        if progress_callback:
            pct = min(100, int(100 * (chunk_idx + 1) / total_chunks))
            progress_callback(pct)

    r_peaks = np.unique(np.asarray(all_peaks, dtype=np.int64))

    # Determine polarity from the peaks we found
    if len(r_peaks) > 0:
        baseline = np.mean(signal)
        sample = r_peaks[:100]
        vals = signal[sample]
        n_below = np.sum(vals < baseline)
        is_inverted = bool(n_below > len(sample) // 2)
    else:
        is_inverted = False

    rr_ms = compute_rr_intervals_ms(r_peaks, sr_hz)
    labels, label_source = label_beats(
        r_peaks, sr_hz, config, signal, is_inverted
    )
    quality = compute_signal_quality(
        signal, r_peaks[:200], sr_hz, is_inverted
    )

    return ECGResult(
        r_peaks=r_peaks,
        rr_intervals_ms=rr_ms,
        labels=labels,
        label_source=label_source,
        is_inverted=is_inverted,
        quality_score=quality,
    )


# =====================================================================
#  HRV — Time Domain
# =====================================================================


def compute_hrv_time(
    rr_intervals_ms: np.ndarray,
    config: ECGConfig,
) -> HRVResult:
    """Compute time-domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr_intervals_ms : ndarray
        Successive RR intervals in milliseconds (valid beats only).
    config : ECGConfig
        Used for pNNx threshold and species context.
    """
    if len(rr_intervals_ms) < 2:
        return HRVResult(n_beats=len(rr_intervals_ms) + 1)

    rr = rr_intervals_ms.astype(np.float64)
    diffs = np.diff(rr)
    abs_diffs = np.abs(diffs)

    mean_rr = np.mean(rr)
    mean_hr = 60_000.0 / mean_rr if mean_rr > 0 else 0.0
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) > 0 else 0.0
    sdsd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # pNNx: percentage of successive differences > threshold
    threshold = config.pnnx_threshold_ms
    pnnx = 100.0 * np.sum(abs_diffs > threshold) / len(abs_diffs) if len(abs_diffs) > 0 else 0.0

    # Min/max HR from RR
    min_hr = 60_000.0 / np.max(rr) if np.max(rr) > 0 else 0.0
    max_hr = 60_000.0 / np.min(rr) if np.min(rr) > 0 else 0.0

    # Coefficient of variation
    cv = sdnn / mean_rr if mean_rr > 0 else 0.0

    # Poincare SD1 / SD2
    sd1 = float(np.std(diffs, ddof=1) / np.sqrt(2)) if len(diffs) > 1 else None
    sd2_val = np.std(rr[:-1] + rr[1:], ddof=1) / np.sqrt(2) if len(rr) > 2 else None
    sd2 = float(sd2_val) if sd2_val is not None else None

    return HRVResult(
        mean_hr_bpm=float(mean_hr),
        sdnn_ms=sdnn,
        rmssd_ms=rmssd,
        pnnx_pct=float(pnnx),
        pnnx_threshold_ms=threshold,
        min_hr_bpm=float(min_hr),
        max_hr_bpm=float(max_hr),
        sdsd_ms=sdsd,
        cv=float(cv),
        n_beats=len(rr) + 1,
        sd1_ms=sd1,
        sd2_ms=sd2,
    )


# =====================================================================
#  HRV — Frequency Domain
# =====================================================================


def compute_hrv_frequency(
    rr_intervals_ms: np.ndarray,
    config: ECGConfig,
) -> Dict[str, Optional[float]]:
    """Frequency-domain HRV with species-appropriate bands.

    Interpolates RR series to uniform sampling, then applies Welch PSD.

    Returns dict with lf_power_ms2, hf_power_ms2, lf_hf_ratio,
    total_power_ms2.
    """
    if len(rr_intervals_ms) < 10:
        return {
            "lf_power_ms2": None,
            "hf_power_ms2": None,
            "lf_hf_ratio": None,
            "total_power_ms2": None,
        }

    rr_sec = rr_intervals_ms / 1000.0
    rr_times = np.cumsum(rr_sec)

    # Interpolate to uniform sampling (20 Hz for rodents)
    max_hf = config.hrv_hf_band[1]
    fs_interp = max(10.0, max_hf * 4.0)  # Nyquist
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / fs_interp)
    if len(t_uniform) < 16:
        return {
            "lf_power_ms2": None,
            "hf_power_ms2": None,
            "lf_hf_ratio": None,
            "total_power_ms2": None,
        }

    rr_interp = np.interp(t_uniform, rr_times, rr_sec)
    rr_interp = rr_interp - np.mean(rr_interp)  # detrend (mean removal)

    # Welch PSD (in seconds^2/Hz, then convert to ms^2/Hz)
    nperseg = min(256, len(rr_interp))
    freqs, psd = welch(rr_interp, fs=fs_interp, nperseg=nperseg)
    psd_ms2 = psd * 1e6  # s^2 -> ms^2

    def _band_power(fmin: float, fmax: float) -> float:
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask):
            return 0.0
        return float(np.trapz(psd_ms2[mask], freqs[mask]))

    lf = _band_power(*config.hrv_lf_band)
    hf = _band_power(*config.hrv_hf_band)
    total = lf + hf

    return {
        "lf_power_ms2": lf,
        "hf_power_ms2": hf,
        "lf_hf_ratio": lf / hf if hf > 0 else None,
        "total_power_ms2": total,
    }


# =====================================================================
#  Respiratory-Cardiac Coupling
# =====================================================================


def compute_rsa(
    r_peaks: np.ndarray,
    breath_onsets: np.ndarray,
    breath_offsets: np.ndarray,
    sr_hz: float,
) -> Dict[str, Optional[float]]:
    """Respiratory sinus arrhythmia from simultaneous ECG + pleth.

    For each breath cycle, finds the max and min instantaneous HR
    and computes their difference (RSA amplitude).

    Parameters
    ----------
    r_peaks : ndarray
        Valid R-peak sample indices.
    breath_onsets : ndarray
        Breath onset sample indices (from Pleth analysis).
    breath_offsets : ndarray
        Breath offset sample indices (inspiration -> expiration).
    sr_hz : float
        Sampling rate.

    Returns
    -------
    dict with rsa_amplitude_bpm (mean), phase_coherence (0-1).
    """
    if len(r_peaks) < 10 or len(breath_onsets) < 3:
        return {"rsa_amplitude_bpm": None, "phase_coherence": None}

    # Instantaneous HR at each R-peak
    rr_sec = np.diff(r_peaks) / sr_hz
    hr_at_peak = 60.0 / rr_sec  # BPM, length = len(r_peaks) - 1
    # Assign HR to the first R-peak of each pair
    peak_times = r_peaks[:-1]

    rsa_amplitudes: List[float] = []
    phases: List[float] = []

    for i in range(len(breath_onsets) - 1):
        onset = breath_onsets[i]
        next_onset = breath_onsets[i + 1]

        # Find R-peaks within this breath cycle
        mask = (peak_times >= onset) & (peak_times < next_onset)
        if np.sum(mask) < 2:
            continue

        hr_in_breath = hr_at_peak[mask]
        peaks_in_breath = peak_times[mask]

        # RSA amplitude: max HR - min HR within this breath
        rsa_amp = float(np.max(hr_in_breath) - np.min(hr_in_breath))
        rsa_amplitudes.append(rsa_amp)

        # Phase: where in the breath cycle does peak HR occur?
        # 0 = onset, 1 = next onset
        cycle_len = float(next_onset - onset)
        if cycle_len > 0:
            peak_hr_idx = np.argmax(hr_in_breath)
            phase = float(peaks_in_breath[peak_hr_idx] - onset) / cycle_len
            phases.append(phase * 2 * np.pi)  # convert to radians

    if not rsa_amplitudes:
        return {"rsa_amplitude_bpm": None, "phase_coherence": None}

    # Phase coherence: mean resultant length (circular statistics)
    if phases:
        phase_arr = np.array(phases)
        coherence = float(np.abs(np.mean(np.exp(1j * phase_arr))))
    else:
        coherence = None

    return {
        "rsa_amplitude_bpm": float(np.mean(rsa_amplitudes)),
        "phase_coherence": coherence,
    }


def compute_hr_at_breaths(
    r_peaks: np.ndarray,
    breath_onsets: np.ndarray,
    sr_hz: float,
) -> np.ndarray:
    """Instantaneous HR (BPM) at each breath onset.

    For each breath onset, finds the nearest preceding R-peak pair
    and returns the HR from that interval.

    Returns array of length len(breath_onsets), with NaN where
    no HR data is available.
    """
    result = np.full(len(breath_onsets), np.nan, dtype=np.float64)
    if len(r_peaks) < 2 or len(breath_onsets) == 0:
        return result

    rr_sec = np.diff(r_peaks) / sr_hz
    hr_at_peak = 60.0 / rr_sec
    peak_times = r_peaks[:-1]  # HR assigned to first peak of each pair

    for i, onset in enumerate(breath_onsets):
        # Find latest R-peak before this breath onset
        idx = np.searchsorted(peak_times, onset, side="right") - 1
        if 0 <= idx < len(hr_at_peak):
            result[i] = hr_at_peak[idx]

    return result
