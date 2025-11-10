# core/metrics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional

"""
Scaffold for y2 metrics derived from breath features.

Each compute_* function must return an array of shape (N,) where N == len(y),
ideally stepwise-constant over breath spans (NaN where undefined).
Inputs:
  - t:        time (s), shape (N,)
  - y:        processed pleth, shape (N,)
  - sr_hz:    sample rate (Hz)
  - peaks:    inspiratory peaks (np.int indices)
  - onsets:   breath onset indices (np.int)
  - offsets:  breath offset indices (np.int)
  - expmins:  expiratory minima indices (np.int)
"""

# Module-level storage for GMM probabilities (set by main window before computing metrics)
_current_gmm_probabilities: Optional[Dict[int, float]] = None

# Module-level storage for normalization statistics (for relative metrics)
# Dict keys: 'amp_insp', 'amp_exp', 'peak_to_trough', 'prominence', 'ibi', 'ti', 'te'
# Dict values: {'mean': float, 'std': float}
_normalization_stats: Optional[Dict[str, Dict[str, float]]] = None

# Module-level storage for auto-threshold model parameters (exp + 2 Gaussians)
# Used to compute P(noise) and P(breath) for each peak
_threshold_model_params: Optional[Dict[str, float]] = None


def set_gmm_probabilities(probs: Optional[Dict[int, float]]):
    """
    Set GMM probabilities for the current sweep.

    Args:
        probs: Dict mapping breath_idx -> sniffing probability (0.0 to 1.0)
               None to clear probabilities
    """
    global _current_gmm_probabilities
    _current_gmm_probabilities = probs


def set_normalization_stats(stats: Optional[Dict[str, Dict[str, float]]]):
    """
    Set normalization statistics for computing relative metrics.

    This should be called by the main window after peak detection, providing
    global mean and std for all detected peaks in the sweep.

    Args:
        stats: Dict with keys like 'amp_insp', 'prominence', etc.
               Each value is a dict with 'mean' and 'std' keys.
               None to clear statistics.

    Example:
        stats = {
            'amp_insp': {'mean': 0.5, 'std': 0.15},
            'prominence': {'mean': 0.3, 'std': 0.1},
            ...
        }
    """
    global _normalization_stats
    _normalization_stats = stats


def set_threshold_model_params(params: Optional[Dict[str, float]]):
    """
    Set auto-threshold model parameters (exponential + 2 Gaussians).

    This should be called by the auto-threshold dialog after fitting the model.

    Args:
        params: Dict with model parameters:
            - 'lambda_exp': Exponential decay rate
            - 'mu1', 'sigma1', 'w_g1': First Gaussian (eupnea)
            - 'mu2', 'sigma2', 'w_g2': Second Gaussian (sniffing)
            - 'w_exp': Weight of exponential component
            None to clear model parameters.

    Example:
        params = {
            'lambda_exp': 5.0,
            'mu1': 0.3, 'sigma1': 0.1, 'w_g1': 0.4,
            'mu2': 0.6, 'sigma2': 0.15, 'w_g2': 0.3,
            'w_exp': 0.3
        }
    """
    global _threshold_model_params
    _threshold_model_params = params

# (human label, key)
METRIC_SPECS: List[Tuple[str, str]] = [
    ("Instantaneous frequency (Hz)",          "if"),
    ("Inspiratory amplitude",                 "amp_insp"),
    ("Expiratory amplitude",                  "amp_exp"),
    ("Inspiratory area",                      "area_insp"),
    ("Expiratory area",                       "area_exp"),
    ("Ti (on→off)",                           "ti"),
    ("Te (off→next on)",                      "te"),
    ("Vent proxy (insp area / cycle dur)",    "vent_proxy"),
    ("d/dt (1st derivative)",                 "d1"),
    ("d²/dt² (2nd derivative)",               "d2"),
    ("Max inspiratory d/dt (peak insp rate)", "max_dinsp"),
    ("Max expiratory d/dt (peak exp rate)",   "max_dexp"),
    ("Eupnic breathing regions",              "eupnic"),
    ("Apnea detection",                       "apnea"),
    ("Breathing regularity score (RMSSD)",    "regularity"),
    ("Sniffing confidence (GMM)",             "sniff_conf"),
    ("Eupnea confidence (GMM)",               "eupnea_conf"),
    # Phase 2.1: Half-width features for ML
    ("FWHM (full width at half max)",        "fwhm"),
    ("Width at 25% of peak",                  "width_25"),
    ("Width at 75% of peak",                  "width_75"),
    ("Width ratio (75% / 25%)",               "width_ratio"),
    # Phase 2.2: Sigh detection features for ML
    ("Inflection point count",                "n_inflections"),
    ("Rise phase variability (std of d/dt)",  "rise_variability"),
    ("Shoulder peak count",                   "n_shoulder_peaks"),
    ("Shoulder prominence (max)",             "shoulder_prominence"),
    ("Rise phase autocorrelation",            "rise_autocorr"),
    ("Peak sharpness (curvature)",            "peak_sharpness"),
    ("Trough sharpness (curvature)",          "trough_sharpness"),
    ("Skewness (asymmetry)",                  "skewness"),
    ("Kurtosis (peakedness, excess)",         "kurtosis"),
    # Phase 2.3 Group A: Shape & Ratio metrics
    ("Peak-to-trough amplitude",              "peak_to_trough"),
    ("Amplitude ratio (insp/exp)",            "amp_ratio"),
    ("Ti/Te ratio (duty cycle)",              "ti_te_ratio"),
    ("Area ratio (insp/exp)",                 "area_ratio"),
    ("Total area (|insp| + |exp|)",           "total_area"),
    ("IBI (inter-breath interval)",           "ibi"),
    # Phase 2.3 Group B: Normalized metrics (z-scores)
    ("Amp insp (normalized)",                 "amp_insp_norm"),
    ("Amp exp (normalized)",                  "amp_exp_norm"),
    ("Peak-to-trough (normalized)",           "peak_to_trough_norm"),
    ("Prominence (normalized)",               "prominence_norm"),
    ("IBI (normalized)",                      "ibi_norm"),
    ("Ti (normalized)",                       "ti_norm"),
    ("Te (normalized)",                       "te_norm"),
    # Phase 2.3 Group C: Probability from auto-threshold
    ("P(noise) - auto-threshold model",       "p_noise"),
    ("P(breath) - auto-threshold model",      "p_breath"),
]


def _step_fill(n: int, spans: List[Tuple[int, int]], values: List[float]) -> np.ndarray:
    """Fill piecewise-constant values across [i0, i1) spans; NaN elsewhere."""
    out = np.full(n, np.nan, dtype=float)
    for (i0, i1), v in zip(spans, values):
        i0c = max(0, int(i0))
        i1c = min(n, int(i1))
        if i1c > i0c:
            out[i0c:i1c] = v
    return out


def _breath_spans_from_onsets(onsets: np.ndarray, N: int) -> List[Tuple[int, int]]:
    """Onset→next onset spans (last one runs to N)."""
    if onsets is None or len(onsets) == 0:
        return []
    spans: List[Tuple[int, int]] = []
    for i in range(len(onsets) - 1):
        spans.append((int(onsets[i]), int(onsets[i + 1])))
    spans.append((int(onsets[-1]), N))
    return spans


def _spans_from_bounds(bounds: np.ndarray, N: int) -> List[Tuple[int, int]]:
    """
    Convert a sorted array of boundary indices (e.g., onsets or peaks)
    into half-open spans [(b0,b1), (b1,b2), ..., (b_last, N)].
    """
    bounds = np.asarray(bounds, dtype=int)
    if bounds.size == 0:
        return []
    spans: List[Tuple[int, int]] = []
    for i in range(len(bounds) - 1):
        spans.append((int(bounds[i]), int(bounds[i + 1])))
    spans.append((int(bounds[-1]), int(N)))
    return spans


# Add near the other helpers
def _trapz_seg(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    """
    Trapezoid integral of y over inclusive indices [i0, i1].
    Returns NaN if the segment is empty or invalid.
    """
    N = len(y)
    i0c = max(0, min(int(i0), N - 1))
    i1c = max(0, min(int(i1), N - 1))
    if i1c <= i0c:
        return np.nan
    return float(np.trapz(y[i0c:i1c + 1], t[i0c:i1c + 1]))


def _safe_array_access(arr: np.ndarray, index: int, default_value=None):
    """
    Safely access array element with bounds checking.
    Returns default_value (or NaN) if index is out of bounds.
    """
    if arr is None or len(arr) == 0:
        return np.nan if default_value is None else default_value
    if 0 <= index < len(arr):
        return arr[index]
    return np.nan if default_value is None else default_value


def _robust_cycle_bounds(onsets, offsets, peaks, i, N):
    """
    Robustly determine cycle boundaries for index i, handling missing data.
    Returns (cycle_start, cycle_end, has_valid_onset, has_valid_offset).
    """
    # Default cycle boundaries based on available data
    if onsets is not None and len(onsets) > i:
        cycle_start = int(onsets[i])
        has_valid_onset = True
        if i + 1 < len(onsets):
            cycle_end = int(onsets[i + 1])
        else:
            cycle_end = N  # Last cycle extends to end
    elif peaks is not None and len(peaks) > i:
        # Fallback to peak-based boundaries
        pk_curr = int(peaks[i])
        cycle_start = max(0, pk_curr - int(0.5 * 20))  # Assume ~20Hz sampling, 0.5s before peak
        has_valid_onset = False
        if i + 1 < len(peaks):
            pk_next = int(peaks[i + 1])
            cycle_end = pk_next
        else:
            cycle_end = min(N, pk_curr + int(0.5 * 20))  # 0.5s after peak
    else:
        # No usable data
        return 0, N, False, False

    # Check for valid offset
    has_valid_offset = (offsets is not None and len(offsets) > i and
                       cycle_start < int(offsets[i]) <= cycle_end)

    return cycle_start, cycle_end, has_valid_onset, has_valid_offset


# ---------- STUB COMPUTATIONS (return NaNs for now) ----------

################################################################################
#####IF Calculation
################################################################################
#     """
#     Instantaneous frequency as a stepwise signal (breaths/min).
#     Prefer onset→onset cycle times; fall back to peak→peak if onsets unavailable.
#     """
# def compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     # choose cycle boundaries

#     if bounds is None or len(bounds) < 2:
#         return out  # not enough info

#     # compute BPM for each cycle
#     for i in range(len(bounds) - 1):
#         vals.append(bpm)
#     # extend last span with last BPM value
#     vals.append(vals[-1] if len(vals) else np.nan)

#     return _step_fill(N, spans, vals)

def compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None):
    """
    Instantaneous frequency in Hz (not breaths/min).
    Stepwise constant over onset→next-onset spans.
    """
    import numpy as np

    N = len(y)
    out = np.full(N, np.nan, dtype=float)
    if onsets is None or len(onsets) < 2:
        return out

    on = np.asarray(onsets, dtype=int)

    # Reuse your helpers if you have them in metrics.py
    spans = _spans_from_bounds(on, N)

    vals = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])
        dt = float(t[i1] - t[i0])
        vals.append((1.0 / dt) if dt > 0 else np.nan)  # Hz

    # extend last span with last value (same behavior as your other metrics)
    vals.append(vals[-1] if vals else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####INSP AMP Calculation
################################################################################
def compute_amp_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Inspiratory amplitude per cycle: peak - onset.
    We take the FIRST peak inside each onset→next-onset span.
    If offsets are provided and that peak is beyond the offset, mark NaN.
    Returns stepwise-constant array over onset-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])
        # first peak in this cycle
        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        # if offsets exist, ensure peak lies before the offset for this cycle
        if offsets is not None and i < len(offsets):
            off_i = int(offsets[i])
            if p > off_i:
                vals.append(np.nan)
                continue

        vals.append(float(y[p] - y[i0]))

    # extend last span to N with the last valid value (consistent with IF)
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####ExpAmP Calculation
################################################################################
def compute_amp_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Expiratory amplitude per cycle: next_onset - expiratory_minimum_in_cycle.
    Prefer the precomputed expiratory minima; if none inside the cycle,
    fall back to argmin(y) over that cycle window.
    Returns stepwise-constant array over onset-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2:
        return out

    on = np.asarray(onsets, dtype=int)
    em = np.asarray(expmins, dtype=int) if (expmins is not None and len(expmins)) else np.array([], dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        exp_idx = None
        if em.size:
            mask = (em >= i0) & (em < i1)
            if np.any(mask):
                exp_idx = int(em[mask][0])

        # fallback: minimum of y within the cycle if no labeled expmin
        if exp_idx is None:
            seg = y[i0:i1]
            if seg.size == 0:
                vals.append(np.nan)
                continue
            exp_idx = int(i0 + int(np.argmin(seg)))

        # guard
        if exp_idx >= i1:
            vals.append(np.nan)
            continue

        # vals.append(float(y[i1] - y[exp_idx]))
        vals.append(float(y[exp_idx] - y[i1]))


    # extend last span with most recent value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)




################################################################################
#####Insp Area Calculation
################################################################################
def compute_area_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Inspiratory area per cycle: integral of y from onset -> offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If an offset is missing or out of range for a cycle, that cycle is NaN.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 1 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    # Step spans are always onset -> next onset (last extends to N)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # Need a valid offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])

        # Guard: offset must lie within (i0, i1]
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        # Trapezoid integral over [i0, oi] (inclusive of oi via +1 slice end)
        left = max(0, i0)
        right = min(N, oi + 1)
        if right - left < 2:
            vals.append(np.nan)
            continue
        area = float(np.trapz(y[left:right], t[left:right]))
        vals.append(area)

    # Extend last span to N with last valid value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####Exp Area Calculation
################################################################################

#     """
#     Expiratory area per cycle: integral of y from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If offset is missing for a cycle, that cycle is NaN.
#     """
# def compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     for i in range(len(on) - 1):
#     vals: list[float] = []
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#             vals.append(np.nan)
#             continue
#         if i >= len(off):
#         oi = int(off[i])

#         # Guard: offset must lie within [i0, i1)
#             vals.append(np.nan)
#             continue
#         if not (i0 <= oi < i1):

#         # Trapezoid integral over [oi, i1]
#         if right - left < 2:
#             vals.append(np.nan)
#             continue
#         vals.append(area)
#         left = max(0, oi)
#         right = min(N, i1 + 1)
#         area = float(np.trapz(y[left:right], t[left:right]))

#     # Extend last span with last value
#     vals.append(vals[-1] if len(vals) else np.nan)
#     return _step_fill(N, spans, vals)

def compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None, debug=False) -> np.ndarray:
    """
    Expiratory area per cycle: integral of y from inspiratory offset -> expiratory offset.
    Returns a stepwise-constant array over onset→next-onset spans.

    If an expiratory offset for a cycle is missing or invalid, that cycle is NaN.
    If any input values (t, y) contain NaN, returns NaN for that cycle.
    If expiratory minimum is positive (above baseline), returns 0 (no valid expiratory phase).
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on  = np.asarray(onsets,  dtype=int)
    off = np.asarray(offsets, dtype=int)
    exo = np.asarray(expoffs, dtype=int) if (expoffs is not None and len(expoffs)) else np.array([], dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    failure_count = 0

    for i in range(len(on) - 1):
        i0 = int(on[i])         # onset (start of cycle)
        i1 = int(on[i + 1])     # next onset (end of cycle)

        # need a valid inspiratory offset for this cycle
        if i >= len(off):
            if debug:
                print(f"  Area_exp cycle {i}: FAILED - missing insp offset (i={i}, len(off)={len(off)})")
            vals.append(np.nan)
            failure_count += 1
            continue
        oi = int(off[i])

        # need a valid expiratory offset for this cycle
        if i >= len(exo):
            if debug:
                print(f"  Area_exp cycle {i}: FAILED - missing exp offset (i={i}, len(exo)={len(exo)})")
            vals.append(np.nan)
            failure_count += 1
            continue
        ei = int(exo[i])

        # Get expiratory minimum for this cycle (if available)
        exp_min_val = None
        if expmins is not None and i < len(expmins):
            exp_min_idx = int(expmins[i])
            if 0 <= exp_min_idx < len(y):
                exp_min_val = y[exp_min_idx]

        # If expiratory peak is positive, there's no valid expiratory phase
        if exp_min_val is not None and exp_min_val > 0:
            if debug:
                print(f"  Area_exp cycle {i}: Set to 0 (exp_min={exp_min_val:.6f} > 0, no valid expiratory phase)")
            vals.append(0.0)
            continue

        # guards: oi must be inside cycle, and expoff must be after oi and before next onset
        if not (i0 <= oi < i1) or not (oi < ei <= i1):
            if debug:
                print(f"  Area_exp cycle {i}: FAILED - bounds check")
                print(f"    Onset: {i0}, Next onset: {i1}")
                print(f"    Insp offset: {oi} (valid: {i0 <= oi < i1})")
                print(f"    Exp offset: {ei} (valid: {oi < ei <= i1})")
            vals.append(np.nan)
            failure_count += 1
            continue

        # trapezoid integral over [oi, ei] (inclusive of ei via +1 end)
        left  = max(0, oi)
        right = min(N, ei + 1)
        if right - left < 2:
            if debug:
                print(f"  Area_exp cycle {i}: FAILED - integration window too small")
                print(f"    left={left}, right={right}, width={right-left}")
            vals.append(np.nan)
            failure_count += 1
            continue

        # Check for NaN values in the integration window
        t_seg = t[left:right]
        y_seg = y[left:right]
        if np.any(np.isnan(t_seg)) or np.any(np.isnan(y_seg)):
            if debug:
                print(f"  Area_exp cycle {i}: FAILED - NaN in integration window")
            vals.append(np.nan)
            failure_count += 1
            continue

        area = float(np.trapz(y_seg, t_seg))
        vals.append(area)

        if debug and area == 0.0:
            print(f"  Area_exp cycle {i}: Zero area calculated")
            print(f"    Integration window: [{oi}, {ei}] ({right-left} points)")
            print(f"    y range: [{y_seg.min():.6f}, {y_seg.max():.6f}]")

    if debug and failure_count > 0:
        print(f"  Area_exp: {failure_count}/{len(on)-1} cycles failed")

    # extend last span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)



################################################################################
#####Ti Calculation
################################################################################
def compute_ti(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Ti duration per cycle (seconds): time from onset -> offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If no valid offset for a cycle, that cycle is NaN.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 1 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])

        # offset must lie in (i0, i1]
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        dt = float(t[oi] - t[i0])
        vals.append(dt if dt > 0 else np.nan)

    # extend last span to N with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
#####Te Calculation
################################################################################
#     """
#     Te duration per cycle (seconds): time from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If no valid offset for a cycle, that cycle is NaN.
#     """
# def compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     for i in range(len(on) - 1):
#     vals: list[float] = []
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#             vals.append(np.nan)
#             continue
#         if i >= len(off):
#         oi = int(off[i])

#         # offset must lie in [i0, i1)
#             vals.append(np.nan)
#             continue
#         if not (i0 <= oi < i1):

#         vals.append(dt if dt > 0 else np.nan)
#         dt = float(t[i1] - t[oi])

#     # extend last span to N with last value
#     vals.append(vals[-1] if len(vals) else np.nan)
#     return _step_fill(N, spans, vals)

def compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None, debug=False) -> np.ndarray:
    """
    Te per cycle (seconds): time from inspiratory offset -> expiratory offset.
    Returns a stepwise-constant array over onset→next-onset spans.
    If either offset or expiratory offset for a cycle is missing/invalid, that cycle is NaN.
    If expiratory minimum is positive (above baseline), returns 0 (no valid expiratory phase).
    If calculated Te is ≤ 0, returns 0 (not NaN).
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if (onsets is None or len(onsets) < 2 or
        offsets is None or len(offsets) < 1 or
        expoffs is None or len(expoffs) < 1):
        return out

    on  = np.asarray(onsets,  dtype=int)
    off = np.asarray(offsets, dtype=int)
    exo = np.asarray(expoffs, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    failure_count = 0

    for i in range(len(on) - 1):
        i0 = int(on[i])         # onset (start of cycle)
        i1 = int(on[i + 1])     # next onset (end of cycle)

        if i >= len(off) or i >= len(exo):
            if debug:
                print(f"  Te cycle {i}: FAILED - missing offset/expoff (i={i}, len(off)={len(off)}, len(exo)={len(exo)})")
            vals.append(np.nan)
            failure_count += 1
            continue

        oi = int(off[i])        # inspiratory offset
        ei = int(exo[i])        # expiratory offset

        # Get expiratory minimum for this cycle (if available)
        exp_min_val = None
        if expmins is not None and i < len(expmins):
            exp_min_idx = int(expmins[i])
            if 0 <= exp_min_idx < len(y):
                exp_min_val = y[exp_min_idx]

        # If expiratory peak is positive, there's no valid expiratory phase
        if exp_min_val is not None and exp_min_val > 0:
            if debug:
                print(f"  Te cycle {i}: Set to 0 (exp_min={exp_min_val:.6f} > 0, no valid expiratory phase)")
            vals.append(0.0)
            continue

        # Check for NaN values in the time array
        if np.isnan(t[oi]) or np.isnan(t[ei]):
            if debug:
                print(f"  Te cycle {i}: FAILED - NaN in time array (t[{oi}]={t[oi]}, t[{ei}]={t[ei]})")
            vals.append(np.nan)
            failure_count += 1
            continue

        # Guards: offset inside cycle; expoff strictly after offset and before/equal next onset
        if not (i0 <= oi < i1) or not (oi < ei <= i1):
            if debug:
                print(f"  Te cycle {i}: FAILED - bounds check")
                print(f"    Onset: {i0}, Next onset: {i1}")
                print(f"    Insp offset: {oi} (valid: {i0 <= oi < i1})")
                print(f"    Exp offset: {ei} (valid: {oi < ei <= i1})")
            vals.append(np.nan)
            failure_count += 1
            continue

        dt = float(t[ei] - t[oi])
        # If Te is ≤ 0, clamp to 0 (not NaN)
        result = max(0.0, dt)
        vals.append(result)

        if debug and result == 0.0:
            print(f"  Te cycle {i}: Clamped to 0 (calculated dt={dt:.6f})")
            print(f"    t[{oi}]={t[oi]:.6f}, t[{ei}]={t[ei]:.6f}")

    if debug and failure_count > 0:
        print(f"  Te: {failure_count}/{len(on)-1} cycles failed")

    # Extend last span with last value
    vals.append(vals[-1] if vals else np.nan)
    return _step_fill(N, spans, vals)





################################################################################
#####Vent proxy Calculation
################################################################################
def compute_vent_proxy(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Vent proxy = (inspiratory area) / (cycle duration), in arbitrary units per second.
    Inspiratory area: integral from onset -> offset.
    Cycle duration: onset -> next onset.
    Returns a stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)

    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # need a valid offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])
        # offset must lie inside the cycle
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        area_insp = _trapz_seg(t, y, i0, oi)
        dur = float(t[i1] - t[i0])
        vals.append(area_insp / dur if (dur > 0 and np.isfinite(area_insp)) else np.nan)

    # extend final span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)

################################################################################
#####d/dt(pleth signal) Calculation
################################################################################
def compute_d1(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    First derivative dy/dt using np.gradient with the time vector (same length as y).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y) or len(t) < 2:
        return np.full_like(y, np.nan, dtype=float)
    return np.gradient(y, t)

################################################################################
#####d^2/dt^2(pleth signal) Calculation
################################################################################
def compute_d2(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Second derivative d²y/dt² via two successive gradients with the time vector.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y) or len(t) < 3:
        return np.full_like(y, np.nan, dtype=float)
    d1 = np.gradient(y, t)
    return np.gradient(d1, t)


################################################################################
##### Maximum Inspiratory Derivative (Peak Inhalation Rate)
################################################################################
def compute_max_dinsp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Maximum positive derivative during inspiration (peak inhalation rate).

    For each breath cycle, finds the maximum value of dy/dt during the
    inspiratory phase (onset → offset).

    Returns:
        Stepwise-constant array where each breath span has its maximum
        positive derivative value.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 1 or offsets is None or len(offsets) < 1:
        return out

    # Compute first derivative
    dy_dt = compute_d1(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])      # onset
        i1 = int(on[i + 1])  # next onset

        # Need a valid offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])  # offset (end of inspiration)

        # Offset must lie within cycle
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        # Find maximum derivative during inspiration [i0, oi]
        insp_segment = dy_dt[i0:oi+1]
        valid_values = insp_segment[~np.isnan(insp_segment)]

        if len(valid_values) == 0:
            vals.append(np.nan)
        else:
            vals.append(float(np.max(valid_values)))

    # Extend last span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
##### Maximum Expiratory Derivative (Peak Exhalation Rate)
################################################################################
def compute_max_dexp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Maximum negative derivative during expiration (peak exhalation rate).

    For each breath cycle, finds the minimum value of dy/dt during the
    expiratory phase (offset → expiratory offset or next onset).
    Returns as absolute value for easier interpretation.

    Returns:
        Stepwise-constant array where each breath span has its maximum
        negative derivative value (as absolute value).
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    # Compute first derivative
    dy_dt = compute_d1(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)
    exo = np.asarray(expoffs, dtype=int) if (expoffs is not None and len(expoffs)) else np.array([], dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])      # onset
        i1 = int(on[i + 1])  # next onset

        # Need a valid inspiratory offset for this cycle
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])  # offset (start of expiration)

        # Offset must lie within cycle
        if not (i0 <= oi < i1):
            vals.append(np.nan)
            continue

        # Determine end of expiration
        # Prefer expiratory offset if available, otherwise use next onset
        if exo.size > 0 and i < len(exo):
            ei = int(exo[i])
            # Validate expiratory offset
            if not (oi < ei <= i1):
                ei = i1  # Fallback to next onset
        else:
            ei = i1  # Use next onset

        # Find minimum (most negative) derivative during expiration [oi, ei]
        exp_segment = dy_dt[oi:ei+1]
        valid_values = exp_segment[~np.isnan(exp_segment)]

        if len(valid_values) == 0:
            vals.append(np.nan)
        else:
            min_derivative = float(np.min(valid_values))
            # Return as absolute value for easier interpretation
            vals.append(abs(min_derivative))

    # Extend last span with last value
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
##### Eupnic Breathing Detection
################################################################################

def _merge_nearby_regions(mask: np.ndarray, t: np.ndarray, max_gap_sec: float = 0.5) -> np.ndarray:
    """
    Merge nearby True regions in a boolean mask if gaps are smaller than max_gap_sec.
    """
    if np.sum(mask) == 0:
        return mask.copy()

    # Find transitions
    diff_mask = np.diff(mask.astype(int))
    starts = np.where(diff_mask == 1)[0] + 1  # Start of True regions
    ends = np.where(diff_mask == -1)[0] + 1   # End of True regions

    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    if len(starts) == 0 or len(ends) == 0:
        return mask.copy()

    # Merge regions with small gaps
    merged_mask = np.zeros_like(mask, dtype=bool)

    current_start = starts[0]
    current_end = ends[0]

    for i in range(1, len(starts)):
        gap_duration = t[starts[i]] - t[current_end - 1] if current_end < len(t) and starts[i] < len(t) else float('inf')

        if gap_duration <= max_gap_sec:
            # Merge with current region
            current_end = ends[i]
        else:
            # Finalize current region and start new one
            merged_mask[current_start:current_end] = True
            current_start = starts[i]
            current_end = ends[i]

    # Finalize last region
    merged_mask[current_start:current_end] = True

    return merged_mask


def _filter_duration(mask: np.ndarray, t: np.ndarray, min_duration_sec: float) -> np.ndarray:
    """
    Keep only True regions that last at least min_duration_sec.
    """
    if np.sum(mask) == 0:
        return mask.copy()

    # Find connected regions
    diff_mask = np.diff(mask.astype(int))
    starts = np.where(diff_mask == 1)[0] + 1
    ends = np.where(diff_mask == -1)[0] + 1

    # Handle edge cases
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])

    if len(starts) == 0 or len(ends) == 0:
        return np.zeros_like(mask, dtype=bool)

    # Filter by duration
    filtered_mask = np.zeros_like(mask, dtype=bool)

    for start, end in zip(starts, ends):
        if start < len(t) and end <= len(t):
            duration = t[end - 1] - t[start] if end > start else 0
            if duration >= min_duration_sec:
                filtered_mask[start:end] = True

    return filtered_mask


def detect_eupnic_regions(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None,
    freq_threshold_hz: float = 5.0,
    min_duration_sec: float = 2.0,
    sniff_regions: list = None
) -> np.ndarray:
    """
    Detect regions of eupnic (normal, regular) breathing.

    Simplified eupnic breathing criteria:
    - Respiratory rate below freq_threshold_hz (default: 5 Hz)
    - Sustained for at least min_duration_sec (default: 2 seconds)
    - Excluded from sniffing regions (if provided)

    NOTE: This is a FALLBACK method used when GMM-based detection is unavailable.
    The default detection mode is GMM-based (eupnea_detection_mode = "gmm").
    This frequency-based method does NOT use coefficient of variation (CV).

    Args:
        sniff_regions: List of (start_time, end_time) tuples marking sniffing bouts

    Returns:
        Binary array (0/1) same length as y, where 1 indicates eupnic breathing
    """
    N = len(y)
    eupnic_mask = np.zeros(N, dtype=bool)

    # Early exit for insufficient data
    if N < 100 or onsets is None or len(onsets) < 3:
        return eupnic_mask.astype(float)

    # Get instantaneous frequency
    freq_hz = compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)

    # Step 1: Apply frequency threshold (only criterion now)
    freq_valid = (freq_hz > 0) & (freq_hz < freq_threshold_hz) & (~np.isnan(freq_hz))

    if np.sum(freq_valid) == 0:
        return eupnic_mask.astype(float)

    # Step 2: Pre-filter chunks by minimum duration (must be ≥ 2s to qualify for merging)
    prefiltered_mask = _filter_duration(freq_valid, t, min_duration_sec)

    # Step 3: Merge nearby pre-qualified regions (fill small gaps between long chunks)
    final_mask = _merge_nearby_regions(prefiltered_mask, t, max_gap_sec=0.5)

    # Step 4: Exclude sniffing regions with buffer (sniffing overrides eupnea)
    if sniff_regions:
        buffer_sec = 0.5  # Add 0.5s buffer on each side of sniffing region
        for (start_time, end_time) in sniff_regions:
            # Expand region by buffer on both sides
            buffered_start = start_time - buffer_sec
            buffered_end = end_time + buffer_sec
            # Find indices corresponding to this buffered time range
            sniff_idx = (t >= buffered_start) & (t <= buffered_end)
            final_mask[sniff_idx] = False

    return final_mask.astype(float)


def detect_apneas(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None,
    min_apnea_duration_sec: float = 0.5
) -> np.ndarray:
    """
    Detect apneic periods (absence of breathing) longer than threshold.

    Apnea criteria:
    - Gap between consecutive breath onsets > min_apnea_duration_sec
    - Marked as binary signal where 1 = apneic period

    Returns:
        Binary array (0/1) same length as y, where 1 indicates apnea
    """
    N = len(y)
    apnea_mask = np.zeros(N, dtype=bool)

    # Early exit for insufficient data
    if N < 10 or onsets is None or len(onsets) < 2:
        return apnea_mask.astype(float)

    onsets = np.asarray(onsets, dtype=int)

    # Find gaps between consecutive onsets
    for i in range(len(onsets) - 1):
        onset_current = int(onsets[i])
        onset_next = int(onsets[i + 1])

        # Calculate gap duration
        if onset_next < len(t) and onset_current < len(t):
            gap_duration = t[onset_next] - t[onset_current]

            # If gap is longer than threshold, mark as apnea
            if gap_duration > min_apnea_duration_sec:
                # Mark the gap region (from current onset to next onset)
                start_idx = max(0, onset_current)
                end_idx = min(N, onset_next)
                apnea_mask[start_idx:end_idx] = True

    return apnea_mask.astype(float)


def compute_regularity_score(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None
) -> np.ndarray:
    """
    Compute breathing regularity score using RMSSD (Root Mean Square of Successive Differences).

    Lower values = more regular breathing patterns.
    Higher values = more irregular/variable breathing patterns.

    Returns:
        Stepwise-constant array over breath cycles, where each cycle gets a regularity score
        based on a sliding window of recent frequency measurements.
    """
    N = len(y)

    # Early exit for insufficient data
    if N < 100 or onsets is None or len(onsets) < 3:
        return np.full(N, np.nan, dtype=float)

    # Get instantaneous frequency signal
    freq_hz = compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)

    # Extract valid frequency values (skip NaNs and zeros)
    valid_mask = (freq_hz > 0) & (~np.isnan(freq_hz))
    if np.sum(valid_mask) < 3:
        return np.full(N, np.nan, dtype=float)

    # Compute RMSSD in sliding windows across breath cycles
    onsets = np.asarray(onsets, dtype=int)
    spans = _spans_from_bounds(onsets, N)
    regularity_values = []

    window_size = 5  # Number of consecutive cycles to analyze

    for i in range(len(onsets)):
        # Define window of cycles around current cycle
        start_cycle = max(0, i - window_size // 2)
        end_cycle = min(len(onsets), i + window_size // 2 + 1)

        if end_cycle - start_cycle < 3:  # Need at least 3 cycles for meaningful RMSSD
            regularity_values.append(np.nan)
            continue

        # Extract frequency values for cycles in window
        window_freqs = []
        for j in range(start_cycle, end_cycle):
            cycle_start = onsets[j]
            cycle_end = onsets[j + 1] if j + 1 < len(onsets) else N

            # Get frequency value for this cycle (should be constant across cycle)
            cycle_freq_vals = freq_hz[cycle_start:cycle_end]
            valid_cycle_freqs = cycle_freq_vals[~np.isnan(cycle_freq_vals)]

            if len(valid_cycle_freqs) > 0:
                window_freqs.append(valid_cycle_freqs[0])  # Take first valid value (they should all be the same)

        # Compute RMSSD for this window
        if len(window_freqs) >= 3:
            freq_diffs = np.diff(window_freqs)
            rmssd = np.sqrt(np.mean(freq_diffs**2))
            regularity_values.append(float(rmssd))
        else:
            regularity_values.append(np.nan)

    # Extend to match number of spans (including final span)
    if len(regularity_values) < len(spans):
        regularity_values.append(regularity_values[-1] if regularity_values else np.nan)

    return _step_fill(N, spans, regularity_values)


# Debug wrappers for diagnostic output
def compute_sniff_confidence(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None
) -> np.ndarray:
    """
    Return GMM-computed sniffing probability for each breath cycle.

    Uses probabilities from automatic GMM clustering (if available).
    Returns NaN if GMM has not been run or no probabilities available.

    Returns:
        Stepwise-constant array where each breath span has its sniffing probability (0.0 to 1.0)
    """
    N = len(y)

    # If no GMM probabilities available, return NaN
    if _current_gmm_probabilities is None:
        return np.full(N, np.nan, dtype=float)

    # Handle case where _current_gmm_probabilities is a numpy scalar (0-d array)
    if isinstance(_current_gmm_probabilities, np.ndarray) and _current_gmm_probabilities.ndim == 0:
        return np.full(N, np.nan, dtype=float)

    # Check if empty (for dict or array-like)
    try:
        if len(_current_gmm_probabilities) == 0:
            return np.full(N, np.nan, dtype=float)
    except TypeError:
        # If len() fails, treat as invalid and return NaN
        return np.full(N, np.nan, dtype=float)

    # Get breath spans from onsets
    if onsets is None or len(onsets) == 0:
        return np.full(N, np.nan, dtype=float)

    spans = _breath_spans_from_onsets(onsets, N)

    # Extract probabilities for each breath
    probs = []
    for breath_idx in range(len(onsets)):
        if breath_idx in _current_gmm_probabilities:
            probs.append(_current_gmm_probabilities[breath_idx])
        else:
            probs.append(np.nan)  # No GMM data for this breath

    return _step_fill(N, spans, probs)


def compute_eupnea_confidence(
    t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None
) -> np.ndarray:
    """
    Return GMM-computed eupnea probability for each breath cycle.

    This is simply (1 - sniffing_probability) for 2-cluster GMM.
    Returns NaN if GMM has not been run or no probabilities available.

    Returns:
        Stepwise-constant array where each breath span has its eupnea probability (0.0 to 1.0)
    """
    N = len(y)

    # If no GMM probabilities available, return NaN
    if _current_gmm_probabilities is None:
        return np.full(N, np.nan, dtype=float)

    # Handle case where _current_gmm_probabilities is a numpy scalar (0-d array)
    if isinstance(_current_gmm_probabilities, np.ndarray) and _current_gmm_probabilities.ndim == 0:
        return np.full(N, np.nan, dtype=float)

    # Check if empty (for dict or array-like)
    try:
        if len(_current_gmm_probabilities) == 0:
            return np.full(N, np.nan, dtype=float)
    except TypeError:
        # If len() fails, treat as invalid and return NaN
        return np.full(N, np.nan, dtype=float)

    # Get breath spans from onsets
    if onsets is None or len(onsets) == 0:
        return np.full(N, np.nan, dtype=float)

    spans = _breath_spans_from_onsets(onsets, N)

    # Extract eupnea probabilities (1 - sniffing probability)
    probs = []
    for breath_idx in range(len(onsets)):
        if breath_idx in _current_gmm_probabilities:
            sniff_prob = _current_gmm_probabilities[breath_idx]
            probs.append(1.0 - sniff_prob)
        else:
            probs.append(np.nan)  # No GMM data for this breath

    return _step_fill(N, spans, probs)


def compute_te_debug(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None):
    """Wrapper to enable debug output for Te calculation."""
    print("\n=== DEBUG: Te Calculation ===")
    result = compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs, debug=True)
    print("=============================\n")
    return result

def compute_area_exp_debug(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None):
    """Wrapper to enable debug output for expiratory area calculation."""
    print("\n=== DEBUG: Expiratory Area Calculation ===")
    result = compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs, debug=True)
    print("==========================================\n")
    return result

################################################################################
##### Phase 2.1: Half-Width Features (ML Breath Classification)
################################################################################

def compute_fwhm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Full Width at Half Maximum (FWHM) per breath cycle.

    Measures breath duration at 50% of peak amplitude (robust to noise).
    For each breath: find time span where signal > (baseline + 0.5 * peak_amplitude).

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        # Find first peak in this cycle
        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        # Half-maximum level
        baseline = y[i0]
        peak_amp = y[p] - baseline
        half_max = baseline + 0.5 * peak_amp

        # Find crossings of half-max level
        breath_segment = y[i0:i1]
        above_half = breath_segment >= half_max

        if not np.any(above_half):
            vals.append(np.nan)
            continue

        # Find first and last sample above half-max
        above_indices = np.where(above_half)[0]
        i_start = i0 + above_indices[0]
        i_end = i0 + above_indices[-1]

        # FWHM in seconds
        fwhm_sec = float(t[i_end] - t[i_start])
        vals.append(fwhm_sec if fwhm_sec > 0 else np.nan)

    # Extend last span
    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_width_25(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Width at 25% of peak amplitude (broader than FWHM, more stable for small peaks).
    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        baseline = y[i0]
        peak_amp = y[p] - baseline
        level_25 = baseline + 0.25 * peak_amp

        breath_segment = y[i0:i1]
        above_25 = breath_segment >= level_25

        if not np.any(above_25):
            vals.append(np.nan)
            continue

        above_indices = np.where(above_25)[0]
        i_start = i0 + above_indices[0]
        i_end = i0 + above_indices[-1]

        width_sec = float(t[i_end] - t[i_start])
        vals.append(width_sec if width_sec > 0 else np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_width_75(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Width at 75% of peak amplitude (narrower than FWHM, sensitive to peak shape).
    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        baseline = y[i0]
        peak_amp = y[p] - baseline
        level_75 = baseline + 0.75 * peak_amp

        breath_segment = y[i0:i1]
        above_75 = breath_segment >= level_75

        if not np.any(above_75):
            vals.append(np.nan)
            continue

        above_indices = np.where(above_75)[0]
        i_start = i0 + above_indices[0]
        i_end = i0 + above_indices[-1]

        width_sec = float(t[i_end] - t[i_start])
        vals.append(width_sec if width_sec > 0 else np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_width_ratio(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Width ratio: width_75 / width_25 (shape descriptor, ~0.3-0.5 for normal breaths).

    - Ratio near 1.0: rectangular/flat peak (unusual)
    - Ratio near 0.3-0.5: normal Gaussian-like peak
    - Ratio near 0: very sharp peak (may be noise/artifact)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        baseline = y[i0]
        peak_amp = y[p] - baseline
        level_25 = baseline + 0.25 * peak_amp
        level_75 = baseline + 0.75 * peak_amp

        breath_segment = y[i0:i1]

        # Width at 25%
        above_25 = breath_segment >= level_25
        if not np.any(above_25):
            vals.append(np.nan)
            continue
        above_indices_25 = np.where(above_25)[0]
        width_25 = float(t[i0 + above_indices_25[-1]] - t[i0 + above_indices_25[0]])

        # Width at 75%
        above_75 = breath_segment >= level_75
        if not np.any(above_75):
            vals.append(np.nan)
            continue
        above_indices_75 = np.where(above_75)[0]
        width_75 = float(t[i0 + above_indices_75[-1]] - t[i0 + above_indices_75[0]])

        # Ratio (avoid division by zero)
        ratio = (width_75 / width_25) if width_25 > 0 else np.nan
        vals.append(ratio)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
##### Phase 2.2: Sigh Detection Features (ML Breath Classification)
################################################################################

def compute_n_inflections(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Count inflection points in rising phase (onset → peak).

    Inflection points = sign changes in 2nd derivative (d²/dt²).
    Normal breaths: 0-1 inflections
    Sighs (double-hump): 2-4 inflections

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        # Rising phase: onset → peak
        if p - i0 < 5:  # Too short to compute 2nd derivative
            vals.append(np.nan)
            continue

        rising_phase = y[i0:p+1]

        # Compute 2nd derivative (simple finite difference)
        d2 = np.diff(rising_phase, n=2)

        # Count sign changes in d2 (inflection points)
        sign_changes = np.sum(np.diff(np.sign(d2)) != 0)
        vals.append(float(sign_changes))

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_rise_variability(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Variability of derivative in rising phase (standard deviation of d/dt).

    High variability → irregular, bumpy rise (sigh or noise)
    Low variability → smooth rise (normal breath)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        if p - i0 < 3:
            vals.append(np.nan)
            continue

        rising_phase = y[i0:p+1]
        d1 = np.diff(rising_phase)

        # Standard deviation of derivative
        rise_var = float(np.std(d1))
        vals.append(rise_var)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_n_shoulder_peaks(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Count secondary peaks (shoulders) in rising phase.

    Uses scipy.signal.find_peaks to detect local maxima in rising phase.
    Normal breaths: 0-1 shoulder peaks
    Sighs: 1-3 shoulder peaks (double/triple hump)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    try:
        from scipy.signal import find_peaks as scipy_find_peaks
    except ImportError:
        # Fallback: return NaN if scipy not available
        return np.full(len(y), np.nan, dtype=float)

    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        if p - i0 < 10:  # Too short to reliably detect shoulders
            vals.append(np.nan)
            continue

        rising_phase = y[i0:p+1]

        # Find local maxima (excluding the main peak at end)
        # Use low prominence threshold to catch subtle shoulders
        min_prom = np.std(rising_phase) * 0.1  # 10% of local variability
        shoulders, _ = scipy_find_peaks(rising_phase[:-1], prominence=min_prom)

        vals.append(float(len(shoulders)))

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_shoulder_prominence(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Maximum prominence of secondary peaks (shoulders) in rising phase.

    High shoulder prominence → strong double-hump (sigh)
    Low/zero prominence → smooth rise (normal breath)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    try:
        from scipy.signal import find_peaks as scipy_find_peaks
    except ImportError:
        return np.full(len(y), np.nan, dtype=float)

    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        if p - i0 < 10:
            vals.append(np.nan)
            continue

        rising_phase = y[i0:p+1]
        min_prom = np.std(rising_phase) * 0.1
        shoulders, props = scipy_find_peaks(rising_phase[:-1], prominence=min_prom)

        if len(shoulders) > 0 and 'prominences' in props:
            max_prom = float(np.max(props['prominences']))
            vals.append(max_prom)
        else:
            vals.append(0.0)  # No shoulders = zero prominence

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_rise_autocorr(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Autocorrelation of rising phase at lag = 25% of rise duration.

    High autocorrelation → periodic/oscillatory rise (double-hump sigh)
    Low autocorrelation → aperiodic smooth rise (normal breath)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        if p - i0 < 20:  # Too short for meaningful autocorrelation
            vals.append(np.nan)
            continue

        rising_phase = y[i0:p+1]

        # Zero-mean the signal
        rising_centered = rising_phase - np.mean(rising_phase)

        # Lag = 25% of rise duration (to detect double-hump periodicity)
        lag = max(1, int(len(rising_centered) * 0.25))

        if lag >= len(rising_centered):
            vals.append(np.nan)
            continue

        # Autocorrelation at lag
        c0 = np.dot(rising_centered, rising_centered)  # Variance
        c_lag = np.dot(rising_centered[:-lag], rising_centered[lag:])

        autocorr = (c_lag / c0) if c0 > 0 else 0.0
        vals.append(float(autocorr))

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_peak_sharpness(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Curvature at peak (negative of 2nd derivative at peak index).

    High sharpness → pointy peak (normal breath or noise)
    Low sharpness → rounded peak (sigh, slow breath)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        mask = (pk >= i0) & (pk < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        p = int(pk[mask][0])

        # Need at least 2 samples on each side of peak for 2nd derivative
        if p < 2 or p >= N - 2:
            vals.append(np.nan)
            continue

        # 2nd derivative at peak (central difference)
        d2_peak = y[p+1] + y[p-1] - 2*y[p]

        # Curvature = -d2 (negative because peak is a local maximum)
        curvature = float(-d2_peak)
        vals.append(curvature)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_trough_sharpness(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Curvature at expiratory minimum (2nd derivative at expmin index).

    High sharpness → pointy trough (abrupt expiration)
    Low sharpness → rounded trough (gradual expiration)

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or expmins is None or len(expmins) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    em = np.asarray(expmins, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        # Find expmin in this cycle
        mask = (em >= i0) & (em < i1)
        if not np.any(mask):
            vals.append(np.nan)
            continue
        expmin_idx = int(em[mask][0])

        if expmin_idx < 2 or expmin_idx >= N - 2:
            vals.append(np.nan)
            continue

        # 2nd derivative at expmin
        d2_trough = y[expmin_idx+1] + y[expmin_idx-1] - 2*y[expmin_idx]

        # Curvature = +d2 (positive because trough is a local minimum)
        curvature = float(d2_trough)
        vals.append(curvature)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_skewness(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Statistical skewness of breath segment (onset → next onset).

    Skewness > 0: Right-skewed (long tail toward peak) - gradual rise, fast fall
    Skewness < 0: Left-skewed (long tail toward trough) - fast rise, gradual fall
    Skewness ≈ 0: Symmetric breath

    Returns stepwise-constant array over onset→next-onset spans.
    """
    try:
        from scipy.stats import skew
    except ImportError:
        # Fallback: manual skewness calculation
        def skew(x):
            x_centered = x - np.mean(x)
            std = np.std(x)
            if std == 0:
                return 0.0
            return np.mean((x_centered / std) ** 3)

    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        if i1 - i0 < 5:  # Too short for meaningful statistics
            vals.append(np.nan)
            continue

        breath_segment = y[i0:i1]
        skewness = float(skew(breath_segment))
        vals.append(skewness)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_kurtosis(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Statistical kurtosis of breath segment (onset → next onset).

    Kurtosis > 3: Leptokurtic (heavy tails, sharp peak) - spiky breath
    Kurtosis = 3: Normal distribution (Gaussian-like)
    Kurtosis < 3: Platykurtic (light tails, flat peak) - rounded sigh

    Returns excess kurtosis (kurtosis - 3) for easier interpretation:
    - Excess > 0: Sharper than Gaussian
    - Excess = 0: Gaussian-like
    - Excess < 0: Flatter than Gaussian

    Returns stepwise-constant array over onset→next-onset spans.
    """
    try:
        from scipy.stats import kurtosis as scipy_kurtosis
    except ImportError:
        # Fallback: manual kurtosis calculation
        def scipy_kurtosis(x, fisher=True):
            x_centered = x - np.mean(x)
            std = np.std(x)
            if std == 0:
                return 0.0
            kurt = np.mean((x_centered / std) ** 4)
            return (kurt - 3.0) if fisher else kurt

    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        if i1 - i0 < 5:
            vals.append(np.nan)
            continue

        breath_segment = y[i0:i1]
        # Fisher=True returns excess kurtosis (kurtosis - 3)
        kurt = float(scipy_kurtosis(breath_segment, fisher=True))
        vals.append(kurt)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


################################################################################
##### Phase 2.3: Relative Features & Shape Metrics (ML Breath Classification)
################################################################################

# ========== GROUP A: SHAPE & RATIO METRICS (context-independent) ==========

def compute_peak_to_trough(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Full breath amplitude: inspiratory peak → expiratory minimum.

    Measures total vertical excursion of breath cycle.
    Sighs typically have 2-3× larger peak-to-trough than normal breaths.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out
    if expmins is None or len(expmins) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    em = np.asarray(expmins, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        # Find peak in this cycle
        mask_pk = (pk >= i0) & (pk < i1)
        if not np.any(mask_pk):
            vals.append(np.nan)
            continue
        p = int(pk[mask_pk][0])

        # Find expmin in this cycle
        mask_em = (em >= i0) & (em < i1)
        if not np.any(mask_em):
            vals.append(np.nan)
            continue
        e = int(em[mask_em][0])

        # Peak to trough amplitude
        amplitude = float(y[p] - y[e])
        vals.append(amplitude)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_amp_ratio(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Amplitude ratio: amp_insp / amp_exp (breath symmetry).

    - Ratio > 1: Larger inspiration than expiration
    - Ratio = 1: Symmetric breath
    - Ratio < 1: Larger expiration (unusual)

    Sighs often have different symmetry than normal breaths.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) == 0 or peaks is None or len(peaks) == 0:
        return out
    if expmins is None or len(expmins) == 0:
        return out

    on = np.asarray(onsets, dtype=int)
    pk = np.asarray(peaks, dtype=int)
    em = np.asarray(expmins, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0, i1 = int(on[i]), int(on[i + 1])

        # Inspiratory amplitude
        mask_pk = (pk >= i0) & (pk < i1)
        if not np.any(mask_pk):
            vals.append(np.nan)
            continue
        p = int(pk[mask_pk][0])
        amp_insp = y[p] - y[i0]

        # Expiratory amplitude
        mask_em = (em >= i0) & (em < i1)
        if not np.any(mask_em):
            vals.append(np.nan)
            continue
        e = int(em[mask_em][0])

        # Next onset for expiratory amplitude
        if i + 1 < len(on):
            next_onset = int(on[i + 1])
            amp_exp = y[next_onset] - y[e]
        else:
            amp_exp = np.nan

        # Ratio (avoid division by zero)
        if amp_exp > 0:
            ratio = amp_insp / amp_exp
            vals.append(float(ratio))
        else:
            vals.append(np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_ti_te_ratio(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Duty cycle ratio: Ti / Te (inspiration time / expiration time).

    - Ratio > 1: Longer inspiration (unusual in mice)
    - Ratio = 0.5-0.8: Normal mouse breathing
    - Ratio < 0.5: Very short inspiration

    Sighs may have different duty cycle than normal breaths.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # Ti: onset → offset
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue
        ti = float(t[oi] - t[i0])

        # Te: offset → next onset
        te = float(t[i1] - t[oi])

        # Ratio (avoid division by zero)
        if te > 0:
            ratio = ti / te
            vals.append(float(ratio))
        else:
            vals.append(np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_area_ratio(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Area ratio: area_insp / area_exp (volume symmetry).

    Measures relative "volume" of inspiration vs expiration.
    - Ratio > 1: More inspiratory volume
    - Ratio = 1: Symmetric volume
    - Ratio < 1: More expiratory volume

    Sighs often have different volume ratios than normal breaths.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # Check offset exists and is valid
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        # Inspiratory area (onset → offset)
        area_insp = _trapz_seg(t, y, i0, oi)

        # Expiratory area (offset → next onset), take absolute value
        area_exp = abs(_trapz_seg(t, y, oi, i1))

        # Ratio (avoid division by zero)
        if area_exp > 0 and not np.isnan(area_insp) and not np.isnan(area_exp):
            ratio = area_insp / area_exp
            vals.append(float(ratio))
        else:
            vals.append(np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


def compute_ibi(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Inter-breath interval (IBI): time to next peak in seconds.

    Measures spacing between consecutive breaths.
    - Normal breathing: ~0.3-0.5s IBI (2-3 Hz)
    - Post-sigh pause: Often 1.5-2× longer IBI

    Critical for detecting post-sigh breathing pattern changes.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if peaks is None or len(peaks) < 2:
        return out

    pk = np.asarray(peaks, dtype=int)

    # Use peaks as boundaries for spans (not onsets)
    spans = _spans_from_bounds(pk, N)

    vals: list[float] = []
    for i in range(len(pk) - 1):
        p0 = int(pk[i])
        p1 = int(pk[i + 1])

        # Time between peaks
        ibi = float(t[p1] - t[p0])
        vals.append(ibi)

    # Last breath has no next peak
    vals.append(np.nan)
    return _step_fill(N, spans, vals)


def compute_total_area(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Total breath area: |area_insp| + |area_exp| (total volume per breath).

    Measures total "volume" of breath cycle (inspiration + expiration).
    Sighs have much larger total area than normal breaths.

    More robust than individual area metrics since it doesn't depend on
    accurate onset/offset detection - just sums all movement in cycle.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
        return out

    on = np.asarray(onsets, dtype=int)
    off = np.asarray(offsets, dtype=int)
    spans = _spans_from_bounds(on, N)

    vals: list[float] = []
    for i in range(len(on) - 1):
        i0 = int(on[i])
        i1 = int(on[i + 1])

        # Check offset exists and is valid
        if i >= len(off):
            vals.append(np.nan)
            continue
        oi = int(off[i])
        if not (i0 < oi <= i1):
            vals.append(np.nan)
            continue

        # Inspiratory area (onset → offset)
        area_insp = _trapz_seg(t, y, i0, oi)

        # Expiratory area (offset → next onset), take absolute value
        area_exp = abs(_trapz_seg(t, y, oi, i1))

        # Total area
        if not np.isnan(area_insp) and not np.isnan(area_exp):
            total = abs(area_insp) + area_exp  # Take abs of insp too for consistency
            vals.append(float(total))
        else:
            vals.append(np.nan)

    vals.append(vals[-1] if len(vals) else np.nan)
    return _step_fill(N, spans, vals)


# ========== GROUP B: NORMALIZED METRICS (ML-ready, context-dependent) ==========

def _compute_normalized_metric(base_metric_data: np.ndarray, stat_key: str) -> np.ndarray:
    """
    Helper: Compute z-score normalization for a metric using global statistics.

    Args:
        base_metric_data: Raw metric values (can contain NaN)
        stat_key: Key to look up in _normalization_stats (e.g., 'amp_insp')

    Returns:
        Z-score normalized array (same shape as input)
    """
    if _normalization_stats is None or stat_key not in _normalization_stats:
        # No normalization stats available - return NaN
        return np.full_like(base_metric_data, np.nan, dtype=float)

    stats = _normalization_stats[stat_key]
    mean_val = stats.get('mean', 0.0)
    std_val = stats.get('std', 1.0)

    # Z-score: (x - mean) / std
    if std_val > 0:
        return (base_metric_data - mean_val) / std_val
    else:
        # Zero std - all values identical, return zeros
        return np.zeros_like(base_metric_data, dtype=float)


def compute_amp_insp_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized inspiratory amplitude (z-score relative to all peaks in sweep).

    Z-score = (amp_insp - mean_all) / std_all

    Sighs typically have amp_insp_norm > 2.0 (2+ standard deviations above mean).
    Normal breaths: -1.0 to 1.0
    Small/noise breaths: < -1.0

    Returns stepwise-constant array over onset→next-onset spans.
    """
    # Compute raw metric
    raw = compute_amp_insp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    # Normalize
    return _compute_normalized_metric(raw, 'amp_insp')


def compute_amp_exp_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized expiratory amplitude (z-score relative to all peaks in sweep).

    Z-score = (amp_exp - mean_all) / std_all

    Returns stepwise-constant array over onset→next-onset spans.
    """
    raw = compute_amp_exp(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    return _compute_normalized_metric(raw, 'amp_exp')


def compute_peak_to_trough_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized peak-to-trough amplitude (z-score relative to all peaks).

    Z-score = (peak_to_trough - mean_all) / std_all

    Most robust amplitude metric for sigh detection.
    Sighs: norm > 2.0
    Normal: -1.0 to 1.0

    Returns stepwise-constant array over onset→next-onset spans.
    """
    raw = compute_peak_to_trough(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    return _compute_normalized_metric(raw, 'peak_to_trough')


def compute_prominence_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized peak prominence (z-score relative to all peaks).

    Z-score = (prominence - mean_all) / std_all

    CRITICAL for ML: This is what auto-threshold uses!
    Peaks with norm < 0 are typically classified as noise.
    Peaks with norm > 0 are typically real breaths.

    Returns stepwise-constant array over peak-bounded spans.
    """
    # We need to compute prominence for each peak
    # This requires accessing scipy.signal.find_peaks with return_properties
    from scipy.signal import find_peaks, peak_prominences

    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if peaks is None or len(peaks) == 0:
        return out

    pk = np.asarray(peaks, dtype=int)

    # Compute prominences
    proms = peak_prominences(y, pk)[0]

    # Use peaks as boundaries for spans
    spans = _spans_from_bounds(pk, N)

    # Fill with prominence values
    for i, ((i0, i1), prom) in enumerate(zip(spans, proms)):
        out[i0:i1] = float(prom)

    # Normalize
    return _compute_normalized_metric(out, 'prominence')


def compute_ibi_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized inter-breath interval (z-score relative to all peaks).

    Z-score = (ibi - mean_all) / std_all

    CRITICAL for post-sigh detection!
    - Normal IBI: -0.5 to 0.5
    - Post-sigh pause: > 1.5 (significantly longer than average)

    Returns stepwise-constant array over peak-bounded spans.
    """
    raw = compute_ibi(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    return _compute_normalized_metric(raw, 'ibi')


def compute_ti_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized inspiratory time (z-score relative to all peaks).

    Z-score = (ti - mean_all) / std_all

    Sighs often have longer Ti than normal breaths.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    raw = compute_ti(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    return _compute_normalized_metric(raw, 'ti')


def compute_te_norm(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Normalized expiratory time (z-score relative to all peaks).

    Z-score = (te - mean_all) / std_all

    Post-sigh breaths often have longer Te.

    Returns stepwise-constant array over onset→next-onset spans.
    """
    raw = compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs)
    return _compute_normalized_metric(raw, 'te')


# ========== GROUP C: PROBABILITY FROM AUTO-THRESHOLD MODEL ==========

def _evaluate_threshold_model(h: float) -> tuple[float, float]:
    """
    Evaluate exponential + 2 Gaussians model at height h.

    Returns:
        (p_noise, p_breath): Probabilities that peak is noise vs real breath
    """
    if _threshold_model_params is None:
        return (np.nan, np.nan)

    p = _threshold_model_params

    # Exponential component (noise)
    lambda_exp = p.get('lambda_exp', 1.0)
    w_exp = p.get('w_exp', 0.3)
    exp_val = lambda_exp * np.exp(-lambda_exp * h) * w_exp

    # Gaussian 1 (eupnea)
    mu1 = p.get('mu1', 0.3)
    sigma1 = p.get('sigma1', 0.1)
    w_g1 = p.get('w_g1', 0.4)
    g1_val = (1.0 / (np.sqrt(2 * np.pi) * sigma1)) * np.exp(-0.5 * ((h - mu1) / sigma1) ** 2) * w_g1

    # Gaussian 2 (sniffing)
    mu2 = p.get('mu2', 0.6)
    sigma2 = p.get('sigma2', 0.15)
    w_g2 = p.get('w_g2', 0.3)
    g2_val = (1.0 / (np.sqrt(2 * np.pi) * sigma2)) * np.exp(-0.5 * ((h - mu2) / sigma2) ** 2) * w_g2

    # Normalize to probabilities
    total = exp_val + g1_val + g2_val
    if total > 0:
        p_noise = exp_val / total
        p_breath = (g1_val + g2_val) / total
    else:
        p_noise = 0.5
        p_breath = 0.5

    return (float(p_noise), float(p_breath))


def compute_p_noise(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Probability that peak is noise based on auto-threshold model.

    Uses exponential + 2 Gaussians fit from auto-threshold dialog.
    P(noise | peak_height) = exp_component / (exp + gauss1 + gauss2)

    - p_noise near 1.0: Almost certainly noise
    - p_noise near 0.5: Ambiguous (near threshold)
    - p_noise near 0.0: Almost certainly real breath

    Returns stepwise-constant array over peak-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if peaks is None or len(peaks) == 0:
        return out
    if _threshold_model_params is None:
        # Model not available - return NaN
        return out

    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(pk, N)

    for i, (p_idx, (i0, i1)) in enumerate(zip(pk, spans)):
        peak_height = float(y[p_idx])
        p_noise, _ = _evaluate_threshold_model(peak_height)
        out[i0:i1] = p_noise

    return out


def compute_p_breath(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
    """
    Probability that peak is a real breath based on auto-threshold model.

    Uses exponential + 2 Gaussians fit from auto-threshold dialog.
    P(breath | peak_height) = (gauss1 + gauss2) / (exp + gauss1 + gauss2)

    - p_breath near 1.0: Almost certainly real breath
    - p_breath near 0.5: Ambiguous (near threshold)
    - p_breath near 0.0: Almost certainly noise

    Returns stepwise-constant array over peak-bounded spans.
    """
    N = len(y)
    out = np.full(N, np.nan, dtype=float)

    if peaks is None or len(peaks) == 0:
        return out
    if _threshold_model_params is None:
        # Model not available - return NaN
        return out

    pk = np.asarray(peaks, dtype=int)
    spans = _spans_from_bounds(pk, N)

    for i, (p_idx, (i0, i1)) in enumerate(zip(pk, spans)):
        peak_height = float(y[p_idx])
        _, p_breath = _evaluate_threshold_model(peak_height)
        out[i0:i1] = p_breath

    return out


# Set this to True to enable detailed diagnostic output for Te and area_exp
ENABLE_DEBUG_OUTPUT = False

# Registry: key -> function
METRICS: Dict[str, Callable] = {
    "if":          compute_if,
    "amp_insp":    compute_amp_insp,
    "amp_exp":     compute_amp_exp,
    "area_insp":   compute_area_insp,
    "area_exp":    compute_area_exp_debug if ENABLE_DEBUG_OUTPUT else compute_area_exp,
    "ti":          compute_ti,
    "te":          compute_te_debug if ENABLE_DEBUG_OUTPUT else compute_te,
    "vent_proxy":  compute_vent_proxy,
    "d1":          compute_d1,
    "d2":          compute_d2,
    "max_dinsp":   compute_max_dinsp,
    "max_dexp":    compute_max_dexp,
    "eupnic":      detect_eupnic_regions,
    "apnea":       detect_apneas,
    "regularity":  compute_regularity_score,
    "sniff_conf":  compute_sniff_confidence,
    "eupnea_conf": compute_eupnea_confidence,
    # Phase 2.1: Half-width features for ML
    "fwhm":        compute_fwhm,
    "width_25":    compute_width_25,
    "width_75":    compute_width_75,
    "width_ratio": compute_width_ratio,
    # Phase 2.2: Sigh detection features for ML
    "n_inflections":       compute_n_inflections,
    "rise_variability":    compute_rise_variability,
    "n_shoulder_peaks":    compute_n_shoulder_peaks,
    "shoulder_prominence": compute_shoulder_prominence,
    "rise_autocorr":       compute_rise_autocorr,
    "peak_sharpness":      compute_peak_sharpness,
    "trough_sharpness":    compute_trough_sharpness,
    "skewness":            compute_skewness,
    "kurtosis":            compute_kurtosis,
    # Phase 2.3 Group A: Shape & Ratio metrics
    "peak_to_trough":      compute_peak_to_trough,
    "amp_ratio":           compute_amp_ratio,
    "ti_te_ratio":         compute_ti_te_ratio,
    "area_ratio":          compute_area_ratio,
    "total_area":          compute_total_area,
    "ibi":                 compute_ibi,
    # Phase 2.3 Group B: Normalized metrics
    "amp_insp_norm":       compute_amp_insp_norm,
    "amp_exp_norm":        compute_amp_exp_norm,
    "peak_to_trough_norm": compute_peak_to_trough_norm,
    "prominence_norm":     compute_prominence_norm,
    "ibi_norm":            compute_ibi_norm,
    "ti_norm":             compute_ti_norm,
    "te_norm":             compute_te_norm,
    # Phase 2.3 Group C: Probability from auto-threshold
    "p_noise":             compute_p_noise,
    "p_breath":            compute_p_breath,
}

# Optional: Enable robust metrics mode
# Uncomment the following lines to use enhanced robust metrics with fallback strategies
# try:
#     print("Enhanced metrics with robust fallbacks enabled.")
# except ImportError:
#     print("Robust metrics module not available. Using standard metrics.")
#     from core.robust_metrics import enhance_metrics_with_robust_fallbacks
#     METRICS = enhance_metrics_with_robust_fallbacks(METRICS)
