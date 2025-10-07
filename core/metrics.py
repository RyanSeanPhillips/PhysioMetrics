# core/metrics.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Callable

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
    ("Eupnic breathing regions",              "eupnic"),
    ("Apnea detection",                       "apnea"),
    ("Breathing regularity score (RMSSD)",    "regularity"),
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
# def compute_if(t, y, sr_hz, peaks, onsets, offsets, expmins, expoffs=None) -> np.ndarray:
#     """
#     Instantaneous frequency as a stepwise signal (breaths/min).
#     Prefer onset→onset cycle times; fall back to peak→peak if onsets unavailable.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     # choose cycle boundaries
#     bounds = None
#     if onsets is not None and len(onsets) >= 2:
#         bounds = np.asarray(onsets, dtype=int)
#     elif peaks is not None and len(peaks) >= 2:
#         bounds = np.asarray(peaks, dtype=int)

#     if bounds is None or len(bounds) < 2:
#         return out  # not enough info

#     # compute BPM for each cycle
#     vals = []
#     spans = _spans_from_bounds(bounds, N)
#     for i in range(len(bounds) - 1):
#         i0, i1 = int(bounds[i]), int(bounds[i + 1])
#         dt = float(t[i1] - t[i0])
#         bpm = 60.0 / dt if dt > 0 else np.nan
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

# def compute_area_exp(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     """
#     Expiratory area per cycle: integral of y from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If offset is missing for a cycle, that cycle is NaN.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     vals: list[float] = []
#     for i in range(len(on) - 1):
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#         if i >= len(off):
#             vals.append(np.nan)
#             continue
#         oi = int(off[i])

#         # Guard: offset must lie within [i0, i1)
#         if not (i0 <= oi < i1):
#             vals.append(np.nan)
#             continue

#         # Trapezoid integral over [oi, i1]
#         left = max(0, oi)
#         right = min(N, i1 + 1)
#         if right - left < 2:
#             vals.append(np.nan)
#             continue
#         area = float(np.trapz(y[left:right], t[left:right]))
#         vals.append(area)

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
# def compute_te(t, y, sr_hz, peaks, onsets, offsets, expmins) -> np.ndarray:
#     """
#     Te duration per cycle (seconds): time from offset -> next onset.
#     Returns a stepwise-constant array over onset→next-onset spans.
#     If no valid offset for a cycle, that cycle is NaN.
#     """
#     N = len(y)
#     out = np.full(N, np.nan, dtype=float)

#     if onsets is None or len(onsets) < 2 or offsets is None or len(offsets) < 1:
#         return out

#     on = np.asarray(onsets, dtype=int)
#     off = np.asarray(offsets, dtype=int)

#     spans = _spans_from_bounds(on, N)

#     vals: list[float] = []
#     for i in range(len(on) - 1):
#         i0 = int(on[i])
#         i1 = int(on[i + 1])

#         if i >= len(off):
#             vals.append(np.nan)
#             continue
#         oi = int(off[i])

#         # offset must lie in [i0, i1)
#         if not (i0 <= oi < i1):
#             vals.append(np.nan)
#             continue

#         dt = float(t[i1] - t[oi])
#         vals.append(dt if dt > 0 else np.nan)

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
##### Eupnic Breathing Detection
################################################################################

def _sliding_window_cv(signal: np.ndarray, t: np.ndarray, window_sec: float) -> np.ndarray:
    """
    Compute coefficient of variation (CV = std/mean) efficiently using scipy.ndimage.
    Falls back to simple decimation if scipy not available.
    Returns array same length as signal with CV at each point.
    """
    N = len(signal)
    cv = np.full(N, np.nan, dtype=float)

    if N < 2 or window_sec <= 0:
        return cv

    # For efficiency with large datasets, use a decimated approach
    # Sample every ~0.1 seconds instead of every sample
    dt_mean = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    decimate_factor = max(1, int(0.1 / dt_mean))  # Sample every ~0.1s

    # Create decimated indices
    indices = np.arange(0, N, decimate_factor)
    if indices[-1] != N - 1:
        indices = np.append(indices, N - 1)  # Always include last point

    window_samples = max(2, int(window_sec / dt_mean))
    half_window = window_samples // 2

    # Compute CV only at decimated points
    for idx in indices:
        start = max(0, idx - half_window)
        end = min(N, idx + half_window + 1)

        window_data = signal[start:end]
        valid_data = window_data[~np.isnan(window_data)]

        if len(valid_data) >= 3:  # Need at least 3 points for meaningful CV
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data, ddof=1)
            if mean_val > 0:
                cv[idx] = std_val / mean_val

    # Forward-fill CV values between computed points
    last_valid = np.nan
    for i in range(N):
        if not np.isnan(cv[i]):
            last_valid = cv[i]
        elif not np.isnan(last_valid):
            cv[i] = last_valid

    return cv


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
    - Respiratory rate below freq_threshold_hz
    - Sustained for at least 2 seconds
    - Excluded from sniffing regions (if provided)

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

# Set this to True to enable detailed diagnostic output for Te and area_exp
ENABLE_DEBUG_OUTPUT = True

# Registry: key -> function
METRICS: Dict[str, Callable] = {
    "if":         compute_if,
    "amp_insp":   compute_amp_insp,
    "amp_exp":    compute_amp_exp,
    "area_insp":  compute_area_insp,
    "area_exp":   compute_area_exp_debug if ENABLE_DEBUG_OUTPUT else compute_area_exp,
    "ti":         compute_ti,
    "te":         compute_te_debug if ENABLE_DEBUG_OUTPUT else compute_te,
    "vent_proxy": compute_vent_proxy,
    "d1":         compute_d1,
    "d2":         compute_d2,
    "eupnic":     detect_eupnic_regions,
    "apnea":      detect_apneas,
    "regularity": compute_regularity_score,
}

# Optional: Enable robust metrics mode
# Uncomment the following lines to use enhanced robust metrics with fallback strategies
# try:
#     from core.robust_metrics import enhance_metrics_with_robust_fallbacks
#     METRICS = enhance_metrics_with_robust_fallbacks(METRICS)
#     print("Enhanced metrics with robust fallbacks enabled.")
# except ImportError:
#     print("Robust metrics module not available. Using standard metrics.")
