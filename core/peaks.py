import numpy as np
from typing import Dict, Tuple
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


try:
    from core import filters as core_filters
except Exception:
    core_filters = None

try:
    from scipy.signal import butter, filtfilt
except Exception:
    butter = filtfilt = None


def detect_peaks(y: np.ndarray, sr_hz: float,
                 thresh: float = None,
                 prominence: float = None,
                 min_dist_samples: int = None,
                 direction: str = "up",
                 return_all: bool = False) -> np.ndarray:
    """
    y: 1D vector
    Returns np.ndarray of peak indices (int)

    If return_all=True, returns ALL detected peaks without threshold filtering.
    If return_all=False (default), filters by threshold for backward compatibility.
    """
    y_use = -y if direction == "down" else y

    if find_peaks is None:
        # Minimal fallback: threshold crossing on upward slope
        if thresh is None:
            thresh = np.percentile(y_use, 90)
        above = y_use > thresh
        inds = np.where(np.diff(above.astype(int)) == 1)[0] + 1
        return inds

    kwargs = {}
    if prominence is not None:
        kwargs["prominence"] = prominence
    if min_dist_samples is not None:
        kwargs["distance"] = int(min_dist_samples)

    # IMPORTANT: Filter out peaks below baseline (height > 0)
    # This matches the auto-detect dialog behavior and prevents detection of
    # negative "peaks" in filtered signals with DC offset or mean subtraction.
    # Breathing signals should only have positive deflections as peaks.
    kwargs["height"] = 0

    pks, _ = find_peaks(y_use, **kwargs)

    # For ML training: return ALL peaks without filtering
    if return_all:
        return pks

    # For backward compatibility: filter by threshold
    if thresh is not None:
        pks = pks[y_use[pks] >= thresh]
    return pks


# ---------- Breath markers ----------

def _zero_crossings_left_indices(x: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Return indices i where sign changes between x[i] and x[i+1].
    We return the *left* index of each crossing.
    eps: treat values with |x| < eps as zero to avoid chatter.
    """
    if eps > 0:
        x = np.where(np.abs(x) < eps, 0.0, x)
    s = np.signbit(x)  # False for >=0, True for <0
    return np.where(s[:-1] != s[1:])[0]


def _first_zc_before(x: np.ndarray, idx: int, left_bound: int, eps: float = 0.0) -> int | None:
    """
    Nearest zero crossing before 'idx' (>= left_bound). Returns the sample
    on the right side of the crossing (i+1), or None if none found.
    """
    zc = _zero_crossings_left_indices(x, eps=eps)
    zc = zc[(zc >= left_bound) & (zc < idx)]
    if zc.size:
        return int(zc[-1] + 1)
    return None


def _first_zc_after(x: np.ndarray, idx: int, right_bound: int, eps: float = 0.0) -> int | None:
    """
    Nearest zero crossing after 'idx' (<= right_bound). Returns the sample
    on the right side of the crossing (i+1), or None if none found.
    """
    zc = _zero_crossings_left_indices(x, eps=eps)
    zc = zc[(zc > idx) & (zc <= right_bound)]
    if zc.size:
        return int(zc[0] + 1)
    return None


#                           peaks_idx: np.ndarray,
#     """
#     Given a processed 1D signal y and inspiratory peaks (indices),
#     compute breath onsets, offsets, and expiratory extrema.
# def compute_breath_events(y: np.ndarray,
#                           direction: str = "up",
#                           eps: float = 0.0) -> Dict[str, np.ndarray]:

#     Rules:
#       - Onset: nearest zero crossing *before* each peak in y; if none found
#         between previous and current peak, fallback to zero crossing in dy/dt.
#       - Offset: nearest zero crossing *after* each peak in y; if none found
#         before the next peak, fallback to zero crossing in dy/dt.
#       - Expiratory "peak": for direction='up' -> minimum of y between peaks;
#                            for direction='down' -> maximum of y between peaks.

#     Returns dict with keys: 'onsets', 'offsets', 'expmins' (np.int arrays).
#     (For direction='down', 'expmins' holds those inter-peak maxima.)
#     """
#                 "offsets": np.array([], dtype=int),
#                 "expmins": np.array([], dtype=int)}
#     peaks_idx = np.asarray(peaks_idx, dtype=int)
#     if peaks_idx.size == 0:
#         return {"onsets": np.array([], dtype=int),

#     y = np.asarray(y, dtype=float)
#     dy = np.gradient(y)

#     onsets, offsets, expmins = [], [], []
#     N = len(y)
#     n = len(peaks_idx)

#     for i, pk in enumerate(peaks_idx):
#         left_bound  = peaks_idx[i-1] if i > 0     else 0
#         right_bound = peaks_idx[i+1] if i < n - 1 else (N - 1)

#         # Onset (before pk)
#         if onset is None:
#         if onset is not None:
#             onsets.append(onset)
#         onset = _first_zc_before(y, pk, left_bound, eps=eps)
#             onset = _first_zc_before(dy, pk, left_bound, eps=eps)

#         # Offset (after pk)
#         if offset is None:
#         if offset is not None:
#             offsets.append(offset)
#         offset = _first_zc_after(y, pk, right_bound, eps=eps)
#             offset = _first_zc_after(dy, pk, right_bound, eps=eps)

#         # Expiratory extremum strictly between pk and next pk
#         if i < n - 1:
#                 if seg.size > 0:
#                     else:
#                     expmins.append(j)
#                         j = int(pk + 1 + np.argmin(seg))  # minima between peaks

#         "onsets":  np.asarray(onsets,  dtype=int),
#         "offsets": np.asarray(offsets, dtype=int),
#         "expmins": np.asarray(expmins, dtype=int),
#     }
#     return {

#     """
#     Given a processed 1D signal y and inspiratory peaks (indices),
#     compute breath onsets, offsets, and expiratory minima.
# def compute_breath_events(y: np.ndarray, peaks_idx: np.ndarray,
#                           sr_hz: float, exclude_sec: float = 0.010) -> Dict[str, np.ndarray]:

#     Rules:
#       - Onset: nearest zero crossing *before* each peak in y; if none found
#         between previous and current peak, fallback to zero crossing in dy/dt,
#         but ignoring ±exclude_sec around the peak.
#       - Offset: nearest zero crossing *after* each peak in y; if none found
#         before the next peak, fallback to zero crossing in dy/dt,
#         but ignoring ±exclude_sec around the peak.
#       - Expiratory minimum: argmin(y) strictly between consecutive peaks.
#     """
#                 "offsets": np.array([], dtype=int),
#                 "expmins": np.array([], dtype=int)}
#     if peaks_idx is None or len(peaks_idx) == 0:
#         return {"onsets": np.array([], dtype=int),


#     if excl_n < 0:
#     # convert exclusion to samples (ensure >= 1 if sr is valid)
#     excl_n = int(round(exclude_sec * sr_hz)) if (sr_hz and sr_hz > 0) else 0
#         excl_n = 0

#     for i, pk in enumerate(peaks_idx):
#         left_bound  = peaks_idx[i-1] if i > 0     else 0
#         right_bound = peaks_idx[i+1] if i < n - 1 else (N - 1)

#         # Onset (raw y)
#         if onset is None:
#             # fallback to dy, but exclude ±excl_n around pk
#         if onset is not None:
#             onsets.append(onset)
#         onset = _first_zc_before(y, pk, left_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             onset = _first_zc_before(dy, pk, left_bound, exclude_lo=lo, exclude_hi=hi)

#         # Offset (raw y)
#         if offset is None:
#         if offset is not None:
#             offsets.append(offset)
#         offset = _first_zc_after(y, pk, right_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             offset = _first_zc_after(dy, pk, right_bound, exclude_lo=lo, exclude_hi=hi)
        
#         # --- Expiratory offsets: first zero-cross in y or d1 after offset, whichever first ---
#             for i in range(ncyc):
#                     continue
#                 # search window: (offset .. next onset)
#                     if k is not None:
#                         expoffs.append(int(k))

#         # Expiratory minimum between pk and next pk
#         if i < n - 1:
#                 if seg.size > 0:
#                     expmins.append(j)
#             nxt = peaks_idx[i + 1]
#             if nxt - pk >= 2:
#                 seg = y[pk + 1:nxt]
#                     j = int(pk + 1 + np.argmin(seg))

#         # "onsets":  np.asarray(onsets,  dtype=int),
#         # "offsets": np.asarray(offsets, dtype=int),
#         # "expmins": np.asarray(expmins, dtype=int),
#         "onsets":  np.asarray(onsets,  dtype=int) if onsets  is not None else np.array([], dtype=int),
#         "offsets": np.asarray(offsets, dtype=int) if offsets is not None else np.array([], dtype=int),
#         "expmins": np.asarray(expmins, dtype=int) if expmins is not None else np.array([], dtype=int),
#         "expoffs": np.asarray(sorted(set(expoffs)), dtype=int),   
#     }
#     return {

#     """
#     Given a processed 1D signal y and inspiratory peaks (indices),
#     compute breath onsets, offsets, expiratory minima, and expiratory offsets.
# def compute_breath_events(y: np.ndarray, peaks_idx: np.ndarray,
#                           sr_hz: float, exclude_sec: float = 0.010) -> Dict[str, np.ndarray]:

#     Rules:
#       - Onset : nearest zero crossing *before* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Offset: nearest zero crossing *after* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Exp min: argmin(y) strictly between consecutive peaks.
#       - Exp offset: earliest of {zero-cross in y, zero-cross in dy/dt} in (offset .. next onset).

#     Returns dict with keys: 'onsets', 'offsets', 'expmins', 'expoffs' (np.int arrays).
#     """
#             "onsets":  np.array([], dtype=int),
#             "offsets": np.array([], dtype=int),
#             "expmins": np.array([], dtype=int),
#             "expoffs": np.array([], dtype=int),
#         }
#     peaks_idx = np.asarray(peaks_idx, dtype=int)
#     if peaks_idx.size == 0:
#         return {

#     y  = np.asarray(y, dtype=float)
#     dy = np.gradient(y)
#     N  = len(y)
#     n  = len(peaks_idx)

#     # Precompute zero-crossings (left indices) once
#     zc_y  = _zero_crossings_left_indices(y)
#     zc_dy = _zero_crossings_left_indices(dy)

#     # Convert exclusion window (for derivative fallback) to samples
#     excl_n = int(round(exclude_sec * sr_hz)) if (sr_hz and sr_hz > 0) else 0
#     excl_n = max(0, excl_n)

#     onsets:  list[int] = []
#     offsets: list[int] = []
#     expmins: list[int] = []

#     # Pass 1: onsets/offsets/expmins in a single O(n) pass
#     for i, pk in enumerate(peaks_idx):
#         left_bound  = peaks_idx[i - 1] if i > 0     else 0
#         right_bound = peaks_idx[i + 1] if i < n - 1 else (N - 1)

#         # Onset (prefer y; fallback to dy/dt with exclusion)
#         if onset is None:
#         if onset is not None:
#             onsets.append(onset)
#         onset = _first_zc_before_from_zc(zc_y, pk, left_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             onset = _first_zc_before_from_zc(zc_dy, pk, left_bound, exclude_lo=lo, exclude_hi=hi)

#         # Offset (prefer y; fallback to dy/dt with exclusion)
#         if offset is None:
#         if offset is not None:
#             offsets.append(offset)
#         offset = _first_zc_after_from_zc(zc_y, pk, right_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             offset = _first_zc_after_from_zc(zc_dy, pk, right_bound, exclude_lo=lo, exclude_hi=hi)

#         # Expiratory minimum strictly between pk and next pk
#         if i < n - 1:
#                 if seg.size:
#                     expmins.append(int(pk + 1 + int(np.argmin(seg))))
#             nxt = peaks_idx[i + 1]
#             if nxt - pk >= 2:
#                 seg = y[pk + 1 : nxt]

#     # Pass 2: expiratory offsets once (no rework inside the loop)
#         # Use the same “either y or d1” rule as before,
#         # but do it once per cycle.
#         for i in range(ncyc):
#                 # # earliest of y-cross or d1-cross
#                 # if cand:
#                 #     expoffs.append(int(min(cand)))
#                 # first zero-cross in y, SECOND zero-cross in dy/dt
#                 if cand:
#                     expoffs.append(int(min(cand)))  # whichever occurs earlier after offset
#                 k_y  = _first_zc_after_from_zc(zc_y,  i_start - 1, i_end)
#                 k_d1 = _kth_zc_after_from_zc(zc_dy, i_start - 1, i_end, k=2)
#                 cand = [k for k in (k_y, k_d1) if k is not None]

#         "onsets":  np.asarray(onsets,  dtype=int),
#         "offsets": np.asarray(offsets, dtype=int),
#         "expmins": np.asarray(expmins, dtype=int),
#         "expoffs": np.asarray(expoffs, dtype=int),
#     }
#     return {


#     """
#     Given a processed 1D signal y and inspiratory peaks (indices),
#     compute breath onsets, offsets, expiratory minima (now via d1 ZC after offset),
#     and expiratory offsets.
# def compute_breath_events(y: np.ndarray, peaks_idx: np.ndarray,
#                           sr_hz: float, exclude_sec: float = 0.010) -> Dict[str, np.ndarray]:

#     Rules:
#       - Onset : nearest zero crossing *before* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Offset: nearest zero crossing *after* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Exp min: FIRST zero crossing of dy/dt after the inspiratory offset
#                  (search window: offset .. next onset).
#       - Exp offset: earliest of {zero-cross in y, SECOND zero-cross in dy/dt}
#                     in (offset .. next onset).

#     Returns dict with keys: 'onsets', 'offsets', 'expmins', 'expoffs' (np.int arrays).
#     """
#             "onsets":  np.array([], dtype=int),
#             "offsets": np.array([], dtype=int),
#             "expmins": np.array([], dtype=int),
#             "expoffs": np.array([], dtype=int),
#         }
#     peaks_idx = np.asarray(peaks_idx, dtype=int)
#     if peaks_idx.size == 0:
#         return {


#     # Precompute zero-crossings (left indices) once
#     zc_y  = _zero_crossings_left_indices(y)
#     zc_dy = _zero_crossings_left_indices(dy)

#     # Convert exclusion window (for derivative fallback) to samples
#     excl_n = int(round(exclude_sec * sr_hz)) if (sr_hz and sr_hz > 0) else 0
#     excl_n = max(0, excl_n)

#     onsets:  list[int] = []
#     offsets: list[int] = []

#     # Pass 1: onsets/offsets
#     for i, pk in enumerate(peaks_idx):
#         left_bound  = peaks_idx[i - 1] if i > 0     else 0
#         right_bound = peaks_idx[i + 1] if i < n - 1 else (N - 1)

#         # Onset (prefer y; fallback to dy/dt with exclusion)
#         if onset is None:
#         if onset is not None:
#             onsets.append(onset)
#         onset = _first_zc_before_from_zc(zc_y, pk, left_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             onset = _first_zc_before_from_zc(zc_dy, pk, left_bound, exclude_lo=lo, exclude_hi=hi)

#         # Offset (prefer y; fallback to dy/dt with exclusion)
#         if offset is None:
#         if offset is not None:
#             offsets.append(offset)
#         offset = _first_zc_after_from_zc(zc_y, pk, right_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             offset = _first_zc_after_from_zc(zc_dy, pk, right_bound, exclude_lo=lo, exclude_hi=hi)

#     # Pass 2: expmins (FIRST d1 ZC after offset) and expoffs (earliest of y-ZC, SECOND d1-ZC)
#     expmins: list[int] = []
#     expoffs: list[int] = []

#     if len(onsets) >= 2 and len(offsets) >= 1:
#         on  = np.asarray(onsets,  dtype=int)
#         off = np.asarray(offsets, dtype=int)
#         ncyc = min(len(off), len(on) - 1)

#         for i in range(ncyc):
#             # Search window: from offset to next onset
#             i_start = max(0, int(off[i]))
#             i_end   = min(int(on[i + 1]), N)  # inclusive right bound behavior handled by helpers

#                 # Expiratory "min": FIRST zero-cross in dy/dt after the offset
#                 if k_d1_first is not None:
#                     expmins.append(int(k_d1_first))
#             if i_end - i_start >= 1:
#                 k_d1_first = _first_zc_after_from_zc(zc_dy, i_start, i_end)

#                 # Expiratory offset: earliest of {y-ZC, SECOND d1-ZC} after the offset
#                 if cand:
#                     expoffs.append(int(min(cand)))
#                 k_y        = _first_zc_after_from_zc( zc_y,  i_start - 1, i_end)
#                 k_d1_second= _kth_zc_after_from_zc(zc_dy, i_start - 1, i_end, k=2)
#                 cand = [k for k in (k_y, k_d1_second) if k is not None]

#         "onsets":  np.asarray(onsets,  dtype=int),
#         "offsets": np.asarray(offsets, dtype=int),
#         "expmins": np.asarray(expmins, dtype=int),
#         "expoffs": np.asarray(expoffs, dtype=int),
#     }
#     return {


#     """
#     Given a processed 1D signal y and inspiratory peaks (indices),
#     compute breath onsets, offsets, expiratory minima (via first d1 ZC after offset),
#     and expiratory offsets.
# def compute_breath_events(y: np.ndarray, peaks_idx: np.ndarray,
#                           sr_hz: float, exclude_sec: float = 0.010) -> Dict[str, np.ndarray]:

#     Rules:
#       - Onset : nearest zero crossing *before* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Offset: nearest zero crossing *after* each peak in y; fallback to dy/dt crossing,
#                 ignoring ±exclude_sec around the peak for the derivative fallback.
#       - Exp min: FIRST zero crossing of dy/dt after the inspiratory offset
#                  (search window: offset .. next onset).
#       - Exp offset: FIRST zero crossing in y after the inspiratory offset
#                     (search window: offset .. next onset).   <-- simplified
#     """
#             "onsets":  np.array([], dtype=int),
#             "offsets": np.array([], dtype=int),
#             "expmins": np.array([], dtype=int),
#             "expoffs": np.array([], dtype=int),
#         }
#     peaks_idx = np.asarray(peaks_idx, dtype=int)
#     if peaks_idx.size == 0:
#         return {

#     y  = np.asarray(y, dtype=float)
#     dy = np.gradient(y)
#     N  = len(y)
#     n  = len(peaks_idx)

#     # Precompute zero-crossings (left indices) once
#     zc_y  = _zero_crossings_left_indices(y)
#     zc_dy = _zero_crossings_left_indices(dy)

#     # Convert exclusion window (for derivative fallback) to samples
#     excl_n = int(round(exclude_sec * sr_hz)) if (sr_hz and sr_hz > 0) else 0
#     excl_n = max(0, excl_n)

#     onsets:  list[int] = []
#     offsets: list[int] = []

#     # Pass 1: onsets/offsets
#     for i, pk in enumerate(peaks_idx):
#         left_bound  = peaks_idx[i - 1] if i > 0     else 0
#         right_bound = peaks_idx[i + 1] if i < n - 1 else (N - 1)

#         # Onset (prefer y; fallback to dy/dt with exclusion)
#         if onset is None:
#         if onset is not None:
#             onsets.append(onset)
#         onset = _first_zc_before_from_zc(zc_y, pk, left_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             onset = _first_zc_before_from_zc(zc_dy, pk, left_bound, exclude_lo=lo, exclude_hi=hi)

#         # Offset (prefer y; fallback to dy/dt with exclusion)
#         if offset is None:
#         if offset is not None:
#             offsets.append(offset)
#         offset = _first_zc_after_from_zc(zc_y, pk, right_bound)
#             lo = max(left_bound,  pk - excl_n)
#             hi = min(right_bound, pk + excl_n)
#             offset = _first_zc_after_from_zc(zc_dy, pk, right_bound, exclude_lo=lo, exclude_hi=hi)

#     # Pass 2: expmins (FIRST d1 ZC after offset) and expoffs (FIRST y ZC after offset)
#     expmins: list[int] = []
#     expoffs: list[int] = []

#     if len(onsets) >= 2 and len(offsets) >= 1:
#         on  = np.asarray(onsets,  dtype=int)
#         off = np.asarray(offsets, dtype=int)
#         ncyc = min(len(off), len(on) - 1)

#         for i in range(ncyc):
#             # Search window: from offset to next onset
#             i_start = max(0, int(off[i]))
#             i_end   = min(int(on[i + 1]), N)

#                 # Exp "min": FIRST zero-cross in dy/dt after the offset (unchanged)
#                 if k_d1_first is not None:
#                     expmins.append(int(k_d1_first))
#             if i_end - i_start >= 1:
#                 k_d1_first = _first_zc_after_from_zc(zc_dy, i_start, i_end)

#                 # Expiratory offset: FIRST zero-cross in y after the offset (simplified)
#                 if k_y is not None:
#                     expoffs.append(int(k_y))
#                 k_y = _first_zc_after_from_zc(zc_y, i_start - 1, i_end)

#         "onsets":  np.asarray(onsets,  dtype=int),
#         "offsets": np.asarray(offsets, dtype=int),
#         "expmins": np.asarray(expmins, dtype=int),
#         "expoffs": np.asarray(expoffs, dtype=int),
#     }
#     return {

def compute_breath_events(y: np.ndarray, peaks_idx: np.ndarray,
                          sr_hz: float, exclude_sec: float = 0.005) -> Dict[str, np.ndarray]:
    """
    ROBUST breath event detection with multiple fallback strategies.

    Given a processed 1D signal y and inspiratory peaks (indices),
    compute breath onsets, offsets, expiratory minima (via FIRST d1 ZC after offset),
    and expiratory offsets (FIRST y ZC after offset but strictly before next peak; else trough between peaks).

    ROBUSTNESS FEATURES:
    - Multiple fallback strategies for each event type
    - Protection against edge effects (peaks near start/end)
    - Graceful handling of noisy or irregular signals
    - Consistent array lengths in output

    Rules:
      - Onset : nearest zero crossing *before* each peak in y; fallback to dy/dt crossing,
                then to fixed fraction of inter-peak distance if needed.
      - Offset: nearest zero crossing *after* each peak in y; fallback to dy/dt crossing,
                then to fixed fraction of inter-peak distance if needed.
      - Exp min: FIRST zero crossing of dy/dt after the inspiratory offset
                 (search window: offset .. next onset); fallback to actual minimum.
      - Exp offset: ENHANCED dual method - finds both:
                    1) First y zero crossing after offset (before next peak)
                    2) First dy zero crossing after expiratory peak meeting 50% amplitude threshold
                    Then selects whichever occurs EARLIER (smaller index).
                    Fallback to minimum between peaks if both methods fail.
    """
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    if peaks_idx.size == 0:
        return {
            "onsets":  np.array([], dtype=int),
            "offsets": np.array([], dtype=int),
            "expmins": np.array([], dtype=int),
            "expoffs": np.array([], dtype=int),
        }

    y  = np.asarray(y, dtype=float)
    dy = np.gradient(y)
    N  = len(y)
    n  = len(peaks_idx)

    # Precompute zero-crossings (left indices) once
    zc_y  = _zero_crossings_left_indices(y)
    zc_dy = _zero_crossings_left_indices(dy)

    # Convert exclusion window (for derivative fallback) to samples
    excl_n = int(round(exclude_sec * sr_hz)) if (sr_hz and sr_hz > 0) else 0
    excl_n = max(0, excl_n)

    onsets:  list[int] = []
    offsets: list[int] = []

    # Track which detection methods were used for quality assessment
    onset_methods: list[str] = []   # Track fallback usage
    offset_methods: list[str] = []  # Track fallback usage

    # Pass 1: onsets/offsets with robust fallbacks
    for i, pk in enumerate(peaks_idx):
        left_bound  = peaks_idx[i - 1] if i > 0     else 0
        right_bound = peaks_idx[i + 1] if i < n - 1 else (N - 1)

        # ROBUST ONSET DETECTION with multiple fallbacks
        onset = None
        onset_method = "failed"

        # Method 1: Zero crossing in y signal
        onset = _first_zc_before_from_zc(zc_y, pk, left_bound)
        if onset is not None:
            onset_method = "signal_zc"

        # Method 2: Zero crossing in dy/dt (with exclusion around peak)
        if onset is None:
            lo = max(left_bound,  pk - excl_n)
            hi = min(right_bound, pk + excl_n)
            onset = _first_zc_before_from_zc(zc_dy, pk, left_bound, exclude_lo=lo, exclude_hi=hi)
            if onset is not None:
                onset_method = "derivative_zc"

        # Method 3: Fixed fraction fallback (25% back from peak)
        if onset is None:
            dist_back = max(1, int(0.25 * (pk - left_bound)))
            onset = max(left_bound, pk - dist_back)
            onset_method = "fraction_fallback"

        # Method 4: Last resort - use left bound + small offset
        if onset is None:
            onset = min(pk - 1, left_bound + 1)
            onset_method = "boundary_fallback"

        if onset is not None:
            onsets.append(max(0, min(onset, N - 1)))  # Ensure valid bounds
            onset_methods.append(onset_method)

        # ROBUST OFFSET DETECTION with multiple fallbacks
        # IMPORTANT: Offset must occur BEFORE next peak (not just right_bound)
        offset = None
        offset_method = "failed"

        # Determine search bound: use next peak if available, otherwise right_bound
        pk_next = int(peaks_idx[i + 1]) if i < n - 1 else right_bound
        search_bound = min(pk_next, right_bound)

        # Method 1: Zero crossing in y signal (must be before next peak)
        offset = _first_zc_after_from_zc(zc_y, pk, search_bound)
        if offset is not None and offset < pk_next:
            offset_method = "signal_zc"
        else:
            offset = None

        # Method 2: Zero crossing in dy/dt (with exclusion around peak, must be before next peak)
        if offset is None:
            lo = max(left_bound,  pk - excl_n)
            hi = min(right_bound, pk + excl_n)
            offset = _first_zc_after_from_zc(zc_dy, pk, search_bound, exclude_lo=lo, exclude_hi=hi)
            if offset is not None and offset < pk_next:
                offset_method = "derivative_zc"
            else:
                offset = None

        # Method 3: Use actual minimum between this peak and next peak
        if offset is None and pk_next is not None and pk_next - pk >= 2:
            seg = y[pk + 1:pk_next]
            if seg.size > 0:
                offset = int(pk + 1 + np.argmin(seg))
                offset_method = "minimum_fallback"

        # Method 4: Fixed fraction fallback (25% forward from peak, before next peak)
        if offset is None:
            dist_forward = max(1, int(0.25 * (pk_next - pk)))
            offset = min(pk_next - 1, pk + dist_forward)
            offset_method = "fraction_fallback"

        # Method 5: Last resort - use position before next peak
        if offset is None:
            offset = max(pk + 1, pk_next - 1)
            offset_method = "boundary_fallback"

        if offset is not None:
            offsets.append(max(0, min(offset, N - 1)))  # Ensure valid bounds
            offset_methods.append(offset_method)

    # Pass 2: Robust expiratory event detection
    expmins: list[int] = []
    expoffs: list[int] = []

    # Track expiratory detection methods
    expmin_methods: list[str] = []   # Track expiratory minimum detection methods
    expoff_methods: list[str] = []   # Track expiratory offset detection methods

    # Ensure we have enough data for meaningful calculations
    if len(onsets) >= 1 and len(offsets) >= 1:
        on  = np.asarray(onsets,  dtype=int)
        off = np.asarray(offsets, dtype=int)

        # Work with available data, don't require perfect matching
        ncyc = min(len(off), len(onsets), len(peaks_idx))
        if len(onsets) > 1:
            ncyc = min(ncyc, len(onsets) - 1)

        for i in range(ncyc):
            pk_curr = int(peaks_idx[i]) if i < len(peaks_idx) else None
            pk_next = int(peaks_idx[i + 1]) if i + 1 < len(peaks_idx) else None

            # ROBUST EXPIRATORY MINIMUM DETECTION
            # IMPORTANT: Expiratory minimum must occur BEFORE next peak
            exp_min_idx = None

            if i < len(off):
                i_start = max(0, int(off[i]))

                # Define search window end: must be before next peak
                if pk_next is not None:
                    # Search window: inspiratory offset to next peak (exclusive)
                    i_end_search = min(pk_next, N)
                elif i + 1 < len(on):
                    i_end_search = min(int(on[i + 1]), N)
                else:
                    i_end_search = min(i_start + int(2.0 * sr_hz), N)  # Default: 2 seconds ahead

                if i_end_search - i_start >= 1:
                    # Method 1: First dy/dt zero crossing after offset (must be before next peak)
                    # AND must be lower than both current and next peak
                    k_d1_first = _first_zc_after_from_zc(zc_dy, i_start, i_end_search)
                    if k_d1_first is not None and (pk_next is None or k_d1_first < pk_next):
                        # Additional constraint: value at this point must be lower than both peaks
                        valid = True
                        if pk_curr is not None and k_d1_first < len(y):
                            if y[k_d1_first] >= y[pk_curr]:
                                valid = False
                        if pk_next is not None and k_d1_first < len(y):
                            if y[k_d1_first] >= y[pk_next]:
                                valid = False

                        if valid:
                            exp_min_idx = int(k_d1_first)

                    # Method 2: Fallback to actual minimum in the search window (before next peak)
                    if exp_min_idx is None and pk_curr is not None and pk_next is not None:
                        if pk_next - pk_curr >= 2:
                            seg_start = max(i_start, pk_curr + 1)
                            seg_end = min(pk_next, i_end_search)  # Ensure we stay before next peak
                            if seg_end > seg_start:
                                seg = y[seg_start:seg_end]
                                if seg.size > 0:
                                    exp_min_idx = int(seg_start + np.argmin(seg))

                    # Method 3: Last resort - midpoint between offset and next peak
                    if exp_min_idx is None:
                        if pk_next is not None:
                            midpoint = i_start + (pk_next - i_start) // 2
                        else:
                            midpoint = i_start + (i_end_search - i_start) // 2
                        exp_min_idx = max(0, min(midpoint, N - 1))

                if exp_min_idx is not None:
                    expmins.append(exp_min_idx)

            # ROBUST EXPIRATORY OFFSET DETECTION - RESTORED ORIGINAL ALGORITHM
            exp_offset_idx = None

            if i < len(off) and pk_curr is not None:
                i_start = max(0, int(off[i]))

                # Find the expiratory peak (minimum amplitude) for this cycle first
                exp_peak_idx = None
                exp_peak_val = None
                if pk_next is not None and pk_next - pk_curr >= 2:
                    exp_peak_idx = int(pk_curr + 1 + int(np.argmin(y[pk_curr + 1:pk_next])))
                    exp_peak_val = y[exp_peak_idx]

                # Collect BOTH candidate methods, then choose the EARLIER one
                candidates = []

                # Candidate 1: First y ZC after offset but before next peak
                if pk_next is not None:
                    i_end_peak = min(pk_next, N)
                    k_y = _first_zc_after_from_zc(zc_y, i_start - 1, i_end_peak)
                    if k_y is not None and k_y < pk_next:
                        candidates.append(k_y)

                # Candidate 2: First dy ZC after expiratory peak (plus padding) with amplitude constraint
                if exp_peak_idx is not None and exp_peak_val is not None:
                    # Add small padding after expiratory peak to avoid finding the peak itself
                    padding_samples = max(1, int(0.010 * sr_hz))  # 10ms padding
                    search_start = max(i_start, exp_peak_idx + padding_samples)

                    # Define search window end
                    if i + 1 < len(on):
                        i_end_onset = min(int(on[i + 1]), N)
                    else:
                        i_end_onset = min(pk_next, N) if pk_next else N

                    # For negative expiratory peaks, we want y values closer to zero (less negative)
                    # Set threshold as 50% of the way from expiratory peak toward zero
                    if exp_peak_val < 0:
                        threshold_closer_to_zero = 0.5 * exp_peak_val  # 50% of negative value = closer to zero
                    else:
                        threshold_closer_to_zero = 0.5 * exp_peak_val  # 50% of positive value

                    # Find dy zero crossings after expiratory peak + padding and before next onset
                    dy_crossings = zc_dy[(zc_dy >= search_start - 1) & (zc_dy <= i_end_onset)]

                    for zc_idx in dy_crossings:
                        crossing_point = int(zc_idx + 1)  # Right side of crossing
                        if crossing_point < len(y):
                            y_val = y[crossing_point]
                            # Check if y value is closer to zero than the threshold
                            if (exp_peak_val < 0 and y_val > threshold_closer_to_zero) or \
                               (exp_peak_val >= 0 and y_val < threshold_closer_to_zero):
                                candidates.append(crossing_point)
                                break  # Take the first one that meets criteria

                # ORIGINAL LOGIC: Choose the EARLIER of the two valid candidates
                if candidates:
                    exp_offset_idx = int(min(candidates))  # Earlier time point
                # Fallback 1: Use actual minimum between peaks
                elif pk_curr is not None and pk_next is not None and pk_next - pk_curr >= 2:
                    exp_offset_idx = int(pk_curr + 1 + int(np.argmin(y[pk_curr + 1:pk_next])))
                # Fallback 2: Last resort - use a reasonable fraction after expiratory minimum
                elif i < len(expmins):
                    exp_min_pos = expmins[-1] if expmins else i_start
                    if pk_next is not None:
                        remaining_dist = pk_next - exp_min_pos
                        exp_offset_idx = min(exp_min_pos + max(1, remaining_dist // 3), N - 1)
                    else:
                        exp_offset_idx = min(exp_min_pos + int(0.5 * sr_hz), N - 1)  # 0.5s ahead

                if exp_offset_idx is not None:
                    expoffs.append(max(0, min(exp_offset_idx, N - 1)))

    # Ensure all arrays have consistent, reasonable lengths
    # If we have peaks but no breath events, create minimal fallbacks
    if len(peaks_idx) > 0 and len(onsets) == 0:
        # Emergency fallback: create basic onsets/offsets
        for i, pk in enumerate(peaks_idx):
            onset = max(0, pk - max(1, int(0.3 * sr_hz)))  # 0.3s before peak
            offset = min(N - 1, pk + max(1, int(0.3 * sr_hz)))  # 0.3s after peak
            onsets.append(onset)
            offsets.append(offset)

    # Track detection quality for error highlighting
    detection_quality = []
    for i in range(len(peaks_idx)):
        quality = {
            "peak_idx": i,
            "onset_method": "signal_zc" if i < len(onsets) else "failed",
            "offset_method": "signal_zc" if i < len(offsets) else "failed",
            "expmin_method": "derivative_zc" if i < len(expmins) else "failed",
            "expoff_method": "dual_candidate" if i < len(expoffs) else "failed",
            "severity": "good"  # Will be updated based on fallback usage
        }
        detection_quality.append(quality)

    return {
        "onsets":  np.asarray(onsets,  dtype=int),
        "offsets": np.asarray(offsets, dtype=int),
        "expmins": np.asarray(expmins, dtype=int),
        "expoffs": np.asarray(expoffs, dtype=int),
        "quality": detection_quality,  # NEW: Quality tracking for error highlighting
    }



#     y: np.ndarray, sr_hz: float,
# ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
#     """
#     Convenience: run peak detection then compute breath events.
#     """
#         y=y, sr_hz=sr_hz,
#         thresh=thresh,
#         prominence=prominence,
#         min_dist_samples=min_dist_samples,
#         direction=direction,
#     )
#     pks = detect_peaks(
#     breaths = compute_breath_events(y, pks, direction=direction)
#     return pks, breaths


def detect_peaks_and_breaths(
    y: np.ndarray, sr_hz: float,
    thresh: float | None = None,
    prominence: float | None = None,
    min_dist_samples: int | None = None,
    direction: str = "up",
    exclude_sec: float = 0.030,  # ignore ±this window around pk for derivative fallback
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    pks = detect_peaks(
        y=y, sr_hz=sr_hz,
        thresh=thresh,
        prominence=prominence,
        min_dist_samples=min_dist_samples,
        direction=direction,
    )

    breaths = compute_breath_events(y, pks, sr_hz=sr_hz, exclude_sec=exclude_sec)
    return pks, breaths


def label_peaks_by_threshold(
    y: np.ndarray,
    peak_indices: np.ndarray,
    thresh: float | None = None,
    direction: str = "up"
) -> Dict[str, np.ndarray]:
    """
    Label detected peaks as breath (1) or noise (0) based on threshold.

    Args:
        y: Signal data
        peak_indices: Already-detected peak indices
        thresh: Height threshold for labeling (peaks >= thresh are breaths)
        direction: 'up' or 'down'

    Returns dictionary with:
        'indices': Peak indices (same as input)
        'labels': Binary labels (1 = breath, 0 = noise)
        'label_source': Array of 'auto' for each peak
        'prominences': Prominence values for each peak
    """
    from scipy.signal import peak_prominences

    if len(peak_indices) == 0:
        return {
            'indices': np.array([], dtype=int),
            'labels': np.array([], dtype=int),
            'label_source': np.array([], dtype=object),
            'prominences': np.array([], dtype=float)
        }

    y_use = -y if direction == "down" else y

    # Compute prominences for all peaks
    proms = peak_prominences(y_use, peak_indices)[0]

    # Auto-label based on threshold
    if thresh is not None:
        peak_heights = y_use[peak_indices]
        labels = (peak_heights >= thresh).astype(int)
    else:
        # No threshold: all peaks labeled as breaths
        labels = np.ones(len(peak_indices), dtype=int)

    # All labels initially from auto-detection
    label_source = np.array(['auto'] * len(peak_indices), dtype=object)

    return {
        'indices': peak_indices.copy(),
        'labels': labels,
        'label_source': label_source,
        'prominences': proms
    }


def _first_zc_before(x: np.ndarray, idx: int, left_bound: int,
                     exclude_lo: int | None = None, exclude_hi: int | None = None) -> int | None:
    """
    Nearest zero crossing before 'idx' (>= left_bound). Returns the sample index
    on the right side of the crossing (i+1). If exclude_* are given, crossings
    whose right index (i+1) falls inside [exclude_lo, exclude_hi] are ignored.
    """
    zc = _zero_crossings_left_indices(x)  # left indices of crossings
    mask = (zc >= left_bound) & (zc < idx)
    if exclude_lo is not None and exclude_hi is not None:
        right_idx = zc + 1
        mask &= ~((right_idx >= exclude_lo) & (right_idx <= exclude_hi))
    zc = zc[mask]
    if zc.size:
        return int(zc[-1] + 1)
    return None


def _first_zc_after(x: np.ndarray, idx: int, right_bound: int,
                    exclude_lo: int | None = None, exclude_hi: int | None = None) -> int | None:
    """
    Nearest zero crossing after 'idx' (<= right_bound). Returns the sample index
    on the right side of the crossing (i+1). If exclude_* are given, crossings
    whose right index (i+1) falls inside [exclude_lo, exclude_hi] are ignored.
    """
    zc = _zero_crossings_left_indices(x)
    mask = (zc > idx) & (zc <= right_bound)
    if exclude_lo is not None and exclude_hi is not None:
        right_idx = zc + 1
        mask &= ~((right_idx >= exclude_lo) & (right_idx <= exclude_hi))
    zc = zc[mask]
    if zc.size:
        return int(zc[0] + 1)
    return None

# --- zero-crossing helpers ---
def _first_zero_cross(y: np.ndarray, i_start: int, i_end: int) -> int | None:
    """
    Return the **sample index** of the first zero crossing in y between
    [i_start+1, i_end) (inclusive start+1, exclusive end). If y equals 0 at any
    sample first, that index is returned. None if no crossing.
    """
    i_start = int(max(0, i_start))
    i_end   = int(min(len(y), i_end))
    if i_end - i_start < 2:
        return None

    seg  = y[i_start:i_end]
    # exact zero first?
    z = np.where(seg == 0.0)[0]
    if z.size:
        return i_start + int(z[0])

    sgn = np.sign(seg)
    prod = sgn[:-1] * sgn[1:]
    # sign change across adjacent samples -> crossing between k and k+1; return k+1
    k = np.where(prod < 0)[0]
    if k.size:
        return i_start + int(k[0] + 1)
    return None


def _first_zero_cross_either(y: np.ndarray, i_start: int, i_end: int, sr_hz: float) -> int | None:
    """
    Earliest of: zero-cross in y OR zero-cross in dy/dt, in (i_start, i_end).
    """
    iz_y  = _first_zero_cross(y, i_start, i_end)
    # dy/dt using uniform dt=1/sr (good enough here)
    # d1 = np.gradient(y, 1.0 / float(sr_hz)) if sr_hz else np.gradient(y)
    d1 = _d1_lowpass20(y, sr_hz)
    iz_d = _first_zero_cross(d1, i_start, i_end)
    cands = [i for i in (iz_y, iz_d) if i is not None]
    return min(cands) if cands else None



def _first_zc_before_from_zc(zc_left: np.ndarray, idx: int, left_bound: int,
                             exclude_lo: int | None = None, exclude_hi: int | None = None) -> int | None:
    """
    zc_left are LEFT indices where a sign change occurs in x (i.e., crossing between k and k+1).
    Return the right index (k+1) of the last crossing in [left_bound, idx).
    """
    mask = (zc_left >= left_bound) & (zc_left < idx)
    if exclude_lo is not None and exclude_hi is not None:
        right_idx = zc_left + 1
        mask &= ~((right_idx >= exclude_lo) & (right_idx <= exclude_hi))
    zc = zc_left[mask]
    if zc.size:
        return int(zc[-1] + 1)
    return None


def _first_zc_after_from_zc(zc_left: np.ndarray, idx: int, right_bound: int,
                            exclude_lo: int | None = None, exclude_hi: int | None = None) -> int | None:
    """
    Return the right index (k+1) of the first crossing in (idx, right_bound].
    """
    mask = (zc_left > idx) & (zc_left <= right_bound)
    if exclude_lo is not None and exclude_hi is not None:
        right_idx = zc_left + 1
        mask &= ~((right_idx >= exclude_lo) & (right_idx <= exclude_hi))
    zc = zc_left[mask]
    if zc.size:
        return int(zc[0] + 1)
    return None

def _kth_zc_after_from_zc(zc_left: np.ndarray, idx: int, right_bound: int, k: int = 1,
                          exclude_lo: int | None = None, exclude_hi: int | None = None) -> int | None:
    """
    Return the right index (k+1) of the k-th crossing in (idx, right_bound].
    zc_left are LEFT indices of zero crossings (between k and k+1).
    """
    mask = (zc_left > idx) & (zc_left <= right_bound)
    if exclude_lo is not None and exclude_hi is not None:
        right_idx = zc_left + 1
        mask &= ~((right_idx >= exclude_lo) & (right_idx <= exclude_hi))
    cand = zc_left[mask]
    if cand.size >= k:
        return int(cand[k - 1] + 1)
    return None

# add near the top of peaks.py, after existing imports
try:
    from core import filters as core_filters
except Exception:
    core_filters = None

try:
    from scipy.signal import butter, filtfilt
except Exception:
    butter = filtfilt = None


def _d1_lowpass20(y: np.ndarray, sr_hz: float, cutoff_hz: float = 20.0) -> np.ndarray:
    """
    Always compute dy/dt from a 20 Hz low-passed version of y.
    - Prefers core.filters.apply_all_1d (LP only)
    - Else uses SciPy butter+filtfilt if present
    - Else falls back to a moving-average
    """
    y = np.asarray(y, dtype=float)

    if sr_hz and sr_hz > 0:
        # 1) Prefer the app's filter pipeline (LP only)
        if core_filters is not None:
            try:
                y_lp = core_filters.apply_all_1d(
                    y, sr_hz,
                    True,  cutoff_hz,   # use_low, low_hz
                    False, None,        # use_high, high_hz
                    False, 0.0,         # use_mean_sub, mean_val
                    False               # use_invert
                )
            except Exception:
                y_lp = y  # safe fallback
        # 2) SciPy fallback
        elif butter is not None and filtfilt is not None:
            nyq = 0.5 * float(sr_hz)
            wn = min(0.99, float(cutoff_hz) / nyq)
            b, a = butter(4, wn, btype="low")
            y_lp = filtfilt(b, a, y, method="pad")
        # 3) Simple moving-average fallback
        else:
            w = max(1, int(round(float(sr_hz) / float(cutoff_hz))))
            if w > 1:
                y_lp = np.convolve(y, np.ones(w, dtype=float)/w, mode="same")
            else:
                y_lp = y

        # dy/dt (time-scaled gradient; scaling doesn’t affect ZCs but keeps units right)
        return np.gradient(y_lp, 1.0/float(sr_hz))

    # No sampling rate — unscaled gradient (ZCs still valid)
    return np.gradient(y)


def compute_peak_candidate_metrics(y: np.ndarray,
                                   all_peak_indices: np.ndarray,
                                   breath_events: Dict[str, np.ndarray],
                                   sr_hz: float,
                                   p_noise: np.ndarray = None,
                                   p_breath: np.ndarray = None) -> list[dict]:
    """
    Compute comprehensive metrics for each peak to enable ML-driven
    merge detection, noise classification, and sigh detection.

    NO hardcoded rules - just collect the data for ML to learn from!

    Args:
        y: Processed signal (1D array)
        all_peak_indices: All detected peaks (including noise)
        breath_events: Dict with 'onsets', 'offsets', 'expmins', 'expoffs'
        sr_hz: Sampling rate
        p_noise: Optional P(noise) probability for each peak
        p_breath: Optional P(breath) probability for each peak

    Returns:
        List of dicts, one per peak, with comprehensive metrics
    """
    all_peak_indices = np.asarray(all_peak_indices, dtype=int)
    y = np.asarray(y, dtype=float)

    if len(all_peak_indices) == 0:
        return []

    # Extract breath event arrays
    onsets = breath_events.get('onsets', np.array([]))
    offsets = breath_events.get('offsets', np.array([]))
    expmins = breath_events.get('expmins', np.array([]))

    # Pre-compute gap statistics for normalization
    all_gaps = np.diff(all_peak_indices) / sr_hz if len(all_peak_indices) > 1 else np.array([])
    median_gap = np.median(all_gaps) if len(all_gaps) > 0 else 1.0

    # Pre-compute amplitude statistics for normalization
    peak_amplitudes = y[all_peak_indices]
    median_amplitude = np.median(peak_amplitudes) if len(peak_amplitudes) > 0 else 1.0

    metrics = []

    for i, pk_idx in enumerate(all_peak_indices):
        # Basic features
        peak_amplitude = y[pk_idx]

        # Left/right neighbors
        prev_pk = all_peak_indices[i-1] if i > 0 else None
        next_pk = all_peak_indices[i+1] if i < len(all_peak_indices)-1 else None

        # === TIMING METRICS ===

        # Gap to previous peak (absolute seconds)
        gap_to_prev = (pk_idx - prev_pk) / sr_hz if prev_pk is not None else None

        # Gap to next peak (absolute seconds)
        gap_to_next = (next_pk - pk_idx) / sr_hz if next_pk is not None else None

        # NORMALIZED gap metrics (relative to recording distribution)
        gap_to_prev_normalized = gap_to_prev / median_gap if gap_to_prev is not None else None
        gap_to_next_normalized = gap_to_next / median_gap if gap_to_next is not None else None

        # === TROUGH METRICS (for merge detection) ===

        # Trough depth to PREVIOUS peak
        trough_depth_prev = None
        trough_ratio_prev = None
        trough_value_prev = None
        if prev_pk is not None:
            trough_idx_prev = prev_pk + np.argmin(y[prev_pk:pk_idx])
            trough_value_prev = y[trough_idx_prev]
            trough_depth_prev = min(y[prev_pk], y[pk_idx]) - trough_value_prev
            combined_amplitude = (y[prev_pk] + y[pk_idx]) / 2
            trough_ratio_prev = trough_depth_prev / combined_amplitude if combined_amplitude > 0 else 0

        # Trough depth to NEXT peak
        trough_depth_next = None
        trough_ratio_next = None
        trough_value_next = None
        if next_pk is not None:
            trough_idx_next = pk_idx + np.argmin(y[pk_idx:next_pk])
            trough_value_next = y[trough_idx_next]
            trough_depth_next = min(y[pk_idx], y[next_pk]) - trough_value_next
            combined_amplitude = (y[pk_idx] + y[next_pk]) / 2
            trough_ratio_next = trough_depth_next / combined_amplitude if combined_amplitude > 0 else 0

        # === PROMINENCE ASYMMETRY ===

        # Left prominence (relative to previous trough)
        left_prominence = None
        if prev_pk is not None:
            left_trough = np.min(y[prev_pk:pk_idx])
            left_prominence = y[pk_idx] - left_trough

        # Right prominence (relative to next trough)
        right_prominence = None
        if next_pk is not None:
            right_trough = np.min(y[pk_idx:next_pk])
            right_prominence = y[pk_idx] - right_trough

        # Asymmetry ratio (0 = highly asymmetric, 1 = symmetric)
        prom_asymmetry = None
        if left_prominence is not None and right_prominence is not None and max(left_prominence, right_prominence) > 0:
            prom_asymmetry = min(left_prominence, right_prominence) / max(left_prominence, right_prominence)

        # === ONSET POSITION (strong indicator of shoulder peak) ===

        # Does the onset start above zero?
        onset_idx = onsets[i] if i < len(onsets) else None
        onset_value = y[onset_idx] if onset_idx is not None else None
        onset_above_zero = onset_value > 0 if onset_value is not None else None

        # How far above zero? (normalized by peak amplitude)
        onset_height_ratio = onset_value / peak_amplitude if (onset_value is not None and peak_amplitude > 0) else None

        # === AMPLITUDE METRICS (for noise detection) ===

        # Absolute amplitude
        amp_absolute = peak_amplitude

        # Normalized amplitude (relative to recording median)
        amp_normalized = peak_amplitude / median_amplitude if median_amplitude > 0 else 0

        # === BREATH EVENT METRICS ===

        # Get offset for this breath
        offset_idx = offsets[i] if i < len(offsets) else None
        offset_value = y[offset_idx] if offset_idx is not None else None

        # Get expiratory minimum for this breath
        expmin_idx = expmins[i] if i < len(expmins) else None
        expmin_value = y[expmin_idx] if expmin_idx is not None else None

        # Inspiratory time (onset to peak)
        ti = (pk_idx - onset_idx) / sr_hz if onset_idx is not None else None

        # Expiratory time (offset to next onset)
        next_onset_idx = onsets[i+1] if i+1 < len(onsets) else None
        te = (next_onset_idx - offset_idx) / sr_hz if (offset_idx is not None and next_onset_idx is not None) else None

        # === NEIGHBOR COMPARISON FEATURES (for merge and sigh detection) ===

        # Neighbor amplitudes
        next_peak_amplitude = y[next_pk] if next_pk is not None else None
        prev_peak_amplitude = y[prev_pk] if prev_pk is not None else None

        # Amplitude ratios (who's bigger?)
        amplitude_ratio_to_next = peak_amplitude / next_peak_amplitude if next_peak_amplitude and next_peak_amplitude > 0 else None
        amplitude_ratio_to_prev = peak_amplitude / prev_peak_amplitude if prev_peak_amplitude and prev_peak_amplitude > 0 else None

        # Signed amplitude differences (-1 to +1, symmetric)
        amplitude_diff_to_next_signed = None
        if next_peak_amplitude is not None and (peak_amplitude + next_peak_amplitude) > 0:
            amplitude_diff_to_next_signed = (peak_amplitude - next_peak_amplitude) / (peak_amplitude + next_peak_amplitude)

        amplitude_diff_to_prev_signed = None
        if prev_peak_amplitude is not None and (peak_amplitude + prev_peak_amplitude) > 0:
            amplitude_diff_to_prev_signed = (peak_amplitude - prev_peak_amplitude) / (peak_amplitude + prev_peak_amplitude)

        # Total prominence (left + right, for neighbor comparisons)
        total_prominence = None
        if left_prominence is not None and right_prominence is not None:
            total_prominence = (left_prominence + right_prominence) / 2  # Average of both sides

        # Neighbor prominences (need to compute for neighbors)
        next_peak_prominence = None
        if next_pk is not None and i+1 < len(all_peak_indices)-1:
            # Next peak's left prominence (from current to next)
            next_left_trough = np.min(y[pk_idx:next_pk])
            next_left_prom = y[next_pk] - next_left_trough
            # Next peak's right prominence (from next to one after)
            next_next_pk = all_peak_indices[i+2] if i+2 < len(all_peak_indices) else None
            if next_next_pk is not None:
                next_right_trough = np.min(y[next_pk:next_next_pk])
                next_right_prom = y[next_pk] - next_right_trough
                next_peak_prominence = (next_left_prom + next_right_prom) / 2
            else:
                next_peak_prominence = next_left_prom  # Only left side available

        prev_peak_prominence = None
        if prev_pk is not None and i > 1:
            # Prev peak's right prominence (from prev to current)
            prev_right_trough = np.min(y[prev_pk:pk_idx])
            prev_right_prom = y[prev_pk] - prev_right_trough
            # Prev peak's left prominence (from one before to prev)
            prev_prev_pk = all_peak_indices[i-2] if i >= 2 else None
            if prev_prev_pk is not None:
                prev_left_trough = np.min(y[prev_prev_pk:prev_pk])
                prev_left_prom = y[prev_pk] - prev_left_trough
                prev_peak_prominence = (prev_left_prom + prev_right_prom) / 2
            else:
                prev_peak_prominence = prev_right_prom  # Only right side available

        # Prominence ratios
        prominence_ratio_to_next = total_prominence / next_peak_prominence if (total_prominence is not None and next_peak_prominence and next_peak_prominence > 0) else None
        prominence_ratio_to_prev = total_prominence / prev_peak_prominence if (total_prominence is not None and prev_peak_prominence and prev_peak_prominence > 0) else None

        # SIGNED prominence asymmetry (replace unsigned version)
        prom_asymmetry_signed = None
        if left_prominence is not None and right_prominence is not None and (left_prominence + right_prominence) > 0:
            prom_asymmetry_signed = (left_prominence - right_prominence) / (left_prominence + right_prominence)

        # SIGNED trough asymmetry
        trough_asymmetry_signed = None
        if trough_depth_prev is not None and trough_depth_next is not None and (trough_depth_prev + trough_depth_next) > 0:
            trough_asymmetry_signed = (trough_depth_prev - trough_depth_next) / (trough_depth_prev + trough_depth_next)

        # Neighbor onset height ratios (critical for merge detection!)
        next_peak_onset_height_ratio = None
        if i+1 < len(onsets):
            next_onset_idx = onsets[i+1]
            next_onset_value = y[next_onset_idx] if next_onset_idx is not None else None
            if next_onset_value is not None and next_peak_amplitude and next_peak_amplitude > 0:
                next_peak_onset_height_ratio = next_onset_value / next_peak_amplitude

        prev_peak_onset_height_ratio = None
        if i > 0 and i-1 < len(onsets):
            prev_onset_idx = onsets[i-1]
            prev_onset_value = y[prev_onset_idx] if prev_onset_idx is not None else None
            if prev_onset_value is not None and prev_peak_amplitude and prev_peak_amplitude > 0:
                prev_peak_onset_height_ratio = prev_onset_value / prev_peak_amplitude

        # Neighbor Ti values
        next_peak_ti = None
        if i+1 < len(onsets) and next_pk is not None:
            next_onset_idx = onsets[i+1]
            if next_onset_idx is not None:
                next_peak_ti = (next_pk - next_onset_idx) / sr_hz

        prev_peak_ti = None
        if i > 0 and prev_pk is not None:
            prev_onset_idx = onsets[i-1]
            if prev_onset_idx is not None:
                prev_peak_ti = (prev_pk - prev_onset_idx) / sr_hz

        # Ti ratios
        ti_ratio_to_next = ti / next_peak_ti if (ti is not None and next_peak_ti and next_peak_ti > 0) else None
        ti_ratio_to_prev = ti / prev_peak_ti if (ti is not None and prev_peak_ti and prev_peak_ti > 0) else None

        # Neighbor Te values
        next_peak_te = None
        if i+1 < len(offsets) and i+2 < len(onsets):
            next_offset_idx = offsets[i+1]
            next_next_onset_idx = onsets[i+2]
            if next_offset_idx is not None and next_next_onset_idx is not None:
                next_peak_te = (next_next_onset_idx - next_offset_idx) / sr_hz

        prev_peak_te = None
        if i > 0 and i-1 < len(offsets):
            prev_offset_idx = offsets[i-1]
            prev_next_onset_idx = onsets[i] if i < len(onsets) else None
            if prev_offset_idx is not None and prev_next_onset_idx is not None:
                prev_peak_te = (prev_next_onset_idx - prev_offset_idx) / sr_hz

        # Te ratios
        te_ratio_to_next = te / next_peak_te if (te is not None and next_peak_te and next_peak_te > 0) else None
        te_ratio_to_prev = te / prev_peak_te if (te is not None and prev_peak_te and prev_peak_te > 0) else None

        # Store all metrics
        metrics.append({
            # Identifiers
            'peak_idx': int(pk_idx),
            'peak_number': int(i),

            # Timing (absolute)
            'gap_to_prev': float(gap_to_prev) if gap_to_prev is not None else None,
            'gap_to_next': float(gap_to_next) if gap_to_next is not None else None,

            # Timing (normalized) - NORMALIZED BY RECORDING DISTRIBUTION
            'gap_to_prev_norm': float(gap_to_prev_normalized) if gap_to_prev_normalized is not None else None,
            'gap_to_next_norm': float(gap_to_next_normalized) if gap_to_next_normalized is not None else None,

            # Trough depth (merge indicators)
            'trough_depth_prev': float(trough_depth_prev) if trough_depth_prev is not None else None,
            'trough_ratio_prev': float(trough_ratio_prev) if trough_ratio_prev is not None else None,
            'trough_value_prev': float(trough_value_prev) if trough_value_prev is not None else None,
            'trough_depth_next': float(trough_depth_next) if trough_depth_next is not None else None,
            'trough_ratio_next': float(trough_ratio_next) if trough_ratio_next is not None else None,
            'trough_value_next': float(trough_value_next) if trough_value_next is not None else None,

            # Prominence asymmetry
            'left_prominence': float(left_prominence) if left_prominence is not None else None,
            'right_prominence': float(right_prominence) if right_prominence is not None else None,
            'prom_asymmetry': float(prom_asymmetry) if prom_asymmetry is not None else None,  # OLD: unsigned (0 to 1)
            'prom_asymmetry_signed': float(prom_asymmetry_signed) if prom_asymmetry_signed is not None else None,  # NEW: signed (-1 to +1)
            'total_prominence': float(total_prominence) if total_prominence is not None else None,

            # Trough asymmetry (signed)
            'trough_asymmetry_signed': float(trough_asymmetry_signed) if trough_asymmetry_signed is not None else None,

            # Onset position - STRONG INDICATOR OF SHOULDER PEAK
            'onset_above_zero': bool(onset_above_zero) if onset_above_zero is not None else None,
            'onset_height_ratio': float(onset_height_ratio) if onset_height_ratio is not None else None,
            'onset_value': float(onset_value) if onset_value is not None else None,

            # Amplitude (for noise detection)
            'amplitude_absolute': float(amp_absolute),
            'amplitude_normalized': float(amp_normalized),

            # Breath event values
            'offset_value': float(offset_value) if offset_value is not None else None,
            'expmin_value': float(expmin_value) if expmin_value is not None else None,

            # Timing metrics
            'ti': float(ti) if ti is not None else None,
            'te': float(te) if te is not None else None,

            # Probability metrics (from threshold model)
            # Note: p_noise/p_breath are signal-length arrays, sample at peak index
            'p_noise': (
                float(p_noise[pk_idx])
                if (p_noise is not None and pk_idx < len(p_noise) and not np.isnan(p_noise[pk_idx]))
                else None
            ),
            'p_breath': (
                float(p_breath[pk_idx])
                if (p_breath is not None and pk_idx < len(p_breath) and not np.isnan(p_breath[pk_idx]))
                else None
            ),

            # Edge case probability (uncertainty metric)
            # p_edge = 4*p_noise*p_breath peaks at 1.0 when p_noise=0.5 (maximum uncertainty)
            # Useful for identifying breaths near the classification boundary
            'p_edge': (
                float(4.0 * p_noise[pk_idx] * p_breath[pk_idx])
                if (p_noise is not None and p_breath is not None and
                    pk_idx < len(p_noise) and
                    not np.isnan(p_noise[pk_idx]) and
                    not np.isnan(p_breath[pk_idx]))
                else None
            ),

            # === NEIGHBOR COMPARISON FEATURES ===

            # Neighbor amplitudes
            'next_peak_amplitude': float(next_peak_amplitude) if next_peak_amplitude is not None else None,
            'prev_peak_amplitude': float(prev_peak_amplitude) if prev_peak_amplitude is not None else None,

            # Amplitude comparisons
            'amplitude_ratio_to_next': float(amplitude_ratio_to_next) if amplitude_ratio_to_next is not None else None,
            'amplitude_ratio_to_prev': float(amplitude_ratio_to_prev) if amplitude_ratio_to_prev is not None else None,
            'amplitude_diff_to_next_signed': float(amplitude_diff_to_next_signed) if amplitude_diff_to_next_signed is not None else None,
            'amplitude_diff_to_prev_signed': float(amplitude_diff_to_prev_signed) if amplitude_diff_to_prev_signed is not None else None,

            # Neighbor prominences
            'next_peak_prominence': float(next_peak_prominence) if next_peak_prominence is not None else None,
            'prev_peak_prominence': float(prev_peak_prominence) if prev_peak_prominence is not None else None,

            # Prominence comparisons
            'prominence_ratio_to_next': float(prominence_ratio_to_next) if prominence_ratio_to_next is not None else None,
            'prominence_ratio_to_prev': float(prominence_ratio_to_prev) if prominence_ratio_to_prev is not None else None,

            # Neighbor onset height ratios (critical for merge detection!)
            'next_peak_onset_height_ratio': float(next_peak_onset_height_ratio) if next_peak_onset_height_ratio is not None else None,
            'prev_peak_onset_height_ratio': float(prev_peak_onset_height_ratio) if prev_peak_onset_height_ratio is not None else None,

            # Neighbor Ti values
            'next_peak_ti': float(next_peak_ti) if next_peak_ti is not None else None,
            'prev_peak_ti': float(prev_peak_ti) if prev_peak_ti is not None else None,

            # Ti comparisons
            'ti_ratio_to_next': float(ti_ratio_to_next) if ti_ratio_to_next is not None else None,
            'ti_ratio_to_prev': float(ti_ratio_to_prev) if ti_ratio_to_prev is not None else None,

            # Neighbor Te values
            'next_peak_te': float(next_peak_te) if next_peak_te is not None else None,
            'prev_peak_te': float(prev_peak_te) if prev_peak_te is not None else None,

            # Te comparisons
            'te_ratio_to_next': float(te_ratio_to_next) if te_ratio_to_next is not None else None,
            'te_ratio_to_prev': float(te_ratio_to_prev) if te_ratio_to_prev is not None else None,
        })

    return metrics
